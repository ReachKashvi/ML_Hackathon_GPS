"""End-to-end inference for one trip: raw files -> matched arcs + metrics.

This is the production path the API serves (Laptop 1). For a trip folder
containing ``<id>.track``, ``<id>.nodes``, ``<id>.arcs`` (``<id>.route`` is
optional) it runs exactly the pipeline the reported results came from:

1. ``candidates.generate_candidates``  -- K=10 arcs within R=50 m per GPS point
2. ``features.build_features``         -- the 23 Phase 2 features
   (+ ``hmm/context_features.py`` when the configured model uses them)
3. LightGBM ``predict_proba``          -- emission probability per pair
4. ``hmm/hmm_v2.decode``               -- Viterbi decoding; the default config
   is the Phase 4 decoder (sigma=20, beta=16), verified identical
5. metrics, computed **server-side**: Top-1 against the monotone route
   alignment (only when ``.route`` exists -- never sent to the UI), route
   continuity, counts and wall-clock inference time.

No route-break resets are used: ``.route`` is not needed to decode.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "hmm"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import candidates as cd  # noqa: E402
import features as ft  # noqa: E402
import geo_utils as geo  # noqa: E402
import hmm_graph as hg  # noqa: E402
import io_utils as io  # noqa: E402
import labels as lb  # noqa: E402
from context_features import add_context_features  # noqa: E402
from hmm_v2 import Config, decode, prepare  # noqa: E402

CONFIG_PATH = Path(os.environ.get("PATHSNAP_CONFIG", Path(__file__).with_name("config.yaml")))


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


class Predictor:
    """Loads the model once; ``predict(trip_id)`` is thread-safe."""

    def __init__(self, config_path: Path = CONFIG_PATH):
        self.cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        self.data_dir = _resolve(os.environ.get("PATHSNAP_DATA_DIR", self.cfg["data_dir"]))
        bundle = joblib.load(_resolve(self.cfg["model"]))
        self.model = bundle["model"] if isinstance(bundle, dict) else bundle
        cols = bundle.get("feature_columns") if isinstance(bundle, dict) else None
        self.feature_columns: List[str] = list(cols or self.cfg.get("feature_columns")
                                               or ft.FEATURE_COLUMNS)
        self.use_context = any(c.startswith("x_") for c in self.feature_columns)
        keys = {f.name for f in fields(Config)}
        self.hmm = Config(**{k: v for k, v in (self.cfg.get("hmm") or {}).items() if k in keys})
        self.model_name = self.cfg.get("model_name", "PathSnap lgbm + HMM/Viterbi")
        self.K = int(self.cfg.get("candidates", {}).get("K", cd.DEFAULT_K))
        self.R = float(self.cfg.get("candidates", {}).get("R", cd.DEFAULT_R_M))
        self.road_radius_m = float(self.cfg.get("road_network_radius_m", 150.0))
        self._lock = threading.Lock()
        self._cache: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    def list_trips(self) -> List[str]:
        return io.list_trips(self.data_dir)

    def describe(self) -> Dict[str, Any]:
        return {"model": self.model_name, "n_features": len(self.feature_columns),
                "context_features": self.use_context, "hmm": self.hmm.as_dict(),
                "candidates": {"K": self.K, "R_m": self.R}, "data_dir": str(self.data_dir)}

    # ------------------------------------------------------------------
    def emissions(self, trip: Dict[str, Any]) -> pd.DataFrame:
        cands = cd.generate_candidates(trip, K=self.K, R=self.R)
        feats = ft.build_features(trip, cands)
        if self.use_context:
            feats = add_context_features(feats, trip)
        X = np.ascontiguousarray(feats[self.feature_columns].to_numpy(dtype="float32"))
        X = pd.DataFrame(X, columns=self.feature_columns)
        feats["y_prob"] = self.model.predict_proba(X)[:, 1] if len(X) else []
        feats["trip_id"] = str(trip["trip_id"])
        return feats[["trip_id", "point_idx", "arc_id", "candidate_rank",
                      "frac_along_arc", "y_prob"]]

    def predict(self, trip_id: str) -> Dict[str, Any]:
        if trip_id not in set(self.list_trips()):
            raise KeyError(trip_id)
        with self._lock:
            if trip_id in self._cache:
                return self._cache[trip_id]
            out = self._predict(trip_id)
            if self.cfg.get("cache_predictions", True):
                self._cache[trip_id] = out
            return out

    def _predict(self, trip_id: str) -> Dict[str, Any]:
        # per-trip caches are keyed by trip id; clear them so a trip id that
        # also exists in another dataset can never reuse a stale projection
        hg.clear_caches()
        geo.clear_transformer_cache()
        t0 = time.perf_counter()
        trip = io.load_trip(self.data_dir / trip_id)
        em = self.emissions(trip)
        path = decode(prepare(trip_id, em, trip), self.hmm)
        hg.clear_caches()
        inference_s = time.perf_counter() - t0

        arcs = trip["arcs"]
        fr, to = arcs["from_node"].to_numpy(), arcs["to_node"].to_numpy()
        nodes = trip["nodes"]
        nlon, nlat = nodes["lon"].to_numpy(), nodes["lat"].to_numpy()
        track = trip["track"]

        pts = np.sort(em["point_idx"].unique())          # points with >= 1 candidate
        seq = path[pts] if len(pts) else np.array([], "int64")
        same_or_linked = (seq[:-1] == seq[1:]) | (to[seq[:-1]] == fr[seq[1:]])
        continuity = float(same_or_linked.mean()) if len(seq) > 1 else None

        top1 = None
        if len(trip.get("route", [])) and len(pts):
            truth = lb.align_gps_to_route(trip).sort_values("point_idx")["truth_arc_id"].to_numpy()
            t = truth[pts]
            top1 = float(((seq == t) & (t >= 0)).mean())

        collapsed = [int(a) for k, a in enumerate(path) if a >= 0 and (k == 0 or a != path[k - 1])]
        arc_coords = [[[float(nlon[fr[a]]), float(nlat[fr[a]])],
                       [float(nlon[to[a]]), float(nlat[to[a]])]] for a in collapsed]

        return {
            "trip_id": trip_id,
            "model": self.model_name,
            "gps_points": track[["lon", "lat", "t_sec"]].astype(float).to_numpy().round(7).tolist(),
            "point_arcs": [int(a) for a in path],
            "predicted_arcs": collapsed,
            "predicted_arc_coords": arc_coords,
            "road_network": self._road_network(trip),
            "metrics": {
                "top1_accuracy": top1,
                "route_continuity": continuity,
                "n_gps_points": int(len(track)),
                "n_points_with_candidates": int(len(pts)),
                "n_predicted_arcs": len(collapsed),
                "inference_time_sec": round(inference_s, 3),
            },
        }

    def _road_network(self, trip: Dict[str, Any]) -> Dict[str, Any]:
        """Arcs with both ends within ``road_network_radius_m`` of the track."""
        proj = geo.project_trip(trip)
        tree = cKDTree(np.c_[proj["track_x"], proj["track_y"]])
        d, _ = tree.query(np.c_[proj["node_x"], proj["node_y"]],
                          distance_upper_bound=self.road_radius_m)
        near = np.isfinite(d)
        arcs = trip["arcs"]
        fr, to = arcs["from_node"].to_numpy(), arcs["to_node"].to_numpy()
        keep = near[fr] & near[to]
        node_ids = np.flatnonzero(near)
        remap = -np.ones(len(near), dtype="int64")
        remap[node_ids] = np.arange(len(node_ids))
        nodes = trip["nodes"]
        return {
            "node_coords": np.c_[nodes["lon"].to_numpy()[node_ids],
                                 nodes["lat"].to_numpy()[node_ids]].round(7).tolist(),
            "arc_pairs": np.c_[remap[fr[keep]], remap[to[keep]]].tolist(),
        }

