"""Tune the HMM's sigma_trans and beta on training trips only (Phase 4, Step 2).

Leakage and bias
----------------
* Only ``data/train`` trips are used. ``student_test`` is never opened here.
* Emissions are **out-of-fold**. ``lgbm.joblib`` was fitted on all 80
  training trips, so predicting on any of them is in-sample: trip 00000001
  scores 0.949 Top-1 in-sample against ~0.76 under cross-validation. Tuning
  beta on those overconfident probabilities would teach the decoder to trust
  emissions far more than it should on unseen trips. Instead one LightGBM
  with the identical untuned configuration is fitted on the trips that are
  *not* being decoded, and the decoded trips are scored by that model.

Trips decoded
-------------
* **tune**  -- 20 training trips drawn with seed 42; the grid is scored on
  these.
* **break** -- the 10 training trips with route-chain breaks, used to check
  that the restart-at-segment-start rule does not hurt accuracy around the
  breaks. They are excluded from the OOF model's training set too.

Memory
------
Arc graphs reach 300k edges, so the loop is trip-outer: each trip's graph and
neighbour cache are built once, reused across every grid cell, then freed.

Outputs
-------
* ``outputs/predictions/lgbm_train_sample.parquet`` -- cached OOF emissions
* ``outputs/hmm_tuning.csv``      -- full grid, per-cell mean/pooled Top-1, time
* ``outputs/hmm_tuning_per_trip.csv``
* ``outputs/best_hmm_params.json``
* ``outputs/hmm_break_check_train.csv`` -- accuracy around route breaks
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import features as ft  # noqa: E402
import hmm_graph as hg  # noqa: E402
import hmm_viterbi as hv  # noqa: E402
import io_utils as io  # noqa: E402
import labels as lb  # noqa: E402
import model_zoo as mz  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "outputs"
SIGMAS = [10.0, 20.0, 40.0, 80.0]
BETAS = [0.5, 1.0, 2.0, 4.0]
BREAK_WINDOW = 30  # GPS points either side of a reset point


def classifier_pick(sub: pd.DataFrame) -> pd.Series:
    """Argmax arc per point, ties to the nearest candidate."""
    s = sub.sort_values(["point_idx", "y_prob", "candidate_rank"],
                        ascending=[True, False, True])
    return s.groupby("point_idx", sort=True)["arc_id"].first()


def oof_predictions(pairs: pd.DataFrame, decode_ids: List[str]) -> pd.DataFrame:
    """Fit untuned lgbm on the other trips, score the decoded ones."""
    from lightgbm import LGBMClassifier

    cols = [c for c in ft.FEATURE_COLUMNS if c in pairs.columns]
    fit_mask = ~pairs["trip_id"].isin(decode_ids)
    Xf = np.ascontiguousarray(pairs.loc[fit_mask, cols].astype("float32").to_numpy())
    yf = np.ascontiguousarray(pairs.loc[fit_mask, "label"].to_numpy().astype("int32"))
    print(f"OOF model: fitting on {pairs.loc[fit_mask, 'trip_id'].nunique()} trips "
          f"({len(Xf):,} rows), scoring {len(decode_ids)} held-out trips")
    model = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        scale_pos_weight=mz.compute_scale_pos_weight(yf),
        random_state=mz.RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    t0 = time.perf_counter()
    model.fit(Xf, yf)
    print(f"  fitted in {time.perf_counter() - t0:.0f}s")

    sub = pairs[pairs["trip_id"].isin(decode_ids)].copy()
    Xs = np.ascontiguousarray(sub[cols].astype("float32").to_numpy())
    sub["y_prob"] = model.predict_proba(Xs)[:, 1]
    return sub[["trip_id", "point_idx", "arc_id", "candidate_rank",
                "frac_along_arc", "y_prob", "label", "truth_arc_id"]]


def top1(picks: pd.Series, truth: pd.Series) -> float:
    """Fraction of points whose pick equals the aligned truth arc."""
    t = truth.loc[picks.index]
    return float(((picks == t) & (t >= 0)).mean())


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-tune", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--sigmas", type=float, nargs="*", default=SIGMAS)
    parser.add_argument("--betas", type=float, nargs="*", default=BETAS)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    train_ids = io.list_trips(DATA / "train")
    test_ids = set(io.list_trips(DATA / "student_test"))
    assert not (set(train_ids) & test_ids), "train/student_test overlap"

    rng = np.random.default_rng(args.seed)
    tune_ids = sorted(rng.choice(train_ids, size=args.n_tune, replace=False).tolist())

    # break trips among the training set
    break_ids = []
    for tid in train_ids:
        t = io.load_trip(DATA / "train" / tid)
        if len(io.route_segments(t["route"], t["arcs"])) > 1:
            break_ids.append(tid)
    decode_ids = sorted(set(tune_ids) | set(break_ids))
    print(f"tune trips ({len(tune_ids)}): {tune_ids}")
    print(f"break trips ({len(break_ids)}): {break_ids}")
    print(f"student_test trips touched: 0 (only data/train is read)\n")

    cache = OUT / "predictions" / "lgbm_train_sample.parquet"
    if cache.exists() and not args.refresh_cache:
        preds = pd.read_parquet(cache)
        if set(preds["trip_id"].unique()) != set(decode_ids):
            preds = None
    else:
        preds = None
    if preds is None:
        pairs = pd.read_parquet(OUT / "training_pairs.parquet")
        preds = oof_predictions(pairs, decode_ids)
        del pairs
        preds["role"] = np.where(preds["trip_id"].isin(tune_ids), "tune", "break")
        cache.parent.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(cache, index=False)
        print(f"cached OOF emissions -> {cache}")
    else:
        print(f"using cached OOF emissions {cache}")

    cells = list(itertools.product(args.sigmas, args.betas))
    per_trip: List[Dict] = []
    cell_time = {c: 0.0 for c in cells}
    cell_time_ablate = {"no_position": 0.0, "no_breaks": 0.0}
    break_rows: List[Dict] = []

    for k, tid in enumerate(decode_ids, 1):
        trip = io.load_trip(DATA / "train" / tid)
        sub = preds[preds["trip_id"] == tid]
        truth = sub.groupby("point_idx")["truth_arc_id"].first()
        cls = classifier_pick(sub)
        cls_top1 = top1(cls, truth)
        is_tune = tid in tune_ids
        segments = hg.load_route_segments(trip)
        alignment = lb.align_gps_to_route(trip) if len(segments) > 1 else None

        t_trip = time.perf_counter()
        if is_tune:
            for sigma, beta in cells:
                t0 = time.perf_counter()
                dec = hv.decode_trip(tid, sub, trip, sigma, beta, alignment=alignment)
                cell_time[(sigma, beta)] += time.perf_counter() - t0
                picks = dec.set_index("point_idx")["arc_id_hmm"].loc[cls.index]
                per_trip.append({
                    "trip_id": tid, "sigma": sigma, "beta": beta,
                    "use_position": True, "use_breaks": True,
                    "n_points": len(cls), "classifier_top1": cls_top1,
                    "hmm_top1": top1(picks, truth),
                })
        print(f"[{k:2d}/{len(decode_ids)}] {tid} {'tune ' if is_tune else 'break'} "
              f"{len(cls):5d} pts  classifier {cls_top1:.4f}  "
              f"({time.perf_counter() - t_trip:.0f}s)")

        # free this trip's graph and neighbour cache before the next one
        hg.clear_caches()

    grid = pd.DataFrame(per_trip)
    summary = (grid.groupby(["sigma", "beta"])
               .agg(mean_top1=("hmm_top1", "mean"),
                    std_top1=("hmm_top1", "std"),
                    classifier_mean_top1=("classifier_top1", "mean"),
                    n_trips=("trip_id", "nunique"))
               .reset_index())
    # pooled (point-weighted) top1 alongside the per-trip mean
    pooled = (grid.assign(correct=grid.hmm_top1 * grid.n_points)
              .groupby(["sigma", "beta"])
              .apply(lambda g: g.correct.sum() / g.n_points.sum(), include_groups=False)
              .rename("pooled_top1").reset_index())
    summary = summary.merge(pooled, on=["sigma", "beta"])
    summary["delta_vs_classifier"] = summary.mean_top1 - summary.classifier_mean_top1
    summary["elapsed_sec"] = [round(cell_time[(s, b)], 2)
                              for s, b in zip(summary.sigma, summary.beta)]
    summary = summary.sort_values("mean_top1", ascending=False).reset_index(drop=True)

    best = summary.iloc[0]
    sigma_b, beta_b = float(best.sigma), float(best.beta)
    print("\n== grid (mean Top-1 across tune trips) ==")
    print(summary.round(4).to_string(index=False))

    # ---- ablations at the best cell + break check, second pass ------------
    ablate_rows: List[Dict] = []
    for tid in decode_ids:
        trip = io.load_trip(DATA / "train" / tid)
        sub = preds[preds["trip_id"] == tid]
        truth = sub.groupby("point_idx")["truth_arc_id"].first()
        cls = classifier_pick(sub)
        segments = hg.load_route_segments(trip)
        alignment = lb.align_gps_to_route(trip) if len(segments) > 1 else None

        variants = {"full": dict(use_position=True, use_breaks=True)}
        if tid in tune_ids:
            variants["no_position"] = dict(use_position=False, use_breaks=True)
        if len(segments) > 1:
            variants["no_breaks"] = dict(use_position=True, use_breaks=False)

        decoded = {}
        for name, kw in variants.items():
            t0 = time.perf_counter()
            dec = hv.decode_trip(tid, sub, trip, sigma_b, beta_b, alignment=alignment, **kw)
            if name in cell_time_ablate:
                cell_time_ablate[name] += time.perf_counter() - t0
            decoded[name] = dec.set_index("point_idx")["arc_id_hmm"].loc[cls.index]

        if tid in tune_ids:
            ablate_rows.append({
                "trip_id": tid, "n_points": len(cls),
                "full": top1(decoded["full"], truth),
                "no_position": top1(decoded["no_position"], truth),
                "classifier": top1(cls, truth),
            })

        if len(segments) > 1:
            resets = sorted(hv.reset_points_from_segments(segments, trip, alignment))
            idx = cls.index.to_numpy()
            window = np.zeros(len(idx), dtype=bool)
            for r in resets:
                window |= (idx >= r - BREAK_WINDOW) & (idx <= r + BREAK_WINDOW)
            w_idx = cls.index[window]
            if len(w_idx):
                tw = truth.loc[w_idx]
                break_rows.append({
                    "trip_id": tid, "role": "tune" if tid in tune_ids else "break",
                    "n_breaks": len(resets), "window_points": int(window.sum()),
                    "classifier_acc": float(((cls.loc[w_idx] == tw) & (tw >= 0)).mean()),
                    "hmm_with_resets_acc": float(((decoded["full"].loc[w_idx] == tw) & (tw >= 0)).mean()),
                    "hmm_no_resets_acc": float(((decoded["no_breaks"].loc[w_idx] == tw) & (tw >= 0)).mean()),
                    "trip_classifier_top1": top1(cls, truth),
                    "trip_hmm_top1": top1(decoded["full"], truth),
                    "trip_hmm_no_resets_top1": top1(decoded["no_breaks"], truth),
                })
        hg.clear_caches()

    ablate = pd.DataFrame(ablate_rows)
    breaks = pd.DataFrame(break_rows)

    print(f"\n== ablation at sigma={sigma_b:g}, beta={beta_b:g} (tune trips) ==")
    print(f"  classifier             {ablate.classifier.mean():.4f}")
    print(f"  HMM, position-aware    {ablate.full.mean():.4f}")
    print(f"  HMM, arc-end distance  {ablate.no_position.mean():.4f}  (the brief's formulation)")

    print(f"\n== route-break check (+/-{BREAK_WINDOW} points around each reset) ==")
    print(breaks.round(4).to_string(index=False))

    # ---- persist ---------------------------------------------------------
    summary.to_csv(OUT / "hmm_tuning.csv", index=False)
    grid.to_csv(OUT / "hmm_tuning_per_trip.csv", index=False)
    breaks.to_csv(OUT / "hmm_break_check_train.csv", index=False)
    payload = {
        "sigma_trans": sigma_b,
        "beta": beta_b,
        "use_position": bool(ablate.full.mean() >= ablate.no_position.mean()),
        "max_gap_s": hv.DEFAULT_MAX_GAP_S,
        "cutoff_m": hg.DEFAULT_CUTOFF_M,
        "tune_mean_top1": float(best.mean_top1),
        "tune_pooled_top1": float(best.pooled_top1),
        "tune_classifier_mean_top1": float(best.classifier_mean_top1),
        "ablation_position_aware": float(ablate.full.mean()),
        "ablation_arc_end": float(ablate.no_position.mean()),
        "tune_trip_ids": tune_ids,
        "break_check_trip_ids": break_ids,
        "emissions": "out-of-fold lgbm (untuned config) fitted on trips not decoded",
        "student_test_used": False,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    (OUT / "best_hmm_params.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nbest: sigma={sigma_b:g} beta={beta_b:g} "
          f"(mean Top-1 {best.mean_top1:.4f} vs classifier {best.classifier_mean_top1:.4f})")
    print(f"wrote hmm_tuning.csv, hmm_tuning_per_trip.csv, best_hmm_params.json, "
          f"hmm_break_check_train.csv  [{time.perf_counter() - started:.0f}s total]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
