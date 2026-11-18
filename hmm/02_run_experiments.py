"""Decode all 80 training trips under every experimental HMM config (HMM-improvement).

Emissions are the out-of-fold lgbm probabilities from ``01_oof_emissions.py``
(each trip scored by a model that never saw it). Results are appended per
trip to ``outputs/hmm/runs_<oof>.csv``, so an interrupted run resumes
where it stopped; ``03_summarize.py`` turns them into the comparison table.

Usage::

    python hmm/02_run_experiments.py --check          # v2 == Phase 4 decoder?
    python hmm/02_run_experiments.py --oof base
    python hmm/02_run_experiments.py --oof base --families temp --temperature 1.7
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from typing import Dict

import numpy as np
import pandas as pd

from common import DATA, OUT, continuity
import hmm_graph as hg
import hmm_viterbi as hv
import io_utils as io
from hmm_v2 import Config, decode, prepare


def registry(temperature: float = 1.0) -> Dict[str, Dict[str, Config]]:
    """Experiment families -> {config name: Config}. Grids are small on purpose."""
    beta_grid = [(20, 4), (20, 8), (20, 16), (20, 32), (20, 64), (20, 128), (10, 16), (40, 16)]
    lap_grid = [(s, b) for s in (2, 5, 10) for b in (1, 2, 4, 8)]
    fam: Dict[str, Dict[str, Config]] = {
        "A_base": {f"A_s{s}_b{b}": Config(sigma=s, beta=b) for s, b in beta_grid},
        "B_signed": {f"B_s{s}_b{b}": Config(sigma=s, beta=b, same_arc="signed")
                     for s, b in beta_grid},
        "C_signed_uturn": {f"C_b{b}_u{u}": Config(beta=b, same_arc="signed", uturn_penalty=u)
                           for b in (16, 32, 64) for u in (2, 5, 10, 20)},
        "D_laplace": {f"D_s{s}_b{b}": Config(sigma=s, beta=b, dist="laplace")
                      for s, b in lap_grid},
        "E_laplace_signed": {f"E_s{s}_b{b}": Config(sigma=s, beta=b, dist="laplace",
                                                    same_arc="signed") for s, b in lap_grid},
        "F_norm": {f"F_{m}_b{b}": Config(beta=b, same_arc=m, emission="norm")
                   for m in ("abs", "signed") for b in (8, 16, 32)},
        # E1 refinement: backward slack up to tol metres costs like Phase 4
        "H_signed_tol": {f"H_tol{tol}_b{b}": Config(beta=b, same_arc="signed_tol",
                                                     jitter_tol_m=tol)
                         for tol in (5, 10, 20) for b in (16, 32, 64)},
        # (a, b) are filled in per trip from that trip's fold (fitted on the others)
        "G_platt": {f"G_{m}_b{b}": Config(beta=b, same_arc=m, emission="platt")
                    for m in ("abs", "signed") for b in (2, 4, 8, 16, 32)},
    }
    return fam


FIELDS = ["trip_id", "fold", "config", "family", "n_points", "n_correct", "cls_correct",
          "n_conn", "n_trans", "tp", "fp", "fn", "tn", "err_twin", "err_adjacent",
          "err_other", "decode_s"]


def score(path: np.ndarray, pts: np.ndarray, truth: np.ndarray, n_pos: np.ndarray,
          n_pairs: np.ndarray, label_of: Dict, fr: np.ndarray, to: np.ndarray) -> Dict:
    pick = path[pts]
    ok = (pick == truth) & (truth >= 0)
    tp = int(sum(label_of.get((int(p), int(a)), 0) for p, a in zip(pts, pick)))
    fp = len(pts) - tp
    fn = int(n_pos.sum()) - tp
    tn = int(n_pairs.sum()) - tp - fp - fn
    wrong = ~ok & (truth >= 0)
    h, t = pick[wrong], truth[wrong]
    twin = (fr[h] == to[t]) & (to[h] == fr[t])
    adj = ~twin & ((fr[h] == fr[t]) | (fr[h] == to[t]) | (to[h] == fr[t]) | (to[h] == to[t]))
    conn, trans = continuity(pick, fr, to)
    return dict(n_points=len(pts), n_correct=int(ok.sum()), n_conn=conn, n_trans=trans,
                tp=tp, fp=fp, fn=fn, tn=tn, err_twin=int(twin.sum()),
                err_adjacent=int(adj.sum()), err_other=int((~twin & ~adj).sum()))


def check(oof: pd.DataFrame, n: int = 4) -> None:
    """The v2 decoder with default Config must equal Phase 4's decoder."""
    for tid in sorted(oof.trip_id.unique())[:n]:
        trip = io.load_trip(DATA / "train" / tid)
        sub = oof[oof.trip_id == tid]
        ref = hv.decode_trip(tid, sub, trip, 20.0, 16.0, use_breaks=False)["arc_id_hmm"].to_numpy()
        new = decode(prepare(tid, sub, trip), Config())
        print(f"{tid}: {len(ref)} pts, identical={np.array_equal(ref, new)}, "
              f"diff={(ref != new).sum()}", flush=True)
        hg.clear_caches()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="base")
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    oof = pd.read_parquet(OUT / f"oof_{args.oof}.parquet")
    oof["trip_id"] = oof["trip_id"].astype(str)
    if args.check:
        check(oof)
        return 0

    fams = registry(args.temperature)
    names = args.families or [f for f in fams if f != "G_platt"]
    platt_path = OUT / f"platt_{args.oof}.json"
    platt = ({int(k): v for k, v in json.loads(platt_path.read_text()).items()}
             if platt_path.exists() else {})
    configs = {c: (f, cfg) for f in names for c, cfg in fams[f].items()}
    out_csv = OUT / f"runs_{args.oof}.csv"
    done = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv, dtype={"trip_id": str})
        done = set(zip(prev.trip_id, prev.config))
    new_file = not out_csv.exists()
    fh = open(out_csv, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()

    trips = sorted(oof.trip_id.unique())
    print(f"{len(configs)} configs x {len(trips)} trips -> {out_csv.name}", flush=True)
    t_all = time.perf_counter()
    for k, tid in enumerate(trips, 1):
        todo = [c for c in configs if (tid, c) not in done]
        if not todo:
            continue
        t0 = time.perf_counter()
        trip = io.load_trip(DATA / "train" / tid)
        sub = oof[oof.trip_id == tid]
        fr = trip["arcs"]["from_node"].to_numpy()
        to = trip["arcs"]["to_node"].to_numpy()
        per_pt = sub.groupby("point_idx").agg(truth=("truth_arc_id", "first"),
                                              n_pos=("label", "sum"), n_pairs=("label", "size"))
        pts = per_pt.index.to_numpy()
        truth = per_pt["truth"].to_numpy()
        label_of = {(int(p), int(a)): 1 for p, a in
                    zip(sub.point_idx[sub.label == 1], sub.arc_id[sub.label == 1])}
        cls = (sub.sort_values(["point_idx", "y_prob", "candidate_rank"],
                               ascending=[True, False, True])
               .groupby("point_idx")["arc_id"].first().loc[pts].to_numpy())
        cls_correct = int(((cls == truth) & (truth >= 0)).sum())
        prep = prepare(tid, sub, trip)
        fold = int(sub["fold"].iloc[0])
        for c in todo:
            fam, cfg = configs[c]
            if cfg.emission == "platt":
                cfg = replace(cfg, platt_a=platt[fold]["a"], platt_b=platt[fold]["b"])
            t1 = time.perf_counter()
            path = decode(prep, cfg)
            row = score(path, pts, truth, per_pt.n_pos.to_numpy(), per_pt.n_pairs.to_numpy(),
                        label_of, fr, to)
            row.update(trip_id=tid, fold=fold, config=c, family=fam, cls_correct=cls_correct,
                       decode_s=round(time.perf_counter() - t1, 3))
            writer.writerow(row)
        fh.flush()
        hg.clear_caches()
        print(f"[{k:2d}/{len(trips)}] {tid} {len(pts):6d} pts  {len(todo)} configs  "
              f"{time.perf_counter() - t0:.0f}s  (total {time.perf_counter() - t_all:.0f}s)",
              flush=True)
    fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
