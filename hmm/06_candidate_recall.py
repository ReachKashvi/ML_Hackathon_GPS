"""Candidate recall vs search radius R and cap K on the 80 training trips (E6).

The HMM can only pick arcs that are in a point's candidate set, so the
share of aligned points whose true arc is missing is a hard ceiling. This
measures that ceiling for a few (R, K) settings using the same truth
alignment as the labels. Measurement only: nothing downstream changes.
"""

from __future__ import annotations

import time

import pandas as pd

from common import DATA, OUT
import candidates as cd
import io_utils as io
import labels as lb

SETTINGS = [(50.0, 10), (50.0, 15), (75.0, 10), (75.0, 15), (100.0, 20)]


def main() -> int:
    rows = []
    trips = io.list_trips(DATA / "train")
    for k, tid in enumerate(trips, 1):
        t0 = time.perf_counter()
        trip = io.load_trip(DATA / "train" / tid)
        truth = lb.align_gps_to_route(trip).sort_values("point_idx")["truth_arc_id"].to_numpy()
        for R, K in SETTINGS:
            c = cd.generate_candidates(trip, K=K, R=R)
            m = cd.candidate_miss_rate(trip, c, truth_arc_by_point=truth)
            m.update(R=R, K=K, n_pairs=len(c))
            rows.append(m)
        print(f"[{k}/{len(trips)}] {tid} {time.perf_counter() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "candidate_recall_train.csv", index=False)
    g = df.groupby(["R", "K"]).agg(n_aligned=("n_aligned", "sum"), n_missed=("n_missed", "sum"),
                                   n_pairs=("n_pairs", "sum"))
    ex = df[df.trip_id != "00000054"].groupby(["R", "K"]).agg(
        n_aligned=("n_aligned", "sum"), n_missed=("n_missed", "sum"))
    g["miss_rate"] = g.n_missed / g.n_aligned
    g["miss_rate_excl_00000054"] = ex.n_missed / ex.n_aligned
    g["pairs_vs_R50K10"] = g.n_pairs / g.loc[(50.0, 10), "n_pairs"]
    g.to_csv(OUT / "candidate_recall_summary.csv")
    print(g.round(5).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
