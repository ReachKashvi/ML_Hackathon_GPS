"""Build the (GPS point, candidate arc) training table (Step 9).

Runs the full Phase-1 pipeline over the **80 training trips only**:

    load -> generate candidates -> align GPS to the ground-truth route
         -> build features -> assign labels

and writes:

* ``outputs/training_pairs.parquet``   one row per candidate pair
* ``outputs/candidate_miss_rate.csv``  per-trip and overall miss rates
* ``outputs/alignment_report.csv``     per-trip alignment diagnostics

Usage
-----
::

    python scripts/build_training_pairs.py
    python scripts/build_training_pairs.py --partition train --K 10 --R 50
    python scripts/build_training_pairs.py --limit 5        # quick smoke run

The ``--partition`` switch exists so the identical pipeline can later produce
student-test pairs without duplicating any logic; it refuses anything that is
not a directory under ``data/``, and the evaluator's test data is never
present there.

No model is trained here.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import candidates as cnd  # noqa: E402
import features as ft  # noqa: E402
import geo_utils as geo  # noqa: E402
import io_utils as io  # noqa: E402
import labels as lb  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "outputs"

#: Columns written to the parquet, in order. ``trip_id`` and ``point_idx``
#: are preserved so Phase 2 can do group-aware cross-validation by trip.
OUTPUT_COLUMNS = (
    "trip_id",
    "point_idx",
    "arc_id",
    *ft.FEATURE_COLUMNS,   # includes candidate_rank
    "truth_arc_id",
    "point_assigned",
    "label",
)


def build_partition(
    partition: str,
    k: int,
    radius: float,
    max_assign_dist_m: float,
    limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the pipeline over one partition.

    Returns
    -------
    (pairs, miss, align) : tuple of DataFrame
        The candidate-pair table, the per-trip miss-rate table and the
        per-trip alignment diagnostics.
    """
    partition_dir = DATA / partition
    trip_ids = io.list_trips(partition_dir)
    if limit:
        trip_ids = trip_ids[:limit]
    print(f"{partition}: {len(trip_ids)} trips, K={k}, R={radius} m")

    pair_frames: List[pd.DataFrame] = []
    miss_rows: List[dict] = []
    align_rows: List[dict] = []

    for trip_id in tqdm(trip_ids, desc=f"building {partition} pairs", unit="trip"):
        t0 = time.perf_counter()
        trip = io.load_trip(partition_dir / trip_id)

        cands = cnd.generate_candidates(trip, K=k, R=radius)
        alignment = lb.align_gps_to_route(trip, max_assign_dist_m=max_assign_dist_m)
        feats = ft.build_features(trip, cands)
        labelled = lb.label_candidates(trip, cands, alignment=alignment)

        # Features and labels are both row-aligned to `cands`, so this is a
        # positional join; asserted rather than assumed.
        assert len(feats) == len(labelled) == len(cands), (
            f"{trip_id}: row count mismatch "
            f"{len(feats)} / {len(labelled)} / {len(cands)}"
        )
        assert np.array_equal(
            feats["point_idx"].to_numpy(), labelled["point_idx"].to_numpy()
        ), f"{trip_id}: point_idx order diverged between features and labels"
        assert np.array_equal(
            feats["arc_id"].to_numpy(), labelled["arc_id"].to_numpy()
        ), f"{trip_id}: arc_id order diverged between features and labels"

        frame = feats.copy()
        frame.insert(0, "trip_id", trip_id)
        frame["truth_arc_id"] = labelled["truth_arc_id"].to_numpy()
        frame["point_assigned"] = labelled["point_assigned"].to_numpy()
        frame["label"] = labelled["label"].to_numpy()
        pair_frames.append(frame[list(OUTPUT_COLUMNS)])

        miss = cnd.candidate_miss_rate(
            trip, cands, truth_arc_by_point=alignment["truth_arc_id"].to_numpy()
        )
        miss["partition"] = partition
        miss["n_pairs"] = len(frame)
        miss["n_positives"] = int(frame["label"].sum())
        miss["seconds"] = round(time.perf_counter() - t0, 3)
        miss_rows.append(miss)

        report = lb.alignment_report(trip, alignment)
        report["partition"] = partition
        report["utm_epsg"] = geo.get_trip_epsg(trip)
        align_rows.append(report)

        # One transformer per trip is cached; 80 of them is harmless, but
        # clearing keeps memory flat if this is ever run over more.
        geo.clear_transformer_cache()

    pairs = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame(
        columns=list(OUTPUT_COLUMNS)
    )
    return pairs, pd.DataFrame(miss_rows), pd.DataFrame(align_rows)


def validate(pairs: pd.DataFrame, miss: pd.DataFrame, partition: str,
             limit: int | None = None) -> List[str]:
    """Run the Step 9 validation checklist. Returns a list of result lines."""
    lines: List[str] = []

    def check(ok: bool, text: str) -> None:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {text}")

    # 1. no evaluator test data was accessed
    accessible = sorted(p.name for p in DATA.iterdir() if p.is_dir())
    check(
        set(accessible) <= {"train", "student_test"},
        f"only train/student_test partitions exist under data/ (found: {accessible})",
    )

    # 2. the built partition is the intended one
    check(
        pairs["trip_id"].nunique() > 0 and partition == "train",
        f"pairs built from partition '{partition}' only",
    )
    train_ids = set(io.list_trips(DATA / "train"))
    test_ids = set(io.list_trips(DATA / "student_test"))
    present = set(pairs["trip_id"].unique())
    check(present <= train_ids, f"all {len(present)} trip ids belong to train/")
    check(
        not (present & test_ids),
        f"no student_test trip id leaked in ({len(present & test_ids)} found)",
    )

    # 3. trip ids intact for group-aware CV
    # pandas 3 gives string columns a dedicated str dtype rather than object,
    # so test for "is a string dtype" instead of a specific one.
    check(
        pd.api.types.is_string_dtype(pairs["trip_id"])
        and all(isinstance(t, str) and len(t) == 8 for t in present),
        "trip_id kept as 8-character zero-padded strings",
    )
    if limit:
        lines.append(f"  [SKIP] trip count check (--limit {limit} in effect, "
                     f"{len(present)} of {len(train_ids)} trips)")
    else:
        check(len(present) == len(train_ids),
              f"all {len(train_ids)} training trips present")

    # 4. candidates grouped correctly
    grp = pairs.groupby(["trip_id", "point_idx"])["candidate_rank"]
    ranks_ok = bool(
        (grp.min() == 0).all()
        and (grp.apply(lambda s: s.is_unique)).all()
        and (grp.max() == grp.count() - 1).all()
    )
    check(ranks_ok, "candidate_rank is 0..n-1 and unique within every (trip, point)")

    # 5. labels consistent with the alignment
    pos = pairs[pairs["label"] == 1]
    check(
        bool((pos["arc_id"] == pos["truth_arc_id"]).all()),
        "every positive row has arc_id == truth_arc_id",
    )
    per_point_pos = pairs.groupby(["trip_id", "point_idx"])["label"].sum()
    check(bool((per_point_pos <= 1).all()), "at most one positive per GPS point")
    check(
        bool((pairs.loc[~pairs["point_assigned"], "label"] == 0).all()),
        "unassigned points carry no positive label",
    )

    # 6. no duplicate candidate rows
    dups = int(pairs.duplicated(subset=["trip_id", "point_idx", "arc_id"]).sum())
    check(dups == 0, f"no duplicate (trip_id, point_idx, arc_id) rows ({dups} found)")

    # 7. feature block is clean
    nan_cols = [c for c in ft.FEATURE_COLUMNS if pairs[c].isna().any()]
    check(not nan_cols, f"no NaNs in feature columns ({nan_cols})")

    return lines


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--partition", default="train",
                        help="partition directory under data/ (default: train)")
    parser.add_argument("--K", type=int, default=cnd.DEFAULT_K,
                        help="max candidates per GPS point")
    parser.add_argument("--R", type=float, default=cnd.DEFAULT_R_M,
                        help="candidate search radius in metres")
    parser.add_argument("--max-assign-dist", type=float,
                        default=lb.DEFAULT_MAX_ASSIGN_DIST_M,
                        help="alignment distance beyond which a point is left unassigned")
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N trips (smoke runs)")
    parser.add_argument("--out", default=None,
                        help="parquet path (default: outputs/<partition>_pairs.parquet, "
                             "or outputs/training_pairs.parquet for train)")
    args = parser.parse_args(argv)

    if not (DATA / args.partition).is_dir():
        parser.error(f"no such partition: data/{args.partition}")

    OUT.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    pairs, miss, align = build_partition(
        args.partition, args.K, args.R, args.max_assign_dist, args.limit
    )
    elapsed = time.perf_counter() - t0

    if args.out:
        parquet_path = Path(args.out)
    elif args.partition == "train":
        parquet_path = OUT / "training_pairs.parquet"
    else:
        parquet_path = OUT / f"{args.partition}_pairs.parquet"

    pairs.to_parquet(parquet_path, index=False)
    suffix = "" if args.partition == "train" else f"_{args.partition}"
    miss_path = OUT / f"candidate_miss_rate{suffix}.csv"
    align_path = OUT / f"alignment_report{suffix}.csv"

    # Overall row: pooled across trips, not a mean of per-trip rates, so a
    # 19k-point trip is not weighted the same as a 262-point one.
    overall = {
        "trip_id": "ALL",
        "partition": args.partition,
        "n_points": int(miss["n_points"].sum()),
        "n_aligned": int(miss["n_aligned"].sum()),
        "n_with_candidates": int(miss["n_with_candidates"].sum()),
        "n_missed": int(miss["n_missed"].sum()),
        "miss_rate": float(miss["n_missed"].sum() / max(miss["n_aligned"].sum(), 1)),
        "n_no_candidates": int(miss["n_no_candidates"].sum()),
        "mean_rank_of_truth": float(
            np.average(miss["mean_rank_of_truth"], weights=miss["n_aligned"].clip(lower=1))
        ),
        "n_pairs": int(miss["n_pairs"].sum()),
        "n_positives": int(miss["n_positives"].sum()),
        "seconds": round(elapsed, 3),
    }
    pd.concat([miss, pd.DataFrame([overall])], ignore_index=True).to_csv(miss_path, index=False)
    align.to_csv(align_path, index=False)

    print(f"\nwrote {parquet_path}  ({len(pairs):,} rows x {len(pairs.columns)} cols)")
    print(f"wrote {miss_path}")
    print(f"wrote {align_path}")
    print(f"elapsed {elapsed:.1f}s for {miss.shape[0]} trips")

    print("\nvalidation")
    results = validate(pairs, miss, args.partition, args.limit)

    # 8. parquet round-trips
    back = pd.read_parquet(parquet_path)
    same = (
        back.shape == pairs.shape
        and list(back.columns) == list(pairs.columns)
        and pd.api.types.is_string_dtype(back["trip_id"])
        and int(back["label"].sum()) == int(pairs["label"].sum())
    )
    results.append(f"  [{'PASS' if same else 'FAIL'}] parquet reads back identical "
                   f"({back.shape[0]:,} rows, trip_id preserved as {back['trip_id'].iloc[0]!r})")
    print("\n".join(results))

    print(f"\ncandidate miss rate (pooled over {len(miss)} trips): "
          f"{overall['miss_rate']:.4f}  "
          f"({overall['n_missed']:,} of {overall['n_aligned']:,} aligned points)")
    print(f"positives: {overall['n_positives']:,} of {overall['n_pairs']:,} pairs "
          f"({100 * overall['n_positives'] / max(overall['n_pairs'], 1):.2f}%)")

    worst = miss.sort_values("miss_rate", ascending=False).head(5)
    print("\nworst trips by miss rate:")
    print(worst[["trip_id", "n_points", "n_aligned", "n_missed", "miss_rate"]]
          .to_string(index=False))

    return 0 if all("FAIL" not in line for line in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
