"""Build candidate pairs for the 10 student_test trips (Step 12).

Runs the **identical** pipeline used for training -- same ``candidates.py``,
``features.py`` and ``labels.py``, same parameters -- by importing
``build_training_pairs.build_partition`` rather than reimplementing it. If the
two ever diverged, the models would be scored on features they were never
trained on, which is the quietest and most damaging kind of bug in this
project.

Writes:

* ``outputs/student_test_pairs.parquet``        same schema as training pairs
* ``outputs/candidate_miss_rate_student_test.csv``
* ``outputs/alignment_report_student_test.csv``

On leakage
----------
This script is evaluation-only. It never writes to ``training_pairs.parquet``
and never reads the evaluator's test data, which is not present under
``data/`` at all. Student-test labels are produced solely so the models can be
*scored*; nothing here is fitted, and :func:`validate_student_test` asserts
that no training trip id appears in the output.

Usage
-----
::

    python scripts/build_student_test_pairs.py
    python scripts/build_student_test_pairs.py --K 10 --R 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (str(SRC), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import candidates as cnd  # noqa: E402
import features as ft  # noqa: E402
import io_utils as io  # noqa: E402
import labels as lb  # noqa: E402
from build_training_pairs import OUTPUT_COLUMNS, build_partition  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "outputs"

PARTITION = "student_test"


def validate_student_test(pairs: pd.DataFrame, miss: pd.DataFrame) -> List[str]:
    """Validation checklist specific to the student-test split."""
    lines: List[str] = []

    def check(ok: bool, text: str) -> None:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {text}")

    train_ids = set(io.list_trips(DATA / "train"))
    test_ids = set(io.list_trips(DATA / PARTITION))
    present = set(pairs["trip_id"].unique())

    accessible = sorted(p.name for p in DATA.iterdir() if p.is_dir())
    check(
        set(accessible) <= {"train", PARTITION},
        f"only train/student_test partitions exist under data/ (found: {accessible})",
    )
    check(present <= test_ids, f"all {len(present)} trip ids belong to {PARTITION}/")
    check(not (present & train_ids), "no training trip id leaked into the test pairs")
    check(len(present) == len(test_ids), f"all {len(test_ids)} student-test trips present")
    check(
        pd.api.types.is_string_dtype(pairs["trip_id"])
        and all(isinstance(t, str) and len(t) == 8 for t in present),
        "trip_id kept as 8-character zero-padded strings",
    )

    # Schema must match the training table exactly, column for column.
    train_path = OUT / "training_pairs.parquet"
    if train_path.exists():
        train_cols = list(pd.read_parquet(train_path).columns)
        check(
            list(pairs.columns) == train_cols,
            "schema is identical to training_pairs.parquet",
        )
    else:
        lines.append("  [SKIP] schema comparison (training_pairs.parquet not found)")

    grp = pairs.groupby(["trip_id", "point_idx"])["candidate_rank"]
    check(
        bool((grp.min() == 0).all() and (grp.max() == grp.count() - 1).all()
             and grp.apply(lambda s: s.is_unique).all()),
        "candidate_rank is 0..n-1 and unique within every (trip, point)",
    )
    pos = pairs[pairs["label"] == 1]
    check(bool((pos["arc_id"] == pos["truth_arc_id"]).all()),
          "every positive row has arc_id == truth_arc_id")
    check(bool((pairs.groupby(["trip_id", "point_idx"])["label"].sum() <= 1).all()),
          "at most one positive per GPS point")
    dups = int(pairs.duplicated(subset=["trip_id", "point_idx", "arc_id"]).sum())
    check(dups == 0, f"no duplicate (trip_id, point_idx, arc_id) rows ({dups} found)")
    nan_cols = [c for c in ft.FEATURE_COLUMNS if pairs[c].isna().any()]
    check(not nan_cols, f"no NaNs in feature columns ({nan_cols})")

    return lines


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--K", type=int, default=cnd.DEFAULT_K,
                        help="max candidates per GPS point (must match training)")
    parser.add_argument("--R", type=float, default=cnd.DEFAULT_R_M,
                        help="candidate search radius in metres (must match training)")
    parser.add_argument("--max-assign-dist", type=float,
                        default=lb.DEFAULT_MAX_ASSIGN_DIST_M,
                        help="alignment distance beyond which a point is unassigned")
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N trips (smoke runs)")
    args = parser.parse_args(argv)

    OUT.mkdir(exist_ok=True)
    print(f"Pipeline parameters must match the training build: "
          f"K={args.K}, R={args.R} m, max_assign_dist={args.max_assign_dist} m")

    t0 = time.perf_counter()
    pairs, miss, align = build_partition(
        PARTITION, args.K, args.R, args.max_assign_dist, args.limit
    )
    elapsed = time.perf_counter() - t0

    parquet_path = OUT / "student_test_pairs.parquet"
    miss_path = OUT / f"candidate_miss_rate_{PARTITION}.csv"
    align_path = OUT / f"alignment_report_{PARTITION}.csv"

    pairs = pairs[list(OUTPUT_COLUMNS)]
    pairs.to_parquet(parquet_path, index=False)

    overall = {
        "trip_id": "ALL",
        "partition": PARTITION,
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
    print(f"elapsed {elapsed:.1f}s for {len(miss)} trips")

    print("\nvalidation")
    results = validate_student_test(pairs, miss)
    back = pd.read_parquet(parquet_path)
    ok = (
        back.shape == pairs.shape
        and list(back.columns) == list(pairs.columns)
        and int(back["label"].sum()) == int(pairs["label"].sum())
    )
    results.append(f"  [{'PASS' if ok else 'FAIL'}] parquet reads back identical "
                   f"({back.shape[0]:,} rows)")
    print("\n".join(results))

    print(f"\ncandidate miss rate (pooled over {len(miss)} trips): "
          f"{overall['miss_rate']:.4f} "
          f"({overall['n_missed']:,} of {overall['n_aligned']:,} aligned points)")
    print(f"positives: {overall['n_positives']:,} of {overall['n_pairs']:,} pairs "
          f"({100 * overall['n_positives'] / max(overall['n_pairs'], 1):.2f}%)")
    print("\nper trip:")
    print(miss[["trip_id", "n_points", "n_aligned", "n_missed", "miss_rate",
                "n_pairs", "n_positives"]].to_string(index=False))

    return 0 if all("FAIL" not in line for line in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
