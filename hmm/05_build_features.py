"""Build the E5 context-feature parquets (HMM improvement, E5).

The features themselves are defined in ``context_features.py`` (shared with
the API). This script appends them to the existing training / student_test
pair tables.

Usage::

    python hmm/05_build_features.py --partition train
    python hmm/05_build_features.py --partition student_test
"""

from __future__ import annotations

import argparse

import pandas as pd

from common import DATA, OUT, ROOT
from context_features import add_context_features
import io_utils as io


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", choices=["train", "student_test"], required=True)
    args = ap.parse_args()
    src = {"train": "training_pairs.parquet", "student_test": "student_test_pairs.parquet"}
    pairs = pd.read_parquet(ROOT / "outputs" / src[args.partition])
    pairs["trip_id"] = pairs["trip_id"].astype(str)
    parts = []
    for k, (tid, g) in enumerate(pairs.groupby("trip_id", sort=True), 1):
        trip = io.load_trip(DATA / args.partition / tid)
        parts.append(add_context_features(g, trip))
        if k % 10 == 0:
            print(f"{k} trips", flush=True)
    out = pd.concat(parts, ignore_index=True)
    xcols = [c for c in out.columns if c.startswith("x_")]
    assert not out[xcols].isna().any().any(), "NaN in new features"
    out[xcols] = out[xcols].astype("float32")
    tag = {"train": "training_pairs_feat.parquet", "student_test": "student_test_pairs_feat.parquet"}
    out.to_parquet(OUT / tag[args.partition], index=False)
    print(f"wrote {tag[args.partition]}: {len(out):,} rows, new features {xcols}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
