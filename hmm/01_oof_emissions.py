"""Out-of-fold LightGBM emissions for all 80 training trips (HMM-improvement, step 1).

The Phase 4 HMM was tuned on out-of-fold emissions for 20 trips. Every
HMM-improvement experiment is scored on all 80 training trips instead, with
5-fold GroupKFold by trip: each trip's probabilities come from an lgbm
fitted on the other four folds only. The model config is the untuned one
chosen in Phase 2 (300 trees, lr 0.05, 63 leaves, scale_pos_weight per
fold). student_test is never read.

Usage::

    python hmm/01_oof_emissions.py --tag base
    python hmm/01_oof_emissions.py --tag feat --pairs outputs/hmm/training_pairs_feat.parquet --extra-features
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from common import OUT, ROOT, SEED, fold_of_trips
import eval_utils as eu
import features as ft
import model_zoo as mz

KEEP = ["trip_id", "point_idx", "arc_id", "candidate_rank", "frac_along_arc",
        "label", "truth_arc_id"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    ap.add_argument("--pairs", default=str(ROOT / "outputs" / "training_pairs.parquet"))
    ap.add_argument("--extra-features", action="store_true",
                    help="use every column starting with 'x_' as an extra feature")
    args = ap.parse_args()
    from lightgbm import LGBMClassifier

    pairs = pd.read_parquet(args.pairs)
    pairs["trip_id"] = pairs["trip_id"].astype(str)
    cols = list(ft.FEATURE_COLUMNS)
    if args.extra_features:
        cols += sorted(c for c in pairs.columns if c.startswith("x_"))
    assert "truth_arc_id" not in cols and "point_assigned" not in cols
    folds = fold_of_trips(pairs["trip_id"].unique())
    pairs["fold"] = pairs["trip_id"].map(folds).astype("int8")
    print(f"{len(pairs):,} pairs, {pairs.trip_id.nunique()} trips, {len(cols)} features", flush=True)

    X = np.ascontiguousarray(pairs[cols].to_numpy(dtype="float32"))
    y = pairs["label"].to_numpy().astype("int32")
    fold = pairs["fold"].to_numpy()
    out = pairs[KEEP + ["fold"]].copy()
    del pairs
    out["y_prob"] = np.nan
    rows, importances = [], []
    for k in range(5):
        tr, te = fold != k, fold == k
        t0 = time.perf_counter()
        model = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                               scale_pos_weight=mz.compute_scale_pos_weight(y[tr]),
                               random_state=SEED, n_jobs=-1, verbose=-1)
        model.fit(np.ascontiguousarray(X[tr]), y[tr])
        p = model.predict_proba(np.ascontiguousarray(X[te]))[:, 1]
        out.loc[te, "y_prob"] = p
        m = eu.per_pair_metrics(y[te], p)
        m.update(fold=k, fit_s=round(time.perf_counter() - t0, 1))
        rows.append(m)
        importances.append(pd.Series(model.booster_.feature_importance("gain"), index=cols, name=k))
        print(f"fold {k}: macro_f1 {m['macro_f1']:.4f} log_loss {m['log_loss']:.4f} ({m['fit_s']}s)", flush=True)

    out.to_parquet(OUT / f"oof_{args.tag}.parquet", index=False)
    pd.DataFrame(rows).to_csv(OUT / f"oof_{args.tag}_pair_metrics.csv", index=False)
    pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False) \
        .to_csv(OUT / f"oof_{args.tag}_importance.csv", header=["gain"])
    (OUT / f"oof_{args.tag}_meta.json").write_text(json.dumps(
        {"features": cols, "pairs": args.pairs, "folds": folds, "seed": SEED}, indent=1))
    print(f"wrote oof_{args.tag}.parquet", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
