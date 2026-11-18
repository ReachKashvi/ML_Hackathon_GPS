"""Rank the models and shortlist five for the report (Step 15).

Score
-----
::

    score = 0.50 * top1_accuracy
          + 0.20 * macro_f1
          + 0.20 * route_continuity
          + 0.10 * (1 - log_loss_normalised)

``log_loss`` is min-max normalised **across the models being ranked**, so the
best log loss contributes the full 0.10 and the worst contributes 0. The other
three terms are already on a 0-1 scale and are used as-is.

Top-1 carries half the weight because it is the only metric that measures the
actual task: picking the right arc for a GPS point. Macro F1 and continuity
check that the model is not winning Top-1 by luck, and log loss rewards
calibrated probabilities -- which matters because these scores feed an HMM in
Phase 3, where a badly calibrated emission probability does real damage.

What the ranking does and does not say
--------------------------------------
The score is computed on the **student-test** metrics. That split contains no
GPS-offset trip, so every model scores higher there than in CV. The ranking is
therefore a comparison *between models on equal footing*, not an estimate of
field accuracy. ``cv_top1_mean`` and its fold std are carried into the output
so a model whose CV and test numbers disagree is visible rather than hidden,
and :func:`stability_flags` marks any model whose test Top-1 sits more than two
CV standard deviations from its CV mean.

Usage
-----
::

    python scripts/select_top5.py
    python scripts/select_top5.py --top-n 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

OUT = ROOT / "outputs"

WEIGHTS = {
    "top1": 0.50,
    "macro_f1": 0.20,
    "route_continuity": 0.20,
    "log_loss": 0.10,
}


def minmax_normalise(values: pd.Series) -> pd.Series:
    """Scale to [0, 1]. A degenerate range maps everything to 0.5.

    Returning 0.5 rather than 0 keeps a tie neutral: if every model has the
    same log loss, that term should neither reward nor punish any of them.
    """
    lo, hi = float(values.min()), float(values.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(hi, lo):
        return pd.Series(np.full(len(values), 0.5), index=values.index)
    return (values - lo) / (hi - lo)


def score_models(results: pd.DataFrame) -> pd.DataFrame:
    """Add the weighted score and its components, ranked best first."""
    df = results.copy()
    df["log_loss_norm"] = minmax_normalise(df["test_log_loss"])

    df["term_top1"] = WEIGHTS["top1"] * df["test_top1"]
    df["term_macro_f1"] = WEIGHTS["macro_f1"] * df["test_macro_f1"]
    df["term_continuity"] = WEIGHTS["route_continuity"] * df["test_route_continuity"]
    df["term_log_loss"] = WEIGHTS["log_loss"] * (1.0 - df["log_loss_norm"])

    df["score"] = (
        df["term_top1"] + df["term_macro_f1"]
        + df["term_continuity"] + df["term_log_loss"]
    )
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def stability_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flag models whose test Top-1 is far from their CV Top-1.

    A gap beyond two CV fold standard deviations means the model behaves
    differently on held-out trips than the folds predicted -- worth knowing
    before committing tuning budget to it.
    """
    df = df.copy()
    if {"cv_top1_mean", "cv_top1_std"} <= set(df.columns):
        gap = df["test_top1"] - df["cv_top1_mean"]
        std = df["cv_top1_std"].replace(0, np.nan)
        df["cv_test_gap"] = gap
        df["cv_test_gap_in_stds"] = gap / std
        df["unstable"] = df["cv_test_gap_in_stds"].abs() > 2
    else:
        df["cv_test_gap"] = np.nan
        df["cv_test_gap_in_stds"] = np.nan
        df["unstable"] = False
    return df


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test-results", default=str(OUT / "test_results.csv"))
    parser.add_argument("--cv-results", default=str(OUT / "cv_results.csv"))
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--out", default=str(OUT / "top5_shortlist.csv"))
    args = parser.parse_args(argv)

    test_path = Path(args.test_results)
    if not test_path.is_file():
        parser.error(f"{test_path} not found; run Step 14 first")

    results = pd.read_csv(test_path)
    cv_path = Path(args.cv_results)
    if cv_path.is_file():
        cv = pd.read_csv(cv_path)
        keep = [c for c in ("model", "cv_macro_f1_mean", "cv_top1_mean",
                            "cv_top1_std", "cv_route_continuity_mean",
                            "cv_log_loss_mean", "fit_time_sec") if c in cv.columns]
        results = results.merge(cv[keep], on="model", how="left")

    ranked = stability_flags(score_models(results))

    print(f"ranked {len(ranked)} models on the weighted score")
    print("  score = 0.50*top1 + 0.20*macro_f1 + 0.20*continuity "
          "+ 0.10*(1 - log_loss_normalised)\n")

    show = ["rank", "model", "score", "test_top1", "test_macro_f1",
            "test_route_continuity", "test_log_loss", "log_loss_norm",
            "inference_sec_per_trip"]
    print(ranked[[c for c in show if c in ranked.columns]].round(4).to_string(index=False))

    print("\nscore decomposition:")
    terms = ["rank", "model", "term_top1", "term_macro_f1", "term_continuity",
             "term_log_loss", "score"]
    print(ranked[terms].round(4).to_string(index=False))

    if ranked["unstable"].any():
        print("\nCV/test disagreement beyond 2 CV standard deviations:")
        cols = ["model", "cv_top1_mean", "cv_top1_std", "test_top1",
                "cv_test_gap_in_stds"]
        print(ranked.loc[ranked["unstable"], [c for c in cols if c in ranked.columns]]
              .round(4).to_string(index=False))

    top_n = min(args.top_n, len(ranked))
    shortlist = ranked.head(top_n)
    shortlist.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")

    print(f"\n--- shortlist: top {top_n} for the report's main comparison ---")
    for _, r in shortlist.iterrows():
        print(f"  {int(r['rank'])}. {r['model']:8s} score {r['score']:.4f}  "
              f"(top1 {r['test_top1']:.4f}, macroF1 {r['test_macro_f1']:.4f}, "
              f"cont {r['test_route_continuity']:.4f}, "
              f"logloss {r['test_log_loss']:.4f}, "
              f"{r['inference_sec_per_trip']:.3f}s/trip)")

    best = shortlist.iloc[0]
    runner = shortlist.iloc[1] if len(shortlist) > 1 else None
    print(f"\n--- recommended for Phase 3 Bayesian tuning: {best['model']} ---")
    print(f"  score {best['score']:.4f}"
          + (f", ahead of {runner['model']} by {best['score'] - runner['score']:.4f}"
             if runner is not None else ""))
    if runner is not None and (best["score"] - runner["score"]) < 0.01:
        print(f"  NOTE: the margin over {runner['model']} is under 0.01 — treat the "
              f"top two as tied and let tuning cost decide.")
    if pd.notna(best.get("cv_top1_std")) and best["cv_top1_std"] > 0.05:
        print(f"  NOTE: CV Top-1 std is {best['cv_top1_std']:.4f} across folds, so "
              f"differences smaller than that between models are not meaningful.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
