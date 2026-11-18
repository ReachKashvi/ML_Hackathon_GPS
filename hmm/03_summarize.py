"""Turn ``runs_<oof>.csv`` into the HMM-improvement experiment comparison tables.

Two tables:

``configs_<oof>.csv``
    One row per decoded config: pooled and mean-per-trip Top-1, mean/std of
    the per-fold pooled Top-1, continuity, decoded-pair macro F1, error types,
    trips improved/regressed against the Phase 4 baseline (``A_s20_b16``),
    decode time.

``experiment_comparison_<oof>.csv``
    One row per experiment *family*, scored by **nested** selection: for
    each fold the family's best config is chosen on the other four folds and
    scored on the held-out one. That is the number to compare families on;
    picking each family's best cell on all 80 trips would flatter families
    with bigger grids.

Every Top-1 uses the project definition: correct / points with >= 1
candidate, unassigned or missed truth counting as wrong.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import OUT

BASELINE = "A_s20_b16"


def macro_f1(tp, fp, fn, tn) -> float:
    f1_pos = 2 * tp / max(2 * tp + fp + fn, 1)
    f1_neg = 2 * tn / max(2 * tn + fn + fp, 1)
    return (f1_pos + f1_neg) / 2


def per_config(runs: pd.DataFrame) -> pd.DataFrame:
    base = runs[runs.config == BASELINE].set_index("trip_id")
    base_acc = base.n_correct / base.n_points
    rows = []
    for c, g in runs.groupby("config"):
        s = g.sum(numeric_only=True)
        acc = (g.set_index("trip_id").n_correct / g.set_index("trip_id").n_points)
        delta = (acc - base_acc.reindex(acc.index)).dropna()
        folds = g.groupby("fold").apply(lambda f: f.n_correct.sum() / f.n_points.sum(),
                                        include_groups=False)
        rows.append(dict(
            config=c, family=g.family.iloc[0], n_trips=g.trip_id.nunique(),
            top1_pooled=s.n_correct / s.n_points,
            top1_mean_trip=float(acc.mean()),
            fold_mean=float(folds.mean()), fold_std=float(folds.std(ddof=1)),
            **{f"fold{k}": float(v) for k, v in folds.items()},
            continuity=s.n_conn / s.n_trans,
            macro_f1_decoded=macro_f1(s.tp, s.fp, s.fn, s.tn),
            err_twin=int(s.err_twin), err_adjacent=int(s.err_adjacent),
            err_other=int(s.err_other),
            trips_improved=int((delta > 1e-9).sum()), trips_regressed=int((delta < -1e-9).sum()),
            decode_s_total=float(s.decode_s),
        ))
    return pd.DataFrame(rows).sort_values("top1_pooled", ascending=False)


def nested(runs: pd.DataFrame, cfg: pd.DataFrame) -> pd.DataFrame:
    folds = sorted(runs.fold.unique())
    by = runs.groupby(["config", "fold"])[["n_correct", "n_points"]].sum()
    fold_acc = (by.n_correct / by.n_points).unstack("fold")
    tot = runs.groupby(["config", "fold"])[["n_correct", "n_points"]].sum()
    rows = []
    fams = list(cfg.groupby("family").groups.items())
    fams.append(("A_base_fixed_s20_b16", [BASELINE]))
    for fam, members in fams:
        members = list(cfg.loc[members, "config"]) if fam != "A_base_fixed_s20_b16" else members
        chosen, held = [], []
        for k in folds:
            others = [f for f in folds if f != k]
            inner = {c: tot.loc[c].loc[others].n_correct.sum() / tot.loc[c].loc[others].n_points.sum()
                     for c in members}
            best = max(inner, key=inner.get)
            chosen.append(best)
            held.append(float(fold_acc.loc[best, k]))
        base = [float(fold_acc.loc[BASELINE, k]) for k in folds]
        diff = np.array(held) - np.array(base)
        rows.append(dict(family=fam, nested_fold_mean=float(np.mean(held)),
                         nested_fold_std=float(np.std(held, ddof=1)),
                         delta_vs_baseline_mean=float(diff.mean()),
                         folds_better=int((diff > 1e-9).sum()), folds_worse=int((diff < -1e-9).sum()),
                         chosen_per_fold=json.dumps(chosen),
                         **{f"fold{k}": v for k, v in zip(folds, held)}))
    return pd.DataFrame(rows).sort_values("nested_fold_mean", ascending=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="base")
    args = ap.parse_args()
    runs = pd.read_csv(OUT / f"runs_{args.oof}.csv", dtype={"trip_id": str})
    n_trips = runs.groupby("config").trip_id.nunique()
    complete = n_trips[n_trips == n_trips.max()].index
    runs = runs[runs.config.isin(complete)]
    cfg = per_config(runs).reset_index(drop=True)

    # classifier argmax reference row, same denominator
    b = runs[runs.config == BASELINE]
    cls_folds = b.groupby("fold").apply(lambda f: f.cls_correct.sum() / f.n_points.sum(),
                                        include_groups=False)
    cls_row = dict(config="classifier_argmax", family="classifier", n_trips=len(b),
                   top1_pooled=b.cls_correct.sum() / b.n_points.sum(),
                   top1_mean_trip=float((b.cls_correct / b.n_points).mean()),
                   fold_mean=float(cls_folds.mean()), fold_std=float(cls_folds.std(ddof=1)))
    cfg = pd.concat([pd.DataFrame([cls_row]), cfg], ignore_index=True)

    # candidate recall (identical for every config of one emission set)
    oof = pd.read_parquet(OUT / f"oof_{args.oof}.parquet",
                          columns=["trip_id", "point_idx", "label"])
    per_pt = oof.groupby(["trip_id", "point_idx"]).label.max()
    cfg["candidate_recall"] = float(per_pt.mean())

    pm = pd.read_csv(OUT / f"oof_{args.oof}_pair_metrics.csv")
    cfg["emission_macro_f1_mean"] = pm.macro_f1.mean()
    cfg["emission_log_loss_mean"] = pm.log_loss.mean()

    cfg.to_csv(OUT / f"configs_{args.oof}.csv", index=False)
    nest = nested(runs, cfg[cfg.family != "classifier"].set_index("config", drop=False))
    nest.to_csv(OUT / f"experiment_comparison_{args.oof}.csv", index=False)

    pd.set_option("display.width", 250)
    show = ["config", "top1_pooled", "top1_mean_trip", "fold_mean", "fold_std", "continuity",
            "macro_f1_decoded", "err_twin", "err_adjacent", "err_other",
            "trips_improved", "trips_regressed", "decode_s_total"]
    print(cfg[show].head(25).round(4).to_string(index=False))
    print()
    print(nest.drop(columns=["chosen_per_fold"]).round(4).to_string(index=False))
    print()
    for _, r in nest.iterrows():
        print(f"{r.family}: {r.chosen_per_fold}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
