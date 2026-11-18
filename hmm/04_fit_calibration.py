"""Fit emission calibration per fold, on the *other* folds only (HMM-improvement, E4).

LightGBM was trained with ``scale_pos_weight`` (~6), which shifts every
positive probability upwards. Its probabilities therefore rank well but are
biased, and the HMM adds ``beta * transition`` to ``log p`` as if ``p`` were
calibrated. Platt scaling, ``p' = sigmoid(a * logit(p) + b)``, removes both a
scale and a bias error (a pure temperature only fixes the scale).

For held-out fold k, ``(a, b)`` is fitted by logistic regression of the label
on ``logit(p)`` over the OOF pairs of the other four folds. Reliability
before/after is written for the report.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import OUT
import eval_utils as eu


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="base")
    args = ap.parse_args()
    oof = pd.read_parquet(OUT / f"oof_{args.oof}.parquet", columns=["fold", "label", "y_prob"])
    z = logit(oof.y_prob.to_numpy())
    y = oof.label.to_numpy()
    fold = oof.fold.to_numpy()
    params, rows = {}, []
    for k in sorted(np.unique(fold)):
        tr, te = fold != k, fold == k
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(z[tr, None], y[tr])
        a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        params[int(k)] = {"a": a, "b": b}
        p_cal = 1 / (1 + np.exp(-(a * z[te] + b)))
        before = eu.per_pair_metrics(y[te], oof.y_prob.to_numpy()[te])
        after = eu.per_pair_metrics(y[te], p_cal)
        rows.append(dict(fold=int(k), a=a, b=b, log_loss_before=before["log_loss"],
                         log_loss_after=after["log_loss"], macro_f1_before=before["macro_f1"],
                         macro_f1_after=after["macro_f1"]))
    df = pd.DataFrame(rows)
    print(df.round(4).to_string(index=False))
    (OUT / f"platt_{args.oof}.json").write_text(json.dumps(params, indent=1))
    df.to_csv(OUT / f"platt_{args.oof}_metrics.csv", index=False)

    # reliability table: predicted vs observed positive rate per probability decile
    bins = np.linspace(0, 1, 11)
    p = oof.y_prob.to_numpy()
    cal = np.empty_like(p)
    for k, ab in params.items():
        m = fold == k
        cal[m] = 1 / (1 + np.exp(-(ab["a"] * z[m] + ab["b"])))
    rel = []
    for name, q in (("raw", p), ("platt", cal)):
        idx = np.clip(np.digitize(q, bins) - 1, 0, 9)
        for i in range(10):
            m = idx == i
            if m.any():
                rel.append(dict(emission=name, bin_lo=bins[i], bin_hi=bins[i + 1], n=int(m.sum()),
                                mean_pred=float(q[m].mean()), observed=float(y[m].mean())))
    pd.DataFrame(rel).to_csv(OUT / f"reliability_{args.oof}.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
