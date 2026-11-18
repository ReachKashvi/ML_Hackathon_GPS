"""Verify the untuned lgbm, then compare it against the HPO-tuned one.

Runs after ``16_bayesian_hpo.py``.

Step (a) -- verify the untuned baseline
---------------------------------------
``outputs/models/lgbm.joblib`` was fitted under LightGBM **4.7.0**, whose
binary later began crashing on every fit (a corrupted install, repaired by
downgrading to **4.6.0**). A model pickled by one version and scored by
another is not guaranteed to reproduce its own numbers, and the whole
tuned-vs-untuned comparison rests on that baseline being real.

So the baseline is re-scored under 4.6.0 and checked against the 0.8223
recorded in ``test_results.csv``. If it moved by more than ``--tolerance``
(default 0.005) the untuned model is **refitted from scratch** under 4.6.0 on
all 80 training trips, and both ``test_results.csv`` and
``predictions/lgbm.parquet`` are rewritten. Otherwise the existing numbers
stand. Either way the delta is reported rather than assumed away.

Step (b) -- score the tuned model
---------------------------------
Evaluates ``lgbm_tuned`` on the student-test pairs, writes its predictions
for the Phase 4 HMM, appends its row to ``test_results.csv`` and plots the
four-metric comparison.

Usage
-----
::

    python scripts/17_tuned_vs_untuned.py
    python scripts/17_tuned_vs_untuned.py --tolerance 0.005
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (str(SRC), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import eval_utils as ev  # noqa: E402
import features as ft  # noqa: E402
import io_utils as io  # noqa: E402
import model_zoo as mz  # noqa: E402
from eval_all_models import (  # noqa: E402
    GRID, INK, INK_MUTED, SERIES_1, SERIES_2, SURFACE, style_axes,
)

DATA = ROOT / "data"
OUT = ROOT / "outputs"
MODELS_DIR = OUT / "models"
PRED_DIR = OUT / "predictions"
FIG_DIR = OUT / "figures"


def score_model(model, pairs: pd.DataFrame, feature_cols: List[str],
                arcs_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Full metric set for one fitted model on the student-test pairs."""
    X = pairs[feature_cols].astype("float32")
    y = pairs["label"].to_numpy().astype(int)
    X_arr = np.ascontiguousarray(X.to_numpy())

    t0 = time.perf_counter()
    y_prob = model.predict_proba(X_arr)[:, 1]
    total_s = time.perf_counter() - t0

    preds = pairs[["trip_id", "point_idx", "arc_id"]].copy()
    preds["y_prob"] = y_prob
    preds["label"] = y

    pair_metrics = ev.per_pair_metrics(y, y_prob)
    top1 = ev.top1_accuracy(preds)
    cont = ev.route_continuity(preds, arcs_dict)

    # per-trip timing, as deployment would run it
    times = []
    for trip_id in pairs["trip_id"].unique():
        mask = (pairs["trip_id"] == trip_id).to_numpy()
        chunk = np.ascontiguousarray(X_arr[mask])
        t1 = time.perf_counter()
        model.predict_proba(chunk)
        times.append(time.perf_counter() - t1)

    # per-trip Top-1, so the mean-per-trip figure is available too
    per_trip = []
    for trip_id, grp in preds.groupby("trip_id"):
        per_trip.append(ev.top1_accuracy(grp)["top1_accuracy"])

    return {
        "preds": preds,
        "test_macro_f1": pair_metrics["macro_f1"],
        "test_accuracy": pair_metrics["accuracy"],
        "test_precision": pair_metrics["precision"],
        "test_recall": pair_metrics["recall"],
        "test_log_loss": pair_metrics["log_loss"],
        "test_top1": top1["top1_accuracy"],
        "test_top1_resolvable": top1["top1_accuracy_resolvable"],
        "test_top1_mean_per_trip": float(np.mean(per_trip)),
        "test_top1_std_per_trip": float(np.std(per_trip)),
        "test_candidate_miss_rate": top1["candidate_miss_rate"],
        "test_route_continuity": cont["continuity"],
        "test_strict_continuity": cont["strict_continuity"],
        "inference_sec_per_trip": float(np.mean(times)),
        "inference_sec_per_1k_pairs": float(1000.0 * total_s / max(len(pairs), 1)),
        "n_pairs": int(len(pairs)),
        "n_points": top1["n_points"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def upsert_results(row: Dict[str, Any], name: str, path: Path) -> None:
    """Replace this model's row in test_results.csv, keeping column order."""
    record = {"model": name, "family": mz.MODEL_FAMILIES.get("lgbm"),
              "key_config": row.pop("key_config", ""), **row}
    record.pop("preds", None)
    frame = pd.DataFrame([record])
    if path.exists():
        prior = pd.read_csv(path)
        prior = prior[prior["model"] != name]
        frame = pd.concat([prior, frame], ignore_index=True)
    frame.to_csv(path, index=False)


def improvement_figure(untuned: Dict[str, Any], tuned: Dict[str, Any],
                       path: Path) -> None:
    """Grouped bars: untuned vs tuned on the four headline metrics.

    Log loss is plotted as-is and labelled "lower is better" rather than
    inverted: a reader comparing this chart against log_loss_comparison.png
    should see the same number, not a transformed one.
    """
    metrics = [
        ("Top-1 accuracy", "test_top1", "higher better"),
        ("Macro F1", "test_macro_f1", "higher better"),
        ("Route continuity", "test_route_continuity", "higher better"),
        ("Log loss", "test_log_loss", "lower better"),
    ]
    x = np.arange(len(metrics))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 4.4), facecolor=SURFACE)
    u = [untuned[k] for _, k, _ in metrics]
    t = [tuned[k] for _, k, _ in metrics]

    ax.bar(x - w / 2 - 0.01, u, width=w, color=SERIES_1, label="lgbm (untuned)")
    ax.bar(x + w / 2 + 0.01, t, width=w, color=SERIES_2, label="lgbm_tuned (HPO)")
    for xi, (uv, tv) in enumerate(zip(u, t)):
        ax.text(xi - w / 2 - 0.01, uv + 0.012, f"{uv:.4f}", ha="center",
                fontsize=8, color=INK)
        ax.text(xi + w / 2 + 0.01, tv + 0.012, f"{tv:.4f}", ha="center",
                fontsize=8, color=INK)

    ax.set_xticks(x, [f"{n}\n({d})" for n, _, d in metrics], color=INK)
    ax.set_ylim(0, max(max(u), max(t)) * 1.18)
    style_axes(ax, xgrid=False)
    ax.set_title("Hyperparameter tuning: untuned vs tuned LightGBM",
                 color=INK, fontsize=12, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED, loc="upper right")
    delta = tuned["test_top1"] - untuned["test_top1"]
    fig.text(0.01, -0.03,
             f"Student-test Top-1 delta: {delta:+.4f}. "
             "Tuning searched CV Top-1 on training trips only; student test "
             "was never scored during the study.",
             fontsize=8, color=INK_MUTED)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tolerance", type=float, default=0.005,
                        help="|delta| above which the untuned model is refitted")
    parser.add_argument("--results", default=str(OUT / "test_results.csv"))
    args = parser.parse_args(argv)

    pairs = pd.read_parquet(OUT / "student_test_pairs.parquet")
    feature_cols = [c for c in ft.FEATURE_COLUMNS if c in pairs.columns]
    trip_ids = sorted(pairs["trip_id"].unique())
    arcs_dict = {t: io.load_arcs(DATA / "student_test" / t / f"{t}.arcs")
                 for t in trip_ids}
    results_path = Path(args.results)
    recorded = pd.read_csv(results_path).set_index("model")
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    import lightgbm
    print(f"lightgbm {lightgbm.__version__} | student test: {len(pairs):,} pairs, "
          f"{len(trip_ids)} trips\n")

    # ---- (a) verify the untuned baseline under the current LightGBM -------
    print("== (a) re-scoring the untuned lgbm under the current LightGBM ==")
    recorded_top1 = float(recorded.loc["lgbm", "test_top1"])
    untuned_bundle = joblib.load(MODELS_DIR / "lgbm.joblib")
    untuned_model = untuned_bundle["model"] if isinstance(untuned_bundle, dict) else untuned_bundle

    refit_reason = None
    try:
        untuned = score_model(untuned_model, pairs, feature_cols, arcs_dict)
        delta = untuned["test_top1"] - recorded_top1
        print(f"  recorded {recorded_top1:.4f} -> re-scored {untuned['test_top1']:.4f} "
              f"(delta {delta:+.4f}, tolerance +/-{args.tolerance})")
        if abs(delta) > args.tolerance:
            refit_reason = f"re-scored Top-1 moved by {delta:+.4f}"
    except Exception as exc:
        delta = float("nan")
        refit_reason = f"loading/scoring the saved model failed: {type(exc).__name__}"
        print(f"  {refit_reason}")

    if refit_reason:
        print(f"  refitting untuned lgbm on all 80 training trips ({refit_reason})")
        train = pd.read_parquet(OUT / "training_pairs.parquet")
        Xtr = np.ascontiguousarray(train[feature_cols].astype("float32").to_numpy())
        ytr = np.ascontiguousarray(train["label"].to_numpy().astype("int32"))
        spw = mz.compute_scale_pos_weight(ytr)
        untuned_model = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=63,
            scale_pos_weight=spw, random_state=mz.RANDOM_STATE, n_jobs=-1, verbose=-1,
        )
        t0 = time.perf_counter()
        untuned_model.fit(Xtr, ytr)
        print(f"  refit in {time.perf_counter() - t0:.1f}s")
        joblib.dump({"model": untuned_model, "feature_columns": feature_cols,
                     "trained_on": "outputs/training_pairs.parquet",
                     "n_rows": int(len(Xtr)), "refit_reason": refit_reason},
                    MODELS_DIR / "lgbm.joblib")
        untuned = score_model(untuned_model, pairs, feature_cols, arcs_dict)
        delta = untuned["test_top1"] - recorded_top1
        print(f"  after refit: Top-1 {untuned['test_top1']:.4f} (delta vs recorded {delta:+.4f})")
        untuned["preds"].to_parquet(PRED_DIR / "lgbm.parquet", index=False)
        row = {k: v for k, v in untuned.items() if k != "preds"}
        row["key_config"] = mz.key_configuration("lgbm")
        upsert_results(row, "lgbm", results_path)
        print(f"  updated {results_path.name} and predictions/lgbm.parquet")
    else:
        print("  within tolerance; existing results and predictions left unchanged")

    # ---- (b) score the tuned model ---------------------------------------
    print("\n== (b) scoring lgbm_tuned ==")
    params_path = OUT / "best_hpo_params.json"
    if not params_path.exists():
        print(f"  {params_path} not found -- has the HPO finished?")
        return 1
    best = json.loads(params_path.read_text(encoding="utf-8"))

    tuned_bundle = joblib.load(MODELS_DIR / "lgbm_tuned.joblib")
    tuned_model = tuned_bundle["model"]
    tuned = score_model(tuned_model, pairs, feature_cols, arcs_dict)
    tuned["preds"].to_parquet(PRED_DIR / "lgbm_tuned.parquet", index=False)

    row = {k: v for k, v in tuned.items() if k != "preds"}
    row["key_config"] = (
        f"HPO: {best['params'].get('n_estimators')} trees, "
        f"lr={best['params'].get('learning_rate'):.4f}, "
        f"leaves={best['params'].get('num_leaves')}, "
        f"depth={best['params'].get('max_depth')}"
    )
    upsert_results(row, "lgbm_tuned", results_path)
    improvement_figure(untuned, tuned, FIG_DIR / "hpo_improvement.png")

    # ---- report ----------------------------------------------------------
    top1_delta = tuned["test_top1"] - untuned["test_top1"]
    print(f"  predictions -> {PRED_DIR / 'lgbm_tuned.parquet'}")
    print(f"  figure      -> {FIG_DIR / 'hpo_improvement.png'}")
    print(f"  appended lgbm_tuned to {results_path.name}")

    print("\n== comparison ==")
    cmp = pd.DataFrame({
        "untuned": {k: untuned[k] for k in
                    ("test_top1", "test_top1_mean_per_trip", "test_macro_f1",
                     "test_log_loss", "test_route_continuity", "inference_sec_per_trip")},
        "tuned": {k: tuned[k] for k in
                  ("test_top1", "test_top1_mean_per_trip", "test_macro_f1",
                   "test_log_loss", "test_route_continuity", "inference_sec_per_trip")},
    })
    cmp["delta"] = cmp["tuned"] - cmp["untuned"]
    print(cmp.round(4).to_string())

    print(f"\nHPO best CV Top-1 : {best['best_mean_top1_cv']:.4f} "
          f"(trial {best['best_trial_number']}, "
          f"{best['n_trials_complete']} complete / {best['n_trials_pruned']} pruned)")
    print(f"student-test Top-1: untuned {untuned['test_top1']:.4f} -> "
          f"tuned {tuned['test_top1']:.4f}  (delta {top1_delta:+.4f})")
    verdict = "tuned" if top1_delta >= 0 else "untuned"
    print(f"\nRECOMMENDATION for Phase 4 HMM: use the **{verdict}** model "
          f"({'tuning held or improved' if top1_delta >= 0 else 'HPO regressed on the test split'}).")
    if abs(top1_delta) < 0.005:
        print("  NOTE: the delta is under 0.005 -- smaller than the fold-to-fold "
              "variation in CV, so treat the two as equivalent on accuracy and "
              "prefer whichever has the better log loss for HMM emissions:")
        better = "tuned" if tuned["test_log_loss"] < untuned["test_log_loss"] else "untuned"
        print(f"    better log loss: {better} "
              f"({min(tuned['test_log_loss'], untuned['test_log_loss']):.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
