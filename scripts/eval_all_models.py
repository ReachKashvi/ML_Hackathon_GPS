"""Evaluate every fitted model on the student-test pairs (Step 14).

For each model saved by Step 13:

1. ``predict_proba`` on ``outputs/student_test_pairs.parquet``.
2. Write ``outputs/predictions/<name>.parquet`` with
   ``[trip_id, point_idx, arc_id, y_prob, label]`` -- the input the HMM stage
   will consume later.
3. Score per-pair metrics, top-1 accuracy, route continuity and inference time
   per trip.
4. Append to ``outputs/test_results.csv``.

Then render the report figures into ``outputs/figures/`` and write
``outputs/comparison_table.md``.

Reading the CV-vs-test chart
----------------------------
Student-test Top-1 is expected to come out **above** CV Top-1 for every model,
and that gap is a property of the split, not evidence of good generalisation:
the CV folds contain training trips ``00000054`` and ``00000033``, whose GPS
tracks sit 100-250 m off their own road network (pooled training candidate
miss rate 2.11%), while the student-test split has no such trip (miss rate
0.17%). The chart therefore also plots ``top1_resolvable`` -- accuracy over
points whose true arc is actually in the candidate set -- which removes that
ceiling difference and is the fairer generalisation comparison.

Usage
-----
::

    python scripts/eval_all_models.py
    python scripts/eval_all_models.py --models lgbm rforest
"""

from __future__ import annotations

import argparse
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import eval_utils as ev  # noqa: E402
import features as ft  # noqa: E402
import io_utils as io  # noqa: E402
import model_zoo as mz  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "outputs"
MODELS_DIR = OUT / "models"
PRED_DIR = OUT / "predictions"
FIG_DIR = OUT / "figures"
CONF_DIR = FIG_DIR / "confusion"

# --- palette -------------------------------------------------------------
# Validated categorical slots 1 and 2 (blue, orange): adjacent CVD dE 24.7,
# normal-vision dE 33.6, both >= 3:1 on the light surface. Single-series charts
# use slot 1 only. Text never wears a series colour.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
GRID = "#deddd8"
# Sequential blue ramp, light -> dark, for the confusion heatmaps.
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUES = LinearSegmentedColormap.from_list("pathsnap_blues", SEQ_BLUE)

TEST_RESULT_COLUMNS = (
    "model", "family", "key_config",
    "test_macro_f1", "test_accuracy", "test_precision", "test_recall",
    "test_log_loss", "test_top1", "test_top1_resolvable",
    "test_candidate_miss_rate", "test_route_continuity", "test_strict_continuity",
    "inference_sec_per_trip", "inference_sec_per_1k_pairs",
    "n_pairs", "n_points", "timestamp",
)


def style_axes(ax: plt.Axes, xgrid: bool = True) -> None:
    """Recessive grid and axes; data stays the loudest thing on the page."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    ax.grid(axis="x" if xgrid else "y", color=GRID, lw=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def hbar_chart(
    frame: pd.DataFrame,
    value_col: str,
    title: str,
    xlabel: str,
    path: Path,
    fmt: str = "{:.3f}",
    ascending: bool = False,
    log_x: bool = False,
) -> None:
    """Single-series horizontal bar chart: magnitude ranked by model.

    One series, so no legend -- the title names the measure. Every bar is
    directly labelled: there are few enough of them that selective labelling
    would just make the reader hunt.
    """
    data = frame.sort_values(value_col, ascending=ascending).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 0.46 * len(data) + 1.7), facecolor=SURFACE)
    y = np.arange(len(data))
    # height < 1 leaves a surface gap between adjacent bars
    ax.barh(y, data[value_col], height=0.68, color=SERIES_1)
    ax.set_yticks(y, data["model"], color=INK)
    ax.invert_yaxis()
    if log_x:
        ax.set_xscale("log")
    style_axes(ax)

    span = data[value_col].max() if len(data) else 1.0
    for yi, v in zip(y, data[value_col]):
        ax.text(v + (span * 0.015 if not log_x else v * 0.08), yi, fmt.format(v),
                va="center", ha="left", fontsize=9, color=INK)
    ax.set_xlim(right=(span * 1.18) if not log_x else None)
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def cv_vs_test_chart(merged: pd.DataFrame, path: Path) -> None:
    """Grouped bars: CV Top-1 (with std error bars) against student-test Top-1.

    Two series, so a legend is present; both are also direct-labelled, so
    identity never rests on colour alone.
    """
    data = merged.sort_values("cv_top1_mean", ascending=False).reset_index(drop=True)
    x = np.arange(len(data))
    w = 0.38
    fig, ax = plt.subplots(figsize=(1.35 * len(data) + 3.2, 5), facecolor=SURFACE)

    ax.bar(x - w / 2 - 0.01, data["cv_top1_mean"], width=w, color=SERIES_1,
           yerr=data["cv_top1_std"], capsize=3,
           error_kw={"ecolor": INK_MUTED, "lw": 1.1}, label="CV (5-fold, by trip)")
    ax.bar(x + w / 2 + 0.01, data["test_top1"], width=w, color=SERIES_2,
           label="Student test")

    for xi, (cv, te) in enumerate(zip(data["cv_top1_mean"], data["test_top1"])):
        ax.text(xi - w / 2 - 0.01, cv + 0.012, f"{cv:.3f}", ha="center",
                fontsize=8, color=INK)
        ax.text(xi + w / 2 + 0.01, te + 0.012, f"{te:.3f}", ha="center",
                fontsize=8, color=INK)

    ax.set_xticks(x, data["model"], color=INK)
    ax.set_ylim(0, min(1.05, max(data["test_top1"].max(), data["cv_top1_mean"].max()) + 0.14))
    style_axes(ax, xgrid=False)
    ax.set_ylabel("Top-1 accuracy", color=INK_MUTED, fontsize=9)
    ax.set_title("Top-1 accuracy: cross-validation vs student test",
                 color=INK, fontsize=12, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED, loc="upper right")
    fig.text(0.01, -0.02,
             "Student test runs higher partly because it contains no "
             "GPS-offset trip: its candidate miss rate is 0.17% vs 2.11% in CV.",
             fontsize=8, color=INK_MUTED)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def confusion_figure(name: str, y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    """2x2 confusion matrix as a sequential-blue heatmap with counts."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.1, 3.7), facecolor=SURFACE)
    norm = cm / max(cm.sum(), 1)
    ax.imshow(norm, cmap=BLUES, vmin=0, vmax=norm.max() if norm.max() else 1)

    for i in range(2):
        for j in range(2):
            # ink on light cells, surface on dark ones, so text always reads
            dark = norm[i, j] > 0.55 * (norm.max() if norm.max() else 1)
            ax.text(j, i, f"{cm[i, j]:,}\n{100 * norm[i, j]:.1f}%",
                    ha="center", va="center", fontsize=10,
                    color=SURFACE if dark else INK)
    ax.set_xticks([0, 1], ["pred 0", "pred 1"], color=INK)
    ax.set_yticks([0, 1], ["true 0", "true 1"], color=INK)
    ax.tick_params(length=0, labelsize=9)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(f"{name} — student test", color=INK, fontsize=11, loc="left", pad=10)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def evaluate_model(
    name: str,
    pairs: pd.DataFrame,
    arcs_dict: Dict[str, pd.DataFrame],
    save: bool = True,
) -> Dict[str, Any] | None:
    """Score one saved model on the student-test pairs."""
    model_path = MODELS_DIR / f"{name}.joblib"
    if not model_path.is_file():
        print(f"  [SKIP] {name}: no fitted model at {model_path}")
        return None

    bundle = joblib.load(model_path)
    model = bundle["model"] if isinstance(bundle, dict) else bundle
    feature_cols = (
        bundle.get("feature_columns") if isinstance(bundle, dict) else None
    ) or [c for c in ft.FEATURE_COLUMNS if c in pairs.columns]

    missing = [c for c in feature_cols if c not in pairs.columns]
    if missing:
        raise KeyError(f"{name}: student-test pairs lack feature columns {missing}")

    X = pairs[feature_cols]
    y = pairs["label"].to_numpy().astype(int)

    t0 = time.perf_counter()
    y_prob = model.predict_proba(X)[:, 1]
    total_s = time.perf_counter() - t0

    preds = pairs[["trip_id", "point_idx", "arc_id"]].copy()
    preds["y_prob"] = y_prob
    preds["label"] = y
    if save:
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(PRED_DIR / f"{name}.parquet", index=False)

    pair_metrics = ev.per_pair_metrics(y, y_prob)
    top1 = ev.top1_accuracy(preds)
    cont = ev.route_continuity(preds, arcs_dict)
    timing = ev.inference_time_per_trip(model, X, pairs["trip_id"].to_numpy())

    if save:
        CONF_DIR.mkdir(parents=True, exist_ok=True)
        confusion_figure(name, y, (y_prob >= 0.5).astype(int),
                         CONF_DIR / f"{name}.png")

    print(f"  {name:8s} macroF1 {pair_metrics['macro_f1']:.4f} | "
          f"top1 {top1['top1_accuracy']:.4f} | cont {cont['continuity']:.4f} | "
          f"logloss {pair_metrics['log_loss']:.4f} | "
          f"{timing['mean_seconds_per_trip']:.3f}s/trip ({total_s:.1f}s total)")

    return {
        "model": name,
        "family": mz.MODEL_FAMILIES.get(name, "?"),
        "key_config": mz.key_configuration(name) if name in mz.ALL_MODELS else "?",
        "test_macro_f1": pair_metrics["macro_f1"],
        "test_accuracy": pair_metrics["accuracy"],
        "test_precision": pair_metrics["precision"],
        "test_recall": pair_metrics["recall"],
        "test_log_loss": pair_metrics["log_loss"],
        "test_top1": top1["top1_accuracy"],
        "test_top1_resolvable": top1["top1_accuracy_resolvable"],
        "test_candidate_miss_rate": top1["candidate_miss_rate"],
        "test_route_continuity": cont["continuity"],
        "test_strict_continuity": cont["strict_continuity"],
        "inference_sec_per_trip": timing["mean_seconds_per_trip"],
        "inference_sec_per_1k_pairs": timing["seconds_per_1k_pairs"],
        "n_pairs": int(len(pairs)),
        "n_points": top1["n_points"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


#: Models whose CV *fit* set was capped, and the row budget used. Every model
#: is still scored on complete held-out folds, so the metrics are comparable;
#: only the reference data these models learned from was reduced. Recorded
#: here so the comparison table cannot be read as a like-for-like contest.
TRAINING_CAPS: Dict[str, int] = {"knn": 100_000}


def write_comparison_table(merged: pd.DataFrame, path: Path) -> None:
    """Markdown table in the column order the report brief requires."""
    order = merged.sort_values("test_top1", ascending=False)
    header = (
        "| Model | Algorithm Family | Key Configuration | CV Macro F1 | "
        "Top-1 Accuracy | Route Continuity | Student-Test Macro F1 | Log Loss | "
        "Inference Time |"
    )
    sep = "|" + "---|" * 9
    lines = [
        "# PathSnap — model comparison (Section 3.1)",
        "",
        f"Generated {datetime.now().isoformat(timespec='seconds')} from "
        f"`outputs/cv_results.csv` and `outputs/test_results.csv`.",
        "",
        "**Top-1 Accuracy**, **Route Continuity** and **Log Loss** are measured on the "
        "student-test split; **CV Macro F1** is the 5-fold GroupKFold mean (± std) over "
        "the 80 training trips. **Inference Time** is mean wall-clock `predict_proba` "
        "seconds per trip.",
        "",
        header,
        sep,
    ]
    for _, r in order.iterrows():
        cv_f1 = (
            f"{r['cv_macro_f1_mean']:.4f} ± {r['cv_macro_f1_std']:.4f}"
            if pd.notna(r.get("cv_macro_f1_mean")) else "—"
        )
        name = r["model"] + (" †" if r["model"] in TRAINING_CAPS else "")
        config = r["key_config"]
        if r["model"] in TRAINING_CAPS:
            config += f" — **trained on {TRAINING_CAPS[r['model']]:,} rows only †**"
        lines.append(
            f"| {name} | {r['family']} | {config} | {cv_f1} | "
            f"{r['test_top1']:.4f} | {r['test_route_continuity']:.4f} | "
            f"{r['test_macro_f1']:.4f} | {r['test_log_loss']:.4f} | "
            f"{r['inference_sec_per_trip']:.3f} s/trip |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- **Route Continuity** counts \"stayed on the same arc\" as connected. "
        "The literal `arc[i].to_node == arc[i+1].from_node` rule scores a *perfect* "
        "map match at 0.097, because 88% of consecutive 1 Hz fixes fall on one arc; "
        "the strict figure is in `test_strict_continuity` for reference.",
        "- **Top-1 Accuracy** counts candidate misses as incorrect. "
        "`test_top1_resolvable` excludes them (student-test miss rate: 0.17%).",
        "- CV and student test are not equally hard: the CV folds include trips "
        "`00000054` and `00000033`, whose GPS tracks are 100-250 m off their own "
        "road network. Student test contains no such trip.",
    ]
    if any(m in TRAINING_CAPS for m in order["model"].values):
        capped = ", ".join(
            f"`{m}` ({n:,} rows)" for m, n in TRAINING_CAPS.items()
            if m in set(order["model"].values)
        )
        lines += [
            "",
            f"> **† Training-set cap.** {capped} was fitted on a reduced training "
            "set, sampled as **whole trips** from each CV fold, against the full "
            "~983,000 rows per fold used by every other model. Its cross-validated "
            "scores therefore understate what it could achieve with the full data "
            "and must not be read as a like-for-like comparison. Validation folds "
            "were complete for every model, so the *held-out* data is identical "
            "throughout.",
            "",
            "> The cap exists because k-NN's cost is at query time, not fit time: "
            "one fold took roughly 20 minutes at full width versus about 40 seconds "
            "at this cap. That cost is itself the finding — k-NN is ~500x slower "
            "than every other model at prediction and is not viable for "
            "production map matching regardless of its accuracy.",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", default=str(OUT / "student_test_pairs.parquet"))
    parser.add_argument("--cv-results", default=str(OUT / "cv_results.csv"))
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--out", default=str(OUT / "test_results.csv"))
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    pairs = pd.read_parquet(args.pairs)
    trip_ids = sorted(pairs["trip_id"].unique())
    arcs_dict = {
        t: io.load_arcs(DATA / "student_test" / t / f"{t}.arcs") for t in trip_ids
    }
    print(f"student test: {len(pairs):,} pairs, {len(trip_ids)} trips, "
          f"positive rate {pairs['label'].mean():.4f}\n")

    names = args.models or sorted(
        p.stem for p in MODELS_DIR.glob("*.joblib")
    ) if MODELS_DIR.is_dir() else []
    if not names:
        parser.error(f"no fitted models found in {MODELS_DIR}; run Step 13 first")

    rows = [
        r for r in (
            evaluate_model(n, pairs, arcs_dict, save=not args.no_save) for n in names
        ) if r is not None
    ]
    if not rows:
        parser.error("no model could be evaluated")

    results = pd.DataFrame(rows)[list(TEST_RESULT_COLUMNS)]
    if not args.no_save:
        out_path = Path(args.out)
        if out_path.exists():
            prior = pd.read_csv(out_path)
            prior = prior[~prior["model"].isin(results["model"])]
            results = pd.concat([prior, results], ignore_index=True)
        results.to_csv(out_path, index=False)
        print(f"\nwrote {out_path}")

    # --- figures ---------------------------------------------------------
    cv_path = Path(args.cv_results)
    cv = pd.read_csv(cv_path) if cv_path.exists() else pd.DataFrame()
    merged = results.merge(cv, on="model", how="left", suffixes=("", "_cv"))
    if "family_cv" in merged.columns:
        merged["family"] = merged["family"].fillna(merged["family_cv"])

    if not args.no_save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        if "cv_top1_mean" in merged.columns and merged["cv_top1_mean"].notna().any():
            cv_vs_test_chart(merged.dropna(subset=["cv_top1_mean"]),
                             FIG_DIR / "cv_vs_test_top1.png")
        else:
            print("  [SKIP] cv_vs_test_top1.png (no CV results to compare against)")

        hbar_chart(results, "test_macro_f1", "Student-test Macro F1",
                   "macro F1 (higher is better)", FIG_DIR / "macro_f1_comparison.png")
        hbar_chart(results, "test_log_loss", "Student-test log loss",
                   "log loss (lower is better)", FIG_DIR / "log_loss_comparison.png",
                   fmt="{:.4f}", ascending=True)
        hbar_chart(results, "test_route_continuity", "Student-test route continuity",
                   "fraction of connected transitions (higher is better)",
                   FIG_DIR / "route_continuity_comparison.png")
        hbar_chart(results, "inference_sec_per_trip", "Inference time per trip",
                   "seconds per trip, log scale (lower is better)",
                   FIG_DIR / "inference_time.png", fmt="{:.3f}s",
                   ascending=True, log_x=True)
        write_comparison_table(merged, OUT / "comparison_table.md")
        print(f"wrote {OUT / 'comparison_table.md'} and figures in {FIG_DIR}")

    show = ["model", "test_macro_f1", "test_top1", "test_top1_resolvable",
            "test_route_continuity", "test_log_loss", "inference_sec_per_trip"]
    print("\n" + results[show].sort_values("test_top1", ascending=False)
          .round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
