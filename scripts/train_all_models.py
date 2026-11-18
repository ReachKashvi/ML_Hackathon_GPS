"""Train and cross-validate the whole model zoo (Step 13).

For every model in ``model_zoo.ALL_MODELS``:

1. 5-fold **GroupKFold by trip_id** -- fit on the training folds, score the
   held-out trips, record per-pair metrics, top-1 accuracy, route continuity
   and fit time.
2. Aggregate across folds as mean +/- std.
3. Refit on **all** training pairs and save to ``outputs/models/<name>.joblib``
   for the Step 14 student-test evaluation.
4. Append a row to ``outputs/cv_results.csv``.

A full log, including the resolved configuration and per-fold timings, goes to
``outputs/logs/train_<timestamp>.log``.

Feature selection and leakage
-----------------------------
Features are taken from ``features.FEATURE_COLUMNS``, which **includes**
``candidate_rank`` as a predictor and excludes ``arc_id``.

Three columns in the parquet are deliberately **not** features:

``truth_arc_id``
    The aligned ground-truth arc. Using it would hand the model the answer --
    ``label`` is literally ``arc_id == truth_arc_id``.
``point_assigned``
    Derived from the same alignment, so it leaks which points have a positive.
``trip_id`` / ``point_idx``
    Identifiers, and ``trip_id`` is the CV grouping key.

The spec's phrasing ("everything except trip_id, point_idx, arc_id,
candidate_rank, label") predates those two alignment columns existing; taking
it literally would leak the label outright.

Usage
-----
::

    python scripts/train_all_models.py
    python scripts/train_all_models.py --models lgbm rforest
    python scripts/train_all_models.py --limit-trips 12 --no-save   # smoke run
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

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
LOGS_DIR = OUT / "logs"

#: Columns that identify a row or encode ground truth; never predictors.
NON_FEATURE_COLUMNS = (
    "trip_id", "point_idx", "arc_id", "truth_arc_id", "point_assigned", "label"
)

CV_RESULT_COLUMNS = (
    "model", "family", "key_config", "scaling",
    "cv_macro_f1_mean", "cv_macro_f1_std",
    "cv_top1_mean", "cv_top1_std",
    "cv_top1_resolvable_mean",
    "cv_route_continuity_mean", "cv_route_continuity_std",
    "cv_strict_continuity_mean",
    "cv_log_loss_mean", "cv_log_loss_std",
    "cv_accuracy_mean", "cv_precision_mean", "cv_recall_mean",
    "cv_candidate_miss_rate_mean",
    "fit_time_sec", "fit_time_sec_std", "score_time_sec",
    "final_fit_time_sec", "n_folds", "n_train_rows", "n_features", "timestamp",
)


def write_results(new_rows: pd.DataFrame, path: Path) -> None:
    """Merge rows into ``cv_results.csv`` by model name, writing immediately.

    Called after each model finishes rather than once at the end, so an
    interrupted sweep keeps everything already computed.
    """
    rows = new_rows[[c for c in CV_RESULT_COLUMNS if c in new_rows.columns]]
    if path.exists():
        prior = pd.read_csv(path)
        prior = prior[~prior["model"].isin(rows["model"])]
        rows = pd.concat([prior, rows], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, index=False)


def setup_logging(tag: str) -> Path:
    """Configure file + console logging; returns the log path."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"train_{tag}.log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)
    return log_path


def load_arcs_for(trip_ids: List[str], cache: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Load (and memoise) the arc tables needed for route continuity."""
    for trip_id in trip_ids:
        if trip_id in cache:
            continue
        for partition in ("train", "student_test"):
            path = DATA / partition / trip_id / f"{trip_id}.arcs"
            if path.is_file():
                cache[trip_id] = io.load_arcs(path)
                break
        else:
            raise FileNotFoundError(f"no .arcs file found for trip {trip_id}")
    return cache


def evaluate_fold(
    model: Any,
    pairs: pd.DataFrame,
    val_idx: np.ndarray,
    y_prob: np.ndarray,
    arcs_cache: Dict[str, pd.DataFrame],
) -> Dict[str, float]:
    """Per-pair + per-point metrics for one validation fold."""
    val = pairs.iloc[val_idx][["trip_id", "point_idx", "arc_id", "label"]].copy()
    val["y_prob"] = y_prob

    pair_metrics = ev.per_pair_metrics(val["label"].to_numpy(), y_prob)
    top1 = ev.top1_accuracy(val)
    load_arcs_for(list(val["trip_id"].unique()), arcs_cache)
    cont = ev.route_continuity(val, arcs_cache)

    return {
        "macro_f1": pair_metrics["macro_f1"],
        "log_loss": pair_metrics["log_loss"],
        "accuracy": pair_metrics["accuracy"],
        "precision": pair_metrics["precision"],
        "recall": pair_metrics["recall"],
        "top1": top1["top1_accuracy"],
        "top1_resolvable": top1["top1_accuracy_resolvable"],
        "candidate_miss_rate": top1["candidate_miss_rate"],
        "route_continuity": cont["continuity"],
        "strict_continuity": cont["strict_continuity"],
    }


def subsample_by_group(
    train_idx: np.ndarray,
    groups: np.ndarray,
    max_rows: int,
    seed: int = mz.RANDOM_STATE,
) -> np.ndarray:
    """Cut a training fold to ``max_rows`` by dropping **whole trips**.

    Row-level random sampling would put parts of the same trip in and out of
    the training set, breaking the group discipline the rest of the pipeline
    maintains and leaving a trip partially represented. Trips are instead
    shuffled and taken whole until the row budget is spent, so every included
    trip keeps its full point sequence.

    Returns ``train_idx`` unchanged when it already fits.
    """
    if max_rows is None or len(train_idx) <= max_rows:
        return train_idx

    fold_groups = groups[train_idx]
    unique = np.unique(fold_groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)

    keep_trips: List[str] = []
    total = 0
    for trip in unique:
        n = int((fold_groups == trip).sum())
        if total + n > max_rows and keep_trips:
            continue
        keep_trips.append(trip)
        total += n
        if total >= max_rows:
            break

    mask = np.isin(fold_groups, keep_trips)
    return train_idx[mask]


def run_model(
    name: str,
    pairs: pd.DataFrame,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    arcs_cache: Dict[str, pd.DataFrame],
    save: bool = True,
    subsample_rows: int | None = None,
    refit: bool = True,
) -> Dict[str, Any]:
    """Cross-validate one model, then refit it on everything."""
    logging.info("=" * 72)
    logging.info("model %s  (%s)", name, mz.MODEL_FAMILIES[name])
    logging.info("  config:    %s", mz.key_configuration(name))
    logging.info("  imbalance: %s", mz.imbalance_strategy(name))
    logging.info("  scaling:   %s", mz.expected_scaling_behavior(name))

    gkf = ev.make_group_kfold(n_splits)
    fold_metrics: List[Dict[str, float]] = []
    fit_times: List[float] = []
    score_times: List[float] = []
    all_warnings: set[str] = set()

    splits = list(gkf.split(X, y, groups))
    for fold, (train_idx, val_idx) in enumerate(
        tqdm(splits, desc=f"{name} CV", unit="fold", leave=False)
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            fit_idx = subsample_by_group(train_idx, groups, subsample_rows)
            if len(fit_idx) < len(train_idx):
                logging.info(
                    "    subsampled fit set: %d -> %d rows (%d -> %d whole trips)",
                    len(train_idx), len(fit_idx),
                    len(set(groups[train_idx])), len(set(groups[fit_idx])),
                )

            model = mz.build_model(name, y[fit_idx])
            t0 = time.perf_counter()
            model.fit(X.iloc[fit_idx], y[fit_idx])
            fit_s = time.perf_counter() - t0

            t1 = time.perf_counter()
            y_prob = model.predict_proba(X.iloc[val_idx])[:, 1]
            score_s = time.perf_counter() - t1

            all_warnings.update(
                f"{w.category.__name__}: {str(w.message)[:120]}" for w in caught
            )

        metrics = evaluate_fold(model, pairs, val_idx, y_prob, arcs_cache)
        fold_metrics.append(metrics)
        fit_times.append(fit_s)
        score_times.append(score_s)

        logging.info(
            "  fold %d/%d  %d train trips / %d val trips | fit %6.1fs score %6.1fs | "
            "macroF1 %.4f top1 %.4f cont %.4f logloss %.4f",
            fold + 1, n_splits,
            len(set(groups[train_idx])), len(set(groups[val_idx])),
            fit_s, score_s,
            metrics["macro_f1"], metrics["top1"],
            metrics["route_continuity"], metrics["log_loss"],
        )

    for w in sorted(all_warnings):
        logging.warning("  [%s] %s", name, w)

    frame = pd.DataFrame(fold_metrics)

    # Final refit on every training pair, for Step 14.
    final_fit_s = float("nan")
    final_model = None
    if refit:
        final_idx = subsample_by_group(np.arange(len(X)), groups, subsample_rows)
        if len(final_idx) < len(X):
            logging.info("  final fit subsampled: %d -> %d rows (whole trips)",
                         len(X), len(final_idx))
        t0 = time.perf_counter()
        final_model = mz.build_model(name, y[final_idx])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            final_model.fit(X.iloc[final_idx], y[final_idx])
            for w in caught:
                logging.warning("  [%s final] %s: %s", name, w.category.__name__,
                                str(w.message)[:120])
        final_fit_s = time.perf_counter() - t0
    else:
        logging.info("  --no-refit: keeping the existing %s.joblib untouched", name)

    if save and refit:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(
            {"model": final_model, "feature_columns": list(X.columns),
             "trained_on": "outputs/training_pairs.parquet", "n_rows": int(len(X))},
            model_path,
        )
        logging.info("  saved %s (%.1f MB)", model_path,
                     model_path.stat().st_size / 1e6)

    logging.info(
        "  CV summary: macroF1 %.4f+/-%.4f | top1 %.4f+/-%.4f | cont %.4f | "
        "logloss %.4f | fit %.1fs | final fit %.1fs",
        frame.macro_f1.mean(), frame.macro_f1.std(),
        frame.top1.mean(), frame.top1.std(),
        frame.route_continuity.mean(), frame.log_loss.mean(),
        float(np.mean(fit_times)), final_fit_s,
    )

    return {
        "model": name,
        "family": mz.MODEL_FAMILIES[name],
        "key_config": mz.key_configuration(name),
        "scaling": (
            "scaled"
            if "scaler" in dict((final_model or mz.build_model(name)).steps)
            else "unscaled"
        ),
        "cv_macro_f1_mean": frame.macro_f1.mean(),
        "cv_macro_f1_std": frame.macro_f1.std(),
        "cv_top1_mean": frame.top1.mean(),
        "cv_top1_std": frame.top1.std(),
        "cv_top1_resolvable_mean": frame.top1_resolvable.mean(),
        "cv_route_continuity_mean": frame.route_continuity.mean(),
        "cv_route_continuity_std": frame.route_continuity.std(),
        "cv_strict_continuity_mean": frame.strict_continuity.mean(),
        "cv_log_loss_mean": frame.log_loss.mean(),
        "cv_log_loss_std": frame.log_loss.std(),
        "cv_accuracy_mean": frame.accuracy.mean(),
        "cv_precision_mean": frame.precision.mean(),
        "cv_recall_mean": frame.recall.mean(),
        "cv_candidate_miss_rate_mean": frame.candidate_miss_rate.mean(),
        "fit_time_sec": float(np.mean(fit_times)),
        "fit_time_sec_std": float(np.std(fit_times)),
        "score_time_sec": float(np.mean(score_times)),
        "final_fit_time_sec": final_fit_s,
        "n_folds": n_splits,
        "n_train_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", default=str(OUT / "training_pairs.parquet"))
    parser.add_argument("--models", nargs="*", default=None,
                        help="subset of model names (default: all available)")
    parser.add_argument("--n-splits", type=int, default=ev.DEFAULT_N_SPLITS)
    parser.add_argument("--limit-trips", type=int, default=None,
                        help="use only the first N trips (smoke runs)")
    parser.add_argument("--no-save", action="store_true",
                        help="skip writing joblib files and cv_results.csv")
    parser.add_argument("--skip", "--skip-existing", dest="skip_existing",
                        action="store_true",
                        help="skip models whose .joblib already exists in outputs/models/")
    parser.add_argument("--no-refit", action="store_true",
                        help="cross-validate only; leave any existing .joblib alone")
    parser.add_argument("--subsample-models", nargs="*", default=[],
                        help="models whose FIT set is capped (default: none)")
    parser.add_argument("--subsample-rows", type=int, default=300_000,
                        help="row cap for those models, taken as whole trips")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                        help="feature matrix dtype; float32 halves peak memory")
    parser.add_argument("--out", default=str(OUT / "cv_results.csv"))
    args = parser.parse_args(argv)

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = setup_logging(tag)

    pairs = pd.read_parquet(args.pairs)
    if args.limit_trips:
        keep = sorted(pairs.trip_id.unique())[: args.limit_trips]
        pairs = pairs[pairs.trip_id.isin(keep)].reset_index(drop=True)

    feature_cols = [c for c in ft.FEATURE_COLUMNS if c in pairs.columns]
    excluded = [c for c in pairs.columns if c not in feature_cols]
    X = pairs[feature_cols]

    # float32 halves the feature matrix (181 MB -> 90 MB at 983k x 23) and,
    # more importantly, halves every intermediate copy sklearn makes during a
    # fit. This machine has ~16 GB with only a few GB free, and the first full
    # run died with numpy ArrayMemoryError, not a timeout. Tree models cast to
    # float32 internally anyway, so this costs no accuracy there; for the
    # scaled linear/MLP models the loss is far below the noise between folds.
    if args.dtype == "float32":
        X = X.astype("float32")
    logging.info("feature matrix: %s rows x %s cols, %.0f MB (%s)",
                 f"{len(X):,}", X.shape[1],
                 X.memory_usage(deep=True).sum() / 1e6, args.dtype)
    y = pairs["label"].to_numpy().astype(int)
    groups = pairs["trip_id"].to_numpy()

    names = args.models or mz.model_names()
    unknown = [n for n in names if n not in mz.ALL_MODELS]
    if unknown:
        parser.error(f"unknown model(s): {unknown}")

    config = {
        "pairs_file": args.pairs,
        "n_rows": int(len(pairs)),
        "n_trips": int(pairs.trip_id.nunique()),
        "n_features": len(feature_cols),
        "positive_rate": float(y.mean()),
        "n_splits": args.n_splits,
        "models": names,
        "feature_columns": feature_cols,
        "excluded_columns": excluded,
        "lightgbm_available": mz.HAS_LIGHTGBM,
        "random_state": mz.RANDOM_STATE,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    logging.info("PathSnap Step 13 -- train_all_models")
    logging.info("config:\n%s", json.dumps(config, indent=2))
    if not mz.HAS_LIGHTGBM:
        logging.warning("lightgbm unavailable -- 'lgbm' skipped")

    if args.skip_existing:
        already = {p.stem for p in MODELS_DIR.glob("*.joblib")} if MODELS_DIR.is_dir() else set()
        skipped = [n for n in names if n in already]
        if skipped:
            logging.info("--skip-existing: skipping %s (joblib already saved)", skipped)
        names = [n for n in names if n not in already]
        if not names:
            logging.info("nothing left to train")
            return 0

    rows: List[Dict[str, Any]] = []
    arcs_cache: Dict[str, pd.DataFrame] = {}
    out_path = Path(args.out)
    t_start = time.perf_counter()

    for name in tqdm(names, desc="models", unit="model"):
        try:
            cap = args.subsample_rows if name in (args.subsample_models or []) else None
            row = run_model(name, pairs, X, y, groups, args.n_splits, arcs_cache,
                            save=not args.no_save, subsample_rows=cap,
                            refit=not args.no_refit)
            rows.append(row)
            # Persist after EVERY model. The first full run was killed by the
            # OS mid-sweep and lost four completed folds, because results were
            # only written at the end; a kill is not an exception, so the
            # try/except below cannot help. This can.
            if not args.no_save:
                write_results(pd.DataFrame([row]), out_path)
                logging.info("  appended %s to %s", name, out_path)
        except Exception as exc:  # keep the sweep going; report at the end
            logging.exception("model %s FAILED: %s", name, exc)

    elapsed = time.perf_counter() - t_start
    results = pd.DataFrame(rows)
    if not results.empty:
        results = results[[c for c in CV_RESULT_COLUMNS if c in results.columns]]

    logging.info("total elapsed %.1f s (%.1f min)", elapsed, elapsed / 60)
    logging.info("log written to %s", log_path)

    if not results.empty:
        show = ["model", "cv_macro_f1_mean", "cv_top1_mean", "cv_top1_std",
                "cv_route_continuity_mean", "cv_log_loss_mean", "fit_time_sec"]
        logging.info("\n%s", results[show].round(4).to_string(index=False))

    return 0 if len(rows) == len(names) else 1


if __name__ == "__main__":
    raise SystemExit(main())
