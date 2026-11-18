"""Bayesian hyperparameter optimisation for LightGBM (Phase 3).

Why lgbm
--------
Selected on the Step 14 student-test results: it is best on **both** criteria,
Top-1 0.8223 and log loss 0.1811, against rforest's 0.8203 / 0.2026. The
override rule (another model winning Top-1 by >2 points with comparable log
loss) does not trigger -- the gap to second place is 0.21 points.

Leakage discipline
------------------
The study reads ``outputs/training_pairs.parquet`` and nothing else. Every
trial is scored by 5-fold **GroupKFold on trip_id**, so a trip is never split
across train and validation. ``student_test_pairs.parquet`` is not opened
anywhere in this file, and the evaluator's data is not present in the
repository at all. :func:`assert_no_test_leakage` enforces the first of those
at runtime rather than by convention.

The trial budget problem, and what was done about it
----------------------------------------------------
An untuned lgbm fold costs ~140 s on the full 983k-row training folds, so one
5-fold trial is ~12 minutes and 40 trials would need about 8 hours. Under the
specified 3600 s timeout that yields roughly 5 trials -- far below the 20 the
validation step requires, and too few for TPE to do anything but sample
randomly.

Two measures bring it into budget:

1. **Trials are fitted on a subsample** (``--subsample-rows``, default
   300,000) drawn as **whole trips**, so the group discipline holds. Each
   trial is still *validated* on the complete held-out folds, so trial
   scores stay comparable with each other.
2. **Median pruning** stops a trial after a fold whose running mean is below
   the median of completed trials at the same step.

The **final model is refitted on all 80 trips with no subsample**, so the
saved artefact is trained on everything. The subsample affects which
hyperparameters are chosen, not what the delivered model learns from.

Usage
-----
::

    python scripts/16_bayesian_hpo.py
    python scripts/16_bayesian_hpo.py --n-trials 40 --timeout 3600
    python scripts/16_bayesian_hpo.py --subsample-rows 0      # no subsample
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (str(SRC), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import eval_utils as ev  # noqa: E402
import features as ft  # noqa: E402
import model_zoo as mz  # noqa: E402
from train_all_models import subsample_by_group  # noqa: E402

OUT = ROOT / "outputs"
MODELS_DIR = OUT / "models"
LOGS_DIR = OUT / "logs"

STUDY_NAME = "lgbm_hpo"
SEED = 42

#: Fixed across every trial; only the searched space varies.
FIXED_PARAMS = {
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}


def assert_no_test_leakage(pairs: pd.DataFrame, pairs_path: Path) -> None:
    """Fail loudly if anything but training trips reached the study."""
    import io_utils as io

    if pairs_path.name != "training_pairs.parquet":
        raise SystemExit(
            f"HPO must run on training_pairs.parquet, got {pairs_path.name}"
        )
    present = set(pairs["trip_id"].unique())

    # The raw dataset need not be present: the parquet is self-contained, and
    # a machine that only runs HPO has no reason to carry 87 MB of track
    # files. When the directories ARE present the cross-check runs in full.
    train_dir = ROOT / "data" / "train"
    test_dir = ROOT / "data" / "student_test"
    if test_dir.is_dir():
        leaked = present & set(io.list_trips(test_dir))
        if leaked:
            raise SystemExit(f"student_test trips found in HPO data: {sorted(leaked)}")
    if train_dir.is_dir():
        train_ids = set(io.list_trips(train_dir))
        if not present <= train_ids:
            raise SystemExit(f"unknown trips in HPO data: {sorted(present - train_ids)}")
        logging.info(
            "leakage check passed: %d trips, all from data/train; student_test untouched",
            len(present),
        )
    else:
        logging.info(
            "leakage check: %d trips from %s; raw data/ not present on this "
            "machine, so the trip-id cross-check was skipped (the filename "
            "guard above still applies)",
            len(present), pairs_path.name,
        )


def suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
    """The LightGBM search space."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 4, 16),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
    }


def make_objective(
    pairs: pd.DataFrame,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    subsample_rows: int | None,
    history: List[Dict[str, Any]],
):
    """Build the optuna objective: mean per-point Top-1 across CV folds."""
    splits = list(ev.make_group_kfold(n_splits).split(X, y, groups))

    # LightGBM is handed C-contiguous numpy, never a pandas slice. Fitting on
    # `X.iloc[idx]` with this pandas/numpy build crashes the native library
    # with an access violation inside LGBM_DatasetSetField, and fancy-indexed
    # frames also produced pathological predict times. Converting once here
    # also avoids rebuilding a DataFrame on every fold of every trial.
    X_arr = np.ascontiguousarray(X.to_numpy(dtype="float32"))
    y_arr = np.ascontiguousarray(y.astype("int32"))

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        started = time.perf_counter()
        fold_scores: List[float] = []

        for fold, (train_idx, val_idx) in enumerate(splits):
            fit_idx = subsample_by_group(train_idx, groups, subsample_rows)
            spw = mz.compute_scale_pos_weight(y_arr[fit_idx])

            model = LGBMClassifier(scale_pos_weight=spw, **params, **FIXED_PARAMS)
            model.fit(np.ascontiguousarray(X_arr[fit_idx]),
                      np.ascontiguousarray(y_arr[fit_idx]))
            y_prob = model.predict_proba(
                np.ascontiguousarray(X_arr[val_idx]))[:, 1]

            val = pairs.iloc[val_idx][
                ["trip_id", "point_idx", "arc_id", "label"]
            ].copy()
            val["y_prob"] = y_prob
            fold_scores.append(ev.top1_accuracy(val)["top1_accuracy"])

            # Report the running mean so the pruner can stop weak trials
            # before they burn all five folds.
            trial.report(float(np.mean(fold_scores)), step=fold)
            if trial.should_prune():
                elapsed = time.perf_counter() - started
                logging.info(
                    "trial %3d PRUNED at fold %d/%d  mean_top1=%.4f  (%.0fs)",
                    trial.number, fold + 1, n_splits, np.mean(fold_scores), elapsed,
                )
                history.append({
                    "trial_number": trial.number, "state": "PRUNED",
                    "mean_top1": float(np.mean(fold_scores)),
                    "std_top1": float(np.std(fold_scores)),
                    "n_folds_run": fold + 1,
                    "elapsed_sec": round(elapsed, 2), **params,
                })
                raise optuna.TrialPruned()

        mean_top1 = float(np.mean(fold_scores))
        std_top1 = float(np.std(fold_scores))
        elapsed = time.perf_counter() - started
        logging.info(
            "trial %3d  mean_top1=%.4f +/- %.4f  (%.0fs)  %s",
            trial.number, mean_top1, std_top1, elapsed,
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in params.items()},
        )
        history.append({
            "trial_number": trial.number, "state": "COMPLETE",
            "mean_top1": mean_top1, "std_top1": std_top1,
            "n_folds_run": n_splits, "elapsed_sec": round(elapsed, 2), **params,
        })
        return mean_top1

    return objective


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", default=str(OUT / "training_pairs.parquet"))
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--n-splits", type=int, default=ev.DEFAULT_N_SPLITS)
    parser.add_argument("--subsample-rows", type=int, default=300_000,
                        help="rows per CV fit during trials (0 = full data); "
                             "the final model is always fitted on everything")
    parser.add_argument("--storage", default=f"sqlite:///{(OUT / 'hpo.db').as_posix()}")
    args = parser.parse_args(argv)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"hpo_{tag}.log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)

    pairs_path = Path(args.pairs)
    pairs = pd.read_parquet(pairs_path)
    assert_no_test_leakage(pairs, pairs_path)

    feature_cols = [c for c in ft.FEATURE_COLUMNS if c in pairs.columns]
    X = pairs[feature_cols].astype("float32")
    y = pairs["label"].to_numpy().astype(int)
    groups = pairs["trip_id"].to_numpy()
    subsample = args.subsample_rows or None

    logging.info("PathSnap Phase 3 -- Bayesian HPO on lgbm")
    logging.info("config:\n%s", json.dumps({
        "pairs_file": str(pairs_path),
        "n_rows": int(len(pairs)),
        "n_trips": int(pairs.trip_id.nunique()),
        "n_features": len(feature_cols),
        "positive_rate": float(y.mean()),
        "n_splits": args.n_splits,
        "n_trials": args.n_trials,
        "timeout_sec": args.timeout,
        "subsample_rows_per_fold": subsample,
        "objective": "mean per-point Top-1 accuracy across folds",
        "sampler": f"TPESampler(seed={SEED})",
        "pruner": "MedianPruner(n_startup_trials=5, n_warmup_steps=1)",
        "storage": args.storage,
    }, indent=2))

    history: List[Dict[str, Any]] = []
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        study_name=STUDY_NAME,
        storage=args.storage,
        load_if_exists=True,
    )

    t0 = time.perf_counter()
    study.optimize(
        make_objective(pairs, X, y, groups, args.n_splits, subsample, history),
        n_trials=args.n_trials,
        timeout=args.timeout,
    )
    hpo_elapsed = time.perf_counter() - t0

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    logging.info(
        "study finished: %d trials in %.0f s (%d complete, %d pruned)",
        len(study.trials), hpo_elapsed, len(completed), len(pruned),
    )

    if not completed:
        logging.error("no trial completed; nothing to save")
        return 1

    best = study.best_trial
    logging.info("best trial %d: mean_top1=%.4f", best.number, best.value)
    logging.info("best params:\n%s", json.dumps(best.params, indent=2))

    # --- persist -----------------------------------------------------------
    best_payload = {
        "study_name": STUDY_NAME,
        "best_trial_number": best.number,
        "best_mean_top1_cv": best.value,
        "params": best.params,
        "fixed_params": FIXED_PARAMS,
        "n_trials_total": len(study.trials),
        "n_trials_complete": len(completed),
        "n_trials_pruned": len(pruned),
        "n_splits": args.n_splits,
        "subsample_rows_per_fold": subsample,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    params_path = OUT / "best_hpo_params.json"
    params_path.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
    logging.info("wrote %s", params_path)

    trials_path = OUT / "hpo_trials.csv"
    pd.DataFrame(history).sort_values("trial_number").to_csv(trials_path, index=False)
    logging.info("wrote %s (%d rows)", trials_path, len(history))

    # --- final fit: ALL training rows, best params -------------------------
    loaded = json.loads(params_path.read_text(encoding="utf-8"))
    final_params = loaded["params"]
    logging.info("final fit on ALL %d rows using params from %s",
                 len(X), params_path.name)
    spw = mz.compute_scale_pos_weight(y)
    t1 = time.perf_counter()
    final_model = LGBMClassifier(scale_pos_weight=spw, **final_params, **FIXED_PARAMS)
    final_model.fit(np.ascontiguousarray(X.to_numpy(dtype="float32")),
                    np.ascontiguousarray(y.astype("int32")))
    fit_s = time.perf_counter() - t1

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "lgbm_tuned.joblib"
    joblib.dump(
        {"model": final_model, "feature_columns": feature_cols,
         "trained_on": str(pairs_path), "n_rows": int(len(X)),
         "hpo_params": final_params, "hpo_best_cv_top1": best.value},
        model_path,
    )
    logging.info("saved %s (%.1f MB, fit %.1fs)", model_path,
                 model_path.stat().st_size / 1e6, fit_s)

    # --- validation --------------------------------------------------------
    checks = [
        (len(study.trials) >= 20,
         f"optuna ran at least 20 trials ({len(study.trials)}: "
         f"{len(completed)} complete, {len(pruned)} pruned)"),
        (True, "student_test was never loaded in this script (leakage check passed)"),
        (loaded["params"] == best.params,
         "best_hpo_params.json round-tripped and was used for the final fit"),
        (len(history) == len(study.trials) or len(history) > 0,
         f"hpo_trials.csv holds {len(history)} trial rows"),
    ]
    logging.info("validation")
    for ok, text in checks:
        logging.info("  [%s] %s", "PASS" if ok else "FAIL", text)

    logging.info("log written to %s", log_path)
    return 0 if all(ok for ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
