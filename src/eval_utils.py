"""Evaluation utilities for PathSnap model comparison (Step 10).

Two levels of metric, because they answer different questions:

* **Per-pair metrics** score the binary problem the models actually solve:
  "is this (GPS point, candidate arc) pair a match?". Useful for training
  diagnostics, but a model can look good here while picking the wrong arc,
  since 85% of pairs are negatives.
* **Per-point metrics** score the problem the task cares about:
  :func:`top1_accuracy` asks whether the highest-scoring candidate for a
  point is the right arc, and :func:`route_continuity` asks whether the
  chosen arcs form a drivable sequence.

Grouping rule
-------------
Every split in this project is **group-aware by ``trip_id``**. A trip's GPS
points are heavily autocorrelated -- consecutive fixes a second apart sit on
the same arc with near-identical features -- so splitting within a trip puts
near-duplicates on both sides and inflates scores. :func:`make_group_kfold`
is the only splitter used.

Ceiling on accuracy
-------------------
Step 9 measured a pooled candidate miss rate of 2.11%: for those points the
true arc is not in the candidate set at all, so no model can get them right.
:func:`top1_accuracy` counts them as incorrect and reports them separately,
so the gap between "model error" and "data ceiling" stays visible.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold

__all__ = [
    "DEFAULT_N_SPLITS",
    "make_group_kfold",
    "per_pair_metrics",
    "top1_accuracy",
    "route_continuity",
    "inference_time_per_trip",
]

DEFAULT_N_SPLITS = 5


def make_group_kfold(n_splits: int = DEFAULT_N_SPLITS) -> GroupKFold:
    """Return a group-aware K-fold splitter.

    Parameters
    ----------
    n_splits
        Number of folds (default 5). With 80 training trips that is 16 trips
        held out per fold.

    Returns
    -------
    sklearn.model_selection.GroupKFold
        Use as ``splitter.split(X, y, groups=trip_ids)``. ``GroupKFold`` is
        deterministic and needs no ``random_state``: it balances fold sizes
        while keeping every group wholly on one side.

    Notes
    -----
    Pass ``trip_id`` as ``groups``. Never split within a trip -- see the
    module docstring for why.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    return GroupKFold(n_splits=n_splits)


def per_pair_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Binary classification metrics for the pair problem.

    Parameters
    ----------
    y_true
        True labels, 0/1, one per candidate pair.
    y_prob
        Predicted probability of class 1.
    threshold
        Decision threshold for the hard-label metrics.

    Returns
    -------
    dict
        ``accuracy``, ``macro_f1``, ``log_loss``, ``precision``, ``recall``,
        plus ``n_pairs``, ``n_positives`` and ``positive_rate`` for context.

    Notes
    -----
    ``macro_f1`` averages the F1 of both classes equally, which is why it is
    the headline pair metric here: plain accuracy is ~85% for a model that
    predicts "no match" every time.

    ``log_loss`` is computed with an explicit ``labels=[0, 1]`` so a fold
    that happens to contain one class only still returns a number instead of
    raising. Probabilities are clipped away from 0 and 1 to keep the loss
    finite when a model is confidently wrong.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype="float64")
    if y_true.shape != y_prob.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_prob.shape}")

    y_pred = (y_prob >= threshold).astype(int)
    eps = 1e-15
    clipped = np.clip(y_prob, eps, 1.0 - eps)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "n_pairs": int(y_true.size),
        "n_positives": int(y_true.sum()),
        "positive_rate": float(y_true.mean()) if y_true.size else float("nan"),
    }


def top1_accuracy(pairs_df: pd.DataFrame, prob_col: str = "y_prob") -> Dict[str, Any]:
    """Fraction of GPS points whose highest-scoring candidate is the true arc.

    Parameters
    ----------
    pairs_df
        Must contain ``trip_id``, ``point_idx``, ``arc_id``, ``label`` and
        the probability column. One row per candidate pair.
    prob_col
        Name of the predicted-probability column.

    Returns
    -------
    dict
        ``top1_accuracy``        correct picks / all points, **counting
                                 candidate misses as incorrect**. This is the
                                 headline number.
        ``top1_accuracy_resolvable``
                                 correct picks / points that *have* a true
                                 candidate -- the model's own hit rate, with
                                 the data ceiling removed.
        ``n_points``             points evaluated.
        ``n_correct``            points where the argmax row is the labelled one.
        ``n_candidate_miss``     points with no ``label == 1`` row at all.
        ``candidate_miss_rate``  that count as a fraction.

    Notes
    -----
    Ties in probability are broken by taking the first row in frame order,
    which is ``candidate_rank`` order from Step 6 -- i.e. a tie falls to the
    geometrically nearest arc. That is a deliberate, documented tiebreak
    rather than an arbitrary one, and it matters for models such as the
    decision tree that emit many identical probabilities.
    """
    required = {"trip_id", "point_idx", "arc_id", "label", prob_col}
    missing = required - set(pairs_df.columns)
    if missing:
        raise KeyError(f"top1_accuracy needs columns {sorted(missing)}")

    df = pairs_df[["trip_id", "point_idx", "arc_id", "label", prob_col]].reset_index(drop=True)

    # argmax per point; stable sort keeps candidate_rank order within ties.
    order = np.lexsort((df.index.to_numpy(), -df[prob_col].to_numpy()))
    ranked = df.iloc[order]
    picks = ranked.groupby(["trip_id", "point_idx"], sort=False).head(1)

    truth = df[df["label"] == 1][["trip_id", "point_idx", "arc_id"]].rename(
        columns={"arc_id": "true_arc_id"}
    )
    merged = picks.merge(truth, on=["trip_id", "point_idx"], how="left")

    has_truth = merged["true_arc_id"].notna()
    correct = has_truth & (merged["arc_id"] == merged["true_arc_id"])

    n_points = int(len(merged))
    n_miss = int((~has_truth).sum())
    n_correct = int(correct.sum())
    n_resolvable = n_points - n_miss

    return {
        "top1_accuracy": float(n_correct / n_points) if n_points else float("nan"),
        "top1_accuracy_resolvable": (
            float(n_correct / n_resolvable) if n_resolvable else float("nan")
        ),
        "n_points": n_points,
        "n_correct": n_correct,
        "n_candidate_miss": n_miss,
        "candidate_miss_rate": float(n_miss / n_points) if n_points else float("nan"),
    }


def route_continuity(
    pairs_df: pd.DataFrame,
    arcs_dict: Mapping[str, pd.DataFrame],
    prob_col: str = "y_prob",
) -> Dict[str, Any]:
    """How often consecutive top-1 picks form a connected sequence.

    For each trip, the top-1 arc is taken per GPS point in ``point_idx``
    order, and consecutive pairs are checked for connectivity.

    Parameters
    ----------
    pairs_df
        As for :func:`top1_accuracy`.
    arcs_dict
        ``trip_id -> arcs DataFrame`` (as loaded by ``io_utils.load_arcs``),
        used to look up ``from_node``/``to_node``.
    prob_col
        Name of the predicted-probability column.

    Returns
    -------
    dict
        ``continuity``        headline score: fraction of consecutive pairs
                              that are connected, **counting "stayed on the
                              same arc" as connected**.
        ``strict_continuity`` the literal ``arc[i].to_node ==
                              arc[i+1].from_node`` rule, with no same-arc
                              exemption.
        ``n_transitions``     consecutive pairs evaluated.
        ``n_same_arc``        of those, how many stayed on one arc.
        ``n_trips``           trips contributing at least one transition.

    Notes
    -----
    **Deliberate deviation from a literal reading of the spec.** A vehicle
    usually stays on the same arc between consecutive 1 Hz fixes -- this
    dataset averages 6.5 candidates per point and up to 57 GPS points per
    route arc. For a repeated arc, ``to_node != from_node`` holds by
    definition, so the strict rule scores a *perfect* map match near zero.
    ``continuity`` therefore treats staying put as continuous, and
    ``strict_continuity`` is returned alongside so the literal number is
    still available.
    """
    required = {"trip_id", "point_idx", "arc_id", prob_col}
    missing = required - set(pairs_df.columns)
    if missing:
        raise KeyError(f"route_continuity needs columns {sorted(missing)}")

    df = pairs_df[["trip_id", "point_idx", "arc_id", prob_col]].reset_index(drop=True)
    order = np.lexsort((df.index.to_numpy(), -df[prob_col].to_numpy()))
    picks = (
        df.iloc[order]
        .groupby(["trip_id", "point_idx"], sort=False)
        .head(1)
        .sort_values(["trip_id", "point_idx"])
    )

    n_conn = n_strict = n_trans = n_same = 0
    trips = 0
    for trip_id, group in picks.groupby("trip_id", sort=False):
        arcs = arcs_dict.get(trip_id)
        if arcs is None:
            raise KeyError(f"arcs_dict has no entry for trip {trip_id!r}")
        seq = group["arc_id"].to_numpy()
        if seq.size < 2:
            continue
        trips += 1
        to_node = arcs["to_node"].to_numpy()[seq[:-1]]
        from_node = arcs["from_node"].to_numpy()[seq[1:]]
        same = seq[:-1] == seq[1:]
        strict = to_node == from_node
        n_trans += seq.size - 1
        n_same += int(same.sum())
        n_strict += int(strict.sum())
        n_conn += int((strict | same).sum())

    return {
        "continuity": float(n_conn / n_trans) if n_trans else float("nan"),
        "strict_continuity": float(n_strict / n_trans) if n_trans else float("nan"),
        "n_transitions": n_trans,
        "n_same_arc": n_same,
        "n_trips": trips,
    }


def inference_time_per_trip(
    model: Any,
    X: pd.DataFrame,
    trip_ids: Sequence[str],
) -> Dict[str, float]:
    """Mean wall-clock ``predict_proba`` time per trip.

    Prediction is timed **per trip**, as it would run in deployment: one
    trip's candidate pairs arrive together and are scored in one call.

    Parameters
    ----------
    model
        Fitted estimator exposing ``predict_proba``.
    X
        Feature frame, rows aligned with ``trip_ids``.
    trip_ids
        Trip id per row of ``X``.

    Returns
    -------
    dict
        ``mean_seconds_per_trip``, ``median_seconds_per_trip``,
        ``total_seconds``, ``n_trips``, ``mean_pairs_per_trip`` and
        ``seconds_per_1k_pairs``.

    Notes
    -----
    Timing is single-threaded only insofar as the estimator is; models built
    with ``n_jobs=-1`` will report their parallel time, which is the honest
    deployment number for this machine.
    """
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"{type(model).__name__} has no predict_proba")

    trip_ids = np.asarray(trip_ids)
    if len(trip_ids) != len(X):
        raise ValueError(f"X has {len(X)} rows but got {len(trip_ids)} trip ids")

    times, sizes = [], []
    for trip_id in pd.unique(trip_ids):
        mask = trip_ids == trip_id
        chunk = X.loc[mask] if isinstance(X, pd.DataFrame) else X[mask]
        start = time.perf_counter()
        model.predict_proba(chunk)
        times.append(time.perf_counter() - start)
        sizes.append(int(mask.sum()))

    times_arr = np.asarray(times)
    sizes_arr = np.asarray(sizes)
    return {
        "mean_seconds_per_trip": float(times_arr.mean()) if times_arr.size else float("nan"),
        "median_seconds_per_trip": float(np.median(times_arr)) if times_arr.size else float("nan"),
        "total_seconds": float(times_arr.sum()),
        "n_trips": int(times_arr.size),
        "mean_pairs_per_trip": float(sizes_arr.mean()) if sizes_arr.size else float("nan"),
        "seconds_per_1k_pairs": (
            float(1000.0 * times_arr.sum() / sizes_arr.sum()) if sizes_arr.sum() else float("nan")
        ),
    }
