"""Shared paths, fold assignment and metrics for the HMM-improvement experiments.

Every metric here follows the definitions already used in the project:

* **Top-1** -- correct picks / GPS points that have at least one candidate,
  where a point whose aligned truth is ``-1`` (unassigned) or absent from
  its candidates counts as incorrect. Same denominator as
  ``eval_utils.top1_accuracy`` and ``scripts/20_eval_hmm.py``.
* **Route continuity** -- consecutive picks that stay on one arc or follow
  ``to_node == from_node`` (``eval_utils.route_continuity``).
* **Macro F1 / log loss** -- ``eval_utils.per_pair_metrics`` on the
  candidate pairs. For a decoded path the pair prediction is 1 for the
  decoded arc and 0 for every other candidate of that point.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
HMM_DIR = Path(__file__).resolve().parent
for p in (SRC, HMM_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import eval_utils as eu  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "outputs" / "hmm"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
N_FOLDS = 5


def fold_of_trips(trip_ids) -> Dict[str, int]:
    """Deterministic GroupKFold assignment of training trips (one trip = one group)."""
    ids = np.asarray(sorted(set(map(str, trip_ids))))
    folds = {}
    for k, (_, test) in enumerate(eu.make_group_kfold(N_FOLDS).split(ids, groups=ids)):
        for t in ids[test]:
            folds[str(t)] = k
    return folds


def continuity(seq: np.ndarray, from_node: np.ndarray, to_node: np.ndarray) -> tuple:
    """(n_connected, n_transitions) for one trip's pick sequence."""
    if len(seq) < 2:
        return 0, 0
    a, b = seq[:-1], seq[1:]
    linked = (a == b) | (to_node[a] == from_node[b])
    return int(linked.sum()), int(len(a))
