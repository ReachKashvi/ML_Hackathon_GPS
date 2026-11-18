"""Experimental Viterbi decoder for HMM-improvement.

A superset of ``src/hmm_viterbi.py``: with the default :class:`Config`
(``same_arc="abs"``, ``dist="gauss"``, ``emission="raw"``, no U-turn) it
reproduces the Phase 4 decoder exactly (checked by ``02_run_experiments.py
--check``). Each switch is one experiment:

``same_arc="signed"`` (E1)
    Phase 4 scores staying on an arc as ``|t_b - t_a| * len``, so sliding
    *backwards* along a directed arc costs nothing. Two-way streets are
    stored as twin arcs (94% of arcs have one), so this makes A->B and its
    twin B->A equally plausible for a moving vehicle -- 64% of the Phase 4
    student_test errors are exactly that confusion. Signed progress
    ``(t_b - t_a) * len`` keeps forward motion free and makes backward motion
    cost twice its length in the residual, while GPS jitter of a few metres
    still costs little.

``uturn_penalty=U`` (E2)
    The arc graph only lets a vehicle reverse by driving to the end node and
    back along the twin. Vehicles also turn mid-arc (trip 00000077 stops and
    turns on a 77 m arc). With U set, ``a -> twin(a)`` may instead cost the
    along-road distance between the two positions, ``|t_a - (1 - t_b)| *
    len``, minus a fixed ``U`` nats; the better of the two is used.

``dist="laplace"`` (E3)
    The measured residual (road distance - GPS displacement) on true
    transitions has a ~2 m core and a heavy tail; a Laplace density
    ``-|r|/sigma`` fits that shape better than a Gaussian.

``emission="norm" | "temp" | "platt"`` (E4)
    ``log y_prob`` from a per-pair binary classifier is not a distribution
    over a point's candidates, and lgbm's ``scale_pos_weight`` inflates it.
    ``norm`` renormalises per point; ``temp`` rescales the logit by ``1/T``;
    ``platt`` maps it to ``a * logit + b`` with ``(a, b)`` fitted on the
    other training folds (``04_fit_calibration.py``).

Route-break resets are not supported: every HMM-improvement number is the
production setting (no ``.route`` at decode time).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

import geo_utils as geo
import hmm_graph as hg
import hmm_viterbi as hv

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


@dataclass(frozen=True)
class Config:
    sigma: float = 20.0
    beta: float = 16.0
    same_arc: str = "abs"           # "abs" (Phase 4) | "signed" | "signed_tol"
    jitter_tol_m: float = 10.0      # "signed_tol": backward slack treated as GPS jitter
    uturn_penalty: Optional[float] = None
    dist: str = "gauss"             # "gauss" | "laplace"
    emission: str = "raw"           # "raw" | "norm" | "temp" | "platt"
    temperature: float = 1.0
    platt_a: float = 1.0
    platt_b: float = 0.0
    max_gap_s: float = hv.DEFAULT_MAX_GAP_S
    cutoff_m: float = hg.DEFAULT_CUTOFF_M

    def as_dict(self) -> Dict:
        return asdict(self)


def twin_arcs(trip: Mapping) -> np.ndarray:
    """``twin[a]`` = id of the arc with swapped endpoints, or -1."""
    arcs = trip["arcs"]
    fr = arcs["from_node"].to_numpy()
    to = arcs["to_node"].to_numpy()
    key = pd.Series(np.arange(len(fr)), index=pd.MultiIndex.from_arrays([fr, to]))
    key = key[~key.index.duplicated()]
    twin = key.reindex(pd.MultiIndex.from_arrays([to, fr])).to_numpy(dtype="float64")
    return np.where(np.isnan(twin), -1, twin).astype("int64")


def _emissions(logp: np.ndarray, cfg: Config) -> np.ndarray:
    if cfg.emission == "raw":
        return logp
    if cfg.emission == "norm":
        m = logp.max()
        return logp - (m + math.log(np.exp(logp - m).sum()))
    if cfg.emission == "temp":
        p = np.exp(logp)
        z = logp - np.log1p(-np.minimum(p, 1 - 1e-12))
        return -np.logaddexp(0.0, -z / cfg.temperature)
    if cfg.emission == "platt":
        p = np.exp(logp)
        z = logp - np.log1p(-np.minimum(p, 1 - 1e-12))
        return -np.logaddexp(0.0, -(cfg.platt_a * z + cfg.platt_b))
    raise ValueError(cfg.emission)


def _log_density(r: np.ndarray, sigma_eff: float, cfg: Config) -> np.ndarray:
    if cfg.dist == "gauss":
        return -(r ** 2) / (2.0 * sigma_eff ** 2) - math.log(sigma_eff) - _LOG_SQRT_2PI
    if cfg.dist == "laplace":
        return -np.abs(r) / sigma_eff - math.log(2.0 * sigma_eff)
    raise ValueError(cfg.dist)


def _transitions(prev_arcs, prev_frac, cur_arcs, cur_frac, d, dt, neighbors_of,
                 lengths, twin, cfg: Config) -> np.ndarray:
    """beta-weighted transition log-probabilities, shape (P, C)."""
    P, C = len(prev_arcs), len(cur_arcs)
    network = np.empty((P, C), dtype="float64")
    for i, a in enumerate(prev_arcs.tolist()):
        nb = neighbors_of(a)
        network[i] = [nb.get(b, math.inf) for b in cur_arcs.tolist()]
    same = prev_arcs[:, None] == cur_arcs[None, :]
    network[same] = 0.0

    len_a = lengths[prev_arcs][:, None]
    len_b = lengths[cur_arcs][None, :]
    ta = prev_frac[:, None]
    tb = cur_frac[None, :]
    corr_diff = (1.0 - ta) * len_a - (1.0 - tb) * len_b
    if cfg.same_arc == "signed":
        corr_same = (tb - ta) * len_a
    elif cfg.same_arc == "signed_tol":
        # like "abs" for backward moves up to tol (jitter), then increasingly
        # penalised: p + 2*min(back, tol) is continuous and equals |p| inside
        p = (tb - ta) * len_a
        corr_same = p + 2.0 * np.minimum(np.maximum(-p, 0.0), cfg.jitter_tol_m)
    else:
        corr_same = np.abs(tb - ta) * len_a
    corr = np.where(same, corr_same, corr_diff)
    corr = np.where(np.isfinite(corr), corr, 0.0)
    network = network + corr

    reachable = np.isfinite(network)
    if cfg.same_arc in ("signed", "signed_tol"):
        # only same-arc progress may be negative (moving backwards); arc
        # changes keep Phase 4's clamp at zero
        network = np.where(reachable,
                           np.where(same, network, np.maximum(network, 0.0)), 0.0)
    else:
        network = np.where(reachable, np.maximum(network, 0.0), 0.0)
    sigma_eff = cfg.sigma * max(1.0, math.sqrt(max(dt, 0.0) / cfg.max_gap_s))
    logp = np.clip(_log_density(network - d, sigma_eff, cfg), hv.LOG_FLOOR, 0.0)
    logp = np.where(reachable, logp, hv.LOG_FLOOR)
    trans = cfg.beta * logp

    if cfg.uturn_penalty is not None:
        is_twin = twin[prev_arcs][:, None] == cur_arcs[None, :]
        if is_twin.any():
            alt = np.abs(ta - (1.0 - tb)) * len_a
            lp = np.clip(_log_density(alt - d, sigma_eff, cfg), hv.LOG_FLOOR, 0.0)
            trans = np.where(is_twin,
                             np.maximum(trans, cfg.beta * lp - cfg.uturn_penalty), trans)
    return trans


def prepare(trip_id: str, pairs_df: pd.DataFrame, trip: Mapping,
            prob_col: str = "y_prob") -> Dict:
    """Per-trip arrays shared by every config decoded on this trip."""
    n = len(trip["track"])
    proj = geo.project_trip(trip)
    return {
        "trip_id": str(trip_id),
        "cols": hv._trellis_columns(str(trip_id), pairs_df, n, prob_col),
        "px": proj["track_x"], "py": proj["track_y"],
        "t": trip["track"]["t_sec"].to_numpy(dtype="float64"),
        "lengths": hg.arc_lengths_m(trip),
        "twin": twin_arcs(trip),
        "graph": hg.build_arc_graph(trip),
    }


def decode(prep: Dict, cfg: Config) -> np.ndarray:
    """Most probable arc per GPS point (``len == n_points``)."""
    col_arcs, col_logp, col_frac = prep["cols"]
    n = len(col_arcs)
    px, py, t_sec = prep["px"], prep["py"], prep["t"]
    lengths, twin, graph, tid = prep["lengths"], prep["twin"], prep["graph"], prep["trip_id"]

    def neighbors_of(arc):
        return hg.precompute_neighbors(graph, arc, cutoff_m=cfg.cutoff_m, trip_id=tid)

    states: List[Optional[np.ndarray]] = [None] * n
    scores: List[Optional[np.ndarray]] = [None] * n
    back: List[Optional[np.ndarray]] = [None] * n
    prev_t = None
    prev_states = prev_scores = prev_frac = None
    for t in range(n):
        arcs = col_arcs[t]
        if arcs.size == 0:
            if prev_states is not None:
                states[t], scores[t] = prev_states, prev_scores
                back[t] = np.arange(prev_states.size)
            continue
        em = _emissions(col_logp[t], cfg)
        frac = col_frac[t]
        if prev_states is None:
            cur = em.copy()
            bp = np.full(arcs.size, -1, dtype="int64")
        else:
            d = float(math.hypot(px[t] - px[prev_t], py[t] - py[prev_t]))
            dt = float(t_sec[t] - t_sec[prev_t])
            total = prev_scores[:, None] + _transitions(
                prev_states, prev_frac, arcs, frac, d, dt, neighbors_of,
                lengths, twin, cfg)
            bp = np.argmax(total, axis=0)
            cur = em + total[bp, np.arange(arcs.size)]
        states[t], scores[t], back[t] = arcs, cur, bp
        prev_states, prev_scores, prev_frac, prev_t = arcs, cur, frac, t

    path = np.full(n, -1, dtype="int64")
    last = max((t for t in range(n) if states[t] is not None), default=None)
    if last is None:
        return path
    idx = int(np.argmax(scores[last]))
    for t in range(last, -1, -1):
        if states[t] is None:
            break
        path[t] = states[t][idx]
        nxt = int(back[t][idx])
        if nxt < 0 and t > 0 and states[t - 1] is not None:
            nxt = int(np.argmax(scores[t - 1]))
        idx = nxt
    first = next((t for t in range(n) if path[t] >= 0), None)
    if first is not None:
        path[:first] = path[first]
    for t in range(1, n):
        if path[t] < 0:
            path[t] = path[t - 1]
    return path
