"""HMM / Viterbi map matching over candidate arcs (Phase 4).

The classifier scores every (GPS point, candidate arc) pair independently, so
nothing stops it picking arc A at one fix and an unconnected arc B one second
later: route continuity of its argmax is ~0.90, i.e. about one consecutive
pick in ten is a jump no vehicle could make. Viterbi decoding chooses the
single arc *sequence* that best balances

* **emission** -- how strongly the classifier believes in each candidate,
  ``log(y_prob)``; and
* **transition** -- how plausible it is to get from the previous arc to this
  one, given how far the GPS actually moved.

Everything is in log space; probabilities are never multiplied.

Transition model
----------------
For consecutive GPS fixes ``t-1 -> t`` with straight-line displacement ``d``
and interval ``dt``, moving from arc ``a`` to arc ``b`` scores::

    log N(network_dist ; d, sigma_eff),  sigma_eff = sigma * max(1, sqrt(dt / max_gap_s))

capped to ``[-20, 0]``. An unreachable pair (no path within the Dijkstra
cutoff) gets the floor, ``-20``, rather than ``-inf``: the road graphs are
fragmented (median 19 components per trip), so a hard zero would make some
trips undecodable. ``sigma_eff`` widens with long gaps because the vehicle's
possible paths diverge the longer it is unobserved.

Position-aware network distance
-------------------------------
``hmm_graph`` measures arc-to-arc distance from the **end** of ``a`` to the
**end** of ``b``. The vehicle is somewhere *along* each arc, though, and Step 6
already records where (``frac_along_arc``). When positions are available the
distance actually driven is used instead::

    a == b : |t_b - t_a| * len(a)
    a != b : (1 - t_a) * len(a) + (network_dist - len(b)) + t_b * len(b)

On a 20 m arc the difference is small; on the 100 m+ arcs common outside city
centres it is most of the signal. Pass ``use_position=False`` to get the plain
arc-end formulation from the brief.

Route breaks
------------
At GPS points where a new contiguous segment of the ground-truth route begins
(``hmm_graph.load_route_segments``) the decoder **restarts**: emission only,
no transition, no backpointer. Note that this uses ``.route``, which exists for
training and student-test trips but will not exist for unseen data; callers
should report results both with and without it.

Points with no candidates
-------------------------
A GPS point with an empty candidate set (true arc not within R) cannot be
scored. The decoder **coasts**: the previous column's states are carried
forward unchanged, the point inherits the decoded arc, and it is flagged
``coasted``. The next real transition is then measured from the last point
that did have candidates, so the gap does not distort displacement.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

import geo_utils as geo
import hmm_graph as hg

__all__ = [
    "PROB_CLIP",
    "LOG_FLOOR",
    "DEFAULT_SIGMA",
    "DEFAULT_BETA",
    "DEFAULT_MAX_GAP_S",
    "build_trellis",
    "transition_log_prob",
    "reset_points_from_segments",
    "viterbi_decode",
    "decode_trip",
]

PROB_CLIP = 1e-6
LOG_FLOOR = -20.0
DEFAULT_SIGMA = 20.0
DEFAULT_BETA = 1.0
DEFAULT_MAX_GAP_S = 10.0
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


# --------------------------------------------------------------------------
# trellis
# --------------------------------------------------------------------------
def _trellis_columns(
    trip_id: str,
    pairs_df: pd.DataFrame,
    n_points: int,
    prob_col: str = "y_prob",
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Optional[np.ndarray]]]:
    """Per-point arrays of candidate arcs, log emissions and arc positions.

    Ordered by ``y_prob`` descending within each point, ties broken by
    ``candidate_rank`` (geometric nearest first) when that column exists --
    the same tiebreak ``eval_utils.top1_accuracy`` uses.
    """
    sub = pairs_df
    if "trip_id" in sub.columns:
        sub = sub[sub["trip_id"] == trip_id]

    pts = sub["point_idx"].to_numpy(dtype="int64")
    arcs = sub["arc_id"].to_numpy(dtype="int64")
    logp = np.log(np.clip(sub[prob_col].to_numpy(dtype="float64"),
                          PROB_CLIP, 1.0 - PROB_CLIP))
    frac = (sub["frac_along_arc"].to_numpy(dtype="float64")
            if "frac_along_arc" in sub.columns else None)
    rank = (sub["candidate_rank"].to_numpy(dtype="int64")
            if "candidate_rank" in sub.columns else np.zeros(len(sub), "int64"))

    # primary: point, then -logp, then rank (lexsort keys are last-primary)
    order = np.lexsort((rank, -logp, pts))
    pts, arcs, logp, rank = pts[order], arcs[order], logp[order], rank[order]
    if frac is not None:
        frac = frac[order]

    col_arcs: List[np.ndarray] = [np.empty(0, "int64")] * n_points
    col_logp: List[np.ndarray] = [np.empty(0, "float64")] * n_points
    col_frac: List[Optional[np.ndarray]] = [None] * n_points
    if len(pts):
        bounds = np.flatnonzero(np.diff(pts)) + 1
        starts = np.concatenate([[0], bounds])
        stops = np.concatenate([bounds, [len(pts)]])
        for s, e in zip(starts, stops):
            t = int(pts[s])
            if 0 <= t < n_points:
                col_arcs[t] = arcs[s:e]
                col_logp[t] = logp[s:e]
                col_frac[t] = frac[s:e] if frac is not None else None
    return col_arcs, col_logp, col_frac


def build_trellis(
    trip_id: str,
    pairs_df: pd.DataFrame,
    trip_data: Mapping[str, Any],
    prob_col: str = "y_prob",
) -> List[List[Tuple[int, float]]]:
    """Candidate lattice for one trip.

    Parameters
    ----------
    trip_id
        Trip to extract from ``pairs_df`` (ignored if the frame has no
        ``trip_id`` column).
    pairs_df
        Rows with ``point_idx``, ``arc_id`` and the probability column.
    trip_data
        The trip dict; used for the number of GPS points.

    Returns
    -------
    list of list of (arc_id, log_emission)
        One inner list per GPS point, in ``point_idx`` order, sorted by
        probability descending. Points with no candidates get ``[]``.
        Probabilities are clipped to ``[1e-6, 1 - 1e-6]`` before the log.
    """
    n = len(trip_data["track"])
    arcs, logp, _ = _trellis_columns(str(trip_id), pairs_df, n, prob_col)
    return [list(zip(a.tolist(), lp.tolist())) for a, lp in zip(arcs, logp)]


# --------------------------------------------------------------------------
# transitions
# --------------------------------------------------------------------------
def transition_log_prob(
    arc_a: int,
    arc_b: int,
    gps_disp_m: float,
    dt_s: float,
    neighbors_dict: Mapping[int, float],
    sigma_trans: float = DEFAULT_SIGMA,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
    along_correction_m: float = 0.0,
) -> float:
    """Log probability of moving from ``arc_a`` to ``arc_b`` between two fixes.

    Parameters
    ----------
    arc_a, arc_b
        Previous and current arc.
    gps_disp_m
        Straight-line distance the GPS moved between the two fixes.
    dt_s
        Seconds between the two fixes.
    neighbors_dict
        ``hmm_graph.precompute_neighbors`` output for ``arc_a``.
    sigma_trans
        Standard deviation of the mismatch between road distance and GPS
        displacement, in metres.
    max_gap_s
        Gaps longer than this widen ``sigma`` by ``sqrt(dt / max_gap_s)``.
    along_correction_m
        Added to the arc-end network distance to account for where on each
        arc the vehicle actually is (see module docstring). 0 reproduces the
        plain arc-end model.

    Returns
    -------
    float
        Log probability in ``[-20, 0]``.
    """
    if int(arc_a) == int(arc_b):
        network = 0.0
    else:
        network = neighbors_dict.get(int(arc_b), math.inf)
    if not math.isfinite(network):
        return LOG_FLOOR
    network = max(0.0, network + along_correction_m)

    sigma_eff = sigma_trans * max(1.0, math.sqrt(max(dt_s, 0.0) / max_gap_s))
    log_prob = (
        -((network - gps_disp_m) ** 2) / (2.0 * sigma_eff ** 2)
        - math.log(sigma_eff) - _LOG_SQRT_2PI
    )
    return float(min(0.0, max(LOG_FLOOR, log_prob)))


def _transition_matrix(
    prev_arcs: np.ndarray,
    prev_frac: Optional[np.ndarray],
    cur_arcs: np.ndarray,
    cur_frac: Optional[np.ndarray],
    gps_disp_m: float,
    dt_s: float,
    neighbors_of,
    lengths: np.ndarray,
    sigma_trans: float,
    max_gap_s: float,
    use_position: bool,
) -> np.ndarray:
    """Vectorised :func:`transition_log_prob` over all (prev, cur) pairs."""
    P, C = len(prev_arcs), len(cur_arcs)
    network = np.empty((P, C), dtype="float64")
    for i, a in enumerate(prev_arcs.tolist()):
        nb = neighbors_of(a)
        network[i] = [nb.get(b, math.inf) for b in cur_arcs.tolist()]
    same = prev_arcs[:, None] == cur_arcs[None, :]
    network[same] = 0.0

    if use_position and prev_frac is not None and cur_frac is not None:
        len_a = lengths[prev_arcs][:, None]
        len_b = lengths[cur_arcs][None, :]
        ta = prev_frac[:, None]
        tb = cur_frac[None, :]
        corr_diff = (1.0 - ta) * len_a - (1.0 - tb) * len_b
        corr_same = np.abs(tb - ta) * len_a
        corr = np.where(same, corr_same, corr_diff)
        corr = np.where(np.isfinite(corr), corr, 0.0)
        network = network + corr

    reachable = np.isfinite(network)
    network = np.where(reachable, np.maximum(network, 0.0), 0.0)
    sigma_eff = sigma_trans * max(1.0, math.sqrt(max(dt_s, 0.0) / max_gap_s))
    logp = (-((network - gps_disp_m) ** 2) / (2.0 * sigma_eff ** 2)
            - math.log(sigma_eff) - _LOG_SQRT_2PI)
    logp = np.clip(logp, LOG_FLOOR, 0.0)
    return np.where(reachable, logp, LOG_FLOOR)


# --------------------------------------------------------------------------
# route breaks
# --------------------------------------------------------------------------
def reset_points_from_segments(
    route_segments: Sequence[Tuple[int, int]],
    trip_data: Mapping[str, Any],
    alignment: Optional[pd.DataFrame] = None,
) -> Set[int]:
    """GPS point indices at which a new route segment begins.

    ``route_segments`` index the ``.route`` array, not the GPS track, so the
    monotone alignment from ``labels.align_gps_to_route`` is used to find, for
    each segment start ``s``, the first GPS point aligned at or beyond ``s``.
    """
    if len(route_segments) <= 1:
        return set()
    if alignment is None:
        import labels  # local: only needed for break trips
        alignment = labels.align_gps_to_route(trip_data)
    pos = alignment.sort_values("point_idx")["route_pos_raw"].to_numpy()
    resets: Set[int] = set()
    for start, _ in route_segments[1:]:
        t = int(np.searchsorted(pos, start, side="left"))
        if 0 < t < len(pos):
            resets.add(t)
    return resets


# --------------------------------------------------------------------------
# Viterbi
# --------------------------------------------------------------------------
def _viterbi(
    col_arcs: List[np.ndarray],
    col_logp: List[np.ndarray],
    col_frac: List[Optional[np.ndarray]],
    trip_data: Mapping[str, Any],
    graph: nx.DiGraph,
    reset_points: Set[int],
    sigma_trans: float,
    beta: float,
    max_gap_s: float,
    use_position: bool,
    cutoff_m: float,
) -> Dict[str, np.ndarray]:
    """Core decoder; returns arcs, per-point path log prob and coast flags."""
    trip_id = str(trip_data["trip_id"])
    n = len(col_arcs)
    proj = geo.project_trip(trip_data)
    px, py = proj["track_x"], proj["track_y"]
    t_sec = trip_data["track"]["t_sec"].to_numpy(dtype="float64")
    lengths = hg.arc_lengths_m(trip_data)

    def neighbors_of(arc: int) -> Mapping[int, float]:
        return hg.precompute_neighbors(graph, arc, cutoff_m=cutoff_m, trip_id=trip_id)

    states: List[Optional[np.ndarray]] = [None] * n
    scores: List[Optional[np.ndarray]] = [None] * n
    back: List[Optional[np.ndarray]] = [None] * n
    coasted = np.zeros(n, dtype=bool)

    prev_t_real: Optional[int] = None          # last point with candidates
    prev_states = prev_scores = prev_frac = None

    for t in range(n):
        arcs = col_arcs[t]
        if arcs.size == 0:
            coasted[t] = True
            if prev_states is not None:
                states[t], scores[t] = prev_states, prev_scores
                back[t] = np.arange(prev_states.size)
            continue

        em = col_logp[t]
        frac = col_frac[t]
        if prev_states is None or t in reset_points:
            cur = em.copy()
            bp = np.full(arcs.size, -1, dtype="int64")
        else:
            d = float(math.hypot(px[t] - px[prev_t_real], py[t] - py[prev_t_real]))
            dt = float(t_sec[t] - t_sec[prev_t_real])
            trans = _transition_matrix(
                prev_states, prev_frac, arcs, frac, d, dt, neighbors_of,
                lengths, sigma_trans, max_gap_s, use_position,
            )
            total = prev_scores[:, None] + beta * trans
            bp = np.argmax(total, axis=0)
            cur = em + total[bp, np.arange(arcs.size)]

        states[t], scores[t], back[t] = arcs, cur, bp
        prev_states, prev_scores, prev_frac = arcs, cur, frac
        prev_t_real = t

    path = np.full(n, -1, dtype="int64")
    path_logp = np.full(n, np.nan)
    last = max((t for t in range(n) if states[t] is not None), default=None)
    if last is None:
        return {"arcs": path, "log_prob": path_logp, "coasted": coasted}

    idx = int(np.argmax(scores[last]))
    for t in range(last, -1, -1):
        if states[t] is None:
            break
        path[t] = states[t][idx]
        path_logp[t] = scores[t][idx]
        nxt = int(back[t][idx])
        if nxt < 0 and t > 0 and states[t - 1] is not None:
            # segment start: the previous segment ends on its own best state
            nxt = int(np.argmax(scores[t - 1]))
        idx = nxt

    # leading points before the first candidate inherit the first decoded arc
    first = next((t for t in range(n) if path[t] >= 0), None)
    if first is not None:
        path[:first] = path[first]
        coasted[:first] = True
    # trailing coasted points already carry states; fill any gap defensively
    for t in range(1, n):
        if path[t] < 0:
            path[t] = path[t - 1]
            coasted[t] = True
    return {"arcs": path, "log_prob": path_logp, "coasted": coasted}


def viterbi_decode(
    trellis: List[List[Tuple[int, float]]],
    trip_data: Mapping[str, Any],
    graph: nx.DiGraph,
    route_segments: Sequence[Tuple[int, int]],
    sigma_trans: float = DEFAULT_SIGMA,
    beta: float = DEFAULT_BETA,
    *,
    positions: Optional[List[Optional[np.ndarray]]] = None,
    alignment: Optional[pd.DataFrame] = None,
    use_breaks: bool = True,
    use_position: bool = True,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
    cutoff_m: float = hg.DEFAULT_CUTOFF_M,
) -> List[int]:
    """Most probable arc sequence for one trip.

    Parameters
    ----------
    trellis
        Output of :func:`build_trellis`.
    trip_data, graph
        The trip and its ``hmm_graph.build_arc_graph``.
    route_segments
        ``hmm_graph.load_route_segments`` output; segment starts become
        forced restarts when ``use_breaks`` is True.
    sigma_trans, beta
        Transition spread (m) and transition weight relative to emission.
    positions
        Optional per-point ``frac_along_arc`` arrays parallel to the
        trellis; enables the position-aware distance.

    Returns
    -------
    list of int
        One arc_id per GPS point, ``len == n_points``.
    """
    col_arcs = [np.asarray([a for a, _ in col], dtype="int64") for col in trellis]
    col_logp = [np.asarray([lp for _, lp in col], dtype="float64") for col in trellis]
    col_frac = positions if positions is not None else [None] * len(trellis)
    resets = (reset_points_from_segments(route_segments, trip_data, alignment)
              if use_breaks else set())
    out = _viterbi(col_arcs, col_logp, col_frac, trip_data, graph, resets,
                   sigma_trans, beta, max_gap_s,
                   use_position and positions is not None, cutoff_m)
    return out["arcs"].tolist()


def decode_trip(
    trip_id: str,
    pairs_df: pd.DataFrame,
    trip_data: Mapping[str, Any],
    sigma_trans: float = DEFAULT_SIGMA,
    beta: float = DEFAULT_BETA,
    *,
    use_breaks: bool = True,
    use_position: bool = True,
    alignment: Optional[pd.DataFrame] = None,
    prob_col: str = "y_prob",
    max_gap_s: float = DEFAULT_MAX_GAP_S,
    cutoff_m: float = hg.DEFAULT_CUTOFF_M,
) -> pd.DataFrame:
    """Decode one trip end to end.

    Returns
    -------
    pandas.DataFrame
        One row per GPS point: ``point_idx``, ``arc_id_hmm``,
        ``log_prob_hmm`` (cumulative path log probability at that point, NaN
        for coasted points before any candidate), ``coasted``.
    """
    n = len(trip_data["track"])
    graph = hg.build_arc_graph(trip_data)
    col_arcs, col_logp, col_frac = _trellis_columns(str(trip_id), pairs_df, n, prob_col)
    resets: Set[int] = set()
    if use_breaks:
        segments = hg.load_route_segments(trip_data)
        resets = reset_points_from_segments(segments, trip_data, alignment)
    has_pos = all(f is not None for f, a in zip(col_frac, col_arcs) if a.size)
    out = _viterbi(col_arcs, col_logp, col_frac, trip_data, graph, resets,
                   sigma_trans, beta, max_gap_s, use_position and has_pos, cutoff_m)
    return pd.DataFrame({
        "point_idx": np.arange(n, dtype="int64"),
        "arc_id_hmm": out["arcs"],
        "log_prob_hmm": out["log_prob"],
        "coasted": out["coasted"],
    })
