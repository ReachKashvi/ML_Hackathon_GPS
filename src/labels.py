"""Ground-truth labelling by route alignment (Step 8).

The problem
-----------
``.route`` is an *ordered* sequence of traversed arcs, but it is **not**
one-to-one with the GPS track. Trip ``00000001`` has 1315 GPS points and 525
route arcs; across the dataset the ratio runs from 1.3 to 57.0 points per arc.
Assigning ``route[i]`` to GPS point ``i`` would therefore be wrong for almost
every point of almost every trip.

The alignment algorithm
-----------------------
A **monotone sequence alignment of GPS points onto route arc-length**, solved
exactly by dynamic programming.

1. *Build the reference geometry.* Walk the ordered route through ``.nodes``
   to get one segment per route position, and give each position a cumulative
   along-route coordinate ``s`` in metres. The route is split into contiguous
   chains first (``io_utils.route_segments``), because 11 trips have genuine
   breaks of 10-600 m; ``s`` accumulates arc lengths only, so a gap never
   contributes phantom travelled distance.

2. *Project.* GPS points and route nodes go into the trip's own UTM CRS
   (``geo_utils.project_local``).

3. *Score every plausible (point, route position) pairing.* For each GPS
   point, the ``M`` nearest route positions are found with a KD-tree over
   segment endpoints and midpoints, then scored by exact point-to-segment
   distance. A sparse candidate set keeps the DP at ``O(n_points * M)``
   instead of ``O(n_points * n_route_arcs)``, which matters for the 19k-point
   trips.

4. *Enforce monotone progress.* The DP chooses one route position per GPS
   point minimising total perpendicular distance, subject to the chosen
   position never decreasing::

       cost[i][j] = dist(i, j) + min over j' <= j of cost[i-1][j']

   The inner minimum is a prefix minimum, so each step is linear in ``M``.
   Monotonicity is what stops a GPS point being matched to an earlier part
   of the route when the route passes near itself.

5. *Assign the arc.* Each GPS point takes the arc at its chosen route
   position. Points whose best distance exceeds ``max_assign_dist_m`` are
   left **unassigned** (``truth_arc_id = -1``) rather than forced onto an
   implausible arc.

6. *Label.* A candidate row is ``1`` when its ``arc_id`` equals the aligned
   truth arc for that point, ``0`` otherwise. Unassigned points get all-zero
   labels and are reported separately, never silently dropped.

Failure modes, and what is done about each
------------------------------------------
**Long stops on one arc.** A vehicle idling at a light emits hundreds of fixes
that all belong to one arc. *Risk*: an alignment that rewards forward progress
drags the match along the route while the vehicle is stationary. *Mitigation*:
the cost function contains **no forward-progress reward and no penalty for
staying put** -- it is pure perpendicular distance under a non-decreasing
constraint, so repeated points at the same place keep the same position.
GPS jitter while stopped moves the match by at most the jitter.

**Nearby loop-backs.** Where the route passes close to itself, or the vehicle
drives the same roads repeatedly, a nearest-arc rule flips between the two
passes. *Mitigation*: monotonicity makes a backward jump impossible, so a
spurious flip to an earlier pass cannot happen. The residual risk is the
opposite error, and it is real: 11 trips have a GPS/route length ratio above 2
(trip ``00000002``: ratio 7.0, 27.6 points per arc) which means the track laps
a route stored as a simple path. For those trips the monotone constraint will
push later laps forward onto route positions the vehicle reaches only later,
producing systematically wrong labels near the end of each lap.
:func:`alignment_report` surfaces this as a rising residual distance, and the
``length_ratio_gps_route`` column from Step 4 identifies the affected trips in
advance. They are kept and flagged rather than silently trusted.

**Points more than 50 m off-route.** Two training trips carry a systematic
GPS-to-network offset (``00000054``: 12.7% of points within 50 m of their
route; ``00000033``: 41.0%). *Mitigation*: ``max_assign_dist_m`` leaves those
points unassigned instead of inventing a label, and the per-trip
``n_unassigned`` count makes the loss explicit. A forced label there would
teach the model that a 140 m perpendicular distance is a positive match --
far more damaging than a missing label.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import geo_utils as geo
import io_utils as io

__all__ = [
    "DEFAULT_MAX_ASSIGN_DIST_M",
    "DEFAULT_M_CANDIDATES",
    "build_route_geometry",
    "align_gps_to_route",
    "label_candidates",
    "alignment_report",
]

#: Beyond this perpendicular distance a GPS point is left unassigned.
DEFAULT_MAX_ASSIGN_DIST_M = 100.0

#: Route positions scored per GPS point before the DP.
DEFAULT_M_CANDIDATES = 20


def build_route_geometry(trip_data: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """Ordered route geometry with a cumulative along-route coordinate.

    Parameters
    ----------
    trip_data
        A trip dict as returned by ``io_utils.load_trip``.

    Returns
    -------
    dict
        ``arc_id``   : arc at each route position, in route order.
        ``ax, ay``   : projected start coordinates per position (m).
        ``bx, by``   : projected end coordinates per position (m).
        ``length_m`` : arc length per position (m).
        ``s_start_m``: cumulative along-route distance at the position start.
        ``chain_id`` : which contiguous chain the position belongs to.

    Notes
    -----
    ``s_start_m`` accumulates arc lengths **continuously across chains**: a
    break contributes nothing, so the gap is not counted as travel, but the
    coordinate stays globally increasing, which is exactly what the monotone
    DP needs.
    """
    arcs = trip_data["arcs"]
    route = np.asarray(trip_data["route"], dtype="int64")
    proj = geo.project_trip(trip_data)
    node_x, node_y = proj["node_x"], proj["node_y"]

    n_arcs = len(arcs)
    valid = (route >= 0) & (route < n_arcs)
    route = route[valid]

    chains = io.route_segments(route, arcs)
    chain_id = np.concatenate(
        [np.full(len(c), k, dtype="int64") for k, c in enumerate(chains)]
    ) if chains else np.array([], dtype="int64")

    a = arcs["from_node"].to_numpy()[route]
    b = arcs["to_node"].to_numpy()[route]
    ax, ay = node_x[a], node_y[a]
    bx, by = node_x[b], node_y[b]
    length = np.hypot(bx - ax, by - ay)

    s_start = np.concatenate([[0.0], np.cumsum(length)[:-1]]) if length.size else length

    return {
        "arc_id": route,
        "ax": ax, "ay": ay, "bx": bx, "by": by,
        "length_m": length,
        "s_start_m": s_start,
        "chain_id": chain_id,
    }


def align_gps_to_route(
    trip_data: Mapping[str, Any],
    max_assign_dist_m: float = DEFAULT_MAX_ASSIGN_DIST_M,
    m_candidates: int = DEFAULT_M_CANDIDATES,
) -> pd.DataFrame:
    """Align each GPS point to a position along the ordered route.

    Implements the monotone dynamic programme described in the module
    docstring. See there for the failure modes this does and does not
    survive.

    Parameters
    ----------
    trip_data
        A trip dict as returned by ``io_utils.load_trip``.
    max_assign_dist_m
        Perpendicular distance beyond which a point is left unassigned.
    m_candidates
        Route positions scored per GPS point.

    Returns
    -------
    pandas.DataFrame
        One row per GPS point, columns:

        ``point_idx``     GPS row index.
        ``route_pos``     chosen position in the route sequence, -1 if none.
        ``truth_arc_id``  arc at that position, **-1 when unassigned**.
        ``align_dist_m``  perpendicular distance to the chosen position.
        ``route_s_m``     along-route coordinate of the match, metres.
        ``assigned``      bool, convenience for filtering.
    """
    track = trip_data["track"]
    n = len(track)
    geom = build_route_geometry(trip_data)
    n_pos = geom["arc_id"].size

    empty = pd.DataFrame(
        {
            "point_idx": np.arange(n, dtype="int64"),
            "route_pos_raw": np.full(n, -1, dtype="int64"),
            "route_pos": np.full(n, -1, dtype="int64"),
            "truth_arc_id": np.full(n, -1, dtype="int64"),
            "align_dist_m": np.full(n, np.nan),
            "route_s_m": np.full(n, np.nan),
            "assigned": np.zeros(n, dtype=bool),
        }
    )
    if n == 0 or n_pos == 0:
        return empty

    proj = geo.project_trip(trip_data)
    px, py = proj["track_x"], proj["track_y"]
    ax, ay, bx, by = geom["ax"], geom["ay"], geom["bx"], geom["by"]

    # --- step 3: sparse candidate route positions per GPS point -----------
    reps = np.column_stack([
        np.concatenate([ax, bx, (ax + bx) / 2.0]),
        np.concatenate([ay, by, (ay + by) / 2.0]),
    ])
    owner = np.tile(np.arange(n_pos), 3)
    tree = cKDTree(reps)
    k = min(m_candidates * 3, len(reps))
    _, nn = tree.query(np.column_stack([px, py]), k=k, workers=-1)
    nn = np.atleast_2d(nn)

    cand_pos = np.empty((n, m_candidates), dtype="int64")
    cand_dist = np.empty((n, m_candidates), dtype="float64")
    cand_t = np.empty((n, m_candidates), dtype="float64")

    for i in range(n):
        pos = np.unique(owner[nn[i]])
        d, t = geo.point_to_segment_dist_m(
            px[i], py[i], ax[pos], ay[pos], bx[pos], by[pos], return_t=True
        )
        if pos.size > m_candidates:
            sel = np.argpartition(d, m_candidates - 1)[:m_candidates]
            pos, d, t = pos[sel], d[sel], t[sel]
        order = np.argsort(pos, kind="stable")       # DP needs sorted positions
        pos, d, t = pos[order], d[order], t[order]
        pad = m_candidates - pos.size
        if pad > 0:
            # Pad with the last position at infinite cost: never selected,
            # keeps the arrays rectangular and the DP branch-free.
            pos = np.concatenate([pos, np.full(pad, pos[-1] if pos.size else 0)])
            d = np.concatenate([d, np.full(pad, np.inf)])
            t = np.concatenate([t, np.full(pad, 0.0)])
        cand_pos[i], cand_dist[i], cand_t[i] = pos, d, t

    # --- step 4: monotone DP ---------------------------------------------
    INF = np.inf
    cost = cand_dist[0].copy()
    back = np.full((n, m_candidates), -1, dtype="int32")

    for i in range(1, n):
        prev_pos = cand_pos[i - 1]
        prev_cost = cost
        # prefix minimum over the previous row, which is sorted by position
        pref_min = np.minimum.accumulate(prev_cost)
        pref_arg = np.zeros(m_candidates, dtype="int32")
        best = INF
        for kk in range(m_candidates):
            if prev_cost[kk] < best:
                best = prev_cost[kk]
                pref_arg[kk] = kk
            else:
                pref_arg[kk] = pref_arg[kk - 1] if kk else kk

        # for each current position, the best previous position <= it
        slot = np.searchsorted(prev_pos, cand_pos[i], side="right") - 1
        ok = slot >= 0
        add = np.where(ok, pref_min[np.clip(slot, 0, m_candidates - 1)], INF)
        cost = cand_dist[i] + add
        back[i] = np.where(ok, pref_arg[np.clip(slot, 0, m_candidates - 1)], -1)

        if not np.isfinite(cost).any():
            # The monotone constraint has painted us into a corner (can happen
            # when the track jumps far backwards). Restart the DP from here
            # rather than returning nothing, and record the discontinuity.
            cost = cand_dist[i].copy()
            back[i] = -1

    # --- backtrack ---------------------------------------------------------
    chosen = np.full(n, -1, dtype="int64")
    kk = int(np.argmin(cost)) if np.isfinite(cost).any() else -1
    for i in range(n - 1, -1, -1):
        if kk < 0:
            kk = int(np.argmin(cand_dist[i]))
        chosen[i] = kk
        kk = int(back[i][kk])

    rows = np.arange(n)
    route_pos = cand_pos[rows, chosen]
    dist = cand_dist[rows, chosen]
    frac = cand_t[rows, chosen]

    assigned = np.isfinite(dist) & (dist <= max_assign_dist_m)
    out = pd.DataFrame(
        {
            "point_idx": np.arange(n, dtype="int64"),
            # raw position is kept even when unassigned: it is what the
            # monotonicity sanity check in alignment_report inspects
            "route_pos_raw": route_pos.astype("int64"),
            "route_pos": np.where(assigned, route_pos, -1).astype("int64"),
            "truth_arc_id": np.where(assigned, geom["arc_id"][route_pos], -1).astype("int64"),
            "align_dist_m": dist,
            "route_s_m": geom["s_start_m"][route_pos] + frac * geom["length_m"][route_pos],
            "assigned": assigned,
        }
    )
    return out


def label_candidates(
    trip_data: Mapping[str, Any],
    candidates_df: pd.DataFrame,
    alignment: pd.DataFrame | None = None,
    max_assign_dist_m: float = DEFAULT_MAX_ASSIGN_DIST_M,
) -> pd.DataFrame:
    """Attach a binary ``label`` to every candidate row.

    Parameters
    ----------
    trip_data
        A trip dict as returned by ``io_utils.load_trip``.
    candidates_df
        Output of ``candidates.generate_candidates`` for the same trip.
    alignment
        Optional precomputed :func:`align_gps_to_route` result, so callers
        that already aligned the trip do not pay for it twice.
    max_assign_dist_m
        Passed through when the alignment has to be computed.

    Returns
    -------
    pandas.DataFrame
        ``candidates_df`` with three columns appended:

        ``label``          1 when this arc is the aligned ground truth.
        ``truth_arc_id``   the aligned arc for the point, -1 if unassigned.
        ``point_assigned`` False for points the alignment declined.

        Row order and all input columns are preserved.
    """
    if alignment is None:
        alignment = align_gps_to_route(trip_data, max_assign_dist_m=max_assign_dist_m)

    if len(candidates_df) == 0:
        out = candidates_df.copy()
        for col, dtype in (("label", "int8"), ("truth_arc_id", "int64"),
                           ("point_assigned", "bool")):
            out[col] = pd.Series(dtype=dtype)
        return out

    truth_by_point = alignment.set_index("point_idx")["truth_arc_id"]
    assigned_by_point = alignment.set_index("point_idx")["assigned"]

    out = candidates_df.copy()
    pt = out["point_idx"].to_numpy()
    truth = truth_by_point.reindex(pt).to_numpy()
    assigned = assigned_by_point.reindex(pt).fillna(False).to_numpy().astype(bool)
    truth = np.where(np.isfinite(pd.to_numeric(truth, errors="coerce")), truth, -1).astype("int64")

    out["truth_arc_id"] = truth
    out["point_assigned"] = assigned
    out["label"] = ((out["arc_id"].to_numpy() == truth) & assigned).astype("int8")
    return out


def alignment_report(
    trip_data: Mapping[str, Any],
    alignment: pd.DataFrame | None = None,
) -> Dict[str, Any]:
    """Diagnostics for one trip's alignment.

    The residual distance and the monotone-progress profile are what reveal
    the loop-back failure mode described in the module docstring: on a
    lapping trip the residual grows steadily through each lap instead of
    staying flat.

    Returns
    -------
    dict
        ``trip_id``, ``n_points``, ``n_assigned``, ``n_unassigned``,
        ``frac_assigned``, ``align_dist_median_m``, ``align_dist_p95_m``,
        ``align_dist_max_m``, ``n_distinct_arcs_used``, ``n_route_arcs``,
        ``frac_route_arcs_used``, ``route_pos_monotone`` (sanity check on
        the DP's non-decreasing constraint; the only legitimate False is a
        trip where the DP had to restart after the constraint became
        infeasible, which is reported by ``n_dp_restarts``).
    """
    if alignment is None:
        alignment = align_gps_to_route(trip_data)

    d = alignment["align_dist_m"].to_numpy()
    assigned = alignment["assigned"].to_numpy()
    truth = alignment.loc[assigned, "truth_arc_id"].to_numpy()
    pos_raw = alignment["route_pos_raw"].to_numpy()
    n_route = int(np.asarray(trip_data["route"]).size)

    return {
        "trip_id": str(trip_data["trip_id"]),
        "n_points": len(alignment),
        "n_assigned": int(assigned.sum()),
        "n_unassigned": int((~assigned).sum()),
        "frac_assigned": float(assigned.mean()) if len(alignment) else float("nan"),
        "align_dist_median_m": float(np.nanmedian(d)) if len(d) else float("nan"),
        "align_dist_p95_m": float(np.nanpercentile(d, 95)) if len(d) else float("nan"),
        "align_dist_max_m": float(np.nanmax(d)) if len(d) else float("nan"),
        "n_distinct_arcs_used": int(np.unique(truth).size),
        "n_route_arcs": n_route,
        "frac_route_arcs_used": float(np.unique(truth).size / n_route) if n_route else float("nan"),
        "route_pos_monotone": bool(np.all(np.diff(pos_raw) >= 0)),
        "n_dp_restarts": int(np.count_nonzero(np.diff(pos_raw) < 0)),
    }
