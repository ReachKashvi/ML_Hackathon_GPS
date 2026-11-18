"""Candidate arc generation for PathSnap (Step 6).

For every GPS observation this module finds the road arcs that could plausibly
have produced it: the arcs within ``R`` metres, ranked by exact perpendicular
distance, capped at ``K``.

Method
------
1. Project the GPS track and the graph nodes to the trip's own UTM CRS
   (``geo_utils.project_local`` -- the dataset is worldwide, so a fixed CRS
   would corrupt metre distances).
2. Build an R-tree over **arc bounding boxes**.
3. Query the index with a ``2R x 2R`` box around each GPS point. This is a
   cheap superset: a box can overlap while the segment itself is further
   than ``R`` away.
4. Compute the **exact** point-to-segment distance for that superset.
5. Keep arcs within ``R`` metres, sort by exact distance, return the nearest
   ``K``. Fewer than ``K`` qualifying arcs simply yields fewer rows.

Why a bounding-box index rather than nearest-neighbour on arc midpoints: arcs
vary from under a metre to several hundred metres long, so a midpoint-based
search silently misses long arcs that pass right next to the point.

What the numbers say about ``R``
--------------------------------
Step 4 measured, per trip, the fraction of GPS points whose ground-truth route
passes within 50 m (``frac_gps_within_50m_of_route``). The median trip is at
100%, but two training trips are not: ``00000054`` (12.7%) and ``00000033``
(41.0%) carry a systematic GPS-to-network offset of 100-250 m. For those
trips no value of ``K`` can help -- the true arc is simply not within ``R`` --
so their contribution to the miss rate is a property of the data, not of this
code. :func:`candidate_miss_rate` reports per-trip rates so that stays visible.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd
from rtree import index as rtree_index

import geo_utils as geo

__all__ = [
    "CANDIDATE_COLUMNS",
    "DEFAULT_K",
    "DEFAULT_R_M",
    "build_arc_index",
    "generate_candidates",
    "candidate_miss_rate",
]

#: Columns of the frame returned by :func:`generate_candidates`.
CANDIDATE_COLUMNS = (
    "point_idx",
    "arc_id",
    "perp_dist_m",
    "endpoint_dist_m",
    "candidate_rank",
)

DEFAULT_K = 10
DEFAULT_R_M = 50.0


def build_arc_index(
    arcs: pd.DataFrame,
    node_x: np.ndarray,
    node_y: np.ndarray,
) -> rtree_index.Index:
    """Build an R-tree over the bounding box of every arc.

    Parameters
    ----------
    arcs
        Arc table as returned by ``io_utils.load_arcs``.
    node_x, node_y
        Projected node coordinates in metres, indexed by node_id.

    Returns
    -------
    rtree.index.Index
        Index whose ids are ``arc_id``. Built by bulk stream loading, which
        is what keeps the 307k-arc graphs tractable -- inserting one at a
        time is roughly an order of magnitude slower.

    Notes
    -----
    Arcs with an out-of-range node reference are skipped rather than
    projected to garbage coordinates. The dataset currently has none, but
    the loader deliberately does not reject them, so this stays defensive.
    """
    a = arcs["from_node"].to_numpy()
    b = arcs["to_node"].to_numpy()
    n_nodes = node_x.size
    valid = (a >= 0) & (a < n_nodes) & (b >= 0) & (b < n_nodes)

    ax, ay = node_x[a], node_y[a]
    bx, by = node_x[b], node_y[b]
    minx = np.minimum(ax, bx)
    maxx = np.maximum(ax, bx)
    miny = np.minimum(ay, by)
    maxy = np.maximum(ay, by)
    finite = np.isfinite(minx) & np.isfinite(miny) & np.isfinite(maxx) & np.isfinite(maxy)
    keep = valid & finite

    def stream():
        for arc_id in np.flatnonzero(keep):
            yield (
                int(arc_id),
                (minx[arc_id], miny[arc_id], maxx[arc_id], maxy[arc_id]),
                None,
            )

    props = rtree_index.Property()
    props.dimension = 2
    return rtree_index.Index(stream(), properties=props)


def generate_candidates(
    trip_data: Mapping[str, Any],
    K: int = DEFAULT_K,
    R: float = DEFAULT_R_M,
) -> pd.DataFrame:
    """Find the ``K`` nearest arcs within ``R`` metres of each GPS point.

    Parameters
    ----------
    trip_data
        A trip dict as returned by ``io_utils.load_trip``.
    K
        Maximum candidates per GPS point. Fewer are returned where fewer
        arcs qualify; a point with no arc within ``R`` contributes no rows
        at all, which :func:`candidate_miss_rate` counts as a miss.
    R
        Search radius in metres.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`CANDIDATE_COLUMNS`:

        ``point_idx``        row index of the GPS observation
        ``arc_id``           candidate arc (trip-local id)
        ``perp_dist_m``      exact distance to the closest point *on the
                             segment*; equals the endpoint distance when the
                             perpendicular foot falls outside the segment
        ``endpoint_dist_m``  distance to the **nearer** of the two arc
                             endpoints, a cheap proxy for "am I near a
                             junction?"
        ``candidate_rank``   0 = nearest, ascending by ``perp_dist_m``

        Sorted by ``point_idx`` then ``candidate_rank``.

    Notes
    -----
    Also attaches ``frac_t`` (position along the arc, 0 at ``from_node``,
    1 at ``to_node``) as an extra column: Step 7 needs it for the
    distance-from-start/end features and recomputing it would mean a second
    pass over the same geometry.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if R <= 0:
        raise ValueError(f"R must be > 0, got {R}")

    arcs = trip_data["arcs"]
    proj = geo.project_trip(trip_data)
    px, py = proj["track_x"], proj["track_y"]
    node_x, node_y = proj["node_x"], proj["node_y"]

    a = arcs["from_node"].to_numpy()
    b = arcs["to_node"].to_numpy()
    ax, ay = node_x[a], node_y[a]
    bx, by = node_x[b], node_y[b]

    tree = build_arc_index(arcs, node_x, node_y)

    point_idx_out: list[np.ndarray] = []
    arc_id_out: list[np.ndarray] = []
    dist_out: list[np.ndarray] = []
    endpoint_out: list[np.ndarray] = []
    rank_out: list[np.ndarray] = []
    t_out: list[np.ndarray] = []

    for i in range(px.size):
        x, y = px[i], py[i]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue

        # Step 3: cheap bounding-box superset ...
        hits = np.fromiter(
            tree.intersection((x - R, y - R, x + R, y + R)),
            dtype="int64",
        )
        if hits.size == 0:
            continue

        # ... Step 4: exact geometry on the superset only.
        d, t = geo.point_to_segment_dist_m(
            x, y, ax[hits], ay[hits], bx[hits], by[hits], return_t=True
        )

        within = d <= R
        if not within.any():
            continue
        hits, d, t = hits[within], d[within], t[within]

        # Step 6/7: rank by exact distance, keep at most K.
        if d.size > K:
            part = np.argpartition(d, K - 1)[:K]
            hits, d, t = hits[part], d[part], t[part]
        order = np.argsort(d, kind="stable")
        hits, d, t = hits[order], d[order], t[order]

        d_a = np.hypot(x - ax[hits], y - ay[hits])
        d_b = np.hypot(x - bx[hits], y - by[hits])

        point_idx_out.append(np.full(d.size, i, dtype="int64"))
        arc_id_out.append(hits.astype("int64"))
        dist_out.append(d)
        endpoint_out.append(np.minimum(d_a, d_b))
        rank_out.append(np.arange(d.size, dtype="int64"))
        t_out.append(t)

    if not point_idx_out:
        return pd.DataFrame(
            {
                "point_idx": np.array([], dtype="int64"),
                "arc_id": np.array([], dtype="int64"),
                "perp_dist_m": np.array([], dtype="float64"),
                "endpoint_dist_m": np.array([], dtype="float64"),
                "candidate_rank": np.array([], dtype="int64"),
                "frac_t": np.array([], dtype="float64"),
            }
        )

    return pd.DataFrame(
        {
            "point_idx": np.concatenate(point_idx_out),
            "arc_id": np.concatenate(arc_id_out),
            "perp_dist_m": np.concatenate(dist_out),
            "endpoint_dist_m": np.concatenate(endpoint_out),
            "candidate_rank": np.concatenate(rank_out),
            "frac_t": np.concatenate(t_out),
        }
    )


def candidate_miss_rate(
    trip_data: Mapping[str, Any],
    candidates_df: pd.DataFrame,
    truth_arc_by_point: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Fraction of GPS points whose true arc is absent from their candidates.

    This is the **ceiling on achievable accuracy**: a point whose
    ground-truth arc never enters the candidate set can never be classified
    correctly, however good the model is.

    Ground truth comes from the route alignment in ``labels.py``, *not* from
    assuming ``route[i]`` belongs to GPS point ``i`` -- the two sequences are
    not one-to-one (trip ``00000001``: 1315 points, 525 route arcs).

    Parameters
    ----------
    trip_data
        A trip dict as returned by ``io_utils.load_trip``.
    candidates_df
        Output of :func:`generate_candidates` for the same trip.
    truth_arc_by_point
        Optional precomputed alignment: one ground-truth arc_id per GPS
        point, ``-1`` where the alignment declines to assign one. When
        omitted it is computed with ``labels.align_gps_to_route``.

    Returns
    -------
    dict
        ``trip_id``           str
        ``n_points``          GPS points in the trip
        ``n_aligned``         points the alignment assigned an arc to
        ``n_with_candidates`` points that received at least one candidate
        ``n_missed``          aligned points whose true arc is not among
                              their candidates
        ``miss_rate``         ``n_missed / n_aligned``
        ``n_no_candidates``   points with an empty candidate set
        ``mean_rank_of_truth``mean ``candidate_rank`` of the true arc, over
                              the points where it was found (1.0 would mean
                              the nearest arc is usually correct)
    """
    import labels  # local import: labels imports candidates-adjacent helpers

    trip_id = str(trip_data["trip_id"])
    n_points = len(trip_data["track"])

    if truth_arc_by_point is None:
        truth_arc_by_point = labels.align_gps_to_route(trip_data)["truth_arc_id"].to_numpy()
    truth = np.asarray(truth_arc_by_point, dtype="int64")

    aligned = truth >= 0
    n_aligned = int(aligned.sum())

    has_cand = np.zeros(n_points, dtype=bool)
    if len(candidates_df):
        has_cand[np.unique(candidates_df["point_idx"].to_numpy())] = True

    # A (point, arc) pair is a "hit" when the aligned truth arc is present.
    found_rank = np.full(n_points, -1, dtype="int64")
    if len(candidates_df):
        cand_pt = candidates_df["point_idx"].to_numpy()
        cand_arc = candidates_df["arc_id"].to_numpy()
        cand_rank = candidates_df["candidate_rank"].to_numpy()
        match = aligned[cand_pt] & (cand_arc == truth[cand_pt])
        found_rank[cand_pt[match]] = cand_rank[match]

    hit = aligned & (found_rank >= 0)
    n_missed = int(n_aligned - hit.sum())

    return {
        "trip_id": trip_id,
        "n_points": n_points,
        "n_aligned": n_aligned,
        "n_with_candidates": int(has_cand.sum()),
        "n_missed": n_missed,
        "miss_rate": float(n_missed / n_aligned) if n_aligned else float("nan"),
        "n_no_candidates": int(n_points - has_cand.sum()),
        "mean_rank_of_truth": float(found_rank[hit].mean()) if hit.any() else float("nan"),
    }
