"""Arc-level road graph and reachability for the HMM stage (Phase 4 groundwork).

The per-point classifier scores each (GPS point, arc) pair independently, so
nothing stops it choosing arc A at one fix and a disconnected arc B a second
later. Measured route continuity is 0.89-0.90, i.e. roughly **one in ten
consecutive picks is a physically impossible jump**. Viterbi decoding fixes
exactly that, and it needs two things this module provides:

* a graph whose **nodes are arcs** (not junctions), so a transition is
  "could the vehicle have gone from arc A to arc B?";
* **road distance** between two arcs, which is what a transition probability
  is a function of -- a 30 m hop is plausible between consecutive 1 Hz
  fixes, a 1500 m one is not.

Why arcs as nodes
-----------------
The HMM's hidden state is *which arc the vehicle is on*. Modelling junctions
as nodes would force a second mapping at every step; with arcs as nodes a
transition is a single edge lookup. Edge ``A -> B`` exists when
``arcs[A].to_node == arcs[B].from_node``, and its weight is the **length of
B**, so a path's cost is the distance actually travelled along it after
leaving A.

Route breaks
------------
11 of the 90 trips contain genuine discontinuities in the ground-truth chain
(gaps of 10-600 m, established in Phase 1). :func:`load_route_segments`
returns the contiguous stretches as index ranges so the decoder can treat a
break as a **forced restart** -- no transition penalty across a gap, because
there is no road connecting the two sides and penalising the jump would push
the decoder into inventing a detour that never happened.

Caching
-------
Graph construction and neighbour expansion are memoised per ``trip_id`` in
module-level dicts. A Viterbi pass queries the same arc's neighbourhood once
per candidate per time step, so recomputing would dominate runtime.
Call :func:`clear_caches` when processing many trips in one process.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

import networkx as nx
import numpy as np
import pandas as pd

import io_utils as io

__all__ = [
    "DEFAULT_CUTOFF_M",
    "build_arc_graph",
    "arc_lengths_m",
    "shortest_arc_distance",
    "precompute_neighbors",
    "load_route_segments",
    "clear_caches",
    "cache_stats",
]

#: Transitions beyond this road distance are treated as unreachable. At the
#: dataset's 1 Hz sampling and a 95th-percentile speed near 20 m/s, 2 km
#: covers even a 100 s gap; searching further costs time for paths no
#: vehicle could have taken between two fixes.
DEFAULT_CUTOFF_M = 2000.0

_GRAPH_CACHE: Dict[str, nx.DiGraph] = {}
_NEIGHBOR_CACHE: Dict[str, Dict[int, Dict[int, float]]] = {}
_LENGTH_CACHE: Dict[str, np.ndarray] = {}


def arc_lengths_m(trip_data: Mapping[str, Any]) -> np.ndarray:
    """Haversine length of every arc in metres, indexed by ``arc_id``.

    Haversine rather than the projected length so the weights do not depend
    on the per-trip UTM choice; over a 20 m arc the two agree to well under
    a centimetre.
    """
    trip_id = str(trip_data["trip_id"])
    cached = _LENGTH_CACHE.get(trip_id)
    if cached is not None:
        return cached

    arcs, nodes = trip_data["arcs"], trip_data["nodes"]
    lon = nodes["lon"].to_numpy()
    lat = nodes["lat"].to_numpy()
    a = arcs["from_node"].to_numpy()
    b = arcs["to_node"].to_numpy()
    n = len(nodes)
    ok = (a >= 0) & (a < n) & (b >= 0) & (b < n)

    lengths = np.full(len(arcs), np.inf)
    lengths[ok] = io._haversine_m(lon[a[ok]], lat[a[ok]], lon[b[ok]], lat[b[ok]])
    _LENGTH_CACHE[trip_id] = lengths
    return lengths


def build_arc_graph(trip_data: Mapping[str, Any]) -> nx.DiGraph:
    """Build the arc-adjacency graph for one trip.

    Parameters
    ----------
    trip_data
        A trip dict as returned by ``io_utils.load_trip``.

    Returns
    -------
    networkx.DiGraph
        Nodes are ``arc_id``s. An edge ``A -> B`` exists when
        ``arcs[A].to_node == arcs[B].from_node``; its ``weight`` is the
        length of **B** in metres, so summing weights along a path gives the
        distance travelled after leaving A.

    Notes
    -----
    Built by grouping arcs on ``from_node`` and joining, rather than by
    nested loops over arc pairs: the largest trip has 307k arcs, where the
    pairwise approach is quadratic and hopeless. Arcs with an out-of-range
    node reference are skipped.

    Memoised per ``trip_id``.
    """
    trip_id = str(trip_data["trip_id"])
    cached = _GRAPH_CACHE.get(trip_id)
    if cached is not None:
        return cached

    arcs = trip_data["arcs"]
    n_nodes = len(trip_data["nodes"])
    from_node = arcs["from_node"].to_numpy()
    to_node = arcs["to_node"].to_numpy()
    valid = (
        (from_node >= 0) & (from_node < n_nodes)
        & (to_node >= 0) & (to_node < n_nodes)
    )
    lengths = arc_lengths_m(trip_data)

    arc_ids = np.flatnonzero(valid)
    # successors of arc A = every arc starting at A's to_node
    starts = pd.DataFrame({"arc_id": arc_ids, "from_node": from_node[arc_ids]})
    ends = pd.DataFrame({"arc_id": arc_ids, "to_node": to_node[arc_ids]})
    joined = ends.merge(
        starts, left_on="to_node", right_on="from_node",
        suffixes=("_a", "_b"), how="inner",
    )

    graph = nx.DiGraph()
    graph.add_nodes_from(arc_ids.tolist())
    if len(joined):
        src = joined["arc_id_a"].to_numpy()
        dst = joined["arc_id_b"].to_numpy()
        w = lengths[dst]
        finite = np.isfinite(w)
        graph.add_weighted_edges_from(
            zip(src[finite].tolist(), dst[finite].tolist(), w[finite].tolist())
        )

    _GRAPH_CACHE[trip_id] = graph
    return graph


def shortest_arc_distance(
    graph: nx.DiGraph,
    arc_from: int,
    arc_to: int,
    cutoff_m: float = DEFAULT_CUTOFF_M,
) -> float:
    """Road distance from ``arc_from`` to ``arc_to``, in metres.

    Parameters
    ----------
    graph
        Output of :func:`build_arc_graph`.
    arc_from, arc_to
        Arc ids. ``arc_from == arc_to`` returns 0.0 -- staying on the same
        arc is free, which is the common case between consecutive 1 Hz fixes.
    cutoff_m
        Dijkstra stops expanding beyond this distance.

    Returns
    -------
    float
        Distance in metres, or ``inf`` when unreachable within ``cutoff_m``
        or when either arc is absent from the graph. ``inf`` is the right
        answer for a transition the vehicle could not have made; the decoder
        turns it into zero probability.
    """
    arc_from, arc_to = int(arc_from), int(arc_to)
    if arc_from == arc_to:
        return 0.0
    if arc_from not in graph or arc_to not in graph:
        return float("inf")
    try:
        lengths = nx.single_source_dijkstra_path_length(
            graph, arc_from, cutoff=cutoff_m, weight="weight"
        )
    except nx.NodeNotFound:
        return float("inf")
    return float(lengths.get(arc_to, float("inf")))


def precompute_neighbors(
    graph: nx.DiGraph,
    arc_id: int,
    cutoff_m: float = DEFAULT_CUTOFF_M,
    trip_id: str | None = None,
) -> Dict[int, float]:
    """All arcs reachable from ``arc_id`` within ``cutoff_m``, with distances.

    One Dijkstra expansion answers every transition query out of this arc at
    once, which is what makes a Viterbi pass affordable: the decoder asks
    "distance from this candidate to each candidate at the next time step",
    and the same source arc recurs at every step the vehicle stays on it.

    Parameters
    ----------
    graph
        Output of :func:`build_arc_graph`.
    arc_id
        Source arc.
    cutoff_m
        Search radius in metres.
    trip_id
        When given, the result is memoised in a module-level cache keyed by
        ``(trip_id, cutoff_m, arc_id)``. Omit it for a one-off query.

    Returns
    -------
    dict
        ``{arc_id: distance_m}``, always including the source at 0.0.
        Empty only when the arc is not in the graph.
    """
    arc_id = int(arc_id)
    key = None
    if trip_id is not None:
        key = f"{trip_id}@{cutoff_m:g}"
        hit = _NEIGHBOR_CACHE.get(key, {}).get(arc_id)
        if hit is not None:
            return hit

    if arc_id not in graph:
        result: Dict[int, float] = {}
    else:
        result = {
            int(k): float(v)
            for k, v in nx.single_source_dijkstra_path_length(
                graph, arc_id, cutoff=cutoff_m, weight="weight"
            ).items()
        }
        result.setdefault(arc_id, 0.0)

    if key is not None:
        _NEIGHBOR_CACHE.setdefault(key, {})[arc_id] = result
    return result


def load_route_segments(trip_data: Mapping[str, Any]) -> List[Tuple[int, int]]:
    """Contiguous stretches of the ground-truth route, as index ranges.

    Parameters
    ----------
    trip_data
        A trip dict as returned by ``io_utils.load_trip``.

    Returns
    -------
    list of (start_idx, end_idx)
        Half-open ranges into the ``.route`` array: segment ``k`` covers
        ``route[start_idx:end_idx]``. A trip with no breaks yields a single
        ``(0, len(route))``.

    Notes
    -----
    Each boundary is a position where ``arc[i].to_node != arc[i+1].from_node``
    -- a real gap of 10-600 m in this dataset, not a rounding artefact. The
    decoder should treat a boundary as a **forced restart**: no transition
    probability is applied across it, since no road connects the two sides
    and any finite penalty would invite a fabricated detour.
    """
    chains = io.route_segments(trip_data["route"], trip_data["arcs"])
    bounds: List[Tuple[int, int]] = []
    cursor = 0
    for chain in chains:
        bounds.append((cursor, cursor + len(chain)))
        cursor += len(chain)
    return bounds


def clear_caches() -> None:
    """Drop every memoised graph, neighbour set and length array."""
    _GRAPH_CACHE.clear()
    _NEIGHBOR_CACHE.clear()
    _LENGTH_CACHE.clear()


def cache_stats() -> Dict[str, int]:
    """Current cache occupancy, for tests and diagnostics."""
    return {
        "graphs": len(_GRAPH_CACHE),
        "length_arrays": len(_LENGTH_CACHE),
        "neighbor_keys": len(_NEIGHBOR_CACHE),
        "neighbor_entries": sum(len(v) for v in _NEIGHBOR_CACHE.values()),
    }
