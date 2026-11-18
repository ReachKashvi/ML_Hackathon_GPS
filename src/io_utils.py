"""I/O helpers for the PathSnap map-matching dataset.

Dataset layout (verified against all 90 trips in ``data/train`` and
``data/student_test``)::

    data/<partition>/<trip_id>/<trip_id>.track
    data/<partition>/<trip_id>/<trip_id>.nodes
    data/<partition>/<trip_id>/<trip_id>.arcs
    data/<partition>/<trip_id>/<trip_id>.route

Verified file formats -- none of this is assumed:

``.track``
    3 whitespace-separated columns ``longitude latitude timestamp_sec``,
    all floats, no header. Tab-separated, CRLF line endings.
``.nodes``
    2 columns ``longitude latitude``, floats, no header. The **row index is
    the node_id**.
``.arcs``
    2 columns ``from_node_id to_node_id``, integers, no header. The **row
    index is the arc_id**. An arc is a straight segment between two nodes.
``.route``
    1 column: the ordered sequence of traversed arc_ids, one per line.

Facts that shape the rest of the pipeline:

* **Each trip carries its own road graph.** All 80 training trips have
  distinct ``.nodes``/``.arcs`` files. node_ids and arc_ids are therefore
  *trip-local* row indices and carry no meaning across trips.
* **``.route`` is not one-to-one with ``.track``.** Trip ``00000001`` has
  1315 GPS points and 525 route arcs. Aligning the two is the job of
  ``labels.py`` (Step 8), never positional indexing.
* **The route chain is only piecewise-continuous.** 11 of the 90 trips
  contain positions where ``arc[i].to_node != arc[i + 1].from_node``, with
  real gaps of ~10 m to ~600 m. Use :func:`route_segments` to walk the route
  as connected chains instead of assuming one unbroken polyline.
* Graph sizes span two orders of magnitude (898 to 88,809 nodes), so
  per-trip work must stay vectorised.
* **Trips are spread worldwide, not confined to Darmstadt** (longitude
  -100.4 to +137.2, latitude -34.6 to +61.7; 23 distinct UTM zones). Every
  metre-based calculation must therefore use a *per-trip* projection --
  see ``geo_utils.project_local``.

Nothing in this module mutates data on disk; the loaders are pure reads.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

__all__ = [
    "TripDataError",
    "TRACK_COLUMNS",
    "NODE_COLUMNS",
    "ARC_COLUMNS",
    "TRIP_EXTENSIONS",
    "load_track",
    "load_nodes",
    "load_arcs",
    "load_route",
    "list_trips",
    "load_trip",
    "route_segments",
    "route_chain_breaks",
    "validate_trip",
]

PathLike = Union[str, os.PathLike]

#: Column names used for a loaded ``.track`` frame.
TRACK_COLUMNS = ("lon", "lat", "t_sec")
#: Column names used for a loaded ``.nodes`` frame (index = node_id).
NODE_COLUMNS = ("lon", "lat")
#: Column names used for a loaded ``.arcs`` frame (index = arc_id).
ARC_COLUMNS = ("from_node", "to_node")
#: The four per-trip files, by extension.
TRIP_EXTENSIONS = ("track", "nodes", "arcs", "route")


class TripDataError(ValueError):
    """Raised when a trip file is missing, empty, or structurally malformed.

    Structural problems (wrong column count, non-numeric token, empty file)
    raise. *Semantic* problems -- non-monotonic timestamps, dangling node or
    arc references, breaks in the route chain -- do not raise; they are
    reported by :func:`validate_trip` so that EDA can quantify them without
    the loader refusing to read the trip.
    """


# --------------------------------------------------------------------------
# low-level reader
# --------------------------------------------------------------------------
def _read_numeric_table(
    path: PathLike,
    n_cols: int,
    columns: tuple,
    dtype: str,
    *,
    what: str,
) -> pd.DataFrame:
    """Read a headerless whitespace-separated numeric table.

    Parameters
    ----------
    path
        File to read.
    n_cols
        Exact number of columns every row must have.
    columns
        Names to assign, in file order.
    dtype
        ``"float64"`` or ``"int64"``. For ``int64`` the parsed values must be
        integral, otherwise :class:`TripDataError` is raised.
    what
        Human-readable file kind, used in error messages.

    Returns
    -------
    pandas.DataFrame
        One row per data line, with a clean ``RangeIndex``. For ``.nodes``
        and ``.arcs`` that index *is* the node_id / arc_id.

    Raises
    ------
    TripDataError
        If the file is missing, empty, ragged, or non-numeric.
    """
    path = Path(path)
    if not path.is_file():
        raise TripDataError(f"{what} file not found: {path}")
    if path.stat().st_size == 0:
        raise TripDataError(f"{what} file is empty: {path}")

    # ``sep=r"\s+"`` covers the tab separation actually used as well as any
    # stray spaces, and tolerates the CRLF line endings the .track files use.
    try:
        frame = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            comment="#",
            skip_blank_lines=True,
            dtype=str,  # parse strictly ourselves, below
            keep_default_na=False,
        )
    except pd.errors.EmptyDataError as exc:  # whitespace-only file
        raise TripDataError(f"{what} file contains no data rows: {path}") from exc
    except pd.errors.ParserError as exc:
        raise TripDataError(f"{what} file is malformed: {path}: {exc}") from exc

    if frame.empty:
        raise TripDataError(f"{what} file contains no data rows: {path}")

    if frame.shape[1] != n_cols:
        raise TripDataError(
            f"{what} file {path} has {frame.shape[1]} column(s), expected "
            f"{n_cols}. First row: {frame.iloc[0].tolist()!r}"
        )

    frame.columns = list(columns)

    # Strict numeric conversion: locate the offending row rather than
    # silently producing NaN.
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    bad_mask = numeric.isna().any(axis=1)
    if bool(bad_mask.any()):
        bad_lines = (np.flatnonzero(bad_mask.to_numpy())[:5] + 1).tolist()
        raise TripDataError(
            f"{what} file {path} has non-numeric value(s) on row(s) "
            f"{bad_lines} (1-based): "
            f"{frame[bad_mask].head().to_dict('records')!r}"
        )

    if dtype == "int64":
        values = numeric.to_numpy(dtype="float64")
        if not np.array_equal(values, np.round(values)):
            raise TripDataError(
                f"{what} file {path} contains non-integer values where "
                f"integer ids are required"
            )
        numeric = numeric.round().astype("int64")
    else:
        numeric = numeric.astype(dtype)

    return numeric.reset_index(drop=True)


# --------------------------------------------------------------------------
# per-file loaders
# --------------------------------------------------------------------------
def load_track(path: PathLike) -> pd.DataFrame:
    """Load a ``.track`` file of GPS observations.

    Returns
    -------
    pandas.DataFrame
        Columns ``point_idx`` (int64), ``lon``, ``lat``, ``t_sec`` (float64),
        in file order. ``point_idx`` is the row index and is the identifier
        used for a GPS observation throughout the pipeline.

    Notes
    -----
    Row order is preserved exactly as stored; timestamps are *not* sorted or
    de-duplicated here. Step 4 flags non-monotonic and duplicate timestamps.
    """
    track = _read_numeric_table(path, 3, TRACK_COLUMNS, "float64", what="track")
    track.insert(0, "point_idx", np.arange(len(track), dtype="int64"))
    return track


def load_nodes(path: PathLike) -> pd.DataFrame:
    """Load a ``.nodes`` file of road-graph vertices.

    Returns
    -------
    pandas.DataFrame
        Columns ``lon``, ``lat`` (float64), indexed by ``node_id`` -- the
        file's row index, which is how ``.arcs`` refers to nodes.
    """
    nodes = _read_numeric_table(path, 2, NODE_COLUMNS, "float64", what="nodes")
    nodes.index = pd.RangeIndex(len(nodes), name="node_id")
    return nodes


def load_arcs(path: PathLike) -> pd.DataFrame:
    """Load an ``.arcs`` file of directed road segments.

    Returns
    -------
    pandas.DataFrame
        Columns ``from_node``, ``to_node`` (int64), indexed by ``arc_id`` --
        the file's row index, which is how ``.route`` refers to arcs.

    Notes
    -----
    Node references are *not* bounds-checked here (that needs the ``.nodes``
    file); :func:`validate_trip` does it once the whole trip is loaded.
    """
    arcs = _read_numeric_table(path, 2, ARC_COLUMNS, "int64", what="arcs")
    arcs.index = pd.RangeIndex(len(arcs), name="arc_id")
    return arcs


def load_route(path: PathLike) -> np.ndarray:
    """Load a ``.route`` file: the ordered ground-truth arc sequence.

    Returns
    -------
    numpy.ndarray
        1-D int64 array of arc_ids in traversal order. Order is significant
        and is preserved exactly; repeated values are legitimate (a route may
        revisit an arc) and are not collapsed.
    """
    route = _read_numeric_table(path, 1, ("arc_id",), "int64", what="route")
    return route["arc_id"].to_numpy(dtype="int64")


# --------------------------------------------------------------------------
# trip discovery and assembly
# --------------------------------------------------------------------------
def list_trips(partition_dir: PathLike) -> List[str]:
    """List trip ids in a partition directory, sorted ascending.

    A subdirectory counts as a trip only if it contains all four
    ``<trip_id>.<ext>`` files. That naming convention was verified to hold
    for every one of the 90 trips present, but it is re-checked here rather
    than assumed, so a partially populated folder is skipped instead of
    failing later.

    Parameters
    ----------
    partition_dir
        e.g. ``data/train`` or ``data/student_test``.

    Returns
    -------
    list of str
        Trip ids, as directory names (zero-padded strings such as
        ``"00000001"``). Kept as strings so the padding survives round-trips
        through parquet and CSV.
    """
    partition_dir = Path(partition_dir)
    if not partition_dir.is_dir():
        raise TripDataError(f"partition directory not found: {partition_dir}")

    trips = []
    for child in sorted(partition_dir.iterdir()):
        if not child.is_dir():
            continue
        if all((child / f"{child.name}.{ext}").is_file() for ext in TRIP_EXTENSIONS):
            trips.append(child.name)
    return trips


def load_trip(trip_dir: PathLike) -> Dict[str, Any]:
    """Load all four components of one trip.

    Parameters
    ----------
    trip_dir
        Path to a single trip directory, e.g. ``data/train/00000001``. The
        four files are expected to be named after the directory.

    Returns
    -------
    dict
        ``trip_id``  : str, the directory name.
        ``track``    : DataFrame from :func:`load_track`.
        ``nodes``    : DataFrame from :func:`load_nodes`.
        ``arcs``     : DataFrame from :func:`load_arcs`.
        ``route``    : ndarray from :func:`load_route`.
        ``trip_dir`` : pathlib.Path, kept for provenance in error messages.

    Raises
    ------
    TripDataError
        If any component is missing, empty, or malformed.
    """
    trip_dir = Path(trip_dir)
    if not trip_dir.is_dir():
        raise TripDataError(f"trip directory not found: {trip_dir}")

    trip_id = trip_dir.name
    stem = trip_dir / trip_id
    return {
        "trip_id": trip_id,
        "track": load_track(f"{stem}.track"),
        "nodes": load_nodes(f"{stem}.nodes"),
        "arcs": load_arcs(f"{stem}.arcs"),
        "route": load_route(f"{stem}.route"),
        "trip_dir": trip_dir,
    }


# --------------------------------------------------------------------------
# route chain structure
# --------------------------------------------------------------------------
#: Mean Earth radius (metres), WGS-84 authalic sphere.
_EARTH_RADIUS_M = 6371008.8


def _haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Great-circle distance in metres (vectorised, degrees in).

    Kept private and local so that this module stays dependency-free.
    ``geo_utils.haversine_m`` is the public implementation and must agree
    with this one.
    """
    lon1, lat1, lon2, lat2 = (
        np.radians(np.asarray(v, dtype="float64")) for v in (lon1, lat1, lon2, lat2)
    )
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def route_chain_breaks(
    route: np.ndarray,
    arcs: pd.DataFrame,
    nodes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Locate discontinuities in the ground-truth arc chain.

    Consecutive route arcs are expected to share a node
    (``arcs[route[i]].to_node == arcs[route[i + 1]].from_node``). In this
    dataset that does not always hold: 11 of the 90 trips contain genuine
    jumps of roughly 10 m to 600 m, so the route polyline is
    *piecewise*-continuous and any arc-length accumulation must reset
    across a break.

    Parameters
    ----------
    route
        Ordered arc_ids, as returned by :func:`load_route`.
    arcs
        Arc table, as returned by :func:`load_arcs`.
    nodes
        Optional node table. When given, the gap distance in metres is
        measured between the two unconnected nodes.

    Returns
    -------
    pandas.DataFrame
        One row per break, with columns ``route_pos`` (index ``i`` in
        ``route`` such that the break is between ``route[i]`` and
        ``route[i + 1]``), ``prev_arc_id``, ``next_arc_id``, ``prev_to_node``,
        ``next_from_node`` and ``gap_m`` (NaN when ``nodes`` is omitted).
        Empty with those columns when the chain is unbroken.
    """
    columns = [
        "route_pos",
        "prev_arc_id",
        "next_arc_id",
        "prev_to_node",
        "next_from_node",
        "gap_m",
    ]
    route = np.asarray(route, dtype="int64")
    n_arcs = len(arcs)
    if len(route) < 2 or ((route < 0) | (route >= n_arcs)).any():
        # Out-of-range arc ids are reported separately by validate_trip;
        # chain analysis would be meaningless here.
        return pd.DataFrame(columns=columns)

    to_nodes = arcs["to_node"].to_numpy()[route[:-1]]
    from_nodes = arcs["from_node"].to_numpy()[route[1:]]
    positions = np.flatnonzero(to_nodes != from_nodes)

    frame = pd.DataFrame(
        {
            "route_pos": positions.astype("int64"),
            "prev_arc_id": route[positions],
            "next_arc_id": route[positions + 1],
            "prev_to_node": to_nodes[positions],
            "next_from_node": from_nodes[positions],
        }
    )
    if nodes is None or frame.empty:
        frame["gap_m"] = np.full(len(frame), np.nan)
        return frame

    node_lon = nodes["lon"].to_numpy()
    node_lat = nodes["lat"].to_numpy()
    a, b = frame["prev_to_node"].to_numpy(), frame["next_from_node"].to_numpy()
    in_range = (
        (a >= 0) & (a < len(nodes)) & (b >= 0) & (b < len(nodes))
    )
    gap = np.full(len(frame), np.nan)
    if in_range.any():
        gap[in_range] = _haversine_m(
            node_lon[a[in_range]],
            node_lat[a[in_range]],
            node_lon[b[in_range]],
            node_lat[b[in_range]],
        )
    frame["gap_m"] = gap
    return frame


def route_segments(route: np.ndarray, arcs: pd.DataFrame) -> List[List[int]]:
    """Split a route into contiguous arc chains.

    The route is cut at every point where consecutive arcs do not share a
    node, so each returned chain *is* a connected polyline and can be walked
    through node coordinates without introducing a phantom straight line
    across a gap.

    Parameters
    ----------
    route
        Ordered arc_ids, as returned by :func:`load_route`.
    arcs
        Arc table, as returned by :func:`load_arcs`.

    Returns
    -------
    list of list of int
        Contiguous chains of arc_ids, in route order. Concatenating them
        reproduces ``route`` exactly. A route with no breaks yields a single
        chain; an empty route yields ``[]``.

    Examples
    --------
    A route of 262 arcs with one break at position 53 returns two chains of
    54 and 208 arcs.
    """
    route = np.asarray(route, dtype="int64")
    if route.size == 0:
        return []

    breaks = route_chain_breaks(route, arcs)
    # cut *after* each break position
    cut_after = breaks["route_pos"].to_numpy(dtype="int64")
    bounds = [0, *(cut_after + 1).tolist(), len(route)]
    return [
        route[start:stop].tolist()
        for start, stop in zip(bounds[:-1], bounds[1:])
        if stop > start
    ]


# --------------------------------------------------------------------------
# non-fatal integrity checks
# --------------------------------------------------------------------------
def validate_trip(trip: Dict[str, Any]) -> List[str]:
    """Check a loaded trip for semantic problems, without raising.

    Covers the cheap cross-file invariants that need more than one component
    to verify. Geometric and temporal profiling (gap counts, trajectory
    length, connected components) belongs to the Step 4 EDA notebook.

    Parameters
    ----------
    trip
        A dict as returned by :func:`load_trip`.

    Returns
    -------
    list of str
        Human-readable issue descriptions; empty when the trip is clean.
    """
    issues: List[str] = []
    track, nodes, arcs, route = (
        trip["track"],
        trip["nodes"],
        trip["arcs"],
        trip["route"],
    )

    # --- GPS sanity ---
    if track[["lon", "lat"]].isna().to_numpy().any():
        issues.append("track contains NaN coordinates")
    if not track["t_sec"].is_monotonic_increasing:
        issues.append("track timestamps are not monotonically increasing")
    n_dup = int(track["t_sec"].duplicated().sum())
    if n_dup:
        issues.append(f"track has {n_dup} duplicate timestamp(s)")

    # --- arcs -> nodes references ---
    n_nodes = len(nodes)
    node_refs = arcs[list(ARC_COLUMNS)].to_numpy()
    bad_nodes = (node_refs < 0) | (node_refs >= n_nodes)
    if bad_nodes.any():
        issues.append(
            f"{int(bad_nodes.any(axis=1).sum())} arc(s) reference node ids "
            f"outside [0, {n_nodes - 1}]"
        )

    # --- route -> arcs references ---
    n_arcs = len(arcs)
    bad_arcs = (route < 0) | (route >= n_arcs)
    if bad_arcs.any():
        issues.append(
            f"{int(bad_arcs.sum())} route entr(ies) reference arc ids "
            f"outside [0, {n_arcs - 1}]"
        )

    # --- route connectivity -------------------------------------------------
    # A WARNING, not an error: 11 of the 90 trips genuinely have gaps in the
    # ground-truth chain, so these trips are usable and must not be dropped.
    # Downstream code handles them via route_segments().
    if not bad_arcs.any() and not bad_nodes.any():
        breaks = route_chain_breaks(route, arcs, nodes)
        if not breaks.empty:
            gaps = breaks["gap_m"].to_numpy()
            detail = ", ".join(
                f"pos {int(p)}: {g:.1f} m"
                for p, g in zip(breaks["route_pos"], gaps)
            )
            issues.append(
                f"WARNING: {len(breaks)} break(s) in route arc chain "
                f"(max gap {np.nanmax(gaps):.1f} m) [{detail}]"
            )

    return issues


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    import sys

    partition = sys.argv[1] if len(sys.argv) > 1 else "data/train"
    trip_ids = list_trips(partition)
    print(f"{partition}: {len(trip_ids)} trips")
    for tid in trip_ids[:3]:
        loaded = load_trip(Path(partition) / tid)
        print(
            f"  {tid}: track={len(loaded['track'])} nodes={len(loaded['nodes'])} "
            f"arcs={len(loaded['arcs'])} route={len(loaded['route'])} "
            f"issues={validate_trip(loaded) or 'none'}"
        )
