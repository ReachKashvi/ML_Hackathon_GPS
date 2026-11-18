"""Geometry and projection helpers for PathSnap.

Why a *per-trip* projection
---------------------------
The dataset is worldwide, not a single city: GPS longitude spans -100.4 to
+137.2 and latitude -34.6 to +61.7 across the 90 trips, covering 23 distinct
UTM zones. A single fixed CRS (such as EPSG:32632, UTM 32N) would be wrong
for 89 of the 90 trips and would make metre distances meaningless -- which
would silently break the 50 m candidate radius in Step 6 and the route
alignment in Step 8.

Each trip therefore gets its own UTM CRS, chosen from the centroid of its
``.nodes`` bounding box::

    zone = floor((centroid_lon + 180) / 6) + 1
    epsg = 32600 + zone   if centroid_lat >= 0
           32700 + zone   otherwise

Two trips (``00000011`` and ``00000097``) spill across a zone boundary.
These are deliberately *not* special-cased: UTM scale distortion at the edge
of a neighbouring zone is under a metre over 125 km, negligible against the
50 m candidate radius, and because GPS points and arc nodes pass through the
*same* per-trip transformer the residual distortion largely cancels in the
distance comparisons that actually drive the pipeline.

Transformers are cached module-level, one per ``trip_id``, built on first
use. :func:`get_trip_epsg` exposes the chosen code so it can be logged.

Conventions
-----------
* Longitude/latitude are always in degrees, WGS-84 (EPSG:4326).
* Projected coordinates ``(x, y)`` are easting/northing in metres.
* Bearings are degrees clockwise from true north, in ``[0, 360)``.
* All functions accept scalars or array-likes and are vectorised.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import numpy as np
from pyproj import CRS, Transformer

__all__ = [
    "WGS84_EPSG",
    "EARTH_RADIUS_M",
    "utm_epsg_for",
    "get_trip_epsg",
    "register_trip",
    "project_local",
    "project_trip",
    "clear_transformer_cache",
    "cached_trip_ids",
    "haversine_m",
    "point_to_segment_dist_m",
    "bearing_deg",
    "angular_diff_deg",
]

#: Source CRS of every coordinate in the dataset.
WGS84_EPSG = 4326

#: Mean Earth radius (metres), WGS-84 authalic sphere. Matches
#: ``io_utils._haversine_m``.
EARTH_RADIUS_M = 6371008.8

# --------------------------------------------------------------------------
# per-trip projection
# --------------------------------------------------------------------------
#: trip_id -> pyproj Transformer (lon/lat degrees -> easting/northing metres)
_TRANSFORMERS: Dict[str, Transformer] = {}
#: trip_id -> chosen UTM EPSG code, kept for logging and inspection
_TRIP_EPSG: Dict[str, int] = {}


def utm_epsg_for(lon: float, lat: float) -> int:
    """Return the UTM EPSG code whose zone contains ``(lon, lat)``.

    Parameters
    ----------
    lon, lat
        A representative point in degrees, typically a trip centroid.

    Returns
    -------
    int
        ``32600 + zone`` in the northern hemisphere, ``32700 + zone`` in the
        southern, with ``zone = floor((lon + 180) / 6) + 1``.

    Notes
    -----
    Longitude is wrapped into ``[-180, 180)`` first, so the zone stays in
    ``1..60`` even for a centroid reported as exactly 180 degrees.
    """
    lon = float(lon)
    lat = float(lat)
    if not np.isfinite(lon) or not np.isfinite(lat):
        raise ValueError(f"non-finite centroid for UTM selection: ({lon}, {lat})")

    wrapped = ((lon + 180.0) % 360.0) - 180.0
    zone = int(np.floor((wrapped + 180.0) / 6.0)) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


def get_trip_epsg(trip_data: Mapping[str, Any]) -> int:
    """Return the UTM EPSG code chosen for a trip.

    The trip's CRS is derived from the centroid of its ``.nodes`` bounding
    box -- the road graph rather than the GPS track, since the graph is the
    wider of the two and is what gets indexed in Step 6.

    Parameters
    ----------
    trip_data
        A trip dict as returned by ``io_utils.load_trip``; needs the
        ``"trip_id"`` and ``"nodes"`` entries.

    Returns
    -------
    int
        EPSG code, e.g. ``32632`` for UTM 32N.

    Notes
    -----
    Cheap and side-effect free apart from memoising the code, so it is safe
    to call for logging on every trip.
    """
    trip_id = str(trip_data["trip_id"])
    if trip_id in _TRIP_EPSG:
        return _TRIP_EPSG[trip_id]

    nodes = trip_data["nodes"]
    lon = nodes["lon"].to_numpy(dtype="float64")
    lat = nodes["lat"].to_numpy(dtype="float64")
    if lon.size == 0:
        raise ValueError(f"trip {trip_id}: .nodes is empty, cannot choose a CRS")

    centroid_lon = 0.5 * (np.nanmin(lon) + np.nanmax(lon))
    centroid_lat = 0.5 * (np.nanmin(lat) + np.nanmax(lat))

    epsg = utm_epsg_for(centroid_lon, centroid_lat)
    _TRIP_EPSG[trip_id] = epsg
    return epsg


def register_trip(trip_data: Mapping[str, Any]) -> int:
    """Build and cache the projection transformer for one trip.

    Idempotent: the ``Transformer`` is constructed the first time a trip is
    seen and reused for every subsequent :func:`project_local` call.

    Parameters
    ----------
    trip_data
        A trip dict as returned by ``io_utils.load_trip``.

    Returns
    -------
    int
        The EPSG code this trip was registered with.
    """
    trip_id = str(trip_data["trip_id"])
    epsg = get_trip_epsg(trip_data)
    if trip_id not in _TRANSFORMERS:
        _TRANSFORMERS[trip_id] = Transformer.from_crs(
            CRS.from_epsg(WGS84_EPSG),
            CRS.from_epsg(epsg),
            always_xy=True,
        )
    return epsg


def project_local(
    lon,
    lat,
    trip_id: str,
    trip_data: Mapping[str, Any] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project lon/lat degrees to the trip's local UTM metres.

    Parameters
    ----------
    lon, lat
        Scalars or array-likes of degrees (EPSG:4326). Shapes must match.
    trip_id
        Identifies which cached transformer to use. The trip must have been
        registered already, or ``trip_data`` must be supplied so it can be
        registered on the spot.
    trip_data
        Optional trip dict, used only if ``trip_id`` is not yet cached.

    Returns
    -------
    (x, y) : tuple of numpy.ndarray
        Easting and northing in metres, same shape as the inputs (0-d arrays
        for scalar input). NaN inputs propagate as NaN rather than raising.

    Raises
    ------
    KeyError
        If the trip is unknown and no ``trip_data`` was given.

    Examples
    --------
    >>> trip = load_trip("data/train/00000001")        # doctest: +SKIP
    >>> register_trip(trip)                            # doctest: +SKIP
    32632
    >>> x, y = project_local(trip["track"].lon, trip["track"].lat,
    ...                      trip["trip_id"])          # doctest: +SKIP
    """
    trip_id = str(trip_id)
    transformer = _TRANSFORMERS.get(trip_id)
    if transformer is None:
        if trip_data is None:
            raise KeyError(
                f"trip {trip_id!r} has no cached transformer; call "
                f"register_trip(trip_data) first, or pass trip_data="
            )
        register_trip(trip_data)
        transformer = _TRANSFORMERS[trip_id]

    lon_arr = np.asarray(lon, dtype="float64")
    lat_arr = np.asarray(lat, dtype="float64")
    if lon_arr.shape != lat_arr.shape:
        raise ValueError(
            f"lon/lat shape mismatch: {lon_arr.shape} vs {lat_arr.shape}"
        )

    x, y = transformer.transform(lon_arr, lat_arr)
    return np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64")


def project_trip(trip_data: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """Project a whole trip's GPS track and road-graph nodes in one pass.

    Convenience wrapper used by candidate generation (Step 6) and route
    alignment (Step 8), both of which need the same projected arrays.

    Parameters
    ----------
    trip_data
        A trip dict as returned by ``io_utils.load_trip``.

    Returns
    -------
    dict
        ``epsg``      : int, the CRS used.
        ``track_x``   : ndarray, easting per GPS point, metres.
        ``track_y``   : ndarray, northing per GPS point, metres.
        ``node_x``    : ndarray, easting per node (indexed by node_id).
        ``node_y``    : ndarray, northing per node (indexed by node_id).
    """
    epsg = register_trip(trip_data)
    trip_id = str(trip_data["trip_id"])
    track, nodes = trip_data["track"], trip_data["nodes"]

    track_x, track_y = project_local(
        track["lon"].to_numpy(), track["lat"].to_numpy(), trip_id
    )
    node_x, node_y = project_local(
        nodes["lon"].to_numpy(), nodes["lat"].to_numpy(), trip_id
    )
    return {
        "epsg": epsg,
        "track_x": track_x,
        "track_y": track_y,
        "node_x": node_x,
        "node_y": node_y,
    }


def clear_transformer_cache() -> None:
    """Drop all cached transformers and EPSG codes.

    Useful in tests and when processing many trips in one long-running
    process, where the cache would otherwise grow unboundedly.
    """
    _TRANSFORMERS.clear()
    _TRIP_EPSG.clear()


def cached_trip_ids() -> Tuple[str, ...]:
    """Return the trip ids that currently have a cached transformer."""
    return tuple(_TRANSFORMERS)


# --------------------------------------------------------------------------
# distances and angles
# --------------------------------------------------------------------------
def haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Great-circle distance in metres between lon/lat points.

    Parameters
    ----------
    lon1, lat1, lon2, lat2
        Scalars or array-likes of degrees; broadcast against each other.

    Returns
    -------
    numpy.ndarray
        Distance in metres on a sphere of radius :data:`EARTH_RADIUS_M`.
        Accurate to roughly 0.3% against the WGS-84 ellipsoid -- fine for
        EDA summaries, while metre-exact work goes through
        :func:`project_local`.

    Notes
    -----
    Used for projection-independent summaries (trip length, route-gap sizes)
    so that EDA numbers do not depend on any CRS choice.
    """
    lon1, lat1, lon2, lat2 = (
        np.radians(np.asarray(v, dtype="float64")) for v in (lon1, lat1, lon2, lat2)
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def point_to_segment_dist_m(px, py, ax, ay, bx, by, *, return_t: bool = False):
    """Perpendicular distance from point P to segment AB, in projected metres.

    All inputs must already be in the same projected CRS (use
    :func:`project_local`), since this is plain planar geometry.

    Parameters
    ----------
    px, py
        Point coordinates (metres). Scalars or array-likes.
    ax, ay, bx, by
        Segment endpoints (metres). Broadcast against the point.
    return_t
        If True, also return the clamped projection parameter.

    Returns
    -------
    dist : numpy.ndarray
        Distance in metres from P to the *closest point on the segment*
        (not on the infinite line), so a point beyond an endpoint gets the
        endpoint distance.
    t : numpy.ndarray, optional
        Returned when ``return_t=True``. Position of the foot of the
        perpendicular along AB, clamped to ``[0, 1]``: 0 at A, 1 at B.
        Multiply by the segment length to get distance-along-arc.

    Notes
    -----
    Degenerate segments (A == B, zero length) are handled explicitly: ``t``
    is 0 and the distance is the point-to-A distance. No division by zero
    and no NaN is produced for them.
    """
    px = np.asarray(px, dtype="float64")
    py = np.asarray(py, dtype="float64")
    ax = np.asarray(ax, dtype="float64")
    ay = np.asarray(ay, dtype="float64")
    bx = np.asarray(bx, dtype="float64")
    by = np.asarray(by, dtype="float64")

    abx = bx - ax
    aby = by - ay
    seg_len_sq = abx * abx + aby * aby

    # Guard the degenerate case before dividing, then zero those t values.
    safe_len_sq = np.where(seg_len_sq > 0.0, seg_len_sq, 1.0)
    t = ((px - ax) * abx + (py - ay) * aby) / safe_len_sq
    t = np.where(seg_len_sq > 0.0, t, 0.0)
    t = np.clip(t, 0.0, 1.0)

    foot_x = ax + t * abx
    foot_y = ay + t * aby
    dist = np.hypot(px - foot_x, py - foot_y)

    if return_t:
        return dist, t
    return dist


def bearing_deg(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Initial great-circle bearing from point 1 to point 2.

    Parameters
    ----------
    lon1, lat1, lon2, lat2
        Scalars or array-likes of degrees; broadcast against each other.

    Returns
    -------
    numpy.ndarray
        Forward azimuth in degrees clockwise from true north, in
        ``[0, 360)``. **NaN where the two points coincide**, since the
        bearing is genuinely undefined there -- callers must decide what an
        undefined heading means rather than silently receiving 0 degrees
        (which would read as "due north").

    Notes
    -----
    Undefined bearings are expected in this dataset: a stationary vehicle
    reports identical consecutive fixes, and zero-length arcs would give an
    undefined arc bearing.
    """
    lon1_r, lat1_r, lon2_r, lat2_r = (
        np.radians(np.asarray(v, dtype="float64")) for v in (lon1, lat1, lon2, lat2)
    )
    dlon = lon2_r - lon1_r
    y = np.sin(dlon) * np.cos(lat2_r)
    x = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon)

    bearing = np.degrees(np.arctan2(y, x)) % 360.0

    coincident = (y == 0.0) & (x == 0.0)
    if np.ndim(bearing) == 0:
        return np.float64(np.nan) if coincident else bearing
    bearing = np.where(coincident, np.nan, bearing)
    return bearing


def angular_diff_deg(a, b) -> np.ndarray:
    """Smallest absolute difference between two angles, in degrees.

    Parameters
    ----------
    a, b
        Angles in degrees (any range; wrapped internally). Scalars or
        array-likes, broadcast against each other.

    Returns
    -------
    numpy.ndarray
        Difference in ``[0, 180]``, handling wrap-around correctly: 350 deg
        and 10 deg differ by 20, not 340. NaN in either input propagates, so
        an undefined bearing from :func:`bearing_deg` yields an undefined
        difference rather than a misleading number.

    Notes
    -----
    Unsigned by design. For a *heading* feature we only care how much two
    directions disagree, not which way round; the sign would also flip with
    arc digitisation order, which carries no information about the vehicle.
    A one-way arc traversed backwards shows up as a difference near 180.
    """
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    diff = np.abs((a - b) % 360.0)
    return np.minimum(diff, 360.0 - diff)
