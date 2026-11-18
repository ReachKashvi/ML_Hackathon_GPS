"""Feature engineering for (GPS point, candidate arc) pairs (Step 7).

:func:`build_features` joins three sources into one row per candidate pair:

* **GPS-derived**, one value per point, broadcast to its candidates: speed,
  heading, sampling interval.
* **Candidate geometry**, from Step 6: perpendicular distance, position
  along the arc.
* **Arc / graph**, one value per arc: length, heading, endpoint degrees.

Design decisions worth stating
------------------------------
*No ``arc_id`` as a predictor.* Arc ids are trip-local row indices, so their
numeric value is meaningless across trips; using it would be pure leakage of
the file layout. It is carried as an identifier column only.

*Bearings are fed as sine/cosine, never as raw degrees.* 359 deg and 1 deg
are two degrees apart but 358 units apart on a linear scale, which would
mislead every distance- or coefficient-based model in the zoo. The single
quantity that genuinely matters -- **how much** the vehicle heading disagrees
with the road heading -- is supplied as ``bearing_diff_deg`` in ``[0, 180]``.

*Missing values are filled, with an indicator column beside each fill.* A
pipeline can then drop the indicators or use them, and models that cannot
accept NaN (GaussianNB, LogisticRegression, KNN) still work. Nothing is
silently filled without a flag:

====================== =========================================== ==========
column                 undefined when                              filled with
====================== =========================================== ==========
``dt_s``               first point of the trip                     trip median
``speed_mps``          first point, or dt == 0                     next valid
``bearing_diff_deg``   GPS or arc heading undefined                90.0
``gps_bearing_*``      consecutive fixes coincide (vehicle still)  0.0
====================== =========================================== ==========

90 degrees is chosen as the neutral fill for a heading disagreement: it is
the midpoint of the range and says "this pair carries no heading evidence",
rather than 0 (perfect agreement) or 180 (perfectly wrong), both of which
would be assertions the data does not support.
"""

from __future__ import annotations

from typing import Any, List, Mapping

import numpy as np
import pandas as pd

import geo_utils as geo

__all__ = [
    "FEATURE_COLUMNS",
    "ID_COLUMNS",
    "NEUTRAL_BEARING_DIFF_DEG",
    "build_features",
    "feature_descriptions",
]

#: Columns that identify a row but must never be used as predictors.
ID_COLUMNS = ("point_idx", "arc_id")

#: Neutral fill for an undefined heading disagreement (see module docstring).
NEUTRAL_BEARING_DIFF_DEG = 90.0

#: Predictive columns produced by :func:`build_features`, in order.
FEATURE_COLUMNS = (
    # --- candidate geometry ---
    "perp_dist_m",
    "endpoint_dist_m",
    "candidate_rank",
    "dist_from_arc_start_m",
    "dist_from_arc_end_m",
    "frac_along_arc",
    # --- arc / graph ---
    "arc_length_m",
    "arc_bearing_sin",
    "arc_bearing_cos",
    "from_node_degree",
    "to_node_degree",
    "min_node_degree",
    "max_node_degree",
    # --- GPS dynamics ---
    "dt_s",
    "speed_mps",
    "accel_mps2",
    "gps_bearing_sin",
    "gps_bearing_cos",
    # --- interaction ---
    "bearing_diff_deg",
    # --- missingness indicators ---
    "dt_was_missing",
    "speed_was_missing",
    "gps_bearing_undefined",
    "arc_bearing_undefined",
)


def _node_degrees(arcs: pd.DataFrame, n_nodes: int) -> np.ndarray:
    """Total incident arc count per node (in + out, undirected view).

    Degree is the cheapest signal for "is this a junction or mid-block?",
    which is what disambiguates candidates near intersections. Direction is
    already carried by the bearing features, so an undirected count adds
    information rather than repeating it.
    """
    deg = np.zeros(n_nodes, dtype="int64")
    for col in ("from_node", "to_node"):
        ids = arcs[col].to_numpy()
        ok = (ids >= 0) & (ids < n_nodes)
        deg += np.bincount(ids[ok], minlength=n_nodes)
    return deg


def _gps_dynamics(trip_data: Mapping[str, Any]) -> pd.DataFrame:
    """Per-GPS-point speed, acceleration, heading and sampling interval.

    Backward differences are used (point i compared with i-1), so a value
    describes the motion that *arrived at* the point. The first point has no
    predecessor; its values are filled and flagged.
    """
    track = trip_data["track"]
    trip_id = str(trip_data["trip_id"])
    lon = track["lon"].to_numpy()
    lat = track["lat"].to_numpy()
    t = track["t_sec"].to_numpy()
    n = len(track)

    x, y = geo.project_local(lon, lat, trip_id, trip_data=trip_data)

    dt = np.full(n, np.nan)
    step = np.full(n, np.nan)
    if n > 1:
        dt[1:] = np.diff(t)
        step[1:] = np.hypot(np.diff(x), np.diff(y))

    dt_missing = ~np.isfinite(dt)
    median_dt = np.nanmedian(dt) if np.isfinite(dt).any() else 1.0
    dt_filled = np.where(dt_missing, median_dt, dt)

    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt_filled > 0, step / dt_filled, np.nan)
    speed_missing = ~np.isfinite(speed)
    # Fill from the next valid reading: a trip's first speed is best
    # approximated by its second, not by zero (vehicles are usually already
    # moving when logging starts).
    speed_filled = pd.Series(speed).bfill().ffill().fillna(0.0).to_numpy()

    accel = np.full(n, 0.0)
    if n > 1:
        accel[1:] = np.diff(speed_filled) / np.where(dt_filled[1:] > 0, dt_filled[1:], 1.0)

    bearing = np.full(n, np.nan)
    if n > 1:
        bearing[1:] = geo.bearing_deg(lon[:-1], lat[:-1], lon[1:], lat[1:])
    bearing_undefined = ~np.isfinite(bearing)
    rad = np.radians(np.where(bearing_undefined, 0.0, bearing))

    return pd.DataFrame(
        {
            "point_idx": np.arange(n, dtype="int64"),
            "dt_s": dt_filled,
            "speed_mps": speed_filled,
            "accel_mps2": accel,
            "gps_bearing_deg": bearing,
            "gps_bearing_sin": np.where(bearing_undefined, 0.0, np.sin(rad)),
            "gps_bearing_cos": np.where(bearing_undefined, 0.0, np.cos(rad)),
            "dt_was_missing": dt_missing.astype("int8"),
            "speed_was_missing": speed_missing.astype("int8"),
            "gps_bearing_undefined": bearing_undefined.astype("int8"),
        }
    )


def _arc_properties(trip_data: Mapping[str, Any]) -> pd.DataFrame:
    """Per-arc length, heading and endpoint degrees."""
    arcs = trip_data["arcs"]
    nodes = trip_data["nodes"]
    trip_id = str(trip_data["trip_id"])
    n_nodes = len(nodes)

    a = arcs["from_node"].to_numpy()
    b = arcs["to_node"].to_numpy()
    valid = (a >= 0) & (a < n_nodes) & (b >= 0) & (b < n_nodes)

    nx_, ny_ = geo.project_local(
        nodes["lon"].to_numpy(), nodes["lat"].to_numpy(), trip_id, trip_data=trip_data
    )
    lon = nodes["lon"].to_numpy()
    lat = nodes["lat"].to_numpy()

    length = np.full(len(arcs), np.nan)
    bearing = np.full(len(arcs), np.nan)
    if valid.any():
        av, bv = a[valid], b[valid]
        length[valid] = np.hypot(nx_[bv] - nx_[av], ny_[bv] - ny_[av])
        bearing[valid] = geo.bearing_deg(lon[av], lat[av], lon[bv], lat[bv])

    # A zero-length arc has no direction; so does an arc with a bad node ref.
    bearing_undefined = ~np.isfinite(bearing)
    rad = np.radians(np.where(bearing_undefined, 0.0, bearing))

    deg = _node_degrees(arcs, n_nodes)
    from_deg = np.where(valid, deg[np.clip(a, 0, max(n_nodes - 1, 0))], 0)
    to_deg = np.where(valid, deg[np.clip(b, 0, max(n_nodes - 1, 0))], 0)

    return pd.DataFrame(
        {
            "arc_id": np.arange(len(arcs), dtype="int64"),
            "arc_length_m": np.where(np.isfinite(length), length, 0.0),
            "arc_bearing_deg": bearing,
            "arc_bearing_sin": np.where(bearing_undefined, 0.0, np.sin(rad)),
            "arc_bearing_cos": np.where(bearing_undefined, 0.0, np.cos(rad)),
            "from_node_degree": from_deg.astype("int64"),
            "to_node_degree": to_deg.astype("int64"),
            "arc_bearing_undefined": bearing_undefined.astype("int8"),
        }
    )


def build_features(
    trip_data: Mapping[str, Any],
    candidates_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the feature table for one trip's candidate pairs.

    Parameters
    ----------
    trip_data
        A trip dict as returned by ``io_utils.load_trip``.
    candidates_df
        Output of ``candidates.generate_candidates`` for the same trip.
        ``frac_t`` is used when present and recomputed from distances
        otherwise, so hand-built candidate frames still work.

    Returns
    -------
    pandas.DataFrame
        One row per candidate pair, with :data:`ID_COLUMNS` followed by
        :data:`FEATURE_COLUMNS`. No NaNs remain in the feature columns.
        Row order follows ``candidates_df``.

    Notes
    -----
    ``arc_id`` is present as an identifier and is **excluded** from
    :data:`FEATURE_COLUMNS` on purpose -- see the module docstring.
    """
    if len(candidates_df) == 0:
        return pd.DataFrame(columns=list(ID_COLUMNS) + list(FEATURE_COLUMNS))

    gps = _gps_dynamics(trip_data)
    arcs = _arc_properties(trip_data)

    df = (
        candidates_df.merge(gps, on="point_idx", how="left", validate="many_to_one")
        .merge(arcs, on="arc_id", how="left", validate="many_to_one")
    )

    # Position along the arc, in metres from each end.
    if "frac_t" in df.columns:
        frac = df["frac_t"].to_numpy()
    else:
        frac = np.full(len(df), 0.5)
    frac = np.clip(np.where(np.isfinite(frac), frac, 0.5), 0.0, 1.0)
    arc_len = df["arc_length_m"].to_numpy()
    df["frac_along_arc"] = frac
    df["dist_from_arc_start_m"] = frac * arc_len
    df["dist_from_arc_end_m"] = (1.0 - frac) * arc_len

    # Heading disagreement: the one interaction feature that matters.
    diff = geo.angular_diff_deg(
        df["gps_bearing_deg"].to_numpy(), df["arc_bearing_deg"].to_numpy()
    )
    df["bearing_diff_deg"] = np.where(np.isfinite(diff), diff, NEUTRAL_BEARING_DIFF_DEG)

    df["min_node_degree"] = np.minimum(df["from_node_degree"], df["to_node_degree"])
    df["max_node_degree"] = np.maximum(df["from_node_degree"], df["to_node_degree"])

    out = df[list(ID_COLUMNS) + list(FEATURE_COLUMNS)].copy()

    # Contract: the feature block is numeric and complete.
    for col in FEATURE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out[list(FEATURE_COLUMNS)].isna().to_numpy().any():
        bad = [c for c in FEATURE_COLUMNS if out[c].isna().any()]
        raise ValueError(f"build_features produced NaNs in {bad} -- fix the fill rules")
    return out.reset_index(drop=True)


def feature_descriptions() -> pd.DataFrame:
    """One-line description of every feature, for the report and notebooks."""
    rows: List[tuple] = [
        ("perp_dist_m", "candidate", "Exact distance from the GPS point to the closest point on the arc (m)."),
        ("endpoint_dist_m", "candidate", "Distance to the nearer arc endpoint (m); small near junctions."),
        ("candidate_rank", "candidate", "0 = nearest arc for this point, ascending by perp_dist_m."),
        ("dist_from_arc_start_m", "candidate", "Distance along the arc from its from_node to the matched position (m)."),
        ("dist_from_arc_end_m", "candidate", "Distance along the arc from the matched position to its to_node (m)."),
        ("frac_along_arc", "candidate", "Matched position as a fraction of arc length, 0 at from_node, 1 at to_node."),
        ("arc_length_m", "arc", "Straight-line length of the arc (m)."),
        ("arc_bearing_sin", "arc", "Sine of the arc heading; circular-safe encoding."),
        ("arc_bearing_cos", "arc", "Cosine of the arc heading; circular-safe encoding."),
        ("from_node_degree", "graph", "Number of arcs incident to the arc's from_node."),
        ("to_node_degree", "graph", "Number of arcs incident to the arc's to_node."),
        ("min_node_degree", "graph", "Smaller of the two endpoint degrees."),
        ("max_node_degree", "graph", "Larger of the two endpoint degrees; high = complex junction."),
        ("dt_s", "gps", "Seconds since the previous GPS fix."),
        ("speed_mps", "gps", "Speed over the previous interval, in projected metres per second."),
        ("accel_mps2", "gps", "Change in speed over the previous interval."),
        ("gps_bearing_sin", "gps", "Sine of the vehicle heading; 0 when the heading is undefined."),
        ("gps_bearing_cos", "gps", "Cosine of the vehicle heading; 0 when the heading is undefined."),
        ("bearing_diff_deg", "interaction", "Unsigned heading disagreement between vehicle and arc, 0-180; 90 when undefined."),
        ("dt_was_missing", "indicator", "1 where dt_s was filled (first point of a trip)."),
        ("speed_was_missing", "indicator", "1 where speed_mps was filled."),
        ("gps_bearing_undefined", "indicator", "1 where consecutive fixes coincide, so heading is undefined."),
        ("arc_bearing_undefined", "indicator", "1 where the arc has zero length, so its heading is undefined."),
    ]
    return pd.DataFrame(rows, columns=["feature", "group", "description"])
