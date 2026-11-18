"""Leakage-free context features for the emission model (HMM improvement E5).

Diagnosis: on noisy tracks the 1-step GPS heading is dominated by jitter
(trip 00000077: median step 8.6 m while its 30 s-subsampled length matches
the route), so ``bearing_diff_deg`` cannot tell an arc from its reverse twin
-- the error type behind 64% of Phase 4's student_test errors. Each feature
below uses the GPS track and the candidate geometry only; no label, truth
arc or route information is read. Windows are centred (they use later GPS
points too), which is legitimate for offline map matching: the whole track
is available at inference time.

=====================  ==========================================================
``x_head_w{k}_sin/cos`` heading of the displacement from point i-k to i+k
``x_bdiff_w{k}``        disagreement (0-180 deg) of that heading with the arc
``x_head_w{k}_undef``   1 when that displacement is under 2 m (heading undefined;
                        sin/cos 0 and bdiff 90, as in Phase 2's fill rule)
``x_speed_w10``         displacement over i-10..i+10 / elapsed time -- low when
                        stopped, whatever the per-step jitter
``x_perp_ctx_mean``     mean perpendicular distance of this same arc over the
                        neighbouring points i-5..i+5 where it is a candidate
``x_ctx_presence``      fraction of those 10 neighbours that have this arc as a
                        candidate (an arc that is only near one fix is unlikely)
=====================  ==========================================================

Shared by ``05_build_features.py`` (training / student_test parquets) and
the API's inference path, so both compute exactly the same columns.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

import geo_utils as geo

WINDOWS = (3, 10)
CTX = 5
MIN_DISP_M = 2.0


def gps_context(trip: Mapping[str, Any]) -> pd.DataFrame:
    """Per-GPS-point windowed heading and speed."""
    track = trip["track"]
    tid = str(trip["trip_id"])
    lon, lat = track["lon"].to_numpy(), track["lat"].to_numpy()
    t = track["t_sec"].to_numpy(dtype="float64")
    x, y = geo.project_local(lon, lat, tid, trip_data=trip)
    n = len(lon)
    i = np.arange(n)
    out = {"point_idx": i}
    for k in WINDOWS:
        a, b = np.clip(i - k, 0, n - 1), np.clip(i + k, 0, n - 1)
        disp = np.hypot(x[b] - x[a], y[b] - y[a])
        head = geo.bearing_deg(lon[a], lat[a], lon[b], lat[b])
        undef = (disp < MIN_DISP_M) | ~np.isfinite(head)
        rad = np.radians(np.where(undef, 0.0, head))
        out[f"head_w{k}"] = np.where(undef, np.nan, head)
        out[f"x_head_w{k}_sin"] = np.where(undef, 0.0, np.sin(rad))
        out[f"x_head_w{k}_cos"] = np.where(undef, 0.0, np.cos(rad))
        out[f"x_head_w{k}_undef"] = undef.astype("int8")
    a, b = np.clip(i - 10, 0, n - 1), np.clip(i + 10, 0, n - 1)
    span = np.maximum(t[b] - t[a], 1.0)
    out["x_speed_w10"] = np.hypot(x[b] - x[a], y[b] - y[a]) / span
    return pd.DataFrame(out)


def arc_context(pairs: pd.DataFrame) -> pd.DataFrame:
    """Same-arc perpendicular distance over the neighbouring +-CTX points."""
    s = np.zeros(len(pairs))
    c = np.zeros(len(pairs))
    key = pd.MultiIndex.from_arrays([pairs.point_idx.to_numpy(), pairs.arc_id.to_numpy()])
    lookup = pd.Series(pairs.perp_dist_m.to_numpy(), index=key)
    lookup = lookup[~lookup.index.duplicated()]
    for o in range(-CTX, CTX + 1):
        if o == 0:
            continue
        q = pd.MultiIndex.from_arrays([pairs.point_idx.to_numpy() + o, pairs.arc_id.to_numpy()])
        v = lookup.reindex(q).to_numpy()
        hit = np.isfinite(v)
        s[hit] += v[hit]
        c[hit] += 1
    mean = np.where(c > 0, s / np.maximum(c, 1), pairs.perp_dist_m.to_numpy() + 50.0)
    return pd.DataFrame({"x_perp_ctx_mean": mean, "x_ctx_presence": c / (2 * CTX)},
                        index=pairs.index)


def add_context_features(pairs: pd.DataFrame, trip: Mapping[str, Any]) -> pd.DataFrame:
    """Append every ``x_*`` column to one trip's feature table."""
    g = pairs.merge(gps_context(trip), on="point_idx", how="left", validate="many_to_one")
    arc_bearing = (np.degrees(np.arctan2(g.arc_bearing_sin, g.arc_bearing_cos)) % 360).to_numpy()
    for w in WINDOWS:
        diff = geo.angular_diff_deg(g[f"head_w{w}"].to_numpy(), arc_bearing)
        ok = np.isfinite(diff) & (g.arc_bearing_undefined.to_numpy() == 0)
        g[f"x_bdiff_w{w}"] = np.where(ok, diff, 90.0)
        g = g.drop(columns=[f"head_w{w}"])
    return pd.concat([g, arc_context(g)], axis=1)
