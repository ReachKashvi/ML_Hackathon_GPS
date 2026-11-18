"""Diagnostics for student_test trip 00000077 (HMM-improvement, Phase B deliverable).

Diagnosis only: nothing here is tuned on this trip, and the trip stays in
every headline number.

Questions answered, each with evidence written to ``outputs/hmm/trip77/``:

1. *Is the 2.27x GPS/route length ratio repeated laps?* Track length at 1, 5,
   10, 30 s subsampling vs route length (``length_vs_sampling.csv``).
2. *Where are the errors, and what kind?* Contiguous error runs of the
   Phase 4 HMM (and the Improved HMM if its predictions exist), each with:
   share of points on the truth arc's reverse twin, median 21 s windowed
   speed, whether the truth arc's twin is itself on the route (out-and-back),
   GPS distance to truth vs predicted arc (``error_runs.csv``).
3. *Model error or ground-truth ambiguity?* A run is flagged
   ``label_ambiguous`` when the prediction is the truth arc's reverse twin
   **and** the vehicle is stopped (< 1 m/s over 21 s): twin arcs share their
   geometry, so position cannot say which direction a stationary vehicle is
   facing, and the label alignment (direction-blind, monotone) resolves the
   tie by staying on the earlier route position. Everything else is counted
   as a model error.

Figures: route position over time with errors, windowed speed with error
runs shaded, position along the arc inside the two largest runs, and an
interactive folium map with GPS, truth, classifier and HMM paths.
"""

from __future__ import annotations

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import DATA, OUT, ROOT  # noqa: E402
import geo_utils as geo  # noqa: E402
import io_utils as io  # noqa: E402
import labels as lb  # noqa: E402

TID = "00000077"
DIR = OUT / "trip77"
DIR.mkdir(exist_ok=True)
STOP_MPS = 1.0


def runs_of(mask: np.ndarray):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def seg_dist(px, py, ax, ay, bx, by):
    d, t = geo.point_to_segment_dist_m(px, py, ax, ay, bx, by, return_t=True)
    return np.asarray(d), np.asarray(t)


def main() -> int:
    trip = io.load_trip(DATA / "student_test" / TID)
    tr = trip["track"]
    proj = geo.project_trip(trip)
    px, py, nx_, ny_ = proj["track_x"], proj["track_y"], proj["node_x"], proj["node_y"]
    arcs = trip["arcs"]
    fr, to = arcs.from_node.to_numpy(), arcs.to_node.to_numpy()
    route = np.asarray(trip["route"])
    n = len(px)
    t_sec = tr.t_sec.to_numpy(dtype="float64")
    al = lb.align_gps_to_route(trip).sort_values("point_idx")
    truth = al.truth_arc_id.to_numpy()
    rpos = al.route_pos_raw.to_numpy()

    p4 = pd.read_parquet(ROOT / "outputs/predictions/hmm_viterbi.parquet")
    p4 = p4[p4.trip_id == TID].sort_values("point_idx")
    preds = {"classifier": p4.arc_id_classifier.to_numpy(), "phase4_hmm": p4.arc_id_hmm.to_numpy()}
    final_path = OUT / "predictions_final_student_test.parquet"
    if final_path.exists():
        f = pd.read_parquet(final_path)
        f = f[f.trip_id == TID].set_index("point_idx").reindex(np.arange(n))
        preds["improved_hmm"] = f.arc_id_final.to_numpy().astype("int64")

    # 1. laps or jitter?
    rlen = float(np.hypot(nx_[to[route]] - nx_[fr[route]], ny_[to[route]] - ny_[fr[route]]).sum())
    rows = []
    for k in (1, 5, 10, 30, 60):
        s = np.arange(0, n, k)
        L = float(np.hypot(np.diff(px[s]), np.diff(py[s])).sum())
        rows.append(dict(subsample_s=k, gps_length_km=L / 1000, route_length_km=rlen / 1000,
                         ratio=L / rlen))
    lv = pd.DataFrame(rows)
    lv.to_csv(DIR / "length_vs_sampling.csv", index=False)
    print(lv.round(3).to_string(index=False))

    i = np.arange(n)
    a, b = np.clip(i - 10, 0, n - 1), np.clip(i + 10, 0, n - 1)
    speed = np.hypot(px[b] - px[a], py[b] - py[a]) / np.maximum(t_sec[b] - t_sec[a], 1.0)
    step = np.r_[np.nan, np.hypot(np.diff(px), np.diff(py))]
    route_pairs = set(zip(fr[route].tolist(), to[route].tolist()))

    # 2/3. error runs per model
    all_runs = []
    for model, pick in preds.items():
        wrong = (pick != truth) & (truth >= 0)
        for s, e in runs_of(wrong):
            seg = np.arange(s, e + 1)
            th, hm = truth[seg], pick[seg]
            twin = (fr[hm] == to[th]) & (to[hm] == fr[th])
            outback = np.mean([(to[t], fr[t]) in route_pairs for t in th])
            dt_, _ = seg_dist(px[seg], py[seg], nx_[fr[th]], ny_[fr[th]], nx_[to[th]], ny_[to[th]])
            dh_, _ = seg_dist(px[seg], py[seg], nx_[fr[hm]], ny_[fr[hm]], nx_[to[hm]], ny_[to[hm]])
            med_speed = float(np.median(speed[seg]))
            twin_frac = float(twin.mean())
            if twin_frac >= 0.8 and med_speed < STOP_MPS:
                verdict = "label_ambiguous: stopped on reverse-twin arcs"
            elif twin_frac >= 0.8:
                verdict = "model_error: wrong direction while moving"
            else:
                verdict = "model_error: junction timing / other arc"
            all_runs.append(dict(
                model=model, start=s, end=e, length=e - s + 1,
                route_pos=f"{rpos[s]}-{rpos[e]}", twin_frac=twin_frac,
                truth_twin_on_route_frac=float(outback), median_speed_w10_mps=med_speed,
                median_step_m=float(np.nanmedian(step[seg])),
                median_dist_gps_truth_m=float(np.median(dt_)),
                median_dist_gps_pred_m=float(np.median(dh_)), verdict=verdict))
    runs = pd.DataFrame(all_runs).sort_values(["model", "length"], ascending=[True, False])
    runs.to_csv(DIR / "error_runs.csv", index=False)
    summ = runs.groupby(["model", "verdict"]).length.agg(["count", "sum"]).rename(
        columns={"count": "runs", "sum": "points"})
    summ.to_csv(DIR / "error_verdicts.csv")
    print(summ.to_string())
    print(runs[runs.model == "phase4_hmm"].head(8).round(2).to_string(index=False))

    # figures -------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax = axes[0]
    ax.plot(i, rpos, color="#333333", lw=1, label="aligned route position (label)")
    for model, col in (("phase4_hmm", "#6a8fc7"), ("improved_hmm", "#d9822b")):
        if model in preds:
            w = (preds[model] != truth) & (truth >= 0)
            ax.scatter(i[w], rpos[w], s=3, color=col, label=f"{model} error")
    ax.set_ylabel("route position")
    ax.legend(frameon=False, fontsize=8, markerscale=4)
    ax.set_title("Trip 00000077 — errors concentrate where the vehicle is stopped")
    ax = axes[1]
    ax.plot(i, speed, color="#333333", lw=0.8, label="speed over ±10 s window")
    ax.plot(i, step, color="#bbbbbb", lw=0.4, alpha=0.7, label="per-step displacement (m)")
    ax.axhline(STOP_MPS, color="#c0392b", lw=0.8, ls="--", label="stopped threshold")
    for _, r in runs[runs.model == "phase4_hmm"].iterrows():
        if r.length >= 20:
            ax.axvspan(r.start, r.end, color="#6a8fc7", alpha=0.15)
    ax.set_ylim(0, 25)
    ax.set_ylabel("m/s  |  m")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax = axes[2]
    for model, col in (("classifier", "#b0b0b0"), ("phase4_hmm", "#6a8fc7"), ("improved_hmm", "#d9822b")):
        if model in preds:
            w = ((preds[model] != truth) & (truth >= 0)).astype(float)
            ax.plot(i, pd.Series(w).rolling(60, center=True, min_periods=1).mean(), color=col,
                    lw=1.2, label=model)
    ax.set_ylabel("error rate (60-pt window)")
    ax.set_xlabel("GPS point (s)")
    ax.legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(DIR / "trip77_timeline.png", dpi=150)
    plt.close(fig)

    big = runs[runs.model == "phase4_hmm"].head(2)
    fig, axes = plt.subplots(1, len(big), figsize=(12, 3.8))
    for ax, (_, r) in zip(np.atleast_1d(axes), big.iterrows()):
        lo, hi = max(r.start - 30, 0), min(r.end + 30, n - 1)
        seg = np.arange(lo, hi + 1)
        arc = int(pd.Series(truth[r.start:r.end + 1]).mode()[0])
        L = float(np.hypot(nx_[to[arc]] - nx_[fr[arc]], ny_[to[arc]] - ny_[fr[arc]]))
        _, tt = seg_dist(px[seg], py[seg], np.full(len(seg), nx_[fr[arc]]), np.full(len(seg), ny_[fr[arc]]),
                         np.full(len(seg), nx_[to[arc]]), np.full(len(seg), ny_[to[arc]]))
        ax.plot(seg, tt * L, color="#333333", lw=1)
        ax.axvspan(r.start, r.end, color="#6a8fc7", alpha=0.15, label="Phase 4 error run")
        ax.set_title(f"pts {r.start}-{r.end}: position along truth arc {arc} ({L:.0f} m)\n"
                     f"median speed {r.median_speed_w10_mps:.2f} m/s, twin share {r.twin_frac:.2f}",
                     fontsize=9)
        ax.set_xlabel("GPS point")
        ax.set_ylabel("metres from arc start")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(DIR / "trip77_largest_runs.png", dpi=150)
    plt.close(fig)

    # folium map ----------------------------------------------------------
    lat, lon = tr.lat.to_numpy(), tr.lon.to_numpy()
    nlat, nlon = trip["nodes"].lat.to_numpy(), trip["nodes"].lon.to_numpy()
    m = folium.Map(location=[float(lat.mean()), float(lon.mean())], zoom_start=14,
                   tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
    g = folium.FeatureGroup(name="ground-truth route (green)")
    for arc in route:
        folium.PolyLine([[nlat[fr[arc]], nlon[fr[arc]]], [nlat[to[arc]], nlon[to[arc]]]],
                        color="#2e8b57", weight=6, opacity=0.6).add_to(g)
    g.add_to(m)
    colors = {"classifier": "#888888", "phase4_hmm": "#3b6fb6", "improved_hmm": "#d9822b"}
    for model, pick in preds.items():
        g = folium.FeatureGroup(name=f"{model} path", show=model != "classifier")
        seq = [a for k, a in enumerate(pick) if a >= 0 and (k == 0 or a != pick[k - 1])]
        for arc in seq:
            folium.PolyLine([[nlat[fr[arc]], nlon[fr[arc]]], [nlat[to[arc]], nlon[to[arc]]]],
                            color=colors[model], weight=3, opacity=0.8).add_to(g)
        g.add_to(m)
    g = folium.FeatureGroup(name="GPS (red = Phase 4 HMM error)")
    w4 = (preds["phase4_hmm"] != truth) & (truth >= 0)
    for k in range(0, n, 3):
        folium.CircleMarker([lat[k], lon[k]], radius=2, color="#c0392b" if w4[k] else "#222222",
                            fill=True, opacity=0.7,
                            tooltip=f"pt {k} | speed {speed[k]:.1f} m/s | truth {truth[k]} | "
                                    f"P4 {preds['phase4_hmm'][k]}").add_to(g)
    g.add_to(m)
    g = folium.FeatureGroup(name="largest error runs")
    for _, r in runs[runs.model == "phase4_hmm"].head(6).iterrows():
        folium.Marker([lat[r.start], lon[r.start]],
                      tooltip=f"pts {r.start}-{r.end} ({r.length}): {r.verdict}").add_to(g)
    g.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(DIR / "trip77_map.html"))
    print(f"wrote {DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
