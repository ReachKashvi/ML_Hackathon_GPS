"""Evaluate the tuned HMM on student_test (Phase 4, Steps 3-5).

Step 3 -- decode every student_test trip with the parameters in
``outputs/best_hmm_params.json`` and compare against the classifier's argmax.
Step 4 -- folium maps for the worst, median and best trips by HMM gain.
Step 5 -- ``outputs/hmm_summary.md``.

Comparability
-------------
Top-1 is computed over GPS points that **have at least one candidate**, which
is exactly the denominator behind the classifier's published 0.8223. Points
with no candidate are still decoded (they coast on the previous arc) and are
reported separately as coverage, so the headline comparison is like for like.

Truth is the same monotone route alignment used to build the labels
(``labels.align_gps_to_route``). It is used here for *scoring only*; the
decoder never sees it.

Route-break resets use ``.route`` and are therefore not available on unseen
data. Every trip with a break is also decoded without resets, and the
production-realistic pooled figure is reported beside the headline one.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import folium
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import hmm_graph as hg  # noqa: E402
import hmm_viterbi as hv  # noqa: E402
import io_utils as io  # noqa: E402
import labels as lb  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "outputs"
PRED_DIR = OUT / "predictions"
MAPS = OUT / "maps"
BREAK_WINDOW = 30
TIE_TOL = 0.0005


def continuity(seq: np.ndarray, arcs: pd.DataFrame) -> float:
    """Share of consecutive picks that stay on an arc or follow a road link."""
    if len(seq) < 2:
        return float("nan")
    a, b = seq[:-1], seq[1:]
    ok = (a >= 0) & (b >= 0)
    to_n = arcs["to_node"].to_numpy()
    fr_n = arcs["from_node"].to_numpy()
    linked = np.zeros(len(a), dtype=bool)
    linked[ok] = to_n[a[ok]] == fr_n[b[ok]]
    return float(((a == b) | linked).mean())


def arc_polyline(arc_seq: np.ndarray, trip: dict) -> List[List[List[float]]]:
    """One 2-point segment per distinct consecutive arc, for a multi-polyline."""
    nodes, arcs = trip["nodes"], trip["arcs"]
    lat, lon = nodes["lat"].to_numpy(), nodes["lon"].to_numpy()
    fr, to = arcs["from_node"].to_numpy(), arcs["to_node"].to_numpy()
    seq = [int(a) for i, a in enumerate(arc_seq)
           if a >= 0 and (i == 0 or a != arc_seq[i - 1])]
    return [[[float(lat[fr[a]]), float(lon[fr[a]])],
             [float(lat[to[a]]), float(lon[to[a]])]] for a in seq]


def build_map(trip: dict, rows: pd.DataFrame, label: str, stats: dict) -> folium.Map:
    track = trip["track"]
    lat, lon = track["lat"].to_numpy(), track["lon"].to_numpy()
    fmap = folium.Map(location=[float(lat.mean()), float(lon.mean())],
                      zoom_start=14, tiles="OpenStreetMap", control_scale=True)
    fmap.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])

    nodes, arcs = trip["nodes"], trip["arcs"]
    nlat, nlon = nodes["lat"].to_numpy(), nodes["lon"].to_numpy()

    ref = folium.FeatureGroup(name="Reference route (green)")
    for start, end in hg.load_route_segments(trip):
        chain = np.asarray(trip["route"][start:end])
        walk = [arcs["from_node"].to_numpy()[chain[0]], *arcs["to_node"].to_numpy()[chain]]
        folium.PolyLine([[float(nlat[n]), float(nlon[n])] for n in walk],
                        color="#1a9850", weight=8, opacity=0.55).add_to(ref)
    ref.add_to(fmap)

    cls = folium.FeatureGroup(name="Classifier argmax (orange)")
    folium.PolyLine(arc_polyline(rows["arc_id_classifier"].to_numpy(), trip),
                    color="#eb6834", weight=4, opacity=0.9).add_to(cls)
    cls.add_to(fmap)

    hmm = folium.FeatureGroup(name="HMM / Viterbi (red)")
    folium.PolyLine(arc_polyline(rows["arc_id_hmm"].to_numpy(), trip),
                    color="#d7191c", weight=3, opacity=0.95).add_to(hmm)
    hmm.add_to(fmap)

    gps = folium.FeatureGroup(name=f"GPS points ({len(track):,}, blue)")
    step = max(1, len(track) // 4000)
    for i in range(0, len(track), step):
        folium.CircleMarker([float(lat[i]), float(lon[i])], radius=2, color="#2166ac",
                            fill=True, fill_opacity=0.8, weight=0,
                            popup=f"point {i}").add_to(gps)
    gps.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    legend = (
        '<div style="position:fixed;bottom:24px;left:12px;z-index:9999;background:white;'
        'padding:8px 12px;border:1px solid #999;font:12px sans-serif;line-height:1.6">'
        f'<b>trip {trip["trip_id"]}</b> ({label})<br>'
        f'classifier Top-1 {stats["classifier_top1"]:.4f} &rarr; '
        f'HMM {stats["hmm_top1"]:.4f} ({stats["delta_top1"]:+.4f})<br>'
        '<span style="color:#1a9850">&#9644;</span> reference route &nbsp;'
        '<span style="color:#eb6834">&#9644;</span> classifier &nbsp;'
        '<span style="color:#d7191c">&#9644;</span> HMM &nbsp;'
        '<span style="color:#2166ac">&#9679;</span> GPS</div>'
    )
    fmap.get_root().html.add_child(folium.Element(legend))
    return fmap


def main() -> int:
    started = time.perf_counter()
    params = json.loads((OUT / "best_hmm_params.json").read_text(encoding="utf-8"))
    sigma, beta = float(params["sigma_trans"]), float(params["beta"])
    use_position = bool(params.get("use_position", True))
    assert params.get("student_test_used") is False, "HMM params were tuned on student_test"
    tune_ids = set(params["tune_trip_ids"])
    test_ids = io.list_trips(DATA / "student_test")
    assert not (tune_ids & set(test_ids)), "tuning trips overlap student_test"
    print(f"HMM params: sigma={sigma:g} beta={beta:g} position_aware={use_position} "
          f"(tuned on {len(tune_ids)} training trips)\n")

    preds = pd.read_parquet(PRED_DIR / "lgbm.parquet")
    pairs = pd.read_parquet(OUT / "student_test_pairs.parquet",
                            columns=["trip_id", "point_idx", "arc_id",
                                     "candidate_rank", "frac_along_arc"])
    df = preds.merge(pairs, on=["trip_id", "point_idx", "arc_id"], how="left",
                     validate="one_to_one")
    assert len(df) == len(preds) == 131544, len(df)

    point_rows: List[pd.DataFrame] = []
    trip_rows: List[Dict] = []
    break_rows: List[Dict] = []
    error_rows: List[Dict] = []
    trips: Dict[str, dict] = {}

    for tid in test_ids:
        trip = io.load_trip(DATA / "student_test" / tid)
        trips[tid] = trip
        n = len(trip["track"])
        sub = df[df["trip_id"] == tid]
        alignment = lb.align_gps_to_route(trip)
        truth_all = alignment.sort_values("point_idx")["truth_arc_id"].to_numpy()

        t0 = time.perf_counter()
        dec = hv.decode_trip(tid, sub, trip, sigma, beta, alignment=alignment,
                             use_position=use_position, use_breaks=True)
        segs = hg.load_route_segments(trip)
        dec_nb = (hv.decode_trip(tid, sub, trip, sigma, beta, alignment=alignment,
                                 use_position=use_position, use_breaks=False)
                  if len(segs) > 1 else dec)
        el = time.perf_counter() - t0

        s = sub.sort_values(["point_idx", "y_prob", "candidate_rank"],
                            ascending=[True, False, True])
        cls_pick = s.groupby("point_idx")["arc_id"].first()
        cls_all = np.full(n, -1, dtype="int64")
        cls_all[cls_pick.index.to_numpy()] = cls_pick.to_numpy()
        has_cand = np.zeros(n, dtype=bool)
        has_cand[cls_pick.index.to_numpy()] = True

        hmm_all = dec["arc_id_hmm"].to_numpy()
        hmm_nb_all = dec_nb["arc_id_hmm"].to_numpy()
        hmm_ok = (hmm_all == truth_all) & (truth_all >= 0)
        hmm_nb_ok = (hmm_nb_all == truth_all) & (truth_all >= 0)
        cls_ok = (cls_all == truth_all) & (truth_all >= 0)

        rows = pd.DataFrame({
            "trip_id": tid, "point_idx": np.arange(n),
            "arc_id_hmm": hmm_all, "arc_id_classifier": cls_all,
            "truth_arc_id": truth_all, "hmm_correct": hmm_ok,
            "classifier_correct": cls_ok, "coasted": dec["coasted"].to_numpy(),
        })
        point_rows.append(rows)

        c_top1 = float(cls_ok[has_cand].mean())
        h_top1 = float(hmm_ok[has_cand].mean())
        trip_rows.append({
            "trip_id": tid, "n_points": int(has_cand.sum()),
            "n_points_total": n, "n_coasted": int(dec["coasted"].sum()),
            "classifier_top1": c_top1, "hmm_top1": h_top1,
            "delta_top1": h_top1 - c_top1,
            "hmm_no_resets_top1": float(hmm_nb_ok[has_cand].mean()),
            "classifier_continuity": continuity(cls_all[has_cand], trip["arcs"]),
            "hmm_continuity": continuity(hmm_all[has_cand], trip["arcs"]),
            "n_route_segments": len(segs),
            "hmm_fixed_pts": int((hmm_ok & ~cls_ok & has_cand).sum()),
            "hmm_broke_pts": int((~hmm_ok & cls_ok & has_cand).sum()),
            "decode_sec": round(el, 2),
        })
        trip_rows[-1]["delta_continuity"] = (trip_rows[-1]["hmm_continuity"]
                                             - trip_rows[-1]["classifier_continuity"])

        # error anatomy: when HMM is wrong, how far (by road) is it from truth?
        graph = hg.build_arc_graph(trip)
        wrong = np.flatnonzero(~hmm_ok & has_cand & (truth_all >= 0))
        for t in wrong[:: max(1, len(wrong) // 400)]:
            a, b = int(hmm_all[t]), int(truth_all[t])
            d_fwd = hg.precompute_neighbors(graph, a, trip_id=tid).get(b, np.inf)
            d_back = hg.precompute_neighbors(graph, b, trip_id=tid).get(a, np.inf)
            error_rows.append({"trip_id": tid, "point_idx": int(t),
                               "road_gap_m": float(min(d_fwd, d_back)),
                               "classifier_right": bool(cls_ok[t])})

        if len(segs) > 1:
            resets = sorted(hv.reset_points_from_segments(segs, trip, alignment))
            idx = np.arange(n)
            win = np.zeros(n, dtype=bool)
            for r in resets:
                win |= (idx >= r - BREAK_WINDOW) & (idx <= r + BREAK_WINDOW)
            win &= has_cand
            break_rows.append({
                "trip_id": tid, "reset_points": resets, "window_points": int(win.sum()),
                "classifier_acc": float(cls_ok[win].mean()),
                "hmm_with_resets_acc": float(hmm_ok[win].mean()),
                "hmm_no_resets_acc": float(hmm_nb_ok[win].mean()),
            })
        hg.clear_caches()
        r = trip_rows[-1]
        print(f"{tid}: {r['n_points']:5d} pts  classifier {r['classifier_top1']:.4f} -> "
              f"HMM {r['hmm_top1']:.4f} ({r['delta_top1']:+.4f})  "
              f"cont {r['classifier_continuity']:.4f} -> {r['hmm_continuity']:.4f}  "
              f"[{el:.0f}s]")

    points = pd.concat(point_rows, ignore_index=True)
    per_trip = pd.DataFrame(trip_rows)
    errors = pd.DataFrame(error_rows)

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    points.to_parquet(PRED_DIR / "hmm_viterbi.parquet", index=False)
    per_trip[["trip_id", "n_points", "classifier_top1", "hmm_top1", "delta_top1",
              "classifier_continuity", "hmm_continuity", "delta_continuity",
              "hmm_no_resets_top1", "n_coasted", "hmm_fixed_pts", "hmm_broke_pts",
              "n_route_segments"]].to_csv(OUT / "hmm_vs_classifier.csv", index=False)

    scored = points[points["arc_id_classifier"] >= 0]
    cls_overall = float(scored["classifier_correct"].mean())
    hmm_overall = float(scored["hmm_correct"].mean())
    hmm_nb_overall = float(
        (per_trip.hmm_no_resets_top1 * per_trip.n_points).sum() / per_trip.n_points.sum())
    cls_cont = float((per_trip.classifier_continuity * per_trip.n_points).sum()
                     / per_trip.n_points.sum())
    hmm_cont = float((per_trip.hmm_continuity * per_trip.n_points).sum()
                     / per_trip.n_points.sum())
    helped = int((per_trip.delta_top1 > TIE_TOL).sum())
    hurt = int((per_trip.delta_top1 < -TIE_TOL).sum())
    tied = len(per_trip) - helped - hurt
    regressed = per_trip[per_trip.delta_top1 < -0.02]

    overall = {
        "classifier_top1_overall": cls_overall,
        "hmm_top1_overall": hmm_overall,
        "delta_top1_overall": hmm_overall - cls_overall,
        "hmm_top1_overall_no_route_resets": hmm_nb_overall,
        "classifier_mean_per_trip_top1": float(per_trip.classifier_top1.mean()),
        "hmm_mean_per_trip_top1": float(per_trip.hmm_top1.mean()),
        "classifier_continuity": cls_cont, "hmm_continuity": hmm_cont,
        "n_trips_helped": helped, "n_trips_hurt": hurt, "n_trips_tied": tied,
        "n_points_scored": int(len(scored)), "n_points_total": int(len(points)),
        "n_points_coasted": int(points["coasted"].sum()),
        "sigma_trans": sigma, "beta": beta, "use_position": use_position,
    }
    (OUT / "hmm_overall.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")

    # ---- Step 4: maps ------------------------------------------------------
    MAPS.mkdir(parents=True, exist_ok=True)
    order = per_trip.sort_values("delta_top1").reset_index(drop=True)
    picks = {"worst": order.iloc[0], "median": order.iloc[len(order) // 2],
             "best": order.iloc[-1]}
    map_paths = {}
    for label, row in picks.items():
        tid = row.trip_id
        fmap = build_map(trips[tid], points[points.trip_id == tid], label, row.to_dict())
        path = MAPS / f"hmm_trip_{tid}.html"
        fmap.save(str(path))
        map_paths[label] = (tid, path)

    # ---- validation ----------------------------------------------------------
    checks = [
        (params.get("student_test_used") is False and not (tune_ids & set(test_ids)),
         f"HMM params tuned only on training trips ({len(tune_ids)} trips, none in student_test)"),
        (params.get("emissions", "").startswith("out-of-fold"),
         "tuning emissions were out-of-fold; no student_test labels or predictions used"),
        (bool((points["arc_id_hmm"] >= 0).all()),
         f"every student_test GPS point has an HMM arc ({len(points):,} points, "
         f"{int(points['coasted'].sum())} coasted)"),
        (helped > len(per_trip) / 2,
         f"HMM >= classifier on a majority of trips ({helped} helped / {tied} tied / {hurt} hurt)"),
    ]
    for b in break_rows:
        checks.append((b["hmm_with_resets_acc"] >= b["classifier_acc"],
                       f"break trip {b['trip_id']}: accuracy within +/-{BREAK_WINDOW} pts of "
                       f"breaks {b['hmm_with_resets_acc']:.4f} (HMM) vs "
                       f"{b['classifier_acc']:.4f} (classifier)"))
    train_breaks = OUT / "hmm_break_check_train.csv"
    if train_breaks.exists():
        tb = pd.read_csv(train_breaks)
        ok = int((tb.hmm_with_resets_acc >= tb.classifier_acc).sum())
        checks.append((ok == len(tb),
                       f"training break trips: HMM >= classifier near breaks on {ok}/{len(tb)}"))

    print("\n== validation ==")
    for ok, text in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {text}")

    # ---- Step 5: summary ---------------------------------------------------
    near = errors[errors.road_gap_m <= 60]
    far = errors[np.isinf(errors.road_gap_m)]
    lines = [
        "# PathSnap — HMM / Viterbi route reconstruction",
        "",
        f"_Generated {datetime.now().isoformat(timespec='seconds')} by "
        "`scripts/20_eval_hmm.py`._",
        "",
        "## Tuned parameters",
        "",
        f"- **sigma_trans = {sigma:g} m**, **beta = {beta:g}**, position-aware "
        f"network distance: **{use_position}**",
        f"- Tuned on {len(tune_ids)} training trips with **out-of-fold** emissions "
        "(an lgbm with the untuned config fitted on the trips not being decoded). "
        "student_test was not used.",
        "",
        "## Headline (student_test, 10 trips)",
        "",
        "| | classifier argmax | HMM / Viterbi | delta |",
        "|---|---|---|---|",
        f"| Top-1, pooled | {cls_overall:.4f} | **{hmm_overall:.4f}** | "
        f"**{hmm_overall - cls_overall:+.4f}** |",
        f"| Top-1, mean per trip | {per_trip.classifier_top1.mean():.4f} | "
        f"{per_trip.hmm_top1.mean():.4f} | "
        f"{per_trip.hmm_top1.mean() - per_trip.classifier_top1.mean():+.4f} |",
        f"| Route continuity | {cls_cont:.4f} | {hmm_cont:.4f} | {hmm_cont - cls_cont:+.4f} |",
        "",
        f"Without the ground-truth route-break resets (the setting available on unseen "
        f"data) pooled HMM Top-1 is **{hmm_nb_overall:.4f}**.",
        "",
        f"Trips helped / tied / hurt: **{helped} / {tied} / {hurt}**. "
        f"Every one of the {len(points):,} GPS points has a decoded arc "
        f"({int(points['coasted'].sum())} coasted through empty candidate sets).",
        "",
        "## Per trip",
        "",
        "| trip | points | classifier | HMM | delta | continuity (cls → HMM) | fixed | broke |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in per_trip.sort_values("delta_top1", ascending=False).iterrows():
        lines.append(
            f"| {r.trip_id} | {r.n_points:,} | {r.classifier_top1:.4f} | {r.hmm_top1:.4f} | "
            f"{r.delta_top1:+.4f} | {r.classifier_continuity:.4f} → {r.hmm_continuity:.4f} | "
            f"{r.hmm_fixed_pts} | {r.hmm_broke_pts} |")
    lines += [
        "",
        "_fixed_ = points the classifier got wrong and the HMM got right; "
        "_broke_ = the reverse.",
        "",
        "## Observed failure modes",
        "",
        "<!-- FAILURE_MODES -->",
        "",
        "## Error anatomy",
        "",
        f"Of a sample of {len(errors):,} HMM errors, **{len(near) / max(len(errors), 1):.0%}** "
        f"pick an arc within 60 m of road distance of the true one — the decoder is on the "
        f"right road but one arc early or late, typically at a junction. "
        f"**{len(far) / max(len(errors), 1):.0%}** pick an arc with no path to the truth within "
        f"{hg.DEFAULT_CUTOFF_M:g} m (a different road or a disconnected graph component). "
        f"In {errors.classifier_right.mean():.0%} of HMM errors the classifier had it right.",
        "",
        "## Limitations",
        "",
        "- The transition model is a Gaussian over *(network distance − GPS displacement)*. "
        "It does not model speed limits, turn restrictions, or one-way streets beyond the "
        "direction already encoded in the arc graph; a U-turn costs the same as going straight "
        "if the distances match.",
        "- Unreachable transitions get a floor of −20 rather than −∞, because the road graphs "
        "are fragmented (median 19 components per trip). This keeps every trip decodable but "
        "lets the decoder teleport between components when emissions demand it.",
        "- Route-break restarts read the ground-truth `.route`, which unseen data will not "
        "have; the no-reset figure above is the honest production number.",
        "- sigma and beta were tuned on 20 trips. The grid optimum is flat near the top, "
        "so the exact values matter less than their order of magnitude.",
        "",
        "## Validation",
        "",
    ]
    lines += [f"- [{'x' if ok else ' '}] {text}" for ok, text in checks]
    (OUT / "hmm_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n== overall ==")
    for k, v in overall.items():
        print(f"  {k:36s} {v:.4f}" if isinstance(v, float) else f"  {k:36s} {v}")
    print("\nbreak windows:", json.dumps(break_rows, default=str))
    print("\nregressed by >2 pts:",
          "none" if regressed.empty else regressed[["trip_id", "delta_top1"]].to_dict("records"))
    print("\nmaps:", {k: str(v[1].name) for k, v in map_paths.items()})
    print(f"error anatomy: n={len(errors)} near(<=60m)={len(near)} unreachable={len(far)}")
    print(f"done in {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
