"""Final student_test comparison: Phase 4 baseline HMM vs the HMM-improvement selection.

Run **once**, after the configuration in ``outputs/hmm/final_config.json``
has been chosen on training cross-validation only. Nothing here feeds back
into a choice: the numbers are reported, not optimised.

Emissions on student_test come from a model fitted on all 80 training trips:

* ``base`` -- the existing Phase 2 lgbm (``outputs/predictions/lgbm.parquet``),
  i.e. exactly the emissions behind the Phase 4 0.9273;
* ``feat`` -- the same lgbm config refitted on the E5 feature set.

Platt parameters, if the final config uses them, are fitted on all 80
training trips' out-of-fold predictions (``04_fit_calibration.py`` logic).

Outputs (``outputs/hmm/``): ``final_student_test_per_trip.csv``,
``final_student_test_summary.json``, ``predictions_final_student_test.parquet``
and the comparison figures under ``figures/``.
"""

from __future__ import annotations

import json
import time
from dataclasses import fields

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from common import DATA, OUT, ROOT, SEED, continuity  # noqa: E402
import features as ft  # noqa: E402
import geo_utils as geo  # noqa: E402
import hmm_graph as hg  # noqa: E402
import hmm_viterbi as hv  # noqa: E402
import io_utils as io  # noqa: E402
import labels as lb  # noqa: E402
import model_zoo as mz  # noqa: E402
from hmm_v2 import Config, decode, prepare  # noqa: E402

FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
STOP_MPS = 1.0


def student_emissions(tag: str) -> pd.DataFrame:
    pairs_cols = ["trip_id", "point_idx", "arc_id", "candidate_rank", "frac_along_arc"]
    if tag == "base":
        preds = pd.read_parquet(ROOT / "outputs/predictions/lgbm.parquet")
        pairs = pd.read_parquet(ROOT / "outputs/student_test_pairs.parquet", columns=pairs_cols)
        df = preds.merge(pairs, on=["trip_id", "point_idx", "arc_id"], validate="one_to_one")
        return df
    from lightgbm import LGBMClassifier
    import joblib
    meta = json.loads((OUT / f"oof_{tag}_meta.json").read_text())
    cols = meta["features"]
    tr = pd.read_parquet(OUT / "training_pairs_feat.parquet", columns=cols + ["label"])
    y = tr.label.to_numpy().astype("int32")
    model = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                           scale_pos_weight=mz.compute_scale_pos_weight(y),
                           random_state=SEED, n_jobs=-1, verbose=-1)
    model.fit(np.ascontiguousarray(tr[cols].to_numpy(dtype="float32")), y)
    del tr
    joblib.dump(model, ROOT / "outputs/models/lgbm_hmm_feat.joblib")
    te = pd.read_parquet(OUT / "student_test_pairs_feat.parquet")
    te["trip_id"] = te["trip_id"].astype(str)
    te["y_prob"] = model.predict_proba(np.ascontiguousarray(te[cols].to_numpy(dtype="float32")))[:, 1]
    return te[pairs_cols + ["y_prob", "label"]]


def platt_all_folds(tag: str):
    oof = pd.read_parquet(OUT / f"oof_{tag}.parquet", columns=["label", "y_prob"])
    p = np.clip(oof.y_prob.to_numpy(), 1e-6, 1 - 1e-6)
    z = np.log(p) - np.log1p(-p)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z[:, None], oof.label.to_numpy())
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def speed_w10(trip) -> np.ndarray:
    tr = trip["track"]
    x, y = geo.project_local(tr.lon.to_numpy(), tr.lat.to_numpy(), str(trip["trip_id"]), trip_data=trip)
    t = tr.t_sec.to_numpy(dtype="float64")
    n = len(x)
    i = np.arange(n)
    a, b = np.clip(i - 10, 0, n - 1), np.clip(i + 10, 0, n - 1)
    return np.hypot(x[b] - x[a], y[b] - y[a]) / np.maximum(t[b] - t[a], 1.0)


def error_types(pick, truth, fr, to, moving):
    wrong = (pick != truth) & (truth >= 0)
    h, t = pick[wrong], truth[wrong]
    twin = (fr[h] == to[t]) & (to[h] == fr[t])
    adj = ~twin & ((fr[h] == fr[t]) | (fr[h] == to[t]) | (to[h] == fr[t]) | (to[h] == to[t]))
    stop = ~moving[wrong]
    return dict(errors=int(wrong.sum()), err_twin=int(twin.sum()),
                err_twin_stopped=int((twin & stop).sum()), err_adjacent=int(adj.sum()),
                err_other=int((~twin & ~adj).sum()))


def main() -> int:
    final = json.loads((OUT / "final_config.json").read_text())
    tag = final["emissions"]
    keys = {f.name for f in fields(Config)}
    cfg = Config(**{k: v for k, v in final["config"].items() if k in keys})
    if cfg.emission == "platt":
        a, b = platt_all_folds(tag)
        cfg = Config(**{**cfg.as_dict(), "platt_a": a, "platt_b": b})
    print(f"final: {final['name']}  emissions={tag}  {cfg}")

    base_em = student_emissions("base")
    final_em = base_em if tag == "base" else student_emissions(tag)
    trips = sorted(base_em.trip_id.unique())
    rows, points = [], []
    for tid in trips:
        trip = io.load_trip(DATA / "student_test" / tid)
        fr = trip["arcs"].from_node.to_numpy()
        to = trip["arcs"].to_node.to_numpy()
        truth_all = lb.align_gps_to_route(trip).sort_values("point_idx")["truth_arc_id"].to_numpy()
        moving = speed_w10(trip) >= STOP_MPS
        sb, sf = base_em[base_em.trip_id == tid], final_em[final_em.trip_id == tid]
        pts = np.sort(sb.point_idx.unique())
        truth = truth_all[pts]

        cls = (sb.sort_values(["point_idx", "y_prob", "candidate_rank"], ascending=[True, False, True])
               .groupby("point_idx").arc_id.first().loc[pts].to_numpy())
        t0 = time.perf_counter()
        base = hv.decode_trip(tid, sb, trip, 20.0, 16.0, use_breaks=True)["arc_id_hmm"].to_numpy()
        base_nb = hv.decode_trip(tid, sb, trip, 20.0, 16.0, use_breaks=False)["arc_id_hmm"].to_numpy()
        t_base = time.perf_counter() - t0
        t0 = time.perf_counter()
        new = decode(prepare(tid, sf, trip), cfg)
        t_new = time.perf_counter() - t0
        hg.clear_caches()

        r = dict(trip_id=tid, n_points=len(pts))
        for name, path in (("classifier", None), ("baseline", base), ("baseline_no_resets", base_nb),
                           ("final", new)):
            pick = cls if path is None else path[pts]
            ok = (pick == truth) & (truth >= 0)
            c, n = continuity(pick, fr, to)
            r[f"{name}_correct"] = int(ok.sum())
            r[f"{name}_top1"] = float(ok.mean())
            r[f"{name}_continuity"] = c / n
            for k, v in error_types(pick, truth, fr, to, moving[pts]).items():
                r[f"{name}_{k}"] = v
        r["baseline_s"], r["final_s"] = t_base, t_new
        r["fixed"] = int(((new[pts] == truth) & (base[pts] != truth) & (truth >= 0)).sum())
        r["broke"] = int(((new[pts] != truth) & (base[pts] == truth) & (truth >= 0)).sum())
        rows.append(r)
        points.append(pd.DataFrame({
            "trip_id": tid, "point_idx": pts, "truth_arc_id": truth, "arc_id_classifier": cls,
            "arc_id_baseline": base[pts], "arc_id_final": new[pts], "moving": moving[pts]}))
        print(f"{tid}: baseline {r['baseline_top1']:.4f} -> final {r['final_top1']:.4f} "
              f"(fixed {r['fixed']}, broke {r['broke']})", flush=True)

    df = pd.DataFrame(rows)
    df["delta"] = df.final_top1 - df.baseline_top1
    df.to_csv(OUT / "final_student_test_per_trip.csv", index=False)
    pts_df = pd.concat(points, ignore_index=True)
    pts_df.to_parquet(OUT / "predictions_final_student_test.parquet", index=False)

    N = df.n_points.sum()
    summ = {"final_config": final, "resolved_config": cfg.as_dict(), "n_points": int(N)}
    for name in ("classifier", "baseline", "baseline_no_resets", "final"):
        summ[f"{name}_top1_pooled"] = float(df[f"{name}_correct"].sum() / N)
        summ[f"{name}_top1_mean_trip"] = float(df[f"{name}_top1"].mean())
        for k in ("errors", "err_twin", "err_twin_stopped", "err_adjacent", "err_other"):
            summ[f"{name}_{k}"] = int(df[f"{name}_{k}"].sum())
    for name, col in (("baseline", "arc_id_baseline"), ("final", "arc_id_final"),
                      ("classifier", "arc_id_classifier")):
        c = n = 0
        for tid, g in pts_df.groupby("trip_id"):
            trip_arcs = io.load_arcs(DATA / "student_test" / tid / f"{tid}.arcs")
            cc, nn = continuity(g[col].to_numpy(), trip_arcs.from_node.to_numpy(),
                                trip_arcs.to_node.to_numpy())
            c, n = c + cc, n + nn
        summ[f"{name}_continuity"] = c / n
    summ["delta_top1_pooled"] = summ["final_top1_pooled"] - summ["baseline_top1_pooled"]
    summ["trips_improved"] = int((df.delta > 1e-9).sum())
    summ["trips_regressed"] = int((df.delta < -1e-9).sum())
    summ["final_top1_excl_00000077"] = float(
        df[df.trip_id != "00000077"].final_correct.sum() / df[df.trip_id != "00000077"].n_points.sum())
    summ["baseline_top1_excl_00000077"] = float(
        df[df.trip_id != "00000077"].baseline_correct.sum() / df[df.trip_id != "00000077"].n_points.sum())
    (OUT / "final_student_test_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps({k: v for k, v in summ.items() if not isinstance(v, dict)}, indent=1))

    # figures
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(df))
    ax.bar(x - 0.27, df.classifier_top1, 0.27, label="classifier argmax", color="#b0b0b0")
    ax.bar(x, df.baseline_top1, 0.27, label="Phase 4 HMM (σ=20, β=16)", color="#6a8fc7")
    ax.bar(x + 0.27, df.final_top1, 0.27, label=f"Improved HMM ({final['name']})", color="#d9822b")
    ax.set_xticks(x, df.trip_id, rotation=45, ha="right")
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title(f"student_test per trip — pooled {summ['baseline_top1_pooled']:.4f} → "
                 f"{summ['final_top1_pooled']:.4f}")
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "final_per_trip_top1.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    cats = [("err_twin_stopped", "reverse twin, stopped"), ("twin_moving", "reverse twin, moving"),
            ("err_adjacent", "adjacent arc"), ("err_other", "other")]
    colors = ["#c9c9c9", "#d9822b", "#6a8fc7", "#555555"]
    for j, name in enumerate(("classifier", "baseline", "final")):
        vals = [summ[f"{name}_err_twin_stopped"], summ[f"{name}_err_twin"] - summ[f"{name}_err_twin_stopped"],
                summ[f"{name}_err_adjacent"], summ[f"{name}_err_other"]]
        left = 0
        for (key, lab), v, col in zip(cats, vals, colors):
            ax.barh(j, v, left=left, color=col, label=lab if j == 0 else None)
            left += v
    ax.set_yticks(range(3), ["classifier", "Phase 4 HMM", "Improved HMM"])
    ax.set_xlabel("errors on student_test (19,160 points)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "final_error_types.png", dpi=150)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
