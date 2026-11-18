"""Check that the API's inference path reproduces the evaluated results.

For each student_test trip given (default: all), compares

* the API's emission probabilities with ``outputs/predictions/lgbm.parquet``
  (the Phase 2 evaluation), and
* the API's Top-1 with the Phase 4 per-trip HMM result
  (``outputs/predictions/hmm_viterbi.parquet``).

Only meaningful while ``api/config.yaml`` serves the Phase 4 configuration.

    python api/verify_api.py 00000091 00000052
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from inference import ROOT, Predictor


def main(argv) -> int:
    pred = Predictor()
    ids = argv or pred.list_trips()
    ref_p = pd.read_parquet(ROOT / "outputs/predictions/lgbm.parquet")
    ref_h = pd.read_parquet(ROOT / "outputs/predictions/hmm_viterbi.parquet")
    ok = True
    for tid in ids:
        em = pred.emissions(__import__("io_utils").load_trip(pred.data_dir / tid))
        r = ref_p[ref_p.trip_id == tid].merge(em, on=["trip_id", "point_idx", "arc_id"],
                                              suffixes=("_ref", "_api"))
        dmax = float(np.abs(r.y_prob_ref - r.y_prob_api).max())
        out = pred.predict(tid)
        h = ref_h[ref_h.trip_id == tid]
        ref_top1 = float(h.hmm_correct.mean())
        api_top1 = out["metrics"]["top1_accuracy"]
        same_arcs = np.array_equal(np.asarray(out["point_arcs"])[h.point_idx.to_numpy()],
                                   h.arc_id_hmm.to_numpy())
        good = len(r) == len(em) and dmax < 1e-4 and same_arcs
        ok &= good
        print(f"{tid}: pairs {len(em)} matched {len(r)}  max|dp| {dmax:.2e}  "
              f"top1 api {api_top1:.4f} vs phase4 {ref_top1:.4f}  identical path {same_arcs}  "
              f"{'PASS' if good else 'FAIL'}  ({out['metrics']['inference_time_sec']}s)", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
