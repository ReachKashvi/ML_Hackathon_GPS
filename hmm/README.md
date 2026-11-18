# hmm/ — HMM accuracy improvements

Everything for the HMM improvement work lives in `hmm/` (code) and `outputs/hmm/` (results).
The Phase 1–4 pipeline in `src/` and `scripts/` is untouched; `hmm/hmm_v2.py`
imports it and reproduces the Phase 4 decoder exactly with its default config
(`02_run_experiments.py --check`).

Rules followed: student_test is read only by `07` and `08`. Every selection
uses 5-fold GroupKFold by trip on the 80 training trips. No labels were changed
and no trip is excluded. Examiner data is never accessed. Seed 42 everywhere.

## Run order

```bash
python hmm/01_oof_emissions.py --tag base                     # OOF lgbm emissions, 80 trips (~2 min)
python hmm/04_fit_calibration.py --oof base                   # per-fold Platt params (E4)
python hmm/02_run_experiments.py --check                      # v2 decoder == Phase 4 decoder
python hmm/02_run_experiments.py --oof base                   # E0-E4 transition/emission grid (~1 h)
python hmm/02_run_experiments.py --oof base --families G_platt
python hmm/05_build_features.py --partition train             # E5 context features
python hmm/05_build_features.py --partition student_test
python hmm/01_oof_emissions.py --tag feat --pairs outputs/hmm/training_pairs_feat.parquet --extra-features
python hmm/04_fit_calibration.py --oof feat
python hmm/02_run_experiments.py --oof feat --families A_base B_signed G_platt
python hmm/06_candidate_recall.py                             # E6 (measurement only)
python hmm/03_summarize.py --oof base                         # comparison tables
python hmm/03_summarize.py --oof feat
# choose final config from CV -> outputs/hmm/final_config.json, then once:
python hmm/08_final_student_test.py
python hmm/07_trip77_diagnostics.py
```

`02_run_experiments.py` appends per trip to `runs_<oof>.csv` and resumes if
interrupted.

## Files

| file | role |
|---|---|
| `common.py` | paths, fold assignment (GroupKFold over trip ids), continuity |
| `hmm_v2.py` | experimental decoder: signed same-arc progress, mid-arc U-turns, Laplace transitions, emission calibration |
| `01_oof_emissions.py` | out-of-fold lgbm probabilities for every training pair |
| `02_run_experiments.py` | decodes every config on every training trip |
| `03_summarize.py` | per-config table and nested-CV family comparison |
| `04_fit_calibration.py` | Platt scaling fitted on the other folds; reliability table |
| `05_build_features.py` | windowed heading/speed and arc-context features (GPS only) |
| `06_candidate_recall.py` | candidate recall for several (R, K) |
| `07_trip77_diagnostics.py` | trip 00000077 error runs, verdicts, figures, map |
| `08_final_student_test.py` | one-off final comparison on student_test |

The results and their interpretation are in `outputs/hmm/hmm_improvement_report.md`.
