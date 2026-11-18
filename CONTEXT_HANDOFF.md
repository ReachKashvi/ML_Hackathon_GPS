# PathSnap — context handoff (as of 2026-11-18)

Read this first in a new chat. It describes the whole state of the project.

## What this is

**PathSnap**: GPS map-matching for the PES University ML hackathon. Each GPS point
must be assigned to the road arc it was driven on.

- Project dir: `C:\Users\LENOVO\ML Hackathon` (Windows 11, Git Bash + PowerShell)
- GitHub: https://github.com/ReachKashvi/ML_Hackathon_GPS (account `ReachKashvi`, `gh` authenticated)
- Hackathon PDFs (brief, deliverables, policy): `~/Downloads/PathSnap_student_materials/` (not read yet)
- Original data zip: `~/Downloads/Student folder.zip`, extracted to `data/`

## Data (verified, not assumed)

- `data/train/` 80 trips, `data/student_test/` 10 trips. **Evaluator test data is never touched.**
- Per trip `<id>/<id>.{track,nodes,arcs,route}`: tab-separated, no header.
  `.track` = lon lat t_sec (CRLF); `.nodes` row = node_id; `.arcs` row = arc_id (from, to); `.route` = ordered arc_ids.
- **Trips are worldwide**, not Darmstadt (lon −100→+137, 23 UTM zones) → every metre calc uses a
  **per-trip UTM** (zone from the centroid of the `.nodes` bbox). 2 trips straddle zones; not special-cased (<1 m error).
- **Every trip has its own road graph**; arc/node ids are trip-local → `arc_id` is never a feature.
- **`.route` is not 1:1 with `.track`** (1.3–57 GPS points per arc) → labels come from alignment.
- **`.route` covers only part of some tracks** (GPS/route length up to 7.5×). Points off the annotated route
  are left **unassigned** (no label). NOTE: I once wrongly called trips 00000054/00000033 "broken"; they are fine.
- **11 trips have route-chain breaks** (10–600 m gaps): train 02,19,33,41,42,69,75,80,86,89; student_test 76.
- Δt is 1 s almost everywhere; graphs are fragmented (median 19 components).

## Pipeline (all code exists)

| file | role |
|---|---|
| `src/io_utils.py` | loaders, `validate_trip`, `route_segments`, `route_chain_breaks` |
| `src/geo_utils.py` | `project_local(lon,lat,trip_id)` cached per-trip Transformer, `get_trip_epsg`, haversine, point-to-segment, bearings |
| `src/candidates.py` | R-tree candidates, K=10, R=50 m; `candidate_miss_rate` |
| `src/features.py` | 23 features (`FEATURE_COLUMNS`); bearings as sin/cos; fills flagged |
| `src/labels.py` | monotone DP alignment GPS→route; `label_candidates`; unassigned if >100 m |
| `src/eval_utils.py` | GroupKFold by trip, `per_pair_metrics`, `top1_accuracy`, `route_continuity` |
| `src/model_zoo.py` | `ALL_MODELS` = logreg, gnb, dtree, rforest, lgbm (knn/linsvm/mlp in `RETIRED_MODELS`) |
| `src/hmm_graph.py` | arc-adjacency graph, Dijkstra neighbours (cached), `load_route_segments` |
| `src/hmm_viterbi.py` | **Phase 4 decoder**: `build_trellis`, `transition_log_prob`, `viterbi_decode`, `decode_trip` |
| `notebooks/01_eda.ipynb`, `02_folium_viz.ipynb` | EDA → `outputs/eda_summary.csv`; maps |
| `scripts/build_training_pairs.py` | → `outputs/training_pairs.parquet` (1,228,307 rows × 29) |
| `scripts/build_student_test_pairs.py` | → `outputs/student_test_pairs.parquet` (131,544 rows) |
| `scripts/train_all_models.py` | 5-fold CV + final fits → `cv_results.csv`, `models/*.joblib` (`--skip-existing`, `--dtype float32`) |
| `scripts/eval_all_models.py` | student_test eval → `test_results.csv`, figures, `comparison_table.md` |
| `scripts/select_top5.py` | weighted ranking → `top5_shortlist.csv` |
| `scripts/16_bayesian_hpo.py` | optuna HPO on lgbm (sqlite `outputs/hpo.db`) |
| `scripts/17_tuned_vs_untuned.py` | tuned vs untuned on student_test |
| `scripts/19_tune_hmm.py` | tune HMM σ/β on training trips (out-of-fold emissions) |
| `scripts/20_eval_hmm.py` | HMM on student_test + maps + `hmm_summary.md` |

Parquet/models/maps/predictions/logs are gitignored except what was force-added for remote HPO.

## Results

**Phase 2 (student_test, classifier argmax per point)**

| model | Top-1 | Macro F1 | Log loss | Continuity |
|---|---|---|---|---|
| **lgbm** (untuned, chosen) | **0.8223** | 0.8671 | **0.1811** | 0.9025 |
| rforest (80 trees, depth 12, leaf 20) | 0.8203 | 0.8590 | 0.2026 | 0.9000 |
| logreg | 0.8090 | 0.7938 | 0.2780 | 0.8910 |
| dtree | 0.8042 | 0.8541 | 0.2473 | 0.8913 |
| gnb | 0.7644 | 0.7830 | 0.7835 | 0.8909 |

CV Top-1 ~0.74–0.76 (±0.06–0.07 across folds). Candidate miss rate 2.11% train, 0.17% student_test.

**Phase 3 (HPO)**: 40 trials (27 complete/13 pruned), best CV 0.7718. Tuned lgbm **regressed** on
student_test (Top-1 0.8159, log loss 0.2064) → **keep untuned lgbm**.

**Phase 4 (HMM/Viterbi) — current best result**

| | classifier | HMM |
|---|---|---|
| Top-1 pooled | 0.8223 | **0.9273** (+0.1050) |
| Top-1 mean per trip | 0.8381 | 0.9418 |
| Continuity | 0.9025 | 0.9823 |

- Params: **σ=20, β=16, position-aware distance**, tuned on 20 OOF training trips (ridge: β/σ² ≈ 0.04).
- 10/10 trips improved; no regressions; all validation checks PASS; no student_test leakage.
- Same 0.9273 **without** ground-truth route-break resets (production-realistic).
- Trip **00000077** holds 59% of remaining errors (laps the route, 2.27× length; likely label/alignment
  issue). Excluding it: **0.9624**. Other errors: junction timing (61% within 60 m of truth).

## Decisions / deviations from the original briefs (all intentional)

- `route_continuity` counts "stayed on same arc" as connected (strict rule scores a perfect match 0.097).
- `top1_accuracy` also reports `top1_accuracy_resolvable` (excludes points with no true candidate).
- `truth_arc_id` and `point_assigned` are **excluded from features** (they leak the label).
- HMM transitions use **position along arc** (`frac_along_arc`); the brief's arc-end version scored 0.7527 (worse than classifier).
- HMM tuning used **out-of-fold** lgbm predictions (in-sample ones are overconfident).
- float32 feature matrices (memory).

## Environment gotchas (important)

- Python 3.12.10, numpy 2.2.3, pandas 3.0.2, sklearn 1.9.0, optuna 5.1.0.
- **lightgbm pinned to 4.6.0.** 4.7.0's binary crashed every fit (`access violation reading 0x0`) after
  pip installs were interrupted. 4.5.0 is incompatible with sklearn 1.9.
- RAM: 16.7 GB, often nearly full. Earlier crashes were numpy `MemoryError`. Run models sequentially.
- Python processes are named **`python3.12.exe`** (not `python.exe`) — filter on that when checking/killing.
- **Never `Stop-Process` while pip is running** — that corrupted installs before.
- Background jobs die when the Claude session ends; long scripts write results incrementally.
- Feed LightGBM C-contiguous numpy, not pandas slices.
- Read `trip_id` from CSV with `dtype={"trip_id": str}` (zero padding).

## Git state

Pushed (4 commits on `main`): Phase 1–2 code + results, Phase 6 UI (`ui/`, pre-existing, from the user),
requirements fix, remote-HPO bundle (training/student_test parquets, `lgbm.joblib`, `data/student_test`).

**Not yet committed**: `src/hmm_viterbi.py`, `scripts/19_tune_hmm.py`, `scripts/20_eval_hmm.py`,
`outputs/best_hpo_params.json`, `hpo_trials.csv`, `figures/hpo_improvement.png`, all `hmm_*` outputs,
modified `test_results.csv`. Commit attribution line: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Open next steps

1. Commit + push Phase 3/4 work.
2. Optionally investigate trip 00000077 labels (lapping vs monotone alignment).
3. Production inference path for unseen trips: load → candidates → features → lgbm → `decode_trip(use_breaks=False)`.
4. API / UI (a Streamlit UI already exists under `ui/`; check its `api_client.py` contract).
5. Read the hackathon PDFs for deliverable/report requirements.

## Working style the user prefers

Fast, direct, no fluff. Often wants things run end to end quickly. Flag problems with evidence,
don't silently change scope. Keep the session open while background jobs run.
