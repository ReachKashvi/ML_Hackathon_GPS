# HMM improvement: error analysis and experiments

Code: `hmm/` (run order in `hmm/README.md`). Every number below comes from a file in
`outputs/hmm/`; the file is named next to each table.

## Summary

- **Baseline:** the Phase 4 decoder (lgbm emissions, Gaussian transitions, σ = 20 m,
  β = 16, position-aware road distance). Top-1 0.9273 on student_test.
- **Diagnosis:** most remaining errors pick the **reverse twin** of the correct arc, i.e.
  the same road in the opposite direction (two-way streets are stored as two directed
  arcs). On trip 00000077 these are mostly long stops, where the direction cannot be
  observed and the label itself is ambiguous.
- **Experiments:** five transition and emission changes were tested on the 80 training
  trips with 5-fold GroupKFold. **None improved on the baseline.**
- **What helped:** new context features for the classifier (E5): GPS heading and speed
  over a ±3 s / ±10 s window, and how close the same arc is to neighbouring fixes.
  Classifier-only Top-1 rises from 0.759 to 0.880; after the HMM the gain is
  **+0.0040** fold mean (better on 3 of 5 folds; 50 trips better, 23 worse).
- **Selected configuration:** E5 features with the unchanged Phase 4 decoder, recorded in
  `final_config.json` before any student_test run. **The one-off student_test comparison
  (`hmm/08_final_student_test.py`) has not been run yet**, and the API still serves the
  Phase 4 model.

## Setup

- **Emissions:** out-of-fold. Each training trip's probabilities come from an lgbm (the
  Phase 2 config: 300 trees, lr 0.05, 63 leaves, `scale_pos_weight` per fold) fitted on
  the other four folds. Folds are GroupKFold by trip, so no trip is split.
- **Selection:** nested. For each held-out fold, an experiment's best setting is chosen on
  the other four folds and scored on the held-out one. This avoids rewarding experiments
  that simply have a larger grid.
- **Metrics:** the project's definitions.
  - Top-1 = correct / points with ≥ 1 candidate; unassigned or missed truth counts as wrong.
  - Continuity = same arc, or `to_node == from_node`.
  - Decoded macro F1: the decoded arc is the positive prediction among a point's candidates.
- **Decoding:** no route-break resets (the production setting).
- **Reproducibility:** `hmm/hmm_v2.py` with its default settings reproduces the Phase 4
  decoder exactly (`02_run_experiments.py --check`: identical paths on every trip
  checked). Seed 42 throughout.

## Diagnosis

**Error types, Phase 4 HMM on student_test** (`predictions/hmm_viterbi.parquet`):

| | errors | reverse twin | shares a node with the truth | other |
|---|---|---|---|---|
| all 10 trips | 1,375 | 877 (64%) | 406 | 92 |
| excluding 00000077 | 557 | 167 (30%) | 300 (54%) | 90 |

**Road graph.** On a 15-trip training sample, 94% of arcs have a reverse twin. No true
transition between consecutive fixes is unreachable in the arc graph, so the graph itself
is not a source of errors.

**Transition fit.** On the same sample (35,330 true transitions), the residual is road
distance minus GPS displacement:
- median −0.9 m, robust σ (IQR / 1.349) 1.8 m;
- heavy negative tail: 5th percentile −7.5 m, 1st percentile −11.8 m.

σ = 20 with β = 16 behaves like a Gaussian with σ ≈ 5 m.

**Candidate recall** (`candidate_recall_summary.csv`, 80 training trips):

| R, K | miss rate | excluding offset trip 00000054 | pairs vs R50/K10 |
|---|---|---|---|
| 50 m, 10 (used) | 2.11% | 1.57% | 1.00× |
| 75 m, 15 | 1.39% | 0.91% | 1.52× |
| 100 m, 20 | 0.77% | 0.49% | 2.12× |

On student_test the miss rate is already 0.17%, so wider candidate sets could add at most
0.17 points there. They would also more than double the pair table, which this machine's
memory cannot handle through the pipeline. Measured only; not changed.

## Trip 00000077: model errors vs label ambiguity

Files: `trip77/` (`error_runs.csv`, `error_verdicts.csv`, `length_vs_sampling.csv`,
`trip77_timeline.png`, `trip77_largest_runs.png`, `trip77_map.html`).

**Not laps.** The track is 2.27× the route length only at 1 s sampling:

| sampled every | 1 s | 5 s | 10 s | 30 s | 60 s |
|---|---|---|---|---|---|
| GPS length / route length | 2.27 | 1.17 | 1.07 | 0.97 | 0.92 |

The excess is GPS jitter: median 8.6 m per 1 s step, even when the vehicle is stopped.

**Where the errors are.** Each error run was classified from the GPS alone, using the
windowed speed and whether the prediction is the truth arc's reverse twin:

| Phase 4 HMM errors | runs | points |
|---|---|---|
| stopped (< 1 m/s over ±10 s) on the reverse twin: **label ambiguous** | 6 | **680 (83%)** |
| junction timing / other arc: model error | 28 | 110 |
| wrong direction while moving: model error | 17 | 28 |

**Why the labels are ambiguous.** In the two largest runs (275 and 240 points, both stops
of about 4 minutes):
- the route goes out along an arc and back along its twin;
- the GPS shows the vehicle arriving, waiting, then driving back to the arc's start;
- the labels keep it on the outbound arc throughout, including the drive back.

The label alignment (`src/labels.py`) scores distance only and keeps the earlier route
position when two positions are equidistant. While the vehicle is stopped, nothing in the
GPS shows which twin is correct.

The labels were **not** changed, and the trip stays in every headline number.

## Experiments

All experiments use 80 training trips with out-of-fold emissions. The nested fold mean is
the comparison metric (`experiment_comparison_base.csv`, `experiment_comparison_feat.csv`).

| Experiment | Change | Nested fold mean | vs baseline | Folds better / worse |
|---|---|---|---|---|
| Baseline | σ = 20, β = 16 (fixed) | 0.9156 | — | — |
| β/σ grid (A) | baseline grid re-tuned | 0.9123 | −0.0034 | 0 / 4 |
| E1 direction-aware cost (B) | same-arc cost uses signed progress | 0.8982 | −0.0175 | 0 / 5 |
| E1 + jitter tolerance (H) | backward moves ≤ 5/10/20 m treated as jitter | 0.9121 | −0.0035 | 0 / 4 |
| E2 mid-arc U-turns (C) | twin switch without reaching the node | 0.8920 | −0.0236 | 0 / 5 |
| E3 Laplace (D) | Laplace instead of Gaussian | 0.8986 | −0.0170 | 0 / 5 |
| E3 Laplace + signed (E) | | 0.9006 | −0.0150 | 0 / 5 |
| E4 per-point normalisation (F) | emissions renormalised per point | 0.9135 | −0.0021 | 0 / 1 |
| **E5 context features** | 11 GPS-only features, same decoder | **0.9196** | **+0.0040** | 3 / 2 |
| E5 + E4 Platt calibration (G) | `a·logit + b`, fitted on other folds | 0.9193 | +0.0037 | — |
| E5 + H | | 0.9167 | +0.0011 | — |

The last three rows compare against the base baseline. Against the E5 baseline (0.9196),
Platt is −0.0004 and H is −0.0030.

**Why E1 failed.** Reverse-twin errors fell (5,324 → 5,022), but errors on neighbouring
arcs rose (4,060 → 6,425). A stopped vehicle jitters backwards along its arc and pays for
it. Its position cannot go below 0 at the start of the next arc, so that arc absorbs the
jitter for free. The decoder therefore switches arcs too early at junctions. The jitter
tolerance (H) removes most of the damage but does not produce a net gain.

**Why E4 normalisation changes nothing.** Subtracting the same constant from every
candidate at a time step cannot change the Viterbi path. The small difference in the
table comes from selecting β.

**E5 emissions in detail** (`oof_*_pair_metrics.csv`, per fold):

| | macro F1 (folds 0–4) | mean | log loss mean |
|---|---|---|---|
| base | 0.881, 0.899, 0.794, 0.806, 0.874 | 0.851 | 0.221 |
| E5 | 0.936, 0.956, 0.900, 0.907, 0.938 | 0.927 | 0.109 |

The ±10 s heading disagreement (`x_bdiff_w10`) is the second most important feature by
gain (`oof_feat_importance.csv`).

**Classifier only vs HMM, 80 training trips** (`configs_*.csv`):

| emissions | classifier Top-1 | HMM Top-1 (pooled) | HMM fold mean ± std | continuity | decoded macro F1 |
|---|---|---|---|---|---|
| base | 0.7585 | 0.9096 | 0.9156 ± 0.046 | 0.9832 | 0.9556 |
| E5 | 0.8802 | 0.9143 | 0.9196 ± 0.042 | 0.9845 | 0.9585 |

**Where E5 does not help.** On trip 00000010 the E5 classifier is worse (0.607 → 0.538),
and the HMM result then depends heavily on β: 0.64 at β = 16, 0.85 at β = 64 (base
emissions: 0.93). The HMM's gain from E5 is therefore small and not uniform across trips.

**Calibration.** Raw lgbm probabilities are overconfident because of `scale_pos_weight`:
pairs predicted at 0.55 are positive 24% of the time (`reliability_base.csv`). Platt
scaling fixes the reliability curve. Fitted parameters are stable across folds
(a = 0.75–0.91, b = −1.26 to −1.44). It does not change the decoded path enough to matter.

**Runtime.** Decoding all 80 training trips takes about 31 s for the baseline config,
excluding graph building. The U-turn variant takes 83 s.

## Evaluator test set (served model)

Running the API with the Phase 4 model on the 10 evaluator trips gives **Top-1 0.9543
pooled** (0.9540 mean per trip) and continuity 0.9759, computed server-side against each
trip's `.route` (`evaluator_test_api_metrics.csv`). These numbers were used for reporting
only, after the model choice was recorded.

## Limitations

- The E5 gain after the HMM (+0.004) is within fold-to-fold variation (fold std ≈ 0.04),
  although it is positive on 50 of 73 trips that changed.
- Top-1 on trips with long stops on two-way roads is bounded by label ambiguity, not by
  the decoder. A direction-aware label alignment would be needed to measure those trips
  fairly; the labels were left unchanged.
- The selected E5 configuration has not yet been compared on student_test.
