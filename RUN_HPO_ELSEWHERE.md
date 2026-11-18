# Running the Bayesian HPO on another machine

Everything needed is in this repo. The raw dataset is **not** required for the
HPO itself — `outputs/training_pairs.parquet` already contains the 1,228,307
feature rows built from all 80 training trips.

## 1. Setup

```bash
git clone https://github.com/ReachKashvi/ML_Hackathon_GPS.git
cd ML_Hackathon_GPS
python -m venv .venv && .venv/Scripts/activate      # Windows
# python3 -m venv .venv && source .venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
```

**`lightgbm` is pinned to 4.6.0 deliberately.** On the original machine 4.7.0's
Windows binary crashed every fit with `access violation reading 0x0` inside
`LGBM_DatasetSetField`. Do not upgrade it without testing a fit first:

```bash
python -c "
import numpy as np
from lightgbm import LGBMClassifier
X = np.ascontiguousarray(np.random.rand(5000, 23).astype('float32'))
y = (np.random.rand(5000) > 0.85).astype(int)
LGBMClassifier(n_estimators=20, verbose=-1).fit(X, y)
print('lightgbm OK')"
```

## 2. Run the HPO

```bash
python scripts/16_bayesian_hpo.py --n-trials 40 --timeout 3600
```

Full-fidelity cost on the original machine was roughly **78 s per fold** on
983k-row folds, so about **400 s per trial** and ~9 trials per hour. On a
machine with free RAM, raise the budget rather than the fidelity:

| machine | suggested command |
|---|---|
| plenty of RAM, time to spare | `--n-trials 40 --timeout 14400` (4 h, full fidelity) |
| moderate | `--n-trials 40 --timeout 3600 --subsample-rows 300000` |
| tight | `--n-trials 40 --timeout 1800 --subsample-rows 150000` |

`--subsample-rows` caps only the **fit** set per CV fold, sampled as whole
trips so no trip is split. Validation folds are always complete, and the final
model is always refitted on all 80 trips regardless of the cap.

The study is persisted to `outputs/hpo.db` (SQLite). **It resumes**: rerun the
same command and optuna adds trials to the existing `lgbm_hpo` study rather
than starting over. Delete the file to start clean.

### Memory

Peak usage is roughly **1–2 GB** with `--dtype float32` (the default in
training). If you hit `numpy ArrayMemoryError`, lower `--subsample-rows`
rather than reducing folds — fold count is what keeps the estimate honest.

## 3. Compare tuned vs untuned

```bash
python scripts/17_tuned_vs_untuned.py
```

This re-scores the saved untuned `lgbm.joblib` under the local LightGBM,
refits it if its Top-1 has drifted more than 0.005 from the recorded 0.8223,
then scores `lgbm_tuned` and writes:

- `outputs/predictions/lgbm_tuned.parquet` — the Phase 4 HMM's input
- `outputs/figures/hpo_improvement.png`
- an `lgbm_tuned` row in `outputs/test_results.csv`

## 4. Bring the results back

```bash
git add -f outputs/best_hpo_params.json outputs/hpo_trials.csv \
           outputs/models/lgbm_tuned.joblib outputs/test_results.csv \
           outputs/predictions/lgbm_tuned.parquet \
           outputs/figures/hpo_improvement.png
git commit -m "HPO results from <machine>"
git push
```

## What is in the repo for this

| file | size | why |
|---|---|---|
| `outputs/training_pairs.parquet` | 61 MB | the HPO's only required input |
| `outputs/student_test_pairs.parquet` | 6.8 MB | needed by step 3 |
| `outputs/models/lgbm.joblib` | 2.0 MB | untuned baseline for the comparison |
| `data/student_test/` | 4.3 MB | `.arcs` files, needed for route continuity |

`data/train/` (87 MB) is **not** included — nothing in the HPO path reads it.
The leakage check detects its absence and skips the trip-id cross-check,
while still refusing to run on anything but `training_pairs.parquet`.

## Expectation setting

On the original machine, 5 full-fidelity trials moved CV Top-1 from **0.7616**
(untuned) to **0.7691** — **+0.0075**, which is well inside the ±0.074
fold-to-fold variation. Tuning is unlikely to be the step that matters here.
The remaining accuracy is in the **HMM/Viterbi stage**: route continuity sits
at 0.90, meaning about one in ten consecutive arc picks is a physically
impossible jump, and that is what sequence decoding repairs.
