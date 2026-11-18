# PathSnap — GPS map matching

Matches raw GPS traces to road-network arcs. PES University ML hackathon.

## Pipeline

| phase | what | entry point |
|---|---|---|
| 1 | load, EDA, candidates, features, labels | `src/`, `notebooks/` |
| 2 | train and compare 5 model families | `scripts/train_all_models.py` |
| 3 | Bayesian HPO on the winner | `scripts/16_bayesian_hpo.py` |
| 4 | HMM / Viterbi smoothing (groundwork) | `src/hmm_graph.py` |

## Modules

- `src/io_utils.py` — loaders, validation, route-chain splitting
- `src/geo_utils.py` — **per-trip UTM** projection, geometry, bearings
- `src/candidates.py` — R-tree candidate arc generation (K=10, R=50 m)
- `src/features.py` — 23 features per (GPS point, candidate arc) pair
- `src/labels.py` — monotone DP alignment of GPS points onto the route
- `src/eval_utils.py` — group-aware CV, Top-1, route continuity
- `src/model_zoo.py` — 5 models across 5 algorithm families
- `src/hmm_graph.py` — arc-adjacency graph and reachability

## Reproducing

```bash
pip install -r requirements.txt
# place the dataset at data/train/ and data/student_test/
python scripts/build_training_pairs.py          # -> outputs/training_pairs.parquet
python scripts/build_student_test_pairs.py
python scripts/train_all_models.py
python scripts/eval_all_models.py
python scripts/select_top5.py
```

## Findings that shaped the design

- **Trips are worldwide**, not one city (lon −100→+137, 23 UTM zones), so every
  metre calculation uses a per-trip CRS. A fixed EPSG would be wrong for 89 of 90 trips.
- **Each trip ships its own road graph**; `arc_id`/`node_id` are trip-local and
  are never used as numeric features.
- **`.route` is not one-to-one with `.track`** (1.3–57 GPS points per arc), so
  labels come from a monotone alignment, never positional indexing.
- **`.route` annotates only part of some tracks**, so points off the annotated
  route are left unlabelled rather than force-matched.
- **11 trips have genuine route-chain breaks** (10–600 m) handled as segments.

## Results (student test, 10 trips)

| model | Top-1 | Macro F1 | Log loss | Continuity |
|---|---|---|---|---|
| lgbm | 0.8223 | 0.8671 | 0.1811 | 0.9025 |
| rforest | 0.8203 | 0.8590 | 0.2026 | 0.9000 |
| logreg | 0.8090 | 0.7938 | 0.2780 | 0.8910 |
| dtree | 0.8042 | 0.8541 | 0.2473 | 0.8913 |
| gnb | 0.7644 | 0.7830 | 0.7835 | 0.8909 |

Candidate miss rate: 2.11% (train), 0.17% (student test).
