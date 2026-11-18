# PathSnap — model comparison (Section 3.1)

Generated 2026-11-18T13:15:50 from `outputs/cv_results.csv` and `outputs/test_results.csv`.

**Top-1 Accuracy**, **Route Continuity** and **Log Loss** are measured on the student-test split; **CV Macro F1** is the 5-fold GroupKFold mean (± std) over the 80 training trips. **Inference Time** is mean wall-clock `predict_proba` seconds per trip.

| Model | Algorithm Family | Key Configuration | CV Macro F1 | Top-1 Accuracy | Route Continuity | Student-Test Macro F1 | Log Loss | Inference Time |
|---|---|---|---|---|---|---|---|---|
| lgbm | Ensemble (gradient-boosted trees) | 300 trees, lr=0.05, num_leaves=63, scale_pos_weight=auto | 0.8467 ± 0.0335 | 0.8223 | 0.9025 | 0.8671 | 0.1811 | 0.090 s/trip |
| rforest | Ensemble (bagged trees) | 80 trees, max_depth=12, min_samples_leaf=20, class_weight=balanced | 0.8404 ± 0.0284 | 0.8203 | 0.9000 | 0.8590 | 0.2026 | 0.094 s/trip |
| logreg | Linear (logistic regression) | StandardScaler + class_weight=balanced, max_iter=1000 | 0.7830 ± 0.0203 | 0.8090 | 0.8910 | 0.7938 | 0.2780 | 0.007 s/trip |
| dtree | Tree (single decision tree) | max_depth=12, class_weight=balanced | 0.8350 ± 0.0282 | 0.8042 | 0.8913 | 0.8541 | 0.2473 | 0.005 s/trip |
| gnb | Probabilistic (Gaussian naive Bayes) | defaults, no scaler | 0.7742 ± 0.0409 | 0.7644 | 0.8909 | 0.7830 | 0.7835 | 0.009 s/trip |

## Notes

- **Route Continuity** counts "stayed on the same arc" as connected. The literal `arc[i].to_node == arc[i+1].from_node` rule scores a *perfect* map match at 0.097, because 88% of consecutive 1 Hz fixes fall on one arc; the strict figure is in `test_strict_continuity` for reference.
- **Top-1 Accuracy** counts candidate misses as incorrect. `test_top1_resolvable` excludes them (student-test miss rate: 0.17%).
- CV and student test are not equally hard: the CV folds include trips `00000054` and `00000033`, whose GPS tracks are 100-250 m off their own road network. Student test contains no such trip.
