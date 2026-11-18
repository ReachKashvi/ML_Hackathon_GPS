# PathSnap — HMM / Viterbi route reconstruction

_Generated 2026-11-18T14:39:19 by `scripts/20_eval_hmm.py`._

## Tuned parameters

- **sigma_trans = 20 m**, **beta = 16**, position-aware network distance: **True**
- Tuned on 20 training trips with **out-of-fold** emissions (an lgbm with the untuned config fitted on the trips not being decoded). student_test was not used.

## Headline (student_test, 10 trips)

| | classifier argmax | HMM / Viterbi | delta |
|---|---|---|---|
| Top-1, pooled | 0.8223 | **0.9273** | **+0.1050** |
| Top-1, mean per trip | 0.8381 | 0.9418 | +0.1037 |
| Route continuity | 0.9025 | 0.9823 | +0.0798 |

Without the ground-truth route-break resets (the setting available on unseen data) pooled HMM Top-1 is **0.9273**.

Trips helped / tied / hurt: **10 / 0 / 0**. Every one of the 19,160 GPS points has a decoded arc (0 coasted through empty candidate sets).

## Per trip

| trip | points | classifier | HMM | delta | continuity (cls → HMM) | fixed | broke |
|---|---|---|---|---|---|---|---|
| 00000077 | 3,911 | 0.6295 | 0.7908 | +0.1613 | 0.8923 → 0.9926 | 902 | 271 |
| 00000076 | 1,486 | 0.8143 | 0.9650 | +0.1507 | 0.8444 → 0.9684 | 238 | 14 |
| 00000091 | 874 | 0.8043 | 0.9485 | +0.1442 | 0.8293 → 0.9702 | 135 | 9 |
| 00000007 | 1,354 | 0.8774 | 1.0000 | +0.1226 | 0.9904 → 1.0000 | 166 | 0 |
| 00000065 | 768 | 0.8086 | 0.9115 | +0.1029 | 0.8905 → 0.9844 | 99 | 20 |
| 00000056 | 3,958 | 0.8631 | 0.9586 | +0.0955 | 0.8661 → 0.9664 | 423 | 45 |
| 00000009 | 1,049 | 0.8093 | 0.9018 | +0.0925 | 0.8397 → 0.9408 | 115 | 18 |
| 00000052 | 262 | 0.9313 | 0.9924 | +0.0611 | 0.9349 → 0.9617 | 16 | 0 |
| 00000040 | 2,995 | 0.9142 | 0.9723 | +0.0581 | 0.9516 → 0.9963 | 190 | 16 |
| 00000000 | 2,503 | 0.9289 | 0.9772 | +0.0483 | 0.9564 → 0.9968 | 124 | 3 |

_fixed_ = points the classifier got wrong and the HMM got right; _broke_ = the reverse.

## Observed failure modes

1. **Trip 00000077 — long stops on reverse-twin arcs (the dominant error).**
   This one trip holds **818 of the 1,392 HMM errors (59%)** across all 19,160 points.
   671 of those 818 sit in just four contiguous runs (the longest 275 points), concentrated
   mid-trip (error rate 0.017 in the first fifth, 0.25–0.36 in the middle, 0.10 in the last).

   *Correction (HMM improvement analysis, `outputs/hmm/trip77/`):* an earlier version of
   this summary attributed these errors to the vehicle lapping the route. That is wrong.
   The GPS track is 2.27× the route length only at 1 s sampling; subsampled every 30 s it
   is 0.97× (15.5 km vs 16.0 km), so the excess is GPS jitter (median 8.6 m per 1 s step),
   not extra laps. The error runs are **stops** of up to 4.5 minutes (speed over a ±10 s
   window < 1 m/s) on two-way roads, where the HMM picks the arc's reverse twin (same
   geometry, opposite direction). In the two largest runs the route goes out along an arc
   and back along its twin, and the GPS shows the vehicle arriving, waiting, then driving
   back; the labels keep it on the outbound arc throughout, including the drive back,
   because the label alignment uses distance only and keeps the earlier route position when
   the two are equidistant. **680 of the 818 errors (83%) are such stopped reverse-twin
   points, where the direction cannot be observed and the label is ambiguous**; 138 are
   ordinary model errors. The trip stays in every headline number. Excluding it, HMM Top-1
   on the other nine is 0.9624.

2. **Junction timing — right road, one arc early or late.** Of a 966-point sample of HMM
   errors, **61% pick an arc within 60 m of road distance of the truth**. The decoder has
   the road but switches arcs a fix too early or too late around a junction. It is worst
   where sampling is sparse relative to arc length: trip 00000056 (2.99 GPS points per route
   arc, 164 errors) and trip 00000009 (2.45 points per arc, 103 errors), where short arcs
   can be crossed entirely between two fixes.

3. **Route-break handling works — and does not need the ground truth.** Trip 00000076 has
   a genuine break in its route, reset at GPS point 54. Within ±30 points of it the HMM
   scores **0.918 against the classifier's 0.787**. Decoding it *without* the reset gives
   the identical 0.918: the −20 floor on unreachable transitions lets the decoder cross the
   gap on emission strength alone. That is why the production-realistic figure (no
   `.route` resets) equals the headline one, 0.9273. On the eight training break trips
   checked, the HMM matched or beat the classifier near every break.

Smallest gain: trip 00000000 (+0.048), simply because the classifier was already at
0.929 — the HMM fixed 124 points there and broke only 3.

## Error anatomy

Of a sample of 966 HMM errors, **61%** pick an arc within 60 m of road distance of the true one — the decoder is on the right road but one arc early or late, typically at a junction. **4%** pick an arc with no path to the truth within 2000 m (a different road or a disconnected graph component). In 27% of HMM errors the classifier had it right.

## Limitations

- The transition model is a Gaussian over *(network distance − GPS displacement)*. It does not model speed limits, turn restrictions, or one-way streets beyond the direction already encoded in the arc graph; a U-turn costs the same as going straight if the distances match.
- Unreachable transitions get a floor of −20 rather than −∞, because the road graphs are fragmented (median 19 components per trip). This keeps every trip decodable but lets the decoder teleport between components when emissions demand it.
- Route-break restarts read the ground-truth `.route`, which unseen data will not have; the no-reset figure above is the honest production number.
- sigma and beta were tuned on 20 trips. The grid optimum is flat near the top, so the exact values matter less than their order of magnitude.

## Validation

- [x] HMM params tuned only on training trips (20 trips, none in student_test)
- [x] tuning emissions were out-of-fold; no student_test labels or predictions used
- [x] every student_test GPS point has an HMM arc (19,160 points, 0 coasted)
- [x] HMM >= classifier on a majority of trips (10 helped / 0 tied / 0 hurt)
- [x] break trip 00000076: accuracy within +/-30 pts of breaks 0.9180 (HMM) vs 0.7869 (classifier)
- [x] training break trips: HMM >= classifier near breaks on 8/8
