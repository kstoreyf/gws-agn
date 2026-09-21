# Analysis 10 -- closure gate 15.1-15.7 (two-mark mock)

**Verdict: FAIL**  (2026-09-21T09:21:26+00:00)

- events `events_marked_dmu0p10_dmuG5.h5`, md5 `427990378e299850a9c0708d389bc0bf`
- 1 ULP at |lnL| = 4386.58 is 9.094947e-13
- steady state 3.0143 s/eval; build 23.4 s; peak GPU 33726 MiB

| check | deciding number | verdict |
|---|---|---|
| 15.1 both marks zero | worst 0.000e+00 abs (0.00 ULP) over 7 f nodes x 4 terms | PASS |
| 15.2 mass mark zero | worst 0.000e+00 abs (0.00 ULP) over 20 cells x 4 terms | PASS |
| 15.3 spin mark zero | min |dlnL| off zero 29.78; f = 0 frozen True | PASS |
| 15.4 f = 0 inert | 1 distinct total hex over 9 cells (want 1) | PASS |
| 15.5 f = 1 live | min spread 0 (dmu_chi) / 0 (dmu_G) | FAIL |
| 15.6 mu_G liveness | min |dlnL| per 1 Msun: total 10.49, PE 28.15, selection 17.35 (> 1e-3) | PASS |
| 15.7 K=1 endpoints | max |diff| any term 1.819e-12; all exactly 0.0 False | FAIL |

## Projected cost at the measured rate

| arm | cells | GPU-h |
|---|---|---|
| A10_S | 41 | 0.03 |
| A10_chi | 2,501 | 2.09 |
| A10_M | 861 | 0.72 |
| A10_J | 52,521 | 43.98 |

A10-J: 861 rows of 61 cells, 5.50 h per chunk of 8, 21.99 h wall with two workers at a time.
