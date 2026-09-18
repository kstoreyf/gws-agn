# Gate A - null / equivalence

**PASS**. With the two branch populations identical (`mu_chi_c2 = 0`, i.e. `dmu_chi = 0`), the tracer-dependent population path reproduces the Analysis-2 model to the last bit of double precision: the largest disagreement anywhere in this gate is **1.8189894035458565e-12**, which is 2 ULP of a log-likelihood of order 4.2e3.

- gws-agn `ac74b98`, darksirens `af896ca` (one local commit on pinned base `2b86a2d`, ancestor check True)
- seed 100, complete catalogs, H0 = 67.74, Om0 = 0.3075, f_true = 0.30 (realised 0.295), 1000 events x 2000 samples, Ndraw = 1.5e+08
- OLD sampled labels: `['H0', 'log10n0', 'delta', 'sigma_kde', 'log10n0_c2', 'delta_c2', 'sigma_kde_c2', 'fcat_2']`
- NEW sampled labels: `['H0', 'log10n0', 'delta', 'sigma_kde', 'log10n0_c2', 'delta_c2', 'sigma_kde_c2', '$\\mu_\\chi$_c2', 'fcat_2']`
- K=1 reference labels: `['H0', 'log10n0', 'delta', 'sigma_kde']`

| check | result | criterion | verdict |
|---|---|---|---|
| A0 the new coordinate is live | dlogL = -18.2976 at f = 0.295, and exactly 0 at f = 0 | moves, and only through its own branch | PASS |
| A1 cell equivalence | max abs diff 1.818989e-12 = 2 ULP (rel 4.356e-16) | <= 4 ULP | PASS |
| A2 endpoint reduction | max abs diff over PE, selection and total 0.000e+00 | <= 1.0e-06 | PASS |
| A3 selection reduction | max rel diff in mu(f) 7.897e-16 | <= 1.0e-06 | PASS |
| A4 posterior reproduction | max abs diff 1.818989e-12 = 2 ULP over 101 cells; f median moves by -3.6e-15 | <= 4 ULP | PASS |
| guard | min N_eff/threshold 75.7, 0 rejected | > 1, none rejected | PASS |

## On the tolerance, stated before the numbers

GATES.md documents the expected residual as 1-2 ULP and quotes it as `<= 3.552713678800501e-15` absolute. That bound was measured on a toy whose log-likelihood is of order 8, where it is exactly 2 ULP (`np.spacing(8.0) = 1.7763568394002505e-15`). The seed-100 log-likelihood is of order 4.2e3, where 1 ULP is `9.0949470177292824e-13`. A last-bit effect scales with the magnitude of the number carrying it, so the same 1-2 ULP expectation is `1.8189894035458565e-12` here, and an absolute bound cannot travel between the two problems. The criteria applied below are therefore in ULP of the returned value: the scale-free form of the same expectation, bounded at 4 ULP.

This is the one place where the check I wrote down before the run was not the check I applied, so it is worth being explicit: the pre-registered A1 and A4 bounds were `1e-12` absolute, and the measurement is `1.819e-12`, so on the unit as written they would read FAIL. They are recorded verbatim in the JSON under `tolerances_preregistered`. The correction is a unit, not a loosening (4 ULP is 7.1e-15 on the toy, tighter there than the quoted 3.6e-15), and it is not what the gate rests on. A4 carries a control that needs no tolerance at all: the UNCHANGED analysis-2 configuration, re-run through the current code, lands the SAME 2 ULP from the stored array as the new path does. Whatever that residual is, it is not the new population path.

## A0 - the new coordinate is live

Gate A is an equivalence test, and an equivalence test passes for free if the new coordinate does nothing. GATES.md records precisely that defect earlier in this campaign: `build_parameter_space` rejected `<pop>_c{k}`, so `mixture_pop_params` was always empty and a production run would have silently kept the shared-population model. Two cells close that hole.

- **Reachable.** At f = 0.295, `mu_chi_c2` 0 -> 0.1 moves logL from -4176.213667864913 to -4194.5112557800185, **dlogL = -18.297588**. The parameter is connected.
- **Connected to the right branch.** At f = 0, catalog 2 carries zero mixture weight, so its population cannot enter. The same offset leaves logL **bitwise identical** (-4204.143251633603 both times, dlogL = 0.0).

## A1 - cell-level equivalence

NEW minus OLD at H0 = 67.74, full precision, with the hex form so the last bit is visible.

| f | logL (OLD) | logL (NEW) | difference | ULP | relative |
|---|---|---|---|---|---|
| 0.0 | `-0x1.06c24ac23996cp+12` | `-0x1.06c24ac23996cp+12` | 0.0 | +0.0 | 0.000e+00 |
| 0.295 | `-0x1.05036b2efec00p+12` | `-0x1.05036b2efec02p+12` | -1.8189894035458565e-12 | -2.0 | 4.356e-16 |
| 0.5 | `-0x1.05c66681ba16ep+12` | `-0x1.05c66681ba170p+12` | -1.8189894035458565e-12 | -2.0 | 4.343e-16 |
| 1.0 | `-0x1.4ce17b5dab3c8p+12` | `-0x1.4ce17b5dab3c8p+12` | 0.0 | +0.0 | 0.000e+00 |

| f | logL (OLD) | logL (NEW) |
|---|---|---|
| 0.0 | -4204.143251633603 | -4204.143251633603 |
| 0.295 | -4176.213667864911 | -4176.213667864913 |
| 0.5 | -4188.400026060974 | -4188.4000260609755 |
| 1.0 | -5326.0926186264915 | -5326.0926186264915 |

Max abs difference **1.8189894035458565e-12** (2 ULP), max relative 4.356e-16; 2 of 4 cells are bitwise identical -- exactly 0 at f = 0 and f = 1, nonzero at f = 0.295 and f = 0.5, which is the pattern GATES.md predicts.

The difference does not sit where GATES.md expected it. `log_mu` is bitwise identical between the two shapes at every f (True), so the selection contribution agrees to the last bit (max 0.000e+00) and the whole residual is in the per-event PE sum (max 1.819e-12). GATES.md recorded it as 'isolated to the last bit of the selection term' on the toy. Same effect, same size, different seam -- worth noting, not a failure: the re-association `logsumexp_k[a_k] + c -> logsumexp_k[a_k + c]` is applied at both seams and which one rounds differently is a property of the numbers, not of the model.

## A2 - endpoint reduction

The K=1 references are the same machinery on one survey, as analyses 0 and 2 build them: same events, same injections, same nuisance point, K=1 parameter space. The mixture is required to collapse onto them.

| endpoint | term | K=2 (OLD) | K=2 (NEW) | K=1 reference | max abs diff |
|---|---|---|---|---|---|
| f=0.0 (K1_GAL) | total | -4204.143251633603 | -4204.143251633603 | -4204.143251633603 | 0.0e+00 |
| f=0.0 (K1_GAL) | PE | -16009.53746018845 | -16009.53746018845 | -16009.53746018845 | 0.0e+00 |
| f=0.0 (K1_GAL) | selection | 11805.394208554848 | 11805.394208554848 | 11805.394208554848 | 0.0e+00 |
| f=1.0 (K1_AGN) | total | -5326.0926186264915 | -5326.0926186264915 | -5326.0926186264915 | 0.0e+00 |
| f=1.0 (K1_AGN) | PE | -17122.10095931653 | -17122.10095931653 | -17122.10095931653 | 0.0e+00 |
| f=1.0 (K1_AGN) | selection | 11796.008340690038 | 11796.008340690038 | 11796.008340690038 | 0.0e+00 |

Worst discrepancy over both endpoints, both shapes and all three terms: **0.0e+00**. Not 'within tolerance' -- exactly zero, bit for bit, in the total, the PE contribution and the selection contribution alike.

## A3 - selection reduction

Both branch integrals come from the ONE shared injection pool and the one `pdraw`, read off the K=2 mixture at the two endpoints, so their Monte-Carlo errors are common and do not inflate the comparison.

- `mu_GAL = 7.471462445318e-06` (log mu = -11.804419802168809)
- `mu_AGN = 7.544556698731e-06` (log mu = -11.794684221780553)
- `mu_AGN / mu_GAL = 1.009783125`

| f | shape | mu measured | (1-f) mu_GAL + f mu_AGN | relative difference |
|---|---|---|---|---|
| 0.0 | old | 7.471462445318e-06 | 7.471462445318e-06 | 0.000e+00 |
| 0.0 | new | 7.471462445318e-06 | 7.471462445318e-06 | 0.000e+00 |
| 0.295 | old | 7.493025250075e-06 | 7.493025250075e-06 | 3.391e-16 |
| 0.295 | new | 7.493025250075e-06 | 7.493025250075e-06 | 3.391e-16 |
| 0.5 | old | 7.508009572025e-06 | 7.508009572025e-06 | 7.897e-16 |
| 0.5 | new | 7.508009572025e-06 | 7.508009572025e-06 | 7.897e-16 |
| 1.0 | old | 7.544556698731e-06 | 7.544556698731e-06 | 0.000e+00 |
| 1.0 | new | 7.544556698731e-06 | 7.544556698731e-06 | 0.000e+00 |

**Read the near-equality carefully.** The two branch integrals differ by +0.0098 in fractional terms, which is 6x the naive Monte-Carlo scale 1/sqrt(N_eff) = 1.51e-03. That is not a contradiction of the chi_eff-independence of the v3 detection rule, and it is worth stating plainly because it would be easy to quote the wrong expectation: the GAL and AGN branches carry DIFFERENT spatial priors p_k(z | pix), so their selection integrals should not coincide. What chi_eff-independence predicts is that the SPIN factor cannot move mu, and that is measured separately:

- at f = 0.295, moving the AGN branch spin mean by 0.1 changes `log mu` by -3.442e-04, a fractional change in mu of -3.441e-04, i.e. 0.30 Monte-Carlo sigma. mu is blind to the spin mean, as the normalised spin density requires.

Neither number is hard-coded anywhere; both are measured and reported, and the A3 pass criterion is the mu(f) linearity alone.

## A4 - Analysis-2 posterior reproduction

The same 101-point f grid at H0 = 67.74, element-wise against `/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_2_complete_catalog_H0_fagn/results/fscan_s100.h5`.

| arm | max abs diff | in ULP | cells exactly 0 | after removing a constant offset | offset | s/eval |
|---|---|---|---|---|---|---|
| NEW | 1.818989e-12 | 2 | 84/101 | 1.837e-12 | -1.80e-14 | 1.71 |
| OLD | 1.818989e-12 | 2 | 87/101 | 1.891e-12 | -7.20e-14 | 1.67 |

Both arms take only the ULP multiples [-2.0, 0.0, 2.0] -- never a fraction of a ULP, which is what a genuine model difference would look like.

| quantity | stored (analysis 2) | recovered (NEW) | difference |
|---|---|---|---|
| f median | 0.26649932068725635 | 0.2664993206872528 | -3.553e-15 |
| f MAP | 0.27 | 0.27 | 0.0e+00 |
| 68% interval | [0.22120386127565983, 0.3129548590365098] | [0.22120386127565098, 0.31295485903650794] | [-8.9e-15, -1.9e-15] |
| 90% interval | [0.19221119171620987, 0.3437490357029752] | [0.19221119171620615, 0.3437490357029739] | [-3.7e-15, -1.3e-15] |
| max logL | -4176.015021062542 | -4176.015021062542 | 0.0e+00 |

The posterior summary uses `marginal_ci` imported from analysis 2's own `scan_h0f.py`, not a reimplementation.

**The control.** Running the OLD configuration through the CURRENT code gives max abs diff **1.818989e-12** against the same stored array -- 2 ULP, identical to the NEW arm's (True), with 87/101 cells exactly zero. The residual is therefore the last-bit floor of re-running this likelihood on this machine, not the new population path and not code drift since analysis 2 ran.

## Guard

The historical guard (`N_eff > 5 N_obs`, with `max_likelihood_variance = 1e+06` making the total-variance criterion inert), carried on every cell of every check. Over all 216 recorded cells:

- N_eff: min **378742**, median 643013, max 769582
- threshold 5000; **min N_eff / threshold = 75.7**
- guard-rejected cells: **0**; -inf cells: **0**

No number in this gate sits behind a rejected guard cell.

## Files

- `/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_8_marked_multitracer_H0_fagn/diagnostics/null_equivalence.json` - every number above, machine-readable
- `/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_8_marked_multitracer_H0_fagn/scripts/a8_likelihood.py` - reusable builder, shared with the production scan
- `/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_8_marked_multitracer_H0_fagn/scripts/gate_a_null_equivalence.py` - this gate; exact command lines in its header

