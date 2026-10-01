# RESULTS — experiment_completeness_free

**The previous experiment's robustness was bought entirely by knowing n₀.** With the AGN
density pinned at truth, taking the survey from complete to 18% completeness costs almost
nothing — the AGN-hosted fraction is still detected at 5.9σ. Free the density and the same
degradation is catastrophic: **7.5σ → 3.1σ at complete, and 6.9σ → 0.9σ the moment the
catalog is anything less than complete.** Completeness and density knowledge are not
separable axes; the completion can substitute a modelled budget for missing hosts only if
you know how many hosts there should be.

The practical threshold is encouraging, though: **knowing n₀_AGN to 10% costs almost
nothing at any completeness** (7.5 → 7.1σ complete, 5.9 → 5.5σ at C = 0.18). The
measurement survives realistic anchoring; it does not survive ignorance.

Setup, arms and scope limits in `DESIGN.md`. Everything — mock, events, surveys, targeted
injection sets — is reused from `../experiment_twotracer_incomplete`, so this adds exactly
one degree of freedom and nothing else. No grid cell is guard-rejected at any rung.

## Detection significance of f_AGN (median / σ)

| C(z ≤ 0.30) | n₀ exact | 5% | 10% | 30% | factor 2 | free |
|---|---|---|---|---|---|---|
| 1.00 | 7.5 | 7.4 | 7.1 | 5.7 | 3.9 | **3.1** |
| 0.94 | 6.9 | 6.8 | 6.4 | 4.6 | 2.5 | **0.9** |
| 0.76 | 7.0 | 6.8 | 6.4 | 4.6 | 2.5 | **1.0** |
| 0.38 | 6.5 | 6.4 | 6.0 | 4.4 | 2.5 | **1.1** |
| 0.18 | 5.9 | 5.8 | 5.5 | 4.2 | 2.7 | **1.6** |

Read across a row and you see what density knowledge is worth; read down a column and you
see what completeness is worth. **The interaction is the result**: along the top row
completeness costs 21%; along the last column it costs 71%, and almost all of that is
spent in the first step away from a complete catalog.

The cliff between "factor 2" and "free" is where the measurement stops being a
measurement. A factor-2 prior — plausible for a real AGN luminosity function — already
sits at 2.5σ once the survey is incomplete.

## σ(f_AGN), 68% half-width

| C | n₀ exact | 5% | 10% | 30% | factor 2 | free |
|---|---|---|---|---|---|---|
| 1.00 | 0.061 | 0.062 | 0.063 | 0.071 | 0.077 | 0.047 |
| 0.94 | 0.059 | 0.060 | 0.063 | 0.075 | 0.087 | 0.072 |
| 0.76 | 0.061 | 0.062 | 0.065 | 0.079 | 0.096 | 0.091 |
| 0.38 | 0.062 | 0.063 | 0.066 | 0.080 | 0.100 | 0.103 |
| 0.18 | 0.067 | 0.069 | 0.072 | 0.091 | 0.124 | 0.161 |

Note that σ alone is a poor summary here and would mislead if quoted on its own: the free
arm at the complete rung is the *narrowest* entry in the whole table (0.047) yet the least
significant of the fixed-n₀ column, because the posterior has collapsed onto small f. That
is why the significance table above is the headline and this one is support.

## The degeneracy

ρ(f_AGN, log10 n₀_AGN) under a flat density prior: **+0.76, +0.87, +0.89, +0.89, +0.89**
across the ladder. It is strong even with a complete catalog and saturates as soon as the
survey thins — exactly the anticipated failure mode: a missing AGN host and an AGN-hosted
event both explain an event in a pixel with no observed AGN, and once most hosts are
missing the two parameters are the same parameter.

`figs/fig_n0_degeneracy.pdf` shows the shape: a banana from (f ≈ 0, low n₀) curving up to
the truth, which sits at its **upper tip** at every rung. Since the tip is where the
density is highest, the direction of the degeneracy is what produces the fixed-anchor
bias reported previously.

Under a flat prior the density itself is recovered **below** truth at every rung —
by −1.22, −0.96, −0.80, −0.66, −0.26 dex (a factor of 17 down to 1.8). That is the
mirror image of the fixed-anchor result: pinned at the true density the completion
manufactures a missing-AGN budget out of the shot noise of a sparse tracer and pushes
f_AGN up; allowed to move, it kills that budget by driving the density down and pulls
f_AGN below truth instead.

**Caveat on the free arm at the top of the ladder.** The flat-prior n₀ posterior has 9.7%
(complete) and 7.5% (m < 21) of its mass against the low edge of the scanned range
[−9.6, −7.1]; at m ≤ 20 the edge mass is ≤ 0.3%. So the two topmost free-arm entries are
partly statements about the range, not the data, and their significances (3.1σ, 0.9σ)
should be treated as range-dependent. The rest of the table is not affected.

## How the centre moves with the assumption

| C | n₀ exact | 5% | 10% | 30% | factor 2 | free |
|---|---|---|---|---|---|---|
| 1.00 | 0.450 | 0.450 | 0.443 | 0.398 | 0.297 | 0.175 |
| 0.94 | 0.403 | 0.403 | 0.394 | 0.340 | 0.221 | 0.071 |
| 0.76 | 0.420 | 0.420 | 0.411 | 0.360 | 0.243 | 0.094 |
| 0.38 | 0.401 | 0.402 | 0.395 | 0.352 | 0.250 | 0.114 |
| 0.18 | 0.394 | 0.396 | 0.394 | 0.376 | 0.325 | 0.240 |

Truth (0.300) falls inside the 68% interval only for the 30% and factor-2 arms, and for
the free arm at the faintest rung. **This is not evidence that a factor-2 prior is the
"right" one.** The fixed-anchor bias is high and the degeneracy pulls low, so some
intermediate prior width necessarily crosses the truth; where it crosses is set by the
size of the unresolved bias, not by anything physical. The honest reading is that the
recovered value is a strong function of an assumption, over a range of assumptions that
are all defensible a priori — which is itself the argument for measuring n₀_AGN
externally rather than marginalising it.

## What this changes for the programme

1. **Anchoring the tracer density to ~10% is the requirement.** That is the design
   specification this experiment produces, and it is a statement about the *AGN survey*,
   not the GW data.
2. **Do not quote σ(f_AGN) from a fixed-n₀ analysis as a forecast.** At fixed n₀ the
   width is nearly completeness-independent and reads as a robust measurement; the same
   configuration with a factor-2 density prior is a 2.5σ result.
3. **The completion's missing-host budget is doing more work than the observed hosts** in
   the incomplete regime, and its amplitude is set entirely by n₀. Everything downstream
   of the completion inherits the density's uncertainty, including the previously reported
   completeness-robustness.

Still deferred, unchanged: the absolute offsets (H₀ from `../experiment_matched_mock`, the
f_AGN high bias under a fixed anchor). Statements here are differential across arms and
rungs, all of which carry both.

## Next

* **3-D (H₀, f_AGN, n₀_AGN).** H₀ is pinned at truth here; the same `--scan fn0`
  machinery generalises, and the (H₀, f) correlation already measured is weak
  (ρ ≈ −0.1 to −0.5), so the marginalised H₀ cost is probably modest — but it is
  untested.
* **Free the galaxy density too**, and free δ as well as n₀ — the density *shape* is
  currently anchored, which is a second favourable assumption of the same kind.
* **A sharper "dies" criterion** than median/σ, e.g. the Bayes factor against f_AGN = 0,
  which does not depend on the posterior being unimodal.

## Reproducing

```
./scripts/run_fn0_ladder.sh     # one (f, log10n0_c2) grid per rung, H0 pinned
python scripts/analyze_n0_arms.py   # every arm by reweighting those grids
python scripts/make_figures.py
```

Figures: `figs/fig_n0_degeneracy.pdf`, `figs/fig_n0_arms.pdf`. Every number regenerates
into `results/n0_arms_summary.json`.
