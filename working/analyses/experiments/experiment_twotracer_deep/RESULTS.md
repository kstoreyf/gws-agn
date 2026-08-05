# RESULTS — experiment_twotracer_deep

**The K=2 deep two-tracer mock is unblocked, and the measurement it was hiding is a
low one.** A catalog-targeted injection lane raises the selection integral's effective
sample size by up to three orders of magnitude, admits the full 200-event sample where
the previous proposal admitted 14 of 41 grid cells, and turns the earlier apparent
recovery of the AGN-hosted fraction into a **1.8σ low** one. The joint plane is now
interpretable and puts H₀ **2.7σ low**, matching the unresolved baseline bias measured
independently in `../experiment_matched_mock`.

## The mock

Assembled from darksirens' own generator components (`scripts/build_twotracer_mock.py`)
so the host draw, detection statistic, population samplers and posterior construction
are all gmd's own — including PR #332's corrected posterior samples:

| | |
|---|---|
| GAL tracer | 1,000,000 hosts, 0.00% empty pixels, log10 n₀ = −5.80 |
| AGN tracer | 12,000 hosts (nested subset), **37.5% empty pixels**, log10 n₀ = −7.72 |
| events | 140 GAL-hosted + 60 AGN-hosted = 200, planted f_AGN = 0.300 |
| detection | gmd noisy network SNR ≥ 8; events span z ∈ [0.023, 0.291] |
| PE | `pe_centering="observed"`, σ_dL = 0.10 |
| pixelisation | nside 32 (nside 16 washes out the sparse tracer's contrast, nside 64 collapses N_eff) |

**Scope limit, stated because it changes what the mock can test:** gmd's catalog is
unclustered (uniform sky, uniform in comoving volume). Here f_AGN is identified purely
by the number-density/sparsity contrast, with none of the clustering-bias contrast the
GLASS mock carries (b = 1.2 vs 2.0). The two mocks probe different parts of the same
channel and are complementary rather than redundant.

## Why the lane was blocked, and the fix

Under `field` sky weighting the selection integral is *catalog-conditioned*: an
injection carries weight only if its redshift lands within a few σ of an actual catalog
host **in its own HEALPix pixel**. As f_AGN rises the integral leans on the sparse
tracer, and a population proposal essentially never lands on it. Measured on this mock
at the full N_obs = 200, against the historical floor N_eff > 5·N_obs = 1000:

| f_AGN | population + uniform | with AGN-targeted branch |
|---|---|---|
| 0.0 | 2,696 ✓ | 2,170 ✓ |
| 0.3 | **497 ✗** | 4,977 ✓ |
| 0.7 | **92 ✗** | 25,198 ✓ |
| 1.0 | **41 ✗** | 65,509 ✓ |

The old proposal decays monotonically in f by a factor of 66 across the grid; the
targeted one *rises* by a factor of 30. The honest cost is at f = 0, where moving a
quarter of the proposal weight off the dense tracer costs 20% of its N_eff — still
2.2× clear of the floor.

`scripts/build_targeted_injections_k2.py` draws a three-branch mixture (0.65 population
+ 0.10 uniform + 0.25 AGN-object-targeted) and stores, for every row, the **exact**
mixture density at that row's coordinates. Two properties make it safe to use:

* the targeted branch reads the **pixelated survey file** — the very object the
  likelihood conditions on, whose `zgals`/`dzgals` *are* the KDE centres and widths of
  the target density — so the proposal cannot drift out of step with the inference
  through a pixelisation or kernel-width mismatch;
* the stored `pdraw` was recomputed by an independent code path (full flat-array pixel
  scan rather than the padded lookup, redshift recovered from the stored canonical
  coordinates) and agrees to **5.5 × 10⁻¹⁴** maximum relative error.

120,000,000 proposals give 351,919 detected injections, of which 20.1% sit on AGN
catalog support — against 1.0% for population draws and 2.6% for uniform ones. The
detected fraction, 2.93 × 10⁻³, is within 2% of gmd's own, so the targeting buys its
coverage without distorting the campaign.

## What the guard was hiding

| run | f_AGN | 68% interval | truth inside? | cells admitted |
|---|---|---|---|---|
| population + uniform, N = 80 | 0.3152 | [0.289, 0.338] | yes — **but see below** | 14/41 |
| AGN-targeted, N = 80 | 0.2157 | [0.166, 0.269] | no (1.6σ low) | 41/41 |
| AGN-targeted, N = 200 | **0.2353** | [0.201, 0.272] | **no (1.8σ low)** | 41/41 |
| GLASS (clustered, N = 1000) | 0.3221 | [0.300, 0.346] | yes | 41/41 |

The first row is the trap this experiment set out to test. Its posterior peaked exactly
at the upper admitted edge with 96.7% of its mass in the top two admitted cells, so
"truth inside the 68% interval" carried no information — the interval was set by where
the guard cut. With the same 80 events and only the injection set changed, the peak
moves from 0.325 to 0.20 and truth leaves the interval. The N = 200 measurement, which
the targeting is what makes possible at all, lands 0.065 below the planted value on a
0.036 half-width.

## The joint plane

MAP (66.25, 0.250); H₀ = 66.40 [65.92, 66.91], f_AGN = 0.2466 [0.210, 0.286], ρ = −0.09.

The scan rejects 1478 of 3321 cells, but **all of them lie at H₀ outside [64.5, 71.25]**,
far from the peak: the posterior mass adjacent to the rejected mask is **0.00000** (it
was 0.459), and **0** of the 68% region's cells touch it (it was 13 of 30). The
correlation ρ = −0.09 replaces the earlier ρ = +0.72, which is now confirmed to have
been entirely a boundary artefact.

So the two-tracer plane is interpretable for the first time, and it says:

* **H₀ is 1.34 low, 2.7σ** — truth outside the 90% interval. This is the same size and
  sign as the unresolved baseline bias measured on the K=1 matched mock (−1.61 ± 0.49
  over five seeds, same code, same corrected PE), so it is almost certainly the same
  defect rather than anything about the two-tracer estimand.
* **f_AGN is low too**, and with ρ ≈ 0 the two offsets are not simply the same degree of
  freedom seen twice. A plausible mechanism is that a systematic shift in the inferred
  distance scale moves events off the redshift support of the sparse tracer's hosts
  preferentially — an AGN host is a rare coincidence to begin with — but that is a
  hypothesis, not a measurement. Testing it requires rebuilding this mock once the
  baseline bias is resolved.

## What is established

- The two-tracer mock builds correctly from the library's own components, with the
  intended density contrast (37.5% empty AGN pixels) and corrected PE.
- The catalog-targeted lane removes the injection-campaign blocker outright: the full
  event sample and the full f grid are now admissible, with margin.
- The previous deep-mock f_AGN number was an artefact of the guard boundary and should
  not be quoted; the number that replaces it is biased low.
- The two-tracer H₀ offset is consistent with the K=1 baseline bias, which remains the
  programme's blocking systematic. **No absolute closure statement is available from
  this mock until that is resolved** — the f_AGN offset in particular cannot yet be
  attributed to the multi-tracer estimand.

## Reproducing

```
python scripts/build_targeted_injections_k2.py --out_path data_derived/injections_targeted_k2.h5 \
    --ndraw 120000000 --validation_json results/injections_targeted_k2_validation.json
./scripts/run_targeted_scans.sh          # f-scan at N=80 and N=200, joint at N=200
python scripts/diag_variance_guard.py ... --f_at {0,0.3,0.7,1.0}   # the N_eff table
python scripts/make_figures.py
```

Figures: `figs/fig_twotracer_targeted.pdf` (the N_eff mechanism, and the f_AGN
posterior before and after), `figs/fig_twotracer_joint.pdf` (the joint plane with the
inadmissible region shaded). Every number regenerates into `results/summary.json`.
