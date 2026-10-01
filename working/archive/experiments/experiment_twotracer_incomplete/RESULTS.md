# RESULTS — experiment_twotracer_incomplete

**The AGN-hosted fraction is remarkably robust to incompleteness, and the expansion rate
is not.** Taking the host survey from complete down to 18% completeness within the GW
horizon — 12,000 AGN hosts down to 28, of which 27 lie inside the horizon —
**σ(f_AGN) grows by 1.15× while σ(H₀) grows by 1.8×.** The measurement stays evaluable
throughout: the likelihood's validity guard never binds at any rung, and in fact becomes
*less* marginal as the survey empties.

The centres are biased (f_AGN recovers ≈ 0.41–0.48 against a planted 0.300); that is not
what this experiment is about and is treated as inherited, not new — see *How to read the
centres*.

## The ladder

Isotropic flux limit on both tracers of the deep two-tracer mock; events untouched.
Completeness quoted within z ≤ 0.30, the range the 200 events occupy. Full setup in
`DESIGN.md`; the complete-catalog pixelation is **bit-identical** to
`../experiment_twotracer_deep`'s own survey files, so the zeroth rung is provably the
same object.

| level | C(z ≤ 0.30) | AGN hosts in horizon | GAL empty pix | AGN empty pix | selection N_eff |
|---|---|---|---|---|---|
| complete | 1.000 | 154 | 0.0% | 37.5% | 146,781 |
| m < 21 | 0.942 | 145 | 1.1% | 94.5% | 165,611 |
| m < 20 | 0.760 | 117 | 18.7% | 97.9% | 173,181 |
| m < 19 | 0.377 | 58 | 56.2% | 99.4% | 191,963 |
| m < 18 | 0.175 | 27 | 83.3% | 99.8% | 220,561 |

## How much the width degrades — the answer

From the joint (H₀, f_AGN) scan, 68% half-widths, and the same numbers as a factor
against the complete rung:

| level | σ(H₀) | × | σ(f_AGN) | × | ρ | cells rejected | mass on the wall |
|---|---|---|---|---|---|---|---|
| complete | 0.656 | 1.00 | 0.0662 | 1.00 | −0.33 | 716/3321 | 0.006 |
| m < 21 | 0.588 | **0.90** | 0.0625 | 0.94 | −0.10 | 728/3321 | 0.000 |
| m < 20 | 0.538 | **0.82** | 0.0624 | 0.94 | −0.04 | 692/3321 | 0.000 |
| m < 19 | 1.094 | 1.67 | 0.0663 | 1.00 | −0.37 | 573/3321 | 0.003 |
| m < 18 | 1.192 | **1.82** | 0.0760 | **1.15** | −0.47 | 440/3321 | 0.011 |

Three things worth stating plainly.

**σ(H₀) is non-monotonic — it improves by 18% before it degrades.** Removing faint hosts
first *helps*: those hosts sit beyond the GW horizon and contribute prior weight at
redshifts no event can occupy, so the flux limit acts as a free noise cut down to
C ≈ 0.76. Past that the cut starts removing hosts the events actually need, and by
C = 0.175 the width is 1.8× the complete-catalog value. If a real survey has to pick a
depth, the interesting implication is that deeper is not automatically better.

**σ(f_AGN) barely moves at all.** This is the surprise. What identifies f_AGN is the
*contrast* between the two completed tracer priors, and the completion preserves that
contrast: as hosts are removed, each tracer's prior migrates from its observed KDE onto
its own smooth `n₀ₖ dV_c/dz (1+z)^δₖ` field, and the two fields still differ by the
number-density ratio that carried the signal in the first place. So the informativeness
of the *design* is nearly completeness-independent — as long as n₀ is known.

**Every rung stays interpretable.** Rejected cells fall from 716 to 440 and never
approach the peak (posterior mass adjacent to the inadmissible region ≤ 1.1%). N_eff
*rises* along the ladder, because the incomplete model's target is increasingly dominated
by the smooth out-of-catalog term, which the population branch of the proposal covers
well. This is the opposite of the failure mode that blocked
`../experiment_twotracer_deep`.

## The null test: is that width real?

A flat width across a 6× change in completeness is exactly the kind of number this
campaign has twice found to be an artefact, so it was checked. `shuffle_event_sky.py`
permutes the per-event (ra, dec) blocks among events: same sky patches, same distances,
same localisation areas, but no event's distance is paired with its own host's redshift.

| level | data f_AGN | null f_AGN | null width / data width | separation |
|---|---|---|---|---|
| complete | 0.4522 ± 0.0619 | 0.1554 ± 0.0525 | 0.85 | **4.79 widths** |
| m < 20 | 0.4227 ± 0.0621 | 0.1200 ± 0.0452 | 0.73 | **4.87 widths** |
| m < 18 | 0.3971 ± 0.0683 | 0.1211 ± 0.0501 | 0.73 | **4.04 widths** |

The null is *equally narrow* but sits 4–5 widths away. Read correctly, that validates the
measurement rather than undermining it: for a mixture weight the likelihood's curvature is
a property of how distinguishable the two priors are — a design quantity, not a
realisation quantity — so a data-independent width is expected. What must be
data-driven is the **location**, and it is, by ~4.8 widths.

It also gives the honest degradation statistic for the AGN identification itself, which
combines width and displacement: **4.79 → 4.04 widths, a 16% loss of significance** from
complete to C = 0.175. That is the number to quote for "how much does incompleteness cost
the AGN detection", and it is close to the 1.15× width figure rather than to the H₀ one.

## How to read the centres

f_AGN recovers high (0.41–0.48 vs 0.300) and H₀ low (65.7–67.0 vs 67.74) at **every**
rung, including the complete one, so neither is a completeness effect. Two separate
inheritances:

- **H₀ low** is the unresolved offset of `../experiment_matched_mock` (−0.80 ± 0.16 over
  20 realisations after both known generator fixes), which
  `../experiment_twotracer_deep` already showed propagating into this exact mock.
- **f_AGN high** is a change of *estimator*, not of data: the same mock and the same
  events under the complete-catalog estimator (`dark_sirens` at log10 n₀ → −12) gave
  f = 0.2353, and switching to the incomplete model with the true per-tracer anchor gives
  0.4522. The completion cannot distinguish "no AGN here" from "no AGN observed here",
  and with only 154 AGN inside the horizon the shot noise in dN_obs/dz is large enough
  that C_AGN(z) < 1 in many bins even when the catalog is genuinely complete — so a
  missing-AGN budget is created out of Poisson noise and deposited in empty pixels, where
  it lets events be AGN-hosted. This is the sparse-tracer form of the
  (f_AGN, n₀_AGN) degeneracy the ladder plan anticipated, and it is the natural subject of
  `experiment_completeness_free`.

Both are deferred by design; the statements above are differential against the complete
rung, which carries both.

## Methodological note worth carrying forward

**Holding the mixture weights fixed across a completeness ladder is the wrong invariant.**
A flux limit leaves only bright, nearby hosts, so the catalog-targeted branch's detection
efficiency rises by ~200× from the complete rung to m < 18 (1.8 × 10⁻³ → 3.8 × 10⁻¹).
At fixed weights that produces ~18M detected rows at the faint end — almost all carrying
negligible importance weight — while starving the population branch, which is the only
branch covering the out-of-catalog term, i.e. exactly the term that grows as the survey
empties. `calibrate_mixture.py` therefore holds the **detected-row split** fixed instead
(10% GAL-targeted + 15% AGN-targeted of detected rows, ~350k detected per rung), solving
for the weights from a per-rung pilot. At the complete rung it returns
0.546/0.100/0.157/0.197 — essentially the hand-picked weights of the previous experiment —
so it generalises that choice rather than departing from it.

Also: **the incomplete estimator with a true per-tracer n₀ anchor is far better
conditioned than the log10 n₀ → −12 complete-catalog trick** — N_eff 147k against 5k on
the same mock. The trick buys an unbiased complete-catalog limit at a real cost in
selection-MC resolution.

Minor caveat: `diag_variance_guard.py` evaluates N_eff at δ = 0 rather than at the
anchored δ (+0.019 / −0.003); the scans use the anchored values. The difference is far
below the margins reported.

## Reproducing

```
python scripts/materialise_tracer_catalogs.py      # + verifies against the deep mock
./scripts/build_ladder.sh                          # pixelate both tracers, anchor n0
python scripts/calibrate_mixture.py                # per-rung proposal weights
./scripts/run_injections.sh                        # one targeted set per rung
./scripts/run_scans.sh                             # guard + f-scan + joint per rung
python scripts/shuffle_event_sky.py ... && ./scripts/run_null.sh
python scripts/make_figures.py
```

Figures: `figs/fig_incomplete_widths.pdf` (posteriors with their nulls; the degradation
curves), `figs/fig_incomplete_joint.pdf` (the plane at each rung). Every number
regenerates into `results/summary.json`.
