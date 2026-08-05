# DESIGN — experiment_completeness_free

**Question: how much of the previous experiment's answer was the assumption?**

`../experiment_twotracer_incomplete` found σ(f_AGN) almost independent of survey
completeness — 1.15× degradation across a 6× loss of hosts — but with n₀ pinned at the
true best fit of the density model at every rung. That is the most favourable case
available, and the mechanism makes clear why it flatters: the completion converts a
*known* n₀ into a missing-host budget, and it is that budget which keeps the two tracer
priors distinguishable once their observed hosts are gone. Take the knowledge away and a
missing AGN host and an AGN-hosted event explain the same observation.

So: sample the AGN density instead of fixing it, and ask at what combination of
completeness and density knowledge the AGN-hosted fraction stops being measurable.

## Design

The same five rungs, the same mock, the same events, the same targeted injection sets —
everything is reused from `../experiment_twotracer_incomplete`, so this experiment adds
exactly one degree of freedom and nothing else.

A new scan mode `--scan fn0` maps the 2-D likelihood **L(f_AGN, log10 n₀_AGN)** at each
rung with H₀ pinned at truth. That single grid serves every arm, because a level of prior
knowledge about n₀ is a *reweighting* of one likelihood, never a separate run:

    p(f) ∝ ∫ dg L(f, g) π(g),    g = log10 n₀_AGN

* **fixed** — π a delta at the truth. This is exactly the previous experiment's slice.
* **5% / 10% / 30% / factor 2** — π Gaussian of width log10(1+ε) dex about the truth:
  0.021, 0.041, 0.114, 0.301 dex.
* **free** — π flat over the scanned range.

Grid: f ∈ [0, 1] × 51, log10 n₀_AGN ∈ [−9.6, −7.1] × 201 (0.0125 dex, ≈ 3.3 points per
10%-prior width). Truth is log10 n₀_AGN = −7.7200.

## What is reported

* **σ(f_AGN)** per (rung, arm) — the width.
* **detection significance, median/σ** — whether an AGN-hosted fraction was measured at
  all. This is the metric that answers "where does it die"; a width alone cannot, because
  a posterior can be narrow and centred anywhere.
* **the (f_AGN, n₀_AGN) correlation** and the free-prior n₀ marginal, including how much
  mass sits against the edges of the scanned range — where the free arm rails, its width
  is a statement about the range, not about the data, and must be flagged as such.

## Scope limits, stated up front

* **H₀ is pinned at truth.** This isolates the (f_AGN, n₀_AGN) degeneracy from the
  (H₀, f_AGN) one already measured. A 3-D scan is the natural follow-up and is cheap on
  the same machinery.
* **Only the AGN density is freed.** The galaxy density is far better determined
  observationally, and the sparse tracer is where the degeneracy lives. Freeing both is a
  separate arm.
* **δ stays anchored** at its per-tracer fit. Freeing the density *shape* as well as its
  amplitude is `experiment_density_evolution`.
* **Absolute bias is inherited, not new** — see `../experiment_matched_mock` (H₀) and
  `../experiment_twotracer_incomplete` (f_AGN high under a fixed anchor). Statements here
  are differential across arms and rungs.
