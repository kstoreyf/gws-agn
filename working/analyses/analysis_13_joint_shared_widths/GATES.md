# Analysis 13 gates

Registered 2026-10-01, before any Analysis-13 likelihood value. A miss is a FAIL to diagnose,
never a bound to widen.

| gate | status | the number that decides it |
|---|---|---|
| K closure + joint selection support | **PASS** (job 1350473, 4 min) | K1 4/4 bitwise against A11 at the fiducial widths; K2 both widths live at f = 0, 0.266, 1; K3 6 rejections, all at f = 1 with AGN spin mean ≥ 0.237 (≈ 1,600 nats below the posterior). At the posterior f the logL along Δμ_χ 0.24–0.30 flattens ~30 nats below the peak as N_eff/threshold falls to 1.5 (selection-noise regime; negligible weight) (`diagnostics/a13_closure.json`) |
| S smoke test (single GPU, seed 100) | WITHDRAWN 2026-10-06: no second sampler (owner). Seed 1 (dynesty, core bf58aa6, rslice from it 2885) is production seed 1 | |
| P two-seed production | OWNER-GATED | |
| C calibration campaign | OWNER-GATED, after P | |

## K — closure (before sampling)

- **K1** with both widths at their fiducials, the joint cell equals the Analysis-11 cell in logL,
  logL_pe and logL_selection, within 1e-8 (bitwise expected).
- **K2** each width moves logL at f = 0, at the 11D median f and at f = 1.
- **K3** the live guard on a (σ_G, σ_χ) grid around the Analysis-12 posteriors, and along the
  extended Δμ_χ range (0.20–0.30) at the posterior f and at f = 1. Rejections are recorded with
  how far they sit below the posterior peak; a rejection inside the posterior bulk is a FAIL.

## S — smoke test

**Withdrawn by the owner 2026-10-06: no second sampler.** Criteria kept as registered.

One run per sampler on one GPU, seed 100, problem 13, flat priors as in README. Pass if, between
dynesty (multi/unif, nlive 200, dlogz 0.1, first bound after 2·nlive calls) and the second
sampler:

- logZ agrees within 3 × the combined quoted error;
- every median and 90% end of the eight coordinates agrees within 0.3 posterior sd (the
  dynesty seed-to-seed spread in Analyses 11–12 was ≤ 0.25 sd);
- every pairwise correlation agrees within 0.07.

## P — production (owner-gated)

Two seeds, merged. Seeds agree within 0.3 sd; no prior edge carries more than 1e-3 of the
posterior within 2% of the box; then the comparison with 11D, 12M and 12χ.
