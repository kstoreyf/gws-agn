# Analysis 5 — owner report

**What it asked.** Analyses 3/4 treat the two completion densities as known
inputs. Promote both to sampled parameters and fit the full 4D posterior
**(H0, log10n0 GAL, log10n0_c2 AGN, f_AGN)** at every rung: does the estimator
self-calibrate its own anchors, and what does the honesty cost?

**Scope.** Seed 100, targeted lane, rungs `m21..m18`, `dynesty` static NS
(nlive 1000, dlogz 0.1). Flat priors: H0 [50,100], f [0,1], AGN [-6,-4],
GAL [-4,-1] (widened from the pilot's [-4,-2], which railed).

**Sources.** `results/campaign_summary.json`,
`results/campaign_<rung>_dynesty[_r2]_s100.{h5,json}`,
`figs/fig_campaign_ladder.*`, `fig_anchor_cost.*`, `fig_campaign_corner.*`,
`fig_anchor_degeneracy.*`, `fig_gate0_corner.*`, `results/gate0_summary.json`.

## Result 1 — the cost is paid entirely by `f_AGN`

| rung | σ(H0) free/pinned | σ(f_AGN) free/pinned |
|---|---|---|
| m<21 | 1.06 | **3.36** |
| m<20 | 1.05 | **3.48** |
| m<19 | 1.08 | **2.13** |
| m<18 | 1.01 | **3.19** |

Not knowing the completion densities costs the distance ladder essentially
nothing and the AGN fraction a factor of three. This is the cleanest statement in
the whole campaign of which parameter is exposed to survey-modelling ignorance.

## Result 2 — `H0` is untouched by freeing the anchors

With both densities free: `H0` = 69.15 ± 0.98, 69.12 ± 0.98, 68.79 ± 1.03,
69.65 ± 1.21 down the ladder, against a truth of 67.74. Freeing the anchors moves
the median by less than 0.5 half-widths at `m<21`, `m<20`, `m<19`; only at `m<18`
does it shift appreciably (+1.33, 1.1 half-widths), and that is precisely the rung
where the galaxy anchor breaks (below).

**Read this against the seed scatter, not against truth.** Seed 100 is a high
realisation: analysis 2's five-seed closure on complete catalogs gives
`+0.41 ± 0.55` (`t = 0.73`, consistent with zero) with per-seed offsets spanning
−1.21 to +1.82, and seed 100 alone contributes +1.48. So the `+1.0` to `+1.9`
seen here is this realisation's own draw, and the analysis-5 statement is the
*differential* one: **the anchoring axis contributes nothing to `H0`** — neither
to its width (≤ 8 %) nor to its median (≤ 0.5 half-widths, except at `m<18`).

The one place freeing the anchors changes the `H0` verdict is coverage at the
bright rungs: with anchors pinned, seed 100 puts truth outside 90 % at `m<21` and
`m<20` (pull +2.01, +1.86); with them free, truth is inside 90 % at all four
rungs. On one seed that is a coverage repair worth noting, not a bias measurement.

## Result 3 — `f_AGN` recovers coverage, by widening rather than by moving

Free-anchor medians: 0.465, 0.484, 0.389, 0.384; realised 0.295 inside 68 % at
**all four rungs**. With anchors pinned the same seed misses 68 % at `m<19` and
`m<18` (pulls +1.01, +0.99) at medians 0.361 and 0.368. Freeing the densities
moves those medians by only +0.03 and +0.02 — the coverage comes from the interval,
not from a better central value. The estimator is not correcting its bias; it is
correctly reporting that it cannot resolve it.

## Result 4 — the two anchors behave completely differently

`log10n0_c2` (AGN) is **measured** and recovers truth (−5) inside 68 % at every
rung: −5.17, −5.10, −4.97, −4.89.

`log10n0` (GAL) does not:

| rung | GAL posterior | shape |
|---|---|---|
| m<21 | one-sided, `< -3.12` (90 %) | flat below truth (KS vs prior 0.06) |
| m<20 | one-sided, `< -3.07` (90 %) | flat below truth (KS 0.03) |
| m<19 | **−3.04 ± 0.08** | measured, on truth |
| m<18 | **−1.81 (+0.55, −0.59)** | railed high; 9 % of mass at the prior edge |

At the bright rungs the galaxy anchor is invisible to the data — the completion
budget is negligible, so any small density does nothing, and the quoted median
(−3.53, −3.45) reports *prior volume, not a measurement*. Those two "offsets"
should never be quoted as biases. At `m<19` there is a narrow window where the
completion matters enough to constrain the anchor and it lands on truth to
0.08 dex. At `m<18` the likelihood wants **ten times the true galaxy density**.

## Result 5 — the degeneracy that does the damage

corr(`n_0^{AGN}`, `f_AGN`) = +0.68, +0.67, +0.83, **+0.89** from `m<21` to
`m<18`. As the catalog empties, the missing-AGN budget and the AGN fraction
become the same number — the mechanism analysis 4 measured one arm at a time,
here visible as a single banana. `fig_anchor_degeneracy` is the figure that makes
analyses 4 and 5 one story.

## Result 6 — an external AGN-density prior helps at the faint end and *hurts* at the bright end

Result 5 suggested the obvious follow-up: the inflation is a degeneracy, so an
external constraint on `n_0^{AGN}` — which a real analysis has from an AGN
luminosity function — should buy σ(`f_AGN`) back. Priced by importance-reweighting
the chains already on disk (no new likelihood calls,
`scripts/prior_sensitivity.py`, `results/prior_sensitivity.json`,
`figs/fig_prior_sensitivity.*`). **It does not work the way I expected.**

σ(`f_AGN`) under a Gaussian prior on `log10n0_c2` centred on truth:

| rung | no prior | 0.05 dex | 0.10 | 0.20 | 0.30 | density pinned |
|---|---|---|---|---|---|---|
| m<21 | 0.210 | 0.162 | 0.168 | 0.178 | 0.184 | 0.062 |
| m<20 | 0.219 | 0.167 | 0.174 | 0.184 | 0.192 | 0.063 |
| m<19 | 0.139 | 0.080 | 0.095 | 0.115 | 0.125 | 0.065 |
| m<18 | 0.236 | **0.065** | 0.088 | 0.132 | 0.166 | 0.074 |

And the accuracy, which is the part that matters — `f_AGN` median (truth 0.295)
at a 0.05 dex prior: **m<21 0.584, m<20 0.575, m<19 0.378, m<18 0.297.**

- **At `m<18` it works beautifully.** σ falls 3.6× to 0.065 and the median lands
  on 0.297 against a realised 0.295. That is *better than pinning the density*
  (0.368 ± 0.074, truth outside 68 %) — the best `f_AGN` measurement anywhere in
  the campaign, and the only one that is both tight and unbiased.
- **At `m<21`/`m<20` it is actively harmful.** σ falls only to ~0.16 (a quarter of
  the way to the pinned value) while the median is dragged from 0.46 to 0.58,
  putting the truth **outside 90 %** at 0.05 and 0.10 dex. The pull goes from
  +0.8 to +1.8: the answer gets tighter and wrong.

The mechanism is Result 4. At the bright rungs the completion barely matters, so
the AGN anchor is only weakly identified and the posterior prefers −5.17, below
the truth. With corr = +0.68, forcing the density up to the truth drags `f_AGN`
up with it. At `m<18` the anchor is genuinely identified (−4.885, truth inside
68 %) and the prior is consistent with the likelihood, so it acts as pure
information. **An external prior is only safe on a parameter the data already
identify.**

Attribution, same machinery, 0.05 dex: knowing the *galaxy* density instead
recovers more than knowing the AGN density at the bright rungs (0.124 vs 0.162 at
`m<21`) and less at `m<19` (0.129 vs 0.080). The "both anchors known" cases are
reported but **not usable** at `m<21`, `m<20` and `m<18` (N_eff = 67, 140, 5): a
0.05 dex prior at the truth lands in the far tail of a posterior that railed, so
reweighting cannot reach it. Only `m<19` supports the both-known statement
(0.072, essentially the pinned 0.065). N_eff is reported for every case and
anything below 200 is marked unusable rather than quoted.

## Interpretation

Three claims that stand on their own:

1. **`H0` from dark sirens with AGN tracers is robust to not knowing your
   survey's completeness normalisation; `f_AGN` is not.** That is a design
   statement about what this class of measurement is good for.
2. **The `f_AGN` inflation is a degeneracy, but external information is not a
   free fix.** The posterior is a banana in (`n_0^{AGN}`, `f_AGN`), so an AGN
   luminosity function prior looks like the obvious remedy — and Result 6 shows it
   only *is* one where the data already identify the density. Deep survey
   (`m<18`): a 0.05 dex prior gives `f_AGN` = 0.297 ± 0.065 against a realised
   0.295, better than knowing the density exactly. Shallow survey (`m<21`): the
   same prior tightens σ by a quarter and pushes the truth outside 90 %. The
   actionable statement is conditional, and stating it unconditionally would be
   wrong.
3. **The `m<18` galaxy anchor railing to 10× truth is a real pathology worth its
   own sentence.** Combined with analysis 4's oracle probe (complete AGN + `m<18`
   galaxies inflates `f_AGN` to +0.197), it says the fit compensates a
   badly-incomplete galaxy catalog by inventing galaxies, and the +1.33 `H0` shift
   at that rung — the only rung where freeing the anchors moves `H0` at all — is
   the visible price. This is the one place in the campaign where the anchoring
   axis and the `H0` channel touch, and it is a depth limit, not a bias.

## Diagnostics (internal, not for the reader)

Every run reproduced the stored analysis-3/4 grid log-likelihoods pointwise
before sampling; worst disagreement across the campaign **3.6e-12**, on runs
spanning an A100-80, an H100 and an A100-40. No run hit the 500k call cap; zero
likelihood calls rejected by the variance guard. The `m<18` rung was run twice
with different sampler seeds: medians agree to 0.005–0.014 half-widths on all
four parameters. Gate 0's two independent samplers (`dynesty`, `emcee`) agreed to
0.12 half-widths. Total cost 35.4 GPU-h, 346k likelihood calls.

## Recommendation

**Main text for Results 1, 2, 4 and 5; Result 3 folded into 1.** Specifically:

- The σ table (Result 1) as a small table or one sentence with the four ratios —
  this is the paper's answer to "what if you don't know your completeness?"
- Result 2 as one sentence in Validation: marginalising over the completion
  densities changes `H0` by less than a tenth of its width. This removes a
  systematic a referee will otherwise propose, and it must be phrased
  differentially — seed 100's high `H0` is realisation scatter (analysis 2:
  `+0.41 ± 0.55` over five seeds) and must not be presented as a bias.
- Result 4 as the physics of what a dark-siren fit can and cannot self-calibrate,
  with the `m<18` railing stated as a limit of the method at that depth.
- Result 5 with `fig_anchor_degeneracy` in the main text.
- **Result 6 in the main text, and phrased as the conditional it is.** "An
  external AGN density prior recovers σ(`f_AGN`) where the survey is deep enough
  for the data to identify that density, and biases it where it is not" is a more
  useful and more defensible sentence than the unconditional version, and
  `fig_prior_sensitivity`'s middle panel is the evidence. Quote the `m<18` result
  (0.297 ± 0.065 against 0.295) as the headline number.

**Do not put in the paper:** the wiring-check numbers, the duplicate-rstate
agreement, the Gate-0 two-lane comparison, the call counts, or the GPU hours.
Those are convergence, and per the standing editorial rule they earn no
reader-facing space.

**Caveat to state once:** one seed. The free-vs-pinned *ratios* are clean (same
events, same estimator, one configuration change), the absolute offsets carry
seed 100's own realisation offset.

**Natural next question this creates.** Quantify claim 2: how tight an external
prior on `n_0^{AGN}` is needed to recover σ(`f_AGN`)? That is a cheap
re-marginalisation of chains already on disk — no new likelihood calls. See `../PROPOSAL_analysis_6.md`.
