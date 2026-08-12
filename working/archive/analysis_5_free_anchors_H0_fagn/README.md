# Analysis 5 — joint 4-parameter fit: (H0, f_AGN) with BOTH completion-density anchors free

Analyses 3/4 fix the completion densities (`log10n0` GAL, `log10n0_c2` AGN); a
real-data estimator can't. This analysis frees both and samples the 4D
posterior **(H0, log10n0, log10n0_c2, fcat_2 = f_AGN)** under flat priors:
does the estimator self-calibrate the anchors, and at what cost in σ?

Status: **COMPLETE** — Gate-0 pilot passed 2026-08-06, the four-rung campaign
finished 2026-08-08. Proposal of record:
`~/.claude/plans/yes-analysis-5-proposal-lovely-eich.md`. Internal record only;
paper wiring is a separate owner gate.

<!-- RESULTS_BANNER -->
> **Not knowing the completion densities costs $H_0$ nothing and $f_{\rm AGN}$ a factor of three.** Freeing both anchors leaves the $H_0$ width untouched at every rung — $\sigma(H_0)$ ratios 1.06, 1.05, 1.08, 1.01 from $m<21$ to $m<18$ — while $\sigma(f_{\rm AGN})$ inflates by 3.36, 3.48, 2.13 and 3.19. The AGN fraction is the parameter that pays for ignorance of the survey's incompleteness; the distance ladder is not.
>
> **$H_0$ does not care.** With both densities free, $H_0$ = $69.15 \pm 0.98$, $69.12 \pm 0.98$, $68.79 \pm 1.03$, $69.65 \pm 1.21$ down the ladder; freeing the anchors moves the median by under 0.5 half-widths at $m<21$, $m<20$ and $m<19$, and only at $m<18$ — the rung where the galaxy anchor breaks — does it shift appreciably ($+1.33$, 1.1 half-widths). Seed 100 sits high against the truth 67.74, but that is this realisation's own draw, not a systematic: analysis 2's five-seed closure on complete catalogs is $+0.41 \pm 0.55$ ($t = 0.73$) with per-seed offsets from $-1.21$ to $+1.82$. The differential statement is the result — **the anchoring axis contributes nothing to $H_0$.** What the extra freedom does buy is coverage on this seed: the pinned-anchor fit puts the truth outside 90 % at $m<21$ and $m<20$ (pull $+2.01$, $+1.86$), the free-anchor fit keeps it inside 90 % at all four rungs.
>
> **$f_{\rm AGN}$ stays honest, but only just.** The realised 0.295 lies inside 68 % at all four rungs once the anchors are free, at medians 0.465, 0.484, 0.389 and 0.384 — the widened interval, not a better median, is what buys the coverage. With the anchors pinned the same seed misses 68 % at $m<19$ and $m<18$ (pulls $+1.01$, $+0.99$) at medians only 0.36-0.37; freeing the densities barely moves those medians (by $+0.03$ and $+0.02$) and simply admits how little the data pin the fraction.
>
> **The AGN anchor is measured; the galaxy anchor is only measured in a narrow window.** $\log_{10} n_0^{\rm AGN}$ recovers its truth ($-5$) inside 68 % at every rung. $\log_{10} n_0^{\rm GAL}$ is invisible to the data at $m<21$ and $m<20$ — a flat posterior below the truth and a one-sided bound $\log_{10} n_0^{\rm GAL} < -3.1$ — is sharply recovered at $m<19$ ($-3.04 \pm 0.08$), and at $m<18$ is driven to $-1.81^{+0.55}_{-0.59}$, ten times the true galaxy density, with 9 % of its mass against the prior edge.
>
> **An external AGN-density prior is only safe where the data already identify that density.** Reweighting these posteriors under a Gaussian prior on $\log_{10} n_0^{\rm AGN}$ centred on truth: at $m<18$ a 0.05 dex prior gives $f_{\rm AGN} = 0.297 \pm 0.065$ against a realised 0.295 — tighter *and* more accurate than pinning the density ($0.368 \pm 0.074$, truth outside 68 %), the best AGN-fraction measurement in the campaign. At $m<21$ and $m<20$ the same prior tightens $\sigma$ only from 0.21 to 0.16 while dragging the median from 0.46 to 0.58, putting the truth **outside 90 %**: where the anchor is weakly identified and the posterior prefers $-5.17$, forcing it to the truth drags the correlated fraction with it. The remedy for the degeneracy is conditional on survey depth, not automatic.
>
> **The degeneracy that inflates $\sigma(f_{\rm AGN})$ is the AGN anchor itself,** correlation $+0.68$, $+0.67$, $+0.83$, $+0.89$ down the ladder: as the catalog empties, the missing-AGN budget and the AGN fraction become the same number, exactly the mechanism analysis 4 measured one arm at a time.
<!-- /RESULTS_BANNER -->

## Configuration of record

Seed 100, targeted injections, rungs `m21 m20 m19 m18`. The exact analysis-3/4
darksirens configuration (K=2 `dark_sirens`, field weighting, hard guard @
`max_likelihood_variance 1e6`, W=4096, Om0 pinned, KDE/delta nuisances at 0),
with the two completion-density anchors promoted from fixed inputs to sampled
parameters.

Flat priors: H0 [50,100], f [0,1], `log10n0_c2` [-6,-4], and **`log10n0`
[-4,-1]** — widened from the pilot's [-4,-2] because the Gate-0 posterior railed
at -2.

Sampler: `dynesty` static nested sampling, nlive 1000, dlogz 0.1, maxcall 500k.
No run hit the call cap, no likelihood call was rejected by the variance guard,
and every run reproduced the stored analysis-3/4 grid log-likelihoods pointwise
before sampling (worst disagreement across the whole campaign:
3.6e-12, on runs spanning three GPU platforms).

## Gate 0

Rung m18, seed 100, two independent lanes — `dynesty` (nlive 200) and `emcee`
(32x1500, peak-initialised) — agreeing to 0.12 half-widths on every parameter,
which is what licensed the single-sampler campaign. Verdict and the cost model
that priced the campaign: `results/gate0_summary.json`,
`figs/fig_gate0_corner.*`.

## Layout

- `scripts/sample_4d.py` — closure build (mirrors analysis-4 `scan_h0f.py`
  block, imports its constants) + wiring check + sampler + summaries.
  Checkpointing via `--checkpoint_file/--checkpoint_every` (added mid-campaign
  so the m21 rung could be migrated between machines).
- `scripts/run_gate0.sh dynesty|emcee`, `scripts/submit_gate0_rita.sbatch` — pilot.
- `scripts/run_campaign_hilda.sh`, `scripts/submit_campaign_henon.sbatch` —
  the campaign, sequential, skipping finished tags and resuming from checkpoints.
- `scripts/make_figures.py` — Gate-0 corner + `results/gate0_summary.json`.
- `scripts/make_campaign_figures.py` — campaign aggregation
  (`results/campaign_summary.json`) + the four campaign figures. Deterministic;
  rerun after any new rung lands.
- `scripts/prior_sensitivity.py` — prices external information on the completion
  densities by importance-reweighting the stored chains (`results/prior_sensitivity.json`,
  `figs/fig_prior_sensitivity.*`). No GPU, no new likelihood calls. Every case
  carries its effective sample size; anything below 200 is marked unusable rather
  than quoted.
- `results/campaign_<rung>_dynesty[_r2]_s100.{h5,json}` — samples + marginals,
  call counts, measured s/eval.

## Figures

| file | what it shows |
| --- | --- |
| `fig_campaign_ladder` | all four parameters vs rung, free anchors against the pinned-anchor reference |
| `fig_anchor_cost` | σ(H0) and σ(f_AGN) inflation from freeing the anchors |
| `fig_campaign_corner` | the four rungs' 4D posteriors overlaid |
| `fig_anchor_degeneracy` | the (n0_AGN, f_AGN) banana per rung, and where the GAL anchor goes |
| `fig_prior_sensitivity` | σ(f_AGN) and its pull vs the width of an external AGN-density prior |
| `fig_gate0_corner` | Gate-0 two-lane agreement |

## Reproducibility

The m18 rung was run twice with different sampler seeds (rstate 7 and 23):
medians agree to 0.005-0.014 half-widths on all four parameters
(`campaign_summary.json:m18_duplicate_rstate`). Campaign cost: 35.4 GPU-h,
346k likelihood calls, across an A100-80 (Gate 0), an H100 (m18-m20) and an
A100-40 (m21).
