# Analysis 5 — owner report (selection-mode redo)

**What it asked.** Analyses 3, 4 and 6 pin the completion densities at the
mock's truths, so the estimator is the only moving part. Analysis 5 removes that
support: **both** densities free under flat priors, sampled jointly with `H0`
and `f_AGN`, at every rung of the ladder. It is the only configuration in the
campaign where the data alone have to identify the densities. The archived
campaign's finding was that at `m<18` the galaxy anchor **railed** — it ran to
`log10 n0 = −1.81` against a truth of −3.0, a 1.19 dex error hard against the
prior wall. This redo asks whether the estimator was responsible.

4 rungs × 4 free parameters, dynesty, `nlive = 1000`, `dlogz = 0.1`, flat priors
`H0 ∈ [50,100]`, `log10 n0 ∈ [−4,−1]`, `log10 n0_c2 ∈ [−6,−4]`, `f_AGN ∈ [0,1]`.
The luminosity-function shape parameters are pinned at their offline fit centres
in every arm, so all arms sample the same four parameters.

**Sources.** `results/campaign_{rung}_dynesty_s100.json`,
`results/free_anchor_summary.json`, `figs/fig1_free_anchors.*`,
`figs/fig2_degeneracy.*`. darksirens `0c5b3db`, K = 2 field mixture, 1000
events, targeted-injection lane, seed 100. Total 13.6 + 6.4 + 3.4 + 2.7 h on one
H100.

## Result

**Selection recovers the galaxy anchor at every rung. per_pixel does not.**

| rung | log10 n0, per_pixel | log10 n0, selection | truth |
|---|---|---|---|
| m<21 | −3.535 | −3.198 | −3.0 |
| m<20 | −3.454 | −3.256 | −3.0 |
| m<19 | −3.038 | −3.043 | −3.0 |
| m<18 | **−1.808** | **−3.118** | −3.0 |

The archived `m<18` posterior does not merely miss — it piles against the upper
prior wall (`fig2_degeneracy`, rightmost panel, blue). Under selection the same
data put the anchor 0.12 dex from truth with truth inside the 90 % interval.

**The evidence separates the estimators exactly where completion matters.**

| rung | lnZ, per_pixel | lnZ, selection | Δ lnZ |
|---|---|---|---|
| m<21 | −4180.42 | −4180.76 | −0.34 |
| m<20 | −4180.77 | −4181.13 | −0.36 |
| m<19 | −4190.57 | −4183.91 | **+6.7** |
| m<18 | −4205.24 | −4187.54 | **+17.7** |

At `C ≈ 100 %` and `81 %` the two are tied — as they must be, since there is
almost nothing to complete. At `C = 32 %` and `9.5 %` selection is decisively
preferred. The estimators agree when the choice cannot matter and diverge, in
selection's favour, when it does.

**`H0` is flat under selection and wanders under per_pixel.**

| rung | H0, per_pixel | H0, selection |
|---|---|---|
| m<21 | 69.148 | 69.202 |
| m<20 | 69.123 | 69.150 |
| m<19 | 68.791 | 69.154 |
| m<18 | 69.652 | 69.075 |

Selection spans 0.13 across the whole ladder, centred on seed 100's own
complete-catalog draw of 69.22; per_pixel spans 0.86. This holds with the
anchors *free*, which is the harder test — analysis 3 showed the same flatness
with them pinned.

**`f_AGN` is not measured by this configuration, and that is the point.** With
both anchors free the 90 % interval is 0.39–0.41 wide out of a unit prior, at
every rung, under both estimators. The medians (selection: 0.394, 0.438, 0.460,
0.497 from `m<21` to `m<18`) sit high of truth by 0.10–0.20 with truth inside
90 % everywhere. What moves is the *correlation*:

| rung | ρ(f_AGN, log10 n0), per_pixel | selection |
|---|---|---|
| m<21 | −0.60 | −0.56 |
| m<20 | −0.67 | −0.54 |
| m<19 | −0.36 | −0.92 |
| m<18 | −0.38 | −0.94 |

Under selection the degeneracy tightens monotonically as the catalog thins —
−0.54 at 81 % completeness to −0.94 at 9.5 %. Under per_pixel it appears to
*loosen* at the faint end, which is an artifact: a chain railed against a prior
wall cannot express its own correlation.

## Interpretation

This arm answers a different question from the rest of the campaign, and it is
the question a referee will ask: *if you did not know the completion densities,
would this method find them?*

Under selection, yes for the galaxy anchor — 0.12 dex at 9.5 % completeness,
with the truth covered at 90 % — and the evidence prefers it by 17.7 in log
over the legacy estimator on the same data. Under per_pixel, no: the anchor
rails by 1.19 dex, and because it rails it also stops reporting an honest
correlation, so the archived `ρ = −0.38` at `m<18` reads as *less* degeneracy
when the truth is more.

The AGN anchor is the weaker half. Selection puts it at −4.72 to −4.85 against
a truth of −5.0, high at every rung, while per_pixel is closer at the faint end
(−4.85 at `m<18`). With 1514 AGN in the complete catalog against 151 million
galaxies, the AGN side simply has less to identify itself with, and the
`ρ(f_AGN, log10 n0_c2) = +0.90` at `m<18` says what little there is comes
tangled with `f_AGN`.

**The honest statement about `f_AGN` is a conditional one.** Free anchors cost
a factor ~5 on `σ(f_AGN)` versus the pinned-anchor analyses (90 % half-width
0.39–0.41 here against 0.075–0.085 in analysis 3). Combined with analysis 4's
measured sensitivity of 0.43–0.47 per dex, the two numbers say the same thing
from two directions: `f_AGN` is only as well determined as the AGN density
prior, and with a flat prior over two dex it is not determined at all.

**Provenance.** The archived series ran on darksirens `2b86a2d` and this one on
`0c5b3db`, so this ladder is not strictly a one-variable comparison. Two things
bound the drift. The SHA-controlled three-arm re-check in
`experiments/experiment_dsmaster_4d_recheck` reproduces this arm's `m<18`
contrast on one code base (per_pixel `f = 0.384`, `log10 n0 = −1.806`;
selection `f = 0.510`, `log10 n0 = −3.126`), matching the numbers here to 0.013
in `f` and 0.008 dex in the anchor. And that re-check's per_pixel `lnZ`
(−4205.15, SHA `e8d5035`) agrees with the archived per_pixel `lnZ` here
(−4205.24, SHA `2b86a2d`) to 0.09 — across two different SHAs. The estimator is
doing the work, not the drift.

## Recommendation

**Main text.** The anchor-recovery table and the Δ lnZ column together are the
strongest single argument in the campaign for the selection estimator: it
recovers a parameter the legacy estimator rails on, and the data say so by 17.7
in log evidence, on the rungs where the choice matters and not on the rungs
where it cannot.

**Main text, as the scope limit on `f_AGN`.** Quote the factor-5 width penalty
for free anchors beside analysis 4's sensitivity slope. Any `f_AGN` measurement
this method reports is a statement about the AGN density prior first.

**Do not quote the medians as a bias.** At one seed with a 0.4-wide interval,
the +0.10 to +0.20 offsets are not resolved and should not appear as numbers
with a sign attached.

**Next.** Seeds, not rungs. The two `m<21`/`m<20` arms are tied in evidence and
cost 20 GPU-h between them; the informative repeat is `m<18` on seeds 101 and
102, which is where both the anchor recovery and the evidence separation live.
