# Proposal — what to run next (awaiting owner gate)

Written 2026-08-08, after analyses 3, 4 and 5 closed. Nothing here is launched.

## Where the campaign actually stands

| axis | status |
|---|---|
| completeness, both tracers together | **measured** (analysis 3): σ(H0) ×1.14, σ(f) ×1.22 to 10 % completeness |
| AGN density anchoring error | **measured** (analysis 4): `f_AGN` responds linearly, `H0` does not |
| both anchors unknown | **measured** (analysis 5): σ(f) ×2–3.5, σ(H0) ×1.0 |
| `H0` accuracy | **not resolvable at 5 seeds** — analysis 2 closure `+0.41 ± 0.55` |
| **relative completeness between the two tracers** | **never scanned** |

The last row is the gap, and analysis 4's oracle probe is the reason to care. Its
one off-diagonal point — galaxies at `m<18` (9.5 % complete inside the horizon),
AGN survey complete — did not remove the faint-rung `f_AGN` bias, it **tripled**
it, from `+0.073` to `+0.197`. Analysis 5 shows the same asymmetry breaking the
galaxy anchor at `m<18` (railed to 10× the true density) and driving the only
`H0` shift in the whole campaign (+1.33).

Crucially, every rung of analysis 3's ladder has `C_GAL = C_AGN` **by
construction** — the two tracers share a magnitude distribution, so the survey
metadata gives 0.0954 vs 0.0960 at `m<18`, 0.3151 vs 0.3170 at `m<19`. The whole
existing campaign lives on the diagonal of a two-dimensional plane. Real surveys
never do: AGN are selected in different bands to different depths than galaxies.

## Recommended: run these two, in this order

### 6a — the free one, today: how tight an AGN density prior buys back σ(f_AGN)?

Analysis 5's actionable claim is that the `f_AGN` inflation is a *degeneracy*
(corr +0.68 → +0.89), so an external constraint on `n_0^{AGN}` — which a real
analysis has, from an AGN luminosity function — should recover most of the factor
of three. Quantify it by **importance-reweighting the chains already on disk**:
multiply the stored flat-prior weights by a Gaussian on `log10n0_c2` of width
σ ∈ {0.05, 0.1, 0.2, 0.3} dex centred on truth, and on a deliberately *offset*
centre (±0.15 dex) to show what a wrong external prior costs.

- Cost: **zero GPU, minutes of CPU.** `campaign_*.h5` stores `raw/logl`,
  `raw/logwt`, `raw/samples`.
- Deliverable: one curve, σ(f_AGN) vs prior width, per rung; one figure; a
  paragraph. Validity is bounded by effective sample size, which is worth
  reporting internally and refusing below ~200.
- Why it is worth doing first: it turns analysis 5's main result from a warning
  into a recommendation, at no cost.

### 6b — analysis 6: the relative-completeness surface

Scan the plane instead of the diagonal: **GAL depth × AGN depth**, `(H0, f_AGN)`
grid at each cell, anchors at truth, everything else the analysis-3/4
configuration of record.

|  | AGN complete | AGN m20 | AGN m19 | AGN m18 |
|---|---|---|---|---|
| **GAL m20** | new | *have (a3)* | new | new |
| **GAL m19** | new | new | *have (a3)* | new |
| **GAL m18** | *have (a4 oracle)* | new | new | *have (a3)* |

**8 new grids.** Every survey file already exists
(`working/data/seed100/surveys/survey_{gal,agn}_{complete,m20,m19,m18}_ns32.h5`)
— no mock regeneration, no new events, no new injections. Cost is set entirely by
the galaxy catalog size: the a4 oracle grid (GAL m18 + AGN complete) ran in
**1.18 h** and a GAL m21 grid in 2.35 h, so **~12–16 GPU-h total**, one RITA
array, comparable to analysis 4 and cheaper than analysis 5.

What it answers: is the `f_AGN` bias a function of the *ratio* `C_AGN / C_GAL`
rather than of either depth alone? If the 8 cells collapse onto one curve in that
ratio, the campaign gets a quotable law and a correction a real analysis can
apply. If they do not, the effect is depth-specific and the paper should say so
and stop.

Secondary reads from the same grids, free: whether the galaxy anchor's `m<18`
pathology tracks `C_GAL` alone or the ratio (re-run the 4D fit on the two most
extreme cells only — 2 extra sampler runs, ~4 GPU-h, decide after seeing 6b).

## Considered and not recommended now

- **More seeds.** The only way to make an `H0` accuracy statement below
  ~0.5 km s⁻¹ Mpc⁻¹, since 5 realisations of a 1.05-half-width statistic cannot.
  But it is the most expensive thing available (analysis 5 is 35 GPU-h *per
  seed*), and the campaign's `H0` claims are currently differential — "freeing
  the anchors does not move it" — which do not need it. Revisit only if the paper
  wants to claim `H0` is unbiased rather than that its systematics are bounded.
- **Chasing the residual `H0` channel further.** `analysis_1/CLOSURE.md` §16
  closed both matched-host controls onto truth under the v3 measurement family.
  There is no open per-event residual to chase; the earlier "4.4σ" statement is
  pre-v3 and superseded.
- **Freeing the evolution indices `delta`, `delta_c2`.** Defensible eventually,
  but the mock has `delta = 0` exactly and measured consistent with zero, so it
  would test robustness to a parameter that is not wrong. Lower value than the
  relative-completeness plane.

## The gate

6a is free and reversible — reweighting chains, no compute, no paper edits. 6b is
~14 GPU-h and follows the analysis-4 pattern exactly. Both need the owner's word
per the standing approval gates; 6b in particular is a new analysis directory.
