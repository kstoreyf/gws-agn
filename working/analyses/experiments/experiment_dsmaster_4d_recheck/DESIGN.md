# DESIGN — experiment_dsmaster_4d_recheck

**Question: what does the 4-parameter fit say under the current completeness
estimator?**

> **Scope revised mid-flight (2026-08-08).** This started as a pure regression
> check under the assumption that `per_pixel` was still the operative
> completeness mode. It is not: master's `aggregate` mode is the current
> estimator and `per_pixel` is retained for legacy support only. The
> experiment is therefore a **two-arm comparison**, and the headline arm is
> `aggregate`. The `per_pixel` arm is kept as the control that separates
> "the code drifted" from "the estimator changed".

## The two estimators

| | formula | behaviour |
|---|---|---|
| `per_pixel` (legacy) | `C(z\|pix) = clip(dN_obs_s(z\|pix) / dN_exp_s(z), 0, 1)` | per-pixel numerator over an **isotropic** denominator, so it really estimates `C_sel(z) * (1 + delta_obs(pix,z))` — angular clustering is absorbed into the completeness, `(1-C)` anti-tracks the true missing density, and overdense complete pixels clip at 1, silently discarding their excess |
| `aggregate` (current) | `Cbar(z) = clip(Sum_p dN_obs_s(z\|p) / (N_pix_total * dN_exp_s(z)), 0, 1)` | ONE sky-anchored radial budget broadcast to every pixel, `N_pix_total` counting empty pixels too. No per-pixel clipping — a single global clip. Angular structure moves to the mean-one `Q` field: "C says how much is missing, Q says where it goes" |

`aggregate` requires the field convention (it needs `field_dN_obs_s`, built over
the full sky); on a compact catalog it raises rather than silently biasing
`Cbar` low. We already run `--catalog_sky_weighting field`, so this is
satisfied. `selection` mode is a third option but is rejected outright for
`n_catalogs >= 2`, so it is unavailable to a K=2 mixture.

**The upstream caveat is aimed directly at this experiment.** From
`completion.py`: *"Cbar is proportional to 1/n0 through a single GLOBAL clip,
so near full completeness the likelihood can be kinked in n0 (one clip boundary
for the whole sky instead of per-pixel boundaries that engage pixel by pixel) —
flagged for the inference stage, watch sampler behaviour in n0."* This fit
samples `log10n0` freely over 3 dex and the m18 galaxy anchor is already known
to rail to ~−1.8. Sampler behaviour in `n0` is therefore a primary read of this
experiment, not a footnote.

## Original framing (retained — the regression question is still answered)

**Does the 4-parameter fit still say the same thing on current
darksirens master?**

Analysis 5 fitted the joint posterior
**(H0, log10n0 [GAL], log10n0_c2 [AGN], f_AGN)** of the K=2 mixture with both
completion-density anchors free, on darksirens `2b86a2d`. Master has since
moved 23 commits ahead, and eight of them touch completeness. This is a
cross-check, not a new result: rerun one rung, unchanged in every other
respect, and ask whether anything moved.

## What changed upstream, and why most of it should not reach us

`2b86a2d..origin/master` (`e8d5035`) is +1995/−100 lines, concentrated in
`redshift/completion.py`, `redshift/selection.py` (new),
`redshift/lognormal_completion.py` and `cli/build_lognormal_completion.py`.
Read against this configuration — `dark_sirens`, K=2, `use_LSS=False`,
field weighting, homogeneous `log10n0`/`log10n0_c2`, `per_pixel` completeness:

| upstream change | reaches this config? |
|---|---|
| `c_mode` aggregate + parametric selection modes | **no** — default is `per_pixel`, every library read is `getattr(opts, "c_mode", None) or "per_pixel"`; and `build_parameter_space` rejects `c_mode="selection"` outright for `n_catalogs >= 2` |
| Q per-z mean-one budget renormalisation (1850757) | **no** — offline Q-table builder + LSS loader only; `maybe_load_lss_completion` is never entered with `lss_completion=None` |
| `--no-budget-renorm` (c4c6996) | **no** — flag on the Q-builder CLI, not `darksirens_inference` |
| expected-side smoothing truncated at survey depth (9e64001) | **no, conditionally** — this one *is* inside the homogeneous per-pixel estimator, but it is gated on `survey.z_depth is not None`. **None of our ten survey files carries a `z_depth` attribute** (checked: attrs are `empty_pixel_fraction, mag_limit, n_hosts, nside, source_complete_catalog, tracer, z_error`), and we never pass `--survey_z_depth`. Upstream pins the `z_depth=None` path bit-identical in `tests/test_completion_depth.py`. |
| zgrid endpoint pin (4566bdc) | **no** — clamps the last node only where the jnp value exceeds the numpy one by an ulp; at our `zMax=5` the two agree exactly |
| per-galaxy apparent magnitudes in pixelated files (e831290) | **no** — purely additive to the writer; `load_survey` is unchanged, so pre-built survey files load without regeneration |
| `gwtc5_fiducial_bpl2peaks` refit to LVK GWTC-5.0 medians (f6740a5) | **no** — we run `powerlaw+peak`; the diff touches only the gwtc5 registry entry. Would have mattered a lot otherwise: `gamma` moved 0.0 → 2.54. |

So the *prediction* is that the likelihood is bit-identical for this
configuration. The point of the experiment is to check that prediction rather
than trust the reading, because a silent completeness change is exactly the
kind of thing that would invalidate analyses 3–6 without anyone noticing.

## Design — two stages, cheap gate first

**Stage A, the probe (minutes).** Rebuild the analysis-5 closure against master
and evaluate it at deterministic cells of analysis 3's stored `m18` grid —
mid-H0 at f = 0.5, and both endpoint cells f = 0 and f = 1. Analysis 5 recorded
`wiring_check_max_abs_diff = 1.8e-12` for exactly this comparison on `2b86a2d`,
i.e. float64 round-off. If master reproduces that, the likelihood has not
changed and the 4D posterior *cannot* move. If it does not, the probe reports
how far it moved for a few GPU-minutes instead of a few GPU-hours.

The wiring check is made **non-fatal** here (`--wiring_nonfatal`): in analysis 5
a mismatch was a fatal misconfiguration, but in this experiment the size of the
mismatch is the measurement.

**Stage B, the fit (~2–3 h).** The full dynesty fit at analysis-5 settings —
`nlive 1000`, `dlogz 0.1`, `maxcall 500000`, rstate 7 — on rung `m18`.

Why `m18`: it is the cheapest rung (5731 s on the js2 H100; the galaxy catalog
sets the cost) *and* the most diagnostic, because it is where the GAL anchor
rails to ~10× the true density and where analysis 6 later found the response to
completeness asymmetry steepening to +0.22/dex. If a completeness change is
going to show up anywhere, it shows up here.

## How the comparison is judged

Against `analysis_5/results/campaign_m18_dynesty_s100.json` (same rstate, same
rung, `2b86a2d`). The noise floor is analysis 5's own rstate-23 replicate:

| quantity | rstate 7 | rstate 23 | replicate Δ |
|---|---|---|---|
| H0 median | 69.6515 | 69.6421 | 0.009 |
| log10n0 median | −1.8081 | −1.8034 | 0.005 |
| log10n0_c2 median | −4.8852 | −4.8840 | 0.001 |
| f_AGN median | 0.3837 | 0.3871 | 0.003 |
| log Z | −4205.24 | −4205.21 | 0.03 |

A master-vs-`2b86a2d` shift has to clear these to mean anything. Anything at or
below them is dynesty, not darksirens.

## Driver deltas

`scripts/sample_4d.py` is analysis 5's, with four changes marked `# [exp]`: the
`A4_SCRIPTS` path gains a `.parent` (this copy is one level deeper), `opts`
states `c_mode="per_pixel"` / `selection_prior=None` explicitly instead of
relying on library defaults that could move, and the two new flags
`--wiring_nonfatal` and `--probe_only`. Priors, guard, KDE window, nuisance
defaults, sampler call and output contract are untouched.

The darksirens switch is `PYTHONPATH`, which resolves ahead of the conda env's
setuptools egg-link — verified, not assumed. `DARKSIRENS_SRC` points the
provenance guard at the same tree so the recorded SHA is the code that ran.
Master is checked out as a **git worktree** at
`/hildafs/projects/phy230014p/magana/src/darksirens-master`, leaving the owner's
`src/darksirens` checkout (parked on `feat/likelihood-2d-scan`) untouched.

## Scope

One rung, one seed, one rstate. This is a regression check on the code, not a
new physics result, and nothing here is destined for the paper.
