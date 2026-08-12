# analysis_6_relative_completeness_H0_fagn — when the two tracers are not surveyed to the same depth

Round 6 of the campaign on `working/data/seed100` (**dataset v3 + D3, float64**).
The same likelihood, the same joint `(H0, f_AGN)` grid and the same
magnitude-limited surveys as
[`analysis_3`](../analysis_3_incomplete_catalog_H0_fagn/README.md) — but the
**galaxy and AGN survey depths are varied independently**.

Analysis 3's ladder, and every arm of analyses 4 and 5, sit on the *diagonal* of a
two-dimensional plane. The mock's two tracers share a magnitude distribution, so
each rung has `C_GAL = C_AGN` by construction (within the horizon,
`surveys_meta.json`: `m20` 0.8143/0.8143, `m19` 0.3151/0.3170, `m18`
0.0954/0.0960). Real surveys never do — AGN are selected in different bands, to
different depths, than galaxies.

The reason to care is analysis 4's single off-diagonal point. Its oracle probe —
galaxies at `m<18`, AGN survey **complete**, both densities at truth — did not
remove the faint-rung `f_AGN` bias, it **tripled** it, `+0.073 → +0.197`. Analysis
5 saw the same asymmetry break the galaxy anchor at `m<18` (railed to ten times
the true density) and produce the one `H0` shift in the whole campaign. This
directory asks whether the bias is a function of the **ratio** `C_AGN / C_GAL`
rather than of either depth alone — and therefore whether a real analysis can
correct for it.

Status: **COMPLETE** — 12/12 cells, 2026-08-08 (8 new grids in 6.57 GPU-h on
RITA-GPU array 1118657; both workers finished inside 3.5 h). Owner report in
`REPORT.md`. Internal record only — no paper wiring — per the standing approval
gates.

<!-- RESULTS_BANNER -->
> **Relative completeness sets both the size and the sign of the AGN-fraction bias.** Holding the galaxy survey at $m<20$ and varying only the AGN depth, $f_{\rm AGN}$ runs $-0.037$, $+0.031$, $+0.052$, $+0.051$ as $C_{\rm AGN}/C_{\rm GAL}$ goes $0.12 \to 0.39 \to 1.00 \to 1.23$. An AGN catalog *shallower* than the galaxy catalog makes the fraction under-estimated, deeper over-estimated, and matched depths give the smallest bias — the diagonal that analyses 3-5 happened to sit on is the favourable ridge of this surface, not a generic point.
>
> **It is the ratio, not either depth.** Across all twelve cells $\log_{10}(C_{\rm AGN}/C_{\rm GAL})$ explains the bias with $R^2 = 0.89$, against $0.63$ for galaxy completeness alone and $0.30$ for AGN completeness alone.
>
> **But the surface is genuinely two-dimensional.** The single global line leaves an rms of 0.024 in $f_{\rm AGN}$, above this realisation's own binomial scatter (0.0145), because each galaxy depth has its own response: the low-ratio slope is $+0.13$ per dex at $m<20$ and $m<19$ but steepens to $+0.22$ at $m<18$, and the bias saturates once the AGN catalog is a few times more complete than the galaxy catalog — at $+0.091$ for $m<19$ and $+0.194$ for $m<18$. A one-parameter correction in the ratio is good to about $0.02$ in $f_{\rm AGN}$ and no better.
>
> **$H_0$ is completely indifferent:** $R^2 = 0.0002$ against relative completeness, over a surface where $f_{\rm AGN}$ moves by 0.24. The offsets ($+0.58$ to $+1.98$ on this seed) track the galaxy depth, not the ratio. Relative completeness is an $f_{\rm AGN}$ systematic and nothing else.
>
> **The detection always survives:** the significance of a non-zero AGN component runs 4.6σ to 7.7σ over every cell. As in analysis 4, what is at risk is the *value* of the fraction, never its existence.
<!-- /RESULTS_BANNER -->

## Scope of record (owner-approved 2026-08-08)

Seed 100 only, targeted injections. GAL depth × AGN depth over
`{complete, m20, m19, m18}`, restricted to GAL ∈ `{m20, m19, m18}` — `m21` is
100 % complete inside the horizon, so a GAL `m21` row would duplicate `complete`.

|  | AGN complete | AGN m20 | AGN m19 | AGN m18 |
|---|---|---|---|---|
| **GAL m20** | new | *have (a3)* | new | new |
| **GAL m19** | new | new | *have (a3)* | new |
| **GAL m18** | *have (a4 oracle)* | new | new | *have (a3)* |

**8 new grids.** The four cells already on disk are **referenced, never rerun**
(`../analysis_3_incomplete_catalog_H0_fagn/results/joint_{m20,m19,m18}_s100.h5`
and `../analysis_4_density_anchoring_H0_fagn/results/joint_m18_oracle_s100.h5`),
so the surface and its diagonal share one estimator by construction — all copies
of `scan_h0f.py` across analyses 3, 4 and 6 are byte-identical
(md5 `02acecc6f73d5ae0bd31985e2b7ac1c3`).

**Both completion densities stay at the mock's truth** (`log10n0 = -3`,
`log10n0_c2 = -5`) in every cell. The anchoring axis was measured in analysis 4
and the cost of not knowing it in analysis 5; mixing either in would confound the
completeness ratio with them.

No mock regeneration, no new events, no new injections — every survey file
already exists.

## Configuration

Byte-identical to analysis 3 (`scripts/env.sh` there) except the two survey
paths: darksirens @ `2b86a2d`, K = 2 mixture, `catalog_sky_weighting = field`,
survey order `[GAL, AGN]` (`fcat_2` **is** `f_AGN`), `delta = delta_c2 = 0`,
`sigma_kde = sigma_kde_c2 = 0`, population fixed at the mock fiducial, `Om0`
pinned, guard `hard` with `max_likelihood_variance 1e6`, `W = 4096`
(`n_sigma = 8`), grid `H0 ∈ [50,100] × 201` × `f ∈ [0,1] × 41`, all 1000
events × 2000 PE samples, `injections_targeted.h5`.

## Running

```bash
./scripts/make_queue.sh                          # 8 tasks, most expensive first
sbatch --array=0-1 scripts/submit_rita.sbatch    # one worker per A100-80
./scripts/status.sh                              # progress at any point
```

Workers claim tasks by atomic `mkdir`; a killed worker's queue is re-runnable
after `rm -rf queue/claim_*` (existing result files are skipped). The sbatch is
sized against the real node — rita is 2× A100-80, 256 CPUs, 250 G, partition
limit 7 days — so both array elements start immediately (2×110 G, 2×32 CPUs) and
each holds 24 h against a worst-case task priced at ~7 h, which means a worker
never exits leaving work unclaimed.

Measured on the first two cells: **1.8 eval/s**, so ~1.3 h for a GAL `m20` grid
and less below it; ~7 GPU-h for the eight.

## Figures

| file | what it shows |
| --- | --- |
| `fig_surface_f` | the f_AGN offset over the full GAL x AGN depth matrix |
| `fig_surface_h0` | the same for H0 — flat, by contrast |
| `fig_ratio_collapse` | the offset against relative completeness (per-row lines show the saturation) vs against galaxy completeness alone |

Regenerate with `python scripts/aggregate_and_figures.py` (deterministic, CPU
only, run in the `jax` conda env).

## Scope limitations, stated plainly

- **One seed.** Cell-to-cell differences share events, injections and estimator,
  so the *shape* of the surface is trustworthy; the absolute offsets carry seed
  100's own realisation offset.
- **No ratio below 1 at `m<18`.** `m18` is the shallowest survey level built, so
  the galaxy-`m18` row cannot be given a shallower AGN catalog. The sign change
  is demonstrated at `m<20` and `m<19` only.
- **`m<20` never reaches saturation** (its largest available ratio is 1.23,
  because a complete AGN catalog is only 1/0.814 times its galaxy completeness),
  so the saturation level is measured at `m<19` and `m<18` only.

## A free wiring check the queue gives us

At `f_AGN = 0` the AGN catalog is switched off entirely, so every cell sharing a
GAL depth must return the *same* log-likelihood there regardless of AGN depth.
The first two cells (`GAL m20 × AGN complete` and `GAL m20 × AGN m19`) both
reported `logL = -4245.7682` at the base coordinate — the `f = 0` endpoint
identity that validates K = 2 wiring, reproduced across two independent survey
pairs.
