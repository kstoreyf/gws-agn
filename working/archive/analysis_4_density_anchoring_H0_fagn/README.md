# analysis_4_density_anchoring_H0_fagn — what mis-anchoring the AGN density costs down the ladder

Round 4 of the campaign on `working/data/seed100` (**dataset v3 + D3, float64**).
The same likelihood, the same joint `(H0, f_AGN)` grid, and the same
magnitude-limited surveys as
[`analysis_3`](../analysis_3_incomplete_catalog_H0_fagn/README.md) — but the
completion's AGN density anchor `log10n0_c2` is now set *off* the mock's truth
by a known factor, one scalar per arm.

Analysis 3 ran the most favourable case: the missing-host budget anchored at the
mock's exact densities, isolating completeness from anchoring error. The
prototype (`../experiments/experiment_completeness_free`) showed anchoring is
the dominant axis on a 200-event deep mock — a factor-2 density error already
halves the detection significance of `f_AGN` once the survey is incomplete, and
a free density collapses it entirely. This directory measures that axis at
production scale, on the v3 catalogs and the 1000-event mixture, with the
estimator of record.

<!-- RESULTS_BANNER -->
> Mis-anchoring the completion's AGN density propagates almost entirely into $f_{\rm AGN}$ itself: at m < 21, halving the assumed density moves the recovered fraction to 0.191 and doubling it to 0.502, against 0.342 at the true anchor and a realised 0.295 — a -2.4σ and +2.6σ shift in units of the exact arm's own 68 % half-width.
>
> The error moves with the median, so the *detection* of an AGN component survives: the significance runs 4.8σ → 5.5σ → 5.6σ across the same factor-4 range in assumed density. $H_0$ is the resilient parameter, shifting -0.26 to +0.53 of its own half-width over the same range.
>
> The oracle probe settles the faintest rung: handing the model a complete AGN survey while the galaxies stay at m < 18 gives $f_{\rm AGN} = 0.492 \pm 0.064$, offset +0.197 — -169 % of the m < 18 bias removed.
<!-- /RESULTS_BANNER -->

## Scope of record (owner-approved 2026-08-05)

- **Seed 100 only**, targeted injections, rungs `m21 m20 m19 m18`.
- **Six anchoring arms** per rung: `log10n0_c2 = -5 + log10(factor)`,
  `factor ∈ {0.5, 0.7, 0.9, 1.1, 1.3, 2.0}` (tags `a05 a07 a09 a11 a13 a20`).
  The GAL density stays at truth (`log10n0 = -3`) everywhere.
- **The exact arm is analysis 3's own seed-100 grids** — referenced, never
  rerun, so arms and reference share one estimator by construction.
- **One oracle probe**: GAL at `m < 18`, AGN survey *complete*, both densities
  at truth (tag `oracle`). If the `+0.084 ± 0.019` `f_AGN` bias at the faintest
  rung is manufactured by the sparse AGN completion (5 hosts per occupied
  pixel, 52.8 % of pixels empty), handing the model every AGN host while the
  galaxies stay 10 %-complete should remove most of it.
- 25 grids total, ~28 GPU-h at analysis 3's measured per-rung s/eval.

## Configuration

Byte-identical to analysis 3 (`scripts/env.sh` there) except `log10n0_c2`
(the arms) or the AGN survey level (the oracle probe): darksirens @ `2b86a2d`,
K = 2 mixture, `catalog_sky_weighting = field`, survey order `[GAL, AGN]`
(`fcat_2` **is** `f_AGN`), `delta = delta_c2 = 0`,
`sigma_kde = sigma_kde_c2 = 0`, population fixed at the mock fiducial, `Om0`
pinned, guard `hard` with `max_likelihood_variance 1e6`, `W = 4096`
(`n_sigma = 8`), grid `H0 ∈ [50,100] × 201` × `f ∈ [0,1] × 41`, all 1000
events × 2000 PE samples, `injections_targeted.h5`.

## Running

```bash
./scripts/make_arm_queue.sh                      # 25 tasks, most expensive first
sbatch --array=0-5 scripts/submit_arms_rita.sbatch   # workers on RITA-GPU (2x A100-80)
./scripts/status.sh                              # progress at any point
```

Workers share the queue by atomic `mkdir` claims exactly as in analysis 3; a
killed worker's queue is re-runnable after `rm -rf queue/claim_*` (existing
result files are skipped).

## Status

**COMPLETE** — 25/25 grids on disk, aggregated (`results/arms_summary.json`),
figures rendered (`figs/`, deterministic `scripts/make_figures.py`), results
tables below, owner report in `REPORT.md`. Rerun everything downstream of the
grids with `./scripts/finalize.sh`. Internal record only — no paper wiring —
per the standing approval gates.

## Results

<!-- ARM_TABLES -->
All numbers: seed 100, targeted-injection lane, truth $H_0 = 67.74$, realised host fraction 0.295 (295/1000), planted 0.3.  The exact arm is analysis_3's own grid, referenced not rerun.  Significance is median / 68 % half-width; the last column is the shift from the exact arm in units of that arm's half-width.

**m < 21**

| assumed / true $n_{0,\rm AGN}$ | $\log_{10} n_{0,c2}$ | $H_0$ | offset | $f_{\rm AGN}$ | offset | $f$ significance | $\Delta f$ vs exact |
|---:|---:|:---|---:|:---|---:|---:|---:|
| 0.5 | -5.301 | $69.37^{+0.93}_{-0.92}$ | +1.63 | $0.191^{+0.042}_{-0.038}$ | -0.104 | 4.8σ | -2.41σ |
| 0.7 | -5.155 | $69.44^{+0.93}_{-0.92}$ | +1.70 | $0.256^{+0.051}_{-0.049}$ | -0.039 | 5.1σ | -1.37σ |
| 0.9 | -5.046 | $69.53^{+0.93}_{-0.92}$ | +1.79 | $0.315^{+0.059}_{-0.057}$ | +0.020 | 5.4σ | -0.43σ |
| **1.0** (exact) | -5.000 | $69.61^{+0.93}_{-0.93}$ | +1.87 | $0.342^{+0.064}_{-0.061}$ | +0.047 | 5.5σ | — |
| 1.1 | -4.959 | $69.66^{+0.94}_{-0.93}$ | +1.92 | $0.366^{+0.068}_{-0.064}$ | +0.071 | 5.6σ | +0.40σ |
| 1.3 | -4.886 | $69.79^{+0.94}_{-0.95}$ | +2.05 | $0.410^{+0.074}_{-0.072}$ | +0.115 | 5.6σ | +1.09σ |
| 2 | -4.699 | $70.10^{+1.00}_{-1.00}$ | +2.36 | $0.502^{+0.089}_{-0.089}$ | +0.207 | 5.6σ | +2.56σ |

**m < 20**

| assumed / true $n_{0,\rm AGN}$ | $\log_{10} n_{0,c2}$ | $H_0$ | offset | $f_{\rm AGN}$ | offset | $f$ significance | $\Delta f$ vs exact |
|---:|---:|:---|---:|:---|---:|---:|---:|
| 0.5 | -5.301 | $69.25^{+0.94}_{-0.91}$ | +1.51 | $0.192^{+0.042}_{-0.038}$ | -0.103 | 4.8σ | -2.46σ |
| 0.7 | -5.155 | $69.31^{+0.93}_{-0.92}$ | +1.57 | $0.259^{+0.052}_{-0.049}$ | -0.036 | 5.1σ | -1.40σ |
| 0.9 | -5.046 | $69.40^{+0.94}_{-0.91}$ | +1.66 | $0.320^{+0.060}_{-0.058}$ | +0.025 | 5.4σ | -0.44σ |
| **1.0** (exact) | -5.000 | $69.47^{+0.94}_{-0.92}$ | +1.73 | $0.347^{+0.065}_{-0.062}$ | +0.052 | 5.5σ | — |
| 1.1 | -4.959 | $69.53^{+0.94}_{-0.93}$ | +1.79 | $0.373^{+0.068}_{-0.066}$ | +0.078 | 5.6σ | +0.41σ |
| 1.3 | -4.886 | $69.66^{+0.96}_{-0.93}$ | +1.92 | $0.418^{+0.074}_{-0.072}$ | +0.123 | 5.7σ | +1.12σ |
| 2 | -4.699 | $70.01^{+1.01}_{-1.00}$ | +2.27 | $0.511^{+0.089}_{-0.089}$ | +0.216 | 5.7σ | +2.59σ |

**m < 19**

| assumed / true $n_{0,\rm AGN}$ | $\log_{10} n_{0,c2}$ | $H_0$ | offset | $f_{\rm AGN}$ | offset | $f$ significance | $\Delta f$ vs exact |
|---:|---:|:---|---:|:---|---:|---:|---:|
| 0.5 | -5.301 | $68.63^{+0.96}_{-0.96}$ | +0.89 | $0.185^{+0.039}_{-0.038}$ | -0.110 | 4.8σ | -2.70σ |
| 0.7 | -5.155 | $68.75^{+0.95}_{-0.95}$ | +1.01 | $0.258^{+0.053}_{-0.050}$ | -0.037 | 5.0σ | -1.57σ |
| 0.9 | -5.046 | $68.89^{+0.96}_{-0.95}$ | +1.15 | $0.329^{+0.063}_{-0.060}$ | +0.034 | 5.4σ | -0.50σ |
| **1.0** (exact) | -5.000 | $68.97^{+0.96}_{-0.95}$ | +1.23 | $0.361^{+0.067}_{-0.064}$ | +0.066 | 5.5σ | — |
| 1.1 | -4.959 | $69.06^{+0.95}_{-0.95}$ | +1.32 | $0.392^{+0.071}_{-0.068}$ | +0.097 | 5.6σ | +0.46σ |
| 1.3 | -4.886 | $69.25^{+0.97}_{-0.96}$ | +1.51 | $0.444^{+0.078}_{-0.076}$ | +0.149 | 5.8σ | +1.27σ |
| 2 | -4.699 | $69.72^{+1.03}_{-1.03}$ | +1.98 | $0.562^{+0.093}_{-0.093}$ | +0.267 | 6.0σ | +3.07σ |

**m < 18**

| assumed / true $n_{0,\rm AGN}$ | $\log_{10} n_{0,c2}$ | $H_0$ | offset | $f_{\rm AGN}$ | offset | $f$ significance | $\Delta f$ vs exact |
|---:|---:|:---|---:|:---|---:|---:|---:|
| 0.5 | -5.301 | $67.73^{+1.32}_{-1.44}$ | -0.01 | $0.141^{+0.039}_{-0.036}$ | -0.154 | 3.8σ | -3.06σ |
| 0.7 | -5.155 | $67.91^{+1.25}_{-1.37}$ | +0.17 | $0.227^{+0.055}_{-0.051}$ | -0.068 | 4.3σ | -1.90σ |
| 0.9 | -5.046 | $68.17^{+1.18}_{-1.30}$ | +0.43 | $0.321^{+0.070}_{-0.066}$ | +0.026 | 4.7σ | -0.63σ |
| **1.0** (exact) | -5.000 | $68.32^{+1.14}_{-1.26}$ | +0.58 | $0.368^{+0.076}_{-0.072}$ | +0.073 | 5.0σ | — |
| 1.1 | -4.959 | $68.48^{+1.12}_{-1.23}$ | +0.74 | $0.414^{+0.081}_{-0.079}$ | +0.119 | 5.2σ | +0.61σ |
| 1.3 | -4.886 | $68.74^{+1.09}_{-1.20}$ | +1.00 | $0.495^{+0.089}_{-0.087}$ | +0.200 | 5.6σ | +1.71σ |
| 2 | -4.699 | $69.31^{+1.09}_{-1.14}$ | +1.57 | $0.699^{+0.098}_{-0.101}$ | +0.404 | 7.0σ | +4.45σ |

**Oracle probe** — galaxies at m < 18, AGN survey complete, both densities at truth

| | $H_0$ | offset | $f_{\rm AGN}$ | offset |
|:--|:--|---:|:--|---:|
| oracle | $69.51^{+0.96}_{-0.98}$ | +1.77 | $0.492^{+0.065}_{-0.064}$ | +0.197 |
<!-- /ARM_TABLES -->
