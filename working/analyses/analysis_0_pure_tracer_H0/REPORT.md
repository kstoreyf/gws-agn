# analysis 0 -- pure-tracer H0 constraining power, and the bias check

ten independent single-tracer event draws for analysis 0: for each of the five v3 catalog realisations, one 1000-event set with every host a galaxy (f_agn = 0) and one with every host an AGN (f_agn = 1), drawn on the SAME signed-off catalogs and surveys with event-noise streams independent of each other and of the record's mixture events.

The question analysis 1 could not answer.  Its two single-tracer numbers came from splitting ONE 1000-event mixture draw on host type, so the galaxy arm carried 705 events and the AGN arm 295, and the event noise was shared with the analysis of record.  Nothing in that pair compares the two tracers' constraining power, because they are not the same measurement.  Here each tracer gets its own independent draw of N = 1000 detected events on the same catalogs and surveys, so the widths are directly comparable and the offsets are a fresh look at the bias.

**What it found.**  At a matched N = 1000 the AGN catalog constrains H0 about 5.1 times more tightly than the galaxy catalog -- a mean 68% half-width of 0.45 against 2.29 km/s/Mpc.  Both tracers recover truth: the mean offset over five realisations is -0.001 +- 0.227 km/s/Mpc for AGN (t(4) = -0.01) and +0.594 +- 0.864 for GAL (t(4) = +0.69).  On independent event draws the H0 bias does not reappear.

## 1. The event sets

`working/data/generate_dataset.py` gained two options, both defaulting to the behaviour of record:

* `--f_agn` -- the planted AGN-hosted fraction used by the events stage.  Unset, it is the module constant `F_AGN`.
* `--seed_events` -- an explicit events-stage sub-seed.  Unset, it is the record's derivation `SEED*1000+3`.

Both flow into the events RNG and into the recorded metadata (`planted_f_agn`, `seed_events` on the file and in `metadata_json`).  `sub_seeds()` spends offsets 1-7 on the record (1 glass_field), (2 magnitudes), (3 events), (4 injections_targeted), (5 injections_popuni), (6 validation), (7 photoz); offsets 8 and 9 are unused by the generator and carry the two draws here, so they are independent of every recorded stream and of each other.

**Bit-identity gate.**  Seed 100's events stage was rerun with no new flags into a scratch output root and compared against the record file dataset by dataset: **50 of 50 datasets byte-identical (SHA-256), 0 failures** -- `PASS`.  The only differences are metadata: the generation timestamp, the new provenance keys the extension records, and the new top-level `seed_events` attribute.  `results/gate_events_bitidentity.json` carries the per-dataset digests.

Nothing in the signed-off dataset was modified: `--events_suffix` writes `events_pure{gal,agn}.h5` beside `events.h5` and suppresses the `META.json` merge, `--overwrite` is never passed, and the catalogs, surveys and both injection lanes are reused exactly as they are on disk.

| seed | tracer | seed_events | N | nsamp | hosts GAL | hosts AGN | unique hosts | max mult | min SNR | median z | max z | checks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | gal | 100008 | 1000 | 2000 | 1000 | 0 | 1000 | 1 | 8.003 | 0.132 | 0.339 | PASS |
| 100 | agn | 100009 | 1000 | 2000 | 0 | 1000 | 973 | 3 | 8.002 | 0.135 | 0.528 | PASS |
| 101 | gal | 101008 | 1000 | 2000 | 1000 | 0 | 1000 | 1 | 8.000 | 0.129 | 0.348 | PASS |
| 101 | agn | 101009 | 1000 | 2000 | 0 | 1000 | 986 | 2 | 8.000 | 0.131 | 0.391 | PASS |
| 102 | gal | 102008 | 1000 | 2000 | 1000 | 0 | 999 | 2 | 8.001 | 0.130 | 0.331 | PASS |
| 102 | agn | 102009 | 1000 | 2000 | 0 | 1000 | 983 | 2 | 8.001 | 0.132 | 0.299 | PASS |
| 103 | gal | 103008 | 1000 | 2000 | 1000 | 0 | 999 | 2 | 8.004 | 0.134 | 0.380 | PASS |
| 103 | agn | 103009 | 1000 | 2000 | 0 | 1000 | 970 | 3 | 8.001 | 0.130 | 0.287 | PASS |
| 105 | gal | 105008 | 1000 | 2000 | 1000 | 0 | 1000 | 1 | 8.001 | 0.132 | 0.399 | PASS |
| 105 | agn | 105009 | 1000 | 2000 | 0 | 1000 | 980 | 3 | 8.004 | 0.135 | 0.319 | PASS |

All ten sets pass every check (`scripts/check_pure_tracer_events.py`, overall `PASS`): the declared count and sample depth, every host of the declared type, every recorded SNR above the threshold of 8, the requested sub-seed and planted fraction on the file, and ten distinct streams.

## 2. The scans

Twenty K=1 `dark_sirens` H0 scans -- five realisations x two tracers x two injection lanes -- with analysis 1's configuration copied verbatim; only the events file changes.

| setting | value |
|---|---|
| estimator | dark_sirens at log10n0 = -24 (complete-catalog limit) |
| sky weighting | field |
| H0 grid | [50, 100] x 201 |
| truth H0 | 67.74 |
| Om0 | 0.3075 |
| population + nuisances | fixed at truth |
| selection guard | hard N_eff wall, max_likelihood_variance = 1e6 (variance criterion inert) |
| catalog KDE window | 4096 on the GAL survey; module default on the AGN survey |
| injection lane of record | targeted |

`scripts/scan_h0f.py` is analysis 1's driver copied byte for byte; `scripts/run_scans.sh` and `scripts/submit_scans.sbatch` are its `run_scans.sh` / `submit_v3_controls.sbatch` with the event paths and tags changed.

## 3. Constraining power at equal N

68% half-width of the H0 posterior, pure-GAL vs pure-AGN, at the same N = 1000 detected events on the same catalog realisation; lane of record = targeted

| seed | N (GAL) | N (AGN) | 68% half-width GAL | 68% half-width AGN | AGN / GAL |
|---|---|---|---|---|---|
| 100 | 1000 | 1000 | 1.96 | 0.50 | 0.26 |
| 101 | 1000 | 1000 | 1.72 | 0.38 | 0.22 |
| 102 | 1000 | 1000 | 1.56 | 0.49 | 0.31 |
| 103 | 1000 | 1000 | 1.95 | 0.44 | 0.23 |
| 105 | 1000 | 1000 | 4.25 | 0.45 | 0.11 |

Mean 68% half-width: **2.29 km/s/Mpc (GAL)** against **0.45 km/s/Mpc (AGN)**, a ratio of means of **0.20**; the per-seed ratios average 0.22 +- 0.03 over 5 realisations.

## 4. Closure on truth

### pure-GAL

| seed | N | median | 68% interval | 90% interval | offset | truth in 68% | truth in 90% | cells rejected |
|---|---|---|---|---|---|---|---|---|
| 100 | 1000 | 67.13 | [65.28, 69.20] | [64.29, 70.84] | -0.61 | yes | yes | 0 |
| 101 | 1000 | 66.40 | [64.79, 68.22] | [63.79, 69.60] | -1.34 | yes | yes | 0 |
| 102 | 1000 | 69.83 | [68.44, 71.57] | [67.67, 72.85] | +2.09 | no | yes | 0 |
| 103 | 1000 | 70.91 | [69.06, 72.97] | [67.80, 74.46] | +3.17 | no | no | 0 |
| 105 | 1000 | 67.40 | [62.51, 71.00] | [60.92, 72.96] | -0.34 | yes | yes | 0 |

Mean offset **+0.594 +- 0.864 km/s/Mpc** over 5 realisations (sd 1.932), t(4) = +0.69, p = 0.5295.  Truth falls inside the 68% interval in 3 of 5 realisations and inside the 90% interval in 4 of 5.  The scatter of the five medians is 0.84 times the mean quoted 68% half-width.

### pure-AGN

| seed | N | median | 68% interval | 90% interval | offset | truth in 68% | truth in 90% | cells rejected |
|---|---|---|---|---|---|---|---|---|
| 100 | 1000 | 68.36 | [67.85, 68.86] | [67.52, 69.18] | +0.62 | no | yes | 0 |
| 101 | 1000 | 67.42 | [67.05, 67.81] | [66.81, 68.06] | -0.32 | yes | yes | 0 |
| 102 | 1000 | 67.45 | [66.97, 67.95] | [66.64, 68.28] | -0.29 | yes | yes | 0 |
| 103 | 1000 | 67.25 | [66.82, 67.71] | [66.56, 68.02] | -0.49 | no | yes | 0 |
| 105 | 1000 | 68.21 | [67.76, 68.66] | [67.43, 68.94] | +0.47 | no | yes | 0 |

Mean offset **-0.001 +- 0.227 km/s/Mpc** over 5 realisations (sd 0.507), t(4) = -0.01, p = 0.9955.  Truth falls inside the 68% interval in 2 of 5 realisations and inside the 90% interval in 5 of 5.  The scatter of the five medians is 1.12 times the mean quoted 68% half-width.

## 5. Injection-lane cross-check

The two lanes are the same detection rule under different proposals, so a difference large against the 68% half-width would mean the selection integral is setting digits of the answer.

| tracer | seed | targeted | popuni | difference | as % of one half-width |
|---|---|---|---|---|---|
| gal | 100 | 67.132 | 67.159 | +0.027 | 1.4% |
| gal | 101 | 66.400 | 67.084 | +0.684 | 39.9% |
| gal | 102 | 69.830 | 69.731 | -0.099 | -6.3% |
| gal | 103 | 70.908 | 69.953 | -0.955 | -48.9% |
| gal | 105 | 67.400 | 68.905 | +1.505 | 35.4% |
| agn | 100 | 68.358 | 68.089 | -0.268 | -53.3% |
| agn | 101 | 67.424 | 67.446 | +0.022 | 5.8% |
| agn | 102 | 67.445 | 67.668 | +0.223 | 45.4% |
| agn | 103 | 67.254 | 66.880 | -0.375 | -84.5% |
| agn | 105 | 68.212 | 68.252 | +0.040 | 8.9% |

GAL: largest lane shift 48.9% of one 68% half-width.

AGN: largest lane shift 84.5% of one 68% half-width.

## 6. Guard and shape (internal)

Selection-validity guard: hard N_eff wall at 5 * N_obs with the total-variance criterion made inert (max_likelihood_variance = 1e6).  Across all 20 scans **every cell was accepted** (`all_cells_accepted = True`), the smallest per-cell N_eff sat 6.8x above the wall, and the largest posterior density reached at a grid edge was 2.2e-10 of the peak, so no posterior is censored by the scanned range.

Genuinely multimodal posteriors (a second mode above 1% of the peak) -- their 68% interval spans the gap between the modes, which is why the width is large:

| scan | modes (relative height) |
|---|---|
| `h0_puregal_targeted_s105` | 62.00 (0.70); 69.75 (1.00) |

## Figures

`python scripts/make_figures.py` renders all five from `results/` (PDF + PNG, deterministic -- a rerun on unchanged results reproduces both files byte for byte); `python scripts/make_figures.py <name>` renders one.

| figure | what it shows |
|---|---|
| `figs/fig_posteriors.{pdf,png}` | the ten record-lane (targeted) H0 posteriors overlaid, each scaled to its own peak -- the AGN densities are ~5x narrower and would otherwise flatten the galaxy curves; blue = galaxies, orange = AGN, seed 100 at full strength and the other four at a lighter step of the same hue.  Drawn on the window holding >= 99.99% of every curve's mass, not the full scanned [50, 100].  The bimodal galaxy realisation is drawn as it is, with its second mode marked |
| `figs/fig_recovery.{pdf,png}` | per-realisation medians +- 68% for both tracers against truth, the two tracers dodged either side of each seed, with each tracer's five-realisation mean offset +- standard error as a band |
| `figs/fig_lanes.{pdf,png}` | the targeted vs popuni median shift for all 20 scans as 10 same-events pairs: in units of that scan's 68% half-width (upper) and in km/s/Mpc (lower).  The two panels rank the scans differently, which is the point -- the largest AGN shift is 0.85 half-widths but only 0.37 km/s/Mpc |
| `figs/fig_diagnostics.{pdf,png}` | internal: the selection guard.  Per-scan minimum N_eff against the 5 N_obs floor (log), and the per-scan PE variance sum as its range over the 201 cells with the median marked; filled = targeted, open = popuni |
| `figs/fig_bimodal.{pdf,png}` | the s105 galaxy case on its own: the bimodal targeted posterior against the unimodal popuni one, both modes labelled with their relative height and both 68% intervals drawn as bars under the curves -- the single-scan look for a future reader who meets that wide interval in the closure table |

Colour is the project data-viz standard: identity is the tracer and only the tracer (categorical slots 1 and 2), the five realisations inside a tracer are an ordinal step of one hue rather than five hues, and every pair used together was checked with `../analysis_2_complete_catalog_H0_fagn/scripts/validate_palette.py` -- the header of `scripts/make_figures.py` records the measured separations.

## Layout

| path | what |
|---|---|
| `scripts/make_pure_tracer_events.sh` | draws the ten event sets |
| `scripts/check_pure_tracer_events.py` | sanity checks them -> `results/event_sets.json` |
| `scripts/scan_h0f.py` | analysis 1's likelihood-grid driver, byte for byte |
| `scripts/run_scans.sh` | the four scans of one realisation |
| `scripts/submit_scans.sbatch` | the original 5-task array (one task per realisation) |
| `scripts/submit_one_seed.sbatch` | one realisation, partition/QOS left to the command line so it can be aimed at whichever GPU is free |
| `scripts/bitcheck_events.py` | the dataset-level bit-identity gate |
| `scripts/aggregate_pure_tracer.py` | closure + constraining power -> `results/h0_pure_tracer.json` |
| `scripts/make_figures.py` | renders `figs/` from `results/` (PDF + PNG, one function per figure) |
| `scripts/make_readme.py` | renders this file from the JSON |
| `results/` | one `.h5` (grid + logL) and one `.json` (posterior summary) per scan, plus the aggregates |
| `figs/` | the five rendered figures, PDF + PNG |
| `logs/` | generation, per-scan and SLURM logs |

Event files live beside the record in `/hildafs/projects/phy220048p/magana/gws-agn-data-v3/seed<S>/events/`.
