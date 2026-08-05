# diagnostics/ — index of the closed investigation

**2026-08-01.**  The diagnostic campaign that ran on top of `analysis_1_complete_catalog_H0` is closed.  Its products were moved out of the analysis-of-record directories (`scripts/`, `results/`, `figs/`, `logs/`) into this subtree, organised by stage.  **Nothing was deleted.**  The analysis of record is unchanged and still lives at the top level; in particular `results/h0_single_tracer.json` — the only file `working/paper` reads from this analysis — is at its original path, byte for byte.

442 files moved into `diagnostics/`.  Stage totals:

| stage | scripts | results | figs | logs | data_derived | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `probes` | 9 | 9 | 2 | 6 | 0 | 26 |
| `attribution` | 39 | 149 | 16 | 100 | 4 | 308 |
| `endgame` | 14 | 56 | 0 | 38 | 0 | 108 |

---

## How to read a path

Every move is mechanical: `<sub>/<name>` at the top level became
`diagnostics/<stage>/<sub>/<name>`.  Each stage directory therefore mirrors the
layout of the analysis root, and the scripts' own convention — `cd "$(dirname
"$0")/.."` in the shell drivers, `ROOT = Path(__file__).parent.parent` in the
Python — resolves `results/`, `figs/` and `logs/` inside the stage, which is
where that stage's products now are.  `data_derived` is a symlink (or, in
`attribution/`, a directory containing symlinks) back to the record's
`data_derived/`, so the matched-host event subsets still resolve.

### Re-running an archived diagnostic

Same-stage work runs unchanged from the stage directory:

```bash
cd diagnostics/attribution && python scripts/attr_score_terms.py --tracer gal
```

Four cross-references survive the split and would need an explicit path:

* `probes/scripts/probe2_kde_window.py` and `probes/scripts/probe4_continuum_survey.py`
  shell out to `scan_h0f.py` **next to themselves**; it is a record script and stayed
  in `scripts/`.
* `endgame/scripts/run_dzscan.sh`, `run_endgame_tail.sh` and `run_v3_pilot.sh` call
  `attr_selmu_oracle.py` / `attr_selmu_pdet.py`, which are in `attribution/scripts/`,
  and `run_v3_pilot.sh` also calls the record's `build_hosttype_subset.py`.
* `attribution/scripts/submit_postfix_aux.sbatch` calls the record's
  `run_guard_diag.sh`.
* The `submit_*.sbatch` files `cd "$SLURM_SUBMIT_DIR"`; submit them from their own
  stage directory, not from the analysis root.

Two scripts were patched for the reorganisation, because their path constants
named directories that moved:

* `attribution/scripts/closure_summary.py` — cross-cutting; it now searches the
  record `results/` first, then its own stage, then `probes/results/`, and points
  `PRE`/`ATTIC` at `attic/results_prefix2` / `attic/results_dsc_attic`.
* `attic/scripts_onhold/compare_dsc_attic.py` — `OLD` now points at
  `attic/results_dsc_attic`.

The record's `scripts/fig_closure_after_fix.py` and `scripts/run_v3_analysis.sh`
were likewise repointed at `attic/results_prefix2`, `attic/results_dsc_attic` and
`attic/results_v2postfix`; `fig_closure_v3` regenerates from them unchanged.

---

## `diagnostics/probes/`

The four mechanism probes for the galaxy-catalog closure failure, plus the
survey-resolution (nside) study they led to.  **The probe 1-4 numeric outputs and
figures were archived before this reorganisation and are NOT here** -- they were
produced in the `dark_sirens_complete` era and live in
`attic/results_dsc_attic/`, `attic/figs_dsc_attic/` and `attic/logs_dsc_attic/`
(`probe1_*`, `probe2_*`, `probe3_*`, `probe4*`).  What moved into this stage is the
probe *code* plus the nside study's own scans, figure and logs.

Cited by: PROBES.md (probes 1-4) and CLOSURE.md 8 (the nside study).

### `probes/scripts/`

| old path | new path | what it is | cited in |
| --- | --- | --- | --- |
| `scripts/build_nside_surveys.py` | `diagnostics/probes/scripts/build_nside_surveys.py` | builds the nside-64 / nside-128 complete surveys for the resolution study | CLOSURE.md 8 |
| `scripts/probe1_pixelation_audit.py` | `diagnostics/probes/scripts/probe1_pixelation_audit.py` | probe 1: healpix pixelation audit of the catalog/survey build (CPU) | PROBES.md 'Probe 1' |
| `scripts/probe2_kde_window.py` | `diagnostics/probes/scripts/probe2_kde_window.py` | probe 2 driver: catalog-KDE window sweep; shells out to the record's scan_h0f.py per window | PROBES.md 'Probe 2' |
| `scripts/probe3_decomposition.py` | `diagnostics/probes/scripts/probe3_decomposition.py` | probe 3: numerator / selection decomposition of the H0 likelihood (per-run and --aggregate) | PROBES.md 'Probe 3' |
| `scripts/probe4_continuum_survey.py` | `diagnostics/probes/scripts/probe4_continuum_survey.py` | probe 4: analytic continuum survey (build / scan / analyse) | PROBES.md 'Probe 4' |
| `scripts/run_nside_scans.sh` | `diagnostics/probes/scripts/run_nside_scans.sh` | the resolution study's four matched-host control scans | CLOSURE.md 8 |
| `scripts/run_probe4.sh` | `diagnostics/probes/scripts/run_probe4.sh` | probe 4 scan + analyse driver | PROBES.md 'Probe 4' |
| `scripts/run_probe4_decomp.sh` | `diagnostics/probes/scripts/run_probe4_decomp.sh` | probe 3's decomposition re-run on probe 4's three surveys | PROBES.md 'Probe 4' |
| `scripts/submit_probe3.sbatch` | `diagnostics/probes/scripts/submit_probe3.sbatch` | TWIG-GPU submission for probe 3 (job 1058123) | PROBES.md 'Probe 3' |
### `probes/results/`

| old path | new path | what it is | cited in |
| --- | --- | --- | --- |
| `results/ctrl_agn_matched_ns128.h5` | `diagnostics/probes/results/ctrl_agn_matched_ns128.h5` | matched-host control scan on the nside-64/128 complete survey (residual vs survey resolution) | CLOSURE.md 8 |
| `results/ctrl_agn_matched_ns128.json` | `diagnostics/probes/results/ctrl_agn_matched_ns128.json` | matched-host control scan on the nside-64/128 complete survey (residual vs survey resolution) | CLOSURE.md 8 |
| `results/ctrl_agn_matched_ns64.h5` | `diagnostics/probes/results/ctrl_agn_matched_ns64.h5` | matched-host control scan on the nside-64/128 complete survey (residual vs survey resolution) | CLOSURE.md 8 |
| `results/ctrl_agn_matched_ns64.json` | `diagnostics/probes/results/ctrl_agn_matched_ns64.json` | matched-host control scan on the nside-64/128 complete survey (residual vs survey resolution) | CLOSURE.md 8 |
| `results/ctrl_gal_matched_ns128.h5` | `diagnostics/probes/results/ctrl_gal_matched_ns128.h5` | matched-host control scan on the nside-64/128 complete survey (residual vs survey resolution) | CLOSURE.md 8 |
| `results/ctrl_gal_matched_ns128.json` | `diagnostics/probes/results/ctrl_gal_matched_ns128.json` | matched-host control scan on the nside-64/128 complete survey (residual vs survey resolution) | CLOSURE.md 8 |
| `results/ctrl_gal_matched_ns64.h5` | `diagnostics/probes/results/ctrl_gal_matched_ns64.h5` | matched-host control scan on the nside-64/128 complete survey (residual vs survey resolution) | CLOSURE.md 8 |
| `results/ctrl_gal_matched_ns64.json` | `diagnostics/probes/results/ctrl_gal_matched_ns64.json` | matched-host control scan on the nside-64/128 complete survey (residual vs survey resolution) | CLOSURE.md 8 |
| `results/surveys_nside.json` | `diagnostics/probes/results/surveys_nside.json` | provenance and row statistics of the nside-64/128 surveys | CLOSURE.md 8 |
### `probes/figs/`

| old path | new path | what it is | cited in |
| --- | --- | --- | --- |
| `figs/fig_nside_curve.pdf` | `diagnostics/probes/figs/fig_nside_curve.pdf` | matched-host offset against survey resolution | CLOSURE.md 8 |
| `figs/fig_nside_curve.png` | `diagnostics/probes/figs/fig_nside_curve.png` | matched-host offset against survey resolution | CLOSURE.md 8 |
### `probes/logs/`

| old path | new path | what it is | cited in |
| --- | --- | --- | --- |
| `logs/build_nside_surveys.log` | `diagnostics/probes/logs/build_nside_surveys.log` | stdout of the nside survey build | CLOSURE.md 8 |
| `logs/ctrl_agn_matched_ns128.log` | `diagnostics/probes/logs/ctrl_agn_matched_ns128.log` | stdout of the nside control scan | CLOSURE.md 8 |
| `logs/ctrl_agn_matched_ns64.log` | `diagnostics/probes/logs/ctrl_agn_matched_ns64.log` | stdout of the nside control scan | CLOSURE.md 8 |
| `logs/ctrl_gal_matched_ns128.log` | `diagnostics/probes/logs/ctrl_gal_matched_ns128.log` | stdout of the nside control scan | CLOSURE.md 8 |
| `logs/ctrl_gal_matched_ns64.log` | `diagnostics/probes/logs/ctrl_gal_matched_ns64.log` | stdout of the nside control scan | CLOSURE.md 8 |
| `logs/run_nside_scans.log` | `diagnostics/probes/logs/run_nside_scans.log` | stdout of the resolution study's scan driver | CLOSURE.md 8 |

## `diagnostics/attribution/`

Attribution of the per-event score residual `r`: the term-by-term split, the
named mass-PE defect and what repairing it is worth, the (m1, m2, dL) quadrature
oracle, the exact host-galaxy sky oracle, the selection-integral sweep
(`P_det`, `mu(H0)`, injections, the `G(b)` battery), the `chi_eff` clip and the
host-acceptance convention.  Also the v2 (b2)/(c2) post-fix closure accounting,
which this stage's `closure_summary.py` gathers.

Cited by: ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13.

### `attribution/scripts/`

| old path | new path | what it is | cited in |
| --- | --- | --- | --- |
| `scripts/attr_chieff_clip.py` | `diagnostics/attribution/scripts/attr_chieff_clip.py` | task 2: the chi_eff clip substitution | CLOSURE.md 12 |
| `scripts/attr_ds_bridge.py` | `diagnostics/attribution/scripts/attr_ds_bridge.py` | shared anchored darksirens loader + weight rebuild; imported by every attr_* script | ATTRIBUTION.md 'Files written (appendix)' |
| `scripts/attr_figures.py` | `diagnostics/attribution/scripts/attr_figures.py` | the attribution figure (where r lives, the PE arms, the posterior mass bias) | ATTRIBUTION.md 'Files written' |
| `scripts/attr_fix_figures.py` | `diagnostics/attribution/scripts/attr_fix_figures.py` | the three appendix figures | ATTRIBUTION.md 'Files written (appendix)' |
| `scripts/attr_fix_summary.py` | `diagnostics/attribution/scripts/attr_fix_summary.py` | the combined appendix verdict JSON | ATTRIBUTION.md 'A4' |
| `scripts/attr_hostw.py` | `diagnostics/attribution/scripts/attr_hostw.py` | task 3: the host-acceptance convention | CLOSURE.md 13 |
| `scripts/attr_mass_pe.py` | `diagnostics/attribution/scripts/attr_mass_pe.py` | mass-channel PE arms, truth-point split, paired (C-A) | ATTRIBUTION.md 'Stage 2' |
| `scripts/attr_oracle.py` | `diagnostics/attribution/scripts/attr_oracle.py` | task 3: the (m1, m2, dL) quadrature oracle, 5 arms | ATTRIBUTION.md 'A3' |
| `scripts/attr_sampler_ratio.py` | `diagnostics/attribution/scripts/attr_sampler_ratio.py` | task 1: 1.2e8 population-sampler draws, closed-form density, the prediction | ATTRIBUTION.md 'A1' |
| `scripts/attr_score_terms.py` | `diagnostics/attribution/scripts/attr_score_terms.py` | term-by-term split of the per-event score residual r (GPU); bit-exact anchors | ATTRIBUTION.md 'Stage 1' |
| `scripts/attr_selmu_gconv.py` | `diagnostics/attribution/scripts/attr_selmu_gconv.py` | the G(b) quadrature convergence battery (CPU, 6 tasks) | CLOSURE.md 11.3 |
| `scripts/attr_selmu_gencheck.py` | `diagnostics/attribution/scripts/attr_selmu_gencheck.py` | the generative replay of stage_events | CLOSURE.md 11.3 |
| `scripts/attr_selmu_inj.py` | `diagnostics/attribution/scripts/attr_selmu_inj.py` | darksirens' injection-based log mu(H0) curves, per-branch estimators, MC-error bootstrap | CLOSURE.md 11.4 |
| `scripts/attr_selmu_mcerr.py` | `diagnostics/attribution/scripts/attr_selmu_mcerr.py` | lean Poisson bootstrap of the selection estimator's sigma_MC | CLOSURE.md 16.6 |
| `scripts/attr_selmu_oracle.py` | `diagnostics/attribution/scripts/attr_selmu_oracle.py` | mu(H0) and d ln mu/dH0 by quadrature, four host measures, every catalog galaxy | CLOSURE.md 11.2, 16.2 |
| `scripts/attr_selmu_pdet.py` | `diagnostics/attribution/scripts/attr_selmu_pdet.py` | P_det in closed form + brute-force validation against the generator's own observe() | CLOSURE.md 11.1, 16.2 |
| `scripts/attr_selmu_summary.py` | `diagnostics/attribution/scripts/attr_selmu_summary.py` | the task-1 verdict JSON | CLOSURE.md 11 |
| `scripts/attr_sky_oracle.py` | `diagnostics/attribution/scripts/attr_sky_oracle.py` | the exact host-galaxy sky oracle, 4 arms (+ opt-in --host_prior_arms) | CLOSURE.md 7, 13 |
| `scripts/attr_toy_masswidth.py` | `diagnostics/attribution/scripts/attr_toy_masswidth.py` | closed-form toy for the mass-measurement convention (CPU) | ATTRIBUTION.md 'Closed-form confirmation' |
| `scripts/build_catalog_skyindex.py` | `diagnostics/attribution/scripts/build_catalog_skyindex.py` | galaxy positions in the survey's row order -- the sky oracle's input index | CLOSURE.md 7 |
| `scripts/closure_summary.py` | `diagnostics/attribution/scripts/closure_summary.py` | gathers the whole v2 closure campaign into one JSON (patched for the reorg: RES now searches record -> stage -> probes) | CLOSURE.md 1-10 |
| `scripts/fig_selmu_oracle.py` | `diagnostics/attribution/scripts/fig_selmu_oracle.py` | the selection-oracle figure | CLOSURE.md 11 |
| `scripts/fig_sky_oracle.py` | `diagnostics/attribution/scripts/fig_sky_oracle.py` | the sky-oracle figures and the nside curve | CLOSURE.md 7, 8 |
| `scripts/make_pe_corrected_events.py` | `diagnostics/attribution/scripts/make_pe_corrected_events.py` | task 2: p_pe -> p_pe/rho corrected copies of the matched event files | ATTRIBUTION.md 'A2' |
| `scripts/run_fix_scans.sh` | `diagnostics/attribution/scripts/run_fix_scans.sh` | task 2: the four repaired matched scans | ATTRIBUTION.md 'A2' |
| `scripts/run_gal_conv_local.sh` | `diagnostics/attribution/scripts/run_gal_conv_local.sh` | the GAL sky-oracle convergence battery (needs the 80 GB card) | CLOSURE.md 7 |
| `scripts/run_hostw_chieff.sh` | `diagnostics/attribution/scripts/run_hostw_chieff.sh` | tasks 2 and 3 driver (chi_eff clip, host-acceptance convention) | CLOSURE.md 12, 13 |
| `scripts/run_local_sweep.sh` | `diagnostics/attribution/scripts/run_local_sweep.sh` | the GAL half of the final sweep (80 GB card) | CLOSURE.md 11-13 |
| `scripts/run_oracle.sh` | `diagnostics/attribution/scripts/run_oracle.sh` | task 3: oracle production + the 8-run convergence battery | ATTRIBUTION.md 'A3' |
| `scripts/run_postfix_attr.sh` | `diagnostics/attribution/scripts/run_postfix_attr.sh` | post-fix chain: score terms on the regenerated events, then the sky oracle | CLOSURE.md 6, 7 |
| `scripts/run_selmu.sh` | `diagnostics/attribution/scripts/run_selmu.sh` | final-sweep driver (P_det, the exact oracle) | CLOSURE.md 11 |
| `scripts/run_selmu_seeds.sh` | `diagnostics/attribution/scripts/run_selmu_seeds.sh` | the exact selection oracle B on every realisation | CLOSURE.md 15.3 |
| `scripts/run_sky_oracle.sh` | `diagnostics/attribution/scripts/run_sky_oracle.sh` | sky-oracle production + convergence battery | CLOSURE.md 7 |
| `scripts/submit_agn_aux.sbatch` | `diagnostics/attribution/scripts/submit_agn_aux.sbatch` | AGN injection curves + the task-3 host-prior arms (HENON-GPU) | CLOSURE.md 11, 13 |
| `scripts/submit_agn_inj2.sbatch` | `diagnostics/attribution/scripts/submit_agn_inj2.sbatch` | AGN injection curves with the MC-error bootstrap (HENON-GPU) | CLOSURE.md 11.4 |
| `scripts/submit_gal_conv.sbatch` | `diagnostics/attribution/scripts/submit_gal_conv.sbatch` | GAL sky-oracle convergence battery on HENON-GPU | CLOSURE.md 7 |
| `scripts/submit_gal_hostw.sbatch` | `diagnostics/attribution/scripts/submit_gal_hostw.sbatch` | task 3, GAL: the sky oracle with --host_prior_arms | CLOSURE.md 13 |
| `scripts/submit_gconv.sbatch` | `diagnostics/attribution/scripts/submit_gconv.sbatch` | the G(b) convergence battery, 6 CPU array tasks | CLOSURE.md 11.3 |
| `scripts/submit_postfix_aux.sbatch` | `diagnostics/attribution/scripts/submit_postfix_aux.sbatch` | post-fix auxiliaries: guard diagnostics + sky-oracle convergence | CLOSURE.md 7 |
### `attribution/results/`

| old path | new path | what it is | cited in |
| --- | --- | --- | --- |
| `results/attr_chieff.json` | `diagnostics/attribution/results/attr_chieff.json` | the chi_eff clip substitution and its per-event arrays | CLOSURE.md 12 |
| `results/attr_chieff_agn.npz` | `diagnostics/attribution/results/attr_chieff_agn.npz` | the chi_eff clip substitution and its per-event arrays | CLOSURE.md 12 |
| `results/attr_chieff_gal.npz` | `diagnostics/attribution/results/attr_chieff_gal.npz` | the chi_eff clip substitution and its per-event arrays | CLOSURE.md 12 |
| `results/attr_fix_summary.json` | `diagnostics/attribution/results/attr_fix_summary.json` | every number quoted in the ATTRIBUTION.md appendix | ATTRIBUTION.md 'A4' |
| `results/attr_hostw.json` | `diagnostics/attribution/results/attr_hostw.json` | the host-acceptance convention substitution | CLOSURE.md 13 |
| `results/attr_mass_pe_agn_s100.json` | `diagnostics/attribution/results/attr_mass_pe_agn_s100.json` | 7 PE arms, the truth-point split, per-event arrays | ATTRIBUTION.md 'Stage 2' |
| `results/attr_mass_pe_agn_s100.npz` | `diagnostics/attribution/results/attr_mass_pe_agn_s100.npz` | 7 PE arms, the truth-point split, per-event arrays | ATTRIBUTION.md 'Stage 2' |
| `results/attr_mass_pe_gal_s100.json` | `diagnostics/attribution/results/attr_mass_pe_gal_s100.json` | 7 PE arms, the truth-point split, per-event arrays | ATTRIBUTION.md 'Stage 2' |
| `results/attr_mass_pe_gal_s100.npz` | `diagnostics/attribution/results/attr_mass_pe_gal_s100.npz` | 7 PE arms, the truth-point split, per-event arrays | ATTRIBUTION.md 'Stage 2' |
| `results/attr_oracle_agn.json` | `diagnostics/attribution/results/attr_oracle_agn.json` | the (m1, m2, dL) quadrature oracle, production run | ATTRIBUTION.md 'A3' |
| `results/attr_oracle_agn.npz` | `diagnostics/attribution/results/attr_oracle_agn.npz` | the (m1, m2, dL) quadrature oracle, production run | ATTRIBUTION.md 'A3' |
| `results/attr_oracle_agn_conv_nm.json` | `diagnostics/attribution/results/attr_oracle_agn_conv_nm.json` | the (m1, m2, dL) quadrature oracle's convergence battery | ATTRIBUTION.md 'A3' |
| `results/attr_oracle_agn_conv_nm.npz` | `diagnostics/attribution/results/attr_oracle_agn_conv_nm.npz` | the (m1, m2, dL) quadrature oracle's convergence battery | ATTRIBUTION.md 'A3' |
| `results/attr_oracle_agn_conv_nz.json` | `diagnostics/attribution/results/attr_oracle_agn_conv_nz.json` | the (m1, m2, dL) quadrature oracle's convergence battery | ATTRIBUTION.md 'A3' |
| `results/attr_oracle_agn_conv_nz.npz` | `diagnostics/attribution/results/attr_oracle_agn_conv_nz.npz` | the (m1, m2, dL) quadrature oracle's convergence battery | ATTRIBUTION.md 'A3' |
| `results/attr_oracle_agn_conv_sh.json` | `diagnostics/attribution/results/attr_oracle_agn_conv_sh.json` | the (m1, m2, dL) quadrature oracle's convergence battery | ATTRIBUTION.md 'A3' |
| `results/attr_oracle_agn_conv_sh.npz` | `diagnostics/attribution/results/attr_oracle_agn_conv_sh.npz` | the (m1, m2, dL) quadrature oracle's convergence battery | ATTRIBUTION.md 'A3' |
| `results/attr_oracle_agn_conv_sky.json` | `diagnostics/attribution/results/attr_oracle_agn_conv_sky.json` | the (m1, m2, dL) quadrature oracle's convergence battery | ATTRIBUTION.md 'A3' |
| `results/attr_oracle_agn_conv_sky.npz` | `diagnostics/attribution/results/attr_oracle_agn_conv_sky.npz` | the (m1, m2, dL) quadrature oracle's convergence battery | ATTRIBUTION.md 'A3' |
| `results/attr_oracle_gal.json` | `diagnostics/attribution/results/attr_oracle_gal.json` | the (m1, m2, dL) quadrature oracle, production run | ATTRIBUTION.md 'A3' |
| `results/attr_oracle_gal.npz` | `diagnostics/attribution/results/attr_oracle_gal.npz` | the (m1, m2, dL) quadrature oracle, production run | ATTRIBUTION.md 'A3' |
| `results/attr_oracle_gal_conv_nm.json` | `diagnostics/attribution/results/attr_oracle_gal_conv_nm.json` | the (m1, m2, dL) quadrature oracle's convergence battery | ATTRIBUTION.md 'A3' |
| `results/attr_oracle_gal_conv_nm.npz` | `diagnostics/attribution/results/attr_oracle_gal_conv_nm.npz` | the (m1, m2, dL) quadrature oracle's convergence battery | ATTRIBUTION.md 'A3' |
| `results/attr_oracle_gal_conv_nz.json` | `diagnostics/attribution/results/attr_oracle_gal_conv_nz.json` | the (m1, m2, dL) quadrature oracle's convergence battery | ATTRIBUTION.md 'A3' |
| `results/attr_oracle_gal_conv_nz.npz` | `diagnostics/attribution/results/attr_oracle_gal_conv_nz.npz` | the (m1, m2, dL) quadrature oracle's convergence battery | ATTRIBUTION.md 'A3' |
| `results/attr_oracle_gal_conv_sh.json` | `diagnostics/attribution/results/attr_oracle_gal_conv_sh.json` | the (m1, m2, dL) quadrature oracle's convergence battery | ATTRIBUTION.md 'A3' |
| `results/attr_oracle_gal_conv_sh.npz` | `diagnostics/attribution/results/attr_oracle_gal_conv_sh.npz` | the (m1, m2, dL) quadrature oracle's convergence battery | ATTRIBUTION.md 'A3' |
| `results/attr_oracle_gal_conv_sky.json` | `diagnostics/attribution/results/attr_oracle_gal_conv_sky.json` | the (m1, m2, dL) quadrature oracle's convergence battery | ATTRIBUTION.md 'A3' |
| `results/attr_oracle_gal_conv_sky.npz` | `diagnostics/attribution/results/attr_oracle_gal_conv_sky.npz` | the (m1, m2, dL) quadrature oracle's convergence battery | ATTRIBUTION.md 'A3' |
| `results/attr_sampler_draws.npz` | `diagnostics/attribution/results/attr_sampler_draws.npz` | task 1: the sampler-ratio test and its raw draws | ATTRIBUTION.md 'A1' |
| `results/attr_sampler_ratio.json` | `diagnostics/attribution/results/attr_sampler_ratio.json` | task 1: the sampler-ratio test and its raw draws | ATTRIBUTION.md 'A1' |
| `results/attr_sampler_ratio.npz` | `diagnostics/attribution/results/attr_sampler_ratio.npz` | task 1: the sampler-ratio test and its raw draws | ATTRIBUTION.md 'A1' |
| `results/attr_selmu_agn.json` | `diagnostics/attribution/results/attr_selmu_agn.json` | the exact selection oracle B, seed 100 | CLOSURE.md 11.2 |
| `results/attr_selmu_agn.npz` | `diagnostics/attribution/results/attr_selmu_agn.npz` | the exact selection oracle B, seed 100 | CLOSURE.md 11.2 |
| `results/attr_selmu_agn_regress.json` | `diagnostics/attribution/results/attr_selmu_agn_regress.json` | the no-override bitwise regression of the selection oracle | CLOSURE.md 15.4 |
| `results/attr_selmu_agn_regress.npz` | `diagnostics/attribution/results/attr_selmu_agn_regress.npz` | the no-override bitwise regression of the selection oracle | CLOSURE.md 15.4 |
| `results/attr_selmu_agn_s101.json` | `diagnostics/attribution/results/attr_selmu_agn_s101.json` | the exact selection oracle B on a further realisation | CLOSURE.md 15.3 |
| `results/attr_selmu_agn_s101.npz` | `diagnostics/attribution/results/attr_selmu_agn_s101.npz` | the exact selection oracle B on a further realisation | CLOSURE.md 15.3 |
| `results/attr_selmu_agn_s102.json` | `diagnostics/attribution/results/attr_selmu_agn_s102.json` | the exact selection oracle B on a further realisation | CLOSURE.md 15.3 |
| `results/attr_selmu_agn_s102.npz` | `diagnostics/attribution/results/attr_selmu_agn_s102.npz` | the exact selection oracle B on a further realisation | CLOSURE.md 15.3 |
| `results/attr_selmu_agn_s103.json` | `diagnostics/attribution/results/attr_selmu_agn_s103.json` | the exact selection oracle B on a further realisation | CLOSURE.md 15.3 |
| `results/attr_selmu_agn_s103.npz` | `diagnostics/attribution/results/attr_selmu_agn_s103.npz` | the exact selection oracle B on a further realisation | CLOSURE.md 15.3 |
| `results/attr_selmu_agn_s105.json` | `diagnostics/attribution/results/attr_selmu_agn_s105.json` | the exact selection oracle B on a further realisation | CLOSURE.md 15.3 |
| `results/attr_selmu_agn_s105.npz` | `diagnostics/attribution/results/attr_selmu_agn_s105.npz` | the exact selection oracle B on a further realisation | CLOSURE.md 15.3 |
| `results/attr_selmu_agn_v3_s100.json` | `diagnostics/attribution/results/attr_selmu_agn_v3_s100.json` | the exact selection oracle B under the v3 measurement family | CLOSURE.md 16.2 |
| `results/attr_selmu_agn_v3_s100.npz` | `diagnostics/attribution/results/attr_selmu_agn_v3_s100.npz` | the exact selection oracle B under the v3 measurement family | CLOSURE.md 16.2 |
| `results/attr_selmu_gal.json` | `diagnostics/attribution/results/attr_selmu_gal.json` | the exact selection oracle B, seed 100 | CLOSURE.md 11.2 |
| `results/attr_selmu_gal.npz` | `diagnostics/attribution/results/attr_selmu_gal.npz` | the exact selection oracle B, seed 100 | CLOSURE.md 11.2 |
| `results/attr_selmu_gal_s101.json` | `diagnostics/attribution/results/attr_selmu_gal_s101.json` | the exact selection oracle B on a further realisation | CLOSURE.md 15.3 |
| `results/attr_selmu_gal_s101.npz` | `diagnostics/attribution/results/attr_selmu_gal_s101.npz` | the exact selection oracle B on a further realisation | CLOSURE.md 15.3 |
| `results/attr_selmu_gal_s102.json` | `diagnostics/attribution/results/attr_selmu_gal_s102.json` | the exact selection oracle B on a further realisation | CLOSURE.md 15.3 |
| `results/attr_selmu_gal_s102.npz` | `diagnostics/attribution/results/attr_selmu_gal_s102.npz` | the exact selection oracle B on a further realisation | CLOSURE.md 15.3 |
| `results/attr_selmu_gal_s103.json` | `diagnostics/attribution/results/attr_selmu_gal_s103.json` | the exact selection oracle B on a further realisation | CLOSURE.md 15.3 |
| `results/attr_selmu_gal_s103.npz` | `diagnostics/attribution/results/attr_selmu_gal_s103.npz` | the exact selection oracle B on a further realisation | CLOSURE.md 15.3 |
| `results/attr_selmu_gal_s105.json` | `diagnostics/attribution/results/attr_selmu_gal_s105.json` | the exact selection oracle B on a further realisation | CLOSURE.md 15.3 |
| `results/attr_selmu_gal_s105.npz` | `diagnostics/attribution/results/attr_selmu_gal_s105.npz` | the exact selection oracle B on a further realisation | CLOSURE.md 15.3 |
| `results/attr_selmu_gal_v3_s100.json` | `diagnostics/attribution/results/attr_selmu_gal_v3_s100.json` | the exact selection oracle B under the v3 measurement family | CLOSURE.md 16.2 |
| `results/attr_selmu_gal_v3_s100.npz` | `diagnostics/attribution/results/attr_selmu_gal_v3_s100.npz` | the exact selection oracle B under the v3 measurement family | CLOSURE.md 16.2 |
| `results/attr_selmu_gconv.json` | `diagnostics/attribution/results/attr_selmu_gconv.json` | the G(b) quadrature convergence battery (one knob per arm) | CLOSURE.md 11.3 |
| `results/attr_selmu_gconv_base.json` | `diagnostics/attribution/results/attr_selmu_gconv_base.json` | the G(b) quadrature convergence battery (one knob per arm) | CLOSURE.md 11.3 |
| `results/attr_selmu_gconv_dv_half.json` | `diagnostics/attribution/results/attr_selmu_gconv_dv_half.json` | the G(b) quadrature convergence battery (one knob per arm) | CLOSURE.md 11.3 |
| `results/attr_selmu_gconv_n_ghx2.json` | `diagnostics/attribution/results/attr_selmu_gconv_n_ghx2.json` | the G(b) quadrature convergence battery (one knob per arm) | CLOSURE.md 11.3 |
| `results/attr_selmu_gconv_n_m1x2.json` | `diagnostics/attribution/results/attr_selmu_gconv_n_m1x2.json` | the G(b) quadrature convergence battery (one knob per arm) | CLOSURE.md 11.3 |
| `results/attr_selmu_gconv_n_qx2.json` | `diagnostics/attribution/results/attr_selmu_gconv_n_qx2.json` | the G(b) quadrature convergence battery (one knob per arm) | CLOSURE.md 11.3 |
| `results/attr_selmu_gconv_range_wide.json` | `diagnostics/attribution/results/attr_selmu_gconv_range_wide.json` | the G(b) quadrature convergence battery (one knob per arm) | CLOSURE.md 11.3 |
| `results/attr_selmu_gencheck.json` | `diagnostics/attribution/results/attr_selmu_gencheck.json` | the generative replay check of stage_events | CLOSURE.md 11.3 |
| `results/attr_selmu_inj_agn_popuni.json` | `diagnostics/attribution/results/attr_selmu_inj_agn_popuni.json` | darksirens' injection-based log mu(H0) on the given lane | CLOSURE.md 11.4 |
| `results/attr_selmu_inj_agn_popuni.npz` | `diagnostics/attribution/results/attr_selmu_inj_agn_popuni.npz` | darksirens' injection-based log mu(H0) on the given lane | CLOSURE.md 11.4 |
| `results/attr_selmu_inj_agn_targeted.json` | `diagnostics/attribution/results/attr_selmu_inj_agn_targeted.json` | darksirens' injection-based log mu(H0) on the given lane | CLOSURE.md 11.4 |
| `results/attr_selmu_inj_agn_targeted.npz` | `diagnostics/attribution/results/attr_selmu_inj_agn_targeted.npz` | darksirens' injection-based log mu(H0) on the given lane | CLOSURE.md 11.4 |
| `results/attr_selmu_inj_gal_popuni.json` | `diagnostics/attribution/results/attr_selmu_inj_gal_popuni.json` | darksirens' injection-based log mu(H0) on the given lane | CLOSURE.md 11.4 |
| `results/attr_selmu_inj_gal_popuni.npz` | `diagnostics/attribution/results/attr_selmu_inj_gal_popuni.npz` | darksirens' injection-based log mu(H0) on the given lane | CLOSURE.md 11.4 |
| `results/attr_selmu_inj_gal_targeted.json` | `diagnostics/attribution/results/attr_selmu_inj_gal_targeted.json` | darksirens' injection-based log mu(H0) on the given lane | CLOSURE.md 11.4 |
| `results/attr_selmu_inj_gal_targeted.npz` | `diagnostics/attribution/results/attr_selmu_inj_gal_targeted.npz` | darksirens' injection-based log mu(H0) on the given lane | CLOSURE.md 11.4 |
| `results/attr_selmu_mcerr_agn_popuni_v3_s100.json` | `diagnostics/attribution/results/attr_selmu_mcerr_agn_popuni_v3_s100.json` | sigma_MC of the selection estimator under v3 | CLOSURE.md 16.6 |
| `results/attr_selmu_mcerr_agn_targeted_v3_s100.json` | `diagnostics/attribution/results/attr_selmu_mcerr_agn_targeted_v3_s100.json` | sigma_MC of the selection estimator under v3 | CLOSURE.md 16.6 |
| `results/attr_selmu_mcerr_gal_popuni_v3_s100.json` | `diagnostics/attribution/results/attr_selmu_mcerr_gal_popuni_v3_s100.json` | sigma_MC of the selection estimator under v3 | CLOSURE.md 16.6 |
| `results/attr_selmu_mcerr_gal_targeted_v3_s100.json` | `diagnostics/attribution/results/attr_selmu_mcerr_gal_targeted_v3_s100.json` | sigma_MC of the selection estimator under v3 | CLOSURE.md 16.6 |
| `results/attr_selmu_pdet.json` | `diagnostics/attribution/results/attr_selmu_pdet.json` | P_det in closed form (v2) against the generator's own observe() | CLOSURE.md 11.1 |
| `results/attr_selmu_pdet.npz` | `diagnostics/attribution/results/attr_selmu_pdet.npz` | P_det in closed form (v2) against the generator's own observe() | CLOSURE.md 11.1 |
| `results/attr_selmu_pdet_v3.json` | `diagnostics/attribution/results/attr_selmu_pdet_v3.json` | P_det in closed form under the v3 detection rule, against observe_v3/detect_v3 | CLOSURE.md 16.2 |
| `results/attr_selmu_summary.json` | `diagnostics/attribution/results/attr_selmu_summary.json` | the task-1 verdict for the selection integral | CLOSURE.md 11 |
| `results/attr_sky_oracle_agn.json` | `diagnostics/attribution/results/attr_sky_oracle_agn.json` | the exact host-galaxy sky oracle, production run | CLOSURE.md 7 |
| `results/attr_sky_oracle_agn.npz` | `diagnostics/attribution/results/attr_sky_oracle_agn.npz` | the exact host-galaxy sky oracle, production run | CLOSURE.md 7 |
| `results/attr_sky_oracle_agn_conv_ap4.json` | `diagnostics/attribution/results/attr_sky_oracle_agn_conv_ap4.json` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_agn_conv_ap4.npz` | `diagnostics/attribution/results/attr_sky_oracle_agn_conv_ap4.npz` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_agn_conv_ap8.json` | `diagnostics/attribution/results/attr_sky_oracle_agn_conv_ap8.json` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_agn_conv_ap8.npz` | `diagnostics/attribution/results/attr_sky_oracle_agn_conv_ap8.npz` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_agn_conv_base.json` | `diagnostics/attribution/results/attr_sky_oracle_agn_conv_base.json` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_agn_conv_base.npz` | `diagnostics/attribution/results/attr_sky_oracle_agn_conv_base.npz` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_agn_conv_nm.json` | `diagnostics/attribution/results/attr_sky_oracle_agn_conv_nm.json` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_agn_conv_nm.npz` | `diagnostics/attribution/results/attr_sky_oracle_agn_conv_nm.npz` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_agn_conv_nz.json` | `diagnostics/attribution/results/attr_sky_oracle_agn_conv_nz.json` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_agn_conv_nz.npz` | `diagnostics/attribution/results/attr_sky_oracle_agn_conv_nz.npz` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_agn_conv_sf5.json` | `diagnostics/attribution/results/attr_sky_oracle_agn_conv_sf5.json` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_agn_conv_sf5.npz` | `diagnostics/attribution/results/attr_sky_oracle_agn_conv_sf5.npz` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_agn_conv_sf7.json` | `diagnostics/attribution/results/attr_sky_oracle_agn_conv_sf7.json` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_agn_conv_sf7.npz` | `diagnostics/attribution/results/attr_sky_oracle_agn_conv_sf7.npz` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_agn_conv_shift.json` | `diagnostics/attribution/results/attr_sky_oracle_agn_conv_shift.json` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_agn_conv_shift.npz` | `diagnostics/attribution/results/attr_sky_oracle_agn_conv_shift.npz` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_agn_conv_sub4.json` | `diagnostics/attribution/results/attr_sky_oracle_agn_conv_sub4.json` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_agn_conv_sub4.npz` | `diagnostics/attribution/results/attr_sky_oracle_agn_conv_sub4.npz` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_agn_conv_sub6.json` | `diagnostics/attribution/results/attr_sky_oracle_agn_conv_sub6.json` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_agn_conv_sub6.npz` | `diagnostics/attribution/results/attr_sky_oracle_agn_conv_sub6.npz` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_agn_hostw.json` | `diagnostics/attribution/results/attr_sky_oracle_agn_hostw.json` | the sky oracle's host-acceptance (task 3) arms | CLOSURE.md 13 |
| `results/attr_sky_oracle_agn_hostw.npz` | `diagnostics/attribution/results/attr_sky_oracle_agn_hostw.npz` | the sky oracle's host-acceptance (task 3) arms | CLOSURE.md 13 |
| `results/attr_sky_oracle_gal.json` | `diagnostics/attribution/results/attr_sky_oracle_gal.json` | the exact host-galaxy sky oracle, production run | CLOSURE.md 7 |
| `results/attr_sky_oracle_gal.npz` | `diagnostics/attribution/results/attr_sky_oracle_gal.npz` | the exact host-galaxy sky oracle, production run | CLOSURE.md 7 |
| `results/attr_sky_oracle_gal_conv_ap4.json` | `diagnostics/attribution/results/attr_sky_oracle_gal_conv_ap4.json` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_gal_conv_ap4.npz` | `diagnostics/attribution/results/attr_sky_oracle_gal_conv_ap4.npz` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_gal_conv_base.json` | `diagnostics/attribution/results/attr_sky_oracle_gal_conv_base.json` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_gal_conv_base.npz` | `diagnostics/attribution/results/attr_sky_oracle_gal_conv_base.npz` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_gal_conv_nm.json` | `diagnostics/attribution/results/attr_sky_oracle_gal_conv_nm.json` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_gal_conv_nm.npz` | `diagnostics/attribution/results/attr_sky_oracle_gal_conv_nm.npz` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_gal_conv_nz.json` | `diagnostics/attribution/results/attr_sky_oracle_gal_conv_nz.json` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_gal_conv_nz.npz` | `diagnostics/attribution/results/attr_sky_oracle_gal_conv_nz.npz` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_gal_conv_sf5.json` | `diagnostics/attribution/results/attr_sky_oracle_gal_conv_sf5.json` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_gal_conv_sf5.npz` | `diagnostics/attribution/results/attr_sky_oracle_gal_conv_sf5.npz` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_gal_conv_sf7.json` | `diagnostics/attribution/results/attr_sky_oracle_gal_conv_sf7.json` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_gal_conv_sf7.npz` | `diagnostics/attribution/results/attr_sky_oracle_gal_conv_sf7.npz` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_gal_conv_shift.json` | `diagnostics/attribution/results/attr_sky_oracle_gal_conv_shift.json` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_gal_conv_shift.npz` | `diagnostics/attribution/results/attr_sky_oracle_gal_conv_shift.npz` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_gal_conv_sub4.json` | `diagnostics/attribution/results/attr_sky_oracle_gal_conv_sub4.json` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_gal_conv_sub4.npz` | `diagnostics/attribution/results/attr_sky_oracle_gal_conv_sub4.npz` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_gal_conv_sub6.json` | `diagnostics/attribution/results/attr_sky_oracle_gal_conv_sub6.json` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_gal_conv_sub6.npz` | `diagnostics/attribution/results/attr_sky_oracle_gal_conv_sub6.npz` | the sky oracle's convergence battery arm | CLOSURE.md 7 |
| `results/attr_sky_oracle_gal_hostw.json` | `diagnostics/attribution/results/attr_sky_oracle_gal_hostw.json` | the sky oracle's host-acceptance (task 3) arms | CLOSURE.md 13 |
| `results/attr_sky_oracle_gal_hostw.npz` | `diagnostics/attribution/results/attr_sky_oracle_gal_hostw.npz` | the sky oracle's host-acceptance (task 3) arms | CLOSURE.md 13 |
| `results/attr_terms_agn_s100.json` | `diagnostics/attribution/results/attr_terms_agn_s100.json` | the three-term split of r + per-event arrays (pre-fix) | ATTRIBUTION.md 'Stage 1' |
| `results/attr_terms_agn_s100.npz` | `diagnostics/attribution/results/attr_terms_agn_s100.npz` | the three-term split of r + per-event arrays (pre-fix) | ATTRIBUTION.md 'Stage 1' |
| `results/attr_terms_agn_s100_postfix.json` | `diagnostics/attribution/results/attr_terms_agn_s100_postfix.json` | the three-term split of r on the regenerated (b2)/(c2) events | CLOSURE.md 6 |
| `results/attr_terms_agn_s100_postfix.npz` | `diagnostics/attribution/results/attr_terms_agn_s100_postfix.npz` | the three-term split of r on the regenerated (b2)/(c2) events | CLOSURE.md 6 |
| `results/attr_terms_gal_s100.json` | `diagnostics/attribution/results/attr_terms_gal_s100.json` | the three-term split of r + per-event arrays (pre-fix) | ATTRIBUTION.md 'Stage 1' |
| `results/attr_terms_gal_s100.npz` | `diagnostics/attribution/results/attr_terms_gal_s100.npz` | the three-term split of r + per-event arrays (pre-fix) | ATTRIBUTION.md 'Stage 1' |
| `results/attr_terms_gal_s100_postfix.json` | `diagnostics/attribution/results/attr_terms_gal_s100_postfix.json` | the three-term split of r on the regenerated (b2)/(c2) events | CLOSURE.md 6 |
| `results/attr_terms_gal_s100_postfix.npz` | `diagnostics/attribution/results/attr_terms_gal_s100_postfix.npz` | the three-term split of r on the regenerated (b2)/(c2) events | CLOSURE.md 6 |
| `results/attr_toy_masswidth.json` | `diagnostics/attribution/results/attr_toy_masswidth.json` | the toy: stored / obswidth / exact mass-width conventions | ATTRIBUTION.md 'Closed-form confirmation' |
| `results/attribution_summary.json` | `diagnostics/attribution/results/attribution_summary.json` | every number quoted in the body of ATTRIBUTION.md | ATTRIBUTION.md 'Files written' |
| `results/closure_after_fix.json` | `diagnostics/attribution/results/closure_after_fix.json` | the v2 (b2)/(c2) before/after five-realisation closure table | CLOSURE.md 5 |
| `results/closure_summary.json` | `diagnostics/attribution/results/closure_summary.json` | the whole v2 closure campaign gathered into one object | CLOSURE.md 1-10 |
| `results/fix_named_defect_agn.h5` | `diagnostics/attribution/results/fix_named_defect_agn.h5` | the repaired matched scans (task 2), m1m2 and m1-only arms | ATTRIBUTION.md 'A2' |
| `results/fix_named_defect_agn.json` | `diagnostics/attribution/results/fix_named_defect_agn.json` | the repaired matched scans (task 2), m1m2 and m1-only arms | ATTRIBUTION.md 'A2' |
| `results/fix_named_defect_agn_m1.h5` | `diagnostics/attribution/results/fix_named_defect_agn_m1.h5` | the repaired matched scans (task 2), m1m2 and m1-only arms | ATTRIBUTION.md 'A2' |
| `results/fix_named_defect_agn_m1.json` | `diagnostics/attribution/results/fix_named_defect_agn_m1.json` | the repaired matched scans (task 2), m1m2 and m1-only arms | ATTRIBUTION.md 'A2' |
| `results/fix_named_defect_gal.h5` | `diagnostics/attribution/results/fix_named_defect_gal.h5` | the repaired matched scans (task 2), m1m2 and m1-only arms | ATTRIBUTION.md 'A2' |
| `results/fix_named_defect_gal.json` | `diagnostics/attribution/results/fix_named_defect_gal.json` | the repaired matched scans (task 2), m1m2 and m1-only arms | ATTRIBUTION.md 'A2' |
| `results/fix_named_defect_gal_m1.h5` | `diagnostics/attribution/results/fix_named_defect_gal_m1.h5` | the repaired matched scans (task 2), m1m2 and m1-only arms | ATTRIBUTION.md 'A2' |
| `results/fix_named_defect_gal_m1.json` | `diagnostics/attribution/results/fix_named_defect_gal_m1.json` | the repaired matched scans (task 2), m1m2 and m1-only arms | ATTRIBUTION.md 'A2' |
| `results/pe_corrected_events.json` | `diagnostics/attribution/results/pe_corrected_events.json` | provenance of the p_pe-corrected event copies | ATTRIBUTION.md 'A2' |
### `attribution/figs/`

| old path | new path | what it is | cited in |
| --- | --- | --- | --- |
| `figs/attr_attribution.pdf` | `diagnostics/attribution/figs/attr_attribution.pdf` | (a) where r lives, (b) the PE arms, (c) the posterior mass bias | ATTRIBUTION.md 'Files written' |
| `figs/attr_attribution.png` | `diagnostics/attribution/figs/attr_attribution.png` | (a) where r lives, (b) the PE arms, (c) the posterior mass bias | ATTRIBUTION.md 'Files written' |
| `figs/attr_oracle.pdf` | `diagnostics/attribution/figs/attr_oracle.pdf` | per-event validation, the arms, the closure ladder | ATTRIBUTION.md 'A3' |
| `figs/attr_oracle.png` | `diagnostics/attribution/figs/attr_oracle.png` | per-event validation, the arms, the closure ladder | ATTRIBUTION.md 'A3' |
| `figs/attr_sampler_ratio.pdf` | `diagnostics/attribution/figs/attr_sampler_ratio.pdf` | the exact log-ratio map, its validation, the prediction | ATTRIBUTION.md 'A1' |
| `figs/attr_sampler_ratio.png` | `diagnostics/attribution/figs/attr_sampler_ratio.png` | the exact log-ratio map, its validation, the prediction | ATTRIBUTION.md 'A1' |
| `figs/fig_before_after_fix.pdf` | `diagnostics/attribution/figs/fig_before_after_fix.pdf` | record vs reweighted H0 posteriors, both controls | ATTRIBUTION.md 'A2' |
| `figs/fig_before_after_fix.png` | `diagnostics/attribution/figs/fig_before_after_fix.png` | record vs reweighted H0 posteriors, both controls | ATTRIBUTION.md 'A2' |
| `figs/fig_closure_after_fix.pdf` | `diagnostics/attribution/figs/fig_closure_after_fix.pdf` | the v2 (b2)/(c2) before/after closure strip (superseded by figs/fig_closure_v3) | CLOSURE.md 5 |
| `figs/fig_closure_after_fix.png` | `diagnostics/attribution/figs/fig_closure_after_fix.png` | the v2 (b2)/(c2) before/after closure strip (superseded by figs/fig_closure_v3) | CLOSURE.md 5 |
| `figs/fig_selmu_oracle.pdf` | `diagnostics/attribution/figs/fig_selmu_oracle.pdf` | the selection oracle | CLOSURE.md 11 |
| `figs/fig_selmu_oracle.png` | `diagnostics/attribution/figs/fig_selmu_oracle.png` | the selection oracle | CLOSURE.md 11 |
| `figs/fig_sky_oracle_agn.pdf` | `diagnostics/attribution/figs/fig_sky_oracle_agn.pdf` | the exact host-galaxy sky oracle | CLOSURE.md 7 |
| `figs/fig_sky_oracle_agn.png` | `diagnostics/attribution/figs/fig_sky_oracle_agn.png` | the exact host-galaxy sky oracle | CLOSURE.md 7 |
| `figs/fig_sky_oracle_gal.pdf` | `diagnostics/attribution/figs/fig_sky_oracle_gal.pdf` | the exact host-galaxy sky oracle | CLOSURE.md 7 |
| `figs/fig_sky_oracle_gal.png` | `diagnostics/attribution/figs/fig_sky_oracle_gal.png` | the exact host-galaxy sky oracle | CLOSURE.md 7 |
### `attribution/logs/`

| old path | new path | what it is | cited in |
| --- | --- | --- | --- |
| `logs/attr_chieff.log` | `diagnostics/attribution/logs/attr_chieff.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_mass_pe_gal.log` | `diagnostics/attribution/logs/attr_mass_pe_gal.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_oracle_agn.log` | `diagnostics/attribution/logs/attr_oracle_agn.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_oracle_agn_conv_nm.log` | `diagnostics/attribution/logs/attr_oracle_agn_conv_nm.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_oracle_agn_conv_nz.log` | `diagnostics/attribution/logs/attr_oracle_agn_conv_nz.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_oracle_agn_conv_sh.log` | `diagnostics/attribution/logs/attr_oracle_agn_conv_sh.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_oracle_agn_conv_sky.log` | `diagnostics/attribution/logs/attr_oracle_agn_conv_sky.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_oracle_gal.log` | `diagnostics/attribution/logs/attr_oracle_gal.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_oracle_gal_conv_nm.log` | `diagnostics/attribution/logs/attr_oracle_gal_conv_nm.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_oracle_gal_conv_nz.log` | `diagnostics/attribution/logs/attr_oracle_gal_conv_nz.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_oracle_gal_conv_sh.log` | `diagnostics/attribution/logs/attr_oracle_gal_conv_sh.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_oracle_gal_conv_sky.log` | `diagnostics/attribution/logs/attr_oracle_gal_conv_sky.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_sampler_ratio.log` | `diagnostics/attribution/logs/attr_sampler_ratio.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_selmu_agn.log` | `diagnostics/attribution/logs/attr_selmu_agn.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_selmu_agn_regress.log` | `diagnostics/attribution/logs/attr_selmu_agn_regress.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_selmu_agn_s101.log` | `diagnostics/attribution/logs/attr_selmu_agn_s101.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_selmu_agn_s102.log` | `diagnostics/attribution/logs/attr_selmu_agn_s102.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_selmu_agn_s103.log` | `diagnostics/attribution/logs/attr_selmu_agn_s103.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_selmu_agn_s105.log` | `diagnostics/attribution/logs/attr_selmu_agn_s105.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_selmu_gal.log` | `diagnostics/attribution/logs/attr_selmu_gal.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_selmu_gal_s101.log` | `diagnostics/attribution/logs/attr_selmu_gal_s101.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_selmu_gal_s102.log` | `diagnostics/attribution/logs/attr_selmu_gal_s102.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_selmu_gal_s103.log` | `diagnostics/attribution/logs/attr_selmu_gal_s103.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_selmu_gal_s105.log` | `diagnostics/attribution/logs/attr_selmu_gal_s105.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_selmu_gencheck.log` | `diagnostics/attribution/logs/attr_selmu_gencheck.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_selmu_inj_agn_popuni.log` | `diagnostics/attribution/logs/attr_selmu_inj_agn_popuni.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_selmu_inj_agn_targeted.log` | `diagnostics/attribution/logs/attr_selmu_inj_agn_targeted.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_selmu_inj_gal_popuni.log` | `diagnostics/attribution/logs/attr_selmu_inj_gal_popuni.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_selmu_inj_gal_targeted.log` | `diagnostics/attribution/logs/attr_selmu_inj_gal_targeted.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_selmu_pdet.log` | `diagnostics/attribution/logs/attr_selmu_pdet.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_sky_oracle_agn.log` | `diagnostics/attribution/logs/attr_sky_oracle_agn.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_sky_oracle_agn_conv_ap4.log` | `diagnostics/attribution/logs/attr_sky_oracle_agn_conv_ap4.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_sky_oracle_agn_conv_ap8.log` | `diagnostics/attribution/logs/attr_sky_oracle_agn_conv_ap8.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_sky_oracle_agn_conv_base.log` | `diagnostics/attribution/logs/attr_sky_oracle_agn_conv_base.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_sky_oracle_agn_conv_nm.log` | `diagnostics/attribution/logs/attr_sky_oracle_agn_conv_nm.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_sky_oracle_agn_conv_nz.log` | `diagnostics/attribution/logs/attr_sky_oracle_agn_conv_nz.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_sky_oracle_agn_conv_sf5.log` | `diagnostics/attribution/logs/attr_sky_oracle_agn_conv_sf5.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_sky_oracle_agn_conv_sf7.log` | `diagnostics/attribution/logs/attr_sky_oracle_agn_conv_sf7.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_sky_oracle_agn_conv_shift.log` | `diagnostics/attribution/logs/attr_sky_oracle_agn_conv_shift.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_sky_oracle_agn_conv_sub4.log` | `diagnostics/attribution/logs/attr_sky_oracle_agn_conv_sub4.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_sky_oracle_agn_conv_sub6.log` | `diagnostics/attribution/logs/attr_sky_oracle_agn_conv_sub6.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_sky_oracle_agn_hostw.log` | `diagnostics/attribution/logs/attr_sky_oracle_agn_hostw.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_sky_oracle_gal.log` | `diagnostics/attribution/logs/attr_sky_oracle_gal.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_sky_oracle_gal_conv_ap4.log` | `diagnostics/attribution/logs/attr_sky_oracle_gal_conv_ap4.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_sky_oracle_gal_conv_base.log` | `diagnostics/attribution/logs/attr_sky_oracle_gal_conv_base.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_sky_oracle_gal_conv_nm.log` | `diagnostics/attribution/logs/attr_sky_oracle_gal_conv_nm.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_sky_oracle_gal_conv_nz.log` | `diagnostics/attribution/logs/attr_sky_oracle_gal_conv_nz.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_sky_oracle_gal_conv_sf5.log` | `diagnostics/attribution/logs/attr_sky_oracle_gal_conv_sf5.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_sky_oracle_gal_conv_sf7.log` | `diagnostics/attribution/logs/attr_sky_oracle_gal_conv_sf7.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_sky_oracle_gal_conv_shift.log` | `diagnostics/attribution/logs/attr_sky_oracle_gal_conv_shift.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_sky_oracle_gal_conv_sub4.log` | `diagnostics/attribution/logs/attr_sky_oracle_gal_conv_sub4.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_sky_oracle_gal_conv_sub6.log` | `diagnostics/attribution/logs/attr_sky_oracle_gal_conv_sub6.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_sky_oracle_gal_hostw.log` | `diagnostics/attribution/logs/attr_sky_oracle_gal_hostw.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_terms_agn_postfix.log` | `diagnostics/attribution/logs/attr_terms_agn_postfix.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_terms_gal.log` | `diagnostics/attribution/logs/attr_terms_gal.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/attr_terms_gal_postfix.log` | `diagnostics/attribution/logs/attr_terms_gal_postfix.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/build_skyindex_s100.log` | `diagnostics/attribution/logs/build_skyindex_s100.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/fix_named_defect_agn.log` | `diagnostics/attribution/logs/fix_named_defect_agn.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/fix_named_defect_agn_m1.log` | `diagnostics/attribution/logs/fix_named_defect_agn_m1.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/fix_named_defect_gal.log` | `diagnostics/attribution/logs/fix_named_defect_gal.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/fix_named_defect_gal_m1.log` | `diagnostics/attribution/logs/fix_named_defect_gal_m1.log` | stdout of the run that produced the like-named product | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/run_fix_scans.log` | `diagnostics/attribution/logs/run_fix_scans.log` | stdout of this stage's driver script | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/run_gal_conv_local.log` | `diagnostics/attribution/logs/run_gal_conv_local.log` | stdout of this stage's driver script | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/run_gal_conv_tail.log` | `diagnostics/attribution/logs/run_gal_conv_tail.log` | stdout of this stage's driver script | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/run_local_sweep.log` | `diagnostics/attribution/logs/run_local_sweep.log` | stdout of this stage's driver script | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/run_oracle.log` | `diagnostics/attribution/logs/run_oracle.log` | stdout of this stage's driver script | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/run_postfix_attr.log` | `diagnostics/attribution/logs/run_postfix_attr.log` | stdout of this stage's driver script | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/run_selmu_seeds.log` | `diagnostics/attribution/logs/run_selmu_seeds.log` | stdout of this stage's driver script | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/run_sky_oracle_prod.log` | `diagnostics/attribution/logs/run_sky_oracle_prod.log` | stdout of this stage's driver script | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/slurm_a1_agnaux_1059131.err` | `diagnostics/attribution/logs/slurm_a1_agnaux_1059131.err` | SLURM job log for this stage | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/slurm_a1_agnaux_1059131.out` | `diagnostics/attribution/logs/slurm_a1_agnaux_1059131.out` | SLURM job log for this stage | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/slurm_a1_agnaux_1059132.err` | `diagnostics/attribution/logs/slurm_a1_agnaux_1059132.err` | SLURM job log for this stage | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/slurm_a1_agnaux_1059132.out` | `diagnostics/attribution/logs/slurm_a1_agnaux_1059132.out` | SLURM job log for this stage | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/slurm_a1_agninj2_1059133.err` | `diagnostics/attribution/logs/slurm_a1_agninj2_1059133.err` | SLURM job log for this stage | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/slurm_a1_agninj2_1059133.out` | `diagnostics/attribution/logs/slurm_a1_agninj2_1059133.out` | SLURM job log for this stage | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/slurm_a1_galconv_1059125.err` | `diagnostics/attribution/logs/slurm_a1_galconv_1059125.err` | SLURM job log for this stage | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/slurm_a1_galconv_1059125.out` | `diagnostics/attribution/logs/slurm_a1_galconv_1059125.out` | SLURM job log for this stage | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/slurm_a1_galhostw_1059130.err` | `diagnostics/attribution/logs/slurm_a1_galhostw_1059130.err` | SLURM job log for this stage | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/slurm_a1_galhostw_1059130.out` | `diagnostics/attribution/logs/slurm_a1_galhostw_1059130.out` | SLURM job log for this stage | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/slurm_a1_gconv_1059134_0.err` | `diagnostics/attribution/logs/slurm_a1_gconv_1059134_0.err` | SLURM job log for this stage | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/slurm_a1_gconv_1059134_0.out` | `diagnostics/attribution/logs/slurm_a1_gconv_1059134_0.out` | SLURM job log for this stage | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/slurm_a1_gconv_1059134_1.err` | `diagnostics/attribution/logs/slurm_a1_gconv_1059134_1.err` | SLURM job log for this stage | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/slurm_a1_gconv_1059134_1.out` | `diagnostics/attribution/logs/slurm_a1_gconv_1059134_1.out` | SLURM job log for this stage | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/slurm_a1_gconv_1059134_2.err` | `diagnostics/attribution/logs/slurm_a1_gconv_1059134_2.err` | SLURM job log for this stage | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/slurm_a1_gconv_1059134_2.out` | `diagnostics/attribution/logs/slurm_a1_gconv_1059134_2.out` | SLURM job log for this stage | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/slurm_a1_gconv_1059134_3.err` | `diagnostics/attribution/logs/slurm_a1_gconv_1059134_3.err` | SLURM job log for this stage | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/slurm_a1_gconv_1059134_3.out` | `diagnostics/attribution/logs/slurm_a1_gconv_1059134_3.out` | SLURM job log for this stage | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/slurm_a1_gconv_1059134_4.err` | `diagnostics/attribution/logs/slurm_a1_gconv_1059134_4.err` | SLURM job log for this stage | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/slurm_a1_gconv_1059134_4.out` | `diagnostics/attribution/logs/slurm_a1_gconv_1059134_4.out` | SLURM job log for this stage | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/slurm_a1_gconv_1059134_5.err` | `diagnostics/attribution/logs/slurm_a1_gconv_1059134_5.err` | SLURM job log for this stage | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/slurm_a1_gconv_1059134_5.out` | `diagnostics/attribution/logs/slurm_a1_gconv_1059134_5.out` | SLURM job log for this stage | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/slurm_a1_postfix_aux_1059118.err` | `diagnostics/attribution/logs/slurm_a1_postfix_aux_1059118.err` | SLURM job log for this stage | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/slurm_a1_postfix_aux_1059118.out` | `diagnostics/attribution/logs/slurm_a1_postfix_aux_1059118.out` | SLURM job log for this stage | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/slurm_a1_postfix_aux_1059119.err` | `diagnostics/attribution/logs/slurm_a1_postfix_aux_1059119.err` | SLURM job log for this stage | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/slurm_a1_postfix_aux_1059119.out` | `diagnostics/attribution/logs/slurm_a1_postfix_aux_1059119.out` | SLURM job log for this stage | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/v3_mcerr_gal_popuni.log` | `diagnostics/attribution/logs/v3_mcerr_gal_popuni.log` | stdout of the v3-era run of this stage's step | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/v3_pdet.log` | `diagnostics/attribution/logs/v3_pdet.log` | stdout of the v3-era run of this stage's step | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/v3_selmu_agn_s100.log` | `diagnostics/attribution/logs/v3_selmu_agn_s100.log` | stdout of the v3-era run of this stage's step | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/v3_selmu_gal_s100.log` | `diagnostics/attribution/logs/v3_selmu_gal_s100.log` | stdout of the v3-era run of this stage's step | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
| `logs/v3_selmu_inj_gal_targeted.log` | `diagnostics/attribution/logs/v3_selmu_inj_gal_targeted.log` | stdout of the v3-era run of this stage's step | ATTRIBUTION.md (all sections) and CLOSURE.md 5-7 and 11-13 |
### `attribution/data_derived/`

| old path | new path | what it is | cited in |
| --- | --- | --- | --- |
| `data_derived/events_agn_hosted_pefix_m1.h5` | `diagnostics/attribution/data_derived/events_agn_hosted_pefix_m1.h5` | p_pe-corrected copy of the matched event file (task 2 input; working/data untouched) | ATTRIBUTION.md 'A2' |
| `data_derived/events_agn_hosted_pefix_m1m2.h5` | `diagnostics/attribution/data_derived/events_agn_hosted_pefix_m1m2.h5` | p_pe-corrected copy of the matched event file (task 2 input; working/data untouched) | ATTRIBUTION.md 'A2' |
| `data_derived/events_gal_hosted_pefix_m1.h5` | `diagnostics/attribution/data_derived/events_gal_hosted_pefix_m1.h5` | p_pe-corrected copy of the matched event file (task 2 input; working/data untouched) | ATTRIBUTION.md 'A2' |
| `data_derived/events_gal_hosted_pefix_m1m2.h5` | `diagnostics/attribution/data_derived/events_gal_hosted_pefix_m1m2.h5` | p_pe-corrected copy of the matched event file (task 2 input; working/data untouched) | ATTRIBUTION.md 'A2' |

## `diagnostics/endgame/`

The endgame: the (A - B) / (C - A) split, the truncation / exchangeability audit
of the event proposal stream, the declared-photo-z-kernel scan, and the v3 pilot
gate that authorised the redesign.  `abc_*` products carry A, B, C per
realisation; the `*_mega*` products carry A on the redraw campaigns.

Cited by: CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate).

### `endgame/scripts/`

| old path | new path | what it is | cited in |
| --- | --- | --- | --- |
| `scripts/abc_summary.py` | `diagnostics/endgame/scripts/abc_summary.py` | the five-realisation split summary | CLOSURE.md 15.3 |
| `scripts/attr_abc_split.py` | `diagnostics/endgame/scripts/attr_abc_split.py` | A (score at the events' TRUE parameters), B (injection estimate), C (posterior-averaged) and the (A-B)/(C-A) split | CLOSURE.md 15.3, 16.2 |
| `scripts/endgame_summary.py` | `diagnostics/endgame/scripts/endgame_summary.py` | one JSON for section 15 | CLOSURE.md 15 |
| `scripts/make_dz_survey.py` | `diagnostics/endgame/scripts/make_dz_survey.py` | a survey block identical to the record's except for the DECLARED photo-z width | CLOSURE.md 15.4 |
| `scripts/regen_events_notrunc.py` | `diagnostics/endgame/scripts/regen_events_notrunc.py` | replays stage_events' proposal loop with [:N_EVENTS] lifted; --verify audits bitwise, --replicas redraws | CLOSURE.md 15.2, 16.2 |
| `scripts/run_abc.sh` | `diagnostics/endgame/scripts/run_abc.sh` | the (A-B)/(C-A) split on every post-fix realisation, both matched configurations | CLOSURE.md 15.3 |
| `scripts/run_dzscan.sh` | `diagnostics/endgame/scripts/run_dzscan.sh` | the declared-photo-z-kernel scan | CLOSURE.md 15.4 |
| `scripts/run_endgame_tail.sh` | `diagnostics/endgame/scripts/run_endgame_tail.sh` | the oracle no-override regression + the GAL half of the dz scan | CLOSURE.md 15.4 |
| `scripts/run_s103_replay_A.sh` | `diagnostics/endgame/scripts/run_s103_replay_A.sh` | (A - B) on a second catalog (seed 103) | CLOSURE.md 15.4 |
| `scripts/run_v3_abmega.sh` | `diagnostics/endgame/scripts/run_v3_abmega.sh` | the v3 (A - B) redraw campaign on a fixed catalog | CLOSURE.md 16.2 |
| `scripts/run_v3_pilot.sh` | `diagnostics/endgame/scripts/run_v3_pilot.sh` | the v3 gate: P_det, the exact oracle, A/B/C on both matched controls | CLOSURE.md 16.2 |
| `scripts/submit_abc.sbatch` | `diagnostics/endgame/scripts/submit_abc.sbatch` | HENON-GPU submission of the (A-B)/(C-A) split | CLOSURE.md 15.3 |
| `scripts/trunc_diag.py` | `diagnostics/endgame/scripts/trunc_diag.py` | proposal-stream exchangeability: gap law, autocorrelations, slot/rank correlations | CLOSURE.md 15.2 |
| `scripts/v3_pilot_summary.py` | `diagnostics/endgame/scripts/v3_pilot_summary.py` | the v3 gate table and verdict | CLOSURE.md 16.2 |
### `endgame/results/`

| old path | new path | what it is | cited in |
| --- | --- | --- | --- |
| `results/abc_agn_mega.json` | `diagnostics/endgame/results/abc_agn_mega.json` | A on 1500 replays, seed 100 | CLOSURE.md 15.2 |
| `results/abc_agn_mega.npz` | `diagnostics/endgame/results/abc_agn_mega.npz` | A on 1500 replays, seed 100 | CLOSURE.md 15.2 |
| `results/abc_agn_mega_dzx0p5.json` | `diagnostics/endgame/results/abc_agn_mega_dzx0p5.json` | A on the replays under a rescaled declared photo-z kernel | CLOSURE.md 15.4 |
| `results/abc_agn_mega_dzx0p5.npz` | `diagnostics/endgame/results/abc_agn_mega_dzx0p5.npz` | A on the replays under a rescaled declared photo-z kernel | CLOSURE.md 15.4 |
| `results/abc_agn_mega_dzx2.json` | `diagnostics/endgame/results/abc_agn_mega_dzx2.json` | A on the replays under a rescaled declared photo-z kernel | CLOSURE.md 15.4 |
| `results/abc_agn_mega_dzx2.npz` | `diagnostics/endgame/results/abc_agn_mega_dzx2.npz` | A on the replays under a rescaled declared photo-z kernel | CLOSURE.md 15.4 |
| `results/abc_agn_mega_dzx3.json` | `diagnostics/endgame/results/abc_agn_mega_dzx3.json` | A on the replays under a rescaled declared photo-z kernel | CLOSURE.md 15.4 |
| `results/abc_agn_mega_dzx3.npz` | `diagnostics/endgame/results/abc_agn_mega_dzx3.npz` | A on the replays under a rescaled declared photo-z kernel | CLOSURE.md 15.4 |
| `results/abc_agn_mega_s103.json` | `diagnostics/endgame/results/abc_agn_mega_s103.json` | A on 500 replays, seed 103 | CLOSURE.md 15.4 |
| `results/abc_agn_mega_s103.npz` | `diagnostics/endgame/results/abc_agn_mega_s103.npz` | A on 500 replays, seed 103 | CLOSURE.md 15.4 |
| `results/abc_agn_s100.json` | `diagnostics/endgame/results/abc_agn_s100.json` | the (A-B)/(C-A) split on one realisation | CLOSURE.md 15.3 |
| `results/abc_agn_s100.npz` | `diagnostics/endgame/results/abc_agn_s100.npz` | the (A-B)/(C-A) split on one realisation | CLOSURE.md 15.3 |
| `results/abc_agn_s101.json` | `diagnostics/endgame/results/abc_agn_s101.json` | the (A-B)/(C-A) split on one realisation | CLOSURE.md 15.3 |
| `results/abc_agn_s101.npz` | `diagnostics/endgame/results/abc_agn_s101.npz` | the (A-B)/(C-A) split on one realisation | CLOSURE.md 15.3 |
| `results/abc_agn_s102.json` | `diagnostics/endgame/results/abc_agn_s102.json` | the (A-B)/(C-A) split on one realisation | CLOSURE.md 15.3 |
| `results/abc_agn_s102.npz` | `diagnostics/endgame/results/abc_agn_s102.npz` | the (A-B)/(C-A) split on one realisation | CLOSURE.md 15.3 |
| `results/abc_agn_s103.json` | `diagnostics/endgame/results/abc_agn_s103.json` | the (A-B)/(C-A) split on one realisation | CLOSURE.md 15.3 |
| `results/abc_agn_s103.npz` | `diagnostics/endgame/results/abc_agn_s103.npz` | the (A-B)/(C-A) split on one realisation | CLOSURE.md 15.3 |
| `results/abc_agn_s105.json` | `diagnostics/endgame/results/abc_agn_s105.json` | the (A-B)/(C-A) split on one realisation | CLOSURE.md 15.3 |
| `results/abc_agn_s105.npz` | `diagnostics/endgame/results/abc_agn_s105.npz` | the (A-B)/(C-A) split on one realisation | CLOSURE.md 15.3 |
| `results/abc_agn_v3_mega.json` | `diagnostics/endgame/results/abc_agn_v3_mega.json` | A on the v3 redraw campaign (1500 replays, seed 100) | CLOSURE.md 16.2 |
| `results/abc_agn_v3_mega.npz` | `diagnostics/endgame/results/abc_agn_v3_mega.npz` | A on the v3 redraw campaign (1500 replays, seed 100) | CLOSURE.md 16.2 |
| `results/abc_agn_v3_s100.json` | `diagnostics/endgame/results/abc_agn_v3_s100.json` | the v3 pilot's A/B/C split | CLOSURE.md 16.2 |
| `results/abc_agn_v3_s100.npz` | `diagnostics/endgame/results/abc_agn_v3_s100.npz` | the v3 pilot's A/B/C split | CLOSURE.md 16.2 |
| `results/abc_gal_mega.json` | `diagnostics/endgame/results/abc_gal_mega.json` | A on 1500 replays, seed 100 | CLOSURE.md 15.2 |
| `results/abc_gal_mega.npz` | `diagnostics/endgame/results/abc_gal_mega.npz` | A on 1500 replays, seed 100 | CLOSURE.md 15.2 |
| `results/abc_gal_mega_dzx2.json` | `diagnostics/endgame/results/abc_gal_mega_dzx2.json` | A on the replays under a rescaled declared photo-z kernel | CLOSURE.md 15.4 |
| `results/abc_gal_mega_dzx2.npz` | `diagnostics/endgame/results/abc_gal_mega_dzx2.npz` | A on the replays under a rescaled declared photo-z kernel | CLOSURE.md 15.4 |
| `results/abc_gal_mega_s103.json` | `diagnostics/endgame/results/abc_gal_mega_s103.json` | A on 500 replays, seed 103 | CLOSURE.md 15.4 |
| `results/abc_gal_mega_s103.npz` | `diagnostics/endgame/results/abc_gal_mega_s103.npz` | A on 500 replays, seed 103 | CLOSURE.md 15.4 |
| `results/abc_gal_s100.json` | `diagnostics/endgame/results/abc_gal_s100.json` | the (A-B)/(C-A) split on one realisation | CLOSURE.md 15.3 |
| `results/abc_gal_s100.npz` | `diagnostics/endgame/results/abc_gal_s100.npz` | the (A-B)/(C-A) split on one realisation | CLOSURE.md 15.3 |
| `results/abc_gal_s101.json` | `diagnostics/endgame/results/abc_gal_s101.json` | the (A-B)/(C-A) split on one realisation | CLOSURE.md 15.3 |
| `results/abc_gal_s101.npz` | `diagnostics/endgame/results/abc_gal_s101.npz` | the (A-B)/(C-A) split on one realisation | CLOSURE.md 15.3 |
| `results/abc_gal_s102.json` | `diagnostics/endgame/results/abc_gal_s102.json` | the (A-B)/(C-A) split on one realisation | CLOSURE.md 15.3 |
| `results/abc_gal_s102.npz` | `diagnostics/endgame/results/abc_gal_s102.npz` | the (A-B)/(C-A) split on one realisation | CLOSURE.md 15.3 |
| `results/abc_gal_s103.json` | `diagnostics/endgame/results/abc_gal_s103.json` | the (A-B)/(C-A) split on one realisation | CLOSURE.md 15.3 |
| `results/abc_gal_s103.npz` | `diagnostics/endgame/results/abc_gal_s103.npz` | the (A-B)/(C-A) split on one realisation | CLOSURE.md 15.3 |
| `results/abc_gal_s105.json` | `diagnostics/endgame/results/abc_gal_s105.json` | the (A-B)/(C-A) split on one realisation | CLOSURE.md 15.3 |
| `results/abc_gal_s105.npz` | `diagnostics/endgame/results/abc_gal_s105.npz` | the (A-B)/(C-A) split on one realisation | CLOSURE.md 15.3 |
| `results/abc_gal_v3_mega.json` | `diagnostics/endgame/results/abc_gal_v3_mega.json` | A on the v3 redraw campaign (1500 replays, seed 100) | CLOSURE.md 16.2 |
| `results/abc_gal_v3_mega.npz` | `diagnostics/endgame/results/abc_gal_v3_mega.npz` | A on the v3 redraw campaign (1500 replays, seed 100) | CLOSURE.md 16.2 |
| `results/abc_gal_v3_s100.json` | `diagnostics/endgame/results/abc_gal_v3_s100.json` | the v3 pilot's A/B/C split | CLOSURE.md 16.2 |
| `results/abc_gal_v3_s100.npz` | `diagnostics/endgame/results/abc_gal_v3_s100.npz` | the v3 pilot's A/B/C split | CLOSURE.md 16.2 |
| `results/abc_summary.json` | `diagnostics/endgame/results/abc_summary.json` | the five-realisation split, gathered | CLOSURE.md 15.3 |
| `results/attr_selmu_agn_dzx0p5.json` | `diagnostics/endgame/results/attr_selmu_agn_dzx0p5.json` | the EXACT selection oracle B on a rescaled-kernel survey block (dz scan) | CLOSURE.md 15.4 |
| `results/attr_selmu_agn_dzx0p5.npz` | `diagnostics/endgame/results/attr_selmu_agn_dzx0p5.npz` | the EXACT selection oracle B on a rescaled-kernel survey block (dz scan) | CLOSURE.md 15.4 |
| `results/attr_selmu_agn_dzx2.json` | `diagnostics/endgame/results/attr_selmu_agn_dzx2.json` | the EXACT selection oracle B on a rescaled-kernel survey block (dz scan) | CLOSURE.md 15.4 |
| `results/attr_selmu_agn_dzx2.npz` | `diagnostics/endgame/results/attr_selmu_agn_dzx2.npz` | the EXACT selection oracle B on a rescaled-kernel survey block (dz scan) | CLOSURE.md 15.4 |
| `results/attr_selmu_agn_dzx3.json` | `diagnostics/endgame/results/attr_selmu_agn_dzx3.json` | the EXACT selection oracle B on a rescaled-kernel survey block (dz scan) | CLOSURE.md 15.4 |
| `results/attr_selmu_agn_dzx3.npz` | `diagnostics/endgame/results/attr_selmu_agn_dzx3.npz` | the EXACT selection oracle B on a rescaled-kernel survey block (dz scan) | CLOSURE.md 15.4 |
| `results/attr_selmu_gal_dzx2.json` | `diagnostics/endgame/results/attr_selmu_gal_dzx2.json` | the EXACT selection oracle B on a rescaled-kernel survey block (dz scan) | CLOSURE.md 15.4 |
| `results/attr_selmu_gal_dzx2.npz` | `diagnostics/endgame/results/attr_selmu_gal_dzx2.npz` | the EXACT selection oracle B on a rescaled-kernel survey block (dz scan) | CLOSURE.md 15.4 |
| `results/endgame_summary.json` | `diagnostics/endgame/results/endgame_summary.json` | every number quoted in section 15 | CLOSURE.md 15 |
| `results/trunc_diag.json` | `diagnostics/endgame/results/trunc_diag.json` | the proposal-stream exchangeability audit | CLOSURE.md 15.2 |
| `results/v3_pilot_summary.json` | `diagnostics/endgame/results/v3_pilot_summary.json` | the v3 gate verdict | CLOSURE.md 16.2 |
### `endgame/logs/`

| old path | new path | what it is | cited in |
| --- | --- | --- | --- |
| `logs/abc_agn_mega_dzx0p5.log` | `diagnostics/endgame/logs/abc_agn_mega_dzx0p5.log` | stdout of the run that produced the like-named product | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/abc_agn_mega_dzx2.log` | `diagnostics/endgame/logs/abc_agn_mega_dzx2.log` | stdout of the run that produced the like-named product | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/abc_agn_mega_dzx3.log` | `diagnostics/endgame/logs/abc_agn_mega_dzx3.log` | stdout of the run that produced the like-named product | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/abc_agn_mega_s103.log` | `diagnostics/endgame/logs/abc_agn_mega_s103.log` | stdout of the run that produced the like-named product | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/abc_agn_s101.log` | `diagnostics/endgame/logs/abc_agn_s101.log` | stdout of the run that produced the like-named product | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/abc_agn_s102.log` | `diagnostics/endgame/logs/abc_agn_s102.log` | stdout of the run that produced the like-named product | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/abc_agn_s103.log` | `diagnostics/endgame/logs/abc_agn_s103.log` | stdout of the run that produced the like-named product | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/abc_agn_s105.log` | `diagnostics/endgame/logs/abc_agn_s105.log` | stdout of the run that produced the like-named product | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/abc_gal_mega.log` | `diagnostics/endgame/logs/abc_gal_mega.log` | stdout of the run that produced the like-named product | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/abc_gal_mega_dzx2.log` | `diagnostics/endgame/logs/abc_gal_mega_dzx2.log` | stdout of the run that produced the like-named product | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/abc_gal_mega_s103.log` | `diagnostics/endgame/logs/abc_gal_mega_s103.log` | stdout of the run that produced the like-named product | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/abc_gal_s100.log` | `diagnostics/endgame/logs/abc_gal_s100.log` | stdout of the run that produced the like-named product | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/abc_gal_s101.log` | `diagnostics/endgame/logs/abc_gal_s101.log` | stdout of the run that produced the like-named product | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/abc_gal_s102.log` | `diagnostics/endgame/logs/abc_gal_s102.log` | stdout of the run that produced the like-named product | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/abc_gal_s103.log` | `diagnostics/endgame/logs/abc_gal_s103.log` | stdout of the run that produced the like-named product | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/abc_gal_s105.log` | `diagnostics/endgame/logs/abc_gal_s105.log` | stdout of the run that produced the like-named product | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/attr_selmu_agn_dzx0p5.log` | `diagnostics/endgame/logs/attr_selmu_agn_dzx0p5.log` | stdout of the run that produced the like-named product | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/attr_selmu_agn_dzx2.log` | `diagnostics/endgame/logs/attr_selmu_agn_dzx2.log` | stdout of the run that produced the like-named product | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/attr_selmu_agn_dzx3.log` | `diagnostics/endgame/logs/attr_selmu_agn_dzx3.log` | stdout of the run that produced the like-named product | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/attr_selmu_gal_dzx2.log` | `diagnostics/endgame/logs/attr_selmu_gal_dzx2.log` | stdout of the run that produced the like-named product | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/regen_replicas_s100.log` | `diagnostics/endgame/logs/regen_replicas_s100.log` | stdout of the run that produced the like-named product | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/regen_replicas_s103.log` | `diagnostics/endgame/logs/regen_replicas_s103.log` | stdout of the run that produced the like-named product | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/run_abc.log` | `diagnostics/endgame/logs/run_abc.log` | stdout of this stage's driver script | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/run_dzscan_agn.log` | `diagnostics/endgame/logs/run_dzscan_agn.log` | stdout of this stage's driver script | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/run_endgame_tail.log` | `diagnostics/endgame/logs/run_endgame_tail.log` | stdout of this stage's driver script | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/run_s103_replay_A.log` | `diagnostics/endgame/logs/run_s103_replay_A.log` | stdout of this stage's driver script | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/slurm_a1_abc_1059144.err` | `diagnostics/endgame/logs/slurm_a1_abc_1059144.err` | SLURM job log for this stage | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/slurm_a1_abc_1059144.out` | `diagnostics/endgame/logs/slurm_a1_abc_1059144.out` | SLURM job log for this stage | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/slurm_a1_abc_1059145.err` | `diagnostics/endgame/logs/slurm_a1_abc_1059145.err` | SLURM job log for this stage | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/slurm_a1_abc_1059145.out` | `diagnostics/endgame/logs/slurm_a1_abc_1059145.out` | SLURM job log for this stage | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/slurm_a1_abc_1059146.err` | `diagnostics/endgame/logs/slurm_a1_abc_1059146.err` | SLURM job log for this stage | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/slurm_a1_abc_1059146.out` | `diagnostics/endgame/logs/slurm_a1_abc_1059146.out` | SLURM job log for this stage | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/v3_abc_agn_s100.log` | `diagnostics/endgame/logs/v3_abc_agn_s100.log` | stdout of the v3-era run of this stage's step | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/v3_abc_gal_s100.log` | `diagnostics/endgame/logs/v3_abc_gal_s100.log` | stdout of the v3-era run of this stage's step | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/v3_abc_mega_agn_s100.log` | `diagnostics/endgame/logs/v3_abc_mega_agn_s100.log` | stdout of the v3-era run of this stage's step | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/v3_abc_mega_gal_s100.log` | `diagnostics/endgame/logs/v3_abc_mega_gal_s100.log` | stdout of the v3-era run of this stage's step | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/v3_regen_replicas_s100.log` | `diagnostics/endgame/logs/v3_regen_replicas_s100.log` | stdout of the v3-era run of this stage's step | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |
| `logs/v3_regen_verify_s100.log` | `diagnostics/endgame/logs/v3_regen_verify_s100.log` | stdout of the v3-era run of this stage's step | CLOSURE.md 15 (the endgame) and 16.2 (the v3 pilot gate) |

