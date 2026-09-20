# Analysis 9 — marked multi-tracer inference with \(H_0\) free

## Question

Analysis 8 measured \((f_{\rm AGN}, \Delta\mu_\chi)\) from one marked seed-100
realisation with \(H_0\) pinned at the planted 67.74. That pin is the strongest
assumption in the result: the spatial channel identifies the host label through
the redshift-space structure of two tracers, and that structure moves with
\(H_0\). Analysis 9 frees it and asks three questions this one realisation can
answer.

1. **Does the joint recovery survive?** With \(H_0\) sampled, are \(f_{\rm AGN}\)
   and the AGN-branch spin offset still recovered, against both the planted and
   the realised truth?
2. **What does marginalising over \(H_0\) cost?** Measured, not asserted: the
   interval widths and the induced correlations
   \(\rho(H_0, f_{\rm AGN})\) and \(\rho(H_0, \Delta\mu_\chi)\), against
   Analysis 8's fixed-\(H_0\) widths (68%: 0.0921 in \(f\), 0.0394 in
   \(\Delta\mu_\chi\)).
3. **Does the intrinsic mark change what the data say about \(H_0\)?** The same
   realisation, the same catalogs and the same selection were measured without
   the mark by Analysis 2 (\(H_0 = 69.22^{+1.0}_{-1.0}\)), so this is a
   differential statement, not an absolute one.

## Free parameters — exactly three

| coordinate | meaning |
|---|---|
| `H0` | the Hubble constant. Sampled here; Analysis 8 held it at 67.74. |
| `fcat_2` | \(f_{\rm AGN}\), the AGN mixture weight (stick-breaking at \(K=2\)). |
| `mu_chi_c2` | the AGN branch's **absolute** effective-spin mean. It equals \(\Delta\mu_\chi\) **only** because \(\mu_{\chi,{\rm GAL}}\) is pinned at the powerlaw+peak fiducial 0.0. Both spellings appear in every output. |

Everything else is the Analysis-8 configuration: \(\Omega_{m,0} = 0.3075\),
common mass model, common mass ratio, common spin width, common redshift rate,
all twelve base population parameters pinned by name through
`fixed_parameter_values`, `per_catalog_pop_params=('mu_chi_c2',)`,
`log10n0 = log10n0_c2 = -24`, hard `selection_neff_guard`,
`max_likelihood_variance = 1e6`, `kde_window = 4096`, `kde_window_nsigma = 8`,
`sel_batch_size = 50000`, `pe_event_block = 25`, `catalog_sky_weighting = field`,
`universe_model = dark_sirens`.

## Scope limits

Analysis 9 is Analysis 8 with \(H_0\) freed, and nothing else.

- **Seed 100 only, one realisation.** No data, no injections, no new mock are
  generated; the marked Gate-B mock and the existing targeted injection set are
  reused exactly.
- **No population degrees of freedom beyond the three above.** No mass mark, no
  incompleteness, no GP/HSGP, no extra intrinsic parameters.
- `darksirens` is not modified, `generate_dataset.py` is not modified, nothing is
  pushed.
- `working/data/seed100/**` and
  `working/analyses/analysis_8_marked_multitracer_H0_fagn/**` are **read-only
  inputs**. Every write this analysis makes lands under this directory (the
  driver refuses to write anywhere else).
- No likelihood is reimplemented here. The Analysis-8 builder is imported.

## Data, reused exactly

    events     working/data/seed100/events/events_marked_dmu0p10.h5
    surveys    working/data/seed100/surveys/survey_{gal,agn}_complete_ns32.h5
    selection  working/data/seed100/injections/injections_targeted.h5

## Truths — carry both, everywhere

| quantity | planted | realised |
|---|---|---|
| \(H_0\) | 67.74 | seed 100's own draw sits high: Analysis 2 recovered 69.217, 68% [68.249, 70.191], on the **unmarked** data. A property of this realisation, not a bias — quote \(H_0\) differentially as well. |
| \(f_{\rm AGN}\) | 0.30 | 0.295 |
| \(\Delta\mu_\chi\) | +0.100000 | +0.111924 ± 0.006815 (of which +0.011924 predates the mark) |

## Files

- `STATE.md` — provenance, the registered grid and its sizing evidence, the plan.
- `GATES.md` — the pass/fail ledger.
- `scripts/a9_scan.py` — the driver (staged, resumable); `scripts/env_a9.sh` and
  the two `.sbatch` harnesses run it on RITA.
- `results/`, `diagnostics/`, `figs/` — outputs, written only by the driver.

Compute runs on RITA via SLURM (`--partition=RITA-GPU --qos=rita
--account=phy220048p --gres=gpu:a100-80:1`). The local H100 stays free; the
driver refuses to run a GPU stage anywhere but RITA.
