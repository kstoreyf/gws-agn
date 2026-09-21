# Analysis 10 — a mass mark on top of the spin mark

> **STATUS: STOPPED at specification §9, before generation** (2026-09-21). The
> registered mark is inadmissible under the production parameterisation:
> \(\lambda_{\rm peak}^{\rm fid} = 0.90\), so \(0.90 + 0.15 = 1.05 > 1\). No mock,
> no injections and no likelihood were run. See `REPORT.md` for the owner
> decision, `GATES.md` for the ledger, `STATE.md` for the verified facts.

## Question

Analyses 8 and 9 gave the AGN branch **one** intrinsic mark — an effective-spin
offset \(\Delta\mu_\chi = +0.10\) — and showed that the GAL/AGN label selects a
distinct \(\chi_{\rm eff}\) population as well as a distinct spatial tracer.
Analysis 10 asks whether the marked multi-tracer model survives a **second,
independent mark in a different physical channel**, the mass function:

\[
  \lambda_{\rm peak}^{\rm AGN} = \lambda_{\rm peak}^{\rm GAL} + \Delta\lambda_{\rm peak},
  \qquad \Delta\lambda_{\rm peak}^{\rm plant} = +0.15,
\]

where \(\lambda_{\rm peak}\) is the Gaussian-peak mixture fraction of the
production `powerlaw+peak` implementation, alongside the existing
\(\Delta\mu_\chi = +0.10\). Three questions:

1. **Are two marks separable?** With \(H_0\) fixed at the planted 67.74, are
   \(f_{\rm AGN}\), \(\Delta\mu_\chi\) and \(\Delta\lambda_{\rm peak}\) recovered
   jointly, and what is \(\rho(\Delta\mu_\chi, \Delta\lambda_{\rm peak})\)?
2. **Which channel carries the label?** Ablation, measured rather than asserted:
   what each mark contributes to the identification of \(f_{\rm AGN}\) on its own
   and in combination.
3. **What does a mass mark do to selection?** The detection rule is
   \(\rho_{\rm obs} \ge 8\) with \(\rho_{\rm opt} \propto \mathcal{M}_{\rm det}^{5/6}/d_L\),
   so a mass mark moves detectability branch by branch — unlike the spin mark,
   which does not enter \(\rho_{\rm opt}\) at all.

## Arms — \(H_0\) fixed first

| arm | free | cells | what it isolates |
|---|---|---|---|
| **A10-S** | \(f_{\rm AGN}\) | 41 | spatial only; both marks off |
| **A10-χ** | \(f_{\rm AGN}, \Delta\mu_\chi\) | 41 × 61 | the Analysis-8 arm J, re-run on the two-mark mock |
| **A10-M** | \(f_{\rm AGN}, \Delta\lambda_{\rm peak}\) | 41 × \(N_\lambda\) | the mass mark alone |
| **A10-J** | \(f_{\rm AGN}, \Delta\mu_\chi, \Delta\lambda_{\rm peak}\) | 41 × 61 × \(N_\lambda\) | the joint measurement |

\(H_0\) stays pinned at 67.74 in all four. **Releasing \(H_0\) is a separate,
gated step** (the Analysis-9 move applied to the two-mark model), taken only
after the fixed-\(H_0\) owner gate, and never in the same campaign.

## Scope locks

- **Seed 100 only**, one realisation.
- **Complete catalogs only.** No incompleteness, no completion path.
- **No GP/HSGP**, anywhere.
- **No free common hyperparameters.** The twelve base population parameters stay
  pinned by name at the `powerlaw+peak` fiducial; only the per-catalog
  \(c_2\) copies of the two marked slots are ever free.
- **No branch-dependent mass ratio.** \(\beta\) is common.
- **No additional mass parameters.** Exactly ONE mass coordinate is marked. In
  particular \(\mu_{\rm G}\), \(\sigma_{\rm G}\), \(\alpha_{\rm PL}\),
  \(m_{\min}\), \(m_{\max}\) and the taper widths stay common and pinned.
- \(\Omega_{m,0} = 0.3075\); \(H_0 = 67.74\) until the release gate.
- `darksirens` is not modified; `working/data/generate_dataset.py` is not
  modified; nothing is pushed.
- `working/data/seed100/**`, `analysis_8_marked_multitracer_H0_fagn/**` and
  `analysis_9_marked_multitracer_H0_fagn/**` are **read-only inputs**. Every
  write lands under this directory.
- No likelihood is reimplemented: the Analysis-8 builder is imported.

## Pre-registration — the mark check

The specification registers one check that runs **before any generation**:

> Verify \(0 < \lambda_{\rm peak}^{\rm fid} + 0.15 < 1\). If this is not true
> under the actual production parameterization, STOP and report the real
> fiducial value. Do not silently alter the registered mark.

`scripts/check_mass_mark_feasibility.py` is that check. It is CPU-only, loads no
survey and no likelihood, runs no sampler, and writes one file,
`diagnostics/a10_mass_mark_feasibility.json`. It reads the twelve parameter
slots off `model.param_specs`, evaluates the stick-breaking map and its inverse
as darksirens defines them, establishes the composition order twice
independently (the curated registry entry **and** a numerical experiment on the
mass density), and reports the check in both the human-readable peak fraction
and the sampled coordinate `v1_c2`.

**It fired.** \(\lambda_{\rm peak}^{\rm fid} = 0.90\) — the sampled coordinate is
the *power-law* weight \(v_1 = 0.10\), and \(\lambda_{\rm peak} = 1 - v_1\) — so
the planted AGN branch would need \(\lambda_{\rm peak}^{\rm AGN} = 1.05\), i.e.
\(v_{1,c2} = -0.05\), outside the \([0, 1]\) prior. The mark was not altered and
nothing downstream was run.

## Files

- `STATE.md` — the stop, the verified facts, what was not done, provenance.
- `GATES.md` — the ledger: Gate 0 PASS, Gate M FAIL, everything after NOT RUN.
- `REPORT.md` — the owner-facing report and the options table.
- `scripts/check_mass_mark_feasibility.py` — the pre-registered check.
- `diagnostics/a10_mass_mark_feasibility.json` — its output, the evidence of
  record.
- `results/`, `figs/` — empty; nothing was measured and nothing was drawn.
