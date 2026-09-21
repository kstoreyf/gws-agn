# Analysis 10 — a mass mark on top of the spin mark

> **STATUS: owner decision recorded (2026-09-21).** The originally registered
> peak-fraction mark \(\Delta\lambda_{\rm peak} = +0.15\) was rejected before
> generation — it lies outside the mixture simplex — and the owner replaced it
> with a Gaussian-peak **location** mark, \(\Delta\mu_{\rm G} = +5\,M_\odot\).
> Nothing has been run under the new mark: the generator extension, the mock and
> the selection-support gate are in progress. The rejected mark's record is kept
> as provenance in `STATE.md`, `GATES.md` and `REPORT.md`.

## Question

Analyses 8 and 9 gave the AGN branch **one** intrinsic mark — an effective-spin
offset \(\Delta\mu_\chi = +0.10\) — and showed that the GAL/AGN label selects a
distinct \(\chi_{\rm eff}\) population as well as a distinct spatial tracer.
Analysis 10 asks whether the marked multi-tracer model survives a **second,
independent mark in a different physical channel**, the source-mass function.

The mark is a difference in the **location of the Gaussian component** of the
PL+G source-mass population:

\[
  \mu_{\rm G}^{\rm GAL} = 35\,M_\odot, \qquad
  \mu_{\rm G}^{\rm AGN} = 40\,M_\odot, \qquad
  \Delta\mu_{\rm G} \equiv \mu_{\rm G}^{\rm AGN} - \mu_{\rm G}^{\rm GAL}
  = +5\,M_\odot,
\]

alongside the unchanged \(\Delta\mu_\chi = +0.10\). The peak fraction is
\(\lambda_{\rm peak} = 0.90\) in **both** branches; the power-law component
(slope, mass limits, tapers) and the mass-ratio distribution \(q\) are
**identical** in both branches. Only the peak location moves.

Reporting coordinate: \(\Delta\mu_{\rm G} = \mu_{{\rm G},c_2} - 35\). The
internal coordinate is the sampled `$\mu_{\rm G}$_c2`; the GAL branch stays
pinned at 35 through the base block.

Three questions:

1. **Are two marks separable?** With \(H_0\) fixed at the planted 67.74, are
   \(f_{\rm AGN}\), \(\Delta\mu_\chi\) and \(\Delta\mu_{\rm G}\) recovered
   jointly, and what is \(\rho(\Delta\mu_\chi, \Delta\mu_{\rm G})\)?
2. **Which channel carries the label?** Ablation, measured rather than asserted:
   what each mark contributes to the identification of \(f_{\rm AGN}\) on its own
   and in combination.
3. **What does a mass mark do to selection?** The detection rule is
   \(\rho_{\rm obs} \ge 8\) with \(\rho_{\rm opt} \propto \mathcal{M}_{\rm det}^{5/6}/d_L\),
   so a mass mark moves detectability branch by branch — unlike the spin mark,
   which does not enter \(\rho_{\rm opt}\) at all.

## The two cosmological channels of a mass mark

A spin mark has exactly one route to \(H_0\). A mass mark has two, and
**separating them is the point of Analysis 10's cosmology arms**:

1. **Event-level tracer routing.** This is the Analysis-9 mechanism. A mark
   sharpens \(P_i({\rm AGN})\) event by event and re-sorts events between the
   GAL and AGN redshift structures; the \(H_0\) gain comes from that routing, not
   from a tighter global \(f_{\rm AGN}\) (Analysis 9: the global-mixture factor
   was 0.99997 at 68%). A mass mark should feed this channel exactly as the spin
   mark does.
2. **The spectral-siren coupling.** Masses redshift,
   \(m_{\rm det} = (1+z)\,m_{\rm src}\), so a feature at a known source-frame
   mass is itself a distance ladder. A **branch-dependent peak location** gives
   the two branches two different spectral-siren rulers, and the location of
   each branch's peak in the detector frame constrains \(H_0\) directly, with no
   reference to the galaxy catalog at all. The spin mark had nothing of the kind:
   \(\chi_{\rm eff}\) does not redshift and does not enter \(\rho_{\rm opt}\).

Any \(H_0\) improvement from the mass mark is therefore a **sum** of a routing
term and a spectral-siren term, and reporting it as one number would be a wrong
attribution. The C10 arms below exist to separate them.

## Arms — \(H_0\) fixed first

| arm | free | cells | what it isolates |
|---|---|---|---|
| **A10-S** | \(f_{\rm AGN}\) | 41 | spatial only; both marks off |
| **A10-χ** | \(f_{\rm AGN}, \Delta\mu_\chi\) | 41 × 61 | the Analysis-8 arm J, re-run on the two-mark mock |
| **A10-M** | \(f_{\rm AGN}, \Delta\mu_{\rm G}\) | 41 × 21 | the mass mark alone |
| **A10-J** | \(f_{\rm AGN}, \Delta\mu_\chi, \Delta\mu_{\rm G}\) | 41 × 61 × 21 = 52,521 | the joint measurement |

\(H_0\) stays pinned at 67.74 in all four.

**Cosmology arms — gated, not scheduled.** Releasing \(H_0\) is a separate step
(the Analysis-9 move applied to the two-mark model), taken **only** after the
fixed-\(H_0\) owner gate passes and never in the same campaign:

| arm | free | what it isolates |
|---|---|---|
| **C10-S** | \(H_0, f_{\rm AGN}\) | the mark-free \(H_0\) baseline on the two-mark mock |
| **C10-J** | \(H_0, f_{\rm AGN}, \Delta\mu_\chi, \Delta\mu_{\rm G}\) | the full \(H_0\) measurement, from which the routing and spectral-siren shares are separated |

## Scope locks

- **Seed 100 only**, one realisation. Science exploration, **not** calibration:
  no extra realisations, no seed ensemble, no error bar on the gain.
- **Complete catalogs only.** No incompleteness, no completion path.
- **No GP/HSGP**, anywhere.
- **No free common hyperparameters.** The twelve base population parameters stay
  pinned by name at the `powerlaw+peak` fiducial; only the per-catalog
  \(c_2\) copies of the two marked slots — `G.mu_c2` and `mu_chi_c2` — are ever
  free.
- **No peak-fraction coordinate.** \(v_1\) / \(\lambda_{\rm peak}\) is **not**
  marked, not sampled and not per-catalog. It is 0.10 / 0.90 in both branches.
- **No third mark.** Exactly two marks: one mass, one spin.
- **No additional mass hyperparameters.** Exactly ONE mass coordinate is marked.
  \(\sigma_{\rm G}\), \(\alpha_{\rm PL}\), \(m_{\min}\), \(m_{\max}\) and the
  taper widths stay common and pinned.
- **No branch-dependent mass ratio.** \(\beta\) is common; \(q\) is identical in
  both branches.
- \(\Omega_{m,0} = 0.3075\); \(H_0 = 67.74\) until the release gate.
- `darksirens` is **not** modified — the per-catalog resolver already carries the
  second-catalog Gaussian mean (Gate R). `working/data/generate_dataset.py`
  gains exactly one new flag, `--dmu_G_agn`, whose default 0.0 leaves every
  existing path **bitwise** unchanged. Nothing is pushed.
- `working/data/seed100/**` as it stands, `analysis_8_marked_multitracer_H0_fagn/**`
  and `analysis_9_marked_multitracer_H0_fagn/**` are **read-only inputs**. The
  new event set is a new file; no existing file is overwritten. Every analysis
  write lands under this directory.
- No likelihood is reimplemented: the Analysis-8 builder is imported.

## The mark is admissible

`G.mu` is slot 6 of `powerlaw+peak`: plain name `G.mu`, LaTeX label
`$\mu_{\rm G}$`, fiducial **35.0**, prior bounds **[20, 50]**. The planted AGN
value 40 sits well inside, and the exploratory axis
\(\Delta\mu_{\rm G} \in [-10, +10]\) (i.e. \(\mu_{{\rm G},\rm AGN} \in [25, 45]\))
leaves \(5\,M_\odot\) of margin at each prior edge. `STATE.md` carries the
verified facts and `GATES.md` the ledger (Gate M2, Gate R).

## Files

- `STATE.md` — the owner decision, the verified facts, current stage; below it,
  the provenance record of the rejected peak-fraction mark.
- `GATES.md` — the ledger: Gate 0 PASS, Gate M FAIL (the rejected mark), Gate M2
  and Gate R PASS, everything after NOT RUN with criteria registered.
- `REPORT.md` — the owner decision, then the full report on the rejected mark.
- `scripts/check_mass_mark_feasibility.py` — the pre-registered check that
  rejected \(\Delta\lambda_{\rm peak}\).
- `diagnostics/a10_mass_mark_feasibility.json` — its output, the evidence of
  record.
- `results/`, `figs/` — empty; nothing has been measured under the new mark and
  nothing has been drawn.
