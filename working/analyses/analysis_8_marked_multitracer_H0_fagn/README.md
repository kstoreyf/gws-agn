# Analysis 8 — marked multi-tracer population inference

## Science question

Analyses 0–7 infer the GAL/AGN host mixture from two correlated large-scale-structure tracers while using one common BBH intrinsic population. Analysis 8 couples those two pieces of the hierarchy:

\[
p(\theta,z,\Omega) =
(1-f_{\rm AGN})\,p_{\rm GAL}(z,\Omega)\,p_{\rm pop}^{\rm GAL}(\theta)
+
f_{\rm AGN}\,p_{\rm AGN}(z,\Omega)\,p_{\rm pop}^{\rm AGN}(\theta),
\]

with \(\theta=(m_1,q,\chi_{\rm eff})\).

The first controlled experiment asks whether the same latent host label can be recovered simultaneously from the spatial tracer field and a single intrinsic mark.

## First production scope

This directory is intentionally narrower than the eventual \(H_0\)+population analysis.

For the first gate:

- seed **100 only**;
- complete GAL and AGN catalogs only;
- \(H_0=67.74\) fixed;
- common mass, mass-ratio and redshift population fixed at the signed-off fiducial;
- one differential intrinsic parameter only,
  \[
  \mu_{\chi,{\rm AGN}}=\mu_{\chi,{\rm GAL}}+\Delta\mu_\chi;
  \]
- common \(\sigma_\chi\);
- infer \((f_{\rm AGN},\Delta\mu_\chi)\);
- use production `darksirens` for PE reweighting, source-frame conversion, population densities and selection;
- compare spatial-only, intrinsic-only and joint spatial+intrinsic information.

The existing seed-100 Analysis 2 dataset is the **null/equivalence dataset**. Do not regenerate a second null mock. The new likelihood must reduce to Analysis 2 when \(\Delta\mu_\chi=0\).

After that equivalence gate passes, generate **one** new seed-100 marked mock with the registered non-zero spin offset in `CLAUDE_FABLE_SPEC.md`.

## Not in the first gate

Do not, without an explicit owner instruction:

- run seeds 101/102/103/105;
- release \(H_0\);
- add a mass mark;
- add catalog incompleteness;
- add \(\chi_p\) or component-spin vectors;
- replace the parametric population with a GP/HSGP;
- move to GWTC data;
- run a planted-effect ladder.

Those are later gates.

## Files

- `CLAUDE_FABLE_SPEC.md` — authoritative execution specification.
- `GATES.md` — pass/fail contract and owner stop.
- `STATE.md` — current state, decisions and provenance to update as work proceeds.
- `scripts/README.md` — conventions for the thin experiment driver layer.

The existing analyses and signed-off datasets are read-only references unless the specification explicitly says otherwise.
