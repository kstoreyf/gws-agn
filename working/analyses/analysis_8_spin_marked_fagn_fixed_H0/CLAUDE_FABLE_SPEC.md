# Claude Fable execution specification — Analysis 8

## Authority and mission

You are driving **Analysis 8: marked multi-tracer population inference** in `kstoreyf/gws-agn`.

Work from the repository state you actually find. The code under `working/` is the authoritative analysis implementation. Archived directories are provenance only. Do not reconstruct the experiment from old toy code.

The scientific goal of this first campaign is deliberately narrow:

> On **one realization only (seed 100)**, extend the existing GAL/AGN multi-tracer dark-siren likelihood so that the same latent GAL/AGN branch also selects a different \(\chi_{\rm eff}\) population. Demonstrate exact/null reduction to Analysis 2, generate one non-zero marked seed-100 mock, recover \((f_{\rm AGN},\Delta\mu_\chi)\), and quantify spatial-only versus intrinsic-only versus joint information. Then **STOP for the owner gate**.

Do not launch additional realizations. Do not infer \(H_0\). Do not add mass-population differences. Do not add incompleteness. Those are explicitly deferred to Ignacio's gate.

---

# 0. Non-negotiable execution rules

1. **Seed 100 only.**
   - Existing seed-100 Analysis 2 files are the null/equivalence dataset.
   - Generate at most one new marked seed-100 event/injection family.
   - Do not run seeds 101, 102, 103, 105 or any new random realization.
   - If you believe another realization is necessary, record why in `STATE.md` and STOP. Do not run it.

2. **Use production `darksirens`.**
   - The mass, \(q\), \(\chi_{\rm eff}\), source-frame conversion, PE-prior division, population model and GW selection machinery already exist there.
   - Do not build a parallel HBI implementation in `gws-agn`.
   - Do not revive `src/darksirens_multitracer` as the production path.
   - If tracer-dependent population blocks require a `darksirens` change, make the smallest general upstream change and test it there. Record the exact darksirens SHA used.

3. **Preserve analyses 0–7.**
   - Existing result files, event files and survey files are read-only.
   - Do not overwrite `working/data/seed100/events/events.h5` or existing injection files.
   - New files must use explicit Analysis-8 names/suffixes or live under the Analysis-8 data/results area.
   - Any generator option added to shared code must default to the existing behavior and pass a regression/bit-identity check on the old path.

4. **One scientific change at a time.**
   - First mark = \(\chi_{\rm eff}\) mean only.
   - Same mass model in GAL and AGN.
   - Same \(q\) model.
   - Same redshift evolution.
   - Same \(\chi_{\rm eff}\) width.
   - \(H_0\) fixed.
   - Complete catalogs.
   - Do not simultaneously introduce another population degree of freedom.

5. **No open-ended looping.**
   - Follow the ordered phases below.
   - Each phase has a concrete pass/fail condition.
   - If a hard gate fails, diagnose enough to identify the failing assumption, write the result to `STATE.md`/`GATES.md`, and STOP rather than launching a parameter sweep.
   - At most one targeted diagnostic rerun is allowed per failed hard gate unless the owner explicitly authorizes more.

6. **Do not optimize away the physics.**
   - The GAL/AGN branch sum must occur only after the matching spatial and intrinsic factors are multiplied within each branch.
   - Do not compute a spatial Bayes factor and an intrinsic Bayes factor independently and multiply post hoc as the production likelihood.
   - Selection must use the same branch-dependent population model as the event likelihood.

7. **Keep the first campaign finite.**
   - No broad effect-size ladder.
   - No multiple seeds.
   - No sampler/model comparison zoo.
   - No publication rewrite.
   - End with the seed-100 closure package and an owner-facing verdict.

---

# 1. Read these files before editing code

Read, in this order:

1. `working/data/README.md`
2. `working/data/DESIGN_PE.md`
3. `working/data/generate_dataset.py`
4. `working/analyses/analysis_2_complete_catalog_H0_fagn/README.md`
5. `working/analyses/analysis_2_complete_catalog_H0_fagn/scripts/scan_h0f.py`
6. `working/analyses/EXECUTIVE_SUMMARY.md`
7. `working/analyses/selection_redo/FU_REPORT.md`
8. current production `darksirens`:
   - `darksirens/likelihood/factory.py`
   - `darksirens/likelihood/core.py`
   - `darksirens/inference/utils.py`
   - `darksirens/gw/populations/base.py`
   - `darksirens/gw/populations/registry.py`
   - `docs/source/guide/populations.md`

Before changing anything, write a short implementation map into `STATE.md` containing:

- current gws-agn HEAD;
- current darksirens HEAD;
- exact existing Analysis-2 likelihood call path;
- exact current population function signature;
- exact location where catalog mixture weights are applied;
- exact location where the selection integral is assembled;
- whether current darksirens already has any per-catalog population API.

Do not start coding until this map is written.

---

# 2. Statistical model

The current model is effectively

\[
p_{\rm pop}(\theta\mid\Lambda)
\left[
(1-f_{\rm AGN})p_{\rm GAL}(x)
+
f_{\rm AGN}p_{\rm AGN}(x)
\right],
\]

where \(x=(z,\Omega)\) and \(\theta=(m_1,q,\chi_{\rm eff},z)\).

Analysis 8 requires

\[
\begin{aligned}
p(\theta,x\mid f_{\rm AGN},\Lambda_{\rm GAL},\Lambda_{\rm AGN})
={}&(1-f_{\rm AGN})\,
p_{\rm GAL}(x)\,
p_{\rm pop}(\theta\mid\Lambda_{\rm GAL})\\
&+f_{\rm AGN}\,
p_{\rm AGN}(x)\,
p_{\rm pop}(\theta\mid\Lambda_{\rm AGN}).
\end{aligned}
\]

For the first experiment the two population vectors are identical except for the spin mean:

\[
\mu_{\chi,{\rm GAL}}=\mu_\chi,
\qquad
\mu_{\chi,{\rm AGN}}=\mu_\chi+\Delta\mu_\chi.
\]

Everything else is fixed to the signed-off seed-100 fiducial.

Reuse the existing darksirens \(\chi_{\rm eff}\) distribution family exactly. Do not invent a second spin family.

The two free scientific coordinates for the first production run are

\[
f_{\rm AGN}\in[0,1],
\qquad
\Delta\mu_\chi.
\]

Set \(H_0=67.74\), \(\Omega_m=0.3075\), and all common population/nuisance coordinates to the existing signed-off values.

---

# 3. Required darksirens architecture

Prefer a general implementation, not an Analysis-8-specific conditional hidden inside gws-agn.

A clean interface is conceptually:

- one spatial/catalog component per tracer \(k\);
- one population parameter block or population view per tracer \(k\);
- event weight for branch \(k\),
  \[
  w_{ijk}\propto
  p_{{\rm cat},k}(z_{ij},\Omega_{ij})
  p_{{\rm pop},k}(m_{1,ij}^{\rm src},q_{ij},z_{ij},\chi_{ij})
  /\pi_{\rm PE}(\theta_{ij});
  \]
- event likelihood,
  \[
  \mathcal L_i=\sum_k f_k\,\mathcal L_{ik};
  \]
- selection expectation,
  \[
  \mu=\sum_k f_k\,\mu_k.
  \]

Do not abuse `shared_spin=False` to mean GAL versus AGN. In current darksirens that flag controls sharing among **mass components inside one population mixture**, not sharing across spatial tracer branches.

The new API should preserve the old path exactly when no per-catalog population split is requested.

If a darksirens change is needed:

- add unit tests there first;
- preserve backwards compatibility;
- keep the existing single-population path bitwise/numerically unchanged to the tightest practical tolerance;
- make one focused commit;
- record its SHA in `STATE.md`.

Do not refactor unrelated darksirens code.

---

# 4. Hard Gate A — null/equivalence on the existing Analysis-2 data

**Do this before generating any new mock.**

Use the existing seed-100 Analysis-2 event, catalog and selection files.

Configure the new tracer-dependent population path with

\[
\Lambda_{\rm GAL}=\Lambda_{\rm AGN}
\quad\Longleftrightarrow\quad
\Delta\mu_\chi=0.
\]

Compare directly against the existing Analysis-2 production likelihood.

## A1. Cell-level likelihood equivalence

On representative cells including \(f=0\), \(f=f_{\rm true}\), \(f=0.5\), and \(f=1\), with \(H_0=67.74\), compare old and new log likelihoods after accounting only for any explicitly known additive constant.

Target: numerical equality at machine/JAX reduction precision. If the old and new code traverse the same arithmetic path, require bit identity. Otherwise document the measured floating-point tolerance before accepting it.

## A2. Endpoint reduction

At \(f=0\), the result must equal the GAL single-branch likelihood.

At \(f=1\), the result must equal the AGN single-branch likelihood.

Check both PE contribution and selection contribution.

## A3. Selection reduction

With identical populations in both branches,

\[
\mu(f)=(1-f)\mu_{\rm GAL}+f\mu_{\rm AGN}
\]

must reproduce the current Analysis-2 K=2 selection result.

## A4. Analysis-2 posterior reproduction

Run the seed-100 Analysis-2 \(f_{\rm AGN}\) scan, with \(H_0\) fixed at truth, through the new code at \(\Delta\mu_\chi=0\).

The \(f_{\rm AGN}\) posterior must reproduce the existing Analysis-2 result within the measured numerical tolerance.

Write:

- `diagnostics/null_equivalence.json`
- `diagnostics/null_equivalence.md`

and update `GATES.md`.

**If Gate A fails: STOP after one focused diagnostic. Do not generate the marked mock.**

---

# 5. New marked seed-100 mock

Only after Gate A passes, make one marked version of seed 100.

## 5.1 Preserve the existing universe

Reuse the existing seed-100:

- GLASS/LSS realization;
- GAL catalog;
- AGN catalog;
- complete survey files;
- tracer densities;
- tracer biases;
- host positions/redshifts;
- cosmology;
- measurement family.

Do not regenerate the entire universe unless technically unavoidable.

## 5.2 Planted host fraction

Use the existing planted value

\[
f_{\rm AGN}^{\rm plant}=0.30
\]

with the realized fraction recorded from the detected set exactly as analyses 0–2 do.

## 5.3 Planted intrinsic mark

Register one nonzero effect:

\[
\boxed{\Delta\mu_\chi^{\rm plant}=+0.10}
\]

relative to the existing fiducial spin mean.

The GAL branch uses the existing fiducial \(\mu_\chi\).

The AGN branch uses

\[
\mu_{\chi,\rm AGN}=\mu_{\chi,\rm GAL}+0.10.
\]

Keep the existing \(\sigma_\chi\) fixed and identical between branches.

If \(+0.10\) would violate the support or make the current truncated spin model malformed at the fiducial, do not silently change the number. Record the issue and STOP for owner input.

## 5.4 Event generation

Use the same host-label draw and host-selection logic as the existing seed-100 mixture, but condition the \(\chi_{\rm eff}\) draw on the host label.

Masses and \(q\) are drawn from the same population in both branches.

The v3 PE measurement family remains unchanged.

Write a new explicitly named file, for example:

`data/seed100/events/events_marked_dmu0p10.h5`

or an equivalent Analysis-8-local path. Never overwrite the signed-off `events.h5`.

Record the true host label and planted branch spin parameters in file metadata.

## 5.5 Selection injections

The selection integral must be valid for both branch populations.

Prefer reusing the existing injection proposal architecture and modifying only the target population weights if the proposal already covers the required \(\chi_{\rm eff}\) support.

Do not blindly regenerate \(10^8\)-scale injections if existing injections have adequate support and exact proposal densities. First prove whether reweighting the existing selection files is sufficient.

If new injections are required, create **one** marked seed-100 targeted lane first. Only create the existing population/uniform cross-check lane if the production targeted lane passes and the additional lane is needed for the selection cross-check. Do not create more proposals.

Record injection support diagnostics in `diagnostics/selection_support.json`.

Important physical check: the current v3 mock detection rule may be independent of \(\chi_{\rm eff}\). If so, the spin-only mark should not create a physical detection-efficiency difference after integrating a normalized spin distribution. The branch-dependent selection implementation must nevertheless be mathematically correct; use the expected near-equality of \(\mu_{\rm GAL}\) and \(\mu_{\rm AGN}\) as a diagnostic rather than hard-coding them equal.

---

# 6. Hard Gate B — marked mock integrity

Before inference, verify:

1. exactly one master realization: seed 100;
2. same catalog/LSS realization as the existing seed-100 record;
3. host-label fraction is consistent with the requested mixture;
4. GAL and AGN truth \(\chi_{\rm eff}\) distributions have the registered mean offset;
5. masses and \(q\) have no planted channel difference;
6. the v3 measurement-family contract is unchanged;
7. all PE prior densities are in the canonical darksirens basis;
8. all detected events pass the same recorded detection rule;
9. selection proposal support covers both branch population supports;
10. no existing seed-100 record file changed.

Write `diagnostics/marked_mock_validation.json`.

If any item fails, STOP.

---

# 7. Inference runs — exactly three scientific arms

Use the same marked seed-100 dataset for all three arms.

Do not add extra model variants.

## Arm S — spatial only

Force the intrinsic populations to be identical:

\[
\Delta\mu_\chi=0.
\]

Infer \(f_{\rm AGN}\) from GAL versus AGN tracer structure only.

This is the Analysis-2 analogue on the marked data.

## Arm I — intrinsic only

Remove the branch-dependent spatial information while retaining the two intrinsic populations.

Implement this as a **diagnostic** branch in which GAL and AGN use the same fixed common spatial prior. Construct the common prior once from the realized seed-100 host field and keep it independent of the sampled \(f_{\rm AGN}\), so the branch label is identified only by intrinsic data.

Document the exact common-prior construction in the result metadata.

Infer

\[
(f_{\rm AGN},\Delta\mu_\chi).
\]

## Arm J — joint spatial + intrinsic

Use the full model

\[
(1-f)p_{\rm GAL}(x)p_{\rm pop,GAL}(\theta)
+
fp_{\rm AGN}(x)p_{\rm pop,AGN}(\theta).
\]

Infer

\[
(f_{\rm AGN},\Delta\mu_\chi).
\]

This is the production Analysis-8 arm.

---

# 8. Parameter grids and compute discipline

This first campaign is only two-dimensional.

Use a deterministic grid unless the measured cost makes it clearly unreasonable.

Registered production ranges:

\[
f_{\rm AGN}\in[0,1],
\]

\[
\Delta\mu_\chi\in[-0.20,+0.25].
\]

Use enough resolution to resolve the posterior but do not run iterative grid-refinement loops.

Allowed workflow:

1. one cheap timing/smoke evaluation on a small set of cells;
2. choose one production grid;
3. run that grid once for Arm I and Arm J;
4. Arm S is one-dimensional.

A reasonable initial production target is \(41\times61\) for \((f,\Delta\mu_\chi)\), but measure one cell/block first and reduce/increase only once if memory/runtime requires it.

Do not run a coarse-grid, medium-grid, fine-grid sequence unless a hard numerical problem requires it.

Carry the existing selection \(N_{\rm eff}\) and likelihood-variance bookkeeping. No result hidden behind a rejected guard cell is publishable.

---

# 9. Hard Gate C — seed-100 recovery

The campaign passes only if all of the following are true.

## C1. Joint recovery

Arm J must recover the planted realized \(f_{\rm AGN}\) and

\[
\Delta\mu_\chi=+0.10
\]

within the declared credible intervals.

Report both median/MAP and 68/90% equal-tailed intervals using the same conventions as Analysis 2.

## C2. Null is disfavored only when warranted

On the marked mock, report the posterior density/credible support at \(\Delta\mu_\chi=0\).

Do not translate this into a sigma claim unless the posterior construction justifies it.

## C3. Spatial arm behaves as expected

Arm S should recover the host fraction using only tracer information and should not have access to the planted spin mark.

## C4. Intrinsic arm behaves as expected

Arm I should recover the sign and scale of the spin offset without using relative GAL/AGN spatial structure.

## C5. Joint information is coherent

Compare the \(f_{\rm AGN}\) and \(\Delta\mu_\chi\) widths between I and J. The joint model should not become pathologically broader or shift away from truth without a diagnosed reason.

Do **not** require a predetermined improvement factor. Measure it.

## C6. Selection validity

All posterior-support cells used in the result must satisfy the existing \(N_{\rm eff}\)/variance guard policy.

If the targeted and population/uniform lanes are both run, their posterior shift must be reported relative to one posterior half-width.

---

# 10. Event-level decomposition

For the final seed-100 joint posterior, compute per-event branch information at a representative hyperparameter point, posterior median or MAP, and record which.

For each event save:

- true host label, mock diagnostic only;
- spatial log Bayes factor,
  \[
  \log {\rm BF}_{i,\rm spatial};
  \]
- intrinsic log Bayes factor,
  \[
  \log {\rm BF}_{i,\rm intrinsic};
  \]
- combined posterior \(P_i({\rm AGN})\).

The exact decomposition must be defined from the same factors used inside the production branch likelihood.

Write a compact table/JSON and make one figure,

\[
\log {\rm BF}_{\rm spatial}
\quad {\rm vs}\quad
\log {\rm BF}_{\rm intrinsic},
\]

with points distinguished by true mock host label and/or combined \(P_i({\rm AGN})\).

Do not build an elaborate classifier analysis.

---

# 11. Required output package

At minimum produce:

## Code and provenance

- thin Analysis-8 driver scripts under `scripts/`;
- any darksirens tests/commit needed for tracer-dependent population blocks;
- exact gws-agn and darksirens SHAs;
- exact command lines used.

## Diagnostics

- `diagnostics/null_equivalence.json`
- `diagnostics/null_equivalence.md`
- `diagnostics/marked_mock_validation.json`
- `diagnostics/selection_support.json`
- event-level decomposition data

## Results

Compact result summaries for:

- spatial-only;
- intrinsic-only;
- joint.

Respect the repository's existing policy for large HDF5 chains/grids. Commit code, documentation, compact JSON and production figures; do not force-add huge binary products unless the repo's current convention explicitly does so.

## Figures

1. `fig_joint_f_dmu.{pdf,png}` — joint posterior in \((f_{\rm AGN},\Delta\mu_\chi)\), truth marked clearly.
2. `fig_ablation_fagn.{pdf,png}` — spatial-only, intrinsic-only and joint \(p(f_{\rm AGN})\).
3. `fig_ablation_dmu.{pdf,png}` — intrinsic-only and joint \(p(\Delta\mu_\chi)\).
4. `fig_event_evidence_plane.{pdf,png}` — spatial versus intrinsic per-event evidence.
5. `fig_selection_marked.{pdf,png}` — only if selection diagnostics are nontrivial enough to justify a figure; otherwise keep them numeric.

Use the existing paper/analysis plotting conventions. No decorative AI-style figures.

## Report

Write `REPORT.md` containing:

- question;
- exact model;
- implementation change;
- null/equivalence result;
- marked mock definition;
- three-arm result;
- event-level interpretation;
- selection diagnostics;
- limitations;
- explicit statement that this is **seed 100 only**;
- exact next owner gate.

---

# 12. Version-control discipline

Commit in small scientific units, not every file touch.

Recommended commits:

1. `analysis8: specify marked multitracer experiment`
2. `darksirens: support tracer-dependent population blocks` — only if needed, in darksirens repo
3. `analysis8: add seed100 marked mock and validation`
4. `analysis8: run seed100 marked multitracer inference`
5. `analysis8: add report and production figures`

Do not rewrite or squash existing history.

Do not modify analyses 0–7 to make Analysis 8 easier.

---

# 13. State logging

Update `STATE.md` after every hard gate, not after every command.

It must always state:

- current phase;
- last passing gate;
- exact gws-agn SHA;
- exact darksirens SHA;
- data files used;
- completed outputs;
- unresolved issue, if any;
- next allowed action.

`GATES.md` is the concise pass/fail ledger.

---

# 14. STOP condition — mandatory owner gate

After the seed-100 package is complete, **STOP**.

Do not proceed automatically to:

- more realizations;
- effect-size ladders;
- \(H_0\) release;
- mass-dependent marks;
- free common population parameters;
- incomplete catalogs;
- real GWTC data.

End your run with a short owner-facing summary containing exactly:

1. whether Gates A, B and C passed;
2. recovered \(f_{\rm AGN}\) and \(\Delta\mu_\chi\);
3. spatial-only / intrinsic-only / joint width comparison;
4. minimum selection \(N_{\rm eff}\) / guard status;
5. darksirens SHA and gws-agn SHA;
6. any caveat that changes the interpretation;
7. the sentence:

> **OWNER GATE: seed-100 Analysis 8 is complete. I have not run additional realizations.**

Wait for Ignacio to decide whether to authorize the next phase.

---

# 15. Claims forbidden from a successful seed-100 run

Do not claim from one realization that:

- the estimator is calibrated across realizations;
- the coverage is correct;
- the measured offset is unbiased in expectation;
- a sensitivity scaling with \(N\) is established;
- real BBHs have an AGN-specific spin distribution;
- \(H_0\) improves;
- incomplete catalogs are safe.

A successful seed-100 run establishes implementation closure and proof of concept only. Multiple realizations are deliberately deferred to the owner gate.
