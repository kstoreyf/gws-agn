# Analysis 10 gates

The concise pass/fail ledger. Detailed evidence belongs in `diagnostics/`,
`results/` and `REPORT.md`. Every criterion below is registered **before** the
run; a measurement that misses one is a FAIL to be diagnosed, not a bound to be
widened.

**The ladder is blocked at Gate M** (2026-09-21). The registered mass mark is
inadmissible under the production parameterisation —
\(\lambda_{\rm peak}^{\rm fid} + \Delta\lambda_{\rm peak} = 0.90 + 0.15 = 1.05\)
— so nothing downstream ran. Every gate after M is **NOT RUN**, not failed: it
has no data to be run against. The criteria are nevertheless registered here, so
that whatever mark the owner chooses inherits a ledger written before its first
number exists.

| gate | status | the number that decides it |
|---|---|---|
| 0 provenance and scope | **PASS** | darksirens HEAD `af896ca`, clean, imports from `darksirens-a8`; base `2b86a2d` an ancestor; gws-agn `408990a` |
| **M mark admissibility** | **FAIL** | \(\lambda_{\rm peak}^{\rm fid} = 0.90\); \(0.90 + 0.15 = \mathbf{1.05} \not\in (0,1)\); sampled coordinate \(v_{1,c2} = -0.05 \notin [0,1]\) |
| G generation | NOT RUN | blocked by M |
| B selection support | NOT RUN | blocked by M |
| 10 closure identities | NOT RUN | blocked by M |
| A10-S spatial-only arm | NOT RUN | blocked by M |
| A10-χ spin-mark arm | NOT RUN | blocked by M |
| A10-M mass-mark arm | NOT RUN | blocked by M |
| A10-J joint arm | NOT RUN | blocked by M |
| D figures and report | NOT RUN | blocked by M |
| **fixed-\(H_0\) owner gate** | **NOT REACHED** | blocked by M |
| \(H_0\) release | NOT REACHED | gated behind the fixed-\(H_0\) owner gate, which is itself blocked |

## Scope lock

- [x] Seed 100 only, one realisation.
- [x] Complete catalogs only; no incompleteness, no completion path.
- [x] No GP/HSGP anywhere.
- [x] No free common hyperparameters: the twelve base population parameters stay
      pinned by name at the `powerlaw+peak` fiducial; only per-catalog `_c2`
      copies of marked slots are ever free.
- [x] No branch-dependent mass ratio; \(\beta\) common.
- [x] Exactly ONE mass coordinate marked; no additional mass parameters.
- [x] \(\Omega_{m,0} = 0.3075\); \(H_0 = 67.74\) until the release gate.
- [x] `darksirens` not modified; `generate_dataset.py` not modified; nothing
      pushed.
- [x] Writes confined to this directory; `working/data/**`, the Analysis-8 tree
      and the Analysis-9 tree are read-only.
- [x] No likelihood reimplemented: the Analysis-8 builder is imported.
- [x] All GPU work, when there is any, on rita via SLURM. **None ran.**

## Gate 0 — provenance and scope

Status: **PASS** (2026-09-21, CPU, no GPU). Evidence:
`diagnostics/a10_mass_mark_feasibility.json` → `provenance`.

| check | measured | verdict |
|---|---|---|
| `PYTHONPATH` carries darksirens-a8 | `/hildafs/projects/phy230014p/magana/src/darksirens-a8` | PASS |
| darksirens HEAD | `af896cae6f3f3dd1f87dec50046e3a8228f59b39` | PASS |
| worktree clean | clean (`git status --porcelain` empty) | PASS |
| import actually resolves there | `.../darksirens-a8/darksirens/__init__.py` | PASS |
| pinned base `2b86a2d` is an ancestor | yes | PASS |
| same pin as the seed-100 mock | events `metadata_json.darksirens_sha` = `af896ca…` | PASS |
| gws-agn HEAD | `408990af79afd82d403d600621aa6525e97e6d6d`, dirty only outside this analysis | PASS |
| CPU only | `JAX_PLATFORMS=cpu`; no survey loaded, no likelihood built | PASS |
| no write outside `analysis_10/**` | no file under `analysis_8/**` or `working/data/**` newer than its close; no `__pycache__` added | PASS |

## Gate M — mark admissibility

Status: **FAIL** (2026-09-21). Evidence:
`diagnostics/a10_mass_mark_feasibility.json` → `admissibility`,
`stick_breaking`, `composition_order`, `mock`. Driver
`scripts/check_mass_mark_feasibility.py` (exit status 2).

The registered criterion, verbatim from the specification: *verify
\(0 < \lambda_{\rm peak}^{\rm fid} + 0.15 < 1\); if this is not true under the
actual production parameterization, STOP and report the real fiducial value; do
not silently alter the registered mark.*

| check | measured | verdict |
|---|---|---|
| **M1** the coordinate exists as specified | it does **not** — `powerlaw+peak` has no \(\lambda_{\rm peak}\) slot; slot 0 is the stick-breaking input `v1` with fiducial 0.10, and \(\lambda_{\rm peak} = 1 - v_1\) | noted |
| **M2** composition order (registry) | `Curated(weights=(0.10,))  # w_PL=0.10, w_G=0.90`; `Curated.weights` holds desired final fractions | PASS |
| **M3** composition order (numerical, independent) | peak-band [25, 45] mass falls 0.8132 → 0.6322 → 0.1146 and power-law-band [5, 20] mass rises 0.1397 → 0.3168 → 0.8231 as \(v_1\) = 0.10 → 0.25 → 0.90; monotone, asserted not assumed | PASS |
| **M4** stick-breaking round trip | `_w_to_v([0.1, 0.9]) = [0.1]`, max \(\|\Delta v\|\) = 0.0 | PASS |
| **M5** the fiducial | \(\lambda_{\rm peak}^{\rm fid} = 1 - 0.10 = \mathbf{0.90}\) | PASS |
| **M6 the registered check** | \(0.90 + 0.15 = \mathbf{1.05}\); \(0 < 1.05 < 1\) is **False** | **FAIL** |
| **M7** the same statement in the sampled coordinate | the AGN branch needs \(v_{1,c2} = 1 - 1.05 = \mathbf{-0.05}\), outside the `v1` prior \([0, 1]\) | **FAIL** |
| **M8** independent corroboration from the data | the seed-100 mock's generator record carries `"peak_fraction": 0.9`; 92.4% of 1000 detected `true_m1src` lie in [25, 45] (GAL 93.2%, AGN 90.5%), 0.6% below 20 \(M_\odot\) | PASS |

M6 and M7 are one failure, not two: the mark asks for 105% of the AGN branch's
mass in the Gaussian peak, i.e. a negative power-law weight. It is outside the
simplex, so no widening of a grid, prior or axis can admit it.

**Action taken.** The specification's own instruction was followed: STOP, report
the real fiducial, do not alter the mark. `REPORT.md` carries the options for
the owner; none was executed.

## Gate G — generation (registered, NOT RUN)

Status: **NOT RUN** — blocked by Gate M. No mock, no injections, no PE.

- [ ] **G1.** The two-mark mock is generated at the pin `af896ca` on seed 100,
      with \(\Delta\mu_\chi = +0.10\) unchanged and the admitted
      \(\Delta\lambda_{\rm peak}\), and the realised mark is measured and
      recorded beside the planted one (the realised \(\Delta\mu_\chi\) in the
      current mock is +0.111924 ± 0.006815 against a planted +0.100000; a mass
      mark needs the same treatment).
- [ ] **G2.** The generator's branch-dependent mass path is stated explicitly.
      Today `generate_dataset.py` carries only `--dmu_chi_agn`, applied to a
      **completed** spin draw so the RNG stream, the host labels, the masses,
      the sky, the distances and the detected set are untouched. A mass mark
      cannot use that trick: changing a mixture weight changes which component
      an event is drawn from, hence the masses, hence — through
      \(\rho_{\rm opt} \propto \mathcal{M}_{\rm det}^{5/6}/d_L\) — the detected
      set. Whatever is done instead is recorded as part of G1.
- [ ] **G3.** The unmarked control is regenerated on the same stream, so every
      mark-on/mark-off comparison is differential on one realisation.

## Gate B — selection support (registered, NOT RUN)

Status: **NOT RUN** — blocked by Gate M. **This gate is not optional for a mass
mark.** The detection rule is `observed-data` with \(\rho_{\rm obs} \ge 8\) and
\(\rho_{\rm opt} \propto \mathcal{M}_{\rm det}^{5/6}/d_L\), so a mass mark
changes detectability branch by branch. The spin mark did not: \(\chi_{\rm eff}\)
does not enter \(\rho_{\rm opt}\), which is why Analyses 8 and 9 could reuse
`injections_targeted.h5` without re-deriving its support.

- [ ] **B1.** \(N_{\rm eff}\) of the existing `injections_targeted.h5` is
      measured under **both** branch populations separately — GAL at
      \(\lambda_{\rm peak} = 0.90\) and AGN at
      \(\lambda_{\rm peak} = 0.90 + \Delta\lambda_{\rm peak}\) — across the
      registered grid, not only at the planted cell.
- [ ] **B2.** The hard `selection_neff_guard` (threshold 5000) is cleared over
      the cells that carry the posterior, with the margin quoted at 90%, 99%,
      99.9% and 99.999% of the mass (Analysis 8's arm J: 85.5x, 70.5x, 55.3x,
      30.7x).
- [ ] **B3.** Any guard-rejected region is bounded: the rejected posterior mass
      is reported with a monotonicity-free bound.
- [ ] **B4.** If B1 fails on the existing injection set, a **targeted injection
      set for the marked branch** is generated; reweighting an injection set
      whose proposal never covered the shifted mass distribution is not an
      acceptable substitute, and the failure is reported to the owner before any
      inference.

## Gate 10 — closure identities (registered, NOT RUN)

Status: **NOT RUN** — blocked by Gate M. These are the identities the
specification lists; they are registered here so they bind whatever mark is
chosen. Residuals are quoted in ULP of the returned value alongside the absolute
bound (at the seed-100 scale, 1 ULP = 9.094947017729282e-13 at
\(\ln L \approx -4.2\times10^3\)).

- [ ] **10.1 zero-mark reduction.** At \(\Delta\mu_\chi = 0\) **and**
      \(\Delta\lambda_{\rm peak} = 0\) the two-mark model reproduces the
      spatial-only (Analysis-2-shape) likelihood over a registered set of
      \(f\) nodes. Criterion \(\le 10^{-6}\) absolute; same-hardware expectation
      \(\le 4\) ULP. The reference must be a genuinely different model — 8
      sampled labels, no `_c2` population coordinate at all — not the same model
      evaluated at zero.
- [ ] **10.2 one-mark reduction.** At \(\Delta\lambda_{\rm peak} = 0\) the
      two-mark model reproduces Analysis 8's arm J cell for cell on the shared
      \((f, \Delta\mu_\chi)\) lattice, and at \(\Delta\mu_\chi = 0\) it
      reproduces the mass-mark-only arm. Same criterion.
- [ ] **10.3 \(f_{\rm AGN} = 0\): both marks inert.** With the AGN weight at
      zero, `logL` is **bitwise frozen** as \(\Delta\mu_\chi\) and
      \(\Delta\lambda_{\rm peak}\) vary over their full axes — one distinct hex
      value per row — in the total, the PE term and the selection term alike.
- [ ] **10.4 \(f_{\rm AGN} = 1\): only the AGN branch.** The same rows give a
      distinct value per node, and the spread is reported.
- [ ] **10.5 endpoint identity.** \(f = 0\) equals a K=1 GAL build and
      \(f = 1\) equals a K=1 AGN build **exactly 0.0** in the total, the PE term,
      the selection term and `log_mu`. This is the identity that validates any
      K=2 wiring; it is registered as an exact equality, not a tolerance.
- [ ] **10.6 mark liveness — the Analysis-8 trap.** Moving each mark off its
      planted value by one grid node changes \(\ln L\) by far more than the 10.1
      residual (registered threshold: min \(|\Delta\ln L| > 10^{-3}\)).
      **10.1–10.5 all pass for free if a new coordinate is silently ignored**,
      which is exactly the defect Analysis 8's Gate A caught, so 10.6 is not
      optional and must be run for \(\Delta\lambda_{\rm peak}\) specifically —
      the coordinate the resolver has never carried in production.
- [ ] **10.7 grid registration.** The \(\Delta\lambda_{\rm peak}\) axis is
      registered before the run with its node count and range, and every node
      lies strictly inside \([0, 1]\) in the branch coordinate
      \(\lambda_{\rm peak}^{\rm fid} + \Delta\lambda_{\rm peak}\); an axis that
      would clip at the prior edge is reported, not silently truncated.

## Gates A10-S, A10-χ, A10-M, A10-J — the four fixed-\(H_0\) arms (registered, NOT RUN)

Status: **NOT RUN** — blocked by Gate M. \(H_0\) is pinned at 67.74 in all four.

| gate | arm | free | criterion |
|---|---|---|---|
| **A10-S** | spatial only | \(f_{\rm AGN}\) | \(f_{\rm AGN}\) recovers the planted 0.30 and the realised value at 90%; this is the mark-free baseline every later width is quoted against |
| **A10-χ** | spin mark | \(f_{\rm AGN}, \Delta\mu_\chi\) | reproduces Analysis 8's arm J on the shared lattice within the 10.2 bound where the data are unchanged, and recovers both truths at 90% |
| **A10-M** | mass mark | \(f_{\rm AGN}, \Delta\lambda_{\rm peak}\) | \(\Delta\lambda_{\rm peak}\) recovers the planted **and** realised mark at 90%, with the realised value carried everywhere |
| **A10-J** | joint | \(f_{\rm AGN}, \Delta\mu_\chi, \Delta\lambda_{\rm peak}\) | all three recover at 90%; \(\rho(\Delta\mu_\chi, \Delta\lambda_{\rm peak})\), \(\rho(f, \Delta\mu_\chi)\) and \(\rho(f, \Delta\lambda_{\rm peak})\) are reported; the two marks are shown to be separable rather than a single re-labelled direction |

- [ ] **A10-1.** Every plotted interval, band and contour is **90%**.
- [ ] **A10-2.** Both truths — planted and realised — are carried in every table
      and figure.
- [ ] **A10-3.** Widths are quoted **differentially** against A10-S and against
      Analysis 8, never as an absolute calibrated gain from one realisation.
- [ ] **A10-4.** \(H_0\) is quoted differentially if it is ever quoted at all:
      seed 100's own draw sits high (Analysis 2 recovered 69.217 unmarked), and
      that is a property of the realisation, not a bias.
- [ ] **A10-5.** Each arm's cube is checked for duplicate rows and its guard
      rejections bounded, as in Analysis 9's Gate B.

## Gate D — figures and report (registered, NOT RUN)

- [ ] **D1.** Every result figure ships as **both** `.pdf` and `.png`, rendered
      by a deterministic `make_figures.py` that reads `results/` and
      `diagnostics/` only.
- [ ] **D2.** Drawn values agree with the JSON they came from to \(\le 10^{-12}\).
- [ ] **D3.** `REPORT.md` is written from `results/` and `diagnostics/` alone;
      nothing is recomputed for it.

`figs/` and `results/` hold only `.gitkeep`. Nothing was measured, so nothing was
drawn.

## The fixed-\(H_0\) owner gate

Status: **NOT REACHED**. It sits behind Gate M, and Gate M failed before
generation. The \(H_0\) release is a further, separate gate behind it.

The decision in front of the owner today is not a result. It is which
\(\Delta\lambda_{\rm peak}\) — if any — replaces +0.15. `REPORT.md` lays out the
options; none was executed.
