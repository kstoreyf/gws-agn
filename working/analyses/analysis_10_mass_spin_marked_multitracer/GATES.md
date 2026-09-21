# Analysis 10 gates

The concise pass/fail ledger. Detailed evidence belongs in `diagnostics/`,
`results/` and `REPORT.md`. Every criterion below is registered **before** the
run; a measurement that misses one is a FAIL to be diagnosed, not a bound to be
widened.

**The mark changed on 2026-09-21.** The registered peak-fraction mark
\(\Delta\lambda_{\rm peak} = +0.15\) failed Gate M before generation — it lies
outside the mixture simplex — and the owner replaced it with a Gaussian-peak
**location** mark, \(\Delta\mu_{\rm G} = +5\,M_\odot\)
(\(\mu_{\rm G}: 35 \to 40\)), with \(\Delta\mu_\chi = +0.10\) unchanged. Gate M
stays in this ledger as a FAIL: it is the rejected mark's gate and part of the
provenance record, not something to be reopened. Gates M2 and R are the new
mark's admissibility and wiring gates, and both **PASS**. Everything downstream
is re-keyed to \(\Delta\mu_{\rm G}\) and is **NOT RUN** — nothing has been
generated or evaluated under the new mark.

| gate | status | the number that decides it |
|---|---|---|
| 0 provenance and scope | **PASS** | darksirens HEAD `af896ca`, clean, imports from `darksirens-a8`; base `2b86a2d` an ancestor; gws-agn `408990a` |
| **M** mark admissibility — *rejected peak-fraction mark* | **FAIL** | \(\lambda_{\rm peak}^{\rm fid} = 0.90\); \(0.90 + 0.15 = \mathbf{1.05} \not\in (0,1)\); sampled coordinate \(v_{1,c2} = -0.05 \notin [0,1]\) |
| **M2** mark admissibility — \(\Delta\mu_{\rm G} = +5\) | **PASS** | `G.mu` fiducial 35.0, bounds [20, 50]; \(20 < 40 < 50\); axis \(\mu_{{\rm G},\rm AGN} \in [25, 45]\) inside with \(5\,M_\odot\) margin each side |
| **R** resolver emits the second-catalog Gaussian mean | **PASS** | 10 labels, `$\mu_{\rm G}$_c2` at [20, 50] and `$\mu_\chi$_c2` at [−1, 1]; no darksirens change |
| G generation (`--dmu_G_agn`) | NOT RUN | — |
| V mock validation (12 checks) | NOT RUN | — |
| B selection support | NOT RUN | — |
| 15 closure identities (15.1–15.7) | NOT RUN | — |
| A10-S spatial-only arm | NOT RUN | — |
| A10-χ spin-mark arm | NOT RUN | — |
| A10-M mass-mark arm | NOT RUN | — |
| A10-J joint arm | NOT RUN | — |
| D figures and report | NOT RUN | — |
| **fixed-\(H_0\) owner gate** | **NOT REACHED** | — |
| C10-S, C10-J cosmology arms | NOT REACHED | gated behind the fixed-\(H_0\) owner gate |

## Scope lock

- [x] Seed 100 only, one realisation. Science exploration, **not** calibration:
      no extra realisations, no seed ensemble.
- [x] Complete catalogs only; no incompleteness, no completion path.
- [x] No GP/HSGP anywhere.
- [x] No free common hyperparameters: the twelve base population parameters stay
      pinned by name at the `powerlaw+peak` fiducial; only the per-catalog
      copies `G.mu_c2` and `mu_chi_c2` are ever free.
- [x] **No peak-fraction coordinate.** \(v_1\) / \(\lambda_{\rm peak}\) is not
      marked, not sampled and not per-catalog: 0.10 / 0.90 in both branches.
- [x] **No third mark.** Exactly two: one mass, one spin.
- [x] **No additional mass hyperparameters.** Exactly ONE mass coordinate is
      marked; \(\sigma_{\rm G}\), \(\alpha_{\rm PL}\), \(m_{\min}\),
      \(m_{\max}\) and the taper widths stay common and pinned.
- [x] No branch-dependent mass ratio; \(\beta\) common, \(q\) identical in both
      branches.
- [x] \(\Omega_{m,0} = 0.3075\); \(H_0 = 67.74\) until the release gate.
- [x] `darksirens` not modified. `generate_dataset.py` gains exactly one flag,
      `--dmu_G_agn`, default 0.0 leaving every existing path bitwise unchanged.
      Nothing pushed.
- [x] Analysis writes confined to this directory; `working/data/seed100/**` as
      it stands, the Analysis-8 tree and the Analysis-9 tree are read-only, and
      the new event set is a **new file**, overwriting nothing.
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

## Gate M — mark admissibility of the REJECTED peak-fraction mark

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

## Gate M2 — mark admissibility of \(\Delta\mu_{\rm G} = +5\,M_\odot\)

Status: **PASS** (2026-09-21, CPU, no GPU). Verified on the pinned checkout
`/hildafs/projects/phy230014p/magana/src/darksirens-a8`, HEAD `af896ca`, clean.

| check | measured | verdict |
|---|---|---|
| **M2.1** the coordinate exists as specified | `powerlaw+peak` slot 6: plain name `G.mu`, LaTeX label `$\mu_{\rm G}$` | PASS |
| **M2.2** the fiducial | \(\mu_{\rm G}^{\rm fid} = \mathbf{35.0}\,M_\odot\) — the GAL branch value, unchanged | PASS |
| **M2.3** the prior bounds | \([20, 50]\) | PASS |
| **M2.4** the planted AGN value is interior | \(20 < \mathbf{40} < 50\) | PASS |
| **M2.5** the scan axis is interior | \(\Delta\mu_{\rm G} \in [-10, +10] \Rightarrow \mu_{{\rm G},\rm AGN} \in [25, 45]\); \(5\,M_\odot\) of margin at each edge, no clipping, no node dropped | PASS |
| **M2.6** nothing else in the mass sector moves | slot 7 `G.sigma` = 5.0 shared; slot 0 `v1` = 0.10 shared, so \(\lambda_{\rm peak} = 0.90\) in **both** branches; PL slope, limits and tapers shared; \(\beta\) shared | PASS |
| **M2.7** the spin mark is untouched | slot 9 `mu_chi` = 0.0, slot 10 `sigma_chi` = 0.10; mark \(\Delta\mu_\chi = +0.10\) | PASS |

**Coordinates.** Reporting coordinate \(\Delta\mu_{\rm G} = \mu_{{\rm G},c_2} - 35\);
internal coordinate the sampled `$\mu_{\rm G}$_c2`. The GAL branch stays pinned
at 35 through the base block, so the mark is carried entirely by the \(c_2\)
copy and \(\Delta\mu_{\rm G} = 0\) is exactly the shared-population model.

Unlike Gate M, this is a bounds question, not a simplex question: the axis has
room on both sides and no node needs to be dropped.

## Gate R — the resolver emits the second-catalog Gaussian mean

Status: **PASS** (2026-09-21, CPU, no data loaded).

`build_parameter_space(..., n_catalogs=2, per_catalog_pop_params=('G.mu_c2', 'mu_chi_c2'))`,
called exactly as `analysis_8/scripts/a8_likelihood.py::build` calls it, emits
**10** labels:

    ['H0', 'log10n0', 'delta', 'sigma_kde',
     'log10n0_c2', 'delta_c2', 'sigma_kde_c2',
     '$\mu_{\rm G}$_c2', '$\mu_\chi$_c2', 'fcat_2']

| check | measured | verdict |
|---|---|---|
| **R1** the space is built | 10 labels against Analysis 8's 9 | PASS |
| **R2** the Gaussian mean appears per catalog | `$\mu_{\rm G}$_c2` present | PASS |
| **R3** its bounds carry over from the base slot | \([20, 50]\) | PASS |
| **R4** the spin mark still appears | `$\mu_\chi$_c2` at \([-1, 1]\) | PASS |
| **R5** no darksirens change is required | the generic per-catalog resolver accepts `G.mu_c2`; worktree clean at `af896ca` | PASS |

**Liveness is NOT shown by this gate.** An emitted coordinate can be dead in the
likelihood — that is precisely the defect Analysis 8's Gate A caught. Gaussian-mean
liveness is registered as closure gate **15.6** and is **mandatory**.

## Registered grid

Registered before any run. One refinement of one axis is allowed **at most**;
there is no refine-and-rerun loop, and a result that needs a second refinement
is reported as such.

| axis | nodes | range | spacing |
|---|---|---|---|
| \(f_{\rm AGN}\) | **41** | \([0, 1]\) | 0.025 |
| \(\Delta\mu_\chi\) | **61** | \([-0.20, +0.25]\) | 0.0075 — Analysis 8's axis, unchanged; widened only if containment requires it, and the widening is reported |
| \(\Delta\mu_{\rm G}\) | **21** | \([-10, +10]\,M_\odot\) | \(1\,M_\odot\); brackets the planted \(+5\) with 5 nodes of headroom above it |

Cell counts: A10-S 41; A10-χ \(41 \times 61 = 2{,}501\); A10-M
\(41 \times 21 = 861\); **A10-J \(41 \times 61 \times 21 = 52{,}521\)**. At the
measured rita A100-80 rate of 3.0203 s/eval that is 0.03 / 2.10 / 0.72 /
**44.1** GPU-h, **46.9 GPU-h** for all four.

## Registered tolerances

Residuals are quoted in **ULP** of the returned value alongside the absolute
bound, as Analyses 8 and 9 did. At the seed-100 scale
(\(\ln L \approx -4.2\times10^{3}\)) **1 ULP = 9.094947017729282e-13**.

- Closure reductions: criterion \(\le 10^{-6}\) **absolute**; same-hardware
  expectation \(\le 4\) ULP (Analysis 8's Gate A measured 2 ULP; Analysis 9's
  cross-hardware worst cell 6 ULP, which was two 2-ULP term errors compounding).
- Endpoint identities (15.7): registered as **exactly 0.0**, not a tolerance.
- Bitwise-frozen checks (15.4): one distinct hex value per row, not a tolerance.
- Liveness (15.6): min \(|\Delta\ln L| > 10^{-3}\) for a one-node move.
- Figures against their JSON (D2): \(\le 10^{-12}\).

## Gate G — generation (registered, NOT RUN)

Status: **NOT RUN**. No mock, no injections, no PE under the new mark. The
generator extension is in progress.

The spin mark's trick does not transfer. `--dmu_chi_agn` shifts an
**already-drawn** value, so the RNG stream, the host labels, the masses, the sky,
the distances and the detected set come through untouched. A peak-location mark
needs a **branch-conditioned draw** — the AGN branch draws its Gaussian component
about \(40\,M_\odot\), the GAL branch about 35 — which changes the masses and
therefore, through \(\rho_{\rm opt} \propto \mathcal{M}_{\rm det}^{5/6}/d_L\),
the detected set. **A new event set is generated; the Analysis-8 file is not
reusable.**

- [ ] **G1.** The new flag `--dmu_G_agn` is added to
      `working/data/generate_dataset.py` with default **0.0**, and its default
      path is **bitwise unchanged**: regenerating seed 100 with
      `--dmu_G_agn 0.0 --dmu_chi_agn 0.10` reproduces the existing
      `working/data/seed100/events/events_marked_dmu0p10.h5` **bitwise** on every
      dataset and attribute that is not the new flag's own record. This is the
      control, and it is the gate on the generator edit.
- [ ] **G2.** The two-mark mock is generated at the pin `af896ca` on seed 100 to
      `working/data/seed100/events/events_marked_dmu0p10_dmuG5.h5`, with
      \(\Delta\mu_\chi = +0.10\) and \(\Delta\mu_{\rm G} = +5\).
- [ ] **G3.** The branch-conditioned mass draw is stated explicitly in the
      events file's `metadata_json` — which branch draws from which \(\mu_{\rm G}\),
      at what point in the RNG stream, and what is shared — so the mark is
      readable off the data without reading the generator.
- [ ] **G4.** The **realised** marks are measured and recorded beside the
      planted ones, in both channels. (The current mock's realised
      \(\Delta\mu_\chi\) is \(+0.111924 \pm 0.006815\) against a planted
      \(+0.100000\); \(\Delta\mu_{\rm G}\) needs the same treatment, measured on
      the *drawn* population, not on the detected set, with the detected-set
      value reported separately because selection shifts it.)
- [ ] **G5.** No existing file under `working/data/seed100/**` is overwritten or
      modified; the new event set is a new path.

## Gate V — mock validation (registered, NOT RUN)

Status: **NOT RUN** — blocked by Gate G. The owner's twelve checks, each
measured on the generated file, not asserted from the command line.

- [ ] **V1.** One realisation, **seed 100**, and the seed is the one recorded in
      the file.
- [ ] **V2.** The **same LSS and survey catalogs** as Analyses 0–9: identical
      catalog paths and hashes, identical pixelisation, no re-draw.
- [ ] **V3.** The GAL branch draws its Gaussian component from
      \(\mu_{\rm G} = \mathbf{35}\,M_\odot\).
- [ ] **V4.** The AGN branch draws its Gaussian component from
      \(\mu_{\rm G} = \mathbf{40}\,M_\odot\).
- [ ] **V5.** **Shared peak fraction 0.90** — both branches, measured on the
      drawn masses, not read off the config alone.
- [ ] **V6.** **Shared power law**: slope, \(m_{\min}\), \(m_{\max}\) and both
      taper widths identical in the two branches.
- [ ] **V7.** **Shared \(q\)**: the mass-ratio distribution is identical in the
      two branches (\(\beta\) common).
- [ ] **V8.** **Spin difference \(+0.10\)**: the \(\chi_{\rm eff}\) mark is
      unchanged from Analyses 8 and 9, realised value recorded.
- [ ] **V9.** **v3 PE unchanged**: the measurement family is the v3 family, bit
      for bit the same code path as Analyses 8 and 9.
- [ ] **V10.** **Observed-SNR detection unchanged**: `observed-data` with
      \(\rho_{\rm obs} \ge 8\); the rule, the PSD and the threshold are identical.
- [ ] **V11.** **Analyses 0–9 files unchanged**: no file under
      `working/data/seed100/**` other than the new event set (and its rejected
      sample and meta) has a changed hash or mtime; no Analysis 0–9 result moves.
- [ ] **V12.** **The control behaves**: the \(\Delta\mu_{\rm G} = 0\) regeneration
      is bitwise equal to `events_marked_dmu0p10.h5` (Gate G1), and the two-mark
      mock differs from it **only** where the mass mark can reach — masses,
      SNRs, the detected set and everything they propagate to — with the
      difference reported rather than assumed.

## Gate B — selection support (registered, NOT RUN)

Status: **NOT RUN** — blocked by Gate G. **This gate is not optional for a mass
mark.** The detection rule is `observed-data` with \(\rho_{\rm obs} \ge 8\) and
\(\rho_{\rm opt} \propto \mathcal{M}_{\rm det}^{5/6}/d_L\), so a peak-location
mark changes detectability branch by branch. The spin mark did not:
\(\chi_{\rm eff}\) does not enter \(\rho_{\rm opt}\), which is why Analyses 8
and 9 could reuse `injections_targeted.h5` without re-deriving its support.

- [ ] **B1.** \(N_{\rm eff}\) of the existing `injections_targeted.h5` is
      measured under **both** branch populations separately — GAL at
      \(\mu_{\rm G} = 35\), AGN at \(\mu_{\rm G} = 35 + \Delta\mu_{\rm G}\) — at
      \(\Delta\mu_{\rm G} \in \{-10, -5, 0, +5, +10\}\), i.e. both ends of the
      axis and the planted cell, crossed with representative \(f_{\rm AGN}\) and
      \(\Delta\mu_\chi\) nodes (including \(f = 1\), where the AGN branch carries
      the whole selection integral, and the \(\Delta\mu_\chi\) axis ends).
- [ ] **B2.** The hard `selection_neff_guard` (threshold 5000) is cleared over
      the cells that carry the posterior, with the margin quoted at 90%, 99%,
      99.9% and 99.999% of the mass (Analysis 8's arm J: 85.5x, 70.5x, 55.3x,
      30.7x).
- [ ] **B3.** The importance-weight tails are reported, not just \(N_{\rm eff}\):
      the maximum single-injection weight share at the axis ends, where the
      proposal is furthest from the shifted mass distribution.
- [ ] **B4.** Any guard-rejected region is bounded: the rejected posterior mass
      is reported with a monotonicity-free bound.
- [ ] **B5.** If B1–B3 fail on the existing injection set, **one** new targeted
      injection set is generated for the marked branch, and only one. Reweighting
      a proposal that never covered the shifted mass distribution is not an
      acceptable substitute, and the failure is reported to the owner before any
      inference.

## Gate 15 — closure identities (registered, NOT RUN)

Status: **NOT RUN** — blocked by Gate G. Residuals in ULP alongside the absolute
bound, per **Registered tolerances** above.

- [ ] **15.1 both marks zero → shared-population K=2.** At
      \(\Delta\mu_\chi = 0\) **and** \(\Delta\mu_{\rm G} = 0\) the two-mark model
      reproduces the shared-population (spatial-only, Analysis-2-shape) K=2
      likelihood over a registered set of \(f\) nodes. The reference must be a
      genuinely different model — 8 sampled labels, no `_c2` population
      coordinate at all — not the same model evaluated at zero.
- [ ] **15.2 mass mark zero → the spin architecture.** At
      \(\Delta\mu_{\rm G} = 0\) the two-mark model reproduces the one-mark
      (Analysis-8 arm J) model cell for cell on the shared
      \((f, \Delta\mu_\chi)\) lattice, on the **same** data. Where the data
      differ (the new event set), the comparison is the Analysis-8 *builder* on
      the new file, not Analysis 8's recorded cube.
- [ ] **15.3 spin mark zero → the mass-only architecture.** At
      \(\Delta\mu_\chi = 0\) the two-mark model reproduces the mass-mark-only
      (A10-M) build cell for cell on the shared \((f, \Delta\mu_{\rm G})\)
      lattice. Same criterion.
- [ ] **15.4 \(f_{\rm AGN} = 0\): both marks inert.** With the AGN weight at
      zero, `logL` is **bitwise frozen** as \(\Delta\mu_\chi\) and
      \(\Delta\mu_{\rm G}\) vary over their full axes — one distinct hex value
      per row — in the total, the PE term and the selection term alike.
- [ ] **15.5 \(f_{\rm AGN} = 1\): both marks live.** The same rows give a
      distinct value per node in **both** mark axes, and the spread is reported
      for each.
- [ ] **15.6 Gaussian-mean liveness — MANDATORY.** Moving \(\Delta\mu_{\rm G}\)
      off its planted value by one grid node (\(1\,M_\odot\)) changes
      \(\ln L\) by far more than the 15.1 residual: registered threshold
      min \(|\Delta\ln L| > 10^{-3}\), checked in the **PE term and the selection
      term separately** as well as in the total. The same check is run for
      \(\Delta\mu_\chi\). **15.1–15.5 and 15.7 all pass for free if a new
      coordinate is silently ignored**, which is exactly the defect Analysis 8's
      Gate A caught, so 15.6 is not optional and must be run for
      \(\Delta\mu_{\rm G}\) specifically — the coordinate the resolver has never
      carried in production. A failure here stops the analysis; it is not a
      diagnostic to be noted and worked around.
- [ ] **15.7 K=1 endpoint identities.** \(f = 0\) equals a K=1 GAL build and
      \(f = 1\) equals a K=1 AGN build — the latter built at
      \(\mu_{\rm G} = 40\), \(\mu_\chi = +0.10\) — **exactly 0.0** in the total,
      the PE term, the selection term and `log_mu`. This is the identity that
      validates any K=2 wiring; it is registered as an exact equality, not a
      tolerance.

## Gates A10-S, A10-χ, A10-M, A10-J — the four fixed-\(H_0\) arms (registered, NOT RUN)

Status: **NOT RUN** — blocked by Gate G. \(H_0\) is pinned at 67.74 in all four.

| gate | arm | free | cells | criterion |
|---|---|---|---|---|
| **A10-S** | spatial only | \(f_{\rm AGN}\) | 41 | \(f_{\rm AGN}\) recovers the planted 0.30 and the realised value at 90%; this is the mark-free baseline every later width is quoted against |
| **A10-χ** | spin mark | \(f_{\rm AGN}, \Delta\mu_\chi\) | 2,501 | reproduces the Analysis-8 arm-J architecture on the shared lattice within the 15.2 bound, and recovers both truths at 90% |
| **A10-M** | mass mark | \(f_{\rm AGN}, \Delta\mu_{\rm G}\) | 861 | \(\Delta\mu_{\rm G}\) recovers the planted \(+5\,M_\odot\) **and** the realised mark at 90%, with the realised value carried everywhere |
| **A10-J** | joint | \(f_{\rm AGN}, \Delta\mu_\chi, \Delta\mu_{\rm G}\) | 52,521 | all three recover at 90%; \(\rho(\Delta\mu_\chi, \Delta\mu_{\rm G})\), \(\rho(f, \Delta\mu_\chi)\) and \(\rho(f, \Delta\mu_{\rm G})\) are reported; the two marks are shown to be separable rather than a single re-labelled direction |

- [ ] **A10-1.** Every plotted interval, band and contour is **90%**.
- [ ] **A10-2.** Both truths — planted and realised — are carried in every table
      and figure.
- [ ] **A10-3.** Widths are quoted **differentially** against A10-S and against
      Analysis 8, never as an absolute calibrated gain from one realisation.
      This is a single-seed science exploration; no error bar on a gain is
      claimed from it.
- [ ] **A10-4.** \(H_0\) is quoted differentially if it is ever quoted at all:
      seed 100's own draw sits high (Analysis 2 recovered 69.217 unmarked), and
      that is a property of the realisation, not a bias.
- [ ] **A10-5.** Each arm's cube is checked for duplicate rows and its guard
      rejections bounded, as in Analysis 9's Gate B.
- [ ] **A10-6.** Containment: the 90% credible region of each mark lies strictly
      inside its registered axis. If it does not, **one** refinement or widening
      of that axis is allowed, reported as such; a second is a FAIL to report,
      not a third run.

## Gate D — figures and report (registered, NOT RUN)

- [ ] **D1.** Every result figure ships as **both** `.pdf` and `.png`, rendered
      by a deterministic `make_figures.py` that reads `results/` and
      `diagnostics/` only.
- [ ] **D2.** Drawn values agree with the JSON they came from to \(\le 10^{-12}\).
- [ ] **D3.** `REPORT.md` is written from `results/` and `diagnostics/` alone;
      nothing is recomputed for it.

`figs/` and `results/` hold only `.gitkeep`. Nothing has been measured under the
new mark, so nothing has been drawn.

## The fixed-\(H_0\) owner gate

Status: **NOT REACHED**. It sits behind Gates G, V, B, 15 and the four arms.

All six conditions must hold. If any fails, **STOP** and report; do not proceed
to the cosmology arms.

1. **Selection is valid.** Gate B passes on the grid that carries the posterior,
   or a targeted injection set was generated and Gate B passes on it.
2. **Closure holds.** Gates 15.1–15.7 pass, including the exact endpoint
   identities.
3. **The mass mark is live.** Gate 15.6 passes for \(\Delta\mu_{\rm G}\)
   specifically, in the PE and selection terms separately.
4. **\(\Delta\mu_{\rm G} = +5\) is meaningfully recovered.** Not merely contained
   at 90%, but resolved: the 90% interval excludes \(\Delta\mu_{\rm G} = 0\), and
   the planted and realised values both sit inside it.
5. **The mass mark adds information beyond spin-only.** A10-J's \(f_{\rm AGN}\)
   width is measurably tighter than A10-χ's on the same data, and the
   \(\rho(\Delta\mu_\chi, \Delta\mu_{\rm G})\) correlation shows the two marks
   are not one direction re-labelled.
6. **No pathological degeneracy.** No ridge that runs to an axis edge, no
   multimodality that the grid cannot resolve, no \(f_{\rm AGN}\) posterior that
   piles at 0 or 1.

## Gates C10-S, C10-J — the cosmology arms (registered, NOT REACHED)

Status: **NOT REACHED** — gated behind the fixed-\(H_0\) owner gate, and never
run in the same campaign as the fixed-\(H_0\) arms.

| gate | arm | free | criterion |
|---|---|---|---|
| **C10-S** | mark-free \(H_0\) | \(H_0, f_{\rm AGN}\) | the spatial-only \(H_0\) baseline on the two-mark mock; every marked width is quoted against this, differentially |
| **C10-J** | full | \(H_0, f_{\rm AGN}, \Delta\mu_\chi, \Delta\mu_{\rm G}\) | \(H_0\) recovers seed 100's own draw; the marked/unmarked width ratio is reported with its mechanism decomposed |

- [ ] **C10-1. The two channels are separated.** A mass mark reaches \(H_0\) by
      two distinct routes and reporting one number for the gain would be a wrong
      attribution:
      **(i) event-level tracer routing** — the Analysis-9 mechanism, in which the
      mark sharpens \(P_i({\rm AGN})\) and re-sorts events between the GAL and
      AGN redshift structures (Analysis 9 measured the global-mixture factor at
      0.99997 at 68%, i.e. the gain was routing, not a tighter global \(f\));
      **(ii) the spectral-siren coupling** \(m_{\rm det} = (1+z)\,m_{\rm src}\) —
      a branch-dependent peak location gives the two branches two different
      spectral-siren rulers, constraining \(H_0\) with no reference to the galaxy
      catalog at all. \(\chi_{\rm eff}\) has no analogue of (ii): it does not
      redshift and does not enter \(\rho_{\rm opt}\).
- [ ] **C10-2.** The routing share is measured as Analysis 9 measured it: the
      fixed-\(f\) width ratio, the \(|\Delta P_i|\) distribution, the crossers,
      and the event-term / selection-term split of the added \(H_0\) curvature.
- [ ] **C10-3.** The spectral-siren share is measured against a control in which
      the branch labels are scrambled, so that the peak-location information
      survives but the tracer routing does not.
- [ ] **C10-4.** \(H_0\) is quoted **differentially** throughout. Seed 100's draw
      sits high; that is the realisation, not a bias, and one realisation
      calibrates nothing.

