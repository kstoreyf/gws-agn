# Analysis 9 gates

The concise pass/fail ledger. Detailed evidence belongs in `diagnostics/`,
`results/` and `REPORT.md`. Every criterion below is registered **before** the
run; a measurement that misses one is a FAIL to be diagnosed, not a bound to be
widened.

**All gates are closed. The owner gate is REACHED** (2026-09-20): both arms are
measured, every criterion below carries the number that decided it, `REPORT.md`
is written, and nothing further runs without an owner decision.

| gate | status | the number that decides it |
|---|---|---|
| 0 provenance and scope | PASS | darksirens HEAD `af896ca`, clean, 9 `mixture_pop_params` references; free = `H0`, `fcat_2`, `$\mu_\chi$_c2` |
| 9 closure checks | PASS | 9.1 max 2.00 ULP; 9.2 max 2.00 ULP; 9.3 \(f=0\) bitwise frozen, \(f=0/1\) endpoint identity exactly 0.0; \(H_0\) liveness min \(|\Delta\ln L|\) = 0.764 |
| W the J9 \(H_0\) window | PASS | S9's own marked marginal: 7.53e-08 and 9.24e-08 of the peak at 63 and 76, 2.00e-08 of the mass outside (criterion 1e-6) |
| A anchor equivalence + \(H_0\) liveness | PASS | whole 2501-cell slab: 2165 bitwise, max **6 ULP** = 5.457e-12 (1.25e-15 relative) against the 1e-6 bound; mask identical; \(\Delta\ln L_{\max}\) exactly 0.0 |
| A(S) the S9 line vs Analysis 8 arm S | PASS | 19 of 21 shared nodes bitwise, worst 1.819e-12 = **2.00 ULP** |
| B cube integrity and selection | PASS | 78,310 cells, no duplicate rows; S9 0 rejected (46.35x floor), J9 809/70,028 (1.155%); monotonicity-free bound **8.96e-09** |
| S11 what the mark does | PASS | matched-lattice \(H_0\) width ratio **0.8879** (68%) / **0.8996** (90%), median +0.00041 |
| C the measurement | PASS | \(f\) and \(\Delta\mu_\chi\) recover both truths at 68%; \(H_0\) 69.0905, **−0.1265** from Analysis 2's unmarked 69.2170 |
| D figures and report | PASS | 3 figures as `.pdf` **and** `.png`, drawn-vs-JSON \(|d| \le\) 5.6e-17; `REPORT.md` written |
| owner | **REACHED** | seed 100 only, 1 realisation; nothing past closure without a decision |

## Scope lock

- [x] Analysis 9 is Analysis 8 with \(H_0\) freed, and nothing else.
- [x] Free parameters, exactly three: `H0`, `fcat_2` (\(f_{\rm AGN}\)),
      `mu_chi_c2` (the AGN branch's ABSOLUTE spin mean).
- [x] Seed 100 only, one realisation; nothing generated.
- [x] Complete catalogs only; the Gate-B marked mock and the existing targeted
      injections, reused exactly.
- [x] \(\Omega_{m,0} = 0.3075\); twelve base population parameters pinned by
      name; common mass, \(q\), spin width and redshift rate.
- [x] No mass mark, no incompleteness, no GP/HSGP, no extra intrinsic parameters.
- [x] No likelihood is reimplemented: the Analysis-8 builder is imported.
- [x] All GPU work on RITA via SLURM (`--partition=RITA-GPU --qos=rita
      --account=phy220048p --gres=gpu:a100-80:1`); the local H100 stays free and
      the driver refuses a GPU stage anywhere else.
- [x] Writes confined to this directory; `data/seed100/**` and the Analysis-8
      tree are read-only.

## Gate 0 — provenance and scope

Status: **PASS** (2026-09-19, CPU pre-flight, no GPU). Evidence:
`diagnostics/provenance.json`; the assertions are reproduced verbatim in
`STATE.md`.

| check | measured | verdict |
|---|---|---|
| PYTHONPATH carries darksirens-a8 | `/hildafs/projects/phy230014p/magana/src/darksirens-a8` | PASS |
| darksirens-a8 HEAD | `af896cae6f3f3dd1f87dec50046e3a8228f59b39`, worktree clean | PASS |
| tracer-dependent population blocks present | 9 references to `mixture_pop_params` in `darksirens/likelihood/core.py` | PASS |
| import actually resolves there | `.../darksirens-a8/darksirens/__init__.py` | PASS |
| pinned base `2b86a2d` is an ancestor | yes | PASS |
| four inputs present, byte sizes recorded | events 162743429, survey_gal 1650316659, survey_agn 19934497, selection 203346016 | PASS |
| parameter space | 9 labels; free = `H0`, `fcat_2`, `$\mu_\chi$_c2`; 6 survey nuisances at the Analysis-8 values; 13 pinned (12 population + \(\Omega_{m,0}\)) | PASS |
| Analysis-8 tree unmodified | no file under it has an mtime after its 2026-09-18 close | PASS |

## Gate W — the J9 \(H_0\) window, set by S9's own measurement

Status: **PASS** (2026-09-19, rita jobs 1296265 + 1310843, 8,282 cells,
6.96 GPU-h, median 3.0219 s/eval on two A100-80s). Evidence:
`results/s9_spatial.json` → `j9_window_check`.

| check | measured on S9's own marked \(H_0\) marginal | verdict |
|---|---|---|
| peak | 69.00 (median 69.0872, mode 69.00) | — |
| density at 63 | **7.53e-08** of the peak (criterion \(\le\) 1e-6) | PASS |
| density at 76 | **9.24e-08** of the peak | PASS |
| mass outside [63, 76] | **2.00e-08** | PASS |
| required window on the 0.5 lattice | [63.0, 76.0] — **no widening** | PASS |
| axis ends (50, 100) | 3.09e-36 and 9.76e-31 of the peak | PASS |

The proposal came from Analysis 2's UNMARKED marginal (7.67e-08 and 9.16e-08,
2.08e-08 outside). S9's own marked marginal gives 7.53e-08, 9.24e-08 and
2.00e-08 — the same to within a few percent, so the window stands as proposed
and J9 runs on the 28-node axis {63.0, 63.5, …, 76.0} ∪ {67.74}.

S9 has a job beyond its own result: it fixes the J9 window. The proposal
[63, 76] came from Analysis 2's UNMARKED seed-100 marginal (density 7.67e-08 and
9.16e-08 of the peak at the two edges, 2.08e-08 of the mass outside). That is a
different dataset and a different model, so it is a proposal, not a measurement.

- [x] **W1.** On **S9's own** \(H_0\) marginal — the marked mock, two tracer
      populations, \(\Delta\mu_\chi = 0\) — the density at 63 and at 76 is
      \(\le 10^{-6}\) of the peak (the Gate-B3 criterion), and the mass outside
      the window is reported whatever the verdict.
- [x] **W2.** (not fired — W1 passed.) If W1 fails, the window is **widened** on the 0.5 lattice until it
      passes, the widening is stated explicitly, and J9 runs on the widened axis.
      Every J9 node stays a strict S9 node position either way.

## Gate A — anchor equivalence and \(H_0\) liveness

Status: **PASS** (2026-09-19, the whole slab, rita job 1316233 chunk 0).
Evidence: `diagnostics/a9_gate_a_full_slab.json` (and, for \(H_0\) liveness,
`results/a9_closure_summary.json`).

| quantity, 2501 cells (2478 accepted in both) | measured | verdict |
|---|---|---|
| total \(\ln L\) | 2165 bitwise identical, 277 at 2 ULP, 35 at 4 ULP, **1 at 6 ULP**; max \(\|\Delta\|\) = 5.456968e-12 (1.3e-15 relative) | PASS |
| PE term | 2263 bitwise, 193 at 1 ULP, 22 at **2 ULP**; max 3.637979e-12 | PASS |
| selection term | 2359 bitwise, 118 at 1 ULP, 1 at **2 ULP**; max 3.637979e-12 | PASS |
| `log_mu` | max 1.776357e-15 = **1 ULP** | PASS |
| \(N_{\rm eff}\) | 2311 bitwise; max 5.47e-09 on values \(\sim\)7.7e5 = **63 ULP** = 7e-15 relative | PASS |
| guard threshold | all 2478 bitwise identical (5000.0) | PASS |
| **guard-rejected mask** | **identical**: 23 cells in both, all at \(f \ge 0.750\) and \(\mu_{\chi,c2} \ge +0.2350\); 0 cells differ | PASS |
| \(\ln L_{\max}\) and the MAP cell | \(-4211.84259284502\) at \((0.275, +0.1075)\) in both, \(\Delta\) **exactly 0.0** | PASS |
| \(f\) marginal | median 0.26165329946926646 vs 0.26165329946926197, \(\Delta\) 4.5e-15; 68% and 90% ends within 5.5e-15 | PASS |
| \(\mu_{\chi,c2}\) marginal | median 0.10738647338252748 vs 0.10738647338252715, \(\Delta\) 3.3e-16 | PASS |
| \(\rho(f, \mu_{\chi,c2})\) | \(-0.5543646830622379\) vs \(-0.5543646830621719\) | PASS |

**On the 6 ULP.** The 40-cell subset of closure 9.1 measured 2 ULP and that is
the expectation this gate was written against. Over 62x more cells the worst
total reaches 6 ULP in exactly **one** cell. It is not a new disagreement: the
PE and selection terms separately still cap at 2 ULP, and \(\ln L\) is
**exactly** their sum on both sides (max residual 0.0), so 6 ULP is two 2-ULP
parts plus the rounding of the sum. The pre-registered criterion is the
absolute bound 1e-6, cleared by six orders of magnitude.

The earlier check (closure 9.1) used a 40-cell registered subset. Gate A proper
is the WHOLE 2501-cell slab, `--stage gate_a_full`, and it compares the total,
the PE term, the selection term, `log_mu`, \(N_{\rm eff}\), the threshold, the
guard-rejected mask and the recomputed \(f\) and \(\mu_{\chi,c2}\) marginals.

The \(H_0\) axis is anchored **exactly** on 67.74, so the anchor slab is cell for
cell Analysis 8's arm-J grid and the gate costs no extra evaluation.

- [x] **A1 cell equivalence.** The 2501-cell \(H_0 = 67.74\) slab reproduces
      `analysis_8/results/arm_J_joint.h5`. Pre-registered: max \(|\Delta \ln L|
      \le 1\mathrm{e}{-6}\) absolute **and** \(\le 2.4\mathrm{e}{-10}\) relative,
      with the residual ALSO reported in ULP (1 ULP = 9.0949470177292824e-13 at
      \(\ln L \sim -4.2\mathrm{e}3\)). On identical hardware the criterion is
      \(\le 4\) ULP; Analysis 8's own Gate A measured 2 ULP. The looser absolute
      bound exists only because arm J ran on the local H100 NVL and this runs on
      a rita A100-80.
- [x] **A2 rejected mask identical.** The same 23 cells are guard-rejected, all
      at \(f \ge 0.750\), \(\mu_{\chi,c2} \ge +0.2350\).
- [x] **A3 marginals reproduced.** \(f\) and \(\mu_{\chi,c2}\) medians and 68/90%
      intervals recomputed from the slab through Analysis 8's own `_marginals`
      and `marginal_ci` agree with `arm_J_joint.json` to \(\le 1\mathrm{e}{-4}\)
      (Analysis 8: \(f\) 0.261653, 68% [0.216174, 0.308298]; \(\mu_{\chi,c2}\)
      0.107386, 68% [0.088587, 0.127960]).
- [x] **A4 \(H_0\) liveness.** At fixed \((f, \mu_{\chi,c2}) = (0.275, +0.1075)\),
      moving \(H_0\) off 67.74 by one node and to each axis end changes \(\ln L\)
      by far more than the A1 tolerance (registered threshold: min
      \(|\Delta \ln L| > 10^{-3}\)). **A1–A3 pass for free if the new coordinate
      is ignored** — that is the exact defect Analysis 8's Gate A caught, and
      `H0` was already in the label list, so A4 is not optional.

Gate A must pass before the remaining 35 slabs are committed.

## Gate A(S) — the S9 line at \(H_0 = 67.74\) against Analysis 8's arm S

Status: **PASS**, free (no extra evaluation). Evidence:
`diagnostics/s9_vs_a8_arm_s.json`.

Analysis 8's arm S used the 101-node \(f\) convention and S9 uses the 41-node
one, so they share 21 nodes. **19 of 21 are bitwise identical and the worst
disagreement is 1.8189894035458565e-12 = 2.00 ULP** — the same cross-hardware
residual (H100 NVL → A100-80) as closure check 9.1 and as Analysis 8's own
same-node Gate A.

## Gate 9 — specification section 9 closure checks

Status: **PASS** (2026-09-19, rita job 1296051, one A100-80 PCIe, 9 min 23 s).
Evidence: `diagnostics/closure_checks.json`, `diagnostics/_a9_closure_main.json`,
`diagnostics/_a9_closure_k1.json`, `results/a9_closure_summary.json`; driver
`scripts/a9_closure.py`, harness `scripts/submit_a9_closure_rita.sbatch`.

Residuals are quoted in ULP of the returned value. At the seed-100 scale
1 ULP = 9.094947017729282e-13. **Analysis 8's arm J ran on a local H100 NVL and
this ran on a rita A100-80, so every figure below is a CROSS-HARDWARE residual.**

| check | measured | verdict |
|---|---|---|
| **9.1** fixed-\(H_0\) reduction vs `arm_J_joint.h5`, 40-cell registered subset | 37/40 bitwise identical (5 of those are the shared \(-\infty\) of the guard corner); 3 cells at exactly **2 ULP**. max \(\|\Delta\ln L\|\) = 1.8189894035458565e-12 = **2.00 ULP** total, **1 ULP** in the PE term and **1 ULP** in the selection term separately | PASS |
| 9.1 MAP cell \(f=0.275\), \(\mu_{\chi,c2}=+0.1075\) | \(-4211.84259284502\), diff **0.0**, **0 ULP**, against the recorded \(\ln L_{\max}\) | PASS |
| 9.1 guard-rejected mask | identical on all 40 cells, including the four straddling pairs (rejected cell + accepted neighbour) at \((f,\mu)\) = (1.000, +0.2350)/(1.000, +0.2275), (0.925, +0.2350)/(0.900, +0.2350), (0.750, +0.2500)/(0.725, +0.2500) | PASS |
| **9.2** zero-mark reduction to the spatial-only model, 35 \((H_0, f)\) pairs over \(H_0 \in\) {60.24, 64.74, 67.74, 71.74, 77.74} | 27/35 bitwise identical, 8 at exactly **2 ULP**; max \(\|\Delta\ln L\|\) = 1.8189894035458565e-12 in the total, the PE term and the selection term alike | PASS |
| 9.2 the reference really is a different model | spatial-only carries **8** sampled labels and **no** `mu_chi_c2` at all (Analysis 8's `K2_OLD` shape, same builder, same `data` object); the marked model carries 9 | PASS |
| 9.2 vs the recorded `arm_S_spatial.h5` | max 2 ULP over the \(H_0 = 67.74\) line; \(f = 0.26\) returns \(-4233.939503109996\), the recorded arm-S \(\ln L_{\max}\), exactly | PASS |
| **9.3** \(f_{\rm AGN} = 0\): \(\mu_{\chi,c2}\) inert | **bitwise frozen** at all three \(H_0\) (64.74, 67.74, 71.74): one distinct `logL` hex over 5 \(\mu\) nodes spanning \([-0.20, +0.25]\), and the PE and selection terms are each frozen too | PASS |
| 9.3 \(f_{\rm AGN} = 1\): only the AGN branch | 5 distinct values per row, spread 843.4 / 847.6 / 846.3 in \(\ln L\) at the three \(H_0\) | PASS |
| 9.3 endpoint identity vs single-branch builds | \(f=0\) equals K=1 GAL and \(f=1\) equals K=1 AGN **exactly 0.0** in the total, the PE term, the selection term AND `log_mu`, at every one of the three \(H_0\) | PASS |
| \(H_0\) liveness (the Analysis-8 trap) | \(\Delta\ln L\) = +0.7638, \(-1.0604\), \(-33.6201\), \(-23.9563\) at \(H_0\) = 68.24, 67.24, 60.24, 77.74; min \(\|\Delta\ln L\|\) = 0.764, \(7.6\times10^{11}\) times the 9.1 residual | PASS |

**Cost model, measured on the rita A100-80 PCIe (81920 MiB), not extrapolated.**

| quantity | measured |
|---|---|
| build (`load_all_data` + parameter space + `make_likelihood`) | **29.22 s** (24.06 s of it `load_all_data` on a cold cache in the first attempt, which built in 50.16 s) |
| spatial-only build reusing the same `data` object | 3.37 s |
| first evaluation (includes the JIT trace) | 11.16 s |
| **steady state** | **3.0188 s/eval** (median of 20; mean 3.0189, sd 0.0020, range [3.0161, 3.0234]) |
| peak GPU memory | **34254 MiB** (nvidia-smi, this PID; JAX `peak_bytes_in_use` 19.59 GB, `peak_pool_bytes` 35.44 GB) |
| peak host RSS | 8.4 GB |

3.02 s/eval is **faster** than both reference points: 3.74 s/eval for Analysis 2's
K=2 joint scan on an A100 and the 3.5-4 s/eval this task expected. It is 1.77x
the 1.7095 s/eval Analysis 8 measured for the identical likelihood on an H100 NVL.

## Gate B — cube integrity and selection validity

Status: **PASS** (2026-09-20). Evidence: `results/s9_spatial.{h5,json}`,
`results/j9_marked.{h5,json}`, `diagnostics/a9_guard.json`,
`diagnostics/a9_guard_sensitivity.json`.

- [x] **B1 coverage.** S9: 202 rows, 8,282 cells, 6.96 GPU-h. J9: all
      28 x 41 = 1,148 rows, 70,028 cells, 58.72 GPU-h, median 3.0203 s/cell.
      `duplicate_rows` is empty in both arms — no row was computed twice, so the
      bit-for-bit re-agreement clause never fired.
- [x] **B2 selection guard.** S9 rejects **0** of 8,282 cells (min
      \(N_{\rm eff}/{\rm threshold}\) = 46.35). J9 rejects **809** of 70,028
      (1.155%), every one at \(f \ge 0.575\) and \(\mu_{\chi,c2} \ge +0.220\),
      spread over the whole \(H_0\) window. All 809 are listed with their
      \(N_{\rm eff}\) and threshold. The fill bound is
      **8.42e-51** of the posterior mass, and the fill is verified to be an
      over-estimate on **809 of 809** cells.
      **A finding, recorded rather than smoothed:** Analysis 8 at fixed \(H_0\)
      found 11 of 11 columns falling into the guard corner. Over the whole cube
      **25 of 348 columns rise instead**, all at \(H_0 \le 65.0\) — the low
      corner of the window, where the \(H_0\) marginal is already 3.3e-8 of its
      peak. The rises are at most 5.3 nats, and on the 376 accepted boundary
      cells the estimator's own variance `sigma2_total` is 199.9 (median) to
      280.7 (max), a 1-sigma of **14.14 nats**, against 8.21 at the posterior
      peak, and \(N_{\rm eff}\) there is 5007 at worst and 5549 at the median
      against the 5000 floor, with 308 of 376 cells within 1.2x of it. The
      largest rise, 5.28 nats, is inside that noise: monotonicity is not
      resolved on the boundary, which is what the guard is for.
      A bound that **drops the monotonicity assumption entirely** — logL allowed
      to keep climbing at the largest one-step rise on the boundary shell, for
      all 5 grid steps of the rejected region's depth — gives **8.96e-09**, still
      clear of 1e-6. Compounding the largest gradient found three shells out
      (19.4 -> 20.7 nats/step, set by the \(f = 1\) edge and in \(H_0\), not the
      direction of approach) would give 6.3e-6; that extrapolation is not
      physically motivated and is reported for completeness, not adopted.
- [x] **B3 the grids do not clip the posterior.** J9 edge density over peak:
      \(H_0\) 3.31e-08 / 2.18e-08; \(f\) 3.05e-21 / 0.0; \(\mu_{\chi,c2}\)
      3.68e-23 / **9.45e-07**. S9: \(H_0\) 3.09e-36 / 9.76e-31, \(f\) 1.31e-12
      / 0.0. All pass. **The \(\mu_{\chi,c2}\) top edge clears 1e-6 by only 6%**
      — J9 sits at the edge of Analysis 8's \(\mu\) range in a way the fixed-
      \(H_0\) arm did not. The axis is Analysis 8's own and is pinned by the
      matched-axis requirement, so it is not widened here; it is named as the
      one axis a successor run should extend.
- [x] **B4 one code, one checkout.** Every checkpoint header carries
      `af896cae6f3f3dd1f87dec50046e3a8228f59b39`; `assemble` asserted it and
      would have refused a mixture.

## Gate S11 — the comparison that matters

Status: **PASS** (2026-09-20). Evidence: `results/section_11_comparison.json`,
`diagnostics/a9_h0_matched_lattice.json`.

| quantity | 68% width | 90% width | median | MAP |
|---|---|---|---|---|
| \(p(H_0)\) S9 spatial-only | 2.00160 | 3.30183 | 69.08719 | 69.00 |
| \(p(H_0)\) J9 marked | **1.82274** | **3.07988** | 69.09053 | 69.00 |
| ratio J9 / S9 (S9 full axis) | 0.9106 | 0.9328 | +0.00334 | 0.00 |
| ratio J9 / S9 (**matched lattice**) | **0.8879** | **0.8996** | +0.00041 | 0.00 |
| \(p(f_{\rm AGN})\) S9 | 0.096775 | 0.159199 | 0.269472 | 0.275 |
| \(p(f_{\rm AGN})\) J9 | **0.093051** | **0.153631** | 0.263927 | 0.275 |
| ratio J9 / S9 | 0.9615 | 0.9650 | −0.005545 | 0.00 |
| \(p(\Delta\mu_\chi)\) A8 fixed \(H_0\) | 0.0393721 | 0.0650741 | 0.107386 | 0.1075 |
| \(p(\Delta\mu_\chi)\) J9 free \(H_0\) | **0.0394102** | **0.0653427** | 0.108275 | 0.1075 |
| ratio J9 / A8 | 1.00097 | 1.00413 | +0.000889 | 0.00 |

- [x] widths quoted at 68% AND 90%, with the median and MAP shift;
- [x] axes matched and **asserted**: `f_axis_shared` true, the
      \(\mu_{\chi,c2}\) axis is Analysis 8's own object, and
      `every_J9_H0_node_is_an_S9_node` true (checked bitwise). Because S9
      integrates on 202 nodes and J9 on 28, `diagnostics/a9_h0_matched_lattice.json`
      re-marginalises **S9's own cube on exactly the 28 J9 nodes**: the coarser
      lattice *widens* S9 by 1.026 (68%) and 1.037 (90%), so the full-axis ratio
      **under-states** the sharpening and 0.888 / 0.900 is the number to quote.
- [x] direction of information flow, as measured: the mark sharpens \(H_0\) by
      11% (68%) and 10% (90%) and sharpens \(f_{\rm AGN}\) by 4%; freeing
      \(H_0\) costs \(\Delta\mu_\chi\) **0.10% at 68% and 0.41% at 90%** —
      essentially nothing. The spin mark and the cosmology are very nearly
      orthogonal here: \(\rho(H_0, \Delta\mu_\chi) = +0.025\).

## Gate C — the measurement

Status: **PASS** (2026-09-20). Evidence: `results/j9_marked.{h5,json}`,
`results/s9_spatial.{h5,json}`, `results/section_11_comparison.json`,
`diagnostics/a9_guard.json`, `diagnostics/a9_guard_sensitivity.json`.

Every interval at 68% **and** 90%; plots show 90%.

| arm | coordinate | median | 68% | 90% | MAP |
|---|---|---|---|---|---|
| S9 | \(H_0\) | 69.0872 | [68.0865, 70.0881] | [67.4375, 70.7393] | 69.00 |
| S9 | \(f_{\rm AGN}\) | 0.26947 | [0.22195, 0.31872] | [0.19026, 0.34946] | 0.275 |
| J9 | \(H_0\) | 69.0905 | [68.1740, 69.9967] | [67.5978, 70.6777] | 69.00 |
| J9 | \(f_{\rm AGN}\) | 0.26393 | [0.21828, 0.31133] | [0.18892, 0.34255] | 0.275 |
| J9 | \(\Delta\mu_\chi\) | 0.10828 | [0.08937, 0.12878] | [0.07815, 0.14350] | 0.1075 |

- [x] **C1 \(H_0\).** Planted 67.74 is **outside** the 68% and **inside** the 90%
      in both arms; the offset is +1.347 (S9) and +1.351 (J9). Differentially
      against Analysis 2 on the same realisation (69.217, unmarked): J9 sits
      **−0.127** away, i.e. the marked two-tracer measurement reproduces the
      unmarked one. Seed 100's own draw sits high; that is the realisation.
- [x] **C2 \(f_{\rm AGN}\).** Planted 0.30 and realised 0.295 are both inside
      the 68% and the 90% in both arms (offsets −0.036 and −0.031 for J9).
- [x] **C3 \(\Delta\mu_\chi\).** Planted +0.100 and realised +0.111924 both
      inside the 68% (offsets +0.00828 and −0.00365). `mu_chi_c2` is the AGN
      branch's ABSOLUTE spin mean; both spellings are in the output.
- [x] **C4 the cost of freeing \(H_0\), measured.** Width ratios against
      Analysis 8's fixed-\(H_0\) arm J: \(f\) 1.0101 (68%) / 1.0078 (90%),
      \(\Delta\mu_\chi\) 1.00097 (68%) / 1.00413 (90%). Correlations:
      \(\rho(H_0, f) = +0.0148\), \(\rho(H_0, \Delta\mu_\chi) = +0.0247\),
      \(\rho(f, \Delta\mu_\chi) = -0.5594\) (Analysis 8 at fixed \(H_0\):
      −0.5544). Freeing \(H_0\) is nearly free.
- [x] **C5 what the mark does to \(H_0\), measured.** On the matched lattice the
      marked \(H_0\) interval is 0.888x the spatial-only one at 68% and 0.900x
      at 90%, with the median moved by +0.0004. Against Analysis 2's unmarked
      \(H_0\) (width 1.9423 at 68%) the marked J9 width is 0.9385x.
      \(\rho(H_0, f)\) falls from +0.0678 (Analysis 2) through +0.0590 (S9) to
      +0.0148 (J9).
- [x] **C6 no result behind a rejected cell.** The rejected region begins at
      \(f = 0.575\) and \(\mu_{\chi,c2} = +0.220\) — **9 f-nodes and 10
      \(\mu\)-nodes beyond** the 90% credible edges (0.3425 and 0.1435). The
      accepted mass at \(f \ge 0.575\) is 3.36e-11 of the total. No result
      stands behind a rejected cell.
- [x] **C7 \(P(\Delta\mu_\chi \le 0) = 2.94\times10^{-11}\)** — a posterior
      probability under THIS model and THIS grid, with 0.0 interpolated between
      the neighbouring nodes (−0.0050 and +0.0025). It is **not** a sigma. The
      seed carries an accidental finite-realisation difference in the true
      mass-ratio distributions of the two host populations (KS p = 0.0096) while
      the model holds `q` identical in both branches; that creates no branch
      evidence directly, but posterior correlations among `q`, mass, distance and
      `chi_eff` can indirectly affect the spin-mark recovery (wording corrected
      2026-09-21).

## Gate D — figures and report

Status: **PASS** (2026-09-20). Evidence: `figs/fig_{s9_spatial,j9_corner,
section_11}.{pdf,png}` and `REPORT.md`.

- [x] `scripts/make_figures.py` renders `fig_s9_spatial`, `fig_j9_corner` and
      `fig_section_11` as **both** `.pdf` and `.png` under `figs/`, and prints a
      drawn-vs-JSON check on every annotated number (all |d| <= 5.6e-17).
- [x] every plotted interval, band and contour is **90%**;
- [x] `REPORT.md` written from `results/` and `diagnostics/` alone, nothing
      recomputed. It states the measurement (\(H_0\) 69.0905, \(f_{\rm AGN}\)
      0.2639, \(\Delta\mu_\chi\) 0.1083), quotes \(H_0\) **differentially**
      against Analysis 2's 69.2170 on the same realisation, gives the
      matched-lattice width ratios 0.8879 / 0.8996 with the mechanism
      (\(\rho(H_0,\Delta\mu_\chi) = +0.025\), so the gain is not a broken
      degeneracy), the 8.96e-09 monotonicity-free guard bound, and the
      limitations unsoftened.

## Owner gate

Status: **REACHED** (2026-09-20). Seed-100 closure is complete — both arms
measured (78,310 cells), every gate above PASS, figures and `REPORT.md` written —
and the next step is an owner decision, not a run.

Nothing proceeds past seed-100 closure without an explicit owner decision. Still
forbidden: seeds 101/102/103/105, any additional realisation including a repeat
draw of seed 100, effect-size ladders, mass marks, free common population
parameters, incompleteness, GP/HSGP population differences, GWTC data, and any
edit to `darksirens` or `generate_dataset.py`.

Claims one realisation cannot establish, and which this analysis will not make:
calibration across realisations, coverage, unbiasedness in expectation, any
sensitivity scaling with \(N\), anything about real BBH spins in AGN, and any
claim that the mark improves \(H_0\) in general.

The decision this result puts in front of the owner: the \(H_0\) interval
narrows by 11.2% (68%) and 10.0% (90%) when the spin mark is added, on **one**
realisation whose own \(H_0\) draw sits +1.35 from the planted value. That is a
scoping measurement, not a calibrated gain, and only additional realisations can
change which of the two it is. `REPORT.md` also names a wider \(\mu_{\chi,c2}\)
range (the top edge clears containment by 6%) and the `q` confound (KS
p = 0.0096) as the other two extensions the numbers justify.
