# Analysis 9 state

## Current status

**ANALYSIS 9 IS COMPLETE AND THE OWNER GATE IS REACHED** (2026-09-20). Both
production arms are measured, gates 0, 9, W, A, A(S), B, S11, C and D all PASS,
the three figures are rendered as `.pdf` **and** `.png`, and `REPORT.md` is
written from `results/` and `diagnostics/` alone — nothing was recomputed for it.
Nothing further runs without an owner decision.

| arm | cells | rejected | GPU-h | s/cell | jobs |
|---|---|---|---|---|---|
| **S9** spatial-only, \(\Delta\mu_\chi = 0\) | 8,282 | 0 | 6.96 | 3.0219 | 1296265 (0-6), 1310843 (7) |
| **J9** marked joint | 70,028 | 809 | 58.72 | 3.0203 | 1316233 (0-6), 1335284 (7) |

Every GPU evaluation ran on rita via SLURM with `--partition=RITA-GPU
--qos=rita --account=phy220048p --gres=gpu:a100-80:1`, two workers side by side.
The local H100 was never used.

**One infrastructure fault, twice.** J9 chunk 7 (`1316233_7`) died 5 s in with
`slurmstepd: error: TaskProlog failed status=1` — a node-side prolog fault before
any of our code ran, so the chunk produced nothing and there was no checkpoint to
resume. It was re-run as `sbatch --array=7 --export=ALL,N_CHUNKS=8
scripts/submit_a9_j9_rita.sbatch` (job 1335284) with the request unchanged, and
completed in 6:19:09. The identical fault had hit S9 chunk 7 (`1296265_7`) and
was handled the same way (job 1310843). This is not a queue-dodging resubmission:
the request, partition, qos, account and gres are byte-identical, and without it
123 rows / 7,503 cells would simply never have been computed.

### The headline

- \(H_0\) = **69.091**, 68% [68.174, 69.997], 90% [67.598, 70.678], MAP 69.00.
- \(f_{\rm AGN}\) = **0.2639**, 68% [0.2183, 0.3113], 90% [0.1889, 0.3425],
  MAP 0.275.
- \(\Delta\mu_\chi\) = **0.1083**, 68% [0.0894, 0.1288], 90% [0.0782, 0.1435],
  MAP 0.1075.
- \(\rho(H_0, f) = +0.0148\), \(\rho(H_0, \Delta\mu_\chi) = +0.0247\),
  \(\rho(f, \Delta\mu_\chi) = -0.5594\).
- **Information flows from the mark to the cosmology, not the other way.** On the
  matched 28-node lattice the marked \(H_0\) interval is 0.888x the spatial-only
  one at 68% and 0.900x at 90% — a narrowing of 11.2% and 10.0% — with the median
  moved by +0.0004 and the MAP cell unchanged; \(f_{\rm AGN}\) narrows by 3.8%
  (68%) and 3.5% (90%). Freeing \(H_0\) costs \(\Delta\mu_\chi\) 0.10% at
  68% and 0.41% at 90% — two orders of magnitude less than the forward effect.
  The mechanism is **not** a broken degeneracy: \(\rho(H_0, \Delta\mu_\chi)\)
  = +0.025 and \(\rho(H_0, f) = +0.015\), so the mark is nearly orthogonal to
  the cosmology and the gain arrives through \(f_{\rm AGN}\), which sets the
  tracer weighting that carries \(H_0\). \(\rho(H_0, f)\) falls monotonically
  along the ladder: +0.0678 (Analysis 2, unmarked) -> +0.0590 (S9) -> +0.0148
  (J9).
- **\(H_0\) is quoted differentially.** The planted 67.74 sits outside the 68%
  and inside the 90% in both arms (offset +1.347 S9, +1.351 J9); Analysis 2
  recovered 69.2170 on this same realisation with unmarked data, and J9 sits
  0.1265 **below** that, so the marked measurement reproduces the unmarked one.
  The offset is seed 100's own draw, not a bias.
- An 11% width change on **one** realisation is a scoping measurement, not a
  calibrated gain.

### Two things worth naming

1. **The \(\mu_{\chi,c2}\) top edge clears Gate B3 by only 6%** (9.45e-07 of
   the peak against the 1e-6 criterion). Analysis 8's \(\mu\) axis is pinned by
   the matched-axis requirement so it is not widened here, but it is the one axis
   a successor run should extend.
2. **25 of 348 guard columns rise into the corner instead of falling**, all at
   \(H_0 \le 65.0\). Analysis 8 at fixed \(H_0\) saw 11 of 11 falling. The
   rises are at most 5.3 nats against a boundary estimator noise of ~14 nats
   (`sigma2_total` 200-281 there versus 8.2 at the peak), so they are inside the
   noise; a bound that drops the monotonicity assumption entirely still gives
   8.96e-09. Recorded in `diagnostics/a9_guard_sensitivity.json`.

## Scientific contract

Analysis 8 with \(H_0\) freed, and nothing else. Three free coordinates —
`H0`, `fcat_2` (\(f_{\rm AGN}\)) and `mu_chi_c2` (the AGN branch's **absolute**
spin mean, equal to \(\Delta\mu_\chi\) only because \(\mu_{\chi,{\rm GAL}}\) is
pinned at 0.0). Seed 100 only, one realisation, the Gate-B marked mock and the
existing targeted injections reused exactly, nothing generated. Everything else
stays at the Analysis-8 configuration. Stop at the owner gate.

## Provenance — the assertions as printed, not as expected

    == provenance assertions ==
      [OK ] PYTHONPATH carries darksirens-a8: PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-a8
      [OK ] DARKSIRENS_SRC unset or darksirens-a8: DARKSIRENS_SRC='/hildafs/projects/phy230014p/magana/src/darksirens-a8' (a8_likelihood.provenance reads it)
      [OK ] darksirens-a8 HEAD == af896ca: af896cae6f3f3dd1f87dec50046e3a8228f59b39 (expected af896cae6f3f3dd1f87dec50046e3a8228f59b39)
      [OK ] darksirens-a8 worktree clean: clean
      [OK ] tracer-dependent population blocks present: 9 references to mixture_pop_params in /hildafs/projects/phy230014p/magana/src/darksirens-a8/darksirens/likelihood/core.py
      [   ] gws-agn HEAD: 335d9a3bb646f4d2809a492f72e77d223d2a34f2 (dirty=True)
      [OK ] darksirens imports from darksirens-a8: /hildafs/projects/phy230014p/magana/src/darksirens-a8/darksirens/__init__.py
      [OK ] pinned base 2b86a2d is an ancestor: base 2b86a2d
      [OK ] input present: events: .../seed100/events/events_marked_dmu0p10.h5 (162743429 bytes)
      [OK ] input present: survey_gal: .../seed100/surveys/survey_gal_complete_ns32.h5 (1650316659 bytes)
      [OK ] input present: survey_agn: .../seed100/surveys/survey_agn_complete_ns32.h5 (19934497 bytes)
      [OK ] input present: selection: .../seed100/injections/injections_targeted.h5 (203346016 bytes)
      [OK ] parameter space: free = H0, fcat_2, $\mu_\chi$_c2
      [OK ] held at Analysis-8 values: ['log10n0', 'delta', 'sigma_kde', 'log10n0_c2', 'delta_c2', 'sigma_kde_c2']
      [OK ] 13 pinned values (12 population + Om0)

      grid: 36 H0 x 41 f x 61 mu = 90036 cells

- gws-agn HEAD `335d9a3bb646f4d2809a492f72e77d223d2a34f2`. The tree is dirty, but
  only outside this analysis: two untracked archives and `working/paper_codex/`
  at the repository root. Nothing under `working/data/seed100/**` or
  `working/analyses/analysis_8_marked_multitracer_H0_fagn/**` has been touched
  (verified: no file under the Analysis-8 tree has an mtime later than its
  2026-09-18 close).
- darksirens HEAD `af896cae6f3f3dd1f87dec50046e3a8228f59b39` in
  `/hildafs/projects/phy230014p/magana/src/darksirens-a8`, worktree clean, one
  local commit on the pinned base `2b86a2d`, never pushed.
- **`PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-a8` is
  load-bearing.** `DARKSIRENS_SRC` does not steer the import — the editable
  install does — so without it the run silently uses a checkout with no
  tracer-dependent populations and Analysis 9 becomes Analysis 2 without saying
  so. Every stage asserts the import path, the SHA, the clean worktree and the
  presence of the population blocks before it does anything else, and
  `scripts/env_a9.sh` re-checks the SHA before python starts.

## The parameter space, verified before any GPU time

`build_parameter_space` was called on CPU exactly as `a8_likelihood.build` calls
it (no data loaded, no likelihood traced) and returns Analysis 8's nine labels:

    ['H0', 'log10n0', 'delta', 'sigma_kde', 'log10n0_c2', 'delta_c2',
     'sigma_kde_c2', '$\mu_\chi$_c2', 'fcat_2']

`H0` was **already** a sampled coordinate in Analysis 8 — it was simply held at
67.74 in every evaluation. Analysis 9 therefore changes no configuration at all:
it varies a coordinate the Analysis-8 parameter space already carried. The six
survey nuisances stay at the Analysis-8 values (`log10n0 = log10n0_c2 = -24`,
`delta = delta_c2 = 0`, `sigma_kde = sigma_kde_c2 = 0`) and thirteen values are
pinned by name (twelve base population parameters plus \(\Omega_{m,0}\)).

The scanned closure is the PURE likelihood, so the flat prior of record is the
**registered grid range**, not darksirens' own [20, 120] bound on `H0`.

## The registered grid

| axis | nodes | range | source |
|---|---|---|---|
| `H0` | 36 | 60.24 … 77.74, step 0.5, anchored exactly on 67.74 (index 15) | new |
| `fcat_2` | 41 | 0 … 1 | `gate_c_three_arms.F_GRID_2D`, imported |
| `mu_chi_c2` | 61 | −0.20 … +0.25 | `gate_c_three_arms.MU_GRID_2D`, imported |

90,036 cells. The \((f, \mu_\chi)\) plane is Analysis 8's own array object, not a
retyped copy, so the \(H_0 = 67.74\) slab is **cell for cell** the arm-J grid:
the equivalence gate costs nothing extra and 1/36 of the cube is the gate.

**The \(H_0\) range is sized from measurement, not taste.** On Analysis 2's
stored seed-100 joint grid (`results/joint_s100.h5`, \(H_0\in[50,100]\times201\),
unmarked, one shared population) the \(H_0\) marginal is single-moded at 69.25
and

- the mass outside [60.24, 77.74] is **4.50e-11**;
- the marginal density at the two edges is **3.66e-13** and **3.48e-10** of the
  peak.

Freeing \(\Delta\mu_\chi\) can only broaden that through the \((f,\Delta\mu_\chi)\)
correlation, and Gate B3 checks the edges again on the real cube rather than
trusting this.

**Pre-registered cost trim.** If the rita timing stage projects the registered
cube above 80 GPU-h, scan `--h0_axis fallback` instead: 25 nodes, [62.74, 74.74],
62,525 cells, a2 mass outside 6.40e-07 with edge densities 2.65e-08 and 4.62e-06.
Any other change to the axis is an owner decision. Because checkpoints are keyed
on the physical coordinates, a trim wastes nothing and a later extension rescans
nothing.

## Cost model — MEASURED on the rita A100-80 PCIe, 2026-09-19

| quantity | measured |
|---|---|
| build (`load_all_data` + parameter space + `make_likelihood`) | **29.22 s** warm / 50.16 s cold cache |
| spatial-only build reusing the same `data` object | 3.37 s |
| first evaluation (includes the JIT trace) | 11.16 s |
| **steady state** | **3.0188 s/eval** (median of 20; mean 3.0189, sd 0.0020, range [3.0161, 3.0234]) |
| peak GPU memory | **34254 MiB** of 81920 (JAX `peak_bytes_in_use` 19.59 GB, `peak_pool_bytes` 35.44 GB) |
| peak host RSS | 8.4 GB of the 100 GB requested |

3.02 s/eval is faster than both reference points — 3.74 s/eval for Analysis 2's
K=2 joint scan on an A100, and 3.5-4 s/eval expected for this problem — and is
1.77x the 1.7095 s/eval Analysis 8 measured for the identical likelihood on an
H100 NVL. One A100-80 holds the whole problem with 47 GB to spare, so two
workers fit the node's two GPUs without contention.

## Production grids — RELEASED, running

Two arms, deliberately not one cube. **S9** demonstrates where the \(H_0\)
posterior mass lies instead of assuming it; **J9** is sized FROM that
measurement. Every J9 \(H_0\) node is a strict S9 node, so the section-11 width
comparison is resolution-matched and not an artefact of spacing.

| arm | axes | nodes | cells | GPU-h at 3.0188 s | 1 GPU | 2 GPUs |
|---|---|---|---|---|---|---|
| **S9** 2-D, \(\Delta\mu_\chi = 0\) | \(H_0\) [50, 100] step 0.25, **plus 67.74**; \(f\) [0, 1] step 0.025 | 202 x 41 | 8,282 | 6.94 | 6.94 h | 3.47 h |
| **J9** 3-D | \(H_0\) [63, 76] step 0.5 (every 2nd S9 node) **plus 67.74**; \(f\) 41; \(\mu_{\chi,c2}\) 61 over [-0.20, +0.25] | 28 x 41 x 61 | 70,028 | 58.72 | 58.72 h | 29.36 h |
| **total** | | | **78,310** | **65.67** | 65.67 h | **32.83 h** |

- S9 is Analysis 2's own registered joint grid shape, so the two are directly
  comparable, plus the one node 67.74 that makes the anchor an S9 node.
- The J9 window is set by containment, not convenience. On Analysis 2's measured
  seed-100 \(H_0\) marginal (same realisation, unmarked) the density at 63 is
  **7.67e-08** of the peak and at 76 is **9.16e-08** — both more than a decade
  below the Gate-B3 criterion of 1e-6 — with **2.08e-08** of the mass outside.
  [64, 76] would leave the low edge at **3.95e-06** of the peak, ABOVE B3's own
  threshold, so the extra 1.0 in \(H_0\) is what the gate requires. The window is
  confirmed against S9's own marked marginal before J9 is committed.
- \(f\) and \(\mu_{\chi,c2}\) are Analysis 8's own axes, untouched: 4.7 nodes
  across A8's 68% width in \(f\) (0.0921) and 6.3 across its 68% width in
  \(\Delta\mu_\chi\) (0.0394); freeing \(H_0\) can only widen those, so these
  are lower bounds. \(H_0\) at step 0.5 gives 4.9 nodes across Analysis 2's 68%
  width (1.942) and is the thinnest axis — which is why S9 resolves it twice as
  finely and the width comparison is made on the shared subset.
- **No \(\Delta\mu_\chi\) trim is taken.** J9 projects to 58.72 GPU-h, below the
  pre-registered 80 GPU-h threshold, so the trim rule does not fire. The only
  trim specification 6 licenses — dropping the three nodes at
  \(\mu_{\chi,c2} \ge +0.2350\) where Analysis 8's hard \(N_{\rm eff}\) guard
  rejects at \(f \ge 0.750\), behind which the posterior mass is bounded by
  3.86e-104 — would save only 2.89 GPU-h and is not worth the loss of an exactly
  matched \(\mu\) axis. \(\Delta\mu_\chi = 0\) is inside the range either way.
- The \(H_0 = 67.74\) slab is cell for cell Analysis 8's arm-J grid, so the full
  2501-cell equivalence gate is 1/28 of J9 and costs nothing extra (2.10 GPU-h if
  it had to be run standalone).

## What is imported, and what is new

Nothing about the likelihood is reimplemented. `scripts/a9_scan.py` imports:

- `analysis_8/scripts/a8_likelihood.py` — `build` (the whole configuration:
  `load_all_data`, `validate_loaded_survey_shapes`, `build_parameter_space`,
  `get_fixed_population_params`, `make_likelihood`), `LikelihoodCell.evaluate`,
  `set_env`, the guard spy, `configure_kde`, `build_opts`, `provenance`,
  `import_scan_h0f`, `fixed_parameter_values_for`,
  `population_labels_and_fiducial`, `MU_CHI_C2_LABEL`, `SETTINGS`, the data
  paths, `H0_FID`, `OM0_FID`;
- `analysis_8/scripts/gate_c_three_arms.py` — `F_GRID_2D`, `MU_GRID_2D`, `TRUTH`,
  `_cell_record`, `_trapz_weights`, `_marginals`, `_truth_flags`, `_json_default`;
- `analysis_2/scripts/scan_h0f.py` — `marginal_ci`, the posterior convention,
  reached through `a8_likelihood.import_scan_h0f()`.

New here: the \(H_0\) axis, the 3-D marginalisation and guard bound, the
anchor-slab equivalence check, the chunked checkpoint layer and the RITA harness.

`sys.dont_write_bytecode` is set before the Analysis-8 import so not even a
`.pyc` is written into that tree, and `_write` refuses any path outside this
directory.

## Checkpoints

Append-only JSONL, `diagnostics/_a9_scan_<tag>.jsonl`, one line per **row** —
a row is the 61 `mu_chi_c2` cells at fixed \((H_0, f_{\rm AGN})\), about two
minutes of GPU. Rows are keyed on the physical coordinates, not on grid indices,
so trimming or extending the \(H_0\) axis re-uses every completed row. Each
worker appends only to its own file and reads the others read-only at start-up,
so two GPUs never write the same file. A kill mid-write loses at most the last
line, which the reader drops and the worker recomputes. This is Analysis 8's
staged pattern with the whole-file rewrite replaced by an append, because the
cube is 36x the arm-J grid.

## Plan

**Driver stages added for the production arms** (`scripts/a9_scan.py`): `s9`,
`s9_status`, `s9_assemble`, `gate_a_full`, `section_11`, plus the named H0 axes
`s9` (202 nodes) and `j9` (28 nodes) and `--out_stem` on `assemble`.  The S9
checkpoints are `diagnostics/_a9_s9_*.jsonl` (one row = the 41 f cells at one
H0); the J9 checkpoints stay `diagnostics/_a9_scan_*.jsonl` (one row = the 61 mu
cells at one (H0, f)).  `diagnostics/a9_guard.json` carries BOTH arms under
`arms.S9` and `arms.J9`.

| # | stage | where | what it settles |
|---|---|---|---|
| 0 | `provenance` | CPU | **done** — environment, inputs, parameter space |
| 1 | `timing` | rita, 1 GPU | the real s/eval, the projected cube cost, the fallback rule, and the \(H_0\) **liveness** probe |
| 2 | `anchor` | rita, 1 GPU | the \(H_0 = 67.74\) slab (2501 cells) |
| 3 | `compare_a8` | CPU | Gate A: that slab against `arm_J_joint.h5`, cell by cell |
| 4 | `scan` | rita, ≤2 GPUs | the remaining 35 slabs, 8 interleaved chunks |
| 5 | `assemble` | CPU | coverage, marginals, guard bound, comparisons, results |
| 6 | figures + `REPORT.md` | CPU | Gate D |

Gate A must pass before the cube is committed: if the slab does not reproduce
Analysis 8, nothing downstream means anything.

Submission, exactly:

    sbatch --export=ALL,STAGE="timing anchor" scripts/submit_a9_stage_rita.sbatch
    sbatch --array=0-7%2 --export=ALL,N_CHUNKS=8 scripts/submit_a9_rita.sbatch

Both carry `--partition=RITA-GPU --qos=rita --account=phy220048p
--gres=gpu:a100-80:1 --cpus-per-task=8 --mem=100G`. That triple is load-bearing:
this account holds priority on the partition, so a PENDING spell is normal.
Poll with `squeue`; do not resubmit, do not shrink the request, do not fall back
to another node.

## Carried in from Analysis 8

1. **The guard corner is real.** At \(H_0 = 67.74\) the hard \(N_{\rm eff}\) floor
   rejects 23 of 2501 arm-J cells, all at \(f \ge 0.750\) and
   \(\mu_{\chi,c2} \ge +0.2350\), with an upper bound of 3.86e-104 on the
   posterior mass behind them. \(H_0\) moves the selection integral, so the
   rejected set will not be the same at every slab; Gate B2 bounds it over the
   whole cube. Never size that margin from the population-only proxy, and never
   from the injection file's own `Neff` attribute (3714.98, a flat-target
   quantity).
2. **Score against both truths**, planted and realised, for all three
   coordinates, and quote \(H_0\) differentially against Analysis 2 as well.
3. **The `q` confound is live.** Seed 100's detected set separates GAL from AGN
   in mass ratio (KS p = 0.0096; z-stratified Fisher 29.57, the largest of 41
   seeds) and the model holds `q` identical in both branches. Any intrinsic-channel
   statement inherits that.
4. **Quote the equivalence tolerance in ULP, not in absolute logL.** At the
   seed-100 scale 1 ULP is 9.0949470177292824e-13; Analysis 8's Gate A residual
   was 2 ULP = 1.819e-12. Analysis 9's anchor check has the extra burden that
   Analysis 8 ran on the local H100 NVL and this runs on a rita A100-80, so it is
   pre-registered with an absolute bound (1e-6) **and** reported in ULP. An
   excess is a FAIL to diagnose, not a bound to widen.
5. **An equivalence test passes for free if the new coordinate is ignored.** That
   is the exact defect Gate A caught in Analysis 8, and `H0` here is a coordinate
   that was already in the label list and never varied — the same trap. The
   timing stage probes \(H_0\) liveness explicitly and Gate A will not pass
   without it.

## What is complete

Everything this analysis registered. The run is finished, every required output
is written, and `REPORT.md` is built from these files alone:

    results/s9_spatial.{h5,json}          arm S9, 8,282 cells, 0 rejected
    results/j9_marked.{h5,json}           arm J9, 70,028 cells, 809 rejected
    results/section_11_comparison.json    the width comparison, matched axes
    results/a9_closure_summary.json       the section-9 closure checks
    diagnostics/a9_guard.json             both arms, all 809 rejected cells
    diagnostics/a9_guard_sensitivity.json the monotonicity finding + the bound
                                          that does not assume monotonicity
    diagnostics/a9_gate_a_full_slab.json  the whole 2501-cell anchor slab
    diagnostics/a9_h0_matched_lattice.json  S9 re-marginalised on the J9 lattice
    diagnostics/anchor_equivalence.json   the same check from inside the cube
    diagnostics/{provenance,closure_checks,s9_vs_a8_arm_s}.json
    figs/fig_{s9_spatial,j9_corner,section_11}.{pdf,png}
    REPORT.md                             the owner report
    GATES.md                              every criterion with its deciding number

The report closes on the three statements the numbers support: the 3-D model
recovers `f_AGN` and `Δμ_χ` against both truths at 68% and reproduces the
unmarked `H0` measurement of the same realisation; the mark narrows `H0` by
11.2% (68%) and 10.0% (90%) on the matched lattice while freeing `H0` costs
`Δμ_χ` 0.10% and 0.41%, so the flow is one-way; and the mechanism is the mark
tightening `f_AGN`, not a degeneracy being broken, since `H0` is nearly
orthogonal to both other coordinates (+0.015, +0.025).

## Next allowed action

**Nothing computational.** The owner gate is reached and the next step is a
decision, not a run. What `REPORT.md` puts in front of the owner, in the order
the numbers support it:

1. additional realisations — an 11% width change on one draw, whose own `H0`
   offset is +1.35, is a scoping measurement and only replication can make it a
   calibrated gain;
2. a wider `mu_chi_c2` range — its top edge clears the 1e-6 containment
   criterion by only 6% (9.447e-07), where the other axis ends clear it by
   factors of 30 to 46 (`H0`) and by 14 orders of magnitude or more (`f` and the
   `Δμ_χ` low end);
3. handling the `q` confound — KS p = 0.0096 on this seed's detected set against
   a model that holds `q` common.

Any paper folding of this result is likewise owner-gated.

## Explicit prohibitions

No additional realisations, no seeds 101/102/103/105, no regenerated data, no
extra population degrees of freedom, no darksirens or generator edits, no push,
no GPU work on the local H100, no writes outside this directory.
