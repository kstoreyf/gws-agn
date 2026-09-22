# C10 design notes — costing the Analysis-10 cosmology stage

**Scope.** Read-only scoping for the two registered cosmology arms, C10-S
(\(H_0, f_{\rm AGN}\)) and C10-J (\(H_0, f_{\rm AGN}, \Delta\mu_\chi, \Delta\mu_{\rm G}\)),
*conditional* on the fixed-\(H_0\) owner gate passing. Nothing here has been run:
no likelihood was built, traced or evaluated, no GPU second was spent, no job was
submitted, no existing file was edited and `darksirens` was not touched. The only
CPU work was an import-level introspection of `build_parameter_space`
(`JAX_PLATFORMS=cpu`, `PYTHONDONTWRITEBYTECODE=1`), reported in §2.2.

**Rate of record.** 3.0143216848373413 s/evaluation on a rita A100-80
(`diagnostics/a10_closure.json :: cost.steady_state_median_seconds`, n = 20,
sd 1.3 ms), plus a 23.4 s build and an 11.1 s first (compiling) evaluation per
worker. Peak GPU 33.7 GiB of 80 for **one** evaluation
(`cost.peak_gpu_nvidia_smi`). All GPU-hours below are cells × 3.0143 s. Wall
clock assumes the two rita A100-80s the A10-J cube already uses
(`scripts/submit_a10_j_rita.sbatch`, `--array=0-7%2`, `--time=08:00:00`).

---

## 1. Posterior support and resolution, measured

Every number in this section is read from a recorded result, not assumed.

| coordinate | source | 68% interval | width | sd | 90% interval |
|---|---|---|---|---|---|
| \(H_0\) | `analysis_9/results/j9_marked.json` | [68.174, 69.997] | 1.823 | 0.901 | [67.598, 70.678] |
| \(f_{\rm AGN}\) | `results/a10_arm_M.json` | [0.2486, 0.3427] | 0.0941 | 0.0449 | [0.2192, 0.3732] |
| \(f_{\rm AGN}\) | `j9_marked.json` (\(H_0\) free) | [0.2183, 0.3113] | 0.0931 | 0.0439 | [0.1889, 0.3425] |
| \(\Delta\mu_\chi\) | `j9_marked.json` | [0.0894, 0.1288] | 0.0394 | 0.0195 | [0.0782, 0.1435] |
| \(\Delta\mu_{\rm G}\) | `results/a10_arm_M.json` | [3.608, 5.476] | 1.867 | 0.634 | [3.158, 5.907] |

Correlations already measured: \(\rho(f,\Delta\mu_{\rm G}) = -0.6157\) (arm M),
\(\rho(f,\Delta\mu_\chi) = -0.5594\), \(\rho(H_0,f) = +0.0148\),
\(\rho(H_0,\Delta\mu_\chi) = +0.0247\) (J9). \(\rho(H_0,\Delta\mu_{\rm G})\) is
the one entry of the six that no arm has measured — it is the C10-J deliverable.

**Containment envelopes (B3: marginal density \(\le 10^{-6}\) of the peak at
every grid edge), read off the recorded marginals:**

| axis | last node with density \(\ge 10^{-6}\) | proposed edge | density at that edge |
|---|---|---|---|
| \(H_0\) | [64.0, 74.5] | **[63, 76]** (the registered 28-node window) | 3.31e-08 / 2.18e-08 (J9) |
| \(f\) | [0.100, 0.525] (arm M); [0.075, 0.500] (J9) | **[0.050, 0.575]** | 2.0e-11 / 3.46e-08 (arm M); 4.87e-08 / 1.51e-10 (J9) |
| \(\Delta\mu_\chi\) | [0.025, 0.2425] | **[0.010, 0.250]** | 1.28e-08 / 9.45e-07 (J9) |
| \(\Delta\mu_{\rm G}\) | [2, 8] | **[1, 9]** | 2.38e-09 / 1.87e-07 (arm M) |

Two cautions. (i) The \(\Delta\mu_\chi\) upper edge is the registered axis edge
0.25 and J9 already sits at 9.45e-07 there — it clears \(10^{-6}\) by a factor
1.06 and cannot be tightened. (ii) The \(\Delta\mu_{\rm G}\) upper edge at 1.87e-07
clears by a factor 5.3; the mass mark is the spectral-siren channel, so freeing
\(H_0\) could move that marginal. B3 is a post-hoc criterion measured on the run's
own cube, and `scripts/a10_scan.py` already supports **additive node sets**
(`NODE_SETS = {"registered", "refine"}`, assembled together), so a too-tight axis
costs only the added rows, not a re-run. That is the main reason to prefer the
grid over a sampler here.

---

## 2. The darksirens sampler path

### 2.1 What `run_sampler` needs

`darksirens/inference/sampling.py:290-306` fixes the contract:

```
run_sampler(method, likelihood, prior_transform, labels,
            lower_bound, upper_bound, opts, prior_kinds=None)
    likelihood:      function(coord) -> logL, coord a 1-D array
    prior_transform: unit cube -> parameter space, 1-D array
```

Both are plain callables. `a10_likelihood.build_a10` returns an
`A10LikelihoodCell` carrying `.labels` (the 10 emitted labels), `.likelihood`
(the JIT'd function of the 10-vector) and `.coord(**overrides)` — so the
**lowest-risk hand-off is a four-line wrapper**, closing over
`cell.coord(H0=…, **{MU_G_C2_LABEL: …, MU_CHI_C2_LABEL: …}, fcat_2=…)`, with
`make_prior_transform(lower, upper, prior_kinds)`
(`darksirens/inference/prior.py:1275`) built on the four sub-bounds. No
darksirens edit, no Analysis-8 edit, and the evaluated likelihood is bitwise the
one every grid arm uses.

### 2.2 A genuine 4-label space is also available (verified on CPU today)

Adding the six survey nuisances to `fixed_parameter_values` drops them from the
sampled block (`darksirens/inference/prior.py:535` `filter_fixed_parameters`,
applied to the survey blocks at `:1031`, the `_c2` blocks at `:1082` and the
`fcat` stick at `:1108`). Called exactly as `a8_likelihood.build` calls it, with
`per_catalog_pop_params=('G.mu_c2','mu_chi_c2')` and
`fixed_parameter_values = fixed_parameter_values_for('new') ∪ {log10n0: -24,
delta: 0, sigma_kde: 0, log10n0_c2: -24, delta_c2: 0, sigma_kde_c2: 0}`,
`build_parameter_space` emits

```
['H0', '$\mu_{\rm G}$_c2', '$\mu_\chi$_c2', 'fcat_2']
bounds      [20, 120]   [20, 50]   [-1, 1]   [0, 1]
prior_kinds uniform     uniform    uniform   beta(1, 1)
```

`build_parameter_decoder` (`darksirens/inference/parameters.py:291-331`) is built
from the *same* `fixed_parameter_values` and re-derives the ordering from
`build_parameter_space`, so `make_likelihood(..., fixed_parameter_values=fpv)`
returns a likelihood that consumes exactly that 4-vector. The only edit needed is
to steer `a8.fixed_parameter_values_for`, which the existing
`a10_likelihood._steer` context-manager pattern already does for two other
darksirens entry points. **This path is more invasive than the wrapper for no
measured gain; the wrapper is the recommendation.**

**Priors must be overridden to match the grids.** The registered \(H_0\) range is
[20, 120]. A nested run on it is a *different prior* from the C10-S grid and its
\(p(H_0)\) width would not be comparable — this is exactly the quadrature trap
Analysis 9 hit (re-marginalising S9 on J9's 28 nodes widened it by 1.0256).
`prior_overrides` must pin \(H_0 \in [63, 76]\), \(\mu_{{\rm G},c2} \in [25, 45]\)
(i.e. \(\Delta\mu_{\rm G} \in [-10, 10]\), the A10 arms' own axis) and
\(\mu_{\chi,c2} \in [-0.20, 0.25]\).

### 2.3 How many evaluations a 4-D nested run costs here

Analysis 5 is the project's only nested-sampling record, and it is a **4-free-parameter
dynesty run at nlive = 1000, dlogz = 0.1** — the same shape as C10-J
(`analysis_5_free_anchors_H0_fagn/REPORT.md:12`;
`results/campaign_{m18,m19,m20,m21}_dynesty_s100.json :: sampler_meta`):

| rung | ncall_total = n_likelihood_calls | niter | eff % | dlogz reached | s/eval | wall |
|---|---|---|---|---|---|---|
| m18 | 80,749 | 10,941 | 14.79 | 9.67e-05 | 0.122 | 2.7 h |
| m19 | 72,249 | 10,484 | 15.90 | 9.61e-05 | 0.169 | 3.4 h |
| m20 | 78,838 | 8,680 | 12.28 | 9.61e-05 | 0.293 | 6.4 h |
| m21 | 90,905 | 8,302 | 10.23 | 9.59e-05 | 0.538 | 13.6 h |

None hit `maxcall = 500,000`. **72k–91k likelihood calls is the sizing number**,
and the A10 likelihood is 5.6–25× more expensive per call than A5's, so

> **dynesty, nlive = 1000, dlogz = 0.1: 60.5–76.1 GPU-h (round to 60–85 with the
> guard-wall efficiency hit), and because there is no pool, that is also the
> wall clock on ONE GPU.**

`nlive = 500` roughly halves it (ncall scales near-linearly with nlive at fixed
dlogz: ncall/nlive = 72–91 across all four rungs) → ~36k–46k calls, 30–38 GPU-h.

### 2.4 Batching, and the two-GPU question

`run_sampler`'s dynesty branch constructs
`NestedSampler(loglike, ptform, ndims, bound="multi", sample="rwalk",
nlive=opts.nlive, rstate=...)` (`sampling.py:830-838`) with **no `pool` and no
`queue_size` anywhere in the file**: dynesty here is strictly serial, one
evaluation at a time, one GPU. A 60–76 h run therefore needs 8–10 chained 8 h
SLURM jobs on a single GPU, riding the `--checkpoint_interval` / `--resume auto`
path (`cli/inference.py:38-44`, `install_dynesty_checkpointing`,
`restore_dynesty_sampler`) — a contract this project has never exercised (all
four A5 runs finished inside one allocation, on JS2).

tinyns *can* batch: its JAX rwalk kernel `jax.vmap`s the loglike over
`replacement_chains` proposals (`darksirens/likelihood/block_sizing.py:294-325`),
16 in the `heavy_darksirens` / `batched_gpu` / `bounded_*` presets, 1 in
`recommended` (`inference/tinyns_config.py:10-35`). That is **not** free
throughput: one evaluation already costs 3.0 s and 33.7 GiB with the selection
term internally batched at `sel_batch_size = 50,000` and `pe_event_block = 25`,
so the GPU is compute-saturated and a vmap over 16 proposals multiplies FLOPs and
forces the block-size planner to shrink the internal batches. No measurement of
this exists in the project; do not budget a speedup for it.

A grid, by contrast, is embarrassingly parallel over both GPUs and resumes for
free (per-row JSONL checkpoints, `_a10_scan_j_c*of8.jsonl`).

### 2.5 Guard-rejected cells and the sampler

The hard guard returns `logL = -inf`. Concretely, on this dataset:
`a10_closure.json :: checks.15_5_f1_live` records \(-\infty\) at
\((f = 1, \Delta\mu_{\rm G} = \pm 10)\) with \(N_{\rm eff}\) = 1057 and 2954
against the 5000 threshold; J9 rejected 809 of 70,028 cells (1.16%), all at
\(f \ge 0.575\) and \(\Delta\mu_\chi \ge 0.22\).

For a grid this is bookkeeping: rejected cells are listed and the mass behind
them is bounded from above by the boundary-fill construction (A10/A9 `guard`
blocks), and the proposed \(f \le 0.575\) truncation excludes J9's entire rejected
region by construction. For a sampler it is a hard wall:

* `_nested_sampler_preflight` (`sampling.py:196-260`) probes 32 prior draws
  before the run (cost: 32 × 3.0 s = 96 s), raises `RuntimeError` if all 32 are
  \(-\infty\), and warns if \(\le 3\) are finite. It will pass here.
* dynesty rejection-samples the cube until it holds `nlive` finite points; the
  \(-\infty\) region is a few percent of the override box, so initialisation
  costs a few percent extra draws — not a blocker.
* During the run, `bound="multi"` ellipsoids straddle the wall and `rwalk`
  proposals into it are rejected, which is a real efficiency loss on top of A5's
  already-modest 10–16%. Budget the high end of §2.3.
* `--selection_neff_guard soft` (`cli/inference.py:826-836`) would remove the
  wall, but it **changes the likelihood** and would break bitwise comparability
  with every grid arm in Analyses 8, 9 and 10. Do not use it.

---

## 3. Costed comparison

Cells × 3.0143 s; "wall" = on the two rita A100-80s unless noted.

| # | approach | cells / evals | GPU-h | wall (2 GPUs) | resolves | does NOT resolve |
|---|---|---|---|---|---|---|
| **S** | **C10-S**, \(H_0\) on A9's 202-node [50,100] axis × 41 \(f\) | 8,282 | **6.9** | 3.5 h | the mark-free \(H_0\) baseline, directly comparable to S9; the 28-node [63,76] sub-lattice is bitwise contained, so the matched-lattice re-marginalisation is **free** | nothing — this is the registered baseline |
| **P** | \(H_0\) profile at the fixed-\(H_0\) MAP \((f^\*,\Delta\mu_\chi^\*,\Delta\mu_{\rm G}^\*)\), 28 nodes | 28 | **0.02** | 90 s | whether [63,76] still brackets the peak once the mass mark is on — the C10-J pre-flight | nothing else; it is a single conditional slice |
| **A0** | full registered 4-D, 28 × 41 × 61 × 27 | 1,890,756 | **1,583** | 792 h (33 d) | everything | — infeasible |
| **G1** | registered lattices truncated to the \(10^{-6}\) envelope: 28 × 22 × 32 × 17 | 335,104 | **281** | 140 h (5.8 d) | \(H_0\) 3.79, \(f\) 3.76, \(\Delta\mu_\chi\) 5.25, \(\Delta\mu_{\rm G}\) 3.73 nodes/68% — every marginal at the 4-node standard (three of four a whisker under) | still 6 days of both GPUs |
| **G2** | **asymmetric**: \(H_0\) and \(\Delta\mu_{\rm G}\) on their full lattices, \(f\) and \(\Delta\mu_\chi\) at stride 2. 28 × 11 × 17 × 17 | 89,012 | **74.5** | 37 h (1.6 d) | \(H_0\) 3.79 and \(\Delta\mu_{\rm G}\) 3.73 nodes/68% — the two marginals whose **widths** are the deliverable — plus all six correlations and the MAP on the fine axes | \(f\) at 1.88 and \(\Delta\mu_\chi\) at 2.63 nodes/68%: their *widths* carry a few-percent quadrature error. Both are already measured at full resolution by arm M / arm J / J9, and freeing \(H_0\) widened \(f\) by only 1.0101 (A9), so the loss is bounded and quantified |
| **G2b** | G2 with \(\Delta\mu_{\rm G}\) widened to [0,10] and \(f\) to [0.05,0.60]: 28 × 12 × 17 × 21 | 119,952 | **100** | 50 h (2.1 d) | as G2, with \(\approx\)4 extra containment nodes per soft edge | same as G2 |
| **G3** | coarse containment/correlation cube: 14 × 11 × 12 × 9 | 16,632 | **13.9** | 7.0 h | B3 containment on all four edges, all six correlations to a few percent, the MAP cell to 1 coarse node | every marginal at 1.75–1.88 nodes/68%: **no width may be quoted from it** |
| **N1** | dynesty, nlive 1000, dlogz 0.1, priors overridden to the grid windows | 72k–91k evals (A5 measured) | **60–85** | 60–85 h on **1 GPU** (8–11 chained 8 h jobs) | all widths, all six correlations, the MAP and logZ, with no grid-resolution question at all | cannot use the second GPU; needs an untested dynesty resume chain; its \(p(H_0)\) is a Monte-Carlo estimate, not the same quadrature as C10-S; gives no B3 edge statement |
| **N2** | dynesty, nlive 500 | 36k–46k evals | **30–38** | 30–38 h on 1 GPU | as N1 with ~2× the Monte-Carlo error on the widths | as N1 |
| **H1** | **hybrid**: G3 (containment + \(\rho\), GPU A) run concurrently with N1 (widths, GPU B) | 16,632 cells + ~80k evals | **~81** | ~68 h (2.8 d), both GPUs | grid-certified B3 containment *and* sampler widths, each from the method it is good at; the two \(\rho\) estimates cross-check | the resume-chain risk of N1 remains; two different numerical objects must be reconciled in the report |
| **H2** | "profile in \(H_0\)": G2 scheduled with \(H_0\) as the outer loop | = G2 | = G2 | = G2 | identical to G2, plus a free profile-likelihood \(H_0\) curve (max over the three nuisances at each node) as a cross-check on the marginal, and per-\(H_0\)-node partial results | nothing extra; it is a scheduling choice, not a different estimator |

**Mechanism arms (§4), costed separately:** 3 new arms × 28 × 41 = 3,444 cells =
**2.9 GPU-h**, 1.5 h on 2 GPUs.

---

## 4. Recommendation

**Run P (0.02 GPU-h) → S (6.9 GPU-h) → G2 scheduled as H2 (74.5 GPU-h, 37 h on
the two GPUs), and skip nested sampling.** G2 costs about the same as a
nlive = 1000 dynesty run but finishes in half the wall clock because a grid uses
both GPUs and dynesty cannot (`sampling.py:830-838` builds the `NestedSampler`
with no pool), and it lands in the harness that Analyses 8, 9 and 10 have already
proven, with per-row checkpoints and free resume, instead of an 8–11 job dynesty
resume chain this project has never exercised. The asymmetric node allocation is
the honest trade: \(H_0\) and \(\Delta\mu_{\rm G}\) — the two axes whose width
ratio and mutual correlation *are* the deliverable — stay on their full lattices
at 3.7–3.8 nodes per 68%, while \(f\) and \(\Delta\mu_\chi\), which are only
integrated out here and whose widths are already recorded at full resolution by
arm M and J9, drop to stride 2. Every G2 node is a node of an already-run lattice,
so the \(H_0 = 67.74\) slab is cell-for-cell the A10-J cube and the
\(\Delta\mu_{\rm G} = 0\) slab is cell-for-cell J9 — two bitwise closure checks
for free, which no sampler can offer. If the P pre-flight shows the mass mark has
moved the \(H_0\) peak off-centre in [63,76], widen the axis with additive rows
before launching rather than re-running, exactly as the \(\Delta\mu_{\rm G}\)
refinement is being handled now.

---

## 5. The mechanism diagnostic (owner spec §21; gates C10-1 … C10-4)

### The proposal: four 2-D \((H_0, f)\) arms, marks pinned, on the matched lattice

Option (a) of the brief — equalising the two branches' spatial priors — is the
right one, and it is **already implemented**: Analysis 8's arm I passes the *same*
survey file for both catalog slots
(`analysis_8/scripts/gate_c_three_arms.py:11-17` for the construction and its
status as a diagnostic, `:81` `"I": {..., "surveys": "gal_gal"}`, `:113`, `:266`).
`a10_likelihood.build_a10(name, survey_paths, ...)` takes `survey_paths`
positionally, so `[SURVEY_GAL, SURVEY_GAL]` gives the **two-mark population on
spatially identical tracers with no new darksirens code and no new event file**.
When \(p_1(z|{\rm pix}) \equiv p_2(z|{\rm pix})\), the spatial factor is
branch-independent and multiplies out of the branch sum, so tracer routing carries
no information and the only branch-discriminating information left is intrinsic —
for the mass mark, the spectral-siren ruler.

Four arms, each \(H_0\) on the 28-node [63,76] lattice × the 41-node \(f\) axis,
with the two marks **pinned** at their C10-J MAP values via
`A10LikelihoodCell.evaluate_at(H0=…, fcat_2=…, dmu_chi=…, dmu_G=…)`:

| arm | surveys | marks | measures |
|---|---|---|---|
| **P0** | [GAL, AGN] | 0, 0 | the unmarked \(H_0\) width — **free**, it is the [63,76] sub-lattice of C10-S |
| **P1** | [GAL, AGN] | \(\Delta\mu_\chi^\*, \Delta\mu_{\rm G}^\*\) | the total marked \(H_0\) width |
| **I0** | [GAL, GAL] | 0, 0 | the equalised-tracer unmarked baseline |
| **I1** | [GAL, GAL] | \(\Delta\mu_\chi^\*, \Delta\mu_{\rm G}^\*\) | the equalised-tracer marked width |

\(W({\rm P1})/W({\rm P0})\) is the total gain (A9's 0.8879 at 68% is the
spin-only value to beat); \(W({\rm I1})/W({\rm I0})\) is the spectral-siren-only
gain; the routing share is the residual. Add a fifth arm with marks
\((0, \Delta\mu_{\rm G}^\*)\) to split mass from spin — 1,148 more cells.

**Cost:** P0 is free (a slice of S). P1 + I0 + I1 = 3 × 1,148 = **3,444 cells =
2.9 GPU-h, 1.5 h on two GPUs**; with the mass-only fifth arm, 4,592 cells =
3.8 GPU-h.

**Why pinning the marks is legitimate here:** Analysis 9's mechanism follow-up
measured it — "fixing the mark as well changes the marked width by 0.25%"
(`REPORT.md`, *Analysis-9 mechanism follow-up*), and the fixed-\(f\) ratio stayed
between 0.883 and 0.892 across every \(f\) node inside the 90% interval.

**What it measures:** the \(H_0\) information the mass mark carries when the two
tracers are spatially indistinguishable — i.e. an estimate of the spectral-siren
channel — against a baseline built the identical way, so the construction's own
artefacts cancel in the ratio. It satisfies C10-3's "the peak-location information
survives but the tracer routing does not" with a *measured* control rather than a
scramble that would need new generator code.

**What it cannot do.** (i) Arm I is a diagnostic construction, not a physical
model: passing GAL twice also changes the **selection** term (\(\mu\) is
recomputed with the GAL survey in both mixture slots), so the I-ratio is not
literally "the spectral-siren term of the production model" — it is a control
whose *ratio* to its own baseline is the interpretable quantity. (ii) The two
shares need not compose multiplicatively; the residual
\(\ln W_{\rm P1/P0} - \ln W_{\rm I1/I0}\) is an *attribution with an interaction
term*, and that interaction must be reported, not assumed away — the same
discipline `a10_event_decomposition.py` already applies to \(I_i\), the two-mark
interaction in the per-event log Bayes factors. (iii) It is a fixed-mark,
fixed-lattice statement; it does not decompose the marginalised C10-J width.

### The two alternatives, costed

* **Option (b), fix \(\Delta\mu_{\rm G}\) at its inferred value and compare with
  the marginalised result: zero extra cells** once G2 exists — it is the
  \(\Delta\mu_{\rm G} = \Delta\mu_{\rm G}^{\rm MAP}\) slab of the cube (at full
  resolution as a standalone arm it would be 28 × 22 × 32 = 19,712 cells =
  16.5 GPU-h). It is worth reporting because it is free, but it answers a
  different question: *how much \(H_0\) precision is lost to marginalising over
  the mass mark*, not *which of the two channels carries the gain*. It cannot
  separate routing from spectral siren at all, because both channels are active
  in both the pinned and the marginalised posterior.
* **Curvature split at the event level (C10-2's own prescription).**
  `scripts/a10_event_decomposition.py` already computes \(E_i[A|Y]\) for
  \(Y \in \{G, \chi, M, AM\}\) at one hyperparameter point and verifies each
  nested sum against a live production evaluation. Repeating it at ~10 \(H_0\)
  nodes, as Analysis 9 did, gives the event-term / selection-term split of the
  added \(H_0\) curvature and the \(|\Delta P_i|\) / crossers statistics that
  C10-2 registers, for a few tens of evaluations per node. That is the natural
  companion to the four arms above and costs well under 1 GPU-h, but it is
  new script work on top of an existing, verified module — schedule it after the
  arms, not instead of them.

---

## 6. Sources

**darksirens** (`/hildafs/projects/phy230014p/magana/src/darksirens-a8`, HEAD `af896ca`, clean):

* `darksirens/inference/sampling.py:290-306` — `run_sampler` signature and contract.
* `darksirens/inference/sampling.py:196-260` — `_nested_sampler_preflight`: 32 prior draws, `RuntimeError` when all \(-\infty\), warning at \(\le 3\) finite.
* `darksirens/inference/sampling.py:735-750` — dynesty loglike/ptform host wrappers.
* `darksirens/inference/sampling.py:830-838` — `NestedSampler(..., bound="multi", sample="rwalk", nlive, rstate)`; **no `pool`/`queue_size` in the file**.
* `darksirens/inference/sampling.py:348-375` — tinyns branch (JAX-native, no host round-trip).
* `darksirens/likelihood/block_sizing.py:294-325` — `sampler_block_sizing_profile`; tinyns `jax.vmap`s the loglike over `replacement_chains`; dynesty/numpyro concurrency 1.
* `darksirens/inference/tinyns_config.py:10-35` — presets: `replacement_chains` 1 (`recommended`) vs 16 (`heavy_darksirens`, `batched_gpu`, `bounded_*`).
* `darksirens/cli/inference.py:949-954` — `--sampler {tinyns,dynesty,numpyro}`, `--nlive` 1000, `--dlogz` 0.1, `--max_samples` 1e6.
* `darksirens/cli/inference.py:969-972` — `--sampler_preflight`.
* `darksirens/cli/inference.py:826-836` — `--selection_neff_guard {auto,hard,soft}`; hard returns \(-\infty\).
* `darksirens/cli/inference.py:38-44` — checkpoint/resume (`--checkpoint_interval`, `--resume auto`).
* `darksirens/inference/prior.py:572-592` — `build_parameter_space` signature; `:620-636` `per_catalog_pop_params` semantics; `:535` `filter_fixed_parameters`; `:1031`, `:1082`, `:1108` where it is applied; `:1275` `make_prior_transform`.
* `darksirens/inference/parameters.py:291-331` — `build_parameter_decoder` re-derives the ordering from the same `fixed_parameter_values`.

**Past runs and recorded costs (gws-agn):**

* `working/analyses/analysis_5_free_anchors_H0_fagn/REPORT.md:12` — "4 rungs × 4 free parameters, dynesty, `nlive = 1000`, `dlogz = 0.1`, flat priors".
* `working/analyses/analysis_5_free_anchors_H0_fagn/results/campaign_{m18,m19,m20,m21}_dynesty_s100.json` — `sampler_meta` (ncall_total 72,249–90,905; niter 8,302–10,941; eff 10.23–15.90%; `maxcall` 500,000 never hit), `n_likelihood_calls`, `seconds_per_eval_mean`, `wall_seconds_total`.
* `working/analyses/analysis_10_mass_spin_marked_multitracer/diagnostics/a10_closure.json` — `cost.steady_state_median_seconds` 3.0143216848373413, `cost.build_seconds_A10_two_mark` 23.36, `cost.first_eval_seconds` 11.08, `cost.peak_gpu_nvidia_smi` 33,726 MiB, `cost.projection`; `checks.15_5_f1_live` (the \(-\infty\) cells and their \(N_{\rm eff}\)).
* `working/analyses/analysis_10_mass_spin_marked_multitracer/results/a10_arm_M.json`, `a10_arm_S.json` — the fixed-\(H_0\) marginals, moments and `edge_mass` blocks used in §1.
* `working/analyses/analysis_9_marked_multitracer_H0_fagn/results/j9_marked.json` — `grid.H0` (28 nodes, [63,76] + 67.74), `posterior_moments`, `edge_mass`, `guard` (809/70,028 rejected, `rejected_f_min` 0.575, `rejected_mu_min` 0.22); `s9_spatial.json` — the 202-node axis, `j9_window_check`, `timing`.
* `working/analyses/analysis_9_marked_multitracer_H0_fagn/scripts/a9_scan.py:141-152` — `A2_H0_AXIS`, `S9_H0_AXIS` (202), `J9_WINDOW = (63.0, 76.0)`, and the assertion that every J9 node is an S9 node.
* `working/analyses/analysis_8_marked_multitracer_H0_fagn/scripts/gate_c_three_arms.py:11-17, 81, 113, 266` — arm I (`gal_gal`) and its status as a diagnostic construction.
* `working/analyses/analysis_10_mass_spin_marked_multitracer/scripts/a10_likelihood.py` — `build_a10(name, survey_paths, ...)`, `A10LikelihoodCell.evaluate_at`, `_steer`; `scripts/a10_scan.py:55-63` — `NODE_SETS` / additive refinement rows.
* `working/analyses/analysis_10_mass_spin_marked_multitracer/scripts/a10_event_decomposition.py:1-80` — the four nested models and the \(I_i\) interaction term.
* `working/analyses/analysis_10_mass_spin_marked_multitracer/REPORT.md`, *Analysis-9 mechanism follow-up* — the 0.8879/0.8985 fixed-\(f\) ratios, the 0.99997 global-mixture factor, and "fixing the mark as well changes the marked width by 0.25%".
* `working/analyses/analysis_10_mass_spin_marked_multitracer/GATES.md:434-465` — the registered C10-S/C10-J gates C10-1 … C10-4.
* `working/analyses/analysis_10_mass_spin_marked_multitracer/scripts/submit_a10_j_rita.sbatch` — the rita harness: `RITA-GPU`, `a100-80`, 100 G, `--time=08:00:00`, `--array=0-7%2`.
