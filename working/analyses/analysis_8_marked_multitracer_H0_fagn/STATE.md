# Analysis 8 state

## Current status

**Phase 1 complete: implementation map written. Coding not started.**

Last passing gate: none yet (Gate A is the first).

## Scientific contract

Unchanged from the specification. Seed 100 only; complete catalogs; `H0 = 67.74`;
`Om0 = 0.3075`; fixed common mass, `q`, redshift-rate and spin-width parameters;
free `f_AGN`; free differential spin mean `dmu_chi`; planted marked mock
`dmu_chi = +0.10`; existing Analysis-2 seed-100 data are the null/equivalence
case; stop after seed-100 closure for owner approval.

## Repository provenance

- gws-agn HEAD: `9aa3c98ce8d63d693f8b9fd147a981631152dafd`
  (18 uncommitted files, all of them the paper work under `working/paper/`;
  Analysis 8 is entirely untracked under its own directory and commits separately).
- darksirens HEAD in use: **`2b86a2d8d48fdb5173f0ba259b8996104187dd2d`**.
- darksirens worktree: `/hildafs/projects/phy230014p/magana/src/darksirens-a8`,
  branch `analysis8-marked-populations`, created from the pin. Local only; the
  owner has forbidden remote darksirens changes, so this branch is never pushed.

### Why the pin and not the working checkout

Gate A has to reproduce the Analysis-2 likelihood. Analysis 2 (SLURM 1059xxx) and
the seed-100 dataset (`seed101/META.json` carries the same SHA) both ran on
`2b86a2d`, while `src/darksirens` sits at `b324bed` on a feature branch. Building
on `b324bed` would make Gate A a test of code drift rather than of the new path.
The two SHAs were separately measured bit-identical on the Analysis-0 scan path
(max abs diff in logL = 0.000e+00 on identical hardware), and the only importable
file that differs is `gw/populations/registry.py`, in the
`gwtc5_fiducial_bpl2peaks` entry, which this campaign does not use. The pin is
therefore a provenance choice, not a numerical one.

**`DARKSIRENS_SRC` does not steer the import.** `scan_h0f.py:49-51` uses it only
for the ancestry check at `scan_h0f.py:549-556`, which asserts `2b86a2d` is an
*ancestor* of HEAD and so passes for any descendant. The real import at
`scan_h0f.py:252` resolves through the editable install (`darksirens.egg-link` ->
`src/darksirens`). **Every Analysis-8 command must set
`PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-a8`**, which does
win, or it will silently run the wrong code.

## Data files used (read-only; never overwritten)

- events: `working/data/seed100/events/events.h5` (156 MB)
- surveys: `working/data/seed100/surveys/survey_{gal,agn}_complete_ns32.h5` (1.6 GB, 20 MB)
- injections: `working/data/seed100/injections/injections_{targeted,popuni}.h5` (194 MB, 101 MB)
- seed-100 realised: 705 GAL / 295 AGN hosts, `realised_f_agn = 0.295`,
  `planted_f_agn = 0.30`, 1000 events, nsamp 2000.

## Implementation map

### Existing Analysis-2 likelihood call path

`scripts/scan_h0f.py` -> `make_likelihood(opts, data, pop_params_fid,
fixed_parameter_values)` at `scan_h0f.py:416-419` (the ONLY place population
information crosses from gws-agn into darksirens) -> `factory.py:706` dispatches
to `_make_mixture_likelihood` (`factory.py:167`) for `n_catalogs >= 2` ->
jitted `_body` decodes at `factory.py:533-540` -> `darksiren_log_likelihood`
(`core.py:305-394`) with `mixture_log_weights=log_w` (`factory.py:592`).

Analysis 2 runs with `lss_marginalize=False` (`scan_h0f.py:332`), so the
alternative cached path at `core.py:1063` / `core.py:1185` is **not** on this
campaign's code path and does not need the change.

### Current population function signature

`pop_model_parser(pop_model, *, shared_beta, shared_spin, shared_gamma)` returns
`get_model(...).log_p_pop` (`registry.py:689-701`), a stateless pure function of
`theta`. For `powerlaw+peak` with everything shared, `theta` has 12 slots:

    [ 0] v1 0.1   [ 1] PL.alpha 2.3  [ 2] PL.m_min 5.0  [ 3] PL.m_max 80.0
    [ 4] PL.dm_min 3.0  [ 5] PL.dm_max 10.0  [ 6] G.mu 35.0  [ 7] G.sigma 5.0
    [ 8] beta 1.0  [ 9] mu_chi 0.0  [10] sigma_chi 0.1  [11] gamma 0.0

**Index 9 is `mu_chi`.** `theta_AGN` is `theta_GAL` with slot 9 set to
`mu_chi + dmu_chi`. Nothing else differs.

`shared_spin=False` is **not** a tracer split: `grammar.py:289-291` emits one spin
component per MASS SLOT and `base.py:425-428` enforces
`len(spin_components) in {1, k}` over mass components. Confirmed independently by
three readers; matches the specification's own warning. It will not be used.

### Where catalog mixture weights are applied

`decode_mixture` (`parameters.py:178`, returns at `:240`) yields `surveys` and
`mark_params_all` as **per-catalog tuples** but `pop_params` as a **single shared
vector**. That asymmetry is the entire parameter-space gap. Stick-breaking
(`parameters.py:31`) gives `log_w = [log(1-fcat_2), log(fcat_2)]` at K=2, so
**`fcat_2` is `f_AGN`**.

The weights are consumed in exactly two places, both of which multiply `f_k` onto
the **redshift prior only**: `core.py:696` (`_eval_prior_mix`) and `core.py:1036`
(`_log_p_mix`, the unused `lss_marginalize` path).

### Where the selection integral is assembled

`compute_selection_term` (`selection.py:429`) invokes a `log_weight_fn` callable
at `selection.py:455`; `_batch_lse` consumes one fully composed per-injection
`ldw` and is blind to branch structure. `log_evidence_and_mc_variance`
(`selection.py:170`) computes `ln Zhat = -log n + logsumexp(ldw)` with the N_eff
and variance bookkeeping.

### Per-catalog population support already present?

**No.** There is no tracer/AGN/mark concept anywhere in the populations package,
and no `_c{k}` suffix machinery for population parameters (the suffix exists for
survey, mark and `b_miss` blocks only). darksirens *does* already carry
per-catalog **marked-host** models (`eta_<mark>_c{k}`, `test_marks_multitracer.py`),
but those attach to the redshift prior, not to `p_pop(theta)`. Right pattern,
wrong target.

### The defect, stated exactly

At the pin the K>=2 mixture collapses at the **redshift-prior level**:

    core.py:695   lps = [ log w_k + log p_z_k(z, pix_k) for k in range(K) ]
    core.py:703   return _mixture_logsumexp(lps)
    utils.py:127  return log_p_pop(...) + log_prior_z_fn(...) - logJ

so the model evaluated is `p_pop(theta) * sum_k f_k p_k(x)`. That is Analysis 2's
model and exactly the ordering specification section 0.6 forbids. The design
comment at `core.py:494` says so in plain words ("inserted at the prior level so
the population term is not recomputed K times").

### Minimal upstream extension point

Four additive edits in `darksirens/likelihood/core.py`, all inert at K=1:

1. declare `mixture_pop_params: tuple = ()` in the K-catalog kwarg block
   (`core.py:365`), mirroring `mixture_surveys`;
2. build `pop_params_all = (pop_params,) + tuple(mixture_pop_params)` beside
   `surveys_all` (`core.py:553`);
3. split `_eval_prior_mix` (`core.py:681`) into a `_prior_lps(...)` returning the
   per-branch list, stopping just before `_mixture_logsumexp` (`core.py:703`);
4. at `core.py:769` (PE) and its mandatory selection twin `core.py:813`, form the
   per-branch total `log f_k + log p_pop(theta | Lambda_k) + log p_z_k` and only
   then `_mixture_logsumexp` (`core.py:617`, reusable verbatim).

The selection integral itself needs **no** darksirens change: `mu = sum_k f_k mu_k`
then holds by linearity of the Monte-Carlo sum, with N_eff, `log_sigma2` and the
variance guard intact. `mu_GAL` and `mu_AGN` must **not** be computed as two
separate `compute_selection_term` calls: the branches share one injection pool and
one `pdraw`, so their Monte-Carlo errors are strongly correlated.

Cost: `log_p_pop` is stateless and uncached, so K evaluations cost only
arithmetic. Because the density factorises as
`mass_q_density(m1,q) * spin(chieff)` under shared spin
(`tests/test_population_split.py`), only the spin factor actually differs between
branches.

### Marked mock: the generator edit

One line between `generate_dataset.py:1344` and `:1347`:

    chi = chi + np.where(is_agn, _DMU_CHI_AGN, 0.0)

plus a `--dmu_chi_agn` flag (default 0.0, preserving existing behaviour) and two
HDF5 attributes beside the `shared_spin` attr at `:1587`. `is_agn` is already in
scope from `:1334`; `chi` has exactly one downstream consumer (`:1347` ->
`observe_v3:843`) whose RNG consumption does not depend on `chi`'s values, so the
event stream stays identical. Exact here because the truncation sits 9 sigma away.

Known consequence: the `shared_spin = True` HDF5 attribute (`:1587` events,
`:2287` injections) becomes false for a marked mock and any reader trusting it
must be found before the file is written.

## Verified before coding

- **Specification 5.3 cleared.** `TruncatedGaussianSpin` is a Gaussian truncated
  to `[-1, 1]` with fiducial `mu_chi = 0.0`, `sigma_chi = 0.1`. The density
  normalises to `1.00000000` at `dmu_chi = -0.20, 0, +0.10, +0.25`; the plant sits
  9 sigma inside the truncation. No STOP required.
- **The v3 chi_eff measurement is observed-centred, not truth-centred.**
  `DESIGN_PE.md`: `chieff_obs = chieff + sigma_chieff N(0,1)`, PE
  `= N(chieff_obs, sigma_chieff)` truncated to `[-1,1]` (exact), flat PE prior in
  `chieff`, `sigma_chieff = 0.2 * (8/rho_obs)` from GWMockCat
  (Farah et al. 2023). The intrinsic arm is therefore honest.
- **Regression baseline** on the pin, before any edit: 60 passed, 1 skipped
  (`test_multitracer_likelihood`, `test_marks_multitracer`, `test_population_split`,
  `test_population_registry_golden`, `test_analyze_fcat_weights`,
  `test_selection_variance_guard`, `test_population_support_contracts`).
- **Effect size is sensible.** 295 AGN-hosted events at per-event
  `sigma_chieff ~ 0.08-0.20` constrain a branch spin mean to ~0.007, so
  `dmu_chi = +0.10` is invisible per event and strongly stacked.
- **Cost bound.** At the measured K=2 rate (~1.9 s/eval on the local H100 NVL) a
  41 x 61 grid is ~1.3 GPU-h per arm: Arms I and J ~2.6 GPU-h, Arm S minutes.

## Gate state

- Gate A null/equivalence: **PENDING** (next action)
- Gate B marked mock integrity: BLOCKED ON A
- Gate C seed-100 recovery: BLOCKED ON B
- Owner gate: LOCKED

## Last completed outputs

- Implementation map (this file).
- Regression baseline recorded above.

## Next allowed action

Implement the four `core.py` edits plus the parameter-space thread-through in the
`darksirens-a8` worktree, add unit tests there, confirm the K=1 and
`mixture_pop_params=()` paths are unchanged against the baseline, then execute
Gate A only.

## Explicit prohibition

Do not run additional realizations. Multiple seeds are an owner decision after
Gate C. Do not push the darksirens branch.
