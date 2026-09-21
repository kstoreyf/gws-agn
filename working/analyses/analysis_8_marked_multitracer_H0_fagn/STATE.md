# Analysis 8 state

## Current status

**Gates A, B and C PASSED (2026-09-18). The seed-100 package is COMPLETE and work has
STOPPED at the owner gate.**

Last passing gate: **C** (seed-100 marked recovery, C1-C6).

The marked mock is `seed100/events/events_marked_dmu0p10.h5`
(md5 7dcb8bccba7f7a360a6da2741d4cf9d1). Selection reuses the existing
`injections_targeted.h5`; no injections were generated.

## Scientific contract

Unchanged from the specification. Seed 100 only; complete catalogs; `H0 = 67.74`;
`Om0 = 0.3075`; fixed common mass, `q`, redshift-rate and spin-width parameters;
free `f_AGN`; free differential spin mean `dmu_chi`; planted marked mock
`dmu_chi = +0.10`; existing Analysis-2 seed-100 data are the null/equivalence
case; stop after seed-100 closure for owner approval.

## Repository provenance

- gws-agn HEAD at Gate A: `ac74b98b86ffb4028921c2fa8e09fa068bd1cb31`
- gws-agn HEAD when this section was written: `9aa3c98ce8d63d693f8b9fd147a981631152dafd`
  (18 uncommitted files, all of them the paper work under `working/paper/`;
  Analysis 8 is entirely untracked under its own directory and commits separately).
- darksirens base SHA: **`2b86a2d8d48fdb5173f0ba259b8996104187dd2d`** (the pin).
- darksirens HEAD in use: **`af896ca`**, "likelihood: support tracer-dependent
  population blocks", a single local commit on top of the pin. Never pushed; the
  owner has forbidden remote darksirens changes.
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

## The Analysis-8 parameter space (established, not assumed)

Two free coordinates, exactly as specification section 2 requires:

    sampled labels: ['$\mu_\chi$_c2', 'fcat_2']

reached with `fix_population=False` and all twelve base population parameters
pinned explicitly through `fixed_parameter_values`, plus
`per_catalog_pop_params=('mu_chi_c2',)`. `fcat_2` is `f_AGN` (stick-breaking at
K=2 gives `log_w = [log(1-fcat_2), log(fcat_2)]`).

`fix_population=True` together with a per-catalog request is REFUSED by design
(it would remove the population block from sampling), so Analysis 8 pins the
twelve common parameters by name instead. That is more auditable than
`fix_population=True`: every fixed value is written down.

The scanned coordinate is `mu_chi_c2`, the AGN branch's ABSOLUTE spin mean. It
equals `dmu_chi` numerically only because the GAL branch's `mu_chi` is pinned at
0.0; report it that way rather than conflating the two.

## Gate state

- Gate A null/equivalence: **PASS** - evidence `diagnostics/null_equivalence.{json,md}`
- Gate B marked mock integrity: **PASS** 10/10 - evidence
  `diagnostics/marked_mock_validation.json`, `diagnostics/selection_support.json`
- Gate C seed-100 recovery: **PASS** (C1-C6) - evidence `results/arm_*.{h5,json}`,
  `results/event_decomposition.{h5,json}`, `diagnostics/gate_c_guard.json`, `REPORT.md`,
  `figs/fig_*.{pdf,png}`
- Owner gate: **REACHED**. Work stopped. Nothing proceeds without an explicit decision.

## Gate C result

On one marked seed-100 dataset the joint arm recovers both planted quantities, with BOTH
truths (planted and realised) inside the 68% intervals:

    f_AGN     0.261653  [0.216174, 0.308298] 68%   planted 0.30, realised 0.295
    dmu_chi   0.107386  [0.088587, 0.127960] 68%   planted +0.100, realised +0.111924
    correlation -0.554, MAP (0.275, +0.1075)

The information split is asymmetric and was measured, not assumed: `f_AGN` is essentially
all spatial (joint 68% width 0.0921 against spatial-only 0.0915, intrinsic-only 4.85x
worse), while the spin offset is measured 2.38x better jointly than intrinsically alone
(0.0394 against 0.0938). Knowing WHICH events are AGN-hosted sharpens the spin
measurement; the spin mark adds essentially nothing to the host fraction.

Per event the two channels are close to orthogonal (AUC 0.690 spatial, 0.679 intrinsic,
0.742 combined). The spatial channel has a long negative tail to -295 nats, because an
event whose localisation volume holds no AGN candidate is excluded from that branch
outright; the intrinsic channel spans only [-2.93, +2.64], since shifting a spin mean can
move a branch ratio by at most a few nats.

Selection: 23 of 2501 joint cells guard-rejected, all at f >= 0.750 and dmu >= 0.2350,
with an upper bound of 3.86e-104 on the posterior mass behind them, so the REGISTERED
grid stands and no result sits behind a rejected cell. Over the cells that carry the
posterior the margin is wide: min N_eff/threshold 85.5 at 90% of the mass, 30.7 at
99.999%.

`P(dmu_chi <= 0)` is 4.85e-11 in the joint arm. It is a posterior probability under this
model and this grid and is explicitly NOT a sigma claim.

## Carried into Gate C (all three established by Gate B; see GATES.md for numbers)

1. **Scan range.** The registered `dmu_chi` range [-0.20, +0.25] collides with the hard
   N_eff guard: the real selection integral drops below 5000 above `mu_chi_c2` ~ +0.227
   at f = 1 (measured 2766 at f = 1, mu = +0.25, likelihood -inf). Either scan
   `|mu_chi_c2| <= 0.20` or scan the registered range and demonstrate the rejected
   corner carries negligible posterior mass. Size this from the REAL selection integral,
   never from the population-only proxy and never from the injection file's own `Neff`
   attribute (3714.98, a flat-target quantity).
2. **An unplanted confound.** Seed 100's detected set already separates GAL from AGN in
   mass ratio `q` (KS p = 0.0096; z-stratified Fisher 29.57, the largest of 41 seeds).
   It is the record's own property, not this campaign's. Since the inferred GAL and
   AGN branches share the same `q` distribution, it does not directly create branch
   evidence; correlations among `q`, mass, distance and `chi_eff` in the event
   posteriors can nevertheless indirectly affect recovery of `dmu_chi`, so the
   intrinsic arm is not guaranteed to measure it in isolation (wording corrected
   2026-09-21).
3. **Score against both truths.** Planted `dmu_chi` = +0.100000; realised (detected-set
   branch-mean difference) = +0.111924 +/- 0.006815. The +0.011924 excess is the
   record's own finite-sample draw, identical to the last digit in the unmarked run.
- Gate C seed-100 recovery: BLOCKED ON B
- Owner gate: LOCKED

## Last completed outputs

- Implementation map (this file).
- Regression baseline recorded above.
- darksirens `af896ca`, one local commit on the pin: per-tracer population blocks.
- `scripts/a8_likelihood.py` (the two configurations, built through analysis 2's own
  `scan_h0f` machinery; no likelihood is reimplemented) and
  `scripts/gate_a_null_equivalence.py`.
- `diagnostics/null_equivalence.{json,md}` - Gate A evidence.

## Gate A result, in one place

With `mu_chi_c2 = 0` the tracer-dependent path reproduces the Analysis-2 model to the
last bit: the largest disagreement anywhere is 1.8189894035458565e-12, which is
**exactly 2 ULP** of a log-likelihood of order 4.2e3 (verified: measured / 2 ULP =
1.0000). A1 and A4 sit at that floor; A2 (both endpoints, PE and selection terms
separately) and the selection linearity in A3 are **exactly 0.0**. The f posterior
median moves by 3.55e-15, MAP unchanged. Guard: 216 cells, min N_eff 75.7x threshold,
none rejected.

The decisive control needs no tolerance at all: re-running the UNCHANGED Analysis-2
configuration through the current code lands the SAME 2 ULP from the stored August
array, so that residual is the machine's last-bit floor, not the new population path
and not code drift.

Liveness, so the equivalence cannot have passed vacuously: at f = 0.295 moving
`mu_chi_c2` 0 -> +0.10 changes logL by -18.2976, and at f = 0, where catalog 2 carries
zero mixture weight, the same offset leaves logL bitwise identical. The coordinate is
connected and wired to its own branch.

Three corrections to earlier notes in this campaign, all recorded in GATES.md: the
residual sits at the PE seam (log_mu is bitwise identical at every f), not the
selection seam; and mu_GAL and mu_AGN are NOT expected to coincide (+0.98% apart)
because the branches carry different spatial priors. The chi_eff-independence claim
was measured on its own terms and holds: a +0.10 shift in the AGN spin mean moves mu
by -3.44e-04, i.e. 0.30 Monte-Carlo sigma.

The third correction matters for every later gate: **quote this equivalence
tolerance in ULP, never in absolute logL.** GATES.md's `3.552713678800501e-15` is
2 ULP of a TOY logL of order 8; at the seed-100 scale 1 ULP is 9.0949470177292824e-13
and the identical effect is 1.82e-12, 500x larger in absolute terms. The driver's
pre-registered absolute bound (1e-12) was in the wrong unit and would have read FAIL
on a correct measurement; it is recorded verbatim in `null_equivalence.json` under
`tolerances_preregistered`, with the applied ULP criterion beside it.

Also measured, and needed for Gate C's parameter space: `build_parameter_space`
accepts ONLY the LaTeX prior labels for the twelve BASE population parameters
(`'$\mu_\chi$'`, not `'mu_chi'`) -- the plain ASCII spelling is aliased for the
`_c{k}` blocks alone. Gate A kept analysis 2's full label shape and added exactly one
coordinate, so A4 was like-for-like against the stored scan:

    OLD: ['H0', 'log10n0', 'delta', 'sigma_kde', 'log10n0_c2', 'delta_c2', 'sigma_kde_c2', 'fcat_2']
    NEW: ['H0', 'log10n0', 'delta', 'sigma_kde', 'log10n0_c2', 'delta_c2', 'sigma_kde_c2', '$\mu_\chi$_c2', 'fcat_2']

## Next allowed action

Gate B: generate ONE marked seed-100 event family at the registered
`dmu_chi = +0.10`, reusing the existing LSS/catalog realization, and write
`diagnostics/marked_mock_validation.json`. Everything before Gate B is complete.

(The darksirens edits and Gate A, which this line used to point at, are both done:
`af896ca` and `diagnostics/null_equivalence.{json,md}`.)

## Explicit prohibition

Do not run additional realizations. Multiple seeds are an owner decision after
Gate C. Do not push the darksirens branch.
