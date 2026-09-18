# Analysis 8 — owner report (marked GAL/AGN multi-tracer, seed 100)

**Sources.** `results/arm_{J_joint,I_intrinsic,S_spatial}.{h5,json}`,
`results/event_decomposition.{h5,json}`, `diagnostics/gate_c_guard.json`,
`diagnostics/event_decomposition.json`, `diagnostics/null_equivalence.{json,md}`,
`diagnostics/marked_mock_validation.json`, `diagnostics/selection_support.json`.
darksirens `af896ca` on the pinned base `2b86a2d`, tree clean; gws-agn `c3bd534`
with uncommitted files elsewhere in the repository at run time (recorded as
`gws_agn_dirty: true`; nothing under this directory's inputs).
Seed 100, complete catalogs, `H0 = 67.74` fixed, one local H100. Total scan cost
4285 s (J) + 8378 s (I) + 181 s (S).

## Question

Analyses 0–7 infer the GAL/AGN host mixture from two correlated large-scale-structure
tracers while both branches share one BBH intrinsic population. The host label is
therefore identified only by where an event sits in the density field. If AGN-channel
binaries also differ *intrinsically* — the standard expectation for a gas-rich,
hierarchical-assembly channel — then the same latent label is written twice, once in
the sky-and-redshift structure and once in the source parameters, and the two channels
should be readable together.

This analysis asks the narrow version of that question that one realisation can answer:
with a single planted intrinsic mark, a GAL/AGN offset in the effective-spin mean, can
`(f_AGN, Δμ_χ)` be recovered simultaneously, and how much does each channel contribute
to each parameter? The quantitative target is the *measured* information split — not
whether a joint fit is better, but which parameter each channel actually constrains.

## Exact model

The hierarchical likelihood is the production dark-siren one with the branch product
opened up so each spatial tracer carries its own intrinsic population,

```
p(θ, x | f_AGN, Λ_GAL, Λ_AGN)
   = (1 − f_AGN) p_GAL(x) p_pop(θ | Λ_GAL)
   +      f_AGN  p_AGN(x) p_pop(θ | Λ_AGN),
```

with `x = (z, Ω)` and `θ = (m1, q, χ_eff)`. The event likelihood is
`L_i = Σ_k f_k L_ik` and the selection expectation is `μ = Σ_k f_k μ_k`, both built from
the identical per-branch kernel, so the normalisation cannot disagree with the event
weights.

`Λ_GAL` and `Λ_AGN` are the same signed-off seed-100 `powerlaw+peak` vector except for
the effective-spin mean,

```
μ_χ,GAL = 0   (pinned at the fiducial),        μ_χ,AGN = μ_χ,GAL + Δμ_χ,
```

with the common width `σ_χ = 0.1`. The sampled coordinate is `mu_chi_c2`, the AGN
branch's **absolute** spin mean; it coincides with `Δμ_χ` only because `μ_χ,GAL` is
pinned at 0. Both spellings are carried in every output. Thirteen population and
cosmology values are pinned (`Ω_m = 0.3075`, `α_PL = 2.3`, `m_min = 5`, `m_max = 80`,
`δm_min = 3`, `δm_max = 10`, `μ_G = 35`, `σ_G = 5`, `β = 1`, `γ = 0`, `v_1 = 0.1`,
`μ_χ = 0`, `σ_χ = 0.1`; listed in `diagnostics/_gate_c_stage_S.json`), and `H0 = 67.74`.
The two free coordinates are `f_AGN ∈ [0, 1]` and `Δμ_χ`.

Three arms share one marked dataset and one likelihood build:

| arm | construction | grid |
|---|---|---|
| **J** joint | both catalogs carry their own spatial prior *and* their own population block; the production model | `f` 41 nodes on [0,1] (step 0.025) × `Δμ_χ` 61 nodes on [−0.20, +0.25] (step 0.0075) |
| **I** intrinsic-only | **diagnostic**: the same GAL complete survey in *both* catalog slots, so the common spatial factor `p(z\|pix)` multiplies straight out of the branch sum and all remaining `f`-dependence is intrinsic | same 41 × 61 |
| **S** spatial-only | `Δμ_χ ≡ 0` exactly (identical intrinsic populations); `f` from tracer structure alone | 101 nodes on [0,1] (step 0.01), the Analysis-2 convention |

Posteriors are flat-prior, equal-tailed, trapezoid-marginalised; the quantile routine is
Analysis 2's `marginal_ci`, imported rather than reimplemented.

## Implementation change

The per-tracer population block lives in darksirens, not in a gws-agn conditional:
one commit, `af896ca` "likelihood: support tracer-dependent population blocks", on the
pinned base `2b86a2d`, in the local worktree
`/hildafs/projects/phy230014p/magana/src/darksirens-a8` (branch
`analysis8-marked-populations`, never pushed). It adds a `mixture_pop_params` kwarg to
`darksiren_log_likelihood`. Per PE sample `s` and branch `k`, the kernel
`_log_sample_weight_pop_branches` in `darksirens/likelihood/core.py` forms
`log f_k + log p_k(z_s | pix_k,s) + log p_pop(θ_s | Λ_k)`, collapses the branches with
`_mixture_logsumexp`, and subtracts the branch-independent Jacobian and proposal terms
once outside the sum. `f_k` appears exactly once, the branch sum happens only after the
per-branch spatial × intrinsic product, and no spatial-times-intrinsic Bayes-factor
product exists anywhere.

Backward compatibility is structural, not incidental: `pop_params_shared` is a static
Python bool, so with `mixture_pop_params = ()` the empty case re-executes the historical
statement verbatim. Measured, it returns the same hex float `0x1.7701cac65c400p-6` as the
pinned code with the change stashed out and in, at eight separate mixture weights,
through both `make_likelihood` and a direct core call. Regression: 60 passed / 1 skipped
on the seven-file suite against a baseline of exactly 60/1, and 214 passed / 1 skipped on
the tier-0 subset.

One defect was found and fixed before any production run. `build_parameter_space` emitted
`_c{k}` suffixes only for survey, stick and mark labels and hard-rejected every other key,
so `has_per_catalog_pop_params` was always false and a production run would have silently
kept the old shared-population model. Two unit tests missed it because they hand-built the
decoder with a plain label spelling (`"mu_chi"`) that never occurs in production, where
population labels are LaTeX (`'$\mu_\chi$'`). With the fix the sampled labels carry
`'$\mu_\chi$_c2'` (`results/arm_J_joint.json`, `sampled_labels` / `per_catalog_pop_params`),
and the coordinate is demonstrably live and wired to the right branch: at `f = 0.295`,
moving `mu_chi_c2` from 0 to +0.10 moves `logL` by **−18.2976**, while at `f = 0`, where
catalog 2 carries zero mixture weight, the same offset leaves `logL` bitwise identical.

The gws-agn layer is thin — `scripts/a8_likelihood.py` (build and environment),
`gate_a_null_equivalence.py`, `gate_b_selection_support.py`, `gate_c_three_arms.py`,
`event_decomposition.py`, `fig_event_evidence_plane.py` — and reimplements no likelihood.
The exact commands that produced the recorded result:

```
export PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-a8
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
D=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_8_marked_multitracer_H0_fagn

$PY $D/scripts/gate_c_three_arms.py --stage timing --arm J
$PY $D/scripts/gate_c_three_arms.py --stage scan   --arm J
$PY $D/scripts/gate_c_three_arms.py --stage scan   --arm I
$PY $D/scripts/gate_c_three_arms.py --stage scan   --arm S
$PY $D/scripts/gate_c_three_arms.py --stage assemble
$PY $D/scripts/event_decomposition.py
```

`PYTHONPATH` is mandatory: `DARKSIRENS_SRC` only feeds the ancestry assertion, while the
real import resolves through the editable install, so without it the wrong code runs
silently.

## Null/equivalence result (Gate A)

**PASS.** Evidence: `diagnostics/null_equivalence.{json,md}`.

With `Δμ_χ = 0` the new path reproduces the Analysis-2 model to the last bit of double
precision. The largest disagreement anywhere is `1.8189894035458565e-12`, exactly 2 ULP
of a log-likelihood of order 4.2e3 (relative 4.36e-16). Both endpoints reduce exactly:
`f = 0` against a K = 1 GAL population and `f = 1` against a K = 1 AGN population agree at
**0.0 exactly**, in the total, the PE term and the selection term alike. The selection
integral is linear in the mixture weight, `μ(f) = (1−f) μ_GAL + f μ_AGN`, to a maximum
relative difference of 7.90e-16. Re-scanning the seed-100 `f` posterior moves the median
by −3.55e-15 and leaves the MAP unchanged at 0.27.

The decisive control is not the tolerance but the baseline: re-running the **unchanged**
Analysis-2 configuration through the current code lands the same 2 ULP from the stored
array, with 87 of 101 cells exactly 0. The residual is the machine's last-bit floor, not
the new population path and not code drift. The residual sits at the PE seam — `log_mu`
is bitwise identical between the two shapes at every `f`.

## Marked mock definition (Gate B)

**PASS, 10/10.** Evidence: `diagnostics/marked_mock_validation.json`,
`diagnostics/selection_support.json`.

One new seed-100 event family, `seed100/events/events_marked_dmu0p10.h5`
(sha256 `98b0e3f96a5b3716c4e9e287ae598e852e5d6580081435164915eb3032373d86`,
162,743,429 bytes), generated with `--dmu_chi_agn 0.10 --events_suffix _marked_dmu0p10`
and no `--seed_events`, `--f_agn` or `--overwrite`. The LSS realisation, the catalogs and
the host assignment are the existing signed-off ones; no injections were generated.

The mark is exactly where it should be. Against a same-environment unmarked run, 46 of 50
datasets are bit-identical and the 4 that differ are all effective spin (`chieff`,
`true_chieff`, `truth/chieff`, `truth/obs_chieff`), differing on exactly the 295 AGN
events and nowhere else; GAL true spin is bitwise unchanged and the AGN shift is 0.1 to
within 2.8e-17. Host labels, indices, redshifts, distances, sky positions, masses, `q`,
SNRs and every `truth/obs_*` array are bit-identical, as is the rejected-proposal file.

**Two truths, and they are not the same number.** The planted values are
`f_AGN = 0.30` and `Δμ_χ = +0.100000`. The realised values — what this one draw actually
contains — are `f_AGN = 0.295` (705 GAL / 295 AGN detected) and, for the detected-set
branch-mean difference in true `χ_eff`, `Δμ_χ = +0.111924 ± 0.006815`. The record already
carried **+0.011924** of that difference from finite sampling before any mark was applied,
identical to the last digit in the unmarked run. Every recovery below is scored against
both.

One vocabulary caveat for readers of the file: the `shared_spin` HDF5 attribute is still
`True`, which is correct in darksirens' vocabulary (one spin component shared across the
*mass* components of one mixture). Branch spin must be read from `dmu_chi_agn = 0.1`,
`mu_chi_gal = 0.0`, `mu_chi_agn = 0.1`.

## Three-arm result (Gate C)

All three arms read the same marked dataset and the same likelihood build.

| | **J** joint | **I** intrinsic-only | **S** spatial-only |
|---|---|---|---|
| cells | 2501 | 2501 | 101 |
| `logL_max` | −4211.842592845020 | −4249.502089201707 | −4233.939503109996 |
| MAP | `f = 0.275`, `Δμ_χ = +0.1075` | `f = 0.250`, `Δμ_χ = +0.1000` | `f = 0.26` |
| `f_AGN` median | 0.261653 | 0.314428 | 0.264977 |
| `f_AGN` 68% | [0.216174, 0.308298] | [0.170729, 0.614948] | [0.219798, 0.311301] |
| `f_AGN` 90% | [0.187170, 0.339609] | [0.116288, 0.842827] | [0.190908, 0.342058] |
| `Δμ_χ` median | 0.107386 | 0.081723 | — (fixed at 0) |
| `Δμ_χ` 68% | [0.088587, 0.127960] | [0.041726, 0.135511] | — |
| `Δμ_χ` 90% | [0.077621, 0.142695] | [0.028444, 0.172627] | — |
| posterior correlation | −0.5544 | −0.8289 | — |
| rejected cells | 23 | 0 | 0 |
| median s/eval | 1.7095 | 3.3464 | 1.7073 |

**Both truths are inside both intervals, in every arm that measures the parameter.**
Naming them explicitly, for the production arm J: the planted `f_AGN = 0.30` and the
realised `f_AGN = 0.295` both lie inside the 68% interval [0.216174, 0.308298] and inside
the 90% interval [0.187170, 0.339609]; the median sits −0.038347 from planted and −0.033347
from realised. The planted `Δμ_χ = +0.100000` and the realised `+0.111924 ± 0.006815` both
lie inside the 68% interval [0.088587, 0.127960] and inside the 90% interval
[0.077621, 0.142695]; the median sits +0.007386 from planted and −0.004538 from realised.
The joint median therefore falls *between* the two truths in `Δμ_χ`, closer to the planted
value, and just below both in `f_AGN`. In arm I, both `f_AGN` truths lie inside its 68%
[0.170729, 0.614948] and 90% [0.116288, 0.842827], and both `Δμ_χ` truths inside its 68%
[0.041726, 0.135511] and 90% [0.028444, 0.172627]. In arm S, both `f_AGN` truths lie
inside its 68% [0.219798, 0.311301] and 90% [0.190908, 0.342058].

**The null.** Under this model and on this grid, the flat-prior posterior mass at or below
zero spin offset is

```
P(Δμ_χ ≤ 0)  =  4.85e-11   (arm J)            3.37e-06   (arm I)
```

obtained by interpolating the same trapezoid CDF that produces the quoted intervals; the
marginal posterior density at `Δμ_χ = 0` is 3.04e-10 (J) and 3.56e-05 (I) of its peak
value, and the best log-likelihood at the grid node nearest zero is 21.11 (J) and 10.19 (I)
below the MAP. **This is not a sigma claim, and it must not be converted into one.** A
far-tail probability on a bounded flat-prior grid is a property of the grid and of the
model as much as of the data: the range [−0.20, +0.25] and the 0.0075 spacing set the
normalisation, zero is not itself a grid node (the neighbouring nodes are −0.0050 and
+0.0025, so the value at zero is interpolated), and the number is conditional on the rest
of the model being right — in particular on the assumption, false in this realisation,
that GAL and AGN share a common mass-ratio distribution (see Limitations). Read it as
"the null is far out in the tail of this posterior", nothing more.

**The spatial arm behaves as the unmarked analysis does.** Against the recorded unmarked
Analysis-2 `f`-scan on the identical 101-node grid
(`analysis_2_complete_catalog_H0_fagn/results/fscan_s100.{h5,json}`), the `f_AGN` median
moves by only −0.001522 (0.266499 → 0.264977), while the log-likelihood shifts by a nearly
constant offset across all 101 cells: mean −58.137, s.d. 0.291, range [−58.794, −57.625].
The mark changes the evidence level of the data under a model that cannot use it, and
leaves the shape in `f` essentially untouched — which is what "spatial-only" should mean.

**The information split, measured.** 68% interval widths:

| | `f_AGN` | `Δμ_χ` |
|---|---|---|
| S spatial-only | 0.0915 | — |
| J joint | 0.0921 | 0.0394 |
| I intrinsic-only | 0.4442 | 0.0938 |

`f_AGN` is essentially all spatial: the joint width 0.0921 is within 0.7% of the
spatial-only 0.0915, while the intrinsic-only arm is 4.85× worse. The spin offset runs the
other way: it is measured 2.38× better jointly (0.0394) than intrinsically alone (0.0938).
The coupling is therefore **asymmetric**. Knowing which events are AGN-hosted — information
the tracer field supplies — sharpens the spin measurement substantially, while the spin
mark adds essentially nothing to the host fraction at this effect size.

The one apparent anomaly is diagnosed, not pathological: arm J's `f_AGN` interval is 0.68%
*wider* than arm S's. Arm J marginalises over `Δμ_χ` where arm S fixes it at zero, and the
two parameters are anticorrelated (−0.5544), so the joint marginal must be at least as wide.
The cost of that marginalisation, under 1%, is the quantitative statement that the spin mark
carries almost no host-fraction information here.

## Event-level interpretation

The decomposition is evaluated at the arm-J MAP, `f = 0.275`, `Δμ_χ = +0.1075`, chosen
because it is a node of the production grid: the scan evaluated that exact cell, so the
decomposition can be checked against a recorded number. (The posterior median lies 0.3 grid
steps away in `f` and 0.06 in `Δμ_χ`, 0.15σ and 0.007σ.)

The split is exact, not approximate. Writing the production kernel's three pieces as
`A_k(s) = log p_k(z_s|pix_k,s)` (spatial), `B_k(s) = log p_pop(θ_s|Λ_k)` (intrinsic) and
the branch-free `C(s)`, and defining `E_i[XY] = ⟨exp(A_X + B_Y + C)⟩_s` over the same
samples with the same production mask, the production per-event likelihood is exactly
`Z_i = f_GAL E_i[GG] + f_AGN E_i[AA]`, and

```
log BF_spatial   = log E[AG] − log E[GG]
log BF_intrinsic = log E[AA] − log E[AG]
log BF_total     = log BF_spatial + log BF_intrinsic   (exactly)
P_i(AGN)         = sigmoid( log(f_AGN/f_GAL) + log BF_total ).
```

The identity is verified against production to zero: `Σ_i log Z_i = −16014.75553768314`
reproduces both a live production call at the MAP and the recorded
`guard/logL_pe[11,41]`, absolute difference 0.0, and `Σ_i log E_i[GG] = −16067.162166483613`
reproduces the recorded `f = 0` column, absolute difference 0.0.

Over the 1000 events (705 true GAL, 295 true AGN; the label is a mock diagnostic and never
enters the inference):

| quantity | median true-GAL | median true-AGN | AGN − GAL | AUC |
|---|---|---|---|---|
| `log BF_spatial` | −0.0755 | +0.1876 | +0.2630 | 0.690 |
| `log BF_intrinsic` | −0.2131 | +0.1867 | +0.3998 | 0.679 |
| `log BF_total` | −0.2888 | +0.3940 | +0.6827 | 0.742 |
| `P_i(AGN)` | 0.2213 | 0.3600 | +0.1387 | 0.742 |

Both channels separate the true labels in the right direction and by comparable amounts in
the median, and they are close to independent per event, so the combined evidence separates
better than either alone: AUC 0.742 against 0.690 and 0.679. That near-independence is the
event-level counterpart of the population-level result — the two channels are carrying
different information about the same latent label.

The per-event labels themselves remain weak, which is the point worth carrying. At a 0.5
threshold, 61 of 295 true AGN and 674 of 705 true GAL are called correctly (accuracy 0.735,
92 events called AGN); only 6 events exceed `P_i(AGN) = 0.9` and 145 fall below 0.1; and the
summed probability `Σ_i P_i = 274.76` against the realised 295. The population constraint is
not built on confidently labelled individual events, it is built on a small mean shift
across a thousand weakly labelled ones.

The chain-rule ordering is a choice and its size is measured: routing the split through the
GAL-spatial / AGN-intrinsic hybrid instead changes the two pieces by the interaction term
only, whose median is 0.0011, 90th percentile of the absolute value 0.218, maximum 1.68.
Both orders give the identical `log BF_total` and the identical `P_i(AGN)`.

Two caveats attach to this table specifically: it is one hyperparameter point, not
marginalised over the `(f_AGN, Δμ_χ)` posterior, and `log BF_intrinsic` is not a clean
measurement of the spin mark alone, because this seed's detected set also separates the two
branches in mass ratio (see Limitations).

## Selection diagnostics

The selection expectation uses the same per-branch kernel as the event term. The guard is
the existing hard `N_eff` floor: `max(5 N_obs, N_obs² / (1e6 − pe_variance_sum))`, which with
the measured `pe_variance_sum` of order 5–55 is `5 N_obs = 5000` everywhere here; a cell
below it returns `−inf`.

| arm | rejected | min `N_eff` | min `N_eff`/threshold | median `N_eff` |
|---|---|---|---|---|
| J | 23 / 2501 | 2766 | 0.553 | 371400 |
| I | 0 / 2501 | 8541 | 1.708 | 391736 |
| S | 0 / 101 | 378742 | 75.7 | 647178 |

**No result stands behind a rejected cell.** All 23 rejected cells in arm J lie in one dead
corner, at `f ≥ 0.750` and `Δμ_χ ≥ +0.2350`, at the far edge of the registered grid; the
least-rejected of them reaches 0.980× the threshold. Because a rejected cell returns `−inf`
and so carries exactly zero mass on the grid as scanned — a circular test — the mass is
instead bounded from above by *filling* every rejected cell with the largest posterior
density found on the accepted boundary beside it. That boundary density is 5.82e-104 of the
peak, 237.7 in log-likelihood below it, and the resulting upper bound on the rejected
posterior mass fraction is **3.86e-104**. The bound overstates the truth, since the
log-likelihood falls monotonically away from the peak in every one of the 11 boundary
columns (checked). The accepted mass beyond the first rejected `Δμ_χ` node is 1.37e-06.

Over the cells that actually *carry* the posterior, the margin is large, not marginal: the
minimum `N_eff`/threshold is **85.5** over the cells holding 90% of the mass, 70.5 at 99%,
55.3 at 99.9% and 30.7 at 99.999%. The posterior mass sitting in accepted cells with less
than 2× the threshold is 6.8e-52 (cell-weighted; 3.4e-52 under the trapezoid weights the
marginals use — the same conclusion either way). The worst accepted cell anywhere, at
1.015×, is at `f = 0.900`, `Δμ_χ = +0.2350`, inside that same dead corner.

Arm I's thin floor (1.708×) is expected and does not bound the result: its common-prior
construction is deliberately misspecified, so its selection integral is evaluated under a
spatial prior that does not match the branch it weights.

The injection set is the existing 2.2M-injection targeted lane, reused; none were generated.
It spans `χ_eff ∈ [−0.99999507, +0.99997832]`, the whole of the [−1, 1] truncation both
branch populations live on, with zero empty bins carrying target mass for either branch, and
the worst margin at the registered mark over all `f` is 28.2×. The spin reweight is
algebraically exact, and the detection rule is independent of `χ_eff`: measured directly
from the uniform proposal branch, where the number proposed per `χ` bin is known exactly,
the detection efficiency is flat at `χ² = 9.24` on 9 dof (p = 0.42), trend
−0.0027 ± 0.0040 per unit `χ_eff`. A normalised spin density therefore cannot change the
detectable fraction.

Two numbers must never be quoted as selection margins, both of which look like margins and
are not: the population-only proxy, which reads 10,044 at `Δμ_χ = +0.25` where the real
selection integral is 2766, and the injection file's own `Neff` attribute (3714.98), which
is the `N_eff` of a flat target computed at generation time and is two orders of magnitude
below any selection-integral `N_eff`.

Only the targeted lane was run. The population/uniform lane was not, so no cross-lane
posterior shift is reported.

## Limitations

**Seed 100's detected set already separates GAL from AGN in mass ratio, and the model does
not know that.** The separation is the record's own property, present in the unmarked data
and not created here: KS `p = 0.0096` on the detected set, `p = 0.0066` when labels are
permuted within redshift quartiles, and a z-stratified Fisher statistic of 29.57 — the
largest of the 41 seeds examined, against a median of 7.54. The model holds `q` identical in
both branches, so this is an **unmodelled channel difference**. Two consequences follow, and
neither is cosmetic. The branch label is partly identifiable without the spin mark, so arm I
cannot be read as measuring `Δμ_χ` in isolation, and neither can the per-event
`log BF_intrinsic`. And because the `P(Δμ_χ ≤ 0)` tail is conditional on the model being
right, an unmodelled intrinsic difference is exactly the kind of error that tail does not
account for.

**Arm I is a diagnostic construction, not a physical model.** It puts the same GAL survey in
both catalog slots so the common spatial prior multiplies out of the branch sum. Nothing in
nature corresponds to that. Its widths bound the intrinsic-only information available in
*this construction*; they are not an intrinsic-only measurement, and its deliberately
misspecified selection integral is why its `N_eff` floor is thin.

**The marked mock is not bit-identical to the August record.** It differs by a 1-ULP
environment drift in the tapered power-law inverse CDF as well as by the mark, because the
record was written on a node with a different `pow` kernel: 11 of 1000 primary masses differ
at up to 1.42e-14, and everything downstream of them follows. Proven by recomputation — this
node reproduces the marked file's `snr_true` bit-for-bit from its own stored masses and
distances (0/1000 differ) but not the record's (406/1000, max 1.421e-14). Against a
same-environment unmarked run the marked file is 46/50 datasets bit-identical with only the
four `χ_eff` arrays differing. The detected set, its order and every realised statistic are
unchanged to full printed precision, and the generator flag itself is provably inert
(pristine versus patched generator at the default gives 50/50 datasets bit-identical), but
the drift is real and is why comparisons here are made against a same-environment run rather
than against the stored record.

**Two truths, not one.** The planted and realised values differ in both parameters
(`f_AGN` 0.30 vs 0.295; `Δμ_χ` +0.100000 vs +0.111924 ± 0.006815, of which +0.011924 predates
the mark). Any recovery statement that does not name which value and which interval is
incomplete.

**One realisation.** Everything above is a single draw of a single seed. Nothing here
establishes that the estimator is calibrated across realisations, that its coverage is
correct, that the measured offsets are unbiased in expectation, or that any sensitivity
scaling with `N` holds. Nothing here says anything about real BBHs having an AGN-specific
spin distribution, about `H0` improving, or about incomplete catalogs being safe. The
per-event decomposition is additionally evaluated at a single hyperparameter point and is
not marginalised over the posterior.

## This is seed 100 only

**Every number in this report comes from seed 100, and from one marked realisation of it.**
No other seed was run. No second realisation of seed 100 was run. No effect-size ladder was
run. What this establishes is implementation closure and proof of concept: the tracer-
dependent population path reduces exactly to the previous model when the mark is switched
off, it recovers both planted quantities from one marked dataset with both truths inside the
68% intervals, and the spatial/intrinsic information split is measured rather than assumed.
It establishes nothing about the estimator's behaviour in expectation.

## Next owner gate

The next step is a decision, not a run. The seed-100 package is complete and work stops
here pending explicit owner approval of the next phase — which, on the evidence above,
means authorising **additional realisations** (seeds 101/102/103/105, or repeat draws of
seed 100) so that the recovery can be assessed as a distribution rather than as one point,
and deciding whether the mass-ratio confound is handled by modelling `q` per branch or by
selecting seeds on it.

Without that approval, none of the following is run: additional seeds or realisations,
effect-size ladders, free `H0`, mass-dependent marks, free common population parameters,
incomplete catalogs, GP/HSGP population differences, or GWTC data.

**OWNER GATE: seed-100 Analysis 8 is complete. I have not run additional realizations.**
