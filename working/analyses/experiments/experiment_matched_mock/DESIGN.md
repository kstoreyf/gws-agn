# experiment_matched_mock — regenerate the mock so its generative process is the model

Goal: remove every known mock/model mismatch **by construction**, by generating the
events and the injection campaign with darksirens' own mock generator
(`scripts/mock_dark_sirens/generate_mock_data.py`, "gmd") instead of the bespoke
chain, then re-run the `experiment_h0f_baseline` scans and see whether the H₀
offset survives.

Predecessor: `../experiment_h0f_baseline` — f_AGN recovers across the planted
range, H₀ recovers low with the deficit growing in f_AGN (−0.92 at f=0.307, −3.32
at f=0.703).

## What the diagnostics established (and what they refuted)

**1. The likelihood decomposition (`../experiment_h0f_baseline/scripts/diag_h0_offset.py`).**
At fixed mixture weight, logL = Σᵢ ln Zᵢ(H₀) + [−N_obs ln μ(H₀) + MC correction]:

| planted f | full peak | numerator alone | shift from selection term |
|---|---|---|---|
| 0.307 | 66.82 (−0.92) | 70.83 (+3.09) | −4.01 |
| 0.703 | 64.42 (−3.32) | 68.69 (+0.95) | −4.27 |

This localises where the H₀ *sensitivity* lives — the selection term supplies a
~4 km/s/Mpc pull, and ln μ moves 0.36 nats over H₀ ∈ [60,76] against N_obs = 1000,
a ~360-nat lever. **It does not by itself identify an error:** the selection term is
*supposed* to counteract the numerator, and only the total has to be unbiased.
Reading "numerator biased high" or "selection is the cause" off this table was too
quick, and is not the position this experiment is built on.

**2. Relabelling the selection rule cannot fix anything.** A cut on *true* dL and a
cut on *true* z select the same events at the true cosmology (the map is monotonic),
so the injection file and the whole logL surface are bit-identical. The distinction
between the two rules is a counterfactual — what *would* have been detected at other
parameters — and that is fixed by the physical detection mechanism, not by a label.
Hence any real fix must change the generative process, not its description.

**3. The mock's selection rule is not expressible in the injection format.**
darksirens stores injections detected-only in `(m1det, q, dL)` with `p_draw` a
density there, and evaluates μ(θ) by mapping each frozen row to the source frame via
z(dL; θ). That is correct for a real detector, whose detectable region is fixed in
those cosmology-free observables. The mock's rule — a cut on **true redshift** — is a
region fixed in source-frame z, which is not a fixed region in `(m1det, q, dL)`; and
no detector can select on true redshift, so no injection set can represent it. This
is a genuine defect of the mock.

**4. The rate-weighting mismatch is real but NOT the cause — measured and refuted.**
gmd weights host acceptance by `(1+z)**(gamma-1)`, with its own comment noting "the
bare dV_c/dz draw is gamma=1, not 0". Our events' host redshifts match the catalog's
flat count-weighted pool (mean z 0.7051 vs 0.7059 GAL; 0.6977 vs 0.7060 AGN), i.e.
the mock's effective γ was 1 while the inference used γ = 0. Re-running the H₀ scans
with γ forced to 1 (`scan_h0f.py --gamma`, `results/gtest_g*.json`):

| planted f | γ = 0 | γ = 1 |
|---|---|---|
| 0.307 | −0.93 | −0.97 |
| 0.703 | −3.31 | −3.47 |

Null. In field mode at the complete-catalog limit the `(1+z)**(gamma-1)` factor
enters both the per-event host prior and the global normaliser Z, so a smooth
z-reweighting largely cancels. The mock's flat host draw is harmless here.

**5. The dominant lever is the catalog's redshift EDGE — measured, and it matches a
documented open darksirens issue.** PR #215's own commit message, under "Not fixed
here", records: *"A separate low bias for catalogs whose dN/dz rises into a sharp
z_max edge remains after this fix (mock closure recovers 45.3, not 70, on the
volume-limited catalog) and is under separate investigation."*

Our catalog is exactly that shape: dN/dz rises monotonically to z ≈ 1.4 and then
drops off a cliff at z_max ≈ 1.56 (GAL bins per Δz=0.1: … 126k, 133k, 140k, 130k,
28k). Re-running the H₀ scans on the z ≤ 1 truncated catalogs
(`gal_zlt1.h5`/`agn_zlt1.h5`, `DARKSIRENS_ZMAX=1.05`), with the **events
unchanged**, moves the edge and moves H₀ with it (`results/edgetest_zlt1_*.json`):

| planted f | native catalog (edge 1.56) | truncated at z ≤ 1 | Δ |
|---|---|---|---|
| 0.307 | −0.93 | **−5.02** | −4.09 |
| 0.703 | −3.31 | −3.74 | −0.43 |

A 4 km/s/Mpc H₀ shift from relocating a catalog-construction boundary, with the data
fixed, is the signature of the documented pathology. It also explains the original
campaign's otherwise puzzling zlt1 A/B (GAL peak 65.45 → 61.6 on truncation).

**Net position.** The mock's true-z selection rule (defect 3) is a real
inconsistency and worth removing, but there is no evidence it drives the H₀ offset,
and the γ mismatch (defect 4) is measured null. The offset is dominated by
sensitivity to the host catalog's redshift edge — an open upstream issue, not a
property of the mock's selection. **A regeneration that only changed the selection
rule would not have fixed the headline number.**

## Design

Two changes, not one. The selection fix removes a real inconsistency and is a
prerequisite for any clean closure test; the **catalog-depth fix is the one aimed at
the headline number**. Generate the mock with the library's own generator so that
every ingredient — selection, rate weighting, mass distribution, PE — is the
inference's model by construction, AND generate the host catalog deep enough that
its z_max edge sits far beyond the detected events, so the edge cannot shape the
prior over the range the data occupy.

**The catalog-depth fix (primary).** Today the catalog ends at z ≈ 1.56 while events
reach z ≈ 1.0 — the edge is ~0.5 away and still dominant. Regenerate the GLASS host
field to z_max ≈ 3 (events still z ≲ 1), set `DARKSIRENS_ZMAX` to the new depth, and
require the result to be **insensitive** to the depth: re-running at two depths
(e.g. 2.5 and 3.0) must move H₀ by ≪ the statistical error. That insensitivity is the
acceptance criterion, and it is the thing the current mock fails by 4 km/s/Mpc.

| ingredient | baseline mock | this experiment |
|---|---|---|
| detection | hard cut on true z ≤ 1 | gmd's **noisy network SNR ≥ threshold** (`_network_snr`, an observable) |
| host rate weighting | flat over eligible catalog entries (effective γ=1) | gmd's `(1+z)**(gamma-1)` acceptance, γ = 0 |
| masses/spins | regenerated at fiducials, whole-mixture taper | gmd's own population samplers (per-component pairing, post-#205) |
| injections | bespoke, detection replaced by the z-cut | gmd's `_selection_injections` with the **same** `_network_snr` + threshold |
| PE | bespoke obs-centred exact-posterior clouds | gmd's `_posterior_samples` |
| **host field depth** | **GLASS to z_max ≈ 1.56, edge ~0.5 beyond the events** | **regenerated to z_max ≈ 3, edge far beyond the events** |
| survey catalogs | `gal.h5`, `agn.h5` (pixelated at the old depth) | rebuilt from the deeper field, same nside 64, same tracer biases |

The K=2 mixture is built by drawing `round(f·N)` events from the AGN host list and
the remainder from the GAL host list (two `_draw_events_until_detected` calls, one
per host catalog), preserving the `gal_then_agn` ordering and `host_type` label.

Open choices to settle during the build:
- SNR threshold: pick the value giving a comparable N_obs and redshift reach to the
  baseline, so the two experiments are comparable in information content.
- Whether the sparse AGN tracer still needs a catalog-targeted injection lane for
  field-mode N_eff once selection is SNR-based (the baseline required it).

## Validation ladder (before any inference number is quoted)

1. μ from the injections at the true cosmology matches the mock's own detected
   fraction.
2. d ln μ/dH₀ matches the analytic expectation for an SNR-limited survey — the check
   that would have caught defect 3 immediately.
3. Event mass/spin/redshift distributions match the inference's p_pop (KS or
   quantile comparison), and host z now carries the `(1+z)^(γ−1)` tilt.
4. N_eff and the σ²_lnL budget at N_obs = 1000.
5. **Closure test:** H₀ recovered at increasing N_obs; the residual must shrink like
   1/√N. A residual flat in N is remaining mis-specification and the experiment has
   failed to close.

## Arms

- **A (primary):** deep catalog + matched mock (γ = 0, SNR selection, gmd injections)
  → the (H₀, f) scans of `experiment_h0f_baseline`, same grids, same guard
  (`5·N_obs`, variance criterion inert).
- **A′ (the acceptance test):** arm A repeated at a second catalog depth. H₀ must be
  insensitive to the depth. This is the criterion that distinguishes "fixed" from
  "happened to land closer".
- **B (mass channel):** same events, inference with an uninformative mass model, to
  size the spectral-siren contribution separately.
- **C (only if A still shows an offset):** re-run `diag_h0_offset.py` on the matched
  deep mock. A surviving, depth-insensitive offset under a matched generative process
  localises the problem inside darksirens — at which point this experiment becomes
  the reproducer for the PR #215 follow-up rather than a mock fix.

## Coordination note

If arm A′ confirms depth-insensitivity restores H₀, the finding belongs upstream too:
the catalog-depth requirement is a *usage* constraint on darksirens (host catalogs
must extend well beyond the detected horizon), in the same family as the already-known
"field-mode sparse-tracer mixtures need catalog-targeted injections" and
"`DARKSIRENS_ZMAX` must match the survey depth". It is the third such constraint this
campaign has had to discover by measurement.
