# Analysis 10 — owner report

## Owner decision

> The originally registered peak-fraction mark was rejected before generation
> because it lay outside the mixture simplex. The owner replaced it with a
> Gaussian-peak-location mark Δμ_G = +5 Msun. The original failed gate remains
> part of the provenance record; the new mass-mark gate begins from this owner
> decision.

**The replacement mark.** Option (c) of the table below: mark the **location**
of the Gaussian component, not its weight.

\[
  \mu_{\rm G}^{\rm GAL} = 35\,M_\odot, \quad
  \mu_{\rm G}^{\rm AGN} = 40\,M_\odot, \quad
  \Delta\mu_{\rm G} = +5\,M_\odot,
\]

with the spin mark unchanged at \(\Delta\mu_\chi = +0.10\). The peak fraction
stays 0.90 in both branches; the power-law component, the mass limits, the
tapers and the \(q\) distribution are identical in both branches. Single seed
(100), one realisation, science exploration — no calibration, no extra
realisations.

**It is admissible.** `G.mu` is slot 6 of `powerlaw+peak`: fiducial **35.0**,
prior bounds **[20, 50]**, LaTeX label `$\mu_{\rm G}$`. The planted 40 sits
inside with room, and the exploratory axis \(\Delta\mu_{\rm G} \in [-10, +10]\)
(\(\mu_{{\rm G},\rm AGN} \in [25, 45]\), 21 nodes at \(1\,M_\odot\)) leaves
\(5\,M_\odot\) of margin at each prior edge. The per-catalog resolver already
emits the coordinate: with
`per_catalog_pop_params=('G.mu_c2', 'mu_chi_c2')` the space carries 10 labels
including `$\mu_{\rm G}$_c2` at [20, 50] and `$\mu_\chi$_c2` at [−1, 1], with
**no darksirens change** (Gate R). Whether the coordinate is *live* — reaching
the PE and selection terms rather than being emitted and ignored, Analysis 8's
trap — is unproven and is a mandatory closure gate.

**What it costs.** A new event set, because the mass mark needs a
branch-conditioned draw and therefore changes detection; the spin mark's trick
of shifting a completed draw does not transfer. Planned: a new
`--dmu_G_agn` flag (default 0.0, existing paths bitwise unchanged), a new file
`working/data/seed100/events/events_marked_dmu0p10_dmuG5.h5`, and a bitwise
control at `--dmu_G_agn 0.0 --dmu_chi_agn 0.10` against the existing
`events_marked_dmu0p10.h5`. Gate B (selection support of
`injections_targeted.h5` under both branch populations) remains mandatory, for
the same reason.

Nothing has been run under the new mark. `GATES.md` carries the re-keyed
ledger; `STATE.md` the verified facts.

## Fixed-`H0` results: the two-mark environmental population is recovered (2026-09-22)

**Sources.** `results/a10_arm_{S,chi,M,J}.{h5,json}`, `diagnostics/a10_closure.json`,
`diagnostics/a10_selection_support.json`, `diagnostics/a10_mock_validation.json`,
`diagnostics/a10_event_decomposition_map_f0p275_mu0p1150_G5.json`, figures
`figs/fig_a10_{fagn,marks,planes}.{pdf,png}` and `figs/fig_event_mass_vs_spin_map_*.{pdf,png}`.
darksirens `af896ca` on the pinned base `2b86a2d`, clean; the two-mark mock
`events_marked_dmu0p10_dmuG5.h5` (md5 `427990378e29…`), the existing targeted injections.
`H0 = 67.74`, `Ω_m = 0.3075`, twelve base population parameters pinned; free coordinates
`f_AGN`, `Δμ_χ = μ_{χ,c2}`, `Δμ_G = μ_{G,c2} − 35`. Rita jobs 1335495/1335500 (arms),
1335496 + 1335661 + 1335501 + 1335665 (cube, 56.7 GPU-h at 3.019 s/cell), 1336522
(event decomposition). Every number below is read from those files.

### What was planted and what the detected set looks like

The AGN branch's Gaussian peak sits at `μ_G = 40` against `35` in the GAL branch
(`Δμ_G = +5 Msun`; peak fraction 0.90, power law and `q` identical), and its spin mean at
`+0.10`. Because the heavier branch is louder, the detected host fraction is **0.357**
(357 of 1000) against the planted 0.30, and the detected fraction of proposals rises from
7.85e-3 to 8.46e-3. The model's `f_AGN` is the pre-selection host fraction, so its target is
**0.30**; 0.357 is a descriptive number a selection-corrected model should *not* recover.
The detected-set spin-mean difference is +0.0995 ± 0.0067; the detected-set primary-mass
medians are 36.8 (GAL) and 41.6 Msun (AGN), descriptive only.

### The four arms

| arm | free | `f_AGN` median, 68%, 90% | `Δμ_χ` | `Δμ_G` [Msun] | MAP |
|---|---|---|---|---|---|
| A10-S spatial only | `f` | 0.2953 [0.2451, 0.3463] [0.2120, 0.3801] | ≡ 0 | ≡ 0 | 0.300 |
| A10-χ spin mark | `f, Δμ_χ` | 0.3039 [0.2594, 0.3485] [0.2317, 0.3793] | 0.1207 [0.1039, 0.1382] [0.0936, 0.1506] | ≡ 0 | (0.300, +0.1225) |
| A10-M mass mark | `f, Δμ_G` | 0.2938 [0.2487, 0.3420] [0.2191, 0.3728] | ≡ 0 | 4.528 [3.848, 5.247] [3.485, 5.732] | (0.275, +4.5) |
| **A10-J joint** | `f, Δμ_χ, Δμ_G` | **0.2762 [0.2350, 0.3191] [0.2093, 0.3472]** | **0.1172 [0.1018, 0.1334] [0.0927, 0.1442]** | **4.756 [4.109, 5.420] [3.674, 5.888]** | (0.275, +0.115, +5.0) |

**Recovery.** The joint model recovers all three coordinates. `f_AGN`: planted 0.30 inside
the 68% interval (offset −0.024); the detected-set 0.357 is outside the 90% interval, as it
should be. `Δμ_G`: planted +5 inside the 68% interval (offset −0.24, 0.4 posterior sd);
zero is excluded by the whole width of the axis (the marginal density at `Δμ_G = 0` is
below e^−30 of the peak). `Δμ_χ`: planted +0.10 and realised +0.0995 sit inside the 90%
interval and just outside the 68% (offset +0.017, 1.1 posterior sd); the spin-only arm,
which is misspecified on this mock because it ignores the mass mark, sits higher still at
0.1207. Zero is excluded on every mark axis by far more than the axis width.

**Widths, marked against spatial-only** (68% / 90%):

| quantity | χ / S | M / S | J / S | J / χ | J / M |
|---|---|---|---|---|---|
| `f_AGN` | 0.881 / 0.878 | 0.922 / 0.914 | **0.830 / 0.821** | 0.942 / 0.934 | 0.900 / 0.897 |
| `Δμ_χ` | — | — | — | **0.920 / 0.905** | — |
| `Δμ_G` | — | — | — | — | **0.938 / 0.985** |

**Correlations** (joint arm): `ρ(f, Δμ_G) = −0.580`, `ρ(f, Δμ_χ) = −0.521`,
`ρ(Δμ_G, Δμ_χ) = +0.272`; posterior sd 0.039 in `f`, 0.596 Msun in `Δμ_G`, 0.0153 in
`Δμ_χ`. The mass-mark arm alone has `ρ(f, Δμ_G) = −0.612`.

### The answers to the fixed-`H0` questions

1. **Can the Gaussian-peak shift be recovered jointly with the spin shift?** Yes. Both marks
   are recovered against their planted values with the host fraction free, and the joint
   posterior is compact on every axis (edges ≤ 7e-10 of the peak; the `Δμ_χ` top edge that
   Analysis 9 cleared by 6% is cleared here by three orders of magnitude, because the
   mass mark takes over part of the branch identification).
2. **Is the mass mark more informative about the environmental label than the spin mark?**
   Event by event, yes; on the global fraction, no. At the joint MAP the mass mark moves
   `P_i(AGN)` by more than 0.1 for 435 events and across 0.5 for 162 (156 upward), against
   381 and 77 for the spin mark, and it moves the expected AGN count by +61 where the spin
   mark moves it by +3.7; AUC for the true label 0.748 (mass) against 0.708 (spin). But the
   spin mark sharpens `f_AGN` more (0.881 against 0.922 at 68%), because `Δμ_G` is more
   strongly degenerate with `f` (−0.58) than `Δμ_χ` is (−0.52) and the mass mark carries the
   selection function with it: a heavier AGN branch is a louder one, so part of what the
   mass mark learns about the label is spent on the branch-dependent detectability.
3. **Does the spatial tracer improve the measurement of `Δμ_G`?** Not measurable here in
   isolation: unlike Analysis 8, no intrinsic-only arm was run (the spec fixed four arms).
   What is measured is the reverse direction: adding the mass mark to the spatial model
   narrows `f_AGN` by 8% at 68%, and adding the spin mark narrows `Δμ_G` by 6% (J/M 0.938).
4. **Are mass and spin complementary or redundant?** Complementary. The two marks re-route
   largely different events: Pearson correlation +0.14 (Spearman +0.10) between `ΔP_i^mass`
   and `ΔP_i^spin`; of the 435 and 381 events each mark moves by more than 0.1, only 182 are
   shared (Jaccard 0.29); 48 events cross 0.5 under the joint model that neither single mark
   moves across. `ΔP^joint ≈ 0.98 ΔP^spin + 0.90 ΔP^mass` (R² 0.973, against 0.54 and 0.57
   for either alone), and the per-event log-Bayes-factor interaction term has median |I|
   0.033 against a median |log BF_joint| of 0.70 (4.8%; 125 events above 0.1) — near-additive
   in the bulk, not in the tail. On the global fraction the joint gain, 0.830, is slightly
   less than the product of the single-mark gains, 0.812: the marks overlap a little in
   what they say about `f`, and both are anticorrelated with it.
5. **`ρ(Δμ_G, Δμ_χ)`** = +0.272: weak and positive, the sign expected when both marks compete
   for the same AGN branch weight (a larger `Δμ_G` makes the branch louder and lowers `f`,
   which pushes `Δμ_χ` up through its own −0.52 anticorrelation with `f`).
6. **`ρ(f, Δμ_G)`** = −0.580 in the joint arm, −0.612 in the mass-only arm: the strongest
   degeneracy in the problem and the new one. It is the selection coupling: the same data
   are fit by a lighter, more numerous AGN branch or a heavier, rarer one.
7. **Does the combined intrinsic information improve routing beyond spin-only?** Yes.
   Called-AGN counts at 0.5 go 41 (spatial) → 104 (spin) → 191 (mass) → 231 (joint); the
   AUC for the true label goes 0.632 → 0.708 → 0.748 → 0.781; accuracy 0.656 → 0.685 →
   0.718 → 0.734. The joint model is still far from a classifier (231 called AGN against 357
   true, 70 false positives), which is the same statement Analyses 8 and 9 made.
8. **Which events are identified by mass versus spin?** Different ones, by the numbers in 4;
   `figs/fig_event_mass_vs_spin_map_*.png` shows the near-uncorrelated cloud. Read after the
   fact, the true AGN hosts move upward under both marks (median `ΔP` +0.041 spin, +0.113
   mass, +0.148 joint) and true GAL hosts barely move (−0.024, +0.001, −0.005).

### Selection and closure

The existing injections support the whole registered grid: minimum `N_eff`/threshold 8.17
on the posterior-relevant region and 5.58 anywhere in the mass-only arm. The joint cube
rejects 2,500 of 67,527 cells (3.7%), all at `f ≥ 0.35` in the extreme corners of both mark
axes (`Δμ_χ` at −0.20 or ≥ +0.2425, `Δμ_G` at ±10), where the accepted boundary density is
2.2e-27 of the peak; filling every rejected cell with that value bounds the mass behind the
guard at 1.5e-25. No result stands behind a rejected cell. The two-mark likelihood reduces
bitwise to the shared-population model at zero marks and to the spin-only model at zero mass
mark (the χ slab re-check: 7 of 8 on-lattice cells bitwise, worst 2 ULP), the GAL branch is
frozen at `f = 0`, and the mass coordinate moves the likelihood by 10.5 nats per Msun.

### The one refinement, measured

The registered 1-Msun `Δμ_G` axis under-resolved the posterior (fewer than two nodes across
the 68% width), so six half-integer nodes were added inside [2, 8] as additive rows. On the
mass-only arm the median moved by 0.001 and the 68% width fell from 1.87 to 1.40 Msun; on
the joint arm from 1.654 to 1.312, the median by 0.002, `f` and `Δμ_χ` by < 1e-4. The
coarse trapezoid was over-stating the mass-mark width by a third; the refined numbers are
the ones quoted. No second refinement was made.

### Fixed-`H0` owner gate

All six registered conditions hold: selection valid over posterior support; closure;
mass coordinate live; `Δμ_G = +5` recovered with zero excluded; the mass mark adds
identifiable information beyond spin-only (`f` width J/χ 0.94, `Δμ_χ` width J/χ 0.92, and
the mark itself measured to ±0.6 Msun); no pathological degeneracy (largest |ρ| 0.58).
**The gate PASSES**; the `H0` release proceeds as the next stage.

### Limitations

One realisation, seed 100, one hyperparameter point for the event-level statistics (the
MAP; the provisional node `f = 0.350` gave the same picture: Pearson +0.13, Jaccard 0.32).
`Δμ_χ` is recovered at 90% but not at 68% (+1.1 sd); on one draw that is unremarkable, and
the spin-only arm's larger offset shows it is partly the mass mark being absorbed into the
spin coordinate when the mass mark is switched off. The event-level nested-model sums are
exact (12 of 12 at 1 ULP), but the four models share the production sample mask only up to
its model dependence (`keep = valid & isfinite(ldw)`), which changes the kept-sample count
by at most 64 of 2000 in a few events; each model's sum verifies against its own production
call, so the statistics are the production numbers. Nothing here is a statement about real
BBHs in AGN.

## Analysis-9 mechanism follow-up (Part I of this task)

Full write-up: `../analysis_9_marked_multitracer_H0_fagn/MECHANISM_FOLLOWUP.md`.

**Did the `H0` improvement remain when `f_AGN` was fixed? Yes — all of it.** With
`f_AGN = 0.275` (the shared S9/J9 MAP node) held fixed in both arms, the marked
`H0` interval is **0.8879** (68%) and **0.8985** (90%) of the spatial-only one on the
matched 28-node lattice, against 0.8879 and 0.8996 with `f` marginalised. Written
as an exact product, the global-mixture factor is 0.99997 (68%) and 1.0013 (90%):
the tighter `f_AGN` contributes nothing to the gain at 68% and slightly opposes it
at 90%. The fixed-`f` ratio stays between 0.883 and 0.892 (68%) across every `f`
node inside the 90% interval. Fixing the mark as well changes the marked width by
0.25%.

**The improvement is driven by event-level tracer routing.** At
`(H0, f, Δμ_χ) = (69.0, 0.275, +0.1075)` the mark moves `P_i(AGN)` by more than 0.1
for 348 of 1000 events and across 0.5 for 71 (55 up, 16 down) while changing the
expected AGN count by only −1.0: it re-sorts events between the GAL and AGN
redshift structures. The added `H0` curvature at fixed `f` sits 81% in the event
term and 19% in the selection term (whose expectation is spin-independent, so that
share is estimator response, not mechanism); within the event term, the 75 events
with `|ΔP_i| > 0.2` (7.5%) carry 60% of the curvature change and the 71 crossers
61%, with rank correlation +0.34 between `|ΔP_i|` and the per-event curvature
change. Every per-event sum was verified against the recorded cubes at ten `H0`
nodes (34/34, worst one ULP). Analysis 9's `REPORT.md`/`STATE.md` attribution to
the global fraction has been corrected; no measured number changed.

---

## Provenance: the rejected peak-fraction mark (2026-09-21)

*The report as it stood when Gate M failed. It is kept as provenance and is
superseded, not corrected, by the owner decision above. Option (c) of the
options table is the one the owner chose.*

**Sources.** `diagnostics/a10_mass_mark_feasibility.json`, written by
`scripts/check_mass_mark_feasibility.py` (CPU, exit status 2). darksirens
`af896ca` on the pinned base `2b86a2d`, worktree clean; gws-agn `408990a`.
Nothing was generated, no likelihood was built or evaluated, no GPU second was
spent, and `working/data/**`, the Analysis-8 tree and the Analysis-9 tree were
read only. Every number below comes from that one JSON.

### The finding

The specification plants \(\lambda_{\rm peak}^{\rm AGN} = \lambda_{\rm peak}^{\rm GAL}
+ 0.15\) and registers the check \(0 < \lambda_{\rm peak}^{\rm fid} + 0.15 < 1\)
before generation. **The check fails.**

\[
  \lambda_{\rm peak}^{\rm fid} = 0.90, \qquad 0.90 + 0.15 = \mathbf{1.05}.
\]

The production `powerlaw+peak` model is a two-component mixture whose fiducial
puts **90% of the mass in the Gaussian peak and 10% in the power law**. The
specification's \(\Delta\lambda_{\rm peak} = +0.15\) reads naturally as "make the
AGN branch more peak-dominated", but the branch is already at 0.90, so there is
only 0.10 of headroom. The mark asks for 105%.

**The sampled coordinate is the other one.** There is no \(\lambda_{\rm peak}\)
parameter to set. `powerlaw+peak` has twelve slots and slot 0 is `v1`
(`$v_1$`), a stick-breaking input with fiducial **0.10** and prior bounds
\([0, 1]\); the Gaussian-peak fraction is the derived quantity
\(\lambda_{\rm peak} = 1 - v_1\). Planting the registered mark means asking the
AGN branch for

\[
  v_{1,c2} = 1 - 1.05 = \mathbf{-0.05},
\]

a **negative** power-law weight. This is not a grid that is too narrow or a
prior that is too tight. It is outside the simplex, so no widening admits it.

Three independent lines of evidence fix \(\lambda_{\rm peak}^{\rm fid} = 0.90\):

1. **The registry.** `Curated(latex="PL+G", weights=(0.10,))  # w_PL=0.10,
   w_G=0.90`, where `Curated.weights` holds desired *final fractions*, converted
   to stick-breaking inputs by `_w_to_v`. The forward map
   `_stick_breaking_weights([0.1])` returns `[0.1, 0.9]` in composition order
   `(PowerLaw, Gaussian)`, and the inverse round-trips with error exactly 0.0.
2. **The model itself, numerically.** Integrating the fiducial mass density
   (`model.log_p_pop` on an \(m_1\) grid at fixed \(q = 0.9\), \(z = 0.1\),
   \(\chi_{\rm eff} = 0\)), the mass in the peak band [25, 45] falls
   0.8132 → 0.6322 → 0.1146 and the mass in the power-law band [5, 20] rises
   0.1397 → 0.3168 → 0.8231 as \(v_1\) goes 0.10 → 0.25 → 0.90. Raising \(v_1\)
   drains the peak, so \(v_1\) is the power-law weight and the fiducial peak
   fraction is 0.90.
3. **The data.** The seed-100 marked mock,
   `working/data/seed100/events/events_marked_dmu0p10.h5`, records its own
   generation configuration: `"population": {..., "peak_fraction": 0.9, ...}`.
   Its 1000 detected events have **92.4%** of `true_m1src` in [25, 45] (GAL
   93.2%, AGN 90.5%), a median of 36.5 \(M_\odot\) against
   \(\mu_{\rm G} = 35\), and **0.6%** below 20 \(M_\odot\). The detected mass
   distribution is already peak-dominated, exactly as a 0.90 peak fraction
   requires, and there is visibly no room above it for +0.15.

The mark was **not** altered and nothing downstream was run, per the
specification's instruction.

### One thing that is ready, and one that is not

**The inference side is ready.** `build_parameter_space`, called exactly as
`analysis_8/scripts/a8_likelihood.py::build` calls it with `n_catalogs = 2`,
accepts a per-catalog mass weight and emits it:

    per_catalog_pop_params=('v1_c2', 'mu_chi_c2')
      -> ['H0', 'log10n0', 'delta', 'sigma_kde', 'log10n0_c2', 'delta_c2',
          'sigma_kde_c2', '$v_1$_c2', '$\mu_\chi$_c2', 'fcat_2']

Ten labels against Analysis 8's nine; the resolver does not refuse. (Informative
only: the Analysis-8 experience is that a coordinate can be *emitted* and still
be dead in the likelihood, which is why closure identity 10.6 — mark liveness
for \(\Delta\lambda_{\rm peak}\) specifically — is registered in `GATES.md`.)

**The generator is not.** `working/data/generate_dataset.py` carries exactly one
branch-dependent mark, `--dmu_chi_agn`, and it works by shifting an
**already-drawn** spin value, which is why the RNG stream, the host labels, the
masses, the sky, the distances and the detected set come through untouched. A
mass mark cannot borrow that trick: changing a mixture weight changes which
component each event is drawn from, hence the masses, hence — through
\(\rho_{\rm opt} \propto \mathcal{M}_{\rm det}^{5/6}/d_L\) — the detected set
itself. Masses today come from a single shared `PopulationConfig` with one
population-wide `peak_fraction`. Whatever mark is chosen, planting it is a
generator change, and that is a second owner decision beyond the value.

### Options for the owner — none executed

| | mark | AGN peak fraction | \(v_{1,c2}\) | in \([0,1]\) | what it costs the specification |
|---|---|---|---|---|---|
| **(a)** | \(\Delta\lambda_{\rm peak} = -0.15\) | 0.75 | 0.25 | yes, comfortably | keeps the registered **magnitude** and the registered **coordinate**; flips the sign. Physically: AGN mergers are *less* peak-dominated, i.e. more power-law, than field mergers |
| **(b)** | \(\Delta\lambda_{\rm peak} = +0.05\) | 0.95 | 0.05 | yes, but 0.05 from the edge | keeps the **sign**, loses the magnitude. \(v_{1,c2} = 0.05\) sits 0.05 from the prior boundary, so a symmetric scan axis **clips**: the branch coordinate admits only \(\Delta\lambda_{\rm peak} < +0.10\), and any node at or past +0.10 is \(v_{1,c2} \le 0\) and must be dropped, not evaluated |
| **(c) — CHOSEN** | mark a different single mass coordinate, e.g. \(\mu_{\rm G}\) (peak **location**, fiducial 35, bounds [20, 50]) | 0.90 both branches | n/a (`G.mu_c2`) | yes | **changes the specification's coordinate.** The scientific question becomes "do AGN mergers peak at a different mass" rather than "are AGN mergers more peak-dominated". Both are single mass marks and both respect the scope lock, but they are different measurements. Owner's call |
| **(d)** | re-pin the fiducial \(\lambda_{\rm peak}\) so that +0.15 fits | — | — | — | **not admissible.** \(\lambda_{\rm peak} = 0.90\) is the value the seed-100 catalogs, events, injections and every Analysis 0–9 result were generated and measured at. Changing it invalidates the shared dataset, not just Analysis 10 |

**The owner chose (c)**, at \(\Delta\mu_{\rm G} = +5\,M_\odot\) (\(\mu_{\rm G}: 35 \to 40\)). The paragraph below is the recommendation as it
stood before that decision and is kept unchanged.

**Which option preserves the specification's intent.** (a) is the only option
that keeps both the registered coordinate and the registered magnitude
\(|\Delta\lambda_{\rm peak}| = 0.15\); it changes the sign, and with it the
physical claim. (b) keeps the sign and the coordinate but neither the magnitude
nor a clean scan axis. (c) keeps the "one mass mark" scope but not the
coordinate. (d) is ruled out by the dataset, not by taste.

### Cost sketch for option (a) — not a commitment

Measured rates, not extrapolations: **3.0203 s/eval** for the K=2 marked
likelihood on a rita A100-80 (Analysis 9, J9, median over 70,028 cells;
S9 measured 3.0219), and **1.7095 s/eval** for the identical likelihood on the
local H100 NVL (Analysis 8) — the H100 stays free, so everything below is at the
rita rate. A \(41 \times 61\) plane is 2,501 cells = **2.10 GPU-h**; the same
plane cost Analysis 8 1.19 GPU-h on the H100. A 3-D fixed-\(H_0\) arm is
\(41\,f \times 61\,\mu \times N_\lambda\) and scales **linearly** in
\(N_\lambda\).

| arm | cells | GPU-h at 3.0203 s/eval |
|---|---|---|
| A10-S spatial only, both marks off | 41 | 0.03 |
| A10-χ \((f, \Delta\mu_\chi)\), \(41 \times 61\) | 2,501 | 2.10 |
| A10-M \((f, \Delta\lambda_{\rm peak})\), \(41 \times N_\lambda\) | 451 / 861 / 1,681 | 0.38 / 0.72 / 1.41 |
| A10-J joint, \(41 \times 61 \times N_\lambda\) | 27,511 / 52,521 / 102,541 | 23.1 / 44.1 / 86.0 |
| **all four** | | **25.6 / 46.9 / 89.6** |

for \(N_\lambda\) = 11 / 21 / 41 nodes. A \(\Delta\lambda_{\rm peak}\) axis for
option (a) has room: the branch coordinate stays inside \((0, 1)\) for
\(\Delta\lambda_{\rm peak} \in (-0.90, +0.10)\), so an axis such as
\([-0.35, +0.05]\) at 0.01 (41 nodes) or 0.02 (21 nodes) brackets the planted
\(-0.15\) without touching either edge. Add Gate 10's closure suite — Analysis
9's ran in 9 min 23 s — and the \(H_0\) release, if it is ever gated open,
multiplies A10-J by the \(H_0\) node count (Analysis 9 used 28 nodes and spent
58.7 GPU-h on a 2-D-plus-\(H_0\) cube).

**Two costs are not in that table, and one of them is a gate, not an
arithmetic.** Generating the two-mark mock and, if needed, a new injection set
is not costed here because the generator path does not exist yet. And the
detection rule of the seed-100 mock is `observed-data` with
\(\rho_{\rm obs} \ge 8\), where \(\rho_{\rm opt}\) depends on the **chirp mass**
— so unlike the spin mark, a mass mark changes detectability branch by branch.
**Gate B (selection support) must run before any inference**: \(N_{\rm eff}\) of
the existing `injections_targeted.h5` has to be measured under *both* branch
populations across the registered grid, not only at the planted cell, and if the
existing proposal never covered the shifted mass distribution then reweighting
it is not an acceptable substitute and a targeted injection set is required.
Analyses 8 and 9 could skip this only because \(\chi_{\rm eff}\) does not enter
\(\rho_{\rm opt}\) at all.

### Status

Analysis 10 is **stopped at specification §9, before generation**. Gate 0 passes;
Gate M fails at \(0.90 + 0.15 = 1.05\); every gate after it is NOT RUN with its
criteria registered in `GATES.md` so that whatever mark is chosen inherits a
ledger written before its first number exists.

The decision is the owner's.
