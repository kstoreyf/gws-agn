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
