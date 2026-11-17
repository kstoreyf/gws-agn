# The likelihood the gws-agn analyses evaluated (code-backed, 16 Nov 2026)

Purpose: an exact statement of the dark-siren likelihood as the code evaluated it, for
writing the paper's Method section, plus the explanation of why `log10n0` moves
`f_AGN` when the catalogues are complete (Section 8).

How to read this file.
- Every equation is followed by one plain-language sentence and by `file:line` pointers
  at the commit that ran it. Paths are relative to the darksirens package directory of
  the worktree named (`/hildafs/projects/phy230014p/magana/src/darksirens-0c5b3db/darksirens/`
  and so on).
- `likelihood/core.py`, `likelihood/selection.py`, `inference/utils.py`,
  `redshift/prior.py` and `redshift/catalog.py` are byte-identical at 2b86a2d and
  0c5b3db (`git diff --stat 2b86a2d 0c5b3db` lists none of them), so 0c5b3db line
  numbers apply to Analyses 0–7. At af896ca (Analyses 8–12) only `likelihood/core.py`
  (population blocks), `redshift/completion.py` (no `c_mode`), `inference/parameters.py`,
  `likelihood/factory.py`, `inference/loaders.py` and the population registry differ.
- **[inference]** marks a statement not read directly in code.
- Section 7 (darksirens-core e7c3007) was mapped by a separate code read; its pointers are
  under `/hildafs/projects/phy230014p/magana/src/darksirens-core-e7c3007/src/darksirens/`.

---

## 0. Runs of record and the options they passed

| analyses | code | model | catalogues | `c_mode` | `log10n0` (GAL, AGN) | other options |
|---|---|---|---|---|---|---|
| 0, 1 (K = 1), 2 (K = 2) | darksirens 2b86a2d | `dark_sirens`, `catalog_sky_weighting=field` | complete GAL and AGN, nside 32 | none (legacy per-pixel ratio; inert, see below) | (−24, −24) | `use_LSS=False`, fixed population, Ωm = 0.3075 |
| 3, 6, 7 | darksirens 0c5b3db | same | flux-limited rungs m<21…m<18 and complete; equal nside in both catalogues of every cell (32; Analysis 7 also 16 and 64) | `selection`, Schechter fits | (−3, −5), the mock's input densities | same |
| 4 | 0c5b3db | same | same | `selection` | (−3, −5 + log10 x), x ∈ {0.5, 0.7, 0.9, 1.1, 1.3, 2.0} | same |
| 5 | 0c5b3db | same, dynesty | same | `selection` | sampled, flat on [−4, −2] and [−6, −4] | `sample_4d.py` |
| 8–12 | darksirens af896ca | same | complete GAL and AGN, nside 32 | none | (−24, −24) | per-catalogue population blocks (`per_catalog_pop_params`, e.g. `('mu_chi_c2',)`) |
| 13 | darksirens-core e7c3007 | `ds.model(catalog=[GAL, AGN], catalog_sky_weighting="field", completeness=None → "incomplete" (per-row count ratio), per_catalog_population={2: [μ_G, μ_χ]})`, dynesty | complete GAL and AGN, nside 32 | count ratio (= legacy per-pixel; inert) | (−24, −24) | `max_likelihood_variance=20` while sampling; γ pinned to 0 (core's own `fixed=True` default is γ = 2.5); Section 7 |

Common to all legacy runs (drivers `analysis_2_complete_catalog_H0_fagn/scripts/scan_h0f.py:316-347`,
`selection_redo/scripts/scan_h0f.py:359-398`, `selection_redo/scripts/sample_4d.py:183-240`,
`analysis_8_spin_marked_fagn_fixed_H0/scripts/a8_likelihood.py:68-89,196-235`; recorded in the
result HDF5 attributes `arg_*`, `base_coord_labeled`, `darksirens_git_sha`):
- population `powerlaw+peak`, fixed at the registry fiducials except where an analysis frees a
  block: v₁ = 0.10 (power-law weight), α_PL = 2.3, m_min = 5, m_max = 80, δm_min = 3, δm_max = 10,
  μ_G = 35, σ_G = 5, β = 1, μ_χ = 0, σ_χ = 0.1, **γ = 0**;
- δ = 0, σ_kde = 0, b_miss = 1 (inert: no density field, Section 4), Ωm = 0.3075, w0 = −1, wa = 0;
- `selection_neff_guard=hard`, `max_likelihood_variance=1e6` (Section 6);
- catalogue KDE window W = 4096 galaxies, half-width 8 σ_max (`kde_window`, `kde_window_nsigma`);
- `z_depth = None` (the survey files carry no `z_depth` attribute, `catalogs/io.py:90`, and the
  drivers do not set `resolved_survey_z_depths`, `inference/parameters.py:579`);
- the internal redshift grid runs to **z_max = 5** (`redshift/grid.py`, 1000 log-spaced nodes;
  `DARKSIRENS_ZMAX` is not set by any Analysis 2–7 driver; the selection-mode value is confirmed
  numerically in Section 8.3);
- grids: H0 ∈ [50, 100] (201 points), f_AGN ∈ [0, 1] (41 points), flat priors.
- data: 1000 events × 2000 PE samples; 2,205,380 detected injections out of N_draw = 1.5×10⁸
  (`injections_targeted.h5`).

---

## 1. The hierarchical likelihood (Question 1)

Sample coordinates: detector-frame primary mass m₁ᵈᵉᵗ, mass ratio q = m₂ᵈᵉᵗ/m₁ᵈᵉᵗ, luminosity
distance d_L, χ_eff and sky position Ω̂, which selects one HEALPix pixel p_k(Ω̂) in each
catalogue k. The redshift is z = z(d_L; H0, Ωm) and m₁ = m₁ᵈᵉᵗ/(1+z).

**Target density in sample coordinates.**
$$
\pi(\theta\mid\Lambda)=
\frac{p_{\rm pop}(m_1,q,\chi_{\rm eff}\mid\Lambda)\,(1+z)^{\gamma-1}\;p_{\rm mix}(z,\hat\Omega)}
{(1+z)\,\dfrac{d d_L}{dz}(z)} .
$$
The population density in source-frame mass, mass ratio and spin, times the redshift-rate factor,
times the host density of Section 3, converted to detector-frame mass and distance by the
Jacobian |∂(m₁ᵈᵉᵗ, q, d_L)/∂(m₁, q, z)| = (1+z) dd_L/dz.
- `inference/utils.py:64-85` (Jacobian), `:103-131` (target density), `:186-252` (weight);
  dd_L/dz = d_L/(1+z) + c(1+z)/(H0 E(z)) at `utils/cosmology.py:476-483`.
- p_pop is normalised over (m₁, q, χ_eff): mass components by their own grid integrals, the
  pairing q^β normalised at each m₁, the truncated-Gaussian χ_eff by its integral
  (`gw/populations/base.py:493-533`).
- The rate factor (1+z)^{γ−1} is applied inside p_pop (`gw/populations/base.py:741-749`;
  per-component γ broadcast, shared here). With γ = 0 it is (1+z)⁻¹, the source-to-detector
  time dilation, so the merger rate per host is constant in source-frame time.
- The comoving volume element enters only through p_mix (Section 2), never through p_pop.

**Per-sample weight.**
$$ w_{ij}=\frac{\pi(\theta_{ij}\mid\Lambda)}{p_{\rm pe}(\theta_{ij})}. $$
The posterior samples are reweighted from the parameter-estimation prior to the model.
p_pe is the PE prior in (m₁ᵈᵉᵗ, q, d_L, χ_eff) (`gw/utils.py:134-140`); the loader divides it by
its per-event sum (`gw/utils.py:236-238`), which multiplies each event term by a constant and
does not affect inference. Samples with d_L outside the distance table, non-finite weight or
p_pe ≤ 0 get w = 0 but still count in n (`likelihood/core.py:765-788, 850-877`).

**Event term.**
$$ \ln\hat{\mathcal Z}_i=\ln\Big(\frac1{n}\sum_{j=1}^{n}w_{ij}\Big),\qquad n=2000 . $$
Each event contributes the Monte Carlo average of its weights.
`likelihood/selection.py:170-228`; events reduced in blocks of 25 (`likelihood/core.py:912-947`).

**Selection term.**
$$
\hat\mu(\Lambda)=\frac1{N_{\rm draw}}\sum_{k\,\in\,\rm det}\frac{\pi(\theta_k\mid\Lambda)}{p_{\rm draw}(\theta_k)},\qquad
N_{\rm eff}=\frac{\hat\mu^2}{\widehat{\rm Var}(\hat\mu)},\quad
\widehat{\rm Var}(\hat\mu)=\frac{\sum_k w_k^2}{N_{\rm draw}^2}-\frac{\hat\mu^2}{N_{\rm draw}} .
$$
The expected detected fraction is the injection average of the same target density, divided by
the injection draw density p_draw in the same (m₁ᵈᵉᵗ, q, d_L, χ_eff) basis.
`likelihood/selection.py:127-167` (μ̂, N_eff), `:405-512` (batched sum, 50,000 injections per batch);
`likelihood/core.py:790-817, 825-833`. The selection integrand uses **the same** p_mix, with the
same per-catalogue normalisers Z_k, as the event term: for `dark_sirens` the selection redshift
prior is the PE one (`likelihood/core.py:73-89, 471-474, 737-741`), and the PE and selection
prior states are built from the same full-sky field inputs and shared object-for-object
(`likelihood/core.py:652-679`, `likelihood/factory.py:503-509`).

**Total.**
$$
\ln\mathcal L(\Lambda)=\sum_{i=1}^{N}\ln\hat{\mathcal Z}_i-N\ln\hat\mu+\frac{N(N+3)}{2N_{\rm eff}},\qquad N=1000,
$$
set to −∞ unless N_eff > max(5N, N²/max(V_max − Σ_iσ_i², 10⁻¹²)) (Section 6).
The rate is marginalised with a 1/R prior (giving μ̂^{−N}); the last term is the Farr (2019)
correction for the Monte Carlo uncertainty in μ̂, always applied.
`likelihood/selection.py:231-354` (correction at `:350-354`), `likelihood/core.py:948-959`.

Sky: the isotropic sky model adds no factor (`likelihood/core.py:419-438`). The host density is a
probability per pixel (Section 2), so Ω_pix is a constant common to the event and selection
terms and cancels. **[inference, not checked in the mock generator]** the events file states its
PE prior is flat in (α, δ) as well as in (ln M_c, ln q, ρ, χ_eff); the code divides only by the
(m₁ᵈᵉᵗ, q, d_L, χ_eff) part, so each event's weights carry an extra sec δ relative to a
per-steradian host density. It is common to both mixture branches and nearly constant across one
event's localisation.

---

## 2. One catalogue's host intensity under `field` weighting (Question 2)

For catalogue k, pixel p (nside 32, Ω_pix = 4π/12288), galaxies i ∈ p with photometric redshift
z_i, width Δz_i = 0.003(1+z_i) (the survey files' `dzgals`) and weight w_i = 1:
$$
p_{\rm cat}(z\mid p)=g(z)\sum_{i\in p}\tilde w_i\,\frac{\mathcal N(z;z_i,\sigma_i)}{\int_0^{z_{\max}}\mathcal N(z';z_i,\sigma_i)\,g(z')\,dz'},\qquad
g(z)=\frac{dV_c}{dz\,d\Omega}(1+z)^{\delta},\quad
\tilde w_i=\frac{w_i}{\sum_{j\in p}w_j},
$$
with σ_i = max(√(Δz_i² + σ_kde²), 10⁻⁴) = 0.003(1+z_i) and δ = 0.
Each galaxy contributes a unit-mass redshift kernel, its photometric Gaussian tilted by the comoving
volume element; every galaxy in a pixel carries the same host probability.
`redshift/catalog.py:459-516` (weights, widths, per-kernel norm), `:391-415` (norm by Gauss–Legendre),
`:693-765` (per-sample evaluation, windowed to 4096 galaxies within 8σ_max),
`redshift/completion.py:324-335` (g).

**Numerator (hosts per unit redshift in pixel p).**
$$ n_k(z,p)=N_{{\rm obs},k}(p)\,p_{\rm cat}(z\mid p)+\frac{dN_{{\rm miss},k}}{dz}(z,p). $$
The catalogued galaxies of the pixel spread over their kernels, plus the modelled uncatalogued hosts
(Section 4). `redshift/prior.py:653-668`, counts at `:567-579`.

**Normalisation: one survey-global constant per catalogue.**
$$
p_k(z,p)=\frac{n_k(z,p)}{Z_k},\qquad
Z_k=\sum_{\rm all\ pixels}\int_0^{z_{\max}}n_k(z,p)\,dz
=N^{\rm tot}_{{\rm obs},k}+\sum_{\rm all\ pixels}\int_0^{z_{\max}}\frac{dN_{{\rm miss},k}}{dz}\,dz .
$$
p_k is the probability that a host of class k lies in pixel p at redshift z: normalised once over the
whole sky (occupied and empty pixels) and the whole internal grid 0 < z < z_max = 5, not per pixel and
not by a volume. Relative angular host density is kept (a pixel with 100 galaxies carries 100 times
the weight of a pixel with one). `redshift/prior.py:586-598` (state), `:671-672` (division);
`redshift/completion.py:1580-1642` (Z_k), built from the full-sky rows before compaction
(`inference/loaders.py:265-273`, `redshift/completion.py:470-557`).
The per-pixel ("conditional") normaliser Z[p] (`redshift/prior.py:583-584, 673`) is not used by any
run of record.

Analyses 0–2 and 8–12 (log10 n0 = −24): the missing term is ≤ 10⁻²⁴ × V(z<5) ≈ 2×10⁻¹² hosts, so
$$p_k(z,p)=\frac{N_{{\rm obs},k}(p)\,p_{\rm cat}(z\mid p)}{N^{\rm tot}_{{\rm obs},k}}$$
to 10⁻¹² relative: the catalogue's own host distribution, normalised by its total count.

---

## 3. The two-catalogue mixture (Question 3)

$$
p_{\rm mix}(z,\hat\Omega)=(1-f_{\rm AGN})\,p_{\rm GAL}\big(z,p_{\rm GAL}(\hat\Omega)\big)+f_{\rm AGN}\,p_{\rm AGN}\big(z,p_{\rm AGN}(\hat\Omega)\big),\qquad f_{\rm AGN}\equiv{\tt fcat\_2}.
$$
The host density is a weighted sum of the two catalogues' densities, each already normalised by its
own Z_k. Survey order [GAL, AGN]; weights from the stick `fcat_2`, (ln(1−f), ln f)
(`inference/parameters.py:49-74, 405-411`); combined per sample by an all-−∞-safe log-sum-exp
(`likelihood/core.py:617-627, 681-703`). Each catalogue keeps its own pixelisation, survey parameters
(n0, δ, σ_kde, completeness) and Z_k (`likelihood/core.py:553-555, 652-679`).
The same p_mix multiplies the event and the selection integrands (Section 1), so f_AGN enters
μ̂ as well as every event term.
Because both p_k are probabilities per pixel, mixing them is coherent only at equal nside; every cell of
record pairs equal nside (checked in the Analysis 6 and 7 queues). The code does not convert to a
per-steradian density, so a mixed-nside cell would weight the branches by the ratio of pixel areas.

---

## 4. The missing-host term under `c_mode = selection` (Question 4)

$$
\frac{dN_{{\rm miss},k}}{dz}(z,p)=\big[1-C_k(z)\big]\;n_{0,k}\,\Omega_{\rm pix}\,\frac{dV_c}{dz\,d\Omega}(1+z)^{\delta}\;\max\!\big(1+b_{\rm eff}\,\delta_g(p,z),\,0\big).
$$
Uncatalogued hosts of class k are the expected count n0 dV_c not covered by the completeness.
`redshift/completion.py:824-876` (dN_exp = n0 Ω_pix g), `:1252-1290` (assembly), `:1308-1327` (density
factor). n0 is in Mpc⁻³ (`core/types.py` SurveyParams; dV_c in Mpc³, `utils/cosmology.py:464-472`),
so the missing count scales as n0 H0⁻³.

**Density field: none.** With `use_LSS=False` (every selection-mode driver:
`selection_redo/scripts/scan_h0f.py:366-369`, `sample_4d.py:197-200`) δ_g is the (1, N_grid) zero
array (`inference/loaders.py:241-247`) and there is no Q_LSS table, so the factor is 1: **the missing
hosts are uniform in comoving volume (δ = 0) and identical in every pixel, occupied or empty.**

**Completeness.** Under `c_mode=selection` the per-pixel count ratio is replaced by one parametric
curve, the same in every pixel (`redshift/completion.py:894-930, 1226-1231`):
$$
C_k(z)={\rm clip}\!\left[\frac{\Gamma\big(\alpha_k+1,\,x_{\rm lim}(z)\big)}{\Gamma\big(\alpha_k+1,\,x_{\rm faint}\big)},0,1\right],\quad
x_{\rm lim}(z)=10^{-0.4\,[m_{{\rm lim},k}-DM(z)-M^*_k]},\quad
x_{\rm faint}=10^{-0.4\,\Delta M_{\rm faint}},\quad
M^*_k=\hat M^*_k+5\log_{10}h .
$$
The fraction of a Schechter luminosity function, truncated faintward at M* + ΔM_faint, brighter than the
magnitude limit at redshift z. The h-scaling makes C independent of H0 (`redshift/selection.py:169-171,
247-290`).

Values used (fit JSONs in `selection_redo/scripts/selection_fits_truez/`; recorded in the result
attributes `selection_theta_pinned_at`, `selection_fit_paths`):

| rung | catalogue | m_lim (pinned datum) | M̂* (pinned) | α (pinned) | z where C first < 1 (H0 = 67.74) |
|---|---|---|---|---|---|
| complete | GAL | 24.0 | −19.62387 | −1.06978 | 1.16 |
| complete | AGN | 25.0 | −19.62448 | −1.06815 | 1.69 |
| m<21 | GAL / AGN | 21.0 | −19.6244 / −19.6194 | −1.0708 / −1.0598 | 0.37 |
| m<18 | GAL / AGN | 18.0 | −19.6158 / −19.6261 | −1.0494 / −1.0547 | 0.11 |

- M̂* and α are sampled labels under `c_mode=selection` but are **pinned at the fit centres** by the
  drivers (`selection_redo/scripts/scan_h0f.py:508-523`; `sample_4d.py:342-360`).
- m_lim is a fixed datum per catalogue (`cli/inference.py:2505-2521`).
- ΔM_faint = −0.094352906 for every fit: the mock's luminosity cut, −2.5 log10 x_cut
  (`selection_redo/scripts/lf_constants.py`); one value per run, pinned unsuffixed and broadcast to the
  AGN catalogue (`inference/parameters.py:159-168`, equality enforced at `cli/inference.py:2522-2538`).

**Normalisation of the completed intensity.** Because C_k is pixel-independent and δ_g = 0,
$$
Z_k=N^{\rm tot}_{{\rm obs},k}+n_{0,k}\,4\pi\int_0^{z_{\max}}\big[1-C_k(z)\big]\frac{dV_c}{dz\,d\Omega}\,dz,\qquad z_{\max}=5,
$$
(Σ Ω_pix = 4π over occupied and empty pixels; `redshift/completion.py:1645-1812`, row term
`:1716-1730`, empty pixels `:1793-1802`, quadrature `:1639-1641`). The CPU check in Section 8.3
reproduces `field_global_log_Z` with this closed form to 10⁻¹⁵.
Values at H0 = 67.74 and the input densities (n0 = 10⁻³, 10⁻⁵ Mpc⁻³):

| rung | Z_GAL | F_GAL = N_obs/Z | Z_AGN | F_AGN | share of missing mass at z > 1 |
|---|---|---|---|---|---|
| complete | 1.94×10⁹ | 0.0778 | 1.63×10⁷ | 0.0930 | 100 % |
| m<21 | 2.12×10⁹ | 0.0114 | 2.12×10⁷ | 0.0115 | 93 % |
| m<18 | 2.12×10⁹ | 0.00039 | 2.12×10⁷ | 0.00039 | 92 % |

So 90–99.96 % of every Z_k is modelled hosts, and nearly all of that mass lies at z > 1: beyond the
catalogue edge (z ≤ 1.01), beyond every PE sample (z ≤ 0.65 even at H0 = 100) and beyond essentially
every detected injection (99.999 % below z = 1.06 at H0 = 100).

---

## 5. Host-dependent population (af896ca, Analyses 8–12) (Question 7)

With per-catalogue population blocks the population moves inside the branch sum:
$$
\pi(\theta\mid\Lambda)=\frac{\sum_{k}w_k\;p_{\rm pop}(m_1,q,\chi_{\rm eff}\mid\Lambda_k)\,(1+z)^{\gamma-1}\;p_k\big(z,p_k(\hat\Omega)\big)}{(1+z)\,\dfrac{dd_L}{dz}(z)\;p_{\rm pe\ or\ draw}},\qquad (w_1,w_2)=(1-f_{\rm AGN},\,f_{\rm AGN}).
$$
Each branch pairs its own population (e.g. its own μ_χ, or its own Gaussian-peak location) with its own
host density, then the branches are summed.
`likelihood/core.py@af896ca:576-610` (switch; shared model when no block is requested), `:754-772`
(open branches), `:812-856` (weight), used for PE at `:885-893` and for selection at `:933-943`.
Catalogue 2's population vector reads `<label>_c2`, defaulting element by element to catalogue 1's value
(`inference/parameters.py@af896ca:262-276`). Analyses 8–9 request `('mu_chi_c2',)`
(`a8_likelihood.py:9-18`, `a9_scan.py:508`); Analysis 10 adds the Gaussian-peak location `G.mu_c2`
(`a10_likelihood.py:81, 237-254`).

**f_AGN is a fraction of the astrophysical population, not of detected events.** Both p_pop(·|Λ_k) are
unit-normalised over (m₁, q, χ_eff); both p_k are unit-normalised over sky and 0 < z < z_max; the same
mixture appears in the event and the selection integrands; nothing divides a branch by its own detection
fraction. Integrating the source-frame rate, Σ_k f_k ∫p_pop,k ∫p_k (1+z)^γ = Σ_k f_k for γ = 0, so f_k is
the share of mergers per unit source-frame time whose host is of class k, counted over the modelled host
population of 0 < z < z_max (catalogued plus modelled-missing; Section 8 shows why that qualifier matters).
The detected AGN share differs from f_AGN by the branch-dependent detection efficiency, which μ̂ carries.
With the γ shared and equal to 0, the detector-frame shares differ from f_k by ⟨(1+z)⁻¹⟩_k only.

---

## 6. Monte Carlo variances and the guard (Question 8)

Definitions (`likelihood/selection.py`):
- per-event variance of ln Ẑ_i (delta method):
  $$\sigma_i^2=\frac{\sum_j w_{ij}^2}{\big(\sum_j w_{ij}\big)^2}-\frac1n\in\Big[0,1-\frac1n\Big]$$ (`:170-228`);
- `pe_variance_sum` = Σ_i σ_i² (`likelihood/core.py:951-958`);
- selection variance of N ln μ̂: N²σ²(ln μ̂) = N²/N_eff (`:58-83, 127-167`);
- `sigma2_total` = Σ_iσ_i² + N²/N_eff.

The guard (`:307-354`): ln L = −∞ unless
$$N_{\rm eff}>\max\!\Big(5N,\ \frac{N^2}{\max(V_{\max}-\sum_i\sigma_i^2,\ 10^{-12})}\Big),\qquad V_{\max}={\tt max\_likelihood\_variance}.$$
- `max_likelihood_variance` = V_max caps the variance of the total ln L estimator (darksirens default 1,
  the GWTC-4/5 criterion). Every legacy run passed 10⁶, which makes the second argument
  N²/(10⁶ − Σσ_i²) ≈ 1, so **the only active criterion is the effective-injection floor N_eff > 5N = 5000**
  (Vitale et al. 2022). Neither variance is penalised or added to ln L.
- darksirens does not store these numbers. The drivers record them per grid cell with a pass-through spy
  on `selection_log_correction` (`selection_redo/scripts/scan_h0f.py:299-347`, arrays `:612-631`; same in
  the Analysis 2 driver and `a8_likelihood.py:124-160`, which also records the selection contribution to
  ln L): `Neff`, `pe_variance_sum`, `selection_variance_N2_over_Neff`, `sigma2_total`, `threshold`,
  `passes`, `legacy_floor_5N`.
- `soft_guard` is off (`selection_neff_guard=hard`).
- Analysis 13 (core) is the only run with an active variance cap: V_max = 20 (Section 7).

---

## 7. darksirens-core e7c3007 (Analysis 13) (Question 6)

Mapped by a separate full code read of the core worktree, with CPU checks importing from it; pointers spot-checked
(`likelihood/weights.py:281-292`, `selection/gw.py:345-352, 400-409`, `catalog/mixture.py:161-176`,
`cosmology/_grid.py:14-19`). Pointers below are under
`/hildafs/projects/phy230014p/magana/src/darksirens-core-e7c3007/src/darksirens/`.

**Options of record** (`analysis_13_joint_shared_widths/scripts/a13_core_sampler.py:74-117`,
`submit_a13core_gpu.sbatch`, run record `results/a13core_e7c3007_rslice_cap20_historical_n200_s1.json`):
complete GAL and AGN catalogues (nside 32, no empty rows); `completeness=None`, which resolves to the per-row
count ratio for both (`analysis.py:554`); field normaliser in its "direct" form (`analysis.py:585-587`);
log10 n0 = −24 for both, δ = σ_kde = 0; no z_depth (core reads only a `z_depth` attribute, `catalog/io.py:67-71`);
z grid to 5 with 1000 nodes (`cosmology/_grid.py:14-19`; `DARKSIRENS_ZMAX` set nowhere in the A13 scripts,
**[inference]** for the submit-time environment); H0 sampled on [60, 76]; population powerlaw+peak with μ_G, σ_G,
μ_χ, σ_χ, μ_G,c2, μ_χ,c2 and fcat_2 sampled, the rest at the legacy fiducials and γ = 0; hard guard with
`max_likelihood_variance = 20` while sampling (10⁶ in the pre-flight); `kernel_window = 1e-10` (drops kernel
terms below 10⁻¹⁰ of the row's largest, `catalog/redshift.py:1108-1215`); no density-field extension.

**Same form as legacy, term by term.**
- Per-sample weight with the population inside the branch sum (`likelihood/weights.py:237-292`, called from
  `likelihood/hierarchical.py:358-376`):
  ln w = logsumexp_k[ln π_k + ln p_pop(m₁, q, χ_eff, z | Λ_k) + ln n_k(z|p_k) − ln Z_k] − ln(dd_L/dz) − ln(1+z) − ln p̃_pe,
  with Jacobian at `weights.py:67-94`, (1+z)^{γ−1} inside p_pop (`population/base.py:1396-1398, 1418-1434`),
  p_pe renormalised per event (`gw/samples.py:291-293`), the same sample mask (`likelihood/event.py:28-55`).
- Event term and σ_i² as in Section 1 and 6 (`selection/gw.py:209-267`).
- μ̂, N_eff, the N(N+3)/(2N_eff) term and the guard N_eff > max(5N, N²/max(V_max − Σσ_i², 10⁻¹²))
  (`selection/gw.py:130-206, 345-352, 400-409, 474-607`). The same mixture prior serves PE and selection
  (`likelihood/mixture.py:517-522`).
- Catalogue intensity, kernel and per-galaxy normalisation as in Section 2 (`catalog/models.py:140-142, 207-236`;
  `catalog/redshift.py:110-125, 235-262, 307-315, 1351-1413`), divided by one survey-global Z_k per catalogue
  over every full-sky row (`likelihood/mixture.py:456-464, 687-697`; `catalog/mixture.py:1-22`).
  One addition: the kernel is built once at H0_ref = 67.74 and shifted by +3 ln(H0/H0_ref), with a probe
  check that turns a failure into −∞ (`catalog/redshift.py:942-1000`).
- Missing hosts: dN_exp = n0 Ω_pix g (`catalog/completeness.py:228-229`); count-ratio C (`:44, 70-80, 166-183,
  262-290`), Schechter selection C̄ (`selection/catalog.py:260-286, 341-371`) and "complete" (no missing term,
  Z_k = ΣN_obs; `likelihood/mixture.py:145-177, 398-408, 436-442`). In every mode the normaliser integrates to
  the grid end z = 5 (selection "moments" form `catalog/mixture.py:161-176`; direct form `:142-158`).
- Mixture weights by the same stick (`catalog/mixture.py:60-78`); branch 2's population equals branch 1's except
  the absolute μ_G,c2 and μ_χ,c2 (`runtime_binding.py:361-382`); each p_pop(·|Λ_k) separately normalised
  (`population/base.py:231-267, 1026-1039, 1170-1221`). So fcat_2 is again the pre-detection fraction (Section 5).

**The n0-through-Z_k mechanism exists in core unchanged.** With A13's log10 n0 = −24 it is inert: Σ N_miss ≈
1.8×10⁻¹² and Z_k equals the catalogue count (CPU check on the A13 catalogues). Run like Analyses 3–7, core
would reproduce Section 8; core already reproduced the 0c5b3db Analysis 5 selection-mode run at the posterior
level (ΔlnZ = −0.57 ± 1.0, `experiments/experiment_darksirens_core_equivalence/README.md`).

**Differences that change numbers.**
1. Population code: core carries the post-2b86a2d population fixes (peak secondary inherits the power-law low-mass
   edge, `population/base.py:1146-1168`; closed-form truncated-Gaussian normalisers, `population/parametric.py:269`;
   a different pairing quadrature). Measured core − af896ca = −3.55 to −3.69 nats, constant to ±0.3 over 212 cells,
   posteriors unchanged (KS ≤ 0.044, 90 % width ratios 0.998–1.007; equivalence README). Near the Δμ_χ prior
   edge the offset grows to ≈ −4.5 (up to −11) (`STATE.md`, 2026-10-07); cause not attributed **[inference]**.
2. Guard cap 20 in A13 against 10⁶ in Analyses 0–12. Same formula; with N = 1000 the A13 threshold is ≥ 50,000
   effective injections and σ²_sel must stay below 20 − Σσ_i², so the 5N floor never binds.
3. Kernel truncation by value (10⁻¹⁰) instead of a 4096-galaxy window; both checked to drop nothing measurable.
4. Not triggered in A13, but traps: core's `fixed=True` sets γ = 2.5, not the mock's 0; core multiplies the χ_eff
   prior into p_pe whenever `chi_eff_in_p_pe` is False even for mock data and without legacy's −50 clip
   (`gw/samples.py:282-289` against legacy `gw/utils.py:221-226`); core has no "aggregate" mode and no built-in
   field LSS modulation.
5. Diagnostics: `return_diagnostics=True` exposes ln Ẑ_i, σ_i², ln μ̂, N_eff and the correction
   (`likelihood/hierarchical.py:40-48, 433-441`), but `BoundAnalysis` at e7c3007 never requests it
   (`runtime_binding.py:626-688`), so the A13 core runs record only ln L and the count of −∞ calls. The N_eff and
   σ²_sel values in the A13 STATE.md come from af896ca cells (`a13_edge_check.py`).

---

## 8. Why `log10n0` moves f_AGN when C = 1 (Question 5)

### 8.1 Result

**n0 enters only through the survey-global normaliser Z_k, and through it the relative weight of the two
mixture branches.** It does not enter the catalogued numerator and, on the complete rung, the missing-host
numerator is exactly zero wherever an event sample or injection lies. Z_k integrates the modelled hosts over
the whole internal grid 0 < z < 5, where C_k < 1 above z = 1.16 (GAL) and 1.69 (AGN), so 91–92 % of each
Z_k is modelled hosts that no data point can see. Because the two catalogues' normalisers do not cancel at
K = 2, n0 rescales the AGN:GAL odds.

Confidence: high. The identity below is exact algebra on the code as read, and the stored GPU grids satisfy
it to 5×10⁻¹² nats where it needs no interpolation.

### 8.2 Derivation

On the complete rung the data live where C_GAL = C_AGN = 1: every PE sample has z ≤ 0.65 and the detected
injections have z ≤ 1.37 even at H0 = 100 (z ≤ 1.0 at H0 = 67.74), while C_GAL < 1 only above z = 1.16 and
C_AGN above 1.69. So dN_miss,k = 0 at every sample (the exception, at H0 ≳ 90, is ≲ 10⁻⁵ of the injections
reaching z = 1.16–1.37 where C_GAL is just below 1; check 2 below bounds their effect at < 10⁻¹¹ nats) and, with
c_k(z,p) = N_obs,k(p) p_cat,k(z|p) (independent of n0) and F_k = N^tot_obs,k / Z_k,
$$
p_{\rm mix}=(1-f)\frac{c_{\rm G}}{Z_{\rm G}}+f\frac{c_{\rm A}}{Z_{\rm A}}
=S(f)\Big[(1-\tilde f)\frac{c_{\rm G}}{N_{\rm G}}+\tilde f\,\frac{c_{\rm A}}{N_{\rm A}}\Big],\quad
S=(1-f)F_{\rm G}+fF_{\rm A},\quad
\tilde f=\frac{fF_{\rm A}}{(1-f)F_{\rm G}+fF_{\rm A}} .
$$
Every event weight and every injection weight is multiplied by the same S, so ln Ẑ_i gains ln S, ln μ̂ gains
ln S, and N_eff and σ_i² are unchanged (they are scale-free). The N event terms and −N ln μ̂ cancel S:
$$
\ln\mathcal L(H_0,f;\,n_0)=\ln\mathcal L(H_0,\tilde f;\,n_0\to0),\qquad
{\rm logit}\,\tilde f={\rm logit}\,f+\ln\frac{F_{\rm A}}{F_{\rm G}} .
$$
The data constrain f̃, the AGN share of the **catalogued** host mass; the reported f_AGN is f̃ moved by
ln(F_G/F_A). At H0 = 67.74, F_GAL = 0.0778 and F_AGN = 0.0930, so ln(F_G/F_A) = −0.178 (−0.188 at H0 = 50,
−0.150 at H0 = 100: the weak H0 dependence is why H0 does not move). Mapped through logit: 0.273 → 0.239,
the measured +0.033 (FU_REPORT §2).

Why F_G ≠ F_A. Write the effective normalisation volume V_k = Z_k/n0_k = N_obs,k/n0_k + 4π∫₀⁵(1−C_k)dV_c.
With both catalogues drawn at their input densities out to z = 1, N_obs,k/n0_k is the same (0.919 and 0.921
V(z<1)), so the shift is ln(V_AGN/V_GAL). A catalogue that is complete to z = 1 and empty beyond, modelled with
C = 1 up to z_C > 1, loses the volume ∫_{1}^{z_max} C dV_c from its normaliser. The two "complete" fits carry
different nominal limits (m_lim = 24 for GAL, 25 for AGN), so the losses differ: V_GAL = 0.916 V(z<5),
V_AGN = 0.768 V(z<5). Flux-limited catalogues have C ≈ 0 above z = 1 and V_k = V(z<5) to 10⁻⁴ for both.

Analysis 4's slope. Mis-anchoring the AGN density by a factor x changes ln Z_AGN by ln[(F_A + x(1−F_A))]
≈ (1−F_A) ln x, so wherever the in-data missing term is negligible
$$\frac{\partial\,{\rm logit} f}{\partial\ln n_{0,{\rm AGN}}}=1-F_{\rm AGN},\qquad
\frac{\partial f}{\partial\log_{10}n_{0,{\rm AGN}}}=\ln 10\;(1-F_{\rm AGN})\,f(1-f)\approx0.43\text{–}0.46\ \ (f\approx0.25\text{–}0.30).$$
1 − F_AGN = 0.9885 (m<21), 0.9960 (m<20), 0.9987 (m<19), 0.9996 (m<18): the normaliser is dominated by the
modelled population out to z = 5 at **every** rung, which is why the slope is the same at 99.7 % and 9.5 %
completeness. At m<19 and m<18 the in-data missing hosts also change with n0; they add curvature and move H0
(the a4 H0 response) but leave the f slope near the normalisation value.

The four AGN=complete cells. The shift relative to a model whose branches share one normalisation volume is
ln(V_AGN/V_GAL): −0.176 for complete × complete (predicted Δf −0.036; measured excess over the zero-density
cell −0.033) and −0.264 for AGN complete × any flux-limited GAL catalogue (predicted Δf −0.053; measured
excess over the matched cells' flat −0.014: −0.042 oracle, −0.051 and −0.055 for the two Analysis 6 cells).
The probe proposed in the Analysis 4 report (AGN m_lim 25 → 26 on the oracle cell) is predicted to move
f_AGN by a further ≈ −0.07 (ln V ratio −0.62).

### 8.3 Checks (CPU, darksirens-0c5b3db imported from its worktree; scripts and outputs in
`/tmp/claude-88592/-hildafs-projects-phy230014p-magana-gws-agn/a23c33d8-3677-45fa-9c99-8730e7a85188/scratchpad/lik/`,
`check_Z.py`, `check_reparam.py`, `check_Veff.py` and their `.json`)

1. `field_global_log_Z` evaluated with the selection-mode SurveyParams equals the closed form of Section 4 to
   ≤ 9×10⁻¹⁶ relative, for all five rungs and both catalogues.
2. Stored grids (`analysis_3_…/results/joint_complete_s100.h5`, n0 = (−3, −5); `selection_redo/fu_probes/results/joint_complete_n0m24_s100.h5`, n0 = (−24, −24); both GPU, 201 × 41):
   at f = 0 and f = 1 the identity needs no interpolation (one branch; its Z cancels). The two grids agree to
   5.5×10⁻¹² (f = 0) and 7.3×10⁻¹² nats (f = 1) at every H0.
3. Interior: the n0 = (−3, −5) grid predicted from the zero-density grid by f → f̃ (cubic spline in f) matches the
   stored grid to rms 0.005 nats, max 0.033 nats within 15 nats of the peak (spline error on the 41-point
   f grid). Predicted f median 0.240348, stored 0.240348; H0 median 69.2184 both. This also confirms z_max = 5
   for the selection-mode runs: with z_max = 1.5, V_AGN would contain no missing hosts and the shift would be
   different in sign and size.
4. Analysis 4 arms predicted from the factor-1 grid by the normalisation map alone: slopes 0.434 and 0.438
   (stored 0.434, 0.438) at m<21 and m<20, every arm's f median within 2×10⁻⁴; at m<19 and m<18 0.441 and 0.451
   (stored 0.470, 0.455), with the residual and the H0 response coming from the in-data missing hosts.
Not done: no likelihood re-evaluation on CPU. The stored GPU grids at both densities already test the identity
exactly at the endpoints; a CPU evaluation (≈ 1.7×10¹⁰ kernel evaluations per call with the complete GAL
catalogue) would add nothing.

### 8.4 Intended model or defect?

- **The coupling is designed.** `redshift/prior.py:281-291` states that log10n0 enters through each
  catalogue's global normaliser at K ≥ 2, "where each catalog's Z_k enters its mixture branch
  non-cancelling"; this is what makes the mixture weight a host fraction. When the two branches share one
  normalisation volume (all flux-limited rungs: V_GAL = V_AGN = V(z<5)), the model is exactly
  R(z, Ω̂) ∝ (1−f) n_GAL(z,Ω̂)/n̄_GAL + f n_AGN(z,Ω̂)/n̄_AGN with n̄_k = n0_k: f_AGN is the fraction of mergers
  in AGN and needs the mean AGN-to-galaxy density ratio to be defined. The 0.43 per dex is that inherent
  degeneracy, not a bug.
- **Two parts are defects of the model as configured.**
  (i) The normalisation runs to the internal grid limit z_max = 5, chosen for numerics, not for the
  population. 90–99.96 % of Z_k is extrapolated hosts at 1 < z < 5, where the catalogue, the events and the
  mock's universe are empty, so even a complete catalogue's own count does not set its branch's weight; the
  assumed n0 does. For a complete catalogue the correct normaliser is its own count.
  (ii) "Complete" catalogues are represented by a finite nominal m_lim, which asserts C = 1 beyond the
  catalogue's redshift edge (to z = 1.16 and 1.69 for z ≤ 1.0 catalogues), removing different volumes from the
  two normalisers. This is the whole AGN=complete offset (−0.03 to −0.05).
  Fixes available in the code without new features **[not run]**: a survey `z_depth` at the catalogue edge
  (C := 0 beyond, `redshift/completion.py:1252-1305, 1803-1811`), which equalises V_k; or `DARKSIRENS_ZMAX` at the
  catalogue edge, which on complete catalogues makes Z_k = N_obs,k and removes n0 from the likelihood.
- **Correct statement for the paper.** "Each tracer's host density is normalised to unit total over the sky
  and 0 < z < z_max, counting catalogued hosts and the modelled uncatalogued hosts n0,k[1 − C_k(z)] dV_c.
  f_AGN is therefore the fraction of mergers whose hosts are AGN, relative to the assumed mean densities: the
  data constrain f_AGN N_AGN/Z_AGN against (1 − f_AGN) N_gal/Z_gal, and the inferred value moves with the
  assumed AGN density as ∂ logit f_AGN/∂ ln n0,AGN = 1 − N_AGN/Z_AGN ≈ 1 (0.43 per dex at f_AGN ≈ 0.3)."
  The complete-catalogue cells should either be rerun with the catalogue edge as depth or carry the
  representation offset explicitly.

---

## 9. Statements in the analysis reports that the code contradicts

1. FU_REPORT §2 and `fu_probes/results/zerodensity_verdict.json`: "max(1 − C_sel) = 0 on this rung" and
   "C ≡ 1 and the −24 density limit both mean no completion contribution". C = 1 only below z = 1.16 (GAL) and
   1.69 (AGN); on the normalisation domain 0 < z < 5, C_GAL(2) = 0.016 and C_AGN(3) = 0.011. The completion
   numerator is indeed inert on that rung; the completion normaliser is not. "Completion path live" is right
   in effect, wrong in route.
2. Analysis 4 REPORT, Interpretation: "a density error is absorbed by the mixture fraction because both are
   budget-like quantities" and "the steepening was the estimator". Under selection the slope is the normaliser
   map ln 10 (1 − F_AGN) f(1 − f); it is depth-independent because F_AGN ≈ 0 at every rung (the normaliser is
   dominated by z > 1), not because the completion is well estimated.
3. Analysis 4 REPORT, oracle probe: the alternative "the deep-m_lim representation of complete carries a
   constant f_AGN offset" is the explanation; it is not constant but ln(V_AGN/V_GAL)·f(1−f), −0.036 for
   complete × complete and −0.053 for AGN complete × flux-limited GAL.
4. darksirens docstrings describe Analyses 3–7's estimand as a "host fraction"; none says that the fraction is
   over 0 < z < 5 (`redshift/grid.py`), which the analyses' READMEs and reports also never state.
5. `selection_redo/scripts/scan_h0f.py:153-157` (`--gamma` help) says the mock's host draw corresponds to
   γ = 1, while `events.h5` records `gamma = 0.0` and the runs use γ = 0. Not investigated; flagged only.
