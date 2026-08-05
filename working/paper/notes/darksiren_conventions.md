# Dark-siren / GW-cosmology terminology and figure conventions

Evidence gathered 2026-08-02 from ar5iv HTML full texts and the actual figure PNGs of:

- **Soares-Santos et al. 2019** (arXiv:1901.01540, ApJL) — DES + GW170814 dark siren. NOTE: the task brief called this "Fishbach+ 2019 GW170817"; the ID actually resolves to the DES GW170814 paper (Soares-Santos, Palmese et al.).
- **Fishbach et al. 2019** (arXiv:1807.05667, ApJL) — the actual GW170817 statistical standard siren paper; fetched in addition since it was clearly intended.
- **Gray et al. 2020** (arXiv:1908.06050, PRD) — gwcosmo mock data analysis.
- **Palmese et al. 2023** (arXiv:2111.06445, ApJ) — 8 dark sirens + DESI Legacy Survey.
- **LVK GWTC-3 cosmology** (arXiv:2111.03604, ApJ).
- **Chen, Fishbach & Holz 2018** (arXiv:1712.06531, Nature).

All quotes verified against the source text or read off the downloaded figure images.

---

## (a) H0-posterior figure axes: labels, units, scaling

The field convention is to plot a **true probability density with explicit inverse-H0 units**, not a peak-scaled curve. Read directly off the published figures:

| Paper | y-axis label (verbatim from figure) | x-axis label |
|---|---|---|
| Soares-Santos+ 2019 Fig. 2 | `p (km^-1 s Mpc)` | `H0 (km s^-1 Mpc^-1)` |
| Fishbach+ 2019 Fig. 5/6 | `p(H0) (km^-1 s Mpc)` | `H0 (km s^-1 Mpc^-1)` |
| Gray+ 2020 Fig. (MDA results) | `p(H0 | {x_GW}, {D_GW}) (km^-1 s Mpc)` | `H0 (km s^-1 Mpc^-1)` |
| Palmese+ 2023 per-event fig. | `p (km^-1 s Mpc)` | `H0 (km s^-1 Mpc^-1)` |
| GWTC-3 cosmology Fig. 6 | `p(H0|x) [km^-1 s Mpc]` | `H0 [km s^-1 Mpc^-1]` (square brackets, LVK house style) |

Key points:

- Units on the density axis are always written out: **km^-1 s Mpc** (the reciprocal of the H0 unit). Parentheses in ApJ-family papers, square brackets in LVK papers.
- Densities are normalized so curves of different width have different peak heights (e.g. Gray+ 2020 shows "Known host galaxy" peaking at ~0.5 vs "Complete galaxy catalog" at ~0.31 on the same axes). **Peak-scaling to 1 is not the norm.**
- When curves ARE arbitrarily rescaled for a multi-curve display, the caption must say so explicitly. Verbatim declarations:
  - Palmese+ 2023 (Fig. 5 caption): "Posteriors are arbitrarily rescaled only for visualization purposes."
  - Gray+ 2020 (Fig. 2 caption): "Individual likelihoods (normalized and then scaled by an arbitrary value), for each of the 249 events, are shown as thin lines"
  - Soares-Santos+ 2019 (comparison of localization-volume choices): "The PDF computed from the larger volume has been renormalized to have the same value of the 90% localization volume H0 posterior at the maximum, to highlight differences below and beyond the..."
- Truth line in mock-recovery figures: a **dashed black vertical line** at the input H0, declared in the caption (Fishbach+ 2019 Fig. 1: "The dashed black line shows the injected value, H0 = 70 km s^-1 Mpc^-1.").

**Recommendation for our draft:** label the density axis `p(H0 | d) (km^-1 s Mpc)` (or `p(H0)` if unconditioned) and plot true densities; if any panel rescales curves, declare it in the caption with the Palmese wording ("arbitrarily rescaled for visualization purposes").

## (b) Naming simulated truths and simulation inputs

Accepted vocabulary (all attested, roughly in order of frequency):

- **"simulated value"** — Gray+ 2020's standard term for the truth: "converges to the simulated value of H0 = 70 km s^-1 Mpc^-1"; "the width of the 68.3% highest density probability interval divided by the simulated value of H0".
- **"injected value" / "injected"** — Fishbach+ 2019: "The dashed black line shows the injected value, H0 = 70 km s^-1 Mpc^-1"; "matches the injected distance and assumed H0 value"; "matches the injected redshift distribution". Gray+ 2020 uses "injected" (sometimes in quotation marks) for signals added to detector noise: "searching for 'injected' signals in real detector data".
- **"true value"** — used for the abstract notion of truth: Fishbach+ 2019 "approaching the true value of H0"; GWTC-3 "assumed to be lower (or higher) than its true value".
- **"input"** — Soares-Santos+ 2019: "our pipeline was able to reliably reproduce the input cosmology on simulation tests."
- **"simulated" as adjective** for synthetic data: "249 simulated BNS detections" (Fishbach+ 2019), "simulated data set consisting of 30,000 BNS mergers" (Chen+ 2018), "We simulate 70,000 BBH mergers" (Palmese+ 2023).

**"planted" appears in NONE of the six papers** (0 hits in all full texts). Do not use it — replace any "planted" in our draft with "injected" (per-event signals) or "simulated/input" (population- or cosmology-level truths).

## (c) Injection / selection machinery

- **"injections"** = simulated signals processed through search/PE. Palmese+ 2023: "After the 70,000 injections are made, we run a matched-filter search to retrieve the detected events."; "We assume IMRPhenomD waveforms both for the injections and reconstructions." Gray+ 2020: "an end-to-end simulation of approximately 50,000 'injected' events in detector noise".
- **"selection effects"** is the universal umbrella term. Soares-Santos+ 2019: "We now consider the selection effects of GW events and galaxies introduced by the experiments' sensitivities and detection pipelines." GWTC-3: "correctly normalizes the posterior and takes into account selection effects (Mandel et al., 2019)"; "We evaluate GW selection effects using LIGO and Virgo sensitivities during the O1, O2, and O3 runs."
- **p_det** — GWTC-3: "p_det(theta, Phi) is the probability of detecting a GW event with intrinsic parameters theta"; also "detection probability" spelled out. Fishbach+ 2019 instead uses **beta(H0)**: "a normalization term to ensure that the likelihood normalizes to 1 when integrated over all detectable GW and EM datasets"; Palmese+ 2023 likewise uses beta(H0) and a "selection function".
- **"injection sets" / "found injections"**: these phrases do NOT appear in any of these cosmology papers (the GWTC-3 cosmology paper has literally zero occurrences of "injection"). They belong to the LVK *population* papers (GWTC-3 pops, arXiv:2111.03634) and the o3 sensitivity-injection data releases. Safe to use if we literally reweight the LVK injection release, but in dark-siren-style prose the native phrasings are "injections", "detected events", "selection effects", and "p_det".
- Detection criteria are stated concretely: "A detection is made when at least 2 detectors reach a single-detector SNR of..." (Palmese+ 2023); "Binary mergers are detected only if their measured network SNR is greater than 12" (Chen+ 2018).

## (d) Other conventions where a draft might deviate

**"dark siren" vs "dark standard siren".**
- Both are used, interchangeably, by Palmese-lineage and LVK papers. Soares-Santos+ 2019 title: "First measurement of the Hubble constant from a dark standard siren..."; body uses "dark standard sirens". Palmese+ 2023 mixes freely: "eight dark siren events", "the sample of dark standard sirens analyzed". GWTC-3: "By using all of the dark sirens...", and in captions "posterior obtained using all dark standard sirens without any galaxy catalog information".
- Convention: use **"dark standard siren"** at first/formal mention (title, abstract, first use), then **"dark sirens"** in running text. The counterpart case is a **"bright siren"** / "bright standard siren" (GWTC-3: "the bright standard siren GW170817").
- Fishbach+ 2019 and Chen+ 2018 predate the term: they say **"statistical standard siren"** / "the statistical method" / "the statistical approach". Don't mix "statistical" into new text unless citing that era.

**Mock catalogs and realizations.**
- **"mock galaxy catalog"** is the standard noun: "the MICE mock galaxy catalog" (Fishbach+ 2019), "we construct a mock catalog" (Chen+ 2018), "mock universe" (Gray+ 2020). Gray+ 2020's whole framing is a "Mock Data Analysis (MDA)" / "mock data challenge".
- **"realizations"** (US spelling, with z) for repeated draws: "20 realizations of the GW170817 3-dimensional sky map" (Fishbach+ 2019); "The scatter between realizations of the group is indicated by the error bars" (Gray+ 2020). Our draft's British "realisations" should be flipped to "realizations" for ApJ/PRD.

**Credible-interval phrasing.**
- The canonical formula is **"maximum a posteriori and X% highest density (posterior) interval"**:
  - Fishbach+ 2019: "H0 = 76 (+48/-23) km s^-1 Mpc^-1 (maximum a posteriori and 68.3% highest density posterior interval; assuming a flat H0 prior in the range [10, 220] km s^-1 Mpc^-1)".
  - GWTC-3: "credible intervals are reported as maximum posterior and 68.3% highest density intervals"; "We quote the maximum a posteriori probability (MAP) and the corresponding highest density interval (HDI) values".
  - Palmese+ 2023: "(68% Highest Density Interval)" with the prior stated: "for a prior in H0 uniform between [20, 140] km s^-1 Mpc^-1".
  - Gray+ 2020: "68.3% highest density error region", "68.3% highest density posterior intervals", "maximum a-posteriori value".
- Note 68% vs 68.3%: LVK-methodology papers write **68.3%**; Palmese papers write **68%**. Chen+ 2018 uses "68% symmetric credible interval" (symmetric, not HDI — a deliberate, stated choice). Soares-Santos+ 2019 says "maximum a posteriori and its 68% confidence level" — but "confidence level" for a posterior is the outlier; prefer "credible interval"/"highest density interval".
- Always state the H0 prior (range and flat vs flat-in-log) next to the quoted interval; GWTC-3: "We use a flat-in-log prior on H0 only when quoting results combined with GW170817 and its EM counterpart."
- Median-based quotes are declared as such: "540 (+130/-210) Mpc (median value with 90% credible interval)" (Soares-Santos+ 2019).

**Miscellaneous.**
- Unit typography: `km s^-1 Mpc^-1` with thin spaces, never "km/s/Mpc" in formal text.
- Fractional-uncertainty convention (forecast papers): "fractional H0 measurement uncertainties defined as half the width of the symmetric 68% credible interval divided by the median" (Chen+ 2018); Gray+ 2020 divides interval width "by the simulated value of H0".
- Blinding is mentioned as good practice: "Our analysis was blinded to avoid confirmation bias." (Soares-Santos+ 2019).
- Convergence language for mocks: "converges to the simulated value", "recover an unbiased estimate of H0" (Gray+ 2020) — "recover"/"recovery" is the accepted verb for mock closure tests.

## Quick fix-list for our draft

1. "planted" -> "injected" (event-level) or "simulated"/"input" (population/cosmology truth). Truth line: "the dashed line shows the injected (or simulated) value, H0 = ... km s^-1 Mpc^-1".
2. y-axis of H0 posteriors: `p(H0 | d) (km^-1 s Mpc)`; plot true densities; declare any rescaling with "arbitrarily rescaled for visualization purposes".
3. Quote results as "maximum a posteriori and 68.3% highest density interval", with the prior stated inline.
4. "dark standard siren" at first mention, "dark sirens" thereafter; "bright siren" for the counterpart case.
5. "mock galaxy catalog", "mock universe", "realizations" (z-spelling); mock closure = "recover the simulated value of H0 without bias".
6. Reserve "injection set"/"found injections" for text that literally uses the LVK sensitivity-injection products; otherwise say "injections", "detected events", "selection effects", "p_det" (or "beta(H0)" if using the Chen/Fishbach normalization).
