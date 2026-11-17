# Novelty and literature check (16 Nov 2026)

Scope: claims C1 to C9 of `notes/DESIGN.md` section 2. Literature searched through
arXiv listings of 13 Nov 2026. Every paper named here with an arXiv number was opened on
arXiv (identifier, title, authors). The papers the verdicts rest on were read at the
abstract and, where the abstract did not settle the point, in full text; the standard
method references were confirmed by record only. BibTeX for all of them is in
`../refs_candidates.bib` (170 entries, INSPIRE keys), and the `note` of each entry says
which of the three levels applies. Items that could not be opened are listed in section
7 as UNVERIFIED and have no bib entry.

Credible levels differ between papers. Each number quoted below carries the level its
source uses; the paper's own convention is 90%.

Names follow the year in the bib entry, with one exception kept from the literature:
"Veronesi et al. 2025" is the Quaia paper (`Veronesi:2024tef`, MNRAS 536, 375, which
INSPIRE dates 2024 and Pouw et al. cite as 2025) and "Veronesi et al. 2024" is the
flare paper (`Veronesi:2024pjq`, MNRAS 536, 3112). With the exported years both will
print as 2024a and 2024b.

## 1. Summary in five sentences

1. No paper found does what this paper does as a whole: H0 free, mergers assigned to a
   galaxy catalog or to an AGN catalog that trace one density field, flux-limited
   catalogs with the abundance of missing hosts varied or free, and mass and spin
   distributions that depend on the host class.
2. The first step alone, C1, is not new. A joint posterior on H0 and the fraction of
   BBHs hosted by AGN from an AGN catalog is already in Physical Review D (Bom &
   Palmese 2024, arXiv:2307.01330, Sec. III.2), and a two-component host mixture
   inferred jointly with H0 was posted on 19 Oct 2026 (Cross-Parkin et al.,
   arXiv:2610.25866, "Hostbusters"; the project owner is an author).
3. A hierarchical likelihood for f_AGN with GW selection effects and catalog
   incompleteness was posted on 15 Sep 2026 and applied to GWTC-5.0 (Pouw et al.,
   arXiv:2609.17665), at fixed cosmology, with the non-AGN mergers uniform in comoving
   volume and with one mass and spin distribution for both classes.
4. Three groups name the host-dependent population analysis (C8, C9) as their next
   step: Pouw et al. 2026, Cross-Parkin et al. 2026 and Bom & Palmese 2024. C8 and C9
   are new today and are the part of the paper most exposed to being overtaken.
5. The part with no precedent found at all is C6 (host abundances inferred from the
   mergers) and the numbers of C5 (how far f_AGN and H0 move per dex of assumed AGN
   abundance).

## 2. Verdict table

| Claim | Closest prior work | What that work did | What is new here | Verdict | Confidence |
|---|---|---|---|---|---|
| C1 two host populations, one likelihood; H0 and f_AGN recovered and nearly uncorrelated | Bom & Palmese 2024 (2307.01330); Cross-Parkin et al. 2026 (2610.25866); Pouw et al. 2026 (2609.17665) | B&P: joint H0 and lambda_AGN from a static AGN catalog, 80 events, n_AGN = 10^-4.75 Mpc^-3, H0 to 5-11%; the non-AGN fraction is an unclustered background that carries no redshift information; GW selection neglected. Hostbusters: mixture of two weightings (r-band luminosity, g-r color) of one complete galaxy catalog, fraction xi with H0, "only a weak correlation". Pouw: f_AGN with selection and incompleteness, cosmology fixed. | Non-AGN mergers sit in a galaxy catalog; both catalogs trace one density field; both classes inform H0; one normalization per catalog; selection included; H0-f_AGN correlation measured (0.08). | partly anticipated | high |
| C2 galaxy-only analysis of a mixed population is biased; AGN catalog alone fails | Hanselman et al. 2025 (2405.14818); Perna et al. 2025 (2405.07904); Alfradique et al. 2025 (2503.18887) | Wrong host weighting biases dark-siren H0 through a wrong redshift distribution and through weight on wrong hosts (Hanselman); Alfradique: unbiased H0 needs the correct weighting on a complete or volume-limited catalog. The size is disputed: Borghi et al. 2026 (2509.18243) find results robust to mis-weighting, and Kalomenopoulos et al. 2025 (2511.12334) that physical weights add little beyond clustering. | The AGN channel as the mis-weighted population (hosts 100 times rarer, more clustered); sign and size of the shift; failure of the AGN-only analysis; Bayes factor for two host classes. | partly anticipated | high |
| C3 AGN-hosted mergers carry most of the H0 information (factor 4.5 at equal numbers) | Alves et al. 2024 (2009.13739); Bom & Palmese 2024; Zhu et al. 2024 (2403.04950); VanWyngarden et al. 2026 (2511.04786); Beirnaert et al. 2025 (2505.14077) | Rarity of AGN identifies hosts and sharpens H0 (Alves; B&P: lower n_AGN gives tighter H0); for EMRIs an AGN catalog improves H0 from 3.7% to 3.2% (Zhu); the brightest 1% of galaxies suffice because faint galaxies cluster around them (VanWyngarden); cluster catalogs improve GWTC-3 precision by 10% and 38% (Beirnaert). | A like-for-like measurement: same number of events, same density field, two tracers differing in abundance (x100) and bias (2.0 against 1.2), 93 catalogs. | partly anticipated | high |
| C4 flux limits cost little when the abundance of missing hosts is known | Dalang & Baker 2024 (2310.08991); Kalomenopoulos et al. 2025 (2511.12334); Barbieri et al. 2026 (2608.30599); Borghi et al. 2026 (2509.18243) | For one galaxy catalog and H0: completion that follows the large-scale structure recovers most of the information (2 to 4 times more precise than homogeneous completion at 1-10% completeness, Barbieri; 25% completeness competitive, Kalomenopoulos); in Borghi et al. the precision degrades as a sigmoid in completeness, with 2% still reached at 50% completeness. | The same statement for f_AGN and for two flux-limited tracers together; the 13% and 7% numbers at 9.5% completeness. | partly anticipated (H0), new (f_AGN) | high / medium |
| C5 f_AGN is only as well known as the AGN abundance below the flux limit | Zhu et al. 2024 (2403.04950); Zhu & Chen 2025 (2505.02924); Bom & Palmese 2024 | The association likelihood contains the product f_agn x completeness (Zhu & Chen 2025, Eq. 1), and Veronesi et al. 2023 note that their constraining power "scales linearly with the completeness"; Zhu et al. 2024 state "the degeneracy between f_agn and f_compl" and the sensitivity to the AGN density; B&P state that AGN selection biases lambda low but not the cosmology, and that a known AGN density injects cosmological information. All qualitative. | The size: 0.43 to 0.47 in f_AGN per dex of assumed n_AGN; H0 insensitive at completeness of 81% or more and 3.2 km/s/Mpc per dex at 9.5%. | partly anticipated | medium-high |
| C6 the mergers measure the galaxy abundance and not the AGN abundance | Bom & Palmese 2024 (Sec. IV.1) | Say that "one would leave the AGN number density free to vary"; tried it only for lambda_AGN = 0 and only to confirm that the spurious cosmological information disappears. | Both abundances free with H0 and f_AGN; galaxy abundance recovered; AGN abundance degenerate with f_AGN (+0.90); H0 unchanged. | new | medium |
| C7 a host-correlated mass difference biases a common-population analysis in H0 | Mastrogiovanni et al. 2021 (2103.14663); Pierra et al. 2024 (2312.11627); Scarpa et al. 2026 (2603.15332); Borghi et al. 2026 | Mass-spectrum misspecification biases spectral-siren H0 (up to 3 sigma with 2000 events, Pierra); H0 correlates with the mixture fraction of two formation channels because one is heavier (Scarpa, no catalogs); the H0-mass-scale correlation grows as catalogs become incomplete (Borghi). B&P 2024 flag the AGN case as future work. | The subpopulation is tied to a host class, so the bias depends on the catalog used (+3.0 with both catalogs, at least +20 with galaxies alone): catalog information does not protect H0. | partly anticipated | high |
| C8 host-dependent mass peak and spin mean recover H0, f_AGN and the offsets | Pouw et al. 2026 (2609.17665); Cross-Parkin et al. 2026; Gayathri et al. 2023 (2301.04187); Bartos & Haiman 2026 (2605.09351) | Pouw write the formalism and apply it with one fixed intrinsic distribution for both classes, deferring "the inference of intrinsic population hyperparameters to future work"; Hostbusters name it as future work; Gayathri and Bartos & Haiman separate an AGN component by masses or spins only, no positions. | The joint inference carried out, with H0 free, including the eight-parameter model. | new | medium-high |
| C9 position and intrinsic parameters identify the host class nearly independently | Pouw et al. 2026 (Sec. 6); Moncrieff & Panther 2025 (2508.18704) | Pouw find that their position-based ranking of events by AGN probability disagrees with the spin-based ranking of Bartos & Haiman 2026 and conclude that joint inference is necessary; Moncrieff & Panther take the hierarchical mergers as already identified by their masses and spins and then test AGN against non-AGN clustering. Neither measures how the two kinds of information combine. | f_AGN interval unchanged (<1%) by the spin difference; spin offset 2.4 times better with catalogs than from spins alone. | new | medium |

## 3. Capability by paper

The gap is clearest as a table. "yes" means done in that paper, not merely allowed by
its formalism.

| Paper | H0 free | AGN catalog | Non-AGN mergers in a galaxy catalog | GW selection effects | Incomplete catalogs | Host abundance varied or free | Host-dependent mass or spin |
|---|---|---|---|---|---|---|---|
| Bartos et al. 2017; Veronesi et al. 2023, 2025; Zhu & Chen 2025 | no | yes | no (uniform background) | no | completeness factor | no | no |
| Pouw et al. 2026 | no (Planck 2015 fixed) | yes (Quaia) | no (uniform in comoving volume, star-formation-like rate) | yes | yes (luminosity-function completeness, undetected-host term) | no | no (formalism only) |
| Palmese et al. 2021; Bom & Palmese 2024 | yes | yes (flares; static catalog in Sec. III.2) | no (background, no information) | no (stated) | no | density fixed; free only in a lambda = 0 test | no (a high-mass-only AGN scenario is simulated) |
| Cross-Parkin et al. 2026 | yes | no | two weightings of one galaxy catalog | yes | no | no | no (future work) |
| Gray et al. 2026 (2609.04991); Agapito & Mancarella 2026 (2609.38110) | yes | no | one galaxy catalog, weight proportional to L^epsilon | yes (Gray; not checked for Agapito & Mancarella) | yes | no | no |
| LVK GWTC-5.0 cosmology (2605.27227) | yes | no | one class (GLADE+ K band, DES-Y6), uniform or luminosity weights | yes | yes | no | no (one mass-dependent spin model for all) |
| Scarpa et al. 2026 | yes | no catalog | no catalog | yes | n/a | n/a | channel-dependent mass-redshift distributions, fraction free |
| Moncrieff & Panther 2025 | no | yes (Quaia) | yes (GLADE+ without quasars), angular cross-correlation | no | no | no | hierarchical mergers assumed pre-identified |
| Zhu et al. 2024 (EMRIs, space detectors) | two-step: f_agn first, then cosmology | yes | yes, separately | n/a | completeness factors | no | eccentricity differs by channel |
| This paper | yes | yes | yes, same density field | yes | yes | yes | yes |

## 4. Positioning, claim by claim

**C1.** Present the mixture likelihood as the combination of two existing analyses, not
as an invention: the AGN-catalog standard-siren formalism of Palmese et al. 2021
(2103.16069) and Bom & Palmese 2024, which already infers H0 with the AGN-hosted
fraction, and the galaxy-catalog dark-siren likelihood (Gray et al. 2020, 2023;
Mastrogiovanni et al. 2023; Gair et al. 2023). The distinguishing sentence is that in
Bom & Palmese the method has "the disadvantage of not providing constraining power for
events that do not occur in AGNs" and the AGN are unclustered (their Sec. IV.5 asks for
"AGN clustering in the simulations" as future work), while in the association searches
the non-AGN mergers are a background that does not cluster; here they follow a galaxy
catalog drawn from the same density field as the AGN.
Cite Hostbusters for the two-component host mixture and for the weak H0-fraction
correlation, which the 0.08 of C1 confirms for two physically distinct catalogs.
Do not write: "the first joint inference of H0 and the AGN fraction", "the first host
mixture inferred with H0", "a new hierarchical likelihood for f_AGN", or that searches
for the AGN channel have never included selection effects (Pouw et al. 2026 do).

**C2.** This is the AGN-channel case of the host-weighting systematic of Hanselman et al.
2025, Perna et al. 2025 and Alfradique et al. 2025, and the paper should say so in one
sentence. The literature disagrees on how large such biases are once catalogs are
clustered (Hanselman: significant; Borghi et al. 2026 and Kalomenopoulos et al. 2025:
small), so a shift of 2.2 km/s/Mpc against an interval of +1.7 -1.6 on one simulated
catalog will be read as inconclusive; state the shift relative to the two-catalog
posterior of the same catalog, as the combined posterior over catalogs, and with the
Bayes factor. Li, Gray & Heng 2026 (2508.15574) already use Bayes factors to choose between
host-weighting models with about 1000 detections; cite it next to the Savage-Dickey
ratio. Do not write: "we show that the wrong host population biases H0" as a new result.

**C3.** The direction is established and often stated: Bartos et al. 2017 (rare hosts can
be associated statistically, the best-localized 5% of events drive the result), Alves et
al. 2024, Bom & Palmese 2024, Zhu et al. 2024, Beirnaert et al. 2025, and Pouw et al.
2026 ("a denser catalogue more closely resembles the average redshift distribution").
What the paper adds is a controlled number and its physical reason. Give the number of
hosts per localization volume for each class so that the factor 4.5 can be compared
with the modest gain Zhu et al. find for EMRIs (3.7% to 3.2%), where volumes are small.
Do not write: "AGN are better dark-siren hosts" as a discovery, or any sentence implying
that real AGN catalogs would deliver this gain; it holds for AGN-hosted mergers only.

**C4.** For H0 this agrees with a dense 2024 to 2026 literature (Dalang & Baker 2024;
Leyde et al. 2024, 2026; VanWyngarden et al. 2026; Kalomenopoulos et al. 2025; Barbieri
et al. 2026; Tagliazucchi et al. 2026; Ghosh & More 2026) and should be presented as
agreement. The small cost depends on the missing-host term being modulated by the
density field; say plainly which field enters (the true one, or one estimated from the
cataloged objects), because Dalang & Baker 2024 state that multiplicative completion
overestimates the amplitude of structure at low completeness and homogeneous completion
underestimates it. The new content is that f_AGN
shares the robustness when the abundances are known. Do not write: "incompleteness does
not matter" (Bera et al. 2020 own that phrase for cross-correlations) or claim the
first treatment of clustered missing hosts.

**C5.** The degeneracy is built into the likelihood every association search uses, where
f_agn multiplies the completeness, and Zhu et al. 2024 name it explicitly together with
the sensitivity to the AGN density; Cross-Parkin et al. 2026 list incompleteness as
"potentially degenerate" with their mixing fraction. Frame C5 as measuring the size of a
known degeneracy and its consequence for H0, which nobody has quantified. One point
needs an explanation in the text. In the association likelihood the abundance enters in
two places: the completeness factor, which goes to one for a complete catalog, and the
background density of AGN (Zhu et al. 2024 name both). C5 reports the same slope "at
every depth", including 99.7% completeness, so the reader must be told through which
term the assumed abundance acts when almost nothing is missing. Do not write:
"we identify a degeneracy between f_AGN and the AGN abundance".

**C6.** Nothing found infers the abundance of hosts below the flux limit from the mergers.
Position it as the answer to the suggestion in Bom & Palmese 2024 (Sec. IV.1) and as the
reason the conclusion hands the problem to AGN surveys. Related but different: the
merger rate per galaxy of Mastrogiovanni et al. 2023, the host-weighting exponent of
Gray et al. 2026 and Agapito & Mancarella 2026, and the luminosity-function dependence
studied by Turski et al. 2026 (2505.13568). Do not write that the galaxy abundance is
"self-calibrated" without the interval (-3.12 +- 0.29 is a factor of two each way).

**C7.** The mechanism and the sign belong to the spectral-siren literature (Farr et al.
2019; Ezquiaga & Holz 2022; Mastrogiovanni et al. 2021; Mukherjee 2022; Pierra et al.
2024; Agarwal et al. 2025; Gennari et al. 2026; Agapito et al. 2026), and Scarpa et al.
2026 already report that H0 correlates with a formation-channel fraction because one
channel is heavier. The new statement is that when the heavier subpopulation is also a
different host class, the galaxy catalog does not protect H0 and makes it worse when
used alone. State whether the common-population model that returns 70.6 has free or
fixed mass parameters: Pierra et al. attribute their bias to models that cannot adjust,
and the LVK GWTC-5.0 mass model has two free peaks. Do not write: "a new systematic",
or that dark-siren analyses ignore the population (Gray et al. 2023 and Mastrogiovanni
et al. 2023 exist to prevent exactly that).

**C8.** This is the result to lead with. It carries out what Pouw et al. 2026 ("defer the
inference of intrinsic population hyperparameters to future work"), Cross-Parkin et al.
2026 ("extending the likelihood to model host weights and compact-binary population
parameters simultaneously") and Bom & Palmese 2024 ("AGN hosts in concert with spectral
siren analyses") leave open; quote them. Related ideas to cite: Tong et al. 2025
(2502.10780), where spins pick out the mass scale used for H0; Gayathri et al. 2021,
2023, Bartos & Haiman 2026 and Li et al. 2026 (2509.23897) for AGN components found
from masses and spins alone. Do not use the verb "unify" for position and population
information (it is the title of Pouw et al.); do not claim the formalism, claim the
demonstration and the recovered numbers.

**C9.** New as a measurement. Pouw et al. supply the motivation: their position-based and
the spin-based identification of AGN candidates disagree. Qualify "independently":
Veronesi et al. 2024 (2405.05318) show that heavier events have larger localization
volumes, so mass and position information are coupled through the detection; the clean
statement of C9 is for the spin offset, which is not redshifted. Do not write that
position and intrinsic parameters are independent probes without that qualification.

## 5. What the introduction must say about prior work

Paragraph on dark sirens, with numbers. GWTC-4.0 (2509.04348): 142 of 218 sources,
GLADE+, H0 = 75.4 +12.8 -9.1 (68.3%) and 75.4 +24.5 -13.3 (90%), with GW170817.
GWTC-5.0 (2605.27227, arXiv v2): 236 sources, DES-Y6 r band, H0 = 71.7 +9.4 -7.5
(68.3%) and +17.7 -11.4 (90%) with GW170817 (their Table 3, fiducial mass model); dark
sirens without GW170817 68.8 +14.2 -13.2 (68.3%) and +24.7 -21.8 (90%); mass spectrum
alone 70.1 +15.1 -13.3 (68%). Both analyses assume one host class, galaxies, weighted
uniformly or by luminosity in GWTC-5.0; GLADE+ itself includes the SDSS DR16Q quasars
among its galaxies (stated in the GWTC-5.0 paper). In 2026 three papers began to infer the host model with H0
(Gray et al.; Agapito & Mancarella; Cross-Parkin et al.), all with galaxies only.

Paragraph on the AGN channel, with numbers. Bartos et al. 2017: f_agn of 50-100% testable
with 70-300 events, 25% with 1000. Veronesi et al. 2023: at most 0.49 (0.17) of
mergers from AGN above 10^45.5 (10^46) erg/s (95%). Veronesi et al. 2025, Quaia, 159
events: at most 0.21 (0.11) from unobscured AGN above 10^44.5 (10^45) erg/s (95%).
Zhu & Chen 2025, SDSS DR16: f_agn = 0.39 +0.41 -0.32 for AGN of 10^44.5 to 10^45 erg/s
(90%). Pouw et al. 2026, GWTC-5.0, 256 BBHs, Quaia: f_agn below 0.52, 0.41, 0.61 for
L_bol above 10^44.5, 10^45.5, 10^46.5 erg/s (95%); the earlier limits 0.21, 0.13, 0.11
loosen once AGN redshift errors and detection efficiency are included. From masses and
spins alone: about 25% of GWTC-2 events prefer an AGN model (Gayathri et al. 2021),
about 10% in a high-spin component (Bartos & Haiman 2026, 90% interval 1% to 14%).
Flares: fewer than 3% of BBHs produce observable AGN flares, which still allows about
40% to merge in disks (Cabrera et al. 2026, 2510.20767). The spatial searches fix the
cosmology (checked in Veronesi et al. 2025, Zhu & Chen 2025 and Pouw et al. 2026) and
compare the AGN hypothesis with a background that does not cluster (uniform in comoving
volume in Pouw et al.); the population studies use no positions.

The gap, as missing capabilities:

1. No analysis leaves H0 free and assigns the mergers outside AGN to a galaxy catalog.
   Bom & Palmese 2024 leave H0 free but the 1 - lambda_AGN fraction is background; the
   association searches fix H0. Since galaxies and AGN trace the same structure, a
   merger in an ordinary galaxy is correlated with the AGN catalog and a merger in an
   AGN with the galaxy catalog; neither analysis is self-contained if the population
   is mixed.
2. No analysis has put a number on how the unknown abundance of AGN below the flux
   limit propagates to f_AGN and to H0, or asked whether the mergers can measure it.
   Completeness enters existing work as a fixed function from an assumed luminosity
   function (Kulkarni et al. 2019 in Pouw et al.; a Schechter function in the LVK
   analyses).
3. No analysis lets the mass or spin distribution depend on the host class while
   using the host catalogs. Population studies separate an AGN component without
   positions; association studies use positions with one population.

One sentence the introduction can then make: this paper supplies the three together on
simulated catalogs, and finds which of them limits the measurement.

## 6. Risks

Things that could lead a referee to say the central idea exists already, in order.

1. **Bom & Palmese 2024, Sec. III.2 and IV.5** (PRD 110, 083005). Joint H0 and
   lambda_AGN from an AGN catalog, called there "a type of informed dark standard siren
   method". If the paper's abstract reads as "we infer H0 and f_AGN together", this is
   the citation a referee will produce. The defense is the galaxy catalog for the
   non-AGN mergers, the shared density field, selection effects, and C5 to C9.
2. **Cross-Parkin et al. 2026** (2610.25866). Same owner, four weeks old, same structure
   for C1 (two-component host mixture with H0, weak correlation, forecast with 1000
   events). The new paper must cite it in the introduction and state the difference in
   one sentence (two catalogs with different abundance and clustering, flux limits,
   host-dependent populations); otherwise C1 reads as a repeat with relabeled hosts. Its
   discussion already lists incompleteness and host-dependent mass and spin as the open
   problems, which this paper can claim to solve.
3. **Pouw et al. 2026** (2609.17665). The title claims the union of spatial correlation
   and population inference. In fact both classes share one fixed intrinsic
   distribution and the cosmology is fixed, and the paper says so. A referee who has
   read only the abstract may think C8 and C9 are done. Quote their deferral. The same
   group is the most likely to post the host-dependent analysis next.
4. **Gray et al. 2026 and Agapito & Mancarella 2026.** Both claim a first for inferring
   the host model with cosmology. Avoid any "first" on that axis.
5. **Clustering-aware completion** (section 4, C4). A referee from that literature will
   test C4 against it and ask what density field modulates the missing hosts.
6. **Zhu et al. 2024** (2403.04950, ApJS 273, 24). EMRIs with a galaxy catalog and an AGN
   catalog, a statistic for f_agn, and the statement that knowing f_agn improves the
   cosmology. Two-step and for space detectors, but it is the earliest paper that puts
   galaxies, AGN, a host fraction and cosmology in one study. Cite it.
7. **Input values against published limits.** The simulated AGN are luminous (log L_X above
   43.7, 1e-5 per Mpc^3) and host 30% of the mergers. Veronesi et al. 2025 limit
   luminous unobscured AGN to 0.11-0.21 (95%); Pouw et al. 2026 loosen that to 0.41-0.52,
   and Zhu & Chen 2025 place the signal in lower-luminosity AGN. The paper claims
   nothing about real mergers, but one clause should say that 0.30 is inside the
   current limits of Pouw et al. and that X-ray selection includes obscured AGN, to which
   those limits do not apply. The mapping between the X-ray and bolometric thresholds
   was not checked here.
8. **AGN clustering.** C3 rests on b_AGN = 2.0 against 1.2. Powell et al. 2018
   (1803.07589) find that local hard-X-ray AGN live in group-scale halos and "occupy
   their dark matter halos similar to inactive galaxies of comparable stellar mass".
   A referee can ask how much of the factor 4.5 is abundance and how much is bias, and
   whether 2.0 is realistic at z < 0.3. The paper needs a clustering reference for the
   value or a statement that it is a design choice, and ideally the split.
9. **Distances of mergers near a supermassive black hole.** Li & Fan 2025 (2503.18333) and
   Liodis et al. 2026 (2609.24476) discuss environmental biases on H0 for AGN-disk
   mergers. One sentence should place this outside the paper's scope.
10. **Counterpart claims.** Kumar & Torres-Orjuela 2026 (2607.03672, 2607.03674) claim 13
    BBH-AGN associations and H0 to 4.4%; Cabrera et al. 2026 and Veronesi et al. 2024
    find no confident association. If cited, cite both sides.

Papers from 2025 and 2026 the authors most need to read in full: Pouw et al. 2026;
Cross-Parkin et al. 2026; Gray, Williams & Papadopoulos 2026; Agapito & Mancarella 2026;
Zhu & Chen 2025; Veronesi et al. 2025; Cabrera et al. 2026; Borghi et al. 2026;
VanWyngarden et al. 2026; Kalomenopoulos et al. 2025; Barbieri et al. 2026; Scarpa et
al. 2026; Moncrieff & Panther 2025; Watarai et al. 2026 (2610.11045); Tong et al. 2025;
the GWTC-5.0 cosmology paper. From before 2025, Bom & Palmese 2024 and Zhu et al. 2024.

Three points about the mock description, found while checking `working/data/README.md`:

- The catalog redshift error of 0.003 (1+z) is below the 0.01 at which Pouw et al. 2026
  (Sec. 4.2.2) find that AGN redshift errors stop limiting f_AGN. They attribute the
  small gain from Quaia on GWTC-5.0 to its redshift precision, so the paper should say
  that its catalogs are spectroscopic in quality and that photometric AGN catalogs
  would return less.
- The B-band Schechter function quoted there (phi* = 1.6e-2 h^3 Mpc^-3, alpha = -1.07,
  M_B* = -20.47 for h = 0.7) is from Gehrels et al. 2016 (1508.03608, ApJ 820, 136),
  which GLADE (Dalya et al. 2018) adopts and attributes. Cite Gehrels et al. as the
  source and GLADE as the user. All three numbers were confirmed in the text of
  Gehrels et al.
- The AGN density of 1e-5 per Mpc^3 for log L_X(2-10 keV) above 43.7 is attributed to the
  BASS DR2 luminosity function (Ananna et al. 2022, 2201.05603). Only the abstract was
  read; the number was not reproduced. The README itself says it was "pinned, not
  re-integrated". It must be integrated from the published function before it is
  quoted. For comparison, the association literature uses 10^-4.75 per Mpc^3 for AGN
  above about 10^44.3 erg/s bolometric (Veronesi et al. 2022 abstract; Bom & Palmese
  2024, citing Greene & Ho 2007 and Bartos et al. 2017).

## 7. Unverified items (no bib entry)

- The collaboration-server version of the GWTC-5.0 cosmology paper, reported by a web
  search to quote H0 = 71.0 +9.0 -7.1 and a 25.7% improvement. Not opened. The arXiv v2
  numbers above were read directly.
- Works seen only in the reference list of Pouw et al. 2026: Li & Fan 2024; Farah et al.
  2026 (a steeply evolving hierarchical subpopulation); Mandel 2026 and Mould et al.
  2026 (parameter-estimation priors); Kritos et al. 2024; McKernan et al. 2024.
- Hopkins, Richards & Hernquist 2007 (bolometric quasar luminosity function), seen only
  in the reference list of Bom & Palmese 2024.
- Callister et al. 2021 (mass ratio and effective spin correlation), seen only in the
  abstract of McKernan et al. 2022.
- Liu et al. 2025, EMRI dark sirens with galaxy clusters: an INSPIRE listing with no
  eprint; not opened.
- Scelfo et al. 2023 on GW clustering bias: named in a search summary; not opened.
- Whether Finke et al. 2021 is the origin of "multiplicative completion": the term is
  in the abstract of Dalang & Baker 2024 without attribution; not checked.
- The number 1e-5 per Mpc^3 for the AGN density (see section 6).

## 8. Searches that found nothing

Queries were run on the arXiv API (abstract search, newest first, through 13 Nov 2026),
on INSPIRE (papers citing Bom & Palmese 2024, Veronesi et al. 2025, Zhu & Chen 2025,
Hanselman et al. 2025, Zhu et al. 2024) and on the web.

- A likelihood with both a galaxy catalog and an AGN catalog and H0 free.
- Any dark-siren study, on data or simulations, with galaxies and AGN drawn from one
  density field.
- Inference of a host number density or luminosity-function normalization from GW
  events (the arXiv query combining dark sirens, number density or luminosity function,
  hosts and the Hubble constant returned no record).
- A fit of host-class-dependent mass or spin distributions that also uses host
  catalogs, with or without H0.
- A quantified dependence of f_AGN on the assumed AGN luminosity function.
- An LVK cosmology analysis with an AGN or quasar catalog as a separate host class
  (the GWTC-5.0 text was searched for AGN, quasar and host weighting).
- A dark-siren H0 measurement based on Quaia or Milliquas.
- "Multi-tracer" dark sirens meaning two electromagnetic tracer catalogs for H0 (the
  term appears only for cross-correlations with galaxies and 21 cm maps, and for
  pulsar timing).
- A cross-correlation analysis that fits a GW bias and H0 with AGN as the tracer and
  returns an AGN fraction (Chakravarti & Urban 2026, 2602.14825, forecast detectability
  with Quaia and 2MPZ only).
- An information or Fisher analysis of H0 precision against host abundance and
  clustering for a mixed host population.
- A GWTC-5-era detection of a non-zero AGN fraction from spatial correlation other than
  Zhu & Chen 2025.
- Any paper posted between mid-July and 13 Nov 2026 that combines H0, f_AGN and
  host-dependent populations (date-restricted listings of dark-siren and of AGN and
  BBH abstracts were scanned title by title).

## 9. Where the references go (keys in `refs_candidates.bib`)

- Dark sirens, method and status: `Schutz:1986gp`, `Holz:2005df`, `MacLeod:2007jd`,
  `DelPozzo:2011vcw`, `Chen:2017rfc`, `LIGOScientific:2018gmd`, `DES:2019ccw`,
  `Gray:2019ksv`, `Gray:2021sew`, `Gray:2023wgj`, `Mastrogiovanni:2023emh`,
  `Gair:2022zsa`, `Finke:2021aom`, `Borghi:2023opd`, `DES:2020nay`, `Palmese:2021mjm`,
  `Bom:2024afj`, `Alfradique:2026dsq`, `McMahon:2026nhi`, `MacBride:2026ufy`,
  `LIGOScientific:2019zcs`, `LIGOScientific:2021aug`, `LIGOScientific:2025jau`,
  `LIGOScientific:2026uyd`, `Palmese:2025zku`.
- AGN channel and its signatures: `Bartos:2016dgn`, `Stone:2016wzz`, `Mckernan:2017ssq`,
  `Tagawa:2019osr`, `Tagawa:2020dxe`, `Tagawa:2020qll`, `Tagawa:2021ofj`,
  `Yang:2019okq`, `Yang:2019cbr`, `Ford:2021kcw`, `McKernan:2021nwk`, `Cook:2024ajp`,
  `Vajpeyi:2021qsw`, `Tagawa:2026afw`, `Vaccaro:2026nkp`, `Vaccaro:2026ocs`.
- Association searches and AGN fractions: `Bartos:2017ggb`, `Corley:2019pxc`,
  `Veronesi:2022hql`, `Veronesi:2023ugk`, `Veronesi:2024pjq`, `Veronesi:2024tef`,
  `Zhu:2025rdv`, `Zhu:2026egg`, `Pouw:2026lar`, `Palmese:2021wcv`, `Cabrera:2025tzc`,
  `Graham:2020gwr`, `Graham:2022xxu`, `Storey-Fisher:2023gca`, `Gayathri:2021xwb`,
  `Gayathri:2023bha`, `Bartos:2026xlt`, `Li:2025iux`, `Padhyegurjar:2026scg`.
- H0 with AGN: `Bom:2023zgw`, `Alves:2020wnl`, `Zhu:2024qpp`, `Mukherjee:2020kki`,
  `Chen:2020gek`, `Gayathri:2020mra`, `Morton:2023wxg`, `Li:2025eig`, `Kumar:2026bqt`.
- Host weighting and host mixtures: `Hanselman:2024hqy`, `Perna:2024lod`,
  `Alfradique:2025tbj`, `Li:2025hrh`, `Gray:2026loe`, `Agapito:2026pbb`,
  `Cross-Parkin:2026pua`, `Vijaykumar:2023bgs`, `Adhikari:2020wpn`, `Borghi:2025pav`.
- Incompleteness: `Dalang:2023ehp`, `Dalang:2024gfk`, `Leyde:2024tov`, `Leyde:2025rzk`,
  `VanWyngarden:2025ogy`, `Kalomenopoulos:2025qpt`, `Barbieri:2026fdy`,
  `Tagliazucchi:2026qwy`, `Ghosh:2026qop`, `Datrier:2025zdf`, `Turski:2025flk`,
  `Rivera:2026rhu`, `Bera:2020jhx`.
- Clustering and cross-correlation: `Oguri:2016dgk`, `Mukherjee:2018ebj`,
  `Mukherjee:2020hyn`, `Mukherjee:2022afz`, `Scelfo:2018sny`, `Calore:2020bpd`,
  `Libanore:2020fim`, `Vijaykumar:2020pzn`, `Ferri:2024amc`, `Pedrotti:2025tfg`,
  `SantiagodeMatos:2025iyj`, `Dalang:2026jhx`, `Pan:2025iya`, `Cheng:2026atn`,
  `Moncrieff:2025xle`, `Smith:2025bkq`, `Bellomo:2026azt`, `Watarai:2026qzu`.
- Spectral sirens and population misspecification: `Taylor:2011fs`, `Farr:2019twy`,
  `Ezquiaga:2022zkx`, `Mastrogiovanni:2021wsd`, `Mukherjee:2021rtw`, `Pierra:2023deu`,
  `Yu:2022spx`, `Agarwal:2024hld`, `Tagliazucchi:2026dpr`, `Scarpa:2026piy`,
  `Gennari:2026dfy`, `Agapito:2026kak`, `Tong:2025xvd`, `Farah:2024xub`, `Mali:2024wpq`.
- Hierarchical inference and numerical accuracy: `Loredo:2004nn`, `Mandel:2018mve`,
  `Thrane:2018qnx`, `Vitale:2020aaz`, `Farr:2019rap`, `Essick:2022ojx`,
  `Talbot:2023pex`, `Essick:2023upv`, `Heinzel:2025ogf`, `Speagle:2019ivv`.
- Mocks: `Tessore:2023zyk`, `Lewis:1999bs`, `Gehrels:2015uga`, `Dalya:2018cnd`,
  `Dalya:2021ewn`, `Ananna:2022hwn`, `Powell:2018`, `Greene:2007xw`, `Talbot:2018cva`,
  `KAGRA:2021duu`, `LIGOScientific:2025pvj`, `LIGOScientific:2026ctl`, `Farah:2023vsc`,
  `Planck:2015fie`.
