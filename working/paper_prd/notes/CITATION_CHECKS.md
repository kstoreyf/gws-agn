# Citation checks for the first draft of `main.tex` (17 Nov 2026)

`main.tex` cites 85 entries of `refs_candidates.bib`; `refs.bib` holds exactly those. Forty of them
carry the note "abstract not read in this session (standard reference): confirm the specific
statement cited before use" (the "record only" entries). This file says, for each of those forty,
what the paper attributes to it and what was opened to confirm it, then lists the statements that
could not be confirmed and how the text was changed, and last the quotations and numbers taken
from entries that were not record-only.

How the checks were made: the arXiv API (`export.arxiv.org/api/query?id_list=...`) for abstracts,
`arxiv.org/html/<id>` and `ar5iv.labs.arxiv.org/html/<id>` for full text, and the INSPIRE API for
the one entry without an arXiv record. Requests carried arXiv numbers or texkeys only. "Abstract"
below means the abstract returned by the arXiv API on 16 Nov 2026; "full text" means the HTML
version was fetched and searched for the statement.

## 1. Record-only entries that the paper cites

### 1a. Confirmed in the full text

| key | statement in the paper | what was found |
|---|---|---|
| `Planck:2015fie` | the input cosmology (H0 = 67.74, Omega_m = 0.3075) is "close to the Planck 2015 values" | full text (ar5iv), parameter table: H0 = 67.74 +- 0.46 and Omega_m = 0.3089 +- 0.0062 in the last column. 0.3075 itself is not in the paper, hence "close to" |
| `Mandel:2018mve` | Eq. (2): likelihood marginalized over the merger rate with a log-uniform prior | full text (ar5iv): "if a prior pi(N) proportional to 1/N is assumed on the intrinsic event number or rate ... can be marginalised over N" |
| `Farr:2019rap` | last term of Eq. (A3) corrects for the uncertainty of the detected fraction | full text (ar5iv), Eq. (15): the term (3 N_obs + N_obs^2) / (2 N_eff). The text also gives N_eff > 4 N_obs; the paper's own requirement (5N) is not attributed to this reference |
| `Farah:2023vsc` | the measurement model of the mock events is "adapted from" it | full text (ar5iv), Appendix: observed signal-to-noise ratio = true value plus unit-variance Gaussian noise, detection when it exceeds 8, other uncertainties scaling inversely with it |
| `Essick:2023upv` | detection must be decided by the observed data on which the posterior is conditioned | abstract and full text (ar5iv): biases are "associated with the assumption that detectability is independent of the observed data given the true source parameters" |
| `LIGOScientific:2025pvj` | analyses of the observed catalogs limit the variance of ln L to one | full text (ar5iv), appendix: "we adopt a threshold on sigma^2_lnL of 1" |
| `Essick:2022ojx` | cited as the basis of that limit ("following Refs.") | full text (ar5iv): precision requirement stated through the variance of the log of the estimator; cited for it by the GWTC-4 population paper |
| `Talbot:2023pex` | same | full text (ar5iv): sets a threshold on the uncertainty of the log-likelihood estimator "above which we expect to see significant biases" |

### 1b. Confirmed from the abstract (general attributions; no number or quotation rests on them)

| key | statement in the paper | abstract says |
|---|---|---|
| `Schutz:1986gp` | a gravitational-wave signal gives the distance without a distance ladder; statistical assignment to galaxies | INSPIRE abstract: the signal "contains enough information to determine the absolute distance to the binary"; H0 from ten events. The statistical use of galaxies is attributed to this paper by the abstract of `Gair:2022zsa` ("as originally proposed by Schutz (1986)") |
| `Holz:2005df` | standard sirens | defines standard sirens, distance without redshift |
| `DelPozzo:2011vcw`, `LIGOScientific:2018gmd`, `DES:2019ccw`, `Gray:2019ksv`, `DES:2020nay`, `Palmese:2021mjm`, `Gair:2022zsa` | the dark siren method assigns each merger to the galaxies of a catalog in its localization volume | each abstract describes a statistical H0 measurement or its method with galaxies in the localization volume as potential hosts |
| `Taylor:2011fs`, `Farr:2019twy`, `Ezquiaga:2022zkx` | the spectral siren method uses features of the source-frame mass distribution | narrow neutron-star mass distribution; the pair-instability mass scale; "spectral sirens" from the full mass distribution |
| `Borghi:2023opd` | dark and spectral sirens combined in one likelihood | CHIMERA, "combining information from the population properties of compact binary mergers and galaxy catalogs" |
| `Dalya:2021ewn` | GLADE+ is the catalog of the GWTC-4 analysis | the GLADE+ catalog paper; the use in GWTC-4 is in the abstract of `LIGOScientific:2025jau` |
| `Bartos:2016dgn`, `Stone:2016wzz`, `Mckernan:2017ssq`, `Tagawa:2019osr` | AGN disks can capture black holes, pair them and drive them to merge | binaries trapped in the disk merge rapidly; hardening by gas and three-body scattering; merger rates, masses and spins in disks; binary formation by gas capture |
| `Yang:2019okq`, `Yang:2019cbr` | orbital alignment and repeated mergers favor heavier black holes | "orbital alignment with the AGN disks preferentially selects heavier black holes"; hierarchical mergers in disks, 40% with a black hole above 50 Msun, spins aligned or anti-aligned with the orbit |
| `Tagawa:2020dxe`, `Tagawa:2021ofj` | mass and spin signatures of mergers in disks | distribution of the effective spin of mergers in AGN disks; spin and chirp-mass signatures of hierarchical mergers |
| `Storey-Fisher:2023gca` | names the Quaia catalog | the Quaia catalog paper |
| `Graham:2020gwr`, `Graham:2022xxu` | flare searches "have found candidates" | a candidate counterpart to S190521g; nine candidate counterparts in the third run |
| `Loredo:2004nn`, `Thrane:2018qnx`, `Vitale:2020aaz` | hierarchical Bayesian population inference | each is a general treatment of it (selection effects in the first and third) |
| `Talbot:2018cva` | primary mass distribution with a power law and a Gaussian peak | parameterized mass spectrum with "an excess of black holes near 40 Msun" from pair-instability supernovae |
| `Speagle:2019ivv` | nested sampling | the dynesty paper; LIKELIHOOD_MATH Sec. 0 names dynesty as the sampler of the runs |
| `Tessore:2023zyk` | the lognormal field is generated with GLASS | GLASS builds a light cone of matter and galaxies in nested shells, with lognormal fields |
| `Lewis:1999bs` | the matter power spectra are computed with CAMB | this is the reference the CAMB code asks for; its abstract concerns the line-of-sight computation of CMB anisotropies and does not mention the nonlinear matter spectrum. Kept as the code citation; add the nonlinear-correction reference that the generator used if the owner wants it (`working/data/README.md` says only "non-linear") |

## 2. Statements that could not be confirmed, and what the text does instead

| statement first intended | problem | text as written |
|---|---|---|
| The AGN class is "log L_X(2-10 keV) above 43.7" with a density of 10^-5 Mpc^-3 from the BASS luminosity function (`Ananna:2022hwn`; `working/data/README.md`, macro `\MockLXcut`) | NOVELTY Sec. 6 and 7: the number was not reproduced from the published function and the mapping between X-ray and bolometric thresholds was not checked; the abstract gives no density | The density and the bias are called design choices. The anchor given is the 10^-4.75 Mpc^-3 of the association studies (`Veronesi:2022hql`, abstract; `Bom:2023zgw`, full text). `Ananna:2022hwn` is not cited and `\MockLXcut` is not used |
| The power-law-plus-peak mass model is that of the GWTC-3 population analysis (`KAGRA:2021duu`) | record-only; the abstract does not name the model and no full-text HTML could be fetched (ar5iv returned "no content") | cited to `Talbot:2018cva` alone; `KAGRA:2021duu` is not in `refs.bib` |
| Rate marginalization with a log-uniform prior, cited to `Mandel:2018mve` and `Vitale:2020aaz` | the full text of Vitale et al. says the number of events "is marginalized using a Jeffreys prior", which is not unambiguous | the statement cites `Mandel:2018mve` only; `Vitale:2020aaz` stays in the general citation of the method |
| The spatial searches compare the AGN with mergers "uniform in comoving volume" (three references) | confirmed in the full text only for `Pouw:2026lar` (their Eq. 11) | "a background of mergers that does not cluster", the wording NOVELTY Sec. 5 checked for all three |
| The measured galaxy sample is what "a spectroscopic survey reaches" | no source | removed; the sentence now says "the bright end of the galaxy population" |

## 3. Quotations and numbers from entries that were not record-only

Opened in full text (arXiv HTML) for this draft:

- `Bom:2023zgw` (v2): the quotations "the advantage of not requiring full galaxy catalogs, but the
  disadvantage of not providing constraining power for events that do not occur in AGNs", "how to
  use AGN hosts in concert with spectral siren analyses" and "one would leave the AGN number density
  free to vary"; the fiducial density 10^-4.75 Mpc^-3; lower AGN density gives tighter H0; the test
  with a free density removing the extra cosmological information.
- `Cross-Parkin:2026pua` (v1): "only a weak correlation", "extending the likelihood to model host
  weights and compact-binary population parameters simultaneously", "potentially degenerate with
  the astrophysical weighting of the host population".
- `Pouw:2026lar` (v1): "defer the inference of intrinsic population hyperparameters to future
  work"; fixed Planck 2015 cosmology; AGN redshift errors below about 0.01 give little further
  gain; "f_agn depends on the maximum redshift at which GWs can be emitted", which changes the two
  normalizations differently; a denser catalog is less able to distinguish the two populations; the
  limits apply to unobscured AGN only; the position-based and spin-based rankings of events
  disagree. The limits 0.52, 0.41, 0.61 (95%) are from the abstract.
- `LIGOScientific:2026uyd` (v2): Table 3, H0 = 71.7 +17.7 -11.4 at 90% with GW170817; "the main
  driver of the Hubble constant measurement remains the presence of features in the mass
  distribution". Source count 236 and the Dark Energy Survey catalog are from the abstract.
- `Zhu:2024qpp` (v2): "the degeneracy between f_agn and f_compl" and the dependence on the AGN
  density; H0 precision from 3.7% to 3.2% with the AGN catalog.
- `Zhu:2025rdv` (v2): Eq. (1), where f_agn multiplies the completeness factor; the value
  0.39 +0.41 -0.32 (90%) is from the abstract.

From abstracts (arXiv API): `LIGOScientific:2025jau` (142 of 218 sources, GLADE+, 90% interval),
`Bartos:2017ggb` (25% with 1000 events), `Veronesi:2023ugk`, `Veronesi:2024tef` (0.21 at 95% above
10^44.5 erg/s, Quaia), `Veronesi:2022hql` (10^-4.75 Mpc^-3), `Veronesi:2024pjq`, `Palmese:2021wcv`,
`Cabrera:2025tzc`, `Gayathri:2021xwb` (25% of the second catalog), `Gayathri:2023bha`,
`Bartos:2026xlt` (about 10%, accretion), `Li:2025iux`, `McKernan:2021nwk`, `Gray:2026loe`,
`Agapito:2026pbb`, `Mastrogiovanni:2023emh`, `Gray:2023wgj`, `Mastrogiovanni:2021wsd`,
`Pierra:2023deu`, `Scarpa:2026piy` (correlation of H0 with the channel fraction),
`Hanselman:2024hqy`, `Perna:2024lod`, `Alfradique:2025tbj`, `Li:2025hrh`, `Borghi:2025pav`
("robust against host weighting mismodeling"; the correlation of H0 with the mass scale grows as
completeness falls), `Kalomenopoulos:2025qpt`, `VanWyngarden:2025ogy`, `Dalang:2023ehp`,
`Leyde:2025rzk`, `Barbieri:2026fdy`, `Tagliazucchi:2026qwy`, `Alves:2020wnl`, `Beirnaert:2025wcx`,
`Powell:2018` (group environments), `Li:2025eig`, `Liodis:2026xvs`, `Finke:2021aom`. `Gehrels:2015uga`
and `Dalya:2018cnd` rest on the full-text notes of the candidate file.

## 4. Entries dropped for length, not for doubt

`Chen:2017rfc`, `Vajpeyi:2021qsw` and `Ghosh:2026qop` were cited in an earlier pass inside long
lists and were removed to shorten the reference list. Each was consistent with its sentence at the
level of the abstract and can be restored.

## 5. Changes made in `refs.bib` against `refs_candidates.bib`

The `note` fields (verification notes) are not carried over, because `apsrev4-2` prints them; the
erratum note of `Thrane:2018qnx` is kept. Thirteen entries without a `journal` field have the entry
type `misc` instead of `article`, as in the lensing drafts, so that BibTeX raises no error. The
underscore in the DOI of `Vitale:2020aaz` is escaped. Nothing else differs.
