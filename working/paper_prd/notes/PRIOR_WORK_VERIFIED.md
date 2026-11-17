# Prior-work statements checked against the papers (17 Nov 2026)

This file answers part F of `REVIEW.md` (items F1 to F11). For each item it gives (a) the source and
version, (b) the text of the paper, verbatim, with its place, (c) a verdict on the reviewer's claim,
(d) one or two sentences that the paper can safely say, and (e) what could not be opened or verified.
It does not change `main.tex`. The only file changed is `refs.bib` (item F11).

## How the papers were read

Every paper was downloaded as a raw file on 17 Nov 2026: the PDF from `arxiv.org/pdf/<id>` and the HTML
from `arxiv.org/html/<id>`. No summarizing fetch was used. The PDF was turned into text with `pdftotext`
and the HTML with a plain tag stripper (formulas replaced by their LaTeX source). A sentence is given
below in quotation marks only if the same words were found in both texts; the few places where one of
the two failed are marked. Symbols inside quotations are written in plain form (f_agn, z_max, H0, Msun).
Versions and dates are from the arXiv API (`export.arxiv.org/api/query`), journal data from Crossref
(by DOI) and INSPIRE (by arXiv number). Requests carried arXiv numbers, DOIs or the title of the cited
paper only.

| paper | version read | date | journal record |
|---|---|---|---|
| Bom and Palmese, arXiv:2307.01330 | v2 | 19 Sep 2024 | Phys. Rev. D 110, 083005 (2024); v2 comment "accepted by PRD" |
| Pouw et al., arXiv:2609.17665 | v1 (only version) | 15 Sep 2026 | none (preprint in MNRAS format) |
| Cross-Parkin et al., arXiv:2610.25866 | v1 (only version) | 19 Oct 2026 | none |
| Kumar and Torres-Orjuela, arXiv:2607.03672 | v2 (v1: 4 Jul 2026) | 7 Jul 2026 | none |
| Kumar and Torres-Orjuela, arXiv:2607.03674 (companion) | v1 | 4 Jul 2026 | none |
| Moncrieff and Panther, arXiv:2508.18704 | v1 (only version) | 26 Aug 2025 | MNRAS 543 (2), 1833-1841, online 22 Sep 2025 |
| Zhu et al., arXiv:2403.04950 | v2 | 4 Jun 2024 | Astrophys. J. Suppl. 273, 24 (2024) |
| Zhu and Chen, arXiv:2505.02924 | v2 | 24 Jul 2025 | Astrophys. J. Lett. 989, L15 (2025) |
| Veronesi et al., arXiv:2306.09415 | v2 | 24 Oct 2023 | MNRAS 526, 6031 (2023) |
| Veronesi et al., arXiv:2407.21568 | v2 | 16 Dec 2024 | MNRAS 536, 375 |
| Mastrogiovanni et al., arXiv:2305.10488 | v2 | 14 Dec 2023 | Phys. Rev. D 108, 042002 (2023) |
| fifth catalog, cosmology, arXiv:2605.27227 | v2 and v1 | 4 Aug 2026 (v1: 26 May 2026) | none (arXiv and INSPIRE records, 17 Nov 2026) |
| fourth catalog, cosmology, arXiv:2509.04348 | v3, the published version | 20 Aug 2026 | Astrophys. J. Lett. 1007, L17 (2026 August 10) |

Journal texts behind a paywall or a bot check were not opened (Phys. Rev. D, MNRAS at Oxford). Where the
arXiv version is the accepted or published one this is said in the table; otherwise section and equation
numbers below are those of the arXiv version.

## Verdicts in one table

| item | reviewer's claim | verdict |
|---|---|---|
| F1 | Bom and Palmese simulate an AGN channel above 35 Msun and find the H0 precision "little changed" | partly right: true for flares (Sec. IV.2), wrong for the AGN catalog case (Sec. III.2: 5-10% becomes 11-27%) |
| F2 | earlier statements exist that completeness limits fAGN | right, and stronger than stated |
| F3 | arXiv:2607.03672 reports 13 mergers with AGN hosts and H0 = 70.5 (+3.4, -2.9) | right; level is 68.3%, statistical only |
| F4 | Moncrieff and Panther "select hierarchical mergers by mass" | imprecise: the hierarchical population is assumed known, no selection is made; the rest is right |
| F5 | 3.7% to 3.2% holds with every inspiral in an AGN, one detector, one model, zero eccentricity, no redshift errors | right |
| F6 | two values of the fifth-catalog H0 circulate; the 90% interval could not be opened | right; they are arXiv v1 and v2, and the 90% interval is in Table 3 of v2 |
| F7 | "changes the two normalizations differently" is the draft's reading | wrong: it is the authors' own sentence; the other points are right |
| F8 | Eq. (16) of Ref. [15] defines a rate "per year at a reference galaxy luminosity" | right |
| F9 | the three quotations are exact; "none names a population that depends on the host class" | quotations exact; the second claim is wrong for Cross-Parkin et al. and Pouw et al. |
| F10 | the product of fAGN and completeness is the form of Veronesi et al. | right |
| F11 | bibliography | fixed in `refs.bib`; one more published entry found |

---

## F1. Bom and Palmese on high-mass mergers in AGN

**(a) Source.** arXiv:2307.01330v2 (19 Sep 2024). The HTML numbers the subsections IV.1, IV.2, ...; the
PDF, in the journal's style, IV A, IV B, ... . Both forms are given.

**(b) Text.**

Sec. IV.2 (IV B), title "BBH in AGNs as high mass mergers". This section is about the flare case:
- "To take this into account, we conduct two tests. First, we assume a scenario in which BBH mergers only
  occur in AGN disks if the primary mass m_1 is beyond the ~30-35 Msun peak observed in the mass
  distribution. We assume m_1 > 35 Msun for the BBHs in AGNs for simplicity"
- "We present the results for this case in Figure 7. We find that the precision worsens, as expected,
  compared to our previous findings, although only slightly when comparing results for the same values
  of lambda."
- Second test: "so we also sample for each high mass BBH in AGN a low mass BBH in AGN with primary below
  the peak. The results in this case become extremely similar to these in Figure 2 for the same values
  of lambda, so we do not show them in the figure. We conclude that a 2-10% precision in H0 is also
  possible with higher mass BBHs merging in AGNs with a A+ sensitivity, assuming that lambda > 1%."
- Caption of Fig. 7: "for a scenario in which only BBH mergers with higher primary masses produce AGN
  flares".

Sec. III.2, "AGN catalog case" (no flares; this is the case comparable with our catalogs):
- "Similarly to the AGN flares case, we consider the case where only BBHs on the high-mass end of the
  merging BH mass distribution merge in AGN disks. In that case, we find that the precision is
  significantly worsened, going from 5-10% precision to ~11-27%. This is expected as the most massive
  mergers are more likely to be at larger distances and encompass larger localization volumes."
- Table 1 (80 events, n_AGN = 10^-4.75 Mpc^-3, the best-localized half of the events; "median and 68%
  interval from the various runs made"), for input lambda_AGN = 0.3, 0.5, 0.6, 0.7, 0.8:

  | | 0.3 | 0.5 | 0.6 | 0.7 | 0.8 |
  |---|---|---|---|---|---|
  | random masses, sigma(H0)/H0 in % | 11.1 +9.5 -4 | 10.8 +1.6 -4.0 | 6.2 +1.6 -0.7 | 6.0 +2.3 -0.3 | 5.0 +8.8 -0.5 |
  | random masses, recovered lambda_AGN | 0.34 +0.20 -0.13 | 0.49 +0.23 -0.17 | 0.61 +0.22 -0.18 | 0.70 +0.21 -0.17 | 0.80 +0.8 -0.14 |
  | high-mass only, sigma(H0)/H0 in % | 26.5 +2.0 -2.0 | 21.2 +5.9 -5.3 | 18.2 +1.2 -5.7 | 15.4 +5.5 -4.0 | 11.25 +8.8 -1.7 |
  | high-mass only, recovered lambda_AGN | 0.35 +0.12 -0.03 | 0.53 +0.37 -0.07 | 0.56 +0.40 -0.11 | 0.51 +0.38 -0.06 | 0.48 +0.51 -0.06 |

  (The upper error "+0.8" of the last entry of the second row is printed so in both versions.)

Sec. IV.3 (IV C), "Caveats":
- "Another caveat for the forecast presented here is the assumption that some fraction of the observed
  BBH population occurs in AGN, where there is no correlation between this specific formation channel
  and the BBH properties. In reality, it is possible that the AGN formation channel dominates the higher
  mass end of the BH mass distribution, thanks to its ability to retain hierarchical mergers."

Sec. IV.1 (IV A), "Selection effects":
- "Since this study is focused on the precision, we disregard the influence of this selection bias in
  our analysis."
- "Future work should focus on the impact of the mass distribution assumptions on the BBH-AGN formalism
  used here, as well as considering how to use AGN hosts in concert with spectral siren analyses as
  currently proposed with a generic host galaxy population [59, 60]."

The likelihood (Sec. II.1, Eq. 2) takes the sky position and luminosity distance of each event and the
positions and redshifts of the AGN; masses and spins do not enter. (This description is from the HTML
text; the PDF text of that passage was broken by the column layout.)

**(c) Verdict: partly right.** The section number and title, the 35 Msun cut, the statement of Sec. IV.3
and the future work of Sec. IV.1 are as the reviewer says. "Finds the H0 precision little changed" is
true of the flare case only (Sec. IV.2, Fig. 7: "only slightly"). For a static AGN catalog the same
restriction makes the precision "significantly worsened", from 5-10% to about 11-27% (Sec. III.2,
Table 1). The reviewer's replacement sentence would misreport the catalog case, which is the one the
draft is compared with. In both cases the restriction enters the simulated events only; the model that
is fitted has no mass or spin distribution.

**(d) Safe sentences.**
- None of these analyses infers a mass or spin distribution that depends on the host class. Bom and
  Palmese simulate the case in which only mergers with primary masses above 35 Msun occur in AGN: their
  H0 precision is little changed when flares identify the hosts and degrades from 5-10% to 11-27% with
  an AGN catalog alone, and the likelihood contains no mass distribution.
- They also note that the AGN channel may dominate the high-mass end of the mass distribution and leave
  the mass-distribution assumptions to future work.

**(e) Not verified.** The published Phys. Rev. D text was not opened; v2 is marked as accepted.

---

## F2. Earlier statements that completeness limits fAGN

**(a) Sources.** Zhu and Chen, arXiv:2505.02924v2 (draft Ref. [34]); Veronesi et al., arXiv:2306.09415v2
(Ref. [31]) and arXiv:2407.21568v2 (Ref. [33]); Pouw et al., arXiv:2609.17665v1 (Ref. [35]); Zhu et al.,
arXiv:2403.04950v2.

**(b) Text.**

Zhu and Chen, Sec. 3 (Methodology), Eq. (1): the likelihood is the product over events of
[0.9 c_i f_agn S_i + (1 - 0.9 c_i f_agn) B_i], and "c_i accounts for the completeness of the AGN catalog
since faint objects may be missed by observations". After Eq. (3), at the end of the same section:
- "We notice that the two parameters f_agn and c_i always appear as a product in the likelihood.
  Therefore, they are degenerate in our analysis. Breaking this degeneracy requires an accurate
  estimation of c_i, but it is difficult due to the poor observational constraints on the luminosity
  function"
- they then set c_i to the sky-coverage fraction: "This is obviously an optimistic assumption and will
  lead to an overestimation of c_i. Consequently, the resulting f_agn is lower than its real value and
  should be regarded as a conservative estimation."

Veronesi et al. 2023, Sec. 3.1, Eq. (1) (same form with one average completeness c), and:
- "The introduction of the c factor is a novelty with respect to previous similar works that used only
  complete mock AGN catalogues (Bartos et al. 2017; Corley et al. 2019; Veronesi et al. 2022)."
- Sec. 5 (Discussion and conclusion): "the prediction power of the method presented in this work depends mainly on three
  elements: the completeness of the AGN catalogue, the number of GW detections, and the size of their
  localisation volumes."

Veronesi et al. (the Quaia paper), Sec. 2.2:
- "In order to obtain tight constraints of f_AGN with methods based on spatial correlation like the one
  used in this work, is important to use an AGN catalogue that has the highest possible value of
  completeness."

Pouw et al., Sect. 7 (conclusions from the mock data):
- "The constraining power of an AGN catalogue is limited by its completeness and redshift precision. To
  achieve near-optimal constraints, the catalogue must be complete within GW localisations, and typical
  AGN redshift uncertainties should be roughly an order of magnitude smaller than those of the GW
  sample."
- Sect. 4.2.2, where the completeness is varied through a redshift cut of the AGN catalogue (their
  Fig. 6): "Constraints on f_agn only start to improve once z_thresh^EM >~ 0.6, and they do not improve
  further for z_thresh^EM > z_thresh^GW. This shows the necessity of spatial overlap between GWs and AGN
  for EM-informed population inference".

Zhu et al. 2024, Sec. II.3 (2.3), after Eq. (17), for extreme-mass-ratio inspirals: "Several factors
render it exceedingly challenging to accurately infer f_agn directly from Equation (17), including (i)
the degeneracy between f_agn and f_compl,i^agn, (ii) the sensitivity of the likelihood to the accurate
determination of rho_agn".

**(c) Verdict: right.** Every statement the reviewer names exists as described. The earlier statements
are in fact stronger than the review says: Zhu and Chen state the degeneracy in words and fix the
completeness at its most optimistic value because of it, and Pouw et al. measure the width of fAGN
against catalog completeness (their Fig. 6). "We are not aware of an earlier statement for fAGN" cannot
stand.

**(d) Safe sentences.**
- For fAGN the association searches find that the completeness of the AGN catalog limits the
  constraint: the fraction and the completeness enter their likelihood as a product
  [Veronesi 2023, Veronesi Quaia, Zhu and Chen], and Pouw et al. find from mock catalogs that the
  catalog must be complete within the localization volumes.
- What can be claimed as ours is the size of the cost when the number of uncataloged hosts is given to
  the model, and the reason it is small here (the reviewer's wording for the reason is the draft's own
  result and is not a statement of these papers).

**(e) Not verified.** Nothing.

---

## F3. arXiv:2607.03672, an H0 from BBH mergers with claimed AGN hosts

**(a) Source.** D. Kumar and A. Torres-Orjuela, "Hubble constant measurement with 13 bright standard
sirens from binary black hole mergers inside active galactic nuclei", arXiv:2607.03672v2 [astro-ph.CO],
7 Jul 2026 (v1: 4 Jul 2026); 4 pages, 2 figures, 1 table; no journal record at arXiv or INSPIRE on
17 Nov 2026. The associations are made in a companion paper by the same authors, "A population of
LIGO-Virgo-KAGRA mergers happening inside active galactic nuclei", arXiv:2607.03674v1 [astro-ph.GA],
4 Jul 2026, also unpublished. Both were read in full text.

**(b) Text.**

The claim (abstract): "We measure the Hubble constant H0 using 13 gravitational-wave binary black hole
mergers associated with active galactic nucleus hosts. We find H0 = 70.50 +3.37 -2.89 (stat) +- 1.56
(cal) km s^-1 Mpc^-1 (4.4% precision)".

Credible level (caption of Table 1): "H0 is reported as the posterior median with 68.3% and 90% symmetric
credible intervals". The rows of Table 1 (68.3%; 90%):

| sample | H0 | 68.3% | 90% |
|---|---|---|---|
| 13 BBH-AGN | 70.50 | +3.37 -2.89 | +5.77 -4.81 |
| 13 BBH-AGN + GW170817 | 70.31 | +3.00 -2.85 | +5.26 -4.51 |
| 13 BBH-AGN + the dark and bright sirens of Bom et al. | 69.71 | +2.55 -2.40 | +4.21 -3.90 |

The 13 single events give medians of 70.7 to 81.1 with 68.3% widths of 16 to 19%. The "(cal)" term is
a calibration systematic of 2.2%, quoted separately. Priors: H0 in [20, 140], Omega_m in [0.05, 0.6].

How each event is used (Methodology): "Because each event is associated with a single,
spectroscopically identified AGN host, z_AGN,i enters as a fixed quantity rather than the marginalised
host-redshift distribution p(z) required by the galaxy-catalogue dark-siren method". The AGN redshift is
corrected for the Doppler and gravitational shifts of the central black hole. The Conclusion calls the
sample "a catalogue of spectroscopically confirmed, one-to-one BBH-AGN associations".

How the associations are made (Introduction of 2607.03672, describing the companion): "we performed a
Bayesian coincidence-classification analysis of 18 gravitational-wave events from the LVK O3-O4b
catalogue against 28 candidate AGN counterparts, jointly accounting for GW sky-localisation, AGN
positional coincidence, and SMBH-induced environmental redshift corrections. Of the 28 candidate pairs,
21 were found to be positive-to-strong-favoured against the vacuum baseline (log-Bayes factor > 1). 13
unique one-to-one BBH-AGN associations were then identified once competing counterparts for the same
event, or the same AGN claimed by multiple events, were adjudicated by their relative evidence."

Which AGN (companion paper, Introduction): the 28 candidates are AGN already proposed as counterparts in
published searches, "compiled from the searches above": the nine flares of the Zwicky Transient
Facility search of the third run, and candidates proposed for later events by Cabrera et al., He et al.
and Bommireddy et al., and by follow-up of single events. The companion says of its own result: "The
resulting Bayes factors provide the first systematic, GW-side quantification of which proposed
counterparts are favoured or disfavoured by the data itself - ranking each candidate's consistency with
the GW posteriors rather than constituting an exhaustive search over the full AGN sky." (The original
has a long dash where a hyphen stands here.) Its abstract gives a combined ln B of about +81 for the 13
and says "Sky localisation is the dominant driver of model selection".

**(c) Verdict: right.** 13 mergers, AGN hosts, 70.5 (+3.4, -2.9): all as the reviewer says. To be added:
the interval is 68.3% and statistical; the 90% interval is +5.8 -4.8; the hosts are previously proposed
flare and counterpart candidates, and the test is of each candidate against a gravitational-wave-only
baseline, not a chance-coincidence test against the AGN in the localization volume.

**(d) Safe sentences.**
- Searches for AGN flares coincident with mergers have found candidates. Several analyses find no
  confident association [Refs. 36-40]; Kumar and Torres-Orjuela rank 28 proposed AGN counterparts of 18
  mergers by their consistency with the gravitational-wave posteriors, retain 13 one-to-one
  associations, and from these, with each AGN redshift taken as the redshift of the merger, report
  H0 = 70.5 +3.4 -2.9 km s^-1 Mpc^-1 (68%, statistical).
- At l. 145: write that BBH mergers have no generally accepted counterpart, or that their counterparts
  remain candidates, and cite the claim; "no confirmed electromagnetic counterpart" contradicts the
  wording of this paper, which calls its 13 associations confirmed.

BibTeX entries: `Kumar:2026bqt` is now in `refs.bib`. The companion is not; if it is cited, add:

```
@misc{Kumar:2026enp,
    author = "Kumar, Dhruv and Torres-Orjuela, Alejandro",
    title = "{A population of LIGO-Virgo-KAGRA mergers happening inside active galactic nuclei}",
    eprint = "2607.03674",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.GA",
    month = "7",
    year = "2026"
}
```

**(e) Not verified.** Whether either paper has been submitted or refereed. The works from which the 28
candidates are taken were not opened.

---

## F4. Moncrieff and Panther, arXiv:2508.18704

**(a) Source.** arXiv:2508.18704v1 (26 Aug 2025, "submitted to MNRAS"), full text. Journal reference from
Crossref and INSPIRE: J. W. N. Moncrieff and F. H. Panther, Mon. Not. R. Astron. Soc. 543 (2), 1833-1841
(2025), DOI 10.1093/mnras/staf1582, online 22 Sep 2025. The DOI resolves to
`academic.oup.com/mnras/article/543/2/1833/...`, which confirms volume, issue and first page.

**(b) Text.**

What is selected: nothing is selected. The mergers are assumed to be known hierarchical mergers.
- Sec. 2.3: "We assume a population of BBHs with independent masses that are uniformly distributed
  between 10-100 solar masses, with aligned spins that are uniformly distributed chi_z ~ Unif(0, 1).
  While a truly hierarchically merging population is expected to have a different distribution of
  parameters, these do not impact the waveforms significantly enough to affect the recovered SNR"
- "We use this BBH population to represent a future BBH population that is known to derive from
  hierarchical mergers."
- Sec. 4: "We adapt these methods to address the question of whether an already established to be
  hierarchically merging population of BBHs have formed in AGN disks or not."
- The motivation, in the abstract, is the detection of mergers with masses of about 80 to 150 Msun.

What is measured: the angular (two-dimensional) cross-correlation of the summed sky localizations with
two catalogs, Quaia quasars below z = 1.5 as the AGN and GLADE+ with its quasars removed as the other
galaxies. No redshift information is used ("we ignore any three-dimensional correlations in this work")
and the catalogs are assumed complete ("The scenario we present is idealized, in the sense that we
assume that these catalogs are complete"). Detector network: LIGO Hanford, LIGO Livingston and Virgo at
A# sensitivity; populations of 1000, 5000 and 10 000 mergers. Sec. 3 fits a mixture of the two origins
to one realization of 10 000 mergers.

Number of detections:
- Abstract: "We find that if the observed merging population is dominated by binaries formed dynamically
  in AGN, then this could be determined with O(5000) mergers detected at the sensitivity that is
  projected for the upcoming A# gravitational wave detectors."
- Sec. 4: "we demonstrate that with O(5000) events, it is possible to distinguish which evolutionary
  channel dominates the hierarchical merger rate."

They defer to future work the effect of catalog completeness, the use of redshifts, and the bias from a
population with higher chirp masses being better localized ("an excess of mergers with higher chirp
masses, as expected for the AGN channel, could result in a population of events with higher SNRs and
consequently smaller sky localization areas. We defer investigation of these biases to future works").

**(c) Verdict: imprecise on one point.** "Selects hierarchical mergers by mass" is not what the paper
does: it takes the hierarchical population as given and makes no selection (as `notes/NOVELTY.md`
already says). The cross-correlation, the AGN versus other galaxies question, the figure of about 5000
and the journal are right. The number is for hierarchical mergers alone, at A# sensitivity, with
complete catalogs and sky positions only.

**(d) Safe sentences.**
- Moncrieff and Panther take a population of mergers already identified as hierarchical and forecast
  that the angular cross-correlation of their sky localizations with a quasar catalog and a galaxy
  catalog tells which class hosts most of them once of order 5000 such mergers are detected at A#
  sensitivity; they assume complete catalogs and use no redshifts.
- In that work the intrinsic properties select the sample and the positions classify it; the two are
  not inferred together.

**(e) Not verified.** The published MNRAS text (bot check at the publisher). Only the submitted arXiv
version was read; a number changed in refereeing would not be seen here.

---

## F5. Zhu et al. 2024, the 3.7% and 3.2%

**(a) Source.** arXiv:2403.04950v2 (4 Jun 2024, accepted version), Sec. VI.1 in the HTML and 6.1 in the
PDF, "Advantage of combining EMRIs with AGN observation".

**(b) Text.** "Assuming that all EMRIs come from AGNs, and using only the AGN catalog to look for the
candidate hosts for EMRIs, we repeat the simulation shown in Section III.2 and derive the posteriors for
the cosmological parameters. More specifically, we consider the EMRIs detected by TianQin in the M1
population model as an example. [...] More quantitatively, by switching from the galaxy catalog to the
AGN catalog, the precision of H0 improves from about 3.7% to about 3.2%, and the precision of w0
improves from about 25% to about 13%. For a fair comparison, here we do not consider the redshift errors
in either candidate host galaxies or AGNs." Caption of Fig. 8: "The eccentricities at plunges are set to
zero, and the other parameters are derived from the M1 model." They attribute the gain to AGN being
"only about (1-10)% of all galaxies".

**(c) Verdict: right** in every particular (one detector, one population model, every inspiral in an
AGN, zero eccentricity, no redshift errors).

**(d) Safe sentence.** For extreme-mass-ratio inspirals that all occur in AGN, replacing the galaxy
catalog by an AGN catalog improves the H0 precision from about 3.7% to about 3.2% in one simulated case
[Zhu et al.].

**(e) Not verified.** Nothing.

---

## F6. The H0 of the fifth catalog, and of the fourth

**(a) Sources.** Fifth catalog: arXiv:2605.27227, v2 of 4 Aug 2026 (51 pages) and v1 of 26 May 2026
(42 pages), both read; the public document of the collaboration, LIGO-P2600018-v14 (`dcc.ligo.org`),
whose PDF is dated 26 May 2026. Not published in a journal according to the arXiv and INSPIRE records
of 17 Nov 2026. Fourth catalog: arXiv:2509.04348v3 (20 Aug 2026), which is the published version,
Astrophys. J. Lett. 1007, L17 (2026 August 10), DOI 10.3847/2041-8213/ae69dd.

**(b) Text and numbers, fifth catalog.**

| version | H0, dark and bright sirens, 68.3% | 90% | dark sirens only, 68.3% (90%) | improvement on the fourth catalog |
|---|---|---|---|---|
| arXiv v2, 4 Aug 2026 (Table 3 and abstract) | 71.7 +9.4 -7.5 | 71.7 +17.7 -11.4 | 68.8 +14.2 -13.2 (+24.7 -21.8) | 22.0% |
| arXiv v1, 26 May 2026, and LIGO-P2600018-v14 | 71.0 +9.0 -7.1 | 71.0 +17.2 -11.0 | 67.6 +13.6 -12.7 (+24.0 -21.8) | 25.7% |

- Both intervals are in Table 3 of either version, row of the fiducial mass model (FullPop-4.0), column
  "H0 (Dark + bright sirens)". Caption: "quoted as the median together with the symmetric 68.3% and 90%
  CI, with the latter in parentheses. The fourth column shows the corresponding constraints after
  combining the dark-siren analysis with the bright-siren posterior of GW170817." The abstract gives the
  68% interval only.
- Number of sources (v2, Introduction): "This allows us to combine the bright siren event GW170817 with
  an additional 235 GW detections used as dark sirens to obtain our final estimate of H0 with a total of
  236 candidates." The abstract says "236 gravitational-wave (GW) sources"; Table 3 lists 235 in the
  dark siren analysis.
- Galaxy catalog (v2, Sec. 4.1): "we choose as our fiducial model the FullPop-4.0 mass distribution,
  augmented by the DES r-band galaxy catalog using epsilon = 1 luminosity weights". The abstract: "galaxy
  catalog information from Dark Energy Survey Year 6 (DES-Y6)". GLADE+ in the K band is the other
  catalog they test.
- Mass distribution (v2, Sec. 4.1; the same sentence is in v1): "With the current set of data, the main
  driver of the Hubble constant measurement remains the presence of features in the mass distribution."
  It is followed by the size of the gain from the catalog: an improvement of 3.5% in the H0 constraint
  over the mass spectrum alone (2% in one pipeline, 8% in the other). Mass spectrum alone:
  70.1 +15.1 -13.3 (68.3%).

**Fourth catalog** (published version). Abstract: "We analyze data from 142 of the 218
gravitational-wave (GW) sources" and, with "the GLADE+ all-sky galaxy catalog", H0 =
75.4 +12.8 -9.1 (75.4 +24.5 -13.3), "reported as a median with 68.3% (90%) symmetric credible interval,
and includes combination with the H0 measurement from GW170817 and its electromagnetic counterpart".
The 142 are 141 dark sirens and GW170817 (Conclusions: "when 141 events with FAR < 0.25 yr^-1 are
analyzed as dark sirens with the GLADE+ galaxy catalog, and combined with the bright siren GW170817").
This version has no HTML at arXiv: the abstract was seen in the PDF and in the arXiv API record, the
Conclusions sentence in the PDF only.

**(c) Verdict: right, with the cause now known.** The draft's numbers (71.7, +17.7 -11.4, 90%, 236
sources, Dark Energy Survey, with GW170817; the sentence on the mass distribution) all agree with arXiv
v2, Table 3 and Sec. 4.1. The "two versions in circulation" are arXiv v1 and v2: the value 71.0
(+9.0, -7.1) is that of v1 and of the collaboration's public document, which still carries the
26 May 2026 PDF and abstract. The fourth-catalog value the reviewer lists (75.4 +24.5 -13.3 at 90%, 142
of 218 sources, GLADE+) is right.

**(d) Safe sentences.**
- The draft's sentence at l. 155-162 can stand as written. Give the version in the reference, for
  example by writing the eprint as 2605.27227v2 or by a note "version 2", because other papers quote
  version 1 (arXiv:2607.03672 quotes 71.0 +9.0 -7.1).
- If the 68% intervals are wanted for comparison: 75.4 +12.8 -9.1 (fourth) and 71.7 +9.4 -7.5 (fifth).

**(e) Not verified.** The science summary of the collaboration was not found and not opened. There is
no published version of the fifth-catalog paper to check against; check again at submission.

---

## F7. Pouw et al.: maximum redshift, AGN density, completeness from the data

**(a) Source.** arXiv:2609.17665v1 (15 Sep 2026), full text.

**(b) Text.**

Maximum redshift, end of Sect. 3 (Method): "Note that f_agn depends on the maximum redshift at which GWs
can be emitted (z_max). Namely, variations in z_max affect the normalisation of our two GW populations
differently, changing the alternative-origin and AGN-origin redshift models in Fig. 3 relative to each
other. We set z_max = 10 in the rest of this work, but we verified that this choice does not
significantly impact our results for values in the range 5 <= z_max <= 10."

AGN density, Sect. 2.2.3 (Completeness estimation): the completeness is "the ratio of the observed AGN
number density (n_obs) to the expected AGN number density (n_exp)" (their Eq. 1), and "The expected AGN
number density is given by a quasar luminosity function (QLF). In particular, we use the Model 1 QLF as
a function of redshift and absolute magnitude at 1450 A, Phi(z, M_1450), from Kulkarni et al. (2019)."
The same function gives the distribution of the AGN-hosted mergers whose host is not in the catalog
(Sect. 3: "We calculate this prior from the observed QLF (see Sect. 2.2.3), neglecting clustering.").
The parameters they infer are f_agn and, in Sect. 6.2, the three parameters of the rate evolution of
the other mergers. No parameter of the AGN density is inferred or varied anywhere in the text.

Completeness from the data, Appendix B, "Biases from the inconsistent handling of EM data", Sect. B.2
(B2 in the PDF), "Measuring the AGN redshift selection function from the data":
- "As a result, EM selection functions are often determined from the data. Even though this is
  fundamentally an inconsistent approach, there are applications where the error introduced is
  negligible."
- "We conclude that, using our method, the completeness estimation of the L_bol >= 10^46.5 erg s^-1 AGN
  population is most at risk of being biased, especially at low redshifts, where there are few of these
  AGN. Fortunately, in our framework the completeness (P_det^EM(z)) is always multiplied by the AGN
  redshift population prior (pi_agn(z))."

**(c) Verdict: one claim wrong, the rest right.** The reviewer writes that "which changes the two
normalizations differently" is the draft's reading attached to their sentence. It is not: the next
sentence of Pouw et al. says that variations in z_max "affect the normalisation of our two GW
populations differently". The draft's l. 981-983 is a fair paraphrase of two consecutive sentences and
needs no separation. Right: z_max = 10; no significant change between 5 and 10; the AGN density from a
published luminosity function, not inferred; Appendix B.2. On the last, the scope should be stated
exactly: the "same data" are the AGN catalog. They estimate the completeness of the AGN catalog from the
catalog itself (observed over expected density), call that inconsistent in principle, and argue that the
bias is negligible for them. The appendix is not about inferring the AGN density from the mergers.

**(d) Safe sentences.**
- Pouw et al. note that their fAGN depends on the maximum redshift of the merger populations, because
  it changes the normalizations of the two populations differently; they set it to 10 and find no
  significant change down to 5.
- They take the number density of AGN from a published quasar luminosity function, estimate the
  completeness of the catalog as the ratio of the observed to that expected density, and note that
  estimating a selection function from the catalog it applies to is inconsistent in principle though
  harmless in their case.

**(e) Not verified.** Nothing. The paper is a preprint; numbers may change on publication.

---

## F8. The merger rate per galaxy in Ref. [15]

**(a) Source.** Mastrogiovanni et al., arXiv:2305.10488v2 (14 Dec 2023), Sec. III.2, "The CBC merger rate
informed by galaxy surveys". Draft Ref. [15] is `Mastrogiovanni:2023emh`, Phys. Rev. D 108, 042002. The
arXiv title differs from the published one.

**(b) Text.** "The first term in Eq. 15 is the CBC rate per galaxy", which they parametrize as
Eq. (16): dN_CBC / (dN_gal dm_s dtheta dt_s) = R*_gal,0 Psi_L(z, M; Lambda_p)
p_pop(m_s, theta | z, M, Lambda_p), "where R*_gal,0 is the CBC rate per year at a reference galaxy
luminosity (or absolute magnitude) L* = L(M*), while Psi_L(z, M; Lambda_p) is an evolution function for
the CBC merger rate per galaxy such that Psi_L(z = 0, M = M*; Lambda_p) = 1."

The next sentences bear on the draft's lead claim: "In Eq. 16 p_pop has been expressed in a general form
assuming that the probability of having some GW parameters might be dependent on the galaxy redshift and
absolute magnitude. For the analysis presented in this paper, we will not consider this possibility and
we will choose p_pop(m_s, theta | z, M, Lambda_p) = p_pop(m_s, theta | Lambda_p)."

Conclusions: "The proposed CBC merger rate parameterization introduces new astrophysical quantities such
as R*_gal,0, the CBC merger rate per galaxy." They infer log10 R*_gal,0 with H0 (their Fig. 6).

**(c) Verdict: right.** Equation number and words are as the reviewer gives them.

**(d) Safe sentences.**
- The merger rate per galaxy is already a parameter of the dark siren likelihood [Ref. 15, their
  Eq. (16)], defined at a reference galaxy luminosity; Eq. (7) of this paper has one such rate for each
  host class.
- The same equation is written with a source population that may depend on the properties of the host
  galaxy, a freedom that Ref. [15] states and does not use. The sentence "None lets the mass or spin
  distribution depend on the host class" should allow for this: the form exists in print, the inference
  does not.

**(e) Not verified.** The equation number in the published Phys. Rev. D version (v2 was posted after
publication and is the only text read).

---

## F9. The three "next step" quotations

**(a) Sources.** Pouw et al., arXiv:2609.17665v1; Cross-Parkin et al., arXiv:2610.25866v1; Bom and
Palmese, arXiv:2307.01330v2.

**(b) Text.**

Pouw et al. The words occur twice.
- Sect. 1: "This unifies the spatial-correlation method with population studies such as Gayathri et al.
  (2023) and Bartos & Haiman (2026), although we defer the inference of intrinsic population
  hyperparameters to future work."
- Sect. 3: "However, we distinguish the GW populations only based on their different spatial
  distributions, and we defer the inference of intrinsic population hyperparameters to future work."
- Their mixture, Eq. (8) and (9), is written with a population prior for each channel; they then assume
  "an identical reference distribution for both the AGN-origin and alternative-origin GWs".
- Sect. 6.3: "We have assumed that the intrinsic GW parameters of AGN-origin and alternative-origin GWs
  are both distributed identically to the entire BBH population. If alternative formation channels
  dominate the BBH population, this may bias f_agn low as our model is tuned to the alternative-origin
  GW population, and we do not model a preference for AGN-channel signatures such as mass-gap or
  high-spin BHs."
- Sect. 6: "This inconsistency demonstrates the necessity of jointly inferring population distributions
  of intrinsic and spatial GW parameters to more robustly characterise candidate AGN-channel GW events."

Cross-Parkin et al., Sec. VI, "Discussion and outlook", last paragraph: "If more than one type of host
association is identified in real data, the corresponding sub-populations would also be expected to
exhibit different distributions of binary mass, mass ratio, effective spin, or spin-precession
parameters. [...] This connection could be tested by dividing the gravitational-wave catalogue according
to source properties or by extending the likelihood to model host weights and compact-binary population
parameters simultaneously."

Bom and Palmese, Sec. IV.1 (IV A): "Future work should focus on the impact of the mass distribution
assumptions on the BBH-AGN formalism used here, as well as considering how to use AGN hosts in concert
with spectral siren analyses as currently proposed with a generic host galaxy population [59, 60]."

**(c) Verdict.** The three quotations in the draft (l. 223-228) are exact substrings of these sentences.
The reviewer's further claim, "None names a population that depends on the host class", is wrong for two
of the three. Cross-Parkin et al. say in the sentence before the quoted one that the sub-populations
with different hosts are expected to differ in mass, mass ratio and spin. Pouw et al. state that they
assume identical intrinsic distributions for the two channels and that they do not model AGN-channel
signatures in mass and spin. Bom and Palmese name the missing correlation between the channel and the
binary properties as a caveat (Sec. IV.3, see F1) and simulate one such case. What none of them does is
infer it. Two cautions on the draft's own wording: Bom and Palmese do not "ask" a question, they list
this under future work; and "named as their next" (l. 1095) says more than the texts, which say "future
work" (Pouw, Bom and Palmese) and "could be tested" (Cross-Parkin).

**(d) Safe sentences.**
- All three leave this to future work: Pouw et al. assume the same mass and spin distributions for both
  channels and defer the inference of the intrinsic population, Cross-Parkin et al. expect host
  sub-populations to differ in mass and spin and propose to model host weights and population
  parameters together, and Bom and Palmese point to the combination of AGN hosts with spectral sirens.
- In the Conclusions: this is the step that these analyses leave to future work. Claim it for the model
  in which the population parameters of each host class are free (the reviewer's point about the
  headline model is about this draft, not about the cited papers, and was not checked here).

**(e) Not verified.** Nothing.

---

## F10. Origin of the association likelihood

**(a) Sources.** Veronesi et al., arXiv:2306.09415v2 (Ref. [31]) and arXiv:2407.21568v2 (Ref. [33]); Zhu
and Chen, arXiv:2505.02924v2 (Ref. [34]).

**(b) Text.**
- Veronesi et al. 2023, Sec. 3.1, Eq. (1): L(f_AGN) is the product over events of
  [c 0.9 f_AGN S_i + (1 - c 0.90 f_AGN) B_i], with c "the average completeness of the AGN catalogue". "The
  general analytical form of the likelihood function used in this work is based on the one described in
  Braun et al. (2008) and first used to draw conclusions on the detectability of a GW-AGN connection by
  Bartos et al. (2017)." And: "The introduction of the c factor is a novelty with respect to previous
  similar works that used only complete mock AGN catalogues (Bartos et al. 2017; Corley et al. 2019;
  Veronesi et al. 2022)."
- Veronesi et al. (Quaia), Sec. 3, Eq. (5): the same with c_i, "the average completeness of the
  catalogue in the region occupied by the 90 per cent CL localisation volume of the i-th GW event".
- Zhu and Chen, Eq. (1): the same form with c_i; "We adopt the method proposed by Bartos et al. (2017a)".
- Pouw et al., Sect. 3 and Appendix A.2, recover "the functional form of the likelihood presented in
  Veronesi et al. (2023)" from their own.

**(c) Verdict: right.** The product of fAGN and the completeness first appears in Veronesi et al. 2023,
by their own statement; the signal-and-background form without it is that of Bartos et al. 2017 (draft
Ref. [30]).

**(d) Safe sentence.** In the likelihood of the association searches, fAGN multiplies the completeness
of the AGN catalog [Veronesi 2023, Veronesi Quaia, Zhu and Chen]; the form goes back to Bartos et al.
[30], and the completeness factor was added by Veronesi et al. [31].

**(e) Not verified.** Bartos et al. 2017 itself was not opened; its role is as Veronesi et al. and Zhu
and Chen describe it.

---

## F11. Bibliography

**(a) Sources.** Crossref records by DOI, INSPIRE records by arXiv number, arXiv API records, and the
journal page of the Open Journal of Astrophysics.

**(b) What the records say.**

| key (draft Ref.) | record | result |
|---|---|---|
| `Powell:2018` ([62]) | Crossref, DOI 10.3847/1538-4357/aabd7f: The Astrophysical Journal 858 (2), 110, 2018. Journal title: "The Swift/BAT AGN Spectroscopic Survey. IX. The Clustering Environments of an Unbiased Sample of Local AGNs" | reviewer right |
| `Tessore:2023zyk` ([57]) | Crossref and the journal page (`astro.theoj.org/article/73838`): The Open Journal of Astrophysics, Vol. 6, 23 March 2023. The journal record has no page or article number | reviewer right: published |
| `LIGOScientific:2025pvj` ([83]) | INSPIRE lists the collaborations as 'LIGO Scientific', 'VIRGO', 'Virgo,', 'KAGRA' (a fault of the record). Crossref: Astrophys. J. Lett. 1005 (2), L51, 2026 | collaboration field corrected; journal fields were right |
| `LIGOScientific:2026uyd` ([20]) | INSPIRE: 'LIGO Scientific', 'VIRGO', 'KAGRA'; no journal | collaboration field corrected; stays a preprint |
| `Kalomenopoulos:2025qpt` | Crossref, DOI 10.1093/mnras/stag1662: Monthly Notices of the Royal Astronomical Society 552 (1), article stag1662, online 3 Sep 2026 | published; the reviewer's "nine preprints, right as they are" is wrong for this one |
| `Vitale:2020aaz` | Crossref: chapter of "Handbook of Gravitational Wave Astronomy", Springer Singapore, online 2 Mar 2022; arXiv journal reference "Chapter in: Handbook of Gravitational Wave Astronomy, Springer (2022)" | a book chapter, not a journal article; not changed |
| `Essick:2022ojx` | arXiv (v1 only) and INSPIRE: no journal, no DOI | preprint; right as it is |
| the other eight entries without a journal | not looked up one by one; `Pouw:2026lar` and `Cross-Parkin:2026pua` have no journal at INSPIRE or arXiv | unchanged |

**(c) Verdict.** The reviewer is right on Refs. [62], [57], [83] and [20]. Of the thirteen entries
without a journal, ten are preprints with no journal record (if the eight not looked up are as the
reviewer says), and one of the nine the reviewer calls preprints is now published.

**(d) Changes made in `refs.bib`** (the build of a test document with `tectonic` and the edited file
gives no BibTeX error; nothing was removed):
- `Powell:2018`: now an article entry with journal "Astrophys. J.", volume 858, number 2, pages 110;
  title as in the journal. Prints as "Astrophys. J. 858, 110 (2018)".
- `Tessore:2023zyk`: now an article entry with journal "Open J. Astrophys.", volume 6, year 2023, no
  pages. With no page number the style prints the DOI in its place: "Open J. Astrophys. 6,
  10.21105/astro.2302.01942 (2023)".
- `Kalomenopoulos:2025qpt`: now an article entry, "Mon. Not. Roy. Astron. Soc. 552, stag1662 (2026)".
  The key keeps its year.
- `LIGOScientific:2025pvj` and `LIGOScientific:2026uyd`: collaboration = "LIGO Scientific, Virgo, KAGRA",
  as in `LIGOScientific:2025jau`.
- Added `Kumar:2026bqt` (arXiv:2607.03672, preprint) and `Moncrieff:2025xle` (arXiv:2508.18704, MNRAS
  543, 1833-1841, 2025). The keys are the INSPIRE keys, as for the rest of the file. Neither is cited in
  `main.tex` yet.
- A comment at the head of the file records these changes. `refs_candidates.bib` was not touched and
  still has the old forms.

Not applied, for the owner to decide: `Vitale:2020aaz` could become a chapter entry,

```
@incollection{Vitale:2020aaz,
    author = "Vitale, Salvatore and Gerosa, Davide and Farr, Will M. and Taylor, Stephen R.",
    title = "{Inferring the properties of a population of compact binaries in presence of selection effects}",
    booktitle = "{Handbook of Gravitational Wave Astronomy}",
    publisher = "Springer",
    address = "Singapore",
    eprint = "2007.05579",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    doi = "10.1007/978-981-15-4702-7\_45-1",
    year = "2022"
}
```

This form was not test-built, and the editors and the page range of the printed book are not in the
Crossref record, so they are left out.

**(e) Not verified.** An article number for `Tessore:2023zyk` (some indexes give one; the journal does
not). The eight remaining preprints were not looked up one by one for a journal.

---

## Statements of the reviewer's "Verified and correct" paragraph seen while reading

These were not the task, but the texts were open. Confirmed in the full text: for Bom and Palmese, the
joint H0 and AGN fraction from a static AGN catalog (Sec. III.2), the three quotations, the density
10^-4.75 Mpc^-3, selection effects disregarded, a fixed density carrying cosmological information
(Sec. IV.1). For Cross-Parkin et al., "only a weak correlation" (Sec. IV), a complete catalog assumed,
selection effects included in an approximate form (Sec. II.5); that their population is held fixed was
not checked. For Pouw et al., the fixed Planck 2015
cosmology (H0 = 67.9), the other mergers uniform in comoving volume (Eq. 11), AGN redshift errors below
about 0.01 (Sect. 4.2.2), the limits 0.52, 0.41 and 0.61 at 95% for unobscured AGN, the disagreement
with the spin-based ranking and "the necessity of jointly inferring". For the Quaia paper of Veronesi et
al., an upper limit of 0.21 at 95% for L_bol above 10^44.5 erg/s (their table of upper limits). For Zhu
and Chen, 0.39 +0.41 -0.32 at the 90% level (abstract). For Zhu et al. 2024, the degeneracy by name. The
remaining entries of that paragraph (Bartos et al. 2017, Borghi et al., Scarpa et al., Bartos and
Haiman, Li et al., Gayathri et al., Kalomenopoulos et al.) were not opened.
