# Independent review: PRD draft of 17 Nov 2026 (`main.pdf`, 18 pages)

Reviewer had no part in the draft. Nothing was edited; this file is the only output.

**Read.** `main.pdf` in full before anything else (first impressions below), then `main.tex`, the rendered
figures (`figures/*.png`) with `figures/CAPTION_FACTS.md`, `values/results_macros.tex`, `values/MACROS.md`,
`values/values.json` (all 505 macro records, the Bayes-factor ledger and the diagnostics), `notes/LIKELIHOOD_MATH.md`,
`notes/DESIGN.md`, `notes/GAPS.md`, `notes/NOVELTY.md`, `notes/CITATION_CHECKS.md`, `notes/MACRO_REQUESTS.md`,
`placeholders.txt`, `runs.txt`, the owner's `astro-paper-style` skill with `llm_tells.md`, the voice brief of
2 Oct, and the sibling `paper1/REVIEW.md` with the abstract and layout of `paper1/main.pdf`.

**Checked by script.** All 232 macros used in `main.tex` against `values.json`: the macro file and the JSON
agree on every rendered value, and every median and 90% interval re-rounds to the printed string (0 mismatches).
Every containment record was compared with the sentence that uses the macro (part C). Section word counts,
display-equation counts and a banned-phrase count were taken from the source and from `pdftotext main.pdf`.
One toy calculation was run for item B8 (the priors of Table I pushed through Eq. (6); described there).

**Opened on the web** (arXiv numbers only in the requests): 2307.01330 (abstract and full text v2), 2610.25866
(abstract and full text v1), 2609.17665 (abstract and full text v1, all three parts), 2605.27227 (abstract v2),
the public LVK record and science summary of the same paper, 2509.04348, 2403.04950 (abstract and full text v2),
2505.02924, 2407.21568, 2306.09415, 1701.02328, 2509.18243, 2603.15332, 2605.09351, 2509.23897, 2104.10253,
2511.12334, 2508.18704, 2607.03672 (abstracts), 2305.10488 (full text). Two topic searches for work newer than
`notes/NOVELTY.md` that joins H0, an AGN catalog and host-dependent populations found nothing beyond the papers
already listed there. Quotations from these papers came through a fetch that summarizes pages. The ones the
draft already prints were confirmed that way; quotations that are new in this review (F1, F2, F7) should be
checked against the PDFs before they are used.

**Not checked.** The 90% interval and the "main driver" sentence of 2605.27227 (the HTML is cut inside the
author list; see F6). The sentence that `notes/NOVELTY.md` attributes to 2306.09415 on completeness (not in its
abstract). 2609.04991 and 2609.38110 were not opened. No analysis result file was opened and `build_values.py`
was not rerun. There is no `main.log` in the directory, so overfull boxes were not counted. Nothing under
`working/archive/` was opened.

## First impressions, from the PDF alone

1. The abstract is a list of about twenty numbers. After one reading I could not say which one is the result,
   and two of them ("factors of 1.07 and 1.06") cannot be told apart.
2. Page 6: the bias of the single population is stated, explained and compared with the literature, and only
   then does a sentence say that this population was held fixed at the galaxy values. The comparison a referee
   wants is two red numbers.
3. Page 5: nine tenths of the mergers sit in the Gaussian mass peak. Nothing in the text says how unusual
   that is or what it does to the H0 numbers.
4. Page 10: at "100.0%" completeness fAGN still moves by 0.43 per decade of the assumed AGN density. I had to
   read Sec. IV C twice to see that the main number of the last third of the paper follows from normalizing the
   hosts out to z = 5, a choice introduced on page 3 with a forward reference.
5. Page 7: three of the eight inputs of the full model lie outside their 90% intervals, followed by
   "This is one simulated catalog."
6. Fig. 3(a): the orange posteriors scatter by more than their own widths. Fig. 8(b): the input sits on the
   edge of the shaded interval.
7. Fig. 7(b) does not look like "left to the prior": the posterior is three times the prior line at its peak
   and falls to zero at the top of the range.
8. "Catalog" means two different things ("both catalogs", "9 catalogs"), and "the fraction of mergers that
   AGN host" stops the eye every time.
9. Every caption ends "Drawn from the exploratory runs." Section V is mostly red.
10. The text ends on page 13; the reference list fills more than four pages and Fig. 9 a fifth.

---

## (A) Verdict

1. The likelihood is sound and the draft states it as the code evaluates it: Eqs. (1) to (8) and (A1) to (B3)
   agree with `notes/LIKELIHOOD_MATH.md` apart from the small omissions of item B12.
2. There is a paper here. Two host classes on one density field, mass and spin distributions that depend on
   the host class, and the statement that mergers measure a rate per host are three things no published
   analysis has together, and the literature search confirms that.
3. The strongest part is the identity of Sec. IV C read as Eq. (7), the odds that a merger is in an AGN equal
   the ratio of rates per host times the ratio of numbers of hosts, together with the spin result (the
   catalogs narrow the spin offset by a factor 2.2 while the spins leave fAGN alone) and the mock itself
   (exact event posteriors, detection decided on observed data).
4. The weakest part is the lead result. The H0 bias is that of a population held fixed at the galaxy values,
   in a mass model with 90% of the mergers in the peak, on one simulated catalog; the model that removes it has
   its galaxy-class population fixed at the truth; and the model a real analysis would use misses three of
   eight inputs at 90%.
5. The second half overstates its own result. The response of fAGN to the assumed AGN density is the same at
   every depth because the hosts are normalized out to z = 5 and because the number density is treated as
   unknown where the catalog measures it. The conclusions then say that depth is not what limits the
   measurement, which Sec. IV C itself denies.
6. The numerical accuracy is not at the standard the draft cites: a variance of ln L of 8 to 11 (far more for
   AGN-hosted events) against a limit of one, with visible consequences in Figs. 3(a) and 8(b).
7. I would not send it to a referee as it stands, also with every placeholder filled, because the filled
   placeholders can change the headline (items B1, B3, B4). With the fair single-population comparison, the
   numerics settled, and the normalization tied to the catalogs, it is a good PRD paper of 14 to 15 pages.

---

## (B) Physics and logic

The three objections I would expect first from a hostile referee:

- **Objection 1.** "Your bias and its cure are properties of populations you fixed at the truth, in a mass model
  with 90% of the mergers in one Gaussian peak, on one realization. Show the single population with free
  parameters, H0 from the mass distribution alone, and ten realizations."
- **Objection 2.** "The dependence of fAGN on the AGN number density is produced by normalizing the hosts out to
  z = 5 and by treating the density as unknown when the catalog counts it. With a catalog that is 100% complete
  where the events are, you still report 0.43 per decade."
- **Objection 3.** "Your likelihood estimator has a variance of 8 to 11, and much more for AGN-hosted events,
  against the limit of 1 that you cite. The AGN-hosted posteriors scatter by more than their widths, the
  93-catalog combination excludes the input at 90%, and three of eight parameters of the full model miss their
  inputs. Nothing is established until the numerics are."

**B1 (must-fix). The single population of the lead claim is fixed, and so is the reference of the model that
cures it.** Abstract (l. 115-121), Sec. IV B first paragraph (l. 699-719), Table III, Conclusions (l. 1087-1092).
The number 70.6 (+2.3, -1.7) and the shift of 2.93 km/s/Mpc come from a common population whose mass peak and
spin mean are held at the galaxy-class inputs (said only at l. 715-716, after the result has been explained and
anchored to five references). No dark siren analysis holds its mass spectrum fixed, and Pierra et al. attribute
biases of this kind to models that cannot adjust. The curing model (67.7, offsets 4.80 and 0.118, log10 B = 28.3,
all in the abstract) has its galaxy-class population fixed at the true values, which no analysis can do either.
The fair pair, a common population with free peak, width, spin mean and spin width against the eight-parameter
model, is three red macros (`\CommonFreeHzero`, `\CommonFreefAGN`, `\LBHDVsCommonFree`).
Also: "the host catalogs do not protect it" (l. 700) is not what the numbers show. With both catalogs the shift
is 2.93; with no host information it is at least 20.1 (l. 742). The catalogs remove most of it.
Fix: run R8e and R8d on one code and one prior, on several simulated catalogs, with the H0 prior of Table I
([50, 100], not [60, 76]); lead the abstract, Sec. IV B and Table III with those two models; keep the fixed
population as one clause. Replacement for l. 699-703: "If mergers in AGN are heavier, a single population cannot
place its mass peak for both host classes, and H0 follows the error (Fig. 4, Table III). With its mass and spin
parameters free the single population gives H0 = XX; held at the galaxy-class values it gives 70.6 (+2.3, -1.7),
with the input 67.74 at the 99% credible level." Replace "do not protect it" by "reduce the bias and do not
remove it".

**B2 (must-fix). The mass model makes the mass scale carry H0, and the draft never says how much.** Sec. III B
(l. 479-482), Sec. IV A, Sec. IV B, Conclusions (l. 1111-1115). A Gaussian peak at 35 Msun of width 5 holds 0.90
of the mergers, and in Secs. IV A and IV C every population parameter is held at its input (Table I caption).
Fits of this model to the observed mergers put a few percent of them in the peak (from memory, 0.04 in the third
catalog; check before quoting). The consequence can be read off the draft's own numbers: a 30% subpopulation
5 Msun heavier shifts the mean mass by 0.3 x 5/35 = 4.3%; at the median redshift 0.13 a fractional error in
1 + z is multiplied by (1 + z)/z = 8.7 in H0; 4.3% x 8.7 = 37%, or 25 km/s/Mpc, which is what the mock with
galaxy hosts only returns (at least 20.1, cut by the prior at 100). So in that mock the galaxy catalog adds
almost nothing to H0 and the analysis is a spectral siren with a known, very sharp mass scale. That reading
changes how three results should be stated: the "factor 4.7" of Sec. IV A compares an informative AGN catalog
with a galaxy catalog that may be nearly uninformative; "H0 hardly notices" the missing AGN catalog (l. 637-647)
because the mass scale, common to both host classes there, carries it; and the size of the bias and of
log10 B = 28.3 scale with the peak fraction.
Ask for: H0 from the mass distribution alone (hosts uniform in comoving volume) on the same events, as a row of
Table II and a curve of Fig. 3(b); one sentence in Sec. III B that the peak fraction is a design choice that makes
the mass peak a sharp ruler, with the value fits to the observed mergers give; and at least one simulated catalog
with a peak fraction near the observed one, or the statement that the bias scales with it and by how much.

**B3 (must-fix). What limits fAGN: the normalization volume, and a number density the catalog already
measures.** Sec. II B (l. 362-364), Sec. IV C (l. 945-965), abstract (l. 129-133), Conclusions (l. 1097-1105).
(a) The algebra of Eqs. (6) to (8) is correct (B12). The number built on it is not general. With z_max = 5 the
cataloged share is F_AGN < 0.012 at every flux limit (`notes/LIKELIHOOD_MATH.md` Sec. 4), so Eq. (8) is at 1
whatever the depth, including the limit m < 21 at which the catalog is 100.0% complete where the events are.
`notes/LIKELIHOOD_MATH.md` Sec. 8.4 and `notes/GAPS.md` A6 call z_max = 5 a choice made for numerics and "not
what a paper should defend"; the draft defends it and puts its consequence (0.13 for a factor of two) in the
abstract. With the hosts normalized over the volume the events reach, Eq. (8) predicts, for f(1 - f) = 0.21,
slopes of 0, 0.09, 0.33 and 0.44 per decade at the four limits (my arithmetic: ln 10 x 0.21 x (1 - F) with F the
completeness inside z = 0.31). That is a result that depends on depth, the opposite of Fig. 6(b).
(b) The analysis fits the Schechter function to the catalogs to get C_k(z) (l. 566-568) and then treats n_k as an
outside assumption. The same catalog counts n_k where it is complete: the shallowest AGN catalog holds about
4500 objects below z = 0.11, which fixes n_AGN to about 2%. In the mock's own universe (constant comoving
density, constant rate per host) a density "off by a factor of two" contradicts the catalog it is used with. The
uncertainty that is real is in the completeness model (the luminosity function below the limit, which objects
count as hosts), and in host classes that are complete nowhere, such as obscured or faint AGN.
(c) The Conclusions say "What limits the measurement is not the depth of the catalogs" (l. 1097). Sec. IV C says
"The change is the same at every depth because the normalization volume is the same, not because completeness
is of no help" (l. 958-960), and `notes/DESIGN.md` L2 forbids the general statement. The two sentences cannot
both stand.
Fix: (i) normalize the hosts over the catalog depth or the volume the events reach, and take n_k from the
catalog counts where complete (runs R5b and the check listed under "Checks before production"); (ii) keep
Eq. (7) as the result and state its consequence as Eq. (8) with F_AGN a function of depth, with the measured
slopes against the prediction in Fig. 6(b); (iii) replace l. 1097-1103 by: "What limits fAGN is the number of
AGN in the volume the fraction refers to. The mergers measure the merger rate per AGN relative to that per
galaxy [Eq. (7)]. Where a catalog is complete it supplies the number of hosts itself; beyond that, fAGN moves
with the assumed number as 1 - F_AGN, which is XX for a factor of two at the shallowest limit."; (iv) keep the
abstract's 0.13 only if it survives (i), and say for which volume.

**B4 (must-fix). Monte Carlo accuracy.** Appendix A (l. 1177-1189), Sec. IV A (l. 692-694), Sec. V. The draft
reports a variance of ln L of 8.1 to 11.0 for the two-catalog analyses, a red number for the AGN-hosted sets
(`notes/GAPS.md` A1: about 55), a cap of 20 for the eight-parameter model, and in the next sentence the limit of
one used for the observed catalogs. A referee needs nothing else. The effects are visible. In Fig. 3(a) the
AGN-hosted posteriors have peaks spread from about 66.5 to 69.5 km/s/Mpc while each is about 1.5 wide at 90%
(`notes/GAPS.md` A2 measures the scatter of the medians at 1.3 to 1.4 times the quoted width). In Fig. 8(b) the
combined posterior of 93 such sets puts the input at the 91% level. The nine combined give fAGN = 0.283 with the
input at the 70% level, which `notes/GAPS.md` A1 traces to the estimator (0.297 after a first-order correction),
and on catalog A every fAGN median obtained with known densities lies below the input (0.262 to 0.287). Catalog F
has a two-lobed region in Fig. 2 and catalog I has the input fAGN at the 99.6% level; whether these are the same
effect is not known. None of the Bayes factors carries this error (D6).
Fix: run R0 before any other run; choose the number of posterior samples per event so that the variance is at
most one, or state the measured effect on medians and widths per result; until then the factor 4.7, the level
91%, and any statement that fAGN is recovered are not supported. The sentence at l. 692-694 belongs in
Appendix A, not in Results.

**B5 (must-fix). The full model does not contain its inputs, and the draft offers no reason.** Sec. IV B
(l. 807-825), Fig. 5(c), Fig. 9. In the eight-parameter model the inputs of the spin offset, the spin width
and the reference peak lie at the 99.3%, 96.7% and 90.9% levels, and the reference spin mean is at the edge
(upper end 0.003 for an input of 0). The offsets drift as freedom is added: spin offset 0.118, 0.131, 0.158 and
mass offset 4.80, 3.9, 4.2 for an input of 0.10 and 5. Fig. 9 shows the cause to test first: the spin offset and
the spin width are anticorrelated, so a narrower shared width with a larger offset fits the same mixture.
"This is one simulated catalog" is not an explanation, and the next sentence presumes that a combination will
settle it. The three curves of Fig. 5(c) also come from different prior ranges for the spin offset (Table I,
footnote b; `figures/CAPTION_FACTS.md` says all three differ), which a reader cannot see.
Fix: the combined posterior over several simulated catalogs (`\HDEight...CombLevel`) is a condition for the
claim "recovers", not a validation detail; run all three models with one prior range; if the combined level is
above 99% find the cause before writing. Until then replace the section's opening claim (l. 753-754) by what is
shown: "With the galaxy-class population held at its input, a mass peak and a spin mean for the AGN class
return H0, fAGN and both offsets inside their 90% intervals."

**B6 (should-fix). "A consequence of the shared density field" is asserted, not shown.** Sec. IV A
(l. 645-647). The galaxy catalog alone gives 68.9 (+1.3, -1.3) against 68.36 (+0.63, -0.63) with both. The
draft reads this as the galaxy catalog marking where the AGN-hosted mergers are. By B2 the other reading is that
the fixed mass scale, the same for both host classes in this mock, carries H0 and the catalog matters little.
The two are separated by one run each: no catalog at all, and a galaxy catalog drawn from an independent
density field. Without them write the reading as one of two and name the other.

**B7 (should-fix). The reason given for the factor 4.7 rests on two red numbers and an unseparated bias.**
Sec. IV A (l. 674-694), Sec. III A (l. 451-453). "We read the gain as a matter of the number of candidate
hosts" is followed by `\PureGalHostsPerVol` and `\PureAGNHostsPerVol`, both red, and the draft says itself that
the number density and the bias of 2.0 were not separated. Ask for the two counts, and for one set of AGN-hosted
mergers on a catalog with the AGN density and the galaxy bias (or the reverse), so that the split is a number.

**B8 (must-fix). With both densities free, most of what Table IV and Fig. 7 show is the prior.** Sec. IV C
(l. 988-1011), Table IV (last four columns), Fig. 7. By the draft's own identity the likelihood depends on
fAGN, n_gal and n_AGN only through f-tilde wherever no event lies in the incomplete part of a catalog. At
m < 21 that is everywhere, so the "inferred" log10 n_gal = -3.20 (+1.09, -0.73) and log10 n_AGN = -4.85
(+0.75, -0.94) of Table IV cannot be measurements. Check: drawing fAGN, log10 n_gal and log10 n_AGN from the
priors of Table I and weighting by a Gaussian in f-tilde of median 0.273 and standard deviation 0.047 (Eq. (6),
F at 0.0115 for the input densities) gives log10 n_gal = -3.10 (+1.32, -0.80), log10 n_AGN = -4.78 (+0.70, -1.00),
fAGN = 0.37 (+0.47, -0.32), and correlations of fAGN with the two densities of -0.60 and +0.38. Table IV has
-3.20 (+1.09, -0.73), -4.85 (+0.75, -0.94), 0.39 (+0.45, -0.33), and the macro menu has -0.56 and +0.47. The row
is the prior boxes seen through the degeneracy, and the agreement of -3.2 with the input -3.0 is a coincidence
of the ranges U[0, 1] and U[-6, -4].
What the mergers add at m < 18 is an upper edge: the 95% point of log10 n_gal moves from about -1.8 to -2.83, and
that of log10 n_AGN from about -4.1 to -4.40. The lower ends (-3.80, -5.54) are the prior. So (i) the sentence
"the mergers bound the galaxy density and leave the AGN density, and with it fAGN, to the prior" is right in
spirit, but Fig. 7(b) is compared with the wrong prior line (flat at 0.5, where the induced density rises
from 0.2 to 0.66 per decade), and the AGN density is bounded above as well; (ii) fAGN is as strongly tied to the
galaxy density (correlation -0.94, `\FreeRhofGalEighteen`) as to the AGN density (+0.90), and the text gives only
the second; what the identity leaves free is the ratio n_AGN/n_gal; (iii) "with the input inside all three 90%
intervals" (l. 993-994) presents a one-sided bound as a recovered value.
Fix: draw the prior-induced densities in Fig. 7(a) and (b); quote 90% upper limits on the two densities with the
induced limit beside them; drop the m < 21 and m < 20 entries of the last four columns of Table IV or mark them
"prior"; replace l. 988-990 by "With both number densities free, fAGN follows the ratio of the two densities and
is left to its prior, and the mergers place an upper limit on each density (Fig. 7)."

**B9 (should-fix). The hinge sentence of the introduction is contradicted by the results.** Sec. I (l. 204-205):
"In a mixed population neither H0 nor fAGN can be inferred alone." The paper finds a correlation coefficient of
0.11, fAGN = 0.265 from positions alone at fixed H0, fAGN = 0.287 under the biased single population, and an H0
that "hardly notices" the missing AGN catalog. Each can be inferred alone when the populations are equal; what
fails is H0 when the masses differ. Replacement: "A measurement of H0 from a mixed population then depends on
which mergers are assigned to which host class."

**B10 (should-fix). "9.5% completeness" overstates the loss, and the reasons for the small cost are half
red.** Sec. III D (l. 558-564), Sec. IV C (l. 852-869), abstract (l. 126-129). The percentage counts hosts inside
z = 0.31, the redshift of the single most distant detected merger of catalog A, where the volume is and the
mergers are not. The shallowest catalog is complete to z = 0.11 and the median merger is at 0.13
(`notes/GAPS.md` A6). The first of the "two things" (l. 860-862) is the three red macros of Sec. III D; the second
is that the model is given the true density. A reader from the association literature expects the opposite
result (F4) and will look for exactly this.
Fix: name each limit by its magnitude and by the redshift to which the catalog is complete; give the
completeness weighted by the detected mergers as the number that describes the loss; fill the three macros
before the sentence "The mergers are nearer than the hosts that the flux limit removes" is kept, since for about
half of them it is not true.

**B11 (should-fix). The spin result needs its conditions and has a simpler statement.** Sec. IV B
(l. 827-847), abstract (l. 122-123). The factor 2.2 is at fixed H0, on one simulated catalog, for an offset equal
to one width, and with a posterior from spins alone that is cut on both sides. The lower cut is not explained:
in Fig. 5(d) the blue curve rises from zero at about 0.02 to its peak at 0.035. My reading, to be confirmed on
the stored grid: spins alone fix the mean spin of the whole population, fAGN times the offset, about 0.03, and
the sharp edge is fAGN = 1 (`\SpinIntfAGN` is 0.31 (+0.53, -0.20)). If so, say it: "From spins alone the mergers
measure the product of fAGN and the offset; the catalogs supply fAGN and the offset follows." The abstract's
"measure the offset 2.2 times better" should read "narrow the 90% interval of the offset by a factor of 2.2 at
fixed H0".

**B12 (should-fix). Equations against `notes/LIKELIHOOD_MATH.md`, symbol by symbol.**
Agree: Eq. (1) (the factor 1/(1+z), both densities normalized, the same mixture in the detected fraction);
Eq. (2) (rate marginalized with a 1/R prior); Eq. (3); Eq. (4) and (5) (one normalization per catalog over the
sky and 0 < z < z_max, uncataloged hosts uniform); Eq. (6) (LIKELIHOOD_MATH Sec. 8.2); Eq. (7) (follows from
Eq. (6) with r_k proportional to the share of class k among mergers in cataloged hosts divided by N_k); Eq. (8)
(the derivative of ln Z_AGN with respect to ln n_AGN is 1 - F_AGN); Eq. (A1), (A2), (A3) (the correction
N(N+3)/2N_eff); Eq. (B1) to (B3) (the Jacobian of the flat prior is rho/(d_L m_1z q)).
Differences to repair:
- Eq. (A3), l. 1160: w_ij is written as pi(x_ij)/p_PE(x_ij) with pi in (theta, z, Omega) and p_PE in
  (m_1z, q, d_L, chi_eff). The code divides pi by the Jacobian (1+z) d d_L/dz between the two sets of variables
  (LIKELIHOOD_MATH Sec. 1). Add: "with pi expressed in the variables of the samples, that is divided by
  (1+z) d d_L/dz". The same holds for p_draw.
- Appendix B, l. 1220-1221: "flat ... in the sky position". The events are flat in right ascension and
  declination, not per unit solid angle, and the weights do not divide the resulting factor out (LIKELIHOOD_MATH
  Sec. 1, marked inference). Say so and give the size (a factor common to both host classes and constant to
  within the 1 to 2.4 degree localization), since Sec. III B promises an exact posterior.
- Eq. (1), l. 284-286: "the fraction of mergers ... whose host is an AGN" holds over 0 < z < z_max. Say it here;
  it is the point of Sec. IV C.
- l. 910: the first term of Eq. (4) is called c_k, one letter case away from the completeness C_k of the same
  equation. Rename it.
- r_AGN and r_gal (l. 925-926) are defined in words only. Give the one-line definition.
- l. 1175: the requirement N_eff > 5N has no reference (Farr gives 4N).
- Notes, not the paper: `notes/LIKELIHOOD_MATH.md` Sec. 0 gives the prior of log10 n_gal as [-4, -2];
  `values.json` and Table I have [-4, -1]. The same notes and the stored summary give 99.7% for the m < 21
  catalog where the paper prints 100.0% (C9).

**B13 (should-fix). Two oracle comparisons should go.** Sec. IV B (l. 813-816): "Freeing the reference is not
rewarded, since the fixed one sits at its input", with log10 B = -1.12 +/- 0.08. A model fixed at the truth
always wins this; the number tells the reader nothing about the mergers. Table III, "Offsets fixed" rows and
Fig. 4(b), solid curve: offsets held at (5.0, 0.115), where 0.115 is the maximum of the same data at fixed H0
(CAPTION_FACTS, Fig. 4), and at the inputs for the galaxy-hosts-only mock. Replace both by runs with the
offsets free, or remove the rows and the curve.

**B14 (should-fix). The prior on H0 changes in mid-paper and decides one result.** Table I footnote a,
Sec. IV B (l. 736-739), Fig. 4(b). The host-dependent models use U[60, 76]. The statement "its density is largest
at 76, the upper end of the range" is then about the prior, and a referee will ask where the posterior goes on
[50, 100]. Run it there.

**B15 (optional). Eqs. (6) and (7) are one statement.** Keep Eq. (7), which is the physics, and Eq. (8); give
f-tilde in a sentence. Cite the merger rate per galaxy of Ref. [15] at Eq. (7) (F8).

---

## (C) Numbers

**Checked and clean.** No number in the rendered PDF differs from `values.json` in value,
units or rounding of the stored quantity. The items below are about what the sentences say with the numbers,
about quantities printed under the wrong heading, and about precision.

**C1 (must-fix). Table III, row "Two populations, offsets fixed", column fAGN.** The entry 0.281 (+0.065, -0.064)
is `\CtrlModfBranch`, the fraction of mergers in the heavier population of a mock in which no merger is hosted by
an AGN. It is printed under fAGN. Give it its own symbol or a footnote.

**C2 (must-fix). "the input 67.74 lies just below the lower end of the 90% interval, 67.741" (l. 1034-1036,
Fig. 8 caption facts).** The combined posterior of the AGN-hosted sets has a standard deviation of 0.048 km/s/Mpc
and is built from likelihood curves stored every 0.25 km/s/Mpc. The diagnostics give 67.7419 for one interpolant
and 67.7414 for another (`c3_agn_combined_interpolant_check`), and the first-order estimator correction of
`notes/GAPS.md` A1 moves the whole posterior by 0.09. Whether the input is inside or outside is decided in the
fourth decimal. Write: "with the input at the 91% credible level, at the edge of the 90% interval", print
67.82 (+0.08, -0.08), and tie the sentence to the accuracy statement of Appendix A.

**C3 (should-fix). "both inputs inside the 90% intervals" for the nine combined (abstract l. 125-126,
Conclusions l. 1086-1087), "recovered together" (l. 581-582).** Containment record: H0 in the 90% and not in
the 68%, level 0.895, lower end 67.73 against 67.74; fAGN in the 90% and not in the 68%, level 0.70. The
statement is true by 0.01 km/s/Mpc. Give the levels, as Sec. IV A does: "with the inputs at the 89% and 70%
credible levels". The owner's rule asks for "inside the 90% and outside the 68%" wherever that is the case.

**C4 (should-fix). "the four inputs lie inside their 90% intervals" (l. 757-758).** True. The spin offset is
outside the 68% interval (level 0.78). Add the clause. The same holds at l. 810-811, where the offsets of the
six- and eight-parameter models are left out of the sentence and come seven lines later.

**C5 (should-fix). "1.01 times as wide" (l. 761-762).** The macro (1.009) is computed from unrounded intervals
on a common H0 range. The intervals printed two lines above give 3.9/4.0 = 0.98. A reader who checks gets a
different number for an effect that cannot be seen. Write "as wide as" and drop the number.

**C6 (should-fix). Precision the estimator does not have.** log10 B = 2037.9 and 101.2 (l. 639, 657; the
unrefined grid gives 2037.2 for the first), 67.820 (+0.080, -0.079), 4.80 (+1.08, -0.98). Mixed decimals in one
column: Table II log10 B (11.1, 5.97, 9.45, 1.82), Table III mass offset (4.80, 3.9, 4.2). `\LBFreeRefVsCommon`
prints "27.2 +/- 0.08". Use one decimal for every log10 B (27.2 +/- 0.1; "above 2000"), and one rule per column.

**C7 (should-fix). Intervals quoted for posteriors that end at a prior edge.** log10 n_gal at every limit
(lower ends at the prior, B8); fAGN with free densities (0.50 (+0.40, -0.42) covers 0.82 of the range); log10
n_AGN; `\CtrlUnmodHzero` 88.9 (+4.7, -4.2) in Table III (the text says "at least", the table gives an interval);
`\SpinIntDmuChi` 0.082 (+0.091, -0.052) (cut on both sides, B11). The owner's rule: no interval and no width for
such a posterior; give the limit or the median against the boundary.

**C8 (should-fix). Numbers typed by hand.** No result number is typed in the body. The preamble types 37 values
through `\req{}` (l. 56-93). Most are design choices that belong in `build_values.py`. Two groups are not design
choices: `\PinDmuChi` = 0.115 (l. 93), a maximum of the data; and the prior ranges, which differ between runs
(H0 [50, 100] and [60, 76]; spin offset upper end 0.30, 0.20 and 0.25). "0.90 for the prior itself" (l. 1005) and
the range [0, 1] of fAGN (l. 391, 614, 640) are typed as well.

**C9 (should-fix). "100.0%" at m < 21.** The mock metadata give 100.0% inside z = 0.31; the stored summary of
the free-density runs and `notes/LIKELIHOOD_MATH.md` give 99.7%. Settle which, and say in Sec. III D that this
catalog is complete where the events are (first incomplete at z = 0.37), since a flux-limited catalog that is
100.0% complete makes the reader stop.

**C10 (should-fix). Counts that are accidents of the runs.** 93 catalogs (the seeds 100 to 194 without 104 and
121) and 9 catalogs A to I (the seeds that had a two-catalog grid). "For 93 catalogs we also draw" (l. 516) reads
as a record. Production should use designed numbers.

**C11 (must-fix for a, b, c; should-fix for the rest). Placeholders whose sentence already presumes the
answer.**
a. l. 716-719, `\CommonFreeHzero`, `\LBHDVsCommonFree`: "can settle between the two classes" and "is preferred to
   it by". Sign and mechanism are assumed for the number the lead claim depends on (B1).
b. l. 1051-1056, the catalogs without AGN-hosted mergers: "fAGN < 0.0X", "log10 B = -X.X". The test has not been
   run on any catalog (`notes/GAPS.md` B1) and the sentence already reports that it passed.
c. l. 960-963, `\SlopefAGNDepthNorm`, with the Conclusions at l. 1097: the run that decides whether depth matters
   is red, and the Conclusions already say it does not (B3).
d. l. 823-825: the combined levels of the two offsets, placed as reassurance after three excluded inputs.
e. l. 680-684: "a few narrow spikes where that of the galaxies is nearly smooth" follows two red counts.
f. l. 643-645: "a shift smaller than its own interval" (true on the stored numbers, 0.54 against 1.3).
g. l. 558-564 with l. 860-862: the three red numbers of Sec. III D carry the first reason for the small cost.
h. l. 948: "F_AGN < 0.0X" (true, 0.012). l. 954-957: "reproduces the change" (true, 0.434 and 0.438 in
   `notes/LIKELIHOOD_MATH.md` Sec. 8.3). l. 1002-1003: "X.X times the prior density and no more" (B8).
i. l. 1058-1072: "are repeated on several", twelve red numbers, each with "the input at the XX% level".
j. l. 1074-1076 and 1187-1189: the shift under redraws is written as "XX%" of a half-width; `notes/GAPS.md` A2
   records up to 0.85 of a half-width for AGN-hosted sets.

**C12 (should-fix). One symbol, two quantities.** rho is the signal-to-noise ratio (l. 489, Appendix B) and the
correlation coefficient (Table II, l. 612). Use r for the second.

**C13 (optional). Units and headers.** Table IV: "dfAGN" and "dH0" under "Per decade of n_AGN" read as
differentials; write the change per decade. Fig. 7 and Table IV give log10 n without units in the header (they
are in the caption).

---

## (D) Statistics

**D1 (should-fix). A coverage count.** l. 596-598: "the input H0 lies outside the 90% interval of catalog D and
the input fAGN outside that of catalog I." Replace by the combined statement already made, or by the levels:
"The input fAGN lies at the 99.6% level of catalog I, which also has the smallest Bayes factor of the nine."

**D2 (should-fix). A count over simulations.** l. 654-656: "has no interior maximum in the prior range for 8 of
the 9 catalogs." Replacement: "With the AGN catalog alone the combined H0 posterior of the nine has no interior
maximum in the prior range" (to be checked on the summed grid), followed by the Bayes factor.

**D3 (should-fix). A mean of widths over simulations.** l. 674-679: "the mean 90% half-width of H0 is 0.75 ...
against 3.53". The Bayesian form is already in Sec. V: 93 catalogs combined give 67.71 (+0.36, -0.35) for
galaxy-hosted and 67.82 (+0.08, -0.08) for AGN-hosted mergers, a ratio of interval widths of 4.5. Use that
ratio and drop the means. It is subject to B4 all the same.

**D4 (must-fix). Bayes factors quoted without the prior they depend on.**
- Abstract, l. 120-121: log10 B = 28.3. Either drop it from the abstract or add "for uniform priors 14 Msun and
  0.25 wide on the offsets".
- Conclusions, l. 1091-1092: the same number, no prior.
- l. 744: log10 B = 13.9 for two populations against one on the galaxy-hosts-only mock. The only parameter the
  two models do not share is the fraction of the second population, uniform on [0, 1]; the offsets are fixed at
  the inputs (D5).
- Table III caption: four Bayes factors, no prior ranges. Add "for the priors of Table I; the spin offset on
  [-0.05, 0.20]".
- l. 1055-1056: the Bayes factor of the catalogs without AGN-hosted mergers (uniform on [0, 1] in fAGN).
- l. 815-816 and Table I footnote b: "for the priors of Table I" is ambiguous, because Table I gives three
  ranges for the spin offset.

**D5 (must-fix). Bayes factors between models that know the answer.** 28.3 compares offsets free with a
population fixed at wrong values; 13.9 compares offsets fixed at the true values with offsets fixed at zero;
-1.12 compares a free reference with a reference fixed at the truth. None is the comparison a reader can use.
The one that is, the eight-parameter model against a common population with the same freedom, is red and, by the
ledger in `values.json`, cannot be formed from the stored runs (different code, variance cap and prior range).
Make it the Bayes factor of the paper.

**D6 (should-fix). The uncertainty of the Bayes factors.** "Numerical uncertainty below 0.005" is the grid
integration. The Monte Carlo error of ln L is not in it. Quote each log10 B with the spread under redraws of
the samples (run R0), or say that the spread is below the last digit printed.

**D7 (should-fix). "Excludes the input" without the level.** l. 702-703 (99.3%); l. 818-822 (99.3%, 90.9%,
96.7%). The owner's form is the level at which the input lies.

**D8 (should-fix). "One simulated catalog".** Said eight times, three of them defensively: the abstract's
parenthesis (l. 121), "each is one simulated catalog" (l. 546), "(one simulated catalog in each panel)" (Fig. 4),
"(one simulated catalog each)" (Table III), "This is one simulated catalog." (l. 822-823), "(one simulated
catalog)" (Figs. 6 and 9), "that rest on one simulated catalog" (l. 1058-1059). Not said where a number is
carried forward without it: the spin paragraph (l. 827-847), the slopes (l. 950-953), and every number of the
Conclusions except the nine combined (2.93, 28.3, 2.2 at l. 1089-1093; 1.07, 0.13, 0.97 at l. 1098-1105).
Rule to apply: once in the opening sentence of each Results subsection and once in each table caption; delete
l. 822-823 and the repeats in the captions of Figs. 4, 6 and 9; in the abstract make it the subject ("In a
simulated catalog of 1000 detected mergers in which ..."); in the Conclusions one clause, "on one simulated
catalog each", after the first number.

**D9 (should-fix). The credible level of an input on a one-sided posterior.** The definition at l. 374-377
(smallest equal-tailed interval) gives 34% for log10 n_gal at m < 18, which reads as a recovery of a quantity
that is only bounded above (B8). For those posteriors give the limit.

**D10 (optional). Ranges over simulations.** "between -0.05 and 0.23 in single catalogs" (l. 589-590) and
"1.8 to 15.9 per catalog" (l. 639). Both are in Table II; the text needs only the combined values.

**Already in the owner's form, and to keep.** The definition of the credible level; the
combined posteriors of Secs. IV A and V with their levels; the Savage-Dickey ratios with their priors in
Table II; the convention sentence of the roadmap; no p-value, no sigma, no coverage fraction anywhere.

---

## (E) Style and structure

**E1 (must-fix). The abstract.** Nine sentences, about 290 words, about 22 numbers. Sentence 1 is a context
opener. The headline carries its assumption in a parenthesis ("(90% credible intervals, one simulated catalog)").
"Factors of 1.07 and 1.06" are two numbers for an effect that cannot be seen, and the reader cannot tell which
belongs to H0. "Both inputs inside the 90% intervals" hides a level of 89%. "Measure the offset 2.2 times better"
has no condition. Proposed shape, seven sentences and about a dozen numbers (XX where B1 and B5 must be run):

> We infer the Hubble constant H0 from binary black hole mergers together with the fraction fAGN of mergers
> hosted by active galactic nuclei (AGN), assigning each merger to a galaxy catalog or to an AGN catalog that
> trace one density field and letting the mass and spin distributions depend on the host. In a simulated catalog
> of 1000 detected mergers in which AGN-hosted mergers have a mass peak 5 Msun heavier and a mean effective spin
> 0.10 larger, a single population with free mass and spin parameters returns H0 = XX km/s/Mpc for an input of
> 67.74, and one held at the galaxy values 70.6 (+2.3, -1.7), with both catalogs (90% credible intervals). A
> mass peak and a spin mean for each host class return H0 = XX and fAGN = XX for an input of 0.30, with the two
> offsets at the XX% and XX% credible levels over XX simulated catalogs. When only the spins differ, the catalogs
> narrow the 90% interval of the spin offset by a factor of 2.2 at fixed H0, and the spins leave fAGN unchanged.
> With equal populations and complete catalogs, nine simulated catalogs combined place the inputs of H0 and fAGN
> at the 89% and 70% credible levels, and flux limits cost less than 10% of either interval when the number of
> uncataloged hosts is known. The mergers measure the merger rate per AGN relative to that per galaxy, so fAGN
> needs the number of AGN in the volume it refers to: where the catalog does not supply it, a factor of two in
> that number moves fAGN by XX. The fraction of mergers hosted by AGN is known only as well as the hosts the
> catalogs miss.

**E2 (must-fix). Length: 18 pages against 12 to 15.** Words by section (equations and captions stripped):
Introduction 1070; Method 825 with 5 display equations; Mocks 1320; Results 2680 with 3 (Sec. IV C alone 1095,
two figures, one table); Validation 300; Conclusions 450; Appendices 630 with 6. The balance is right (Method
shorter than Mocks; the introduction under 1100 words). The pages are in the floats and the references. Cuts,
with my estimate of each:
a. Fig. 9 and Appendix C: a full page for 28 featureless regions whose content is three sentences of Sec. IV B.
   Move to supplemental material, or keep a column-width corner of H0, the reference peak and the two offsets
   (-1.0 or -0.6 page).
b. Fig. 8: panels (c) and (d) repeat the combined panel of Fig. 2 and Fig. 3(b). Keep (a) and (b) at column
   width, or rebuild the figure around the tests that are not shown elsewhere (-0.25).
c. Fig. 6: panels (a) and (c) are five identical intervals, and Table IV has the numbers. Keep (b) and (d) side
   by side (-0.25).
d. Figs. 4 and 5: drop Fig. 5(a) (three curves that cannot be told apart; one line of text) and Fig. 4(b) (B13);
   one figure of five panels (-0.35).
e. Table II: drop the column of correlation coefficients (the text gives the combined value) and print log10 B
   in each panel of Fig. 2, or keep the table and drop the per-catalog galaxy-catalog column (-0.2 to -0.35).
f. Sec. V, l. 1037-1039 repeat l. 583-586 word for word; l. 692-694 moves to Appendix A (-0.05).
g. Introduction, paragraph 4: three quotations here and the same three papers again in the Conclusions; one
   place is enough (-0.1).
h. Sec. IV C: Eq. (6) into the text; the list of five prior statements of the degeneracy (l. 975-986) down to
   the two that state it (-0.15).
i. References: 85 entries on more than four pages. The list citations at l. 149 (nine entries), 167, 171, 187,
   262-264, 649, 686, 707 and 865-868 can lose about fifteen (-0.5).
j. White space on pages 9, 11, 12 and 13 goes with the smaller floats (-0.5).
Total about -3.5 pages. The red material (null test, free common population, depth normalization) brings back
about 0.4. Result: 15 pages.

**E3 (must-fix). "Drawn from the exploratory runs."** The last sentence of all nine captions, and red and teal
numbers throughout. Correct for a draft; none of it may reach a reader outside the group.

**E4 (should-fix). One object, several names.**
- The unit of the simulation (one density field, two catalogs, 1000 mergers) is "simulated catalog", "catalog A",
  "reference catalog", "mock", "merger catalog", "set", "event set", while "catalog" also means the galaxy
  catalog and the AGN catalog. "Both catalogs" and "9 catalogs" are different things; the abstract has both
  senses in one sentence ("the hosts in the catalogs widen the intervals of one catalog"). Proposal: "realization"
  for the unit (realizations A to I), "catalog" only for galaxy and AGN catalogs, "mock" as the adjective of
  Sec. III.
- "Common population" (17 times), "single population" (abstract), "one population" (l. 699, 1088). Keep one.
- "Reference" is the reference catalog (catalog A) and the reference population (galaxy class), and in Figs. 4
  and 5 "reference fixed" and "reference free". Write "catalog A" and "the galaxy-class population".
- "Uncataloged hosts" (12), "missing hosts" (the title of Sec. IV C), "the hosts the catalog lacks", "hosts below
  the flux limit". Keep "uncataloged hosts" in the body and the title's image for its three landings (title,
  last line of the abstract, last line of the Conclusions: the count is right).
- "Two host classes" and "two populations" both name the larger model in a Bayes factor (l. 638, 743). Models
  named by a count: "eight parameters", "six parameters" (Table III, Figs. 4, 5, 9). Name them by what is free.

**E5 (should-fix). Coinages and collisions.** "Horizon" (l. 512-513, nine uses, the label of every flux limit)
is the redshift of the most distant of 1000 detected mergers. In this field the horizon is a property of the
detector; here it is a sample maximum that changes from 0.31 to 0.39 between realizations. Write "inside
z = 0.31". "Cataloged share" for F_k is defined and used twice; acceptable.

**E6 (should-fix). "that AGN host".** "The fraction of mergers that AGN host", "AGN host a fraction", "AGN
therefore host 357" (l. 132, 174, 208-209, 513, 535, 691, 999) read as a noun phrase each time, and the last
line of the abstract is one of them. Write "hosted by AGN".

**E7 (should-fix). Sentences in the record-of-runs voice**, with replacements.
- l. 371: "obtain the evidence from the same grid or run." -> "and obtain the evidence from the same posterior."
- l. 514-518: "We use 9 such catalogs, A to I ... For 93 catalogs we also draw one set ..." -> "Nine
  realizations, A to I, have their own density fields. A further set of realizations has all mergers hosted by
  galaxies, and another all hosted by AGN."
- l. 540-546: "Two further mocks separate masses, spins and positions ... All three use the density field and
  the catalogs of catalog A, and each is one simulated catalog." -> state the two populations and delete the
  last sentence.
- l. 569-572: "We take these densities at their inputs, assume an AGN density that is off by factors between 0.5
  and 2, or infer both. The merger catalog is that of catalog A at every limit; catalogs B and C add two more at
  the shallowest." -> "The same mergers are used at every limit."
- l. 715-716: the admission about the fixed population (B1).
- l. 822-823: "This is one simulated catalog." (D8).
- l. 1058-1059: "The results of Secs. IV B and IV C that rest on one simulated catalog are repeated on several."
  -> open with the result.
- l. 1182-1183: "the eight-parameter model admits only points with a variance below 20": a setting of one run.
- Table I, footnotes a and b: prior ranges that differ by run. One range per parameter (B14).
- Table III, "5.0, fixed" and "0.115, fixed" (B13).

**E8 (should-fix). Machine-prose tells.**
- Counted triads in adjacent paragraphs of the introduction: "Three kinds of analysis" (l. 172), "three of them
  name the next step" with three quotations (l. 223-228), "Three capabilities are missing. No analysis ... None
  ... And none ..." (l. 228-233). Keep the first; state the gap without counting.
- One template three times: "The direction is known: ... What we add is the size of the effect" (l. 684-689);
  "The degeneracy itself is known. ... Equations (6) to (8) give its size" (l. 975-986); "For H0 this is in line
  with ... We are not aware of an earlier statement for fAGN" (l. 863-869). The positioning is required; vary
  the form, and put the citation in the sentence that states the result.
- Closers written for effect: "and no more" (l. 1003); "Doing so costs H0 little, and it costs fAGN its
  measurement" (l. 1010-1011); "It removes the bias." (l. 763) is fine and earns its place.
- The Conclusions repeat six numbers of the abstract in the same order.

**E9 (should-fix). The last paragraph of the Conclusions is a list of limitations.** l. 1107-1117: nine items in
two chains of semicolons, none with a direction or a size. The owner's rule is a caveat on the claim it limits,
with its size. Keep the three that change a conclusion and give each its sign: AGN that are also galaxies
(the galaxy catalog then contains every AGN host); uncataloged hosts that cluster (completion that follows the
structure recovers more at low completeness, Refs. [76-79]); photometric AGN redshifts (errors above 0.01 limit
fAGN, Ref. [35]). Move the rest to the sentences of Sec. III where each choice is made.

**E10 (should-fix). Captions against the figures as drawn.**
- Fig. 1: the caption says "dashed arcs"; they are dotted. The labels "81.4%" and "31.5%" sit on the curves and
  "100.0%" is pale on a pale line; "(b)" is drawn on the lower subpanel only.
- Fig. 4: the caption says "the host-dependent model with its offsets held fixed"; it does not say at which
  values, and one of them is not the input. The key in panel (a) is a legend block.
- Fig. 5: the caption does not say that the three curves of (c) have different prior ranges. "Which ends at the edge
  of its prior" describes the upper end of the blue curve in (d); the sharp lower end is left unexplained. The
  key in panel (b) is a legend block, and it serves panels (a) and (c).
- Fig. 6: the figure says "all four completeness levels"; the caption says "flux limit".
- Fig. 7: the lower panels have no letters and the caption calls them "Lower panels". "Gray dotted lines are
  the prior densities" names the wrong comparison (B8).
- Fig. 2: the word "combined" is printed across two outlines.

**E11 (should-fix). Figures and tables that do not earn their place.** Fig. 9 (E2a). Fig. 8(c), (d). Fig. 5(a).
Fig. 6(a), (c). Fig. 4(b), solid curve. Table II, column of correlation coefficients. Table IV, the last four
columns at m < 21 and m < 20 (prior, B8). Table III, the two "offsets fixed" rows and the zero entries of a
log10 B column that compares a model with itself.

**E12 (should-fix). Sentences a reader stumbles on.**
- l. 583-586: "gives H0 = ..., which contains the input 67.74 at the 89% credible level". A value does not
  contain. -> "with the input 67.74 at the 89% credible level".
- l. 638-639 and 656-657: "against galaxy hosts only is" / "against AGN hosts only is". -> "The Bayes factor of
  two host classes against galaxy hosts alone is".
- l. 714: "The fraction stays at fAGN = 0.287". Stays relative to what? -> "The single population still returns
  fAGN = 0.287, since the positions set it."
- l. 1000: "Its posterior": the antecedent is two sentences back.
- l. 235: "This paper takes that step": the step was named six lines earlier and three missing capabilities
  stand between.
- l. 126-129 (abstract): "leave 9.5% of the hosts in the catalogs" (inside which volume?).
- Small integers set as numerals from macros: "9 simulated catalogs", "the 9 combined". Spell out nine.

**E13 (optional). Layout.** Fig. 9 lands inside the reference list (page 15); the heading of Appendix C is set
flush against the last line of Appendix B (page 14). Symbols: x is the event parameters in Eq. (2) and a
luminosity in Eq. (A2); n is the number of samples in Eq. (A3) beside the densities n_k.

**Good, and to keep.** The introduction's structure and its absence of headline numbers; the roadmap
paragraph with the cosmology, the 90% convention and the Bayes-factor convention; Method as one likelihood and
its limits; Sec. III opening on the objects; every Results subsection opening on its result; the two-verb system
("we find" for numbers, "we read" and "we argue" for reasons); no em dashes; none of the banned words in the
rendered text; the direct labels and the small set of colours in the figures; the honest statements of the
railed posteriors (l. 736-739) and of the excluded inputs.

---

## (F) Prior work and citations

**F1 (must-fix). "All of these describe the mergers of every host class with one population" (l. 222) is too
strong for Bom and Palmese.** Their Sec. IV.2, "BBH in AGNs as high mass mergers", simulates mergers that occur
in AGN only when the primary mass is above 35 Msun and finds the H0 precision little changed; their Sec. IV.3
says the AGN channel may dominate the high-mass end; their Sec. IV.1 lists the mass assumptions as future work.
That is the closest published statement to the lead claim of this draft, by an author who is a likely referee,
and the draft does not mention it. Replacement: "None of these infers a mass or spin distribution that depends
on the host class. Bom and Palmese simulate an AGN channel restricted to primary masses above 35 Msun and find
their H0 precision little changed, without a population model in the likelihood."

**F2 (must-fix). "We are not aware of an earlier statement for fAGN" (l. 869).** There are earlier statements,
and they point the other way. In the association likelihood fAGN multiplies the completeness (the draft's own
Ref. [34], and Veronesi et al. before it); Pouw et al. state that the constraining power of
an AGN catalogue is limited by its completeness and redshift precision. The draft's result (a 6% wider interval at "9.5%") has to be
set against those and explained (B10), not presented as the first statement. Replacement: "For fAGN the
association searches find that completeness limits the constraint [31, 33-35]; here the cost is small because
the informative mergers lie where the catalog is complete and the model is given the number of uncataloged
hosts."

**F3 (should-fix). A reported H0 from BBH mergers in AGN is missing.** arXiv:2607.03672 reports 13 BBH mergers
associated with AGN hosts and H0 = 70.5 (+3.4, -2.9). The draft says flare searches "have found candidates and
no confident association" (l. 185-187) and that BBH mergers "have no confirmed electromagnetic counterpart"
(l. 145-146). A referee on the AGN channel knows this paper. Cite it beside Refs. [36-40] with the claim it
makes.

**F4 (should-fix). Moncrieff and Panther (arXiv:2508.18704, MNRAS 2025) is missing.** It selects hierarchical
mergers by mass and asks, by cross-correlation with galaxy catalogs, whether they follow AGN or other galaxies,
with about 5000 detections needed. It is the nearest prior work to "where they are and what they are" and
`notes/NOVELTY.md` (its claim nine) asks for it. Cite it in the third paragraph of the introduction and at l. 841-843.

**F5 (should-fix). "an AGN catalog improves H0 from 3.7% to 3.2% for extreme-mass-ratio inspirals" (l. 687-688).**
In the source (Sec. VI.1) this is for one detector and one population model with every inspiral in an AGN, zero
eccentricity and no redshift errors. Add "when all of them are in AGN".

**F6 (should-fix). The H0 of the fifth catalog.** The arXiv abstract (v2, 4 Aug 2026) gives 71.7 (+9.4, -7.5) at
68% from 236 sources with the Dark Energy Survey; the draft's 90% interval (+17.7, -11.4) is from a table I could
not open. The collaboration's public record and science summary give 71.0 (+9.0, -7.1) at 68%. Two versions are
in circulation. Cite the version, and check the number and the sentence on the mass distribution against the
published paper before submission.

**F7 (should-fix). What Pouw et al. say about the maximum redshift (l. 981-983).** They write that fAGN "depends
on the maximum redshift at which GWs can be emitted", use z_max = 10, and report that their results do not
change between 5 and 10. "Which changes the two normalizations differently" is this draft's reading, attached
to their sentence. They also take the AGN density from a published luminosity function and hold it fixed, and
in their Appendix B.2 they discuss the inconsistency of estimating completeness from the same data. Separate
their statements from the reading, and use the last one: it is the point of B3(b).

**F8 (should-fix). The merger rate per galaxy is already a parameter of Ref. [15].** Its Eq. (16) defines the
rate "per year at a reference galaxy luminosity". Cite it at Eq. (7), so that "what the mergers measure is the
merger rate per host" is read as an extension to two classes.

**F9 (should-fix). "the step that the analyses of Refs. [35, 45, 46] named as their next" (l. 1094-1095, and
l. 222-228).** Pouw et al. defer "the inference of intrinsic population hyperparameters"; Cross-Parkin et al.
propose to model "host weights and compact-binary population parameters simultaneously"; Bom and Palmese ask how
to use AGN hosts "in concert with spectral siren analyses". All three quotations are exact. None names a
population that depends on the host class, and the headline model of this draft holds its population fixed.
Write "a step that these analyses point to", and claim it for the model with the population free.

**F10 (optional). The origin of the association likelihood.** l. 975-977 cites Ref. [34] for the product of fAGN
and the completeness. The form is that of Veronesi et al. (Refs. [31, 33]); cite them with it.

**F11 (optional). Bibliography.** Ref. [62] has no journal (Astrophys. J. 858, 110, from its DOI). Ref. [57]
is, I believe, published (Open J. Astrophys.; check). Ref. [83] prints "(LIGO Scientific, VIRGO, Virgo,, KAGRA)"
and Ref. [20] "VIRGO". Thirteen entries have no journal field; nine are 2025-2026 preprints and are right as
they are.

**Verified and correct.** Bom and Palmese: joint H0 and AGN fraction from a static AGN catalog
(their Sec. III.2), the three quotations, the fiducial density, selection effects neglected, a fixed density
carrying cosmological information. Cross-Parkin et al.: one complete catalog with two weightings, "only a weak
correlation", selection included, population fixed. Pouw et al.: fixed cosmology, the deferral, mergers outside
AGN uniform in comoving volume, redshift errors below about 0.01, limits of 0.52, 0.41 and 0.61 at 95% for
unobscured AGN, the disagreement with the spin-based ranking and "the necessity of jointly inferring" intrinsic
and spatial distributions. The fourth catalog: 75.4 (+24.5, -13.3) at 90%, 142 of 218 sources, GLADE+. Zhu and
Chen: 0.39 (+0.41, -0.32) at 90%. Veronesi et al. 2024: at most 0.21 at 95% above 10^44.5 erg/s, unobscured.
Bartos et al. 2017: 25% with 1000 events. Borghi et al.: robust against host weighting, correlation with the
mass scale grows as completeness falls. Scarpa et al.: H0 correlates with a channel fraction. Bartos and Haiman:
about 10%. Li et al.: aligned, rapidly spinning component. Gayathri et al. 2021: 25%. Kalomenopoulos et al.:
physical weights add little beyond clustering. Zhu et al. 2024: the degeneracy with completeness, by name.
The three missing capabilities of the introduction are consistent with what I found.

---

## (G) Figures: changes to the drawings

**G1. Fig. 1 (should-fix).** Move the four completeness labels off the curves, to the right margin in the order
of the curves, and label each by its magnitude limit as well. Mark the median merger redshift (0.13) and the
redshift to which each catalog is complete on the completeness panel; this is the picture of B10. Give the
completeness panel its own letter. Draw the galaxies darker or as a density shading: at printed size they are
a pale texture and their clustering, which the caption relies on, is hard to see. Draw the arcs as the caption
says.

**G2. Fig. 2 (optional).** Move "combined" clear of the outlines. Print log10 B in each panel if Table II is
cut. Keep catalog F's two lobes as they are.

**G3. Fig. 3 (should-fix).** Panel (a): 186 curves; the orange ones merge into one block. Thinner lines and less
opacity. Add the H0 posterior from the mass distribution alone (B2) to panel (b), which is the baseline both
panels are missing.

**G4. Fig. 4 (must-fix with B1 and B13).** Add the common population with free parameters to panel (a); it is
the curve the paper is about. Remove the solid curve of panel (b) or replace it by a run with the offsets free.
Replace the key of panel (a) by labels on the curves. Draw panels (a) and (b) on the prior range of Table I
once the runs are on [50, 100].

**G5. Fig. 5 (should-fix).** Drop panel (a). Label the curves directly in panel (c), where they separate, and
remove the key. In panel (d) mark the prior range and label the lower edge of the blue curve with what sets it
(B11). One prior range for the three curves of (c).

**G6. Fig. 6 (should-fix).** Drop panels (a) and (c). In panel (b) draw the prediction of Eq. (8) as a black
line and the medians as points: four overlapping curves show less than one line with points on it. When the
hosts are normalized over the catalog depth, the four limits separate and the panel carries the result of B3.

**G7. Fig. 7 (must-fix with B8).** Draw the prior-induced densities in (a) and (b) in place of the flat lines,
or beside them. One lower panel with realization A filled and B and C as outlines; use the freed space for
fAGN against log10 n_gal (correlation -0.94, shown nowhere). Draw the curve of Eq. (6) at the median f-tilde
through the region. Letter the lower panels.

**G8. Fig. 8 (should-fix).** Drop (c) and (d). When they exist, show the tests that appear nowhere else: fAGN
for the realizations without AGN-hosted mergers, and the combined offsets of the host-dependent model.

**G9. Fig. 9 (should-fix).** Remove from the paper or reduce to four parameters at column width (E2a).

---

## (H) The ten changes that would most improve the paper, in order

1. Make the lead comparison fair: a common population with free mass and spin parameters against the
   eight-parameter model, on one code, one prior and several realizations, with the H0 prior at [50, 100]; lead
   the abstract, Sec. IV B, Fig. 4 and Table III with it (B1, B5, B13, B14, D5).
2. Settle the Monte Carlo accuracy before any number is final: bring the variance of ln L to the standard the
   draft cites or measure its effect per result, then restate the factor 4.7, the 91% level and the recovery of
   fAGN (B4, C2, D6).
3. Normalize the hosts over a volume tied to the data and take the number densities from the catalog counts
   where complete; restate what limits fAGN as Eq. (7) with a response that depends on depth, and remove "not
   the depth of the catalogs" (B3, F7).
4. Say how much of H0 comes from the mass scale: add the mass-distribution-only baseline, state that 90% of the
   mergers are in the peak and what fits to the observed mergers give, and show how the bias scales (B2, B6, B7).
5. Rewrite the free-density result around what the mergers add: prior-induced densities in Fig. 7, upper limits
   in place of intervals, the prior-dominated entries out of Table IV, the degeneracy with the galaxy density
   stated (B8, C7, D9).
6. Fill Sec. V with the tests it promises, on the production code: realizations without AGN-hosted mergers, and
   combined posteriors for the host-dependent and flux-limited results; until then no sentence may report their
   outcome (C11, D8).
7. Rewrite the abstract: seven sentences, about a dozen numbers, no context opener, levels in place of "inside
   the 90%", the condition on every number (E1, C3, D4).
8. Correct the account of prior work: the high-mass AGN scenario of Bom and Palmese, the earlier statements on
   completeness and fAGN, the reported H0 from 13 AGN associations, Moncrieff and Panther, the conditions of the
   3.7% to 3.2% number, the version of the fifth-catalog H0 (F1 to F9).
9. One name for each object: "realization" for the unit of the simulation, one name for the common population,
   "galaxy-class population" for the reference, "hosted by AGN", no "horizon", r for the correlation coefficient
   (E4, E5, E6, C12).
10. Cut to 15 pages: Fig. 9 and Appendix C, Fig. 8(c, d), Fig. 6(a, c), Fig. 5(a), Fig. 4(b), a column of
    Table II, Eq. (6), the repeated sentences of Sec. V, about fifteen references; remove "Drawn from the
    exploratory runs." and the limitation list of the Conclusions (E2, E3, E9, E11).
