# Review of the text of 17 Nov (evening), with the decisions taken on it

Independent review (Fable) of `main.tex` at commit 06843cf: the text that took in the experiment
(Sec. IV B, Fig. 3, the probability of an AGN host, units of the number densities, abstract,
summary, Validation, Conclusions) and its consistency with the rest. Line numbers are those of
`main.tex` at that commit. Part A is the decision on each finding; Part B is the review as received.

## A. Decisions (all findings accepted unless said)

Numbers and figures (done before the text pass; see `values/MACROS.md` for the names):
- 1.1: a macro for the f_AGN at which the probability of an AGN host is evaluated in each run (its
  posterior median) and for the mean over all mergers. The dashed line of Fig. 3(c) stays the input.
- 1.2: smallest and largest cost over the realizations for every cost factor.
- 1.7: the red placeholders of the inferred number densities are brought inside the prior of Table I.
- 1.11 (Table IV, Fig. 4(a)): the red flux-limited placeholders are re-anchored to the finished
  complete-catalog result so that a red row does not contradict the black one.
- 1.10 (Fig. 2(b) label), 2.10 (lower case in `fig_classes`): changed in the figure script.
- 2.2: half-widths print two significant figures.

Text:
- 1.1 to 1.6, 1.8 to 1.11, 2.1 to 2.6, 2.9, 2.11, 3.1 to 3.11: apply, in the paper's voice; the
  reviewer's replacements are proposals, not wording to paste.
- 1.9: a statement about another paper stays only as far as `notes/PRIOR_WORK_VERIFIED.md` (or a
  full-text entry of `notes/CITATION_CHECKS.md`) supports it.
- 2.1: the abstract goes back to about a dozen numbers in seven sentences.
- 2.7: the equation for the probability of an AGN host and its paragraph move from Sec. II A to
  Sec. IV B, where the quantity is first used, so that the Method is again shorter than the
  section on the simulations and has six display equations.
- 2.8: not applied. The length target of 14 to 15 pages and nine figures (item 8 of
  `REVISION_DECISIONS.md`, skeleton Sec. 3) predates the owner's decision to fold the letter into
  this paper; about 18 pages and ten figures is the size of the one paper.
- 2.11: the text follows the figure, log10[n/(h^3 Mpc^-3)].
- 3.1: one measure of precision, the 90% half-width, defined once and used in every section
  (`...HalfWid` macros exist for every key).
- Seen in passing: HEALPix without a citation; titles printed in the reference list although
  item 8 of `REVISION_DECISIONS.md` keeps them off. Fix both if `refs.bib` and the verified notes
  allow.

## B. The review

### 1. Wrong or unsupported

1.1 Lines 756-760 (Sec. IV B). "...has a mean of 0.33 over the mergers of realization A that AGN
host and of 0.24 over those that galaxies host, where a merger with no information would have
0.30." By Eq. (3) a merger with equal class likelihoods has the f_AGN at which Eq. (3) is
evaluated, the posterior median: 0.2672 for this run, so 0.27, not the input. The offsets are
+0.06 and -0.03, the reverse of what the sentence suggests; the mean over all 1000 mergers is
0.27. Proposed: "...where a merger whose position says nothing would have 0.27, the posterior
median of f_AGN at which Eq. (3) is evaluated." The same applies to Fig. 3(c): the dashed line is
the input 0.30 (the caption says so, correctly), while the top panel was evaluated at f = 0.32 and
the bottom at 0.295.

1.2 Lines 1037-1040 (Sec. IV E) and 1131-1134 (Sec. IV G). "...so the unknown host class costs a
factor of 1.6 in the number of mergers, where it cost 2.3 with equal populations." The cost itself
falls with the share of AGN-hosted mergers (3.4, 2.3, 1.5 at 0.1, 0.3, 0.5), and the host-dependent
set has 0.36 of its detected mergers AGN-hosted against 0.298. Interpolating the equal-population
half-widths to 0.36 gives a cost of about 2.1 (the reviewer's estimate, no run exists), so the
confound named for the half-widths also applies to the cost. Per realization the costs are 1.8 to
3.4 (equal) and 1.3 to 2.3 (host-dependent); the ranges overlap. In IV G the 1.6 follows "Every
row", but no row holds two populations at their inputs. Proposed, IV E: "...costs a factor of 1.6
[1.3, 2.3] in the number of mergers, where it cost 2.3 [1.8, 3.4] with equal populations; part of
that difference is again the larger share of AGN-hosted mergers, with which the cost falls
(Sec. IV B)." Proposed, IV G: "With the population known that costs a factor of 2.3 in the number
of mergers at f_AGN = 0.30; with two known populations that differ, an assumption no row makes, it
costs 1.6."

1.3 Lines 1232-1233 (Conclusions). "Its precision rests on the mergers that AGN host, which
positions alone do not single out." "Its" is the joint inference with the host class unknown and
the populations inferred; the support (90% and 72%) exists only with the host class and the
population known, and the text itself says the division cannot be made with the host class
unknown. Proposed: "With the host class known, most of the constraint on H0 comes from the
AGN-hosted mergers; positions alone do not single them out, and that is what the unknown host
class costs."

1.4 Lines 1023-1024 against 491-492. "Masses and spins that depend on the host also say which
mergers the AGN host" against "so no single merger can be assigned to a class by its mass or its
spin" (and the Fig. 1 caption). The numbers are mean probabilities of 0.51 and 0.28. Proposed:
"Masses and spins that depend on the host add to what the positions say about the host class
[Fig. 3(c)]."

1.5 Lines 725-727. "Averaged over realizations, the AGN catalog adds nothing when AGN host one
merger in ten". What is compared is two event sets (fractions 0.1 and 0), both analyzed with both
catalogs; no analysis with and without the AGN catalog on the same mergers stands behind the
sentence. Per realization the difference runs from -0.73 to +0.60 km/s/Mpc. The Conclusions make
the same point against the galaxy catalog alone, IV B against both catalogs. Proposed: "Averaged
over realizations, mergers of which one in ten is hosted by an AGN constrain H0 no better than
mergers with no AGN host; this is a statement about the means, since ..." Use one pair of numbers
in both places.

1.6 Lines 434-435 against 1253-1255. "an input f_AGN of 0.30 lies below the current upper limits
for unobscured AGN [Pouw]" against "upper limits between 0.21 and 0.61 ... [Veronesi, Pouw]".
0.30 is above 0.21. Both numbers are supported by the verified notes; the two sentences disagree
with each other. Proposed, III A: "lies below the upper limits of 0.41 to 0.61 of Ref. [Pouw]";
Conclusions: the same range, or "inside the range of the current upper limits, 0.21 to 0.61".

1.7 Lines 881-886, Table IV and Fig. 5(a) against Table I. After the unit change the red limits
lie above the prior they are drawn from (numbers work).

1.8 Lines 761-764. "...holds 15 AGN in the median and that of an AGN-hosted merger 17, beside 1400
and 1500 galaxies." The four macros are medians over the sets with every merger hosted by galaxies
or by AGN, not over the f_AGN = 0.30 mergers the sentence explains. Add "(medians over the sets of
realization A with every merger hosted by one class)".

1.9 Other papers. Line 686-687, "That rare hosts are easier to associate is known [Bartos, Bom]":
not in `notes/PRIOR_WORK_VERIFIED.md`, which says Bartos 2017 was not opened;
`notes/CITATION_CHECKS.md` supports Bom and Palmese from the full text ("lower AGN density gives
tighter H0") and Bartos from the abstract only. Proposed: "That a lower density of hosts gives a
tighter H0 is known [Bom]." Line 1259-1260, "like the candidates found so far [Graham:2020gwr]":
that key is the single candidate for S190521g; the nine-candidate search is `Graham:2022xxu`, in
`refs.bib` and uncited.

1.10 Captions. Fig. 3: "with the population known and complete catalogs", but the open squares
have the populations inferred; proposed "with complete catalogs and, except for the open squares,
the population known". Fig. 2: "Each catalog narrows what the known mass scale gives", while the
panel draws the AGN catalog alone, now called biased; proposed "The galaxy catalog narrows what the
known mass scale gives and the AGN catalog narrows it again; the AGN catalog alone is biased."
Fig. 2(b) prints "peak 2.4 off scale", a number with no macro and no unit (figure script).

1.11 Tags. Today's text is clean. Outside the diff: line 780, "We attribute the small cost to two
things. % EXPECTED": as printed the widths are 3.4 (red) against 3.9 (black), so there is no cost;
line 789-791, "the cost is small here because...", rests on the same red number and is untagged;
Table IV and Fig. 4(a): the complete row is black (68.4) and m<21 is red (69.2) under a caption
that says the flux limit hardly moves either parameter (numbers work). Untagged sentences worded
for a red outcome: line 928 ("The catalogs break a degeneracy"), line 1104 ("The two offsets fare
differently"), lines 1115-1117, line 1215 ("is sufficient"), lines 1241-1247.

### 2. Against the style skill

2.1 Abstract: about 35 numbers; item 7 of `REVISION_DECISIONS.md` says about a dozen. Shorter form
of today's sentence: "With the population known and complete catalogs, the 90% half-width of H0
falls from 3.5 to 0.8 km/s/Mpc as the mergers hosted by AGN go from none to all, and with three in
ten hosted by AGN the unknown host class costs the precision of 2.3 times as many mergers."
2.2 Significant figures: half-widths quoted to three figures (3.53 against 3.55, "differ in the
last digit only") while the range over realizations is tens of percent and two injection sets move
realization A's half-width from 1.95 to 2.09. Quote two figures and say "agree to two figures".
2.3 Mechanism stated as fact, line 683, "The difference is one of the number of candidate hosts",
while line 688 says density and bias were not separated. Proposed: "We read the difference as one
of the number of candidate hosts".
2.4 Assumption travels with the number: Conclusions lines 1249-1252 give three half-widths without
"population known, complete catalogs, 1000 mergers"; the cost factors carry no range where the
half-widths carry brackets.
2.5 Run of numbers: lines 1054-1064 put sixteen numbers in three sentences, all drawn in Fig. 3(a,
b). Proposed: "With both catalogs the half-widths are those of the peak-dominated spectrum at every
fraction, 1.98 against 2.01 km/s/Mpc with three mergers in ten hosted by AGN, each difference
inside the range over the realizations [Fig. 3(a) and (b)]."
2.6 Restating: 2.3 is printed five times (abstract, lines 741, 1040, 1132, 1261); the Conclusions
do not restate the body (item 13f).
2.7 Method against Data: Method is 1036 words and 7 display equations; the simulations section is
998 words; the appendices hold 3 equations. Eq. (3) tipped it.
2.8 Length: 18 pages and ten figures against 14 to 15 and nine.
2.9 "AGN host" as noun and as subject plus verb in one sentence (lines 757-759, 1023-1024, 1232).
Use "the AGN-hosted mergers".
2.10 `fig_classes` capitalizes legend and axis labels where the other nine figures use lower case.
2.11 Units typography: text and tables write log10(n/h^3 Mpc^-3); the figure n/(h^3 Mpc^-3).

### 3. Where a gravitational-wave cosmologist would stumble

3.1 Width and half-width: IV A gives widths (12, 6.5, 3.9), IV B half-widths; Validation gives "a
90% width of 1.6" for the analysis IV B quotes as 0.81. "90% half-width" is never defined.
3.2 "From positions alone" is undefined for host-dependent mergers. It is the analysis with one
population held at the galaxy-class inputs (posterior median H0 = 70.9, f_AGN = 0.32, against 67.2
and 0.295 with two populations), so the two panels of Fig. 3(c) differ in evaluation point as well
as in the use of masses and spins, and the reader meets two "positions alone" pairs (0.33, 0.24 in
IV B; 0.37, 0.30 in IV E) with no word on why they differ. Add: "From positions alone, that is with
one population for both classes held at the inputs of the galaxy class, ...".
3.3 The analysis "with the populations known" is not among the limits of Sec. II A, where
"population known" has both offsets at zero, and Table I says the offsets are held at 0. Add: "For
mergers with host-dependent populations the offsets are held at their inputs as well."
"Host-dependent populations" now names an event set, the analysis that infers the offsets, and a
figure series with them held.
3.4 Line 1050, "the mass scale is a blunt ruler": the number is 8.44 against 5.57, a factor of 1.5.
Proposed: "With a mass spectrum like the observed one the known mass scale alone gives a half-width
half again as large, and with the catalogs H0 is constrained as before".
3.5 Lines 1123-1127, "in both the analysis lacks something that the mergers have": the galaxy
catalog alone also lacks the hosts of three mergers in ten and stays on the input (68.6 +1.7 -2.0,
input at the 54% level, both black); the paper never says why. Proposed addition: "The galaxy
catalog alone also lacks the hosts of three mergers in ten and stays on the input, which we read as
the galaxies tracing the density field of the AGN they do not hold."
3.6 Line 1131, "Every row has the host class of the mergers unknown": the rows with one catalog or
none do not infer f_AGN. Proposed: "No row uses the host class of a merger."
3.7 Abstract, "a factor of 2.3 in the number of mergers": the inverse-square-root step and the
direction are missing.
3.8 Brackets are defined "in this subsection" and reused in IV E and IV F.
3.9 Lines 510-515: the sets with twice the offsets at 0.30 are not declared, and Sec. IV F is
missing from the list of sections that use the further sets.
3.10 Caption, "(b) The same": panel (b) has three fractions and no twice-offset series, and the
caption does not say that (b) belongs to Sec. IV F.
3.11 Lines 1259-1262: a flare identifies the host, hence the redshift, which is worth more than
the class; 2.3 is the value of the class alone.

### Clean
Eq. (3) against the likelihood and `notes/LIKELIHOOD_MATH.md` (apart from 1.1); derived macros
(costs, shares, ratio) reproduced from `values.json`; names (apart from 3.3, 3.6); Fig. 3 legend,
point counts and printed means; rendered PDF: banned words, em dashes, author notes 0; build: 0
errors, 0 overfull boxes, 0 undefined; abstract seven sentences; literature numbers in the changed
text supported.

### Not checked
No macro re-derived from run outputs; no cited paper opened; Secs. I to III, IV C to D and the
appendices read for tags and obvious consistency only; figures compared by eye; title-image
landings and the reference list not checked.
