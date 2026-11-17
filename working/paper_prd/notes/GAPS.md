# Gaps between the analyses on disk and the paper (16 Nov 2026)

Each gap says what is missing, why a referee would care, what closes it and what it
costs. "Catalog" means one simulated catalog with its 1000 events (a seed, in the
analysis directories). Costs are GPU-hours on an 80 GB A100 with darksirens-core at
the measured 0.28 s per likelihood call for the complete-catalog two-population model.
Nothing here has been run. All four GPU nodes were full on 16 Nov.

## A. Gaps that decide whether the numbers can be published

**A1. The Monte Carlo variance of ln L is not controlled.** Every analysis ran with the
variance limit switched off (`max_likelihood_variance = 1e6`); the only guard was the
effective-injection floor. Recorded values at the input H0 on the reference catalog:
the posterior-sample term contributes a variance of 5.3 (all galaxy hosts), 6.6 to 10
(f_AGN 0.25 to 0.5) and 54.7 (all AGN hosts); the selection term adds 1.3 to 2.6. The
eight-parameter run was capped at 20 and sits at 3 to 11. The lensing papers, and the
LVK population papers, limit this variance to 1.
Why it matters: (i) it is the first thing a referee who knows Essick & Farr or Talbot
& Golomb will ask; (ii) it is the leading candidate for a measured defect, A2;
(iii) it already biases results. The logarithm of a Monte Carlo mean is biased low by
half its variance, and the variance grows with f_AGN (5.3 at 0, 7.1 at 0.3, 10 at 0.5
on the reference catalog), so the likelihood leans toward low f_AGN. Measured on
16 Nov from the stored variance grids, by adding half the posterior-sample variance
to ln L in every cell (a first-order correction, CPU only):

| quantity | as stored | corrected |
|---|---|---|
| f_AGN, each of 9 catalogs | | +0.008 to +0.021, mean +0.015 |
| f_AGN, 9 catalogs combined (input 0.30) | 0.283 (0.256 to 0.310), input at the 70% level | 0.297 (0.269 to 0.325), input at the 14% level |
| H0, 9 catalogs combined (input 67.74) | 68.36 (67.73 to 68.99), input at the 90% level | 68.34 (67.72 to 68.96), unchanged |
| H0, 94 all-AGN sets combined | 67.828 (67.749 to 67.906), input outside 90% | 67.735 (67.656 to 67.814) |
| H0, 94 all-galaxy sets combined | 67.65 (67.30 to 68.01) | 67.53 (67.17 to 67.89) |

So the small negative f_AGN offset seen in every analysis of the campaign (-0.012 to
-0.022) is this estimator bias, not physics, and it goes away with more posterior
samples per event. The H0 offset of the joint fits is not explained by it.
What closes it: redraw the posterior samples and the injection set several times for
one catalog and measure how much the posterior moves (the direct test the lensing
drafts used); then fix the number of posterior samples per event and of injections for
production from that measurement. The mock posteriors are analytic, so more samples
cost only likelihood time.
Cost: test, about 10 GPU-h. Consequence for production: unknown until the test; if
20 times more samples per event are needed, the eight-parameter model becomes about
ten times more expensive per catalog.

**A2. Intervals from AGN-hosted events are too narrow.** Over 93 catalogs the scatter
of the H0 median for all-AGN event sets is 1.26 times the quoted 68% half-width, and
the input lies inside the 90% interval 76% of the time (galaxy-hosted sets: 1.00 and
89%). The two injection sets of the same catalog move the AGN median by up to 0.85 of
a half-width. This is why C3's "4.5 times narrower" is at most "3.5 times" today.
The bias correction of A1 does not change this: after it the scatter is still 1.38
times the half-width (0.62 against 0.45 km/s/Mpc, 94 catalogs), so the excess is
noise, not offset. The joint fits show the same factor in f_AGN (scatter 0.063
against a half-width of 0.050), and one of the nine catalogs returns f_AGN = 0.154
(0.078 to 0.237). A variance of 28 in ln L with a correlation length of a few km/s/Mpc
in H0 is enough to produce the excess, and the size of the effect relative to the
posterior width depends only on the number of posterior samples per event, not on the
number of events: roughly 16 times more samples (32,000 per event) would bring it
under a few percent.
What closes it: A1's test on an all-AGN set. If more samples restore the width, done.
If not, the catalog kernel or the pixel prior for a sparse catalog needs a look.
Cost: included in A1. If 32,000 samples per event is the answer, every likelihood
call costs 8 to 16 times more, and a different way of evaluating the AGN term (for
example sampling each event's posterior where the AGN are) is worth asking the
darksirens-core developers about before paying that.

**A3. Most claims rest on one catalog.** C4 and C5 (flux limits at fixed abundance,
dependence on the AGN abundance): one catalog. C6 (free abundances): three catalogs at
one depth, one at the other three. C7 to C9 (host-dependent masses and spins): one
catalog, which also has an accidental difference in mass ratio between its two host
classes (KS p = 0.01, the largest of 41 catalogs examined). In the eight-parameter
model the spin offset comes out at 0.158 (0.121 to 0.191) for an input of 0.10 and the
spin width at 0.082 (0.069 to 0.096) for 0.10; one catalog cannot say whether that is
a fluctuation or a bias.
What closes it: the same analyses on several catalogs, combined posteriors against
the inputs (the paper's validation section is built for this).
Cost on core, before A1's answer: two-parameter grids are about 0.6 GPU-h per catalog,
flux-limited ones less; the eight-parameter model about 18 GPU-h per catalog. Ten
catalogs of everything: roughly 250 GPU-h, dominated by the eight-parameter model.

**A4. Four versions of the inference code.** Analyses 0 to 2 ran on darksirens
`2b86a2d`, 3 to 7 on `0c5b3db`, 8 to 12 on `af896ca`, 13 on darksirens-core
(`bf58aa6`, `e7c3007`). The owner's rule since 2 Oct is that new runs use core at one
pinned commit. Equivalence was shown for the Analysis 11 grids only.
What closes it: the production runs of A3 on one core commit, which then are the
paper's numbers. Before that, one flux-limited cell and one free-abundance run on core
against the stored results.
Cost: under 5 GPU-h for the checks.

**A5. RESOLVED by code reading (16 Nov): how the number densities enter.** See
`notes/LIKELIHOOD_MATH.md` section 8. Each catalog's host density is normalised over
the sky and 0 < z < 5 (the code's internal redshift grid), counting catalogued plus
modelled hosts. In a two-catalog mixture the two normalisations do not cancel, so the
likelihood depends on f_AGN and the assumed densities only through
logit f_AGN + ln(F_AGN / F_gal), with F_k the catalogued share of class-k hosts in
that volume. Checked on the stored grids to 5e-12 nats at f = 0 and 1, and it
reproduces Analysis 4's slopes (0.434, 0.438). Consequences:
- the "AGN = complete" offset (-0.03 to -0.05 in f_AGN) is a setup artefact: the two
  "complete" catalogs were given different nominal magnitude limits (24 and 25), which
  removes different volumes from their normalisations;
- the response of f_AGN to the assumed AGN density (0.43 to 0.47 per dex, the same at
  every depth) is this identity with F_AGN < 0.02, because more than 98% of each
  normalisation is modelled hosts at 1 < z < 5 where there are no catalog objects and
  no events. It is not an effect of hosts missing where the events are.

**A6. The normalisation volume and the missing-host model are not what a paper should
defend.** (i) z_max = 5 was chosen for numerics. With it, the assumed number density
sets each class's weight even for a complete catalog. The physically meaningful
choice is the volume the detectors reach or the catalog depth. (ii) The missing hosts
are uniform in comoving volume; the 2024 to 2026 literature shows that completion
which follows large-scale structure recovers much more at low completeness, and a
referee will ask. (iii) "9.5% completeness inside the horizon" overstates the loss:
that catalog is complete below z = 0.11 and the median event is at z = 0.13.
What closes it: decide the convention for production (my recommendation: normalise
over the catalog depth, report f_AGN for that volume, and state the identity); run
R5b; consider one run with density-modulated missing hosts (`use_LSS`) at the
shallowest limit. Cost: R5b about 5 GPU-h. The paper's present text states the
identity and gives the stored numbers as its small-F limit.

**A7. The mocks are not draws from darksirens-core's population below the minimum
mass (found 17 Nov, in the first minutes of the accuracy run).** The mock generator
draws masses with the samplers of legacy darksirens 2b86a2d, in which the secondary of
a binary whose primary is in the Gaussian peak can fall below the minimum mass of
5 Msun (secondaries down to 1.2 Msun exist in the event files). The legacy likelihood
agrees with the generator. darksirens-core cuts those secondaries at the minimum mass
(a fix made after 2b86a2d), so an event whose posterior samples lie almost entirely
below 5 Msun gets an evidence carried by one sample. Measured on the all-AGN set of
the reference catalog: one such event (true m2 = 3.37 Msun) has ln evidence -176.5 in
core against a median of -15.5, which moves the H0 median by 1.1 km/s/Mpc, 2.2
half-widths. The mixed set of the same catalog has no such event and core reproduces
the legacy posterior there (H0 69.26 against 69.22, f_AGN 0.2746 against 0.2734).
Over catalogs 100 to 120, about a third of the event sets hold at least one event
with more than 95% of its samples below 5 Msun.
Why it matters: production on core with the present mocks would be biased on a third
of the catalogs, for a reason that has nothing to do with the paper's physics.
What closes it: regenerate the events (and injections) with core's own population
sampler, so that the mock is the inference's model again, and validate with the
generator's existing checks; or, as a stopgap, restrict production to catalogs
without such events. Cost of regeneration: 12 CPU-minutes per catalog, plus the
validation stage.
**Owner decision, 17 Nov: regenerate on core.** Production catalogs get events and
injections drawn with darksirens-core's population sampler; the exploratory results
stay as they are.

## B. Gaps in what the paper can claim

**B1. No catalog without AGN-hosted mergers.** Every two-population fit was made on
mocks with f_AGN = 0.30. The paper cannot say that mergers hosted only by galaxies do
not produce a spurious AGN fraction, nor show how the Bayes factor behaves there.
Close with the two-population model on the existing all-galaxy event sets (and one
other input fraction, for example 0.10). Cost: about 0.6 GPU-h per catalog.

**B2. Pixel resolution.** f_AGN keeps a slope of +0.028 per dex of pixel occupancy
after the estimator change, and its cause is not known (the redshift-kernel bandwidth
is ruled out). A referee will ask for the analysis at two resolutions. Close with the
complete-catalog fit at nside 64 on two catalogs. Cost: a few GPU-h; needs the
catalogs re-pixelated.

**B3. Evidence for the model comparisons.** The paper's statistics are Bayes factors,
and several are not available from the stored runs (the numbers agent is listing which).
Grids give Savage-Dickey ratios for nested cases; comparisons between the
host-dependent models need nested-sampling runs with matched priors. Folded into A3's
production list.

**B4. H0 gain from the spin difference.** The 11% narrowing of H0 when the spin offset
is modelled is one catalog, and the mass-offset analysis found no gain at all. The
paper does not claim a gain (C8 says so). It would need A3's catalogs to say more.

**B5. The single-population analysis that K1 calls biased has a fixed population.**
The +3.0 km/s/Mpc shift compares the host-dependent model with a common population
whose mass peak and spin mean are fixed at the galaxy-class values. A real analysis
fits its mass spectrum. With a free common peak the fit would land between the two
classes and the H0 shift would probably be smaller. A referee will ask for that
number (Pierra et al. 2024 attribute such biases to models that cannot adjust). It
was never run. Close with run R8e. Cost: about one eight-parameter run per catalog.

**B6. Prior work the earlier sessions did not know about.** The novelty check
(`notes/NOVELTY.md`, 16 Nov) found that the joint H0 and host-fraction step is already
published (Bom & Palmese 2024; Cross-Parkin et al. 2026, which the owner co-authored)
and that f_AGN with selection and incompleteness at fixed cosmology was posted in
September (Pouw et al. 2026). The paper now leads with the host-dependent population
result, which three groups name as their next step. That part rests on one catalog
(A3), so A3's host-dependent runs are the time-sensitive ones.

**B7. Combined H0 in the mixed-population fits sits high.** Nine catalogs combined:
68.36 (67.73 to 69.00) for an input of 67.74, at the 89.5% level; the galaxy-only
analysis of the same events gives 68.9. The all-galaxy and all-AGN event sets recover
the input. Nine catalogs cannot tell a fluctuation from an offset of 0.6. More
catalogs of the joint fit (run R1) settle it; this is cheap.

## C. Limits of the mocks (stated in the paper, not run)

AGN are a separate population and not a subset of the galaxies; one luminosity
function and one flux limit for both catalogs, isotropic depth; redshift errors of
0.003 (1+z) for every object; a single-number signal-to-noise detection rule and
analytic posteriors with 1 to 2.4 degree localisation; all mergers inside z of about
0.3; fixed mass-ratio, redshift-evolution and power-law parameters; hosts weighted by
number only. Each is a sentence in the conclusions. None needs a run for this paper,
but the first one (AGN inside galaxies) is the natural next model change.

## Owner decisions (17 Nov)

Regenerate the mocks on core (A7). The small follow-ups to the accuracy run go right
after it on the same GPU. Production waits for a costed plan and the owner's
approval. New work is committed locally, not pushed.

## D. Order I recommend

1. A1 test (settles A2 and the production settings), A4 and A5 checks: about 15 GPU-h.
2. Production on core at one commit with the settings from step 1: complete catalogs,
   single-catalog and all-galaxy sets, flux-limited ladder, free abundances,
   host-dependent models, on about ten catalogs. Roughly 250 GPU-h at today's settings.
3. B1 and B2 alongside step 2.

## Update, 17 Nov 2026 (after Analysis 15 and the owner's decisions)
- A5 closed by Analysis 15 (`analyses/analysis_15_host_normalisation/REPORT.md`): the identity is exact
  on core where missed hosts lie outside the data; the "AGN = complete" offset is the nominal depth.
- A6 answered in part: normalizing to z = 1.10 removes the density dependence only for complete
  catalogs. At m < 18 a factor of two in the assumed AGN density moves f_AGN by 0.13 and H0 by about
  1 km/s/Mpc. Production convention: densities from each catalog's complete volume.
- B (paper structure): superseded by `notes/PAPER_SKELETON.md` and `REVISION_DECISIONS.md` item 13.
  Two drafts are maintained: `paper_prd/main.tex` and `paper_short/main.tex`.
- Production scope approved by the owner (see `runs.txt`, last block); machinery in
  `analyses/analysis_16_production_runs/`.
- New watch item: in the exploratory eight-parameter run the AGN spin offset is 0.158 (+0.030, -0.037)
  against an input of 0.10 (outside the 90% interval; one realization, 2000 samples per event).
  The production runs on five realizations decide whether this is a draw or a bias.
