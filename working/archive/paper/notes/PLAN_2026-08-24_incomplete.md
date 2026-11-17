# PLAN 2026-08-24: fold the incomplete-catalog results into the paper

Orchestrated execution (Fable orchestrator, Opus workers). Repo at `a0e6335`,
clean, plus the follow-up campaign results (fu_* dirs, FU_REPORT.md, not yet
committed). Owner approved the itemized edit list 2026-08-24 and the
abstract-style conclusion draft; execution style bound by
`astro-paper-style` + `physics-first-results`.

## Headline decision (gated on data already on disk)

New section 4.3 claim, pre-computed from local JSONs:

> Flux-limited catalogs preserve the joint (H0, f_AGN) measurement provided
> the completeness is computed from the survey's flux limit and the catalog's
> luminosity function. With both host densities free, that representation
> recovers the input galaxy density to within 0.2 dex at each of three
> realisations, where completeness estimated from the observed number counts
> overstates it by 0.5 to 1.2 dex; the data prefer the former by 12 to 18 in
> ln evidence. Freeing the densities widens the f_AGN interval by a factor
> ~5, and H0 is insensitive to the assumed AGN density only while the
> catalogs remain more than ~80% complete (at 9.5% completeness a factor-2
> density error moves it by ~1 km/s/Mpc).

Costs (stated with the claims): anchors fixed at the simulation's input
densities in the ladder; luminosity-function fits held fixed across
realisations; single-realisation ladder (the free-density replication is
three realisations); f_AGN conditional on the AGN density everywhere.

EXCLUDED from the paper (owner decisions of 2026-08-12/24): the
relative-completeness law, the faintward f_AGN bias, the oracle "tripled
bias", analysis-7's occupancy axis, and the four AGN=complete cells (their
-0.06 offset is a completion contribution live at C==1, mechanism unresolved;
quote flux-limited cells only).

## Vocabulary (binding for prose, captions, figure legends)

- "completeness computed from the survey's flux limit and the catalog's
  luminosity function" (first use, spelled out); thereafter
  "luminosity-function completeness".
- "completeness estimated from the observed number counts"; thereafter
  "number-count completeness".
- NEVER in reader-facing text: per_pixel, selection mode, c_mode, estimator
  switch narrative, campaign/redo language, code names, seeds-as-process.
  "realisations of the simulated universe" is the sanctioned phrase.
- Intervals: 68% equal-tailed in prose (declared in the intro roadmap);
  figures show 90% bands, labeled in caption and legend.

## Macro contract (WS-A1 implements EXACTLY these names, append-only;
existing 182 macros byte-identical; replace the three pending)

Sources (all local; A1 must assert c_mode=="selection" and, where present,
darksirens_git_sha startswith 0c5b3db in each consumed file, else hard fail):

- LADDER  = analyses/analysis_3_incomplete_catalog_H0_fagn/results/joint_{m21,m20,m19,m18}_s100.json (+ladder_summary.json)
- FREE    = analyses/analysis_5_free_anchors_H0_fagn/results/campaign_m18_dynesty_s100.json
- SEEDPAIR100 = analyses/experiments/experiment_dsmaster_4d_recheck/results/fit_m18_{selection,per_pixel}_s100.json
- SEEDPAIR101 = analyses/selection_redo/fu_seed101/results/campaign_m18_dynesty{,_pp}_s101.json
- SEEDPAIR102 = analyses/selection_redo/fu_seed102/results/campaign_m18_dynesty{,_pp}_s102.json
- SENS    = analyses/analysis_4_density_anchoring_H0_fagn/results/arms_summary.json (+per-arm jsons as needed)
- SURFACE = analyses/analysis_6_relative_completeness_H0_fagn/results/surface_summary.json

Macros (median^{+u}_{-l} formatting per existing conventions unless noted):

| macro | value |
|---|---|
| \HzeroIncomplete        | H0 median+CI68, LADDER m18 |
| \FagnIncomplete         | f_AGN median+CI68, LADDER m18 |
| \FagnWidthRatio         | f_AGN halfwidth68 LADDER m18 / analysis_2 joint (existing \FagnJoint source) |
| \CompletenessShallowAgn | AGN completeness at m18 (percent, from SENS or fit metadata; expect 9.5) |
| \CompletenessImmuneMin  | completeness above which H0 slope consistent with 0 (expect 81) |
| \FagnFree               | f_AGN median+CI68, FREE |
| \FagnFreeWidthRatio     | f_AGN halfwidth68 FREE / LADDER m18 (expect ~5) |
| \GalDensityFree         | log10 n0 median+CI68, FREE |
| \GalDensityFreeOffsetDex| |median - (-3)| of FREE, 1-2 s.f. |
| \GalAnchorOffsetMaxDex  | max over 3 seeds of |sel median + 3| (SEEDPAIRs; expect 0.17) |
| \PixelAnchorBiasMinDex  | min over 3 seeds of (pp median + 3) (expect ~0.5) |
| \PixelAnchorBiasMaxDex  | max over 3 seeds (expect ~1.2) |
| \SeedLnBFmin, \SeedLnBFmax | min/max over 3 seeds of lnZ_sel - lnZ_pp, 2 s.f. (expect 12, 18) |
| \NSeedsIncomplete       | 3 |
| \HzDensitySlopeShallow  | d H0 / d log10(assumed AGN density) at m18, km/s/Mpc per dex (SENS; expect 3.2) |
| \HzDensityShiftFactorTwo| that slope x log10(2) (expect ~1.0) |
| \FagnDensitySlope       | d f_AGN / d dex at fixed depth (SENS; expect 0.45) |
| \RelCompletenessSpanDex | span of log10(C_AGN/C_GAL) over SURFACE flux-limited cells (expect 1.86) |
| \FagnRelSlope           | f_AGN slope over that span, per dex (expect -0.004) |

Also: one NUMBERS.md containment entry per agreement claim
(H0/f_AGN vs input at LADDER m18; galaxy density vs -3 for FREE and each
seed), stating which interval (68/90) contains the value; prose must match by
lookup. audit_values.py must exit 0.

## Workstreams

- WS-A1 (Opus): build_values.py extension per contract; NUMBERS.md; audit green.
- WS-A2 (Opus): scripts/fig_incomplete.py + make_figures.py hook; figure*
  two-panel at \textwidth: (a) ladder H0 and f_AGN vs AGN completeness with
  90% bands and input-value lines; (b) galaxy-density recovery per
  realisation, both completeness treatments, 90% CIs, truth line, per-seed
  ln-evidence difference annotated. Acceptance: legible at \textwidth,
  legend wording == caption wording, no code names, all numerals from the
  same JSONs the macros use. PNG+PDF via figstyle.
- WS-B  (orchestrator): prose. Results 4.3 (replace the \todo), methods 2.3
  addition (the LF form as a variant of Eq. completeness family), validation
  paragraph (three-realisation replication), abstract +2 sentences,
  discussion named-inputs addition.
- WS-C1 (Opus, fresh): integration + compile: apply orchestrator patch file,
  run build_values, make_figures, pdflatex/bibtex x3, zero errors/overfull,
  word + abstract counts from rendered PDF (linenumbers off).
- WS-D1 (Opus, fresh): claims-vs-numbers adversarial review vs JSONs;
  independently recompute one lnBF and the m18 ladder medians.
- WS-D2 (Opus, fresh): style review per astro-paper-style self-check
  (banned words on rendered PDF incl. figures, coinage dump, agreement-verb
  audit by value-grep, em-dash scan, llm-tells pass, caption-vs-pixels).

## Verification (phase D, orchestrator)

audit_values exit 0; compile clean; pdftotext greps 0; abstract <=250 from
render; I recompute \SeedLnBFmax and \HzDensityShiftFactorTwo independently;
all D1/D2 findings fixed or recorded; commit per workstream; memory updated.

## Owner decisions (2026-08-24, post phase A — supersede the contract where they differ)

1. SHA exemption ACCEPTED for `fit_m18_per_pixel_s100.json` (records e8d5035,
   the revision its arm was finished on; its comparison file bounds the
   revision drift at 1.3 replicate-scatter units, and the recorded
   ln-evidence drift bound is 0.09 against a preference of 17.7). The
   exemption is keyed to that exact hash in both scripts.
2. `\FagnRelSlope` (+0.005 over the six flux-limited cells; the contract's
   -0.004 was the eight-cell fit including AGN=complete cells) is kept as a
   macro with its caveat and is NOT quoted in prose; no sentence may lean on
   its sign.
3. Prose quotes the macros as rendered (0.97, 0.455, 0.12, 0.17, 5.9, 1.1);
   the contract's "expected" column is superseded by the rendered values.
4. Post-contract macro addition (phase B): `\HzDensityShiftFactorTwoDeep`,
   the factor-of-two H0 shift at the m<20 rung (|slope_m20| x log10 2,
   expect 0.01), quantifying the deep side of the immunity claim; plus one
   containment row for f_AGN at ladder m18 against the REALISED fraction
   (the figure's dashed line and A1's existing flag are against the input
   0.295; prose references the realised fraction per the paper's convention).
5. Independent recomputation (orchestrator): m18 H0 slope 3.2125/dex,
   x log10(2) = 0.9671; ladder m18 H0 68.66 CI68 [67.60, 69.67], f 0.286
   CI68 [0.235, 0.339]. Matches A1.

## Review adjudication (2026-08-24, phase D)

D1: 0 SEV-1, 8 SEV-2, 9 SEV-3. D2: 12 blockers, 17 advisories. Disposition:

FIXED by orchestrator prose edits: abstract cap + null-test assumption +
"insensitive" quantified (B1/A8/A9, D1-8); handle discipline, one name per
treatment, number-count handle named in methods (B3/B4/B5); caption honesty
on the right panel, medians vs intervals (B6, D1-3); figure-4 realised-label
caption wording (B7, D1-2); "agree in the mean" removed, true-z LF fit
disclosed, LF-known assumption stated (D1-4, D1-7); discussion conditional
re-scoped to the density, H0 immunity re-stated with both sizes (D1-5,
D1-6/B10); agreement claims name their interval, verbatim across sections
(B9); reference-realisation caveat on the ladder (B11); per-rung containment
levels stated (A7); intro roadmap (A6); finance register trimmed to two
pre-existing uses + abstract closer (A2); "heavy lifting" (A12);
pixel-by-pixel sky normalisation (A11); double "of evidence" (D1-16/A13);
width ratio 1.1 reframed as "essentially unchanged" (D1-14); free-f
coherent offset disclosed with sizes (D1-13); \shorttitle added (part
of A14).

FIXED by mechanical worker: \SeedLnBF* at 1 decimal (D1-1); figure label
realised/input strings + hardcodes (B7/B8/A16); fig_joint seed label (B2,
D1-17); realisation renumbering 1..5 (A4); fig_pgm channel collision (A10);
orphan sweep to figures_attic (B12); Schechter + HEALPix cites (A15);
unused completeness macros deleted (D1-10/15); m21/m20/m19 containment rows
(A7); \FagnFreeOffsetMin/Max added.

ACCEPTED, no change: D1-11 (prose free fit vs figure realisation 1 are two
runs of one configuration; both correct); D1-12 (results scopes the slope
"at fixed depth"; discussion quotes the shallowest); D1-14 residual caution
noted in NUMBERS; A3 "standing" kept in its one pre-existing use; A17.

DEFERRED to owner: A5 (figures 3/5/6 draw 68% intervals against the
90%-plots standing rule; changing them alters pre-existing committed
figures); A14 remainder (ORCID, \facility, \software, data-availability
statement); ApJL vs ApJ target (6 figures vs the Letter's soft cap).

## Risks

- arms_summary field names unknown -> A1 inspects, derives slopes by fit if
  not precomputed, and RECORDS the derivation in NUMBERS.md.
- 68 vs 90 convention drift -> contract fixes it; D2 checks every interval
  statement.
- Figure/caption vocabulary drift -> binding vocabulary above; D2 rechecks.
