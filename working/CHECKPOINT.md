# Checkpoint, 18 Nov 2026, 09:00 UTC (read this first after a context clear)

You are the Fable session that owns the gws-agn project (owner handover of 16 Nov 2026: full
technical and creative freedom; Fable agents write and review, Opus agents code and submit jobs,
you orchestrate and check). Everything below is the state at the time of writing. The memory files
`gw-agn-paper-prd.md`, `writer-briefs-must-follow-style-skill.md`, `gw-agn-ownership-handover.md`
hold the owner's words; this file holds the working state.

## 1. What the owner wants (standing)
- ONE paper (owner, 17 Nov ~20:05 UTC: "Lets stick to the long paper. You did a better job there.
  So i would archive the short paper."): `working/paper_prd/`. The letter
  `paper_short_independent` is archived (Sec. 2). Do not propose a short version again unless asked.
- What is new, in the owner's words: "proper multi-tracer darksirens, and correlated gw
  properties with galaxies". The paper is titled and argued for that.
- The draft is written for the expected results at once; anything that waits for a run is red (red
  macros `\ph{}`, red figure data); no note to the author is printed in a figure.
- Form is governed by the owner's style skill `astro-paper-style`. LOAD IT YOURSELF before
  briefing any writer; a brief adds content and never overrides the skill (the owner rejected a
  draft whose brief did). Bayesian statements only, 90% credible intervals, no em dashes.
- Commit locally on `main`, never push (push is the owner's call). Attribution lines:
  `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01L6W2Gr2RxPuB4nQyh4gjGp`.
- Private: never put the owner's name, the project name or local paths into outside requests.
- Agents: no idle waiting; code watchers or Slurm dependencies instead; fresh agents with short
  briefs beat resuming cold ones; keep the number proportionate.

## 2. The paper
- `working/paper_prd/main.tex` (16 pp). Design: `notes/PAPER_SKELETON.md` (Sec. 9 of 17 Nov evening:
  the long paper takes in the letter's experiment; new subsection IV B "The constraint from each
  host class", new figure `fig_classes`, names "host class unknown / known", "probability of an AGN
  host"), `REVISION_DECISIONS.md` (item 13), `notes/OWNER_STYLE_DETAILS.md`.
- Numbers: `scripts/build_values.py`, `production.py`, `values/production_map.json`,
  `audit_values.py`, `values/MACROS.md`. Figures: `scripts/make_figures.py`,
  `figures/CAPTION_FACTS.md`. Shared notes: `refs.bib`, `notes/PRIOR_WORK_VERIFIED.md` (the only
  source for statements about other papers), `notes/LIKELIHOOD_MATH.md`, `notes/GAPS.md`.
- Maintenance loop: `bash working/refresh_papers.sh` (values, audit, figures, PDF, gates, counts)
  -> a Fable consistency pass limited to sentences tagged `% EXPECTED` whose numbers turned real
  -> send the PDF to the owner (SendUserFile) only when it is consistent.
- DONE by 21:15 UTC (commit 9ac3426): the long paper stands alone (`scripts/run_posteriors.py`,
  `scripts/classes.py`; nothing reads the archived letter); the experiment's numbers are macros
  (135, of which 87 real; block `classes` of `values/production_map.json`; audit checks K0 to K2);
  figure `fig_classes` drawn; Fig. 1 from the production event sets; Obs keys mapped to stage 5
  and their placeholders narrowed (skeleton Sec. 7); `refresh_papers.sh` covers one paper; the
  letter is frozen in `working/archive/paper_short_independent/` (`FROZEN.md`).
- DONE by 23:20 UTC, 17 Nov (commits ca7da6e to 5bb62bf): the experiment is in `main.tex` (17 pp,
  ten figures): Sec. IV B `sec:classes` with Fig. 3 `fig_classes` and the equation for the
  probability of an AGN host; the AGN catalog alone as a biased analysis; masses and spins as a
  label (host-dependent subsection); observed-like half-widths for the known population; number
  densities in h^3 Mpc^-3 everywhere; abstract of seven sentences and fifteen numbers; the 90%
  half-width (two significant figures) as the one measure of precision. An independent Fable
  review was applied in full except its length item (`REVIEW_2.md`: findings and decisions). The
  PDF of 5bb62bf minus the HEALPix citation was sent to the owner at 23:12 UTC.
- State of the numbers: 78 of 177 result macros used in the text are real; the experiment's 111
  of 164. Red placeholders of the flux-limited and inferred-density analyses are re-anchored to
  the finished complete-catalog result and kept inside the Table I prior (rules in the header of
  `scripts/placeholder_results.py`).
- Leftovers, none urgent: Table IV prints the AGN-density limit, which in the placeholders shows
  no gain over the priors (galaxy-density limits exist as macros; decide the columns when the
  stage-2 nested runs are real); `fig8_summary` prints "at the prior edge, 100" in a panel with no
  macro behind the number; the abstract has fifteen numbers where a dozen was the aim; the date
  of the fifth observing run in the Conclusions still needs a citable schedule; the conventions
  bullet of `figures/CAPTION_FACTS.md` still says every element drawn from
  `placeholder_results.json` is a placeholder; the refresh can fail once when a run finishes
  between its values step and its audit (rerun it).

## 3. Runs (analysis 16: `working/analyses/analysis_16_production_runs/`, README there)
- Machine: partition RITA-GPU, two A100-80, both ours. (This Claude session itself runs inside
  Slurm job 1412516 on MIKO; never cancel it.)
- Launcher: `PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python`;
  `$PY scripts/launch.py --lanes 2 status`; `... --lanes 2 stage <S ...> [--match REGEX]
  [--only-lane K] [--dry-run]` (always `--lanes 2`); lanes `runs/_lanes/lane1.txt`, `lane2.txt`,
  jobs `agn-prod-l1`, `agn-prod-l2` (each runs its lane file in rank order and chains a
  successor). Ranks: table `LANE_RANKS` at the top of `scripts/make_runlist.py`. Summaries:
  `runs/<run>/summary.json` (block `posterior`; never `grid/result.json`).
- FINISHED: stage 1; every grid of the experiment (E1, E2); the stage-2 grids (flux limits with
  known densities, and m < 20, m < 18 with the AGN density at half and twice).
- QUEUED at 06:15 UTC, 18 Nov, in lane order. Lane 1: stage-2 nested runs (number densities
  inferred; 16 tasks, 3.6 h), then the eight-parameter reference run of realization 100 with
  rslice (33 h; `host_dependent_populations/both_catalogs/complete/host_dependent/realization_100`),
  then the no-catalog runs on host-dependent mergers, E3, and so on. Lane 2: the validation pair
  `.../realization_100_unif_boot5` (stage V8: eight parameters with the six-parameter sampler),
  then stage-2 nested (8 tasks), the single-population runs on host-dependent mergers of
  realizations 100-105 (running), and so on. Everything else that is not an eight-parameter run
  is queued behind (stages 3, 4, 5, E3).
- NOT QUEUED: the other four eight-parameter runs of stage 4, the five of stage 5, and the other
  four V8 pairs. They wait for the sampler decision below.
- THE SAMPLER QUESTION (eight parameters). Test c (job 1414839) is VOID: its uniform-sampling run
  with bootstrap 0 collapsed onto a plane (first bound built from 3 live points, because 198 of
  200 initial points sat on the variance-guard floor at 2000 samples), so its projection of 9000
  calls is the cost of sampling a slice; `tests/results/sampler_test_corrected.json`. Never use
  `--sampler-8 unif_b0`. What is known: rslice costs about 60 calls per iteration (33 GPU-h per
  run); Analysis 13's eight-parameter run with uniform sampling and bootstrap 5 stalled (589k
  calls by iteration 2884), but that run had no variance guard. The validation pair on
  realization 100 measures what uniform sampling with bootstrap 5 costs with the guard. A watcher
  of the session ends 3.5 h after its first part starts (or when it finishes). Then:
  `$PY tests/nested_diagnostics.py runs/<V8 run>/nested_1` (calls per iteration by block,
  corrected insertion-rank test, bound volume against contour volume, flatness of the live set).
  If it runs at well under 60 calls per iteration and passes the diagnostics, let the pair
  finish, compare the two parts and the rslice reference with `tests/compare_sampler_runs.py`,
  and run the other realizations with it; if it stalls, take its two lines out of `lane2.txt`,
  cancel the lane-2 job (its successor carries on), and run all eight-parameter runs with
  rslice (4 x 33 GPU-h more for stage 4, 190 for stage 5).
- EVIDENCE WITH A FLOOR PLATEAU (checked 18 Nov, `tests/test_plateau_evidence.py`,
  `tests/results/plateau_evidence.json`): with the driver's floor of -1e300 for guard rejections,
  ln Z is right at the 75% floor of the six-parameter runs (bias under 0.05 nats); at the 98%
  floor of the eight-parameter runs it is biased by +0.2 to 0.4 nats and its error is 1.5 times
  too small. `scripts/plateau_correction.py` corrects finished runs from what they store; apply
  it (a new field in `prod_summary.py`, by atomic replace) before a Bayes factor with |ln B|
  below about 2 is printed; the paper's are expected near 100 and 35. At a 98% floor 1 to 2.5%
  of runs cannot start (every initial point on the floor, dynesty raises IndexError) and about
  1% crash in the first bound: another seed is the fix.
- STAGE X (queued on lane 1 at rank 2.85, after the stage-2 nested runs): the nside-64 grid of
  realization 100 (turns `\SkyResShiftHzero`, `\SkyResShiftfAGN` black) and two m < 18 grids
  with the AGN density in physical units (expected: the H0 shift under a doubled density turns
  from -0.66 to about +1.1, which would explain Analysis 15's +1.05 by the units).
- TIMING: a stage-2 nested part takes about 45 minutes, three times its estimate, so lane 1's
  block of them ends about 17:30 UTC on 18 Nov and the 33-hour reference run starts then. A
  six-parameter part takes about 2.3 h (0.47 s per call), as estimated. The driver and
  `prod_common.py` were replaced at 07:10 UTC (`--nside`); tasks started since run normally.
- Settings: core e7c3007; hosts normalized over the grid to z = 1.10; mixed sets 8000 posterior
  samples per merger with the half-variance correction; all-AGN sets 32000 samples and six pooled
  injection sets; each realization its own injections; host-dependent analyses use
  `injections/injections_targeted_peak.h5`; observed-like `injections_targeted_obslike.h5`;
  nested runs: nlive 200, dlogz 0.1, six parameters uniform sampling in multi-ellipsoid bounds
  with bootstrap 5, two independent parts merged by the summary.
- Owner-approved scope: stages 1 to 4 with the eight-parameter model on 5 realizations first
  ("Staged: 5, then 10"), plus the observed-like family (stage 5), plus the experiment.

## 4. Results so far (population known unless said; 1000 detected mergers per realization;
90%; inputs H0 = 67.74, f_AGN = 0.30)
- One realization: both catalogs H0 68.4 (+1.9, -2.0), f_AGN 0.27 +- 0.08. Ten combined: H0 68.2
  +- 0.6 (input at the 80% level), f_AGN 0.306 +- 0.026 (30%). No merger in AGN: f_AGN < 0.02.
- 90% half-width of H0, mean over five realizations, against the fraction of mergers in AGN:
  host class unknown 3.58 (0), 3.55 (0.1), 2.04 (0.3), 1.37 (0.5), 0.83 (1); host class known
  3.56, 1.90, 1.34, 1.10, 0.81; no catalog 5.6. With one merger in ten in an AGN the analysis
  gains nothing from the AGN catalog (3.55 against 3.58) although known hosts would give 1.90.
- Probability of an AGN host (f = 0.3, AGN-hosted mergers heavier by 5 Msun and spin +0.10):
  positions alone 0.37 for mergers truly in AGN against 0.30 for those truly in galaxies; with
  masses and spins 0.51 against 0.28.
- Observed-like spectrum: no catalog 8.5, both catalogs at f = 0.3 2.0, all in AGN 0.84.
- AGN catalog alone on mixed mergers: at the prior edge H0 = 100 in four of five realizations;
  realization 102 peaks at 76.8 [75.5, 78.1] and excludes the input.
- Analysis 14 (`analyses/analysis_14_monte_carlo_accuracy/REPORT.md`): 2000 samples per merger
  are too few; the old low f_AGN is the log-mean bias; injections matter as much as samples.
- Analysis 15 (`analyses/analysis_15_host_normalisation/REPORT.md`): the mergers constrain the
  merger rate per host; with complete catalogs no number density enters; at m < 18 an AGN
  density wrong by a factor of two moves f_AGN by 0.13 and H0 by about 1.
- These numbers were made before the summaries moved to the refined lattice; single-realization
  bounds moved by up to 0.09 km/s/Mpc, means over realizations by less. The macros are the source.

## 5. Do next, in order
0. Done 18 Nov 06:50 UTC (commit b37c15d): the flux-limited subsection rests on finished runs:
   all four flux limits with known densities, and the response of f_AGN and H0 to an AGN density
   assumed at half and at twice its value at m < 20 and m < 18 (three realizations; macros from
   `paper_prd/scripts/density.py`, panel (b) of `fig3_flux_limits` redrawn, paragraph, caption
   and abstract sentence rewritten). 137 of 218 result macros in the text are real; the PDF
   (18 pp) went to the owner at 06:50 UTC.
1. Done 07:35 UTC (commit d4af0ed): the evidence test, the sky-resolution run's tooling and
   wiring, the paper chain's selectors (validation stage V8 and check stage X excluded), and
   the two physical-units grids; their three runs are queued (Sec. 3, stage X).
2. The sampler decision of Sec. 3 when the watcher ends (about 11:45 UTC; the validation
   run starts when the single-population run on lane 2 finishes, about 08:10 UTC).
3. Stage-2 nested runs finish (a watcher of the session ends then; lane 1 by about 17:30 UTC,
   lane 2 after its validation pair):
   refresh; the second half of the flux-limited subsection, Table IV and `fig4_densities` turn
   real; decide Table IV's columns (AGN-density limit or galaxy-density limit); consistency
   pass on those sentences (main session, under the skill), PDF to the owner.
4. As six-parameter runs finish (HDSingle, HDNone, then E3, stage 3): refresh and consistency
   pass per subsection.
5. Open points for the text: none that wait for the owner except the confirmations of Sec. 6.
6. Known numerical flags: two catalog-free observed-like grids (all mergers in AGN, realizations
   101 and 103) have ln L jumps of about 100 nats from one heavy merger at the mass cut; no
   number of the paper uses them and audit check K2 guards against it.
7. If an agent dies with an API or billing error (it happened at 06:30 UTC on 18 Nov), its edits
   stay in the working tree: resume it with SendMessage to its id rather than starting over.

## 6. Owner decisions of 18 Nov (multiple choice, about 08:30 UTC) and what is still theirs
- Decided: push (done, 45 commits up to ba8f91f; `origin` is K. Storey-Fisher's GitHub repository,
  each later push is asked for again); no date for the fifth observing run; K. Storey-Fisher's
  affiliation (KIPAC, Stanford); the paper's length is mine to decide ("Whatever makes it
  coherent. Again you own it. You decide."): it stays at about 18 pages; acknowledgments modeled
  on the owner's arXiv:2610.37617 (commit aadae78), with the AI statement adapted to this work.
- Still theirs to confirm (asked 09:00 UTC, none blocks work): that NASA Grant No. 22-LPS22-0025
  (a LISA award, carried over from that paper) covers this work; the AI sentence (it says
  "manuscript drafting and editing" and names Claude models only; add other tools if their other
  sessions used them for code this paper rests on); whether to push as milestones are committed.
- Do not ask the owner editorial questions about the paper (memory: ownership handover).

## 7. Other sessions
`gaia` (left RITA-GPU; nothing owed), `darksirens` (gave the style note
`paper_prd/notes/OWNER_STYLE_DETAILS.md`; nothing owed). Never touch other sessions' jobs.

Commits up to ba8f91f are pushed; local since then: aadae78 plus the one that adds this file; `git log --oneline | head -20`.
