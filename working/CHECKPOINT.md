# Checkpoint, 21 Nov 2026, 07:15 UTC (read this first after a context clear)

You are the Fable session that owns the gws-agn project (owner handover of 16 Nov 2026: full
technical and creative freedom; Fable agents write and review, Opus agents code and submit jobs,
you orchestrate and check). The memory files `gw-agn-paper-prd.md`,
`gw-agn-ownership-handover.md`, `writer-briefs-must-follow-style-skill.md` hold the owner's words;
this file holds the working state.

## THE OFFSET IS A FLUCTUATION OF THE FIRST TEN (21 Nov 12:35 UTC; read first)
- RESULT of the ten new realizations (111 to 120; twenty grids finished 12:33 UTC, none
  failed; `tests/skyres_sequence.py`, block `KnownBoth_production_111_120`; printout in the
  scratchpad `skyres_sequence_1121c.txt`). Two-catalog analysis, population known, combined
  H0 with the level of the input 67.74:
  NEW TEN ALONE (the independent test): nside 32 67.59 (+0.65, -0.63), 30%; nside 64 67.85
  (+0.77, -0.73), 19%. f_AGN 0.282 +- 0.027 (74%) and 0.277 +- 0.026 (85%).
  ALL TWENTY: nside 32 67.91 +- 0.45, 47%; f_AGN 0.294 +- 0.019, 40%. nside 64 68.27 (+0.46,
  -0.48), 93%; f_AGN 0.297 (+0.020, -0.018), 17%.
  FIRST TEN (as before): 68.20, 80%; 68.54, 98%.
  Paired shift of the H0 median, nside 64 minus 32: new ten +0.36 (sd 0.87, se 0.27; seven of
  ten positive: +1.22, +1.50, +1.53, -0.75, -0.27, -0.85, +0.43, +0.64, +0.08, +0.04); all
  twenty +0.31 (sd 0.67, se 0.15); half-width ratio 0.90; f_AGN +0.007 (sd 0.014).
  READING (mine): at the pixels used throughout the analysis recovers the input (twenty, 47%;
  the independent ten, 30%): the +0.5 of the first ten was a fluctuation, and so was the
  unknown-minus-known difference on the first five. With finer pixels the median of these
  sets rises by about 0.3 on average (0.39 +- 0.08 on five at 32,000 samples and six
  injection sets), 15% of one realization's half-width, while sets with half of the mergers
  in AGN move by -0.25 and all-AGN sets by +0.08: no common sign. The twenty at the finer
  pixels hold the input at the 93% level, the independent ten at 19%. Production pixels stay.
  Realization 117: f_AGN 0.163 [0.081, 0.251] for a realized AGN share of 0.285 (outside its
  90% interval; one of twenty). Realization 112: 0.187 for a realized share of 0.258.
- DECIDED FOR THE PAPER (mine): the two-catalog result with the population known is quoted
  over TWENTY realizations everywhere (key `KnownBoth` = stage 1's ten plus the stage-X
  `_nside32` runs of 111 to 120; `\NrealKnown` = twenty; Sec. IV A, Table III, Figs. 2a, 9a,
  summary). The Validation's last paragraph is rewritten: no more "we cannot yet say"; the
  host-class-known comparison goes out of the text (it localized an offset that is not
  there). An Opus agent (started 12:40 UTC) makes the key change, the macros
  `\SkyResAll{Nreal,HzeroComb,HzeroCL,fAGNComb,fAGNCL,MeanShiftHzero,SdShiftHzero,
  MeanShiftfAGN,SdShiftfAGN,NarrowPct,ShiftPctHalfWid}`, `\SkyResHalf{Mean,Sd}ShiftHzero`
  (half of the mergers in AGN), `\SkyResAGN{Mean,Sd}ShiftHzero`, audit checks, figures, and
  two leftovers (`\HDNoneHzeroCombUL`; the in-panel number of `fig8_summary`). It does not
  touch `main.tex`.
- WHEN IT REPORTS: check its changed-macro table against the numbers above; then edit
  `main.tex` (style skill loaded): (1) Validation, first paragraph: drop "and at the
  \SkyResTenHzeroCL\% level with finer sky pixels, a difference that we examine at the end of
  this section"; (2) Validation, last paragraph, DRAFT: "Finally, we repeat the analysis with
  pixels half as wide. [two sentences on the Monte Carlo variance, as now.] With the
  population known and both catalogs, the 90% interval of H0 from one realization narrows by
  \SkyResAllNarrowPct% and its median moves by \SkyResAllMeanShiftHzero on average over the
  \NrealKnown realizations, with a standard deviation of \SkyResAllSdShiftHzero between
  them, against a 90% half-width of \KnownBothHzeroHalfWid from one realization. With
  \MCNsampHi posterior samples per merger and \SkyResMCNsets sets of simulated signals, on
  \SkyResNreal of the realizations, the mean shift is \SkyResMCMeanShiftHzero. The median of
  f_AGN moves by \SkyResAllMeanShiftfAGN on average, with a standard deviation of
  \SkyResAllSdShiftfAGN. The shift has no common sign: with half of the mergers hosted by AGN
  the median of H0 moves by \SkyResHalfMeanShiftHzero on average over \NrealClasses
  realizations, and with all of them hosted by AGN by \SkyResAGNMeanShiftHzero. Combined over
  the \NrealKnown realizations, the finer pixels give H0 = \SkyResAllHzeroComb with the input
  at the \SkyResAllHzeroCL% credible level, where the pixels used throughout give
  \KnownBothHzeroComb with the input at the \KnownBothHzeroCL% level. The pixel size thus
  moves H0 by \SkyResAllShiftPctHalfWid% of the 90% half-width that \MockNev mergers give.";
  (3) Table III, HDNone row: an upper limit, not an interval; Sec. IV F's sentence quoting
  `\HDNoneHzero` (line ~1156): reword as an interval against the prior range, as Sec. IV E
  does. Then `bash working/refresh_papers.sh` (exit 0), re-read the changed sentences and the
  abstract's sentence on \NrealKnown, commit, PDF to the owner (named copy), and the push
  question again.
- RUNS: lane 1 (job 1425268, successor 1494378) resumed the eight-parameter run of
  realization 101 at iteration 3288 at 12:34 UTC (ends about 04:00 UTC 22 Nov), then 102.
  Lane 2 (job 1454491) on realization 103, iteration 1704 at 12:33 UTC. No watcher is armed.
- PENDING WITH THE OWNER: push the corrected Validation paragraph (local commits 8ed540e,
  f1befea and this one; GitHub has 3e060b3, whose last Validation sentence is too general)?
  Options given: A push now (recommended), B after the twenty-realization result, C hold.

## 0. How to resume (the owner clears the context and says "resume from the latest checkpoint")
1. Read this file to the end. Do not re-read the paper or the scripts until a step needs them.
2. Look at the state: `git -C /hildafs/projects/phy230014p/magana/gws-agn log --oneline | head -5`;
   `squeue -u $USER -p RITA-GPU`; `$PY scripts/launch.py --lanes 2 status` in the analysis-16
   directory (Sec. 3 has `PY`).
3. Nothing is owed right now except waiting: lane 1 is on the eight-parameter reference run of
   realization 100, lane 2 on the pixel-size sequence (Sec. 3). The next actions are Sec. 5,
   items 1 and 2, each triggered by a watcher. THE OPEN QUESTION OF THE DAY is Sec. 4, "Pixel
   size": read it before anything else is decided about H0 numbers.
4. WATCHERS survive a context clear and report as background-task notifications that carry no
   context. Two are armed:
   - task `bxzjrwxqb` (21 Nov 07:16 UTC), "Wait until the twenty diagnostic grids have finished
     or failed, or until 10:30 UTC": D1 and D2 (host class known and unknown at nside 64 on
     `peak_f0p30_s0` and `peak_f0p50_s0`, five realizations) were queued on lane 2 at 07:13 UTC;
     lane 2's job 1416415 was cancelled by me at 07:13 so that its successor 1454491 runs them
     first and then RESUMES the eight-parameter run of realization 103 from its checkpoint
     (check in `logs/agn-prod-l2_1454491.out` that it did). Then: `tests/skyres_sequence.py`
     (new blocks by the diagnosis agent) and the agent's `tests/mixture_decomposition.py`.
     D3 (f_AGN held) cannot be run: `prod_model.build` does not fix `fcat_2`; the agent reads
     the f = 0.30 line of the stored two-dimensional grids instead.
     (`b1oyi3kaz` is a timer for 08:30 UTC; no action. Two Opus agents are at work: the
     diagnosis agent, Parts 2 and 3, and the macros agent for the Validation paragraph.)
     The agent's catalogs for seeds 106 to 110 (`data/seedNNN/catalogs`, 5.4 G each, regenerated
     to build the nside-64 survey files; the 100-seed campaign had pruned them) can be deleted
     again when no further pixel size is needed for those seeds.
     LESSON of 20 Nov: a watcher that waits for "all N summaries" never fires when one run
     fails; the decision point of 16:00 UTC passed unseen for that reason. Watchers must also
     end when the lane has moved past the tasks (count finished OR failed OR errored attempts)
     and must carry a time limit a little after the expected end.
   (`bue7u6pya`, both eight-parameter runs of realization 100, reported 20 Nov 07:12 UTC: the
   comparison is done, see Sec. 4; what is left of Sec. 5 item 2 is the agent's part.)
   (`bk2x1ic8d` and `bvv0w6obd`, the sky-resolution pairs of the five realizations, reported
   on 19 Nov 21:45 UTC and 20 Nov 04:40 UTC: done.)
   A notification from any other old task (earlier watchers, earlier agents) needs no action:
   check what it printed and move on. If the session was restarted, the watchers are gone:
   check the two conditions by hand.
5. Tell the owner in a few lines what the state is and what you are waiting for; do not ask
   them editorial questions (Sec. 1).

### Rules that go into every agent brief (they were in force all through 17 and 18 Nov)
- Opus agents for code, numbers, figures and run lists; Fable agents (or yourself) for prose and
  reviews. One agent per coherent job; sequential when two would touch the same scripts.
- No sbatch, scancel or scontrol by an agent: it prepares, gives the launcher command line with
  its `--dry-run` output, and you queue. No git commit by an agent: you commit after checking.
- Two lane jobs start `prod_driver.py` (importing `prod_common.py`, `prod_model.py`) fresh for
  every task and `prod_summary.py` after every run: such files are changed only in a copy,
  tested on CPU, and installed by one atomic `mv`, with every existing option and default
  unchanged; never `lane_worker.py`; lane files and run lists only through `make_runlist.py`
  and `launch.py` (you may reorder a lane file by hand, atomically). After a driver change,
  check that the next task on each lane starts normally.
- Paper agents do not edit `main.tex` unless they are the writer; the writer edits only
  `main.tex`. Every brief for text says: load the skill `astro-paper-style` first, it governs.
- Stages V8 (validation) and X (checks) never feed a paper key (selectors exclude them).
- Check an agent's claims yourself on a sample (recompute a number, read the diff, view the
  figure) before using them, and say in the report to the owner what was verified.
- A PDF goes to the owner with SendUserFile as a named copy in the scratchpad
  (`dark_sirens_galaxies_and_AGN_draft_<date><letter>.pdf`), only after a refresh that exits 0.

## 1. What the owner wants (standing)
- ONE paper: `working/paper_prd/` (owner, 17 Nov: "Lets stick to the long paper"). The letter is
  frozen in `working/archive/paper_short_independent/`. Do not propose a short version.
- What is new, in the owner's words: "proper multi-tracer darksirens, and correlated gw
  properties with galaxies".
- The draft is written for the expected results at once; what waits for a run is red (macros
  `\ph{}`, figure data); no note to the author is printed in a figure or in the text.
- Form: the owner's style skill `astro-paper-style`. LOAD IT before briefing a writer or writing
  text yourself; a brief adds content and never overrides the skill. Bayesian statements only,
  90% credible intervals, precision as the 90% half-width (two significant figures), no em dashes.
- Editorial and structural choices about the paper are YOURS: do not ask the owner (18 Nov:
  "Whatever makes it coherent. Again you own it. You decide."). Ask only for facts they alone
  have and for actions outside the repo.
- Commit locally on `main`. `origin` is K. Storey-Fisher's GitHub repository (she is second
  author); the owner approved one push on 18 Nov (up to ba8f91f); ask before each later push.
  Attribution lines: `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01L6W2GrLCXpR9r5AzCuhHp8` (changed 21 Nov;
  use the one in the latest system reminder).
- Private: never put the owner's name, the project name or local paths into outside requests
  (the owner did ask for the lookup of their own arXiv paper on 18 Nov).
- Agents: no idle waiting; code watchers; fresh agents with short briefs; proportionate numbers.
  If an agent dies with an API or billing error its edits stay in the working tree: resume it
  with SendMessage to its id.

## 2. The paper (state at the commit of 19 Nov that adds this file's date; 18 pp, ten figures)
- `working/paper_prd/main.tex`. Design: `notes/PAPER_SKELETON.md` (Sec. 9: the experiment taken
  in; 9.6 conventions), `REVISION_DECISIONS.md` item 13 (owner's words),
  `notes/OWNER_STYLE_DETAILS.md`, `REVIEW_2.md` (independent review of 17 Nov, applied).
- Numbers chain (`scripts/`): `build_values.py`, `production.py` + `values/production_map.json`
  (keys), `classes.py` (the experiment), `density.py` (assumed AGN density, units), `skyres.py`
  (sky resolution), `placeholder_results.py` (red values and their rules), `audit_values.py`
  (second code path; 78 checks pass), `make_figures.py`, `figures/CAPTION_FACTS.md`,
  `values/MACROS.md`. `notes/PRIOR_WORK_VERIFIED.md` and `notes/CITATION_CHECKS.md` are the only
  sources for statements about other work.
- Loop: `bash working/refresh_papers.sh` (about 6 min; exit 0 required; rerun once if a GPU run
  finished mid-refresh) -> re-read every sentence whose macros turned black (tag `% EXPECTED`
  marks sentences worded for unfinished runs; 36 left) -> commit -> PDF to the owner with
  SendUserFile (a named copy in the scratchpad).
- Real now: 159 of 217 result macros used in the text. Final sections: known population (IV A),
  the constraint from each host class (IV B, Fig. 3; two points wait for the eight-parameter
  runs), flux-limited catalogs (IV C, Table IV, Figs. 4 and 5), most of Validation. Red: unknown
  population (IV D, stage 3), host-dependent populations (IV E, stage 4), observed-like (IV F,
  stage 5), the abstract's headline, Table III rows of those, Figs. 6 to 10 where they use them.
- Front matter: authors I. Magana Hernandez (CMU) and K. Storey-Fisher (KIPAC, Stanford);
  acknowledgments modeled on the owner's arXiv:2610.37617. OWNER, 21 Nov: NASA Grant No.
  22-LPS22-0025 does NOT cover this paper (funding sentence removed, uncommitted edit of
  `main.tex` until the next refresh; the right grant is theirs to supply); the AI sentence stays
  as written; ONE push after the pixel test, once the Validation says what the result means.
- Sentences to revisit when their runs finish: Validation, sky resolution (its numbers are real
  since 20 Nov: H0 median +0.39 on average, sd 0.18, f_AGN +0.005, sd 0.017; the sentence still
  carries `% EXPECTED` because it does not yet say that the shift is common to the realizations
  nor what it means: word it from the pixel-size sequence, Sec. 5 item 1); Sec. IV E and the
  abstract when HDSingle/HDHost are real; `fig8_summary` prints "at the prior edge, 100" with no
  macro behind the number.

## 3. Runs (analysis 16: `working/analyses/analysis_16_production_runs/`, README there)
- Machine: partition RITA-GPU on the Vera cluster (PSC), two A100-80, both ours. This Claude
  session itself runs inside Slurm job 1412516 on MIKO; never cancel it.
- Launcher: `PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python`;
  `$PY scripts/launch.py --lanes 2 status`; `... --lanes 2 stage <S ...> [--match REGEX]
  [--only-lane K] [--dry-run]` (always `--lanes 2`; dry run first). Lanes
  `runs/_lanes/lane{1,2}.txt`, jobs `agn-prod-l1`, `agn-prod-l2` (each runs its lane file in
  order and chains a successor; a cancelled job's successor carries on from checkpoints).
  Order = `lane_rank` per run (table `LANE_RANKS` in `scripts/make_runlist.py`); a lane file may
  be reordered by hand (atomic rewrite), `launch.py` re-sorts it at its next call.
  Summaries: `runs/<run>/summary.json`, block `posterior` (never `grid/result.json`).
  Diagnostics of a nested part: `tests/nested_diagnostics.py <part dir>` (core venv python
  `/hildafs/projects/phy230014p/magana/envs/darksirens-core-e7c3007/bin/python`, `JAX_PLATFORMS=cpu`).
- FINISHED: stage 1; the experiment's grids (E1, E2); all of stage 2 (grids and nested); stage X
  first five runs (nside 64 at 8000 samples; two m < 18 grids with the AGN density in Mpc^-3;
  the sky-resolution pair of realization 100 at 32,000 samples and six injection sets); the
  mixed-sampler eight-parameter run of realization 100 (`..._unif_then_rslice`, 5293 iterations,
  200k calls, ln Z = -2899.18 +- 0.37; finished 19 Nov 20:44 UTC).
- STATE 21 Nov 00:30 UTC: lane 1 (job 1416219, successor 1425268) on the mixed-sampler
  eight-parameter run of realization 101 since 20 Nov 16:16 UTC (about 29 h, nside 32), then
  that of 102. Lane 2 (job 1416415, successor 1454491) on HDSingle realization 105 (second
  part ends about 02:20 UTC), then the mixed-sampler runs of 103 and 105. HDNone is done (nine
  tasks), HDSingle 101 to 103 done. Both realization-100 eight-parameter runs finished.
  The 20 Nov text follows.
- STATE 20 Nov 05:35 UTC (the two bullets RUNNING and QUEUED below are the 18 Nov text, kept for
  the order of the queue): lane 1 on the rslice reference of realization 100 (iteration 4834 at
  04:35 UTC, about 126 per hour; the mixed run needed 5093: ends about 07:00 UTC; job 1416192
  ended at its limit, its successor 1416219 carries on, 1425268 queued behind), then HDNone
  (nine tasks, about 9 h), then the mixed-sampler run of realization 101 (about 29 h; would
  start about 16:00 UTC 20 Nov: DECIDE THE PIXEL SIZE BEFORE THAT, Sec. 5 item 1). Lane 2 (job
  1415023 until its limit at 12:23 UTC, successor 1416415): the first part of HDSingle
  realization 101 until about 06:45 UTC, then the PIXEL-SIZE SEQUENCE queued 20 Nov 05:17 UTC
  (stage X, 50 runs, about 3.5 GPU-h, done about 10:30 UTC), then HDSingle 101 to 105 and the
  rest as below.
- PIXEL-SIZE RUNS (stage X, realizations 100, 101, 102, 103, 105; survey files
  `data/seedNNN/surveys_ns{16,64,128}/`, all checks passed). Finished: KnownBoth at nside 32 and
  64, at 8000 samples and at 32,000 samples with six injection sets (`realization_NNN_nside64`,
  `..._s32000_inj6`, `..._nside64_s32000_inj6`). Queued (the sequence): nside 16, 64, 128 for
  the three one-catalog analyses (`known_population/agn_catalog/complete/agn_hosted/
  realization_NNN_nside<N>_s32000_inj6`; `known_population/galaxy_catalog/complete/
  {galaxy_hosted,equal_populations}/realization_NNN_nside<N>`), then KnownBoth at nside 128
  (`..._nside128_s32000_inj6`). In the run list but NOT queued: KnownBoth at nside 16 (five
  runs, 8 to 11 GPU-h, GPU memory untested; queue only if the trend toward coarse pixels is
  needed: `launch.py --lanes 2 stage X --only-lane 2`, dry run first). Table of all of it:
  `JAX_PLATFORMS=cpu $PY tests/skyres_sequence.py` -> `tests/results/skyres_sequence.json`
  (paired shifts to nside 32, posteriors combined over the five with the paper chain's own
  function, credible level of the inputs, Monte Carlo variances; missing runs are listed).
- DISK: `working/data/seedNNN` are links into `/hildafs/projects/phy230054p/` (15 T free); the
  run outputs and the repo sit on `phy230014p`, which is 92% full (18 G free on 19 Nov).
- RUNNING since 18 Nov: lane 1 (job 1416192, successor 1416219 queued), the eight-parameter
  reference of realization 100 with rslice from the start
  (`host_dependent_populations/both_catalogs/complete/host_dependent/realization_100`, started
  17:50 UTC; at 20:49 UTC iteration 584, 32.6k calls); lane 2 (job 1415023, successor 1416415
  queued), the continuation `.../realization_100_unif_then_rslice` (from the stopped uniform run
  at iteration 1594, rslice from there; started 16:55 UTC; at 20:49 UTC iteration 2112, 60.5k
  calls). About 4700 to 5100 iterations are needed in all; at the present rates (about 195 and
  135 iterations per hour) both end in the evening of 19 Nov, UTC.
- QUEUED behind them, in order. Lane 1: no-catalog single-population runs on host-dependent
  mergers (HDNone), `..._unif_then_rslice` of realizations 101 and 102 (automatic switch, about
  30 h each), E3, HDSingle 106 to 110, stage 3, stage 5 six-parameter runs. Lane 2: the
  sky-resolution pair at 32,000 samples and six injection sets (stage X, 1.2 GPU-h), HDSingle
  of realizations 101 to 105, `..._unif_then_rslice` of 103 and 105, E3, HDSingle 106 to 110,
  stage 3, stage 5 six-parameter runs.
- NOT QUEUED: the pure-rslice stage-4 runs of 101 to 105 and the five eight-parameter runs of
  stage 5 (they wait for the comparison below).
- THE EIGHT-PARAMETER SAMPLER (decided 18 Nov). Uniform sampling (bootstrap 5) is healthy and
  cheap for about 1500 iterations and then stalls (150 calls per iteration against rslice's 60);
  test c was void (bootstrap 0 collapses onto a plane: never use `--sampler-8 unif_b0`). The
  runs therefore sample uniformly while that is cheaper and switch to rslice (driver options
  `--auto-switch-sample rslice --auto-switch-cpi 60`, or `--init-from STATE --switch-sample
  rslice`); stage V8 holds the five such runs (`..._unif_then_rslice`). VALIDATION when both
  realization-100 runs have finished (evening of 19 Nov, UTC):
  `JAX_PLATFORMS=cpu <core venv python> tests/compare_sampler_runs.py reference=runs/.../realization_100/nested_1 mixed=runs/.../realization_100_unif_then_rslice/nested_1 --out tests/results/compare_r100_eight_parameter.json`
  on the two part directories (medians in units of the half-width, distance between posteriors
  against its null, ln Z difference, insertion-rank diagnostics). If they agree: map the paper's
  keys `HDHost` (and `HWHost`) to the `..._unif_then_rslice` runs in
  `paper_prd/scripts/production.py` (stage V8 is excluded from the selectors by design, so this
  is a deliberate edit), queue stage 5's eight-parameter runs the same way (new run-list entries),
  and drop the unqueued pure-rslice entries. If they disagree: run 101 to 105 with rslice from
  the start and find out why first.
- EVIDENCE: with the driver's floor of -1e300 for guard rejections ln Z is right at the 75% floor
  of the six-parameter runs and biased by +0.2 to 0.4 nats (error 1.5 times too small) at the
  98% floor of the eight-parameter runs (`tests/results/plateau_evidence.json`).
  `scripts/plateau_correction.py` corrects finished runs; apply it in `prod_summary.py` (new
  field, atomic replace) before a Bayes factor with |ln B| below about 2 is printed. A run
  that cannot start at a 98% floor (IndexError) needs another seed.
- COST, measured: a six-parameter part 2.3 h; a stage-2 nested part 0.75 h; the eight-parameter
  model 0.50 s per call (the run list's estimates for it are about 14% low). Remaining queue:
  about 65 GPU-h of eight-parameter work in progress for realization 100, 120 for 101 to 105,
  then about 150 for stage 5's five, plus about 150 of six-parameter work; two GPUs, about ten
  days in all, the host-dependent headline about 23 Nov.
- Settings: core e7c3007; hosts normalized over the grid to z = 1.10; number densities in
  h^3 Mpc^-3 from catalog counts; 8000 posterior samples per merger with the half-variance
  correction (all-AGN sets 32,000 samples and six pooled injection sets); nside 32; nested runs
  nlive 200, dlogz 0.1.
- Owner-approved scope: stages 1 to 4 with the eight-parameter model on 5 realizations first
  ("Staged: 5, then 10"), the observed-like family (stage 5), the experiment.

## 4. Results so far (1000 detected mergers per realization; 90%; inputs H0 = 67.74, f_AGN = 0.30)
- Known population, complete catalogs: one realization H0 68.4 (+1.9, -2.0), f_AGN 0.27 +- 0.08;
  ten combined 68.2 +- 0.6 and 0.306 +- 0.026. No merger in AGN: f_AGN < 0.02.
- Half-width of H0 against the fraction of mergers in AGN (means over five realizations): host
  class unknown 3.5 (0), 3.5 (0.1), 2.0 (0.3), 1.4 (0.5), 0.82 (1); host class known 3.5, 1.9,
  1.3, 1.1, 0.81; no catalog 5.6 (observed-like 8.4). Cost of the unknown host class in mergers:
  3.4 [1.6, 7.2] at 0.1, 2.3 [1.8, 3.4] at 0.3; with host-dependent populations known 1.6
  [1.3, 2.3]; with twice the offsets 1.1 [0.8, 1.7].
- Probability of an AGN host: positions alone 0.33 against 0.24 (evaluated at f = 0.27); with
  host-dependent masses and spins 0.51 against 0.28; twice the offsets 0.72 against 0.21.
- AGN catalog alone on mixed mergers: biased (prior edge in four of five realizations, 76.8 in
  the fifth).
- Flux limits with known densities: H0 68.4, 68.4, 68.0, 68.3 at m < 21, 20, 19, 18 (half-width
  1.9 to 2.2). Assumed AGN density at twice its value: f_AGN +0.145 at m < 20 (as the identity
  predicts, H0 unmoved), +0.10 and H0 -0.68 at m < 18; with the density given in Mpc^-3 the H0
  shift is +1.62 (units, not event set, explain the old Analysis 15 sign).
- Number densities inferred (m < 18): rate ratio 41 (+18, -14) for an input of 42.8, f_AGN
  follows its prior, H0 68.1 (+2.3, -2.2), an upper limit on the galaxy density and none on the
  AGN density.
- THE OFFSET AFTER THE DIAGNOSTIC GRIDS (21 Nov 09:15 UTC; read this BEFORE the 07:00 entry
  below, which it corrects). `tests/skyres_sequence.py` has blocks for all of it. Combined H0
  over five realizations (100 to 105), host class unknown / known, level of the input:
  f = 0.3 (set `peak_f0p30_s0`, the same mergers as stage 1's first five, another injection
  set): nside 32 68.46 (82%) / 67.83 (19%); nside 64 68.53 (83%) / 67.77 (6%).
  f = 0.5 (`peak_f0p50_s0`): nside 32 67.90 (35%) / 67.63 (30%); nside 64 67.55 (43%) / 67.33
  (86%, low side). f = 0.1: 67.49; f = 0: 67.83; f = 1: 67.68 +- 0.36.
  Unknown minus known on the same mergers: f = 0.3 +0.56 (nside 32) and +0.79 (nside 64), five
  of five positive at both; f = 0.5 +0.45 and +0.50 (four and three of five). BUT the unknown
  analysis is noisier than the known one, so the null scatter of that difference is about 0.8
  per realization (0.35 for a mean of five): these are 1.5 to 2.5 sd, and the four numbers
  share events or realizations. f_AGN held at its input changes nothing (-0.02, -0.06 over
  ten): not an H0-f_AGN degeneracy.
  The pixel shift is NOT uniform: +0.39 +- 0.08 on stage 1's five at 32,000 samples and six
  sets, +0.26 +- 0.14 on ten at 8000, but +0.12 +- 0.36 on the same five with the other
  injection set, and -0.25 +- 0.19 (unknown) and -0.30 +- 0.13 (known) at f = 0.5. With one
  injection set the selection term at nside 64 (N^2/N_eff 2.5) moves single medians by about
  0.5 between injection sets.
  WHAT IS ESTABLISHED: the sets with f = 0.3 lie +0.5 (nside 32, ten realizations, 80%) to
  +0.8 (nside 64, 98%) above the input; the analysis with the host class known, the AGN-only
  analysis and the f = 0.5, 0.1, 0 sets do not. Property of the mixture analysis or fluctuation
  of ten sets (the first five share events between "stage 1" and "E1"): NOT KNOWN. The
  Validation's last paragraph says exactly this since the local commit after 3e060b3 (the
  pushed 3e060b3 says "the analysis places H0 above its input ... origin not identified",
  which is too general: ASK THE OWNER before pushing the correction; asked 21 Nov 09:20 UTC).
  PER-MERGER DECOMPOSITION (done 21 Nov 09:45 UTC; `tests/mixture_decomposition.py`,
  `tests/results/mixture_decomposition_{103,100}.json`; f_AGN held at 0.30, set
  `peak_f0p30_s0`, reproduces the stored grids to 1e-11). Going from host class known to the
  mixture, change of the posterior mean of H0 (realization 103 at nside 32 / 64; 100 at 32 /
  64): the galaxy-hosted mergers' term, sum of ln[(1 - f) + f A_i/G_i]: +0.50 / +1.26; +0.55 /
  +0.95 (alone on the known posterior +0.18, +0.50, +0.26, +0.54); the AGN-hosted mergers'
  term, sum of ln[f + (1 - f) G_i/A_i]: -0.47 / -0.80; +0.27 / -0.28; the selection term at
  most +0.03; the Monte Carlo correction terms at most +0.14; AGN-hosted mergers at 8000
  instead of 32,000 samples: +0.05 / +0.47; -0.32 / -0.20. Share f A/(f A + (1 - f) G) of
  galaxy-hosted mergers 0.27 on average (AGN-hosted 0.36 to 0.37). READING (mine): no faulty
  term was found. A correct mixture has exactly this structure: the galaxy-hosted mergers'
  AGN term pulls H0 up, the AGN-hosted mergers' galaxy term pulls it down, and only their sum
  over all mergers has zero mean at the input; finer pixels enlarge both pulls. The net left
  over is of the size of the noise of the difference. So the decomposition does not decide
  between a property of the analysis and a fluctuation; the twenty realizations do.
  D3 as GPU runs would need `fixed_survey["fcat_2"] = f` in `prod_model.Model.build` when
  f_AGN is not free (change only when no lane job imports the module).
  NEXT: (1) done, above; (2) TEN MORE REALIZATIONS of the f = 0.3 two-catalog analysis at nside 32 and
  64 (corepop events for seeds 111 to 120 must be drawn: `data/corepop/README.md`; catalogs of
  those seeds were pruned and must be regenerated with
  NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" for bitwise agreement; survey files
  at nside 64; about 2.5 GPU-h as stage-X grids): brief an Opus agent when the node is less
  loaded (load 68 on 32 cores at 08:30 UTC) and queue on lane 1 with `--only-lane 1` BEFORE
  its eight-parameter run of realization 101 ends (about 18:00 UTC 21 Nov), so that they run
  ahead of realization 102 without interrupting anything. Twenty realizations give a combined
  half-width of about 0.4: an offset of 0.8 would then be unmistakable.
- THE H0 OFFSET OF THE TWO-CATALOG ANALYSIS (21 Nov 07:00 UTC; THE OPEN PROBLEM). The pixel test
  on realizations 106 to 110 confirmed it. KnownBoth over ten realizations at the production
  settings: nside 32 68.20 (+0.58, -0.59), input at the 80% level; nside 64 68.54 (+0.53,
  -0.57), input at the 98% level; paired shift +0.26 +- 0.14 over ten (+0.26 +- 0.20 and +0.26
  +- 0.21 in the two groups of five); f_AGN 0.306 -> 0.319 (+-0.026; shift +0.015 +- 0.004 at
  8000 samples, +0.005 +- 0.007 at 32,000). LOCALIZED from existing nside-32 grids of the
  experiment (five realizations, same mergers analysed twice): host class unknown 68.46 (+0.81,
  -0.89) against host class known 67.83 +- 0.61 at f = 0.3, unknown minus known positive in all
  five (+0.49, +0.52, +0.61, +0.09, +1.07; mean +0.56); at f = 0.5 67.90 against 67.63 (+0.45
  +- 0.27); at f = 0.1 no difference (noisy); f = 0: 67.83 (+1.7, -1.4); f = 1 with both
  catalogs 67.68 +- 0.36. So the offset is in the mixture (host class unknown), absent when the
  class is known, and grows with finer pixels; cause unknown. Not Monte Carlo (8000 -> 32,000
  samples and one -> six injection sets change nothing).
  OWNER, 21 Nov 07:00 UTC: "Diagnose, runs continue": interrupt ONE long run for about 4 GPU-h
  of diagnostic grids, keep the production pixels until the cause is known (the switch to finer
  pixels is OFF: I withdrew that recommendation); and "Push now, numbers as measured": write the
  Validation paragraph stating the offset plainly, refresh, push today.
  IN PROGRESS: an Opus agent prepares stage-X diagnostics (D1 host class known at nside 64 for
  f = 0.3 and 0.5; D2 host class unknown at nside 64 on the same sets; D3 KnownBoth with f_AGN
  held at its input, one-dimensional H0 grid, ten realizations, nside 32 and 64), new blocks of
  `tests/skyres_sequence.py`, and a per-merger decomposition on CPU
  (`tests/mixture_decomposition.py`: catalog terms A_i, G_i against H0 by true host class,
  realization 103 then 100). It sends "ready to queue"; I check its proof, queue on lane 2, and
  `scancel` lane 2's RUNNING job only (the successor takes the stage-X tasks first, then resumes
  the eight-parameter run of realization 103 from its checkpoint). Then a second Opus agent
  adds the macros of the Validation paragraph (draft and macro names in the session's notes
  below), I edit `main.tex`, refresh, commit, push, PDF to the owner.
  DRAFT of the Validation paragraph (replaces "Finally, we repeat the analysis with pixels half
  as wide ..."): the variance sentence as now; the high-accuracy comparison on five (rises in
  each, min to max, mean; interval narrower by 12%; f_AGN unmoved); combined over ten at the
  finer pixels with the level of the input beside `\KnownBothHzeroComb` and its level; "The
  offset is absent when the host class of each merger is given": host class known against
  unknown on the same mergers (five realizations), AGN-hosted alone at both pixel sizes; last
  sentence: the analysis with the host class unknown places H0 high by 0.5 (production pixels)
  to 0.8 (finer), less than half of one realization's half-width, origin not identified.
  New macros wanted: `\SkyResMCMinShiftHzero`, `\SkyResMCMaxShiftHzero`, `\SkyResMCNarrowPct`,
  `\SkyResTenNreal`, `\SkyResTenHzeroComb`, `\SkyResTenHzeroCL`, `\SkyResOffsetHzeroProd`,
  `\SkyResOffsetHzeroFine`, `\ClsKnownThirtyHzeroComb`, `\ClsUnknownThirtyHzeroComb`,
  `\SkyResAGNHzeroCombSixtyFour`.
- PIXEL SIZE, THE SEQUENCE (21 Nov, `tests/results/skyres_sequence.json`; paired H0 shift to
  nside 32, mean +- standard error over five; combined H0 with the level of the input):
  AGN-hosted, AGN catalog alone: nside 16 -0.21 +- 0.14 (67.47 +- 0.40, 73%), nside 32 (67.69
  +- 0.36, 18%), nside 64 +0.08 +- 0.13 (67.68 +- 0.37, 20%); nside 128 cannot be evaluated
  (ln L = -inf at every H0 at 32,000 samples: a merger without Monte Carlo support; five tasks
  that fail in a minute and are retried by every lane job until held). Galaxy-hosted, galaxy
  catalog alone: -0.16 +- 0.36, -, -0.30 +- 0.47, -1.2 +- 0.9 (levels 2%, 24%, 9%, 80%).
  Galaxy catalog alone on mixed mergers: -0.11 +- 1.07, -, +0.18 +- 0.89, +0.77 +- 1.48 (34%,
  54%, 16%, 9%). Both catalogs: nside 64 +0.39 +- 0.08 (68.74, 96%), nside 128 +0.23 +- 0.26
  (68.58 +0.83 -0.91, 87%); f_AGN unmoved at both. READING: no one-catalog analysis moves with
  the pixel; the two-catalog analysis moves up by 0.2 to 0.4 and does not go on rising at
  nside 128, where the Monte Carlo variance is three to four times larger (posterior samples
  2.4 to 4.3, selection 1.1 to 1.4) and the scatter of the shift 0.58. So the fine-pixel
  two-catalog posterior of five sits 0.8 to 1.0 above the input at the 87 to 96% level:
  fluctuation of five or offset, not known. OWNER, 21 Nov: "Test first": nside 64 at the
  production settings on realizations 106 to 110 (they have one injection set each), then I
  decide; a switch of the two-catalog analyses to finer pixels is approved if the offset is
  confirmed (about 3 days of both GPUs, headline about 26 Nov instead of 23).
- PIXEL SIZE (state of 20 Nov). KnownBoth at 32,000 samples and six injection sets, nside 64 minus
  nside 32, realizations 100, 101, 102, 103, 105: H0 median +0.42, +0.18, +0.31, +0.67, +0.40
  (mean +0.39, sd 0.18, standard error 0.08: common to the realizations, not scatter); f_AGN
  +0.005 (standard error 0.007); the 90% half-width of H0 shrinks by 12% (mean 2.04 -> 1.80),
  by the lower edge of the interval moving up (+0.2 to +1.3) while the upper edge stays
  (-0.2 to +0.4). At 8000 samples the mean shift is +0.26 +- 0.20 (noisier, same sign in
  three of five). The Monte Carlo variances double with each halving of the pixel (posterior
  samples 0.6 -> 1.1, selection 0.22 -> 0.43 at 32,000 samples), but going from 8000 to 32,000
  samples at nside 32 moves the medians by -0.07 on average, so the shift is not Monte Carlo
  error. Combined over the five (paper chain's function): nside 32 H0 68.22 (+0.89, -0.91),
  input 67.74 at the 62% level; nside 64 68.74 (+0.74, -0.79), input at the 96% level; f_AGN
  0.312 and 0.318 (+-0.037), input at the 41% and 59% levels. The mock's sky localization is
  1 to 1.75 deg per axis (35 deg over the amplitude, clipped), an nside-32 pixel is 1.83 deg
  wide: nside 32 does not resolve it, so more information at nside 64 is expected; a common
  shift of the median is not. NOT KNOWN: whether nside 64 is nearer the truth (then nside 32
  hides part of a positive H0 offset of the two-catalog analysis, about +1.0 at 96%) or whether
  the shift stops; which term carries it (AGN catalog, 31 objects per nside-64 pixel, or galaxy
  catalog, 3075). At nside 32 every combined posterior of stage 1 holds the input (AGN-hosted
  67.69 +- 0.36, input at 18%; galaxy-hosted 68.0, 24%; galaxy catalog alone on mixed mergers
  68.6, 54%). The sequence of Sec. 3 answers both.
- THE EXPECTED HEADLINE OF SEC. IV E IS NOT SHOWING (21 Nov; 4 of 10 and 1 of 5 realizations, so
  no rewrite yet, but word nothing more on the old expectation). Single population (six free
  parameters), both catalogs, host-dependent mergers (HDSingle): H0 67.6 [65.9, 69.7], 66.7
  [64.9, 68.8], 68.9 [66.7, 70.5], 70.5 [68.5, 72.5] for realizations 100 to 103; a rough
  combination is 68.4 +- 1.0, the input inside; the red placeholder says 70.8 +- 0.8, a bias
  of +3.1 (that number came from the exploratory Analysis 10, where the single population was
  held at fixed parameters; with its mass peak and spin free it settles between the classes,
  mu_G 36.2 to 36.8, mu_chi 0.04 to 0.05, and H0 is spared). f_AGN 0.34, 0.28, 0.42, 0.33.
  The host-dependent analysis on realization 100 gives H0 67.5 +- 1.45 against 67.6 +- 1.9
  for the single population (a quarter narrower) and ln B = +19.4 for it (-2899.2 against
  -2918.6). So the section's result is likely: host-dependent masses and spins are detected
  decisively and sharpen H0 by telling the host classes apart; a single population with free
  parameters is not biased in H0 at these offsets. The abstract, the Conclusions' first
  paragraph and Fig. 6's panel (a) are worded for the bias: rewrite them when HDSingle (ten)
  and HDHost (five) are real, about 23 Nov. The owner was told on 21 Nov.
- No catalog, single population, host-dependent mergers (HDNone, five realizations, real): H0
  is barely measured: 78 [61, 94] on realization A, and the lower prior edge in three of the
  other four (57.5 [50.8, 74.5], 59.7 [51, 76], 55 [50.4, 68]; 101: 70 [54.5, 88]). The text
  of Sec. IV E now says so with realization A's interval against the prior range. LEFTOVER:
  Table III prints the combined `\HDNoneHzeroComb` = 59 +8 -8, a posterior that leans on the
  prior edge; by the style rule it gets no interval (needs a limit macro or another cell);
  Sec. IV F's sentence still quotes `\HDNoneHzero` as a measurement beside `\ObsNoneHzero`.
- The five AGN-catalog-alone grids at nside 128 are stopped by the variance guard, not by
  empty pixels: the variance of ln L is 24 to 40 there against the limit of 20 (posterior
  samples 18 to 32; 0.85, 2.2, 6.9, about 20 at nside 16, 32, 64, 128), every merger has
  support. Held in the run list (hold text says "-inf", strictly the guard) and removed from
  `lane2.txt` by hand (copy in the scratchpad) so that no empty successor jobs are chained.
- THE TWO EIGHT-PARAMETER SAMPLERS AGREE (20 Nov, `tests/results/compare_r100_eight_parameter.json`):
  rslice reference against uniform-then-rslice on realization 100: every median within 0.08 of
  the 90% half-width (H0 67.46 against 67.53), ln Z -2899.47 +- 0.38 against -2899.18 +- 0.37,
  distance between the posteriors at its null (p 0.23); only sigma_G differs formally (KS p
  0.002) by 0.03 on a half-width of 0.42. Calls 297k against 233k. The mixed sampler is the
  production sampler. The spin offsets below are in both runs: they come from the data or the
  model, not from the sampler.
- First eight-parameter posterior (mixed sampler, realization 100; inputs in parentheses): H0
  67.53 [66.09, 68.99] (67.74), f_AGN 0.288 [0.222, 0.361] (0.30), mu_G 35.73 [35.12, 36.31]
  (35), sigma_G 4.69 [4.30, 5.09] (5), mu_chi -0.019 [-0.033, -0.004] (0), sigma_chi 0.078
  [0.063, 0.093] (0.10), dmu_G 3.80 [2.42, 5.15] (5), dmu_chi 0.163 [0.126, 0.194] (0.10). The
  inputs sit at the 97% level of the eight-dimensional posterior (Gaussian distance squared
  17.0; spins alone 10.2 on three parameters, masses 6.8 on three, H0 and f_AGN 0.1 on two).
  H0 is uncorrelated with the other seven (|rho| <= 0.10); dmu_chi anticorrelates with mu_chi
  and sigma_chi (-0.5). One realization: no conclusion; see whether the rslice reference and
  realizations 101 to 105 show the same spin offsets (narrower widths, larger separation) before
  Sec. IV E is worded. Mean Monte Carlo variance of ln L over the posterior 4.2 (2.8 from the
  posterior samples).
- Analyses 14 and 15: see their REPORT.md (Monte Carlo accuracy; rate per host and the
  normalization volume).

## 5. Do next, in order
1. THE PIXEL TEST (21 Nov; an Opus agent prepares survey files, five run-list entries, a hold
   on the five failing nside-128 tasks and a new block of `tests/skyres_sequence.py`; it sends
   "ready to queue"; check its proof, queue on lane 2 BEFORE lane 2's task ends at about 02:20
   UTC, or the five wait behind a 29-hour run). When the five are in: `tests/skyres_sequence.py`,
   block for the production settings over ten realizations: paired shift over ten, combined
   posterior of ten at nside 32 and 64 with the level of the input. If the input is inside the
   90% interval of the ten at nside 64 and the shift over 106 to 110 is not again positive in
   all: fluctuation; production pixels stay; write the Validation paragraph (shift, lost
   precision, the one-catalog analyses unmoved); refresh; commit; PUSH (owner approved one push
   at that point); PDF to the owner. If the input is outside at nside 64 over ten: offset of the
   two-catalog likelihood exposed by finer pixels: hold the eight-parameter runs (reorder lanes),
   find the cause (mixture weight of the two catalogs; the in-pixel independence of sky and
   redshift; the completion term), switch the two-catalog analyses to nside 64 with 16,000
   samples (owner approved), tell the owner. (Old text of this item, for the reasoning:) run
   `tests/skyres_sequence.py` and read, per analysis, the paired shifts 16 -> 32 -> 64 -> 128
   and the level of the input in the combined posteriors. Questions in order: (a) does the
   AGN-hosted analysis (AGN catalog alone, combined half-width 0.36) move with the pixel, and
   does it stay on the input? (b) do the galaxy-catalog analyses move? (c) does KnownBoth
   converge (64 -> 128 much smaller than 32 -> 64) and where does the input sit at nside 128?
   Then decide, before lane 1 reaches the mixed-sampler run of realization 101 (about 16:00
   UTC): if the shift stops at nside 64 or 128 and the input stays inside the combined
   intervals, keep nside 32 for the campaign, state the shift and the lost precision in the
   Validation (macros exist) and stop. If the finer pixels put the input outside, there is an
   offset in the two-catalog likelihood or in the mocks that nside 32 hides: hold the queued
   eight-parameter runs (reorder the lane files so that six-parameter work goes first; never
   scancel a running task without need), find the term (the one-catalog runs say where to
   look), and tell the owner what was found, with the cost of rerunning at the finer pixel
   (cost per call does not rise with nside; the Monte Carlo variance doubles per halving, so
   16,000 samples per merger at nside 64 for the variance of today). The paper takes the
   sequence in as a Validation paragraph either way (extend `paper_prd/scripts/skyres.py`
   through an Opus agent; `tests/skyres_sequence.py` is the reference computation).
2. The comparison of Sec. 3 is done and the samplers agree (Sec. 4). STILL TO DO, by an Opus
   agent with a brief, AFTER the pixel-size decision of item 1 (one brief for both): the
   mapping and the stage-5 entries; realization-A numbers of Sec. IV E turn black only when
   their keys' other runs exist (HDHost needs five realizations and their HDSingle partners).
3. As six-parameter runs finish (HDNone, HDSingle 101 to 105, E3, stage 3): refresh and re-read
   the tagged sentences of that subsection; PDF to the owner at each subsection that turns black.
4. Before the paper is called ready: an independent Fable review against the skill's self-check
   (the last one, `REVIEW_2.md`, predates Secs. IV C to F turning real), the build gates, the
   title-image count, the reference list.
5. Small leftovers: `fig8_summary` in-panel number; the conventions bullet of
   `figures/CAPTION_FACTS.md`; realization-A interval macros for the number densities still
   exist although unbounded (unused in the text); the prior-allowed density limits carry a
   Monte Carlo error of about 0.01 dex in their second decimal.

## 6. Waiting for the owner (none blocks work)
Answered 21 Nov (Sec. 2). Still theirs: the right grant number. PUSHED 21 Nov 07:54 UTC up to
3e060b3 (owner: "Push now, numbers as measured"; the Validation's last paragraph states the
offset; PDF `..._draft_2026-11-21a.pdf` sent). Each later push is asked for again.

## 7. Other sessions
`gaia`, `darksirens`: nothing owed. Never touch other sessions' jobs (anything not named `agn-*`).

Local commits up to the one of 20 Nov 05:35 UTC that carries this file; `git log --oneline | head -30`.
