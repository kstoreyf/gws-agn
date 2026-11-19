# Checkpoint, 19 Nov 2026, 22:45 UTC (read this first after a context clear)

You are the Fable session that owns the gws-agn project (owner handover of 16 Nov 2026: full
technical and creative freedom; Fable agents write and review, Opus agents code and submit jobs,
you orchestrate and check). The memory files `gw-agn-paper-prd.md`,
`gw-agn-ownership-handover.md`, `writer-briefs-must-follow-style-skill.md` hold the owner's words;
this file holds the working state.

## 0. How to resume (the owner clears the context and says "resume from the latest checkpoint")
1. Read this file to the end. Do not re-read the paper or the scripts until a step needs them.
2. Look at the state: `git -C /hildafs/projects/phy230014p/magana/gws-agn log --oneline | head -5`;
   `squeue -u $USER -p RITA-GPU`; `$PY scripts/launch.py --lanes 2 status` in the analysis-16
   directory (Sec. 3 has `PY`).
3. Nothing is owed right now except waiting: lane 1 is on the eight-parameter reference run of
   realization 100, lane 2 on the twelve sky-resolution runs. The next actions are Sec. 5,
   items 1 and 2, each triggered by a watcher.
4. WATCHERS survive a context clear and report as background-task notifications that carry no
   context. Two are armed:
   - task `bvv0w6obd` (19 Nov, 22:30 UTC), "Wait until the twelve sky-resolution runs of
     realizations 101, 102, 103 and 105 have summaries": then do Sec. 5 item 1.
   - task `bue7u6pya` (18 Nov), "Wait until both eight-parameter runs of realization 100 have
     finished": then do Sec. 5 item 2.
   (`bk2x1ic8d`, the high-accuracy pair of realization 100, reported on 19 Nov 21:45 UTC: done.)
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
  `Claude-Session: https://claude.ai/code/session_01L6W2Gr2RxPuB4nQyh4gjGp`.
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
  acknowledgments modeled on the owner's arXiv:2610.37617. The owner has been asked to confirm
  (not blocking): that NASA Grant No. 22-LPS22-0025 covers this work; the AI sentence ("drafting
  and editing", Claude models only); whether to push at each milestone.
- Sentences to revisit when their runs finish: Validation, sky resolution (rewritten 19 Nov for
  the mean and the scatter of the shift over five realizations at 32,000 samples and six
  injection sets: macros `\SkyResMCMeanShift...`, `\SkyResMCSdShift...`, red until the twelve
  runs of Sec. 3 are in; tagged `% EXPECTED`); Sec. IV E and the
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
- STATE 19 Nov 22:45 UTC (the two bullets RUNNING and QUEUED below are the 18 Nov text, kept for
  the order of the queue): lane 1 still on the rslice reference of realization 100 (iteration
  3957 at 21:39 UTC, 136 iterations per hour, about 4900 to 5300 needed: ends about 04:30 to
  07:30 UTC 20 Nov; job 1416192 reaches its time limit at 05:28 UTC 20 Nov and the successor
  1416219 carries on from the checkpoint). Lane 2: the second part of HDSingle realization 100
  (until about 23:45 UTC), then the TWELVE SKY-RESOLUTION RUNS queued on 19 Nov 22:29 UTC
  (stage X, realizations 101, 102, 103, 105: `realization_NNN_nside64`, `..._s32000_inj6`,
  `..._nside64_s32000_inj6` under `known_population/both_catalogs/complete/equal_populations/`;
  about 4 GPU-h, done about 04:00 to 05:00 UTC 20 Nov), then HDSingle 101 to 105 and the rest
  as below. Their nside-64 survey files exist (`data/seedNNN/surveys_ns64/`, checks passed).
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
- Sky resolution, realization 100: nside 64 against 32 moves H0 by +0.64 and f_AGN by +0.016 at
  8000 samples (1.9 times the Monte Carlo variance) and by +0.42 and +0.008 at 32,000 samples
  and six injection sets (medians 68.12 -> 68.53; variance of the posterior-sample sums 0.59 ->
  1.07). Redraws alone move the H0 median by 0.20 at 32,000 samples (Analysis 14), so one
  realization does not separate pixel size from Monte Carlo error: four more realizations are
  running. A shift common to all realizations of +0.4 would matter for the combined posterior
  of ten (68.2 +- 0.6, input 67.74).
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
1. The twelve sky-resolution runs (watcher `bvv0w6obd`, about 05:00 UTC 20 Nov): refresh; the
   macros `\SkyResMCMeanShift...`, `\SkyResMCSdShift...` turn black; reword the sentence of the
   Validation by what they show and drop its `% EXPECTED`. The standard error of the mean is
   the scatter over root five. If the mean H0 shift is within about twice that of zero: say the
   pixel size moves no median in common, done. If it is a common shift of some tenths: nside 32
   biases every H0 by that much; say so with its size against the half-width, look at the
   combined posteriors, and cost a rerun of the headline analyses at nside 64 (an nside-64 grid
   costs about half an nside-32 one at equal samples; nested runs unknown) before deciding.
   `\SkyResMeanShift...` (8000 samples) are also produced; not used in the text.
2. Both realization-100 eight-parameter runs finished (watcher `bue7u6pya`): the comparison of
   Sec. 3 (also look at each run's `summary.json`), then, by an Opus agent with a brief, the
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
The three confirmations of Sec. 2 (grant, AI sentence, push policy). Nine local commits since the
push (ba8f91f): ask before pushing.

## 7. Other sessions
`gaia`, `darksirens`: nothing owed. Never touch other sessions' jobs (anything not named `agn-*`).

Local commits up to the one of 19 Nov 22:45 UTC that carries this file; `git log --oneline | head -30`.
