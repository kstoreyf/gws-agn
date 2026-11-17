# Checkpoint, 17 Nov 2026, 22:00 UTC (read this first after a context clear)

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
- DONE by 22:00 UTC (commits ca7da6e, 06843cf): the text of the experiment is in `main.tex` (18 pp):
  new Sec. IV B `sec:classes` with Fig. 3 `fig_classes`, Eq. 3 (probability of an AGN host), the
  AGN catalog alone as a biased analysis, masses and spins as a label (host-dependent
  subsection), the observed-like half-widths for the known population, one abstract sentence
  (seven in all), number densities in h^3 Mpc^-3 in text, tables and `fig4_densities`; every
  macro the text uses exists (`values/fallback_macros.tex` is generated and empty); design
  numbers and hosts per localization volume come from the production event sets.
- IN PROGRESS at 22:00 UTC: an independent Fable review of the changed text (findings only, no
  edits). When it reports: apply the fixes (small ones myself, larger ones by a Fable writer
  under the skill), refresh, commit, send the PDF to the owner with SendUserFile.
- Small leftovers: the hand-written blocks of `figures/CAPTION_FACTS.md` for Figs. 2, 3, 5, 8
  still call finished keys placeholders; `notes/MACRO_REQUESTS.md` Sec. F1 is stale; the red
  placeholder density limits of Table IV lie above the Table I prior edge until the stage-2
  nested runs replace them; Sec. III C says Secs. IV A to IV E use the peak-dominated spectrum
  although Fig. 3(b), placed in IV B, draws the observed-like one; the refresh can fail once
  when a run finishes between its values step and its audit (rerun it).

## 3. Runs (analysis 16: `working/analyses/analysis_16_production_runs/`, README there)
- Machine: partition RITA-GPU, two A100-80, both ours.
- Launcher: `PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python`;
  `$PY scripts/launch.py --lanes 2 status`; `... --lanes 2 stage <S ...> [--match REGEX]
  [--only-lane K] [--dry-run]`; lanes `runs/_lanes/lane1.txt`, `lane2.txt`, jobs `agn-prod-l1`,
  `agn-prod-l2` (each job runs its lane file to the end and chains a successor). Lane order is
  the `lane_rank` of each run (table `LANE_RANKS` at the top of `scripts/make_runlist.py`):
  stage-2 grids, stage 4 on realizations 100-105, E3, stage 4 on 106-110, stage-2 nested,
  stage 3 on 100-105, stage 5, stage 3 on 106-110. Summaries: `runs/<run>/summary.json` (block
  `posterior`, computed on a refined lattice since 20:00 UTC; `grid/result.json` holds older
  node-based bounds, do not use them).
- State at 22:00 UTC: E1 132 of 235, E2 finished; lane 1 ends about 05:30 UTC on 18 Nov, lane 2
  continues by itself into the 32 stage-2 grids (4.7 GPU-h) and the stage-4 single-population
  runs on realizations 100-105 (20.5 GPU-h), queued at 20:10 UTC. Jobs: 1414946 (lane 1),
  1415005 (lane 2), successors 1414969 and 1415023.
- Sampler test for the eight-parameter model, job 1414839 (2 h 15), starts by itself when lane 1
  ends (`afterany:1414946`); result in `tests/results/sampler_test.json`.
- Two background watchers of the session: one ends when all E1/E2 grids are finished, one when
  job 1414839 has ended. If the session was restarted they are gone: check by hand.
- After the sampler test: `$PY scripts/make_runlist.py --timing tests/results/timing.json
  [--sampler-8 X --calls-8 N]`, then `$PY scripts/launch.py --lanes 2 stage E3 2 3 4 5` (dry run
  first). Estimates with the present setting: E3 20.5, stage 2 10, stage 3 60, stage 4 215,
  stage 5 225 GPU-h (stage 4 and E3 about 1.14 times more on their injection set); the
  eight-parameter runs are 165 + 190 of it.
- Settings: core e7c3007; hosts normalized over the grid to z = 1.10; mixed sets 8000 posterior
  samples per merger with the half-variance correction; all-AGN sets 32000 samples and six pooled
  injection sets; each realization its own injections; host-dependent analyses use
  `injections/injections_targeted_peak.h5`; observed-like `injections_targeted_obslike.h5`;
  `working/data/experiment_sets.json` maps event sets to files and injection sets.
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
1. Finish the items of Sec. 2 (writer, review, cleanup, PDF to the owner).
2. When the grids are done (watcher; about 06:45 UTC 18 Nov): refresh; Fable consistency pass on
   the `% EXPECTED` sentences whose numbers turned real; PDF to the owner.
3. When the sampler test has ended (watcher; about 09:00 UTC): read
   `tests/results/sampler_test.json`, regenerate the run list, launch `E3 2 3 4 5`.
4. Stage-2 grids real (about 11:00 UTC): refresh; the flux-limited subsection's numbers turn
   real (they are exploratory and narrower than the complete-catalog width until then).
5. Open points for the text: number densities in h^3 Mpc^-3 in Table I and wherever a prior is
   quoted; the AGN-catalog-alone exception (realization 102); the date of the fifth observing
   run in the conclusions needs a citable schedule; acknowledgments and author list are the
   owner's.
6. Known numerical flags (from the audit fix of 17 Nov): two catalog-free observed-like grids
   (all mergers in AGN, realizations 101 and 103) have ln L jumps of about 100 nats between
   neighbouring H0 nodes; one all-AGN observed-like grid (realization 102) is under-resolved in
   f_AGN. The numbers agent of Sec. 2a reports whether anything printed uses them.

## 6. Decisions waiting for the owner
None.

## 7. Other sessions
`gaia` (left RITA-GPU; nothing owed), `darksirens` (gave the style note
`paper_prd/notes/OWNER_STYLE_DETAILS.md`; nothing owed). Never touch other sessions' jobs.

Local commits up to 06843cf plus the one that adds this file; `git log --oneline | head -20`.
