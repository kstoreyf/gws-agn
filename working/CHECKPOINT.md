# Checkpoint, 17 Nov 2026, 19:50 UTC (read this first after a context clear)

You are the Fable session that owns the gws-agn project (owner handover of 16 Nov 2026: full
technical and creative freedom; Fable agents write and review, Opus agents code and submit jobs,
you orchestrate and check). Everything below is the state at the time of writing. The memory files
`gw-agn-paper-prd.md`, `writer-briefs-must-follow-style-skill.md`, `gw-agn-ownership-handover.md`
hold the owner's words; this file holds the working state.

## 1. What the owner wants (standing)
- Two papers, maintained together as results arrive (owner, 17 Nov): the long paper
  `working/paper_prd/` (the form the owner prefers) and the letter
  `working/paper_short_independent/`. `paper_short` is frozen in `working/archive/paper_short/`.
- What is new, in the owner's words: "proper multi-tracer darksirens, and correlated gw
  properties with galaxies". The papers are titled and argued for that.
- Drafts are written for the expected results at once; anything that waits for a run is red (red
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

## 2. The two papers
| | Long paper | Letter |
|---|---|---|
| Path | `working/paper_prd/main.tex` (16 pp) | `working/paper_short_independent/main.tex` (10 pp) |
| Design | `notes/PAPER_SKELETON.md`, `REVISION_DECISIONS.md` (item 13), `notes/OWNER_STYLE_DETAILS.md` | `BRIEF_v2.md` (binding), `EXPERIMENT.md` |
| Numbers | `scripts/build_values.py`, `production.py`, `values/production_map.json`, `audit_values.py` | `scripts/results.py`, `results_map.json`, `build_values.py`, `notes/STATUS.md` |
| Figures | `scripts/make_figures.py`, nine figures, `figures/CAPTION_FACTS.md` | `scripts/make_figures.py`: `fig_concept`, `fig_anchor`, `fig1_budget`, `fig_populations` |
| Real now | 28 of 122 result macros | 23 of 42 result macros |

Maintenance loop: `bash working/refresh_papers.sh` (rebuilds values, figures and both PDFs, prints
gates and real/waiting counts) -> a Fable consistency pass limited to sentences tagged
`% EXPECTED` whose numbers turned real (brief: skill governs; give the real numbers; each paper
in its own words) -> send the PDF to the owner (SendUserFile) only when it is consistent.
Shared: `paper_prd/refs.bib`, `paper_prd/notes/PRIOR_WORK_VERIFIED.md` (the only source for
statements about other papers), `paper_prd/notes/LIKELIHOOD_MATH.md`, `paper_prd/notes/GAPS.md`.

## 3. Runs (analysis 16: `working/analyses/analysis_16_production_runs/`, README there)
- Machine: partition RITA-GPU, two A100-80, both ours (the gaia session left for good).
- Launcher: `PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python`;
  `$PY scripts/launch.py --lanes 2 status`; `... --lanes 2 stage <S ...>`; lanes are
  `runs/_lanes/lane1.txt`, `lane2.txt`, jobs `agn-prod-l1`, `agn-prod-l2` (each job runs its lane
  file to the end and chains a successor). Summaries: `runs/<run>/summary.json`.
- State at 19:40 UTC: stage 1 finished (35 runs). The letter's grids: E1 105 of 235, E2 53 of 75;
  about 21 GPU-h left, so both lanes end around 06:00 UTC on 18 Nov. Jobs: 1414946 (lane 1),
  1415005 (lane 2), successors 1414969 and 1415023.
- Tests: closure passed (exact against Analyses 14 and 15); the resume test passed. The
  sampler-settings test for the eight-parameter model, job 1414839 (2 h 15), is released with
  `afterany:1414946`, so it starts by itself when lane 1 ends; result in
  `tests/results/sampler_test.json`.
- Not yet launched (nested sampling): E3 (five six-parameter runs at twice the population
  difference, 22 GPU-h), stage 4 (host-dependent events: single population on ten realizations 62,
  eight parameters on five 166), stage 3 (80), stage 2 (27). After the sampler test:
  `$PY scripts/make_runlist.py --timing tests/results/timing.json [--sampler-8 X --calls-8 N]`
  then `$PY scripts/launch.py --lanes 2 stage E3 4 3 2` (check that the lane order puts stage 4
  before 3 and 2; the headline of both papers waits for it). Stage 5 (observed-like spectrum,
  about 200 GPU-h) is approved by the owner and not in the run list yet.
- Settings: core e7c3007; hosts normalized over the grid to z = 1.10; mixed sets 8000 posterior
  samples per merger with the half-variance correction; all-AGN sets 32000 samples and six pooled
  injection sets; each realization its own injections; host-dependent analyses use the lane
  `injections/injections_targeted_peak.h5` (the old lane is too thin for them);
  `working/data/experiment_sets.json` maps event sets to files and injection lanes.
- Owner-approved scope: stages 1 to 4 with the eight-parameter model on 5 realizations first
  ("Staged: 5, then 10"), plus the observed-like family, plus the letter's experiment
  ("more can happen"). About 340 + 200 + 60 GPU-h.

## 4. Results so far (population known unless said; 1000 detected mergers per realization;
90%; inputs H0 = 67.74, f_AGN = 0.30)
- One realization: both catalogs H0 68.4 (+1.9, -2.0), f_AGN 0.27 +- 0.08. Ten combined: H0 68.2
  +- 0.6 (input at the 80% level), f_AGN 0.306 +- 0.026 (30%). No merger in AGN: f_AGN < 0.02.
- 90% half-width of H0, mean over five realizations, against the fraction of mergers in AGN:
  labels unknown 3.58 (0), 3.55 (0.1), 2.04 (0.3), 1.37 (0.5), 0.83 (1); hosts known 3.56, 1.90,
  1.34, 1.10, 0.81; mass spectrum alone 5.6. With one merger in ten in an AGN and equal
  populations the likelihood gains nothing from the AGN catalog (3.55 against 3.58) although known
  hosts would give 1.90: positions alone barely identify the hosts. This is new and belongs in
  both papers.
- Probability of an AGN host (f = 0.3, AGN-hosted mergers heavier by 5 Msun and spin +0.10):
  positions alone 0.37 for mergers truly in AGN against 0.30 for those truly in galaxies; with
  masses and spins 0.51 against 0.28.
- Observed-like spectrum: mass spectrum alone 8.5, both catalogs at f = 0.3 2.0, all in AGN 0.84.
- AGN catalog alone on mixed mergers: at the prior edge H0 = 100 in four of five realizations;
  realization 102 peaks at 76.8 [75.5, 78.1] and excludes the input.
- Analysis 14 (`analyses/analysis_14_monte_carlo_accuracy/REPORT.md`): 2000 samples per merger
  are too few; the old low f_AGN is the log-mean bias; injections matter as much as samples.
- Analysis 15 (`analyses/analysis_15_host_normalisation/REPORT.md`): the mergers constrain the
  merger rate per host; with complete catalogs no number density enters; at m < 18 an AGN
  density wrong by a factor of two moves f_AGN by 0.13 and H0 by about 1.
- My wrong expectation, corrected in both texts: the galaxy catalog does not "add little"; it
  halves the half-width of the mass spectrum alone.

## 5. Do next, in order
1. Two failures in the last refresh (19:45 UTC), both small, for an Opus agent: the long paper's
   audit check J1 (second-code-path recomputation, largest deviation 0.086 on the f = 0.1 control
   key `CtrlTenth`) and the letter's figure check (it compares a rounded printed mean with the
   unrounded value in `fig_populations(c)` and calls it a mismatch).
2. When the grids are done (about 06:00 UTC 18 Nov): refresh; then a Fable pass on BOTH papers
   with the real experiment results (Sec. 4 above). The long paper must take in the letter's
   experiment (half-width against the fraction in AGN, hosts known, per-merger probabilities).
   Carry these open points: the f = 0.1 result; the AGN-catalog-alone exception (realization
   102); "most of the constraint comes from the AGN-hosted mergers" is true as a share of the
   information (about 63%), not of the width; number-density units (runs sample log10 n in
   h^3 Mpc^-3, Table I of the long paper says Mpc^-3); the long paper's flux-limited numbers are
   still exploratory and narrower than the new complete-catalog width; Fig. 1 (c, d) of the long
   paper uses the old event files.
3. Read `tests/results/sampler_test.json`, then launch the nested stages (Sec. 3).
4. Add stage 5 (observed-like, host-dependent) to the run list; its event sets and injections
   exist (`working/data/obslike/README.md`).
5. Before either paper is called ready: an independent Fable review against the skill's
   self-check, and the build gates of the skill.

## 6. Decisions waiting for the owner (not urgent)
- Are the two papers a letter with a companion (then the style skill requires disjoint figures:
  the letter's Figs. 2 and 4 overlap the long paper's recovery and host-dependent figures), or
  alternatives of which one is submitted?

## 7. Other sessions
`gaia` (left RITA-GPU; nothing owed), `darksirens` (gave the style note
`paper_prd/notes/OWNER_STYLE_DETAILS.md`; nothing owed). Never touch other sessions' jobs.

Local commits up to the one that adds this file; `git log --oneline | head -20` for the day's work.
