# Checkpoint, 18 Nov 2026, 19:10 UTC (read this first after a context clear or compaction)

You are the Fable session that owns the gws-agn project (owner handover of 16 Nov 2026: full
technical and creative freedom; Fable agents write and review, Opus agents code and submit jobs,
you orchestrate and check). The memory files `gw-agn-paper-prd.md`,
`gw-agn-ownership-handover.md`, `writer-briefs-must-follow-style-skill.md` hold the owner's words;
this file holds the working state.

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

## 2. The paper (state at commit b29e087; 18 pp, ten figures)
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
- Sentences to revisit when their runs finish: Validation, sky resolution (the 8000-sample pair
  moves H0 by 0.64 with twice the Monte Carlo variance; the pair at 32,000 samples and six
  injection sets decides what to say: macros `\SkyResMC...`, red now); Sec. IV E and the
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
  first three runs (nside 64 at 8000 samples; two m < 18 grids with the AGN density in Mpc^-3).
- RUNNING since 18 Nov: lane 1 (job 1416192), the eight-parameter reference of realization 100
  with rslice from the start (`host_dependent_populations/both_catalogs/complete/host_dependent/
  realization_100`, started 17:50 UTC, about 37 to 40 h); lane 2 (job 1415023), the continuation
  `.../realization_100_unif_then_rslice` (from the stopped uniform run at iteration 1594, rslice
  from there; started 16:55 UTC, about 26 to 28 h).
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
  realization-100 runs have finished (about 20 Nov, midday UTC): `tests/compare_sampler_runs.py`
  on the two run directories (medians in units of the half-width, distance between posteriors
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
- Watchers are background shells of the session; after a restart they are gone: check by hand.

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
- Sky resolution: nside 64 against 32 at 8000 samples moves H0 by 0.64 and f_AGN by 0.016, with
  1.9 times the Monte Carlo variance; not yet separated.
- Analyses 14 and 15: see their REPORT.md (Monte Carlo accuracy; rate per host and the
  normalization volume).

## 5. Do next, in order
1. Lane 2 reaches the sky-resolution pair when the continuation ends (about 19 Nov, 20:00 UTC):
   refresh, then rewrite the sky-resolution sentence of the Validation from `\SkyResMC...`
   (is the shift at high accuracy small against the half-width, or is nside 32 too coarse? If
   it is not small: a resolution study at nside 64 for the headline runs becomes a real item;
   cost it before deciding).
2. Both realization-100 eight-parameter runs finished: the comparison of Sec. 3, then the
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
The three confirmations of Sec. 2 (grant, AI sentence, push policy). Six local commits since the
push (ba8f91f): ask before pushing.

## 7. Other sessions
`gaia`, `darksirens`: nothing owed. Never touch other sessions' jobs (anything not named `agn-*`).

Local commits up to b29e087 plus the one that adds this file; `git log --oneline | head -30`.
