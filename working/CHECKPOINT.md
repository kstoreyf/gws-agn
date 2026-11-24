# Checkpoint, 24 Nov 2026, 00:05 UTC (read this first after a context clear)

You are the Fable session that owns the gws-agn project (owner handover of 16 Nov 2026: full
technical and creative freedom; Fable agents write and review, Opus agents code and submit jobs,
you orchestrate and check). The memory files `gw-agn-paper-prd.md`,
`gw-agn-ownership-handover.md`, `writer-briefs-must-follow-style-skill.md` hold the owner's words;
this file holds the working state.

## STATE 24 Nov 00:05 UTC (READ THIS BLOCK FIRST; it supersedes steps 3 and 4 of Sec. 0)
Local commits up to the one that carries this file; GitHub has e6d5313 (ask before any push).

### The session is back; there was no gap
- The new MIKO holder 1494715 started 16:57 UTC 23 Nov, a day before Slurm's estimate; its limit
  is 11:56 UTC 30 Nov. This session runs inside it: never cancel it. The owner said "continue
  from the latest checkpoint" at 18:55 UTC. The session link for commit messages changed again:
  use the one in the latest system reminder.
- The two lane chains on RITA-GPU ran through untouched. Lane 1: job 1494378 (limit 09:32 UTC
  25 Nov), successor 1496542. Lane 2: job 1482276 (limit 06:30 UTC 25 Nov), successor 1495021.
  At 18:55 UTC: 43 and 40 tasks to do, none failed. Check as before, in the analysis-16
  directory: `$PY scripts/launch.py --lanes 2 status` (Sec. 3 has `PY`).

### Done since 15:10 UTC
1. THE FIFTEEN OBSERVED-LIKE REALIZATIONS ARE IN THE PAPER (step 1 of the old list). All twenty
   grids of realizations 106 to 115 finished (none failed). Refresh 19:03 UTC and again 19:21
   UTC after the edits: exit 0, audit 94 of 94, figures 515 with 0 mismatches, 190 of 255
   result macros of the text real, 19 pp. Both catalogs, set `obslike_f0p30_s1`, H0 combined
   with the level of the input 67.74:
   population assumed, fifteen: 68.5 (+0.5, -0.4), 99% (outside the 90% interval);
   both populations at their inputs, fifteen: 67.4 +- 0.4, 86%; first five 67.0 (+0.7, -0.6),
   94%; later ten 67.6 (+0.7, -0.5), 32%; assumed on the first five 67.8 +- 0.8 (9%), on the
   later ten 69.0 (+0.5, -0.6), 99.8%. f_AGN with the population assumed 0.306 +- 0.017 (43%).
   PAIRED SHIFT, assumed minus inputs, per realization (100, 101, 102, 103, 105, 106 to 115):
   +1.30 +0.68 +1.08 +1.42 +1.11 +0.41 +1.77 +0.88 +1.39 +0.86 +0.54 +0.70 +0.52 +1.24 +0.97:
   mean 0.99, sd 0.39, standard error 0.10, positive in fifteen of fifteen. This is the robust
   statement of the bias for this spectrum (macros: median 1.0, range 0.4 to 1.8).
   `main.tex`: Sec. IV F's sentence is rewritten as two (shift over fifteen with its range; the
   two combined posteriors with their levels; then "The five realizations that carry the
   analyses with inferred populations below lie low ... which we read as the draw of those
   five"), both tagged `% REREAD`; Sec. IV G ("... by 1.0 instead of 3.2, and fifteen combined
   realizations place the input at the 99% credible level", tagged `% REREAD`); Fig. 8's
   caption ("only the two with the population assumed have the input outside that of H0").
2. The analysis-16 README describes the extension of stage E2 to realizations 106 to 115
   (paragraph after "Stage 5"; the stage table's E2 row is 95 runs, 11.4 GPU-h).
3. THE FIVE EIGHT-PARAMETER RUNS ARE IN (step 2 of the old list). Realization 105 finished
   20:09 UTC (204,953 calls, 29.2 GPU-h): H0 67.55 [65.69, 69.26], f_AGN 0.360 [0.280, 0.444],
   mu_G 35.11, sigma_G 5.09, mu_chi -0.018 [-0.039, 0.002], sigma_chi 0.104, dmu_G 4.29 [2.28,
   6.08], dmu_chi 0.098 [0.056, 0.137]: all eight inputs inside; ln Z -2947.33 +- 0.36 against
   -2953.11 for the single population: ln B +5.8 (the lowest of the five), and NO gain in the
   H0 half-width on this realization (1.79 against 1.75).
   Refresh 20:20 UTC: exit 0, audit 94 of 94, 215 of 255 result macros of the text real. Key
   HDHost over five: H0 67.9 (+0.9, -0.8), input at 26%; f_AGN 0.32 +- 0.04 (58%); dmu_G 4.6
   +- 0.8 (59%); dmu_chi 0.126 (+0.020, -0.019), INPUT AT 98%, below the 90% interval; mu_chi
   -0.010 (+0.008, -0.009), INPUT AT 95%, above it; mu_G 46%, sigma_G 3.7%, sigma_chi 16%.
   log10 B 2.5 to 8.4 (combined 27). H0 half-width 1.6 [1.4, 1.8] against 1.9 [1.8, 2.0] for
   the single population: 1.3 [1.0, 1.8] times the mergers. |rho(H0, other)| <= 0.10.
   THE SPINS: galaxy-class spin mean low and AGN spin offset high, an anticorrelated pair
   (realizations 100, 102, 105 have mu_chi near -0.019; 100 and 102 have dmu_chi 0.163 and
   0.147). Two of eight inputs outside the 90% intervals over five: stated in the text as
   measured (Sec. IV E third paragraph; Validation, new sentence "Two inputs lie outside the
   90% intervals of these five realizations, both in the spins"), no hunt on five. It is the
   strongest reason for "5, then 10" (the eight-parameter model on realizations 106 to 110).
   `main.tex` also: abstract ("and in the mean over them the single population needs ...");
   the `% EXPECTED` tags of the sentences that are now real are removed.

### THE STEPS IN ln L OF THE OBSERVED-LIKE FAMILY: DIAGNOSED (20:25 UTC), two grids queued to size it
- FOUND 19:10 UTC: Fig. 7(a)'s curve for the assumed population has a shoulder at 69.2.
  Realization 106's stored ln L along H0 at f_AGN = 0.30 has two upward STEPS, +1.2 nats
  between 68.2 and 68.4 and +1.3 between 69.0 and 69.2, in `two_tracers_one_population` and
  at the same H0 in `two_tracers_two_populations` (+0.8, +1.0). A scan of all 504 stored
  grids flags 12 runs with steps above 0.5 nat within 12 nats of the peak, ALL of the
  observed-like family, runs without a catalog included (`mass_spectrum_alone` on
  `obslike_f0p00_s0` realization 105 and `obslike_f1p00_s0` realizations 101 and 103, about
  100 nats there; `galaxy_catalog_alone` and `two_tracers_one_population` on `obslike_f0p00_s0`
  realization 105 and on `obslike_f0p30_s0` realization 101; `hosts_known` and
  `two_tracers_one_population` on `obslike_f1p00_s0` realization 103). All go up with H0.
- CAUSE (Opus agent, CPU, `tests/obslike_steps_merger.py`, results `tests/results/
  obslike_steps_*`; its per-merger sum reproduces the stored ln L to 4e-12; I checked its
  8000-sample line and intervals against the stored grid): ONE merger, number 691 of
  realization 106 (galaxy-hosted, power law, true m1 80.6 and m2 74.2 in the source frame,
  z 0.35; observed with a low mass ratio, so its posterior has m1 104/171/297 at 5/50/95%).
  The power law of the observed-like spectrum ends at m_max = 87 with a taper of 0.01, a hard
  edge (`sfilter_high` in core): 42 of the merger's 8000 samples lie below it at the input
  H0, and among those the narrow spin population gives nearly all the weight to one or two
  (effective sample number 1.1 to 2.2 over H0 66 to 72). As H0 rises, single samples cross
  below 87 and each adds a step. NOT a fault: Monte Carlo resolution of a real, small
  feature. Merger 691's ln L(70) - ln L(67): +2.20 at 8000 samples (the largest of 32
  independent subsets of 8000: mean 0.42, sd 0.50), +0.49 at 32,000, 0.37 +- 0.05 converged
  (256,000). About one redraw in four at 8000 shows a step above 0.5 nat for this merger.
  The 100-nat steps of the all-AGN no-catalog runs are mergers with NO sample below the edge
  under some H0 (inferred from the crossings, not evaluated).
- CENSUS at the input, fifteen realizations: mergers with effective sample number below 5:
  0 to 2 per realization (16 in all), below 20: 4 to 13, below 100: 45 to 76; only 2 of the
  16 are edge mergers (106's number 691, 100's number 494), the others are light mergers of
  the steep low-mass end and show no steps. The peak-dominated family has as many poorly
  resolved mergers (0 and 2 below 5 on two realizations) and no edge (its taper is 10).
- SIZE (hybrid on CPU: the 18 mergers of realization 106 with effective number below 50 at
  32,000 samples, the rest as stored; line at f_AGN = 0.30): realization 106 alone, assumed
  population, 90% interval [69.06, 71.74] -> [68.41, 71.68]; two populations [68.43, 71.07]
  -> [67.87, 70.81]. Applied to the stored marginals (my estimate): the fifteen with the
  population assumed 68.50 [68.06, 69.05], level 98.9% -> 68.42 [67.94, 68.89], level 97.8%;
  both populations at their inputs 67.37, 86% -> 67.41, 80%. SO THE STATEMENTS OF SEC. IV F
  STAND (input outside the 90% interval with the population assumed, inside with both
  populations at their inputs); the printed 99% carries one merger's Monte Carlo excess of
  about one point. The paired shift is unaffected in substance (106's is the smallest).
- DECIDED (mine): production keys stay at 8000 samples for all fifteen. The Validation's
  Monte Carlo paragraph gets one or two sentences giving the size: the mechanism in physics
  words (a merger whose posterior lies almost wholly above the upper edge of the power law is
  carried by a few samples) and the combination with that realization evaluated at 32,000
  samples beside the production one. For that: realization 106's two grids are re-run at
  32,000 samples as stage-X runs (`.../obslike_f0p30_s1/realization_106_s32000`, same redraw
  key and injection set; about 1 to 2 GPU-h), and the chain gets macros
  `\ObsAssumedMCHzeroComb`, `\ObsAssumedMCHzeroCL`, `\ObsPopsMCHzeroComb`, `\ObsPopsMCHzeroCL`,
  `\ObsAssumedMCShiftHzero[Min|Max]`, `\ObsMCNsamp`, `\ObsMCOneHzero`, `\ObsMCOneHzeroProd`
  (red until the two runs exist).
- DONE 21:05 UTC: an Opus agent added the two stage-X entries (`make_runlist.X_OBS_MC`; run
  lists regenerated: two runs added, none removed, no arguments changed, checked by me from
  the YAML against HEAD; stage 5's eight-parameter estimates moved to 35.35 GPU-h each because
  realization 105 of stage V8 finished) and the macros (`build_values.build_obslike_mc`, map
  block `obslike_mc`, audit J7, `notes/MACRO_REQUESTS.md` Sec. H; no existing macro value
  changed, nine added, red). `\ObsMCNsamp` does not exist: `\MCNsampHi` prints 32000.
  QUEUED by me on lane 2 (rank 2.85, first in its file): the two grids
  `two_tracers_{one_population,two_populations}/both_catalogs/complete/obslike_f0p30_s1/
  realization_106_s32000`, 2008 calls each, 1.3 GPU-h together; they start when lane 2's
  E3 part ends (about 22:35 UTC) and end about 00:00 UTC 24 Nov. WATCHER `bniuy3n5w`: both
  summaries, or an error outcome, or 03:30 UTC 24 Nov.
  `main.tex`: Appendix A's paragraph "Accuracy" has the new sentences ("The observed-like
  spectrum ends sharply at 87 Msun ... One of the fifteen realizations of Sec. IV F holds such
  a merger. With 32000 samples per merger that realization gives ... and the combined
  posterior becomes ..., with the input at the ...% credible level where Sec. IV F quotes
  ...%"), tagged `% EXPECTED`, red until the runs exist; Sec. IV F points to the appendix.
  Refresh 21:03 UTC: exit 0, audit 95 of 95, 216 of 262 result macros of the text real.
- WHEN THE WATCHER FIRES: refresh (exit 0; J7 black). Compare with my estimate (assumed
  population over fifteen 68.42 [67.94, 68.89], level 97.8%; both populations 67.41, 80%;
  realization 106 alone [68.41, 71.68] and [67.87, 70.81]). Read the appendix sentences
  against the black numbers ("where 8000 give" must show a real difference; if the level
  after replacement is still above 90% the wording "outside the 90% interval" of Sec. IV F
  stands), remove their `% EXPECTED` and the three `% REREAD` tags of Secs. IV F and IV G,
  look at the combined curve of Fig. 7(a) (it stays the production one, with its shoulder:
  decide whether the figure should draw the 32,000-sample combination instead; mine).
  Commit. If a run errored: read its log (GPU memory at 32,000 samples with both catalogs is
  the likely cause; the `_s32000_inj6` precedents of the peak family ran).
- FOR STAGE 5 (observed-like nested runs on realizations 100 to 105, 8000 samples, last on
  the lanes): of those sets only realization 100 holds an edge merger with an effective
  number below 5 (number 494), and its stored grids show no step above 0.5 nat near the
  peak. When stage 5's posteriors of realization 100 come in, look at its H0 marginal for a
  wall or shoulder before quoting it.
- THE HALF-WIDTH MEANS ARE AFFECTED TOO (agent report 21:40 UTC; `tests/obslike_steps_halfwidth.py`,
  results `tests/results/obslike_steps_halfwidth_*.json`; every stored line reproduced to
  4e-12). Six stepped runs feed means of H0 half-widths over five realizations (Sec. IV F,
  Fig. 3(b)). Half-width of the realization, stored -> hybrid at four times the samples
  (-> with the step merger at 256,000 or more), and the printed mean:
  1 `mass_spectrum_alone` on `obslike_f0p00_s0` realization 105 (merger 999, 7 to 21 of 8000
    samples below the edge): 7.15 -> 8.14 (-> 8.52): HWObsNone 8.4 -> 8.6 (8.7);
  2 `galaxy_catalog_alone`, same realization: 3.69 -> 4.10 (4.17): HWObsKnownZero 3.5 -> 3.6;
  3 `two_tracers_one_population`, same: 3.65 -> 4.08 (4.12): HWObsUnknownZero 3.5 -> 3.6;
  4 `two_tracers_one_population` on `obslike_f0p30_s0` realization 101 (mergers 872, 142):
    2.03 -> 1.87 (1.91): HWObsUnknownThirty 2.0 -> 2.0;
  5, 6 `hosts_known` and `two_tracers_one_population` on `obslike_f1p00_s0` realization 103,
    already at 32,000 samples (merger 553: 6 of 32,000 samples below the edge, effective
    number 29 even at 1,024,000): 1.06 -> 0.77: HWObsKnownHundred 0.83 -> 0.77,
    HWObsUnknownHundred 0.84 -> 0.78.
  So my expectation was wrong: four of six means move in their second digit, single
  realizations by 10 to 30%, and MORE SAMPLES DO NOT CONVERGE for such mergers.
- DECIDED 21:45 UTC (mine): FIX IT AT THE ROOT BEFORE STAGE 5 RUNS (stage 5 is about 210
  GPU-h on this family and starts about 26 Nov). A DEFENSIVE MIXTURE PROPOSAL: half of each
  merger's samples from its posterior as now, half from the posterior restricted to the
  region A where the spectrum's sharp mass edges can hold for some H0 of the prior range
  (primary source mass below m_max at H0 = 100, and the lower edge at H0 = 50), with the
  exact weight 1 / [(1 - a) + a 1_A / P_A] entering through the PE-prior column. Exact for
  any population (the Gaussian tail above the edge included), no change of the likelihood
  code, no effect where a family has no sharp edge.
  STEP A DONE (report 23:05 UTC; `tests/mixture_redraw/`: `redraw_mixture.py`, the tools
  `mix_eval.py`, `mix_scan.py`, `mix_item3*.py`, `mix_summary.py`, results in its git-ignored
  `results/`). Driver option `--redraw-mixture {off,support}`, default off; when on the run
  needs a `--variant` with the token `mix` or an explicit `--run-name`; config.json and
  summary.json record it only when on; `<part>/redraw_mixture.json` holds the region and the
  per-merger P_A. Region for the observed-like family: m1det / (1 + z(dL; H0 = 100)) < 87.05
  (the lower edge has a taper of 4.9 and adds nothing); for the peak family the region is
  everything and the redraw is bitwise production's. A merger is refilled when a pilot of
  2e5 draws puts less than 99% of its posterior inside (about 190 of 1000 per realization;
  the others keep production's samples bitwise).
  VALIDATION at 8000 samples, eight streams, off -> on against the converged references:
  merger 691: effective number 2.1 -> 17.5, largest step 0.31 (up to 1.48) -> 0.07, preference
  over 66 to 72 1.02 +- 0.91 -> 0.556 +- 0.087 (reference 0.584); merger 999: 7.4 -> 75.8,
  4.23 +- 1.28 -> 3.44 +- 0.23 (3.54); merger 553 (AGN-only): 1.0 -> 3.8 at 8000 and 2.2 -> 18
  at 32,000, 27 +- 46 -> 1.55 +- 0.51 and 1.65 +- 0.35 (1.61). Whole realizations (90% interval,
  half-width; stored / hybrid / mixture): 106 one population [69.06, 71.74] 1.34 / [68.41,
  71.68] 1.64 / [68.33, 71.59] 1.63 (streams 1.82, 1.66); 106 two populations 1.32 / 1.47 /
  1.49; 105 of f0p00_s0 no catalog 7.15 / 8.52 / 8.50 (streams 8.3 to 8.8); 103 of f1p00_s0
  AGN-only at 32,000 1.06 / 0.77 / 0.82 (streams 0.79, 0.76). No steps left. Non-refilled and
  mildly refilled mergers: on minus off is a constant per merger (the p_pe normalisation,
  about 1.1 nats summed for realization 106: COMPARE EVIDENCES ONLY WITHIN ONE SETTING) with
  no H0 dependence (-0.00007 +- 0.00044 per merger across the interval). Cost: 17 to 23 s
  more setup per run, per-call cost unchanged, cap never reached in 25 realizations.
  LIMITS: the gain is bounded by about 1/(2 P_A) (merger 553 stays at an effective number of
  18 at 32,000); mergers poorly resolved for other reasons (narrow spin population, light
  mergers) are untouched: the all-AGN observed-like set at 8000 samples is NOT fixed by it
  (58 mergers below 20) and keeps its 32,000 samples.
  CHECKED BY ME: the copies' diffs against the originals add lines only, all behind the
  option; `redraw_mixture.py` read in full (stratified halves with balance weights
  1 / [(1 - a) + a 1_A / P_A]: unbiased); the copy with the option off reproduces the stored
  grid of realization 106 (two_tracers_one_population, three nodes) to 3.6e-12 in ln L.
  INSTALLED 23:18:48 UTC into `scripts/` by atomic `mv` (order `redraw_mixture.py`,
  `prod_summary.py`, `prod_model.py`, `prod_driver.py`; `prod_common.py` unchanged; md5 of the
  old files in the scratchpad `md5_before_install.txt`, and git has them). Imports checked
  in the core venv. CHECKED 00:00 UTC 24 Nov: lane 1's next task (E3 realization 103, part 1) started
  23:49:57 UTC with the installed scripts and runs normally; the summary step before it
  exited 0.
  THE TWO 32,000-SAMPLE GRIDS OF REALIZATION 106 finished (22:51 and 23:16 UTC): assumed
  population 69.67 [67.89, 71.40] (8000: 70.11 [69.07, 71.73]); two populations 69.12
  [67.39, 70.72] (8000: 69.70 [68.37, 71.06]). All mergers at four times the samples move the
  medians by -0.4 and -0.6, beyond the edge merger's part (hybrid: 70.00 and 69.36): the
  general redraw scatter of 0.28 per realization that the appendix already states. They are
  a cross-check now; no macro reads them.
  STEP B DONE AND QUEUED (00:01 UTC 24 Nov). The Opus agent's work, checked by me from the
  YAML and the macros file against the commit before: `RUNLIST_experiment.yaml` has 95 new
  runs, all stage E2M (`<partner>_mix`: 75 `_mix`, 15 `_agn_s32000_inj1_mix`, 5
  `_s32000_inj1_mix`; arguments = the partner's + the `mix` token in the variant +
  `--redraw-mixture support`, all 95; field `mix_partner`; 12.0 GPU-h); `RUNLIST.yaml`:
  stage 5's fifteen entries are exactly the old arguments + `--redraw-mixture support
  --run-name <their name>` (names, part directories and nested state paths unchanged; no
  stage-5 directory existed), nothing else changed anywhere. `launch.py` has an opt-in
  `--spread` (new tasks balanced among themselves over the lanes) and knows stage E2M.
  Paper chain: keys `ObsAssumed`, `ObsPops` and the classes macros `HWObs{Unknown,Known}
  {Zero,Thirty,Hundred}`, `HWObsNone`, `HWObsPopsThirty`, `HWObsKnownPopsThirty` (with Min,
  Max) read the E2M runs; 89 macros went real -> red, each printing its old value
  (`values/obslike_e2_frozen.json`); the nine `...MC...` macros, the block `obslike_mc` and
  audit J7 are gone; audit J8 (every observed-like run a key reads records
  `redraw_mixture == "support"`, no other run does, stage 5 carries the option, red macros
  equal the frozen values). Refresh 23:55 UTC: exit 0, audit 95 of 95, figures 515 with 0
  mismatches, 19 pp, 201 of 256 result macros of the text real (15 of the re-pointed ones
  are in the text). Figs. 3(b), 7 and 8 draw the observed-like quantities in placeholder
  style with the same values. Committed 6e4cfbf.
  QUEUED by me: `launch.py --lanes 2 stage E2M --spread`: 51 tasks on lane 1 (6.1 h), 44 on
  lane 2 (5.9 h), rank 1.5, at the front of both lanes: they start when each lane's present
  E3 part ends (lane 2 about 01:20 UTC, lane 1 about 01:50 UTC) and end about 07:30 and
  08:00 UTC 24 Nov. Queueing them also makes each lane job re-read the run lists at its
  next task, so stage 5 will run WITH the option (a lane job reads the lists at its start
  and again only when it meets a task it does not know: remember this whenever arguments of
  queued runs are changed).
  WATCHER `bqb11jb9l` (armed 00:02 UTC): all 95 E2M summaries, or an error outcome, or 11:30
  UTC 24 Nov; its list is `tests/results/e2m_runs.txt`.
  WHEN IT FIRES: (1) `bash working/refresh_papers.sh` (exit 0; J8 now compares summaries).
  (2) Repeat on the new grids what found the problem: the step scan (first differences of
  ln L along H0 against the median of their four neighbours, flag above 0.5 nat within 12
  nats of the peak, over `runs/*/*/*/*/realization_*_mix/grid/`: expect none), the
  leave-one-out of the fifteen for `ObsAssumed` (it was 98.5% with realization 106, 89.5%
  without), and a look at Fig. 7(a)'s dotted curve (the shoulder at 69.2 should be gone).
  (3) Re-read EVERY observed-like sentence against the black numbers: Sec. IV F whole (the
  half-widths 8.4, 2.0, 1.6 and "each difference between the spectra is inside the range
  over the realizations"; the shift over fifteen and the two combined levels with "outside
  the 90% interval"; the sentence on the five that lie low), Sec. IV G's sentence, the
  abstract's sentence on this spectrum, Table III's observed-like rows, Fig. 3(b), Fig. 7,
  Fig. 8 and their captions; remove the three `% REREAD` tags and the `% EXPECTED` of the
  Appendix A sentence. EXPECT from the CPU hybrids: HWObsNone 8.4 -> about 8.7,
  HWObsKnownZero and HWObsUnknownZero 3.5 -> 3.6, HWObsUnknownThirty 2.0, the all-AGN pair
  0.83 and 0.84 -> about 0.78; the assumed population over fifteen about 68.4 with the
  input near the 98% level; both populations about 67.4, 80%. Word from the printed values.
  (4) Commit; no PDF to the owner yet (it goes at the Sec. IV E milestone).
  `main.tex` (committed 6e4cfbf): the sentences on the 32,000-sample check and the pointer
  in Sec. IV F are gone; Appendix A "Monte Carlo sums" has the sentence on the mixture
  sampling with `\cite{Hesterberg:1995,Owen:2000}` (verified by web search, recorded in
  `notes/CITATION_CHECKS.md`, entries in `refs.bib`), tagged `% EXPECTED` until the runs exist.
  THE OWNER HAS BEEN TOLD (messages of 21:50, 23:25 UTC and 00:05 UTC 24 Nov), not asked.

### Running now, and what each needs from you
- LANE 1 (job 1494378, limit 09:32 UTC 25 Nov, successor 1496542): E3 realization 103 part 1
  (since 23:50 UTC), then 51 E2M grids, E3 (2 parts), HDSingle 106 to 110 first parts,
  stage 3, stage 5.
- LANE 2 (job 1482276, limit 06:30 UTC 25 Nov, successor 1495021): E3 realization 105 part 2
  (since 23:17 UTC), then 44 E2M grids, E3 (1 part), HDSingle 106 to 110 second parts,
  stage 3, stage 5.
- WATCHER `bqb11jb9l`: the 95 E2M summaries (see "THE STEPS", "WHEN IT FIRES").
- WATCHER `bwyy2o0h3`: the ten runs of the Sec. IV E milestone (E3's five and HDSingle 106 to
  110; all summaries, or an error outcome, or 09:00 UTC 25 Nov). The milestone moved to
  about 12:00 UTC 25 Nov, so this watcher will fire at its limit with some runs open: re-arm
  one then for the rest (list `step3_runs.txt` in the session scratchpad, or rebuild it:
  E3's five are `single_population/both_catalogs/complete/peak_f0p30_s2/realization_{100,
  101,102,103,105}`).
- No agent is running. If the session restarted, the watchers are gone: check by hand.

### Do next, in order
1. The E2M watcher: "THE STEPS", "WHEN IT FIRES".
2. THE SEC. IV E MILESTONE (about 12:00 UTC 25 Nov): when E3's five and HDSingle 106 to 110
   are in: refresh; the single-population sentences and the twice-offset sentence turn
   black; re-read them (the wording "is not biased" must match the level the combined
   posteriors print). Commit, PDF to the owner with SendUserFile
   (`dark_sirens_galaxies_and_AGN_draft_2026-11-25a.pdf`, a named copy in the scratchpad),
   and ONE multiple choice with the push request: "5, then 10" (the eight-parameter model on
   realizations 106 to 110, about 150 GPU-h; say what the five gave: H0 67.9 (+0.9, -0.8) at
   26%, log10 B 2.5 to 8.4, 1.3 [1.0, 1.8] times the mergers, and the two spin inputs outside
   their 90% intervals at 95% and 98%, which is the reason to run it), and stage 5's
   six-parameter runs on realizations 106 to 110 as well (about 36 GPU-h; its first five are
   the low draw). Tell them in the same message what the mixture sampling changed.
3. FIG. 7 WHEN STAGE 5 IS REAL (decide then, mine): its dotted curve (population assumed) is
   combined over fifteen while the single-population and host-dependent curves will be the
   low five; on the same five the assumed population gives 67.8 +- 0.8 (before the mixture
   re-runs). Either draw all three on the five, with the fifteen in the text, or say in the
   caption which curve uses which realizations. Table III states the counts per row already.
4. STAGE 5 (starts about 26 Nov): when realization 100's posteriors come in, look at its H0
   marginal (edge merger 494; with the mixture 1069 of its 8000 samples lie below the edge
   at the input H0, where production had 6).
5. Then Sec. 5 items 3 to 5 of this file (stage 3 and stage 5 subsections as they turn black,
   the independent Fable review against the style skill's self-check, small leftovers).
- OWED TO THE OWNER: nothing blocks. At the milestone: the PDF, the multiple choice, the
  push request (GitHub has e6d5313).
- CONTEXT: this session's context is long (resumed 18:55 UTC 23 Nov, five agents, eight
  refreshes). This block is a complete resume document; a `/clear` and "continue from the
  latest checkpoint" loses nothing.

### What happened on 23 Nov (all committed)
1. EIGHT-PARAMETER RUNS (host-dependent populations, `..._unif_then_rslice`): realizations 100
   to 103 finished; 105 on lane 2 (iteration 3634 at 10:51 UTC, delta ln Z 4.5; ends between
   19:30 UTC 23 Nov and 00:30 UTC 24 Nov). Host-dependent / single population on the same
   mergers, H0 median [90%] and half-width, ln B:
   100: 67.53 [66.09, 68.99], 1.45 / 67.65 [65.86, 69.72], 1.93; +19.4
   101: 66.84 [65.40, 68.44], 1.52 / 66.69 [64.94, 68.85], 1.95; +15.7
   102: 68.47 [66.70, 70.37], 1.84 / 68.87 [66.73, 70.48], 1.88; +8.9
   103: 70.59 [68.80, 72.04], 1.62 / 70.55 [68.49, 72.46], 1.98; +13.0
   105: (running) / 67.57 [65.74, 69.25], 1.75
   log10 B 8.4, 6.8, 3.9, 5.7. Offsets (inputs 5, 0.10): dmu_G 3.80, 4.61, 4.49, 5.71; dmu_chi
   0.163, 0.127, 0.147, 0.104 (only realization 100's dmu_chi has its input outside the 90%
   interval). f_AGN 0.288, 0.306, 0.327, 0.323; single population 0.336, 0.277, 0.418, 0.328,
   0.411 = the AGN share of the DETECTED mergers (0.36, 0.38, 0.364, 0.346, 0.351). Realization
   103 is high in every analysis of those mergers (host class and populations known: 69.88
   [68.39, 71.34]): the draw of that event set.
2. SEC. IV E IS REWRITTEN ON THREE STEPS (commit bfd84d4; PDF `..._draft_2026-11-23a.pdf` sent
   to the owner): (a) population ASSUMED, the galaxy class's mass and spin distribution for
   both host classes (stored grids `two_tracers_one_population`): H0 71.2 +- 0.8 over five,
   input beyond the 99.99% level; the same mergers with both populations at their inputs
   67.7 +- 0.8 (7.9%); paired shift 3.2 [2.2, 20.2] (the 20 is realization 101, whose posterior
   favors few mergers in AGN and reaches the upper prior edge); at twice the offsets no
   interior maximum in any of five (realization A's median 99.66, f_AGN < 0.10). (b) SINGLE
   population with inferred parameters: no H0 bias (five finished: 67.6, 66.7, 68.9, 70.5,
   67.6), peak between the classes (36.2 to 36.8), f_AGN the detected share. (c) HOST-DEPENDENT
   populations inferred: offsets measured, log10 B 3.9 to 8.4, H0 half-width 1.61 against 1.94
   in the mean (a single population needs about 1.4 times the mergers), f_AGN the input.
   With one population for both classes, assumed or inferred, f_AGN is the detected share
   (assumed: 0.35 +- 0.04, input at 97%). Abstract, roadmap sentence, Table III (rows
   "Population assumed"), Secs. IV E, F, G, captions of Figs. 6 to 8 (third curve "population
   assumed", dotted; new rows), Conclusions all follow. Keys HDAssumed, HDAssumedTwice, HDPops,
   HDSingleTwice, ObsAssumed, ObsPops; audit checks J4 to J6; spec in
   `paper_prd/notes/MACRO_REQUESTS.md` Sec. G. Last refresh (08:34 UTC): exit 0, audit 93 of
   93, figures 515 with 0 mismatches, 186 of 251 result macros of the text real, 19 pp,
   abstract 420 words rendered.
3. RUN LIST (commit 201a7e4): keys HDHost/HWHost read the `..._unif_then_rslice` runs (stage-V8
   exception `build_values.STAGE_V_KEY_RUNS`); stage 5's five eight-parameter runs are
   `host_dependent_obslike/realization_NNN_unif_then_rslice` (lane rank 10, 35.8 GPU-h each),
   QUEUED last on the lanes (three on lane 1, two on lane 2); nine never-queued pure-rslice
   entries dropped; twenty unused macros pruned.
4. E3, FIRST RESULT (single population inferred, TWICE the offsets, realization 100): H0 68.17
   [66.12, 70.65], mu_G 38.05, sigma_G 6.76, mu_chi 0.080, f_AGN 0.358 [0.270, 0.448]
   (detected share 0.408): no H0 bias where the assumed population is at the prior edge.
5. THE OBSERVED-LIKE HOST-DEPENDENT SETS (`obslike_f0p30_s1`, realizations 100 to 105, the sets
   of ALL of stage 5) LIE LOW, AND IT IS THEIR DRAW. Host class and both populations known:
   66.64 [66.12, 67.17], input at the 99.91% level, all of it in the AGN-hosted mergers; an
   Opus diagnosis (`tests/obslike_offset_step*.py`, results in the git-ignored
   `tests/results/obslike_offset_*`) found no inconsistency between generation and analysis.
   Ten NEW realizations (seeds 106 to 115, events and injection sets generated on CPU today;
   recipe in the memory `gw-agn-h0-offset-is-seed-scatter`), AGN-hosted mergers with the host
   class known, on CPU: 69.4, 67.9, 67.0, 67.1, 68.1, 67.7, 67.8, 68.7, 68.2, 67.2; combined
   67.87 [67.45, 68.29], input at the 38% level. Rule fixed beforehand (>= 67.2: fluctuation):
   STAGE 5 RUNS AS QUEUED. Stage 5's keys use the low five and will sit about 1 low: the text
   must say so with the fifteen beside them.

### Facts worth knowing
- THIS SESSION'S SLURM JOB is 1494715 on MIKO (limit 11:56 UTC 30 Nov). Background tasks and
  agents die with it.
- In the session shell `grep` is a function: in watcher tests use `command grep` and test for
  `"outcome": "error"` positively (attempt outcomes: finished, deadline, running, error).
- `tests/results/` and `working/data/` are git-ignored; tracked files under `working/data`
  are staged with `git add -u`.
- The agent that knows the paper chain and the run-list builder (three jobs today) is
  `a3140b9762ce96e03`; a fresh agent needs `notes/MACRO_REQUESTS.md` Sec. G and the docstrings
  of `production.py` / `make_runlist.py`.
- Not followed up: `mass_spectrum_alone` on the observed-like first five is low too (61.4
  [57.9, 65.9]); the combined `mass_spectrum_alone` posterior of `obslike_f1p00_s0` looks
  pathological (median 56.5, interval [56.4, 67.7]); the diagnosis agent's inference that the
  observed-like AGN-class intervals are about 10% too narrow (score variance 18 to 25% above
  the curvature; 3% for the peak family); the analysis-16 README does not yet describe the E2
  extension to 106 to 115; `\ObsAssumedHzeroCL`-type levels that round to 100 print ">99.99".
- The five host-class-known posteriors of the peak-dominated host-dependent sets scatter a
  little more than their widths (chi^2 9.9 on 4): no hunt.

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
- DONE 21 Nov 13:45 UTC: the agent's macros checked against the reference table (twenty at
  the pixels used throughout 67.9 +- 0.4, 47%; f_AGN 0.294 +- 0.019, 40%; finer pixels 68.3
  +- 0.5, 93%; audit 92 of 92; figure check 491, 0 mismatches); `main.tex` edited by me
  (Validation's first and last paragraphs; Table III's no-catalog row on host-dependent
  mergers is "< 66", a 90% upper limit, with a caption sentence; Fig. 8 draws that row as a
  cap with a left arrow, caption sentence); refresh exit 0 (18 pp; 163 of 218 result macros
  of the text real); committed; PDF `dark_sirens_galaxies_and_AGN_draft_2026-11-21b.pdf` sent.
  Macros `\SkyResTen...`, `\SkyResOffset...`, `\ClsKnownThirty...`, `\ClsUnknownThirty...`,
  `\Cls...Fifty...`, `\SkyResMCMin/MaxShiftHzero`, `\SkyResMCNarrowPct` are no longer used
  in the text (still defined; `\SkyResOffsetHzeroProd` now mixes twenty and ten: prune them
  with the next macros job). LEFT: Sec. IV F's sentence quoting `\HDNoneHzero` beside
  `\ObsNoneHzero` (wait for stage 5); Fig. 9(a)'s narrow curve shows slight facets (grid
  spacing of the combined posterior); `\MockzHorizonMax` is now 0.49 (unused in the text).
  The catalogs of seeds 106 to 120 (5.4 G each, on phy230054p) can be deleted when no further
  pixel size is wanted.
- NEXT JOB (not urgent; HDHost needs five realizations, about 24 Nov): Sec. 5 item 2 is now
  unblocked (the pixel-size decision is made: production pixels stay): one Opus agent maps the
  keys `HDHost`/`HWHost` to the `..._unif_then_rslice` runs, adds stage 5's eight-parameter
  entries the same way and drops the unqueued pure-rslice entries; with it, prune the unused
  macros named above. Then Sec. 5 items 3 to 5.
- RUNS: lane 1 (job 1425268, successor 1494378) resumed the eight-parameter run of
  realization 101 at iteration 3288 at 12:34 UTC (ends about 04:00 UTC 22 Nov), then 102.
  Lane 2 (job 1454491) on realization 103, iteration 1704 at 12:33 UTC. No watcher is armed.
- OWNER, 22 Nov 08:30 UTC (multiple choice): "Push now": PUSHED up to e6d5313 at 08:30 UTC
  (GitHub now has the twenty-realization Validation); each later push is asked for again.
  Funding: "Leave out for now" (no funding sentence; they supply the grant later; do not list
  it as open at every status).
- STATE 22 Nov 08:30 UTC: lane 1 (job 1425268, limit 10:00 UTC 23 Nov, successor 1494378)
  finished the eight-parameter run of realization 101 at 23:56 UTC 21 Nov and is on
  realization 102 (iteration 1983 at 08:12 UTC, ends about 09:00 UTC 23 Nov); lane 2 (job
  1454491) on realization 103 at iteration 4088 (delta ln Z 2.4; ends about 16:30 UTC 22 Nov),
  then 105 (about 29 h). All five late on 23 Nov. No watcher armed: check by hand.
  REALIZATION 101, eight parameters: H0 66.84 [65.40, 68.44], f_AGN 0.306 [0.230, 0.383],
  mu_G 35.12 [34.35, 35.86], sigma_G 5.24 [4.72, 5.76], mu_chi 0.005 [-0.016, 0.025],
  sigma_chi 0.096 [0.078, 0.111], dmu_G 4.61 [2.80, 6.39], dmu_chi 0.127 [0.082, 0.168]: ALL
  EIGHT INPUTS INSIDE (the spin offsets of realization 100 do not repeat); ln Z -3022.30 +-
  0.35; single population on the same mergers H0 66.69 [64.94, 68.85], ln Z -3037.99: ln B
  +15.7 (realization 100: +19.4); H0 half-width 1.52 against 1.95 (100: 1.45 against 1.93).
  The reading of Sec. 4 ("the expected headline of Sec. IV E is not showing") holds on two of
  five: host dependence detected decisively, H0 a quarter narrower, no H0 bias of the single
  population. Rewrite Sec. IV E, abstract, Conclusions when the five are in.
  COMING TO THE OWNER about 24 Nov: "5, then 10" (extend the eight-parameter model to
  realizations 106 to 110, about 150 GPU-h), with the five results; multiple choice.

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
