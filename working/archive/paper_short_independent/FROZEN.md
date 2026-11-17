# Frozen draft

Archived 2026-11-17 (about 20:05 UTC) at the owner's request: "Lets stick to the long paper. You
did a better job there. So i would archive the short paper." Repository state before the move:
5d4688b.

This was the letter, written from `BRIEF_v2.md` around its own experiment (`EXPERIMENT.md`: the 90%
half-width of H0 against the fraction of mergers hosted by AGN, with the host class unknown and
known, and the probability of an AGN host for each merger). It is a record only and is not
maintained. The one maintained paper is `working/paper_prd/`.

What went into the long paper (design: `paper_prd/notes/PAPER_SKELETON.md`, Sec. 9):
- the experiment's results, as Sec. IV B and parts of the host-dependent and observed-like
  subsections, with the figure `fig_classes` (this draft's `fig1_budget` and panel (c) of
  `fig_populations`);
- the code that computes them: `scripts/results.py` here became `paper_prd/scripts/run_posteriors.py`
  and `paper_prd/scripts/classes.py`; nothing in the long paper reads from this directory.

The experiment's runs are stages E1, E2 and E3 of `working/analyses/analysis_16_production_runs/`;
that directory's README describes the analyses. `values` and `refs.bib` are links into
`working/paper_prd/`; the scripts here are not kept working against later changes of that chain.
