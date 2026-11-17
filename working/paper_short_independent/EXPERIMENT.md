# The experiment behind the short paper (17 Nov 2026)

The short paper answers one question: where does the H0 information of 1000 binary black hole
mergers sit when some of them are hosted by AGN, and what do the black holes' masses and spins
add? It needs its own, mostly cheap, runs. This file is for the people who generate the event sets
and run the analyses. The writer's brief is `BRIEF.md`.

## Settings (as the production runs of `analyses/analysis_16_production_runs/`)
darksirens-core e7c3007; complete catalogs; hosts normalized over the redshift grid to z = 1.10;
8000 posterior samples per event, redrawn with a recorded seed; half-variance correction of ln L;
H0 uniform on [50, 100], f_AGN uniform on [0, 1]; each realization with its own injection set.
Realizations 100, 101, 102, 103, 105.

## Event sets (1000 detected mergers each; core population)
Two numbers describe a set: the fraction f of mergers hosted by AGN, and the size s of the
population difference, AGN-hosted mergers having the mass peak higher by s x 5 Msun and the mean
effective spin higher by s x 0.10. `working/data/experiment_sets.json` maps (family, f, s) to the
events file suffix; the run list reads it.

| s \ f | 0 | 0.1 | 0.3 | 0.5 | 1 |
|---|---|---|---|---|---|
| 0 (equal populations) | exists (`puregal`) | exists (`fagn0p10`) | exists (`corepop`) | new | exists (`pureagn`) |
| 1 | | new | exists (`marked_dmu0p10_dmuG5`) | new | |
| 2 | | new | new | new | |

Observed-like mass spectrum (after its injection set exists): f = 0, 0.3, 1 at s = 0 and f = 0.3 at
s = 1 (the last two of the middle column exist).

## Analyses, named by what they do
All with the populations held at their inputs (grids), except the last line.
- `mass_spectrum_alone`: no catalog. H0, and f where the populations differ.
- `galaxy_catalog_alone`: every merger matched to the galaxy catalog, one population (the galaxy
  class). H0. On the f = 0 sets (where it is the right analysis) and on the mixed sets (where it is
  today's standard analysis applied to mergers of two kinds).
- `two_tracers_one_population`: both catalogs, f_AGN free, both tracers given the galaxy-class
  population. (H0, f_AGN). The right analysis at s = 0 with the labels unknown; at s > 0 the analysis
  that ignores the difference with the population held fixed.
- `two_tracers_two_populations`: both catalogs, f_AGN free, the AGN tracer given its own population
  (offsets at their inputs). (H0, f_AGN). Only at s > 0.
- `hosts_known`: the true host class of every merger is given. AGN-hosted mergers with the AGN
  catalog and the AGN-class population, galaxy-hosted mergers with the galaxy catalog and the
  galaxy-class population, each with its own selection term; the H0 posteriors multiply.
- Membership: for realization 100 at f = 0.3 and s = 0, 1, 2, the probability that each merger is
  AGN-hosted, r_i = f L_i,AGN / (f L_i,AGN + (1 - f) L_i,gal) at the posterior median, from
  `two_tracers_one_population` (positions alone) and `two_tracers_two_populations` (positions with
  masses and spins), stored with the true host class.
- Populations inferred (nested sampling): the six-parameter single population on (f = 0.3, s = 2),
  five realizations. The corresponding runs at s = 0 and s = 1, and the eight-parameter analysis
  at s = 1, are stages 3 and 4 of the production list.

## Cost and order
About twenty minutes of GPU per event set for the grids (two 2-D grids of 1147 cells and up to
four 1-D grids of 132 cells at 0.39 s per call): 55 sets, about 18 GPU-h. The five six-parameter
runs at s = 2: about 22 GPU-h. The observed-like subset: about 8 GPU-h.
Order on the two A100s: production stage 1 (3 GPU-h; it already contains the f = 0.3, s = 0
two-tracer grids, the all-AGN and all-galaxy sets and the no-catalog grid, so do not repeat them),
then this experiment, then production stages 2 to 4.

## What feeds what
- Fig. 1 (H0 half-width against f): every grid above; the open symbol from production stage 4.
- Fig. 2 (labels): the membership output.
- Fig. 3 (bias): production stages 3 and 4 and the s = 2 nested runs.
