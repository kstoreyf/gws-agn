# Analysis 13 — both shared population widths free together

**Status:** set up 2026-10-01; closure and the single-GPU sampler smoke test first, then the
two-seed production run (owner-gated), then the calibration campaign (owner-gated, after the
two seeds).

Seed 100. Science exploration, not calibration. Same mock, surveys, injections, detection and
PE model as Analyses 10–12.

## Question

Analysis 12 freed one shared width at a time and found that neither absorbs its offset. The
model recommended for calibration (Analysis 11 REPORT §11) frees both at once:

(H0, f_AGN, μ_G, Δμ_G, μ_χ, Δμ_χ, σ_G, σ_χ)

with priors H0 [60, 76], f_AGN [0, 1], μ_G [31, 39], Δμ_G [−4, 10], μ_χ [−0.1, 0.1],
Δμ_χ [−0.05, **0.30**] (extended from 0.20; 12χ reached that edge), σ_G [1, 10], σ_χ [0.01, 1].
Does the joint model reproduce the one-at-a-time results (do the two widths interact), and is it
fit to calibrate?

## Likelihood and sampler

`a11_likelihood.build_a11(shared_width=("sigma_G", "sigma_chi"))`; sampler problem `13` in
`../analysis_11_free_common_population/scripts/a11_sampler.py`, run through
`scripts/a13_sampler.py` (outputs into this directory).

## Layout

| path | contents |
|---|---|
| `REPORT.md` | results (not yet) |
| `STATE.md`, `GATES.md` | running state; criteria registered before results |
| `scripts/` | `a13_sampler.py`, `a13_closure.py`, `submit_a13_gpu.sbatch` |
| `results/`, `queue/` | runs and merges; sampler checkpoints (git-ignored) |
| `diagnostics/`, `figs/`, `logs/` | checks, figures, Slurm logs |
