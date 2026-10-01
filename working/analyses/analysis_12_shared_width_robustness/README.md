# Analysis 12 — can one shared population width absorb an environmental offset?

Seed 100 only. Science exploration, not calibration. Same mock, surveys, injections,
detection and PE model as Analyses 10 and 11.

## Question

Analysis 11 inferred the reference population (μ_G, μ_χ) and the AGN offsets (Δμ_G, Δμ_χ)
with H0 free (11D), holding the Gaussian-peak width σ_G and the spin width σ_χ at their
fiducials. A broader common population could, in principle, imitate a branch difference.
Each stress test frees ONE width, shared by GAL and AGN, on top of 11D:

| arm | free parameters | prior on the added width | truth |
|---|---|---|---|
| 12M | 11D + σ_G | [1, 10] M☉ (darksirens GAUSS_SIGMA) | 5 M☉ |
| 12χ | 11D + σ_χ | [0.01, 1] (darksirens CHI_SIGMA) | 0.10 |

No branch-dependent widths, no peak fraction, and never both widths in one fit (brief
§§22–25). Decision metric: R = W(offset with the width free) / W(offset in 11D), the median
shift, and ρ(width, offset).

## Likelihood

`a11_likelihood.build_a11(shared_width=("sigma_G",) | ("sigma_chi",))` in
`../analysis_11_free_common_population/scripts/` releases the base σ label on top of the
Analysis-11 pair. Catalogue 2 has no σ copy (the per-catalogue block is only G.mu_c2 and
mu_chi_c2), so the released width is common to both branches. Closure: `scripts/a12_closure.py`
→ `diagnostics/a12_closure.json`.

## Layout

| path | contents |
|---|---|
| `scripts/a12_sampler.py` | the Analysis-11 sampler (problems `12M`, `12chi`) pointed at this directory |
| `scripts/a12_closure.py` | closure K1–K3 before sampling |
| `scripts/submit_a12_gpu.sbatch` | rita GPU job (RITA-GPU, QOS rita) |
| `results/` | per-run `a11_ns_<problem>_dynesty_n200_s<seed>.{json,npz}` (git-ignored); merged `a12_<problem>.{json,npz}` |
| `queue/` | dynesty checkpoints and pickled Results (git-ignored; merge input) |
| `diagnostics/`, `figs/`, `logs/` | closure record, figures, Slurm logs |

Run tags keep the sampler's `a11_ns_` prefix: they are the run IDs the checkpoints and the
merge refer to. `STATE.md` is the running state, `GATES.md` the criteria, `REPORT.md` the
results.
