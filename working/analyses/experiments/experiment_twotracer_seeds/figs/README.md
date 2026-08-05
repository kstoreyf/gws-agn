# Figures

- `fig_f_recovery_seeds.{pdf,png}` — all 12 post-fix f_AGN posteriors (orange) over the faded pre-fix ensemble (blue), ensemble mean-of-medians markers, truth 0.30; from `scripts/fig_recovery_seeds.py` reading `results/fscan{,_fix}_s73xx.h5` and `results/seeds_summary{,_fix}.json`.
- `fig_h0_recovery_seeds.{pdf,png}` — the 12 joint-grid H0 marginals pre vs post fix (ensemble mean offset -3.22 ± 0.55 → +0.44 ± 0.36), truth 67.74; from `scripts/fig_recovery_seeds.py` reading `results/joint{,_fix}_s73xx.h5` and `results/seeds_summary{,_fix}.json`.
- `fig_joint_medians_seeds.{pdf,png}` — scatter of the 12 joint (H0, f_AGN) medians pre (faded) → post (solid) with per-seed connectors, truth cross, and the mean quoted 68% interval ellipse per ensemble; from `scripts/fig_recovery_seeds.py` reading `results/seeds_summary{,_fix}.json`.
- `seeds_fix_strip.png`, `seeds_fix_widths.png` — earlier pre/post diagnostic panels from `scripts/make_fix_figures.py`.
