# Figures

- `fig_f_recovery_ladder_fix.{pdf,png}` — post-fix f_AGN posteriors per completeness rung (complete, m<21/20/19/18) with the sky-shuffle nulls dashed, truth 0.30; from `scripts/fig_recovery_ladder.py` reading `results/fscan_<lev>_fix.h5` and `results/fscan_null_<lev>_fix.h5`.
- `fig_h0_recovery_ladder_fix.{pdf,png}` — post-fix H0 marginals of the joint grids per rung (legend quotes the 68% half-widths; the width is non-monotonic in depth), truth 67.74; from `scripts/fig_recovery_ladder.py` reading `results/joint_<lev>_fix.{h5,json}`.
- `fig_joint_h0f_ladder_fix.{pdf,png}` — per-rung 68/90% credible regions in the (H0, f_AGN) plane, small multiples on a common zoom, truth cross; from `scripts/fig_recovery_ladder.py` reading `results/joint_<lev>_fix.h5`.
