fu_probes -- the two owner-approved follow-up probes (2026-08-23), seed 100,
darksirens 0c5b3db, run on HildaFS SLURM (HENON-GPU A100-40), one GPU,
sequential.

Task 1: KDE-window pin (a7 residual +0.028/dex).
  Clone of a7's joint_m18_ns64_selection_s100 with --kde_window_nsigma 8 -> 32.
  Rationale, checked against the survey files before launch: at m18 the
  W=4096 galaxy cap NEVER binds (max row occupancy 189 at ns32, 61 at ns64),
  so the only window quantity that varies across nside is the traced z
  half-width n_sigma * max_row(sigma_eff), which is set from the pixel's own
  occupants -- exactly the a7 REPORT's candidate. n_sigma=32 makes every ns64
  window at least as wide as its parent ns32 window (guaranteed: 4*(1+z_child)
  >= 1+z_parent for all z >= 0). Compare against the campaign's
  a7/results/joint_m18_ns64_selection_s100.json:
    - offset moves toward the ns32 value  => the bandwidth window is live;
    - unchanged                            => window ruled out, N_eff guard next.

Task 2: zero-density probe (the AGN=complete offset, exec summary open item 1).
  Clone of a3's joint_complete_s100 with log10n0 = log10n0_c2 = -24.
  Selection's C == 1 and the -24 density limit both mean "no completion
  contribution" and must agree on the same code. If they do not, the
  completion path is live at C == 1 -- the mechanism behind the -0.055..-0.069
  f_AGN offset in the four AGN=complete cells.

Environment differences vs the campaign (J2 H100 -> HildaFS A100), recorded:
same darksirens SHA 0c5b3db (fresh worktree at
/hildafs/projects/phy230014p/magana/src/darksirens-0c5b3db), same data bytes
(working/data symlinks -> phy220048p/gws-agn-data-v3), same fits, same flags.
Cross-GPU float noise is far below every effect probed here.
