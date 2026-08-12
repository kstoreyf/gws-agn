a7 -- the mechanism check: does the estimator's failure scale with OCCUPANCY?

Why this exists. The archived per_pixel f_AGN bias tracks AGN pixel occupancy at
R2 = 0.965, and a peer team (desi_darksirens_selection) independently traced a
bias in THEIR per-pixel-normalised catalog prior to low pixel occupancy, closing
it by coarsening nside 128 -> 64. But our flux-limited ladder cannot test that:
in a fixed-population flux-cut mock, occupancy is completeness times a per-tracer
constant (measured log10 occ = 0.999 log10 C + 1.08 for AGN, R2 0.86), so the
two variables are structurally collinear and any law we fit to the a6 surface
fits both equally well by construction.

Changing HEALPix nside is the one manipulation that moves occupancy while holding
the catalog, the completeness, the population and the events fixed.

    m18, C = 9.5%, identical 8,274 AGN and 821,444 galaxies, only the binning moves
      nside 16  ->  lambda = 2.693 AGN per pixel,   8.4% empty
      nside 32  ->  lambda = 0.673                 52.8% empty   (already have)
      nside 64  ->  lambda = 0.168                 84.8% empty
    1.2 dex of occupancy at fixed everything else.

Prediction registered by the peer team BEFORE the run: their mechanism depends on
counts per pixel and nothing else, so if it is the same disease the offset must
move along this axis. If the offset is flat in nside and only responds to the
flux cut, our estimator has a different mechanism and the cross-code story is
weaker. Either answer is worth having.

`selection` arms are the control: the parametric estimator has no per-pixel
completeness ratio to be shot-noise-limited, so it should be flat in nside.

    python make_a7_queue.py

CONTENTS (6 tasks, ~2.5 GPU-h)
  2 policy-inertness f-scans at nside 64 (zero vs volume) -- must be identical
  2 per_pixel grids at nside 16 and 64   -- the axis under test
  2 selection grids at nside 16 and 64   -- the control, expected flat

The nside-32 midpoints are NOT rerun: per_pixel comes from the archived a3 grid
(archive/analysis_3_.../results/joint_m18_s100.h5) and selection from this
campaign's a3 (analysis_3_.../results/joint_m18_s100.h5). Same cell, already paid for.

READ THE ANSWER AS: offset vs log10(lambda) where lambda = N_AGN / npix is
2.693 / 0.673 / 0.168 at nside 16 / 32 / 64. Slope significantly non-zero under
per_pixel and consistent with zero under selection => occupancy mechanism
confirmed, cross-code. Flat under both => our bias is not the peer's disease.
