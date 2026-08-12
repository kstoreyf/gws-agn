#!/usr/bin/env python
"""a7 -- the mechanism check: does the estimator's failure scale with OCCUPANCY?

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
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEED = 100

DATA = "${DATA_ROOT}/seed%d" % SEED
FITS = "${FITS_DIR}"
SCAN = "${SCRIPTS}/scan_h0f.py"

GRID = "--scan joint --h0_grid 50 100 201 --f_grid 0 1 41"
COMMON = ("--universe_model dark_sirens --catalog_sky_weighting field "
          "--selection_neff_guard hard --max_likelihood_variance 1e6 "
          "--kde_window 4096 --kde_window_nsigma 8 "
          "--sel_batch_size 50000 --pe_event_block 25 "
          "--h0_true 67.74 --f_true 0.295 --device gpu")
ANCHORS = "--log10n0 -3.0 --log10n0_c2 -5.0"


def surveys(ns):
    """nside 32 is the campaign's own directory; 16/64 are the a7 variants."""
    d = "surveys" if ns == 32 else f"surveys_ns{ns}"
    return (f"{DATA}/{d}/survey_gal_m18_ns{ns}.h5 "
            f"{DATA}/{d}/survey_agn_m18_ns{ns}.h5")


def cell(tag, ns, c_mode, extra=""):
    sel = (f"--c_mode selection --selection_fit "
           f"{FITS}/fit_gal_m18_schechter.json,{FITS}/fit_agn_m18_schechter.json"
           if c_mode == "selection" else f"--c_mode {c_mode}")
    return (f"${{PY}} -u {SCAN} {COMMON} {GRID} "
            f"--survey_path {surveys(ns)} "
            f"--gw_path {DATA}/events/events.h5 "
            f"--gwselection_path {DATA}/injections/injections_targeted.h5 "
            f"{sel} {ANCHORS} {extra} --out_tag {tag}").replace("  ", " ")


def main():
    q = []

    # --- the policy-inertness check, FIRST and cheap --------------------------
    # Under FIELD weighting an empty pixel carries zero count weight and is -inf
    # regardless of complete_empty_pixel_policy (prior.py:767-777). That matters
    # because this axis drives the empty fraction 8% -> 85%; if the policy were
    # live, a7 would measure policy x occupancy rather than occupancy. Code says
    # inert; these two f-scans at the WORST case (nside 64, 85% empty) prove it.
    # They must come out bit-identical.
    for pol in ("zero", "volume"):
        q.append(
            (f"${{PY}} -u {SCAN} {COMMON} "
             f"--scan f --f_grid 0 1 11 --h0_fixed 67.74 "
             f"--survey_path {surveys(64)} "
             f"--gw_path {DATA}/events/events.h5 "
             f"--gwselection_path {DATA}/injections/injections_targeted.h5 "
             f"--c_mode per_pixel {ANCHORS} "
             f"--complete_empty_pixel_policy {pol} "
             f"--out_tag policyprobe_ns64_{pol}_s{SEED}").replace("  ", " "))

    # --- the axis ------------------------------------------------------------
    # nside 32 is NOT rerun: per_pixel ns32 is the archived a3 grid and
    # selection ns32 is this campaign's own a3 grid. Both are the same cell.
    for c_mode in ("per_pixel", "selection"):
        for ns in (16, 64):
            q.append(cell(f"joint_m18_ns{ns}_{c_mode}_s{SEED}", ns, c_mode))

    out = ROOT / "a7" / "queue"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tasks.txt").write_text("\n".join(q) + "\n")
    (out / "README.txt").write_text(__doc__.strip() + """

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
""")
    print(f"  a7/queue/tasks.txt   {len(q)} tasks")


if __name__ == "__main__":
    main()
