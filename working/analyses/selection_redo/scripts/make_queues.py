#!/usr/bin/env python
"""Emit the full selection-mode campaign as explicit, auditable command lines.

One task per line, in the order they should run. Nothing here is implicit: each
line carries its own survey paths, anchors and selection fits, so a task can be
copied out of the queue and run by hand to reproduce exactly one grid.

Design is the ARCHIVED campaign's, cell for cell — the only thing that changes
is the completeness estimator (`--c_mode selection` plus the per-catalog
magnitude fits). Anchors, grids, seeds and guards are untouched, because the
question is what the estimator does, not what a new design does.

    python make_queues.py                 # writes ../a{3,4,5,6}/queue/tasks.txt

Order (owner's, 2026-08-10): the a3 ladder runs first because a6's three
diagonal cells ARE the a3 ladder cells; a6 then completes with 8 new grids.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # working/analyses/selection_redo
SEED = 100

# Filled in for J2 by run_campaign.sh; kept as shell vars so one queue file
# works on either machine.
DATA = "${DATA_ROOT}/seed%d" % SEED
FITS = "${FITS_DIR}"
SCAN = "${SCRIPTS}/scan_h0f.py"
SAMPLE = "${SCRIPTS}/sample_4d.py"

GRID = "--scan joint --h0_grid 50 100 201 --f_grid 0 1 41"
COMMON = ("--universe_model dark_sirens --catalog_sky_weighting field "
          "--selection_neff_guard hard --max_likelihood_variance 1e6 "
          "--kde_window 4096 --kde_window_nsigma 8 "
          "--sel_batch_size 50000 --pe_event_block 25 "
          "--h0_true 67.74 --f_true 0.295 --device gpu")
TRUTH_ANCHORS = "--log10n0 -3.0 --log10n0_c2 -5.0"

# The AGN "complete" catalog is represented inside selection mode by a nominal
# m_lim deeper than the population's faintest object (max app_mag 23.59). That
# gives an IDENTICALLY zero missing budget over the whole horizon -- verified,
# max(1 - C_sel) = 0.000e+00, see
# experiments/experiment_dsmaster_4d_recheck/results/
# verify_complete_is_estimator_independent.json -- so this is an exact
# representation of "complete", not an approximation.
LEVEL_FIT = {lvl: f"{FITS}/fit_%s_{lvl}_schechter.json"
             for lvl in ("m18", "m19", "m20", "m21", "complete")}


def survey(tracer, level):
    return f"{DATA}/surveys/survey_{tracer}_{level}_ns32.h5"


def fit(tracer, level):
    return LEVEL_FIT[level] % tracer


def cell(out_tag, gal_lvl, agn_lvl, anchors=TRUTH_ANCHORS, extra=""):
    """One joint (H0, f_AGN) grid. Survey order GAL then AGN, so fcat_2 is f_AGN."""
    return (f"${{PY}} -u {SCAN} {COMMON} {GRID} "
            f"--survey_path {survey('gal', gal_lvl)} {survey('agn', agn_lvl)} "
            f"--gw_path {DATA}/events/events.h5 "
            f"--gwselection_path {DATA}/injections/injections_targeted.h5 "
            f"--c_mode selection "
            f"--selection_fit {fit('gal', gal_lvl)},{fit('agn', agn_lvl)} "
            f"{anchors} {extra} --out_tag {out_tag}").replace("  ", " ")


def write(analysis, lines, note):
    q = ROOT / analysis / "queue"
    q.mkdir(parents=True, exist_ok=True)
    (q / "tasks.txt").write_text("\n".join(lines) + "\n")
    (q / "README.txt").write_text(note.strip() + "\n")
    print(f"  {analysis}/queue/tasks.txt   {len(lines)} tasks")
    return len(lines)


def main():
    total = 0

    # ---- a3: the completeness ladder, seed 100 only (owner, 2026-08-10) ------
    # Rung 0 (complete GAL x complete AGN) is NOT rerun: with both catalogs
    # complete the missing budget is identically zero under every estimator
    # (verified), so the archived per_pixel grid is the same number.
    a3 = [cell(f"joint_{lvl}_s{SEED}", lvl, lvl)
          for lvl in ("m21", "m20", "m19", "m18")]
    total += write("a3", a3, """
a3 -- completeness ladder under c_mode=selection. Seed 100 only.
Both tracers at the SAME depth (the diagonal). Anchors FIXED at truth, as in
the archived campaign. Rung 0 (complete) is reused from the archive: verified
estimator-independent, max(1 - C_sel) = 0.000e+00 for both tracers.
These four cells are also a6's diagonal, which is why this queue runs first.
""")

    # ---- a6: the GAL-depth x AGN-depth surface ------------------------------
    # 8 new grids; the 3 diagonal cells come from a3 above and the GAL m18 x
    # AGN complete cell is a4's oracle probe.
    a6 = []
    for g in ("m20", "m19", "m18"):
        for a in ("complete", "m20", "m19", "m18"):
            if a == g:
                continue                      # diagonal: from the a3 queue
            if g == "m18" and a == "complete":
                continue                      # this is a4's oracle probe
            if g == "m18" and a == "m20" or g == "m18" and a == "m19" \
               or g != "m18":
                a6.append(cell(f"joint_g{g}_a{a}_s{SEED}", g, a))
    total += write("a6", a6, """
a6 -- relative completeness: GAL depth x AGN depth surface, under selection.
8 new grids. The 3 diagonal cells come from a3's queue; GAL m18 x AGN complete
is a4's oracle probe. m21 is excluded (identical to complete inside the horizon).
Anchors FIXED at truth in every cell, deliberately, so the surface is not
confounded with the anchoring axes of a4/a5.
The two AGN=complete cells run under selection with the complete catalog's
nominal deep m_lim -- they are NOT reused from the archive, because their GALAXY
catalog is flux-limited and the estimator does change those cells.
""")

    # ---- a4: AGN density anchoring ------------------------------------------
    # 6 arms x 4 rungs. log10n0_c2 = -5 + log10(factor); the factor-1.0 arm is
    # a3's own grid, referenced not rerun (as in the archived campaign).
    import math
    ARMS = {"a05": 0.5, "a07": 0.7, "a09": 0.9, "a11": 1.1, "a13": 1.3, "a20": 2.0}
    a4 = []
    for lvl in ("m21", "m20", "m19", "m18"):
        for tag, factor in ARMS.items():
            n0c2 = -5.0 + math.log10(factor)
            a4.append(cell(f"joint_{lvl}_{tag}_s{SEED}", lvl, lvl,
                           anchors=f"--log10n0 -3.0 --log10n0_c2 {n0c2:.6f}"))
    # the oracle probe: GAL m18 x AGN complete, both densities at truth
    a4.append(cell(f"joint_m18_oracle_s{SEED}", "m18", "complete"))
    total += write("a4", a4, """
a4 -- AGN density anchoring under selection. Seed 100 only.
24 arms = 4 rungs x 6 mis-anchoring factors (0.5,0.7,0.9,1.1,1.3,2.0), GAL
anchor fixed at truth throughout, plus the oracle probe (GAL m18 x AGN complete,
both densities at truth). The factor-1.0 arm is a3's own grid, referenced.
The oracle runs under selection with the complete AGN catalog's nominal deep
m_lim; it is not reused from the archive, because its galaxy catalog is
flux-limited and the estimator changes that cell.
""")

    # ---- a5: both anchors free, dynesty -------------------------------------
    a5 = []
    for lvl in ("m18", "m19", "m20", "m21"):     # cheapest first
        a5.append(
            f"${{PY}} -u {SAMPLE} {COMMON} "
            f"--survey_path {survey('gal', lvl)} {survey('agn', lvl)} "
            f"--gw_path {DATA}/events/events.h5 "
            f"--gwselection_path {DATA}/injections/injections_targeted.h5 "
            f"--c_mode selection "
            f"--selection_fit {fit('gal', lvl)},{fit('agn', lvl)} "
            f"--n0_prior -4.0 -1.0 --n0c2_prior -6.0 -4.0 "
            f"--sampler dynesty --nlive 1000 --dlogz 0.1 --maxcall 500000 "
            f"--rstate_seed 7 --skip_wiring "
            f"--checkpoint_file ${{RESULTS}}/campaign_{lvl}_dynesty_s{SEED}.ckpt "
            f"--checkpoint_every 900 "
            f"--out_tag campaign_{lvl}_dynesty_s{SEED}")
    total += write("a5", a5, """
a5 -- both completion densities FREE, sampled with dynesty, under selection.
4 rungs, seed 100, rstate 7, nlive 1000, dlogz 0.1 -- the archived campaign's
settings exactly. Flat priors log10n0 [-4,-1] (the widened one: the archived
per_pixel run railed against [-4,-2]) and log10n0_c2 [-6,-4].
The four LF theta labels are PINNED at their fit centres, so this samples the
same 4 parameters the archived campaign did.
m21 is by far the most expensive rung (24.5 h in the archive); it runs last.
""")

    print(f"\n  {total} tasks total")


if __name__ == "__main__":
    main()
