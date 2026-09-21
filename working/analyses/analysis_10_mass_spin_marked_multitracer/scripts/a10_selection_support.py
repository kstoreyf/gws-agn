#!/usr/bin/env python
"""Analysis 10, Gate B: does the EXISTING seed-100 injection set support a second
per-catalog population coordinate -- the AGN branch's Gaussian-peak LOCATION?

Analysis 8's Gate B asked this for the SPIN mark and could answer it almost
analytically: the detection rule does not depend on chi_eff, so a normalised
spin density cannot move the detectable fraction, and the branch-specific
selection factor barely moved.  ``G.mu`` is different in kind.  Detection
depends on the chirp mass, so moving the AGN branch's peak from 35 Msun to
25 or 45 moves BOTH the population density evaluated at every injection AND
the detectable fraction of that branch.  The selection factor mu(Lambda) is
therefore genuinely dmu_G-dependent, and whether the stored proposal still
carries it with enough effective samples is a HARD gate, not a formality.

This script GENERATES NO INJECTIONS.  If the gate fails it writes the
diagnosis (which branch, which dmu_G, which mass range) and a minimal proposal
design into the JSON and stops; the decision to generate belongs to the owner.

What it measures
----------------
(a) LIVE evaluations of the real dark-siren selection term, through the guard
    spy ``a8_likelihood`` installs, at H0 = 67.74:

      * 5 dmu_G x 5 f_AGN x 3 dmu_chi = 75 cells (the registered grid);
      * the full 21-node dmu_G axis at f = 1 -- the PURE AGN branch, where the
        branch-specific selection factor is not diluted -- for dmu_chi in
        {0, +0.10}: 42 cells;
      * an INERTNESS probe at f = 0: the GAL branch carries no per-catalog
        coordinate, so its logL must be BITWISE identical as dmu_G moves.
        That is simultaneously a liveness control (if the f = 1 axis also
        froze, the coordinate would be dead -- Analysis 8's trap).

    Per cell: log_mu, N_eff, threshold, N_eff/threshold, sigma2_total,
    pe_variance_sum, sel_logL, guard pass/fail, seconds.

(b) WEIGHT TAILS from the injection file, per branch population, with the
    generator's own density in the same canonical coordinates as the stored
    pdraw: N_eff, the largest normalised weight, the share of sum(w) carried by
    the top 10/100/1000 injections, and a coverage table in source-frame
    primary mass.  This is the POPULATION-ONLY proxy: it carries the smooth
    uniform-in-comoving-volume redshift prior, not the catalog one, so it
    diagnoses the branch population's demand on the proposal, while (a)
    measures the actual dark-siren selection term.  Labelled as such
    everywhere.

(c) DECISION.  PASS iff every cell of (a) passes the hard guard AND, on the
    posterior-relevant region (f <= 0.5, dmu_chi <= +0.15, all dmu_G in
    [-10, +10]), N_eff/threshold >= 2.

The events file
---------------
The selection term never reads the events file.  It enters the guard only
through nEvents and pe_variance_sum, both of which set the THRESHOLD:

    threshold = max(5 N_obs, N_obs^2 / (max_likelihood_variance - pe_var_sum))

With the analysis-2 setting max_likelihood_variance = 1e6 and N_obs = 1000 the
second term is ~1.0 for any pe_variance_sum up to ~1e6, so the threshold is
5 N_obs = 5000 and is INSENSITIVE to which 1000-event file is used.  The
Analysis-10 mock is being generated in parallel and does not exist yet, so the
Analysis-8 marked file ``events_marked_dmu0p10.h5`` is used for the build and
the measured pe_variance_sum is reported separately, with the margin by which
it would have to change to move the threshold.

EXACT COMMAND LINES
-------------------
    A10=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_10_mass_spin_marked_multitracer
    source .../analysis_9_marked_multitracer_H0_fagn/scripts/env_a9.sh

    "${A9_PY}" $A10/scripts/a10_selection_support.py --stage file       # CPU
    sbatch --export=ALL,STAGE=likelihood $A10/scripts/submit_a10_gate_b_rita.sbatch
    "${A9_PY}" $A10/scripts/a10_selection_support.py --stage assemble   # CPU
"""
import sys

sys.dont_write_bytecode = True          # the A8/A9 trees are READ-ONLY

import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A10_DIR = HERE.parent
DIAG = A10_DIR / "diagnostics"
LOGS = A10_DIR / "logs"

A9_SCRIPTS = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/"
                  "analysis_9_marked_multitracer_H0_fagn/scripts")
DATA_ROOT = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data/seed100")
INJ_TARGETED = DATA_ROOT / "injections" / "injections_targeted.h5"
DARKSIRENS_A8 = "/hildafs/projects/phy230014p/magana/src/darksirens-a8"
GWS_AGN_REPO = "/hildafs/projects/phy230014p/magana/gws-agn"

for p in (str(HERE), str(A9_SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

# --------------------------------------------------------------------------- #
# The registered grids
# --------------------------------------------------------------------------- #
H0_ANCHOR = 67.74
DMU_G_COARSE = [-10.0, -5.0, 0.0, 5.0, 10.0]
F_GRID = [0.0, 0.275, 0.50, 0.75, 1.0]
DMU_CHI_GRID = [0.0, 0.10, 0.25]
DMU_G_AXIS = [round(x, 6) for x in np.linspace(-10.0, 10.0, 21)]
AXIS_DMU_CHI = [0.0, 0.10]
INERT_DMU_G = [-10.0, 0.0, 10.0]
INERT_DMU_CHI = 0.10

DMU_G_PLANT = 5.0
DMU_CHI_PLANT = 0.10
SIGMA_CHI = 0.10
MU_CHI_GAL = 0.0
MU_G_FID = 35.0

# Posterior-relevant region for the margin requirement.
POST_F_MAX = 0.50
POST_DMU_CHI_MAX = 0.15
MARGIN_REQUIRED = 2.0

# Coverage table: 2.5 Msun bins from 20 to 60 in SOURCE-frame primary mass.
COVER_LO, COVER_HI, COVER_DM = 20.0, 60.0, 2.5
COVER_MIN_PER_MSUN = 100.0

STAGE_FILE_JSON = DIAG / "_a10_selsupport_file.json"
STAGE_LIKE_JSON = DIAG / "_a10_selsupport_likelihood.json"
STAGE_LIKE_JSONL = DIAG / "_a10_selsupport_likelihood.jsonl"
OUT_JSON = DIAG / "a10_selection_support.json"
OUT_MD = DIAG / "a10_selection_support.md"


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(repr(o))


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))
    print(f"wrote {path}")


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


def _fin(x):
    """JSON-safe float: non-finite -> None (the companion *_finite flag carries it)."""
    x = float(x)
    return x if np.isfinite(x) else None


# =========================================================================== #
# Stage "file" -- weight tails and mass coverage, from the injection file alone
# =========================================================================== #
def stage_file(args):
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import h5py

    sys.path.insert(0, "/hildafs/projects/phy230014p/magana/gws-agn/working/data")
    import generate_dataset as gd
    gmd = gd.import_gmd(Path(DARKSIRENS_A8))
    import a10_likelihood as a10

    t_start = time.time()
    out = {
        "what": ("population-only weight tails and source-mass coverage for the "
                 "AGN branch as its Gaussian-peak location moves.  PROXY: the "
                 "numerator is the generator's population density in the same "
                 "canonical coordinates as the stored pdraw, carrying the smooth "
                 "uniform-in-comoving-volume redshift prior, NOT the catalog "
                 "redshift prior the dark-siren selection term uses.  The real "
                 "selection term is measured live in stage 'likelihood'."),
        "provenance": {
            "gws_agn_sha": _run(["git", "-C", GWS_AGN_REPO, "rev-parse", "HEAD"]),
            "gws_agn_dirty": bool(_run(["git", "-C", GWS_AGN_REPO, "status",
                                        "--porcelain"])),
            "darksirens_repo": DARKSIRENS_A8,
            "darksirens_sha": _run(["git", "-C", DARKSIRENS_A8, "rev-parse", "HEAD"]),
            "darksirens_dirty": bool(_run(["git", "-C", DARKSIRENS_A8, "status",
                                           "--porcelain"])),
            "python": sys.executable,
            "pythonpath": os.environ.get("PYTHONPATH", ""),
            "host": socket.gethostname(),
            "written_at_utc": _now(),
        },
        "constants": {
            "mu_G_fiducial": MU_G_FID, "dmu_G_plant": DMU_G_PLANT,
            "dmu_G_axis_bounds": [-10.0, 10.0],
            "mu_chi_gal": MU_CHI_GAL, "sigma_chi": SIGMA_CHI,
            "dmu_chi_plant": DMU_CHI_PLANT,
            "H0": gd.H0_FID, "Om0": gd.OM0_FID, "gamma": gd.GAMMA,
        },
    }

    # ---- the coordinate is the generator's peak_mu: assert the alignment ---
    labels, plain, fid = a10.population_labels_and_fiducial()
    pc = gmd.PopulationConfig()
    align = {}
    for gen_name, ds_name in (("peak_mu", "G.mu"), ("peak_sigma", "G.sigma"),
                              ("chi_mu", "mu_chi"), ("chi_sigma", "sigma_chi"),
                              ("alpha", "PL.alpha"), ("mmin", "m_1,low"),
                              ("mmax", "m_1,high")):
        idx = [i for i, p in enumerate(plain) if p == ds_name]
        if not idx:
            continue
        align[ds_name] = {"generator_field": gen_name,
                          "generator_value": float(getattr(pc, gen_name)),
                          "darksirens_slot": int(idx[0]),
                          "darksirens_fiducial": float(fid[idx[0]]),
                          "agree": bool(abs(float(getattr(pc, gen_name))
                                            - float(fid[idx[0]])) < 1e-12)}
    out["population_slot_alignment"] = {
        "darksirens_plain_names": list(plain),
        "darksirens_latex_labels": list(labels),
        "darksirens_fiducial_vector": [float(x) for x in fid],
        "checked": align,
        "all_agree": bool(all(v["agree"] for v in align.values())),
    }
    if not align.get("G.mu", {}).get("agree"):
        raise SystemExit("[fatal] generator peak_mu != darksirens G.mu fiducial; "
                         "dmu_G would not mean the same thing in the two codes.")

    # ---------------------------------------------------------------- load --
    with h5py.File(INJ_TARGETED, "r") as f:
        A = {k: (v.item() if isinstance(v, np.generic) else v)
             for k, v in f.attrs.items() if k not in ("metadata_json",
                                                      "targeted_branch_json")}
        D = {k: f[k][:] for k in ("chieff", "branch", "pdraw", "pdraw_population",
                                  "m1src", "m2src", "m1det", "z")}
    n = int(D["chieff"].size)
    q = D["m2src"] / D["m1src"]
    Ndraw = float(A["Ndraw"])
    br = D["branch"].astype(int)
    BR = {0: "population", 1: "uniform", 2: "targeted_agn"}
    out["injection_file"] = {
        "path": str(INJ_TARGETED), "realpath": str(INJ_TARGETED.resolve()),
        "n_detected": n, "Ndraw": Ndraw, "detected_fraction": float(n / Ndraw),
        "proposal": str(A["selection_proposal"]),
        "proposal_mix": {"population": float(A["proposal_mix_population"]),
                         "uniform": float(A["proposal_mix_uniform"]),
                         "targeted_agn": float(A["proposal_mix_targeted_agn"])},
        "n_detected_by_branch": {BR[b]: int((br == b).sum()) for b in (0, 1, 2)},
        "zmax_proposal": float(A["zmax_proposal"]),
        "m1det_range_uniform": [float(x) for x in np.atleast_1d(A["m1det_range_uniform"])],
        "m1src_range": [float(D["m1src"].min()), float(D["m1src"].max())],
        "m1det_range": [float(D["m1det"].min()), float(D["m1det"].max())],
    }

    cosmo = gmd._build_cosmology(gd.H0_FID, gd.OM0_FID, gd.W0_FID, gd.WA_FID)
    grids = gmd._cosmology_grids(cosmo, float(A["zmax_proposal"]))

    def target_density(dmu_G, mu_chi, chunk=200_000):
        """p_pop(theta_j | Lambda) in the SAME coordinates as the stored pdraw."""
        pop = gmd.PopulationConfig(gamma=gd.GAMMA, chi_mu=float(mu_chi),
                                   peak_mu=MU_G_FID + float(dmu_G))
        o = np.empty(n)
        for i in range(0, n, chunk):
            s = slice(i, min(i + chunk, n))
            o[s] = gmd._selection_pdraw("population", D["m1src"][s], q[s],
                                        D["chieff"][s], D["z"][s], grids, pop)
        return o

    # The convention check: at the fiducial the recomputation must reproduce the
    # STORED population-branch density.  That certifies the coordinates, not a
    # comment.  (Analysis 8's Gate B ran the same check for the spin mark.)
    t0 = time.time()
    p_fid = target_density(0.0, MU_CHI_GAL)
    rel = np.abs(p_fid - D["pdraw_population"]) / np.abs(D["pdraw_population"])
    out["convention_check"] = {
        "what": ("recomputed p_pop at (dmu_G, dmu_chi) = (0, 0) vs the file's "
                 "stored pdraw_population"),
        "max_rel_diff": float(rel.max()), "median_rel_diff": float(np.median(rel)),
        "n_rows": n, "seconds": time.time() - t0,
        "passes": bool(rel.max() < 1e-10),
    }
    if rel.max() >= 1e-10:
        raise SystemExit(f"[fatal] convention check failed: max rel diff "
                         f"{rel.max():.3e}; the recomputed density is not in the "
                         f"same coordinates as the stored pdraw.")

    # ------------------------------------------------------- weight tails ---
    def tails(dmu_G, mu_chi, tag):
        p = target_density(dmu_G, mu_chi)
        w = p / D["pdraw"]
        s1 = float(w.sum())
        s2 = float(np.square(w).sum())
        ne = s1 * s1 / s2
        order = np.argsort(w)[::-1]
        share = {f"top{k}": float(w[order[:k]].sum() / s1)
                 for k in (1, 10, 100, 1000)}
        # which proposal branch supplies the heaviest weights
        heavy = br[order[:1000]]
        return {
            "tag": tag, "dmu_G": float(dmu_G), "mu_chi": float(mu_chi),
            "mu_G": MU_G_FID + float(dmu_G),
            "sum_w": s1, "Pdet": float(s1 / Ndraw),
            "Neff": float(ne), "Neff_over_5Nobs": float(ne / 5000.0),
            "rel_mc_error": float(ne ** -0.5),
            "max_normalised_weight": float(w.max() / s1),
            "share_of_sum_w": share,
            "top1000_by_proposal_branch": {BR[b]: int((heavy == b).sum())
                                           for b in (0, 1, 2)},
            "n_rows_with_zero_weight": int((w <= 0).sum()),
        }

    t0 = time.time()
    rows = [tails(d, MU_CHI_GAL + DMU_CHI_PLANT, f"AGN_dmuG_{d:+.0f}")
            for d in DMU_G_COARSE]
    rows.append(tails(0.0, MU_CHI_GAL, "GAL_fiducial"))
    out["weight_tails"] = {
        "definition": ("w_j = p_pop(theta_j | Lambda_branch) / pdraw_j over the "
                       "2,205,380 DETECTED injections; N_eff = (sum w)^2 / sum w^2; "
                       "shares are of sum(w) after sorting descending.  AGN rows "
                       "carry dmu_chi = +0.10, the registered spin mark."),
        "guard_reference_5Nobs": 5000.0,
        "rows": rows,
        "seconds": time.time() - t0,
    }

    # ----------------------------------------------------- mass coverage ----
    edges = np.arange(COVER_LO, COVER_HI + 0.5 * COVER_DM, COVER_DM)
    cnt, _ = np.histogram(D["m1src"], bins=edges)
    per_msun = cnt / COVER_DM

    # Source-frame primary-mass marginal of the branch population.  The pairing
    # density is normalised in q per m1 (generate_mock_data._pair_pdf) and the
    # spin factor is independent, so the m1 marginal is the mixture of the two
    # primary-mass components.
    mg = np.linspace(1.0, 200.0, 40_000)

    def p_m1(dmu_G):
        pop = gmd.PopulationConfig(gamma=gd.GAMMA,
                                   peak_mu=MU_G_FID + float(dmu_G))
        p_pl = gmd._powerlaw_pdf(mg, pop.alpha, pop.mmin, pop.mmax,
                                 pop.dm_min, pop.dm_max)
        p_pk = gmd._peak_pdf(mg, pop.peak_mu, pop.peak_sigma)
        p = (1.0 - pop.peak_fraction) * p_pl + pop.peak_fraction * p_pk
        return p / np.trapz(p, mg)

    def mass_in(p, lo, hi):
        m = (mg >= lo) & (mg <= hi)
        if m.sum() < 2:
            return 0.0
        return float(np.trapz(p[m], mg[m]))

    sparse = per_msun < COVER_MIN_PER_MSUN
    cover_rows = []
    pm = {d: p_m1(d) for d in (-10.0, -5.0, 0.0, 5.0, 10.0)}
    for i in range(len(cnt)):
        row = {"m1src_lo": float(edges[i]), "m1src_hi": float(edges[i + 1]),
               "n_injections": int(cnt[i]),
               "injections_per_Msun": float(per_msun[i]),
               "sparse_below_100_per_Msun": bool(sparse[i])}
        for d, p in pm.items():
            row[f"pop_mass_dmuG_{d:+.0f}"] = mass_in(p, edges[i], edges[i + 1])
        cover_rows.append(row)

    def uncovered(dmu_G):
        p = pm[dmu_G]
        m_sparse = sum(mass_in(p, edges[i], edges[i + 1])
                       for i in range(len(cnt)) if sparse[i])
        m_in_table = mass_in(p, COVER_LO, COVER_HI)
        return {
            "dmu_G": float(dmu_G), "mu_G": MU_G_FID + float(dmu_G),
            "pop_mass_in_table_range_20_60": m_in_table,
            "pop_mass_below_20": mass_in(p, mg.min(), COVER_LO),
            "pop_mass_above_60": mass_in(p, COVER_HI, mg.max()),
            "pop_mass_in_sparse_bins": m_sparse,
            "fraction_of_total_mass_in_sparse_bins": m_sparse,
        }

    # coverage of the whole m1src range, so the 20-60 table is not read as if
    # nothing lives outside it
    full_edges = np.concatenate(([D["m1src"].min()], edges, [D["m1src"].max()]))
    full_cnt, _ = np.histogram(D["m1src"], bins=np.unique(full_edges))

    out["mass_coverage"] = {
        "definition": (f"detected injections per unit SOURCE-frame primary mass in "
                       f"{COVER_DM} Msun bins from {COVER_LO} to {COVER_HI}; a bin "
                       f"is 'sparse' below {COVER_MIN_PER_MSUN:.0f} injections per "
                       f"Msun.  'pop_mass' is the branch population's m1src "
                       f"marginal integrated over the bin (mixture of the tapered "
                       f"power law and the Gaussian peak; the pairing density is "
                       f"normalised in q, so the marginal is exact)."),
        "n_injections_total": n,
        "bins": cover_rows,
        "n_sparse_bins": int(sparse.sum()),
        "sparse_bins": [[float(edges[i]), float(edges[i + 1])]
                        for i in range(len(cnt)) if sparse[i]],
        "min_injections_per_Msun_in_table": float(per_msun.min()),
        "per_dmu_G": [uncovered(d) for d in (-10.0, -5.0, 0.0, 5.0, 10.0)],
        "note_outside_table": {
            "n_injections_below_20": int((D["m1src"] < COVER_LO).sum()),
            "n_injections_above_60": int((D["m1src"] > COVER_HI).sum()),
            "m1src_min": float(D["m1src"].min()),
            "m1src_max": float(D["m1src"].max()),
        },
    }

    out["seconds_total"] = time.time() - t_start
    _write(STAGE_FILE_JSON, out)

    print("\n== weight tails (population-only proxy) ==")
    print(f"{'tag':>18} {'mu_G':>6} {'Neff':>12} {'Neff/5000':>10} "
          f"{'w_max/sum':>11} {'top10':>8} {'top100':>8} {'top1000':>8}")
    for r in rows:
        print(f"{r['tag']:>18} {r['mu_G']:6.1f} {r['Neff']:12.1f} "
              f"{r['Neff_over_5Nobs']:10.2f} {r['max_normalised_weight']:11.3e} "
              f"{r['share_of_sum_w']['top10']:8.4f} "
              f"{r['share_of_sum_w']['top100']:8.4f} "
              f"{r['share_of_sum_w']['top1000']:8.4f}")
    print("\n== sparse mass bins ==")
    print(f"  {int(sparse.sum())} of {len(cnt)} bins below "
          f"{COVER_MIN_PER_MSUN:.0f}/Msun; min = {per_msun.min():.1f}/Msun")
    for u in out["mass_coverage"]["per_dmu_G"]:
        print(f"  dmu_G={u['dmu_G']:+5.1f}  mass in sparse bins = "
              f"{u['pop_mass_in_sparse_bins']:.3e}  "
              f"mass above 60 = {u['pop_mass_above_60']:.3e}")


# =========================================================================== #
# Stage "likelihood" -- the real dark-siren selection term, on rita's GPU
# =========================================================================== #
def stage_likelihood(args):
    import a9_scan                                  # the A9 rita harness
    import a10_likelihood as a10

    t_start = time.time()
    env = a9_scan._gpu_setup("a10_gate_b_likelihood")

    surveys = [a10.SURVEY_GAL, a10.SURVEY_AGN]
    t0 = time.time()
    cell = a10.build_a10("A10_GATEB", surveys, verbose=True,
                         gw_path=a10.GW_PATH_MARKED)
    t_build = time.time() - t0
    labels = a10.assert_a10_labels(cell)
    print(f"[a10] build: {t_build:.1f}s  labels={labels}")

    out = {
        "what": ("live evaluations of the real dark-siren selection term through "
                 "the a8_likelihood guard spy, on the Analysis-10 two-mark "
                 "parameter space"),
        "provenance": a10.provenance(gw_path=a10.GW_PATH_MARKED,
                                     survey_paths=surveys),
        "environment": env,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "host": socket.gethostname(),
        "labels": labels,
        "events_file_caveat": (
            "The Analysis-10 mock does not exist yet; the build used the "
            "Analysis-8 MARKED events file events_marked_dmu0p10.h5.  The "
            "selection term never reads events: the events file enters only "
            "through nEvents and pe_variance_sum, both of which set the guard "
            "THRESHOLD.  pe_variance_sum is reported per cell."),
        "grids": {"H0": H0_ANCHOR, "dmu_G_coarse": DMU_G_COARSE,
                  "f_grid": F_GRID, "dmu_chi_grid": DMU_CHI_GRID,
                  "dmu_G_axis": DMU_G_AXIS, "axis_dmu_chi": AXIS_DMU_CHI,
                  "inert_dmu_G": INERT_DMU_G, "inert_dmu_chi": INERT_DMU_CHI},
        "build_seconds": t_build,
        "cells": [],
    }

    STAGE_LIKE_JSONL.parent.mkdir(parents=True, exist_ok=True)
    fh = open(STAGE_LIKE_JSONL, "w")

    def ev(block, f, dmu_chi, dmu_G):
        r = cell.evaluate_at(H0=H0_ANCHOR, fcat_2=f, dmu_chi=dmu_chi, dmu_G=dmu_G)
        g = r.get("guard", {})
        rec = {
            "block": block, "H0": H0_ANCHOR, "f_agn": float(f),
            "dmu_chi": float(dmu_chi), "dmu_G": float(dmu_G),
            "mu_chi_c2": float(dmu_chi), "mu_G_c2": MU_G_FID + float(dmu_G),
            "logL": _fin(r["logL"]), "logL_hex": r["logL_hex"],
            "logL_finite": bool(r["finite"]),
            "logL_selection": _fin(r.get("logL_selection", float("nan"))),
            "logL_pe": _fin(r.get("logL_pe", float("nan"))),
            "log_mu": _fin(g.get("log_mu", float("nan"))),
            "Neff": _fin(g.get("Neff", float("nan"))),
            "threshold": _fin(g.get("threshold", float("nan"))),
            "Neff_over_threshold": None,
            "sigma2_total": _fin(g.get("sigma2_total", float("nan"))),
            "pe_variance_sum": _fin(g.get("pe_variance_sum", float("nan"))),
            "selection_variance_N2_over_Neff":
                _fin(g.get("selection_variance_N2_over_Neff", float("nan"))),
            "nEvents": g.get("nEvents"),
            "guard_passes": bool(g.get("passes", False)),
            "n_guard_calls": r["n_guard_calls"],
            "seconds": r["seconds"],
        }
        if rec["Neff"] is not None and rec["threshold"]:
            rec["Neff_over_threshold"] = rec["Neff"] / rec["threshold"]
        if rec["n_guard_calls"] != 1:
            rec["ANOMALY"] = (f"{rec['n_guard_calls']} guard calls in one "
                              f"evaluation; expected exactly 1")
            print(f"  [WARN] {rec['ANOMALY']}")
        out["cells"].append(rec)
        fh.write(json.dumps(rec, default=_json_default) + "\n")
        fh.flush()
        _f = lambda x, w, p: (f"{x:{w}.{p}f}" if isinstance(x, float)  # noqa: E731
                              else f"{str(x):>{w}}")
        print(f"  [{block:>7}] f={f:5.3f} dchi={dmu_chi:+.2f} dG={dmu_G:+6.2f}  "
              f"Neff={_f(rec['Neff'], 12, 1)}  "
              f"ratio={_f(rec['Neff_over_threshold'], 7, 2)}  "
              f"logmu={_f(rec['log_mu'], 9, 4)}  pass={rec['guard_passes']}  "
              f"{rec['seconds']:.2f}s")
        sys.stdout.flush()
        return rec

    # (a1) the registered 5 x 5 x 3 grid
    print("\n== block 'grid75': dmu_G x f x dmu_chi ==")
    for dG in DMU_G_COARSE:
        for f in F_GRID:
            for dc in DMU_CHI_GRID:
                ev("grid75", f, dc, dG)

    # (a2) the full dmu_G axis on the PURE AGN branch
    print("\n== block 'axis_f1': f = 1, 21-node dmu_G axis ==")
    for dc in AXIS_DMU_CHI:
        for dG in DMU_G_AXIS:
            ev("axis_f1", 1.0, dc, dG)

    # (a3) the f = 0 inertness probe
    print("\n== block 'inert_f0': f = 0, dmu_G must not move anything ==")
    for dG in INERT_DMU_G:
        ev("inert_f0", 0.0, INERT_DMU_CHI, dG)

    fh.close()
    out["n_cells"] = len(out["cells"])
    out["seconds_total"] = time.time() - t_start
    out["seconds_first_eval"] = (out["cells"][0]["seconds"] if out["cells"] else None)
    out["seconds_per_eval_mean"] = (
        float(np.mean([c["seconds"] for c in out["cells"][1:]]))
        if len(out["cells"]) > 1 else None)
    out["n_anomalies"] = sum(1 for c in out["cells"] if "ANOMALY" in c)
    _write(STAGE_LIKE_JSON, out)
    print(f"\n[a10] {out['n_cells']} cells in {out['seconds_total']:.1f}s "
          f"(first eval {out['seconds_first_eval']:.1f}s incl. compile; "
          f"{out['seconds_per_eval_mean']:.3f} s/eval after); "
          f"anomalies: {out['n_anomalies']}")


# =========================================================================== #
# Stage "assemble"
# =========================================================================== #
def _summarise(cells, key):
    """min / median N_eff/threshold grouped by ``key``."""
    groups = {}
    for c in cells:
        if c["Neff_over_threshold"] is None:
            continue
        groups.setdefault(round(float(c[key]), 6), []).append(c["Neff_over_threshold"])
    return {str(k): {"n": len(v), "min": float(np.min(v)),
                     "median": float(np.median(v)), "max": float(np.max(v))}
            for k, v in sorted(groups.items())}


def stage_assemble(args):
    fs = json.loads(STAGE_FILE_JSON.read_text())
    lk = json.loads(STAGE_LIKE_JSON.read_text())
    cells = lk["cells"]
    grid75 = [c for c in cells if c["block"] == "grid75"]
    axis = [c for c in cells if c["block"] == "axis_f1"]
    inert = [c for c in cells if c["block"] == "inert_f0"]

    # -------------------------------------------------- the hard guard ------
    unmeasured = [c for c in cells if c["Neff_over_threshold"] is None]
    if unmeasured:
        raise SystemExit(
            f"[fatal] {len(unmeasured)} cells carry no guard record; the spy did "
            f"not fire and the gate is undecidable.  First: {unmeasured[0]}")
    rejected = [c for c in cells if not c["guard_passes"]]
    ratios = [c["Neff_over_threshold"] for c in cells]

    # --------------------------------------- posterior-relevant region ------
    post = [c for c in grid75
            if c["f_agn"] <= POST_F_MAX + 1e-9
            and c["dmu_chi"] <= POST_DMU_CHI_MAX + 1e-9]
    post_min = min(post, key=lambda c: c["Neff_over_threshold"])
    post_rejected = [c for c in post if not c["guard_passes"]]

    # The gate's two conditions.  A cell rejected OUTSIDE the posterior-relevant
    # region is not a failure -- Analysis 9 carried one too -- but it is not a
    # clean pass either, so it is reported as its own verdict with the numbers
    # that decide whether it matters.  The call is the owner's.
    post_ok = (not post_rejected) and (
        post_min["Neff_over_threshold"] >= MARGIN_REQUIRED)
    if not post_ok:
        verdict = "FAIL"
    elif rejected:
        verdict = "PASS_WITH_REJECTED_CORNER"
    else:
        verdict = "PASS"

    # Which coordinate drives the rejections?  If every rejected cell carries the
    # LARGEST spin mark and none carries a large |dmu_G| alone, the wall belongs
    # to Analysis 8's spin axis, not to the new mass coordinate.
    rej_by_chi = {}
    rej_by_dG = {}
    for c in rejected:
        rej_by_chi[f"{c['dmu_chi']:+.2f}"] = rej_by_chi.get(
            f"{c['dmu_chi']:+.2f}", 0) + 1
        rej_by_dG[f"{c['dmu_G']:+.1f}"] = rej_by_dG.get(f"{c['dmu_G']:+.1f}", 0) + 1
    rej_chi_values = sorted({c["dmu_chi"] for c in rejected})
    rej_f_values = sorted({c["f_agn"] for c in rejected})

    # --------------------------------------------- the f = 0 inertness ------
    f0_all = [c for c in cells if c["f_agn"] == 0.0]
    hexes = sorted({c["logL_hex"] for c in f0_all})
    neffs = sorted({c["Neff"] for c in f0_all})
    inert_hexes = sorted({c["logL_hex"] for c in inert})
    frozen = (len(hexes) == 1) and (len(neffs) == 1)

    # ------------------------------------------------ the f = 1 liveness ----
    axis_by_chi = {}
    for c in axis:
        axis_by_chi.setdefault(f"{c['dmu_chi']:+.2f}", []).append(c)
    for k in axis_by_chi:
        axis_by_chi[k] = sorted(axis_by_chi[k], key=lambda c: c["dmu_G"])
    live_span = {}
    for k, rows in axis_by_chi.items():
        ne = [r["Neff"] for r in rows]
        lm = [r["log_mu"] for r in rows]
        live_span[k] = {
            "Neff_min": float(np.min(ne)), "Neff_max": float(np.max(ne)),
            "Neff_min_at_dmu_G": float(rows[int(np.argmin(ne))]["dmu_G"]),
            "Neff_max_over_min": float(np.max(ne) / np.min(ne)),
            "log_mu_min": float(np.min(lm)), "log_mu_max": float(np.max(lm)),
            "log_mu_span": float(np.max(lm) - np.min(lm)),
            "n_distinct_logL": len({r["logL_hex"] for r in rows}),
            "min_Neff_over_threshold": float(
                min(r["Neff_over_threshold"] for r in rows)),
        }

    # ------------------------------------------------------- thresholds -----
    thr = sorted({c["threshold"] for c in cells})
    pev = [c["pe_variance_sum"] for c in cells if c["pe_variance_sum"] is not None]
    nev = sorted({c["nEvents"] for c in cells})
    mlv = 1e6           # SETTINGS['max_likelihood_variance'] (analysis-2 pin)
    n_obs = float(nev[0]) if len(nev) == 1 else None
    pe_breakeven = (mlv - (n_obs * n_obs) / (5.0 * n_obs)) if n_obs else None

    out = {
        "question": ("Analysis 10 Gate B -- does the EXISTING seed-100 injection "
                     "set support the AGN branch's Gaussian-peak location "
                     "G.mu_c2 over dmu_G in [-10, +10], alongside the spin mark?"),
        "verdict": verdict,
        "written_at_utc": _now(),
        "decision_rule": {
            "hard_guard": "every evaluated cell must satisfy Neff > threshold",
            "margin": (f"on the posterior-relevant region (f <= {POST_F_MAX}, "
                       f"dmu_chi <= +{POST_DMU_CHI_MAX}, all dmu_G in [-10, +10]) "
                       f"Neff/threshold >= {MARGIN_REQUIRED}"),
            "n_cells_evaluated": len(cells),
            "n_cells_posterior_relevant": len(post),
        },
        "provenance": {
            "file_stage": fs["provenance"],
            "likelihood_stage": lk["provenance"],
            "likelihood_environment": lk.get("environment"),
            "slurm_job_id": lk.get("slurm_job_id"),
            "host_likelihood": lk.get("host"),
            "build_seconds": lk.get("build_seconds"),
            "likelihood_seconds_total": lk.get("seconds_total"),
            "seconds_per_eval_mean": lk.get("seconds_per_eval_mean"),
            "file_seconds_total": fs.get("seconds_total"),
            "labels": lk["labels"],
        },
        "events_file": {
            "used_for_build": "working/data/seed100/events/events_marked_dmu0p10.h5",
            "why": lk["events_file_caveat"],
            "nEvents_seen_by_guard": nev,
            "pe_variance_sum_min": float(np.min(pev)) if pev else None,
            "pe_variance_sum_max": float(np.max(pev)) if pev else None,
            "pe_variance_sum_median": float(np.median(pev)) if pev else None,
            "threshold_values_seen": thr,
            "max_likelihood_variance": mlv,
            "threshold_is_the_5Nobs_floor": bool(
                len(thr) == 1 and n_obs is not None
                and abs(thr[0] - 5.0 * n_obs) < 1e-6),
            "pe_variance_sum_breakeven": pe_breakeven,
            "statement": (
                f"The guard threshold sat at {thr} for every cell, i.e. the "
                f"5 N_obs floor with N_obs = {nev}.  pe_variance_sum would have "
                f"to exceed {pe_breakeven:.3e} before the variance term could "
                f"raise it, and the measured values span "
                f"[{np.min(pev):.4g}, {np.max(pev):.4g}].  The gate is therefore "
                f"insensitive to which 1000-event realisation is used; only "
                f"N_obs matters."
                if pev and n_obs else "insufficient guard records"),
        },
        "a_live_grid": {
            "n_cells": len(cells),
            "n_guard_rejected": len(rejected),
            "rejected_cells": [
                {k: c[k] for k in ("block", "f_agn", "dmu_chi", "dmu_G", "Neff",
                                   "threshold", "Neff_over_threshold", "log_mu")}
                for c in rejected],
            "rejections_by_dmu_chi": rej_by_chi,
            "rejections_by_dmu_G": rej_by_dG,
            "rejected_dmu_chi_values": rej_chi_values,
            "rejected_f_values": rej_f_values,
            "rejections_confined_to_largest_spin_mark": bool(
                rejected and all(c["dmu_chi"] > POST_DMU_CHI_MAX for c in rejected)),
            "rejections_confined_to_f_above_posterior_region": bool(
                rejected and all(c["f_agn"] > POST_F_MAX for c in rejected)),
            "Neff_over_threshold": {
                "min": float(np.min(ratios)), "median": float(np.median(ratios)),
                "max": float(np.max(ratios))},
            "grid75_by_dmu_G": _summarise(grid75, "dmu_G"),
            "grid75_by_f": _summarise(grid75, "f_agn"),
            "grid75_by_dmu_chi": _summarise(grid75, "dmu_chi"),
            "grid75_cells": grid75,
        },
        "a_f1_axis": {
            "what": ("the pure AGN branch: f = 1 removes the GAL dilution, so the "
                     "branch-specific selection factor is seen undiluted.  This is "
                     "also the LIVENESS probe for the new coordinate."),
            "by_dmu_chi": live_span,
            "rows": {k: [{"dmu_G": r["dmu_G"], "mu_G_c2": r["mu_G_c2"],
                          "Neff": r["Neff"], "threshold": r["threshold"],
                          "Neff_over_threshold": r["Neff_over_threshold"],
                          "log_mu": r["log_mu"],
                          "logL_selection": r["logL_selection"],
                          "guard_passes": r["guard_passes"]}
                         for r in rows]
                     for k, rows in axis_by_chi.items()},
        },
        "a_f0_inertness": {
            "what": ("at f = 0 the mixture puts zero weight on catalog 2, so "
                     "NEITHER per-catalog coordinate may move anything.  Bitwise "
                     "identity is the test; it is also the control that makes the "
                     "f = 1 movement meaningful."),
            "n_cells_at_f0": len(f0_all),
            "distinct_logL_hex": hexes,
            "distinct_Neff": neffs,
            "bitwise_frozen": bool(frozen),
            "dedicated_probe_dmu_G": INERT_DMU_G,
            "dedicated_probe_distinct_logL_hex": inert_hexes,
            "dedicated_probe_frozen": bool(len(inert_hexes) == 1),
        },
        "b_weight_tails": fs["weight_tails"],
        "b_mass_coverage": fs["mass_coverage"],
        "b_convention_check": fs["convention_check"],
        "b_population_slot_alignment": fs["population_slot_alignment"],
        "b_injection_file": fs["injection_file"],
        "b_proxy_caveat": fs["what"],
        "c_decision": {
            "verdict": verdict,
            "verdict_meaning": {
                "PASS": "no cell rejected anywhere and the posterior margin holds",
                "PASS_WITH_REJECTED_CORNER": (
                    "no posterior-relevant cell rejected and the posterior margin "
                    "holds, but cells OUTSIDE the posterior-relevant region are "
                    "rejected.  Analysis 9 carried such a corner.  The numbers "
                    "that decide whether it matters are in "
                    "a_live_grid.rejected_cells and the two "
                    "'rejections_confined_to_*' flags; the call is the owner's."),
                "FAIL": ("a posterior-relevant cell is rejected, or the posterior "
                         "margin is below the requirement"),
            }[verdict],
            "all_cells_pass_hard_guard": not rejected,
            "at_planted_truth": [
                {k: c[k] for k in ("f_agn", "dmu_chi", "dmu_G", "Neff",
                                   "threshold", "Neff_over_threshold", "log_mu",
                                   "guard_passes")}
                for c in grid75
                if abs(c["dmu_G"] - DMU_G_PLANT) < 1e-9
                and abs(c["dmu_chi"] - DMU_CHI_PLANT) < 1e-9],
            "minimum_margin_posterior_relevant": post_min["Neff_over_threshold"],
            "minimum_margin_at": {k: post_min[k] for k in
                                  ("block", "f_agn", "dmu_chi", "dmu_G", "Neff",
                                   "threshold")},
            "margin_required": MARGIN_REQUIRED,
            "n_posterior_relevant_rejected": len(post_rejected),
            "posterior_relevant_rejected": post_rejected,
            "minimum_margin_all_cells": float(np.min(ratios)),
            "minimum_margin_all_cells_at": min(
                (c for c in cells if c["Neff_over_threshold"] is not None),
                key=lambda c: c["Neff_over_threshold"]),
        },
    }

    if verdict == "FAIL":
        # No injections are generated here.  Write the diagnosis and the minimal
        # design, and stop: the decision to generate belongs to the owner.
        worst = sorted((c for c in cells if not c["guard_passes"]),
                       key=lambda c: (c["dmu_G"], c["f_agn"]))
        dgs = sorted({c["dmu_G"] for c in worst})
        cov = fs["mass_coverage"]
        out["c_diagnosis_if_fail"] = {
            "branch": ("catalog 2 (AGN): the rejected cells all carry f_AGN > 0, "
                       "and the f = 0 cells are frozen, so the deficit is in the "
                       "AGN branch's selection factor."
                       if all(c["f_agn"] > 0 for c in worst)
                       else "rejected cells include f = 0: investigate, this "
                            "should be impossible"),
            "dmu_G_values_rejected": dgs,
            "mass_range_lacking_support": {
                "sparse_bins_below_100_per_Msun": cov["sparse_bins"],
                "min_injections_per_Msun": cov["min_injections_per_Msun_in_table"],
                "population_mass_in_sparse_bins_by_dmu_G": [
                    {"dmu_G": u["dmu_G"],
                     "mass_in_sparse_bins": u["pop_mass_in_sparse_bins"],
                     "mass_above_60": u["pop_mass_above_60"]}
                    for u in cov["per_dmu_G"]],
            },
            "minimal_proposal_design": {
                "one_targeted_set_covering_BOTH_mass_populations": (
                    "Add ONE mixture branch to the existing proposal whose "
                    "primary-mass density is broad enough to cover the Gaussian "
                    "peak at every dmu_G in [-10, +10] simultaneously: draw "
                    "m1src from a mixture of the fiducial power law and a WIDE "
                    "Gaussian centred at 35 Msun with sigma = sqrt(5^2 + 10^2) "
                    "= 11.2 Msun, so a peak anywhere in [25, 45] sits inside one "
                    "sigma of the proposal.  Keep the existing spin and redshift "
                    "branches unchanged -- the spin mark is already covered by "
                    "the 10 per cent uniform-in-chi branch, and the redshift "
                    "proposal is unchanged by dmu_G."),
                "weights": ("keep population 0.65 -> 0.50, uniform 0.10, "
                            "targeted-AGN 0.25, new broad-mass branch 0.15; the "
                            "broad branch is the one that bounds the weight "
                            "ratio at the axis endpoints"),
                "sizing": ("the cheapest sufficient Ndraw is set by the worst "
                           "measured N_eff: scale the existing 1.5e8 draws by "
                           "(target margin) / (measured margin) at the worst "
                           "posterior-relevant cell"),
                "note": "NOT generated here; the decision to generate is the owner's",
            },
        }

    _write(OUT_JSON, out)
    _write_md(out)
    print(f"\n==== Gate B: {verdict} ====")
    print(f"  cells evaluated        : {len(cells)}")
    print(f"  guard-rejected         : {len(rejected)}  "
          f"(by dmu_chi: {rej_by_chi}; by dmu_G: {rej_by_dG})")
    print(f"  posterior-relevant rej.: {len(post_rejected)}")
    print(f"  min margin (all cells) : {np.min(ratios):.2f}")
    print(f"  min margin (posterior) : {post_min['Neff_over_threshold']:.2f} "
          f"at f={post_min['f_agn']}, dmu_chi={post_min['dmu_chi']:+.2f}, "
          f"dmu_G={post_min['dmu_G']:+.1f}")
    print(f"  f = 0 bitwise frozen   : {frozen}")
    for k, v in live_span.items():
        print(f"  f = 1 axis dmu_chi={k}: Neff {v['Neff_min']:.4g} .. "
              f"{v['Neff_max']:.4g} (x{v['Neff_max_over_min']:.1f}), "
              f"log_mu span {v['log_mu_span']:.4f}, "
              f"{v['n_distinct_logL']}/21 distinct logL")


def _write_md(out):
    L = []
    A = L.append
    A(f"# Analysis 10 -- Gate B: selection support for `G.mu_c2`\n")
    A(f"**Verdict: {out['verdict']}**  ({out['written_at_utc']})\n")
    d = out["c_decision"]
    A(f"- cells evaluated: {out['a_live_grid']['n_cells']}; "
      f"guard-rejected: {out['a_live_grid']['n_guard_rejected']}")
    A(f"- minimum N_eff/threshold over ALL cells: "
      f"{d['minimum_margin_all_cells']:.2f}")
    mm = d["minimum_margin_at"]
    A(f"- minimum over the posterior-relevant region "
      f"(f <= {POST_F_MAX}, dmu_chi <= +{POST_DMU_CHI_MAX}): "
      f"{d['minimum_margin_posterior_relevant']:.2f} "
      f"at f = {mm['f_agn']}, dmu_chi = {mm['dmu_chi']:+.2f}, "
      f"dmu_G = {mm['dmu_G']:+.1f} (required >= {MARGIN_REQUIRED})")
    A(f"- guard threshold: {out['events_file']['threshold_values_seen']} "
      f"(5 N_obs floor: {out['events_file']['threshold_is_the_5Nobs_floor']})")
    A(f"- f = 0 bitwise frozen: {out['a_f0_inertness']['bitwise_frozen']}")
    g = out["a_live_grid"]
    if g["n_guard_rejected"]:
        A(f"- guard-rejected cells: {g['n_guard_rejected']} "
          f"(by dmu_chi {g['rejections_by_dmu_chi']}; "
          f"by dmu_G {g['rejections_by_dmu_G']}); "
          f"all at dmu_chi > +{POST_DMU_CHI_MAX}: "
          f"{g['rejections_confined_to_largest_spin_mark']}; "
          f"none is posterior-relevant: "
          f"{d['n_posterior_relevant_rejected'] == 0}")
    A("")
    A("## At the planted truth (dmu_G = +5, dmu_chi = +0.10)\n")
    A("| f_AGN | N_eff | N_eff/thr | log_mu | pass |")
    A("|---|---|---|---|---|")
    for c in d["at_planted_truth"]:
        A(f"| {c['f_agn']} | {c['Neff']:.4g} | {c['Neff_over_threshold']:.2f} | "
          f"{c['log_mu']:.4f} | {c['guard_passes']} |")
    A("")

    A("## 75-cell grid: N_eff / threshold\n")
    for tag, key in (("by dmu_G", "grid75_by_dmu_G"), ("by f_AGN", "grid75_by_f"),
                     ("by dmu_chi", "grid75_by_dmu_chi")):
        A(f"### {tag}\n")
        A("| value | n | min | median | max |")
        A("|---|---|---|---|---|")
        for k, v in out["a_live_grid"][key].items():
            A(f"| {k} | {v['n']} | {v['min']:.2f} | {v['median']:.2f} | "
              f"{v['max']:.2f} |")
        A("")

    A("## f = 1 axis (pure AGN branch)\n")
    A("| dmu_G | mu_G,c2 | N_eff | N_eff/thr | log_mu | pass |")
    A("|---|---|---|---|---|---|")
    for k, rows in out["a_f1_axis"]["rows"].items():
        A(f"| **dmu_chi = {k}** | | | | | |")
        for r in rows:
            A(f"| {r['dmu_G']:+.1f} | {r['mu_G_c2']:.1f} | {r['Neff']:.4g} | "
              f"{r['Neff_over_threshold']:.2f} | {r['log_mu']:.4f} | "
              f"{r['guard_passes']} |")
    A("")

    A("## Weight tails (population-only proxy, from the injection file)\n")
    A("| branch | mu_G | N_eff | N_eff/5000 | max w/sum w | top10 | top100 | top1000 |")
    A("|---|---|---|---|---|---|---|---|")
    for r in out["b_weight_tails"]["rows"]:
        s = r["share_of_sum_w"]
        A(f"| {r['tag']} | {r['mu_G']:.0f} | {r['Neff']:.1f} | "
          f"{r['Neff_over_5Nobs']:.2f} | {r['max_normalised_weight']:.3e} | "
          f"{s['top10']:.4f} | {s['top100']:.4f} | {s['top1000']:.4f} |")
    A("")

    A("## Source-mass coverage (2.5 Msun bins)\n")
    A("| m1src | injections | per Msun | sparse | mass(-10) | mass(0) | mass(+10) |")
    A("|---|---|---|---|---|---|---|")
    for b in out["b_mass_coverage"]["bins"]:
        A(f"| {b['m1src_lo']:.1f}-{b['m1src_hi']:.1f} | {b['n_injections']} | "
          f"{b['injections_per_Msun']:.0f} | {b['sparse_below_100_per_Msun']} | "
          f"{b['pop_mass_dmuG_-10']:.4f} | {b['pop_mass_dmuG_+0']:.4f} | "
          f"{b['pop_mass_dmuG_+10']:.4f} |")
    A("")
    for u in out["b_mass_coverage"]["per_dmu_G"]:
        A(f"- dmu_G = {u['dmu_G']:+.0f}: population mass in sparse bins = "
          f"{u['pop_mass_in_sparse_bins']:.3e}; above 60 Msun = "
          f"{u['pop_mass_above_60']:.3e}; below 20 Msun = "
          f"{u['pop_mass_below_20']:.3e}")
    A("")
    A(f"Proxy caveat: {out['b_proxy_caveat']}\n")
    OUT_MD.write_text("\n".join(L))
    print(f"wrote {OUT_MD}")


# =========================================================================== #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True,
                    choices=("file", "likelihood", "assemble"))
    args = ap.parse_args()
    DIAG.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    {"file": stage_file, "likelihood": stage_likelihood,
     "assemble": stage_assemble}[args.stage](args)


if __name__ == "__main__":
    main()
