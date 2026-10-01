#!/usr/bin/env python
"""Analysis 11 -- event-level AGN assignment under reference-population freedom (brief s.16).

P_i(AGN) = f E_i[AA] / ((1 - f) E_i[GG] + f E_i[AA]) is formed with Analysis 8's
``event_decomposition`` (``capture_operands`` + ``build_evaluator``, imported through
``a10_event_decomposition``) at two points, both at H0 = 67.74:

    A10   the Analysis-10 A10-J posterior median, on the Analysis-10 cell
          (reference pinned at mu_G = 35, mu_chi = 0);
    A11   the 11C posterior median (or --point), on the Analysis-11 cell (reference free).

Each pass is verified: sum_i logZ_prod equals the live production logL_pe at the same
point (1e-6 absolute is a FAIL).  Reported: RMS of dP, number with |dP| > 0.1, number
crossing 0.5, Spearman rank correlation, and (interpretation only, read last) the
agreement of each with the true host label.

    python scripts/a11_event_assign.py --a11_point f mu_G dmu_G mu_chi dmu_chi
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
A11 = HERE.parent
sys.path.insert(0, str(HERE))

import a11_likelihood as L                                  # noqa: E402

sys.path.insert(0, str(L.A10_SCRIPTS))
import a10_event_decomposition as ED                        # noqa: E402

A8 = ED.A8
OUT = A11 / "diagnostics" / "a11_event_assignment.json"


def capture(cell, coord, a11):
    if str(ED.A8_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(ED.A8_SCRIPTS))
    import event_decomposition as ed
    saved = A8.GW_PATH_MARKED
    A8.GW_PATH_MARKED = L.GW_PATH_A11
    try:
        if a11:
            with L.A10._steer(L.A10.PER_CATALOG_A10, L.A11LikelihoodCell), L._free_base():
                cap = ed.capture_operands(cell, coord)
        else:
            with L.A10._steer(L.A10.PER_CATALOG_A10, L.A10.A10LikelihoodCell):
                cap = ed.capture_operands(cell, coord)
    finally:
        A8.GW_PATH_MARKED = saved
    return ed, cap


def probs(cell, coord, live, a11, tag):
    ed, cap = capture(cell, coord, a11)
    block = int(A8.SETTINGS["pe_event_block"])
    cols, _nk, nEv, _ns, log_w, _tb, _tp = ED.per_event_pass(ed, cap, block, tag)
    lf0, lf1 = float(log_w[0]), float(log_w[1])
    P = 1.0 / (1.0 + np.exp((lf0 + cols["logE_GG"]) - (lf1 + cols["logE_AA"])))
    ver = {"sum_logZ_prod": float(np.sum(cols["logZ_prod"])), "live_logL_pe": live["logL_pe"],
           "abs_diff": abs(float(np.sum(cols["logZ_prod"])) - live["logL_pe"])}
    ver["pass"] = ver["abs_diff"] <= 1e-6
    print(f"[{tag}] verification {ver}", flush=True)
    return P, ver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a11_point", type=float, nargs=5, required=True,
                    metavar=("f", "mu_G", "dmu_G", "mu_chi", "dmu_chi"))
    ap.add_argument("--a10_point", type=float, nargs=3, default=None,
                    metavar=("f", "dmu_chi", "dmu_G"))
    args = ap.parse_args()
    sys.path.insert(0, str(A11.parent / "analysis_9_spin_marked_H0_fagn" / "scripts"))
    import a9_scan as A9
    env = A9._gpu_setup("a11_event_assign")
    if args.a10_point is None:
        j = json.loads((L.A10_SCRIPTS.parent / "results" / "a10_arm_J.json").read_text())
        args.a10_point = [j["f"]["median"], j["dmu_chi"]["median"], j["dmu_G"]["median"]]
    f10, c10, g10 = args.a10_point
    f11, mG, dG, mC, dC = args.a11_point

    cell11 = L.build_a11("A11ASSIGN")
    cell10 = L.build_a10_reference("A10ASSIGN", data=cell11.data)
    live10 = cell10.evaluate_at(H0=L.H0_FID, fcat_2=f10, dmu_chi=c10, dmu_G=g10)
    live11 = cell11.evaluate_at(H0=L.H0_FID, fcat_2=f11, mu_G=mG, dmu_G=dG, mu_chi=mC,
                                dmu_chi=dC)
    coord10 = cell10.coord(fcat_2=f10, **{L.MU_CHI_C2_LABEL: c10, L.MU_G_C2_LABEL: 35.0 + g10})
    coord11 = cell11.coord(fcat_2=f11, **{L.MU_G_LABEL: mG, L.MU_G_C2_LABEL: mG + dG,
                                          L.MU_CHI_LABEL: mC, L.MU_CHI_C2_LABEL: mC + dC})
    P10, v10 = probs(cell10, coord10, live10, False, "A10")
    P11, v11 = probs(cell11, coord11, live11, True, "A11")

    from scipy.stats import spearmanr
    import h5py
    dP = P11 - P10
    cross = (P10 > 0.5) != (P11 > 0.5)
    with h5py.File(L.GW_PATH_A11, "r") as h:                 # interpretation only
        host = h["host_type"][:].astype(int)
    out = {
        "points": {"A10_median": {"f": f10, "dmu_chi": c10, "dmu_G": g10},
                   "A11_11C": {"f": f11, "mu_G": mG, "dmu_G": dG, "mu_chi": mC, "dmu_chi": dC}},
        "verification": {"A10": v10, "A11": v11},
        "rms_dP": float(np.sqrt(np.mean(dP ** 2))),
        "n_abs_dP_gt_0p1": int((np.abs(dP) > 0.1).sum()),
        "n_abs_dP_gt_0p05": int((np.abs(dP) > 0.05).sum()),
        "n_crossing_0p5": int(cross.sum()),
        "spearman_rho": float(spearmanr(P10, P11)[0]),
        "sum_P": {"A10": float(P10.sum()), "A11": float(P11.sum())},
        "median_abs_dP": float(np.median(np.abs(dP))),
        "max_abs_dP": float(np.max(np.abs(dP))),
        "true_label_agreement_at_0p5 (interpretation only)": {
            "A10": float(np.mean((P10 > 0.5) == (host == 1))),
            "A11": float(np.mean((P11 > 0.5) == (host == 1)))},
        "environment": env, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    with h5py.File(A11 / "diagnostics" / "a11_event_assignment.h5", "w") as h:
        h.create_dataset("P_A10", data=P10)
        h.create_dataset("P_A11", data=P11)
        h.create_dataset("true_host_type", data=host)
    print(json.dumps({k: v for k, v in out.items() if k != "environment"}, indent=1))


if __name__ == "__main__":
    main()
