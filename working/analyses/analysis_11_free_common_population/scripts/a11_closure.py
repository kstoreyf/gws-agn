#!/usr/bin/env python
"""Analysis 11 -- closure identities, live selection support, timing and the
grid-design profiles, in ONE GPU job (one data load, four builds).

CLOSURE (brief s.11).  Every comparison is of the full logL and of its PE and
selection parts; "bitwise" means identical doubles.

  C1  baseline truth reduction.  At (mu_G, mu_chi) = (35, 0) the A11 cell equals
      (a) a LIVE Analysis-10 cell on the same data object, and (b) the RECORDED
      results/a10_arm_J.h5 cells, at 20 (f, dmu_chi, dmu_G) nodes of that cube.
  C2  free vs pinned off the fiducial.  At (mu_G, mu_chi) = (33, -0.05) the A11
      cell equals an Analysis-10-shape cell whose two base slots are PINNED at
      33 / -0.05 (the released coordinate is read through the same
      resolve_parameter_values whether sampled or fixed).
  C3  environmental-zero identity.  At dmu_G = dmu_chi = 0 the two branch
      populations coincide, so the A11 cell equals a K = 2 cell with NO
      per-catalogue population block (one shared population) at the same base
      values, at every f.
  C4  f = 0.  The AGN offsets are irrelevant: logL bitwise constant over
      (dmu_G, dmu_chi) and equal to C3's shared cell at f = 0.
  C5  f = 1.  Only the AGN branch is active, so only mu_G + dmu_G and
      mu_chi + dmu_chi enter: logL constant as (mu_G, mu_chi) move with the
      absolute AGN means held at (40, 0.10).  This is the correct model
      geometry (the decomposition is unidentified at f = 1), not a pathology.
  C6  liveness.  At f = 0.3 the two released base coordinates move logL.

SELECTION (brief s.10, part a): the LIVE dark-siren selection term through the
guard spy at the mass points (mu_G, dmu_G) in {(31,-4),(31,10),(35,5),(39,-4),
(39,10)} and the spin points (mu_chi, dmu_chi) in {(-0.1,-0.05),(-0.1,0.2),
(0,0.1),(0.1,-0.05),(0.1,0.2)}, each at f in {0, 0.3, 1}: f = 0 is the pure
reference branch, f = 1 the pure AGN branch, 0.3 the mixture.

PROFILES (grid design, recorded as such): 1-D conditional profiles through the
planted point along each of the four population axes at f = 0.3, plus a 2-D
four-point stencil for each (mu, dmu) pair, to size the 11A/11B grids.

    sbatch scripts/submit_a11_gpu.sbatch  (A11_SCRIPT=scripts/a11_closure.py)
"""
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
A11 = HERE.parent
sys.path.insert(0, str(HERE))

import a11_likelihood as L                                  # noqa: E402

A10 = L.A10
a8 = L.a8
A10_RESULTS = L.A10_SCRIPTS.parent / "results"
OUT = A11 / "diagnostics" / "a11_closure.json"


def rec_min(r):
    keep = ("logL", "logL_pe", "logL_selection", "Neff", "threshold", "finite",
            "seconds", "log_mu")
    out = {k: r.get(k) for k in keep}
    g = r.get("guard") or {}
    for k in ("sigma2_total", "pe_variance_sum", "passes"):
        out[k] = g.get(k)
    return out


def cmp(a, b):
    """Bitwise / absolute comparison of logL, logL_pe, logL_selection."""
    o = {}
    for k in ("logL", "logL_pe", "logL_selection"):
        x, y = a.get(k), b.get(k)
        if x is None or y is None:
            o[k] = {"a": x, "b": y, "bitwise": x == y}
            continue
        o[k] = {"a": x, "b": y, "abs_diff": abs(x - y),
                "bitwise": float(x).hex() == float(y).hex()}
    o["all_bitwise"] = all(v["bitwise"] for v in o.values() if isinstance(v, dict))
    o["max_abs_diff"] = max((v.get("abs_diff", 0.0) for v in o.values()
                             if isinstance(v, dict)), default=0.0)
    return o


@contextmanager
def pin_base(values):
    """a8.fixed_parameter_values_for('new') with base slots pinned elsewhere."""
    true_fpv = a8.fixed_parameter_values_for

    def fpv(mode):
        out = true_fpv(mode)
        if mode == "new":
            for k, v in values.items():
                if k not in out:
                    raise RuntimeError(k)
                out[k] = float(v)
        return out

    a8.fixed_parameter_values_for = fpv
    try:
        yield
    finally:
        a8.fixed_parameter_values_for = true_fpv


def main():
    import h5py
    t_all = time.time()
    env = L.A10.a8.provenance(gw_path=L.GW_PATH_A11,
                              survey_paths=[L.SURVEY_GAL, L.SURVEY_AGN])
    out = {"what": __doc__.split("    sbatch")[0].strip(), "provenance": env,
           "events_md5_expected": L.GW_MD5_A11, "injections_md5_expected": L.INJ_MD5,
           "slurm_job_id": os.environ.get("SLURM_JOB_ID")}

    t0 = time.time()
    cell = L.build_a11("A11")
    data = cell.data
    out["labels"] = list(cell.labels)
    out["build_seconds"] = time.time() - t0
    ref10 = L.build_a10_reference("A10REF", data=data)
    with pin_base({L.MU_G_LABEL: 33.0, L.MU_CHI_LABEL: -0.05}):
        pinned = A10.build_a10("A10PIN", [L.SURVEY_GAL, L.SURVEY_AGN], data=data,
                               gw_path=L.GW_PATH_A11)
    with A10._steer((), L.A11LikelihoodCell), L._free_base():
        shared = a8.build("A11SHARED", "new", [L.SURVEY_GAL, L.SURVEY_AGN], data=data,
                          gw_path=L.GW_PATH_A11)
    out["labels_shared"] = list(shared.labels)
    out["labels_pinned"] = list(pinned.labels)

    ev = lambda **kw: rec_min(cell.evaluate_at(H0=L.H0_FID, **kw))

    # ---- timing (first call compiles) ----------------------------------- #
    secs = [cell.evaluate_at(H0=L.H0_FID, fcat_2=0.3, mu_G=35, dmu_G=5, mu_chi=0,
                             dmu_chi=0.1)["seconds"] for _ in range(6)]
    out["timing"] = {"first_call_seconds": secs[0],
                     "steady_state_median_seconds": float(np.median(secs[1:]))}
    print(f"[timing] {out['timing']}")

    # ---- C1 ---------------------------------------------------------------- #
    with h5py.File(A10_RESULTS / "a10_arm_J.h5", "r") as h:
        fg, gg, cg = h["f_grid"][:], h["dmu_G_grid"][:], h["dmu_chi_grid"][:]
        rll, rpe, rsel = (h["log_likelihood"][:], h["guard/logL_pe"][:],
                          h["guard/logL_selection"][:])
    c1 = []
    for f in (0.0, 0.1, 0.3, 0.5, 1.0):
        for dchi, dG in ((0.1, 5.0), (0.1075, 4.5), (0.0025, 0.0), (0.1, -3.0)):
            i = int(np.argmin(abs(fg - f)))
            k = int(np.argmin(abs(gg - dG)))
            j = int(np.argmin(abs(cg - dchi)))
            f_, g_, c_ = float(fg[i]), float(gg[k]), float(cg[j])
            a = ev(fcat_2=f_, mu_G=35.0, dmu_G=g_, mu_chi=0.0, dmu_chi=c_)
            b = rec_min(ref10.evaluate_at(H0=L.H0_FID, fcat_2=f_, dmu_chi=c_, dmu_G=g_))
            r = {"logL": float(rll[i, k, j]), "logL_pe": float(rpe[i, k, j]),
                 "logL_selection": float(rsel[i, k, j])}
            c1.append({"f": f_, "dmu_G": g_, "dmu_chi": c_,
                       "vs_live_A10": cmp(a, b), "vs_recorded_a10_arm_J": cmp(a, r)})
    out["C1_baseline_truth_reduction"] = {
        "cells": c1,
        "live_all_bitwise": all(x["vs_live_A10"]["all_bitwise"] for x in c1),
        "recorded_all_bitwise": all(x["vs_recorded_a10_arm_J"]["all_bitwise"] for x in c1),
        "live_max_abs": max(x["vs_live_A10"]["max_abs_diff"] for x in c1),
        "recorded_max_abs": max(x["vs_recorded_a10_arm_J"]["max_abs_diff"] for x in c1)}
    print(f"[C1] live bitwise {out['C1_baseline_truth_reduction']['live_all_bitwise']}, "
          f"recorded bitwise {out['C1_baseline_truth_reduction']['recorded_all_bitwise']} "
          f"(max {out['C1_baseline_truth_reduction']['recorded_max_abs']:.2e})")

    # ---- C2 ---------------------------------------------------------------- #
    c2 = []
    for f, dG, dchi in ((0.3, 5.0, 0.1), (0.0, 7.0, 0.15), (1.0, -2.0, 0.0), (0.6, 0.0, 0.05)):
        a = ev(fcat_2=f, mu_G=33.0, dmu_G=dG, mu_chi=-0.05, dmu_chi=dchi)
        b = rec_min(pinned.evaluate(**{"H0": L.H0_FID, "fcat_2": f,
                                       L.MU_G_C2_LABEL: 33.0 + dG,
                                       L.MU_CHI_C2_LABEL: -0.05 + dchi}))
        c2.append({"f": f, "dmu_G": dG, "dmu_chi": dchi, "cmp": cmp(a, b)})
    out["C2_free_vs_pinned_off_fiducial"] = {
        "cells": c2, "all_bitwise": all(x["cmp"]["all_bitwise"] for x in c2),
        "max_abs": max(x["cmp"]["max_abs_diff"] for x in c2)}
    print(f"[C2] bitwise {out['C2_free_vs_pinned_off_fiducial']['all_bitwise']}")

    # ---- C3 ---------------------------------------------------------------- #
    c3 = []
    for f in (0.0, 0.3, 0.7, 1.0):
        a = ev(fcat_2=f, mu_G=33.0, dmu_G=0.0, mu_chi=-0.05, dmu_chi=0.0)
        b = rec_min(shared.evaluate(**{"H0": L.H0_FID, "fcat_2": f,
                                       L.MU_G_LABEL: 33.0, L.MU_CHI_LABEL: -0.05}))
        c3.append({"f": f, "cmp": cmp(a, b)})
    out["C3_environmental_zero_identity"] = {
        "cells": c3, "all_bitwise": all(x["cmp"]["all_bitwise"] for x in c3),
        "max_abs": max(x["cmp"]["max_abs_diff"] for x in c3)}
    print(f"[C3] bitwise {out['C3_environmental_zero_identity']['all_bitwise']} "
          f"max {out['C3_environmental_zero_identity']['max_abs']:.2e}")

    # ---- C4 ---------------------------------------------------------------- #
    c4 = [dict(dG=dG, dchi=dchi, **ev(fcat_2=0.0, mu_G=33.0, dmu_G=dG, mu_chi=-0.05,
                                      dmu_chi=dchi))
          for dG in (-4.0, 0.0, 5.0, 10.0) for dchi in (-0.05, 0.2)]
    ref0 = c3[0]["cmp"]["logL"]["b"]
    out["C4_f0_offsets_irrelevant"] = {
        "cells": c4,
        "all_bitwise_equal": len({float(x["logL"]).hex() for x in c4}) == 1,
        "equals_shared_f0": all(float(x["logL"]).hex() == float(ref0).hex() for x in c4),
        "spread": float(np.ptp([x["logL"] for x in c4]))}
    print(f"[C4] {out['C4_f0_offsets_irrelevant']['all_bitwise_equal']} "
          f"spread {out['C4_f0_offsets_irrelevant']['spread']:.2e}")

    # ---- C5 ---------------------------------------------------------------- #
    c5 = [dict(mu_G=mg, mu_chi=mc, **ev(fcat_2=1.0, mu_G=mg, dmu_G=40.0 - mg,
                                        mu_chi=mc, dmu_chi=0.10 - mc))
          for mg in (31.0, 35.0, 39.0) for mc in (-0.10, 0.0, 0.10)]
    vals = [x["logL"] for x in c5]
    out["C5_f1_only_sums_enter"] = {
        "cells": c5, "all_bitwise_equal": len({float(v).hex() for v in vals}) == 1,
        "spread": float(np.ptp(vals)),
        "spread_in_ulp_of_logL": float(np.ptp(vals) / np.spacing(abs(vals[0])))}
    print(f"[C5] bitwise {out['C5_f1_only_sums_enter']['all_bitwise_equal']} "
          f"spread {out['C5_f1_only_sums_enter']['spread']:.2e}")

    # ---- profiles (also C6 liveness) --------------------------------------- #
    T = dict(fcat_2=0.3, mu_G=35.0, dmu_G=5.0, mu_chi=0.0, dmu_chi=0.10)
    axes = {"mu_G": np.round(np.arange(31.0, 39.0001, 0.25), 6),
            "dmu_G": np.round(np.arange(-4.0, 10.0001, 0.5), 6),
            "mu_chi": np.round(np.arange(-0.10, 0.10001, 0.005), 6),
            "dmu_chi": np.round(np.arange(-0.05, 0.20001, 0.01), 6)}
    prof = {}
    for name, grid in axes.items():
        rows = [dict(T, **{name: float(x)}) for x in grid]
        res = [ev(**r) for r in rows]
        ll = np.array([x["logL"] for x in res])
        prof[name] = {"grid": grid.tolist(), "logL": ll.tolist(),
                      "logL_selection": [x["logL_selection"] for x in res],
                      "Neff": [x["Neff"] for x in res],
                      "argmax": float(grid[int(np.nanargmax(ll))]),
                      "range_logL": float(np.nanmax(ll) - np.nanmin(ll))}
        # quadratic width about the argmax (conditional sd)
        k = int(np.nanargmax(ll))
        if 0 < k < len(grid) - 1:
            h = grid[1] - grid[0]
            c = -(ll[k + 1] - 2 * ll[k] + ll[k - 1]) / h ** 2
            prof[name]["conditional_sd"] = float(1.0 / np.sqrt(c)) if c > 0 else None
        print(f"[profile] {name}: argmax {prof[name]['argmax']} "
              f"cond sd {prof[name].get('conditional_sd')} range {prof[name]['range_logL']:.1f}")
    out["profiles_at_planted_point"] = prof
    out["C6_liveness"] = {k: prof[k]["range_logL"] for k in ("mu_G", "mu_chi")}

    # 2-D stencils for the (mu, dmu) correlations
    st = {}
    for a_, b_, ha, hb in (("mu_G", "dmu_G", 0.5, 0.5), ("mu_chi", "dmu_chi", 0.01, 0.01)):
        c0 = ev(**T)["logL"]
        v = {}
        for sa in (-1, 1):
            for sb in (-1, 1):
                v[(sa, sb)] = ev(**dict(T, **{a_: T[a_] + sa * ha, b_: T[b_] + sb * hb}))["logL"]
        pa = prof[a_]
        pb = prof[b_]
        def second(p, x0, h):
            g = np.array(p["grid"]); l = np.array(p["logL"])
            k = int(np.argmin(abs(g - x0)))
            step = g[1] - g[0]
            m = int(round(h / step))
            return (l[k + m] - 2 * l[k] + l[k - m]) / (m * step) ** 2
        Haa = second(pa, T[a_], ha)
        Hbb = second(pb, T[b_], hb)
        Hab = (v[(1, 1)] - v[(1, -1)] - v[(-1, 1)] + v[(-1, -1)]) / (4 * ha * hb)
        Hm = -np.array([[Haa, Hab], [Hab, Hbb]])
        try:
            C = np.linalg.inv(Hm)
            sd = np.sqrt(np.diag(C))
            st[f"{a_}|{b_}"] = {"hessian_neg": Hm.tolist(), "marginal_sd": sd.tolist(),
                                "rho": float(C[0, 1] / (sd[0] * sd[1]))}
        except Exception as e:                                   # noqa: BLE001
            st[f"{a_}|{b_}"] = {"hessian_neg": Hm.tolist(), "error": str(e)}
        print(f"[stencil] {a_},{b_}: {st[f'{a_}|{b_}']}")
    out["local_gaussian_at_planted_point"] = st

    # ---- selection support, live --------------------------------------- #
    sel = []
    for (mg, dg) in ((31, -4), (31, 10), (35, 5), (39, -4), (39, 10)):
        for f in (0.0, 0.3, 1.0):
            sel.append(dict(sector="mass", f=f, mu_G=mg, dmu_G=dg, mu_chi=0.0, dmu_chi=0.10,
                            **ev(fcat_2=f, mu_G=mg, dmu_G=dg, mu_chi=0.0, dmu_chi=0.10)))
    for (mc, dc) in ((-0.10, -0.05), (-0.10, 0.20), (0.0, 0.10), (0.10, -0.05), (0.10, 0.20)):
        for f in (0.0, 0.3, 1.0):
            sel.append(dict(sector="spin", f=f, mu_G=35.0, dmu_G=5.0, mu_chi=mc, dmu_chi=dc,
                            **ev(fcat_2=f, mu_G=35.0, dmu_G=5.0, mu_chi=mc, dmu_chi=dc)))
    for s in sel:
        s["Neff_over_threshold"] = (s["Neff"] / s["threshold"]
                                    if s.get("Neff") and s.get("threshold") else None)
    out["selection_live"] = {
        "cells": sel,
        "n_rejected": int(sum(1 for s in sel if not s["finite"])),
        "min_Neff_over_threshold": min(s["Neff_over_threshold"] for s in sel
                                       if s["Neff_over_threshold"] is not None)}
    print(f"[selection] rejected {out['selection_live']['n_rejected']}/{len(sel)}, "
          f"min Neff/thr {out['selection_live']['min_Neff_over_threshold']:.2f}")

    out["verdict"] = {
        "C1": out["C1_baseline_truth_reduction"]["live_all_bitwise"]
              and out["C1_baseline_truth_reduction"]["recorded_all_bitwise"],
        "C2": out["C2_free_vs_pinned_off_fiducial"]["all_bitwise"],
        "C3": out["C3_environmental_zero_identity"]["all_bitwise"],
        "C4": out["C4_f0_offsets_irrelevant"]["all_bitwise_equal"],
        "C5": out["C5_f1_only_sums_enter"]["all_bitwise_equal"],
        "C6": all(v > 1.0 for v in out["C6_liveness"].values()),
    }
    out["seconds_total"] = time.time() - t_all
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {OUT}\nverdict {out['verdict']}")


if __name__ == "__main__":
    main()
