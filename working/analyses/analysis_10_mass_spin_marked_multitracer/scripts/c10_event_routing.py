#!/usr/bin/env python
"""Analysis 10, gate C10-2 -- event-level routing with H0 released.

WHAT THIS COMPUTES
------------------
Analysis 9 measured its routing mechanism per event at ten H0 nodes.  This is
the same measurement for the two-mark model, with the four nested models of
``a10_event_decomposition.py`` (spatial, +spin, +mass, +mass+spin) at every node
instead of one H0 = 67.74 point.

The point is P1's pin: f_AGN = 0.275 (F_GRID node 11), dmu_chi = +0.115,
dmu_G = +5.0.  H0 runs over twelve nodes, [66.0, 71.0] on the 0.5 lattice plus
67.74, which are nodes of both the P1 arm and the C10-S lattice (asserted
bitwise).  They bracket both posteriors: P1's median is 67.58 and C10-S's is
70.60.

At each node and for each model Y in {G, chi, M, AM} (AGN-branch population
vector Lambda_G, Lambda_G[mu_chi], Lambda_G[G.mu], Lambda_G[both]):

    ln Z_i^Y(H0) = logaddexp(log(1-f) + log E_i[GG], log f + log E_i[A|Y])
    P_i^Y(H0)    = f E_i[A|Y] / Z_i^Y

and the live production evaluation of the same model gives its selection term.

VERIFICATION (every node; anything above 1e-6 absolute is a FAIL)
------------------------------------------------------------------
    sum_i ln Z_i^Y            == logL_pe of the live evaluate_at, each Y
    sum_i logZ_prod (AM pass) == the same, for AM
    sum_i ln Z_i^AM           == the recorded P1 cell (H0, f = 0.275)
    sum_i ln Z_i^G            == the recorded C10-S cell (H0, f = 0.275)

THE READOUT (stage ``analyse``, CPU)
------------------------------------
At two stencil centres, 67.5 (the marked peak) and 70.5 (the spatial peak),
with h = 0.5:

    s_i^Y = [ln Z_i^Y(H0+h) - ln Z_i^Y(H0-h)] / 2h          per-event score
    I_i^Y = -[ln Z_i^Y(H0+h) - 2 ln Z_i^Y(H0) + ln Z_i^Y(H0-h)] / h^2
    ds_i^Y = s_i^Y - s_i^G,  dI_i^Y = I_i^Y - I_i^G        what the mark adds

The event-term / selection-term split of the added curvature AND of the added
score (the marks mainly SHIFT H0 on this mock, so the score budget is the
primary one), the |dP| distribution and 0.5 crossers, the share of the added
score and curvature carried by the most re-routed events, and the fixed-f H0
width ratio P1/P0 at every f node inside C10-J's 90% interval (from
results/c10_mech.h5, no GPU).

The true host label is read ONLY in the analyse stage, for interpretation
columns.  It is never an input.

USAGE
-----
    sbatch scripts/submit_c10_event_routing_rita.sbatch            # GPU + analyse
    JAX_PLATFORMS=cpu python scripts/c10_event_routing.py --dry_run
    JAX_PLATFORMS=cpu python scripts/c10_event_routing.py --analyse_only
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE.parent
DIAG = ANALYSIS_DIR / "diagnostics"
RESULTS = ANALYSIS_DIR / "results"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import a10_event_decomposition as ED               # noqa: E402  capture + passes
import c10_scan as CS                              # noqa: E402  C10 inputs of record

A10, A10C, A9, A8 = ED.A10, ED.A10C, ED.A9, ED.A8

OUT_H5 = DIAG / "c10_event_routing.h5"
OUT_JSON = DIAG / "c10_event_routing.json"

F_IDX = 11                                        # F_GRID[11] = 0.275
DMU_CHI = CS.PROFILE_DMU_CHI                      # +0.115, P1's pin
DMU_G = CS.PROFILE_DMU_G                          # +5.0,   P1's pin
H0_NODES = [66.0, 66.5, 67.0, 67.5, 67.74, 68.0, 68.5, 69.0, 69.5, 70.0, 70.5, 71.0]
CENTRES = {"marked_peak": 67.5, "spatial_peak": 70.5}
H = 0.5
MODELS = ("G", "chi", "M", "AM")                  # spatial, +spin, +mass, +both
MODEL_MARKS = {"G": (0.0, 0.0), "chi": (DMU_CHI, 0.0),
               "M": (0.0, DMU_G), "AM": (DMU_CHI, DMU_G)}
FAIL_ABS = 1.0e-6
TOP_K = (25, 75, 150)


# --------------------------------------------------------------------------- #
# Recorded references: P1 (marked, pinned) and C10-S (spatial) at f = 0.275
# --------------------------------------------------------------------------- #
def recorded_references():
    f_val = float(CS.F_GRID[F_IDX])
    if abs(f_val - 0.275) > 1e-15:
        raise SystemExit(f"[fatal] F_GRID[{F_IDX}] = {f_val!r}, not 0.275")
    _h, p1_rows, _d = CS._read_jsonl(DIAG / "_c10_mech_P1.jsonl")
    p1 = {CS._key1(r["H0"]): r["cells"][F_IDX] for r in p1_rows}
    s_done, _hdr, _d = CS._load_done_s()
    ref = {}
    for h0 in H0_NODES:
        k = CS._key1(h0)
        if k not in p1 or k not in s_done:
            raise SystemExit(f"[fatal] H0 = {h0} is missing from P1 or C10-S")
        ref[h0] = {"P1_logL_pe": float(p1[k]["logL_pe"]),
                   "P1_logL_selection": float(p1[k]["logL_selection"]),
                   "S_logL_pe": float(s_done[k][1]["cells"][F_IDX]["logL_pe"]),
                   "S_logL_selection": float(s_done[k][1]["cells"][F_IDX]["logL_selection"])}
    return f_val, ref


# --------------------------------------------------------------------------- #
# GPU stage
# --------------------------------------------------------------------------- #
def run_gpu(args):
    import h5py
    import jax.numpy as jnp

    env = A9._gpu_setup("c10_event_routing")
    env["a10_inputs"] = A10C.assert_a10_inputs(with_md5=True)
    f_agn, ref = recorded_references()
    i_G, _, _ = A10.fiducial_slot("G.mu")
    i_chi, _, _ = A10.fiducial_slot("mu_chi")
    mu_chi_c2 = A10.dmuchi_to_label(DMU_CHI)
    mu_G_c2 = A10.dmuG_to_label(DMU_G)

    t0 = time.time()
    cell = A10.build_a10("C10ROUTE", [A8.SURVEY_GAL, A8.SURVEY_AGN], verbose=True,
                         gw_path=A10C.GW_PATH_A10)
    A10C.assert_cell(cell, A10.EXPECTED_LABELS_A10, "C10 routing")
    print(f"[route] build {time.time() - t0:.1f}s")
    block = int(A8.SETTINGS["pe_event_block"])

    n_nodes = len(H0_NODES)
    logZ = {m: None for m in MODELS}
    logEA = {m: None for m in MODELS}
    logEGG = None
    logZ_prod = None
    live_rec = {m: {"logL": [], "logL_pe": [], "logL_selection": []} for m in MODELS}
    checks = []
    for a, h0 in enumerate(H0_NODES):
        print(f"\n== H0 = {h0} ==")
        live = {}
        for m in MODELS:
            dc, dg = MODEL_MARKS[m]
            r = cell.evaluate_at(H0=h0, fcat_2=f_agn, dmu_chi=dc, dmu_G=dg)
            live[m] = r
            for k in live_rec[m]:
                live_rec[m][k].append(float(r[k]))
            print(f"  live {m:<3s} logL_pe={r['logL_pe']!r} "
                  f"logL_selection={r['logL_selection']!r}")

        coord = cell.coord(H0=float(h0), fcat_2=f_agn,
                           **{A10.MU_CHI_C2_LABEL: mu_chi_c2, A10.MU_G_C2_LABEL: mu_G_c2})
        ed, cap = ED.capture_a10(cell, coord)
        h0_cap = float(np.asarray(cap["args"][0].H0))
        if float(h0_cap).hex() != float(h0).hex():
            raise SystemExit(f"[fatal] captured cosmo.H0 = {h0_cap!r} != {h0!r}")
        if int(cap["kwargs"]["n_catalogs"]) != 2:
            raise SystemExit("[fatal] the capture is not K = 2")
        lam_G = np.asarray(cap["args"][2], dtype=float)
        lam_AM = np.asarray(tuple(cap["kwargs"]["mixture_pop_params"])[0], dtype=float)
        lam = {"G": lam_G.copy(), "chi": lam_G.copy(), "M": lam_G.copy(), "AM": lam_G.copy()}
        lam["chi"][i_chi] = mu_chi_c2
        lam["M"][i_G] = mu_G_c2
        lam["AM"][i_chi] = mu_chi_c2
        lam["AM"][i_G] = mu_G_c2
        if not np.array_equal(lam["AM"], lam_AM):
            raise SystemExit("[fatal] rebuilt Lambda_AM != captured; the slot map is wrong")

        logE_A = {}
        logE_GG = None
        for m in ("AM", "chi", "M", "G"):
            cols, _nk, nEv, _ns, log_w, _tb, _tp = ED.per_event_pass(
                ed, ED.cap_with_agn_population(cap, jnp.asarray(lam[m])), block, m)
            if abs(float(np.exp(log_w[1])) - f_agn) > 1e-12:
                raise SystemExit("[fatal] captured mixture weight != f")
            logE_A[m] = cols["logE_AA"]
            if m == "AM":
                logE_GG, lf = cols["logE_GG"], (float(log_w[0]), float(log_w[1]))
                zp = cols["logZ_prod"]
            elif not np.array_equal(cols["logE_GG"], logE_GG):
                raise SystemExit(f"[fatal] E[GG] depends on the AGN vector (pass {m})")
        del cap
        if logEGG is None:
            logEGG = np.full((n_nodes, nEv), np.nan)
        logEGG[a] = logE_GG
        for m in MODELS:
            if logZ[m] is None:
                logZ[m] = np.full((n_nodes, nEv), np.nan)
                logEA[m] = np.full((n_nodes, nEv), np.nan)
            logEA[m][a] = logE_A[m]
            logZ[m][a] = np.logaddexp(lf[0] + logE_GG, lf[1] + logE_A[m])
        if logZ_prod is None:
            logZ_prod = np.full((n_nodes, nEv), np.nan)
        logZ_prod[a] = zp

        row = {"H0": h0}
        for m in MODELS:
            row[f"sum_{m}_minus_live"] = float(np.sum(logZ[m][a]) - live[m]["logL_pe"])
        row["sum_prod_minus_live_AM"] = float(np.sum(zp) - live["AM"]["logL_pe"])
        row["live_AM_minus_recorded_P1"] = float(live["AM"]["logL_pe"] - ref[h0]["P1_logL_pe"])
        row["live_G_minus_recorded_S"] = float(live["G"]["logL_pe"] - ref[h0]["S_logL_pe"])
        row["sel_AM_minus_recorded_P1"] = float(live["AM"]["logL_selection"]
                                                - ref[h0]["P1_logL_selection"])
        row["sel_G_minus_recorded_S"] = float(live["G"]["logL_selection"]
                                              - ref[h0]["S_logL_selection"])
        worst = max(abs(v) for k, v in row.items() if k != "H0")
        row["worst_abs"] = worst
        row["pass"] = bool(worst <= FAIL_ABS)
        checks.append(row)
        print(f"  verification worst |d| = {worst:.3e} -> {'PASS' if row['pass'] else 'FAIL'}")
        if not row["pass"]:
            print(json.dumps(row, indent=1))
            raise SystemExit("[fatal] a per-event sum does not reproduce production")

    with h5py.File(OUT_H5, "w") as h:
        h.create_dataset("H0_nodes", data=np.array(H0_NODES))
        h.create_dataset("logE_GG", data=logEGG)
        for m in MODELS:
            h.create_dataset(f"logZ_{m}", data=logZ[m])
            h.create_dataset(f"logE_A_{m}", data=logEA[m])
            for k, v in live_rec[m].items():
                h.create_dataset(f"live_{m}_{k}", data=np.array(v))
        h.create_dataset("logZ_prod_AM", data=logZ_prod)
        h.attrs["f_agn"] = f_agn
        h.attrs["dmu_chi"] = DMU_CHI
        h.attrs["dmu_G"] = DMU_G
        h.attrs["log_w"] = json.dumps(list(lf))
        h.attrs["events_file"] = str(A10C.GW_PATH_A10)
        h.attrs["verification"] = json.dumps(checks)
        h.attrs["environment"] = json.dumps(env, default=str)
        h.attrs["slurm_job_id"] = os.environ.get("SLURM_JOB_ID", "")
    print(f"wrote {OUT_H5}")


# --------------------------------------------------------------------------- #
# CPU analyse stage
# --------------------------------------------------------------------------- #
def _fixed_f_width_ratios():
    import h5py
    mci = A8.import_scan_h0f().marginal_ci
    j = json.loads((RESULTS / "c10_arm_J.json").read_text())
    lo, hi = j["f_agn"]["ci90"]
    out = []
    with h5py.File(RESULTS / "c10_mech.h5", "r") as h:
        h0, fg = h["H0_grid"][:], h["f_grid"][:]
        L1, L0 = h["P1/log_likelihood"][:], h["P0/log_likelihood"][:]
    for i, f in enumerate(fg):
        if not lo <= f <= hi:
            continue
        a, b = mci(h0, L1[:, i]), mci(h0, L0[:, i])
        w = lambda c, k: c[k][1] - c[k][0]
        out.append({"f_agn": float(f), "P1_median": a["median"], "P0_median": b["median"],
                    "ratio_68": w(a, "ci68") / w(b, "ci68"),
                    "ratio_90": w(a, "ci90") / w(b, "ci90"),
                    "shift_P1_minus_P0": a["median"] - b["median"]})
    return {"f_range_C10J_90": [lo, hi], "per_f": out}


def analyse(args):
    import h5py
    with h5py.File(OUT_H5, "r") as h:
        nodes = h["H0_nodes"][:]
        logZ = {m: h[f"logZ_{m}"][:] for m in MODELS}
        logEA = {m: h[f"logE_A_{m}"][:] for m in MODELS}
        logEGG = h["logE_GG"][:]
        sel = {m: h[f"live_{m}_logL_selection"][:] for m in MODELS}
        f = float(h.attrs["f_agn"])
        lf = json.loads(h.attrs["log_w"])
        ver = json.loads(h.attrs["verification"])
    with h5py.File(A10C.GW_PATH_A10, "r") as h:            # interpretation only
        host = h["host_type"][:].astype(int)
    is_agn = host == 1
    k_of = {float(x): k for k, x in enumerate(nodes)}
    fin = np.all([np.isfinite(logZ[m]).all(axis=0) for m in MODELS], axis=0)
    n_ex = int((~fin).sum())

    def P(m, a):
        return 1.0 / (1.0 + np.exp((lf[0] + logEGG[a]) - (lf[1] + logEA[m][a])))

    out = {"point": {"f_agn": f, "dmu_chi": DMU_CHI, "dmu_G": DMU_G},
           "H0_nodes": nodes.tolist(),
           "verification": ver,
           "verification_all_pass": bool(all(r["pass"] for r in ver)),
           "verification_worst_abs": float(max(r["worst_abs"] for r in ver)),
           "n_events": int(fin.size), "n_events_excluded_nonfinite": n_ex,
           "centres": {}}

    # the whole-likelihood lines, per model: the H0 profile at fixed (f, marks)
    out["profile_lines"] = {
        m: {"logL_pe_minus_max": (logZ[m].sum(1) - logZ[m].sum(1).max()).tolist(),
            "logL_selection": sel[m].tolist(),
            "total_argmax_H0": float(nodes[int(np.argmax(logZ[m].sum(1) + sel[m]))])}
        for m in MODELS}

    for cname, c in CENTRES.items():
        a, lo, hi = k_of[c], k_of[c - H], k_of[c + H]
        blk = {"H0": c, "stencil": [c - H, c, c + H]}
        s = {m: (logZ[m][hi] - logZ[m][lo]) / (2 * H) for m in MODELS}
        I = {m: -(logZ[m][hi] - 2 * logZ[m][a] + logZ[m][lo]) / H ** 2 for m in MODELS}
        s_sel = {m: (sel[m][hi] - sel[m][lo]) / (2 * H) for m in MODELS}
        I_sel = {m: -(sel[m][hi] - 2 * sel[m][a] + sel[m][lo]) / H ** 2 for m in MODELS}
        PG = P("G", a)
        blk["spatial_totals"] = {
            "score_pe": float(s["G"][fin].sum()), "score_selection": float(s_sel["G"]),
            "curvature_pe": float(I["G"][fin].sum()), "curvature_selection": float(I_sel["G"])}
        for m in ("chi", "M", "AM"):
            ds, dI = (s[m] - s["G"])[fin], (I[m] - I["G"])[fin]
            dP = (P(m, a) - PG)[fin]
            order = np.argsort(-np.abs(dP))
            tot_s, tot_I = float(ds.sum()), float(dI.sum())
            curv_new = float(I[m][fin].sum() + I_sel[m])
            curv_old = float(I["G"][fin].sum() + I_sel["G"])
            score_new = float(s[m][fin].sum() + s_sel[m])
            score_old = float(s["G"][fin].sum() + s_sel["G"])
            mb = {
                "dP": ED.dp_block(m, dP, PG[fin], P(m, a)[fin], int(is_agn[fin].sum())),
                "added_score": {
                    "pe": tot_s, "selection": float(s_sel[m] - s_sel["G"]),
                    "total": tot_s + float(s_sel[m] - s_sel["G"]),
                    "implied_peak_shift_newton": (score_new / curv_new) - (score_old / curv_old),
                    "note": ("score at the stencil centre; the Newton step "
                             "score/curvature from the centre is the local peak "
                             "offset, so its change is the shift the mark induces")},
                "added_curvature": {
                    "pe": tot_I, "selection": float(I_sel[m] - I_sel["G"]),
                    "total": tot_I + float(I_sel[m] - I_sel["G"]),
                    "ratio_total_new_over_spatial": curv_new / curv_old,
                    "implied_width_ratio": float(np.sqrt(curv_old / curv_new))},
                "concentration_by_abs_dP": {
                    str(k): {"share_of_added_score_pe": float(ds[order[:k]].sum() / tot_s)
                             if tot_s else None,
                             "share_of_added_curvature_pe": float(dI[order[:k]].sum() / tot_I)
                             if tot_I else None,
                             "min_abs_dP_in_set": float(np.abs(dP[order[k - 1]]))}
                    for k in TOP_K},
                "abs_dP_gt_0p1_set": {
                    "n": int((np.abs(dP) > 0.1).sum()),
                    "added_score_pe": float(ds[np.abs(dP) > 0.1].sum()),
                    "added_curvature_pe": float(dI[np.abs(dP) > 0.1].sum())},
                "corr_absdP_vs_dscore": ED._corr(np.abs(dP), ds),
                "corr_absdP_vs_dcurv": ED._corr(np.abs(dP), dI),
                "per_event_dscore_pct": ED._pct(ds),
                "per_event_dcurv_pct": ED._pct(dI),
                "by_true_host (interpretation only)": {
                    "added_score_pe_true_AGN": float(ds[is_agn[fin]].sum()),
                    "added_score_pe_true_GAL": float(ds[~is_agn[fin]].sum()),
                    "added_curvature_pe_true_AGN": float(dI[is_agn[fin]].sum()),
                    "added_curvature_pe_true_GAL": float(dI[~is_agn[fin]].sum())},
            }
            blk[m] = mb
        out["centres"][cname] = blk

    out["fixed_f_width_ratio_P1_over_P0"] = _fixed_f_width_ratios()

    with h5py.File(OUT_H5, "a") as h:
        if "true_host_type" not in h:
            h.create_dataset("true_host_type", data=host)
    ED._write(OUT_JSON, out)

    for cname, blk in out["centres"].items():
        print(f"\n== centre {cname} (H0 = {blk['H0']}) ==")
        print(f"  spatial: score pe {blk['spatial_totals']['score_pe']:+.3f} "
              f"sel {blk['spatial_totals']['score_selection']:+.3f}; curvature pe "
              f"{blk['spatial_totals']['curvature_pe']:.3f} sel "
              f"{blk['spatial_totals']['curvature_selection']:.3f}")
        for m in ("chi", "M", "AM"):
            b = blk[m]
            print(f"  {m:<3s} added score pe {b['added_score']['pe']:+.3f} sel "
                  f"{b['added_score']['selection']:+.3f} -> shift "
                  f"{b['added_score']['implied_peak_shift_newton']:+.3f}; added curv pe "
                  f"{b['added_curvature']['pe']:+.3f} sel {b['added_curvature']['selection']:+.3f}"
                  f" -> width x{b['added_curvature']['implied_width_ratio']:.3f}; "
                  f"|dP|>0.1: {b['dP']['n_abs_gt']['0.1']}, crossers "
                  f"{b['dP']['n_crossing_0p5_total']}, top-75 score share "
                  f"{b['concentration_by_abs_dP']['75']['share_of_added_score_pe']}")
    ff = out["fixed_f_width_ratio_P1_over_P0"]["per_f"]
    print("\n== fixed-f H0 width ratio P1/P0 ==")
    for r in ff:
        print(f"  f={r['f_agn']:.3f}  68% {r['ratio_68']:.4f}  90% {r['ratio_90']:.4f}  "
              f"shift {r['shift_P1_minus_P0']:+.3f}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--analyse_only", action="store_true")
    args = ap.parse_args(argv)
    if args.dry_run:
        f_val, ref = recorded_references()
        print(f"[dry-run] f = {f_val!r}; references found at {len(ref)} H0 nodes")
        for h0, r in ref.items():
            print(f"  {h0:6.2f}  P1 logL_pe {r['P1_logL_pe']:.6f}  S logL_pe {r['S_logL_pe']:.6f}")
        return 0
    if not args.analyse_only:
        run_gpu(args)
    analyse(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
