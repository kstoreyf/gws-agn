#!/usr/bin/env python
"""Analysis 9, mechanism follow-up part B -- the event-level routing diagnostic.

WHAT THIS COMPUTES, AND WHY IT NEEDS A GPU JOB
----------------------------------------------
The J9 and S9 cubes store only the SUMMED log-likelihood per cell.  Every
quantity below is per EVENT, so it cannot be read off disk; it is recomputed at
ten H0 nodes with the Analysis-8 machinery, and then CHECKED against the
recorded cubes at every one of those nodes.

The model.  Per event i the marked (J9) likelihood is

    Z_i^J(H0) = f_G E_i[GG] + f_A E_i[AA],
    E_i[XY]   = (1/n) sum_s exp(A_X(s) + B_Y(s) + C(s)),

with A_k the SPATIAL factor of tracer k (H0 enters here, through z(dL; H0)),
B_k the branch population factor and C branch-free -- exactly the decomposition
``analysis_8/scripts/event_decomposition.py`` defines and verifies, whose
``capture_operands`` / ``build_evaluator`` this script imports rather than
reimplements.

The spatial-only arm S9 holds dmu_chi = 0, so BOTH branches carry the GAL
population, B_A == B_G, and therefore

    Z_i^S(H0) = f_G E_i[GG] + f_A E_i[AG]      EXACTLY

(closure 9.2 measured the two code shapes at <= 2 ULP).  One capture at
(H0, f, mu_chi_c2) therefore yields BOTH arms' per-event likelihoods: the mark
changes E[AG] -> E[AA] on the AGN branch and nothing else.

Hence, at fixed f,

    P_i(AGN | spatial)        = sigmoid(log(f_A/f_G) + log E[AG] - log E[GG])
    P_i(AGN | spatial + mark) = sigmoid(log(f_A/f_G) + log E[AA] - log E[GG])
    dP_i                      = the difference          ("routing")
    dlnL_i(H0)                = ln Z_i^J(H0) - ln Z_i^S(H0)

and the per-event H0 curvature (the event's own H0 information at the node)

    I_i^X = -[ln Z_i^X(69.5) - 2 ln Z_i^X(69.0) + ln Z_i^X(68.5)] / 0.25,
    dI_i  = I_i^J - I_i^S.

The question this answers is whether the events whose branch assignment the
mark MOVES are the events whose H0 information the mark CHANGES.

THE POINT
---------
f_AGN = 0.275 (f index 11) and mu_chi_c2 = +0.1075 (mu index 41): the shared
S9/J9 MAP node, and a registered node of both cubes, so every sum below has a
recorded number to be checked against.  H0 runs over ten nodes that are exact
members of BOTH H0 axes (asserted bitwise); the representative point is 69.0.

VERIFICATION (read from the cubes, never recomputed)
----------------------------------------------------
At every node:
    sum_i logaddexp(log f_G + log E[GG], log f_A + log E[AA])
        == j9_marked.h5  guard/logL_pe[iJ9, 11, 41]
    sum_i logaddexp(log f_G + log E[GG], log f_A + log E[AG])
        == s9_spatial.h5 guard/logL_pe[iS9, 11]
    sum_i logZ_prod (the production per-event value)
        == j9_marked.h5  guard/logL_pe[iJ9, 11, 41]
plus ONE live production ``cell.evaluate`` at the representative node.  Both
cubes were computed on rita A100-80s, so the expectation is <= 2 ULP of a value
~4.2e3 (1 ULP = 2^-40 = 9.094947e-13); anything above 1e-6 absolute is a FAIL.

The true host label is read ONLY after every inference quantity is formed, and
only for the interpretation columns.  It is never an input.

USAGE
-----
    sbatch scripts/submit_a9_event_routing_rita.sbatch     # the GPU run
    JAX_PLATFORMS=cpu python scripts/a9_event_routing.py --dry_run   # preflight
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Never write a .pyc into the read-only Analysis-8 tree.
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE.parent
DIAG = ANALYSIS_DIR / "diagnostics"
RESULTS = ANALYSIS_DIR / "results"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import a9_scan as A9S                      # noqa: E402  the environment of record

A8 = A9S.A8                                # analysis_8/scripts/a8_likelihood.py
A8_SCRIPTS = A9S.A8_SCRIPTS

J9_H5 = RESULTS / "j9_marked.h5"
S9_H5 = RESULTS / "s9_spatial.h5"
OUT_H5 = DIAG / "a9_event_routing.h5"
OUT_JSON = DIAG / "a9_event_routing.json"

# --------------------------------------------------------------------------- #
# The point of record
# --------------------------------------------------------------------------- #
H0_NODES_WANTED = [67.0, 67.5, 67.74, 68.0, 68.5, 69.0, 69.5, 70.0, 70.5, 71.0]
H0_REP = 69.0
H0_STEP = 0.5                       # the spacing of the curvature stencil
I_F = 11                            # f_AGN  = 0.275
I_MU = 41                           # mu_chi_c2 = +0.1075
ULP = 2.0 ** -40                    # one ULP of a double near 4.2e3
FAIL_ABS = 1e-6                     # anything above this is a FAIL, not a note

DP_BIN_EDGES = [-1.0, -0.2, -0.1, -0.05, -0.01, 0.01, 0.05, 0.1, 0.2, 1.0]


def _json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o)} is not JSON serialisable")


# --------------------------------------------------------------------------- #
# The recorded cubes: axes, the two logL_pe lines, the two selection lines
# --------------------------------------------------------------------------- #
def load_cube_axes():
    """Axis membership, asserted BITWISE in both cubes, plus the recorded lines."""
    import h5py

    with h5py.File(J9_H5, "r") as h:
        J_H0 = h["H0_grid"][:]
        f_grid = h["f_grid"][:]
        mu_grid = h["mu_chi_c2_grid"][:]
        J_pe = h["guard/logL_pe"][:, I_F, I_MU]
        J_sel = h["guard/logL_selection"][:, I_F, I_MU]
        J_ll = h["log_likelihood"][:, I_F, I_MU]
        J_rej = h["guard/rejected"][:, I_F, I_MU]
    with h5py.File(S9_H5, "r") as h:
        S_H0 = h["H0_grid"][:]
        S_f_grid = h["f_grid"][:]
        S_pe = h["guard/logL_pe"][:, I_F]
        S_sel = h["guard/logL_selection"][:, I_F]
        S_ll = h["log_likelihood"][:, I_F]
        S_rej = h["guard/rejected"][:, I_F]
        S_mu_held = float(h.attrs["mu_chi_c2_held_at"])

    if not np.array_equal(f_grid, S_f_grid):
        raise SystemExit("[fatal] the two cubes do not share the f axis bitwise")
    if S_mu_held != 0.0:
        raise SystemExit(f"[fatal] S9 holds mu_chi_c2 at {S_mu_held}, not 0.0")
    f_agn = float(f_grid[I_F])
    mu_c2 = float(mu_grid[I_MU])
    if abs(f_agn - 0.275) > 1e-15 or abs(mu_c2 - 0.1075) > 1e-15:
        raise SystemExit(f"[fatal] node (f, mu) = ({f_agn!r}, {mu_c2!r}) is not "
                         f"the registered (0.275, +0.1075) MAP node")

    nodes = []
    print("== H0 node membership (bitwise, both axes) ==")
    for want in H0_NODES_WANTED:
        ij = np.nonzero(J_H0 == want)[0]
        isx = np.nonzero(S_H0 == want)[0]
        if ij.size != 1 or isx.size != 1:
            raise SystemExit(f"[fatal] H0 = {want!r} is not a unique node of both "
                             f"axes (J9 {ij.tolist()}, S9 {isx.tolist()})")
        iJ, iS = int(ij[0]), int(isx[0])
        same_bits = float(J_H0[iJ]).hex() == float(S_H0[iS]).hex() == float(want).hex()
        if not same_bits:
            raise SystemExit(f"[fatal] H0 = {want!r} differs in bits between axes")
        nodes.append({"H0": float(J_H0[iJ]), "H0_hex": float(J_H0[iJ]).hex(),
                      "iJ9": iJ, "iS9": iS})
        print(f"  [OK ] H0 {want:>6.2f}  J9[{iJ:>2d}]  S9[{isx[0]:>3d}]  "
              f"hex {float(want).hex()}  J9 rejected={bool(J_rej[iJ])} "
              f"S9 rejected={bool(S_rej[iS])}")
        if bool(J_rej[iJ]) or bool(S_rej[iS]):
            raise SystemExit(f"[fatal] node H0 = {want!r} is guard-rejected in a "
                             f"cube; its logL_pe is not a usable reference")

    return {"nodes": nodes, "f_agn": f_agn, "mu_chi_c2": mu_c2,
            "J_pe": J_pe, "J_sel": J_sel, "J_ll": J_ll,
            "S_pe": S_pe, "S_sel": S_sel, "S_ll": S_ll,
            "f_grid": f_grid, "mu_grid": mu_grid}


def selection_profile(cube):
    """Item 4: the selection term, from the cubes ALONE.

    d_sel(H0) = logL_selection_J(H0; 0.275, +0.1075) - logL_selection_S(H0; 0.275),
    re-centred at 69.0, with the curvature of each line on the 0.5 stencil.
    """
    nodes = cube["nodes"]
    h0 = np.array([n["H0"] for n in nodes])
    sJ = np.array([cube["J_sel"][n["iJ9"]] for n in nodes])
    sS = np.array([cube["S_sel"][n["iS9"]] for n in nodes])
    peJ = np.array([cube["J_pe"][n["iJ9"]] for n in nodes])
    peS = np.array([cube["S_pe"][n["iS9"]] for n in nodes])
    llJ = np.array([cube["J_ll"][n["iJ9"]] for n in nodes])
    llS = np.array([cube["S_ll"][n["iS9"]] for n in nodes])
    irep = int(np.nonzero(h0 == H0_REP)[0][0])

    def _curv(y):
        """-y''(69.0) on the 0.5 stencil, from the recorded line itself."""
        lo = int(np.nonzero(h0 == H0_REP - H0_STEP)[0][0])
        hi = int(np.nonzero(h0 == H0_REP + H0_STEP)[0][0])
        return float(-(y[hi] - 2.0 * y[irep] + y[lo]) / (H0_STEP ** 2))

    d = sJ - sS
    return {
        "definition": ("logL_selection_J(H0; f=0.275, mu_chi_c2=+0.1075) - "
                       "logL_selection_S(H0; f=0.275), both READ from the cubes"),
        "H0_nodes": h0.tolist(),
        "logL_selection_J": sJ.tolist(),
        "logL_selection_S": sS.tolist(),
        "difference": d.tolist(),
        "difference_recentred_at_69": (d - d[irep]).tolist(),
        "range_of_recentred_difference": float(np.ptp(d - d[irep])),
        "logL_pe_J": peJ.tolist(),
        "logL_pe_S": peS.tolist(),
        "logL_pe_difference_recentred_at_69":
            ((peJ - peS) - (peJ[irep] - peS[irep])).tolist(),
        "total_logL_difference_recentred_at_69":
            ((llJ - llS) - (llJ[irep] - llS[irep])).tolist(),
        "curvature_at_69": {
            "selection_J": _curv(sJ), "selection_S": _curv(sS),
            "selection_difference": _curv(d),
            "pe_J": _curv(peJ), "pe_S": _curv(peS), "pe_difference": _curv(peJ - peS),
            "total_J": _curv(llJ), "total_S": _curv(llS),
            "total_difference": _curv(llJ - llS),
            "note": ("curvature I = -d2/dH0^2 on the 0.5 stencil "
                     "(69.0 +- 0.5); positive = information"),
        },
    }


# --------------------------------------------------------------------------- #
# The per-event pass, one H0 node
# --------------------------------------------------------------------------- #
def per_event_at(cell, ed, h0, f_agn, mu_c2, block):
    """Capture the production operands at (h0, f, mu) and run the per-event pass."""
    L = A8.MU_CHI_C2_LABEL
    coord = cell.coord(H0=float(h0), fcat_2=float(f_agn), **{L: float(mu_c2)})
    t0 = time.time()
    cap = ed.capture_operands(cell, coord)
    evaluate, nEvents, nsamp, log_w = ed.build_evaluator(cap)
    t_cap = time.time() - t0

    # the captured cosmology must carry THIS node's H0, and the weights THIS f
    cosmo = cap["args"][0]
    h0_cap = float(np.asarray(cosmo.H0))
    if abs(h0_cap - float(h0)) > 1e-9:
        raise SystemExit(f"[fatal] captured cosmo.H0 = {h0_cap!r} != {h0!r}")
    bitwise_h0 = float(h0_cap).hex() == float(h0).hex()
    f_cap = float(np.exp(np.asarray(log_w)[1]))
    if abs(f_cap - f_agn) > 1e-12:
        raise SystemExit(f"[fatal] captured mixture weight f_A = {f_cap!r} "
                         f"!= {f_agn!r}")

    keys = ("logZ_prod", "logE_GG", "logE_AG", "logE_GA", "logE_AA")
    cols = {k: np.empty(nEvents) for k in keys}
    n_kept = np.empty(nEvents, dtype=np.int64)
    t0 = time.time()
    for s in range(0, nEvents, block):
        m = min(block, nEvents - s)
        out = evaluate(s * nsamp, m)
        for k in keys:
            cols[k][s:s + m] = np.asarray(out[k])
        n_kept[s:s + m] = np.asarray(out["n_kept"])
    t_pass = time.time() - t0
    print(f"  [node] H0={h0:<6.2f} capture+build {t_cap:6.1f}s  "
          f"per-event pass {t_pass:6.1f}s  nEvents={nEvents} nsamp={nsamp}  "
          f"log_w={np.asarray(log_w)}  cosmo.H0 bitwise={bitwise_h0}")

    del cap, evaluate
    gc.collect()
    return cols, n_kept, int(nEvents), int(nsamp), np.asarray(log_w), t_cap, t_pass


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry_run", action="store_true",
                    help="CPU preflight: imports, environment assertions, cube "
                         "axes and the selection-term profile.  No data load, "
                         "no likelihood build, no GPU.")
    args = ap.parse_args(argv)

    stage = "event_routing"
    t_start = time.time()
    job_id = os.environ.get("SLURM_JOB_ID", "")

    cube = load_cube_axes()
    f_agn, mu_c2 = cube["f_agn"], cube["mu_chi_c2"]
    nodes = cube["nodes"]
    sel = selection_profile(cube)
    print(f"\n== the point ==\n  f_AGN = {f_agn!r} (index {I_F})\n"
          f"  mu_chi_c2 = {mu_c2!r} (index {I_MU})\n"
          f"  H0 nodes = {[n['H0'] for n in nodes]}\n"
          f"  representative H0 = {H0_REP}")
    print("\n== selection term, from the cubes alone ==")
    print("   H0      sel_J        sel_S        diff       diff-@69")
    for k, n in enumerate(nodes):
        print(f"  {n['H0']:>6.2f}  {sel['logL_selection_J'][k]:>11.6f}  "
              f"{sel['logL_selection_S'][k]:>11.6f}  "
              f"{sel['difference'][k]:>10.6f}  "
              f"{sel['difference_recentred_at_69'][k]:>+10.6f}")
    print(f"  curvature of the selection difference at 69.0: "
          f"{sel['curvature_at_69']['selection_difference']:+.6f}")
    print(f"  curvature of the PE difference at 69.0:        "
          f"{sel['curvature_at_69']['pe_difference']:+.6f}")

    if args.dry_run:
        env = A9S._cpu_setup(stage, import_darksirens=True)
        print("\n[dry-run] environment OK; no data loaded, no GPU touched.")
        print(f"[dry-run] darksirens {env['darksirens_sha']}  "
              f"module {env.get('darksirens_module_file')}")
        # exercise the import path of the machinery without building anything
        if str(A8_SCRIPTS) not in sys.path:
            sys.path.insert(0, str(A8_SCRIPTS))
        import event_decomposition as ed
        assert callable(ed.capture_operands) and callable(ed.build_evaluator)
        print("[dry-run] event_decomposition.capture_operands / build_evaluator "
              "imported")
        print(f"[dry-run] pe_event_block = {A8.SETTINGS['pe_event_block']}")
        print(f"[dry-run] would run {len(nodes)} nodes x "
              f"{1000 // int(A8.SETTINGS['pe_event_block'])} blocks")
        return 0

    # ----------------------------------------------------------------- GPU --
    env = A9S._gpu_setup(stage)
    if str(A8_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(A8_SCRIPTS))
    import event_decomposition as ed

    t0 = time.time()
    cell, survey_paths = A9S.build_cell(verbose=True)
    print(f"[routing] build_cell {time.time() - t0:.1f}s; labels={cell.labels}")

    # one live production evaluation at the representative node
    L = A8.MU_CHI_C2_LABEL
    rec = cell.evaluate(H0=H0_REP, fcat_2=f_agn, **{L: mu_c2})
    irep = int(np.nonzero(np.array([n["H0"] for n in nodes]) == H0_REP)[0][0])
    nrep = nodes[irep]
    print(f"[routing] live production logL      = {rec['logL']!r}")
    print(f"[routing] live production logL_pe   = {rec['logL_pe']!r}")
    print(f"[routing] recorded J9 logL          = {cube['J_ll'][nrep['iJ9']]!r}")
    print(f"[routing] recorded J9 logL_pe       = {cube['J_pe'][nrep['iJ9']]!r}")

    block = int(A8.SETTINGS["pe_event_block"])
    nH = len(nodes)
    store = {k: None for k in ("logE_GG", "logE_AG", "logE_GA", "logE_AA",
                               "logZ_prod")}
    n_kept_rep = None
    timings = []
    nEvents = nsamp = None
    log_w = None
    for k, n in enumerate(nodes):
        cols, n_kept, nEv, ns, lw, t_cap, t_pass = per_event_at(
            cell, ed, n["H0"], f_agn, mu_c2, block)
        if nEvents is None:
            nEvents, nsamp, log_w = nEv, ns, lw
            for key in store:
                store[key] = np.empty((nH, nEvents))
        elif (nEv, ns) != (nEvents, nsamp):
            raise SystemExit("[fatal] nEvents/nsamp changed between nodes")
        for key in store:
            store[key][k] = cols[key]
        if n["H0"] == H0_REP:
            n_kept_rep = n_kept
        timings.append({"H0": n["H0"], "capture_build_s": t_cap,
                        "per_event_pass_s": t_pass,
                        "n_samples_kept_min": int(n_kept.min()),
                        "n_samples_kept_median": float(np.median(n_kept)),
                        "n_samples_kept_max": int(n_kept.max())})
        try:
            import jax
            jax.clear_caches()
        except Exception:
            pass
        gc.collect()

    lf_gal, lf_agn = float(log_w[0]), float(log_w[1])
    logE = {k[5:]: store[k] for k in store if k.startswith("logE_")}
    logZ_prod = store["logZ_prod"]

    lnZ_marked = np.logaddexp(lf_gal + logE["GG"], lf_agn + logE["AA"])
    lnZ_spatial = np.logaddexp(lf_gal + logE["GG"], lf_agn + logE["AG"])
    delta_lnL = lnZ_marked - lnZ_spatial

    finite = np.isfinite(lnZ_marked) & np.isfinite(lnZ_spatial)
    n_nonfinite = int((~finite).sum())
    if n_nonfinite:
        print(f"[routing] WARNING: {n_nonfinite} non-finite per-event entries")

    # ------------------------------------------------------- 1. verification --
    def _cmp(name, mine, ref, source):
        d = float(mine - ref)
        return {"quantity": name, "mine": float(mine), "reference": float(ref),
                "source": source, "abs_difference": abs(d),
                "ulp": abs(d) / ULP, "signed_difference": d,
                "pass": bool(abs(d) <= FAIL_ABS)}

    verification = []
    print("\n== verification against the recorded cubes (read, never recomputed) ==")
    print("   H0     quantity                 |d| (abs)      ULP")
    for k, n in enumerate(nodes):
        rows = [
            _cmp(f"sum_i lnZ^J_i (H0={n['H0']})", lnZ_marked[k].sum(),
                 cube["J_pe"][n["iJ9"]],
                 f"results/j9_marked.h5 guard/logL_pe[{n['iJ9']},{I_F},{I_MU}]"),
            _cmp(f"sum_i logZ_prod (H0={n['H0']})", logZ_prod[k].sum(),
                 cube["J_pe"][n["iJ9"]],
                 f"results/j9_marked.h5 guard/logL_pe[{n['iJ9']},{I_F},{I_MU}]"),
            _cmp(f"sum_i lnZ^S_i (H0={n['H0']})", lnZ_spatial[k].sum(),
                 cube["S_pe"][n["iS9"]],
                 f"results/s9_spatial.h5 guard/logL_pe[{n['iS9']},{I_F}]"),
        ]
        for r in rows:
            r["H0"] = n["H0"]
            verification.append(r)
            tag = r["quantity"].split(" (")[0]
            print(f"  {n['H0']:>6.2f}  {tag:<22s} {r['abs_difference']:>12.4e}  "
                  f"{r['ulp']:>8.2f}  {'OK' if r['pass'] else 'FAIL'}")

    live = [
        _cmp("live production logL_pe vs the recorded cube", rec["logL_pe"],
             cube["J_pe"][nrep["iJ9"]],
             f"one live cell.evaluate(H0={H0_REP}, fcat_2={f_agn}, "
             f"mu_chi_c2={mu_c2}) vs j9_marked.h5 guard/logL_pe"),
        _cmp("live production logL_pe vs sum_i lnZ^J_i", rec["logL_pe"],
             float(lnZ_marked[irep].sum()),
             "the same live call vs this script's per-event sum"),
        _cmp("live production logL vs the recorded cube", rec["logL"],
             cube["J_ll"][nrep["iJ9"]],
             "the same live call vs j9_marked.h5 log_likelihood"),
        _cmp("live production logL_selection vs the recorded cube",
             rec["logL_selection"], cube["J_sel"][nrep["iJ9"]],
             "the same live call vs j9_marked.h5 guard/logL_selection"),
    ]
    print("\n== the live production call at the representative node ==")
    for r in live:
        print(f"  {r['quantity']:<48s} |d| {r['abs_difference']:.4e}  "
              f"{r['ulp']:>8.2f} ULP  {'OK' if r['pass'] else 'FAIL'}")

    internal = {
        "max_abs_per_event_lnZ_marked_minus_logZ_prod":
            float(np.max(np.abs(lnZ_marked - logZ_prod))),
        "note": ("logZ_prod is the production per-event value; lnZ^J is the same "
                 "branch sum re-associated from the four E's"),
    }
    print(f"  max per-event |lnZ^J - logZ_prod| = "
          f"{internal['max_abs_per_event_lnZ_marked_minus_logZ_prod']:.3e}")

    n_fail = sum(1 for r in verification + live if not r["pass"])
    worst_ulp = max(r["ulp"] for r in verification + live)
    print(f"\n  {len(verification) + len(live)} checks, {n_fail} FAIL, "
          f"worst {worst_ulp:.2f} ULP")

    # ------------------------------------------------------------ 2. routing --
    log_odds_prior = lf_agn - lf_gal
    bf_spatial = logE["AG"][irep] - logE["GG"][irep]
    bf_intrinsic = logE["AA"][irep] - logE["AG"][irep]
    bf_total = logE["AA"][irep] - logE["GG"][irep]
    P_spatial = 1.0 / (1.0 + np.exp(-(log_odds_prior + bf_spatial)))
    P_marked = 1.0 / (1.0 + np.exp(-(log_odds_prior + bf_total)))
    dP = P_marked - P_spatial

    cross_up = (P_spatial <= 0.5) & (P_marked > 0.5)      # GAL -> AGN
    cross_dn = (P_spatial > 0.5) & (P_marked <= 0.5)      # AGN -> GAL
    pct = [1, 5, 16, 50, 84, 95, 99]
    routing = {
        "H0": H0_REP,
        "f_agn": f_agn, "mu_chi_c2": mu_c2,
        "log_prior_odds_log_fA_over_fG": log_odds_prior,
        "delta_P_percentiles": {str(p): float(np.percentile(dP, p)) for p in pct},
        "delta_P_median": float(np.median(dP)),
        "delta_P_mean": float(np.mean(dP)),
        "abs_delta_P_median": float(np.median(np.abs(dP))),
        "abs_delta_P_rms": float(np.sqrt(np.mean(dP ** 2))),
        "abs_delta_P_max": float(np.max(np.abs(dP))),
        "n_abs_delta_P_gt_0p05": int((np.abs(dP) > 0.05).sum()),
        "n_abs_delta_P_gt_0p1": int((np.abs(dP) > 0.1).sum()),
        "n_abs_delta_P_gt_0p2": int((np.abs(dP) > 0.2).sum()),
        "n_crossing_0p5_total": int(cross_up.sum() + cross_dn.sum()),
        "n_crossing_GAL_to_AGN": int(cross_up.sum()),
        "n_crossing_AGN_to_GAL": int(cross_dn.sum()),
        "sum_P_spatial": float(P_spatial.sum()),
        "sum_P_marked": float(P_marked.sum()),
        "sum_P_marked_minus_spatial": float(P_marked.sum() - P_spatial.sum()),
        "realised_AGN_count": 295,
        "P_spatial_range": [float(P_spatial.min()), float(P_spatial.max())],
        "P_marked_range": [float(P_marked.min()), float(P_marked.max())],
        "n_called_AGN_spatial": int((P_spatial > 0.5).sum()),
        "n_called_AGN_marked": int((P_marked > 0.5).sum()),
        "log_BF_spatial_median": float(np.median(bf_spatial)),
        "log_BF_intrinsic_median": float(np.median(bf_intrinsic)),
        "log_BF_total_median": float(np.median(bf_total)),
    }
    print("\n== routing at the representative node ==")
    print(f"  dP percentiles (1,5,16,50,84,95,99): "
          f"{[round(routing['delta_P_percentiles'][str(p)], 5) for p in pct]}")
    print(f"  median |dP| {routing['abs_delta_P_median']:.5f}   "
          f"rms |dP| {routing['abs_delta_P_rms']:.5f}   "
          f"max |dP| {routing['abs_delta_P_max']:.5f}")
    print(f"  N(|dP|>0.05) {routing['n_abs_delta_P_gt_0p05']}   "
          f"N(|dP|>0.1) {routing['n_abs_delta_P_gt_0p1']}   "
          f"N(|dP|>0.2) {routing['n_abs_delta_P_gt_0p2']}")
    print(f"  crossings of 0.5: GAL->AGN {routing['n_crossing_GAL_to_AGN']}, "
          f"AGN->GAL {routing['n_crossing_AGN_to_GAL']}")
    print(f"  sum P_spatial {routing['sum_P_spatial']:.3f}   "
          f"sum P_marked {routing['sum_P_marked']:.3f}   realised 295")

    # ----------------------------------------- 3. is routing where H0 changes --
    ilo = int(np.nonzero(np.array([n["H0"] for n in nodes])
                         == H0_REP - H0_STEP)[0][0])
    ihi = int(np.nonzero(np.array([n["H0"] for n in nodes])
                         == H0_REP + H0_STEP)[0][0])

    def _curv_ev(Y):
        return -(Y[ihi] - 2.0 * Y[irep] + Y[ilo]) / (H0_STEP ** 2)

    I_marked = _curv_ev(lnZ_marked)
    I_spatial = _curv_ev(lnZ_spatial)
    dI = I_marked - I_spatial

    cube_curv_J = float(-(cube["J_pe"][nodes[ihi]["iJ9"]]
                          - 2.0 * cube["J_pe"][nodes[irep]["iJ9"]]
                          + cube["J_pe"][nodes[ilo]["iJ9"]]) / (H0_STEP ** 2))
    cube_curv_S = float(-(cube["S_pe"][nodes[ihi]["iS9"]]
                          - 2.0 * cube["S_pe"][nodes[irep]["iS9"]]
                          + cube["S_pe"][nodes[ilo]["iS9"]]) / (H0_STEP ** 2))
    curv_identity = [
        _cmp("sum_i I^J_i vs the curvature of the recorded J9 PE line",
             float(I_marked.sum()), cube_curv_J,
             "j9_marked.h5 guard/logL_pe at 68.5 / 69.0 / 69.5"),
        _cmp("sum_i I^S_i vs the curvature of the recorded S9 PE line",
             float(I_spatial.sum()), cube_curv_S,
             "s9_spatial.h5 guard/logL_pe at 68.5 / 69.0 / 69.5"),
    ]
    print("\n== per-event H0 curvature (identity against the cubes) ==")
    for r in curv_identity:
        print(f"  {r['quantity']:<56s} mine {r['mine']:+.9f}  "
              f"ref {r['reference']:+.9f}  |d| {r['abs_difference']:.3e}")

    from scipy.stats import pearsonr, spearmanr
    rng_dlnL = np.ptp(delta_lnL, axis=0)
    absdP, absdI = np.abs(dP), np.abs(dI)

    def _corr(x, y, name):
        rho, p_rho = spearmanr(x, y)
        r, p_r = pearsonr(x, y)
        return {"pair": name, "spearman_rho": float(rho),
                "spearman_p": float(p_rho),
                "pearson_r": float(r), "pearson_p": float(p_r)}

    corrs = [
        _corr(absdP, absdI, "|dP_i| vs |dI_i|"),
        _corr(absdP, rng_dlnL, "|dP_i| vs range_H0[dlnL_i]"),
        _corr(absdP, dI, "|dP_i| vs dI_i (signed)"),
        _corr(dP, dI, "dP_i (signed) vs dI_i (signed)"),
    ]
    print("\n== correlations ==")
    for c in corrs:
        print(f"  {c['pair']:<30s} Spearman {c['spearman_rho']:+.4f} "
              f"(p {c['spearman_p']:.2e})   Pearson {c['pearson_r']:+.4f} "
              f"(p {c['pearson_p']:.2e})")

    sum_abs_dI = float(np.abs(dI).sum())
    sum_dI = float(dI.sum())

    def _frac(mask, label):
        return {"selector": label, "n_events": int(mask.sum()),
                "fraction_of_sum_abs_delta_I":
                    float(np.abs(dI[mask]).sum() / sum_abs_dI) if sum_abs_dI else None,
                "fraction_of_sum_delta_I":
                    float(dI[mask].sum() / sum_dI) if sum_dI else None,
                "sum_delta_I_in_selection": float(dI[mask].sum()),
                "sum_abs_delta_I_in_selection": float(np.abs(dI[mask]).sum()),
                "fraction_of_events": float(mask.mean())}

    fractions = [
        _frac(absdP > 0.05, "|dP_i| > 0.05"),
        _frac(absdP > 0.1, "|dP_i| > 0.1"),
        _frac(absdP > 0.2, "|dP_i| > 0.2"),
        _frac(cross_up | cross_dn, "crosses 0.5"),
        _frac(cross_up, "crosses 0.5 GAL -> AGN"),
        _frac(cross_dn, "crosses 0.5 AGN -> GAL"),
    ]
    print("\n== where the change in H0 information sits ==")
    print(f"  sum_i dI_i = {sum_dI:+.6f}   sum_i |dI_i| = {sum_abs_dI:.6f}")
    for fr in fractions:
        print(f"  {fr['selector']:<26s} N={fr['n_events']:>4d} "
              f"({fr['fraction_of_events'] * 100:5.1f}% of events)  "
              f"|dI| share {fr['fraction_of_sum_abs_delta_I']:.4f}   "
              f"dI share {fr['fraction_of_sum_delta_I']:+.4f}")

    edges = np.array(DP_BIN_EDGES)
    which = np.digitize(dP, edges[1:-1], right=False)
    binned = []
    print("\n== binned in dP ==")
    print("   bin                     N    mean dI        sum dI      mean |dP|")
    for b in range(len(edges) - 1):
        m = which == b
        row = {"lo": float(edges[b]), "hi": float(edges[b + 1]),
               "n_events": int(m.sum()),
               "mean_delta_I": float(dI[m].mean()) if m.any() else None,
               "sum_delta_I": float(dI[m].sum()),
               "sum_abs_delta_I": float(np.abs(dI[m]).sum()),
               "mean_abs_delta_P": float(np.abs(dP[m]).mean()) if m.any() else None,
               "mean_delta_P": float(dP[m].mean()) if m.any() else None}
        binned.append(row)
        print(f"  [{edges[b]:+.2f}, {edges[b + 1]:+.2f})  {row['n_events']:>6d}  "
              f"{(row['mean_delta_I'] if row['mean_delta_I'] is not None else 0.0):+11.6f}  "
              f"{row['sum_delta_I']:+12.6f}  "
              f"{(row['mean_abs_delta_P'] if row['mean_abs_delta_P'] is not None else 0.0):10.5f}")

    peak_J = np.argmax(lnZ_marked, axis=0)
    peak_S = np.argmax(lnZ_spatial, axis=0)
    h0_arr = np.array([n["H0"] for n in nodes])
    moved = peak_J != peak_S
    peak_change = {
        "definition": ("the H0 node at which the event's own lnZ peaks over the "
                       "ten nodes; a coarse per-event 'H0 profile' summary"),
        "n_events_changing_peak_node": int(moved.sum()),
        "fraction_changing": float(moved.mean()),
        "mean_shift_in_H0": float(np.mean(h0_arr[peak_J] - h0_arr[peak_S])),
        "median_shift_in_H0": float(np.median(h0_arr[peak_J] - h0_arr[peak_S])),
        "n_moving_up": int((h0_arr[peak_J] > h0_arr[peak_S]).sum()),
        "n_moving_down": int((h0_arr[peak_J] < h0_arr[peak_S]).sum()),
        "n_at_axis_edge_marked": int(((peak_J == 0) | (peak_J == len(nodes) - 1)).sum()),
        "n_at_axis_edge_spatial":
            int(((peak_S == 0) | (peak_S == len(nodes) - 1)).sum()),
        "peak_movers_are_routed": {
            "median_abs_delta_P_movers":
                float(np.median(np.abs(dP)[moved])) if moved.any() else None,
            "median_abs_delta_P_stayers":
                float(np.median(np.abs(dP)[~moved])) if (~moved).any() else None,
        },
    }
    print(f"\n  events changing their peak H0 node: "
          f"{peak_change['n_events_changing_peak_node']} / {len(dP)} "
          f"(up {peak_change['n_moving_up']}, down {peak_change['n_moving_down']})")

    delta_lnL_profile = {
        "H0_nodes": h0_arr.tolist(),
        "sum_delta_lnL": delta_lnL.sum(axis=1).tolist(),
        "sum_delta_lnL_recentred_at_69":
            (delta_lnL.sum(axis=1) - delta_lnL.sum(axis=1)[irep]).tolist(),
        "median_range_H0_delta_lnL": float(np.median(rng_dlnL)),
        "max_range_H0_delta_lnL": float(np.max(rng_dlnL)),
    }

    # ---------------------------------- interpretation only: the true labels --
    import h5py
    with h5py.File(A8.GW_PATH_MARKED, "r") as h:
        host_type = h["host_type"][:].astype(int)
    if host_type.size != nEvents:
        raise SystemExit("[fatal] host_type length does not match nEvents")
    is_agn = host_type == 1

    def _by_label(x):
        return {"true_GAL_median": float(np.median(x[~is_agn])),
                "true_AGN_median": float(np.median(x[is_agn])),
                "true_GAL_mean": float(np.mean(x[~is_agn])),
                "true_AGN_mean": float(np.mean(x[is_agn]))}

    interpretation = {
        "caveat": ("the true host label is read here for the first time, AFTER "
                   "every inference quantity above was formed; it is never an "
                   "input to any of them"),
        "n_true_AGN": int(is_agn.sum()), "n_true_GAL": int((~is_agn).sum()),
        "delta_P": _by_label(dP),
        "abs_delta_P": _by_label(np.abs(dP)),
        "delta_I": _by_label(dI),
        "P_spatial": _by_label(P_spatial),
        "P_marked": _by_label(P_marked),
        "classification_at_0p5": {
            "spatial": {
                "true_AGN_called_AGN": int((is_agn & (P_spatial > 0.5)).sum()),
                "true_AGN_called_GAL": int((is_agn & ~(P_spatial > 0.5)).sum()),
                "true_GAL_called_AGN": int((~is_agn & (P_spatial > 0.5)).sum()),
                "true_GAL_called_GAL": int((~is_agn & ~(P_spatial > 0.5)).sum()),
                "accuracy": float(((P_spatial > 0.5) == is_agn).mean()),
            },
            "marked": {
                "true_AGN_called_AGN": int((is_agn & (P_marked > 0.5)).sum()),
                "true_AGN_called_GAL": int((is_agn & ~(P_marked > 0.5)).sum()),
                "true_GAL_called_AGN": int((~is_agn & (P_marked > 0.5)).sum()),
                "true_GAL_called_GAL": int((~is_agn & ~(P_marked > 0.5)).sum()),
                "accuracy": float(((P_marked > 0.5) == is_agn).mean()),
            },
        },
        "crossings_by_true_label": {
            "GAL_to_AGN_true_AGN": int((cross_up & is_agn).sum()),
            "GAL_to_AGN_true_GAL": int((cross_up & ~is_agn).sum()),
            "AGN_to_GAL_true_AGN": int((cross_dn & is_agn).sum()),
            "AGN_to_GAL_true_GAL": int((cross_dn & ~is_agn).sum()),
        },
    }
    cl = interpretation["classification_at_0p5"]
    print(f"\n  [interpretation] accuracy at 0.5: spatial "
          f"{cl['spatial']['accuracy']:.4f} -> marked {cl['marked']['accuracy']:.4f}")
    print(f"  [interpretation] crossings by truth: "
          f"{interpretation['crossings_by_true_label']}")

    # ------------------------------------------------------------- write out --
    DIAG.mkdir(parents=True, exist_ok=True)
    with h5py.File(OUT_H5, "w") as h:
        h.create_dataset("event_index", data=np.arange(nEvents))
        h.create_dataset("true_host_type", data=host_type)
        h.create_dataset("H0_nodes", data=h0_arr)
        for k, v in logE.items():
            h.create_dataset("logE_" + k, data=v)
        h.create_dataset("lnZ_marked", data=lnZ_marked)
        h.create_dataset("lnZ_spatial", data=lnZ_spatial)
        h.create_dataset("delta_lnL", data=delta_lnL)
        h.create_dataset("logZ_prod", data=logZ_prod)
        h.create_dataset("I_marked", data=I_marked)
        h.create_dataset("I_spatial", data=I_spatial)
        h.create_dataset("delta_I", data=dI)
        h.create_dataset("P_AGN_spatial", data=P_spatial)
        h.create_dataset("P_AGN_marked", data=P_marked)
        h.create_dataset("delta_P", data=dP)
        h.create_dataset("log_BF_spatial", data=bf_spatial)
        h.create_dataset("log_BF_intrinsic", data=bf_intrinsic)
        h.create_dataset("log_BF_total", data=bf_total)
        h.create_dataset("range_H0_delta_lnL", data=rng_dlnL)
        h.create_dataset("peak_node_marked", data=peak_J)
        h.create_dataset("peak_node_spatial", data=peak_S)
        h.create_dataset("n_samples_kept_at_H0_rep", data=n_kept_rep)
        h.attrs["H0_rep"] = H0_REP
        h.attrs["f_agn"] = f_agn
        h.attrs["mu_chi_c2"] = mu_c2
        h.attrs["log_f_gal"] = lf_gal
        h.attrs["log_f_agn"] = lf_agn
        h.attrs["darksirens_sha"] = env["darksirens_sha"]
        h.attrs["gws_agn_sha"] = env["gws_agn_sha"]
        h.attrs["slurm_job_id"] = job_id
        h.attrs["events_file"] = A8.GW_PATH_MARKED
        h.attrs["host_type_encoding"] = "0 = GAL, 1 = AGN (diagnostic only)"
        h.attrs["H0_curvature_stencil"] = H0_STEP
        h.attrs["arrays"] = ("logE_*/lnZ_*/delta_lnL are (n_H0, nEvents); "
                             "P_*/delta_P/log_BF_*/I_*/delta_I are (nEvents,) "
                             "at H0_rep")
    print(f"\n[routing] wrote {OUT_H5}")

    summary = {
        "analysis": "analysis_9_marked_multitracer_H0_fagn",
        "diagnostic": ("mechanism follow-up part B -- event-level routing and "
                       "the per-event dlnL(H0) profile"),
        "point": {"f_agn": f_agn, "f_index": I_F, "mu_chi_c2": mu_c2,
                  "mu_index": I_MU, "H0_nodes": h0_arr.tolist(),
                  "H0_representative": H0_REP,
                  "log_f_gal": lf_gal, "log_f_agn": lf_agn,
                  "nEvents": nEvents, "nsamp": nsamp,
                  "pe_event_block": block,
                  "node_indices": nodes},
        "model": {
            "marked": "Z_i^J = f_G E_i[GG] + f_A E_i[AA]",
            "spatial": ("Z_i^S = f_G E_i[GG] + f_A E_i[AG]; exact because S9 "
                        "holds dmu_chi = 0 so both branches share the GAL "
                        "population (closure 9.2, <= 2 ULP between the two "
                        "code shapes)"),
            "P_spatial": "sigmoid(log(f_A/f_G) + log E[AG] - log E[GG])",
            "P_marked": "sigmoid(log(f_A/f_G) + log E[AA] - log E[GG])",
            "curvature": ("I_i^X = -[lnZ_i^X(69.5) - 2 lnZ_i^X(69.0) + "
                          "lnZ_i^X(68.5)] / 0.25"),
        },
        "verification": {
            "ulp_definition": {"one_ulp_near_4200": ULP,
                               "fail_threshold_absolute": FAIL_ABS},
            "per_node": verification,
            "live_production_call": live,
            "live_production_record": {
                "logL": rec["logL"], "logL_pe": rec["logL_pe"],
                "logL_selection": rec["logL_selection"],
                "Neff": rec.get("Neff"), "guard_passes": rec.get("guard_passes"),
                "seconds": rec["seconds"]},
            "internal": internal,
            "curvature_identity": curv_identity,
            "n_checks": len(verification) + len(live),
            "n_fail": n_fail,
            "worst_ulp": worst_ulp,
            "n_nonfinite_per_event_entries": n_nonfinite,
        },
        "routing_at_H0_rep": routing,
        "h0_information": {
            "sum_delta_I": sum_dI, "sum_abs_delta_I": sum_abs_dI,
            "sum_I_marked": float(I_marked.sum()),
            "sum_I_spatial": float(I_spatial.sum()),
            "correlations": corrs,
            "fractions": fractions,
            "binned_in_delta_P": binned,
            "peak_node_change": peak_change,
            "delta_lnL_profile": delta_lnL_profile,
        },
        "selection_term": sel,
        "interpretation_only": interpretation,
        "timings": {"per_node": timings,
                    "wall_seconds": float(time.time() - t_start)},
        "environment": env,
        "provenance": A8.provenance(gw_path=A8.GW_PATH_MARKED,
                                    survey_paths=survey_paths),
        "slurm_job_id": job_id,
        "inputs_read": [str(J9_H5), str(S9_H5), A8.GW_PATH_MARKED],
        "outputs": {"h5": str(OUT_H5), "json": str(OUT_JSON)},
        "caveats": [
            "Seed 100 only, one realisation.",
            "One (f_AGN, mu_chi_c2) node: the shared S9/J9 MAP.  Nothing here is "
            "marginalised over the posterior.",
            "The true host label enters no inference quantity; it is read once, "
            "at the end, for the interpretation columns.",
            "The per-event curvature is a three-point second difference on the "
            "0.5 lattice, not an analytic Fisher information.",
        ],
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=_json_default))
    print(f"[routing] wrote {OUT_JSON}")
    print(f"[routing] wall {time.time() - t_start:.1f}s  job {job_id or '-'}")
    if n_fail:
        raise SystemExit(f"[fatal] {n_fail} verification checks exceed {FAIL_ABS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
