#!/usr/bin/env python
"""Analysis 10 -- the per-event GAL/AGN decomposition under FOUR NESTED models.

WHAT THIS COMPUTES
------------------
At ONE hyperparameter point (f_AGN, dmu_chi, dmu_G) with H0 pinned at 67.74,
each of the 1000 seed-100 two-mark events gets its AGN-branch probability under
four models that differ ONLY in which marks the AGN branch's population vector
carries:

    spatial              Lambda_A = Lambda_G                (no mark)
    spatial + spin       Lambda_A = Lambda_G[mu_chi -> dmu_chi]
    spatial + mass       Lambda_A = Lambda_G[mu_G   -> 35 + dmu_G]
    spatial + mass+spin  Lambda_A = Lambda_G[both]          (the production model)

THE DEFINITION, AND WHY IT IS THE PRODUCTION ONE
------------------------------------------------
Analysis 8's ``event_decomposition`` established, and verified against recorded
production values, that the per-tracer kernel
``darksirens/likelihood/core.py :: _log_sample_weight_pop_branches`` forms per PE
sample s and branch k

    log f_k + A_k(s) + B_k(s),
    A_k(s) = log p_k(z_s | pix_{k,s})           SPATIAL factor of tracer k
    B_k(s) = log p_pop(theta_s | Lambda_k)      INTRINSIC factor of branch k
    C(s)   = -log|d(m1src,q,z)/d(m1det,q,dL)| - log prior_wt_s     branch-free

so that with E_i[X|Y] = (1/n) sum_s exp(A_X(s) + B_Y(s) + C(s)) over the SAME
samples and the SAME production mask,

    Z_i = (1 - f) E_i[GG] + f E_i[A|Y]

is EXACTLY the production per-event likelihood of the model whose AGN branch
carries Lambda_Y.  A8's ``capture_operands`` (the factory seam) and
``build_evaluator`` (the per-sample kernel, term for term) are IMPORTED here,
not reimplemented; the only Analysis-10 addition is that the AGN population
vector handed to ``build_evaluator`` is swapped between four values, which is a
change of one argument, not of the arithmetic.

    P_i(AGN | spatial)           = f E[A|G]  / ((1-f) E[GG] + f E[A|G])
    P_i(AGN | spatial+spin)      = f E[A|chi]/ ((1-f) E[GG] + f E[A|chi])
    P_i(AGN | spatial+mass)      = f E[A|M]  / ((1-f) E[GG] + f E[A|M])
    P_i(AGN | spatial+mass+spin) = f E[A|AM] / ((1-f) E[GG] + f E[A|AM])

    log BF_i,spatial = log E[A|G]   - log E[GG]
    log BF_i,spin    = log E[A|chi] - log E[A|G]
    log BF_i,mass    = log E[A|M]   - log E[A|G]
    log BF_i,joint   = log E[A|AM]  - log E[A|G]
    I_i              = log BF_joint - log BF_spin - log BF_mass
                     = log E[A|AM] + log E[A|G] - log E[A|chi] - log E[A|M]

I_i is the two-mark interaction.  It is REPORTED, never assumed away: the marks
are additive in the log Bayes factor only to the extent that |I_i| is small
against the Bayes factors themselves, and both numbers are in the output.

THE INTERMEDIATE POPULATION VECTORS ARE BUILT, THEN CHECKED
-----------------------------------------------------------
The capture yields only the two production vectors, Lambda_G (the pinned base
block) and Lambda_AM (both marks).  Lambda_chi and Lambda_M are built here from
Lambda_G by replacing ONE slot each -- ``mu_chi`` and ``G.mu``, whose indices
are looked up by name, never hard-coded -- and the construction is then verified
by rebuilding Lambda_AM from Lambda_G with BOTH replacements and requiring it to
equal the captured Lambda_AM bitwise, slot by slot.  If the slot map were wrong
that assertion could not pass.

VERIFICATION (mandatory; every nested sum against a LIVE production evaluation)
------------------------------------------------------------------------------
Because log f_k enters the branch additively and the AGN population vector is
the ONLY thing that differs between the four models, each model's per-event sum
has a production number to be checked against at the same point:

    sum_i ln[(1-f) E[GG] + f E[A|AM]]  == logL_pe of evaluate_at(67.74, f, dchi, dG)
    sum_i ln[(1-f) E[GG] + f E[A|chi]] == logL_pe of evaluate_at(67.74, f, dchi, 0)
    sum_i ln[(1-f) E[GG] + f E[A|M]]   == logL_pe of evaluate_at(67.74, f, 0, dG)
    sum_i ln[(1-f) E[GG] + f E[A|G]]   == logL_pe of evaluate_at(67.74, f, 0, 0)
    sum_i log E[GG]                    == logL_pe of evaluate_at(67.74, 0, 0, 0)
                                       == the RECORDED a10_closure.json 15.1 f=0 value

Each of the four is checked twice: once from the re-associated branch sum above,
and once from ``logZ_prod``, the production per-sample mixture the evaluator
rebuilds in place (a fourth pass with Lambda_A = Lambda_G supplies the
spatial-only one).  Residuals are quoted in ULP of the PE term, whose magnitude
is ~1.6e4, so one ULP is 1.82e-12; anything above 1e-6 absolute is a FAIL.

The true host label is read ONLY at the end, for the interpretation columns.  It
is never an input to any inference quantity.

USAGE
-----
    sbatch scripts/submit_a10_event_decomposition_rita.sbatch     # the GPU run
    JAX_PLATFORMS=cpu python scripts/a10_event_decomposition.py --dry_run
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True                    # A8/A9 trees are READ-ONLY
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE.parent
DIAG = ANALYSIS_DIR / "diagnostics"

A9_SCRIPTS = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/"
                  "analysis_9_marked_multitracer_H0_fagn/scripts")
for _p in (str(HERE), str(A9_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import a10_likelihood as A10                      # the A10 builder (steers a8)
import a10_closure as A10C                        # inputs of record + asserts
import a9_scan as A9                              # the rita harness

A8 = A9.A8
A8_SCRIPTS = A9.A8_SCRIPTS
GC_ = A9.GC

H0_POINT = A8.H0_FID                              # 67.74, pinned
FAIL_ABS = 1.0e-6                                 # above this is a FAIL
PCT = (1, 5, 16, 50, 84, 95, 99)
DP_THRESH = (0.05, 0.1, 0.2)

# The registered provisional point (GATES.md): f node 14, dmu_chi node 41,
# dmu_G node 15 of the three registered axes.
DEFAULT_POINT = {"f": 0.350, "dmu_chi": 0.1075, "dmu_G": 5.0}
DEFAULT_TAG = "prov_f0p350_mu0p1075_G5"


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


def _write(path, obj):
    path = Path(path)
    if ANALYSIS_DIR not in path.parents:
        raise RuntimeError(f"[fatal] refusing to write outside {ANALYSIS_DIR}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default))
    tmp.replace(path)
    print(f"wrote {path}")
    return path


# --------------------------------------------------------------------------- #
# The point: snapped to the registered axes, bitwise
# --------------------------------------------------------------------------- #
def resolve_point(f, dmu_chi, dmu_G, tol=1e-9):
    """Snap each coordinate to its registered axis node when it is one.

    Returns the BITWISE node value, so a sum compared against a recorded cell is
    compared at the same coordinate the scan evaluated.
    """
    axes = (("f_agn", f, A10C.F_GRID, "linspace(0, 1, 41)"),
            ("dmu_chi", dmu_chi, A10C.MU_GRID, "linspace(-0.20, 0.25, 61)"),
            ("dmu_G", dmu_G, A10C.MG_GRID, "linspace(-10, 10, 21)"))
    out = {}
    for name, want, grid, spec in axes:
        grid = np.asarray(grid, dtype=float)
        k = int(np.argmin(np.abs(grid - float(want))))
        on_node = bool(abs(grid[k] - float(want)) <= tol)
        val = float(grid[k]) if on_node else float(want)
        out[name] = {"value": val, "hex": float(val).hex(), "axis": spec,
                     "is_grid_node": on_node,
                     "node_index": (k if on_node else None),
                     "requested": float(want),
                     "snap_residual": float(abs(grid[k] - float(want)))}
        print(f"  [{'OK ' if on_node else '   '}] {name:<8s} = {val!r} "
              f"({'node %d of %s' % (k, spec) if on_node else 'NOT a grid node'})")
    return out


# --------------------------------------------------------------------------- #
# The capture, on build_a10 rather than a8.build
# --------------------------------------------------------------------------- #
def capture_a10(cell, coord):
    """A8's ``capture_operands`` seam, driven through A10's ``_steer``.

    ``event_decomposition.capture_operands`` rebuilds the cell by calling
    ``a8.build(..., 'new', ...)`` with ``a8.GW_PATH_MARKED``.  Wrapping that call
    in ``a10_likelihood._steer`` is EXACTLY what ``build_a10`` does -- the same
    context manager, the same two per-catalog coordinates, the same cell class --
    and ``a8.GW_PATH_MARKED`` is redirected to the two-mark events file for the
    duration.  Both are restored afterwards; nothing in the A8 tree is written.
    """
    if str(A8_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(A8_SCRIPTS))
    import event_decomposition as ed

    saved_gw = A8.GW_PATH_MARKED
    A8.GW_PATH_MARKED = A10C.GW_PATH_A10
    try:
        with A10._steer(A10.PER_CATALOG_A10, A10.A10LikelihoodCell):
            cap = ed.capture_operands(cell, coord)
    finally:
        A8.GW_PATH_MARKED = saved_gw
    if str(A8.GW_PATH_MARKED) != saved_gw:
        raise SystemExit("[fatal] a8.GW_PATH_MARKED not restored")
    return ed, cap


def cap_with_agn_population(cap, agn_pop):
    """The captured operand bundle with ONE argument changed: Lambda_AGN."""
    kw = dict(cap["kwargs"])
    kw["mixture_pop_params"] = (agn_pop,)
    return {"args": cap["args"], "kwargs": kw}


def per_event_pass(ed, cap_v, block, label):
    """One full per-event pass; returns the five columns plus n_kept."""
    t0 = time.time()
    evaluate, nEvents, nsamp, log_w = ed.build_evaluator(cap_v)
    t_build = time.time() - t0
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
    print(f"  [pass {label:<4s}] build {t_build:6.1f}s  per-event {t_pass:6.1f}s  "
          f"nEvents={nEvents} nsamp={nsamp} log_w={np.asarray(log_w)}")
    sys.stdout.flush()
    del evaluate
    try:
        import jax
        jax.clear_caches()
    except Exception:
        pass
    gc.collect()
    return cols, n_kept, int(nEvents), int(nsamp), np.asarray(log_w), t_build, t_pass


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #
def _pct(x):
    return {str(p): float(np.percentile(x, p)) for p in PCT}


def _auc(score, positive):
    """Rank AUC of ``score`` for the boolean ``positive``, ties averaged."""
    from scipy.stats import rankdata
    r = rankdata(np.asarray(score, dtype=float))
    n1 = int(positive.sum())
    n0 = int((~positive).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[positive].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _corr(x, y):
    from scipy.stats import pearsonr, spearmanr
    r, pr = pearsonr(x, y)
    rho, prho = spearmanr(x, y)
    return {"pearson_r": float(r), "pearson_p": float(pr),
            "spearman_rho": float(rho), "spearman_p": float(prho)}


def dp_block(name, dP, P_ref, P_new, realised_agn):
    up = (P_ref <= 0.5) & (P_new > 0.5)
    dn = (P_ref > 0.5) & (P_new <= 0.5)
    return {
        "name": name,
        "percentiles": _pct(dP),
        "median": float(np.median(dP)),
        "mean": float(np.mean(dP)),
        "median_abs": float(np.median(np.abs(dP))),
        "rms_abs": float(np.sqrt(np.mean(dP ** 2))),
        "max_abs": float(np.max(np.abs(dP))),
        "n_abs_gt": {f"{t}": int((np.abs(dP) > t).sum()) for t in DP_THRESH},
        "n_crossing_0p5_total": int(up.sum() + dn.sum()),
        "n_crossing_GAL_to_AGN": int(up.sum()),
        "n_crossing_AGN_to_GAL": int(dn.sum()),
        "sum_P_reference": float(P_ref.sum()),
        "sum_P_model": float(P_new.sum()),
        "sum_P_model_minus_reference": float(P_new.sum() - P_ref.sum()),
        "realised_AGN_count": int(realised_agn),
        "n_called_AGN_at_0p5": int((P_new > 0.5).sum()),
        "P_range": [float(P_new.min()), float(P_new.max())],
    }


# =========================================================================== #
# Driver
# =========================================================================== #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--f", type=float, default=DEFAULT_POINT["f"])
    ap.add_argument("--dmu_chi", type=float, default=DEFAULT_POINT["dmu_chi"])
    ap.add_argument("--dmu_G", type=float, default=DEFAULT_POINT["dmu_G"])
    ap.add_argument("--tag", type=str, default=DEFAULT_TAG)
    ap.add_argument("--dry_run", action="store_true",
                    help="CPU preflight: imports, the point, the slot map and "
                         "the population-vector construction on the FIDUCIAL "
                         "vector.  No data load, no likelihood, no GPU.")
    ap.add_argument("--no_md5", action="store_true", help="skip the events md5")
    args = ap.parse_args(argv)

    t_start = time.time()
    job_id = os.environ.get("SLURM_JOB_ID", "")
    tag = args.tag
    out_h5 = DIAG / f"a10_event_decomposition_{tag}.h5"
    out_json = DIAG / f"a10_event_decomposition_{tag}.json"

    print("== the point (snapped to the registered axes) ==")
    point = resolve_point(args.f, args.dmu_chi, args.dmu_G)
    f_agn = point["f_agn"]["value"]
    dmu_chi = point["dmu_chi"]["value"]
    dmu_G = point["dmu_G"]["value"]
    mu_chi_c2 = A10.dmuchi_to_label(dmu_chi)          # = dmu_chi (fiducial 0)
    mu_G_c2 = A10.dmuG_to_label(dmu_G)                # = 35 + dmu_G
    print(f"  H0 = {H0_POINT} (pinned)   mu_chi_c2 = {mu_chi_c2!r}   "
          f"mu_G_c2 = {mu_G_c2!r}   tag = {tag}")

    # ---- the slot map, by NAME, and the two intermediate vectors ----------- #
    if args.dry_run:
        env = A9._cpu_setup("event_decomposition", import_darksirens=True)
    else:
        env = A9._gpu_setup("event_decomposition")
    env["a10_inputs"] = A10C.assert_a10_inputs(with_md5=not args.no_md5)

    i_G, lab_G, fid_G = A10.fiducial_slot("G.mu")
    i_chi, lab_chi, fid_chi = A10.fiducial_slot("mu_chi")
    labels_pop, plain_pop, fid_vec = A8.population_labels_and_fiducial()
    slot_map = {
        "G.mu": {"index": i_G, "label": lab_G, "fiducial": fid_G,
                 "set_to": mu_G_c2},
        "mu_chi": {"index": i_chi, "label": lab_chi, "fiducial": fid_chi,
                   "set_to": mu_chi_c2},
        "n_slots": len(fid_vec),
        "plain_names": list(plain_pop),
        "fiducial_vector": [float(v) for v in fid_vec],
    }
    print(f"\n== the population slot map (looked up by name) ==\n"
          f"  G.mu   -> slot {i_G} ({lab_G!r}), fiducial {fid_G} -> {mu_G_c2}\n"
          f"  mu_chi -> slot {i_chi} ({lab_chi!r}), fiducial {fid_chi} -> {mu_chi_c2}")
    if i_G != 6 or i_chi != 9:
        raise SystemExit(f"[fatal] the registered slot map is G.mu = 6, mu_chi = 9; "
                         f"this build gives {i_G} and {i_chi}")

    if args.dry_run:
        lam = np.asarray(fid_vec, dtype=float)
        lam_both = lam.copy()
        lam_both[i_chi] = mu_chi_c2
        lam_both[i_G] = mu_G_c2
        lam_chi = lam.copy(); lam_chi[i_chi] = mu_chi_c2
        lam_M = lam.copy(); lam_M[i_G] = mu_G_c2
        print("\n[dry-run] Lambda_G   =", lam.tolist())
        print("[dry-run] Lambda_chi =", lam_chi.tolist())
        print("[dry-run] Lambda_M   =", lam_M.tolist())
        print("[dry-run] Lambda_AM  =", lam_both.tolist())
        assert np.array_equal(lam_chi != lam, np.arange(len(lam)) == i_chi)
        assert np.array_equal(lam_M != lam, np.arange(len(lam)) == i_G)
        if str(A8_SCRIPTS) not in sys.path:
            sys.path.insert(0, str(A8_SCRIPTS))
        import event_decomposition as ed
        assert callable(ed.capture_operands) and callable(ed.build_evaluator)
        print("[dry-run] event_decomposition.capture_operands / build_evaluator OK")
        print(f"[dry-run] pe_event_block = {A8.SETTINGS['pe_event_block']}, "
              f"would run 4 passes x {1000 // int(A8.SETTINGS['pe_event_block'])} blocks")
        print(f"[dry-run] outputs would be {out_h5} and {out_json}")
        print("[dry-run] environment OK; no data loaded, no GPU touched.")
        return 0

    # ----------------------------------------------------------------- GPU -- #
    import h5py

    t0 = time.time()
    surveys = [A8.SURVEY_GAL, A8.SURVEY_AGN]
    cell = A10.build_a10("A10DECOMP", surveys, verbose=True,
                         gw_path=A10C.GW_PATH_A10)
    build_s = time.time() - t0
    A10C.assert_cell(cell, A10.EXPECTED_LABELS_A10, "A10 two-mark")
    print(f"[decomp] build {build_s:.1f}s")

    # ---- the five live production evaluations ------------------------------ #
    print("\n== live production evaluations (the reference for every sum) ==")
    live = {}
    for key, (ff, dc, dg) in {
        "joint":   (f_agn, dmu_chi, dmu_G),
        "spin":    (f_agn, dmu_chi, 0.0),
        "mass":    (f_agn, 0.0, dmu_G),
        "spatial": (f_agn, 0.0, 0.0),
        "f0":      (0.0, 0.0, 0.0),
        "f0_both_marks": (0.0, dmu_chi, dmu_G),
    }.items():
        r = cell.evaluate_at(H0=H0_POINT, fcat_2=ff, dmu_chi=dc, dmu_G=dg)
        live[key] = r
        print(f"  {key:<14s} f={ff:<6.4g} dchi={dc:+.4f} dG={dg:+5.2f}  "
              f"logL={r['logL']!r}  logL_pe={r['logL_pe']!r}  "
              f"Neff={r.get('Neff')!r}  {r['seconds']:.2f}s")
        sys.stdout.flush()

    # ---- the recorded closure value (read, never recomputed) --------------- #
    recorded = {}
    cl_path = DIAG / "a10_closure.json"
    if cl_path.exists():
        cl = json.loads(cl_path.read_text())
        for row in cl["checks"]["15_1_both_marks_zero"]["per_f"]:
            if float(row["f_agn"]) == 0.0:
                recorded["closure_15_1_f0_logL_pe"] = float(row["two_mark"]["logL_pe"])
                recorded["closure_15_1_f0_source"] = (
                    "diagnostics/a10_closure.json checks.15_1_both_marks_zero"
                    ".per_f[f=0].two_mark.logL_pe")
            if abs(float(row["f_agn"]) - f_agn) <= 1e-12:
                recorded["closure_15_1_f_logL_pe"] = float(row["two_mark"]["logL_pe"])
                recorded["closure_15_1_f_source"] = (
                    f"diagnostics/a10_closure.json 15.1 per_f[f={f_agn}] "
                    f"two_mark.logL_pe (spatial-only: both marks zero)")
    print(f"\n== recorded references found: {sorted(recorded)} ==")

    # ---- the capture -------------------------------------------------------- #
    coord = cell.coord(fcat_2=f_agn, **{A10.MU_CHI_C2_LABEL: mu_chi_c2,
                                        A10.MU_G_C2_LABEL: mu_G_c2})
    t0 = time.time()
    ed, cap = capture_a10(cell, coord)
    print(f"\n[decomp] captured production operands in {time.time() - t0:.1f}s")

    import jax.numpy as jnp
    cosmo = cap["args"][0]
    h0_cap = float(np.asarray(cosmo.H0))
    if float(h0_cap).hex() != float(H0_POINT).hex():
        raise SystemExit(f"[fatal] captured cosmo.H0 = {h0_cap!r} != {H0_POINT!r}")
    if int(cap["kwargs"]["n_catalogs"]) != 2:
        raise SystemExit("[fatal] the capture is not K = 2")
    mix = tuple(cap["kwargs"]["mixture_pop_params"])
    if len(mix) != 1:
        raise SystemExit(f"[fatal] expected ONE mixture population vector, got {len(mix)}")

    lam_G = np.asarray(cap["args"][2], dtype=float)
    lam_AM = np.asarray(mix[0], dtype=float)
    lam_chi = lam_G.copy(); lam_chi[i_chi] = mu_chi_c2
    lam_M = lam_G.copy(); lam_M[i_G] = mu_G_c2
    lam_both = lam_G.copy(); lam_both[i_chi] = mu_chi_c2; lam_both[i_G] = mu_G_c2

    pop_check = {
        "Lambda_G": lam_G.tolist(),
        "Lambda_AM_captured": lam_AM.tolist(),
        "Lambda_chi_built": lam_chi.tolist(),
        "Lambda_M_built": lam_M.tolist(),
        "Lambda_AM_rebuilt_from_Lambda_G": lam_both.tolist(),
        "rebuilt_equals_captured_bitwise": bool(np.array_equal(lam_both, lam_AM)),
        "max_abs_difference": float(np.max(np.abs(lam_both - lam_AM))),
        "slots_that_differ_G_vs_AM": [int(k) for k in
                                      np.nonzero(lam_G != lam_AM)[0]],
        "Lambda_G_equals_pinned_fiducial_bitwise":
            bool(np.array_equal(lam_G, np.asarray(fid_vec, dtype=float))),
        "Lambda_G_minus_pinned_fiducial_max_abs":
            float(np.max(np.abs(lam_G - np.asarray(fid_vec, dtype=float)))),
        "tolerance": 1.0e-9,
    }
    print("\n== the population vectors ==")
    print(f"  Lambda_G  = {lam_G.tolist()}")
    print(f"  Lambda_AM = {lam_AM.tolist()}")
    print(f"  slots differing between them: {pop_check['slots_that_differ_G_vs_AM']} "
          f"(expected [{i_G}, {i_chi}])")
    print(f"  Lambda_G == pinned fiducial: bitwise "
          f"{pop_check['Lambda_G_equals_pinned_fiducial_bitwise']}, max |d| "
          f"{pop_check['Lambda_G_minus_pinned_fiducial_max_abs']:.3e}")
    print(f"  rebuilt Lambda_AM: bitwise "
          f"{pop_check['rebuilt_equals_captured_bitwise']}, max |d| "
          f"{pop_check['max_abs_difference']:.3e}")
    if pop_check["max_abs_difference"] > pop_check["tolerance"]:
        raise SystemExit(
            f"[fatal] Lambda_G with slot {i_chi} -> {mu_chi_c2} and slot {i_G} -> "
            f"{mu_G_c2} is NOT the captured AGN vector "
            f"(max |d| = {pop_check['max_abs_difference']:.3e}); the slot map is wrong")
    if pop_check["Lambda_G_minus_pinned_fiducial_max_abs"] > pop_check["tolerance"]:
        raise SystemExit("[fatal] the GAL branch vector is not the pinned fiducial")
    if sorted(pop_check["slots_that_differ_G_vs_AM"]) != sorted([i_G, i_chi]):
        raise SystemExit("[fatal] the AGN vector differs from the GAL vector in "
                         "slots other than the two marks")

    # ---- the four per-event passes ----------------------------------------- #
    block = int(A8.SETTINGS["pe_event_block"])
    variants = (("AM", jnp.asarray(lam_AM)), ("chi", jnp.asarray(lam_chi)),
                ("M", jnp.asarray(lam_M)), ("G", jnp.asarray(lam_G)))
    passes, timings = {}, []
    mask_stable = True
    nEvents = nsamp = None
    log_w = None
    n_kept_ref = None
    print(f"\n== four per-event passes (block {block}) ==")
    for name, vec in variants:
        cols, n_kept, nEv, ns, lw, t_b, t_p = per_event_pass(
            ed, cap_with_agn_population(cap, vec), block, name)
        if nEvents is None:
            nEvents, nsamp, log_w, n_kept_ref = nEv, ns, lw, n_kept
        elif (nEv, ns) != (nEvents, nsamp):
            raise SystemExit("[fatal] nEvents/nsamp changed between passes")
        elif not np.array_equal(n_kept, n_kept_ref):
            mask_stable = False
            print(f"  [WARN] the production mask changed in pass {name}: the four "
                  f"E's would not share a mask.  The verification below is the gate.")
        passes[name] = cols
        timings.append({"pass": name, "build_seconds": t_b,
                        "per_event_seconds": t_p,
                        "n_samples_kept_min": int(n_kept.min()),
                        "n_samples_kept_median": float(np.median(n_kept)),
                        "n_samples_kept_max": int(n_kept.max())})

    lf_gal, lf_agn = float(log_w[0]), float(log_w[1])
    f_from_weights = float(np.exp(lf_agn))
    if abs(f_from_weights - f_agn) > 1e-12:
        raise SystemExit(f"[fatal] captured mixture weight {f_from_weights!r} != "
                         f"{f_agn!r}")

    # ---- the five E's, and the cross-pass identities ------------------------ #
    logE_GG = passes["AM"]["logE_GG"]
    logE_A_G = passes["AM"]["logE_AG"]
    logE_A_AM = passes["AM"]["logE_AA"]
    logE_G_AM = passes["AM"]["logE_GA"]
    logE_A_chi = passes["chi"]["logE_AA"]
    logE_A_M = passes["M"]["logE_AA"]

    def _same(x, y):
        return {"bitwise": bool(np.array_equal(x, y)),
                "max_abs_difference": float(np.max(np.abs(x - y)))}

    internal = {
        "production_mask_identical_across_passes": bool(mask_stable),
        "logE_GG_across_passes": {
            k: _same(passes[k]["logE_GG"], logE_GG) for k in passes},
        "logE_A_G_across_passes": {
            k: _same(passes[k]["logE_AG"], logE_A_G) for k in passes},
        "pass_G_AA_equals_AG": _same(passes["G"]["logE_AA"], logE_A_G),
        "note": ("E[GG] and E[A|G] must not depend on the AGN population vector; "
                 "with Lambda_A = Lambda_G the pass-G E[A|A] IS E[A|G]"),
        "n_nonfinite_logE": {
            "GG": int((~np.isfinite(logE_GG)).sum()),
            "A_G": int((~np.isfinite(logE_A_G)).sum()),
            "A_chi": int((~np.isfinite(logE_A_chi)).sum()),
            "A_M": int((~np.isfinite(logE_A_M)).sum()),
            "A_AM": int((~np.isfinite(logE_A_AM)).sum()),
        },
    }
    print("\n== cross-pass identities (the GAL branch and the hybrid are shared) ==")
    for name, blk in (("logE_GG", internal["logE_GG_across_passes"]),
                      ("logE_A_G", internal["logE_A_G_across_passes"])):
        for k, v in blk.items():
            print(f"  {name:<9s} pass {k:<4s} bitwise {str(v['bitwise']):<5s} "
                  f"max|d| {v['max_abs_difference']:.3e}")
            if v["max_abs_difference"] > FAIL_ABS:
                raise SystemExit(f"[fatal] {name} differs in pass {k} by "
                                 f"{v['max_abs_difference']:.3e}")
    v = internal["pass_G_AA_equals_AG"]
    print(f"  pass-G E[A|A] == E[A|G]: bitwise {v['bitwise']}, "
          f"max|d| {v['max_abs_difference']:.3e}")
    if v["max_abs_difference"] > FAIL_ABS:
        raise SystemExit("[fatal] with Lambda_A = Lambda_G, E[A|A] must equal E[A|G]")
    if not mask_stable:
        print("  [WARN] the production mask was NOT identical across passes")

    # ---- the four nested per-event likelihoods ----------------------------- #
    lnZ = {
        "joint": np.logaddexp(lf_gal + logE_GG, lf_agn + logE_A_AM),
        "spin": np.logaddexp(lf_gal + logE_GG, lf_agn + logE_A_chi),
        "mass": np.logaddexp(lf_gal + logE_GG, lf_agn + logE_A_M),
        "spatial": np.logaddexp(lf_gal + logE_GG, lf_agn + logE_A_G),
    }
    lnZ_prod = {"joint": passes["AM"]["logZ_prod"], "spin": passes["chi"]["logZ_prod"],
                "mass": passes["M"]["logZ_prod"], "spatial": passes["G"]["logZ_prod"]}

    # ---- verification ------------------------------------------------------- #
    def _cmp(name, mine, ref, source):
        c = A10C._cmp(float(mine), float(ref))
        c.update({"quantity": name, "source": source,
                  "pass": bool(c["abs_diff"] <= FAIL_ABS)})
        return c

    checks = []
    for key, label in (("joint", "spatial+mass+spin"), ("spin", "spatial+spin"),
                       ("mass", "spatial+mass"), ("spatial", "spatial")):
        ref = live[key]["logL_pe"]
        src = (f"live cell.evaluate_at(67.74, f={live[key]['coord']['fcat_2']}, "
               f"dmu_chi={live[key]['dmu_chi']}, dmu_G={live[key]['dmu_G']}) "
               f"logL - logL_selection")
        checks.append(_cmp(f"sum_i ln Z_i [{label}] (re-associated)",
                           lnZ[key].sum(), ref, src))
        checks.append(_cmp(f"sum_i logZ_prod [{label}] (production kernel)",
                           lnZ_prod[key].sum(), ref, src))
    checks.append(_cmp("sum_i log E_i[GG]", logE_GG.sum(), live["f0"]["logL_pe"],
                       "live cell.evaluate_at(67.74, f=0, 0, 0) logL - logL_selection"))
    checks.append(_cmp("sum_i log E_i[GG] (f=0 with BOTH marks set)",
                       logE_GG.sum(), live["f0_both_marks"]["logL_pe"],
                       "live cell.evaluate_at(67.74, f=0, dmu_chi, dmu_G): at f=0 "
                       "the marks must be inert"))
    if "closure_15_1_f0_logL_pe" in recorded:
        checks.append(_cmp("sum_i log E_i[GG] vs the RECORDED closure 15.1 f=0 cell",
                           logE_GG.sum(), recorded["closure_15_1_f0_logL_pe"],
                           recorded["closure_15_1_f0_source"]))
    if "closure_15_1_f_logL_pe" in recorded:
        checks.append(_cmp("sum_i ln Z_i [spatial] vs the RECORDED closure 15.1 cell",
                           lnZ["spatial"].sum(), recorded["closure_15_1_f_logL_pe"],
                           recorded["closure_15_1_f_source"]))

    pe_scale = float(abs(live["joint"]["logL_pe"]))
    one_ulp = float(np.spacing(pe_scale))
    print(f"\n== verification (PE term ~ {pe_scale:.4f}; 1 ULP = {one_ulp:.4e}) ==")
    print(f"  {'quantity':<58s} {'|d| abs':>12s} {'ULP':>10s}")
    for c in checks:
        print(f"  {c['quantity']:<58s} {c['abs_diff']:>12.4e} "
              f"{c['diff_in_ulp']:>10.2f}  {'OK' if c['pass'] else 'FAIL'}")
    per_event_reassoc = {
        k: float(np.max(np.abs(lnZ[k] - lnZ_prod[k]))) for k in lnZ}
    print("  max per-event |re-associated - production| : " +
          ", ".join(f"{k} {v:.3e}" for k, v in per_event_reassoc.items()))
    n_fail = sum(1 for c in checks if not c["pass"])
    worst_ulp = max((abs(c["diff_in_ulp"]) for c in checks
                     if np.isfinite(c["diff_in_ulp"])), default=float("nan"))
    print(f"  {len(checks)} checks, {n_fail} FAIL, worst {worst_ulp:.2f} ULP")

    # ---- the branch probabilities and the log Bayes factors ---------------- #
    log_prior_odds = lf_agn - lf_gal
    bf_spatial = logE_A_G - logE_GG
    bf_spin = logE_A_chi - logE_A_G
    bf_mass = logE_A_M - logE_A_G
    bf_joint = logE_A_AM - logE_A_G
    interaction = bf_joint - bf_spin - bf_mass

    def _P(logE_A):
        return 1.0 / (1.0 + np.exp(-(log_prior_odds + (logE_A - logE_GG))))

    P = {"spatial": _P(logE_A_G), "spin": _P(logE_A_chi),
         "mass": _P(logE_A_M), "joint": _P(logE_A_AM)}
    dP = {"spin": P["spin"] - P["spatial"], "mass": P["mass"] - P["spatial"],
          "joint": P["joint"] - P["spatial"]}

    FIN = (np.isfinite(logE_GG) & np.isfinite(logE_A_G) & np.isfinite(logE_A_chi)
           & np.isfinite(logE_A_M) & np.isfinite(logE_A_AM))
    n_excluded = int((~FIN).sum())
    n_stats = int(FIN.sum())
    if n_excluded:
        print(f"\n[WARN] {n_excluded} events carry a non-finite E; every statistic "
              f"below is formed on the remaining {n_stats}.  The per-event arrays "
              f"on disk keep all {nEvents}.")

    # ---- the per-event arrays go to disk BEFORE any statistic is formed, so a
    # ---- failure in the summary cannot cost the GPU pass ------------------- #
    DIAG.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_h5, "w") as h:
        h.create_dataset("event_index", data=np.arange(nEvents))
        h.create_dataset("logE_GG", data=logE_GG)
        h.create_dataset("logE_A_G", data=logE_A_G)
        h.create_dataset("logE_A_chi", data=logE_A_chi)
        h.create_dataset("logE_A_M", data=logE_A_M)
        h.create_dataset("logE_A_AM", data=logE_A_AM)
        h.create_dataset("logE_G_AM", data=logE_G_AM)
        h.create_dataset("logE_G_chi", data=passes["chi"]["logE_GA"])
        h.create_dataset("logE_G_M", data=passes["M"]["logE_GA"])
        for k in ("spatial", "spin", "mass", "joint"):
            h.create_dataset(f"P_AGN_{k}", data=P[k])
            h.create_dataset(f"lnZ_{k}", data=lnZ[k])
            h.create_dataset(f"lnZ_prod_{k}", data=lnZ_prod[k])
        for k in ("spin", "mass", "joint"):
            h.create_dataset(f"delta_P_{k}", data=dP[k])
        h.create_dataset("log_BF_spatial", data=bf_spatial)
        h.create_dataset("log_BF_spin", data=bf_spin)
        h.create_dataset("log_BF_mass", data=bf_mass)
        h.create_dataset("log_BF_joint", data=bf_joint)
        h.create_dataset("interaction", data=interaction)
        h.create_dataset("n_samples_kept", data=n_kept_ref)
        h.create_dataset("Lambda_G", data=lam_G)
        h.create_dataset("Lambda_chi", data=lam_chi)
        h.create_dataset("Lambda_M", data=lam_M)
        h.create_dataset("Lambda_AM", data=lam_AM)
        h.attrs["tag"] = tag
        h.attrs["H0"] = H0_POINT
        h.attrs["f_agn"] = f_agn
        h.attrs["dmu_chi"] = dmu_chi
        h.attrs["dmu_G"] = dmu_G
        h.attrs["mu_chi_c2"] = mu_chi_c2
        h.attrs["mu_G_c2"] = mu_G_c2
        h.attrs["f_agn_hex"] = float(f_agn).hex()
        h.attrs["dmu_chi_hex"] = float(dmu_chi).hex()
        h.attrs["dmu_G_hex"] = float(dmu_G).hex()
        h.attrs["log_f_gal"] = lf_gal
        h.attrs["log_f_agn"] = lf_agn
        h.attrs["log_prior_odds"] = log_prior_odds
        h.attrs["events_file"] = A10C.GW_PATH_A10
        h.attrs["events_md5"] = env["a10_inputs"].get("events_md5", "")
        h.attrs["darksirens_sha"] = env["darksirens_sha"]
        h.attrs["gws_agn_sha"] = env["gws_agn_sha"]
        h.attrs["slurm_job_id"] = job_id
        h.attrs["models"] = ("spatial | spatial+spin | spatial+mass | "
                             "spatial+mass+spin; the AGN population vector is the "
                             "only difference")
    print(f"[decomp] wrote the per-event arrays to {out_h5}")

    if n_excluded:
        P = {k: v[FIN] for k, v in P.items()}
        dP = {k: v[FIN] for k, v in dP.items()}
        bf_spatial, bf_spin = bf_spatial[FIN], bf_spin[FIN]
        bf_mass, bf_joint = bf_mass[FIN], bf_joint[FIN]
        interaction = interaction[FIN]

    realised_agn = 357
    stats = {k: dp_block(k, dP[k], P["spatial"], P[k], realised_agn)
             for k in ("spin", "mass", "joint")}
    print("\n== dP statistics ==")
    for k in ("spin", "mass", "joint"):
        s = stats[k]
        print(f"  dP^{k:<6s} pct(1,5,16,50,84,95,99) = "
              f"{[round(s['percentiles'][str(p)], 5) for p in PCT]}")
        print(f"            median|dP| {s['median_abs']:.5f}  rms|dP| "
              f"{s['rms_abs']:.5f}  max|dP| {s['max_abs']:.5f}  "
              f"N>0.05/0.1/0.2 {s['n_abs_gt']['0.05']}/{s['n_abs_gt']['0.1']}/"
              f"{s['n_abs_gt']['0.2']}  cross up/down "
              f"{s['n_crossing_GAL_to_AGN']}/{s['n_crossing_AGN_to_GAL']}  "
              f"sum P {s['sum_P_model']:.2f} (spatial {s['sum_P_reference']:.2f}, "
              f"realised {realised_agn})")

    # ---- complementarity ---------------------------------------------------- #
    a = np.abs(dP["mass"]) > 0.1
    b = np.abs(dP["spin"]) > 0.1
    both = (np.abs(dP["mass"]) > 0.05) & (np.abs(dP["spin"]) > 0.05)
    sm, ss = np.sign(dP["mass"]), np.sign(dP["spin"])
    X = np.column_stack([np.ones(n_stats), dP["spin"], dP["mass"]])
    beta, *_ = np.linalg.lstsq(X, dP["joint"], rcond=None)
    resid = dP["joint"] - X @ beta
    sstot = float(np.sum((dP["joint"] - dP["joint"].mean()) ** 2))
    r2 = float(1.0 - np.sum(resid ** 2) / sstot) if sstot > 0 else float("nan")

    def _r2_single(x):
        Xs = np.column_stack([np.ones(n_stats), x])
        bb, *_ = np.linalg.lstsq(Xs, dP["joint"], rcond=None)
        rr = dP["joint"] - Xs @ bb
        return float(1.0 - np.sum(rr ** 2) / sstot), [float(v) for v in bb]

    r2_spin, b_spin = _r2_single(dP["spin"])
    r2_mass, b_mass = _r2_single(dP["mass"])
    call = {k: P[k] > 0.5 for k in P}
    joint_only = ((call["joint"] != call["spatial"]) &
                  (call["spin"] == call["spatial"]) &
                  (call["mass"] == call["spatial"]))
    complementarity = {
        "dP_mass_vs_dP_spin": _corr(dP["mass"], dP["spin"]),
        "logBF_mass_vs_logBF_spin": _corr(bf_mass, bf_spin),
        "quadrants_both_abs_gt_0p05": {
            "n_events": int(both.sum()),
            "mass_pos_spin_pos": int((both & (sm > 0) & (ss > 0)).sum()),
            "mass_pos_spin_neg": int((both & (sm > 0) & (ss < 0)).sum()),
            "mass_neg_spin_pos": int((both & (sm < 0) & (ss > 0)).sum()),
            "mass_neg_spin_neg": int((both & (sm < 0) & (ss < 0)).sum()),
            "same_sign": int((both & (sm * ss > 0)).sum()),
            "opposite_sign": int((both & (sm * ss < 0)).sum()),
        },
        "sets_abs_gt_0p1": {
            "n_mass": int(a.sum()), "n_spin": int(b.sum()),
            "n_intersection": int((a & b).sum()), "n_union": int((a | b).sum()),
            "jaccard": (float((a & b).sum() / (a | b).sum())
                        if (a | b).sum() else float("nan")),
            "n_mass_only": int((a & ~b).sum()), "n_spin_only": int((b & ~a).sum()),
        },
        "share_of_abs_dP_joint": {
            "sum_abs_dP_joint": float(np.abs(dP["joint"]).sum()),
            "sum_abs_dP_spin": float(np.abs(dP["spin"]).sum()),
            "sum_abs_dP_mass": float(np.abs(dP["mass"]).sum()),
            "ratio_spin_over_joint":
                float(np.abs(dP["spin"]).sum() / np.abs(dP["joint"]).sum()),
            "ratio_mass_over_joint":
                float(np.abs(dP["mass"]).sum() / np.abs(dP["joint"]).sum()),
            "additivity_residual_sum_abs":
                float(np.abs(dP["joint"] - dP["spin"] - dP["mass"]).sum()),
            "additivity_residual_share":
                float(np.abs(dP["joint"] - dP["spin"] - dP["mass"]).sum()
                      / np.abs(dP["joint"]).sum()),
            "definition": ("ratios are sum_i |dP^X_i| / sum_i |dP^joint_i|; the "
                           "additivity residual is sum_i |dP^joint - dP^spin - "
                           "dP^mass| over the same denominator"),
        },
        "regression_dP_joint_on_dP_spin_and_dP_mass": {
            "intercept": float(beta[0]), "coef_dP_spin": float(beta[1]),
            "coef_dP_mass": float(beta[2]), "r_squared": r2,
            "r_squared_dP_spin_alone": r2_spin,
            "coefficients_dP_spin_alone": b_spin,
            "r_squared_dP_mass_alone": r2_mass,
            "coefficients_dP_mass_alone": b_mass,
            "residual_rms": float(np.sqrt(np.mean(resid ** 2))),
        },
        "crossings_0p5": {
            "n_called_AGN": {k: int(call[k].sum()) for k in call},
            "n_crossing_under_joint_but_neither_single_mark": int(joint_only.sum()),
            "n_crossing_under_spin_only": int(((call["spin"] != call["spatial"])
                                               & (call["mass"] == call["spatial"])).sum()),
            "n_crossing_under_mass_only": int(((call["mass"] != call["spatial"])
                                               & (call["spin"] == call["spatial"])).sum()),
            "n_crossing_under_both_single_marks":
                int(((call["spin"] != call["spatial"])
                     & (call["mass"] != call["spatial"])).sum()),
        },
    }
    cq = complementarity["quadrants_both_abs_gt_0p05"]
    cs = complementarity["sets_abs_gt_0p1"]
    print("\n== complementarity ==")
    print(f"  dP_mass vs dP_spin: Pearson "
          f"{complementarity['dP_mass_vs_dP_spin']['pearson_r']:+.4f}, Spearman "
          f"{complementarity['dP_mass_vs_dP_spin']['spearman_rho']:+.4f}")
    print(f"  logBF_mass vs logBF_spin: Pearson "
          f"{complementarity['logBF_mass_vs_logBF_spin']['pearson_r']:+.4f}, Spearman "
          f"{complementarity['logBF_mass_vs_logBF_spin']['spearman_rho']:+.4f}")
    print(f"  quadrants (both |dP|>0.05, N={cq['n_events']}): same sign "
          f"{cq['same_sign']}, opposite {cq['opposite_sign']}")
    print(f"  sets |dP|>0.1: mass {cs['n_mass']}, spin {cs['n_spin']}, "
          f"intersection {cs['n_intersection']}, Jaccard {cs['jaccard']:.4f}")
    print(f"  regression dP_joint = {beta[0]:+.5f} + {beta[1]:+.5f} dP_spin "
          f"{beta[2]:+.5f} dP_mass,  R2 = {r2:.5f}")
    print(f"  crossers of 0.5 under joint but neither single mark: "
          f"{complementarity['crossings_0p5']['n_crossing_under_joint_but_neither_single_mark']}")

    # ---- the interaction term ------------------------------------------------ #
    inter = {
        "definition": "I_i = log BF_joint - log BF_spin - log BF_mass",
        "percentiles": _pct(interaction),
        "median": float(np.median(interaction)),
        "mean": float(np.mean(interaction)),
        "rms": float(np.sqrt(np.mean(interaction ** 2))),
        "median_abs": float(np.median(np.abs(interaction))),
        "max_abs": float(np.max(np.abs(interaction))),
        "n_abs_gt_0p01": int((np.abs(interaction) > 0.01).sum()),
        "n_abs_gt_0p1": int((np.abs(interaction) > 0.1).sum()),
        "n_abs_gt_1": int((np.abs(interaction) > 1.0).sum()),
        "scale_comparison": {
            "median_abs_logBF_spin": float(np.median(np.abs(bf_spin))),
            "median_abs_logBF_mass": float(np.median(np.abs(bf_mass))),
            "median_abs_logBF_joint": float(np.median(np.abs(bf_joint))),
            "median_abs_I_over_median_abs_logBF_joint":
                float(np.median(np.abs(interaction)) / np.median(np.abs(bf_joint))),
            "median_ratio_abs_I_over_abs_logBF_joint":
                float(np.median(np.abs(interaction)
                                / np.maximum(np.abs(bf_joint), 1e-300))),
        },
        "sum_over_events": {
            "sum_logBF_spin": float(bf_spin.sum()),
            "sum_logBF_mass": float(bf_mass.sum()),
            "sum_logBF_joint": float(bf_joint.sum()),
            "sum_I": float(interaction.sum()),
        },
        "note": ("additivity of the two marks in the per-event log Bayes factor "
                 "is a MEASURED statement: it holds only to the extent that "
                 "|I_i| is small against |log BF_joint|, and both are quoted"),
    }
    print("\n== interaction I_i = logBF_joint - logBF_spin - logBF_mass ==")
    print(f"  pct {[round(inter['percentiles'][str(p)], 5) for p in PCT]}")
    print(f"  median {inter['median']:+.6f}  mean {inter['mean']:+.6f}  "
          f"rms {inter['rms']:.6f}  median|I| {inter['median_abs']:.6f}  "
          f"max|I| {inter['max_abs']:.6f}")
    print(f"  median|logBF| spin {inter['scale_comparison']['median_abs_logBF_spin']:.5f}  "
          f"mass {inter['scale_comparison']['median_abs_logBF_mass']:.5f}  "
          f"joint {inter['scale_comparison']['median_abs_logBF_joint']:.5f}")
    print(f"  sum_i: logBF_spin {bf_spin.sum():+.4f}  logBF_mass {bf_mass.sum():+.4f}  "
          f"logBF_joint {bf_joint.sum():+.4f}  I {interaction.sum():+.4f}")

    # ---- interpretation only: the true labels, read now for the first time -- #
    with h5py.File(A10C.GW_PATH_A10, "r") as h:
        host_type = h["host_type"][:].astype(int)
        true_z = h["true_z"][:]
        true_chieff = h["true_chieff"][:]
        true_m1src = h["true_m1src"][:]
    if host_type.size != nEvents:
        raise SystemExit("[fatal] host_type length does not match nEvents")
    if int((host_type == 1).sum()) != realised_agn:
        raise SystemExit(f"[fatal] realised AGN count "
                         f"{int((host_type == 1).sum())} != {realised_agn}")
    with h5py.File(out_h5, "a") as h:
        h.create_dataset("true_host_type", data=host_type)
        h.create_dataset("true_z", data=true_z)
        h.create_dataset("true_chieff", data=true_chieff)
        h.create_dataset("true_m1src", data=true_m1src)
        h.attrs["host_type_encoding"] = "0 = GAL, 1 = AGN (diagnostic only)"
        h.attrs["n_events_excluded_from_statistics"] = n_excluded
    print(f"[decomp] appended the true-label columns to {out_h5}")
    is_agn = (host_type == 1)[FIN] if n_excluded else (host_type == 1)

    def _by_label(x):
        return {"true_GAL_median": float(np.median(x[~is_agn])),
                "true_AGN_median": float(np.median(x[is_agn])),
                "true_GAL_mean": float(np.mean(x[~is_agn])),
                "true_AGN_mean": float(np.mean(x[is_agn])),
                "median_difference_AGN_minus_GAL":
                    float(np.median(x[is_agn]) - np.median(x[~is_agn]))}

    def _classify(p):
        pred = p > 0.5
        return {"true_AGN_called_AGN": int((is_agn & pred).sum()),
                "true_AGN_called_GAL": int((is_agn & ~pred).sum()),
                "true_GAL_called_AGN": int((~is_agn & pred).sum()),
                "true_GAL_called_GAL": int((~is_agn & ~pred).sum()),
                "n_called_AGN": int(pred.sum()),
                "accuracy": float((pred == is_agn).mean())}

    interpretation = {
        "caveat": ("the true host label is read here for the first time, AFTER "
                   "every inference quantity above was formed; it is never an "
                   "input to any of them"),
        "n_true_AGN": int(is_agn.sum()), "n_true_GAL": int((~is_agn).sum()),
        "delta_P_by_true_label": {k: _by_label(dP[k]) for k in dP},
        "abs_delta_P_by_true_label": {k: _by_label(np.abs(dP[k])) for k in dP},
        "P_by_true_label": {k: _by_label(P[k]) for k in P},
        "logBF_by_true_label": {
            "spatial": _by_label(bf_spatial), "spin": _by_label(bf_spin),
            "mass": _by_label(bf_mass), "joint": _by_label(bf_joint),
            "interaction": _by_label(interaction)},
        "auc_for_true_AGN": {
            "log_BF_spatial": _auc(bf_spatial, is_agn),
            "log_BF_spin": _auc(bf_spin, is_agn),
            "log_BF_mass": _auc(bf_mass, is_agn),
            "log_BF_joint": _auc(bf_joint, is_agn),
            "log_BF_total_spatial_plus_joint": _auc(bf_spatial + bf_joint, is_agn),
            "P_spatial": _auc(P["spatial"], is_agn),
            "P_spin": _auc(P["spin"], is_agn),
            "P_mass": _auc(P["mass"], is_agn),
            "P_joint": _auc(P["joint"], is_agn),
        },
        "classification_at_0p5": {k: _classify(P[k]) for k in P},
        "crossings_by_true_label": {
            k: {"up_true_AGN": int((((P["spatial"] <= 0.5) & (P[k] > 0.5)) & is_agn).sum()),
                "up_true_GAL": int((((P["spatial"] <= 0.5) & (P[k] > 0.5)) & ~is_agn).sum()),
                "down_true_AGN": int((((P["spatial"] > 0.5) & (P[k] <= 0.5)) & is_agn).sum()),
                "down_true_GAL": int((((P["spatial"] > 0.5) & (P[k] <= 0.5)) & ~is_agn).sum())}
            for k in ("spin", "mass", "joint")},
        "joint_only_crossers_by_true_label": {
            "n": int(joint_only.sum()),
            "true_AGN": int((joint_only & is_agn).sum()),
            "true_GAL": int((joint_only & ~is_agn).sum())},
    }
    print("\n== interpretation only (true labels) ==")
    au = interpretation["auc_for_true_AGN"]
    for k in ("log_BF_spatial", "log_BF_spin", "log_BF_mass", "log_BF_joint"):
        print(f"  AUC {k:<16s} {au[k]:.4f}")
    for k in ("spatial", "spin", "mass", "joint"):
        c = interpretation["classification_at_0p5"][k]
        print(f"  {k:<8s} called AGN {c['n_called_AGN']:>4d}  accuracy "
              f"{c['accuracy']:.4f}  (TP {c['true_AGN_called_AGN']}, FP "
              f"{c['true_GAL_called_AGN']})")


    summary = {
        "analysis": "analysis_10_mass_spin_marked_multitracer",
        "diagnostic": ("per-event GAL/AGN branch decomposition under four nested "
                       "models at one hyperparameter point"),
        "tag": tag,
        "point": {
            "H0": H0_POINT, "H0_hex": float(H0_POINT).hex(),
            "f_agn": f_agn, "dmu_chi": dmu_chi, "dmu_G": dmu_G,
            "mu_chi_c2": mu_chi_c2, "mu_G_c2": mu_G_c2,
            "axes": point,
            "kind": ("provisional registered point; NOT a posterior summary -- "
                     "the A10-J cube that would locate the MAP is still running"),
            "log_f_gal": lf_gal, "log_f_agn": lf_agn,
            "log_prior_odds_log_fA_over_fG": log_prior_odds,
            "nEvents": nEvents, "nsamp": nsamp, "pe_event_block": block,
        },
        "model": {
            "E": ("E_i[X|Y] = (1/n) sum_s exp(A_X(s) + B_Y(s) + C(s)) over the SAME "
                  "samples with the SAME production mask; A_X the spatial factor of "
                  "tracer X, B_Y = log p_pop(theta_s | Lambda_Y), C branch-free"),
            "Z": "Z_i = (1 - f) E_i[GG] + f E_i[A|Y] for the model carrying Lambda_Y",
            "P": "P_i(AGN | Y) = f E[A|Y] / ((1 - f) E[GG] + f E[A|Y])",
            "log_BF_spatial": "log E[A|G] - log E[GG]",
            "log_BF_spin": "log E[A|chi] - log E[A|G]",
            "log_BF_mass": "log E[A|M] - log E[A|G]",
            "log_BF_joint": "log E[A|AM] - log E[A|G]",
            "interaction": "I_i = log BF_joint - log BF_spin - log BF_mass",
            "machinery": ("analysis_8/scripts/event_decomposition.py :: "
                          "capture_operands (through a10_likelihood._steer, i.e. "
                          "build_a10's own steering) and build_evaluator, imported "
                          "not reimplemented; the four models differ only in the "
                          "AGN population vector handed to build_evaluator"),
        },
        "population_vectors": pop_check,
        "slot_map": slot_map,
        "verification": {
            "pe_term_magnitude": pe_scale,
            "one_ulp_of_pe_term": one_ulp,
            "fail_threshold_absolute": FAIL_ABS,
            "checks": checks,
            "n_checks": len(checks), "n_fail": n_fail, "worst_ulp": worst_ulp,
            "max_per_event_reassociated_minus_production": per_event_reassoc,
            "internal": internal,
            "live_production_records": {
                k: {"coord": v["coord"], "logL": v["logL"],
                    "logL_pe": v.get("logL_pe"),
                    "logL_selection": v.get("logL_selection"),
                    "log_mu": v.get("log_mu"), "Neff": v.get("Neff"),
                    "guard_passes": v.get("guard_passes"),
                    "seconds": v["seconds"]}
                for k, v in live.items()},
            "recorded_references": recorded,
        },
        "n_events_total": nEvents,
        "n_events_in_statistics": n_stats,
        "n_events_excluded_non_finite": n_excluded,
        "statistics": stats,
        "complementarity": complementarity,
        "interaction": inter,
        "interpretation_only": interpretation,
        "truth": A10C.TRUTH,
        "timings": {"build_seconds": build_s, "per_pass": timings,
                    "wall_seconds": float(time.time() - t_start)},
        "environment": env,
        "provenance": A8.provenance(gw_path=A10C.GW_PATH_A10,
                                    survey_paths=cell.survey_paths),
        "slurm_job_id": job_id,
        "inputs_read": [A10C.GW_PATH_A10, A8.SURVEY_GAL, A8.SURVEY_AGN,
                        A8.GWSEL_PATH, str(cl_path)],
        "outputs": {"h5": str(out_h5), "json": str(out_json)},
        "caveats": [
            "Seed 100 only, one realisation.",
            "ONE hyperparameter point, and a PROVISIONAL one: the A10-J cube is "
            "still running, so this point is a registered grid node chosen before "
            "the posterior exists, not a MAP.  Nothing here is marginalised.",
            "H0 is pinned at 67.74; the spectral-siren channel of the mass mark is "
            "therefore not exercised, only the event-level routing channel.",
            "The true host label is a mock diagnostic; it enters no inference "
            "quantity and is read only for the interpretation block.",
            "Seed 100's detected set separates GAL from AGN in mass ratio q before "
            "any mark is applied (Analysis 8's caveat), and the mass mark moves the "
            "detected set itself, so log BF_mass is not a clean measurement of the "
            "planted peak-location offset alone.",
            "Additivity of the two marks is not asserted: the interaction term I_i "
            "is reported with the Bayes factors it is to be judged against.",
        ],
    }
    _write(out_json, summary)
    print(f"[decomp] wall {time.time() - t_start:.1f}s  job {job_id or '-'}")
    if n_fail:
        raise SystemExit(f"[fatal] {n_fail} verification checks exceed {FAIL_ABS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
