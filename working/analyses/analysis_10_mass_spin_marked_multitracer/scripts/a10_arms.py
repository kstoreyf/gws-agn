#!/usr/bin/env python
"""Analysis 10 -- the two CHEAP fixed-H0 arms on the two-mark mock.

  A10-S   f_AGN on the registered 41-node axis with BOTH marks held at zero
          (dmu_chi = 0.0 and dmu_G = 0.0 exactly), evaluated with the TWO-MARK
          model.  Closure 15.1 measures that this equals the shared-population
          K=2 model, so the arm is the mark-free baseline every later width is
          quoted against without needing a second build.

  A10-M   f_AGN x dmu_G with dmu_chi = 0.0 exactly: the mass mark alone.  Its
          dmu_G = 0 column is arm A10-S re-evaluated, and the assemble stage
          checks the two BITWISE.

THE ONE ALLOWED GRID REFINEMENT (owner decision, 2026-09-21).  The registered
dmu_G axis is 21 nodes at 1 Msun.  A10-M measured sd(dmu_G) = 0.634 Msun and a
68% width of 1.87 Msun -- fewer than two nodes across the interval -- so six
HALF-INTEGER nodes {2.5, 3.5, 4.5, 5.5, 6.5, 7.5} are added inside [2, 8],
where the measured marginal sits above ~e^-9 of its peak.  They are ADDITIVE
ROWS, not a new grid: rows stay keyed on the physical coordinates, the coarse
rows are untouched and reused, and ``assemble`` merges the two checkpoints into
one 27-node axis (1 Msun outside [2, 8], 0.5 Msun inside) that is integrated
with the SAME non-uniform trapezoid weights.  Every dmu_G result is reported on
the refined axis WITH the coarse-axis number beside it, so the effect of the
refinement is measured rather than assumed.  There is no second refinement.

H0 is pinned at 67.74 and the twelve base population parameters stay pinned at
the powerlaw+peak fiducial; the only free coordinates are ``fcat_2`` and
``$\\mu_{\\rm G}$_c2``.

Rows are checkpointed to JSONL keyed on the PHYSICAL coordinates (Analysis 9's
pattern), so a killed job resumes and costs at most one row.

    sbatch --export=ALL,STAGE="S M" scripts/submit_a10_arms_rita.sbatch
    sbatch --export=ALL,STAGE="M",DMUG_NODES=refine \\
        scripts/submit_a10_arms_rita.sbatch
    python scripts/a10_arms.py --stage assemble          # CPU
"""
import argparse
import json
import os
import socket
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

import a10_closure as C                      # node sets, asserts, writer, truths

A10, A8, GC, A9 = C.A10, C.A8, C.GC, C.A9
L_MU, L_MG = C.L_MU, C.L_MG
F_GRID = C.F_GRID
MG_COARSE = C.MG_GRID                        # the 21 registered nodes
MG_REFINE = C.MG_REFINE                      # the 6 half-integer nodes
NODE_SETS = {"registered": MG_COARSE, "refine": MG_REFINE}
GW_PATH_A10 = C.GW_PATH_A10

ARMS = {
    "S": {"stem": "a10_arm_S", "what": "spatial only; both marks held at zero"},
    "M": {"stem": "a10_arm_M", "what": "the mass mark alone; dmu_chi held at zero"},
}
EDGE_CRITERION = 1.0e-6


# --------------------------------------------------------------------------- #
def _write(path, obj):
    return C._write(path, obj)


def _grid_for(arm, nodes):
    if arm == "S":
        return np.array([0.0])
    return np.asarray(NODE_SETS[nodes], dtype=float)


def _ckpt(arm, nodes="registered"):
    suffix = "" if nodes == "registered" else f"_{nodes}"
    return DIAG / f"_a10_arm_{arm}{suffix}.jsonl"


def _append_jsonl(path, obj):
    path = Path(path)
    if ANALYSIS_DIR not in path.parents:
        raise RuntimeError(f"[fatal] refusing to write outside {ANALYSIS_DIR}")
    path.parent.mkdir(parents=True, exist_ok=True)
    lead = ""
    if path.exists():
        with open(path, "rb") as fh:
            if fh.seek(0, os.SEEK_END):
                fh.seek(-1, os.SEEK_END)
                if fh.read(1) != b"\n":
                    lead = "\n"
    with open(path, "a") as fh:
        fh.write(lead + json.dumps(obj, default=GC._json_default) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _read_jsonl(path):
    header, rows, dropped = None, [], 0
    if not Path(path).exists():
        return header, rows, dropped
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            dropped += 1
            continue
        if obj.get("record") == "header":
            header = obj
        elif obj.get("record") == "row":
            rows.append(obj)
    return header, rows, dropped


def _key(x):
    return "{:.10g}".format(float(x))


def _header(cell, arm, grid, nodes):
    return {
        "record": "header",
        "analysis": "analysis_10_mass_spin_marked_multitracer",
        "arm": arm,
        "dmu_G_node_set": nodes,
        "what": ARMS[arm]["what"],
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "provenance": A8.provenance(gw_path=GW_PATH_A10,
                                    survey_paths=cell.survey_paths),
        "darksirens_sha": A9.DARKSIRENS_A8_SHA,
        "events_file": GW_PATH_A10,
        "events_md5": C.GW_MD5_A10,
        "survey_paths": [str(s) for s in cell.survey_paths],
        "selection_file": str(cell.opts.gwselection_path),
        "sampled_labels": list(cell.labels),
        "per_catalog_pop_params": list(cell.per_catalog_pop_params),
        "fixed_parameter_values": {k: float(v)
                                   for k, v in cell.fixed_parameter_values.items()},
        "base_coord": {k: float(v) for k, v in zip(cell.labels, cell.base)},
        "free_parameters": (["fcat_2"] if arm == "S" else ["fcat_2", L_MG]),
        "dmu_chi_held_at": 0.0,
        "H0_fixed": A8.H0_FID, "Om0_fixed": A8.OM0_FID,
        "nEvents": int(cell.data["nEvents"]), "nsamp": int(cell.data["nsamp"]),
        "Ndraw": float(cell.data["Ndraw"]),
        "f_grid": F_GRID.tolist(),
        "dmu_G_grid": np.asarray(grid, dtype=float).tolist(),
        "mu_G_c2_grid": [A10.dmuG_to_label(x) for x in np.asarray(grid)],
        "truth": dict(C.TRUTH),
        "tolerances": dict(C.TOL),
    }


# =========================================================================== #
# Stage: scan one arm (GPU, rita)
# =========================================================================== #
def stage_scan(args, arm):
    nodes = args.dmuG_nodes
    if arm == "S" and nodes != "registered":
        raise SystemExit("[fatal] arm S has a single dmu_G node (0.0); the "
                         "refinement applies to arm M only")
    env = A9._gpu_setup(f"arm_{arm}_{nodes}")
    env["a10_inputs"] = C.assert_a10_inputs(with_md5=False)
    grid = _grid_for(arm, nodes)
    path = _ckpt(arm, nodes)

    hdr, rows, dropped = _read_jsonl(path)
    done = {r["key"]: r for r in rows}
    todo = [f for f in F_GRID if _key(f) not in done]
    print(f"\narm {arm} [{nodes}]: dmu_G nodes {grid.tolist()}")
    print(f"checkpoint {path.name}: {len(done)}/{F_GRID.size} rows done, "
          f"{dropped} truncated line(s) dropped; {len(todo)} rows to do "
          f"({len(todo) * grid.size} cells)")
    if not todo:
        print("nothing to do")
        return

    t0 = time.time()
    cell = A10.build_a10(f"A10_ARM_{arm}", [A8.SURVEY_GAL, A8.SURVEY_AGN],
                         verbose=True, gw_path=GW_PATH_A10)
    C.assert_cell(cell, A10.EXPECTED_LABELS_A10, f"arm {arm}")
    print(f"build: {time.time() - t0:.1f}s")
    if hdr is None:
        _append_jsonl(path, _header(cell, arm, grid, nodes))

    t0 = time.time()
    for n, f in enumerate(todo, start=1):
        cells = []
        for g in grid:
            rec = cell.evaluate_at(fcat_2=float(f), dmu_chi=0.0, dmu_G=float(g))
            row = GC._cell_record(rec)
            row["logL_hex"] = rec["logL_hex"]
            row["dmu_G"] = float(g)
            row["mu_G_c2"] = A10.dmuG_to_label(g)
            cells.append(row)
        _append_jsonl(path, {"record": "row", "f_agn": float(f), "key": _key(f),
                             "dmu_chi": 0.0, "dmu_G_node_set": nodes,
                             "cells": cells,
                             "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        lls = np.array([c["logL"] for c in cells], dtype=float)
        fin = np.isfinite(lls)
        el = time.time() - t0
        print(f"[arm {arm}/{nodes}] row {n}/{len(todo)} f={f:.4f} "
              f"maxlogL={(lls[fin].max() if fin.any() else float('nan')):.6f} "
              f"rejected={int((~fin).sum())}/{grid.size} "
              f"elapsed={el/60:.1f}min eta={el/n*(len(todo)-n)/60:.1f}min")
        sys.stdout.flush()
    print(f"[arm {arm}/{nodes}] done -> {path}")


# =========================================================================== #
# Stage: assemble (CPU)
# =========================================================================== #
def _collect(arm):
    """Merge every checkpoint of this arm on the PHYSICAL (f, dmu_G) key."""
    hdr0, store, dropped, files, dup = None, {}, 0, [], []
    for nodes in ("registered", "refine"):
        p = _ckpt(arm, nodes)
        if not p.exists():
            continue
        h, rows, d = _read_jsonl(p)
        dropped += d
        files.append(p.name)
        if h is not None and hdr0 is None:
            hdr0 = h
        for r in rows:
            for c in r["cells"]:
                k = (_key(r["f_agn"]), _key(c["dmu_G"]))
                if k in store:
                    dup.append({"key": list(k), "file": p.name,
                                "same_hex": bool(store[k].get("logL_hex")
                                                 == c.get("logL_hex")),
                                "abs_delta_logL": abs(float(store[k]["logL"])
                                                      - float(c["logL"]))})
                store[k] = c
    if hdr0 is None:
        raise SystemExit(f"[fatal] no checkpoint header for arm {arm}")
    axis = np.array(sorted({float(c["dmu_G"]) for c in store.values()}))
    return hdr0, axis, store, dropped, files, dup


def _arrays(store, axis):
    shape = (F_GRID.size, axis.size)
    ll = np.full(shape, np.nan)
    fin = np.zeros(shape, dtype=bool)
    hexes = np.empty(shape, dtype=object)
    grid_cells = {}
    missing = []
    for i, f in enumerate(F_GRID):
        for j, g in enumerate(axis):
            c = store.get((_key(f), _key(g)))
            if c is None:
                missing.append({"f_agn": float(f), "dmu_G": float(g)})
                continue
            grid_cells[(i, j)] = c
            ll[i, j] = c["logL"]
            fin[i, j] = bool(c["finite"])
            hexes[i, j] = c.get("logL_hex")

    def pull(key, default=np.nan):
        out = np.full(shape, default, dtype=float)
        for (i, j), c in grid_cells.items():
            v = c.get(key)
            out[i, j] = default if v is None else v
        return out

    return ll, fin, hexes, pull, missing


def _edges(marg, key, grid):
    lp = np.asarray(marg[key]["marginal_logp"], dtype=float)
    p = np.exp(lp - np.nanmax(lp[np.isfinite(lp)]))
    return {"low_edge_density_over_peak": float(p[0]),
            "high_edge_density_over_peak": float(p[-1]),
            "low_edge_value": float(grid[0]), "high_edge_value": float(grid[-1]),
            "criterion": f"both <= {EDGE_CRITERION:g} of the peak",
            "contained": bool(p[0] <= EDGE_CRITERION and p[-1] <= EDGE_CRITERION)}


def _rename_mu_to_g(block):
    """gate_c_three_arms names the second axis mu_chi_c2; here it is dmu_G."""
    out = dict(block)
    if "mu_chi_c2" in out:
        out["dmu_G"] = out.pop("mu_chi_c2")
    pm = out.get("posterior_moments")
    if pm:
        pm = dict(pm)
        if "mean_mu_chi_c2" in pm:
            pm["mean_dmu_G"] = pm.pop("mean_mu_chi_c2")
        if "sd_mu_chi_c2" in pm:
            pm["sd_dmu_G"] = pm.pop("sd_mu_chi_c2")
        pm["correlation_f_dmu_G"] = pm.get("correlation")
        out["posterior_moments"] = pm
    mp = out.get("map")
    if mp:
        mp = dict(mp)
        if "mu_chi_c2" in mp:
            mp["dmu_G"] = mp.pop("mu_chi_c2")
        mp.pop("dmu_chi", None)
        out["map"] = mp
    return out


def _guard(arm, grid, ll, fin, pull):
    rep = GC._guard_report(arm, F_GRID, grid, ll, fin, pull)
    for c in rep.get("rejected_cells", []):
        c["dmu_G"] = c.pop("mu_chi_c2", None)
        c.pop("dmu_chi", None)
        c["j_dmu_G"] = c.pop("j_mu", None)
    if "first_rejected_mu_chi_c2" in rep:
        rep["first_rejected_dmu_G"] = rep.pop("first_rejected_mu_chi_c2")
    if "accepted_mass_fraction_beyond_first_rejected_mu" in rep:
        rep["accepted_mass_fraction_beyond_first_rejected_dmu_G"] = rep.pop(
            "accepted_mass_fraction_beyond_first_rejected_mu")
    for m in rep.get("falling_towards_the_guard", []):
        if "last_accepted_mu" in m:
            m["last_accepted_dmu_G"] = m.pop("last_accepted_mu")
    return rep


def _truths_f(block):
    """The f_AGN scoring convention: one TARGET, one descriptive number."""
    GC._truth_flags(block, {"planted": C.TRUTH["f_agn_planted"]})
    GC._truth_flags(block,
                    {"detected_set_fraction": C.TRUTH["f_agn_detected_fraction"]})
    block["truth_convention"] = C.TRUTH["f_agn_convention"]
    return block


def refinement_smoothness(axis, values, name, refine_nodes=MG_REFINE):
    """Is each refinement node where the coarse axis said it would be?

    Two statements, both measured:
      * the LINEAR-INTERPOLATION RESIDUAL at each refinement node, i.e. the
        measured value minus the average of its two integer neighbours -- what
        the coarse axis would have assumed;
      * the SECOND DIFFERENCES on the coarse sub-axis (h = 1) and on the refined
        sub-axis (h = 0.5), with the h^2-scaled comparison, since a smooth
        function has Delta^2 ~ h^2 f''.
    """
    axis = np.asarray(axis, dtype=float)
    v = np.asarray(values, dtype=float)
    by = {round(float(x), 6): float(y) for x, y in zip(axis, v)}
    rows = []
    for g in np.asarray(refine_nodes, dtype=float):
        lo, hi = round(g - 0.5, 6), round(g + 0.5, 6)
        k = round(float(g), 6)
        if k not in by or lo not in by or hi not in by:
            continue
        lin = 0.5 * (by[lo] + by[hi])
        rows.append({"dmu_G": float(g), "measured": by[k],
                     "linear_from_integer_neighbours": lin,
                     "residual": by[k] - lin,
                     "neighbours": [by[lo], by[hi]]})
    coarse = np.array([by[round(float(x), 6)] for x in MG_COARSE
                       if round(float(x), 6) in by])
    d2_coarse = np.diff(coarse, 2) if coarse.size >= 3 else np.array([np.nan])
    inner = np.array(sorted(k for k in by if 2.0 - 1e-9 <= k <= 8.0 + 1e-9))
    fine = np.array([by[k] for k in inner])
    d2_fine = np.diff(fine, 2) if fine.size >= 3 else np.array([np.nan])
    out = {
        "quantity": name,
        "per_refinement_node": rows,
        "max_abs_linear_interpolation_residual": (
            float(max(abs(r["residual"]) for r in rows)) if rows
            else float("nan")),
        "median_abs_linear_interpolation_residual": (
            float(np.median([abs(r["residual"]) for r in rows])) if rows
            else float("nan")),
        "second_difference_coarse_h1": {
            "values": d2_coarse.tolist(),
            "max_abs": float(np.nanmax(np.abs(d2_coarse))),
            "median_abs": float(np.nanmedian(np.abs(d2_coarse)))},
        "second_difference_refined_h0p5_inside_2_to_8": {
            "nodes": inner.tolist(),
            "values": d2_fine.tolist(),
            "max_abs": float(np.nanmax(np.abs(d2_fine))),
            "median_abs": float(np.nanmedian(np.abs(d2_fine)))},
        "h2_scaling_check": {
            "note": ("for a smooth function Delta^2 ~ h^2 f'', so the refined "
                     "(h = 0.5) second differences should be about a QUARTER "
                     "of the coarse (h = 1) ones over the same interval"),
            "ratio_median_refined_over_coarse": float(
                np.nanmedian(np.abs(d2_fine)) / np.nanmedian(np.abs(d2_coarse)))
            if np.isfinite(np.nanmedian(np.abs(d2_coarse)))
            and np.nanmedian(np.abs(d2_coarse)) != 0 else float("nan"),
            "expected_if_smooth": 0.25},
    }
    return out


def _block_for_axis(axis, ll, fin, marginal_ci, tag):
    """Marginals / moments / MAP / edges on ONE dmu_G axis."""
    marg = _rename_mu_to_g(GC._marginals(F_GRID, axis, ll, fin, marginal_ci))
    _truths_f(marg["f"])
    out = {"axis_tag": tag, "n_dmu_G_nodes": int(axis.size),
           "dmu_G_nodes": axis.tolist(),
           "logL_max": marg["logL_max"], "map": marg["map"], "f": marg["f"],
           "edge_mass": {"f_agn": _edges(marg, "f", F_GRID)}}
    if "dmu_G" in marg:
        GC._truth_flags(marg["dmu_G"], {"planted": C.TRUTH["dmu_G_planted"]})
        out["dmu_G"] = marg["dmu_G"]
        out["posterior_moments"] = marg["posterior_moments"]
        out["correlation_f_dmu_G"] = marg["posterior_moments"][
            "correlation_f_dmu_G"]
        out["edge_mass"]["dmu_G"] = _edges(marg, "dmu_G", axis)
    return out, marg


def stage_assemble(args):
    import h5py
    env = A9._cpu_setup("arms_assemble", import_darksirens=False)
    scan_h0f = A8.import_scan_h0f()
    marginal_ci = scan_h0f.marginal_ci
    out_all, packed = {}, {}

    for arm in ("S", "M"):
        hdr, axis, store, dropped, files, dup = _collect(arm)
        ll, fin, hexes, pull, missing = _arrays(store, axis)
        if missing:
            raise SystemExit(f"[fatal] arm {arm}: {len(missing)} (f, dmu_G) "
                             f"cells missing, first {missing[:5]}")
        packed[arm] = (hdr, axis, ll, fin, hexes, pull, dropped, files, dup)

    # The dmu_G = 0 column of arm M IS arm S; they must agree bitwise.
    (_hS, aS, llS, finS, hexS, _pS, _dS, _fS, _uS) = packed["S"]
    (_hM, aM, llM, finM, hexM, _pM, _dM, _fM, _uM) = packed["M"]
    jS = int(np.argmin(np.abs(aS - 0.0)))
    jM = int(np.argmin(np.abs(aM - 0.0)))
    cross = {
        "description": ("arm S is the dmu_G = 0 column of arm M, evaluated "
                        "twice; the two must agree BITWISE"),
        "n_nodes": int(F_GRID.size),
        "n_bitwise_identical": int(sum(1 for i in range(F_GRID.size)
                                       if hexS[i, jS] == hexM[i, jM])),
        "max_abs_diff": float(np.nanmax(np.abs(llS[:, jS] - llM[:, jM]))),
        "all_bitwise_identical": bool(all(hexS[i, jS] == hexM[i, jM]
                                          for i in range(F_GRID.size))),
    }
    print(f"[cross] arm S vs arm M dmu_G = 0 column: "
          f"{cross['n_bitwise_identical']}/{cross['n_nodes']} bitwise identical, "
          f"max |diff| {cross['max_abs_diff']:.3e}")

    for arm in ("S", "M"):
        hdr, axis, ll, fin, hexes, pull, dropped, files, dup = packed[arm]
        stem = ARMS[arm]["stem"]
        refined, marg = _block_for_axis(axis, ll, fin, marginal_ci, "refined"
                                        if axis.size > MG_COARSE.size else
                                        "registered")
        guard = _guard(arm, axis, ll, fin, pull)
        llm = np.where(fin, ll, -np.inf)
        mx = float(np.nanmax(llm[np.isfinite(llm)]))
        P = np.where(np.isfinite(llm), np.exp(llm - mx), 0.0)

        coarse_block, smooth = None, None
        if arm == "M" and axis.size > MG_COARSE.size:
            keep = np.array([bool(np.any(np.isclose(MG_COARSE, g, atol=1e-9)))
                             for g in axis])
            coarse_block, _ = _block_for_axis(axis[keep], ll[:, keep],
                                              fin[:, keep], marginal_ci,
                                              "registered_21_node_coarse")
            lp = np.asarray(refined["dmu_G"]["marginal_logp"], dtype=float)
            i_map = int(refined["map"]["index"][0])
            smooth = {
                "refinement_nodes": MG_REFINE.tolist(),
                "why": ("A10-M on the 21-node axis gave sd(dmu_G) = 0.634 and a "
                        "68% width of 1.87 Msun, i.e. fewer than two nodes "
                        "across the interval; the one allowed refinement adds "
                        "0.5 Msun nodes inside [2, 8]"),
                "marginal_log_density": refinement_smoothness(
                    axis, lp, "log of the dmu_G marginal density"),
                "logL_at_MAP_f": refinement_smoothness(
                    axis, llm[i_map], f"logL along dmu_G at f = "
                                       f"{float(F_GRID[i_map]):.4g}"),
            }

        summary = {
            "analysis": "analysis_10_mass_spin_marked_multitracer",
            "arm": f"A10-{arm}", "what": ARMS[arm]["what"],
            "file": f"results/{stem}.h5",
            "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00",
                                            time.gmtime()),
            "environment": env,
            "provenance": hdr.get("provenance"),
            "events_file": GW_PATH_A10, "events_md5": C.GW_MD5_A10,
            "selection_file": hdr.get("selection_file"),
            "survey_paths": hdr.get("survey_paths"),
            "sampled_labels": hdr.get("sampled_labels"),
            "per_catalog_pop_params": hdr.get("per_catalog_pop_params"),
            "base_coord": hdr.get("base_coord"),
            "fixed_parameter_values": hdr.get("fixed_parameter_values"),
            "free_parameters": hdr.get("free_parameters"),
            "dmu_chi_held_at": 0.0,
            "H0_fixed": A8.H0_FID, "Om0_fixed": A8.OM0_FID,
            "grid": {"f_agn": [float(F_GRID[0]), float(F_GRID[-1]),
                               int(F_GRID.size)],
                     "dmu_G_nodes": axis.tolist(),
                     "dmu_G_n": int(axis.size),
                     "dmu_G_spacing": ("1 Msun outside [2, 8], 0.5 Msun inside"
                                       if axis.size > MG_COARSE.size
                                       else "1 Msun")},
            "truth": dict(C.TRUTH),
            "n_cells": int(ll.size), "n_rejected": int((~fin).sum()),
            "logL_max": refined["logL_max"], "map": refined["map"],
            "f": refined["f"],
            "posterior_moments": refined.get("posterior_moments"),
            "edge_mass": refined["edge_mass"],
            "guard": {k: v for k, v in guard.items() if k != "rejected_cells"},
            "guard_rejected_cells": guard.get("rejected_cells", []),
            "arm_S_vs_arm_M_zero_column": cross,
            "duplicate_cells": dup,
            "timing": {"cells": int(ll.size),
                       "gpu_hours": float(np.nansum(pull("seconds")) / 3600.0),
                       "median_seconds_per_cell": float(
                           np.nanmedian(pull("seconds")))},
            "checkpoints": files,
            "truncated_lines_dropped": dropped,
        }
        if "dmu_G" in refined:
            summary["dmu_G"] = refined["dmu_G"]
            summary["correlation_f_dmu_G"] = refined["correlation_f_dmu_G"]
            summary["mu_G_c2"] = dict(
                refined["dmu_G"],
                note="mu_G_c2 = 35 + dmu_G; the reporting coordinate is the offset")
        if coarse_block is not None:
            summary["coarse_axis_comparison"] = {
                "description": ("the SAME cells restricted to the 21 registered "
                                "1 Msun nodes, so the refinement's effect is "
                                "measured rather than assumed"),
                "registered_21_node": coarse_block,
                "refined_27_node": {k: refined[k] for k in
                                    ("n_dmu_G_nodes", "logL_max", "map", "f",
                                     "dmu_G", "posterior_moments",
                                     "correlation_f_dmu_G", "edge_mass")
                                    if k in refined},
                "shifts": {
                    "dmu_G_median": (refined["dmu_G"]["median"]
                                     - coarse_block["dmu_G"]["median"]),
                    "dmu_G_width68": ((refined["dmu_G"]["ci68"][1]
                                       - refined["dmu_G"]["ci68"][0])
                                      - (coarse_block["dmu_G"]["ci68"][1]
                                         - coarse_block["dmu_G"]["ci68"][0])),
                    "dmu_G_width90": ((refined["dmu_G"]["ci90"][1]
                                       - refined["dmu_G"]["ci90"][0])
                                      - (coarse_block["dmu_G"]["ci90"][1]
                                         - coarse_block["dmu_G"]["ci90"][0])),
                    "f_median": refined["f"]["median"] - coarse_block["f"]["median"],
                    "rho_f_dmu_G": (refined["correlation_f_dmu_G"]
                                    - coarse_block["correlation_f_dmu_G"]),
                },
            }
            summary["refinement_smoothness"] = smooth

        h5_path = RESULTS / f"{stem}.h5"
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(h5_path, "w") as h5:
            h5.create_dataset("f_grid", data=F_GRID)
            h5.create_dataset("dmu_G_grid", data=axis)
            h5.create_dataset("mu_G_c2_grid",
                              data=np.array([A10.dmuG_to_label(x) for x in axis]))
            h5.create_dataset("log_likelihood",
                              data=(ll[:, 0] if arm == "S" else ll))
            h5.create_dataset("posterior_unnormalised",
                              data=(P[:, 0] if arm == "S" else P))
            g = h5.create_group("guard")
            for key in ("Neff", "threshold", "pe_variance_sum", "sigma2_total",
                        "logL_selection", "logL_pe", "log_mu", "seconds"):
                d = pull(key)
                g.create_dataset(key, data=(d[:, 0] if arm == "S" else d))
            g.create_dataset("rejected",
                             data=((~fin)[:, 0] if arm == "S" else (~fin)))
            g.attrs["n_rejected"] = int((~fin).sum())
            m = h5.create_group("marginal")
            m.create_dataset("f", data=np.exp(np.asarray(
                refined["f"]["marginal_logp"])))
            if "dmu_G" in refined:
                m.create_dataset("dmu_G", data=np.exp(np.asarray(
                    refined["dmu_G"]["marginal_logp"])))
            if coarse_block is not None:
                cg = h5.create_group("coarse_21_node")
                cg.create_dataset("dmu_G_grid",
                                  data=np.asarray(coarse_block["dmu_G_nodes"]))
                cg.create_dataset("marginal_f", data=np.exp(np.asarray(
                    coarse_block["f"]["marginal_logp"])))
                cg.create_dataset("marginal_dmu_G", data=np.exp(np.asarray(
                    coarse_block["dmu_G"]["marginal_logp"])))
            h5.attrs["analysis"] = "analysis_10_mass_spin_marked_multitracer"
            h5.attrs["arm"] = f"A10-{arm}"
            h5.attrs["free_parameters"] = json.dumps(hdr.get("free_parameters"))
            h5.attrs["labels"] = json.dumps(hdr.get("sampled_labels"))
            h5.attrs["base_coord"] = json.dumps(hdr.get("base_coord"))
            h5.attrs["truth"] = json.dumps(C.TRUTH)
            h5.attrs["H0_fixed"] = A8.H0_FID
            h5.attrs["Om0_fixed"] = A8.OM0_FID
            h5.attrs["dmu_chi_held_at"] = 0.0
            h5.attrs["events_file"] = GW_PATH_A10
            h5.attrs["events_md5"] = C.GW_MD5_A10
            h5.attrs["selection_file"] = str(hdr.get("selection_file"))
            h5.attrs["survey_paths"] = json.dumps(hdr.get("survey_paths"))
            h5.attrs["darksirens_sha"] = A9.DARKSIRENS_A8_SHA
            h5.attrs["settings"] = json.dumps(A8.SETTINGS)
        print(f"wrote {h5_path}")
        _write(RESULTS / f"{stem}.json", summary)
        out_all[arm] = summary

        b = summary["f"]
        print(f"  A10-{arm} f_AGN: median {b['median']:.6f}  68% "
              f"[{b['ci68'][0]:.6f}, {b['ci68'][1]:.6f}]  90% "
              f"[{b['ci90'][0]:.6f}, {b['ci90'][1]:.6f}]  MAP "
              f"{summary['map']['f']:.6f}")
        if "dmu_G" in summary:
            b = summary["dmu_G"]
            print(f"  A10-{arm} dmu_G : median {b['median']:.6f}  68% "
                  f"[{b['ci68'][0]:.6f}, {b['ci68'][1]:.6f}]  90% "
                  f"[{b['ci90'][0]:.6f}, {b['ci90'][1]:.6f}]  MAP "
                  f"{summary['map']['dmu_G']:.6f}  ({axis.size} nodes)")
            print(f"  A10-{arm} rho(f, dmu_G) = "
                  f"{summary['correlation_f_dmu_G']:+.6f}")
        if coarse_block is not None:
            cb = coarse_block["dmu_G"]
            print(f"    coarse 21-node: dmu_G median {cb['median']:.6f}  68% "
                  f"[{cb['ci68'][0]:.6f}, {cb['ci68'][1]:.6f}]  90% "
                  f"[{cb['ci90'][0]:.6f}, {cb['ci90'][1]:.6f}]")
            s = smooth["marginal_log_density"]
            print(f"    refinement smoothness (log marginal): max |linear-"
                  f"interp residual| {s['max_abs_linear_interpolation_residual']:.4g}"
                  f", Delta^2 median h=1 {s['second_difference_coarse_h1']['median_abs']:.4g}"
                  f" vs h=0.5 "
                  f"{s['second_difference_refined_h0p5_inside_2_to_8']['median_abs']:.4g}"
                  f" (ratio {s['h2_scaling_check']['ratio_median_refined_over_coarse']:.3f}"
                  f", 0.25 if smooth)")
    return out_all


# =========================================================================== #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=("S", "M", "assemble"))
    ap.add_argument("--dmuG_nodes", default="registered",
                    choices=("registered", "refine"))
    args = ap.parse_args(argv)
    if args.stage in ("S", "M"):
        return stage_scan(args, args.stage)
    return stage_assemble(args)


if __name__ == "__main__":
    main()
