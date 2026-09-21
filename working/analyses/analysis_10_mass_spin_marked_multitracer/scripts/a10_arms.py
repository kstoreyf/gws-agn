#!/usr/bin/env python
"""Analysis 10 -- the two CHEAP fixed-H0 arms on the two-mark mock.

  A10-S   f_AGN on the registered 41-node axis with BOTH marks held at zero
          (dmu_chi = 0.0 and dmu_G = 0.0 exactly), evaluated with the TWO-MARK
          model.  Closure 15.1 measures that this equals the shared-population
          K=2 model, so the arm is the mark-free baseline every later width is
          quoted against without needing a second build.

  A10-M   f_AGN x dmu_G on 41 x 21 with dmu_chi = 0.0 exactly: the mass mark
          alone.  Its dmu_G = 0 column is arm A10-S re-evaluated, and the
          assemble stage checks the two BITWISE.

H0 is pinned at 67.74 and the twelve base population parameters stay pinned at
the powerlaw+peak fiducial; the only free coordinates are ``fcat_2`` and
``$\\mu_{\\rm G}$_c2``.

Rows are checkpointed to JSONL keyed on the PHYSICAL coordinates (Analysis 9's
pattern), so a killed job resumes and costs at most one row.

    sbatch --export=ALL,STAGE="S M" scripts/submit_a10_arms_rita.sbatch
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
F_GRID, MG_GRID = C.F_GRID, C.MG_GRID
GW_PATH_A10 = C.GW_PATH_A10

ARMS = {
    "S": {"stem": "a10_arm_S", "grid": np.array([0.0]),
          "what": "spatial only; both marks held at zero"},
    "M": {"stem": "a10_arm_M", "grid": MG_GRID,
          "what": "the mass mark alone; dmu_chi held at zero"},
}
EDGE_CRITERION = 1.0e-6


# --------------------------------------------------------------------------- #
def _write(path, obj):
    return C._write(path, obj)


def _ckpt(arm):
    return DIAG / f"_a10_arm_{arm}.jsonl"


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


def _key(f):
    return "{:.10g}".format(float(f))


def _header(cell, arm, grid):
    return {
        "record": "header",
        "analysis": "analysis_10_mass_spin_marked_multitracer",
        "arm": arm,
        "what": ARMS[arm]["what"],
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "provenance": A8.provenance(gw_path=GW_PATH_A10,
                                    survey_paths=cell.survey_paths),
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
    env = A9._gpu_setup(f"arm_{arm}")
    env["a10_inputs"] = C.assert_a10_inputs(with_md5=(arm == "S"))
    grid = np.asarray(ARMS[arm]["grid"], dtype=float)
    path = _ckpt(arm)

    hdr, rows, dropped = _read_jsonl(path)
    done = {r["key"]: r for r in rows}
    todo = [f for f in F_GRID if _key(f) not in done]
    print(f"\ncheckpoint {path.name}: {len(done)}/{F_GRID.size} rows done, "
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
        _append_jsonl(path, _header(cell, arm, grid))

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
                             "dmu_chi": 0.0, "cells": cells,
                             "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        lls = np.array([c["logL"] for c in cells], dtype=float)
        fin = np.isfinite(lls)
        el = time.time() - t0
        print(f"[arm {arm}] row {n}/{len(todo)} f={f:.4f} "
              f"maxlogL={(lls[fin].max() if fin.any() else float('nan')):.6f} "
              f"rejected={int((~fin).sum())}/{grid.size} "
              f"elapsed={el/60:.1f}min eta={el/n*(len(todo)-n)/60:.1f}min")
        sys.stdout.flush()
    print(f"[arm {arm}] done -> {path}")


# =========================================================================== #
# Stage: assemble (CPU)
# =========================================================================== #
def _arrays(arm, grid):
    hdr, rows, dropped = _read_jsonl(_ckpt(arm))
    if hdr is None:
        raise SystemExit(f"[fatal] no header in {_ckpt(arm)}")
    by = {r["key"]: r for r in rows}
    missing = [float(f) for f in F_GRID if _key(f) not in by]
    if missing:
        raise SystemExit(f"[fatal] arm {arm}: {len(missing)} rows missing: "
                         f"{missing[:5]}")
    shape = (F_GRID.size, grid.size)
    ll = np.full(shape, np.nan)
    fin = np.zeros(shape, dtype=bool)
    hexes = np.empty(shape, dtype=object)
    store = {}
    for i, f in enumerate(F_GRID):
        cells = by[_key(f)]["cells"]
        store[i] = cells
        for j, c in enumerate(cells):
            ll[i, j] = c["logL"]
            fin[i, j] = bool(c["finite"])
            hexes[i, j] = c.get("logL_hex")

    def pull(key, default=np.nan):
        out = np.full(shape, default, dtype=float)
        for i, cells in store.items():
            for j, c in enumerate(cells):
                v = c.get(key)
                out[i, j] = default if v is None else v
        return out

    return hdr, ll, fin, hexes, pull, dropped


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


def stage_assemble(args):
    import h5py
    env = A9._cpu_setup("arms_assemble", import_darksirens=False)
    scan_h0f = A8.import_scan_h0f()
    marginal_ci = scan_h0f.marginal_ci
    out_all = {}

    arrays = {}
    for arm in ("S", "M"):
        grid = np.asarray(ARMS[arm]["grid"], dtype=float)
        hdr, ll, fin, hexes, pull, dropped = _arrays(arm, grid)
        arrays[arm] = (hdr, ll, fin, hexes, pull, grid, dropped)

    # The dmu_G = 0 column of arm M IS arm S; they must agree bitwise.
    (hS, llS, finS, hexS, pullS, gS, _dS) = arrays["S"]
    (hM, llM, finM, hexM, pullM, gM, _dM) = arrays["M"]
    j0 = int(np.argmin(np.abs(gM - 0.0)))
    if gM[j0] != 0.0:
        raise SystemExit("[fatal] the dmu_G axis has no exact 0 node")
    cross = {
        "description": ("arm S is the dmu_G = 0 column of arm M, evaluated "
                        "twice; the two must agree BITWISE"),
        "j_index_of_dmu_G_zero": j0,
        "n_nodes": int(F_GRID.size),
        "n_bitwise_identical": int(sum(1 for i in range(F_GRID.size)
                                       if hexS[i, 0] == hexM[i, j0])),
        "max_abs_diff": float(np.nanmax(np.abs(llS[:, 0] - llM[:, j0]))),
        "all_bitwise_identical": bool(all(hexS[i, 0] == hexM[i, j0]
                                          for i in range(F_GRID.size))),
    }
    print(f"[cross] arm S vs arm M dmu_G = 0 column: "
          f"{cross['n_bitwise_identical']}/{cross['n_nodes']} bitwise identical, "
          f"max |diff| {cross['max_abs_diff']:.3e}")

    for arm in ("S", "M"):
        hdr, ll, fin, hexes, pull, grid, dropped = arrays[arm]
        stem = ARMS[arm]["stem"]
        marg = _rename_mu_to_g(GC._marginals(F_GRID, grid, ll, fin, marginal_ci))
        GC._truth_flags(marg["f"], {"planted": C.TRUTH["f_agn_planted"],
                                    "realised": C.TRUTH["f_agn_realised_detected"]})
        if "dmu_G" in marg:
            GC._truth_flags(marg["dmu_G"],
                            {"planted": C.TRUTH["dmu_G_planted"]})
        guard = _guard(arm, grid, ll, fin, pull)
        llm = np.where(fin, ll, -np.inf)
        mx = float(np.nanmax(llm[np.isfinite(llm)]))
        P = np.where(np.isfinite(llm), np.exp(llm - mx), 0.0)

        edges = {"f_agn": _edges(marg, "f", F_GRID)}
        if "dmu_G" in marg:
            edges["dmu_G"] = _edges(marg, "dmu_G", grid)
            if not edges["dmu_G"]["contained"]:
                print(f"\n  *** arm {arm}: the dmu_G POSTERIOR TOUCHES THE AXIS "
                      f"EDGE: low {edges['dmu_G']['low_edge_density_over_peak']:.3e}"
                      f", high {edges['dmu_G']['high_edge_density_over_peak']:.3e} "
                      f"of the peak (criterion {EDGE_CRITERION:g}) ***\n")

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
                     "dmu_G": [float(grid[0]), float(grid[-1]), int(grid.size)]},
            "truth": dict(C.TRUTH),
            "n_cells": int(ll.size), "n_rejected": int((~fin).sum()),
            "logL_max": marg["logL_max"], "map": marg["map"],
            "f": marg["f"],
            "posterior_moments": marg.get("posterior_moments"),
            "edge_mass": edges,
            "guard": {k: v for k, v in guard.items() if k != "rejected_cells"},
            "guard_rejected_cells": guard.get("rejected_cells", []),
            "arm_S_vs_arm_M_zero_column": cross,
            "timing": {"cells": int(ll.size),
                       "gpu_hours": float(np.nansum(pull("seconds")) / 3600.0),
                       "median_seconds_per_cell": float(
                           np.nanmedian(pull("seconds")))},
            "checkpoint": _ckpt(arm).name,
            "truncated_lines_dropped": dropped,
        }
        if "dmu_G" in marg:
            summary["dmu_G"] = marg["dmu_G"]
            summary["mu_G_c2"] = dict(
                marg["dmu_G"],
                note=("mu_G_c2 = 35 + dmu_G; the reporting coordinate is the "
                      "offset"))
            summary["correlation_f_dmu_G"] = marg["posterior_moments"][
                "correlation_f_dmu_G"]

        h5_path = RESULTS / f"{stem}.h5"
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(h5_path, "w") as h5:
            h5.create_dataset("f_grid", data=F_GRID)
            h5.create_dataset("dmu_G_grid", data=grid)
            h5.create_dataset("mu_G_c2_grid",
                              data=np.array([A10.dmuG_to_label(x) for x in grid]))
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
                marg["f"]["marginal_logp"])))
            if "dmu_G" in marg:
                m.create_dataset("dmu_G", data=np.exp(np.asarray(
                    marg["dmu_G"]["marginal_logp"])))
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
                  f"{summary['map']['dmu_G']:.6f}")
            print(f"  A10-{arm} rho(f, dmu_G) = "
                  f"{summary['correlation_f_dmu_G']:+.6f}")
    return out_all


# =========================================================================== #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=("S", "M", "assemble"))
    args = ap.parse_args(argv)
    if args.stage in ("S", "M"):
        return stage_scan(args, args.stage)
    return stage_assemble(args)


if __name__ == "__main__":
    main()
