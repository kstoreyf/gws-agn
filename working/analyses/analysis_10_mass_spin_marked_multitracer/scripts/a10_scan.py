#!/usr/bin/env python
"""Analysis 10 -- ARM A10-J, the joint fixed-H0 cube on the two-mark mock.

    free   f_AGN x dmu_chi x dmu_G  =  41 x 61 x 21  =  52,521 cells
    held   H0 = 67.74, Om0 = 0.3075, the twelve base population parameters
           pinned at the powerlaw+peak fiducial, the spatial nuisances at the
           Analysis-8 values

One checkpoint unit is a ROW: the 61 dmu_chi cells at fixed (f, dmu_G).  There
are 41 x 21 = 861 such rows; they are dealt to 8 INTERLEAVED chunks, so a
partially finished cube still spans the whole (f, dmu_G) plane instead of
stopping halfway across it.  Rows are keyed on the PHYSICAL coordinates, so a
killed worker resumes and costs at most one row.

    THE dmu_G = 0 SLAB (node index 10 of the 21-node axis) IS ARM A10-chi.
    It is the spin-only arm on the same axes -- Analysis 8's arm J re-run on the
    two-mark mock -- and closure 15.2 is exactly the statement that the two-mark
    model reduces to the spin-only model there.  That slab is dealt FIRST, so
    A10-chi closes in the first wave of workers while the rest of the cube is
    still running, and ``--stage assemble`` writes it out as its own arm
    (``results/a10_arm_chi.{h5,json}``) beside the joint cube
    (``results/a10_arm_J.{h5,json}``).

    sbatch --array=0-7%2 --export=ALL,N_CHUNKS=8 scripts/submit_a10_j_rita.sbatch
    python scripts/a10_scan.py --stage status               # CPU
    python scripts/a10_scan.py --stage assemble             # CPU

Nothing is regenerated, no darksirens file is touched, and every write lands
under the Analysis-10 directory.
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

import a10_closure as C

A10, A8, GC, A9 = C.A10, C.A8, C.GC, C.A9
L_MU, L_MG = C.L_MU, C.L_MG
F_GRID, MU_GRID, MG_GRID = C.F_GRID, C.MU_GRID, C.MG_GRID
MG_REFINE = C.MG_REFINE                 # the six 0.5 Msun nodes inside [2, 8]
MG_FULL = C.MG_GRID_REFINED             # the merged 27-node axis
NODE_SETS = {"registered": MG_GRID, "refine": MG_REFINE}
GW_PATH_A10 = C.GW_PATH_A10
N_ROWS = int(F_GRID.size * MG_GRID.size)
N_CELLS = int(N_ROWS * MU_GRID.size)
N_ROWS_REFINE = int(F_GRID.size * MG_REFINE.size)
N_ROWS_FULL = N_ROWS + N_ROWS_REFINE
J_CHI_SLAB = int(np.argmin(np.abs(MG_GRID - 0.0)))       # 10 on the coarse axis
EDGE_CRITERION = 1.0e-6

GUARD_POLICY = A9.GUARD_POLICY
GUARD_CIRCULARITY_NOTE = A9.GUARD_CIRCULARITY_NOTE


# --------------------------------------------------------------------------- #
def _write(path, obj):
    return C._write(path, obj)


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


def _key(f, mg):
    return "{:.10g}|{:.10g}".format(float(f), float(mg))


def _tag_path(tag):
    return DIAG / f"_a10_scan_{tag}.jsonl"


def _all_checkpoints():
    """Both families: the main cube (_a10_scan_j_c*) and the refinement rows
    (_a10_scan_jr_c*).  Rows are keyed on the physical (f, dmu_G), so the two
    merge with no bookkeeping beyond this glob."""
    return sorted(DIAG.glob("_a10_scan_j*_c*of*.jsonl"))


def load_done(exclude=None):
    done, headers, dropped = {}, {}, 0
    for path in _all_checkpoints():
        if exclude is not None and path == Path(exclude):
            continue
        hdr, rows, drop = _read_jsonl(path)
        dropped += drop
        if hdr is not None:
            headers[path.name] = hdr
        for r in rows:
            done.setdefault(_key(r["f_agn"], r["dmu_G"]), (path.name, r))
    return done, headers, dropped


def row_order(nodes="registered"):
    """The (f, dmu_G) rows for one node set.

    'registered' is the 861-row main cube with the A10-chi slab (dmu_G = 0)
    dealt FIRST -- unchanged, byte for byte, because the running array's chunk
    assignment is ``index % n_chunks``.  'refine' is the 246 additive rows of
    the one allowed refinement; dmu_G = 0 is not among them, so arm A10-chi is
    untouched.
    """
    if nodes == "refine":
        return [(float(f), float(g)) for g in MG_REFINE for f in F_GRID]
    zero = [(float(f), 0.0) for f in F_GRID]
    rest = [(float(f), float(g)) for g in MG_GRID if g != 0.0 for f in F_GRID]
    return zero + rest


def _header(cell, tag, extra=None, nodes="registered"):
    hdr = {
        "record": "header",
        "analysis": "analysis_10_mass_spin_marked_multitracer",
        "arm": "A10-J",
        "tag": tag,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "provenance": A8.provenance(gw_path=GW_PATH_A10,
                                    survey_paths=cell.survey_paths),
        "darksirens_sha": A9.DARKSIRENS_A8_SHA,
        "events_file": GW_PATH_A10, "events_md5": C.GW_MD5_A10,
        "selection_file": str(cell.opts.gwselection_path),
        "survey_paths": [str(s) for s in cell.survey_paths],
        "sampled_labels": list(cell.labels),
        "per_catalog_pop_params": list(cell.per_catalog_pop_params),
        "fixed_parameter_values": {k: float(v)
                                   for k, v in cell.fixed_parameter_values.items()},
        "base_coord": {k: float(v) for k, v in zip(cell.labels, cell.base)},
        "free_parameters": ["fcat_2", L_MU, L_MG],
        "H0_fixed": A8.H0_FID, "Om0_fixed": A8.OM0_FID,
        "nEvents": int(cell.data["nEvents"]), "nsamp": int(cell.data["nsamp"]),
        "Ndraw": float(cell.data["Ndraw"]),
        "f_grid": F_GRID.tolist(),
        "dmu_chi_grid": MU_GRID.tolist(),
        "dmu_G_grid": np.asarray(NODE_SETS[nodes], dtype=float).tolist(),
        "dmu_G_node_set": nodes,
        "dmu_G_grid_registered": MG_GRID.tolist(),
        "dmu_G_grid_refinement": MG_REFINE.tolist(),
        "dmu_G_grid_merged": MG_FULL.tolist(),
        "dmu_G_zero_index_on_registered_axis": J_CHI_SLAB,
        "truth": dict(C.TRUTH),
        "tolerances": dict(C.TOL),
    }
    if extra:
        hdr.update(extra)
    return hdr


def eval_row(cell, f, mg):
    """One checkpoint unit: the 61 dmu_chi cells at fixed (f, dmu_G)."""
    cells = []
    for mu in MU_GRID:
        rec = cell.evaluate_at(fcat_2=float(f), dmu_chi=float(mu),
                               dmu_G=float(mg))
        row = GC._cell_record(rec)
        row["logL_hex"] = rec["logL_hex"]
        row["dmu_chi"] = float(mu)
        cells.append(row)
    return cells


# =========================================================================== #
# Stage: scan (GPU, rita)
# =========================================================================== #
def stage_scan(args):
    env = A9._gpu_setup("scan")
    env["a10_inputs"] = C.assert_a10_inputs(with_md5=False)
    if not (0 <= args.chunk < args.n_chunks):
        raise SystemExit(f"[fatal] --chunk must be in [0, {args.n_chunks})")
    nodes = args.dmuG_nodes
    stem = "j" if nodes == "registered" else "jr"
    tag = f"{stem}_c{args.chunk}of{args.n_chunks}"
    path = _tag_path(tag)

    order = row_order(nodes)
    mine = [x for i, x in enumerate(order) if i % args.n_chunks == args.chunk]
    print(f"chunk {args.chunk}/{args.n_chunks} [{nodes}]: {len(mine)} rows "
          f"({len(mine) * MU_GRID.size} cells)"
          + ("; the dmu_G = 0 (A10-chi) rows come first"
             if nodes == "registered" else
             f"; dmu_G nodes {MG_REFINE.tolist()}"))

    done_elsewhere, _headers, dropped = load_done(exclude=path)
    hdr_self, rows_self, drop_self = _read_jsonl(path)
    done_self = {_key(r["f_agn"], r["dmu_G"]): r for r in rows_self}
    print(f"checkpoint {path.name}: {len(done_self)} rows already here, "
          f"{len(done_elsewhere)} in sibling checkpoints, "
          f"{dropped + drop_self} truncated line(s) dropped")

    todo = [(f, g) for (f, g) in mine
            if _key(f, g) not in done_self and _key(f, g) not in done_elsewhere]
    print(f"rows to do in this worker: {len(todo)} "
          f"({len(todo) * MU_GRID.size} cells)")
    if not todo:
        print("nothing to do")
        return

    t_build = time.time()
    cell = A10.build_a10(f"A10_J_c{args.chunk}", [A8.SURVEY_GAL, A8.SURVEY_AGN],
                         verbose=True, gw_path=GW_PATH_A10)
    C.assert_cell(cell, A10.EXPECTED_LABELS_A10, f"A10-J chunk {args.chunk}")
    print(f"build: {time.time() - t_build:.1f}s")
    if hdr_self is None:
        _append_jsonl(path, _header(cell, tag, nodes=nodes, extra={
            "chunk": args.chunk, "n_chunks": args.n_chunks,
            "rows_this_worker": len(mine),
            "jax_devices": str(__import__("jax").devices()),
        }))

    t0 = time.time()
    for n, (f, g) in enumerate(todo, start=1):
        cells = eval_row(cell, f, g)
        _append_jsonl(path, {
            "record": "row", "f_agn": float(f), "dmu_G": float(g),
            "mu_G_c2": A10.dmuG_to_label(g), "key": _key(f, g), "cells": cells,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        lls = np.array([c["logL"] for c in cells], dtype=float)
        fin = np.isfinite(lls)
        el = time.time() - t0
        print(f"[A10-J/{nodes}] row {n}/{len(todo)} f={f:.4f} dmu_G={g:+.2f} "
              f"maxlogL={(lls[fin].max() if fin.any() else float('nan')):.4f} "
              f"rejected={int((~fin).sum())}/{MU_GRID.size} "
              f"elapsed={el/60:.1f}min eta={el/n*(len(todo)-n)/60:.1f}min")
        sys.stdout.flush()
        if args.stop_after_s and el > args.stop_after_s:
            print(f"[A10-J] stopping cleanly at the requested budget; "
                  f"{len(todo) - n} rows left")
            break
    print(f"[A10-J] done -> {path}")


# =========================================================================== #
# Stage: status (CPU)
# =========================================================================== #
def stage_status(args):
    done, headers, dropped = load_done()
    want = {_key(f, g) for f, g in
            (row_order("registered") + row_order("refine"))}
    have = want & set(done)
    per_g = {f"{g:+.1f}": sum(1 for f in F_GRID if _key(f, g) in done)
             for g in MG_FULL}
    secs = [c["seconds"] for _, r in done.values() for c in r["cells"]]
    out = {
        "arm": "A10-J",
        "dmu_G_axis": {"registered": MG_GRID.tolist(),
                       "refinement": MG_REFINE.tolist(),
                       "merged": MG_FULL.tolist(),
                       "rows_registered": N_ROWS, "rows_refinement": N_ROWS_REFINE,
                       "rows_total": N_ROWS_FULL,
                       "cells_total": int(N_ROWS_FULL * MU_GRID.size)},
        "rows_total": len(want), "rows_done": len(have),
        "cells_total": len(want) * MU_GRID.size,
        "cells_done": len(have) * MU_GRID.size,
        "fraction": len(have) / max(len(want), 1),
        "rows_done_off_grid": len(set(done) - want),
        "truncated_lines_dropped": dropped,
        "rows_per_dmu_G": per_g,
        "rows_done_registered": sum(1 for f, g in row_order("registered")
                                    if _key(f, g) in done),
        "rows_done_refinement": sum(1 for f, g in row_order("refine")
                                    if _key(f, g) in done),
        "A10_chi_slab_rows_done": per_g.get("+0.0", 0),
        "A10_chi_slab_rows_total": int(F_GRID.size),
        "checkpoints": [p.name for p in _all_checkpoints()],
        "headers": {k: {"chunk": v.get("chunk"), "host": v.get("host"),
                        "slurm_job_id": v.get("slurm_job_id"),
                        "darksirens_sha": v.get("darksirens_sha")}
                    for k, v in headers.items()},
        "median_seconds_per_cell": float(np.median(secs)) if secs else None,
        "gpu_hours_spent": float(np.sum(secs) / 3600.0) if secs else 0.0,
    }
    if secs:
        out["gpu_hours_remaining"] = float(
            (len(want) - len(have)) * MU_GRID.size * np.median(secs) / 3600.0)
    print(json.dumps(out, indent=2, default=GC._json_default))
    return out


# =========================================================================== #
# Stage: assemble (CPU)
# =========================================================================== #
def _cube(done, mg=None):
    """(f, dmu_G, dmu_chi) -> logL, finite, puller, on the assemble axis."""
    mg = _MG_AXIS if mg is None else np.asarray(mg, dtype=float)
    shape = (F_GRID.size, mg.size, MU_GRID.size)
    ll = np.full(shape, np.nan)
    fin = np.zeros(shape, dtype=bool)
    hexes = np.empty(shape, dtype=object)
    store, missing = {}, []
    for i, f in enumerate(F_GRID):
        for k, g in enumerate(mg):
            hit = done.get(_key(f, g))
            if hit is None:
                missing.append({"f_agn": float(f), "dmu_G": float(g)})
                continue
            cells = hit[1]["cells"]
            store[(i, k)] = cells
            for j, c in enumerate(cells):
                ll[i, k, j] = c["logL"]
                fin[i, k, j] = bool(c["finite"])
                hexes[i, k, j] = c.get("logL_hex")

    def pull(key, default=np.nan):
        out = np.full(shape, default, dtype=float)
        for (i, k), cells in store.items():
            for j, c in enumerate(cells):
                v = c.get(key)
                out[i, k, j] = default if v is None else v
        return out

    return ll, fin, hexes, pull, missing


def _truths_f(block):
    """The f_AGN scoring convention: one TARGET, one descriptive number."""
    GC._truth_flags(block, {"planted": C.TRUTH["f_agn_planted"]})
    GC._truth_flags(block,
                    {"detected_set_fraction": C.TRUTH["f_agn_detected_fraction"]})
    block["truth_convention"] = C.TRUTH["f_agn_convention"]
    return block


def _refine_smoothness(mg, values, name):
    """Does each refinement node sit where the coarse axis said it would?

    Two measured statements: the LINEAR-INTERPOLATION RESIDUAL at each 0.5 Msun
    node (measured minus the mean of its two integer neighbours -- what the
    coarse axis implicitly assumed), and the SECOND DIFFERENCES on the coarse
    sub-axis (h = 1) against the refined sub-axis (h = 0.5) inside [2, 8], where
    a smooth function gives Delta^2 ~ h^2 f'' and hence a ratio near 1/4.
    """
    by = {round(float(x), 6): float(y) for x, y in zip(np.asarray(mg),
                                                       np.asarray(values))}
    rows = []
    for g in MG_REFINE:
        k, lo, hi = round(float(g), 6), round(float(g) - 0.5, 6), round(float(g) + 0.5, 6)
        if k not in by or lo not in by or hi not in by:
            continue
        lin = 0.5 * (by[lo] + by[hi])
        rows.append({"dmu_G": float(g), "measured": by[k],
                     "linear_from_integer_neighbours": lin,
                     "residual": by[k] - lin, "neighbours": [by[lo], by[hi]]})
    coarse = np.array([by[round(float(x), 6)] for x in MG_GRID
                       if round(float(x), 6) in by])
    d2c = np.diff(coarse, 2) if coarse.size >= 3 else np.array([np.nan])
    inner = np.array(sorted(k for k in by if 2.0 - 1e-9 <= k <= 8.0 + 1e-9))
    d2f = (np.diff(np.array([by[k] for k in inner]), 2) if inner.size >= 3
           else np.array([np.nan]))
    mc, mf = float(np.nanmedian(np.abs(d2c))), float(np.nanmedian(np.abs(d2f)))
    return {
        "quantity": name,
        "per_refinement_node": rows,
        "max_abs_linear_interpolation_residual": (
            float(max(abs(r["residual"]) for r in rows)) if rows else float("nan")),
        "median_abs_linear_interpolation_residual": (
            float(np.median([abs(r["residual"]) for r in rows])) if rows
            else float("nan")),
        "second_difference_coarse_h1": {
            "values": d2c.tolist(), "max_abs": float(np.nanmax(np.abs(d2c))),
            "median_abs": mc},
        "second_difference_refined_h0p5_inside_2_to_8": {
            "nodes": inner.tolist(), "values": d2f.tolist(),
            "max_abs": float(np.nanmax(np.abs(d2f))), "median_abs": mf},
        "h2_scaling_check": {
            "note": ("for a smooth function Delta^2 ~ h^2 f'', so the h = 0.5 "
                     "second differences should be about a QUARTER of the h = 1 "
                     "ones over the same interval"),
            "ratio_median_refined_over_coarse": (mf / mc if mc else float("nan")),
            "expected_if_smooth": 0.25},
    }


def _duplicate_report():
    seen, dups = {}, []
    for path in _all_checkpoints():
        _hdr, rows, _d = _read_jsonl(path)
        for r in rows:
            k = _key(r["f_agn"], r["dmu_G"])
            if k in seen:
                a = np.array([c["logL"] for c in seen[k][1]["cells"]], dtype=float)
                b = np.array([c["logL"] for c in r["cells"]], dtype=float)
                both = np.isfinite(a) & np.isfinite(b)
                dups.append({
                    "key": k, "files": [seen[k][0], path.name],
                    "max_abs_delta_logL": (float(np.abs(a[both] - b[both]).max())
                                           if both.any() else None),
                    "same_rejected_mask": bool(np.array_equal(np.isfinite(a),
                                                              np.isfinite(b))),
                })
            else:
                seen[k] = (path.name, r)
    return dups


NAMES = ("f_agn", "dmu_G", "dmu_chi")

# The dmu_G axis the ASSEMBLE path integrates over.  The scan path never reads
# it: a worker is told its node set on the command line.  Default is the merged
# 27-node axis (the registered 21 plus the 6 refinement nodes); a restriction to
# the registered 21 is how the coarse-axis comparison is produced.
_MG_AXIS = MG_FULL


def _set_axis(a):
    global _MG_AXIS
    _MG_AXIS = np.asarray(a, dtype=float)
    return _MG_AXIS


def _marginals_3d(ll, fin, marginal_ci, mg=None):
    mg = _MG_AXIS if mg is None else np.asarray(mg, dtype=float)
    grids = [F_GRID, mg, MU_GRID]
    llm = np.where(fin, ll, -np.inf)
    if not np.isfinite(llm).any():
        raise RuntimeError("[fatal] every cell is rejected")
    mx = float(llm[np.isfinite(llm)].max())
    P = np.where(np.isfinite(llm), np.exp(llm - mx), 0.0)

    m_f = np.trapz(np.trapz(P, MU_GRID, axis=2), mg, axis=1)
    m_g = np.trapz(np.trapz(P, MU_GRID, axis=2), F_GRID, axis=0)
    m_mu = np.trapz(np.trapz(P, mg, axis=1), F_GRID, axis=0)

    out = {"logL_max": mx}
    with np.errstate(divide="ignore"):
        for name, x, m in (("f", F_GRID, m_f), ("dmu_G", mg, m_g),
                           ("dmu_chi", MU_GRID, m_mu)):
            lp = np.log(m)
            out[name] = marginal_ci(x, lp)
            out[name]["marginal_logp"] = lp.tolist()
            out[name]["marginal_mode"] = float(x[int(np.argmax(lp))])

    # _trapz_weights is the general NON-UNIFORM trapezoid rule
    # (w_i = (x_{i+1} - x_{i-1}) / 2), so the merged 1 / 0.5 Msun axis needs no
    # special handling -- verified against np.trapz on this very axis.
    W = (GC._trapz_weights(F_GRID)[:, None, None]
         * GC._trapz_weights(mg)[None, :, None]
         * GC._trapz_weights(MU_GRID)[None, None, :]) * P
    Z = W.sum()
    G = np.meshgrid(*grids, indexing="ij")
    mean = [float((W * g).sum() / Z) for g in G]
    cov = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            cov[a, b] = float((W * (G[a] - mean[a]) * (G[b] - mean[b])).sum() / Z)
    sd = np.sqrt(np.diag(cov))
    out["posterior_moments"] = {
        "names": list(NAMES),
        "mean": {n: m for n, m in zip(NAMES, mean)},
        "sd": {n: float(s) for n, s in zip(NAMES, sd)},
        "cov": cov.tolist(),
        "correlation": {f"{NAMES[a]}|{NAMES[b]}":
                        float(cov[a, b] / (sd[a] * sd[b]))
                        for a in range(3) for b in range(a + 1, 3)},
    }
    i, k, j = np.unravel_index(np.argmax(llm), llm.shape)
    out["map"] = {"f_agn": float(F_GRID[i]), "dmu_G": float(mg[k]),
                  "dmu_chi": float(MU_GRID[j]), "logL": float(llm[i, k, j]),
                  "index": [int(i), int(k), int(j)]}
    out["marginals_2d"] = {
        "f_dmu_G": np.trapz(P, MU_GRID, axis=2),
        "f_dmu_chi": np.trapz(P, mg, axis=1),
        "dmu_G_dmu_chi": np.trapz(P, F_GRID, axis=0),
    }
    out["_P"] = P
    return out


def _edge_mass(marg, mg=None):
    mg = _MG_AXIS if mg is None else np.asarray(mg, dtype=float)
    out = {}
    for name, x, key in (("f_agn", F_GRID, "f"), ("dmu_G", mg, "dmu_G"),
                         ("dmu_chi", MU_GRID, "dmu_chi")):
        lp = np.asarray(marg[key]["marginal_logp"], dtype=float)
        p = np.exp(lp - np.nanmax(lp[np.isfinite(lp)]))
        out[name] = {"low_edge_density_over_peak": float(p[0]),
                     "high_edge_density_over_peak": float(p[-1]),
                     "low_edge_value": float(x[0]), "high_edge_value": float(x[-1]),
                     "criterion": f"both <= {EDGE_CRITERION:g} of the peak",
                     "contained": bool(p[0] <= EDGE_CRITERION
                                       and p[-1] <= EDGE_CRITERION)}
    return out


def _guard_report_cube(ll, fin, pull, mg=None):
    mg = _MG_AXIS if mg is None else np.asarray(mg, dtype=float)
    Neff, thr = pull("Neff"), pull("threshold")
    rejected = ~fin
    rep = {
        "n_cells": int(ll.size), "n_rejected": int(rejected.sum()),
        "rejected_fraction": float(rejected.mean()),
        "Neff_min": float(np.nanmin(Neff)), "Neff_median": float(np.nanmedian(Neff)),
        "Neff_max": float(np.nanmax(Neff)),
        "threshold_min": float(np.nanmin(thr)), "threshold_max": float(np.nanmax(thr)),
        "Neff_over_threshold_min": float(np.nanmin(Neff / thr)),
        "policy": GUARD_POLICY,
        "posterior_mass_in_rejected_region_as_scanned": 0.0,
        "posterior_mass_note": GUARD_CIRCULARITY_NOTE,
        "rejected_cells": [],
    }
    idx = np.argwhere(rejected)
    for i, k, j in idx[:5000]:
        rep["rejected_cells"].append({
            "f_agn": float(F_GRID[i]), "dmu_G": float(mg[k]),
            "dmu_chi": float(MU_GRID[j]), "Neff": float(Neff[i, k, j]),
            "threshold": float(thr[i, k, j]),
            "Neff_over_threshold": float(Neff[i, k, j] / thr[i, k, j])})
    if idx.size:
        rep["rejected_f_min"] = float(F_GRID[idx[:, 0].min()])
        rep["rejected_dmu_G_range"] = [float(mg[idx[:, 1].min()]),
                                       float(mg[idx[:, 1].max()])]
        rep["rejected_dmu_chi_min"] = float(MU_GRID[idx[:, 2].min()])

    llm = np.where(fin, ll, -np.inf)
    mx = float(llm[np.isfinite(llm)].max())
    P = np.where(np.isfinite(llm), np.exp(llm - mx), 0.0)
    W = (GC._trapz_weights(F_GRID)[:, None, None]
         * GC._trapz_weights(mg)[None, :, None]
         * GC._trapz_weights(MU_GRID)[None, None, :])
    if rep["n_rejected"] == 0:
        rep["upper_bound_rejected_mass_fraction"] = 0.0
        rep["passes"] = True
        return rep
    nb, local = [], []
    for i, k, j in idx:
        here = []
        for di in (-1, 0, 1):
            for dk in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    x, y, z = i + di, k + dk, j + dj
                    if (0 <= x < P.shape[0] and 0 <= y < P.shape[1]
                            and 0 <= z < P.shape[2] and fin[x, y, z]):
                        here.append(float(P[x, y, z]))
        nb.extend(here)
        local.append(max(here) if here else 0.0)
    P_bound = float(max(nb)) if nb else 0.0
    P_filled = P.copy()
    P_filled[rejected] = P_bound
    rep["boundary_max_density_relative_to_peak"] = P_bound
    rep["boundary_max_delta_logL_below_peak"] = (float(np.log(P_bound))
                                                 if P_bound > 0 else float("-inf"))
    rep["upper_bound_rejected_mass_fraction"] = float(
        (W * np.where(rejected, P_filled, 0.0)).sum() / (W * P_filled).sum())
    rep["accepted_mass_normalisation"] = float((W * P).sum())
    rep["fill_is_an_over_estimate"] = {
        "fill_value_relative_to_peak": P_bound,
        "n_rejected_cells": int(len(local)),
        "n_cells_where_fill_ge_local_accepted_boundary_max":
            int(sum(1 for v in local if P_bound >= v)),
        "max_local_accepted_boundary_density": float(max(local)) if local else 0.0,
        "verified": bool(all(P_bound >= v for v in local)),
    }
    mono = []
    for i, k in sorted({(int(x[0]), int(x[1])) for x in idx}):
        col = llm[i, k]
        acc = np.where(np.isfinite(col))[0]
        acc = acc[acc >= int(np.argmax(np.where(np.isfinite(col), col, -np.inf)))]
        tail = acc[-4:] if acc.size >= 4 else acc
        d = np.diff(col[tail]) if tail.size > 1 else np.array([np.nan])
        mono.append({"f_agn": float(F_GRID[i]), "dmu_G": float(mg[k]),
                     "last_accepted_dmu_chi": (float(MU_GRID[acc[-1]])
                                               if acc.size else None),
                     "last_diffs_logL": d.tolist(),
                     "monotonically_falling": bool(np.all(d < 0))})
    rep["falling_towards_the_guard"] = mono
    rep["all_columns_falling"] = bool(all(m["monotonically_falling"] for m in mono))
    rep["passes"] = bool(rep["upper_bound_rejected_mass_fraction"] < 1e-6)
    return rep


def _write_chi_arm(hdr, ll, fin, pull, env, marginal_ci, mg=None):
    """The dmu_G = 0 slab: arm A10-chi, Analysis 8's arm J on the two-mark mock.

    The refinement adds no node at dmu_G = 0, so this slab is EXACTLY the one the
    main array computed; only its index on the merged axis is looked up here.
    """
    import h5py
    mg = _MG_AXIS if mg is None else np.asarray(mg, dtype=float)
    k = int(np.argmin(np.abs(mg - 0.0)))
    if mg[k] != 0.0:
        raise SystemExit("[fatal] the dmu_G axis has no exact 0 node")
    llc, finc = ll[:, k, :], fin[:, k, :]

    def pullc(key):
        return pull(key)[:, k, :]

    marg = GC._marginals(F_GRID, MU_GRID, llc, finc, marginal_ci)
    marg["dmu_chi"] = marg["mu_chi_c2"]
    _truths_f(marg["f"])
    GC._truth_flags(marg["dmu_chi"], {"planted": C.TRUTH["dmu_chi_planted"],
                                      "realised": C.TRUTH["dmu_chi_realised"]})
    guard = GC._guard_report("A10-chi", F_GRID, MU_GRID, llc, finc, pullc)
    llm = np.where(finc, llc, -np.inf)
    mx = float(np.nanmax(llm[np.isfinite(llm)]))
    P = np.where(np.isfinite(llm), np.exp(llm - mx), 0.0)
    edges = {}
    for name, x, key in (("f_agn", F_GRID, "f"), ("dmu_chi", MU_GRID, "dmu_chi")):
        lp = np.asarray(marg[key]["marginal_logp"], dtype=float)
        p = np.exp(lp - np.nanmax(lp[np.isfinite(lp)]))
        edges[name] = {"low_edge_density_over_peak": float(p[0]),
                       "high_edge_density_over_peak": float(p[-1]),
                       "criterion": f"both <= {EDGE_CRITERION:g} of the peak",
                       "contained": bool(p[0] <= EDGE_CRITERION
                                         and p[-1] <= EDGE_CRITERION)}
    summary = {
        "analysis": "analysis_10_mass_spin_marked_multitracer",
        "arm": "A10-chi",
        "what": ("the dmu_G = 0 slab of the A10-J cube: the spin mark alone on "
                 "the two-mark mock, i.e. Analysis 8's arm J re-run on this "
                 "events file.  Closure 15.2 is the statement that the "
                 "two-mark model reduces to the spin-only model here."),
        "file": "results/a10_arm_chi.h5",
        "slab_index_on_dmu_G_axis": k,
        "dmu_G_axis_used": mg.tolist(),
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "environment": env,
        "provenance": hdr.get("provenance"),
        "events_file": GW_PATH_A10, "events_md5": C.GW_MD5_A10,
        "selection_file": hdr.get("selection_file"),
        "survey_paths": hdr.get("survey_paths"),
        "sampled_labels": hdr.get("sampled_labels"),
        "base_coord": hdr.get("base_coord"),
        "fixed_parameter_values": hdr.get("fixed_parameter_values"),
        "free_parameters": ["fcat_2", L_MU],
        "dmu_G_held_at": 0.0,
        "H0_fixed": A8.H0_FID, "Om0_fixed": A8.OM0_FID,
        "grid": {"f_agn": [float(F_GRID[0]), float(F_GRID[-1]), int(F_GRID.size)],
                 "dmu_chi": [float(MU_GRID[0]), float(MU_GRID[-1]),
                             int(MU_GRID.size)]},
        "truth": dict(C.TRUTH),
        "n_cells": int(llc.size), "n_rejected": int((~finc).sum()),
        "logL_max": marg["logL_max"], "map": marg["map"],
        "f": marg["f"], "dmu_chi": marg["dmu_chi"],
        "posterior_moments": marg.get("posterior_moments"),
        "edge_mass": edges,
        "guard": {kk: v for kk, v in guard.items() if kk != "rejected_cells"},
        "guard_rejected_cells": guard.get("rejected_cells", []),
    }
    h5_path = RESULTS / "a10_arm_chi.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("f_grid", data=F_GRID)
        h5.create_dataset("dmu_chi_grid", data=MU_GRID)
        h5.create_dataset("log_likelihood", data=llc)
        h5.create_dataset("posterior_unnormalised", data=P)
        g = h5.create_group("guard")
        for key in ("Neff", "threshold", "pe_variance_sum", "sigma2_total",
                    "logL_selection", "logL_pe", "log_mu", "seconds"):
            g.create_dataset(key, data=pullc(key))
        g.create_dataset("rejected", data=(~finc))
        m = h5.create_group("marginal")
        m.create_dataset("f", data=np.exp(np.asarray(marg["f"]["marginal_logp"])))
        m.create_dataset("dmu_chi",
                         data=np.exp(np.asarray(marg["dmu_chi"]["marginal_logp"])))
        h5.attrs["arm"] = "A10-chi"
        h5.attrs["truth"] = json.dumps(C.TRUTH)
        h5.attrs["labels"] = json.dumps(hdr.get("sampled_labels"))
        h5.attrs["base_coord"] = json.dumps(hdr.get("base_coord"))
        h5.attrs["dmu_G_held_at"] = 0.0
        h5.attrs["H0_fixed"] = A8.H0_FID
        h5.attrs["Om0_fixed"] = A8.OM0_FID
        h5.attrs["events_file"] = GW_PATH_A10
        h5.attrs["events_md5"] = C.GW_MD5_A10
        h5.attrs["darksirens_sha"] = A9.DARKSIRENS_A8_SHA
    print(f"wrote {h5_path}")
    _write(RESULTS / "a10_arm_chi.json", summary)
    return summary


def _closure_152_recheck():
    """FREE CHECK: the cube's dmu_G = 0 slab against the 15.2 spin-only values.

    Closure 15.2 evaluated the spin-only (Analysis-8) model at 5 f x 4 dmu_chi
    nodes.  Only the nodes that fall EXACTLY on the cube lattice are comparable
    -- f = 0.357 is not a multiple of 0.025 and dmu_chi = -0.10 / 0.0 are not
    multiples of 0.0075 -- so the off-lattice cells are listed as such rather
    than compared at an interpolated coordinate.  Residuals in ULP of the
    compared value, as the rest of Gate 15 quotes them.
    """
    path = DIAG / "_a10_closure_main.json"
    if not path.exists():
        return {"available": False,
                "why": f"{path.name} not found; run the closure stage first"}
    main = json.loads(path.read_text())
    rows = main.get("ledger_SPIN_ONLY", [])
    out = {"available": True, "source": path.name,
           "description": ("the dmu_G = 0 slab of the A10-J cube against the "
                           "closure-15.2 spin-only ledger, at the registered "
                           "15.2 nodes that lie exactly on the cube lattice"),
           "on_lattice": [], "off_lattice": []}
    # A node "is" the 15.2 coordinate only if it agrees to 1e-12; the grids are
    # linspaces, so +0.10 lands 1 ULP (2.8e-17) off the Python float 0.1 while
    # 0.0 and -0.10 are not dmu_chi nodes at all and f = 0.357 is not an f node.
    atol = 1.0e-12
    out["coordinate_match_atol"] = atol
    for r in rows:
        f, mu = float(r["f_agn"]), float(r["dmu_chi"])
        i = int(np.argmin(np.abs(F_GRID - f)))
        j = int(np.argmin(np.abs(MU_GRID - mu)))
        df, dmu = float(F_GRID[i] - f), float(MU_GRID[j] - mu)
        if abs(df) > atol or abs(dmu) > atol:
            out["off_lattice"].append({
                "f_agn": f, "dmu_chi": mu, "logL_spin_only": r["logL"],
                "nearest_f_node": float(F_GRID[i]),
                "nearest_dmu_chi_node": float(MU_GRID[j]),
                "delta_f": df, "delta_dmu_chi": dmu,
                "why": ("f is not a node of the 0.025 f axis"
                        if abs(df) > atol else
                        "dmu_chi is not a node of the 0.0075 axis")})
            continue
        out["on_lattice"].append({"f_agn": f, "dmu_chi": mu,
                                  "i_f": i, "j_dmu_chi": j,
                                  "node_f": float(F_GRID[i]),
                                  "node_dmu_chi": float(MU_GRID[j]),
                                  "delta_f": df, "delta_dmu_chi": dmu,
                                  "logL_spin_only": r["logL"],
                                  "logL_pe_spin_only": r.get("logL_pe"),
                                  "logL_selection_spin_only": r.get(
                                      "logL_selection"),
                                  "log_mu_spin_only": r.get("log_mu")})
    return out


def _closure_152_compare(recheck, ll, fin, pull):
    """Fill in the cube side of :func:`_closure_152_recheck` and judge it."""
    if not recheck.get("available"):
        return recheck
    k = J_CHI_SLAB
    pe, sel, lmu = pull("logL_pe"), pull("logL_selection"), pull("log_mu")
    blocks = []
    for rec in recheck["on_lattice"]:
        i, j = rec["i_f"], rec["j_dmu_chi"]
        cube = {"logL": float(ll[i, k, j]), "logL_pe": float(pe[i, k, j]),
                "logL_selection": float(sel[i, k, j]),
                "log_mu": float(lmu[i, k, j]), "finite": bool(fin[i, k, j])}
        ref = {"logL": rec["logL_spin_only"],
               "logL_pe": rec["logL_pe_spin_only"],
               "logL_selection": rec["logL_selection_spin_only"],
               "log_mu": rec["log_mu_spin_only"]}
        blk = C._cmp4(cube, ref)
        blk.update({"f_agn": rec["f_agn"], "dmu_chi": rec["dmu_chi"],
                    "cube": cube, "spin_only": ref})
        blocks.append(blk)
    recheck["per_cell"] = blocks
    recheck["n_compared"] = len(blocks)
    recheck["n_off_lattice"] = len(recheck["off_lattice"])
    recheck["worst"] = C._worst4(blocks) if blocks else None
    recheck["n_bitwise_identical_total"] = int(sum(
        1 for b in blocks if b["total"]["bitwise_identical"]))
    recheck["tolerance_ulp"] = C.TOL["ulp_same_hardware"]
    recheck["tolerance_abs"] = C.TOL["abs_logL"]
    recheck["pass"] = bool(
        blocks
        and recheck["worst"]["max_abs_diff_any_term"] <= C.TOL["abs_logL"]
        and recheck["worst"]["max_ulp_any_term"] <= C.TOL["ulp_same_hardware"])
    return recheck


def stage_assemble(args):
    import h5py
    env = A9._cpu_setup("assemble", import_darksirens=False)
    done, headers, dropped = load_done()

    shas = {h.get("darksirens_sha") for h in headers.values()}
    if shas and shas != {A9.DARKSIRENS_A8_SHA}:
        raise RuntimeError(f"[fatal] checkpoints carry SHA {shas}")
    md5s = {h.get("events_md5") for h in headers.values()}
    if md5s and md5s != {C.GW_MD5_A10}:
        raise RuntimeError(f"[fatal] checkpoints carry events md5 {md5s}")

    mg = _set_axis(MG_FULL)
    ll, fin, hexes, pull, missing = _cube(done, mg)
    if missing and not args.allow_partial:
        raise RuntimeError(
            f"[fatal] {len(missing)} of {N_ROWS_FULL} rows are missing; run the "
            f"remaining chunks or pass --allow_partial for a coverage report "
            f"only.  First missing: {missing[:5]}")
    if missing:
        _write(DIAG / "a10_j_coverage.json",
               {"rows_missing": missing, "rows_total": N_ROWS_FULL,
                "checkpoints": [p.name for p in _all_checkpoints()]})
        print(f"[partial] {len(missing)} rows missing; wrote the coverage report")
        return

    scan_h0f = A8.import_scan_h0f()
    marginal_ci = scan_h0f.marginal_ci
    hdr = next(iter(headers.values())) if headers else {}

    recheck = _closure_152_compare(_closure_152_recheck(), ll, fin, pull)
    if recheck.get("available"):
        w = recheck["worst"]
        print(f"[15.2 recheck] the dmu_G = 0 slab against the spin-only ledger: "
              f"{recheck['n_compared']} on-lattice cells, "
              f"{recheck['n_bitwise_identical_total']} bitwise identical, worst "
              f"{w['max_abs_diff_any_term']:.3e} abs / "
              f"{w['max_ulp_any_term']:.2f} ULP "
              f"({recheck['n_off_lattice']} of the 20 registered 15.2 cells are "
              f"off the cube lattice and are listed, not compared)")

    chi = _write_chi_arm(hdr, ll, fin, pull, env, marginal_ci, mg)
    chi["closure_15_2_recheck"] = recheck
    _write(RESULTS / "a10_arm_chi.json", chi)

    marg = _marginals_3d(ll, fin, marginal_ci, mg)
    P = marg.pop("_P")
    m2 = marg.pop("marginals_2d")
    _truths_f(marg["f"])
    GC._truth_flags(marg["dmu_chi"], {"planted": C.TRUTH["dmu_chi_planted"],
                                      "realised": C.TRUTH["dmu_chi_realised"]})
    GC._truth_flags(marg["dmu_G"], {"planted": C.TRUTH["dmu_G_planted"]})
    guard = _guard_report_cube(ll, fin, pull, mg)
    edges = _edge_mass(marg, mg)
    dups = _duplicate_report()

    # ---- the one allowed refinement, MEASURED against the coarse axis ------ #
    keep = np.array([bool(np.any(np.isclose(MG_GRID, g, atol=1e-9))) for g in mg])
    mg_c = mg[keep]
    marg_c = _marginals_3d(ll[:, keep, :], fin[:, keep, :], marginal_ci, mg_c)
    marg_c.pop("_P", None); marg_c.pop("marginals_2d", None)
    _truths_f(marg_c["f"])
    GC._truth_flags(marg_c["dmu_chi"], {"planted": C.TRUTH["dmu_chi_planted"],
                                        "realised": C.TRUTH["dmu_chi_realised"]})
    GC._truth_flags(marg_c["dmu_G"], {"planted": C.TRUTH["dmu_G_planted"]})
    w = lambda b, l: float(b[l][1] - b[l][0])
    refinement = {
        "description": ("the ONE allowed grid refinement: six 0.5 Msun dmu_G "
                        "nodes inside [2, 8] added as ADDITIVE ROWS.  The same "
                        "cells restricted to the registered 21-node axis are "
                        "reported beside the refined 27-node result, so the "
                        "refinement's effect is measured rather than assumed."),
        "refinement_nodes": MG_REFINE.tolist(),
        "registered_21_node": {k: marg_c[k] for k in
                               ("logL_max", "map", "f", "dmu_G", "dmu_chi",
                                "posterior_moments")},
        "registered_21_node_edge_mass": _edge_mass(marg_c, mg_c),
        "shifts_refined_minus_coarse": {
            "dmu_G_median": marg["dmu_G"]["median"] - marg_c["dmu_G"]["median"],
            "dmu_G_width68": w(marg["dmu_G"], "ci68") - w(marg_c["dmu_G"], "ci68"),
            "dmu_G_width90": w(marg["dmu_G"], "ci90") - w(marg_c["dmu_G"], "ci90"),
            "f_median": marg["f"]["median"] - marg_c["f"]["median"],
            "f_width68": w(marg["f"], "ci68") - w(marg_c["f"], "ci68"),
            "dmu_chi_median": (marg["dmu_chi"]["median"]
                               - marg_c["dmu_chi"]["median"]),
            "dmu_chi_width68": (w(marg["dmu_chi"], "ci68")
                                - w(marg_c["dmu_chi"], "ci68")),
            "correlations": {kk: (marg["posterior_moments"]["correlation"][kk]
                                  - marg_c["posterior_moments"]["correlation"][kk])
                             for kk in marg["posterior_moments"]["correlation"]},
        },
        "smoothness": {
            "marginal_log_density": _refine_smoothness(
                mg, np.asarray(marg["dmu_G"]["marginal_logp"], dtype=float),
                "log of the dmu_G marginal density"),
            "logL_at_MAP_cell": _refine_smoothness(
                mg, np.where(fin, ll, -np.inf)[int(marg["map"]["index"][0]), :,
                                               int(marg["map"]["index"][2])],
                "logL along dmu_G through the MAP cell"),
        },
    }
    print(f"[refinement] dmu_G median {marg_c['dmu_G']['median']:.6f} (21 nodes) "
          f"-> {marg['dmu_G']['median']:.6f} (27 nodes); 68% width "
          f"{w(marg_c['dmu_G'], 'ci68'):.6f} -> {w(marg['dmu_G'], 'ci68'):.6f}")

    summary = {
        "analysis": "analysis_10_mass_spin_marked_multitracer",
        "arm": "A10-J",
        "question": ("are the two marks separable?  (f_AGN, dmu_chi, dmu_G) "
                     "from one two-mark seed-100 realisation, H0 pinned."),
        "file": "results/a10_arm_J.h5",
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "environment": env,
        "provenance": hdr.get("provenance"),
        "events_file": GW_PATH_A10, "events_md5": C.GW_MD5_A10,
        "selection_file": hdr.get("selection_file"),
        "survey_paths": hdr.get("survey_paths"),
        "sampled_labels": hdr.get("sampled_labels"),
        "per_catalog_pop_params": hdr.get("per_catalog_pop_params"),
        "base_coord": hdr.get("base_coord"),
        "fixed_parameter_values": hdr.get("fixed_parameter_values"),
        "free_parameters": ["fcat_2", L_MU, L_MG],
        "H0_fixed": A8.H0_FID, "Om0_fixed": A8.OM0_FID,
        "grid": {"f_agn": [float(F_GRID[0]), float(F_GRID[-1]), int(F_GRID.size)],
                 "dmu_G_nodes": mg.tolist(), "dmu_G_n": int(mg.size),
                 "dmu_G_spacing": "1 Msun outside [2, 8], 0.5 Msun inside",
                 "dmu_G_registered_nodes": MG_GRID.tolist(),
                 "dmu_G_refinement_nodes": MG_REFINE.tolist(),
                 "dmu_chi": [float(MU_GRID[0]), float(MU_GRID[-1]),
                             int(MU_GRID.size)]},
        "truth": dict(C.TRUTH),
        "n_cells": int(ll.size), "n_rejected": int((~fin).sum()),
        "logL_max": marg["logL_max"], "map": marg["map"],
        "f": marg["f"], "dmu_G": marg["dmu_G"], "dmu_chi": marg["dmu_chi"],
        "posterior_moments": marg["posterior_moments"],
        "correlations": marg["posterior_moments"]["correlation"],
        "edge_mass": edges,
        "guard": {k: v for k, v in guard.items() if k != "rejected_cells"},
        "guard_rejected_cells": guard.get("rejected_cells", []),
        "duplicate_rows": dups,
        "closure_15_2_recheck": recheck,
        "refinement": refinement,
        "A10_chi_slab": {"file": "results/a10_arm_chi.json",
                         "slab_index": J_CHI_SLAB,
                         "f": chi["f"], "dmu_chi": chi["dmu_chi"],
                         "map": chi["map"], "logL_max": chi["logL_max"]},
        "timing": {"cells": int(ll.size),
                   "gpu_hours": float(np.nansum(pull("seconds")) / 3600.0),
                   "median_seconds_per_cell": float(
                       np.nanmedian(pull("seconds")))},
        "checkpoints": [p.name for p in _all_checkpoints()],
        "truncated_lines_dropped": dropped,
    }

    h5_path = RESULTS / "a10_arm_J.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("f_grid", data=F_GRID)
        h5.create_dataset("dmu_G_grid", data=mg)
        h5.create_dataset("dmu_G_grid_registered", data=MG_GRID)
        h5.create_dataset("dmu_G_grid_refinement", data=MG_REFINE)
        h5.create_dataset("dmu_chi_grid", data=MU_GRID)
        h5.create_dataset("mu_G_c2_grid",
                          data=np.array([A10.dmuG_to_label(x) for x in mg]))
        h5.create_dataset("log_likelihood", data=ll)
        h5.create_dataset("posterior_unnormalised", data=P)
        g = h5.create_group("guard")
        for key in ("Neff", "threshold", "pe_variance_sum", "sigma2_total",
                    "logL_selection", "logL_pe", "log_mu", "seconds"):
            g.create_dataset(key, data=pull(key))
        g.create_dataset("rejected", data=(~fin))
        g.attrs["n_rejected"] = int((~fin).sum())
        m = h5.create_group("marginal")
        for name in ("f", "dmu_G", "dmu_chi"):
            m.create_dataset(name, data=np.exp(np.asarray(
                marg[name]["marginal_logp"])))
        for k, v in m2.items():
            m.create_dataset(k, data=v)
        h5.attrs["analysis"] = "analysis_10_mass_spin_marked_multitracer"
        h5.attrs["arm"] = "A10-J"
        h5.attrs["axis_order"] = json.dumps(list(NAMES))
        h5.attrs["free_parameters"] = json.dumps(["fcat_2", L_MU, L_MG])
        h5.attrs["labels"] = json.dumps(hdr.get("sampled_labels"))
        h5.attrs["base_coord"] = json.dumps(hdr.get("base_coord"))
        h5.attrs["truth"] = json.dumps(C.TRUTH)
        h5.attrs["H0_fixed"] = A8.H0_FID
        h5.attrs["Om0_fixed"] = A8.OM0_FID
        h5.attrs["events_file"] = GW_PATH_A10
        h5.attrs["events_md5"] = C.GW_MD5_A10
        h5.attrs["selection_file"] = str(hdr.get("selection_file"))
        h5.attrs["survey_paths"] = json.dumps(hdr.get("survey_paths"))
        h5.attrs["darksirens_sha"] = A9.DARKSIRENS_A8_SHA
        h5.attrs["settings"] = json.dumps(A8.SETTINGS)
    print(f"wrote {h5_path}")
    _write(RESULTS / "a10_arm_J.json", summary)

    for k in ("f", "dmu_G", "dmu_chi"):
        b = summary[k]
        print(f"  {k:<8} median {b['median']:.6f}  68% [{b['ci68'][0]:.6f}, "
              f"{b['ci68'][1]:.6f}]  90% [{b['ci90'][0]:.6f}, {b['ci90'][1]:.6f}]")
    print(f"  correlations: {json.dumps(summary['correlations'], indent=None)}")
    print(f"  edge mass: {json.dumps(edges, indent=None, default=str)}")
    return summary


# =========================================================================== #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True,
                    choices=("scan", "status", "assemble"))
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--dmuG_nodes", default="registered",
                    choices=("registered", "refine"))
    ap.add_argument("--n_chunks", type=int, default=8)
    ap.add_argument("--stop_after_s", type=float, default=0.0)
    ap.add_argument("--allow_partial", action="store_true")
    args = ap.parse_args(argv)
    return {"scan": stage_scan, "status": stage_status,
            "assemble": stage_assemble}[args.stage](args)


if __name__ == "__main__":
    main()
