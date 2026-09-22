#!/usr/bin/env python
"""Analysis 10 -- the COSMOLOGY-STAGE scan driver: H0 RELEASED.

Implements the design of record (``diagnostics/c10_design_notes.md``, section
4: "P -> S -> G2 (as H2)") plus the section-21 mechanism arms.  Nothing about
the likelihood is reimplemented: every cell is a call to
``a10_likelihood.build_a10`` (the two-mark model, steering a8's 'new' recipe)
and ``A10LikelihoodCell.evaluate_at``.  Analyses 8 and 9 are read-only
(``sys.dont_write_bytecode`` is set before either is reached) and nothing under
``darksirens`` is touched.

STAGES

    profile          GPU, rita.  28 H0 nodes (A9's J9 window) at the fixed-H0
                     joint MAP (f=0.275, dmu_chi=+0.115, dmu_G=+5.0) -- the
                     MARKED profile -- and the same 28 nodes with both marks at
                     zero at f=0.275 -- the SPATIAL profile.  Auto-assembles
                     diagnostics/c10_profile.json when the 28-node checkpoint is
                     complete.  ~56 cells, ~3 min, ONE worker, no array.

    s                GPU, rita.  C10-S: p(H0, f_AGN), both marks held at 0.0
                     EXACTLY, on A9's S9 lattice: H0 202 nodes ([50,100] step
                     0.25, plus 67.74) x f 41 nodes.  8,282 cells, ~6.9 GPU-h.
                     Row = the 41 f cells at one H0.  4 interleaved chunks,
                     checkpoints diagnostics/_c10_s_c{k}of4.jsonl.
    s_assemble       CPU.  -> results/c10_arm_S.{h5,json}, shaped like A9's
                     s9_spatial.{h5,json}, plus j_window_check (Gate-W-style,
                     on THIS mock) and matched_28_node_lattice (the S marginal
                     re-marginalised on the 28 default J nodes).

    j                GPU, rita.  C10-J (the reduced asymmetric cube "G2", run
                     as H2): H0 from --h0_window (default [63,76], 0.5 lattice
                     + 67.74, 28 nodes) x f 11 nodes {0.05,...,0.55} x dmu_chi
                     13 nodes {0.025,...,0.205} x dmu_G 15 nodes
                     {1,2,2.5,...,8,9}.  28x11x15 = 4,620 rows x 13 = 60,060
                     cells, ~50.4 GPU-h at 3.0143 s/eval.  Row = the 13 dmu_chi
                     cells at fixed (H0, f, dmu_G).  8 interleaved chunks,
                     checkpoints diagnostics/_c10_j_c{k}of8.jsonl.  The H0 =
                     67.74 slab is dealt FIRST (it is cell-for-cell a
                     sub-lattice of the fixed-H0 A10-J cube).
    j_assemble       CPU.  -> results/c10_arm_J.{h5,json}: the 4-D cube, six
                     2-D marginals, six correlations, MAP, edge masses on every
                     axis, guard report, the FREE 67.74-slab closure against
                     results/a10_arm_J.h5 (ULP), and the H0 width comparison
                     against C10-S (matched lattice) and, for reference only
                     (a DIFFERENT mock), against Analysis 9's
                     diagnostics/a9_h0_matched_lattice.json.

    mech             GPU, rita.  Section-21 mechanism arms, 2-D (H0, f) on the
                     28 (default window) x 41 (full F_GRID) lattice, marks
                     PINNED at --pin_dmu_chi/--pin_dmu_G (default the fixed-H0
                     MAP, +0.115/+5.0).  --arm {P1,I0,I1} selects one arm per
                     invocation:
                        P1  [GAL, AGN]  marks pinned   (total marked width)
                        I0  [GAL, GAL]  marks 0        (equalised-tracer base)
                        I1  [GAL, GAL]  marks pinned   (equalised-tracer marked)
                     P0 (marks 0, [GAL, AGN]) is the h0_window slice of C10-S
                     and is NOT computed here.  3 x 1,148 = 3,444 cells total,
                     ~2.9 GPU-h.  Row = the 41 f cells at one H0 (28 rows/arm).
                     Checkpoints diagnostics/_c10_mech_{P1,I0,I1}.jsonl.
    mech_assemble    CPU.  -> results/c10_mech.{h5,json}: per-arm H0 marginal
                     (median/MAP/68/90), W(P1)/W(P0) (total gain),
                     W(I1)/W(I0) (spectral-siren-only gain), and the residual
                     routing factor, with the interaction caveat stated.

    status           CPU.  Rows/cells done and GPU-hours spent/remaining for
                     every family that has checkpoints on disk.
    dry_run          CPU, NO DATA, NO likelihood trace.  Resolves the
                     parameter space (via a10_likelihood._selftest) and prints
                     the node sets, bitwise-lattice assertions, row/cell counts
                     and GPU-hour projections for every family named in
                     --which (default "all").

ENVIRONMENT (load-bearing; the same as every other Analysis-10/-9 driver)

    source .../analysis_9_marked_multitracer_H0_fagn/scripts/env_a9.sh

COMPUTE.  Every GPU stage runs on RITA via SLURM and refuses to run anywhere
else (checked by ``a9_scan._gpu_setup`` -> ``assert_rita``); the local H100
stays free.  A ``TaskProlog failed`` death is a node fault: resubmit that
array index once, identically.

    #SBATCH --account=phy220048p --partition=RITA-GPU --qos=rita
    #SBATCH --gres=gpu:a100-80:1 --cpus-per-task=8 --mem=100G

Writes are confined to this analysis directory (diagnostics/, results/,
logs/); Analyses 8, 9 and the existing Analysis-10 fixed-H0 results
(results/a10_arm_J.h5 etc.) are read ONLY, never edited.
"""
import argparse
import itertools
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True                     # A8/A9 trees are READ-ONLY
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE.parent
DIAG = ANALYSIS_DIR / "diagnostics"
RESULTS = ANALYSIS_DIR / "results"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import a10_closure as C                            # noqa: E402

A10, A8, GC, A9 = C.A10, C.A8, C.GC, C.A9           # the established aliases

GW_PATH_A10 = C.GW_PATH_A10
GW_MD5_A10 = C.GW_MD5_A10
TRUTH = dict(C.TRUTH)
TOL = dict(C.TOL)

# The registered fixed-H0 lattices (Analysis-8/-10 objects, not retyped copies).
F_GRID = C.F_GRID                     # 41, linspace(0, 1, 41)
MU_GRID = C.MU_GRID                   # 61, linspace(-0.20, 0.25, 61)
MG_GRID = C.MG_GRID                   # 21, linspace(-10, 10, 21), 1 Msun
MG_REFINE = C.MG_REFINE               # 6, the half-integer nodes inside [2, 8]
MG_FULL = C.MG_GRID_REFINED           # 27, the merged registered+refine axis


def _rate_of_record():
    """3.0143216848373413 s/eval, measured on rita (a10_closure.json), with a
    hard-coded fallback so a missing file never blocks a dry run."""
    try:
        d = json.loads((DIAG / "a10_closure.json").read_text())
        return float(d["cost"]["steady_state_median_seconds"])
    except Exception:
        return 3.0143216848373413


SECONDS_PER_EVAL = _rate_of_record()

# --------------------------------------------------------------------------- #
# The fixed-H0 joint MAP, the default mechanism-arm pin point (owner spec).
# --------------------------------------------------------------------------- #
PROFILE_F = 0.275
PROFILE_DMU_CHI = 0.115
PROFILE_DMU_G = 5.0

DEFAULT_H0_WINDOW = (63.0, 76.0)
J_H0_STEP = 0.5
J_H0_ANCHOR = 67.74


def j_h0_axis(lo, hi):
    """H0 nodes on the 0.5 lattice inside [lo, hi] plus the anchor 67.74.

    Identical construction to A9's J9_H0_AXIS, so at the default window this
    is BITWISE the same array (asserted, not assumed, at every call site that
    cares).
    """
    lat = np.arange(float(lo), float(hi) + 1e-9, J_H0_STEP)
    return np.unique(np.concatenate([lat, [J_H0_ANCHOR]]))


def _nodes_from_formula(grid, formula_values, atol, what):
    """Pick the grid's OWN bitwise values nearest each formula-derived target.

    Float arithmetic on the constructive formula (e.g. -0.20 + 0.015*m) is NOT
    bitwise the same as the registered ``np.linspace`` grid's values at the
    same mathematical point (residuals ~1e-17), so "bitwise a node of the
    grid" is enforced by SELECTING the grid's own array element, with the
    formula value checked to land within ``atol`` of it -- a typo in the
    formula fails loudly instead of silently picking the wrong node.
    """
    idx, out = [], []
    for v in formula_values:
        i = int(np.argmin(np.abs(grid - v)))
        d = float(grid[i] - v)
        if abs(d) > atol:
            raise SystemExit(
                f"[fatal] {what}: {v!r} is not within {atol:g} of any node of "
                f"the registered grid (nearest {grid[i]!r}, diff {d:.3e})")
        idx.append(i)
        out.append(float(grid[i]))
    return np.array(out, dtype=float), np.array(idx, dtype=int)


# ---- C10-J's own node sets, derived and asserted bitwise onto the registered
# axes at IMPORT time (cheap, CPU-only, no data). ---------------------------- #
_J_F_FORMULA = np.arange(0.05, 0.55 + 1e-9, 0.05)                       # 11
J_F_AXIS, J_F_IDX = _nodes_from_formula(F_GRID, _J_F_FORMULA, 1e-9,
                                        "C10-J f_AGN axis")

_J_MU_FORMULA = np.array([-0.20 + 0.015 * m for m in range(15, 28)])    # 13
J_MU_AXIS, J_MU_IDX = _nodes_from_formula(MU_GRID, _J_MU_FORMULA, 1e-9,
                                          "C10-J dmu_chi axis")

_J_MG_FORMULA = np.array([1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0,
                          6.5, 7.0, 7.5, 8.0, 9.0])                     # 15
J_MG_AXIS, J_MG_IDX = _nodes_from_formula(MG_FULL, _J_MG_FORMULA, 1e-9,
                                          "C10-J dmu_G axis")

if J_F_AXIS.size != 11 or J_MU_AXIS.size != 13 or J_MG_AXIS.size != 15:
    raise SystemExit(f"[fatal] C10-J node counts wrong: f={J_F_AXIS.size} "
                     f"mu={J_MU_AXIS.size} mg={J_MG_AXIS.size}")

J_CELLS_PER_H0_NODE = int(J_F_AXIS.size * J_MG_AXIS.size * J_MU_AXIS.size)  # 2145

MECH_ARMS = {
    "P1": {"surveys": "gal_agn", "marks": "pinned",
           "measures": "the total marked H0 width"},
    "I0": {"surveys": "gal_gal", "marks": "zero",
           "measures": "the equalised-tracer unmarked baseline"},
    "I1": {"surveys": "gal_gal", "marks": "pinned",
           "measures": "the equalised-tracer marked width"},
}


# =========================================================================== #
# Small utilities (JSON, checkpoints; writes confined to this directory)
# =========================================================================== #
def _write(path, obj):
    path = Path(path)
    if ANALYSIS_DIR not in path.parents:
        raise RuntimeError(f"[fatal] refusing to write outside {ANALYSIS_DIR}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=GC._json_default))
    tmp.replace(path)
    print(f"wrote {path}")
    return path


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


def _key1(h0):
    return "{:.10g}".format(float(h0))


def _key3(h0, f, g):
    return "{:.10g}|{:.10g}|{:.10g}".format(float(h0), float(f), float(g))


def _s_checkpoints():
    return sorted(DIAG.glob("_c10_s_c*of*.jsonl"))


def _load_done_s(exclude=None):
    done, headers, dropped = {}, {}, 0
    for path in _s_checkpoints():
        if exclude is not None and path == Path(exclude):
            continue
        hdr, rows, drop = _read_jsonl(path)
        dropped += drop
        if hdr is not None:
            headers[path.name] = hdr
        for r in rows:
            done.setdefault(_key1(r["H0"]), (path.name, r))
    return done, headers, dropped


def _j_checkpoints():
    return sorted(DIAG.glob("_c10_j_c*of*.jsonl"))


def _load_done_j(exclude=None):
    done, headers, dropped = {}, {}, 0
    for path in _j_checkpoints():
        if exclude is not None and path == Path(exclude):
            continue
        hdr, rows, drop = _read_jsonl(path)
        dropped += drop
        if hdr is not None:
            headers[path.name] = hdr
        for r in rows:
            done.setdefault(_key3(r["H0"], r["f_agn"], r["dmu_G"]), (path.name, r))
    return done, headers, dropped


def _header(cell, arm, tag, node_info, extra=None):
    hdr = {
        "record": "header",
        "analysis": "analysis_10_mass_spin_marked_multitracer",
        "campaign": "C10 cosmology stage (H0 released)",
        "arm": arm, "tag": tag,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "provenance": A8.provenance(gw_path=GW_PATH_A10, survey_paths=cell.survey_paths),
        "darksirens_sha": A9.DARKSIRENS_A8_SHA,
        "events_file": GW_PATH_A10, "events_md5": GW_MD5_A10,
        "selection_file": str(cell.opts.gwselection_path),
        "survey_paths": [str(s) for s in cell.survey_paths],
        "sampled_labels": list(cell.labels),
        "per_catalog_pop_params": list(cell.per_catalog_pop_params),
        "fixed_parameter_values": {k: float(v)
                                   for k, v in cell.fixed_parameter_values.items()},
        "base_coord": {k: float(v) for k, v in zip(cell.labels, cell.base)},
        "nEvents": int(cell.data["nEvents"]), "nsamp": int(cell.data["nsamp"]),
        "Ndraw": float(cell.data["Ndraw"]),
        "truth": dict(TRUTH), "tolerances": dict(TOL),
        "jax_devices": str(__import__("jax").devices()),
    }
    hdr.update(node_info)
    if extra:
        hdr.update(extra)
    return hdr


def build_c10_cell(tag, survey_paths):
    cell = A10.build_a10(tag, survey_paths, verbose=True, gw_path=GW_PATH_A10)
    C.assert_cell(cell, A10.EXPECTED_LABELS_A10, tag)
    return cell


def _eval(cell, H0, f, dmu_chi, dmu_G):
    rec = cell.evaluate_at(H0=float(H0), fcat_2=float(f),
                           dmu_chi=float(dmu_chi), dmu_G=float(dmu_G))
    row = GC._cell_record(rec)
    row["logL_hex"] = rec.get("logL_hex")
    return row


# =========================================================================== #
# STAGE: profile  (GPU, tiny -- one worker, no array)
# =========================================================================== #
def _profile_h0_axis(verbose=True):
    import h5py
    axis = np.array(A9.J9_H0_AXIS, dtype=float)
    ref_path = A9.RESULTS / "j9_marked.h5"
    with h5py.File(ref_path, "r") as h:
        ref = h["H0_grid"][:]
    ok = bool(np.array_equal(axis, ref))
    if verbose:
        print(f"  [{'OK ' if ok else 'FAIL'}] C10-P H0 axis ({axis.size} nodes) "
              f"bitwise == {ref_path} H0_grid: {ok}")
    if not ok:
        raise SystemExit("[fatal] C10-P H0 axis is not bitwise A9's J9 H0_grid")
    return axis, {"n_nodes": int(axis.size), "source": str(ref_path),
                  "bitwise_equal_to_a9_j9_H0_grid": ok}


def stage_profile(args):
    env = A9._gpu_setup("profile")
    env["a10_inputs"] = C.assert_a10_inputs(with_md5=False)
    axis, axis_check = _profile_h0_axis()
    path = DIAG / "_c10_profile.jsonl"

    hdr, rows, dropped = _read_jsonl(path)
    done = {_key1(r["H0"]): r for r in rows}
    todo = [h0 for h0 in axis if _key1(h0) not in done]
    print(f"[C10-P] {len(done)}/{axis.size} H0 nodes already done, "
          f"{len(todo)} to do ({dropped} truncated line(s) dropped)")

    if todo:
        cell = build_c10_cell("C10_P", [A8.SURVEY_GAL, A8.SURVEY_AGN])
        if hdr is None:
            _append_jsonl(path, _header(cell, "C10-P", "profile", {
                "H0_grid": axis.tolist(), "H0_grid_check": axis_check,
                "marked_point": {"f_agn": PROFILE_F, "dmu_chi": PROFILE_DMU_CHI,
                                 "dmu_G": PROFILE_DMU_G},
                "spatial_point": {"f_agn": PROFILE_F, "dmu_chi": 0.0, "dmu_G": 0.0},
            }))
        t0 = time.time()
        for n, h0 in enumerate(todo, start=1):
            marked = _eval(cell, h0, PROFILE_F, PROFILE_DMU_CHI, PROFILE_DMU_G)
            spatial = _eval(cell, h0, PROFILE_F, 0.0, 0.0)
            _append_jsonl(path, {
                "record": "row", "H0": float(h0), "key": _key1(h0),
                "marked": marked, "spatial": spatial,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            el = time.time() - t0
            print(f"[C10-P] {n}/{len(todo)} H0={h0:.2f} "
                  f"logL_marked={marked['logL']:.4f} logL_spatial={spatial['logL']:.4f} "
                  f"elapsed={el:.1f}s eta={el/n*(len(todo)-n):.1f}s")
            sys.stdout.flush()
    else:
        print("nothing to do")

    _, rows, _ = _read_jsonl(path)
    have = {_key1(r["H0"]) for r in rows}
    want = {_key1(h0) for h0 in axis}
    if want <= have:
        _assemble_profile(axis, rows, axis_check)
    else:
        print(f"[C10-P] {len(want - have)} node(s) still missing; re-run to finish "
              f"and auto-assemble")


def _assemble_profile(axis, rows, axis_check):
    by = {_key1(r["H0"]): r for r in rows}
    order = list(axis)
    ll_m = np.array([by[_key1(h0)]["marked"]["logL"] for h0 in order])
    ll_s = np.array([by[_key1(h0)]["spatial"]["logL"] for h0 in order])
    fin_m = np.isfinite(ll_m)
    fin_s = np.isfinite(ll_s)
    scan_h0f = A8.import_scan_h0f()
    marginal_ci = scan_h0f.marginal_ci

    def block(ll, fin, nodes_key_prefix):
        llm = np.where(fin, ll, -np.inf)
        mx = float(llm[np.isfinite(llm)].max())
        ci = marginal_ci(axis, llm)
        return {
            "logL": ll.tolist(), "finite": fin.tolist(), "logL_max": mx,
            "n_rejected": int((~fin).sum()),
            "argmax_H0": float(axis[int(np.argmax(llm))]),
            "profile_ci68": ci["ci68"], "profile_ci90": ci["ci90"],
            "profile_median": ci["median"],
            "profile_ci_note": ("PROFILE, NOT MARGINAL: exp(logL) on the H0 "
                                "lattice at fixed (f, dmu_chi, dmu_G) treated "
                                "as a 1-D density for a quick 68/90 via "
                                "Analysis 2's marginal_ci; nothing is "
                                "integrated over."),
        }

    marked_block = block(ll_m, fin_m, "marked")
    spatial_block = block(ll_s, fin_s, "spatial")
    nodes_marked = [dict(H0=float(h0), **by[_key1(h0)]["marked"]) for h0 in order]
    nodes_spatial = [dict(H0=float(h0), **by[_key1(h0)]["spatial"]) for h0 in order]

    lo, hi = DEFAULT_H0_WINDOW
    out = {
        "analysis": "analysis_10_mass_spin_marked_multitracer",
        "stage": "profile",
        "question": ("does the [63, 76] window still bracket the H0 peak once "
                     "the mass mark is live at the fixed-H0 joint MAP?"),
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "H0_grid": axis.tolist(), "n_nodes": int(axis.size),
        "H0_grid_check": axis_check,
        "marked_point": {"f_agn": PROFILE_F, "dmu_chi": PROFILE_DMU_CHI,
                         "dmu_G": PROFILE_DMU_G},
        "spatial_point": {"f_agn": PROFILE_F, "dmu_chi": 0.0, "dmu_G": 0.0},
        "marked_profile": marked_block, "nodes_marked": nodes_marked,
        "spatial_profile": spatial_block, "nodes_spatial": nodes_spatial,
        "window_check": {
            "window": [lo, hi],
            "marked_argmax_in_window": bool(lo <= marked_block["argmax_H0"] <= hi),
            "spatial_argmax_in_window": bool(lo <= spatial_block["argmax_H0"] <= hi),
        },
        "truth": dict(TRUTH),
        "events_file": GW_PATH_A10, "events_md5": GW_MD5_A10,
    }
    _write(DIAG / "c10_profile.json", out)
    print(f"[C10-P] marked   argmax H0={marked_block['argmax_H0']:.2f}  "
          f"profile 68% {marked_block['profile_ci68']}  "
          f"90% {marked_block['profile_ci90']}")
    print(f"[C10-P] spatial  argmax H0={spatial_block['argmax_H0']:.2f}  "
          f"profile 68% {spatial_block['profile_ci68']}  "
          f"90% {spatial_block['profile_ci90']}")
    return out


# =========================================================================== #
# STAGE: s  (C10-S, GPU array 0-3%2) / s_assemble (CPU)
# =========================================================================== #
def _s_h0_axis(verbose=True):
    import h5py
    axis = np.array(A9.S9_H0_AXIS, dtype=float)
    ref_path = A9.RESULTS / "s9_spatial.h5"
    with h5py.File(ref_path, "r") as h:
        ref = h["H0_grid"][:]
    ok = bool(np.array_equal(axis, ref))
    if verbose:
        print(f"  [{'OK ' if ok else 'FAIL'}] C10-S H0 axis ({axis.size} nodes) "
              f"bitwise == {ref_path} H0_grid: {ok}")
    if not ok:
        raise SystemExit("[fatal] C10-S H0 axis is not bitwise A9's S9 H0_grid")
    return axis, {"n_nodes": int(axis.size), "source": str(ref_path),
                  "bitwise_equal_to_a9_s9_H0_grid": ok}


def stage_s(args):
    env = A9._gpu_setup("s")
    env["a10_inputs"] = C.assert_a10_inputs(with_md5=False)
    axis, axis_check = _s_h0_axis()
    if not (0 <= args.chunk < args.n_chunks):
        raise SystemExit(f"[fatal] --chunk must be in [0, {args.n_chunks})")
    tag = f"c{args.chunk}of{args.n_chunks}"
    path = DIAG / f"_c10_s_{tag}.jsonl"
    mine = [float(x) for i, x in enumerate(axis) if i % args.n_chunks == args.chunk]

    done_elsewhere, headers, dropped = _load_done_s(exclude=path)
    hdr_self, rows_self, drop_self = _read_jsonl(path)
    done_self = {_key1(r["H0"]): r for r in rows_self}
    todo = [h0 for h0 in mine
            if _key1(h0) not in done_self and _key1(h0) not in done_elsewhere]
    print(f"[C10-S] chunk {args.chunk}/{args.n_chunks}: {len(mine)} H0 nodes, "
          f"{len(done_self)} already here, {len(done_elsewhere)} in siblings, "
          f"{len(todo)} to do ({len(todo) * F_GRID.size} cells), "
          f"{dropped + drop_self} truncated line(s) dropped")
    if not todo:
        print("nothing to do")
        return

    t_build = time.time()
    cell = build_c10_cell(f"C10_S_c{args.chunk}", [A8.SURVEY_GAL, A8.SURVEY_AGN])
    print(f"build: {time.time() - t_build:.1f}s")
    if hdr_self is None:
        _append_jsonl(path, _header(cell, "C10-S", tag, {
            "H0_grid": axis.tolist(), "H0_grid_check": axis_check,
            "f_grid": F_GRID.tolist(),
            "dmu_chi_held_at": 0.0, "dmu_G_held_at": 0.0,
            "chunk": args.chunk, "n_chunks": args.n_chunks,
            "h0_nodes_this_worker": mine,
        }))

    t0 = time.time()
    for n, h0 in enumerate(todo, start=1):
        cells = [_eval(cell, h0, f, 0.0, 0.0) for f in F_GRID]
        _append_jsonl(path, {
            "record": "row", "H0": float(h0), "key": _key1(h0), "cells": cells,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        lls = np.array([c["logL"] for c in cells], dtype=float)
        fin = np.isfinite(lls)
        el = time.time() - t0
        print(f"[C10-S] row {n}/{len(todo)} H0={h0:.2f} "
              f"maxlogL={(lls[fin].max() if fin.any() else float('nan')):.4f} "
              f"rejected={int((~fin).sum())}/{F_GRID.size} "
              f"elapsed={el/60:.1f}min eta={el/n*(len(todo)-n)/60:.1f}min")
        sys.stdout.flush()
        if args.stop_after_s and el > args.stop_after_s:
            print(f"[C10-S] stopping cleanly at the requested budget; "
                  f"{len(todo) - n} rows left")
            break
    print(f"[C10-S] done -> {path}")


def _s_arrays(done, axis):
    shape = (axis.size, F_GRID.size)
    ll = np.full(shape, np.nan)
    fin = np.zeros(shape, dtype=bool)
    store, missing = {}, []
    for a, h0 in enumerate(axis):
        hit = done.get(_key1(h0))
        if hit is None:
            missing.append(float(h0))
            continue
        cells = hit[1]["cells"]
        store[a] = cells
        for i, c in enumerate(cells):
            ll[a, i] = c["logL"]
            fin[a, i] = bool(c["finite"])

    def pull(key, default=np.nan):
        out = np.full(shape, default, dtype=float)
        for a, cells in store.items():
            for i, c in enumerate(cells):
                v = c.get(key)
                out[a, i] = default if v is None else v
        return out

    return ll, fin, pull, missing


def _marginals_h0_f(h0, f, ll, fin, marginal_ci):
    """p(H0, f) with a flat prior on both; returns the 2-D moments too."""
    llm = np.where(fin, ll, -np.inf)
    if not np.isfinite(llm).any():
        raise RuntimeError("[fatal] every cell is rejected")
    mx = float(llm[np.isfinite(llm)].max())
    P = np.where(np.isfinite(llm), np.exp(llm - mx), 0.0)
    m_h0 = np.trapz(P, f, axis=1)
    m_f = np.trapz(P, h0, axis=0)
    out = {"logL_max": mx}
    with np.errstate(divide="ignore"):
        lp_h0, lp_f = np.log(m_h0), np.log(m_f)
    out["H0"] = marginal_ci(h0, lp_h0)
    out["H0"]["marginal_logp"] = lp_h0.tolist()
    out["H0"]["marginal_mode"] = float(
        h0[int(np.nanargmax(np.where(np.isfinite(lp_h0), lp_h0, -np.inf)))])
    out["f"] = marginal_ci(f, lp_f)
    out["f"]["marginal_logp"] = lp_f.tolist()
    out["f"]["marginal_mode"] = float(
        f[int(np.nanargmax(np.where(np.isfinite(lp_f), lp_f, -np.inf)))])

    W = np.outer(GC._trapz_weights(h0), GC._trapz_weights(f)) * P
    Z = W.sum()
    H2, F2 = np.meshgrid(h0, f, indexing="ij")
    Eh, Ef = float((W * H2).sum() / Z), float((W * F2).sum() / Z)
    Vh = float((W * (H2 - Eh) ** 2).sum() / Z)
    Vf = float((W * (F2 - Ef) ** 2).sum() / Z)
    Cov = float((W * (H2 - Eh) * (F2 - Ef)).sum() / Z)
    out["posterior_moments"] = {
        "mean": {"H0": Eh, "f_agn": Ef},
        "sd": {"H0": float(np.sqrt(Vh)), "f_agn": float(np.sqrt(Vf))},
        "cov_H0_f": Cov, "correlation": {"H0|f_agn": float(Cov / np.sqrt(Vh * Vf))},
    }
    a, i = np.unravel_index(np.argmax(llm), llm.shape)
    out["map"] = {"H0": float(h0[a]), "f_agn": float(f[i]),
                  "logL": float(llm[a, i]), "index": [int(a), int(i)]}
    out["_P"] = P
    return out


def _guard_report_2d(name, h0, f, ll, fin, pull, mark_labels):
    Neff, thr = pull("Neff"), pull("threshold")
    rejected = ~fin
    rep = {
        "arm": name, "axes": ["H0", "f_agn"], "marks_held_at": mark_labels,
        "n_cells": int(ll.size), "n_rejected": int(rejected.sum()),
        "rejected_fraction": float(rejected.mean()),
        "Neff_min": float(np.nanmin(Neff)), "Neff_median": float(np.nanmedian(Neff)),
        "Neff_max": float(np.nanmax(Neff)),
        "threshold_min": float(np.nanmin(thr)), "threshold_max": float(np.nanmax(thr)),
        "Neff_over_threshold_min": float(np.nanmin(Neff / thr)),
        "policy": A9.GUARD_POLICY,
        "posterior_mass_in_rejected_region_as_scanned": 0.0,
        "posterior_mass_note": A9.GUARD_CIRCULARITY_NOTE,
        "rejected_cells": [],
    }
    idx = np.argwhere(rejected)
    for a, i in idx[:5000]:
        rep["rejected_cells"].append({
            "H0": float(h0[a]), "f_agn": float(f[i]), **mark_labels,
            "Neff": float(Neff[a, i]), "threshold": float(thr[a, i]),
            "Neff_over_threshold": float(Neff[a, i] / thr[a, i])})
    llm = np.where(fin, ll, -np.inf)
    mx = float(llm[np.isfinite(llm)].max())
    P = np.where(np.isfinite(llm), np.exp(llm - mx), 0.0)
    W = np.outer(GC._trapz_weights(h0), GC._trapz_weights(f))
    rep["accepted_mass_normalisation"] = float((W * P).sum())
    if rep["n_rejected"] == 0:
        rep["upper_bound_rejected_mass_fraction"] = 0.0
        rep["passes"] = True
        return rep
    nb, local = [], []
    for a, i in idx:
        here = []
        for da in (-1, 0, 1):
            for di in (-1, 0, 1):
                x, y = a + da, i + di
                if 0 <= x < P.shape[0] and 0 <= y < P.shape[1] and fin[x, y]:
                    here.append(float(P[x, y]))
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
    rep["fill_is_an_over_estimate"] = {
        "fill_value_relative_to_peak": P_bound,
        "n_rejected_cells": int(len(local)),
        "n_cells_where_fill_ge_local_accepted_boundary_max":
            int(sum(1 for v in local if P_bound >= v)),
        "max_local_accepted_boundary_density": float(max(local)) if local else 0.0,
        "verified": bool(all(P_bound >= v for v in local)),
    }
    rep["passes"] = bool(rep["upper_bound_rejected_mass_fraction"] < 1e-6)
    return rep


def _window_check(name, axis, marg, window, criterion=1e-6, cells_per_h0_node=None):
    """Gate-W-style: does [window] hold THIS marginal's own mass?  If not,
    report (not choose) the smallest 0.5-lattice window that does."""
    lp = np.asarray(marg["H0"]["marginal_logp"], dtype=float)
    p = np.exp(lp - np.nanmax(lp[np.isfinite(lp)]))
    dens = p / np.trapz(p, axis)
    lo, hi = float(window[0]), float(window[1])
    ilo = int(np.flatnonzero(axis == lo)[0]) if (axis == lo).any() else None
    ihi = int(np.flatnonzero(axis == hi)[0]) if (axis == hi).any() else None
    if ilo is None or ihi is None:
        raise RuntimeError(f"[fatal] {name}: window edges {window} are not axis nodes")
    inside = np.trapz(dens[ilo:ihi + 1], axis[ilo:ihi + 1])
    out = {
        "arm": name, "window": [lo, hi],
        "criterion": (f"marginal density at each edge <= {criterion:g} of the "
                      f"peak (GATES.md B3), measured on THIS marginal"),
        "peak_H0": float(axis[int(np.argmax(p))]),
        "density_at_low_edge_over_peak": float(p[ilo]),
        "density_at_high_edge_over_peak": float(p[ihi]),
        "mass_inside": float(inside), "mass_outside": float(1.0 - inside),
        "passes": bool(p[ilo] <= criterion and p[ihi] <= criterion),
    }
    lattice = axis[np.isclose(np.round(axis * 2.0) / 2.0, axis, rtol=0, atol=1e-12)]
    ok = p[np.searchsorted(axis, lattice)] <= criterion
    below = lattice[(lattice <= lo) & ok] if (lattice <= lo).any() else lattice[:0]
    above = lattice[(lattice >= hi) & ok] if (lattice >= hi).any() else lattice[:0]
    need_lo = float(below.max()) if below.size else float(lattice.min())
    need_hi = float(above.min()) if above.size else float(lattice.max())
    n_nodes = int(np.sum((lattice >= need_lo) & (lattice <= need_hi)))
    if not ((need_lo <= J_H0_ANCHOR <= need_hi)
            and np.any(np.isclose(lattice, J_H0_ANCHOR))):
        n_nodes += 1  # + 67.74, if it is not already one of the lattice nodes
    out["required_window_on_the_0p5_lattice"] = [need_lo, need_hi]
    out["required_window_n_H0_nodes_including_67p74"] = n_nodes
    if cells_per_h0_node:
        out["required_window_cells"] = int(n_nodes * cells_per_h0_node)
    out["widening_needed"] = bool(need_lo < lo or need_hi > hi)
    if not out["passes"]:
        print(f"[{name}] WARNING: the [{lo}, {hi}] window FAILS the {criterion:g} "
              f"edge criterion on this mock; the smallest passing 0.5-lattice "
              f"window is {out['required_window_on_the_0p5_lattice']} "
              f"({n_nodes} nodes).  NOT choosing this automatically.")
    return out


def _matched_28_node_lattice(s_axis, s_f, ll_s, j_axis, marginal_ci):
    """C10-S's own cube, re-marginalised on exactly the given 28 J nodes
    (a9_post.h0_matched_lattice's construction, generalised to C10)."""
    idx = []
    for v in j_axis:
        hit = np.where(s_axis == v)[0]
        if hit.size != 1:
            raise RuntimeError(f"[fatal] J node {v} is not a unique C10-S node")
        idx.append(int(hit[0]))
    idx = np.asarray(idx)
    if not np.array_equal(s_axis[idx], j_axis):
        raise RuntimeError("[fatal] the J lattice is not bitwise a subset of C10-S's")

    def marginal(h0, ll):
        llm = np.where(np.isfinite(ll), ll, -np.inf)
        mx = float(llm[np.isfinite(llm)].max())
        P = np.where(np.isfinite(llm), np.exp(llm - mx), 0.0)
        with np.errstate(divide="ignore"):
            return marginal_ci(h0, np.log(np.trapz(P, s_f, axis=1)))

    full = marginal(s_axis, ll_s)
    sub = marginal(j_axis, ll_s[idx, :])
    w = lambda b, lev: float(b[lev][1] - b[lev][0])
    return {
        "description": ("C10-S's OWN cube, re-marginalised on exactly the given "
                        "J H0 nodes, so the S-vs-J width comparison uses "
                        "identical quadrature (A9's h0_matched_lattice "
                        "construction, generalised)."),
        "j_nodes": j_axis.tolist(), "S_indices_used": idx.tolist(),
        "S_full_202_node_axis": {"median": full["median"], "ci68": full["ci68"],
                                 "ci90": full["ci90"], "width68": w(full, "ci68"),
                                 "width90": w(full, "ci90")},
        "S_on_the_matched_J_nodes": {"median": sub["median"], "ci68": sub["ci68"],
                                     "ci90": sub["ci90"], "width68": w(sub, "ci68"),
                                     "width90": w(sub, "ci90")},
        "ratio_matched_over_full_68": w(sub, "ci68") / w(full, "ci68"),
        "ratio_matched_over_full_90": w(sub, "ci90") / w(full, "ci90"),
    }


def stage_s_assemble(args):
    import h5py
    env = A9._cpu_setup("s_assemble", import_darksirens=False)
    axis, axis_check = _s_h0_axis()
    done, headers, dropped = _load_done_s()

    shas = {h.get("darksirens_sha") for h in headers.values()}
    if shas and shas != {A9.DARKSIRENS_A8_SHA}:
        raise RuntimeError(f"[fatal] checkpoints carry SHA {shas}")
    md5s = {h.get("events_md5") for h in headers.values()}
    if md5s and md5s != {GW_MD5_A10}:
        raise RuntimeError(f"[fatal] checkpoints carry events md5 {md5s}")

    ll, fin, pull, missing = _s_arrays(done, axis)
    if missing and not args.allow_partial:
        raise RuntimeError(
            f"[fatal] {len(missing)} of {axis.size} rows missing; run the "
            f"remaining chunks or pass --allow_partial.  First missing: {missing[:5]}")
    if missing:
        _write(DIAG / "c10_s_coverage.json",
               {"rows_missing": missing, "rows_total": int(axis.size),
                "checkpoints": [p.name for p in _s_checkpoints()]})
        print(f"[partial] {len(missing)} S rows missing; wrote the coverage report")
        return

    scan_h0f = A8.import_scan_h0f()
    marginal_ci = scan_h0f.marginal_ci
    marg = _marginals_h0_f(axis, F_GRID, ll, fin, marginal_ci)
    P = marg.pop("_P")
    GC._truth_flags(marg["H0"], {"planted": 67.74})
    GC._truth_flags(marg["f"], {"planted": TRUTH["f_agn_planted"]})
    guard = _guard_report_2d("C10-S", axis, F_GRID, ll, fin, pull,
                             {"dmu_chi": 0.0, "dmu_G": 0.0})
    window = _window_check("C10-S", axis, marg, DEFAULT_H0_WINDOW,
                           cells_per_h0_node=J_CELLS_PER_H0_NODE)
    j_default_axis = j_h0_axis(*DEFAULT_H0_WINDOW)
    matched = _matched_28_node_lattice(axis, F_GRID, ll, j_default_axis, marginal_ci)

    edges = {}
    for name, x, key in (("H0", axis, "H0"), ("f_agn", F_GRID, "f")):
        lp = np.asarray(marg[key]["marginal_logp"], dtype=float)
        pk = np.exp(lp - np.nanmax(lp[np.isfinite(lp)]))
        edges[name] = {"low_edge_density_over_peak": float(pk[0]),
                       "high_edge_density_over_peak": float(pk[-1]),
                       "criterion": "both <= 1e-6 of the peak",
                       "passes": bool(pk[0] <= 1e-6 and pk[-1] <= 1e-6)}

    hdr = next(iter(headers.values())) if headers else {}
    summary = {
        "analysis": "analysis_10_mass_spin_marked_multitracer",
        "arm": "C10-S",
        "question": ("spatial-only cosmology on the TWO-MARK mock: (H0, f_AGN) "
                     "with both marks held at 0.0 exactly, evaluated with the "
                     "two-mark model (closure 15.1 showed this equals the "
                     "shared-population model bitwise)."),
        "file": "results/c10_arm_S.h5",
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "environment": env,
        "provenance": hdr.get("provenance"),
        "free_parameters": ["H0", "fcat_2"],
        "dmu_chi_held_at": 0.0, "dmu_G_held_at": 0.0,
        "sampled_labels": hdr.get("sampled_labels"),
        "base_coord": hdr.get("base_coord"),
        "fixed_parameter_values": hdr.get("fixed_parameter_values"),
        "H0_fixed": None, "Om0_fixed": A8.OM0_FID,
        "events_file": GW_PATH_A10, "events_md5": GW_MD5_A10,
        "selection_file": hdr.get("selection_file"),
        "survey_paths": hdr.get("survey_paths"),
        "truth": dict(TRUTH),
        "grid": {"H0": {"values": axis.tolist(), "n": int(axis.size),
                       "check": axis_check},
                "f_agn": [float(F_GRID[0]), float(F_GRID[-1]), int(F_GRID.size)]},
        "n_cells": int(ll.size), "n_rejected": int((~fin).sum()),
        "logL_max": marg["logL_max"], "map": marg["map"],
        "H0": marg["H0"], "f": marg["f"],
        "posterior_moments": marg["posterior_moments"],
        "edge_mass": edges,
        "j_window_check": window,
        "matched_28_node_lattice": matched,
        "guard": {k: v for k, v in guard.items() if k != "rejected_cells"},
        "guard_rejected_cells": guard.get("rejected_cells", []),
        "timing": {"cells": int(ll.size),
                   "gpu_hours": float(np.nansum(pull("seconds")) / 3600.0),
                   "median_seconds_per_cell": float(np.nanmedian(pull("seconds")))},
        "checkpoints": [p.name for p in _s_checkpoints()],
        "truncated_lines_dropped": dropped,
    }

    h5_path = RESULTS / "c10_arm_S.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("H0_grid", data=axis)
        h5.create_dataset("f_grid", data=F_GRID)
        h5.create_dataset("log_likelihood", data=ll)
        g = h5.create_group("guard")
        for key in ("Neff", "threshold", "pe_variance_sum", "sigma2_total",
                    "logL_selection", "logL_pe", "log_mu", "seconds"):
            g.create_dataset(key, data=pull(key))
        g.create_dataset("rejected", data=(~fin))
        g.attrs["n_rejected"] = int((~fin).sum())
        m = h5.create_group("marginal")
        m.create_dataset("H0", data=np.exp(np.asarray(marg["H0"]["marginal_logp"])))
        m.create_dataset("f", data=np.exp(np.asarray(marg["f"]["marginal_logp"])))
        h5.create_dataset("posterior_unnormalised", data=P)
        h5.attrs["analysis"] = "analysis_10_mass_spin_marked_multitracer"
        h5.attrs["arm"] = "C10-S"
        h5.attrs["free_parameters"] = json.dumps(["H0", "fcat_2"])
        h5.attrs["dmu_chi_held_at"] = 0.0
        h5.attrs["dmu_G_held_at"] = 0.0
        h5.attrs["labels"] = json.dumps(hdr.get("sampled_labels"))
        h5.attrs["truth"] = json.dumps(TRUTH)
        h5.attrs["Om0_fixed"] = A8.OM0_FID
        h5.attrs["events_file"] = GW_PATH_A10
        h5.attrs["events_md5"] = GW_MD5_A10
        h5.attrs["selection_file"] = str(hdr.get("selection_file"))
        h5.attrs["survey_paths"] = json.dumps(hdr.get("survey_paths"))
        h5.attrs["darksirens_sha"] = A9.DARKSIRENS_A8_SHA
    print(f"wrote {h5_path}")
    _write(RESULTS / "c10_arm_S.json", summary)
    for k in ("H0", "f"):
        b = summary[k]
        print(f"  {k:<4} median {b['median']:.6f}  68% [{b['ci68'][0]:.6f}, "
              f"{b['ci68'][1]:.6f}]  90% [{b['ci90'][0]:.6f}, {b['ci90'][1]:.6f}]")
    print(f"  j_window_check: {json.dumps(window, default=GC._json_default)}")
    return summary


# =========================================================================== #
# STAGE: j  (C10-J, GPU array 0-7%2) / j_assemble (CPU)
# =========================================================================== #
def j_row_order(h0_axis, f_axis, mg_axis, anchor=J_H0_ANCHOR):
    """The (H0, f, dmu_G) rows, H0 = 67.74 dealt FIRST -- that slab is
    cell-for-cell a sub-lattice of the fixed-H0 A10-J cube, so it closes the
    free ULP check while the rest of the cube is still running."""
    has_anchor = bool(np.any(h0_axis == anchor))
    rest = [h for h in h0_axis if h != anchor]
    anchor_rows = ([(float(anchor), float(f), float(g))
                    for g in mg_axis for f in f_axis] if has_anchor else [])
    rest_rows = [(float(h0), float(f), float(g))
                for h0 in rest for g in mg_axis for f in f_axis]
    return anchor_rows + rest_rows


def _check_j_h0_axis_default(h0_axis, window):
    if tuple(float(w) for w in window) != DEFAULT_H0_WINDOW:
        return {"checked": False, "why": "non-default --h0_window"}
    ref = np.array(A9.J9_H0_AXIS, dtype=float)
    ok = bool(np.array_equal(h0_axis, ref))
    print(f"  [{'OK ' if ok else 'FAIL'}] C10-J default H0 axis ({h0_axis.size} "
          f"nodes) bitwise == A9.J9_H0_AXIS: {ok}")
    return {"checked": True, "bitwise_equal_to_a9_J9_H0_AXIS": ok}


def stage_j(args):
    env = A9._gpu_setup("j")
    env["a10_inputs"] = C.assert_a10_inputs(with_md5=False)
    h0_axis = j_h0_axis(*args.h0_window)
    axis_check = _check_j_h0_axis_default(h0_axis, args.h0_window)
    if not (0 <= args.chunk < args.n_chunks):
        raise SystemExit(f"[fatal] --chunk must be in [0, {args.n_chunks})")

    order = j_row_order(h0_axis, J_F_AXIS, J_MG_AXIS)
    mine = [x for i, x in enumerate(order) if i % args.n_chunks == args.chunk]
    tag = f"c{args.chunk}of{args.n_chunks}"
    path = DIAG / f"_c10_j_{tag}.jsonl"
    print(f"[C10-J] chunk {args.chunk}/{args.n_chunks}: {len(mine)} rows "
          f"({len(mine) * J_MU_AXIS.size} cells); h0_window={list(args.h0_window)}, "
          f"H0={h0_axis.size} f={J_F_AXIS.size} dmu_G={J_MG_AXIS.size} nodes")

    done_elsewhere, headers, dropped = _load_done_j(exclude=path)
    hdr_self, rows_self, drop_self = _read_jsonl(path)
    done_self = {_key3(r["H0"], r["f_agn"], r["dmu_G"]): r for r in rows_self}
    todo = [(h0, f, g) for (h0, f, g) in mine
            if _key3(h0, f, g) not in done_self
            and _key3(h0, f, g) not in done_elsewhere]
    print(f"rows to do in this worker: {len(todo)} "
          f"({len(todo) * J_MU_AXIS.size} cells), "
          f"{dropped + drop_self} truncated line(s) dropped")
    if not todo:
        print("nothing to do")
        return

    t_build = time.time()
    cell = build_c10_cell(f"C10_J_c{args.chunk}", [A8.SURVEY_GAL, A8.SURVEY_AGN])
    print(f"build: {time.time() - t_build:.1f}s")
    if hdr_self is None:
        _append_jsonl(path, _header(cell, "C10-J", tag, {
            "h0_window": list(args.h0_window), "H0_grid": h0_axis.tolist(),
            "H0_grid_default_check": axis_check,
            "f_grid": J_F_AXIS.tolist(), "dmu_chi_grid": J_MU_AXIS.tolist(),
            "dmu_G_grid": J_MG_AXIS.tolist(),
            "chunk": args.chunk, "n_chunks": args.n_chunks,
            "rows_this_worker": len(mine),
        }))

    t0 = time.time()
    for n, (h0, f, g) in enumerate(todo, start=1):
        cells = [_eval(cell, h0, f, mu, g) for mu in J_MU_AXIS]
        _append_jsonl(path, {
            "record": "row", "H0": float(h0), "f_agn": float(f), "dmu_G": float(g),
            "key": _key3(h0, f, g), "cells": cells,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        lls = np.array([c["logL"] for c in cells], dtype=float)
        fin = np.isfinite(lls)
        el = time.time() - t0
        print(f"[C10-J] row {n}/{len(todo)} H0={h0:.2f} f={f:.4f} dmu_G={g:+.2f} "
              f"maxlogL={(lls[fin].max() if fin.any() else float('nan')):.4f} "
              f"rejected={int((~fin).sum())}/{J_MU_AXIS.size} "
              f"elapsed={el/60:.1f}min eta={el/n*(len(todo)-n)/60:.1f}min")
        sys.stdout.flush()
        if args.stop_after_s and el > args.stop_after_s:
            print(f"[C10-J] stopping cleanly at the requested budget; "
                  f"{len(todo) - n} rows left")
            break
    print(f"[C10-J] done -> {path}")


def _j_cube(done, h0_axis):
    shape = (h0_axis.size, J_F_AXIS.size, J_MG_AXIS.size, J_MU_AXIS.size)
    ll = np.full(shape, np.nan)
    fin = np.zeros(shape, dtype=bool)
    store, missing = {}, []
    for a, h0 in enumerate(h0_axis):
        for i, f in enumerate(J_F_AXIS):
            for k, g in enumerate(J_MG_AXIS):
                hit = done.get(_key3(h0, f, g))
                if hit is None:
                    missing.append({"H0": float(h0), "f_agn": float(f),
                                    "dmu_G": float(g)})
                    continue
                cells = hit[1]["cells"]
                store[(a, i, k)] = cells
                for j, c in enumerate(cells):
                    ll[a, i, k, j] = c["logL"]
                    fin[a, i, k, j] = bool(c["finite"])

    def pull(key, default=np.nan):
        out = np.full(shape, default, dtype=float)
        for (a, i, k), cells in store.items():
            for j, c in enumerate(cells):
                v = c.get(key)
                out[a, i, k, j] = default if v is None else v
        return out

    return ll, fin, pull, missing


NAMES4 = ("H0", "f_agn", "dmu_G", "dmu_chi")


def _marginals_4d(h0, f, mg, mu, ll, fin, marginal_ci):
    grids = [h0, f, mg, mu]
    llm = np.where(fin, ll, -np.inf)
    if not np.isfinite(llm).any():
        raise RuntimeError("[fatal] every C10-J cell is rejected")
    mx = float(llm[np.isfinite(llm)].max())
    P = np.where(np.isfinite(llm), np.exp(llm - mx), 0.0)

    # Three 3-D reductions, each integrating out ONE axis; reused below both for
    # the four 1-D marginals and (further reduced) for the six 2-D marginals.
    P_over_mu = np.trapz(P, mu, axis=3)       # (h0, f, mg) -- dmu_chi integrated out
    P_over_mg = np.trapz(P, mg, axis=2)       # (h0, f, mu) -- dmu_G integrated out
    P_over_f = np.trapz(P, f, axis=1)         # (h0, mg, mu) -- f_agn integrated out

    # Direct, unambiguous 1-D marginals (each integrates out the OTHER three).
    # Axis bookkeeping verified numerically against a brute-force nested-trapz
    # reference on a random 4-D array before this was trusted (see the commit
    # that added this file).
    m_h0 = np.trapz(np.trapz(P_over_mu, mg, axis=2), f, axis=1)
    m_f = np.trapz(np.trapz(P_over_mu, h0, axis=0), mg, axis=1)
    m_mg = np.trapz(np.trapz(P_over_mu, h0, axis=0), f, axis=0)
    m_mu = np.trapz(np.trapz(P_over_mg, h0, axis=0), f, axis=0)

    out = {"logL_max": mx}
    with np.errstate(divide="ignore"):
        for name, x, m in (("H0", h0, m_h0), ("f_agn", f, m_f),
                           ("dmu_G", mg, m_mg), ("dmu_chi", mu, m_mu)):
            lp = np.log(m)
            out[name] = marginal_ci(x, lp)
            out[name]["marginal_logp"] = lp.tolist()
            out[name]["marginal_mode"] = float(x[int(np.argmax(lp))])

    W = (GC._trapz_weights(h0)[:, None, None, None]
         * GC._trapz_weights(f)[None, :, None, None]
         * GC._trapz_weights(mg)[None, None, :, None]
         * GC._trapz_weights(mu)[None, None, None, :]) * P
    Z = W.sum()
    G = np.meshgrid(*grids, indexing="ij")
    mean = [float((W * g).sum() / Z) for g in G]
    cov = np.zeros((4, 4))
    for a in range(4):
        for b in range(4):
            cov[a, b] = float((W * (G[a] - mean[a]) * (G[b] - mean[b])).sum() / Z)
    sd = np.sqrt(np.diag(cov))
    out["posterior_moments"] = {
        "names": list(NAMES4),
        "mean": {n: m for n, m in zip(NAMES4, mean)},
        "sd": {n: float(s) for n, s in zip(NAMES4, sd)},
        "cov": cov.tolist(),
        "correlation": {f"{NAMES4[a]}|{NAMES4[b]}":
                        float(cov[a, b] / (sd[a] * sd[b]))
                        for a in range(4) for b in range(a + 1, 4)},
    }
    idx = np.unravel_index(np.argmax(llm), llm.shape)
    out["map"] = {"H0": float(h0[idx[0]]), "f_agn": float(f[idx[1]]),
                  "dmu_G": float(mg[idx[2]]), "dmu_chi": float(mu[idx[3]]),
                  "logL": float(llm[idx]), "index": [int(x) for x in idx]}

    # The six 2-D marginals, from the three already-computed 3-D reductions.
    out["marginals_2d"] = {
        "H0_f_agn": np.trapz(P_over_mu, mg, axis=2),
        "H0_dmu_G": np.trapz(P_over_mu, f, axis=1),
        "H0_dmu_chi": np.trapz(P_over_mg, f, axis=1),
        "f_agn_dmu_G": np.trapz(P_over_mu, h0, axis=0),
        "f_agn_dmu_chi": np.trapz(P_over_mg, h0, axis=0),
        "dmu_G_dmu_chi": np.trapz(P_over_f, h0, axis=0),
    }
    out["_P"] = P
    return out


def _edge_mass_4(marg, h0, f, mg, mu, criterion=1e-6):
    out = {}
    for name, x, key in (("H0", h0, "H0"), ("f_agn", f, "f_agn"),
                         ("dmu_G", mg, "dmu_G"), ("dmu_chi", mu, "dmu_chi")):
        lp = np.asarray(marg[key]["marginal_logp"], dtype=float)
        p = np.exp(lp - np.nanmax(lp[np.isfinite(lp)]))
        out[name] = {"low_edge_density_over_peak": float(p[0]),
                     "high_edge_density_over_peak": float(p[-1]),
                     "low_edge_value": float(x[0]), "high_edge_value": float(x[-1]),
                     "criterion": f"both <= {criterion:g} of the peak",
                     "contained": bool(p[0] <= criterion and p[-1] <= criterion)}
    if not all(v["contained"] for v in out.values()):
        failed = [k for k, v in out.items() if not v["contained"]]
        print(f"[C10-J] WARNING: edge-mass containment FAILS on axis(es): {failed}")
    return out


_NEIGH4 = [d for d in itertools.product((-1, 0, 1), repeat=4) if any(d)]


def _guard_report_4d(h0, f, mg, mu, ll, fin, pull):
    Neff, thr = pull("Neff"), pull("threshold")
    rejected = ~fin
    rep = {
        "n_cells": int(ll.size), "n_rejected": int(rejected.sum()),
        "rejected_fraction": float(rejected.mean()),
        "Neff_min": float(np.nanmin(Neff)), "Neff_median": float(np.nanmedian(Neff)),
        "Neff_max": float(np.nanmax(Neff)),
        "threshold_min": float(np.nanmin(thr)), "threshold_max": float(np.nanmax(thr)),
        "Neff_over_threshold_min": float(np.nanmin(Neff / thr)),
        "policy": A9.GUARD_POLICY,
        "posterior_mass_in_rejected_region_as_scanned": 0.0,
        "posterior_mass_note": A9.GUARD_CIRCULARITY_NOTE,
        "rejected_cells": [],
    }
    idx = np.argwhere(rejected)
    for a, i, k, j in idx[:5000]:
        rep["rejected_cells"].append({
            "H0": float(h0[a]), "f_agn": float(f[i]), "dmu_G": float(mg[k]),
            "dmu_chi": float(mu[j]), "Neff": float(Neff[a, i, k, j]),
            "threshold": float(thr[a, i, k, j]),
            "Neff_over_threshold": float(Neff[a, i, k, j] / thr[a, i, k, j])})
    if idx.size:
        rep["rejected_H0_range"] = [float(h0[idx[:, 0].min()]), float(h0[idx[:, 0].max()])]
        rep["rejected_f_min"] = float(f[idx[:, 1].min()])
        rep["rejected_dmu_G_range"] = [float(mg[idx[:, 2].min()]), float(mg[idx[:, 2].max()])]
        rep["rejected_dmu_chi_min"] = float(mu[idx[:, 3].min()])

    llm = np.where(fin, ll, -np.inf)
    mx = float(llm[np.isfinite(llm)].max())
    P = np.where(np.isfinite(llm), np.exp(llm - mx), 0.0)
    W = (GC._trapz_weights(h0)[:, None, None, None]
         * GC._trapz_weights(f)[None, :, None, None]
         * GC._trapz_weights(mg)[None, None, :, None]
         * GC._trapz_weights(mu)[None, None, None, :])
    rep["accepted_mass_normalisation"] = float((W * P).sum())
    if rep["n_rejected"] == 0:
        rep["upper_bound_rejected_mass_fraction"] = 0.0
        rep["passes"] = True
        return rep
    shape = P.shape
    nb, local = [], []
    for a, i, k, j in idx:
        here = []
        for da, di, dk, dj in _NEIGH4:
            x, y, z, w = a + da, i + di, k + dk, j + dj
            if (0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]
                    and 0 <= w < shape[3] and fin[x, y, z, w]):
                here.append(float(P[x, y, z, w]))
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
    rep["fill_is_an_over_estimate"] = {
        "fill_value_relative_to_peak": P_bound,
        "n_rejected_cells": int(len(local)),
        "n_cells_where_fill_ge_local_accepted_boundary_max":
            int(sum(1 for v in local if P_bound >= v)),
        "max_local_accepted_boundary_density": float(max(local)) if local else 0.0,
        "verified": bool(all(P_bound >= v for v in local)),
    }
    rep["passes"] = bool(rep["upper_bound_rejected_mass_fraction"] < 1e-6)
    return rep


def _closure_j_slab_check(h0_axis, ll, fin, pull):
    """FREE CLOSURE: the H0 = 67.74 slab of C10-J against the fixed-H0
    results/a10_arm_J.h5 cube -- cell for cell, in ULP."""
    import h5py
    a10_j_path = RESULTS / "a10_arm_J.h5"
    if not a10_j_path.exists():
        return {"available": False, "why": f"{a10_j_path} not found"}
    a = int(np.argmin(np.abs(h0_axis - J_H0_ANCHOR)))
    if h0_axis[a] != J_H0_ANCHOR:
        return {"available": False, "why": "67.74 is not a C10-J axis node"}
    with h5py.File(a10_j_path, "r") as h:
        ref_f, ref_mg, ref_mu = h["f_grid"][:], h["dmu_G_grid"][:], h["dmu_chi_grid"][:]
        ref_ll = h["log_likelihood"][:]
        ref_pe = h["guard/logL_pe"][:]
        ref_sel = h["guard/logL_selection"][:]
        ref_logmu = h["guard/log_mu"][:]
    pe, sel, lmu = pull("logL_pe"), pull("logL_selection"), pull("log_mu")
    blocks = []
    for i, f in enumerate(J_F_AXIS):
        fi = np.where(ref_f == f)[0]
        if fi.size != 1:
            raise RuntimeError(f"[fatal] f node {f} not a unique a10_arm_J f_grid node")
        for k, g in enumerate(J_MG_AXIS):
            gi = np.where(ref_mg == g)[0]
            if gi.size != 1:
                raise RuntimeError(f"[fatal] dmu_G node {g} not a unique a10_arm_J node")
            for j, mu in enumerate(J_MU_AXIS):
                mi = np.where(ref_mu == mu)[0]
                if mi.size != 1:
                    raise RuntimeError(f"[fatal] dmu_chi node {mu} not a unique "
                                       f"a10_arm_J node")
                fI, gI, mI = int(fi[0]), int(gi[0]), int(mi[0])
                cube = {"logL": float(ll[a, i, k, j]), "logL_pe": float(pe[a, i, k, j]),
                        "logL_selection": float(sel[a, i, k, j]),
                        "log_mu": float(lmu[a, i, k, j]), "finite": bool(fin[a, i, k, j])}
                ref = {"logL": float(ref_ll[fI, gI, mI]), "logL_pe": float(ref_pe[fI, gI, mI]),
                       "logL_selection": float(ref_sel[fI, gI, mI]),
                       "log_mu": float(ref_logmu[fI, gI, mI])}
                blk = C._cmp4(cube, ref)
                blk.update({"f_agn": float(f), "dmu_G": float(g), "dmu_chi": float(mu),
                           "cube": cube, "reference": ref})
                blocks.append(blk)
    worst = C._worst4(blocks)
    return {
        "available": True, "reference_file": "results/a10_arm_J.h5",
        "description": ("the H0 = 67.74 slab of C10-J against the fixed-H0 "
                        "A10-J cube, at every one of C10-J's (f, dmu_G, dmu_chi) "
                        "nodes -- all of which are exact nodes of the fixed-H0 "
                        "cube's finer lattice."),
        "n_compared": len(blocks), "worst": worst,
        "n_bitwise_identical_total": int(sum(
            1 for b in blocks if b["total"]["bitwise_identical"])),
        "tolerance_ulp": TOL["ulp_same_hardware"], "tolerance_abs": TOL["abs_logL"],
        "pass": bool(blocks and worst["max_abs_diff_any_term"] <= TOL["abs_logL"]
                    and worst["max_ulp_any_term"] <= TOL["ulp_same_hardware"]),
    }


def _j_vs_s_matched_lattice(h0_axis, marg_j, marginal_ci):
    import h5py
    s_path = RESULTS / "c10_arm_S.h5"
    if not s_path.exists():
        return {"available": False, "why": f"{s_path} not found; run s_assemble first"}
    with h5py.File(s_path, "r") as h:
        s_h0, s_f, s_ll = h["H0_grid"][:], h["f_grid"][:], h["log_likelihood"][:]
    if not np.array_equal(s_f, F_GRID):
        raise RuntimeError("[fatal] C10-S f_grid != F_GRID")
    idx = []
    for v in h0_axis:
        hit = np.where(s_h0 == v)[0]
        if hit.size != 1:
            raise RuntimeError(f"[fatal] C10-J H0 node {v} is not a unique C10-S node")
        idx.append(int(hit[0]))
    idx = np.asarray(idx)

    def marginal(h0, ll):
        llm = np.where(np.isfinite(ll), ll, -np.inf)
        mx = float(llm[np.isfinite(llm)].max())
        P = np.where(np.isfinite(llm), np.exp(llm - mx), 0.0)
        with np.errstate(divide="ignore"):
            return marginal_ci(h0, np.log(np.trapz(P, s_f, axis=1)))

    sub = marginal(h0_axis, s_ll[idx, :])
    w = lambda b, lev: float(b[lev][1] - b[lev][0])
    j = marg_j["H0"]
    out = {
        "available": True,
        "C10_S_on_the_matched_lattice": {"median": sub["median"], "ci68": sub["ci68"],
                                         "ci90": sub["ci90"], "width68": w(sub, "ci68"),
                                         "width90": w(sub, "ci90")},
        "C10_J": {"median": j["median"], "ci68": j["ci68"], "ci90": j["ci90"],
                 "width68": w(j, "ci68"), "width90": w(j, "ci90")},
        "ratio_J_over_S_matched_68": w(j, "ci68") / w(sub, "ci68"),
        "ratio_J_over_S_matched_90": w(j, "ci90") / w(sub, "ci90"),
        "median_shift_J_minus_S_matched": j["median"] - sub["median"],
        "map_H0_J": marg_j["map"]["H0"],
    }
    a9_path = A9.DIAG / "a9_h0_matched_lattice.json"
    if a9_path.exists():
        a9 = json.loads(a9_path.read_text())
        out["reference_analysis_9_DIFFERENT_MOCK"] = {
            "note": ("Analysis 9's numbers: the SPIN-ONLY mark, on a DIFFERENT "
                     "events file (events_marked_dmu0p10.h5, no mass mark). "
                     "Shown for reference only -- not a like-for-like "
                     "comparison to C10-J's two-mark result."),
            "J9_over_S9_matched_68": a9["ratios"]["J9_over_S9_sublattice_68"],
            "J9_over_S9_matched_90": a9["ratios"]["J9_over_S9_sublattice_90"],
            "J9_width68": a9["J9_marked"]["width68"],
            "J9_width90": a9["J9_marked"]["width90"],
        }
    return out


def stage_j_assemble(args):
    import h5py
    env = A9._cpu_setup("j_assemble", import_darksirens=False)
    h0_axis = j_h0_axis(*args.h0_window)
    done, headers, dropped = _load_done_j()

    shas = {h.get("darksirens_sha") for h in headers.values()}
    if shas and shas != {A9.DARKSIRENS_A8_SHA}:
        raise RuntimeError(f"[fatal] checkpoints carry SHA {shas}")
    md5s = {h.get("events_md5") for h in headers.values()}
    if md5s and md5s != {GW_MD5_A10}:
        raise RuntimeError(f"[fatal] checkpoints carry events md5 {md5s}")

    ll, fin, pull, missing = _j_cube(done, h0_axis)
    n_want = int(h0_axis.size * J_F_AXIS.size * J_MG_AXIS.size)
    if missing and not args.allow_partial:
        raise RuntimeError(
            f"[fatal] {len(missing)} of {n_want} rows missing; run the remaining "
            f"chunks or pass --allow_partial.  First missing: {missing[:5]}")
    if missing:
        _write(DIAG / "c10_j_coverage.json",
               {"rows_missing": missing, "rows_total": n_want,
                "checkpoints": [p.name for p in _j_checkpoints()]})
        print(f"[partial] {len(missing)} C10-J rows missing; wrote the coverage report")
        return

    scan_h0f = A8.import_scan_h0f()
    marginal_ci = scan_h0f.marginal_ci
    marg = _marginals_4d(h0_axis, J_F_AXIS, J_MG_AXIS, J_MU_AXIS, ll, fin, marginal_ci)
    P = marg.pop("_P")
    m2 = marg.pop("marginals_2d")
    GC._truth_flags(marg["H0"], {"planted": 67.74})
    GC._truth_flags(marg["f_agn"], {"planted": TRUTH["f_agn_planted"]})
    GC._truth_flags(marg["dmu_chi"], {"planted": TRUTH["dmu_chi_planted"],
                                      "realised": TRUTH["dmu_chi_realised"]})
    GC._truth_flags(marg["dmu_G"], {"planted": TRUTH["dmu_G_planted"]})

    edges = _edge_mass_4(marg, h0_axis, J_F_AXIS, J_MG_AXIS, J_MU_AXIS)
    guard = _guard_report_4d(h0_axis, J_F_AXIS, J_MG_AXIS, J_MU_AXIS, ll, fin, pull)
    slab_check = _closure_j_slab_check(h0_axis, ll, fin, pull)
    if slab_check.get("available"):
        w = slab_check["worst"]
        print(f"[67.74-slab closure] {slab_check['n_compared']} cells, "
              f"{slab_check['n_bitwise_identical_total']} bitwise identical, "
              f"worst {w['max_abs_diff_any_term']:.3e} abs / "
              f"{w['max_ulp_any_term']:.2f} ULP -> "
              f"{'PASS' if slab_check['pass'] else 'FAIL'}")
    h0_vs_s = _j_vs_s_matched_lattice(h0_axis, marg, marginal_ci)

    hdr = next(iter(headers.values())) if headers else {}
    summary = {
        "analysis": "analysis_10_mass_spin_marked_multitracer",
        "arm": "C10-J",
        "question": ("the reduced asymmetric cosmology cube: (H0, f_AGN, "
                     "dmu_chi, dmu_G) with H0 released, on the design's G2 "
                     "node allocation (H0 and dmu_G full-strength, f and "
                     "dmu_chi at stride 2)."),
        "file": "results/c10_arm_J.h5",
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "environment": env,
        "provenance": hdr.get("provenance"),
        "h0_window": list(args.h0_window),
        "free_parameters": ["H0", "fcat_2", A10.MU_CHI_C2_LABEL, A10.MU_G_C2_LABEL],
        "sampled_labels": hdr.get("sampled_labels"),
        "per_catalog_pop_params": hdr.get("per_catalog_pop_params"),
        "base_coord": hdr.get("base_coord"),
        "fixed_parameter_values": hdr.get("fixed_parameter_values"),
        "Om0_fixed": A8.OM0_FID,
        "events_file": GW_PATH_A10, "events_md5": GW_MD5_A10,
        "selection_file": hdr.get("selection_file"),
        "survey_paths": hdr.get("survey_paths"),
        "truth": dict(TRUTH),
        "grid": {"H0": h0_axis.tolist(), "f_agn": J_F_AXIS.tolist(),
                "dmu_G": J_MG_AXIS.tolist(), "dmu_chi": J_MU_AXIS.tolist()},
        "n_cells": int(ll.size), "n_rejected": int((~fin).sum()),
        "logL_max": marg["logL_max"], "map": marg["map"],
        "H0": marg["H0"], "f_agn": marg["f_agn"], "dmu_G": marg["dmu_G"],
        "dmu_chi": marg["dmu_chi"],
        "posterior_moments": marg["posterior_moments"],
        "correlations": marg["posterior_moments"]["correlation"],
        "edge_mass": edges,
        "guard": {k: v for k, v in guard.items() if k != "rejected_cells"},
        "guard_rejected_cells": guard.get("rejected_cells", []),
        "closure_67p74_slab_vs_a10_arm_J": slab_check,
        "H0_width_vs_C10_S_matched_lattice": h0_vs_s,
        "timing": {"cells": int(ll.size),
                   "gpu_hours": float(np.nansum(pull("seconds")) / 3600.0),
                   "median_seconds_per_cell": float(np.nanmedian(pull("seconds")))},
        "checkpoints": [p.name for p in _j_checkpoints()],
        "truncated_lines_dropped": dropped,
    }

    h5_path = RESULTS / "c10_arm_J.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("H0_grid", data=h0_axis)
        h5.create_dataset("f_grid", data=J_F_AXIS)
        h5.create_dataset("dmu_G_grid", data=J_MG_AXIS)
        h5.create_dataset("dmu_chi_grid", data=J_MU_AXIS)
        h5.create_dataset("log_likelihood", data=ll)
        h5.create_dataset("posterior_unnormalised", data=P)
        g = h5.create_group("guard")
        for key in ("Neff", "threshold", "pe_variance_sum", "sigma2_total",
                    "logL_selection", "logL_pe", "log_mu", "seconds"):
            g.create_dataset(key, data=pull(key))
        g.create_dataset("rejected", data=(~fin))
        g.attrs["n_rejected"] = int((~fin).sum())
        m = h5.create_group("marginal")
        for name in ("H0", "f_agn", "dmu_G", "dmu_chi"):
            m.create_dataset(name, data=np.exp(np.asarray(marg[name]["marginal_logp"])))
        m2g = h5.create_group("marginal_2d")
        for k, v in m2.items():
            m2g.create_dataset(k, data=v)
        h5.attrs["analysis"] = "analysis_10_mass_spin_marked_multitracer"
        h5.attrs["arm"] = "C10-J"
        h5.attrs["axis_order"] = json.dumps(list(NAMES4))
        h5.attrs["h0_window"] = json.dumps(list(args.h0_window))
        h5.attrs["labels"] = json.dumps(hdr.get("sampled_labels"))
        h5.attrs["base_coord"] = json.dumps(hdr.get("base_coord"))
        h5.attrs["truth"] = json.dumps(TRUTH)
        h5.attrs["Om0_fixed"] = A8.OM0_FID
        h5.attrs["events_file"] = GW_PATH_A10
        h5.attrs["events_md5"] = GW_MD5_A10
        h5.attrs["selection_file"] = str(hdr.get("selection_file"))
        h5.attrs["survey_paths"] = json.dumps(hdr.get("survey_paths"))
        h5.attrs["darksirens_sha"] = A9.DARKSIRENS_A8_SHA
    print(f"wrote {h5_path}")
    _write(RESULTS / "c10_arm_J.json", summary)
    for k in ("H0", "f_agn", "dmu_G", "dmu_chi"):
        b = summary[k]
        print(f"  {k:<8} median {b['median']:.6f}  68% [{b['ci68'][0]:.6f}, "
              f"{b['ci68'][1]:.6f}]  90% [{b['ci90'][0]:.6f}, {b['ci90'][1]:.6f}]")
    print(f"  correlations: {json.dumps(summary['correlations'])}")
    return summary


# =========================================================================== #
# STAGE: mech  (GPU, one arm per invocation) / mech_assemble (CPU)
# =========================================================================== #
def stage_mech(args):
    if args.arm not in MECH_ARMS:
        raise SystemExit(f"[fatal] --arm must be one of {list(MECH_ARMS)}")
    env = A9._gpu_setup("mech")
    env["a10_inputs"] = C.assert_a10_inputs(with_md5=False)
    h0_axis = j_h0_axis(*args.h0_window)
    spec = MECH_ARMS[args.arm]
    survey_paths = ([A8.SURVEY_GAL, A8.SURVEY_AGN] if spec["surveys"] == "gal_agn"
                    else [A8.SURVEY_GAL, A8.SURVEY_GAL])
    if spec["marks"] == "zero":
        dmu_chi_val, dmu_G_val = 0.0, 0.0
    else:
        dmu_chi_val, dmu_G_val = float(args.pin_dmu_chi), float(args.pin_dmu_G)

    path = DIAG / f"_c10_mech_{args.arm}.jsonl"
    hdr, rows, dropped = _read_jsonl(path)
    done = {_key1(r["H0"]): r for r in rows}
    todo = [h0 for h0 in h0_axis if _key1(h0) not in done]
    print(f"[C10-mech-{args.arm}] surveys={spec['surveys']} marks={spec['marks']} "
          f"(dmu_chi={dmu_chi_val}, dmu_G={dmu_G_val}): {len(done)}/{h0_axis.size} "
          f"H0 nodes done, {len(todo)} to do ({len(todo) * F_GRID.size} cells), "
          f"{dropped} truncated line(s) dropped")
    if not todo:
        print("nothing to do")
        return

    t_build = time.time()
    cell = build_c10_cell(f"C10_MECH_{args.arm}", survey_paths)
    print(f"build: {time.time() - t_build:.1f}s")
    if hdr is None:
        _append_jsonl(path, _header(cell, f"C10-mech-{args.arm}", args.arm, {
            "h0_window": list(args.h0_window), "H0_grid": h0_axis.tolist(),
            "f_grid": F_GRID.tolist(), "survey_construction": spec["surveys"],
            "marks_pinned": spec["marks"], "measures": spec["measures"],
            "dmu_chi_value": dmu_chi_val, "dmu_G_value": dmu_G_val,
        }))

    t0 = time.time()
    for n, h0 in enumerate(todo, start=1):
        cells = [_eval(cell, h0, f, dmu_chi_val, dmu_G_val) for f in F_GRID]
        _append_jsonl(path, {
            "record": "row", "H0": float(h0), "key": _key1(h0), "cells": cells,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        lls = np.array([c["logL"] for c in cells], dtype=float)
        fin = np.isfinite(lls)
        el = time.time() - t0
        print(f"[C10-mech-{args.arm}] row {n}/{len(todo)} H0={h0:.2f} "
              f"maxlogL={(lls[fin].max() if fin.any() else float('nan')):.4f} "
              f"rejected={int((~fin).sum())}/{F_GRID.size} "
              f"elapsed={el/60:.1f}min eta={el/n*(len(todo)-n)/60:.1f}min")
        sys.stdout.flush()
        if args.stop_after_s and el > args.stop_after_s:
            print(f"[C10-mech-{args.arm}] stopping cleanly at the requested budget; "
                  f"{len(todo) - n} rows left")
            break
    print(f"[C10-mech-{args.arm}] done -> {path}")


def stage_mech_assemble(args):
    import h5py
    env = A9._cpu_setup("mech_assemble", import_darksirens=False)
    h0_axis = j_h0_axis(*args.h0_window)
    scan_h0f = A8.import_scan_h0f()
    marginal_ci = scan_h0f.marginal_ci

    arms = {}
    for arm in ("P1", "I0", "I1"):
        path = DIAG / f"_c10_mech_{arm}.jsonl"
        hdr, rows, dropped = _read_jsonl(path)
        done = {_key1(r["H0"]): r for r in rows}
        missing = [float(h0) for h0 in h0_axis if _key1(h0) not in done]
        if missing and not args.allow_partial:
            raise RuntimeError(f"[fatal] mech arm {arm}: {len(missing)} of "
                               f"{h0_axis.size} rows missing")
        if missing:
            _write(DIAG / f"c10_mech_{arm}_coverage.json", {"rows_missing": missing})
            print(f"[partial] mech {arm}: {len(missing)} rows missing")
            return
        ll = np.full((h0_axis.size, F_GRID.size), np.nan)
        fin = np.zeros_like(ll, dtype=bool)
        secs = []
        for a, h0 in enumerate(h0_axis):
            cells = done[_key1(h0)]["cells"]
            for i, cdict in enumerate(cells):
                ll[a, i] = cdict["logL"]
                fin[a, i] = bool(cdict["finite"])
                secs.append(cdict.get("seconds", np.nan))
        marg = _marginals_h0_f(h0_axis, F_GRID, ll, fin, marginal_ci)
        P = marg.pop("_P")
        arms[arm] = {"hdr": hdr, "ll": ll, "fin": fin, "marg": marg, "P": P,
                     "gpu_hours": float(np.nansum(secs) / 3600.0),
                     "spec": MECH_ARMS[arm]}

    # P0: the h0_window slice of results/c10_arm_S.h5 -- NOT recomputed.
    s_path = RESULTS / "c10_arm_S.h5"
    if not s_path.exists():
        raise RuntimeError(f"[fatal] {s_path} not found; run s_assemble first "
                           f"(P0 is its h0_window slice, not recomputed)")
    with h5py.File(s_path, "r") as h:
        s_h0, s_f, s_ll = h["H0_grid"][:], h["f_grid"][:], h["log_likelihood"][:]
    if not np.array_equal(s_f, F_GRID):
        raise RuntimeError("[fatal] C10-S f_grid != F_GRID")
    idx = []
    for v in h0_axis:
        hit = np.where(s_h0 == v)[0]
        if hit.size != 1:
            raise RuntimeError(f"[fatal] mech H0 node {v} is not a unique C10-S node")
        idx.append(int(hit[0]))
    idx = np.asarray(idx)
    ll_p0 = s_ll[idx, :]
    fin_p0 = np.isfinite(ll_p0)
    marg_p0 = _marginals_h0_f(h0_axis, F_GRID, ll_p0, fin_p0, marginal_ci)
    P0 = marg_p0.pop("_P")
    arms["P0"] = {"marg": marg_p0, "P": P0, "gpu_hours": 0.0,
                 "spec": {"surveys": "gal_agn", "marks": "zero",
                         "measures": "the unmarked H0 width; the h0_window "
                                     "slice of results/c10_arm_S.h5"}}

    w = lambda b, lev: float(b[lev][1] - b[lev][0])
    ratios = {}
    for lev in ("ci68", "ci90"):
        wP1, wP0 = w(arms["P1"]["marg"]["H0"], lev), w(arms["P0"]["marg"]["H0"], lev)
        wI1, wI0 = w(arms["I1"]["marg"]["H0"], lev), w(arms["I0"]["marg"]["H0"], lev)
        total_gain = wP1 / wP0
        siren_gain = wI1 / wI0
        ratios[lev] = {
            "width_P0": wP0, "width_P1": wP1, "width_I0": wI0, "width_I1": wI1,
            "W_P1_over_W_P0__total_gain": total_gain,
            "W_I1_over_W_I0__spectral_siren_only_gain": siren_gain,
            "routing_factor__total_over_siren": total_gain / siren_gain,
            "ln_routing_factor": float(np.log(total_gain) - np.log(siren_gain)),
        }

    hdr_any = next((v["hdr"] for v in arms.values() if v.get("hdr")), {})
    out = {
        "analysis": "analysis_10_mass_spin_marked_multitracer",
        "stage": "mech",
        "question": ("section-21 mechanism diagnostic: how much of the marked "
                     "H0 gain is tracer ROUTING vs the intrinsic (mass+spin, "
                     "spectral-siren) channel, at the marks pinned to their "
                     "C10-J MAP (or the fixed-H0 MAP default)."),
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "environment": env,
        "h0_window": list(args.h0_window), "H0_grid": h0_axis.tolist(),
        "pin_dmu_chi": args.pin_dmu_chi, "pin_dmu_G": args.pin_dmu_G,
        "arms": {
            arm: {
                "surveys": v["spec"]["surveys"], "marks": v["spec"]["marks"],
                "measures": v["spec"]["measures"],
                "H0": v["marg"]["H0"], "map": v["marg"]["map"],
                "gpu_hours": v["gpu_hours"],
            } for arm, v in arms.items()
        },
        "ratios": ratios,
        "attribution_caveat": (
            "the two shares need not compose multiplicatively; the residual "
            "ln[W(P1)/W(P0)] - ln[W(I1)/W(I0)] (= ln_routing_factor above) is "
            "an ATTRIBUTION WITH AN INTERACTION TERM and is reported, not "
            "assumed away.  Arm I (I0, I1) is a DIAGNOSTIC construction, not a "
            "physical model: passing the GAL survey twice also changes the "
            "SELECTION term (mu is recomputed with GAL in both mixture slots), "
            "so the I-ratio is not literally the spectral-siren term of the "
            "production model -- it is a control whose ratio to its own "
            "baseline is the interpretable quantity."),
        "checkpoints": {arm: f"_c10_mech_{arm}.jsonl" for arm in ("P1", "I0", "I1")},
    }

    h5_path = RESULTS / "c10_mech.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("H0_grid", data=h0_axis)
        h5.create_dataset("f_grid", data=F_GRID)
        for arm in ("P1", "I0", "I1"):
            g = h5.create_group(arm)
            g.create_dataset("log_likelihood", data=arms[arm]["ll"])
            g.create_dataset("rejected", data=(~arms[arm]["fin"]))
        g0 = h5.create_group("P0")
        g0.create_dataset("log_likelihood", data=ll_p0)
        g0.create_dataset("rejected", data=(~fin_p0))
        h5.attrs["analysis"] = "analysis_10_mass_spin_marked_multitracer"
        h5.attrs["arm"] = "C10-mech"
        h5.attrs["h0_window"] = json.dumps(list(args.h0_window))
        h5.attrs["pin_dmu_chi"] = float(args.pin_dmu_chi)
        h5.attrs["pin_dmu_G"] = float(args.pin_dmu_G)
    print(f"wrote {h5_path}")
    _write(RESULTS / "c10_mech.json", out)
    for lev, blk in ratios.items():
        print(f"  {lev}: total_gain={blk['W_P1_over_W_P0__total_gain']:.4f} "
              f"siren_gain={blk['W_I1_over_W_I0__spectral_siren_only_gain']:.4f} "
              f"routing={blk['routing_factor__total_over_siren']:.4f}")
    return out


# =========================================================================== #
# STAGE: status  (CPU, all families that have checkpoints on disk)
# =========================================================================== #
def stage_status(args):
    out = {}

    p_path = DIAG / "_c10_profile.jsonl"
    if p_path.exists():
        _hdr, rows, dropped = _read_jsonl(p_path)
        axis = np.array(A9.J9_H0_AXIS, dtype=float)
        have = {_key1(r["H0"]) for r in rows} & {_key1(h0) for h0 in axis}
        out["profile"] = {"rows_done": len(have), "rows_total": int(axis.size),
                          "complete": len(have) == axis.size,
                          "truncated_lines_dropped": dropped}

    if _s_checkpoints():
        done, _headers, dropped = _load_done_s()
        axis = np.array(A9.S9_H0_AXIS, dtype=float)
        want = {_key1(h0) for h0 in axis}
        have = want & set(done)
        secs = [c["seconds"] for _, r in done.values() for c in r["cells"]]
        out["s"] = {
            "rows_done": len(have), "rows_total": len(want),
            "cells_done": len(have) * F_GRID.size, "cells_total": len(want) * F_GRID.size,
            "fraction": len(have) / max(len(want), 1),
            "gpu_hours_spent": float(np.sum(secs) / 3600.0) if secs else 0.0,
            "gpu_hours_remaining": (float((len(want) - len(have)) * F_GRID.size
                                          * np.median(secs) / 3600.0) if secs else None),
            "checkpoints": [p.name for p in _s_checkpoints()],
            "truncated_lines_dropped": dropped,
        }

    if _j_checkpoints():
        done, _headers, dropped = _load_done_j()
        h0_axis = j_h0_axis(*args.h0_window)
        want = {_key3(h0, f, g) for h0 in h0_axis for f in J_F_AXIS for g in J_MG_AXIS}
        have = want & set(done)
        secs = [c["seconds"] for _, r in done.values() for c in r["cells"]]
        out["j"] = {
            "h0_window": list(args.h0_window),
            "rows_done": len(have), "rows_total": len(want),
            "cells_done": len(have) * J_MU_AXIS.size,
            "cells_total": len(want) * J_MU_AXIS.size,
            "fraction": len(have) / max(len(want), 1),
            "gpu_hours_spent": float(np.sum(secs) / 3600.0) if secs else 0.0,
            "gpu_hours_remaining": (float((len(want) - len(have)) * J_MU_AXIS.size
                                          * np.median(secs) / 3600.0) if secs else None),
            "checkpoints": [p.name for p in _j_checkpoints()],
            "truncated_lines_dropped": dropped,
        }

    mech_files = sorted(DIAG.glob("_c10_mech_*.jsonl"))
    if mech_files:
        h0_axis = j_h0_axis(*args.h0_window)
        arms_status = {}
        for arm in ("P1", "I0", "I1"):
            path = DIAG / f"_c10_mech_{arm}.jsonl"
            if not path.exists():
                continue
            _hdr, rows, dropped = _read_jsonl(path)
            have = {_key1(r["H0"]) for r in rows} & {_key1(h0) for h0 in h0_axis}
            secs = [c["seconds"] for r in rows for c in r["cells"]]
            arms_status[arm] = {
                "rows_done": len(have), "rows_total": int(h0_axis.size),
                "complete": len(have) == h0_axis.size,
                "gpu_hours_spent": float(np.sum(secs) / 3600.0) if secs else 0.0,
                "truncated_lines_dropped": dropped,
            }
        out["mech"] = arms_status

    print(json.dumps(out, indent=2, default=GC._json_default))
    return out


# =========================================================================== #
# STAGE: dry_run  (CPU, no data, no likelihood trace)
# =========================================================================== #
def _report_profile_grids():
    import h5py
    axis = np.array(A9.J9_H0_AXIS, dtype=float)
    with h5py.File(A9.RESULTS / "j9_marked.h5", "r") as h:
        ref = h["H0_grid"][:]
    ok = bool(np.array_equal(axis, ref))
    n_cells = int(axis.size * 2)
    return {"H0_nodes": int(axis.size), "bitwise_equal_to_a9_j9_H0_grid": ok,
            "n_cells": n_cells, "gpu_hours_projected": n_cells * SECONDS_PER_EVAL / 3600.0}


def _report_s_grids():
    import h5py
    axis = np.array(A9.S9_H0_AXIS, dtype=float)
    with h5py.File(A9.RESULTS / "s9_spatial.h5", "r") as h:
        ref = h["H0_grid"][:]
    ok = bool(np.array_equal(axis, ref))
    n_cells = int(axis.size * F_GRID.size)
    return {"H0_nodes": int(axis.size), "f_nodes": int(F_GRID.size),
            "bitwise_equal_to_a9_s9_H0_grid": ok,
            "n_rows": int(axis.size), "n_cells": n_cells,
            "n_chunks": 4,
            "gpu_hours_projected": n_cells * SECONDS_PER_EVAL / 3600.0,
            "projected_chunk_hours": (n_cells / 4) * SECONDS_PER_EVAL / 3600.0}


def _report_j_grids(window):
    h0 = j_h0_axis(*window)
    n_rows = int(h0.size * J_F_AXIS.size * J_MG_AXIS.size)
    n_cells = int(n_rows * J_MU_AXIS.size)
    default_match = None
    if tuple(float(w) for w in window) == DEFAULT_H0_WINDOW:
        default_match = bool(np.array_equal(h0, np.array(A9.J9_H0_AXIS, dtype=float)))
    chunk_cells = n_cells / 8.0
    return {
        "h0_window": list(window), "H0_nodes": int(h0.size),
        "f_nodes": int(J_F_AXIS.size), "f_axis": J_F_AXIS.tolist(),
        "dmu_chi_nodes": int(J_MU_AXIS.size), "dmu_chi_axis": J_MU_AXIS.tolist(),
        "dmu_G_nodes": int(J_MG_AXIS.size), "dmu_G_axis": J_MG_AXIS.tolist(),
        "H0_axis_equals_a9_J9_axis_when_default_window": default_match,
        "n_rows": n_rows, "n_cells": n_cells, "n_chunks": 8,
        "gpu_hours_projected": n_cells * SECONDS_PER_EVAL / 3600.0,
        "projected_chunk_seconds": chunk_cells * SECONDS_PER_EVAL,
        "projected_chunk_hours": chunk_cells * SECONDS_PER_EVAL / 3600.0,
        "recommended_sbatch_time_1p3x_margin_hours":
            chunk_cells * SECONDS_PER_EVAL / 3600.0 * 1.3,
    }


def _report_mech_grids(window, pin_mu, pin_mg):
    h0 = j_h0_axis(*window)
    n_cells_per_arm = int(h0.size * F_GRID.size)
    return {
        "h0_window": list(window), "H0_nodes": int(h0.size),
        "f_nodes": int(F_GRID.size), "n_arms": 3,
        "pin_dmu_chi": pin_mu, "pin_dmu_G": pin_mg,
        "n_cells_per_arm": n_cells_per_arm, "n_cells_total": n_cells_per_arm * 3,
        "gpu_hours_projected_per_arm": n_cells_per_arm * SECONDS_PER_EVAL / 3600.0,
        "gpu_hours_projected_total": n_cells_per_arm * 3 * SECONDS_PER_EVAL / 3600.0,
        "note": "P0 is a slice of C10-S and is not counted here",
    }


def stage_dry_run(args):
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    print("== C10 dry run: parameter space on CPU, no data, no likelihood trace ==")
    A10._selftest()

    which = (["profile", "s", "j", "mech"] if args.which == "all"
             else [w.strip() for w in args.which.split(",") if w.strip()])
    out = {"which": which, "seconds_per_eval": SECONDS_PER_EVAL}
    if "profile" in which:
        out["profile"] = _report_profile_grids()
    if "s" in which:
        out["s"] = _report_s_grids()
    if "j" in which:
        out["j"] = _report_j_grids(args.h0_window)
    if "mech" in which:
        out["mech"] = _report_mech_grids(args.h0_window, args.pin_dmu_chi, args.pin_dmu_G)
    print(json.dumps(out, indent=2, default=GC._json_default))
    _write(DIAG / "c10_dry_run.json", out)
    return out


# =========================================================================== #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=(
        "profile", "s", "s_assemble", "j", "j_assemble",
        "mech", "mech_assemble", "status", "dry_run"))
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--n_chunks", type=int, default=1)
    ap.add_argument("--h0_window", type=float, nargs=2, default=list(DEFAULT_H0_WINDOW),
                    help="j/j_assemble/mech/mech_assemble/status/dry_run: the "
                         "H0 window on the 0.5 lattice (+ 67.74); default [63, 76]")
    ap.add_argument("--pin_dmu_chi", type=float, default=PROFILE_DMU_CHI,
                    help="mech/mech_assemble/dry_run: the pinned dmu_chi mark "
                         "(default the fixed-H0 joint MAP, +0.115)")
    ap.add_argument("--pin_dmu_G", type=float, default=PROFILE_DMU_G,
                    help="mech/mech_assemble/dry_run: the pinned dmu_G mark "
                         "(default the fixed-H0 joint MAP, +5.0)")
    ap.add_argument("--arm", choices=list(MECH_ARMS), default=None,
                    help="mech: which of P1/I0/I1 this worker computes")
    ap.add_argument("--which", default="all",
                    help="status/dry_run: comma list of profile,s,j,mech or 'all'")
    ap.add_argument("--allow_partial", action="store_true",
                    help="*_assemble: write a coverage report instead of failing")
    ap.add_argument("--stop_after_s", type=float, default=0.0,
                    help="stop cleanly after this many seconds of scanning")
    args = ap.parse_args(argv)

    if args.stage == "mech" and args.arm is None:
        raise SystemExit("[fatal] --stage mech requires --arm {P1,I0,I1}")

    return {
        "profile": stage_profile,
        "s": stage_s, "s_assemble": stage_s_assemble,
        "j": stage_j, "j_assemble": stage_j_assemble,
        "mech": stage_mech, "mech_assemble": stage_mech_assemble,
        "status": stage_status, "dry_run": stage_dry_run,
    }[args.stage](args)


if __name__ == "__main__":
    main()
