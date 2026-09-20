#!/usr/bin/env python
"""Analysis 9 -- Analysis 8 with H0 freed, and nothing else.

Three free coordinates, exactly:

    H0            the Hubble constant (sampled; Analysis 8 pinned it at 67.74)
    fcat_2        f_AGN, the AGN mixture weight (stick-breaking at K=2)
    mu_chi_c2     the AGN branch's ABSOLUTE effective-spin mean.  It equals the
                  planted dmu_chi only because mu_chi_GAL is pinned at the
                  powerlaw+peak fiducial 0.0.  Both spellings are carried in
                  every output; they are not the same quantity.

Everything else is the Analysis-8 configuration, and nothing about the
likelihood is reimplemented here.  The likelihood, the opts namespace, the
guard spy, the KDE window, the twelve pinned base population parameters and the
provenance assertions all come from

    analysis_8_marked_multitracer_H0_fagn/scripts/a8_likelihood.py

and the registered (f, mu_chi_c2) plane, the truth dictionary, the per-cell
record, the trapezoid weights and the 2-D marginal convention come from

    analysis_8_marked_multitracer_H0_fagn/scripts/gate_c_three_arms.py

which in turn imports ``marginal_ci`` from Analysis 2's production driver
``analysis_2_complete_catalog_H0_fagn/scripts/scan_h0f.py``.  The Analysis-8
tree is READ-ONLY: this module imports from it and writes nothing into it
(``sys.dont_write_bytecode`` is set before the import so not even a ``.pyc``
lands there).

ENVIRONMENT (load-bearing, asserted at the start of every stage)

    export PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-a8

Only that checkout carries the tracer-dependent population blocks.  The
editable install -- not DARKSIRENS_SRC -- decides the import, so without
PYTHONPATH the run silently becomes Analysis 2 with extra coordinates.  Every
stage asserts ``darksirens.__file__`` is under ``darksirens-a8`` and that the
checkout is at af896cae6f3f3dd1f87dec50046e3a8228f59b39.

COMPUTE.  Every GPU stage runs on RITA via SLURM and refuses to run anywhere
else; the local H100 stays free.

    #SBATCH --partition=RITA-GPU --qos=rita --account=phy220048p
    #SBATCH --gres=gpu:a100-80:1 --cpus-per-task=8 --mem=100G

STAGES (a queue eviction or a crash costs at most one 61-cell row)

    provenance   CPU only.  Prints and records every environment assertion.
    timing       GPU.  A handful of probe cells: the per-eval rate on rita, the
                 projected cube cost, and the H0 LIVENESS probe (an equivalence
                 test passes for free if the new coordinate is ignored).
    anchor       GPU.  The H0 = 67.74 slab alone: the same 41 x 61 cells
                 Analysis 8 ran as arm J, so it is the equivalence gate and it
                 is also 1/36 of the cube -- nothing is thrown away.
    scan         GPU.  The registered cube, split over the H0 axis with
                 --chunk/--n_chunks; each worker appends to its OWN checkpoint.
    compare_a8   CPU.  The anchor slab against arm_J_joint.h5, cell by cell.
    status       CPU.  Rows done, cells done, ETA.
    assemble     CPU.  Merge, coverage check, marginals, guard bound, results.

Checkpoints are append-only JSONL keyed on the PHYSICAL coordinates
(``diagnostics/_a9_scan_<tag>.jsonl``), not on grid indices, so trimming or
extending the H0 axis re-uses every completed row instead of rescanning it.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Never write a .pyc into the read-only Analysis-8 tree.
sys.dont_write_bytecode = True

# --------------------------------------------------------------------------- #
# Paths of record
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE.parent
DIAG = ANALYSIS_DIR / "diagnostics"
RESULTS = ANALYSIS_DIR / "results"

WORKING = Path("/hildafs/projects/phy230014p/magana/gws-agn/working")
A8_DIR = WORKING / "analyses" / "analysis_8_marked_multitracer_H0_fagn"
A8_SCRIPTS = A8_DIR / "scripts"
A8_ARM_J_H5 = A8_DIR / "results" / "arm_J_joint.h5"
A8_ARM_J_JSON = A8_DIR / "results" / "arm_J_joint.json"
A2_DIR = WORKING / "analyses" / "analysis_2_complete_catalog_H0_fagn"
A2_JOINT_H5 = A2_DIR / "results" / "joint_s100.h5"
A2_JOINT_JSON = A2_DIR / "results" / "h0_fagn_joint.json"     # the summary
A2_SEED100_JSON = A2_DIR / "results" / "joint_s100.json"      # ci68 AND ci90

DARKSIRENS_A8 = "/hildafs/projects/phy230014p/magana/src/darksirens-a8"
DARKSIRENS_A8_SHA = "af896cae6f3f3dd1f87dec50046e3a8228f59b39"
DARKSIRENS_PINNED_BASE = "2b86a2d"
GWS_AGN_REPO = "/hildafs/projects/phy230014p/magana/gws-agn"

GPU_STAGES = ("timing", "anchor", "scan", "s9")
CPU_STAGES = ("provenance", "compare_a8", "status", "assemble",
              "s9_assemble", "gate_a_full", "section_11", "s9_status")


# --------------------------------------------------------------------------- #
# Analysis-8 machinery, imported (never copied)
# --------------------------------------------------------------------------- #
def _import_a8():
    sys.dont_write_bytecode = True
    if str(A8_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(A8_SCRIPTS))
    import a8_likelihood as a8            # the likelihood builder of record
    import gate_c_three_arms as gc        # the registered grid + truths + helpers
    return a8, gc


A8, GC = _import_a8()

# The (f, mu_chi_c2) plane is Analysis 8's own array object, not a retyped copy.
F_GRID = np.asarray(GC.F_GRID_2D, dtype=float)          # linspace(0, 1, 41)
MU_GRID = np.asarray(GC.MU_GRID_2D, dtype=float)        # linspace(-0.20, 0.25, 61)

# The new axis.  Anchored EXACTLY on Analysis 8's fixed H0 so the anchor slab is
# cell-for-cell the arm-J grid.  Range sized from the measured Analysis-2
# seed-100 (H0, f) marginal: 4.50e-11 of its mass falls outside [60.24, 77.74],
# whose edge densities are 3.66e-13 and 3.48e-10 of the peak (single mode, 69.25).
H0_ANCHOR = float(A8.H0_FID)                            # 67.74
H0_STEP = 0.5
H0_AXES = {
    "registered": (-15, 20),      # 36 nodes, [60.24, 77.74]
    "fallback": (-10, 14),        # 25 nodes, [62.74, 74.74]; a2 mass outside 6.40e-07
}

# ---- the two PRODUCTION arms (specification section 11) -------------------- #
# S9 is Analysis 2's own registered joint H0 axis -- [50, 100] on 201 nodes,
# read here as the identical linspace -- PLUS the anchor 67.74, so the anchor is
# an S9 node and S9 is directly comparable to the unmarked Analysis-2 scan.
A2_H0_AXIS = np.linspace(50.0, 100.0, 201)
S9_H0_AXIS = np.unique(np.concatenate([A2_H0_AXIS, [H0_ANCHOR]]))          # 202
# J9 is the confirmed containment window on 0.5 spacing, every node a STRICT S9
# node position, plus the anchor.  The window is set by S9's OWN measured H0
# marginal (stage s9_assemble -> j9_window_check), not by the earlier proposal.
J9_WINDOW = (63.0, 76.0)
J9_H0_AXIS = np.unique(np.concatenate([
    np.arange(J9_WINDOW[0], J9_WINDOW[1] + 1e-9, H0_STEP), [H0_ANCHOR]]))  # 28
H0_AXES_EXPLICIT = {"s9": S9_H0_AXIS, "j9": J9_H0_AXIS}
# Every J9 node must be BITWISE an S9 node or the section-11 width comparison is
# an artefact of spacing rather than a measurement.
assert all(np.any(S9_H0_AXIS == v) for v in J9_H0_AXIS), (
    "J9 H0 nodes are not a strict subset of the S9 nodes")

GUARD_POLICY = (
    "Analysis 8 C6: no result may stand behind a guard-rejected cell.  Every "
    "rejected cell is listed with its N_eff and threshold, and the mass behind "
    "the rejected region is bounded from above by FILLING each rejected cell "
    "with the largest posterior density found on the accepted boundary beside "
    "it.")
GUARD_CIRCULARITY_NOTE = (
    "A rejected cell returns logL = -inf, so on the grid as scanned it carries "
    "exactly zero mass -- that is circular and is reported as 0.0 only to name "
    "the circularity.  The bound below instead fills every rejected cell with "
    "the largest accepted density adjacent to the rejected region, which "
    "over-states the mass there because logL falls monotonically towards the "
    "guard corner (checked, and the fill value is >= each cell's own accepted "
    "boundary maximum by construction).")

TRUTH = dict(GC.TRUTH)
TRUTH.update({
    "H0_planted": 67.74,
    "H0_note": (
        "67.74 is the planted cosmology.  Seed 100's own draw sits high: "
        "Analysis 2 recovered 69.217 (68% [68.249, 70.191]) on the UNMARKED "
        "data with the same catalogs and selection.  That is a property of this "
        "realisation, not a bias, so quote H0 differentially against Analysis 2 "
        "as well as against 67.74."),
})

TOLERANCES = {
    "anchor_equivalence_abs_logL": 1e-6,
    "anchor_equivalence_rel_logL": 2.4e-10,
    "anchor_equivalence_same_hardware_ulp": 4.0,
    "anchor_marginal_abs": 1e-4,
    "note": (
        "Analysis 8's arm J ran on the local H100 NVL and Analysis 9 runs on a "
        "rita A100-80, so the pre-registered criterion is an ABSOLUTE bound of "
        "1e-6 in logL (2.4e-10 relative at logL ~ -4.2e3) with the measured "
        "residual ALSO quoted in ULP (1 ULP = 9.0949470177292824e-13 at that "
        "scale).  Gate A of Analysis 8 measured 2 ULP for the same-node "
        "comparison; if the same node is used the 4 ULP criterion applies.  A "
        "measurement above the bound is a FAIL to be diagnosed, not widened."),
}


# --------------------------------------------------------------------------- #
# Small utilities (JSON, writes confined to this directory)
# --------------------------------------------------------------------------- #
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
        raise RuntimeError(f"[fatal] refusing to write outside {ANALYSIS_DIR}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # A kill mid-write leaves a line with no newline.  Close it before appending,
    # or the fragment and the next row merge into one unparsable line and BOTH
    # are lost -- which is precisely the eviction this checkpoint exists for.
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
    """Header + rows.  A truncated final line (a kill mid-write) is dropped."""
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


def _key(h0, f):
    """Checkpoint key: the PHYSICAL coordinates, so an axis change re-uses rows."""
    return "{:.10g}|{:.10g}".format(float(h0), float(f))


def h0_axis(name="registered", kmin=None, kmax=None):
    """The H0 axis of record.  Named explicit arrays win; the lattice is
    H0 = 67.74 + 0.5k and an explicit (kmin, kmax) override wins over both."""
    if kmin is not None and kmax is not None:
        return H0_ANCHOR + H0_STEP * np.arange(int(kmin), int(kmax) + 1)
    if name in H0_AXES_EXPLICIT:
        return np.array(H0_AXES_EXPLICIT[name], dtype=float)
    kmin, kmax = H0_AXES[name]
    k = np.arange(int(kmin), int(kmax) + 1)
    return H0_ANCHOR + H0_STEP * k


def _axis_from_args(args):
    axis = h0_axis(args.h0_axis, args.h0_kmin, args.h0_kmax)
    if not np.any(np.isclose(axis, H0_ANCHOR, rtol=0, atol=0)):
        raise RuntimeError(
            "[fatal] the H0 axis must contain the Analysis-8 anchor 67.74 exactly; "
            "the anchor slab IS the equivalence gate.")
    return axis


# --------------------------------------------------------------------------- #
# Provenance assertions -- printed, not assumed
# --------------------------------------------------------------------------- #
def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def assert_environment(stage, import_darksirens=True, verbose=True):
    """Every load-bearing environment assertion, printed as it is checked."""
    out = {"stage": stage, "checks": [], "host": socket.gethostname()}

    def check(name, ok, detail):
        out["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
        if verbose:
            print(f"  [{'OK ' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            raise SystemExit(f"[fatal] {name}: {detail}")

    if verbose:
        print("== provenance assertions ==")
    pp = [str(Path(p)) for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    check("PYTHONPATH carries darksirens-a8", DARKSIRENS_A8 in pp,
          f"PYTHONPATH={os.environ.get('PYTHONPATH', '<unset>')}")

    src = os.environ.get("DARKSIRENS_SRC")
    check("DARKSIRENS_SRC unset or darksirens-a8",
          src is None or str(Path(src)) == DARKSIRENS_A8,
          f"DARKSIRENS_SRC={src!r} (a8_likelihood.provenance reads it)")

    head = _run(["git", "-C", DARKSIRENS_A8, "rev-parse", "HEAD"])
    check("darksirens-a8 HEAD == af896ca", head == DARKSIRENS_A8_SHA,
          f"{head} (expected {DARKSIRENS_A8_SHA})")
    out["darksirens_sha"] = head

    dirty = bool(_run(["git", "-C", DARKSIRENS_A8, "status", "--porcelain"]))
    check("darksirens-a8 worktree clean", not dirty,
          "clean" if not dirty else "DIRTY -- the SHA no longer describes the code")

    core = Path(DARKSIRENS_A8) / "darksirens" / "likelihood" / "core.py"
    n_mix = core.read_text().count("mixture_pop_params")
    check("tracer-dependent population blocks present", n_mix >= 9,
          f"{n_mix} references to mixture_pop_params in {core}")
    out["mixture_pop_params_references"] = n_mix

    gws = _run(["git", "-C", GWS_AGN_REPO, "rev-parse", "HEAD"])
    out["gws_agn_sha"] = gws
    out["gws_agn_dirty"] = bool(_run(["git", "-C", GWS_AGN_REPO, "status", "--porcelain"]))
    if verbose:
        print(f"  [   ] gws-agn HEAD: {gws} (dirty={out['gws_agn_dirty']})")

    if import_darksirens:
        import darksirens
        check("darksirens imports from darksirens-a8",
              "darksirens-a8" in darksirens.__file__
              and darksirens.__file__.startswith(DARKSIRENS_A8),
              darksirens.__file__)
        out["darksirens_module_file"] = darksirens.__file__
        out["darksirens_version"] = getattr(darksirens, "__version__", None)
        # a8_likelihood's own assertions: pinned-base ancestry + module path.
        out["a8_provenance"] = A8.provenance(
            gw_path=A8.GW_PATH_MARKED, survey_paths=[A8.SURVEY_GAL, A8.SURVEY_AGN])
        check("pinned base 2b86a2d is an ancestor",
              out["a8_provenance"]["darksirens_pinned_base_is_ancestor"],
              f"base {DARKSIRENS_PINNED_BASE}")

    for tag, path in (("events", A8.GW_PATH_MARKED), ("survey_gal", A8.SURVEY_GAL),
                      ("survey_agn", A8.SURVEY_AGN), ("selection", A8.GWSEL_PATH)):
        p = Path(path)
        check(f"input present: {tag}", p.exists(), f"{p} ({p.stat().st_size} bytes)")
    out["inputs"] = {"events": A8.GW_PATH_MARKED, "survey_gal": A8.SURVEY_GAL,
                     "survey_agn": A8.SURVEY_AGN, "selection": A8.GWSEL_PATH}
    return out


def assert_rita(stage):
    """GPU stages run on RITA via SLURM.  The local H100 stays free."""
    host = socket.gethostname().split(".")[0]
    part = os.environ.get("SLURM_JOB_PARTITION", "")
    if host.startswith("rita") or part == "RITA-GPU":
        print(f"  [OK ] GPU host: {host} (SLURM_JOB_PARTITION={part!r}, "
              f"job {os.environ.get('SLURM_JOB_ID', '-')})")
        return host
    raise SystemExit(
        f"[fatal] stage '{stage}' is GPU work and must run on RITA via SLURM "
        f"(--partition=RITA-GPU --qos=rita --account=phy220048p --gres=gpu:a100-80:1); "
        f"this is {host}.  Submit scripts/submit_a9_rita.sbatch instead.")


def _gpu_setup(stage):
    """Env -> darksirens import -> assertions, in that order (JAX needs it)."""
    assert_rita(stage)
    A8.set_env(guard_record=True)
    import darksirens  # noqa: F401
    import jax
    env = assert_environment(stage, import_darksirens=True)
    env["jax_devices"] = str(jax.devices())
    print(f"  [   ] JAX devices: {env['jax_devices']}")
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise SystemExit("[fatal] no GPU device visible; this stage must run on a GPU.")
    return env


def _cpu_setup(stage, import_darksirens=True):
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return assert_environment(stage, import_darksirens=import_darksirens)


# --------------------------------------------------------------------------- #
# The likelihood: Analysis 8's, built by Analysis 8's own builder
# --------------------------------------------------------------------------- #
def build_cell(verbose=True):
    """The Analysis-8 'new' shape on the marked seed-100 data, H0 left sampled."""
    surveys = [A8.SURVEY_GAL, A8.SURVEY_AGN]
    cell = A8.build("A9", "new", surveys, verbose=verbose, gw_path=A8.GW_PATH_MARKED)
    if "H0" not in cell.labels:
        raise RuntimeError(f"[fatal] H0 is not a sampled coordinate: {cell.labels}")
    mu_fid, mu_label = _fiducial_mu_chi()
    if mu_fid != 0.0:
        raise RuntimeError(
            f"[fatal] mu_chi fiducial is {mu_fid}, not 0.0, so mu_chi_c2 is NOT "
            f"dmu_chi and every truth comparison must subtract it.")
    return cell, surveys


def _fiducial_mu_chi():
    labels, plain, fid = A8.population_labels_and_fiducial()
    idx = [i for i, p in enumerate(plain) if p == "mu_chi"]
    if len(idx) != 1:
        raise RuntimeError(f"expected exactly one 'mu_chi' slot, got {idx} in {plain}")
    return float(fid[idx[0]]), labels[idx[0]]


def eval_row(cell, h0, f, mu_grid):
    """One checkpoint unit: 61 cells at fixed (H0, f_AGN)."""
    L = A8.MU_CHI_C2_LABEL
    cells = []
    for mu in mu_grid:
        rec = cell.evaluate(H0=float(h0), fcat_2=float(f), **{L: float(mu)})
        cells.append(GC._cell_record(rec))
    return cells


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def _tag_path(tag):
    return DIAG / f"_a9_scan_{tag}.jsonl"


def _all_checkpoints():
    return sorted(DIAG.glob("_a9_scan_*.jsonl"))


def load_done(exclude=None):
    """Every row completed anywhere, keyed on the physical coordinates."""
    done, headers, dropped = {}, {}, 0
    for path in _all_checkpoints():
        if exclude is not None and path == Path(exclude):
            continue
        hdr, rows, drop = _read_jsonl(path)
        dropped += drop
        if hdr is not None:
            headers[path.name] = hdr
        for r in rows:
            done.setdefault(_key(r["H0"], r["f_agn"]), (path.name, r))
    return done, headers, dropped


def _header(cell, surveys, h0_grid, tag, extra=None):
    hdr = {
        "record": "header",
        "analysis": "analysis_9_marked_multitracer_H0_fagn",
        "tag": tag,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "provenance": A8.provenance(gw_path=A8.GW_PATH_MARKED, survey_paths=surveys),
        "survey_paths": [str(s) for s in surveys],
        "sampled_labels": cell.labels,
        "per_catalog_pop_params": list(cell.per_catalog_pop_params),
        "fixed_parameter_values": cell.fixed_parameter_values,
        "base_coord": {k: float(v) for k, v in zip(cell.labels, cell.base)},
        "free_parameters": ["H0", "fcat_2", A8.MU_CHI_C2_LABEL],
        "mu_chi_fiducial_GAL_branch": _fiducial_mu_chi()[0],
        "nEvents": int(cell.data["nEvents"]),
        "nsamp": int(cell.data["nsamp"]),
        "Ndraw": float(cell.data["Ndraw"]),
        "truth": dict(TRUTH),
        "H0_grid": h0_axis_record(h0_grid),
        "f_grid": F_GRID.tolist(),
        "mu_grid": MU_GRID.tolist(),
        "tolerances": dict(TOLERANCES),
    }
    if extra:
        hdr.update(extra)
    return hdr


def h0_axis_record(axis):
    axis = np.asarray(axis, dtype=float)
    d = np.diff(axis)
    return {"values": axis.tolist(), "n": int(axis.size),
            "min": float(axis.min()), "max": float(axis.max()),
            "step": H0_STEP, "anchor": H0_ANCHOR,
            "spacing_min": float(d.min()) if d.size else 0.0,
            "spacing_max": float(d.max()) if d.size else 0.0,
            "anchor_index": int(np.argmin(np.abs(axis - H0_ANCHOR)))}


def check_parameter_space():
    """Pre-flight, CPU only: the parameter space is Analysis 8's, with H0 sampled.

    Builds the SAME ``build_parameter_space`` call ``a8_likelihood.build`` makes
    -- no data, no likelihood trace -- so the three free coordinates are proved
    before any GPU time is spent.
    """
    from darksirens.inference.prior import build_parameter_space
    opts = A8.build_opts([A8.SURVEY_GAL, A8.SURVEY_AGN], gw_path=A8.GW_PATH_MARKED)
    opts.fix_population = False
    fpv = A8.fixed_parameter_values_for("new")
    res = build_parameter_space(
        opts.pop_model, opts.fix_population, opts.fix_cosmology, opts.fix_survey,
        fix_de=opts.fix_de, prior_overrides={}, fixed_parameter_values=fpv,
        universe_model=opts.universe_model, shared_beta=opts.shared_beta,
        shared_spin=opts.shared_spin, shared_gamma=opts.shared_gamma,
        sky_model=opts.sky_model, mark_model=opts.mark_model,
        mark_names=opts.mark_names, n_catalogs=2,
        lss_completion_active=[False, False], use_lss=bool(opts.use_LSS),
        mark_names_by_catalog=None, per_catalog_pop_params=("mu_chi_c2",))
    labels = list(res[0])
    free = ["H0", "fcat_2", A8.MU_CHI_C2_LABEL]
    out = {
        "sampled_labels": labels,
        "free_parameters_analysis_9": free,
        "held_at_analysis_8_values": [l for l in labels if l not in free],
        "n_fixed_parameter_values": len(fpv),
        "fixed_parameter_values": {k: float(v) for k, v in fpv.items()},
        "mu_chi_fiducial_GAL_branch": _fiducial_mu_chi()[0],
        "H0_prior_bounds_in_darksirens": [float(res[1][labels.index("H0")]),
                                          float(res[2][labels.index("H0")])],
        "note": ("the scanned closure is the PURE likelihood, so the flat prior "
                 "of record is the REGISTERED GRID RANGE, not these bounds"),
    }
    missing = [l for l in free if l not in labels]
    if missing:
        raise SystemExit(f"[fatal] not sampled: {missing}; labels = {labels}")
    if len(labels) != 9:
        raise SystemExit(f"[fatal] expected Analysis 8's 9 labels, got {labels}")
    print("  [OK ] parameter space: free = " + ", ".join(free))
    print(f"  [OK ] held at Analysis-8 values: {out['held_at_analysis_8_values']}")
    print(f"  [OK ] {len(fpv)} pinned values (12 population + Om0)")
    return out


# =========================================================================== #
# Stage: provenance (CPU)
# =========================================================================== #
def stage_provenance(args):
    env = _cpu_setup("provenance", import_darksirens=True)
    env["grid"] = {
        "H0": h0_axis_record(_axis_from_args(args)),
        "f_agn": {"values": F_GRID.tolist(), "n": int(F_GRID.size),
                  "source": "gate_c_three_arms.F_GRID_2D (imported)"},
        "mu_chi_c2": {"values": MU_GRID.tolist(), "n": int(MU_GRID.size),
                      "source": "gate_c_three_arms.MU_GRID_2D (imported)"},
        "n_cells": int(_axis_from_args(args).size * F_GRID.size * MU_GRID.size),
    }
    env["parameter_space"] = check_parameter_space()
    env["truth"] = dict(TRUTH)
    env["tolerances"] = dict(TOLERANCES)
    env["reuse"] = {
        "likelihood": "analysis_8/scripts/a8_likelihood.py (imported: build, "
                      "provenance, set_env, guard spy, KDE window, opts, pins)",
        "grid_and_truths": "analysis_8/scripts/gate_c_three_arms.py (imported: "
                           "F_GRID_2D, MU_GRID_2D, TRUTH, _cell_record, "
                           "_trapz_weights, _marginals, _truth_flags)",
        "posterior_convention": "analysis_2/scripts/scan_h0f.py marginal_ci "
                                "(imported via a8_likelihood.import_scan_h0f)",
    }
    print(f"\n  grid: {env['grid']['H0']['n']} H0 x {F_GRID.size} f x "
          f"{MU_GRID.size} mu = {env['grid']['n_cells']} cells")
    _write(DIAG / "provenance.json", env)
    return env


# =========================================================================== #
# Stage: timing (GPU, rita)
# =========================================================================== #
def stage_timing(args):
    env = _gpu_setup("timing")
    t0 = time.time()
    cell, surveys = build_cell()
    build_s = time.time() - t0
    axis = _axis_from_args(args)
    L = A8.MU_CHI_C2_LABEL

    # Probes: the anchor, the Analysis-8 MAP, the cube corners, the guard corner.
    probes = [
        (H0_ANCHOR, 0.295, 0.0, "anchor, null mark"),
        (H0_ANCHOR, 0.275, 0.1075, "anchor, Analysis-8 MAP"),
        (float(axis.min()), 0.275, 0.1075, "low-H0 edge"),
        (float(axis.max()), 0.275, 0.1075, "high-H0 edge"),
        (float(axis.min()), 1.0, 0.25, "low-H0 guard corner"),
        (float(axis.max()), 1.0, 0.25, "high-H0 guard corner"),
        (H0_ANCHOR, 0.0, -0.20, "anchor, f = 0 corner"),
    ]
    rows = []
    for h0, f, mu, note in probes:
        r = cell.evaluate(H0=h0, fcat_2=f, **{L: mu})
        rec = {"H0": h0, "f_agn": f, "mu_chi_c2": mu, "dmu_chi": mu, "note": note,
               **GC._cell_record(r)}
        rows.append(rec)
        print(f"  H0={h0:<7.2f} f={f:<5} mu={mu:+.4f}  logL={r['logL']!r}  "
              f"Neff={rec['Neff']}  {r['seconds']:.3f}s   {note}")
        sys.stdout.flush()

    # H0 LIVENESS.  Analysis 8's Gate A caught a feature that was silently
    # ignored; the same trap applies to a coordinate that was already in the
    # label list but never varied.
    live = []
    ref = cell.evaluate(H0=H0_ANCHOR, fcat_2=0.275, **{L: 0.1075})
    for h0 in (H0_ANCHOR + H0_STEP, H0_ANCHOR - H0_STEP,
               float(axis.min()), float(axis.max())):
        r = cell.evaluate(H0=h0, fcat_2=0.275, **{L: 0.1075})
        live.append({"H0": h0, "logL": r["logL"], "seconds": r["seconds"],
                     "delta_logL_vs_anchor": r["logL"] - ref["logL"]})
        print(f"  liveness H0={h0:<7.2f} dlogL={live[-1]['delta_logL_vs_anchor']:+.6f}")
    min_live = min(abs(x["delta_logL_vs_anchor"]) for x in live)

    steady = [r["seconds"] for r in rows[1:]] + [x["seconds"] for x in live]
    med = float(np.median(steady))
    n_reg = int(h0_axis("registered").size * F_GRID.size * MU_GRID.size)
    n_fal = int(h0_axis("fallback").size * F_GRID.size * MU_GRID.size)
    out = {
        "stage": "timing",
        "environment": env,
        "build_seconds": build_s,
        "first_eval_seconds": rows[0]["seconds"],
        "steady_state_median_seconds": med,
        "steady_state_min_seconds": float(np.min(steady)),
        "steady_state_max_seconds": float(np.max(steady)),
        "probes": rows,
        "h0_liveness": {
            "reference": {"H0": H0_ANCHOR, "f_agn": 0.275, "mu_chi_c2": 0.1075,
                          "logL": ref["logL"]},
            "cells": live,
            "min_abs_delta_logL": min_live,
            "criterion": ("H0 must move logL by far more than the anchor "
                          f"tolerance {TOLERANCES['anchor_equivalence_abs_logL']}"),
            "passes": bool(min_live > 1e-3),
        },
        "projection": {
            "registered_cells": n_reg,
            "registered_gpu_hours": n_reg * med / 3600.0,
            "fallback_cells": n_fal,
            "fallback_gpu_hours": n_fal * med / 3600.0,
            "anchor_slab_cells": int(F_GRID.size * MU_GRID.size),
            "anchor_slab_gpu_hours": F_GRID.size * MU_GRID.size * med / 3600.0,
            "fallback_rule": ("pre-registered: if the registered cube projects "
                              "above 80 GPU-h, scan the fallback H0 axis "
                              "[62.74, 74.74] (25 nodes) and record the trim"),
            "fallback_rule_fires": bool(n_reg * med / 3600.0 > 80.0),
        },
    }
    _write(DIAG / "_a9_stage_timing.json", out)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("probes", "environment")},
                     indent=2, default=GC._json_default))
    return out


# =========================================================================== #
# Stages: anchor and scan (GPU, rita)
# =========================================================================== #
def _scan(args, h0_list, tag):
    env = _gpu_setup(args.stage)
    axis = _axis_from_args(args)
    path = _tag_path(tag)

    done_elsewhere, headers, dropped = load_done(exclude=path)
    hdr_self, rows_self, drop_self = _read_jsonl(path)
    done_self = {_key(r["H0"], r["f_agn"]): r for r in rows_self}
    print(f"\ncheckpoint {path.name}: {len(done_self)} rows already here, "
          f"{len(done_elsewhere)} in sibling checkpoints, {dropped + drop_self} "
          f"truncated line(s) dropped")

    todo = [(h0, f) for h0 in h0_list for f in F_GRID
            if _key(h0, f) not in done_self and _key(h0, f) not in done_elsewhere]
    print(f"rows to do in this worker: {len(todo)} "
          f"({len(todo) * MU_GRID.size} cells)")
    if not todo:
        print("nothing to do")
        return

    t_build = time.time()
    cell, surveys = build_cell()
    print(f"build: {time.time() - t_build:.1f}s")
    if hdr_self is None:
        _append_jsonl(path, _header(cell, surveys, axis, tag, extra={
            "chunk": args.chunk, "n_chunks": args.n_chunks,
            "h0_slabs_this_worker": [float(x) for x in h0_list],
            "h0_axis_name": args.h0_axis,
            "jax_devices": str(__import__("jax").devices()),
        }))

    t0 = time.time()
    for n, (h0, f) in enumerate(todo, start=1):
        cells = eval_row(cell, h0, f, MU_GRID)
        _append_jsonl(path, {
            "record": "row", "H0": float(h0), "f_agn": float(f),
            "key": _key(h0, f), "cells": cells,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        lls = np.array([c["logL"] for c in cells], dtype=float)
        fin = np.isfinite(lls)
        el = time.time() - t0
        eta = el / n * (len(todo) - n)
        print(f"[{tag}] row {n}/{len(todo)} H0={h0:.2f} f={f:.4f} "
              f"maxlogL={(lls[fin].max() if fin.any() else float('nan')):.4f} "
              f"rejected={int((~fin).sum())}/{MU_GRID.size} "
              f"elapsed={el/60:.1f}min eta={eta/60:.1f}min")
        sys.stdout.flush()
        if args.stop_after_s and el > args.stop_after_s:
            print(f"[{tag}] stopping cleanly at the requested budget "
                  f"({args.stop_after_s}s); {len(todo) - n} rows left")
            break
    print(f"[{tag}] done -> {path}")


def stage_anchor(args):
    axis = _axis_from_args(args)
    anchor = float(axis[int(np.argmin(np.abs(axis - H0_ANCHOR)))])
    if anchor != H0_ANCHOR:
        raise RuntimeError(f"[fatal] anchor node {anchor} != {H0_ANCHOR}")
    return _scan(args, [anchor], "anchor")


def stage_scan(args):
    axis = _axis_from_args(args)
    if not (0 <= args.chunk < args.n_chunks):
        raise SystemExit(f"[fatal] --chunk must be in [0, {args.n_chunks})")
    # Interleaved, not contiguous: a partially finished cube still spans the
    # whole H0 range instead of stopping halfway up it.  The ANCHOR slab is
    # dealt first, so it belongs to chunk 0 and is computed in the first wave of
    # workers: it is Gate A, and Gate A is closed WHILE the rest of the cube
    # runs, not after it.
    order = [float(H0_ANCHOR)] + [float(x) for x in axis if x != H0_ANCHOR]
    mine = [x for i, x in enumerate(order) if i % args.n_chunks == args.chunk]
    mine.sort(key=lambda x: (x != H0_ANCHOR, x))
    print(f"chunk {args.chunk}/{args.n_chunks}: H0 slabs {mine}")
    return _scan(args, mine, f"{args.h0_axis}c{args.chunk}of{args.n_chunks}")


# =========================================================================== #
# Stage: compare_a8 (CPU) -- the anchor slab against arm J, cell by cell
# =========================================================================== #
def _slab_arrays(done, h0):
    """(logL, finite, puller) on the (f, mu) plane at one H0, from checkpoints."""
    shape = (F_GRID.size, MU_GRID.size)
    ll = np.full(shape, np.nan)
    fin = np.zeros(shape, dtype=bool)
    store = {}
    missing = []
    for i, f in enumerate(F_GRID):
        hit = done.get(_key(h0, f))
        if hit is None:
            missing.append(float(f))
            continue
        cells = hit[1]["cells"]
        for j, c in enumerate(cells):
            ll[i, j] = c["logL"]
            fin[i, j] = bool(c["finite"])
        store[i] = cells

    def pull(key, default=np.nan):
        out = np.full(shape, default, dtype=float)
        for i, cells in store.items():
            for j, c in enumerate(cells):
                v = c.get(key)
                out[i, j] = default if v is None else v
        return out

    return ll, fin, pull, missing


def _anchor_equivalence(done):
    import h5py
    with h5py.File(A8_ARM_J_H5, "r") as h5:
        f8 = h5["f_grid"][:]
        mu8 = h5["mu_chi_c2_grid"][:]
        ll8 = h5["log_likelihood"][:]
        rej8 = h5["guard/rejected"][:].astype(bool)
    if not (np.array_equal(f8, F_GRID) and np.array_equal(mu8, MU_GRID)):
        raise RuntimeError("[fatal] the anchor plane is not Analysis 8's grid")

    ll9, fin9, pull9, missing = _slab_arrays(done, H0_ANCHOR)
    out = {
        "check": "anchor slab (H0 = 67.74) vs analysis_8 results/arm_J_joint.h5",
        "a8_file": str(A8_ARM_J_H5),
        "n_cells": int(ll8.size),
        "n_rows_missing": len(missing),
        "rows_missing": missing,
        "tolerances": dict(TOLERANCES),
    }
    if missing:
        out["verdict"] = "PENDING -- the anchor slab is incomplete"
        return out

    rej9 = ~fin9
    same_mask = bool(np.array_equal(rej9, rej8))
    both = (~rej9) & (~rej8)
    d = np.abs(ll9 - ll8)[both]
    a8_vals = np.abs(ll8)[both]
    ulp = np.spacing(a8_vals)
    out.update({
        "rejected_a8": int(rej8.sum()),
        "rejected_a9": int(rej9.sum()),
        "rejected_masks_identical": same_mask,
        "n_compared": int(both.sum()),
        "max_abs_delta_logL": float(d.max()),
        "max_rel_delta_logL": float((d / np.maximum(a8_vals, 1e-300)).max()),
        "max_delta_in_ulp": float((d / ulp).max()),
        "n_exactly_equal": int((d == 0.0).sum()),
        "one_ulp_at_logL_scale": float(np.spacing(float(np.abs(ll8[both]).max()))),
        "argmax_cell": None,
    })
    if d.size:
        k = int(np.argmax(d))
        idx = np.argwhere(both)[k]
        out["argmax_cell"] = {"i_f": int(idx[0]), "j_mu": int(idx[1]),
                              "f_agn": float(F_GRID[idx[0]]),
                              "mu_chi_c2": float(MU_GRID[idx[1]]),
                              "logL_a8": float(ll8[idx[0], idx[1]]),
                              "logL_a9": float(ll9[idx[0], idx[1]])}

    # the same posterior convention Analysis 8 used, imported from it
    scan_h0f = A8.import_scan_h0f()
    marg9 = GC._marginals(F_GRID, MU_GRID, ll9, fin9, scan_h0f.marginal_ci)
    a8_json = json.loads(Path(A8_ARM_J_JSON).read_text())
    cmp = {}
    for key, block in (("f", "f"), ("mu_chi_c2", "mu_chi_c2")):
        a9b, a8b = marg9[block], a8_json[block]
        cmp[key] = {
            "a9_median": a9b["median"], "a8_median": a8b["median"],
            "delta_median": a9b["median"] - a8b["median"],
            "a9_ci68": a9b["ci68"], "a8_ci68": a8b["ci68"],
            "delta_ci68": [a9b["ci68"][0] - a8b["ci68"][0],
                           a9b["ci68"][1] - a8b["ci68"][1]],
            "a9_ci90": a9b["ci90"], "a8_ci90": a8b["ci90"],
        }
    out["marginals"] = cmp
    out["a8_logL_max"] = float(a8_json["logL_max"])
    out["a9_logL_max"] = float(marg9["logL_max"])
    out["delta_logL_max"] = out["a9_logL_max"] - out["a8_logL_max"]

    worst_marg = max(abs(cmp[k]["delta_median"]) for k in cmp)
    out["max_abs_marginal_delta"] = worst_marg
    out["passes_abs"] = bool(out["max_abs_delta_logL"]
                             <= TOLERANCES["anchor_equivalence_abs_logL"])
    out["passes_rel"] = bool(out["max_rel_delta_logL"]
                             <= TOLERANCES["anchor_equivalence_rel_logL"])
    out["passes_marginals"] = bool(worst_marg <= TOLERANCES["anchor_marginal_abs"])
    out["verdict"] = ("PASS" if (same_mask and out["passes_abs"] and out["passes_rel"]
                                 and out["passes_marginals"]) else "FAIL")
    return out


def stage_compare_a8(args):
    env = _cpu_setup("compare_a8", import_darksirens=False)
    done, headers, dropped = load_done()
    out = _anchor_equivalence(done)
    out["environment"] = env
    out["checkpoint_headers"] = {k: {"tag": v.get("tag"), "host": v.get("host"),
                                     "darksirens_sha": v["provenance"]["darksirens_sha"]}
                                 for k, v in headers.items()}
    timing = DIAG / "_a9_stage_timing.json"
    if timing.exists():
        out["h0_liveness"] = json.loads(timing.read_text())["h0_liveness"]
    _write(DIAG / "anchor_equivalence.json", out)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("environment", "rows_missing")},
                     indent=2, default=GC._json_default))
    return out


# =========================================================================== #
# Stage: status (CPU)
# =========================================================================== #
def stage_status(args):
    axis = _axis_from_args(args)
    done, headers, dropped = load_done()
    want = {_key(h0, f) for h0 in axis for f in F_GRID}
    have = want & set(done)
    per_h0 = {f"{h0:.2f}": sum(1 for f in F_GRID if _key(h0, f) in done)
              for h0 in axis}
    secs = [c["seconds"] for _, r in done.values() for c in r["cells"]]
    out = {
        "rows_total": len(want), "rows_done": len(have),
        "cells_total": len(want) * MU_GRID.size,
        "cells_done": len(have) * MU_GRID.size,
        "fraction": len(have) / max(len(want), 1),
        "rows_done_off_axis": len(set(done) - want),
        "truncated_lines_dropped": dropped,
        "rows_per_H0": per_h0,
        "checkpoints": [p.name for p in _all_checkpoints()],
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
def _cube(done, axis):
    shape = (axis.size, F_GRID.size, MU_GRID.size)
    ll = np.full(shape, np.nan)
    fin = np.zeros(shape, dtype=bool)
    store = {}
    missing, dup = [], []
    for a, h0 in enumerate(axis):
        for i, f in enumerate(F_GRID):
            hit = done.get(_key(h0, f))
            if hit is None:
                missing.append({"H0": float(h0), "f_agn": float(f)})
                continue
            cells = hit[1]["cells"]
            store[(a, i)] = cells
            for j, c in enumerate(cells):
                ll[a, i, j] = c["logL"]
                fin[a, i, j] = bool(c["finite"])

    def pull(key, default=np.nan):
        out = np.full(shape, default, dtype=float)
        for (a, i), cells in store.items():
            for j, c in enumerate(cells):
                v = c.get(key)
                out[a, i, j] = default if v is None else v
        return out

    return ll, fin, pull, missing, dup


def _duplicate_report(axis):
    """Rows computed by more than one worker must agree bit for bit."""
    seen, dups = {}, []
    for path in _all_checkpoints():
        _hdr, rows, _d = _read_jsonl(path)
        for r in rows:
            k = _key(r["H0"], r["f_agn"])
            if k in seen:
                a = np.array([c["logL"] for c in seen[k][1]["cells"]], dtype=float)
                b = np.array([c["logL"] for c in r["cells"]], dtype=float)
                both = np.isfinite(a) & np.isfinite(b)
                dups.append({
                    "key": k, "files": [seen[k][0], path.name],
                    "max_abs_delta_logL": float(np.abs(a[both] - b[both]).max())
                    if both.any() else None,
                    "same_rejected_mask": bool(np.array_equal(
                        np.isfinite(a), np.isfinite(b))),
                })
            else:
                seen[k] = (path.name, r)
    return dups


def _marginals_3d(axis, ll, fin, marginal_ci):
    llm = np.where(fin, ll, -np.inf)
    if not np.isfinite(llm).any():
        raise RuntimeError("[fatal] every cell is rejected")
    mx = float(llm[np.isfinite(llm)].max())
    P = np.where(np.isfinite(llm), np.exp(llm - mx), 0.0)

    m_h0 = np.trapz(np.trapz(P, MU_GRID, axis=2), F_GRID, axis=1)
    m_f = np.trapz(np.trapz(P, MU_GRID, axis=2), axis, axis=0)
    m_mu = np.trapz(np.trapz(P, F_GRID, axis=1), axis, axis=0)

    out = {"logL_max": mx}
    with np.errstate(divide="ignore"):
        out["H0"] = marginal_ci(axis, np.log(m_h0))
        out["H0"]["marginal_logp"] = np.log(m_h0).tolist()
        out["f"] = marginal_ci(F_GRID, np.log(m_f))
        out["f"]["marginal_logp"] = np.log(m_f).tolist()
        out["mu_chi_c2"] = marginal_ci(MU_GRID, np.log(m_mu))
        out["mu_chi_c2"]["marginal_logp"] = np.log(m_mu).tolist()

    W = (GC._trapz_weights(axis)[:, None, None]
         * GC._trapz_weights(F_GRID)[None, :, None]
         * GC._trapz_weights(MU_GRID)[None, None, :]) * P
    Z = W.sum()
    grids = [axis, F_GRID, MU_GRID]
    names = ["H0", "f_agn", "mu_chi_c2"]
    G = np.meshgrid(*grids, indexing="ij")
    mean = [float((W * g).sum() / Z) for g in G]
    cov = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            cov[a, b] = float((W * (G[a] - mean[a]) * (G[b] - mean[b])).sum() / Z)
    sd = np.sqrt(np.diag(cov))
    out["posterior_moments"] = {
        "names": names,
        "mean": {n: m for n, m in zip(names, mean)},
        "sd": {n: float(s) for n, s in zip(names, sd)},
        "cov": cov.tolist(),
        "correlation": {f"{names[a]}|{names[b]}": float(cov[a, b] / (sd[a] * sd[b]))
                        for a in range(3) for b in range(a + 1, 3)},
    }
    a, i, j = np.unravel_index(np.argmax(llm), llm.shape)
    out["map"] = {"H0": float(axis[a]), "f_agn": float(F_GRID[i]),
                  "mu_chi_c2": float(MU_GRID[j]), "dmu_chi": float(MU_GRID[j]),
                  "logL": float(llm[a, i, j]), "index": [int(a), int(i), int(j)]}

    # P(dmu <= 0): 0.0 is NOT a grid node (neighbours -0.0050, +0.0025)
    p_mu = m_mu / np.trapz(m_mu, MU_GRID)
    neg = MU_GRID <= 0.0
    xs = np.concatenate([MU_GRID[neg], [0.0]])
    ys = np.concatenate([p_mu[neg], [float(np.interp(0.0, MU_GRID, p_mu))]])
    out["P_dmu_chi_le_0"] = float(np.trapz(ys, xs))
    out["P_dmu_chi_le_0_note"] = (
        "a posterior probability under THIS model and THIS grid, with 0.0 "
        "interpolated between the neighbouring nodes; it is NOT a sigma claim")

    for key, x in (("H0", axis), ("f", F_GRID), ("mu_chi_c2", MU_GRID)):
        out[key]["marginal_mode"] = _marginal_mode(x, out[key]["marginal_logp"])

    out["marginals_2d"] = {
        "H0_f": np.trapz(P, MU_GRID, axis=2),
        "H0_mu": np.trapz(P, F_GRID, axis=1),
        "f_mu": np.trapz(P, axis, axis=0),
    }
    out["_P"] = P
    return out


def _edge_mass(marg, axis):
    """Gate B3: the cube must not clip posterior mass at any edge."""
    out = {}
    for name, x, key in (("H0", axis, "H0"), ("f_agn", F_GRID, "f"),
                         ("mu_chi_c2", MU_GRID, "mu_chi_c2")):
        lp = np.asarray(marg[key]["marginal_logp"], dtype=float)
        p = np.exp(lp - np.nanmax(lp[np.isfinite(lp)]))
        out[name] = {"low_edge_density_over_peak": float(p[0]),
                     "high_edge_density_over_peak": float(p[-1]),
                     "criterion": "both <= 1e-6 of the peak",
                     "passes": bool(p[0] <= 1e-6 and p[-1] <= 1e-6)}
    return out


def _guard_report_cube(axis, ll, fin, pull):
    Neff = pull("Neff")
    thr = pull("threshold")
    rejected = ~fin
    rep = {
        "n_cells": int(ll.size),
        "n_rejected": int(rejected.sum()),
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
    for a, i, j in idx[:5000]:
        rep["rejected_cells"].append({
            "H0": float(axis[a]), "f_agn": float(F_GRID[i]),
            "mu_chi_c2": float(MU_GRID[j]), "dmu_chi": float(MU_GRID[j]),
            "Neff": float(Neff[a, i, j]), "threshold": float(thr[a, i, j]),
            "Neff_over_threshold": float(Neff[a, i, j] / thr[a, i, j])})
    if idx.size:
        rep["rejected_H0_range"] = [float(axis[idx[:, 0].min()]),
                                    float(axis[idx[:, 0].max()])]
        rep["rejected_f_min"] = float(F_GRID[idx[:, 1].min()])
        rep["rejected_mu_min"] = float(MU_GRID[idx[:, 2].min()])

    llm = np.where(fin, ll, -np.inf)
    mx = float(llm[np.isfinite(llm)].max())
    P = np.where(np.isfinite(llm), np.exp(llm - mx), 0.0)
    W = (GC._trapz_weights(axis)[:, None, None]
         * GC._trapz_weights(F_GRID)[None, :, None]
         * GC._trapz_weights(MU_GRID)[None, None, :])
    if rep["n_rejected"] == 0:
        rep["upper_bound_rejected_mass_fraction"] = 0.0
        return rep
    nb, local = [], []
    for a, i, j in idx:
        here = []
        for da in (-1, 0, 1):
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    x, y, z = a + da, i + di, j + dj
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
        "check": ("the single fill value is the LARGEST accepted density adjacent "
                  "to ANY rejected cell, so it is >= the largest accepted density "
                  "adjacent to EACH of them, cell by cell"),
        "verified": bool(all(P_bound >= v for v in local)),
    }
    # logL must be FALLING into the guard corner for the boundary fill to be an
    # over-estimate; checked per rejected (H0, f) column along mu, as Analysis 8
    # checked it at fixed H0.
    mono = []
    for a, i in sorted({(int(x[0]), int(x[1])) for x in idx}):
        col = llm[a, i]
        acc = np.where(np.isfinite(col))[0]
        acc = acc[acc >= int(np.argmax(np.where(np.isfinite(col), col, -np.inf)))]
        tail = acc[-4:] if acc.size >= 4 else acc
        d = np.diff(col[tail]) if tail.size > 1 else np.array([np.nan])
        mono.append({"H0": float(axis[a]), "f_agn": float(F_GRID[i]),
                     "last_accepted_mu": float(MU_GRID[acc[-1]]) if acc.size else None,
                     "last_diffs_logL": d.tolist(),
                     "monotonically_falling": bool(np.all(d < 0))})
    rep["falling_towards_the_guard"] = mono
    rep["all_columns_falling"] = bool(all(m["monotonically_falling"] for m in mono))
    rep["passes"] = bool(rep["upper_bound_rejected_mass_fraction"] < 1e-6)
    return rep


def _comparisons(marg, axis):
    """Analysis 8 (H0 fixed) and Analysis 2 (unmarked, H0 free), read from file."""
    out = {}
    a8 = json.loads(Path(A8_ARM_J_JSON).read_text())
    w = lambda b, lev: float(b[lev][1] - b[lev][0])
    out["analysis_8_arm_J_fixed_H0"] = {
        "file": str(A8_ARM_J_JSON),
        "f_median": a8["f"]["median"], "f_ci68": a8["f"]["ci68"],
        "f_ci90": a8["f"]["ci90"],
        "mu_median": a8["mu_chi_c2"]["median"], "mu_ci68": a8["mu_chi_c2"]["ci68"],
        "mu_ci90": a8["mu_chi_c2"]["ci90"],
        "correlation_f_mu": a8["posterior_moments"]["correlation"],
        "cost_of_freeing_H0": {
            "f_ci68_width_a8": w(a8["f"], "ci68"),
            "f_ci68_width_a9": w(marg["f"], "ci68"),
            "f_ci68_width_ratio": w(marg["f"], "ci68") / w(a8["f"], "ci68"),
            "f_ci90_width_ratio": w(marg["f"], "ci90") / w(a8["f"], "ci90"),
            "f_median_shift": marg["f"]["median"] - a8["f"]["median"],
            "mu_ci68_width_a8": w(a8["mu_chi_c2"], "ci68"),
            "mu_ci68_width_a9": w(marg["mu_chi_c2"], "ci68"),
            "mu_ci68_width_ratio": (w(marg["mu_chi_c2"], "ci68")
                                    / w(a8["mu_chi_c2"], "ci68")),
            "mu_ci90_width_ratio": (w(marg["mu_chi_c2"], "ci90")
                                    / w(a8["mu_chi_c2"], "ci90")),
            "mu_median_shift": (marg["mu_chi_c2"]["median"]
                                - a8["mu_chi_c2"]["median"]),
        },
    }
    a2 = _analysis_2_seed100()
    if a2:
        a2["h0_median_shift_j9_minus_a2"] = marg["H0"]["median"] - a2["H0"]["median"]
        a2["h0_width68_ratio_j9_over_a2"] = (w(marg["H0"], "ci68")
                                             / a2["H0"]["width68"])
        a2["h0_width90_ratio_j9_over_a2"] = (w(marg["H0"], "ci90")
                                             / a2["H0"]["width90"])
        a2["f_median_shift_j9_minus_a2"] = marg["f"]["median"] - a2["f"]["median"]
        out["analysis_2_unmarked_H0_free"] = a2
    return out


def stage_assemble(args):
    import h5py
    env = _cpu_setup("assemble", import_darksirens=False)
    axis = _axis_from_args(args)
    done, headers, dropped = load_done()

    shas = {h["provenance"]["darksirens_sha"] for h in headers.values()}
    if len(shas) > 1:
        raise RuntimeError(f"[fatal] checkpoints disagree on darksirens SHA: {shas}")
    if shas and shas != {DARKSIRENS_A8_SHA}:
        raise RuntimeError(f"[fatal] checkpoints carry SHA {shas}, not "
                           f"{DARKSIRENS_A8_SHA}")

    ll, fin, pull, missing, _dup = _cube(done, axis)
    if missing and not args.allow_partial:
        raise RuntimeError(
            f"[fatal] {len(missing)} of {axis.size * F_GRID.size} rows are "
            f"missing; run the remaining chunks or pass --allow_partial to write "
            f"a coverage report only.  First missing: {missing[:5]}")
    if missing:
        _write(DIAG / "coverage.json",
               {"rows_missing": missing, "rows_total": int(axis.size * F_GRID.size),
                "checkpoints": [p.name for p in _all_checkpoints()]})
        print(f"[partial] {len(missing)} rows missing; wrote the coverage report "
              f"and stopped before the results.")
        return

    scan_h0f = A8.import_scan_h0f()
    marg = _marginals_3d(axis, ll, fin, scan_h0f.marginal_ci)
    P = marg.pop("_P")
    m2 = marg.pop("marginals_2d")

    GC._truth_flags(marg["H0"], {"planted": TRUTH["H0_planted"]})
    GC._truth_flags(marg["f"], {"planted": TRUTH["f_agn_planted"],
                                "realised": TRUTH["f_agn_realised"]})
    GC._truth_flags(marg["mu_chi_c2"], {"planted": TRUTH["dmu_chi_planted"],
                                        "realised": TRUTH["dmu_chi_realised"]})
    guard = _guard_report_cube(axis, ll, fin, pull)
    anchor = _anchor_equivalence(done)
    dups = _duplicate_report(axis)
    edges = _edge_mass(marg, axis)

    hdr = next(iter(headers.values())) if headers else {}
    summary = {
        "analysis": "analysis_9_marked_multitracer_H0_fagn",
        "question": ("Analysis 8 with H0 freed: (H0, f_AGN, dmu_chi) from one "
                     "marked seed-100 realisation, everything else pinned."),
        "file": f"results/{args.out_stem}.h5",
        "arm": "J9",
        "environment": env,
        "provenance": hdr.get("provenance"),
        "free_parameters": ["H0", "fcat_2", A8.MU_CHI_C2_LABEL],
        "sampled_labels": hdr.get("sampled_labels"),
        "per_catalog_pop_params": hdr.get("per_catalog_pop_params"),
        "base_coord": hdr.get("base_coord"),
        "fixed_parameter_values": hdr.get("fixed_parameter_values"),
        "mu_chi_gal_fixed": hdr.get("mu_chi_fiducial_GAL_branch"),
        "Om0_fixed": A8.OM0_FID,
        "events_file": A8.GW_PATH_MARKED,
        "selection_file": A8.GWSEL_PATH,
        "survey_paths": hdr.get("survey_paths"),
        "truth": dict(TRUTH),
        "grid": {"H0": h0_axis_record(axis),
                 "f_agn": [float(F_GRID[0]), float(F_GRID[-1]), int(F_GRID.size)],
                 "mu_chi_c2": [float(MU_GRID[0]), float(MU_GRID[-1]),
                               int(MU_GRID.size)]},
        "n_cells": int(ll.size), "n_rejected": int((~fin).sum()),
        "logL_max": marg["logL_max"], "map": marg["map"],
        "H0": marg["H0"], "f": marg["f"], "mu_chi_c2": marg["mu_chi_c2"],
        "dmu_chi": dict(marg["mu_chi_c2"],
                        note="identical to mu_chi_c2 only because mu_chi_GAL is "
                             "pinned at the powerlaw+peak fiducial 0.0"),
        "posterior_moments": marg["posterior_moments"],
        "P_dmu_chi_le_0": marg["P_dmu_chi_le_0"],
        "P_dmu_chi_le_0_note": marg["P_dmu_chi_le_0_note"],
        "edge_mass": edges,
        "guard": {k: v for k, v in guard.items() if k != "rejected_cells"},
        "anchor_equivalence": {k: v for k, v in anchor.items()
                               if k not in ("environment", "rows_missing")},
        "duplicate_rows": dups,
        "comparisons": _comparisons(marg, axis),
        "timing": {
            "cells": int(ll.size),
            "gpu_hours": float(np.nansum(pull("seconds")) / 3600.0),
            "median_seconds_per_cell": float(np.nanmedian(pull("seconds"))),
        },
        "checkpoints": [p.name for p in _all_checkpoints()],
        "truncated_lines_dropped": dropped,
    }

    h5_path = RESULTS / f"{args.out_stem}.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("H0_grid", data=axis)
        h5.create_dataset("f_grid", data=F_GRID)
        h5.create_dataset("mu_chi_c2_grid", data=MU_GRID)
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
        m.create_dataset("mu_chi_c2",
                         data=np.exp(np.asarray(marg["mu_chi_c2"]["marginal_logp"])))
        for k, v in m2.items():
            m.create_dataset(k, data=v)
        h5.create_dataset("posterior_unnormalised", data=P)
        h5.attrs["analysis"] = "analysis_9_marked_multitracer_H0_fagn"
        h5.attrs["free_parameters"] = json.dumps(["H0", "fcat_2",
                                                  A8.MU_CHI_C2_LABEL])
        h5.attrs["labels"] = json.dumps(hdr.get("sampled_labels"))
        h5.attrs["base_coord"] = json.dumps(hdr.get("base_coord"))
        h5.attrs["truth"] = json.dumps(TRUTH)
        h5.attrs["Om0_fixed"] = A8.OM0_FID
        h5.attrs["events_file"] = A8.GW_PATH_MARKED
        h5.attrs["selection_file"] = A8.GWSEL_PATH
        h5.attrs["survey_paths"] = json.dumps(hdr.get("survey_paths"))
        h5.attrs["darksirens_sha"] = DARKSIRENS_A8_SHA
        h5.attrs["gws_agn_sha"] = env.get("gws_agn_sha", "")
        h5.attrs["settings"] = json.dumps(A8.SETTINGS)
    print(f"wrote {h5_path}")
    _write(RESULTS / f"{args.out_stem}.json", summary)
    _merge_guard_file("J9", guard, extra={"grid": summary["grid"],
                                          "provenance": hdr.get("provenance")})
    _write(DIAG / "anchor_equivalence.json", anchor)

    print(json.dumps({k: summary[k] for k in
                      ("logL_max", "map", "P_dmu_chi_le_0", "edge_mass")},
                     indent=2, default=GC._json_default))
    for k in ("H0", "f", "mu_chi_c2"):
        b = summary[k]
        print(f"  {k:<11} median {b['median']:.6f}  68% [{b['ci68'][0]:.6f}, "
              f"{b['ci68'][1]:.6f}]  90% [{b['ci90'][0]:.6f}, {b['ci90'][1]:.6f}]")
    return summary


# =========================================================================== #
# Arm S9 -- spatial-only cosmology.  dmu_chi = 0 exactly, free (H0, f_AGN).
#
# The model is Analysis 8's ARM S: the SAME 'new'-shape likelihood with
# mu_chi_c2 held at 0.0, which makes the two branches' intrinsic populations
# identical and leaves f_AGN inferred from tracer structure alone.  Closure
# check 9.2 measured that this is the spatial-only (8-label, no mu_chi_c2)
# model to <= 2 ULP at 35 (H0, f) pairs, so no separate build is needed and
# S9 is a STRICT sub-model of J9: S9's cells are J9's dmu_chi = 0 plane.
# =========================================================================== #
S9_MU = 0.0


def _key_h0(h0):
    """S9 checkpoint key: one row is the 41 f_AGN cells at one H0."""
    return "{:.10g}".format(float(h0))


def _s9_tag_path(tag):
    return DIAG / f"_a9_s9_{tag}.jsonl"


def _s9_checkpoints():
    return sorted(DIAG.glob("_a9_s9_*.jsonl"))


def load_done_s9(exclude=None):
    done, headers, dropped = {}, {}, 0
    for path in _s9_checkpoints():
        if exclude is not None and path == Path(exclude):
            continue
        hdr, rows, drop = _read_jsonl(path)
        dropped += drop
        if hdr is not None:
            headers[path.name] = hdr
        for r in rows:
            done.setdefault(_key_h0(r["H0"]), (path.name, r))
    return done, headers, dropped


def eval_row_s9(cell, h0, f_grid):
    L = A8.MU_CHI_C2_LABEL
    return [GC._cell_record(cell.evaluate(H0=float(h0), fcat_2=float(f),
                                          **{L: S9_MU})) for f in f_grid]


def stage_s9(args):
    env = _gpu_setup("s9")
    axis = h0_axis("s9")
    if not (0 <= args.chunk < args.n_chunks):
        raise SystemExit(f"[fatal] --chunk must be in [0, {args.n_chunks})")
    mine = [float(x) for i, x in enumerate(axis) if i % args.n_chunks == args.chunk]
    tag = f"c{args.chunk}of{args.n_chunks}"
    path = _s9_tag_path(tag)

    done_elsewhere, headers, dropped = load_done_s9(exclude=path)
    hdr_self, rows_self, drop_self = _read_jsonl(path)
    done_self = {_key_h0(r["H0"]): r for r in rows_self}
    todo = [h0 for h0 in mine
            if _key_h0(h0) not in done_self and _key_h0(h0) not in done_elsewhere]
    print(f"\n[S9] chunk {args.chunk}/{args.n_chunks}: {len(mine)} H0 nodes, "
          f"{len(done_self)} already here, {len(done_elsewhere)} in siblings, "
          f"{len(todo)} to do ({len(todo) * F_GRID.size} cells), "
          f"{dropped + drop_self} truncated line(s) dropped")
    if not todo:
        print("nothing to do")
        return

    t_build = time.time()
    cell, surveys = build_cell()
    print(f"build: {time.time() - t_build:.1f}s")
    if hdr_self is None:
        _append_jsonl(path, _header(cell, surveys, axis, tag, extra={
            "arm": "S9",
            "arm_description": ("spatial-only cosmology: the Analysis-8 arm-S "
                                "construction (mu_chi_c2 == 0 exactly) with H0 "
                                "and f_AGN free"),
            "free_parameters_this_arm": ["H0", "fcat_2"],
            "mu_chi_c2_held_at": S9_MU,
            "chunk": args.chunk, "n_chunks": args.n_chunks,
            "h0_nodes_this_worker": mine,
            "h0_axis_name": "s9",
            "jax_devices": str(__import__("jax").devices()),
        }))

    t0 = time.time()
    for n, h0 in enumerate(todo, start=1):
        cells = eval_row_s9(cell, h0, F_GRID)
        _append_jsonl(path, {
            "record": "row", "H0": float(h0), "key": _key_h0(h0),
            "mu_chi_c2": S9_MU, "cells": cells,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        lls = np.array([c["logL"] for c in cells], dtype=float)
        fin = np.isfinite(lls)
        el = time.time() - t0
        print(f"[S9] row {n}/{len(todo)} H0={h0:.2f} "
              f"maxlogL={(lls[fin].max() if fin.any() else float('nan')):.4f} "
              f"rejected={int((~fin).sum())}/{F_GRID.size} "
              f"elapsed={el/60:.1f}min eta={el/n*(len(todo)-n)/60:.1f}min")
        sys.stdout.flush()
        if args.stop_after_s and el > args.stop_after_s:
            print(f"[S9] stopping cleanly at the requested budget; "
                  f"{len(todo) - n} rows left")
            break
    print(f"[S9] done -> {path}")


def stage_s9_status(args):
    axis = h0_axis("s9")
    done, headers, dropped = load_done_s9()
    want = {_key_h0(h0) for h0 in axis}
    have = want & set(done)
    secs = [c["seconds"] for _, r in done.values() for c in r["cells"]]
    out = {"arm": "S9", "rows_total": len(want), "rows_done": len(have),
           "cells_total": len(want) * F_GRID.size,
           "cells_done": len(have) * F_GRID.size,
           "fraction": len(have) / max(len(want), 1),
           "rows_done_off_axis": len(set(done) - want),
           "truncated_lines_dropped": dropped,
           "checkpoints": [p.name for p in _s9_checkpoints()],
           "median_seconds_per_cell": float(np.median(secs)) if secs else None,
           "gpu_hours_spent": float(np.sum(secs) / 3600.0) if secs else 0.0}
    if secs:
        out["gpu_hours_remaining"] = float(
            (len(want) - len(have)) * F_GRID.size * np.median(secs) / 3600.0)
    print(json.dumps(out, indent=2, default=GC._json_default))
    return out


def _s9_arrays(done, axis):
    shape = (axis.size, F_GRID.size)
    ll = np.full(shape, np.nan)
    fin = np.zeros(shape, dtype=bool)
    store, missing = {}, []
    for a, h0 in enumerate(axis):
        hit = done.get(_key_h0(h0))
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


def _marginal_mode(x, logp):
    lp = np.asarray(logp, dtype=float)
    if not np.isfinite(lp).any():
        return float("nan")
    return float(np.asarray(x, dtype=float)[int(np.nanargmax(np.where(
        np.isfinite(lp), lp, -np.inf)))])


def _marginals_h0_f(axis, ll, fin, marginal_ci):
    """The S9 posterior on (H0, f_AGN); marginal_ci is Analysis 2's, imported."""
    llm = np.where(fin, ll, -np.inf)
    if not np.isfinite(llm).any():
        raise RuntimeError("[fatal] every S9 cell is rejected")
    mx = float(llm[np.isfinite(llm)].max())
    P = np.where(np.isfinite(llm), np.exp(llm - mx), 0.0)
    m_h0 = np.trapz(P, F_GRID, axis=1)
    m_f = np.trapz(P, axis, axis=0)
    out = {"logL_max": mx}
    with np.errstate(divide="ignore"):
        lp_h0, lp_f = np.log(m_h0), np.log(m_f)
    out["H0"] = marginal_ci(axis, lp_h0)
    out["H0"]["marginal_logp"] = lp_h0.tolist()
    out["H0"]["marginal_mode"] = _marginal_mode(axis, lp_h0)
    out["f"] = marginal_ci(F_GRID, lp_f)
    out["f"]["marginal_logp"] = lp_f.tolist()
    out["f"]["marginal_mode"] = _marginal_mode(F_GRID, lp_f)

    W = np.outer(GC._trapz_weights(axis), GC._trapz_weights(F_GRID)) * P
    Z = W.sum()
    H2, F2 = np.meshgrid(axis, F_GRID, indexing="ij")
    Eh, Ef = float((W * H2).sum() / Z), float((W * F2).sum() / Z)
    Vh = float((W * (H2 - Eh) ** 2).sum() / Z)
    Vf = float((W * (F2 - Ef) ** 2).sum() / Z)
    C = float((W * (H2 - Eh) * (F2 - Ef)).sum() / Z)
    out["posterior_moments"] = {
        "mean": {"H0": Eh, "f_agn": Ef},
        "sd": {"H0": float(np.sqrt(Vh)), "f_agn": float(np.sqrt(Vf))},
        "cov_H0_f": C,
        "correlation": {"H0|f_agn": float(C / np.sqrt(Vh * Vf))},
    }
    a, i = np.unravel_index(np.argmax(llm), llm.shape)
    out["map"] = {"H0": float(axis[a]), "f_agn": float(F_GRID[i]),
                  "mu_chi_c2": S9_MU, "dmu_chi": S9_MU,
                  "logL": float(llm[a, i]), "index": [int(a), int(i)]}
    out["_P"] = P
    return out


def _guard_report_2d(axis, ll, fin, pull, arm):
    """Every rejected cell and a NON-CIRCULAR upper bound on the mass behind it."""
    Neff, thr = pull("Neff"), pull("threshold")
    rejected = ~fin
    rep = {
        "arm": arm,
        "axes": ["H0", "f_agn"],
        "n_cells": int(ll.size),
        "n_rejected": int(rejected.sum()),
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
    for a, i in idx:
        rep["rejected_cells"].append({
            "H0": float(axis[a]), "f_agn": float(F_GRID[i]),
            "mu_chi_c2": S9_MU, "dmu_chi": S9_MU,
            "Neff": float(Neff[a, i]), "threshold": float(thr[a, i]),
            "Neff_over_threshold": float(Neff[a, i] / thr[a, i])})
    llm = np.where(fin, ll, -np.inf)
    mx = float(llm[np.isfinite(llm)].max())
    P = np.where(np.isfinite(llm), np.exp(llm - mx), 0.0)
    W = np.outer(GC._trapz_weights(axis), GC._trapz_weights(F_GRID))
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
        "check": ("the single fill value is the LARGEST accepted density adjacent "
                  "to ANY rejected cell, so it is >= the largest accepted density "
                  "adjacent to EACH of them, cell by cell"),
        "verified": bool(all(P_bound >= v for v in local)),
    }
    rep["passes"] = bool(rep["upper_bound_rejected_mass_fraction"] < 1e-6)
    return rep


def _j9_window_check(axis, marg, window, criterion=1e-6):
    """Does the proposed J9 H0 window hold S9's OWN measured marginal?"""
    lp = np.asarray(marg["H0"]["marginal_logp"], dtype=float)
    p = np.exp(lp - np.nanmax(lp[np.isfinite(lp)]))          # peak-normalised
    dens = p / np.trapz(p, axis)                             # unit-area density
    lo, hi = float(window[0]), float(window[1])
    ilo = int(np.flatnonzero(axis == lo)[0]) if (axis == lo).any() else None
    ihi = int(np.flatnonzero(axis == hi)[0]) if (axis == hi).any() else None
    if ilo is None or ihi is None:
        raise RuntimeError(f"[fatal] the window edges {window} are not S9 nodes")
    inside = np.trapz(dens[ilo:ihi + 1], axis[ilo:ihi + 1])
    out = {
        "window": [lo, hi],
        "criterion": (f"marginal density at each edge <= {criterion:g} of the "
                      f"peak (GATES.md B3), measured on S9's OWN marked marginal"),
        "peak_H0": float(axis[int(np.argmax(p))]),
        "density_at_low_edge_over_peak": float(p[ilo]),
        "density_at_high_edge_over_peak": float(p[ihi]),
        "mass_inside": float(inside),
        "mass_outside": float(1.0 - inside),
        "passes": bool(p[ilo] <= criterion and p[ihi] <= criterion),
    }
    # If it fails, widen on the 0.5 lattice (J9 nodes must be S9 nodes).
    lattice = axis[np.isclose(np.round(axis * 2.0) / 2.0, axis, rtol=0, atol=1e-12)]
    ok = p[np.searchsorted(axis, lattice)] <= criterion
    below = lattice[(lattice <= lo) & ok] if (lattice <= lo).any() else lattice[:0]
    above = lattice[(lattice >= hi) & ok] if (lattice >= hi).any() else lattice[:0]
    need_lo = float(below.max()) if below.size else float(lattice.min())
    need_hi = float(above.min()) if above.size else float(lattice.max())
    n_nodes = int(np.sum((lattice >= need_lo) & (lattice <= need_hi))) + 1  # + 67.74
    out["required_window_on_the_0p5_lattice"] = [need_lo, need_hi]
    out["required_window_n_H0_nodes_including_67p74"] = n_nodes
    out["required_window_cells"] = int(n_nodes * F_GRID.size * MU_GRID.size)
    out["widening_needed"] = bool(need_lo < lo or need_hi > hi)
    return out


def _merge_guard_file(arm, block, extra=None):
    """diagnostics/a9_guard.json carries BOTH arms; never clobber the other."""
    path = DIAG / "a9_guard.json"
    doc = {}
    if path.exists():
        try:
            doc = json.loads(path.read_text())
        except json.JSONDecodeError:
            doc = {}
    doc.setdefault("analysis", "analysis_9_marked_multitracer_H0_fagn")
    doc.setdefault("truth", dict(TRUTH))
    doc.setdefault("policy", GUARD_POLICY)
    doc.setdefault("arms", {})
    doc["arms"][arm] = block
    if extra:
        doc["arms"][arm].update(extra)
    doc["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _write(path, doc)
    return doc


def stage_s9_assemble(args):
    import h5py
    env = _cpu_setup("s9_assemble", import_darksirens=False)
    axis = h0_axis("s9")
    done, headers, dropped = load_done_s9()
    shas = {h["provenance"]["darksirens_sha"] for h in headers.values()}
    if shas and shas != {DARKSIRENS_A8_SHA}:
        raise RuntimeError(f"[fatal] S9 checkpoints carry SHA {shas}")

    ll, fin, pull, missing = _s9_arrays(done, axis)
    if missing and not args.allow_partial:
        raise RuntimeError(f"[fatal] {len(missing)} of {axis.size} S9 rows missing; "
                           f"first: {missing[:5]}")
    if missing:
        _write(DIAG / "s9_coverage.json", {"rows_missing": missing,
                                           "rows_total": int(axis.size)})
        print(f"[partial] {len(missing)} S9 rows missing")
        return

    scan_h0f = A8.import_scan_h0f()
    marg = _marginals_h0_f(axis, ll, fin, scan_h0f.marginal_ci)
    P = marg.pop("_P")
    GC._truth_flags(marg["H0"], {"planted": TRUTH["H0_planted"]})
    GC._truth_flags(marg["f"], {"planted": TRUTH["f_agn_planted"],
                                "realised": TRUTH["f_agn_realised"]})
    guard = _guard_report_2d(axis, ll, fin, pull, "S9")
    window = _j9_window_check(axis, marg, args.j9_window)
    edges = {}
    for name, x, key in (("H0", axis, "H0"), ("f_agn", F_GRID, "f")):
        lp = np.asarray(marg[key]["marginal_logp"], dtype=float)
        pk = np.exp(lp - np.nanmax(lp[np.isfinite(lp)]))
        edges[name] = {"low_edge_density_over_peak": float(pk[0]),
                       "high_edge_density_over_peak": float(pk[-1]),
                       "criterion": "both <= 1e-6 of the peak",
                       "passes": bool(pk[0] <= 1e-6 and pk[-1] <= 1e-6)}

    # Analysis 2, same realisation, UNMARKED events, one shared population.
    a2 = _analysis_2_seed100()
    if a2:
        a2["h0_median_shift_s9_minus_a2"] = (marg["H0"]["median"]
                                             - a2["H0"]["median"])
        a2["h0_width68_ratio_s9_over_a2"] = (_w(marg["H0"], "ci68")
                                             / a2["H0"]["width68"])
        a2["h0_width90_ratio_s9_over_a2"] = (_w(marg["H0"], "ci90")
                                             / a2["H0"]["width90"])
        a2["f_median_shift_s9_minus_a2"] = marg["f"]["median"] - a2["f"]["median"]
        a2["f_width68_ratio_s9_over_a2"] = (_w(marg["f"], "ci68")
                                            / a2["f"]["width68"])

    hdr = next(iter(headers.values())) if headers else {}
    summary = {
        "analysis": "analysis_9_marked_multitracer_H0_fagn",
        "arm": "S9",
        "question": ("spatial-only cosmology: (H0, f_AGN) from the marked "
                     "seed-100 realisation with dmu_chi = 0 exactly"),
        "file": "results/s9_spatial.h5",
        "model_note": ("Analysis 8's arm-S construction -- the same 'new'-shape "
                       "likelihood with mu_chi_c2 = 0.0, which closure check 9.2 "
                       "measured equal to the 8-label spatial-only model to <= 2 "
                       "ULP.  S9 is therefore J9's dmu_chi = 0 plane."),
        "environment": env,
        "provenance": hdr.get("provenance"),
        "free_parameters": ["H0", "fcat_2"],
        "mu_chi_c2_held_at": S9_MU,
        "sampled_labels": hdr.get("sampled_labels"),
        "base_coord": hdr.get("base_coord"),
        "fixed_parameter_values": hdr.get("fixed_parameter_values"),
        "Om0_fixed": A8.OM0_FID,
        "events_file": A8.GW_PATH_MARKED,
        "selection_file": A8.GWSEL_PATH,
        "survey_paths": hdr.get("survey_paths"),
        "truth": dict(TRUTH),
        "grid": {"H0": h0_axis_record(axis),
                 "f_agn": [float(F_GRID[0]), float(F_GRID[-1]), int(F_GRID.size)],
                 "source": ("H0 = Analysis 2's own [50, 100] 201-node axis plus "
                            "the node 67.74; f = gate_c_three_arms.F_GRID_2D")},
        "n_cells": int(ll.size), "n_rejected": int((~fin).sum()),
        "logL_max": marg["logL_max"], "map": marg["map"],
        "H0": marg["H0"], "f": marg["f"],
        "posterior_moments": marg["posterior_moments"],
        "edge_mass": edges,
        "j9_window_check": window,
        "guard": {k: v for k, v in guard.items() if k != "rejected_cells"},
        "analysis_2_unmarked": a2,
        "timing": {"cells": int(ll.size),
                   "gpu_hours": float(np.nansum(pull("seconds")) / 3600.0),
                   "median_seconds_per_cell": float(np.nanmedian(pull("seconds")))},
        "checkpoints": [p.name for p in _s9_checkpoints()],
        "truncated_lines_dropped": dropped,
    }

    h5_path = RESULTS / "s9_spatial.h5"
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
        h5.attrs["analysis"] = "analysis_9_marked_multitracer_H0_fagn"
        h5.attrs["arm"] = "S9"
        h5.attrs["free_parameters"] = json.dumps(["H0", "fcat_2"])
        h5.attrs["mu_chi_c2_held_at"] = S9_MU
        h5.attrs["labels"] = json.dumps(hdr.get("sampled_labels"))
        h5.attrs["truth"] = json.dumps(TRUTH)
        h5.attrs["Om0_fixed"] = A8.OM0_FID
        h5.attrs["events_file"] = A8.GW_PATH_MARKED
        h5.attrs["selection_file"] = A8.GWSEL_PATH
        h5.attrs["survey_paths"] = json.dumps(hdr.get("survey_paths"))
        h5.attrs["darksirens_sha"] = DARKSIRENS_A8_SHA
        h5.attrs["gws_agn_sha"] = env.get("gws_agn_sha", "")
        h5.attrs["settings"] = json.dumps(A8.SETTINGS)
    print(f"wrote {h5_path}")
    _write(RESULTS / "s9_spatial.json", summary)
    _merge_guard_file("S9", guard, extra={"grid": summary["grid"]})
    print(json.dumps({"logL_max": summary["logL_max"], "map": summary["map"],
                      "edge_mass": edges, "j9_window_check": window,
                      "n_rejected": summary["n_rejected"]},
                     indent=2, default=GC._json_default))
    for k in ("H0", "f"):
        b = summary[k]
        print(f"  {k:<4} median {b['median']:.6f}  mode {b['marginal_mode']:.6f}  "
              f"68% [{b['ci68'][0]:.6f}, {b['ci68'][1]:.6f}] (w "
              f"{b['ci68'][1]-b['ci68'][0]:.6f})  90% [{b['ci90'][0]:.6f}, "
              f"{b['ci90'][1]:.6f}] (w {b['ci90'][1]-b['ci90'][0]:.6f})")
    return summary


# =========================================================================== #
# Gate A, in full: the whole H0 = 67.74 slab against Analysis 8's arm J
# =========================================================================== #
def _ulp_stats(a9, a8, mask):
    d = np.abs(a9 - a8)[mask]
    ref = np.abs(a8)[mask]
    ulp = np.spacing(np.where(ref > 0, ref, 1.0))
    out = {"n_compared": int(mask.sum()),
           "n_bitwise_identical": int((d == 0.0).sum()),
           "max_abs_delta": float(d.max()) if d.size else 0.0,
           "max_delta_in_ulp": float((d / ulp).max()) if d.size else 0.0,
           "mean_abs_delta": float(d.mean()) if d.size else 0.0}
    if d.size:
        k = int(np.argmax(d))
        ij = np.argwhere(mask)[k]
        out["argmax_cell"] = {"i_f": int(ij[0]), "j_mu": int(ij[1]),
                              "f_agn": float(F_GRID[ij[0]]),
                              "mu_chi_c2": float(MU_GRID[ij[1]]),
                              "a8": float(a8[ij[0], ij[1]]),
                              "a9": float(a9[ij[0], ij[1]])}
    return out


def stage_gate_a_full(args):
    import h5py
    env = _cpu_setup("gate_a_full", import_darksirens=False)
    done, headers, dropped = load_done()
    with h5py.File(A8_ARM_J_H5, "r") as h5:
        f8, mu8 = h5["f_grid"][:], h5["mu_chi_c2_grid"][:]
        a8 = {"logL": h5["log_likelihood"][:],
              "logL_pe": h5["guard/logL_pe"][:],
              "logL_selection": h5["guard/logL_selection"][:],
              "log_mu": h5["guard/log_mu"][:],
              "Neff": h5["guard/Neff"][:],
              "threshold": h5["guard/threshold"][:]}
        rej8 = h5["guard/rejected"][:].astype(bool)
    if not (np.array_equal(f8, F_GRID) and np.array_equal(mu8, MU_GRID)):
        raise RuntimeError("[fatal] the anchor plane is not Analysis 8's grid")

    ll9, fin9, pull9, missing = _slab_arrays(done, H0_ANCHOR)
    out = {
        "check": ("Gate A in full: the whole H0 = 67.74 slab of J9 (all 2501 "
                  "cells) against analysis_8/results/arm_J_joint.h5"),
        "a8_file": str(A8_ARM_J_H5),
        "n_cells": int(a8["logL"].size),
        "rows_missing": missing,
        "cross_hardware": ("Analysis 8 ran on a local H100 NVL, Analysis 9 on a "
                           "rita A100-80; 2 ULP is the residual already measured "
                           "in closure check 9.1 on a 40-cell subset"),
        "one_ulp_at_the_seed100_scale": 9.094947017729282e-13,
        "tolerances": dict(TOLERANCES),
        "environment": env,
    }
    if missing:
        out["verdict"] = "PENDING -- the anchor slab is incomplete"
        _write(DIAG / "a9_gate_a_full_slab.json", out)
        print(f"[pending] {len(missing)} of {F_GRID.size} anchor rows missing")
        return out

    rej9 = ~fin9
    both = (~rej9) & (~rej8)
    a9 = {"logL": ll9, "logL_pe": pull9("logL_pe"),
          "logL_selection": pull9("logL_selection"), "log_mu": pull9("log_mu"),
          "Neff": pull9("Neff"), "threshold": pull9("threshold")}
    out["terms"] = {k: _ulp_stats(a9[k], a8[k], both) for k in a8}
    out["rejected_mask"] = {
        "n_rejected_a8": int(rej8.sum()), "n_rejected_a9": int(rej9.sum()),
        "identical": bool(np.array_equal(rej8, rej9)),
        "n_cells_differing": int((rej8 != rej9).sum()),
        "rejected_cells": [{"f_agn": float(F_GRID[i]), "mu_chi_c2": float(MU_GRID[j])}
                           for i, j in np.argwhere(rej9)],
        "f_agn_min_rejected": (float(F_GRID[np.argwhere(rej9)[:, 0].min()])
                               if rej9.any() else None),
        "mu_chi_c2_min_rejected": (float(MU_GRID[np.argwhere(rej9)[:, 1].min()])
                                   if rej9.any() else None),
    }
    scan_h0f = A8.import_scan_h0f()
    marg9 = GC._marginals(F_GRID, MU_GRID, ll9, fin9, scan_h0f.marginal_ci)
    a8j = json.loads(Path(A8_ARM_J_JSON).read_text())
    cmp = {}
    for key in ("f", "mu_chi_c2"):
        b9, b8 = marg9[key], a8j[key]
        cmp[key] = {
            "a9_median": b9["median"], "a8_median": b8["median"],
            "delta_median": b9["median"] - b8["median"],
            "a9_ci68": b9["ci68"], "a8_ci68": b8["ci68"],
            "delta_ci68": [b9["ci68"][0] - b8["ci68"][0],
                           b9["ci68"][1] - b8["ci68"][1]],
            "a9_ci90": b9["ci90"], "a8_ci90": b8["ci90"],
            "delta_ci90": [b9["ci90"][0] - b8["ci90"][0],
                           b9["ci90"][1] - b8["ci90"][1]],
        }
    out["marginals"] = cmp
    out["posterior_moments"] = {"a9": marg9.get("posterior_moments"),
                                "a8": a8j.get("posterior_moments")}
    out["logL_max"] = {"a9": float(marg9["logL_max"]),
                       "a8": float(a8j["logL_max"]),
                       "delta": float(marg9["logL_max"]) - float(a8j["logL_max"]),
                       "a9_map": marg9["map"], "a8_map": a8j["map"]}
    worst = max(abs(cmp[k]["delta_median"]) for k in cmp)
    out["max_abs_marginal_delta_median"] = worst
    out["passes_abs"] = bool(out["terms"]["logL"]["max_abs_delta"]
                             <= TOLERANCES["anchor_equivalence_abs_logL"])
    out["passes_mask"] = out["rejected_mask"]["identical"]
    out["passes_marginals"] = bool(worst <= TOLERANCES["anchor_marginal_abs"])
    out["verdict"] = ("PASS" if (out["passes_abs"] and out["passes_mask"]
                                 and out["passes_marginals"]) else "FAIL")
    timing = DIAG / "_a9_stage_timing.json"
    if timing.exists():
        out["h0_liveness"] = json.loads(timing.read_text())["h0_liveness"]
    _write(DIAG / "a9_gate_a_full_slab.json", out)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("environment", "rows_missing", "rejected_mask",
                                   "tolerances")},
                     indent=2, default=GC._json_default)[:4000])
    print("rejected mask identical:", out["rejected_mask"]["identical"],
          "| a8", out["rejected_mask"]["n_rejected_a8"],
          "| a9", out["rejected_mask"]["n_rejected_a9"])
    return out


# =========================================================================== #
# Specification section 11: the comparison that matters, on matched axes
# =========================================================================== #
def _w(block, lev):
    return float(block[lev][1] - block[lev][0])


def _analysis_2_seed100():
    """Analysis 2's own seed-100 (H0, f) measurement: UNMARKED events, one
    shared population, the SAME 201-node H0 axis, the same marginal_ci."""
    if not Path(A2_SEED100_JSON).exists():
        return {}
    j = json.loads(Path(A2_SEED100_JSON).read_text())
    out = {"file": str(A2_SEED100_JSON), "logL_max": j.get("logL_max"),
           "map": j.get("map"), "rho_H0_f": j.get("rho")}
    for k in ("H0", "f"):
        b = j[k]
        out[k] = {"median": b["median"], "map": b.get("map"),
                  "ci68": b["ci68"], "ci90": b["ci90"],
                  "width68": _w(b, "ci68"), "width90": _w(b, "ci90")}
    out["note"] = ("same realisation and the same H0 axis, but the UNMARKED "
                   "events and ONE shared population: the difference from S9 is "
                   "what the two-tracer marked mock does to the spatial "
                   "measurement, and the difference from J9 is what the spin "
                   "mark does on top of that")
    return out


def stage_section_11(args):
    _cpu_setup("section_11", import_darksirens=False)
    s9 = json.loads((RESULTS / "s9_spatial.json").read_text())
    j9 = json.loads((RESULTS / f"{args.out_stem}.json").read_text())
    a8 = json.loads(Path(A8_ARM_J_JSON).read_text())

    s9_axis = np.asarray(s9["grid"]["H0"]["values"], dtype=float)
    j9_axis = np.asarray(j9["grid"]["H0"]["values"], dtype=float)
    matched = {
        "f_axis_shared": bool(s9["grid"]["f_agn"] == j9["grid"]["f_agn"]),
        "mu_axis_shared_with_analysis_8": True,
        "every_J9_H0_node_is_an_S9_node": bool(
            all(np.any(s9_axis == v) for v in j9_axis)),
        "n_H0_nodes": {"S9": int(s9_axis.size), "J9": int(j9_axis.size)},
        "note": ("all three arms share Analysis 8's f and mu_chi_c2 axes and "
                 "every J9 H0 node is bitwise an S9 node, so the widths below "
                 "are compared on matched axes, not on resampled ones"),
    }
    out = {"analysis": "analysis_9_marked_multitracer_H0_fagn",
           "section": "11 -- what the mark does, measured",
           "matched_axes": matched, "comparisons": {}}

    out["comparisons"]["H0_spatial_S9_vs_marked_J9"] = {
        "S9": {"median": s9["H0"]["median"], "mode": s9["H0"]["marginal_mode"],
               "map": s9["map"]["H0"], "ci68": s9["H0"]["ci68"],
               "ci90": s9["H0"]["ci90"], "width68": _w(s9["H0"], "ci68"),
               "width90": _w(s9["H0"], "ci90")},
        "J9": {"median": j9["H0"]["median"], "mode": j9["H0"]["marginal_mode"],
               "map": j9["map"]["H0"], "ci68": j9["H0"]["ci68"],
               "ci90": j9["H0"]["ci90"], "width68": _w(j9["H0"], "ci68"),
               "width90": _w(j9["H0"], "ci90")},
        "width68_ratio_J9_over_S9": _w(j9["H0"], "ci68") / _w(s9["H0"], "ci68"),
        "width90_ratio_J9_over_S9": _w(j9["H0"], "ci90") / _w(s9["H0"], "ci90"),
        "median_shift_J9_minus_S9": j9["H0"]["median"] - s9["H0"]["median"],
        "map_shift_J9_minus_S9": j9["map"]["H0"] - s9["map"]["H0"],
    }
    out["comparisons"]["f_AGN_S9_vs_J9"] = {
        "S9": {"median": s9["f"]["median"], "mode": s9["f"]["marginal_mode"],
               "map": s9["map"]["f_agn"], "ci68": s9["f"]["ci68"],
               "ci90": s9["f"]["ci90"], "width68": _w(s9["f"], "ci68"),
               "width90": _w(s9["f"], "ci90")},
        "J9": {"median": j9["f"]["median"], "mode": j9["f"]["marginal_mode"],
               "map": j9["map"]["f_agn"], "ci68": j9["f"]["ci68"],
               "ci90": j9["f"]["ci90"], "width68": _w(j9["f"], "ci68"),
               "width90": _w(j9["f"], "ci90")},
        "width68_ratio_J9_over_S9": _w(j9["f"], "ci68") / _w(s9["f"], "ci68"),
        "width90_ratio_J9_over_S9": _w(j9["f"], "ci90") / _w(s9["f"], "ci90"),
        "median_shift_J9_minus_S9": j9["f"]["median"] - s9["f"]["median"],
        "map_shift_J9_minus_S9": j9["map"]["f_agn"] - s9["map"]["f_agn"],
    }
    out["comparisons"]["dmu_chi_A8_fixed_H0_vs_J9_free_H0"] = {
        "analysis_8_fixed_H0": {
            "file": str(A8_ARM_J_JSON), "median": a8["mu_chi_c2"]["median"],
            "map": a8["map"]["mu_chi_c2"], "ci68": a8["mu_chi_c2"]["ci68"],
            "ci90": a8["mu_chi_c2"]["ci90"],
            "width68": _w(a8["mu_chi_c2"], "ci68"),
            "width90": _w(a8["mu_chi_c2"], "ci90")},
        "J9_free_H0": {
            "median": j9["mu_chi_c2"]["median"],
            "mode": j9["mu_chi_c2"]["marginal_mode"],
            "map": j9["map"]["mu_chi_c2"], "ci68": j9["mu_chi_c2"]["ci68"],
            "ci90": j9["mu_chi_c2"]["ci90"],
            "width68": _w(j9["mu_chi_c2"], "ci68"),
            "width90": _w(j9["mu_chi_c2"], "ci90")},
        "width68_ratio_J9_over_A8": (_w(j9["mu_chi_c2"], "ci68")
                                     / _w(a8["mu_chi_c2"], "ci68")),
        "width90_ratio_J9_over_A8": (_w(j9["mu_chi_c2"], "ci90")
                                     / _w(a8["mu_chi_c2"], "ci90")),
        "median_shift_J9_minus_A8": (j9["mu_chi_c2"]["median"]
                                     - a8["mu_chi_c2"]["median"]),
        "map_shift_J9_minus_A8": j9["map"]["mu_chi_c2"] - a8["map"]["mu_chi_c2"],
    }
    out["J9_correlations"] = j9["posterior_moments"]["correlation"]
    out["S9_correlation"] = s9["posterior_moments"]["correlation"]
    out["analysis_2_unmarked_reference"] = _analysis_2_seed100()
    out["analysis_2_as_recorded_in_s9"] = s9.get("analysis_2_unmarked", {})
    _write(RESULTS / "section_11_comparison.json", out)
    print(json.dumps(out, indent=2, default=GC._json_default))
    return out


# =========================================================================== #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=list(CPU_STAGES) + list(GPU_STAGES))
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--n_chunks", type=int, default=1)
    ap.add_argument("--h0_axis",
                    choices=sorted(set(H0_AXES) | set(H0_AXES_EXPLICIT)),
                    default="registered",
                    help="j9 = the 28-node production window (default for the "
                         "production cube); s9 = the 202-node spatial axis; "
                         "registered = 36 nodes [60.24, 77.74]; fallback = 25 "
                         "nodes [62.74, 74.74] (pre-registered cost trim)")
    ap.add_argument("--out_stem", default="j9_marked",
                    help="assemble: results/<stem>.{h5,json}")
    ap.add_argument("--j9_window", type=float, nargs=2, default=list(J9_WINDOW),
                    help="s9_assemble: the proposed J9 H0 window to confirm")
    ap.add_argument("--h0_kmin", type=int, default=None,
                    help="owner-approved override: first node index, H0 = 67.74 + 0.5k")
    ap.add_argument("--h0_kmax", type=int, default=None)
    ap.add_argument("--stop_after_s", type=float, default=0.0,
                    help="stop cleanly after this many seconds of scanning")
    ap.add_argument("--allow_partial", action="store_true",
                    help="assemble: write a coverage report instead of failing")
    args = ap.parse_args(argv)
    return {
        "provenance": stage_provenance, "timing": stage_timing,
        "anchor": stage_anchor, "scan": stage_scan,
        "compare_a8": stage_compare_a8, "status": stage_status,
        "assemble": stage_assemble, "s9": stage_s9,
        "s9_assemble": stage_s9_assemble, "s9_status": stage_s9_status,
        "gate_a_full": stage_gate_a_full, "section_11": stage_section_11,
    }[args.stage](args)


if __name__ == "__main__":
    main()
