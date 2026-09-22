#!/usr/bin/env python
"""Analysis 10 -- the registered closure gate 15.1-15.7 on the TWO-MARK mock,
plus the measured cost model that sizes the joint cube.

The mock of record is

    working/data/seed100/events/events_marked_dmu0p10_dmuG5.h5
    md5 427990378e299850a9c0708d389bc0bf   (1000 events, nsamp 2000)

which is a DIFFERENT file from the one Analyses 8 and 9 ran on
(``events_marked_dmu0p10.h5``): a peak-location mark changes the masses and
therefore, through rho_opt ~ Mc_det^{5/6}/dL, the detected set.  Every number
here is measured on the new file; nothing is read off Analysis 8's cubes.

  15.1  BOTH MARKS ZERO.  At dmu_chi = 0 AND dmu_G = 0 the two-mark model must
        reproduce the shared-population (analysis-2 'old') K=2 likelihood.  The
        reference is a genuinely different model -- 8 sampled labels, no _c2
        population coordinate at all -- built on the SAME data object.

  15.2  MASS MARK ZERO.  At dmu_G = 0 the two-mark model must reproduce the
        spin-only (Analysis-8) model cell for cell on the shared
        (f, dmu_chi) lattice, on THIS file.

  15.3  SPIN MARK ZERO.  At dmu_chi = 0 the two-mark model moves with dmu_G at
        f > 0 and is frozen at f = 0: liveness with the spin mark off.

  15.4  f = 0.  Both marks inert, BITWISE, in the total, the PE term, the
        selection term and log_mu alike.

  15.5  f = 1.  Both marks live: a distinct value per node along each axis.

  15.6  GAUSSIAN-MEAN LIVENESS (MANDATORY).  |dlnL| per 1 Msun step, in the PE
        term and the selection term SEPARATELY as well as in the total.
        15.1-15.5 and 15.7 all pass for free if a coordinate is silently
        ignored; this is the check that cannot.

  15.7  K=1 ENDPOINT IDENTITIES, registered as EXACTLY 0.0.  f = 0 equals a K=1
        GAL build; f = 1 at (dmu_chi = +0.10, dmu_G = +5) equals a K=1 AGN build
        whose pinned base block carries mu_G = 40 and mu_chi = +0.10, and
        f = 1 at (0, -10) equals a K=1 AGN build at mu_G = 25, mu_chi = 0.

  COST MODEL.  Build time, steady-state s/eval and peak GPU memory, MEASURED on
  the rita A100-80, and the projected cost of the registered 41 x 61 x 21 cube.

Stages (GPU stages are rita-only and refuse to run anywhere else):

    sbatch --export=ALL,STAGE="closure k1refs" scripts/submit_a10_closure_rita.sbatch
    python scripts/a10_closure.py --stage assemble        # CPU

Every write lands under the Analysis-10 directory.  Nothing is regenerated,
no darksirens file is touched, and the A8/A9 trees are imported read-only
(``sys.dont_write_bytecode`` is set before they are reached).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True                    # A8/A9 trees are READ-ONLY
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE.parent
DIAG = ANALYSIS_DIR / "diagnostics"
RESULTS = ANALYSIS_DIR / "results"

A9_SCRIPTS = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/"
                  "analysis_9_marked_multitracer_H0_fagn/scripts")
for _p in (str(HERE), str(A9_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import a10_likelihood as A10                      # the A10 builder (steers a8)
import a9_scan as A9                              # the rita harness + asserts
import a9_closure as A9C                          # GpuMemoryPoller, _ulp, _cmp

A8, GC = A9.A8, A9.GC
L_MU = A10.MU_CHI_C2_LABEL                        # '$\mu_\chi$_c2'
L_MG = A10.MU_G_C2_LABEL                          # '$\mu_{\rm G}$_c2'

# --------------------------------------------------------------------------- #
# The data of record -- the NEW two-mark mock, asserted by md5
# --------------------------------------------------------------------------- #
GW_PATH_A10 = ("/hildafs/projects/phy230014p/magana/gws-agn/working/data/seed100/"
               "events/events_marked_dmu0p10_dmuG5.h5")
GW_MD5_A10 = "427990378e299850a9c0708d389bc0bf"
N_EVENTS_EXPECTED = 1000
NSAMP_EXPECTED = 2000

# Truths, both planted and realised, carried into every output.
TRUTH = {
    # The model's f_AGN is the PRE-SELECTION host fraction, so the TARGET is the
    # planted 0.30 (the underlying draw among ~2e5 proposals, 0.30 to +/-0.001).
    # 0.357 is the DETECTED-SET fraction: a descriptive number that a correctly
    # selection-corrected marked model should NOT recover -- a model blind to
    # the branch-dependent detectability would drift towards it.  The two are
    # not two equivalent truths and are never labelled as such.
    "f_agn_planted": 0.30,
    "f_agn_detected_fraction": 0.357,
    "f_agn_convention": (
        "TARGET = the planted / underlying pre-selection host fraction 0.30.  "
        "DESCRIPTIVE = the detected-set fraction 0.357 (357 of 1000 detected "
        "events are AGN-hosted), which the heavier, louder AGN branch inflates "
        "above 0.30 and which a selection-corrected model should NOT recover."),
    "dmu_chi_planted": 0.10,
    "dmu_chi_realised": 0.099502,
    "dmu_chi_realised_err": 0.006690,
    "dmu_G_planted": 5.0,
    "dmu_G_note": ("the planted +5 Msun is a HYPERPARAMETER of the draw "
                   "(mu_G = 35 GAL / 40 AGN); the detected-set medians "
                   "36.813 / 41.557 Msun are descriptive only and are NOT the "
                   "mark"),
    "mu_G_gal_fixed": A10.MU_G_FID,
    "mu_chi_gal_fixed": A10.MU_CHI_FID,
}

TOL = {
    "ulp_same_hardware": 4.0,
    "abs_logL": 1.0e-6,
    "endpoint_abs_logL": 0.0,          # 15.7 is registered as EXACTLY 0.0
    "liveness_min_abs_dlogL": 1.0e-3,
    "note": ("residuals are quoted in ULP of the compared value alongside the "
             "absolute bound; 15.7 is an exact equality, not a tolerance, and "
             "15.4 is a bitwise check (one distinct hex per row)"),
}

# ---- the registered closure node sets (fixed here, before the run) --------- #
F_151 = (0.0, 0.1, 0.275, 0.357, 0.5, 0.75, 1.0)
F_152 = (0.1, 0.275, 0.357, 0.5, 1.0)
MU_152 = (-0.10, 0.0, 0.10, 0.2275)
MG_153 = (-10.0, -5.0, 0.0, 5.0, 10.0)
F_153 = 0.357
MU_GRID_154 = (-0.20, 0.0, 0.25)
MG_GRID_154 = (-10.0, 0.0, 10.0)
MG_156 = (-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0)
LIVE_156 = ((0.357, 0.10), (0.275, 0.0))
# 15.7: (f, dmu_chi, dmu_G, reference tag)
CELLS_157 = (
    (0.0, 0.10, 5.0, "K1_GAL"),
    (0.0, 0.0, -10.0, "K1_GAL"),
    (1.0, 0.10, 5.0, "K1_AGN_mu40_chi010"),
    (1.0, 0.0, -10.0, "K1_AGN_mu25_chi000"),
)
# The K=1 AGN references: pinned base block, by LaTeX label.
K1_REFS = {
    "K1_GAL": {"survey": "gal", "pins": {}},
    "K1_GAL_new": {"survey": "gal", "pins": {}, "mode": "new"},
    "K1_AGN_mu40_chi010": {"survey": "agn", "mode": "new",
                           "pins": {"G.mu": 40.0, "mu_chi": 0.10}},
    "K1_AGN_mu25_chi000": {"survey": "agn", "mode": "new",
                           "pins": {"G.mu": 25.0, "mu_chi": 0.0}},
}

# The registered production grid (GATES.md "Registered grid").
F_GRID = np.asarray(GC.F_GRID_2D, dtype=float)            # linspace(0, 1, 41)
MU_GRID = np.asarray(GC.MU_GRID_2D, dtype=float)          # linspace(-.20,.25,61)
MG_GRID = np.linspace(-10.0, 10.0, 21)                    # 1 Msun spacing
# THE ONE ALLOWED GRID REFINEMENT (owner decision, 2026-09-21): six half-integer
# nodes inside [2, 8], where A10-M's measured dmu_G marginal sits above ~e^-9 of
# its peak.  They are ADDITIVE ROWS on the same physical-coordinate key, not a
# new grid; the merged axis is 1 Msun outside [2, 8] and 0.5 Msun inside, and it
# is integrated with the SAME non-uniform trapezoid weights.  No second
# refinement follows.
MG_REFINE = np.array([2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
MG_GRID_REFINED = np.sort(np.concatenate([MG_GRID, MG_REFINE]))   # 27 nodes


# --------------------------------------------------------------------------- #
# Writes, confined to this analysis directory
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


def _md5(path, block=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def _hex(v):
    try:
        return float(v).hex()
    except (TypeError, ValueError):
        return None


def _ulp(x):
    return A9C._ulp(x)


def _cmp(a, b):
    return A9C._cmp(a, b)


def _cmp4(a, b):
    """The four terms every closure comparison reports."""
    return {
        "total": _cmp(a["logL"], b["logL"]),
        "pe": _cmp(a.get("logL_pe", np.nan), b.get("logL_pe", np.nan)),
        "selection": _cmp(a.get("logL_selection", np.nan),
                          b.get("logL_selection", np.nan)),
        "log_mu": _cmp(a.get("log_mu", np.nan), b.get("log_mu", np.nan)),
    }


def _worst4(blocks):
    out = {}
    for k in ("total", "pe", "selection", "log_mu"):
        vals = [abs(b[k]["abs_diff"]) for b in blocks if np.isfinite(b[k]["abs_diff"])]
        ulps = [abs(b[k]["diff_in_ulp"]) for b in blocks
                if np.isfinite(b[k]["diff_in_ulp"])]
        out[k] = {"max_abs_diff": float(max(vals)) if vals else float("nan"),
                  "max_abs_diff_in_ulp": float(max(ulps)) if ulps else float("nan")}
    out["max_abs_diff_any_term"] = float(max(
        (out[k]["max_abs_diff"] for k in ("total", "pe", "selection", "log_mu")
         if np.isfinite(out[k]["max_abs_diff"])), default=float("nan")))
    out["max_ulp_any_term"] = float(max(
        (out[k]["max_abs_diff_in_ulp"] for k in
         ("total", "pe", "selection", "log_mu")
         if np.isfinite(out[k]["max_abs_diff_in_ulp"])), default=float("nan")))
    return out


# --------------------------------------------------------------------------- #
# Provenance assertions specific to Analysis 10
# --------------------------------------------------------------------------- #
def assert_a10_inputs(verbose=True, with_md5=True):
    out = {"events_file": GW_PATH_A10}
    p = Path(GW_PATH_A10)
    if not p.exists():
        raise SystemExit(f"[fatal] events file missing: {p}")
    out["events_bytes"] = int(p.stat().st_size)
    if with_md5:
        t0 = time.time()
        md5 = _md5(p)
        out["events_md5"] = md5
        out["events_md5_seconds"] = time.time() - t0
        if md5 != GW_MD5_A10:
            raise SystemExit(f"[fatal] events md5 {md5} != {GW_MD5_A10}")
        if verbose:
            print(f"  [OK ] events md5 {md5}")
    out["fiducials"] = A10.assert_fiducials()
    if verbose:
        print(f"  [OK ] fiducials: G.mu = "
              f"{out['fiducials']['G.mu']['fiducial']}, mu_chi = "
              f"{out['fiducials']['mu_chi']['fiducial']}")
    return out


def assert_cell(cell, expect_labels, tag):
    if list(cell.labels) != list(expect_labels):
        raise SystemExit(f"[fatal] {tag} labels {list(cell.labels)} != "
                         f"{list(expect_labels)}")
    if int(cell.data["nEvents"]) != N_EVENTS_EXPECTED:
        raise SystemExit(f"[fatal] {tag} nEvents {cell.data['nEvents']} != "
                         f"{N_EVENTS_EXPECTED}")
    if int(cell.data["nsamp"]) != NSAMP_EXPECTED:
        raise SystemExit(f"[fatal] {tag} nsamp {cell.data['nsamp']} != "
                         f"{NSAMP_EXPECTED}")
    if str(cell.opts.gw_path) != GW_PATH_A10:
        raise SystemExit(f"[fatal] {tag} events file {cell.opts.gw_path} != "
                         f"{GW_PATH_A10}")
    print(f"  [OK ] {tag}: {len(cell.labels)} labels, nEvents="
          f"{int(cell.data['nEvents'])}, events={Path(cell.opts.gw_path).name}")
    return list(cell.labels)


def _config_record(cell):
    return {
        "labels": list(cell.labels),
        "mode": cell.mode,
        "per_catalog_pop_params": list(cell.per_catalog_pop_params),
        "fix_population": bool(cell.opts.fix_population),
        "n_fixed_parameter_values": len(cell.fixed_parameter_values),
        "fixed_parameter_values": {k: float(v)
                                   for k, v in cell.fixed_parameter_values.items()},
        "base_coord": {k: float(v) for k, v in zip(cell.labels, cell.base)},
        "survey_paths": [str(s) for s in cell.survey_paths],
        "events_file": str(cell.opts.gw_path),
        "selection_file": str(cell.opts.gwselection_path),
        "nEvents": int(cell.data["nEvents"]),
        "nsamp": int(cell.data["nsamp"]),
        "Ndraw": float(cell.data["Ndraw"]),
    }


# --------------------------------------------------------------------------- #
# The evaluation ledger: every cell computed ONCE, keyed on its coordinates
# --------------------------------------------------------------------------- #
class Ledger:
    """(f, dmu_chi, dmu_G) -> one evaluated record, for one built cell."""

    def __init__(self, cell, tag, kind):
        self.cell, self.tag, self.kind = cell, tag, kind
        self.store, self.order = {}, []

    @staticmethod
    def key(f, mu, mg):
        return "{:.10g}|{:.10g}|{:.10g}".format(
            float(f), float(0.0 if mu is None else mu),
            float(0.0 if mg is None else mg))

    def _overrides(self, f, mu, mg):
        ov = {"fcat_2": float(f)}
        if self.kind in ("a10", "spin"):
            ov[L_MU] = float(0.0 if mu is None else mu)
        if self.kind == "a10":
            ov[L_MG] = A10.dmuG_to_label(0.0 if mg is None else mg)
        return ov

    def get(self, f, mu=None, mg=None, note="", force=False):
        k = self.key(f, mu, mg)
        if k in self.store and not force:
            return self.store[k]
        r = self.cell.evaluate(**self._overrides(f, mu, mg))
        rec = {
            "config": self.tag, "f_agn": float(f),
            "dmu_chi": (None if mu is None else float(mu)),
            "dmu_G": (None if mg is None else float(mg)),
            "mu_chi_c2": (None if mu is None else float(A10.dmuchi_to_label(mu))),
            "mu_G_c2": (None if mg is None else float(A10.dmuG_to_label(mg))),
            "note": note,
            "logL_hex": r["logL_hex"],
            "logL_pe_hex": _hex(r.get("logL_pe")),
            "logL_selection_hex": _hex(r.get("logL_selection")),
            "log_mu_hex": _hex(r.get("log_mu")),
            "coord": {lbl: float(v) for lbl, v in r["coord"].items()},
            **GC._cell_record(r),
        }
        self.store[k] = rec
        self.order.append(k)
        print(f"    [{self.tag}] f={f:<6.4g} dchi="
              f"{'--' if mu is None else format(mu, '+.4f')} dG="
              f"{'--' if mg is None else format(mg, '+6.2f')}  "
              f"logL={rec['logL']!r}  {r['seconds']:.3f}s  {note}")
        sys.stdout.flush()
        return rec

    def rows(self):
        return [self.store[k] for k in self.order]

    def seconds(self, skip_first=1):
        return [self.store[k]["seconds"] for k in self.order][skip_first:]


# =========================================================================== #
# Stage: closure  (GPU, rita)
# =========================================================================== #
def stage_closure(args):
    env = A9._gpu_setup("closure")
    env["a10_inputs"] = assert_a10_inputs()
    poller = A9C.GpuMemoryPoller()
    poller.start()

    surveys = [A8.SURVEY_GAL, A8.SURVEY_AGN]
    t0 = time.time()
    cell = A10.build_a10("A10", surveys, verbose=True, gw_path=GW_PATH_A10)
    build_a10_s = time.time() - t0
    assert_cell(cell, A10.EXPECTED_LABELS_A10, "A10 two-mark")
    print(f"\n[cost] A10 two-mark build: {build_a10_s:.2f} s")
    mem_after_build = A9C._jax_mem()

    led = Ledger(cell, "A10", "a10")

    # ------------------------------------------------------------- 15.1 ----
    print("\n== 15.1  both marks zero (two-mark side) ==")
    for f in F_151:
        led.get(f, 0.0, 0.0, "15.1 two-mark at both marks zero")

    # ------------------------------------------------------------- 15.2 ----
    print("\n== 15.2  mass mark zero (two-mark side) ==")
    for f in F_152:
        for mu in MU_152:
            led.get(f, mu, 0.0, "15.2 two-mark at dmu_G = 0")

    # ------------------------------------------------------------- 15.3 ----
    print("\n== 15.3  spin mark zero ==")
    for mg in MG_153:
        led.get(F_153, 0.0, mg, "15.3 f = 0.357, dmu_chi = 0")
        led.get(0.0, 0.0, mg, "15.3 f = 0 control, must be frozen")

    # ------------------------------------------------------------- 15.4 ----
    print("\n== 15.4  f = 0: both marks inert, bitwise ==")
    for mu in MU_GRID_154:
        for mg in MG_GRID_154:
            led.get(0.0, mu, mg, "15.4 f = 0")

    # ------------------------------------------------------------- 15.5 ----
    print("\n== 15.5  f = 1: both marks live ==")
    for mu in MU_GRID_154:
        for mg in MG_GRID_154:
            led.get(1.0, mu, mg, "15.5 f = 1")

    # ------------------------------------------------------------- 15.6 ----
    print("\n== 15.6  Gaussian-mean liveness (MANDATORY) ==")
    for f, mu in LIVE_156:
        for mg in MG_156:
            led.get(f, mu, mg, f"15.6 liveness at f = {f:g}, dmu_chi = {mu:+.4f}")

    # ------------------------------------------------------------- 15.7 ----
    print("\n== 15.7  K=1 endpoint identities (two-mark side) ==")
    for f, mu, mg, tag in CELLS_157:
        led.get(f, mu, mg, f"15.7 endpoint against {tag}")

    # ------------------------------------------------------- cost model ----
    print("\n== cost model: steady state (median of 20) ==")
    burn = [led.get(0.5, 0.0, float(MG_GRID[k]), "steady-state timer", force=True)
            for k in range(1, 21)]
    steady = [r["seconds"] for r in burn]
    mem_peak = A9C._jax_mem()
    n_a10_evals = len(led.order)

    # --------------------------------------------- the two reference shapes --
    # Built one at a time on the SAME data object, and released, so at most two
    # compiled likelihoods are resident.
    print("\n== 15.1 reference: the shared-population (analysis-2 'old') shape ==")
    t0 = time.time()
    ref_old = A10.build_spatial_reference("A10_SPATIAL", surveys, data=cell.data,
                                          verbose=True, gw_path=GW_PATH_A10)
    build_old_s = time.time() - t0
    if L_MU in ref_old.labels or L_MG in ref_old.labels:
        raise SystemExit(f"[fatal] the spatial reference carries a per-catalog "
                         f"population coordinate: {ref_old.labels}")
    assert_cell(ref_old, ["H0", "log10n0", "delta", "sigma_kde", "log10n0_c2",
                          "delta_c2", "sigma_kde_c2", "fcat_2"],
                "spatial reference")
    print(f"[cost] spatial-reference build on the SAME data: {build_old_s:.2f} s")
    led_old = Ledger(ref_old, "SPATIAL", "old")
    for f in F_151:
        led_old.get(f, None, None, "15.1 shared-population reference")
    cfg_old = _config_record(ref_old)
    del ref_old
    import gc
    gc.collect()

    print("\n== 15.2 reference: the spin-only (Analysis-8) shape ==")
    t0 = time.time()
    ref_spin = A10.build_spin_only("A10_SPIN_ONLY", surveys, data=cell.data,
                                   verbose=True, gw_path=GW_PATH_A10)
    build_spin_s = time.time() - t0
    if L_MG in ref_spin.labels:
        raise SystemExit(f"[fatal] the spin-only reference carries {L_MG}: "
                         f"{ref_spin.labels}")
    assert_cell(ref_spin, ["H0", "log10n0", "delta", "sigma_kde", "log10n0_c2",
                           "delta_c2", "sigma_kde_c2", L_MU, "fcat_2"],
                "spin-only reference")
    print(f"[cost] spin-only build on the SAME data: {build_spin_s:.2f} s")
    led_spin = Ledger(ref_spin, "SPIN_ONLY", "spin")
    for f in F_152:
        for mu in MU_152:
            led_spin.get(f, mu, None, "15.2 spin-only reference")
    cfg_spin = _config_record(ref_spin)
    del ref_spin
    gc.collect()

    peak = poller.stop()
    out = {
        "stage": "closure",
        "analysis": "analysis_10_mass_spin_marked_multitracer",
        "environment": env,
        "gpu_name": subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip(),
        "provenance": A8.provenance(gw_path=GW_PATH_A10, survey_paths=surveys),
        "configs": {"A10": _config_record(cell), "SPATIAL": cfg_old,
                    "SPIN_ONLY": cfg_spin},
        "node_sets": {
            "F_151": list(F_151), "F_152": list(F_152), "MU_152": list(MU_152),
            "MG_153": list(MG_153), "F_153": F_153,
            "MU_154": list(MU_GRID_154), "MG_154": list(MG_GRID_154),
            "MG_156": list(MG_156), "LIVE_156": [list(x) for x in LIVE_156],
            "CELLS_157": [list(x) for x in CELLS_157],
        },
        "cost": {
            "build_seconds_A10_two_mark": build_a10_s,
            "build_seconds_spatial_reference_same_data": build_old_s,
            "build_seconds_spin_only_same_data": build_spin_s,
            "first_eval_seconds": led.rows()[0]["seconds"],
            "steady_state_median_seconds": float(np.median(steady)),
            "steady_state_mean_seconds": float(np.mean(steady)),
            "steady_state_min_seconds": float(np.min(steady)),
            "steady_state_max_seconds": float(np.max(steady)),
            "steady_state_sd_seconds": float(np.std(steady)),
            "steady_state_n": len(steady),
            "all_evals_median_seconds": float(np.median(led.seconds(skip_first=1))),
            "n_evaluations_A10": n_a10_evals,
            "n_evaluations_spatial": len(led_old.order),
            "n_evaluations_spin_only": len(led_spin.order),
            "jax_memory_stats_after_build": mem_after_build,
            "jax_memory_stats_peak": mem_peak,
            "peak_gpu_nvidia_smi": peak,
        },
        "ledger_A10": led.rows(),
        "ledger_SPATIAL": led_old.rows(),
        "ledger_SPIN_ONLY": led_spin.rows(),
        "truth": dict(TRUTH),
        "tolerances": dict(TOL),
    }
    _write(DIAG / "_a10_closure_main.json", out)
    print(f"\n[cost] build {build_a10_s:.1f} s | steady state "
          f"{out['cost']['steady_state_median_seconds']:.4f} s/eval | peak GPU "
          f"{peak['peak_mib_this_pid']:.0f} MiB (this pid) / "
          f"{peak['peak_mib_all_compute_apps']:.0f} MiB (all apps)")
    return out


# =========================================================================== #
# Stage: k1refs  (GPU, rita) -- the single-branch references for 15.7
# =========================================================================== #
def _pinned_fpv(pins):
    """``fixed_parameter_values`` for mode 'new' with the base block RE-pinned.

    ``pins`` is keyed on the PLAIN population slot name ('G.mu', 'mu_chi'); the
    pin itself is written under the LaTeX prior label, which is the only
    spelling ``build_parameter_space`` accepts for the BASE block.
    """
    fpv = A8.fixed_parameter_values_for("new")
    labels, plain, _fid = A8.population_labels_and_fiducial()
    by_plain = {p: l for p, l in zip(plain, labels)}
    applied = {}
    for name, value in pins.items():
        if name not in by_plain:
            raise SystemExit(f"[fatal] unknown population slot {name!r}")
        fpv[by_plain[name]] = float(value)
        applied[by_plain[name]] = float(value)
    return fpv, applied


class _steer_fpv:
    """Temporarily replace ``a8.fixed_parameter_values_for`` (A8 tree read-only)."""

    def __init__(self, fpv):
        self.fpv, self._true = fpv, None

    def __enter__(self):
        self._true = A8.fixed_parameter_values_for
        A8.fixed_parameter_values_for = lambda mode: dict(self.fpv)
        return self

    def __exit__(self, *exc):
        A8.fixed_parameter_values_for = self._true
        return False


def stage_k1refs(args):
    env = A9._gpu_setup("k1refs")
    env["a10_inputs"] = assert_a10_inputs(with_md5=False)
    poller = A9C.GpuMemoryPoller()
    poller.start()
    out = {"stage": "k1refs", "environment": env, "configs": {}, "cells": {},
           "build_seconds": {}, "pins_applied": {}}
    data_by_survey = {}
    import gc
    for tag, spec in K1_REFS.items():
        survey = A8.SURVEY_GAL if spec["survey"] == "gal" else A8.SURVEY_AGN
        mode = spec.get("mode", "old")
        data = data_by_survey.get(spec["survey"])
        t0 = time.time()
        if mode == "new":
            fpv, applied = _pinned_fpv(spec["pins"])
            out["pins_applied"][tag] = applied
            with _steer_fpv(fpv):
                cell = A8.build(tag, "new", [survey], data=data, verbose=True,
                                gw_path=GW_PATH_A10)
        else:
            out["pins_applied"][tag] = {}
            cell = A8.build(tag, "old", [survey], data=data, verbose=True,
                            gw_path=GW_PATH_A10)
        out["build_seconds"][tag] = time.time() - t0
        data_by_survey.setdefault(spec["survey"], cell.data)
        if "fcat_2" in cell.labels:
            raise SystemExit(f"[fatal] {tag} is not a K=1 build: {cell.labels}")
        if int(cell.data["nEvents"]) != N_EVENTS_EXPECTED:
            raise SystemExit(f"[fatal] {tag} nEvents {cell.data['nEvents']}")
        if str(cell.opts.gw_path) != GW_PATH_A10:
            raise SystemExit(f"[fatal] {tag} events {cell.opts.gw_path}")
        cfg = _config_record(cell)
        cfg["pins_applied"] = out["pins_applied"][tag]
        out["configs"][tag] = cfg
        r = cell.evaluate()
        out["cells"][tag] = {"logL_hex": r["logL_hex"],
                             "logL_pe_hex": _hex(r.get("logL_pe")),
                             "logL_selection_hex": _hex(r.get("logL_selection")),
                             "log_mu_hex": _hex(r.get("log_mu")),
                             "coord": {l: float(v) for l, v in r["coord"].items()},
                             **GC._cell_record(r)}
        print(f"    [{tag}] logL={r['logL']!r} ({r['seconds']:.3f}s) "
              f"labels={cell.labels}")
        sys.stdout.flush()
        del cell
        gc.collect()
    out["peak_gpu_nvidia_smi"] = poller.stop()
    _write(DIAG / "_a10_closure_k1.json", out)
    return out


# =========================================================================== #
# Stage: assemble  (CPU) -- the verdicts
# =========================================================================== #
def _find(rows, f, mu, mg):
    k = Ledger.key(f, mu, mg)
    for r in rows:
        if Ledger.key(r["f_agn"], r["dmu_chi"], r["dmu_G"]) == k:
            return r
    raise KeyError(f"no ledger row at {k}")


def _find_old(rows, f):
    for r in rows:
        if float(r["f_agn"]) == float(f):
            return r
    raise KeyError(f"no reference row at f = {f}")


def _find_spin(rows, f, mu):
    for r in rows:
        if float(r["f_agn"]) == float(f) and float(r["dmu_chi"]) == float(mu):
            return r
    raise KeyError(f"no spin-only row at (f={f}, dmu_chi={mu})")


def _frozen_block(cells):
    """Bitwise-frozen bookkeeping for one row of cells."""
    keys = ("logL_hex", "logL_pe_hex", "logL_selection_hex", "log_mu_hex")
    out = {}
    for k in keys:
        hexes = [str(c.get(k)) for c in cells]
        out[k] = {"n_distinct": len(set(hexes)), "values": sorted(set(hexes)),
                  "frozen": bool(len(set(hexes)) == 1)}
    out["all_frozen"] = bool(all(out[k]["frozen"] for k in keys))
    out["all_distinct"] = bool(all(out[k]["n_distinct"] == len(cells) for k in keys))
    vals = [c["logL"] for c in cells if np.isfinite(c["logL"])]
    out["spread_logL"] = float(max(vals) - min(vals)) if vals else float("nan")
    out["spread_logL_pe"] = _spread([c.get("logL_pe") for c in cells])
    out["spread_logL_selection"] = _spread([c.get("logL_selection") for c in cells])
    return out


def _spread(vals):
    v = [float(x) for x in vals if x is not None and np.isfinite(float(x))]
    return float(max(v) - min(v)) if v else float("nan")


def stage_assemble(args):
    env = A9._cpu_setup("assemble", import_darksirens=False)
    main = json.loads((DIAG / "_a10_closure_main.json").read_text())
    k1 = json.loads((DIAG / "_a10_closure_k1.json").read_text())
    led = main["ledger_A10"]
    led_old = main["ledger_SPATIAL"]
    led_spin = main["ledger_SPIN_ONLY"]

    logl_scale = float(np.median([abs(r["logL"]) for r in led
                                  if np.isfinite(r["logL"])]))
    ulp_here = float(np.spacing(logl_scale))

    res = {
        "analysis": "analysis_10_mass_spin_marked_multitracer",
        "gate": "15 (closure identities)",
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "environment": env,
        "provenance": main.get("provenance"),
        "events_file": GW_PATH_A10,
        "events_md5": main["environment"].get("a10_inputs", {}).get("events_md5"),
        "configs": main["configs"],
        "truth": dict(TRUTH),
        "tolerances": dict(TOL),
        "logL_scale": {"median_abs_logL": logl_scale, "one_ulp": ulp_here,
                       "note": ("1 ULP of the ACTUAL |logL| of this mock; the "
                                "registered tolerance table quotes "
                                "9.094947017729282e-13 at |lnL| ~ 4.2e3")},
        "checks": {},
    }

    # ------------------------------------------------------------- 15.1 ----
    b151 = []
    for f in F_151:
        a = _find(led, f, 0.0, 0.0)
        b = _find_old(led_old, f)
        blk = _cmp4(a, b)
        blk.update({"f_agn": float(f), "two_mark": a, "reference": b,
                    "Neff_two_mark": a.get("Neff"), "Neff_reference": b.get("Neff")})
        b151.append(blk)
    w151 = _worst4(b151)
    res["checks"]["15_1_both_marks_zero"] = {
        "description": ("two-mark at (dmu_chi, dmu_G) = (0, 0) against the "
                        "shared-population K=2 model (8 labels, no _c2 "
                        "population coordinate), same data object"),
        "f_nodes": list(F_151),
        "per_f": b151,
        "worst": w151,
        "arm_S_reference_values": [
            {"f_agn": float(b["f_agn"]), "logL": b["reference"]["logL"],
             "logL_pe": b["reference"].get("logL_pe"),
             "logL_selection": b["reference"].get("logL_selection"),
             "log_mu": b["reference"].get("log_mu"),
             "Neff": b["reference"].get("Neff")} for b in b151],
        "pass": bool(w151["max_abs_diff_any_term"] <= TOL["abs_logL"]
                     and w151["max_ulp_any_term"] <= TOL["ulp_same_hardware"]),
    }

    # ------------------------------------------------------------- 15.2 ----
    b152 = []
    for f in F_152:
        for mu in MU_152:
            a = _find(led, f, mu, 0.0)
            b = _find_spin(led_spin, f, mu)
            blk = _cmp4(a, b)
            blk.update({"f_agn": float(f), "dmu_chi": float(mu),
                        "two_mark": a, "reference": b})
            b152.append(blk)
    w152 = _worst4(b152)
    res["checks"]["15_2_mass_mark_zero"] = {
        "description": ("two-mark at dmu_G = 0 against the spin-only "
                        "(Analysis-8) 9-label model, cell for cell on the "
                        "shared (f, dmu_chi) lattice, on THIS events file"),
        "n_cells": len(b152),
        "per_cell": b152,
        "worst": w152,
        "pass": bool(w152["max_abs_diff_any_term"] <= TOL["abs_logL"]
                     and w152["max_ulp_any_term"] <= TOL["ulp_same_hardware"]),
    }

    # ------------------------------------------------------------- 15.3 ----
    live = [_find(led, F_153, 0.0, mg) for mg in MG_153]
    ctrl = [_find(led, 0.0, 0.0, mg) for mg in MG_153]
    ref0 = _find(led, F_153, 0.0, 0.0)
    d153 = [{"dmu_G": float(mg), "logL": c["logL"],
             "delta_logL_vs_dmuG_0": c["logL"] - ref0["logL"],
             "delta_logL_pe": (c["logL_pe"] - ref0["logL_pe"]),
             "delta_logL_selection": (c["logL_selection"]
                                      - ref0["logL_selection"]),
             "log_mu": c.get("log_mu"), "Neff": c.get("Neff")}
            for mg, c in zip(MG_153, live)]
    off = [abs(x["delta_logL_vs_dmuG_0"]) for x in d153 if x["dmu_G"] != 0.0]
    res["checks"]["15_3_spin_mark_zero"] = {
        "description": ("with the spin mark OFF (dmu_chi = 0) the mass mark "
                        "must still move logL at f = 0.357 and must be frozen "
                        "at f = 0"),
        "f_live": F_153, "dmu_G_nodes": list(MG_153),
        "live_cells": d153,
        "min_abs_delta_logL_off_zero": float(min(off)) if off else float("nan"),
        "f0_control": _frozen_block(ctrl),
        "pass": bool(off and min(off) > TOL["liveness_min_abs_dlogL"]
                     and _frozen_block(ctrl)["all_frozen"]),
    }

    # ------------------------------------------------------------- 15.4 ----
    cells154 = [_find(led, 0.0, mu, mg) for mu in MU_GRID_154 for mg in MG_GRID_154]
    blk154 = _frozen_block(cells154)
    res["checks"]["15_4_f0_inert"] = {
        "description": ("f_AGN = 0: total, PE, selection and log_mu bitwise "
                        "frozen over the 3 x 3 mark lattice"),
        "dmu_chi_nodes": list(MU_GRID_154), "dmu_G_nodes": list(MG_GRID_154),
        "n_cells": len(cells154),
        "bitwise": blk154,
        "cells": cells154,
        "pass": bool(blk154["all_frozen"]),
    }

    # ------------------------------------------------------------- 15.5 ----
    cells155 = {(mu, mg): _find(led, 1.0, mu, mg)
                for mu in MU_GRID_154 for mg in MG_GRID_154}
    rows_mu, rows_mg = [], []
    for mg in MG_GRID_154:
        cs = [cells155[(mu, mg)] for mu in MU_GRID_154]
        b = _frozen_block(cs)
        b.update({"axis": "dmu_chi", "dmu_G_fixed": float(mg),
                  "logL": [c["logL"] for c in cs]})
        rows_mu.append(b)
    for mu in MU_GRID_154:
        cs = [cells155[(mu, mg)] for mg in MG_GRID_154]
        b = _frozen_block(cs)
        b.update({"axis": "dmu_G", "dmu_chi_fixed": float(mu),
                  "logL": [c["logL"] for c in cs]})
        rows_mg.append(b)
    all_live = bool(all(r["all_distinct"] for r in rows_mu + rows_mg))
    res["checks"]["15_5_f1_live"] = {
        "description": ("f_AGN = 1: a distinct value per node in BOTH mark axes "
                        "with the other held, in total / PE / selection / log_mu"),
        "rows_along_dmu_chi": rows_mu,
        "rows_along_dmu_G": rows_mg,
        "min_spread_logL_along_dmu_chi": float(min(r["spread_logL"]
                                                   for r in rows_mu)),
        "min_spread_logL_along_dmu_G": float(min(r["spread_logL"]
                                                 for r in rows_mg)),
        "all_rows_distinct": all_live,
        "cells": [dict(c, dmu_chi_key=float(k[0]), dmu_G_key=float(k[1]))
                  for k, c in cells155.items()],
        "pass": all_live,
    }

    # ------------------------------------------------------------- 15.6 ----
    live_blocks = []
    for f, mu in LIVE_156:
        ref = _find(led, f, mu, 0.0)
        rows = []
        for mg in MG_156:
            c = _find(led, f, mu, mg)
            rows.append({
                "dmu_G": float(mg),
                "logL": c["logL"], "logL_pe": c.get("logL_pe"),
                "logL_selection": c.get("logL_selection"),
                "log_mu": c.get("log_mu"), "Neff": c.get("Neff"),
                "dlnL_total": c["logL"] - ref["logL"],
                "dlnL_pe": c["logL_pe"] - ref["logL_pe"],
                "dlnL_selection": c["logL_selection"] - ref["logL_selection"],
                "per_msun_total": ((c["logL"] - ref["logL"]) / mg
                                   if mg != 0 else None),
            })
        off = [r for r in rows if r["dmu_G"] != 0.0]
        one = [r for r in rows if abs(r["dmu_G"]) == 1.0]
        blk = {
            "f_agn": float(f), "dmu_chi": float(mu),
            "reference_dmu_G_0": {"logL": ref["logL"], "logL_pe": ref.get("logL_pe"),
                                  "logL_selection": ref.get("logL_selection")},
            "rows": rows,
            "min_abs_dlnL_total": float(min(abs(r["dlnL_total"]) for r in off)),
            "min_abs_dlnL_pe": float(min(abs(r["dlnL_pe"]) for r in off)),
            "min_abs_dlnL_selection": float(min(abs(r["dlnL_selection"])
                                                for r in off)),
            "one_msun_step": {
                "abs_dlnL_total": [abs(r["dlnL_total"]) for r in one],
                "abs_dlnL_pe": [abs(r["dlnL_pe"]) for r in one],
                "abs_dlnL_selection": [abs(r["dlnL_selection"]) for r in one],
                "min_abs_dlnL_total_per_msun": float(min(abs(r["dlnL_total"])
                                                         for r in one)),
                "min_abs_dlnL_pe_per_msun": float(min(abs(r["dlnL_pe"])
                                                      for r in one)),
                "min_abs_dlnL_selection_per_msun": float(
                    min(abs(r["dlnL_selection"]) for r in one)),
            },
        }
        blk["pass"] = bool(
            blk["one_msun_step"]["min_abs_dlnL_total_per_msun"]
            > TOL["liveness_min_abs_dlogL"]
            and blk["one_msun_step"]["min_abs_dlnL_pe_per_msun"]
            > TOL["liveness_min_abs_dlogL"]
            and blk["one_msun_step"]["min_abs_dlnL_selection_per_msun"]
            > TOL["liveness_min_abs_dlogL"])
        live_blocks.append(blk)
    res["checks"]["15_6_gaussian_mean_liveness"] = {
        "description": ("MANDATORY.  |dlnL| for a 1 Msun step in dmu_G, in the "
                        "PE term and the selection term SEPARATELY as well as "
                        "in the total; registered threshold 1e-3"),
        "dmu_G_nodes": list(MG_156),
        "blocks": live_blocks,
        "threshold": TOL["liveness_min_abs_dlogL"],
        "pass": bool(all(b["pass"] for b in live_blocks)),
    }

    # ------------------------------------------------------------- 15.7 ----
    b157 = []
    for f, mu, mg, tag in CELLS_157:
        a = _find(led, f, mu, mg)
        ref = k1["cells"][tag]
        blk = _cmp4(a, ref)
        blk.update({"f_agn": float(f), "dmu_chi": float(mu), "dmu_G": float(mg),
                    "reference": tag,
                    "reference_pins": k1["pins_applied"].get(tag, {}),
                    "reference_labels": k1["configs"][tag]["labels"],
                    "two_mark_logL": a["logL"], "reference_logL": ref["logL"]})
        b157.append(blk)
    # The two GAL K=1 builds ('old' and 'new' at the fiducial) against each other.
    gal_cross = _cmp4(k1["cells"]["K1_GAL"], k1["cells"]["K1_GAL_new"])
    w157 = _worst4(b157)
    exact = bool(all(all(b[k]["abs_diff"] == 0.0
                         for k in ("total", "pe", "selection", "log_mu"))
                     for b in b157))
    res["checks"]["15_7_k1_endpoints"] = {
        "description": ("f = 0 equals a K=1 GAL build; f = 1 equals a K=1 AGN "
                        "build whose pinned base block carries the AGN branch's "
                        "own mu_G and mu_chi.  Registered as EXACTLY 0.0."),
        "per_cell": b157,
        "worst": w157,
        "K1_GAL_old_vs_new_fiducial": gal_cross,
        "k1_build_seconds": k1["build_seconds"],
        "k1_configs": k1["configs"],
        "all_exactly_zero": exact,
        "pass": exact,
    }

    # ------------------------------------------------------------- cost ----
    sec = float(main["cost"]["steady_state_median_seconds"])
    n_cube = int(F_GRID.size * MU_GRID.size * MG_GRID.size)
    res["cost"] = dict(main["cost"])
    res["cost"]["projection"] = {
        "seconds_per_eval": sec,
        "cells": {"A10_S": int(F_GRID.size),
                  "A10_chi": int(F_GRID.size * MU_GRID.size),
                  "A10_M": int(F_GRID.size * MG_GRID.size),
                  "A10_J": n_cube},
        "gpu_hours": {"A10_S": F_GRID.size * sec / 3600.0,
                      "A10_chi": F_GRID.size * MU_GRID.size * sec / 3600.0,
                      "A10_M": F_GRID.size * MG_GRID.size * sec / 3600.0,
                      "A10_J": n_cube * sec / 3600.0},
        "A10_J_rows": int(F_GRID.size * MG_GRID.size),
        "A10_J_cells_per_row": int(MU_GRID.size),
        "A10_J_chunks": 8,
        "A10_J_hours_per_chunk_of_8": n_cube * sec / 8.0 / 3600.0,
        "A10_J_wall_hours_on_2_gpus": n_cube * sec / 2.0 / 3600.0,
        "A10_J_wall_hours_8_chunks_2_at_a_time": (n_cube * sec / 8.0 / 3600.0) * 4.0,
    }

    order = ["15_1_both_marks_zero", "15_2_mass_mark_zero", "15_3_spin_mark_zero",
             "15_4_f0_inert", "15_5_f1_live", "15_6_gaussian_mean_liveness",
             "15_7_k1_endpoints"]
    res["verdict_per_check"] = {k: bool(res["checks"][k]["pass"]) for k in order}
    res["verdict"] = "PASS" if all(res["verdict_per_check"].values()) else "FAIL"
    _write(DIAG / "a10_closure.json", res)
    _write_md(res)
    print(json.dumps({"verdict": res["verdict"], **res["verdict_per_check"]},
                     indent=2))
    return res


def _write_md(res):
    c = res["checks"]
    L = []
    A = L.append
    A("# Analysis 10 -- closure gate 15.1-15.7 (two-mark mock)\n")
    A(f"**Verdict: {res['verdict']}**  ({res['written_at_utc']})\n")
    A(f"- events `{Path(res['events_file']).name}`, md5 `{res['events_md5']}`")
    A(f"- 1 ULP at |lnL| = {res['logL_scale']['median_abs_logL']:.6g} is "
      f"{res['logL_scale']['one_ulp']:.6e}")
    A(f"- steady state {res['cost']['steady_state_median_seconds']:.4f} s/eval; "
      f"build {res['cost']['build_seconds_A10_two_mark']:.1f} s; peak GPU "
      f"{res['cost']['peak_gpu_nvidia_smi']['peak_mib_this_pid']:.0f} MiB\n")
    A("| check | deciding number | verdict |")
    A("|---|---|---|")
    w = c["15_1_both_marks_zero"]["worst"]
    A(f"| 15.1 both marks zero | worst {w['max_abs_diff_any_term']:.3e} abs "
      f"({w['max_ulp_any_term']:.2f} ULP) over 7 f nodes x 4 terms | "
      f"{'PASS' if c['15_1_both_marks_zero']['pass'] else 'FAIL'} |")
    w = c["15_2_mass_mark_zero"]["worst"]
    A(f"| 15.2 mass mark zero | worst {w['max_abs_diff_any_term']:.3e} abs "
      f"({w['max_ulp_any_term']:.2f} ULP) over 20 cells x 4 terms | "
      f"{'PASS' if c['15_2_mass_mark_zero']['pass'] else 'FAIL'} |")
    b = c["15_3_spin_mark_zero"]
    A(f"| 15.3 spin mark zero | min |dlnL| off zero "
      f"{b['min_abs_delta_logL_off_zero']:.4g}; f = 0 frozen "
      f"{b['f0_control']['all_frozen']} | {'PASS' if b['pass'] else 'FAIL'} |")
    b = c["15_4_f0_inert"]
    A(f"| 15.4 f = 0 inert | {b['bitwise']['logL_hex']['n_distinct']} distinct "
      f"total hex over {b['n_cells']} cells (want 1) | "
      f"{'PASS' if b['pass'] else 'FAIL'} |")
    b = c["15_5_f1_live"]
    A(f"| 15.5 f = 1 live | min spread {b['min_spread_logL_along_dmu_chi']:.4g} "
      f"(dmu_chi) / {b['min_spread_logL_along_dmu_G']:.4g} (dmu_G) | "
      f"{'PASS' if b['pass'] else 'FAIL'} |")
    b = c["15_6_gaussian_mean_liveness"]
    mins = [x["one_msun_step"] for x in b["blocks"]]
    A(f"| 15.6 mu_G liveness | min |dlnL| per 1 Msun: total "
      f"{min(m['min_abs_dlnL_total_per_msun'] for m in mins):.4g}, PE "
      f"{min(m['min_abs_dlnL_pe_per_msun'] for m in mins):.4g}, selection "
      f"{min(m['min_abs_dlnL_selection_per_msun'] for m in mins):.4g} "
      f"(> 1e-3) | {'PASS' if b['pass'] else 'FAIL'} |")
    b = c["15_7_k1_endpoints"]
    A(f"| 15.7 K=1 endpoints | max |diff| any term "
      f"{b['worst']['max_abs_diff_any_term']:.3e}; all exactly 0.0 "
      f"{b['all_exactly_zero']} | {'PASS' if b['pass'] else 'FAIL'} |")
    A("")
    p = res["cost"]["projection"]
    A("## Projected cost at the measured rate\n")
    A("| arm | cells | GPU-h |")
    A("|---|---|---|")
    for k in ("A10_S", "A10_chi", "A10_M", "A10_J"):
        A(f"| {k} | {p['cells'][k]:,} | {p['gpu_hours'][k]:.2f} |")
    A("")
    A(f"A10-J: {p['A10_J_rows']} rows of {p['A10_J_cells_per_row']} cells, "
      f"{p['A10_J_hours_per_chunk_of_8']:.2f} h per chunk of 8, "
      f"{p['A10_J_wall_hours_8_chunks_2_at_a_time']:.2f} h wall with two "
      f"workers at a time.")
    path = DIAG / "a10_closure.md"
    path.write_text("\n".join(L) + "\n")
    print(f"wrote {path}")
    return path


# =========================================================================== #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True,
                    choices=("closure", "k1refs", "assemble"))
    args = ap.parse_args(argv)
    return {"closure": stage_closure, "k1refs": stage_k1refs,
            "assemble": stage_assemble}[args.stage](args)


if __name__ == "__main__":
    main()
