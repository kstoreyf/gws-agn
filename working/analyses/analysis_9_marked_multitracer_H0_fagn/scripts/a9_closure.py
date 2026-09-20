#!/usr/bin/env python
"""Analysis 9 -- the three closure checks of specification section 9, plus a
measured cost model.

  9.1  FIXED-H0 REDUCTION.  With H0 pinned at 67.74 the Analysis-9 marked
       likelihood must reproduce Analysis 8's arm-J grid, cell for cell, in the
       TOTAL logL and separately in the PE and selection terms.  Reported in ULP
       of the returned value, because an absolute bound does not travel between
       problems.

  9.2  ZERO-MARK REDUCTION.  With dmu_chi = 0 the marked model must reduce to
       the spatial-only H0-f_AGN model.  'Spatial-only' is Analysis 8's own
       K2_OLD shape -- the SAME builder, the SAME data object, with no
       mu_chi_c2 coordinate in the parameter space at all (Analysis 2's
       configuration).  This is a reduction to a DIFFERENT model, not a re-read
       of the mu = 0 column of the same one.

  9.3  BRANCH ENDPOINTS.  At f_AGN = 0 the AGN spin parameter must be
       irrelevant: mu_chi_c2 must leave logL BITWISE unchanged.  At f_AGN = 1
       only the AGN branch remains, tested against a K=1 AGN reference built on
       the same events and injections.  Both at more than one H0, because H0 is
       now free.

  COST MODEL.  Build time, steady-state s/eval and peak GPU memory, MEASURED on
  the rita A100-80.

Nothing here is a production grid.  Nothing is regenerated.  Every write lands
under the Analysis-9 directory.

  sbatch --export=ALL,STAGE="closure k1refs" scripts/submit_a9_closure_rita.sbatch
  python scripts/a9_closure.py --stage closure_assemble      # CPU
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE.parent
DIAG = ANALYSIS_DIR / "diagnostics"
RESULTS = ANALYSIS_DIR / "results"

sys.dont_write_bytecode = True
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import a9_scan as A9                      # the Analysis-9 driver, imported

A8, GC = A9.A8, A9.GC
L_MU = A8.MU_CHI_C2_LABEL                 # '$\mu_\chi$_c2'
F_GRID, MU_GRID = A9.F_GRID, A9.MU_GRID   # Analysis 8's own arrays
H0_ANCHOR = A9.H0_ANCHOR                  # 67.74

A8_DIR = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/"
              "analysis_8_marked_multitracer_H0_fagn")
ARM_J = A8_DIR / "results" / "arm_J_joint.h5"
ARM_S = A8_DIR / "results" / "arm_S_spatial.h5"

# Recorded Analysis-8 headline numbers, for the record and for the checks.
A8_ARM_J_LOGL_MAX = -4211.842592845020
A8_ARM_J_MAP = (0.275, 0.1075)
A8_ARM_S_LOGL_MAX = -4233.939503109996
A8_ARM_S_MAP_F = 0.26

# The extra H0 values the multi-H0 checks use.  On the registered 0.5-step axis.
H0_LOW, H0_HIGH = 64.74, 71.74
H0_MULTI = (H0_LOW, H0_ANCHOR, H0_HIGH)
H0_WIDE = (60.24, 64.74, H0_ANCHOR, 71.74, 77.74)

TOL = {
    "ulp_same_code_same_data": 4.0,     # Analysis 8's Gate A measured 2 ULP
    "abs_logL_cross_hardware": 1.0e-6,  # arm J ran on an H100 NVL; this is an A100-80
    "endpoint_abs_logL": 1.0e-6,        # Analysis 8's A2 tolerance, verbatim
    "liveness_min_abs_dlogL": 1.0e-3,
}


# --------------------------------------------------------------------------- #
def _ulp(x):
    """One unit in the last place of |x| -- the scale a last-bit effect lives on."""
    v = abs(float(x))
    if not np.isfinite(v):
        return float("nan")
    return float(np.spacing(v))


def _cmp(a, b):
    """Compare two logL-like numbers, tolerating a shared -inf."""
    fa, fb = float(a), float(b)
    both_ninf = (fa == -np.inf) and (fb == -np.inf)
    if both_ninf:
        return {"a": fa, "b": fb, "diff": 0.0, "abs_diff": 0.0, "ulp": float("nan"),
                "diff_in_ulp": 0.0, "bitwise_identical": True, "both_neg_inf": True}
    d = fa - fb
    u = _ulp(fb)
    return {
        "a": fa, "b": fb, "diff": float(d), "abs_diff": float(abs(d)),
        "ulp": u, "diff_in_ulp": float(d / u) if (u and np.isfinite(u)) else float("nan"),
        "bitwise_identical": bool(fa == fb),
        "both_neg_inf": False,
    }


class GpuMemoryPoller(threading.Thread):
    """Peak GPU memory as the driver sees it, not as JAX reports it."""

    def __init__(self, period=0.25):
        super().__init__(daemon=True)
        self.period, self.peak_mib, self._halt = period, 0.0, threading.Event()
        self.peak_any_mib, self.matched = 0.0, False

    def run(self):
        while not self._halt.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout
                mine = str(os.getpid())
                total = 0.0
                for line in out.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) != 2:
                        continue
                    try:
                        used = float(parts[1])
                    except ValueError:
                        continue
                    total += used
                    if parts[0] == mine:
                        self.peak_mib = max(self.peak_mib, used)
                        self.matched = True
                self.peak_any_mib = max(self.peak_any_mib, total)
            except Exception:
                pass
            self._halt.wait(self.period)

    def stop(self):
        try:
            self._halt.set()
            self.join(timeout=5)
        except Exception as exc:                                # never kill a run
            print(f"  [warn] GPU memory poller shutdown: {exc!r}")
        return {"peak_mib_this_pid": self.peak_mib,
                "peak_mib_all_compute_apps": self.peak_any_mib,
                "pid_matched_in_nvidia_smi": self.matched}


def _jax_mem():
    try:
        import jax
        st = jax.devices()[0].memory_stats() or {}
        return {k: int(v) for k, v in st.items() if isinstance(v, (int, float))}
    except Exception as exc:                                   # pragma: no cover
        return {"error": repr(exc)}


# --------------------------------------------------------------------------- #
# The evaluation ledger: every cell computed once, keyed on its coordinates.
# --------------------------------------------------------------------------- #
class Ledger:
    def __init__(self, cell, tag):
        self.cell, self.tag, self.store, self.order = cell, tag, {}, []

    @staticmethod
    def key(h0, f, mu):
        return "{:.10g}|{:.10g}|{:.10g}".format(float(h0), float(f), float(mu))

    def get(self, h0, f, mu=None, note="", force=False):
        k = self.key(h0, f, 0.0 if mu is None else mu)
        if k in self.store and not force:
            return self.store[k]
        over = {"H0": float(h0), "fcat_2": float(f)}
        if mu is not None:
            over[L_MU] = float(mu)
        r = self.cell.evaluate(**over)
        # GC._cell_record drops logL_hex; every bitwise test here needs it.
        rec = {"config": self.tag, "H0": float(h0), "f_agn": float(f),
               "mu_chi_c2": (None if mu is None else float(mu)),
               "dmu_chi": (None if mu is None else float(mu)),
               "note": note, "logL_hex": r["logL_hex"],
               "coord": {lbl: float(v) for lbl, v in r["coord"].items()},
               **GC._cell_record(r)}
        self.store[k] = rec
        self.order.append(k)
        print(f"    [{self.tag}] H0={h0:<7.3f} f={f:<6.4g} "
              f"mu={'--' if mu is None else format(mu, '+.4f')}  "
              f"logL={rec['logL']!r}  {r['seconds']:.3f}s  {note}")
        sys.stdout.flush()
        return rec

    def rows(self):
        return [self.store[k] for k in self.order]

    def seconds(self, skip_first=1):
        s = [self.store[k]["seconds"] for k in self.order]
        return s[skip_first:]


# --------------------------------------------------------------------------- #
# The cell subsets -- registered here, before the run.
# --------------------------------------------------------------------------- #
def cells_91():
    """A representative subset of the 2501 arm-J cells (indices into A8's grids).

    Registered before the run: the MAP cell, the whole f = 0 and f = 1 rows at
    five mu nodes each, four cells straddling the recorded guard boundary (a
    rejected cell and its accepted neighbour, three times over), and an interior
    scatter drawn from a fixed seed.
    """
    i_map = int(np.argmin(np.abs(F_GRID - A8_ARM_J_MAP[0])))
    j_map = int(np.argmin(np.abs(MU_GRID - A8_ARM_J_MAP[1])))
    picks = [(i_map, j_map, "arm-J MAP")]
    for j in (0, 15, 30, j_map, 60):
        picks.append((0, j, "f = 0 row"))
        picks.append((F_GRID.size - 1, j, "f = 1 row"))
    # The guard boundary, from the recorded arm-J mask: each rejected cell
    # paired with its accepted neighbour, so the mask is tested from both sides.
    for i, j, note in ((40, 58, "guard boundary: A8 rejected"),
                       (40, 57, "guard boundary: A8 accepted, one mu node below"),
                       (37, 58, "guard boundary: A8 rejected, lowest f at this mu"),
                       (36, 58, "guard boundary: A8 accepted, one f node below"),
                       (30, 60, "guard boundary: A8 rejected, lowest f at mu = +0.25"),
                       (29, 60, "guard boundary: A8 accepted, one f node below")):
        picks.append((i, j, note))
    rng = np.random.default_rng(9)                     # fixed: the subset is registered
    seen = {(i, j) for i, j, _ in picks}
    while len(picks) < 40:
        i = int(rng.integers(1, F_GRID.size - 1))
        j = int(rng.integers(0, MU_GRID.size))
        if (i, j) in seen:
            continue
        seen.add((i, j))
        picks.append((i, j, "interior scatter"))
    return picks


F_91_2 = (0.0, 0.25, 0.26, 0.30, 0.50, 0.75, 1.0)      # 0.26 is arm S's MAP node
MU_93 = (-0.20, -0.05, 0.0, 0.1075, 0.25)


# =========================================================================== #
# Stage: closure  (GPU, rita)
# =========================================================================== #
def stage_closure(args):
    env = A9._gpu_setup("closure")
    poller = GpuMemoryPoller()
    poller.start()

    t0 = time.time()
    cell, surveys = A9.build_cell()
    build_new_s = time.time() - t0
    print(f"\n[cost] K2_NEW build (load_all_data + parameter space + "
          f"make_likelihood): {build_new_s:.2f} s")

    t0 = time.time()
    old = A8.build("A9_SPATIAL_ONLY", "old", surveys, data=cell.data,
                   gw_path=A8.GW_PATH_MARKED)
    build_old_s = time.time() - t0
    print(f"[cost] spatial-only build on the SAME data object: {build_old_s:.2f} s")
    if L_MU in old.labels:
        raise SystemExit("[fatal] the spatial-only reference still carries a "
                         f"mu_chi_c2 coordinate: {old.labels}")
    if "fcat_2" not in old.labels or "H0" not in old.labels:
        raise SystemExit(f"[fatal] spatial-only labels are wrong: {old.labels}")

    led_new, led_old = Ledger(cell, "K2_NEW"), Ledger(old, "SPATIAL")
    mem_after_build = _jax_mem()

    # ---------------------------------------------------------------- 9.1 ----
    print("\n== 9.1  fixed-H0 reduction against Analysis 8 arm J ==")
    picks = cells_91()
    c91 = []
    for i, j, note in picks:
        f, mu = float(F_GRID[i]), float(MU_GRID[j])
        rec = led_new.get(H0_ANCHOR, f, mu, note)
        c91.append({"i": i, "j": j, "f_agn": f, "mu_chi_c2": mu, "dmu_chi": mu,
                    "note": note, "cell": rec})

    # ---------------------------------------------------------------- 9.2 ----
    print("\n== 9.2  zero-mark reduction to the spatial-only model ==")
    c92 = []
    for h0 in H0_WIDE:
        for f in F_91_2:
            rn = led_new.get(h0, f, 0.0, "marked model at dmu_chi = 0")
            ro = led_old.get(h0, f, None, "spatial-only model")
            c92.append({"H0": h0, "f_agn": f, "new": rn, "old": ro})

    # ---------------------------------------------------------------- 9.3 ----
    print("\n== 9.3  branch endpoints, at more than one H0 ==")
    c93 = []
    for h0 in H0_MULTI:
        for f in (0.0, 1.0):
            for mu in MU_93:
                c93.append({"H0": h0, "f_agn": f, "mu_chi_c2": mu, "dmu_chi": mu,
                            "cell": led_new.get(h0, f, mu,
                                                f"endpoint f = {f:g}")})

    # H0 liveness at the endpoints and at the MAP -- the trap Analysis 8 caught.
    print("\n== H0 liveness ==")
    live = []
    ref = led_new.get(H0_ANCHOR, A8_ARM_J_MAP[0], A8_ARM_J_MAP[1], "arm-J MAP")
    for h0 in (H0_ANCHOR + 0.5, H0_ANCHOR - 0.5, 60.24, 77.74):
        r = led_new.get(h0, A8_ARM_J_MAP[0], A8_ARM_J_MAP[1], "H0 liveness")
        live.append({"H0": h0, "logL": r["logL"],
                     "delta_logL_vs_anchor": r["logL"] - ref["logL"]})
        print(f"    dlogL(H0={h0}) = {live[-1]['delta_logL_vs_anchor']:+.6f}")

    # --------------------------------------------------------- cost model ----
    print("\n== cost model: steady state ==")
    burn = [led_new.get(H0_ANCHOR, 0.5, float(MU_GRID[k]), "steady-state timer",
                        force=True)
            for k in range(20, 40)]
    steady = [r["seconds"] for r in burn]
    all_s = led_new.seconds(skip_first=1)
    mem_peak = _jax_mem()
    peak = poller.stop()

    out = {
        "stage": "closure",
        "environment": env,
        "gpu_name": subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip(),
        "configs": {
            "K2_NEW": {"labels": cell.labels, "mode": cell.mode,
                       "per_catalog_pop_params": list(cell.per_catalog_pop_params),
                       "n_fixed_parameter_values": len(cell.fixed_parameter_values),
                       "survey_paths": cell.survey_paths,
                       "events_file": str(cell.opts.gw_path),
                       "selection_file": str(cell.opts.gwselection_path)},
            "SPATIAL": {"labels": old.labels, "mode": old.mode,
                        "per_catalog_pop_params": list(old.per_catalog_pop_params),
                        "n_fixed_parameter_values": len(old.fixed_parameter_values),
                        "fix_population": bool(old.opts.fix_population),
                        "survey_paths": old.survey_paths,
                        "events_file": str(old.opts.gw_path),
                        "selection_file": str(old.opts.gwselection_path)},
        },
        "data": {"nEvents": int(cell.data["nEvents"]),
                 "nsamp": int(cell.data["nsamp"]),
                 "Ndraw": float(cell.data["Ndraw"])},
        "cost": {
            "build_seconds_K2_NEW": build_new_s,
            "build_seconds_spatial_only_same_data": build_old_s,
            "first_eval_seconds": led_new.rows()[0]["seconds"],
            "steady_state_median_seconds": float(np.median(steady)),
            "steady_state_mean_seconds": float(np.mean(steady)),
            "steady_state_min_seconds": float(np.min(steady)),
            "steady_state_max_seconds": float(np.max(steady)),
            "steady_state_sd_seconds": float(np.std(steady)),
            "steady_state_n": len(steady),
            "all_evals_median_seconds": float(np.median(all_s)),
            "n_evaluations_K2_NEW": len(led_new.order),
            "n_evaluations_spatial": len(led_old.order),
            "jax_memory_stats_after_build": mem_after_build,
            "jax_memory_stats_peak": mem_peak,
            "peak_gpu_nvidia_smi": peak,
        },
        "checks": {"c91_cells": c91, "c92_pairs": c92, "c93_cells": c93,
                   "h0_liveness": {"reference": ref, "cells": live,
                                   "min_abs_delta_logL": float(min(
                                       abs(x["delta_logL_vs_anchor"]) for x in live))}},
        "ledger_K2_NEW": led_new.rows(),
        "ledger_SPATIAL": led_old.rows(),
        "truth": dict(A9.TRUTH),
        "tolerances": TOL,
    }
    A9._write(DIAG / "_a9_closure_main.json", out)
    print(f"\n[cost] build {build_new_s:.1f} s | steady state "
          f"{out['cost']['steady_state_median_seconds']:.4f} s/eval | peak GPU "
          f"{peak['peak_mib_this_pid']:.0f} MiB (this pid) / "
          f"{peak['peak_mib_all_compute_apps']:.0f} MiB (all apps)")
    return out


# =========================================================================== #
# Stage: k1refs  (GPU, rita) -- the single-branch references for 9.3
# =========================================================================== #
def stage_k1refs(args):
    env = A9._gpu_setup("k1refs")
    poller = GpuMemoryPoller()
    poller.start()
    out = {"stage": "k1refs", "environment": env, "configs": {}, "cells": {},
           "build_seconds": {}}
    for tag, survey in (("K1_GAL", A8.SURVEY_GAL), ("K1_AGN", A8.SURVEY_AGN)):
        t0 = time.time()
        cell = A8.build(tag, "old", [survey], gw_path=A8.GW_PATH_MARKED)
        out["build_seconds"][tag] = time.time() - t0
        out["configs"][tag] = {
            "mode": "old", "labels": cell.labels,
            "per_catalog_pop_params": list(cell.per_catalog_pop_params),
            "fix_population": bool(cell.opts.fix_population),
            "survey_paths": cell.survey_paths,
            "events_file": str(cell.opts.gw_path),
            "selection_file": str(cell.opts.gwselection_path),
            "nEvents": int(cell.data["nEvents"]), "Ndraw": float(cell.data["Ndraw"]),
        }
        out["cells"][tag] = {}
        for h0 in H0_MULTI:
            r = cell.evaluate(H0=float(h0))
            out["cells"][tag][f"{h0:.10g}"] = {"H0": float(h0),
                                               "logL_hex": r["logL_hex"],
                                               **GC._cell_record(r)}
            print(f"    [{tag}] H0={h0:<7.3f} logL={r['logL']!r} "
                  f"({r['seconds']:.3f}s)")
            sys.stdout.flush()
        del cell
        import gc
        gc.collect()
    out["peak_gpu_nvidia_smi"] = poller.stop()
    A9._write(DIAG / "_a9_closure_k1.json", out)
    return out


# =========================================================================== #
# Stage: closure_assemble  (CPU)
# =========================================================================== #
def _read_arm(path, keys):
    import h5py
    with h5py.File(path, "r") as h:
        d = {k: np.asarray(h[k]) for k in keys}
        d["attrs"] = {k: (v.decode() if isinstance(v, bytes) else v)
                      for k, v in h.attrs.items()}
    return d


def _grid_proposal(sec_per_eval):
    """The S9 / J9 arithmetic, from the MEASURED rate.

    S9 is Analysis 2's own joint (H0, f) grid shape -- [50, 100] x 201, step
    0.25 -- plus the single node 67.74, so the Analysis-8 anchor is ON the grid
    and every J9 node is a strict S9 node.  J9 takes every SECOND S9 node over
    the H0 window, plus the anchor, and keeps Analysis 8's own f and mu_chi_c2
    axes untouched so the fixed-H0 comparison of section 11 is resolution-
    matched in all three coordinates.

    The H0 window [63, 76] is set by containment, not taste: on Analysis 2's
    measured seed-100 H0 marginal (unmarked, same catalogs, same selection) the
    density at 63 is 7.67e-08 of the peak and at 76 is 9.16e-08, both more than
    a decade below the Gate-B3 criterion of 1e-6, with 2.08e-08 of the mass
    outside.  The [64, 76] window sits at 3.95e-06 of the peak at its low edge
    -- above B3's own threshold -- so the extra 1.0 in H0 is what the gate
    requires, not padding.  S9 measures this axis on the real marked data and
    the window is confirmed against S9's own marginal before J9 is committed.
    """
    s9_h0 = np.round(np.linspace(50.0, 100.0, 201), 10)          # A2's own shape
    s9_nodes = np.sort(np.unique(np.concatenate([s9_h0, [H0_ANCHOR]])))
    j9_core = np.round(np.arange(63.0, 76.0 + 1e-9, 0.5), 10)    # every 2nd S9 node
    j9_nodes = np.sort(np.unique(np.concatenate([j9_core, [H0_ANCHOR]])))
    subset = bool(np.all([np.any(np.isclose(s9_nodes, x, rtol=0, atol=1e-9))
                          for x in j9_nodes]))
    n_f, n_mu = int(F_GRID.size), int(MU_GRID.size)
    s9_cells = int(s9_nodes.size * n_f)
    j9_cells = int(j9_nodes.size * n_f * n_mu)

    def hours(n):
        return n * sec_per_eval / 3600.0

    # What a mu trim justified by the Analysis-8 SELECTION diagnostics would buy:
    # the hard N_eff guard rejects at mu >= +0.2350 for f >= 0.750, and the
    # posterior mass behind that corner is bounded by 3.86e-104.  Dropping those
    # three nodes is the only trim specification 6 licenses; it is 5% of the mu
    # axis.  dmu_chi = 0 stays inside any range: figure 3 compares against it.
    n_mu_trim = int((MU_GRID < 0.2350 - 1e-9).sum())
    j9_cells_trim = int(j9_nodes.size * n_f * n_mu_trim)

    return {
        "seconds_per_eval_used": sec_per_eval,
        "S9": {
            "shape": "2-D (H0, f_AGN) at dmu_chi = 0",
            "H0_nodes": int(s9_nodes.size),
            "H0_range": [float(s9_nodes.min()), float(s9_nodes.max())],
            "H0_step": 0.25,
            "H0_note": ("Analysis 2's own joint grid [50, 100] x 201 (step 0.25) "
                        "plus the single node 67.74, so the Analysis-8 anchor is "
                        "on the grid and the J9 nodes stay a strict subset.  Full "
                        "registered production range: it DEMONSTRATES where the "
                        "H0 posterior mass lies instead of assuming it."),
            "f_nodes": n_f, "f_range": [0.0, 1.0], "f_step": 0.025,
            "cells": s9_cells,
            "gpu_hours": hours(s9_cells),
            "wall_hours_1_gpu": hours(s9_cells),
            "wall_hours_2_gpu": hours(s9_cells) / 2.0,
        },
        "J9": {
            "shape": "3-D (H0, f_AGN, mu_chi_c2)",
            "H0_nodes": int(j9_nodes.size),
            "H0_range": [float(j9_nodes.min()), float(j9_nodes.max())],
            "H0_step": 0.5,
            "H0_is_subset_of_S9": subset,
            "H0_nodes_list": [float(x) for x in j9_nodes],
            "H0_window_evidence": {
                "source": ("analysis_2/results/joint_s100.h5 -- the SAME "
                           "realisation, unmarked, one shared population"),
                "median": 69.217036, "ci68": [68.249133, 70.191466],
                "ci90": [67.599776, 70.836534],
                "ci999": [65.922882, 72.594041],
                "mass_outside_63_76": 2.076e-08,
                "edge_density_over_peak_at_63": 7.666e-08,
                "edge_density_over_peak_at_76": 9.164e-08,
                "edge_density_over_peak_at_64": 3.952e-06,
                "gate_B3_criterion": 1.0e-6,
                "why_not_64": ("[64, 76] leaves the low edge at 3.95e-06 of the "
                               "peak, ABOVE the B3 criterion; 63 is where the "
                               "measured marginal clears it with a decade of "
                               "margin"),
                "confirm_against": ("S9's own marked marginal, before J9 is "
                                    "committed"),
            },
            "f_nodes": n_f, "f_range": [0.0, 1.0], "f_step": 0.025,
            "mu_nodes": n_mu, "mu_range": [float(MU_GRID.min()), float(MU_GRID.max())],
            "mu_step": 0.0075,
            "resolution_vs_A8_widths": {
                "A8_ci68_width_f": 0.0921,
                "nodes_across_A8_ci68_f": 0.0921 / 0.025 + 1.0,
                "A8_ci68_width_dmu": 0.0394,
                "nodes_across_A8_ci68_dmu": 0.0394 / 0.0075 + 1.0,
                "A2_ci68_width_H0": 1.942,
                "nodes_across_A2_ci68_H0_at_step_0p5": 1.942 / 0.5 + 1.0,
                "note": ("freeing H0 can only widen f and dmu_chi, so these node "
                         "counts are lower bounds; H0 at step 0.5 is the thinnest "
                         "axis, which is why S9 resolves it at 0.25 and the "
                         "section-11 width comparison is made on the shared "
                         "subset"),
            },
            "cells": j9_cells,
            "gpu_hours": hours(j9_cells),
            "wall_hours_1_gpu": hours(j9_cells),
            "wall_hours_2_gpu": hours(j9_cells) / 2.0,
            "mu_trim_if_needed": {
                "justification": ("Analysis-8 SELECTION diagnostics only: the "
                                  "hard N_eff guard rejects at mu_chi_c2 >= "
                                  "+0.2350 for f >= 0.750 (23 of 2501 arm-J "
                                  "cells) and the posterior mass behind that "
                                  "corner is bounded by 3.86e-104"),
                "mu_nodes_kept": n_mu_trim,
                "mu_range_kept": [float(MU_GRID.min()), float(MU_GRID[n_mu_trim - 1])],
                "cells": j9_cells_trim,
                "gpu_hours": hours(j9_cells_trim),
                "saving_gpu_hours": hours(j9_cells) - hours(j9_cells_trim),
                "dmu_chi_zero_inside": True,
                "taken": bool(hours(j9_cells) > 80.0),
            },
        },
        "total_cells": s9_cells + j9_cells,
        "total_gpu_hours": hours(s9_cells + j9_cells),
        "total_wall_hours_1_gpu": hours(s9_cells + j9_cells),
        "total_wall_hours_2_gpu": hours(s9_cells + j9_cells) / 2.0,
        "preregistered_trim_threshold_gpu_hours": 80.0,
        "preregistered_trim_fires": bool(hours(j9_cells) > 80.0),
        "anchor_slab_cells": n_f * n_mu,
        "anchor_slab_gpu_hours": hours(n_f * n_mu),
        "anchor_slab_note": ("the H0 = 67.74 slab is cell for cell Analysis 8's "
                             "arm-J grid, so the full equivalence gate is 1/28 "
                             "of J9 and costs nothing extra.  Dropping 67.74 "
                             "from both grids would save this much and force the "
                             "gate to be run standalone."),
    }


def stage_closure_assemble(args):
    A9._cpu_setup("closure_assemble", import_darksirens=False)
    main = json.loads((DIAG / "_a9_closure_main.json").read_text())
    k1 = json.loads((DIAG / "_a9_closure_k1.json").read_text())

    J = _read_arm(ARM_J, ["f_grid", "mu_chi_c2_grid", "log_likelihood",
                          "guard/logL_pe", "guard/logL_selection",
                          "guard/log_mu", "guard/Neff", "guard/rejected"])
    S = _read_arm(ARM_S, ["f_grid", "log_likelihood", "guard/logL_pe",
                          "guard/logL_selection", "guard/rejected"])

    res = {"stage": "closure_assemble", "tolerances": TOL,
           "sources": {"arm_J": str(ARM_J), "arm_S": str(ARM_S),
                       "closure_main": str(DIAG / "_a9_closure_main.json"),
                       "closure_k1": str(DIAG / "_a9_closure_k1.json")},
           "cost": main["cost"], "gpu_name": main.get("gpu_name"),
           "configs": main["configs"], "checks": {}}

    # ---------------------------------------------------------------- 9.1 ----
    rows, worst = [], {"total": 0.0, "pe": 0.0, "sel": 0.0}
    worst_ulp = {"total": 0.0, "pe": 0.0, "sel": 0.0}
    n_bitwise, n_ninf = 0, 0
    mask_ok = True
    for c in main["checks"]["c91_cells"]:
        i, j, cel = c["i"], c["j"], c["cell"]
        ref_tot = float(J["log_likelihood"][i, j])
        ref_pe = float(J["guard/logL_pe"][i, j])
        ref_sel = float(J["guard/logL_selection"][i, j])
        rej_ref = bool(J["guard/rejected"][i, j])
        rej_new = (not cel["finite"])
        mask_ok = mask_ok and (rej_ref == rej_new)
        ct = _cmp(cel["logL"], ref_tot)
        cp = _cmp(cel.get("logL_pe", np.nan), ref_pe)
        cs = _cmp(cel.get("logL_selection", np.nan), ref_sel)
        n_bitwise += int(ct["bitwise_identical"])
        n_ninf += int(ct["both_neg_inf"])
        for key, blk in (("total", ct), ("pe", cp), ("sel", cs)):
            if np.isfinite(blk["abs_diff"]):
                worst[key] = max(worst[key], blk["abs_diff"])
            if np.isfinite(blk["diff_in_ulp"]):
                worst_ulp[key] = max(worst_ulp[key], abs(blk["diff_in_ulp"]))
        rows.append({"i": i, "j": j, "f_agn": c["f_agn"],
                     "mu_chi_c2": c["mu_chi_c2"], "dmu_chi": c["dmu_chi"],
                     "note": c["note"], "a9_logL_hex": cel["logL_hex"],
                     "rejected_a8": rej_ref, "rejected_a9": rej_new,
                     "total": ct, "pe": cp, "selection": cs,
                     "Neff_a9": cel.get("Neff"),
                     "Neff_a8": float(J["guard/Neff"][i, j]),
                     "log_mu": _cmp(cel.get("log_mu", np.nan),
                                    float(J["guard/log_mu"][i, j]))})
    map_i = int(np.argmin(np.abs(np.asarray(J["f_grid"]) - A8_ARM_J_MAP[0])))
    map_j = int(np.argmin(np.abs(np.asarray(J["mu_chi_c2_grid"]) - A8_ARM_J_MAP[1])))
    map_row = [r for r in rows if (r["i"], r["j"]) == (map_i, map_j)][0]
    c91 = {
        "description": ("the Analysis-9 marked likelihood with H0 pinned at 67.74, "
                        "against the recorded arm-J grid, cell for cell"),
        "n_cells": len(rows),
        "n_bitwise_identical": n_bitwise,
        "n_both_neg_inf": n_ninf,
        "max_abs_diff_total": worst["total"],
        "max_abs_diff_pe": worst["pe"],
        "max_abs_diff_selection": worst["sel"],
        "max_abs_diff_in_ulp_total": worst_ulp["total"],
        "max_abs_diff_in_ulp_pe": worst_ulp["pe"],
        "max_abs_diff_in_ulp_selection": worst_ulp["sel"],
        "ulp_at_the_seed100_scale": _ulp(A8_ARM_J_LOGL_MAX),
        "rejected_mask_matches": bool(mask_ok),
        "map_cell": {"f_agn": A8_ARM_J_MAP[0], "mu_chi_c2": A8_ARM_J_MAP[1],
                     "a8_recorded_logL_max": A8_ARM_J_LOGL_MAX,
                     "a8_grid_value": map_row["total"]["b"],
                     "a9_value": map_row["total"]["a"],
                     "diff": map_row["total"]["diff"],
                     "diff_in_ulp": map_row["total"]["diff_in_ulp"]},
        "per_cell": rows,
        "pass": bool(mask_ok
                     and worst["total"] <= TOL["abs_logL_cross_hardware"]
                     and worst["pe"] <= TOL["abs_logL_cross_hardware"]
                     and worst["sel"] <= TOL["abs_logL_cross_hardware"]),
        "pass_same_hardware_ulp_criterion": bool(
            max(worst_ulp.values()) <= TOL["ulp_same_code_same_data"]),
    }
    res["checks"]["c91_fixed_H0_reduction"] = c91

    # ---------------------------------------------------------------- 9.2 ----
    pairs, w2, w2u = [], {"total": 0.0, "pe": 0.0, "sel": 0.0}, 0.0
    nb2 = 0
    for p in main["checks"]["c92_pairs"]:
        n, o = p["new"], p["old"]
        ct = _cmp(n["logL"], o["logL"])
        cp = _cmp(n.get("logL_pe", np.nan), o.get("logL_pe", np.nan))
        cs = _cmp(n.get("logL_selection", np.nan), o.get("logL_selection", np.nan))
        nb2 += int(ct["bitwise_identical"])
        for key, blk in (("total", ct), ("pe", cp), ("sel", cs)):
            if np.isfinite(blk["abs_diff"]):
                w2[key] = max(w2[key], blk["abs_diff"])
        if np.isfinite(ct["diff_in_ulp"]):
            w2u = max(w2u, abs(ct["diff_in_ulp"]))
        pairs.append({"H0": p["H0"], "f_agn": p["f_agn"],
                      "marked_dmu0_logL_hex": n["logL_hex"],
                      "spatial_only_logL_hex": o["logL_hex"],
                      "total": ct, "pe": cp, "selection": cs,
                      "log_mu": _cmp(n.get("log_mu", np.nan),
                                     o.get("log_mu", np.nan))})
    # cross-check against the recorded arm S (the fixed-H0 dmu = 0 line)
    sf = np.asarray(S["f_grid"])
    armS = []
    for p in main["checks"]["c92_pairs"]:
        if abs(p["H0"] - H0_ANCHOR) > 1e-12:
            continue
        k = int(np.argmin(np.abs(sf - p["f_agn"])))
        if abs(sf[k] - p["f_agn"]) > 1e-9:
            continue
        armS.append({"f_agn": p["f_agn"], "arm_S_index": k,
                     "marked_dmu0_vs_armS": _cmp(p["new"]["logL"],
                                                 float(S["log_likelihood"][k])),
                     "spatial_only_vs_armS": _cmp(p["old"]["logL"],
                                                  float(S["log_likelihood"][k]))})
    armS_worst = max([abs(a["marked_dmu0_vs_armS"]["abs_diff"]) for a in armS]
                     + [abs(a["spatial_only_vs_armS"]["abs_diff"]) for a in armS])
    armS_worst_ulp = max(
        [abs(a["marked_dmu0_vs_armS"]["diff_in_ulp"]) for a in armS]
        + [abs(a["spatial_only_vs_armS"]["diff_in_ulp"]) for a in armS])
    res["checks"]["c92_zero_mark_reduction"] = {
        "description": ("the marked model at dmu_chi = 0 against the spatial-only "
                        "H0-f_AGN model -- Analysis 8's K2_OLD shape, the same "
                        "builder and the same data object, with NO mu_chi_c2 "
                        "coordinate in the parameter space at all"),
        "spatial_only_labels": main["configs"]["SPATIAL"]["labels"],
        "marked_labels": main["configs"]["K2_NEW"]["labels"],
        "n_pairs": len(pairs), "n_bitwise_identical": nb2,
        "H0_values": list(H0_WIDE), "f_values": list(F_91_2),
        "max_abs_diff_total": w2["total"], "max_abs_diff_pe": w2["pe"],
        "max_abs_diff_selection": w2["sel"], "max_abs_diff_in_ulp_total": w2u,
        "per_pair": pairs,
        "arm_S_crosscheck": {
            "description": ("the H0 = 67.74 line of both models against the "
                            "recorded arm_S_spatial.h5"),
            "a8_recorded_logL_max": A8_ARM_S_LOGL_MAX,
            "a8_recorded_map_f": A8_ARM_S_MAP_F,
            "max_abs_diff": armS_worst, "max_abs_diff_in_ulp": armS_worst_ulp,
            "per_f": armS},
        "pass": bool(w2["total"] <= TOL["abs_logL_cross_hardware"]
                     and w2["pe"] <= TOL["abs_logL_cross_hardware"]
                     and w2["sel"] <= TOL["abs_logL_cross_hardware"]
                     and armS_worst <= TOL["abs_logL_cross_hardware"]),
    }

    # ---------------------------------------------------------------- 9.3 ----
    by = {}
    for c in main["checks"]["c93_cells"]:
        by.setdefault((c["H0"], c["f_agn"]), []).append(c)
    f0, f1 = [], []
    for (h0, f), cs in sorted(by.items()):
        cs = sorted(cs, key=lambda c: c["mu_chi_c2"])
        hexes = [c["cell"]["logL_hex"] for c in cs]
        vals = [c["cell"]["logL"] for c in cs]
        blk = {"H0": h0, "f_agn": f,
               "mu_chi_c2": [c["mu_chi_c2"] for c in cs],
               "logL": vals, "logL_hex": hexes,
               "n_distinct_hex": len(set(hexes)),
               "bitwise_frozen": bool(len(set(hexes)) == 1),
               "spread_logL": float(np.nanmax([v for v in vals if np.isfinite(v)])
                                    - np.nanmin([v for v in vals if np.isfinite(v)]))
               if any(np.isfinite(v) for v in vals) else float("nan"),
               "pe_hex": [c["cell"].get("logL_pe") for c in cs],
               "selection_hex": [c["cell"].get("logL_selection") for c in cs],
               "n_distinct_pe": len({repr(c["cell"].get("logL_pe")) for c in cs}),
               "n_distinct_selection": len({repr(c["cell"].get("logL_selection"))
                                            for c in cs}),
               "n_rejected": int(sum(1 for c in cs if not c["cell"]["finite"]))}
        (f0 if f == 0.0 else f1).append(blk)

    k1cmp = []
    for h0 in H0_MULTI:
        key = f"{h0:.10g}"
        for f, tag in ((0.0, "K1_GAL"), (1.0, "K1_AGN")):
            ref = k1["cells"][tag][key]
            # the K=2 cell at mu_chi_c2 = 0, the point where the two shapes agree
            cand = [c for c in main["checks"]["c93_cells"]
                    if c["H0"] == h0 and c["f_agn"] == f and c["mu_chi_c2"] == 0.0]
            if not cand:
                continue
            c = cand[0]["cell"]
            k1cmp.append({
                "H0": h0, "f_agn": f, "reference": tag,
                "total": _cmp(c["logL"], ref["logL"]),
                "pe": _cmp(c.get("logL_pe", np.nan), ref.get("logL_pe", np.nan)),
                "selection": _cmp(c.get("logL_selection", np.nan),
                                  ref.get("logL_selection", np.nan)),
                "log_mu": _cmp(c.get("log_mu", np.nan), ref.get("log_mu", np.nan)),
            })
    k1_worst = max(max(abs(b[k]["abs_diff"]) for k in ("total", "pe", "selection"))
                   for b in k1cmp) if k1cmp else float("nan")
    res["checks"]["c93_branch_endpoints"] = {
        "description": ("f = 0: mu_chi_c2 must be inert, bitwise.  f = 1: only "
                        "the AGN branch remains, against a K=1 AGN reference "
                        "built on the same events and injections.  Both at "
                        f"H0 = {list(H0_MULTI)}."),
        "mu_nodes_probed": list(MU_93),
        "f0_rows": f0,
        "f1_rows": f1,
        "f0_all_bitwise_frozen": bool(all(b["bitwise_frozen"] for b in f0)),
        "f1_any_bitwise_frozen": bool(any(b["bitwise_frozen"] for b in f1)),
        "f1_min_spread_logL": float(min(b["spread_logL"] for b in f1))
        if f1 else float("nan"),
        "k1_reference_comparison": k1cmp,
        "k1_max_abs_diff_any_term": k1_worst,
        "k1_build_seconds": k1["build_seconds"],
        "pass": bool(all(b["bitwise_frozen"] for b in f0)
                     and not any(b["bitwise_frozen"] for b in f1)
                     and np.isfinite(k1_worst)
                     and k1_worst <= TOL["endpoint_abs_logL"]),
    }

    # ------------------------------------------------------------ liveness ---
    lv = main["checks"]["h0_liveness"]
    res["checks"]["h0_liveness"] = {
        "description": ("H0 was already a label in the Analysis-8 parameter "
                        "space and was never varied; an equivalence test passes "
                        "for free if the new coordinate is ignored."),
        "reference": {k: lv["reference"][k] for k in ("H0", "f_agn", "mu_chi_c2",
                                                      "logL")},
        "cells": lv["cells"], "min_abs_delta_logL": lv["min_abs_delta_logL"],
        "pass": bool(lv["min_abs_delta_logL"] > TOL["liveness_min_abs_dlogL"]),
    }

    # -------------------------------------------------------- grid proposal --
    sec = float(main["cost"]["steady_state_median_seconds"])
    res["grid_proposal"] = _grid_proposal(sec)

    # Analysis-8 evidence for the mu range, read off arm J itself.
    llm = np.where(np.isfinite(J["log_likelihood"]), J["log_likelihood"], -np.inf)
    P = np.where(np.isfinite(llm), np.exp(llm - llm[np.isfinite(llm)].max()), 0.0)
    m_mu = np.trapz(P, np.asarray(J["f_grid"]), axis=0)
    m_f = np.trapz(P, np.asarray(J["mu_chi_c2_grid"]), axis=1)
    rej = np.asarray(J["guard/rejected"])
    fg, mg = np.asarray(J["f_grid"]), np.asarray(J["mu_chi_c2_grid"])
    res["a8_range_evidence"] = {
        "mu_marginal_edge_density_over_peak": [float(m_mu[0] / m_mu.max()),
                                               float(m_mu[-1] / m_mu.max())],
        "f_marginal_edge_density_over_peak": [float(m_f[0] / m_f.max()),
                                              float(m_f[-1] / m_f.max())],
        "n_rejected_cells": int(rej.sum()),
        "rejected_min_f": float(fg[rej.any(axis=1)].min()) if rej.any() else None,
        "rejected_min_mu": float(mg[rej.any(axis=0)].min()) if rej.any() else None,
        "note": ("the guard-rejected corner is where a mu trim would be "
                 "justified by SELECTION diagnostics; dmu_chi = 0 must stay "
                 "inside any restricted range because figure 3 compares to it"),
    }

    verdict = all(res["checks"][k]["pass"] for k in res["checks"])
    res["verdict"] = "PASS" if verdict else "STOP"
    A9._write(DIAG / "closure_checks.json", res)
    A9._write(RESULTS / "a9_closure_summary.json", {
        "verdict": res["verdict"],
        "cost": res["cost"],
        "gpu_name": res.get("gpu_name"),
        "c91": {k: v for k, v in res["checks"]["c91_fixed_H0_reduction"].items()
                if k != "per_cell"},
        "c92": {k: v for k, v in res["checks"]["c92_zero_mark_reduction"].items()
                if k != "per_pair"},
        "c93": res["checks"]["c93_branch_endpoints"],
        "h0_liveness": res["checks"]["h0_liveness"],
        "grid_proposal": res["grid_proposal"],
        "a8_range_evidence": res["a8_range_evidence"],
    })
    print(json.dumps({"verdict": res["verdict"],
                      "c91": {k: res["checks"]["c91_fixed_H0_reduction"][k]
                              for k in ("n_cells", "n_bitwise_identical",
                                        "max_abs_diff_total",
                                        "max_abs_diff_in_ulp_total",
                                        "rejected_mask_matches", "pass")},
                      "c92": {k: res["checks"]["c92_zero_mark_reduction"][k]
                              for k in ("n_pairs", "n_bitwise_identical",
                                        "max_abs_diff_total", "pass")},
                      "c93": {k: res["checks"]["c93_branch_endpoints"][k]
                              for k in ("f0_all_bitwise_frozen",
                                        "f1_any_bitwise_frozen",
                                        "k1_max_abs_diff_any_term", "pass")},
                      "cost": res["cost"],
                      "grid_proposal": res["grid_proposal"]},
                     indent=2, default=GC._json_default))
    return res


# =========================================================================== #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=("closure", "k1refs", "closure_assemble"))
    args = ap.parse_args(argv)
    return {"closure": stage_closure, "k1refs": stage_k1refs,
            "closure_assemble": stage_closure_assemble}[args.stage](args)


if __name__ == "__main__":
    main()
