#!/usr/bin/env python
"""Per-merger decomposition of the host-class-unknown (mixture) likelihood against the host-class-known one, on CPU.

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= <core venv python> tests/mixture_decomposition.py --realization 103 --nside 32
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= <core venv python> tests/mixture_decomposition.py --realization 103 --nside 64
    <core venv or jax env python> tests/mixture_decomposition.py --realization 103 --analyze
        -> tests/mixture_decomposition_runs/r<R>_ns<N>.npz (per-merger terms, resumable), tests/results/mixture_decomposition_<R>.json

The model is the experiment's two_tracers_one_population grid of the set peak_f0p30_s0 (both complete catalogs, f_AGN free,
the galaxy-class population for both catalogs; E1 in runs/RUNLIST_experiment.yaml) at its own Monte Carlo settings (8000
redrawn samples per merger, the set's injection lane, the run's redraw stream): the driver's own configuration, resolved by
prod_driver.parse_args/resolve from the run list's arguments (nside 64: stage X's realization_NNN_nside64 entry, the same
arguments plus --nside 64), built and bound by prod_driver.setup_model (prod_model.Model: import_core, build, bind), and
evaluated by prod_model.Model.evaluate_events, which runs core's field_mixture_log_likelihood with return_diagnostics=True
(the program of every grid cell) and returns, besides ln L, pe_var, n_eff, log_mu and sel_corr, every merger's ln evidence and
Monte Carlo variance. Nothing in scripts/ is modified; the modules are only imported.

Core's mixture weight of a posterior sample or an injection is linear in f: w(f) = (1 - f) w_gal + f w_agn, each branch
normalised over its own catalog (complete catalogs: Z_k = N_obs,k). Hence, per merger i and per H0,
    G_i = merger i's evidence at f = 0 (galaxy catalog),  A_i = at f = 1 (AGN catalog),
    L_i(f) = (1 - f) G_i + f A_i,   mu(f) = (1 - f) mu_gal + f mu_agn,
and the per-merger mixture identity is checked against a direct evaluation at f = f0 (default 0.30). G_i, A_i are what
hosts_known uses for galaxy-hosted (galaxy catalog) and AGN-hosted (AGN catalog) mergers (experiment check E: the mixture at
f = 0 / 1 is the one-catalog analysis; with equal populations the class populations are the same); mu_gal, mu_agn are its
subsets' selection integrals.

Per H0 (default 64 to 72 in steps of 0.5, plus 67.74), at f0:
  S_mix,AGN = sum_{AGN-hosted} ln[f A_i + (1 - f) G_i],  S_mix,gal = the same over galaxy-hosted;
  S_known,AGN = sum_{AGN-hosted} ln A_i,  S_known,gal = sum_{galaxy-hosted} ln G_i;
  selection: mixture -N ln mu(f), known -N_AGN ln mu_agn - N_gal ln mu_gal; their Monte Carlo terms
  N (N + 3) / (2 N_eff) (mixture: from the direct evaluation at f0) and sum_c N_c (N_c + 3) / (2 N_eff,c) (known), and the
  half-variance corrections (sum sigma_i^2 / 2 - N / (2 N_eff); known: per class).
Checks: the direct evaluation at f0 against the stored grid of the nside-32 run at the cells it shares (f = f0, H0 on both
grids; ln L, pe_var, n_eff, log_mu), and the galaxy-hosted factor against hosts_known's stored lnL_gal (8000 samples there too).
The AGN-hosted factor of hosts_known is at 32000 samples per merger (--agn-subset-nsamp): --known-agn evaluates that subset at
its own settings (prod_model.HostsKnownModel from the hosts_known run's arguments, subs['agn'].evaluate_events) per H0, compared
with the stored lnL_agn and with the same mergers' ln A_i at 8000 samples (block known_agn_at_32000).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

A16 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(A16 / "scripts"))
import prod_common as P  # noqa: E402

RUNS = P.RUNS
OUT_DIR = A16 / "tests" / "mixture_decomposition_runs"
RESULTS = A16 / "tests" / "results"
SET = "peak_f0p30_s0"
MIX = "two_tracers_one_population/both_catalogs/complete/{set}/realization_{R:03d}"
KNOWN = "hosts_known/both_catalogs/complete/{set}/realization_{R:03d}_agn_s32000_inj1"
H0_IN = 67.74
H0_EXTRA = (67.74,)
H0_R = (67.74, 69.5)          # the H0 at which the distribution of the AGN share is reported
SLOPE_WINDOW = (66.0, 70.0)   # per-merger slopes: least squares over the nodes in this window


def run_name(R, nside, which="mix"):
    base = (MIX if which == "mix" else KNOWN).format(set=SET, R=R)
    if nside == P.SURVEY_NSIDE:
        return base
    if which == "mix":
        return base + f"_nside{nside}"
    return base.replace("_agn_s32000_inj1", f"_nside{nside}_agn_s32000_inj1")


def runlist_args(name):
    import yaml
    for f in ("RUNLIST_experiment.yaml", "RUNLIST.yaml"):
        doc = yaml.safe_load((RUNS / f).read_text())
        for st in doc["stages"]:
            for r in st["runs"]:
                if r["name"] == name:
                    return list(r["args"]), f
    raise SystemExit(f"[fatal] {name} is in neither run list")


def h0_grid(spec, extra=H0_EXTRA):
    lo, hi, step = (float(x) for x in spec.split(","))
    g = np.round(lo + step * np.arange(int(round((hi - lo) / step)) + 1), 10)
    return np.unique(np.round(np.concatenate([g, np.asarray(extra, float)]), 10))


# ============================================================================ compute (core venv)
def compute(a):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import prod_driver as PD
    name = run_name(a.realization, a.nside)
    args, src = runlist_args(name)
    argv = args + (["--limit-events", str(a.limit_events)] if a.limit_events else [])
    cfg = PD.resolve(PD.parse_args(argv))
    if cfg["run_name"] != name:
        raise SystemExit(f"[fatal] resolved run name {cfg['run_name']} != {name}")
    stored = RUNS / name / "config.json"
    cfg_check = None
    if stored.exists() and not a.limit_events:
        sc = json.loads(stored.read_text())
        keys = ("events_file", "injections_files", "surveys", "nsamp", "redraw_key", "free", "population_values",
                "class_populations", "half_variance_correction", "max_var", "pe_event_block", "sel_batch",
                "catalog_evaluation", "zmax", "completeness_model", "Om0", "planted_f_agn")
        diff = {k: (sc.get(k), cfg.get(k)) for k in keys if sc.get(k) != cfg.get(k)}
        cfg_check = {"stored_config": str(stored), "keys_compared": list(keys), "differences": diff}
        if diff:
            raise SystemExit(f"[fatal] resolved configuration differs from the stored run's: {diff}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"r{a.realization}_ns{a.nside}" + (f"_lim{a.limit_events}" if a.limit_events else "")
    path = OUT_DIR / f"{tag}.npz"
    H = h0_grid(a.h0)
    f0 = float(a.f)
    mix_at = H if a.mix_at == "all" else np.asarray([h for h in H if h in mix_nodes(a.realization, a.nside, f0, H)])
    t0 = time.time()
    m = PD.setup_model(cfg)
    t_setup = time.time() - t0
    n = m.n_events
    import prod_model as PM
    ht = PM.event_host_types(cfg["events_file"], cfg.get("limit_events"))
    if ht.shape != (n,):
        raise SystemExit(f"[fatal] {ht.shape} host types for {n} mergers")
    scal = ("lnL", "pe_var", "n_eff", "log_mu", "sel_corr")
    rec = {"H0": H, "f0": f0, "host_type": ht}
    for tag_f in ("f0", "f1", "fm"):
        rec[f"ev_{tag_f}"] = np.full((len(H), n), np.nan)
        rec[f"var_{tag_f}"] = np.full((len(H), n), np.nan)
        for k in scal:
            rec[f"{k}_{tag_f}"] = np.full(len(H), np.nan)
    rec["seconds"] = np.full((len(H), 3), np.nan)
    if path.exists():
        z = np.load(path)
        if np.array_equal(z["H0"], H) and float(z["f0"]) == f0:
            for k in rec:
                if k in z.files:
                    rec[k] = np.array(z[k])
            print(f"[mixdec] resuming {path}: {int(np.isfinite(rec['seconds'][:, 0]).sum())}/{len(H)} H0 done", flush=True)
    info = {"run": name, "args_from": src, "argv": argv, "events_file": cfg["events_file"], "n_events": n, "nsamp": m.nsamp, "n_draw": m.n_draw,
            "n_agn_hosted": int(ht.sum()), "n_gal_hosted": int((ht == 0).sum()), "setup_seconds": t_setup,
            "config_check": cfg_check, "labels": m.labels, "model_info": {k: m.info.get(k) for k in (
                "core_commit", "core_worktree_dirty", "zmax_grid", "n_detected_injections", "catalog_model", "events_samples")}}
    for j, h in enumerate(H):
        todo = [(0, "f0", 0.0), (1, "f1", 1.0)] + ([(2, "fm", f0)] if h in mix_at else [])
        for col, tf, fv in todo:
            if np.isfinite(rec["seconds"][j, col]):
                continue
            tc = time.time()
            r = m.evaluate_events({"H0": float(h), "f_agn": fv})
            rec["seconds"][j, col] = time.time() - tc
            rec[f"ev_{tf}"][j] = r["event_log_evidence"]
            rec[f"var_{tf}"][j] = r["event_mc_variance"]
            for k in scal:
                rec[f"{k}_{tf}"][j] = r[k]
            np.savez(path, **rec, info=json.dumps(info, default=str))
            print(f"[mixdec] nside {a.nside} H0 {h:g} f {fv:g}: lnL {r['lnL']:.4f} log_mu {r['log_mu']:.6f} "
                  f"n_eff {r['n_eff']:.0f} ({rec['seconds'][j, col]:.0f} s)", flush=True)
    np.savez(path, **rec, info=json.dumps(info, default=str))
    print(f"[mixdec] -> {path} ({time.time() - t0:.0f} s)", flush=True)


def compute_known_agn(a):
    """--known-agn: the AGN-hosted subset of the hosts_known run of the set (its own Monte Carlo settings: 32000 samples per
    merger), per H0: every AGN-hosted merger's ln evidence with the AGN catalog alone (prod_model.HostsKnownModel, built and bound
    by prod_driver.setup_model from the run list's arguments; subs['agn'].evaluate_events). A one-catalog model does not divide by
    the catalog's size (core: with one catalog Z_1 cancels and is not evaluated), so its ln A_i and ln mu_agn exceed the mixture's
    by ln N_obs,AGN, a constant that cancels in -N ln mu + sum ln A_i and in every slope."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import prod_driver as PD
    name = run_name(a.realization, a.nside, "known")
    args, src = runlist_args(name)
    cfg = PD.resolve(PD.parse_args(args))
    if cfg["run_name"] != name:
        raise SystemExit(f"[fatal] resolved run name {cfg['run_name']} != {name}")
    path = OUT_DIR / f"r{a.realization}_ns{a.nside}_knownagn.npz"
    H = h0_grid(a.h0)
    t0 = time.time()
    m = PD.setup_model(cfg)
    sub = m.subs["agn"]
    n = sub.n_events
    scal = ("lnL", "pe_var", "n_eff", "log_mu", "sel_corr")
    rec = {"H0": H, "ev": np.full((len(H), n), np.nan), "var": np.full((len(H), n), np.nan), "seconds": np.full(len(H), np.nan),
           "agn_mask": m.masks["agn"]}
    for k in scal:
        rec[k] = np.full(len(H), np.nan)
    if path.exists():
        z = np.load(path)
        if np.array_equal(z["H0"], H):
            rec = {k: np.array(z[k]) for k in rec}
    info = {"run": name, "args_from": src, "nsamp_agn_subset": sub.nsamp, "n_agn_hosted": n, "setup_seconds": time.time() - t0}
    for j, h in enumerate(H):
        if np.isfinite(rec["seconds"][j]):
            continue
        tc = time.time()
        r = sub.evaluate_events({"H0": float(h)})
        rec["seconds"][j] = time.time() - tc
        rec["ev"][j], rec["var"][j] = r["event_log_evidence"], r["event_mc_variance"]
        for k in scal:
            rec[k][j] = r[k]
        np.savez(path, **rec, info=json.dumps(info, default=str))
        print(f"[mixdec known-agn] nside {a.nside} H0 {h:g}: lnL_agn {r['lnL']:.4f} log_mu {r['log_mu']:.6f} n_eff {r['n_eff']:.0f} "
              f"({rec['seconds'][j]:.0f} s)", flush=True)
    np.savez(path, **rec, info=json.dumps(info, default=str))
    print(f"[mixdec known-agn] -> {path} ({time.time() - t0:.0f} s)", flush=True)


def _known_agn_rows(R, nside, H, ht):
    """The --known-agn arrays on the nodes H (None unless every node is there and the AGN-hosted mergers agree)."""
    p = OUT_DIR / f"r{R}_ns{nside}_knownagn.npz"
    if not p.exists():
        return None
    z = dict(np.load(p, allow_pickle=False))
    if not np.array_equal(z["agn_mask"].astype(bool), ht == 1):
        return None
    idx = [int(np.argmin(np.abs(z["H0"] - h))) for h in H]
    if not all(abs(z["H0"][i] - h) < 1e-9 and np.isfinite(z["seconds"][i]) for i, h in zip(idx, H)):
        return None
    return {k: np.asarray(z[k])[idx] for k in ("ev", "var", "lnL", "pe_var", "n_eff", "log_mu", "sel_corr")}


def known_agn_block(R, nside, H, lA8, ht):
    """The AGN-hosted factor of hosts_known at its own 32000 samples against the same mergers' ln A_i at 8000 (the mixture
    model's), and against the stored hosts_known lnL_agn."""
    import h5py
    p = OUT_DIR / f"r{R}_ns{nside}_knownagn.npz"
    if not p.exists():
        return None
    z = dict(np.load(p, allow_pickle=False))
    info = json.loads(str(z.pop("info")))
    if not np.array_equal(z["agn_mask"].astype(bool), ht == 1):
        return {"error": "the AGN-hosted mergers differ from the mixture run's"}
    out = {"info": info}
    idx = [int(np.argmin(np.abs(z["H0"] - h))) for h in H]
    ok = np.array([abs(z["H0"][i] - h) < 1e-9 and np.isfinite(z["seconds"][i]) for i, h in zip(idx, H)])
    if not ok.any():
        return {"info": info, "error": "no common H0"}
    Hc = H[ok]
    ev32 = z["ev"][np.asarray(idx)[ok]]
    s32 = ev32.sum(1)
    s8 = lA8[ok][:, ht == 1].sum(1)
    NA = int((ht == 1).sum())
    lmu32 = z["log_mu"][np.asarray(idx)[ok]]
    ne32 = z["n_eff"][np.asarray(idx)[ok]]
    j0 = int(np.argmin(np.abs(Hc - H0_IN)))
    out["H0"] = Hc.tolist()
    out["S_known_AGN_32000_minus_67.74"] = (s32 - s32[j0]).tolist()
    out["S_known_AGN_8000_minus_67.74"] = (s8 - s8[j0]).tolist()
    out["S32000_minus_S8000_minus_67.74"] = ((s32 - s8) - (s32 - s8)[j0]).tolist()
    out["pe_var_32000"] = z["pe_var"][np.asarray(idx)[ok]].tolist()
    out["sel_known_AGN_32000_minus_67.74"] = (-NA * (lmu32 - lmu32[j0])).tolist()
    out["mc_sel_known_AGN_32000"] = (NA * (NA + 3) / (2 * ne32)).tolist()
    for k, y in (("S_known_AGN_32000", s32), ("S_known_AGN_8000", s8), ("S32000_minus_S8000", s32 - s8)):
        out[f"slope_at_67.74_{k}"] = local_slope(Hc, y, H0_IN)
        out[f"slope_at_69.5_{k}"] = local_slope(Hc, y, 69.5)
    gk = RUNS / run_name(R, nside, "known") / "grid"
    rows = []
    for nm in ("fine", "fine_widened", "coarse"):
        fp = gk / f"{nm}.h5"
        if not fp.exists():
            continue
        with h5py.File(fp, "r") as f:
            ha = np.asarray(f["axis_H0"])
            lnl_agn = np.asarray(f["lnL_agn"])
        for i, h in zip(np.asarray(idx)[ok], Hc):
            ih = np.where(np.abs(ha - h) < 1e-9)[0]
            if ih.size:
                rows.append({"pass": nm, "H0": float(h), "lnL_agn_stored": float(lnl_agn[ih[0]]),
                             "lnL_agn_here": float(z["lnL"][i]), "diff": float(z["lnL"][i] - lnl_agn[ih[0]])})
    out["check_vs_stored_lnL_agn"] = {"n_cells": len(rows), "max_abs_diff": max((abs(r["diff"]) for r in rows), default=None),
                                      "rows": rows}
    return out


def mix_nodes(R, nside, f0, H):
    """The H0 of H at which the direct evaluation at f0 is made with --mix-at matching: the stored grid's nodes (fine and
    coarse) at f = f0, else every node."""
    import h5py
    g = RUNS / run_name(R, nside) / "grid"
    out = set()
    for nm in ("fine", "fine_widened", "coarse"):
        p = g / f"{nm}.h5"
        if not p.exists():
            continue
        with h5py.File(p, "r") as f:
            ha, fa = np.asarray(f["axis_H0"]), np.asarray(f["axis_f_agn"])
        if np.any(np.abs(fa - f0) < 1e-9):
            out |= {h for h in H if np.any(np.abs(ha - h) < 1e-9)}
    return out or set(H)


# ============================================================================ analysis (any python with numpy, h5py)
def _lse(a, b):
    return np.logaddexp(a, b)


def peak_of(x, y):
    """H0 of the maximum of y on the nodes x, refined by a parabola through the three nodes around it; at an edge the slope
    there (per km/s/Mpc) instead."""
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    i = int(np.argmax(y))
    lin = np.polyfit(x, y, 1)[0]
    if 0 < i < len(x) - 1:
        c = np.polyfit(x[i - 1:i + 2], y[i - 1:i + 2], 2)
        xp = -c[1] / (2 * c[0]) if c[0] < 0 else x[i]
        return {"peak_H0": float(xp), "at_edge": False, "slope_over_grid": float(lin)}
    return {"peak_H0": None, "at_edge": "low" if i == 0 else "high",
            "edge_slope": float((y[1] - y[0]) / (x[1] - x[0]) if i == 0 else (y[-1] - y[-2]) / (x[-1] - x[-2])),
            "slope_over_grid": float(lin)}


def local_slope(x, y, x0):
    """dy/dx at x0 from a quadratic through the five nodes nearest x0."""
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    idx = np.argsort(np.abs(x - x0))[:5]
    c = np.polyfit(x[idx], y[idx], 2)
    return float(2 * c[0] * x0 + c[1])


def gauss_fit(x, lnl):
    """Peak and sd of exp(lnl) on the nodes: moments of the normalised density (trapezoid)."""
    ok = np.isfinite(lnl)
    x, l = x[ok], lnl[ok]
    p = np.exp(l - l.max())
    nrm = np.trapz(p, x)
    mu = float(np.trapz(p * x, x) / nrm)
    sd = float(np.sqrt(np.trapz(p * (x - mu) ** 2, x) / nrm))
    return {"mean": mu, "sd": sd, "max_node": float(x[int(np.argmax(l))]), "parabola_peak": peak_of(x, l)["peak_H0"]}


def window_slopes(H, D, lo, hi):
    """Least-squares slope of every column of D (H0 nodes x mergers) over the nodes in [lo, hi]."""
    sel = (H >= lo - 1e-9) & (H <= hi + 1e-9)
    x = H[sel] - H[sel].mean()
    return (x[:, None] * (D[sel] - D[sel].mean(0))).sum(0) / (x ** 2).sum()


def quantiles(v):
    q = np.quantile(v, (0.05, 0.25, 0.5, 0.75, 0.95, 0.99))
    return {"n": int(v.size), "mean": float(v.mean()), "q05": float(q[0]), "q25": float(q[1]), "median": float(q[2]),
            "q75": float(q[3]), "q95": float(q[4]), "q99": float(q[5]), "frac_gt_0.1": float((v > 0.1).mean()),
            "frac_gt_0.5": float((v > 0.5).mean()), "frac_gt_0.9": float((v > 0.9).mean()),
            "hist_edges": [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
            "hist_counts": np.histogram(v, bins=[0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0 + 1e-12])[0].tolist()}


def stored_checks(R, nside, z, f0):
    """The direct evaluation at f0 against the stored mixture grid at shared cells; the galaxy-hosted factor against
    hosts_known's stored lnL_gal at shared H0."""
    import h5py
    out = {"mixture": [], "hosts_known_gal": []}
    H = z["H0"]
    g = RUNS / run_name(R, nside) / "grid"
    for nm in ("fine", "fine_widened", "coarse"):
        p = g / f"{nm}.h5"
        if not p.exists():
            continue
        with h5py.File(p, "r") as f:
            ha, fa = np.asarray(f["axis_H0"]), np.asarray(f["axis_f_agn"])
            d = {k: np.asarray(f[k]) for k in ("lnL", "pe_var", "n_eff", "log_mu", "sel_corr")}
        jf = np.where(np.abs(fa - f0) < 1e-9)[0]
        if not jf.size:
            continue
        for j, h in enumerate(H):
            ih = np.where(np.abs(ha - h) < 1e-9)[0]
            if not ih.size or not np.isfinite(z["lnL_fm"][j]):
                continue
            row = {"pass": nm, "H0": float(h), "f": f0}
            for k in d:
                row[f"{k}_stored"] = float(d[k][ih[0], jf[0]])
                row[f"{k}_here"] = float(z[f"{k}_fm"][j])
                row[f"{k}_diff"] = row[f"{k}_here"] - row[f"{k}_stored"]
            out["mixture"].append(row)
    gk = RUNS / run_name(R, nside, "known") / "grid"
    ht = z["host_type"]
    ng = int((ht == 0).sum())
    for nm in ("fine", "fine_widened", "coarse"):
        p = gk / f"{nm}.h5"
        if not p.exists():
            continue
        with h5py.File(p, "r") as f:
            ha = np.asarray(f["axis_H0"])
            d = {k: np.asarray(f[k]) for k in ("lnL_gal", "log_mu_gal", "n_eff_gal", "pe_var_gal")}
        for j, h in enumerate(H):
            ih = np.where(np.abs(ha - h) < 1e-9)[0]
            if not ih.size or not np.isfinite(z["seconds"][j, 0]):
                continue
            here = (float(np.sum(z["ev_f0"][j][ht == 0])) - ng * float(z["log_mu_f0"][j])
                    + ng * (ng + 3) / (2 * float(z["n_eff_f0"][j])))
            out["hosts_known_gal"].append({
                "pass": nm, "H0": float(h), "lnL_gal_stored": float(d["lnL_gal"][ih[0]]), "lnL_gal_here": here,
                "diff": here - float(d["lnL_gal"][ih[0]]),
                "log_mu_gal_diff": float(z["log_mu_f0"][j]) - float(d["log_mu_gal"][ih[0]]),
                "n_eff_gal_rel_diff": float(z["n_eff_f0"][j]) / float(d["n_eff_gal"][ih[0]]) - 1,
                "pe_var_gal_diff": float(np.sum(z["var_f0"][j][ht == 0])) - float(d["pe_var_gal"][ih[0]])})
    for k in ("mixture", "hosts_known_gal"):
        rows = out[k]
        key = "lnL_diff" if k == "mixture" else "diff"
        out[f"{k}_max_abs_diff"] = max((abs(r[key]) for r in rows), default=None)
        out[f"{k}_n_cells"] = len(rows)
    return out


def analyze_one(R, nside, limit=None):
    p = OUT_DIR / (f"r{R}_ns{nside}" + (f"_lim{limit}" if limit else "") + ".npz")
    if not p.exists():
        return None
    z = dict(np.load(p, allow_pickle=False))
    info = json.loads(str(z.pop("info")))
    H = z["H0"]
    f = float(z["f0"])
    ht = z["host_type"].astype(int)
    done = np.isfinite(z["seconds"][:, 0]) & np.isfinite(z["seconds"][:, 1])
    if not done.all():
        info["incomplete"] = f"{int(done.sum())}/{len(H)} H0 nodes have f = 0 and f = 1"
    H = H[done]
    lG, lA = z["ev_f0"][done], z["ev_f1"][done]
    vG, vA = z["var_f0"][done], z["var_f1"][done]
    lmG, lmA = z["log_mu_f0"][done], z["log_mu_f1"][done]
    neG, neA = z["n_eff_f0"][done], z["n_eff_f1"][done]
    n = lG.shape[1]
    a, g = ht == 1, ht == 0
    NA, NG = int(a.sum()), int(g.sum())
    lmix = _lse(math.log(f) + lA, math.log1p(-f) + lG)            # ln[f A + (1 - f) G]
    r = np.exp(math.log(f) + lA - lmix)                            # share of the AGN branch
    lmu_mix = _lse(math.log(f) + lmA, math.log1p(-f) + lmG)
    T = {"H0": H.tolist()}
    T["S_mix_AGN"] = lmix[:, a].sum(1)
    T["S_mix_gal"] = lmix[:, g].sum(1)
    T["S_known_AGN"] = lA[:, a].sum(1)
    T["S_known_gal"] = lG[:, g].sum(1)
    T["sel_mix"] = -n * lmu_mix                                    # -N ln mu(f)
    T["sel_known"] = -NA * lmA - NG * lmG                          # -N_AGN ln mu_agn - N_gal ln mu_gal
    T["sel_known_AGN"] = -NA * lmA
    T["sel_known_gal"] = -NG * lmG
    T["ln_mu_gal"], T["ln_mu_agn"], T["ln_mu_mix"] = lmG, lmA, lmu_mix
    # Monte Carlo terms: the mixture's from the direct evaluation at f0 (NaN where not made); known: per class
    fm = np.isfinite(z["seconds"][:, 2])[done]
    neM = np.where(fm, z["n_eff_fm"][done], np.nan)
    pvM = np.where(fm, z["pe_var_fm"][done], np.nan)
    T["mc_sel_mix"] = n * (n + 3) / (2 * neM)
    T["mc_sel_known"] = NA * (NA + 3) / (2 * neA) + NG * (NG + 3) / (2 * neG)
    T["halfvar_mix"] = 0.5 * pvM - 0.5 * n / neM
    T["halfvar_known"] = 0.5 * (vA[:, a].sum(1) + vG[:, g].sum(1)) - 0.5 * NA / neA - 0.5 * NG / neG
    T["lnL_mix"] = T["S_mix_AGN"] + T["S_mix_gal"] + T["sel_mix"] + T["mc_sel_mix"]
    T["lnL_known"] = T["S_known_AGN"] + T["S_known_gal"] + T["sel_known"] + T["mc_sel_known"]
    T["lnL_mix_corrected"] = T["lnL_mix"] + T["halfvar_mix"]
    T["lnL_known_corrected"] = T["lnL_known"] + T["halfvar_known"]
    # without the Monte Carlo terms (every node): the sums and the selection integrals alone
    T["core_mix"] = T["S_mix_AGN"] + T["S_mix_gal"] + T["sel_mix"]
    T["core_known"] = T["S_known_AGN"] + T["S_known_gal"] + T["sel_known"]
    D = {"d_AGN": T["S_mix_AGN"] - T["S_known_AGN"], "d_gal": T["S_mix_gal"] - T["S_known_gal"],
         "d_sel": T["sel_mix"] - T["sel_known"]}
    D["d_core"] = D["d_AGN"] + D["d_gal"] + D["d_sel"]
    D["d_mc_sel"] = T["mc_sel_mix"] - T["mc_sel_known"]
    D["d_halfvar"] = T["halfvar_mix"] - T["halfvar_known"]
    T.update(D)
    # the known analysis at hosts_known's own settings: its AGN-hosted factor at 32000 samples (--known-agn), the galaxy-hosted
    # factor as above (8000 there too); a one-catalog model's ln A_i and ln mu_agn carry + ln N_obs,AGN (a constant)
    k32 = _known_agn_rows(R, nside, H, ht)
    has32 = k32 is not None
    if has32:
        T["S_known_AGN_32000"] = k32["ev"][:, :].sum(1)
        T["sel_known_AGN_32000"] = -NA * k32["log_mu"]
        T["mc_sel_known_AGN_32000"] = NA * (NA + 3) / (2 * k32["n_eff"])
        T["core_known_32000"] = T["S_known_AGN_32000"] + T["S_known_gal"] + T["sel_known_AGN_32000"] + T["sel_known_gal"]
        T["lnL_known_32000_corrected"] = (T["core_known_32000"] + T["mc_sel_known_AGN_32000"] + NG * (NG + 3) / (2 * neG)
                                         + 0.5 * (k32["pe_var"] + vG[:, g].sum(1)) - 0.5 * NA / k32["n_eff"] - 0.5 * NG / neG)
        T["d_mc_AGN_32000_minus_8000"] = T["S_known_AGN_32000"] - T["S_known_AGN"]
        T["d_AGN_vs_32000"] = T["S_mix_AGN"] - T["S_known_AGN_32000"]
        T["d_sel_vs_32000"] = T["sel_mix"] - T["sel_known_gal"] - T["sel_known_AGN_32000"]
        T["d_core_vs_32000"] = T["core_mix"] - T["core_known_32000"]
    j0 = int(np.argmin(np.abs(H - H0_IN)))
    rel = {k: (np.asarray(v) - np.asarray(v)[j0]).tolist() for k, v in T.items() if k != "H0" and not k.startswith("ln_mu")}
    # peaks and slopes
    pieces = ("S_mix_AGN", "S_mix_gal", "S_known_AGN", "S_known_gal", "sel_mix", "sel_known", "sel_known_AGN", "sel_known_gal",
              "core_mix", "core_known", "lnL_mix", "lnL_known", "lnL_mix_corrected", "lnL_known_corrected",
              "d_AGN", "d_gal", "d_sel", "d_core", "d_mc_sel", "d_halfvar", "S_known_AGN_32000", "sel_known_AGN_32000",
              "core_known_32000", "lnL_known_32000_corrected", "d_mc_AGN_32000_minus_8000", "d_AGN_vs_32000", "d_sel_vs_32000",
              "d_core_vs_32000")
    shape = {}
    for k in pieces:
        if k not in T:
            continue
        y = np.asarray(T[k], float)
        if np.isfinite(y).sum() < 3:
            shape[k] = {"n_nodes": int(np.isfinite(y).sum())}
            continue
        s = peak_of(H, y)
        s["slope_at_67.74"] = local_slope(H, y, H0_IN)
        s["slope_at_69.5"] = local_slope(H, y, 69.5)
        s["n_nodes"] = int(np.isfinite(y).sum())
        shape[k] = s
    post = {k: gauss_fit(H, np.asarray(T[k], float)) for k in ("core_mix", "core_known", "lnL_mix_corrected", "lnL_known_corrected",
                                                              "core_known_32000", "lnL_known_32000_corrected")
            if k in T and np.isfinite(T[k]).sum() >= 5}
    # from the known analysis to the mixture one piece at a time (posterior means on the nodes; each step adds one difference to
    # the corrected ln L of the step before; the last step is the mixture's corrected ln L exactly), and each piece alone added
    # to the known analysis at 8000 samples
    steps = None
    if has32 and np.isfinite(T["lnL_mix_corrected"]).all():
        seq = [("known, hosts_known's settings (AGN-hosted at 32000 samples)", np.asarray(T["lnL_known_32000_corrected"], float)),
               ("AGN-hosted at 8000 samples (the mixture's), its Monte Carlo terms", np.asarray(T["lnL_known_corrected"], float))]
        for k, lbl in (("d_AGN", "+ AGN-hosted mergers: ln[f + (1 - f) G/A]"), ("d_gal", "+ galaxy-hosted mergers: ln[(1 - f) + f A/G]"),
                       ("d_sel", "+ selection: -N ln mu(f) + N_AGN ln mu_agn + N_gal ln mu_gal"),
                       ("d_mc", "+ Monte Carlo terms (N(N+3)/2N_eff, half-variance) of the mixture minus the known")):
            add = (np.asarray(T["d_mc_sel"], float) + np.asarray(T["d_halfvar"], float)) if k == "d_mc" else np.asarray(T[k], float)
            seq.append((lbl, seq[-1][1] + add))
        steps = {"cumulative": [], "alone_on_known_8000": {}}
        prev = None
        for lbl, l in seq:
            g_ = gauss_fit(H, l)
            steps["cumulative"].append({"step": lbl, "mean": g_["mean"], "sd": g_["sd"],
                                        "change": None if prev is None else g_["mean"] - prev})
            prev = g_["mean"]
        base = np.asarray(T["lnL_known_corrected"], float)
        m0 = gauss_fit(H, base)["mean"]
        for k in ("d_AGN", "d_gal", "d_sel"):
            steps["alone_on_known_8000"][k] = gauss_fit(H, base + np.asarray(T[k], float))["mean"] - m0
        steps["end_minus_mixture_mean"] = prev - gauss_fit(H, np.asarray(T["lnL_mix_corrected"], float))["mean"]
    # the shift each piece of the difference would give the known analysis: slope at its mean times its variance
    shifts = {}
    if "core_known" in post:
        mk, sk = post["core_known"]["mean"], post["core_known"]["sd"]
        for k in ("d_AGN", "d_gal", "d_sel", "d_core", "d_mc_sel", "d_halfvar", "d_mc_AGN_32000_minus_8000", "d_AGN_vs_32000",
                  "d_sel_vs_32000", "d_core_vs_32000"):
            if k not in T:
                continue
            y = np.asarray(T[k], float)
            if np.isfinite(y).sum() >= 5:
                sl = local_slope(H, y, mk)
                shifts[k] = {"slope_at_known_mean": sl, "slope_times_var_known": sl * sk ** 2}
        shifts["_known"] = {"mean": mk, "sd": sk, "from": "core_known (the sums and selection integrals, no Monte Carlo terms)"}
    # the AGN share over each class at 67.74 and 69.5
    share = {}
    for h in H0_R:
        j = int(np.argmin(np.abs(H - h)))
        if abs(H[j] - h) > 1e-6:
            continue
        share[f"{h:g}"] = {"galaxy_hosted": quantiles(r[j, g]), "agn_hosted": quantiles(r[j, a]),
                           "sum_r_galaxy_hosted": float(r[j, g].sum()), "sum_1_minus_r_agn_hosted": float((1 - r[j, a]).sum())}
    # per-merger contributions to the slope of the difference over SLOPE_WINDOW
    dA_i = lmix[:, a] - lA[:, a]                 # ln[f + (1 - f) G_i / A_i]
    dG_i = lmix[:, g] - lG[:, g]                 # ln[(1 - f) + f A_i / G_i]
    sA, sG = window_slopes(H, dA_i, *SLOPE_WINDOW), window_slopes(H, dG_i, *SLOPE_WINDOW)
    ia, ig = np.where(a)[0], np.where(g)[0]
    j0 = int(np.argmin(np.abs(H - H0_IN)))
    ev = event_truth(info, n)

    def top(idx, s, d, k=10):
        o = np.argsort(-s)[:k]
        tot = float(s.sum())
        rows = []
        for q in o:
            i = int(idx[q])
            rows.append({"merger": i, "slope": float(s[q]), "share_of_class_slope": float(s[q] / tot) if tot else None,
                         "r_at_67.74": float(r[j0, i]), "lnA_minus_lnG_at_67.74": float(lA[j0, i] - lG[j0, i]),
                         "delta_at_67.74": float(d[j0, q]), **{k2: (float(v[i]) if v is not None else None) for k2, v in ev.items()}})
        return rows

    contrib = {"window": list(SLOPE_WINDOW),
               "agn_hosted": {"sum_slope": float(sA.sum()), "n_positive": int((sA > 0).sum()),
                              "sum_top10": float(np.sort(sA)[::-1][:10].sum()), "top10": top(ia, sA, dA_i),
                              "bottom3": top(ia, -sA, dA_i, 3)},
               "galaxy_hosted": {"sum_slope": float(sG.sum()), "n_positive": int((sG > 0).sum()),
                                 "sum_top10": float(np.sort(sG)[::-1][:10].sum()), "top10": top(ig, sG, dG_i),
                                 "bottom3": top(ig, -sG, dG_i, 3)}}
    for c in ("agn_hosted", "galaxy_hosted"):
        for row in contrib[c]["bottom3"]:
            row["slope"] = -row["slope"]
            row["share_of_class_slope"] = None
    # the per-merger identity at the direct evaluations
    ident = None
    if fm.any():
        diff = z["ev_fm"][done][fm] - lmix[fm]
        mu_d = z["log_mu_fm"][done][fm] - lmu_mix[fm]
        ident = {"n_nodes": int(fm.sum()), "max_abs_event_identity": float(np.max(np.abs(diff))),
                 "max_abs_log_mu_identity": float(np.max(np.abs(mu_d))),
                 "max_abs_lnL_direct_minus_sum": float(np.nanmax(np.abs(z["lnL_fm"][done][fm] - (
                     lmix[fm].sum(1) + z["sel_corr_fm"][done][fm]))))}
    out = {"run": info["run"], "nside": nside, "f": f, "n_events": n, "n_agn_hosted": NA, "n_gal_hosted": NG,
           "info": info, "tables": {k: (np.asarray(v).tolist() if not isinstance(v, list) else v) for k, v in T.items()},
           "tables_minus_value_at_67.74": rel, "shape": shape, "posterior_on_nodes": post, "implied_shift": shifts,
           "agn_share": share, "slope_contributions": contrib, "identity": ident, "known_to_mixture_steps": steps,
           "checks": None if limit else stored_checks(R, nside, z, f),
           "known_agn_at_32000": None if limit else known_agn_block(R, nside, H, lA, ht),
           "seconds": {"per_eval_median": float(np.nanmedian(z["seconds"])), "total": float(np.nansum(z["seconds"]))}}
    return out


def event_truth(info, n):
    """Per-merger truth from the events file (true redshift, observed distance, SNR, sky width)."""
    import h5py
    path = info.get("events_file")
    out = {}
    if not path or not Path(path).exists():
        return {"true_z": None, "obs_dL": None, "snr_obs": None, "obs_sigma_ang": None}
    with h5py.File(path, "r") as f:
        for k, src in (("true_z", "truth/z"), ("obs_dL", "truth/obs_dL"), ("snr_obs", "truth/snr_obs"),
                       ("obs_sigma_ang", "truth/obs_sigma_ang")):
            out[k] = np.asarray(f[src])[:n] if src in f else None
    return out


def compare(o32, o64):
    """nside 64 minus nside 32: slopes and peaks of every piece."""
    if not (o32 and o64):
        return None
    out = {}
    for k in o32["shape"]:
        s32, s64 = o32["shape"][k], o64["shape"].get(k, {})
        row = {}
        for q in ("slope_at_67.74", "slope_at_69.5", "slope_over_grid", "peak_H0"):
            if s32.get(q) is not None and s64.get(q) is not None:
                row[q] = {"32": s32[q], "64": s64[q], "64_minus_32": s64[q] - s32[q]}
        out[k] = row
    for k in ("implied_shift",):
        out[k] = {p: {"32": o32[k].get(p, {}).get("slope_times_var_known"), "64": o64[k].get(p, {}).get("slope_times_var_known")}
                  for p in o32[k] if not p.startswith("_")}
    return out


def analyze(a):
    import warnings
    warnings.simplefilter("ignore")
    res = {"_what": "tests/mixture_decomposition.py: per-merger decomposition of the host-class-unknown likelihood "
                    "(two_tracers_one_population) against the host-class-known one (hosts_known) on the set " + SET,
           "realization": a.realization, "f": float(a.f), "nsides": {}}
    for ns in (32, 64):
        o = analyze_one(a.realization, ns, a.limit_events)
        if o is not None:
            res["nsides"][str(ns)] = o
    res["nside64_minus_32"] = compare(res["nsides"].get("32"), res["nsides"].get("64"))
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"mixture_decomposition_{a.realization}.json"
    if a.limit_events:
        out = OUT_DIR / f"mixture_decomposition_{a.realization}_lim{a.limit_events}.json"
    out.write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)) + "\n")
    report(res)
    print(f"-> {out}")


def report(res):
    for ns, o in res["nsides"].items():
        print(f"\n=== realization {res['realization']} nside {ns}: {o['n_events']} mergers ({o['n_agn_hosted']} AGN-hosted, "
              f"{o['n_gal_hosted']} galaxy-hosted), f = {o['f']}")
        if o["info"].get("incomplete"):
            print("   INCOMPLETE:", o["info"]["incomplete"])
        T, R = o["tables"], o["tables_minus_value_at_67.74"]
        cols = ("S_mix_AGN", "S_known_AGN", "d_AGN", "S_mix_gal", "S_known_gal", "d_gal", "sel_mix", "sel_known", "d_sel",
                "d_core", "d_mc_sel")
        print("   H0     " + " ".join(f"{c:>11s}" for c in cols) + "   (each minus its value at 67.74)")
        for j, h in enumerate(T["H0"]):
            print(f"   {h:6.2f} " + " ".join(f"{R[c][j]:11.3f}" for c in cols))
        print("   shape (peak H0, or edge and slope; slope at 67.74 and 69.5 per km/s/Mpc):")
        for k, s in o["shape"].items():
            if "slope_at_67.74" not in s:
                continue
            pk = f"peak {s['peak_H0']:.2f}" if s.get("peak_H0") is not None else f"edge {s['at_edge']} (slope {s['edge_slope']:+.3f})"
            print(f"     {k:20s} {pk:30s} slope@67.74 {s['slope_at_67.74']:+8.3f}  slope@69.5 {s['slope_at_69.5']:+8.3f}  "
                  f"grid slope {s['slope_over_grid']:+8.3f}")
        for k, p in o["posterior_on_nodes"].items():
            print(f"     posterior of {k:22s} mean {p['mean']:.3f} sd {p['sd']:.3f} parabola peak {p['parabola_peak']}")
        for k, s in o["implied_shift"].items():
            if not k.startswith("_"):
                print(f"     {k:10s} slope at the known mean {s['slope_at_known_mean']:+.4f} x var {o['implied_shift']['_known']['sd'] ** 2:.3f}"
                      f" = {s['slope_times_var_known']:+.3f} km/s/Mpc")
        st = o.get("known_to_mixture_steps")
        if st:
            print("   from the known analysis to the mixture (posterior mean on the nodes, sd, change):")
            for row in st["cumulative"]:
                ch = "" if row["change"] is None else f"  {row['change']:+.3f}"
                print(f"     {row['mean']:8.3f} {row['sd']:6.3f}{ch:>9s}  {row['step']}")
            print("     each alone added to the known at 8000: " + ", ".join(f"{k} {v:+.3f}" for k, v in st["alone_on_known_8000"].items()))
        for h, d in o["agn_share"].items():
            gq = d["galaxy_hosted"]
            print(f"     AGN share over galaxy-hosted at {h}: mean {gq['mean']:.4f} median {gq['median']:.4f} q95 {gq['q95']:.3f} "
                  f"q99 {gq['q99']:.3f} >0.5 {gq['frac_gt_0.5']:.3f}; sum {d['sum_r_galaxy_hosted']:.1f}; AGN-hosted mean "
                  f"{d['agn_hosted']['mean']:.3f}, sum(1-r) {d['sum_1_minus_r_agn_hosted']:.1f}")
        c = o["slope_contributions"]
        for cl in ("agn_hosted", "galaxy_hosted"):
            print(f"     {cl}: slope of the difference over {c['window']}: sum {c[cl]['sum_slope']:+.3f}, top ten "
                  f"{c[cl]['sum_top10']:+.3f}, {c[cl]['n_positive']} positive")
        if o.get("identity"):
            print(f"     identity: {o['identity']}")
        ch = o.get("checks") or {}
        print(f"     check vs stored mixture grid: {ch.get('mixture_n_cells')} cells, max |d lnL| {ch.get('mixture_max_abs_diff')}; "
              f"vs hosts_known lnL_gal: {ch.get('hosts_known_gal_n_cells')} H0, max |d| {ch.get('hosts_known_gal_max_abs_diff')}")
        ka = o.get("known_agn_at_32000")
        if ka and "H0" in ka:
            print(f"     AGN-hosted factor at 32000 samples (hosts_known's setting): slope of sum ln A at 67.74 "
                  f"{ka['slope_at_67.74_S_known_AGN_32000']:+.4f} (8000: {ka['slope_at_67.74_S_known_AGN_8000']:+.4f}; "
                  f"32000 minus 8000 {ka['slope_at_67.74_S32000_minus_S8000']:+.4f}); check vs stored lnL_agn: "
                  f"{ka['check_vs_stored_lnL_agn']['n_cells']} cells, max |d| {ka['check_vs_stored_lnL_agn']['max_abs_diff']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--realization", type=int, default=103)
    ap.add_argument("--nside", type=int, default=32)
    ap.add_argument("--h0", default="64,72,0.5", help="LO,HI,STEP of the H0 nodes (67.74 is added)")
    ap.add_argument("--f", type=float, default=0.30)
    ap.add_argument("--mix-at", choices=("all", "matching"), default="all",
                    help="direct evaluation at f0 at every node, or only where the stored grid has a cell")
    ap.add_argument("--limit-events", type=int, default=None, help="first N mergers only (tests of this script)")
    ap.add_argument("--analyze", action="store_true", help="tables from the stored per-merger terms, no evaluation")
    ap.add_argument("--known-agn", action="store_true",
                    help="evaluate the AGN-hosted subset of hosts_known at its own settings (32000 samples) per H0")
    a = ap.parse_args()
    if a.analyze:
        analyze(a)
    elif a.known_agn:
        compute_known_agn(a)
    else:
        compute(a)


if __name__ == "__main__":
    main()
