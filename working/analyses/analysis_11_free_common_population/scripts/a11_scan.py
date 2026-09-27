#!/usr/bin/env python
"""Analysis 11A / 11B -- deterministic (f, mu, dmu) grids at H0 = 67.74.

    11A  free (f, mu_G, dmu_G);   spin pinned at the planted (mu_chi, dmu_chi) = (0, +0.10)
    11B  free (f, mu_chi, dmu_chi); mass pinned at the planted (mu_G, dmu_G) = (35, +5)

A ROW is one (mu, dmu) pair evaluated at every f node of scripts/a11_grid_axes.json,
all of which are nodes of Analysis 10's f lattice (linspace(0, 1, 41)), so every 11A cell at mu_G = 35 and every 11B cell at
mu_chi = 0 whose dmu is an Analysis-10 node is cell-for-cell a node of the recorded
results/a10_arm_J.h5 cube: a free closure checked at assembly.  The reference slab
(mu_G = 35 / mu_chi = 0) is dealt first in every chunk's row order.

Stages
    scan      GPU, rita: --arm {11A,11B} --chunk k --n_chunks n.  Checkpoint
              diagnostics/_a11_{arm}_c{k}of{n}.jsonl, resumable, one line per row.
    status    CPU: rows done / total, GPU-hours.
    assemble  CPU: results/a11_{arm}.{h5,json} -- marginals (trapezoid), medians,
              68/90% intervals, MAP, correlations, edge densities on every axis,
              the density at dmu = 0 against the peak, the guard map, and the
              slab closure against a10_arm_J.h5.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
A11 = HERE.parent
DIAG, RESULTS = A11 / "diagnostics", A11 / "results"
sys.path.insert(0, str(HERE))

F_AXIS = np.linspace(0.0, 1.0, 41)


def _axis(lo, hi, step):
    n = int(round((hi - lo) / step)) + 1
    return np.round(lo + step * np.arange(n), 10)


# Production resolution (set from diagnostics/a11_closure.json's profiles; see
# STATE.md for the timing check and the reasoning).  The domains are the brief's.
ARMS = {
    "11A": {"mu": ("mu_G", None), "dmu": ("dmu_G", None),
            "pinned": {"mu_chi": 0.0, "dmu_chi": 0.10}, "ref_mu": 35.0},
    "11B": {"mu": ("mu_chi", None), "dmu": ("dmu_chi", None),
            "pinned": {"mu_G": 35.0, "dmu_G": 5.0}, "ref_mu": 0.0},
}


def set_axes(arm, mu_axis, dmu_axis):
    ARMS[arm]["mu"] = (ARMS[arm]["mu"][0], np.asarray(mu_axis, dtype=float))
    ARMS[arm]["dmu"] = (ARMS[arm]["dmu"][0], np.asarray(dmu_axis, dtype=float))


def _load_axes():
    global F_AXIS
    cfg = json.loads((A11 / "scripts" / "a11_grid_axes.json").read_text())
    F_AXIS = np.asarray(cfg["f_axis"], dtype=float)
    for arm, ax in cfg["arms"].items():
        set_axes(arm, ax["mu"], ax["dmu"])
    return cfg


def _key(mu, dmu):
    return f"{float(mu):.10g}|{float(dmu):.10g}"


def _append(path, obj):
    with open(path, "a") as fh:
        fh.write(json.dumps(obj) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _read(path):
    hdr, rows, bad = None, [], 0
    if not Path(path).exists():
        return hdr, rows, bad
    for line in Path(path).read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        if r.get("record") == "header":
            hdr = r
        elif r.get("record") == "row":
            rows.append(r)
    return hdr, rows, bad


def _ckpts(arm):
    return sorted(DIAG.glob(f"_a11_{arm}_c*of*.jsonl"))


def row_order(arm):
    a = ARMS[arm]
    mu_ax, dmu_ax = a["mu"][1], a["dmu"][1]
    ref = [(a["ref_mu"], d) for d in dmu_ax] if np.any(mu_ax == a["ref_mu"]) else []
    rest = [(m, d) for m in mu_ax for d in dmu_ax if m != a["ref_mu"]]
    return ref + rest


def stage_scan(args):
    import a11_likelihood as L
    sys.path.insert(0, str(A11.parent / "analysis_9_marked_multitracer_H0_fagn" / "scripts"))
    import a9_scan as A9
    cfg = _load_axes()
    arm = ARMS[args.arm]
    env = A9._gpu_setup(f"a11_{args.arm}")
    order = row_order(args.arm)
    mine = [x for i, x in enumerate(order) if i % args.n_chunks == args.chunk]
    path = DIAG / f"_a11_{args.arm}_c{args.chunk}of{args.n_chunks}.jsonl"
    done = set()
    for p in _ckpts(args.arm):
        done |= {_key(r["mu"], r["dmu"]) for r in _read(p)[1]}
    todo = [(m, d) for (m, d) in mine if _key(m, d) not in done]
    print(f"[{args.arm}] chunk {args.chunk}/{args.n_chunks}: {len(mine)} rows, "
          f"{len(todo)} to do ({len(todo) * F_AXIS.size} cells)", flush=True)
    if not todo:
        return
    t0 = time.time()
    cell = L.build_a11(f"A11_{args.arm}_c{args.chunk}")
    print(f"build {time.time() - t0:.1f}s", flush=True)
    if _read(path)[0] is None:
        _append(path, {"record": "header", "arm": args.arm, "axes_config": cfg,
                       "labels": list(cell.labels), "pinned": arm["pinned"],
                       "f_axis": F_AXIS.tolist(), "events_md5": L.GW_MD5_A11,
                       "provenance": L.A10.a8.provenance(gw_path=L.GW_PATH_A11,
                                                         survey_paths=[L.SURVEY_GAL, L.SURVEY_AGN]),
                       "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "environment": env})
    mu_name, dmu_name = arm["mu"][0], arm["dmu"][0]
    t0 = time.time()
    for n, (m, d) in enumerate(todo, start=1):
        cells = []
        for f in F_AXIS:
            kw = dict(arm["pinned"], **{mu_name: float(m), dmu_name: float(d)})
            r = cell.evaluate_at(H0=L.H0_FID, fcat_2=float(f), **kw)
            g = r.get("guard") or {}
            cells.append({"logL": r["logL"], "logL_hex": r["logL_hex"], "finite": r["finite"],
                          "logL_pe": r.get("logL_pe"), "logL_selection": r.get("logL_selection"),
                          "Neff": r.get("Neff"), "threshold": r.get("threshold"),
                          "sigma2_total": g.get("sigma2_total"),
                          "pe_variance_sum": g.get("pe_variance_sum"),
                          "seconds": r["seconds"]})
        _append(path, {"record": "row", "mu": float(m), "dmu": float(d),
                       "key": _key(m, d), "cells": cells,
                       "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        ll = np.array([c["logL"] for c in cells])
        el = time.time() - t0
        print(f"[{args.arm}] row {n}/{len(todo)} {mu_name}={m:g} {dmu_name}={d:g} "
              f"max logL {np.nanmax(ll):.3f} rejected {int((~np.isfinite(ll)).sum())} "
              f"eta {el / n * (len(todo) - n) / 60:.1f} min", flush=True)


def _cube(arm):
    a = ARMS[arm]
    mu_ax, dmu_ax = a["mu"][1], a["dmu"][1]
    rows = {}
    for p in _ckpts(arm):
        for r in _read(p)[1]:
            rows.setdefault(r["key"], r)
    shape = (F_AXIS.size, mu_ax.size, dmu_ax.size)
    out = {k: np.full(shape, np.nan) for k in ("logL", "logL_pe", "logL_selection",
                                               "Neff", "threshold", "seconds")}
    missing = 0
    for i, m in enumerate(mu_ax):
        for j, d in enumerate(dmu_ax):
            r = rows.get(_key(m, d))
            if r is None:
                missing += 1
                continue
            for k in out:
                out[k][:, i, j] = [np.nan if c.get(k) is None else c[k] for c in r["cells"]]
    return out, missing


def stage_status(args):
    _load_axes()
    for arm in ARMS:
        if not _ckpts(arm):
            continue
        c, missing = _cube(arm)
        n = c["logL"].shape[1] * c["logL"].shape[2]
        print(f"{arm}: rows {n - missing}/{n}; GPU-h {np.nansum(c['seconds']) / 3600:.2f}")


def _summ(x, p):
    p = np.asarray(p, float)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
    cdf /= cdf[-1]
    q = lambda t: float(np.interp(t, cdf, x))
    return {"median": q(0.5), "ci68": [q(0.16), q(0.84)], "ci90": [q(0.05), q(0.95)],
            "mode": float(x[int(np.argmax(p))])}


def stage_assemble(args):
    import h5py
    cfg = _load_axes()
    arm = args.arm
    a = ARMS[arm]
    mu_name, mu_ax = a["mu"]
    dmu_name, dmu_ax = a["dmu"]
    c, missing = _cube(arm)
    if missing:
        raise SystemExit(f"[fatal] {missing} rows missing")
    ll = c["logL"]
    fin = np.isfinite(ll)
    mx = np.max(ll[fin])
    P = np.where(fin, np.exp(np.where(fin, ll, -np.inf) - mx), 0.0)
    axes = {"f_agn": F_AXIS, mu_name: mu_ax, dmu_name: dmu_ax}
    names = list(axes)
    T = lambda arr, x, ax: np.trapz(arr, x, axis=ax)
    marg = {
        "f_agn": T(T(P, dmu_ax, 2), mu_ax, 1),
        mu_name: T(T(P, dmu_ax, 2), F_AXIS, 0),
        dmu_name: T(T(P, mu_ax, 1), F_AXIS, 0),
    }
    m2 = {f"f_agn|{mu_name}": T(P, dmu_ax, 2), f"f_agn|{dmu_name}": T(P, mu_ax, 1),
          f"{mu_name}|{dmu_name}": T(P, F_AXIS, 0)}
    # moments with trapezoid weights
    W = (np.gradient(F_AXIS)[:, None, None] * np.gradient(mu_ax)[None, :, None]
         * np.gradient(dmu_ax)[None, None, :])
    w = P * W
    w /= w.sum()
    G = np.meshgrid(F_AXIS, mu_ax, dmu_ax, indexing="ij")
    mean = [float((w * g).sum()) for g in G]
    cov = np.array([[float((w * (G[i] - mean[i]) * (G[j] - mean[j])).sum())
                     for j in range(3)] for i in range(3)])
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    out = {"arm": arm, "question": __doc__.split("\n")[2].strip(),
           "axes": {k: v.tolist() for k, v in axes.items()}, "pinned": a["pinned"],
           "axes_config": cfg, "n_cells": int(ll.size), "n_rejected": int((~fin).sum()),
           "gpu_hours": float(np.nansum(c["seconds"]) / 3600),
           "seconds_per_cell_median": float(np.nanmedian(c["seconds"])),
           "logL_max": float(mx)}
    idx = np.unravel_index(int(np.argmax(np.where(fin, ll, -np.inf))), ll.shape)
    out["map"] = {n: float(axes[n][i]) for n, i in zip(names, idx)}
    for n in names:
        p = marg[n] / marg[n].max()
        s = _summ(axes[n], marg[n])
        s["sd"] = float(sd[names.index(n)])
        s["edge_over_peak"] = [float(p[0]), float(p[-1])]
        s["contained_1e-6"] = bool(max(p[0], p[-1]) <= 1e-6)
        s["marginal"] = p.tolist()
        out[n] = s
    out["correlations"] = {f"{names[i]}|{names[j]}": float(corr[i, j])
                           for i in range(3) for j in range(i + 1, 3)}
    # the environmental-zero density
    pd = marg[dmu_name] / marg[dmu_name].max()
    out[f"{dmu_name}_zero_density_over_peak"] = float(np.interp(0.0, dmu_ax, pd))
    # the offset's marginal on the pinned-reference slab (the A10 comparison)
    kref = np.where(mu_ax == a["ref_mu"])[0]
    if kref.size:
        ps = T(P[:, kref[0], :], F_AXIS, 0)
        out[f"{dmu_name}_on_reference_slab_{mu_name}={a['ref_mu']}"] = _summ(dmu_ax, ps)
    # slab closure against the recorded A10-J cube
    try:
        with h5py.File(A11.parent / "analysis_10_mass_spin_marked_multitracer/results/a10_arm_J.h5",
                       "r") as h:
            fg, gg, cg = h["f_grid"][:], h["dmu_G_grid"][:], h["dmu_chi_grid"][:]
            R = h["log_likelihood"][:]
        n_cmp = n_bit = 0
        worst = 0.0
        if kref.size:
            for j, d in enumerate(dmu_ax):
                if arm == "11A":
                    kk, jj = np.where(np.isclose(gg, d, atol=1e-12))[0], \
                        np.where(np.isclose(cg, 0.10, atol=1e-12))[0]
                else:
                    kk, jj = np.where(np.isclose(gg, 5.0, atol=1e-12))[0], \
                        np.where(np.isclose(cg, d, atol=1e-12))[0]
                if not (kk.size and jj.size):
                    continue
                for i in range(F_AXIS.size):
                    fi = np.where(np.isclose(fg, F_AXIS[i], atol=1e-12))[0]
                    if not fi.size:
                        continue
                    x, y = ll[i, kref[0], j], R[fi[0], kk[0], jj[0]]
                    n_cmp += 1
                    n_bit += int(float(x).hex() == float(y).hex())
                    if np.isfinite(x) and np.isfinite(y):
                        worst = max(worst, abs(x - y))
        out["slab_closure_vs_a10_arm_J"] = {"n_compared": n_cmp, "n_bitwise": n_bit,
                                            "worst_abs": worst,
                                            "pass": bool(n_cmp and n_bit == n_cmp)}
    except FileNotFoundError:
        out["slab_closure_vs_a10_arm_J"] = {"available": False}
    RESULTS.mkdir(exist_ok=True)
    with h5py.File(RESULTS / f"a11_{arm}.h5", "w") as h:
        for n, x in axes.items():
            h.create_dataset(f"axis/{n}", data=x)
        for k, v in c.items():
            h.create_dataset(k, data=v)
        h.create_dataset("posterior_unnormalised", data=P)
        for k, v in marg.items():
            h.create_dataset(f"marginal/{k}", data=v)
        for k, v in m2.items():
            h.create_dataset(f"marginal_2d/{k.replace('|', '__')}", data=v)
        h.attrs["axis_order"] = json.dumps(names)
    (RESULTS / f"a11_{arm}.json").write_text(json.dumps(out, indent=2))
    print(f"wrote results/a11_{arm}.{{h5,json}}")
    for n in names:
        s = out[n]
        print(f"  {n:<8s} median {s['median']:.4f} 68% [{s['ci68'][0]:.4f}, {s['ci68'][1]:.4f}] "
              f"90% [{s['ci90'][0]:.4f}, {s['ci90'][1]:.4f}] edges {s['edge_over_peak']}")
    print(f"  MAP {out['map']}  corr {out['correlations']}")
    print(f"  {dmu_name}=0 density/peak {out[f'{dmu_name}_zero_density_over_peak']:.3e}; "
          f"slab closure {out['slab_closure_vs_a10_arm_J']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("scan", "status", "assemble"))
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--n_chunks", type=int, default=1)
    args = ap.parse_args(argv)
    {"scan": stage_scan, "status": stage_status, "assemble": stage_assemble}[args.stage](args)


if __name__ == "__main__":
    main()
