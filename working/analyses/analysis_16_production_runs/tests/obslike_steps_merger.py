#!/usr/bin/env python
"""The upward steps of ln L(H0) in the observed-like runs: which merger, which samples, how well resolved (CPU only).

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH= <core venv python> tests/obslike_steps_merger.py full ANALYSIS SET R \
        [--h0 68.2,68.4,69.0,69.2] [--f 0.30] [--limit-inj N] [--tag T] [--nsamp N]
    ... single ANALYSIS SET R IDX [--blocks 64] [--h0 66,72,0.02] [--f 0.30]
    ... edge SET R [--blocks 4]                      (numpy only: per merger, samples whose m1src crosses m_max on 66..72)

full: the run's own configuration (prod_driver.parse_args/resolve from the run list, compared with the stored config.json),
built and bound by prod_driver.setup_model, evaluated with prod_model.Model.evaluate_events at (H0, f_agn) nodes: every merger's
ln L_i and Monte Carlo variance. The per-sample log weights of the PE sum are captured as core computes them (a
jax.debug.callback attached, in this process only, to darksirens.likelihood.event._masked_log_weights, the function whose
output core reduces with log_evidence_and_mc_variance); from them, per merger, the effective sample number (sum w)^2 / sum w^2,
and for the mergers that change most between consecutive nodes the full per-sample rows with each sample's source-frame m1
(z from core's z_of_dL at the node). With the full injection set the sum is checked against the stored grid node by node.
--limit-inj N: the first N detected injections only (the per-merger terms do not depend on them; ln L then does not match).
-> tests/results/obslike_steps/full_<analysis>_<SET>_r<R><tag>.npz and .json

single: one merger alone with B blocks of 2000 redrawn samples (block b drawn for the whole set with the run's key
redraw_key + [b], exactly as the production redraw, then the merger's row kept), the run's population/catalog model, at the
nodes of an H0 line: the per-sample log weights at every node; then ln L_i(H0) for the first 8000, 16000, 32000, all samples
and for independent 8000-sample subsets (blocks 4k..4k+3), with each subset's own p_pe normalisation (core normalises p_pe per
event over the event's samples: ldw_S = ldw_all + ln(sum_S p / sum_all p)). The first 8000 are checked against full's ln L_i.
-> tests/results/obslike_steps/single_<analysis>_<SET>_r<R>_i<IDX>.npz and .json

Nothing in scripts/ or runs/ is written; scripts/ modules are imported only.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

A16 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(A16 / "scripts"))
import prod_common as _P  # noqa: E402

_P.ensure_paths()             # analysis 14's mc_redraw (the production redraw)
RUNS = A16 / "runs"
OUT = A16 / "tests" / "results" / "obslike_steps"
EDGE = {"obslike": 87.0, "corepop": 80.0}       # m_max of the power law (dm_max 0.01 obslike, 10 corepop)
RUN = "{analysis}/both_catalogs/complete/{set}/realization_{R:03d}"
CMP_KEYS = ("events_file", "injections_files", "surveys", "nsamp", "redraw_key", "free", "population_values",
            "class_populations", "fixed_offsets", "half_variance_correction", "max_var", "pe_event_block", "sel_batch",
            "catalog_evaluation", "zmax", "completeness_model", "Om0", "planted_f_agn")


def runlist_args(name):
    import yaml
    for f in ("RUNLIST_experiment.yaml", "RUNLIST.yaml"):
        doc = yaml.safe_load((RUNS / f).read_text())
        for st in doc["stages"]:
            for r in st["runs"]:
                if r["name"] == name:
                    return list(r["args"]), f
    raise SystemExit(f"[fatal] {name} is in neither run list")


def make_cfg(name):
    import prod_driver as PD
    args, src = runlist_args(name)
    cfg = PD.resolve(PD.parse_args(args))
    if cfg["run_name"] != name:
        raise SystemExit(f"[fatal] resolved run name {cfg['run_name']} != {name}")
    sc_path = RUNS / name / "config.json"
    check = None
    if sc_path.exists():
        sc = json.loads(sc_path.read_text())
        diff = {k: (sc.get(k), cfg.get(k)) for k in CMP_KEYS if sc.get(k) != cfg.get(k)}
        if diff:
            raise SystemExit(f"[fatal] resolved configuration differs from the stored run's: {diff}")
        check = {"stored_config": str(sc_path), "keys_compared": list(CMP_KEYS), "differences": {}}
    return cfg, src, check


# ---------------------------------------------------------------------------- per-sample capture
CAP = {"blocks": {}}


def _cb(start, ldw):
    CAP["blocks"][int(start)] = np.array(ldw, dtype=float)


def install_capture():
    """Attach a host callback to core's per-sample PE log weights (this process only; traces made after the call)."""
    import jax
    import jax.numpy as jnp
    import darksirens.likelihood.event as E
    if getattr(E, "_obslike_steps_capture", False):
        return
    orig = E._masked_log_weights

    def patched(event, start, size, log_weight_fn, sky_log_weight_fn=None):
        ldw = orig(event, start, size, log_weight_fn, sky_log_weight_fn)
        jax.debug.callback(_cb, jnp.asarray(start), ldw)
        return ldw

    E._masked_log_weights = patched
    E._obslike_steps_capture = True


def captured(n_total):
    out = np.full(n_total, np.nan)
    for s, v in CAP["blocks"].items():
        out[s:s + v.size] = v
    CAP["blocks"] = {}
    if np.isnan(out).any():
        raise SystemExit(f"[fatal] {int(np.isnan(out).sum())} per-sample weights not captured")
    return out


def lse(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def ess_of(ldw, axis=-1):
    """(sum w)^2 / sum w^2 of log weights."""
    return np.exp(2 * lse(ldw, axis) - lse(2 * ldw, axis))


def m1src_at(m1det, dL, h0, om0):
    import jax.numpy as jnp
    from darksirens.cosmology.distances import z_of_dL
    z = np.asarray(z_of_dL(jnp.asarray(dL), float(h0), float(om0)))
    return m1det / (1.0 + z), z


def h0_cross(m1det, dL, edge, om0):
    """The H0 at which a sample's source-frame m1 = m1det / (1 + z(dL; H0)) equals edge: dL(z*; H0) = dL with
    z* = m1det/edge - 1, and dL(z; H0) proportional to 1/H0 at fixed z, Om0 (core's distance table at H0 = 100)."""
    import jax.numpy as jnp
    from darksirens.cosmology.distances import dL_of_z
    zs = m1det / edge - 1.0
    out = np.full(m1det.shape, -np.inf)        # m1det < edge: inside at every H0
    ok = zs > 0
    out[ok] = 100.0 * np.asarray(dL_of_z(jnp.asarray(zs[ok]), 100.0, float(om0))) / dL[ok]
    return out


def stored_node(name, h, f, files=("fine_widened.h5", "fine.h5", "coarse.h5")):
    import h5py
    for fn in files:
        p = RUNS / name / "grid" / fn
        if not p.exists():
            continue
        with h5py.File(p, "r") as F:
            ax = np.asarray(F["axis_H0"][()])
            af = np.asarray(F["axis_f_agn"][()]) if "axis_f_agn" in F else None
            i = int(np.argmin(np.abs(ax - h)))
            if abs(ax[i] - h) > 1e-6:
                continue
            if af is not None:
                j = int(np.argmin(np.abs(af - f)))
                if abs(af[j] - f) > 1e-6:
                    continue
                return {k: float(F[k][()][i, j]) for k in ("lnL", "pe_var", "n_eff", "log_mu", "sel_corr")} | {"file": fn}
            return {k: float(F[k][()][i]) for k in ("lnL", "pe_var", "n_eff", "log_mu", "sel_corr")} | {"file": fn}
    return None


def setup(name, overrides=None, select=None):
    import prod_driver as PD
    cfg, src, check = make_cfg(name)
    if overrides:
        cfg.update(overrides)
    t0 = time.time()
    if select is None:
        m = PD.setup_model(cfg)
    else:                               # setup_model's steps, binding the selected mergers only (the run's redraw, then subset)
        os.environ["DARKSIRENS_ZMAX"] = repr(float(cfg["zmax"]))
        import prod_model as PM
        m = PM.make_model(cfg)
        m.import_core()
        import jax
        jax.config.update("jax_enable_x64", True)
        m.build()
        m.bind(select=select)
    install_capture()
    return m, cfg, src, check, time.time() - t0


def truth_table(events_file, idx):
    import h5py
    with h5py.File(events_file, "r") as f:
        t = f["truth"]
        row = {k: float(np.asarray(t[k][idx])) for k in t.keys() if np.asarray(t[k]).ndim == 1 and np.asarray(t[k]).shape[0] == int(f.attrs["nobs"])}
    return row


# ============================================================================ full
def full(analysis, label, R, h0s, f0, limit_inj=None, tag="", nsamp=None, limit_events=None, mergers=None):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    name = RUN.format(analysis=analysis, set=label, R=R)
    ov = {}
    if limit_inj:
        ov["limit_injections"] = int(limit_inj)
    if nsamp:
        ov["nsamp"] = int(nsamp)
    if limit_events:
        ov["limit_events"] = int(limit_events)
    sel = None
    if mergers is not None:
        sel = np.zeros(1000 if not limit_events else int(limit_events), bool)
        sel[np.asarray(mergers, int)] = True
    m, cfg, src, check, t_setup = setup(name, ov, sel)
    n, ns = m.n_events, m.nsamp
    index = np.arange(n) if sel is None else np.where(sel)[0]
    pw = np.asarray(m.b.gw_pe.prior_wt, dtype=float).reshape(n, ns)
    fam = cfg["population_family"]
    edge = EDGE[fam]
    om0 = float(cfg["Om0"])
    gw = m.b.gw_pe
    m1det = np.asarray(gw.m1det, dtype=float).reshape(n, ns)
    dL = np.asarray(gw.dL, dtype=float).reshape(n, ns)
    H = np.asarray(h0s, float)
    rec = {"H0": H, "f": f0, "index": index, "ev8000": np.full((len(H), n), np.nan), "ev": np.full((len(H), n), np.nan), "var": np.full((len(H), n), np.nan),
           "ess": np.full((len(H), n), np.nan), "n_in": np.zeros((len(H), n), int), "wfrac_in": np.full((len(H), n), np.nan),
           "lnL": np.full(len(H), np.nan), "pe_var": np.full(len(H), np.nan), "n_eff": np.full(len(H), np.nan),
           "log_mu": np.full(len(H), np.nan), "sel_corr": np.full(len(H), np.nan), "seconds": np.full(len(H), np.nan)}
    ldw_keep = {}
    stored = []
    for j, h in enumerate(H):
        tc = time.time()
        r = m.evaluate_events({"H0": float(h), "f_agn": float(f0)})
        rec["seconds"][j] = time.time() - tc
        ldw = captured(n * ns).reshape(n, ns)
        ev_np = lse(ldw) - np.log(ns)
        dmax = float(np.max(np.abs(ev_np - r["event_log_evidence"])))
        rec["ev"][j], rec["var"][j] = r["event_log_evidence"], r["event_mc_variance"]
        rec["ess"][j] = ess_of(ldw)
        if ns > 8000:                   # the first 8000 samples (the production run's), p_pe normalised over them as core does
            rec["ev8000"][j] = lse(ldw[:, :8000] + np.log(pw[:, :8000].sum(1, keepdims=True))) - np.log(8000)
        ms, _ = m1src_at(m1det.ravel(), dL.ravel(), h, om0)
        inside = (ms < edge).reshape(n, ns)
        rec["n_in"][j] = inside.sum(1)
        w = np.exp(ldw - np.max(ldw, axis=1, keepdims=True))
        rec["wfrac_in"][j] = (w * inside).sum(1) / w.sum(1)
        ldw_keep[j] = ldw
        for k in ("lnL", "pe_var", "n_eff", "log_mu", "sel_corr"):
            rec[k][j] = r[k]
        st = None if limit_inj else stored_node(name, h, f0)
        stored.append(st)
        msg = (f"[full] {analysis} {label} r{R} H0 {h:g} f {f0:g}: lnL {r['lnL']:.6f} pe_var {r['pe_var']:.5f} sum ev {r['event_log_evidence'].sum():.6f}"
               f" (numpy from captured weights: max |diff| {dmax:.1e}) ({rec['seconds'][j]:.0f} s)")
        if st:
            msg += (f" | stored ({st['file']}) lnL {st['lnL']:.6f} diff {r['lnL'] - st['lnL']:+.2e}, pe_var diff {r['pe_var'] - st['pe_var']:+.2e},"
                    f" log_mu diff {r['log_mu'] - st['log_mu']:+.2e}")
        print(msg, flush=True)
    # mergers that change most between consecutive nodes: keep their per-sample rows
    top = set()
    for j in range(len(H) - 1):
        d = rec["ev"][j + 1] - rec["ev"][j]
        top.update(int(i) for i in np.argsort(-np.abs(d - np.median(d)))[:8])
    top.update(int(i) for i in np.where(np.nanmin(rec["ess"], axis=0) < 5)[0])
    top = np.array(sorted(top))
    rec["top"] = top
    rec["top_ldw"] = np.stack([ldw_keep[j][top] for j in range(len(H))])        # (nH, ntop, ns)
    rec["top_m1det"], rec["top_dL"] = m1det[top], dL[top]
    rec["top_h0cross"] = h0_cross(m1det[top].ravel(), dL[top].ravel(), edge, om0).reshape(len(top), ns)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"full_{analysis}_{label}_r{R}{tag}"
    np.savez_compressed(OUT / f"{stem}.npz", **rec)
    info = {"run": name, "args_from": src, "config_check": check, "overrides": ov, "n_events": n, "nsamp": ns,
            "setup_seconds": t_setup, "stored_nodes": stored, "H0": H.tolist(), "f": f0, "edge_m1": edge,
            "lnL": rec["lnL"].tolist(), "pe_var": rec["pe_var"].tolist(),
            "lnL_minus_stored": [None if s is None else rec["lnL"][j] - s["lnL"] for j, s in enumerate(stored)],
            "pe_var_minus_stored": [None if s is None else rec["pe_var"][j] - s["pe_var"] for j, s in enumerate(stored)]}
    for j in range(len(H) - 1):
        d = rec["ev"][j + 1] - rec["ev"][j]
        o = np.argsort(-np.abs(d - np.median(d)))[:6]
        info[f"largest_change_{H[j]:g}_to_{H[j + 1]:g}"] = [
            {"i": int(i), "d_ev": float(d[i]), "d_var": float(rec["var"][j + 1, i] - rec["var"][j, i]),
             "ess": [float(rec["ess"][j, i]), float(rec["ess"][j + 1, i])], "n_in": [int(rec["n_in"][j, i]), int(rec["n_in"][j + 1, i])]}
            for i in o]
        info[f"sum_d_ev_{H[j]:g}_to_{H[j + 1]:g}"] = float(d.sum())
        print(f"[full] {H[j]:g} -> {H[j + 1]:g}: sum d ev {d.sum():+.4f}, median {np.median(d):+.5f}; largest: "
              + "; ".join(f"i={x['i']} d={x['d_ev']:+.3f} dvar={x['d_var']:+.3f} ess {x['ess'][0]:.2f}->{x['ess'][1]:.2f} n_in {x['n_in'][0]}->{x['n_in'][1]}"
                          for x in info[f"largest_change_{H[j]:g}_to_{H[j + 1]:g}"][:4]), flush=True)
    (OUT / f"{stem}.json").write_text(json.dumps(info, indent=1, default=float))
    print(f"[full] -> {OUT / stem}.npz/.json", flush=True)


# ============================================================================ single
def redraw_one(cfg, idx, blocks):
    """Merger idx's samples of blocks 0..blocks-1 of the production redraw (whole set drawn per block), raw columns."""
    import mc_redraw as M
    import prod_common as P
    obs, attrs, nobs, _ = M.read_event_set(cfg["events_file"])
    g = M.generator()
    rows = {}
    for b in range(blocks):
        rng = np.random.default_rng(np.random.SeedSequence(list(cfg["redraw_key"]) + [b]))
        p = g.posterior_samples_v3(rng, obs, P.REDRAW_BLOCK)
        for k, v in p.items():
            rows.setdefault(k, []).append(v.reshape(nobs, P.REDRAW_BLOCK)[idx])
    return {k: np.concatenate(v) for k, v in rows.items()}, attrs


def single(analysis, label, R, idx, blocks, h0spec, f0):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import mc_redraw as M
    name = RUN.format(analysis=analysis, set=label, R=R)
    t0 = time.time()
    cfg, src, check = make_cfg(name)
    cfg["limit_injections"] = 5000          # the per-merger term does not involve the injections
    raw, attrs = redraw_one(cfg, idx, blocks)
    N = raw["dL"].size
    t_draw = time.time() - t0
    post = M.postprocess({k: v.copy() for k, v in raw.items()}, 1, N)
    sub = {k: v for k, v in post.items() if k in M.STORE_COLUMNS}
    import prod_driver as PD
    import prod_model as PM
    os.environ["DARKSIRENS_ZMAX"] = repr(float(cfg["zmax"]))
    m = PM.make_model(cfg)
    m.import_core()
    import jax
    jax.config.update("jax_enable_x64", True)
    m.build()
    store = M.load_store(m.ds, sub, attrs, 1, N)
    m.bind(events_store=store)
    install_capture()
    lo, hi, st = (float(x) for x in h0spec.split(","))
    H = np.round(lo + st * np.arange(int(round((hi - lo) / st)) + 1), 10)
    H = np.unique(np.round(np.concatenate([H, [67.74, 68.2, 68.4, 69.0, 69.2]]), 10))
    ldw = np.full((len(H), N), np.nan)
    ev = np.full(len(H), np.nan)
    tc = time.time()
    for j, h in enumerate(H):
        r = m.evaluate_events({"H0": float(h), "f_agn": float(f0)})
        ldw[j] = captured(N)
        ev[j] = r["event_log_evidence"][0]
    t_eval = time.time() - tc
    chk = float(np.max(np.abs(lse(ldw) - np.log(N) - ev)))
    om0 = float(cfg["Om0"])
    edge = EDGE[cfg["population_family"]]
    hc = h0_cross(raw["m1det"], raw["dL"], edge, om0)
    ppe = post["p_pe"]                          # proportional to core's per-event normalised p_pe
    lnS_all = np.log(ppe.sum())

    def curve(sel):
        ldw_s = ldw[:, sel] + (np.log(ppe[sel].sum()) - lnS_all)
        return lse(ldw_s) - np.log(sel.sum()), ess_of(ldw_s)

    idxs = np.arange(N)
    subsets = {"first8000": idxs < 8000, "first16000": idxs < 16000, "first32000": idxs < 32000, f"all{N}": idxs < N}
    nb8 = N // 8000
    for k in range(nb8):
        subsets[f"sub8000_{k}"] = (idxs >= 8000 * k) & (idxs < 8000 * (k + 1))
    nb32 = N // 32000
    for k in range(nb32):
        subsets[f"sub32000_{k}"] = (idxs >= 32000 * k) & (idxs < 32000 * (k + 1))
    curves = {}
    for s, sel in subsets.items():
        if sel.sum() > N:
            continue
        c, e = curve(sel)
        curves[s] = {"lnL": c, "ess": e}
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"single_{analysis}_{label}_r{R}_i{idx}"
    # per-sample rows kept for samples that ever carry weight (within 25 nats of the merger's best sample at some node)
    rel = ldw - np.max(ldw, axis=1, keepdims=True)
    keep = np.where(np.max(rel, axis=0) > -25)[0]
    np.savez_compressed(OUT / f"{stem}.npz", H0=H, ev=ev, keep=keep, ldw_keep=ldw[:, keep], h0cross=hc, p_pe=ppe,
                        **{f"raw_{k}": v for k, v in raw.items()},
                        **{f"lnL_{s}": c["lnL"] for s, c in curves.items()}, **{f"ess_{s}": c["ess"] for s, c in curves.items()})
    info = {"run": name, "args_from": src, "config_check": check, "merger": idx, "samples": N, "blocks": blocks,
            "seconds_draw": t_draw, "seconds_eval": t_eval, "max_abs_numpy_minus_core": chk, "edge_m1": edge,
            "H0": H.tolist(), "f": f0, "n_keep": int(keep.size)}
    for s, c in curves.items():
        info[s] = {"lnL": c["lnL"].tolist(), "ess": c["ess"].tolist()}
    (OUT / f"{stem}.json").write_text(json.dumps(info, default=float))
    print(f"[single] {name} merger {idx}: {N} samples drawn in {t_draw:.0f} s, {len(H)} nodes in {t_eval:.0f} s; numpy vs core "
          f"max |diff| {chk:.1e}; -> {OUT / stem}.npz", flush=True)


# ============================================================================ edge (numpy + core distances)
def edge(label, R, blocks=4, analysis="two_tracers_one_population", run=None, lo=66.0, hi=72.0):
    """Every merger's samples of the production redraw (blocks 0..blocks-1): the H0 at which each sample's m1src crosses
    m_max; per merger the number of samples in the power law's support at H0 = 66, 67.74, 72 and the crossings on 66..72."""
    import mc_redraw as M
    import prod_common as P
    name = run or RUN.format(analysis=analysis, set=label, R=R)
    cfg, src, check = make_cfg(name)
    os.environ["DARKSIRENS_ZMAX"] = repr(float(cfg["zmax"]))
    import prod_model as PM
    PM.Model(cfg).import_core()
    import jax
    jax.config.update("jax_enable_x64", True)
    obs, attrs, nobs, _ = M.read_event_set(cfg["events_file"])
    g = M.generator()
    t0 = time.time()
    bl = []
    for b in range(blocks):
        rng = np.random.default_rng(np.random.SeedSequence(list(cfg["redraw_key"]) + [b]))
        p = g.posterior_samples_v3(rng, obs, P.REDRAW_BLOCK)
        bl.append({k: p[k].reshape(nobs, P.REDRAW_BLOCK) for k in ("m1det", "dL")})
    m1det = np.concatenate([b["m1det"] for b in bl], axis=1)
    dL = np.concatenate([b["dL"] for b in bl], axis=1)
    e = EDGE[cfg["population_family"]]
    hc = h0_cross(m1det.ravel(), dL.ravel(), e, float(cfg["Om0"])).reshape(m1det.shape)
    # m1src < edge  <=>  H0 > hc (z grows with H0 at fixed dL)
    res = {"run": name, "samples": int(m1det.shape[1]), "edge": e, "seconds": time.time() - t0}
    n_in = {h: (hc < h).sum(1) for h in (lo, 67.74, hi)}
    ncross = ((hc > lo) & (hc < hi)).sum(1)
    rows = []
    for i in np.where(ncross > 0)[0]:
        rows.append({"i": int(i), f"n_in_{lo:g}": int(n_in[lo][i]), "n_in_67.74": int(n_in[67.74][i]), f"n_in_{hi:g}": int(n_in[hi][i]),
                     f"n_cross_{lo:g}_{hi:g}": int(ncross[i]), "cross_h0_first": sorted(float(x) for x in hc[i][(hc[i] > lo) & (hc[i] < hi)])[:12]})
    rows.sort(key=lambda r: r["n_in_67.74"])
    res["mergers_with_crossings"] = rows
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"edge_{label}_r{R}" + ("" if run is None else "_" + run.split("/")[0])
    (OUT / f"{stem}.json").write_text(json.dumps(res, indent=1))
    print(f"[edge] {name}: {len(rows)} mergers have samples crossing m1src = {e} between H0 {lo:g} and {hi:g} ({res['seconds']:.0f} s)")
    for r in rows[:15]:
        print("   ", r)


# ============================================================================ single report
def single_report(analysis, label, R, idx, a=67.0, b=70.0):
    """ln L_i(H0) of one merger against the number of samples: steps (largest rise between neighbouring nodes), the
    preference D = ln L_i(b) - ln L_i(a), its spread over independent subsets, the effective sample number at the step
    nodes; the first 8000 samples checked against full's ln L_i at the shared nodes."""
    z = np.load(OUT / f"single_{analysis}_{label}_r{R}_i{idx}.npz")
    H = z["H0"]
    ia, ib = int(np.argmin(np.abs(H - a))), int(np.argmin(np.abs(H - b)))
    reg = H[(H >= 66.0) & (H <= 72.0)]
    names = sorted(k[4:] for k in z.files if k.startswith("lnL_"))
    out = {"merger": idx, "H0_pref": [float(H[ia]), float(H[ib])], "curves": {}}

    def smooth_resid(h, y):
        c = np.polyfit(h - 69.0, y, 3)
        return y - np.polyval(c, h - 69.0)

    for nm in names:
        y = z["lnL_" + nm]
        e = z["ess_" + nm]
        d = np.diff(y)
        r = smooth_resid(H, y)
        j = int(np.argmax(d))
        out["curves"][nm] = {"D": float(y[ib] - y[ia]), "largest_rise": float(d[j]), "at": [float(H[j]), float(H[j + 1])],
                             "rises_gt_0.1": int((d > 0.1).sum()), "rms_resid_cubic": float(np.sqrt(np.mean(r ** 2))),
                             "ess_67.74": float(e[int(np.argmin(np.abs(H - 67.74)))]),
                             "ess_min_max_66_72": [float(e.min()), float(e.max())]}
    for size in ("8000", "32000"):
        sub = [nm for nm in names if nm.startswith(f"sub{size}_")]
        if len(sub) >= 2:
            D = np.array([out["curves"][nm]["D"] for nm in sub])
            out[f"D_over_sub{size}"] = {"n": len(sub), "mean": float(D.mean()), "sd": float(D.std(ddof=1)),
                                         "sd_of_mean": float(D.std(ddof=1) / np.sqrt(len(sub))), "values": D.tolist(),
                                         "largest_rise_median": float(np.median([out["curves"][nm]["largest_rise"] for nm in sub]))}
    fp = OUT / f"full_{analysis}_{label}_r{R}.npz"
    if fp.exists():
        f = np.load(fp)
        chk = []
        for jf, h in enumerate(f["H0"]):
            js = int(np.argmin(np.abs(H - h)))
            if abs(H[js] - h) < 1e-9:
                chk.append({"H0": float(h), "full": float(f["ev"][jf, idx]), "single_first8000": float(z["lnL_first8000"][js]),
                            "diff": float(z["lnL_first8000"][js] - f["ev"][jf, idx])})
        out["check_against_full"] = chk
    p = OUT / f"single_{analysis}_{label}_r{R}_i{idx}_report.json"
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "curves"}, indent=1))
    for nm in ("first8000", "first16000", "first32000") + tuple(n for n in names if n.startswith("all")):
        if nm in out["curves"]:
            print(nm, out["curves"][nm])


# ============================================================================ hybrid line (question 5 on CPU)
def hybrid_report(analysis, label, R, tag, f0=0.30):
    """The stored 8000-sample H0 line at f0 with the mergers of a full ... --mergers --nsamp 32000 run replaced by their
    32000-sample terms: lnL_hyb = lnL_stored - sum_K ev_i(8000) + sum_K ev_i(32000) (the selection term, sel_corr, is the
    stored one: it does not depend on the PE samples); pe_var likewise; the half-variance corrected lines; the conditional
    posterior along H0 (Akima-refined as the run summaries) of both. ev_i(8000) is recomputed from the first 8000 of the
    32000 captured samples and checked against full's 8000-sample run at its nodes."""
    import h5py
    from scipy.interpolate import Akima1DInterpolator
    import prod_common as P
    z = np.load(OUT / f"full_{analysis}_{label}_r{R}{tag}.npz")
    H, K = z["H0"], z["index"]
    name = RUN.format(analysis=analysis, set=label, R=R)
    with h5py.File(RUNS / name / "grid" / "fine_widened.h5", "r") as F:
        ax, af = F["axis_H0"][()], F["axis_f_agn"][()]
        j = int(np.argmin(np.abs(af - f0)))
        st = {k: F[k][()][:, j] for k in ("lnL", "pe_var", "n_eff", "sel_corr")}
    idx = [int(np.argmin(np.abs(ax - h))) for h in H]
    if max(abs(ax[i] - h) for i, h in zip(idx, H)) > 1e-6:
        raise SystemExit("[fatal] hybrid nodes are not on the stored grid")
    st = {k: v[idx] for k, v in st.items()}
    out = {"run": name, "mergers_replaced": K.tolist(), "n_replaced": int(K.size), "H0": H.tolist(), "f": f0}
    fp = OUT / f"full_{analysis}_{label}_r{R}.npz"
    if fp.exists():
        f8 = np.load(fp)
        d = []
        for jf, h in enumerate(f8["H0"]):
            jj = np.where(np.abs(H - h) < 1e-9)[0]
            if jj.size:
                d.append(float(np.max(np.abs(z["ev8000"][jj[0]] - f8["ev"][jf][K]))))
        out["check_ev8000_vs_full_max_abs"] = max(d) if d else None
    d_ev = (z["ev"] - z["ev8000"]).sum(1)
    lnL8, lnLh = st["lnL"], st["lnL"] + d_ev
    out["lnL_8000_minus_max"] = (lnL8 - lnL8.max()).tolist()
    out["lnL_hybrid_minus_max"] = (lnLh - lnLh.max()).tolist()
    out["d_ev_sum"] = d_ev.tolist()
    out["d_ev_by_merger_at_67.0_70.0"] = {}
    ia, ib = int(np.argmin(np.abs(H - 67.0))), int(np.argmin(np.abs(H - 70.0)))
    dd = (z["ev"][ib] - z["ev"][ia]) - (z["ev8000"][ib] - z["ev8000"][ia])
    for k in np.argsort(-np.abs(dd))[:8]:
        out["d_ev_by_merger_at_67.0_70.0"][str(int(K[k]))] = float(dd[k])
    out["pref_70_minus_67"] = {"8000": float(lnL8[ib] - lnL8[ia]), "hybrid": float(lnLh[ib] - lnLh[ia])}
    lat = np.linspace(H[0], H[-1], (len(H) - 1) * 16 + 1)
    for nm, y in (("8000", lnL8), ("hybrid", lnLh)):
        sm = P.summarize_1d(lat, Akima1DInterpolator(H, y)(lat))
        lv = Akima1DInterpolator(H, y)(lat)
        w = np.exp(lv - lv.max())
        cdf = np.concatenate([[0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(lat))])
        cdf /= cdf[-1]
        sm["cdf_at_67.74"] = float(np.interp(67.74, lat, cdf))
        out[f"posterior_along_H0_{nm}"] = sm
        print(f"[hybrid] {name} {nm}: median {sm['median']:.3f} 90% [{sm['ci90'][0]:.3f}, {sm['ci90'][1]:.3f}] "
              f"CDF(67.74) {sm['cdf_at_67.74']:.4f}; lnL(70)-lnL(67) {out['pref_70_minus_67'][nm]:+.3f}")
    print("[hybrid] H0   8000   hybrid (relative to max)")
    for h, a, b in zip(H, out["lnL_8000_minus_max"], out["lnL_hybrid_minus_max"]):
        print(f"   {h:6.2f} {a:+8.3f} {b:+8.3f}")
    p = A16 / "tests" / "results" / f"obslike_steps_hybrid_{analysis}_r{R}.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    print("wrote", p, "| check ev8000:", out.get("check_ev8000_vs_full_max_abs"), "| largest per-merger changes of the 70-67 preference:",
          out["d_ev_by_merger_at_67.0_70.0"])


# ============================================================================ census report
def census_report(analysis, labels_reals, tag="_census", h_in=67.74):
    """Per realization (outputs of full ... --h0 67.54,67.74,67.94 --limit-inj N --tag _census): per merger the effective
    sample number of its sum at the input (mean of the two side nodes' and the input node's), how many mergers fall below
    5, 20, 100; how many of their samples lie inside the power law's support (m1src < m_max) at the input; their summed
    contribution to d lnL/dH0 at the input (central difference over the side nodes) against the sum over all mergers."""
    import h5py
    res = {}
    for label, reals in labels_reals:
        per = {}
        for R in reals:
            p = OUT / f"full_{analysis}_{label}_r{R}{tag}.npz"
            if not p.exists():
                print(f"[census] missing {p.name}")
                continue
            z = np.load(p)
            H = z["H0"]
            j0 = int(np.argmin(np.abs(H - h_in)))
            ess = z["ess"][j0]
            ns = None
            info = json.loads((OUT / f"full_{analysis}_{label}_r{R}{tag}.json").read_text())
            ns = int(info["nsamp"])
            slope = (z["ev"][-1] - z["ev"][0]) / (H[-1] - H[0])
            n_in = z["n_in"][j0]
            ev_file = json.loads((RUNS / info["run"] / "config.json").read_text())["events_file"]
            with h5py.File(ev_file, "r") as f:
                ht = np.asarray(f["truth/host_type"][:])
                tm1 = np.asarray(f["truth/m1src"][:])
            row = {"n_mergers": int(ess.size), "samples": ns, "edge_m1": info["edge_m1"]}
            for c in (5, 20, 100):
                lo = ess < c
                row[f"n_ess_lt_{c}"] = int(lo.sum())
                row[f"sum_slope_ess_lt_{c}"] = float(slope[lo].sum())
                row[f"sum_pe_var_ess_lt_{c}"] = float(z["var"][j0][lo].sum())
                row[f"median_frac_inside_ess_lt_{c}"] = float(np.median(n_in[lo] / ns)) if lo.any() else None
                row[f"n_ess_lt_{c}_with_lt_1pct_inside"] = int((lo & (n_in < 0.01 * ns)).sum())
            row["sum_slope_all"] = float(slope.sum())
            row["pe_var_all"] = float(z["var"][j0].sum())
            row["stored_total_slope_at_input_f0.30"] = None
            for fn in ("fine_widened.h5", "fine.h5"):
                gp = RUNS / info["run"] / "grid" / fn
                if gp.exists():
                    with h5py.File(gp, "r") as G:
                        ax, af = G["axis_H0"][()], G["axis_f_agn"][()]
                        jf = int(np.argmin(np.abs(af - 0.30)))
                        near = np.abs(ax - h_in) <= 0.6
                        if abs(af[jf] - 0.30) < 0.011 and near.sum() >= 2:
                            row["stored_total_slope_at_input_f0.30"] = float(np.polyfit(ax[near], G["lnL"][()][near, jf], 1)[0])
                            row["stored_slope_from"] = [fn, float(af[jf]), [float(ax[near][0]), float(ax[near][-1])]]
                            break
            row["sd_slope_all"] = float(slope.std())
            row["n_lt_1pct_inside"] = int((n_in < 0.01 * ns).sum())
            row["n_lt_10pct_inside"] = int((n_in < 0.10 * ns).sum())
            row["sum_slope_lt_1pct_inside"] = float(slope[n_in < 0.01 * ns].sum())
            row["ess_quantiles_1_5_50pct"] = [float(x) for x in np.quantile(ess, [0.01, 0.05, 0.5])]
            o = np.argsort(ess)[:8]
            row["lowest_ess"] = [{"i": int(i), "ess": float(ess[i]), "n_inside": int(n_in[i]), "slope": float(slope[i]),
                                  "host": int(ht[i]), "true_m1src": float(tm1[i]), "pe_var": float(z["var"][j0, i])} for i in o]
            per[str(R)] = row
            print(f"[census] {label} r{R}: ESS<5 {row['n_ess_lt_5']} (<1% inside: {row['n_ess_lt_5_with_lt_1pct_inside']}), "
                  f"<20 {row['n_ess_lt_20']} ({row['n_ess_lt_20_with_lt_1pct_inside']}), <100 {row['n_ess_lt_100']} "
                  f"({row['n_ess_lt_100_with_lt_1pct_inside']}); mergers with <1% inside {row['n_lt_1pct_inside']}; slope sum "
                  f"ESS<5 {row['sum_slope_ess_lt_5']:+.3f}, <20 {row['sum_slope_ess_lt_20']:+.3f}, <100 {row['sum_slope_ess_lt_100']:+.3f}, "
                  f"all {row['sum_slope_all']:+.3f}; lowest {[(x['i'], round(x['ess'], 2), x['n_inside']) for x in row['lowest_ess'][:4]]}", flush=True)
        res[label] = per
    p = A16 / "tests" / "results" / f"obslike_steps_census_{analysis}.json"
    p.write_text(json.dumps(res, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("pos", nargs="*")
    ap.add_argument("--h0", default=None)
    ap.add_argument("--f", type=float, default=0.30)
    ap.add_argument("--limit-inj", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--nsamp", type=int, default=None)
    ap.add_argument("--blocks", type=int, default=None)
    ap.add_argument("--limit-events", type=int, default=None)
    ap.add_argument("--mergers", default=None, help="comma list of merger indices, or a .json list")
    a = ap.parse_args()
    if a.mode == "full":
        an, label, R = a.pos[0], a.pos[1], int(a.pos[2])
        h0s = [float(x) for x in (a.h0 or "68.2,68.4,69.0,69.2").split(",")]
        mg = None
        if a.mergers:
            mg = json.loads(Path(a.mergers).read_text()) if a.mergers.endswith(".json") else [int(x) for x in a.mergers.split(",")]
        full(an, label, R, h0s, a.f, a.limit_inj, a.tag, a.nsamp, a.limit_events, mg)
    elif a.mode == "single":
        an, label, R, idx = a.pos[0], a.pos[1], int(a.pos[2]), int(a.pos[3])
        single(an, label, R, idx, a.blocks or 64, a.h0 or "66,72,0.02", a.f)
    elif a.mode == "single_report":
        single_report(a.pos[0], a.pos[1], int(a.pos[2]), int(a.pos[3]))
    elif a.mode == "hybrid_report":
        hybrid_report(a.pos[0], a.pos[1], int(a.pos[2]), a.tag)
    elif a.mode == "census_report":
        sets = [(x.split(":")[0], [int(r) for r in x.split(":")[1].split(",")]) for x in a.pos[1:]]
        census_report(a.pos[0], sets)
    elif a.mode == "edge":
        rng = [float(x) for x in (a.h0 or "66,72").split(",")]
        edge(a.pos[0], int(a.pos[1]), a.blocks or 4, run=a.pos[2] if len(a.pos) > 2 else None, lo=rng[0], hi=rng[1])
    else:
        raise SystemExit(f"unknown mode {a.mode}")
