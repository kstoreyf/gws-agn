#!/usr/bin/env python
"""Why the peak_f0p30_s1 two-population family scatters in H0 more than its widths allow: CPU evaluations (25 Nov).

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH= <core venv python> tests/hd_scatter_eval.py MODE RUN [options]

RUN is a run directory under runs/. Points are H0:f pairs; f may be 'fmax' (the f_AGN node of largest posterior weight of
the run's fine pass, obslike_steps_halfwidth.stored_line) or 'in' (the stored f node nearest 0.30).

nodes RUN --pts 64:fmax,67.75:fmax,... [--redraw K] [--tag T]
    The run's model (run-list arguments, resolved configuration compared with config.json; with --redraw K the driver's
    --redraw-index K, i.e. an independent production redraw of every merger), the first 5000 detected injections only
    (the per-merger terms do not involve them), evaluate_events at each point with the per-sample PE log weights captured
    (obslike_steps_merger.install_capture): per merger ln L_i, sigma_i^2, effective sample number. Check: sum_i ln L_i
    against the stored lnL - sel_corr at the same node, pe_var against the stored pe_var.
    -> tests/results/hd_scatter/nodes_<slug><tag>.npz

hybrid RUN --ess-from NODES_NPZ --ess-point 64:fmax --ess-cut 50 --mult 4 --top i,j,... --top-mult 32 [--step 0.5] [--tag T]
    The stored H0 line at fmax (every node at spacing --step) with (a) every merger whose effective sample number at
    --ess-point (from NODES_NPZ) is below --ess-cut re-evaluated at mult x 8000 samples, (b) the --top mergers at
    top-mult x 8000: blocks of 2000 of the production redraw (whole set per block with the run's key, rows kept; the
    first 8000 are the run's), bound alone with 5000 injections; per merger ln L_i from all samples and from the first
    8000 (obslike_steps_halfwidth.per_merger). lnL_hyb = lnL_stored + sum (ln L_i(M) - ln L_i(8000)), pe_var likewise.
    -> tests/results/hd_scatter/hybrid_<slug><tag>.npz

sel RUN --pts ... [--inj FILE[,FILE]] [--tag T]
    The selection term alone (first 5 mergers, every detected injection of the run's lane or of FILE): the per-injection
    log weights captured (core's compute_selection_term replaced, in this process only, by a copy that scans the same
    batches and also hands each batch's weights to a host callback; log mu is checked against the stored log_mu).
    -> tests/results/hd_scatter/sel_<slug><tag>.npz (log weights at each point, float32)

Nothing in scripts/ or runs/ is written; scripts/ modules are imported only.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import obslike_steps_merger as S  # noqa: E402  (adds scripts/ to the path)
import obslike_steps_halfwidth as HW  # noqa: E402

A16 = S.A16
RUNS = S.RUNS
OUT = A16 / "tests" / "results" / "hd_scatter"
F_IN = 0.30


def slug(name):
    p = name.split("/")
    return f"{p[0]}_{p[3]}_{p[4]}"


def resolve_pts(name, spec):
    import h5py
    line = HW.stored_line(name)
    g = RUNS / name / "grid" / line["file"]
    with h5py.File(g, "r") as F:
        af = np.asarray(F["axis_f_agn"][()], float)
    f_in = float(af[int(np.argmin(np.abs(af - F_IN)))])
    out = []
    for tok in spec.split(","):
        h, f = tok.split(":")
        f = line["f"] if f == "fmax" else (f_in if f == "in" else float(f))
        out.append((float(h), float(f)))
    return out, line


def make_cfg(name, redraw=0):
    import prod_driver as PD
    args, src = S.runlist_args(name)
    if redraw:
        args = list(args) + ["--redraw-index", str(redraw)]
    cfg = PD.resolve(PD.parse_args(args))
    if cfg["run_name"] != name:
        raise SystemExit(f"[fatal] resolved run name {cfg['run_name']} != {name}")
    sc = json.loads((RUNS / name / "config.json").read_text())
    keys = [k for k in S.CMP_KEYS if not (redraw and k == "redraw_key")]
    sc = dict(sc)
    if sc.get("injections_files") is None and sc.get("injections_file"):      # older configs: the single lane only
        sc["injections_files"] = [sc["injections_file"]]
    diff = {k: (sc.get(k), cfg.get(k)) for k in keys if sc.get(k) != cfg.get(k)}
    if diff:
        raise SystemExit(f"[fatal] resolved configuration differs from the stored run's: {diff}")
    if redraw:
        exp = list(sc["redraw_key"][:-1]) + [int(redraw)]
        if list(cfg["redraw_key"]) != exp:
            raise SystemExit(f"[fatal] redraw key {cfg['redraw_key']} != {exp}")
    return cfg, src


def build(cfg, bind=True):
    os.environ["DARKSIRENS_ZMAX"] = repr(float(cfg["zmax"]))
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import prod_model as PM
    m = PM.make_model(cfg)
    m.import_core()
    import jax
    jax.config.update("jax_enable_x64", True)
    m.build()
    if bind:
        m.bind()
    return m


# ============================================================================ nodes
def nodes(name, spec, redraw=0, tag=""):
    t0 = time.time()
    pts, line = resolve_pts(name, spec)
    cfg, src = make_cfg(name, redraw)
    cfg["limit_injections"] = 5000
    m = build(cfg)
    S.install_capture()
    n, ns = m.n_events, m.nsamp
    t_setup = time.time() - t0
    print(f"[nodes] {name} redraw {redraw}: built + bound in {t_setup:.0f} s; {n} mergers x {ns}; f_max {line['f']}", flush=True)
    P = len(pts)
    rec = {"H0": np.array([p[0] for p in pts]), "f": np.array([p[1] for p in pts]), "ev": np.full((P, n), np.nan),
           "var": np.full((P, n), np.nan), "ess": np.full((P, n), np.nan), "sum_ev": np.full(P, np.nan),
           "pe_var": np.full(P, np.nan), "stored_lnL_minus_selcorr": np.full(P, np.nan), "stored_pe_var": np.full(P, np.nan),
           "stored_sel_corr": np.full(P, np.nan), "stored_log_mu": np.full(P, np.nan), "stored_n_eff": np.full(P, np.nan),
           "seconds": np.full(P, np.nan)}
    what = []                                   # per point the normalised per-sample weights (float32), for the pairs below
    rec["mcvar_pair"] = np.full((P, P, n), np.nan)
    for j, (h, f) in enumerate(pts):
        tc = time.time()
        r = m.evaluate_events({"H0": h, "f_agn": f})
        ldw = S.captured(n * ns).reshape(n, ns)
        what.append(np.exp(ldw - S.lse(ldw)[:, None]).astype(np.float32))
        for k in range(j):                      # Monte Carlo variance of ln L_i(point j) - ln L_i(point k), common samples
            d = what[j].astype(np.float64) - what[k]
            rec["mcvar_pair"][j, k] = rec["mcvar_pair"][k, j] = (d * d).sum(1) * ns / (ns - 1.0)
        rec["ev"][j], rec["var"][j] = r["event_log_evidence"], r["event_mc_variance"]
        rec["ess"][j] = S.ess_of(ldw)
        rec["sum_ev"][j], rec["pe_var"][j] = float(np.sum(r["event_log_evidence"])), r["pe_var"]
        st = S.stored_node(name, h, f)
        if st:
            rec["stored_lnL_minus_selcorr"][j] = st["lnL"] - st["sel_corr"]
            rec["stored_pe_var"][j], rec["stored_sel_corr"][j] = st["pe_var"], st["sel_corr"]
            rec["stored_log_mu"][j], rec["stored_n_eff"][j] = st["log_mu"], st["n_eff"]
        rec["seconds"][j] = time.time() - tc
        print(f"[nodes] H0 {h:g} f {f:g}: sum ev {rec['sum_ev'][j]:.6f} (stored lnL - sel_corr {rec['stored_lnL_minus_selcorr'][j]:.6f}, "
              f"diff {rec['sum_ev'][j] - rec['stored_lnL_minus_selcorr'][j]:+.2e}); pe_var {r['pe_var']:.5f} (stored "
              f"{rec['stored_pe_var'][j]:.5f}) ({rec['seconds'][j]:.0f} s)", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / f"nodes_{slug(name)}{tag}.npz", **rec,
                            meta=json.dumps({"run": name, "args_from": src, "redraw": redraw, "f_max": line["f"],
                                             "setup_seconds": t_setup, "events_file": cfg["events_file"], "nsamp": ns}))
    print(f"[nodes] -> {OUT}/nodes_{slug(name)}{tag}.npz ({time.time() - t0:.0f} s)", flush=True)


# ============================================================================ hybrid
def draw_rows(cfg, idxs, blocks):
    import mc_redraw as M
    import prod_common as P
    obs, attrs, nobs, _ = M.read_event_set(cfg["events_file"])
    g = M.generator()
    rows = {}
    for b in range(blocks):
        rng = np.random.default_rng(np.random.SeedSequence(list(cfg["redraw_key"]) + [b]))
        p = g.posterior_samples_v3(rng, obs, P.REDRAW_BLOCK)
        for k, v in p.items():
            rows.setdefault(k, []).append(v.reshape(nobs, P.REDRAW_BLOCK)[idxs])
        if b in (0, 3, 15, 63) or b == blocks - 1:
            print(f"[draw] block {b + 1}/{blocks}", flush=True)
    return {k: np.concatenate(v, axis=1) for k, v in rows.items()}, attrs        # (nI, blocks*2000)


def bind_rows(m, raw, attrs, ncols):
    import mc_redraw as M
    nI = next(iter(raw.values())).shape[0]
    flat = {k: np.ascontiguousarray(v[:, :ncols]).ravel() for k, v in raw.items()}
    post = M.postprocess(flat, nI, ncols)
    sub = {k: v for k, v in post.items() if k in M.STORE_COLUMNS}
    store = M.load_store(m.ds, sub, attrs, nI, ncols)
    m.cfg["nsamp"] = ncols
    m.bind(events_store=store)


def hybrid(name, ess_from, ess_point, ess_cut, mult, top, top_mult, step, tag=""):
    t0 = time.time()
    line = HW.stored_line(name)
    H_all = line["H0"]
    f0 = line["f"]
    z = np.load(ess_from)
    (hp, fp), = resolve_pts(name, ess_point)[0]
    keep = np.isclose(np.round((H_all - hp) / step) * step, H_all - hp)     # nodes at spacing step through the ess point
    H = H_all[keep]
    jp = int(np.where(np.isclose(z["H0"], hp) & np.isclose(z["f"], fp))[0][0])
    K = np.where(z["ess"][jp] < ess_cut)[0]
    top = np.asarray(sorted(set(int(i) for i in top)), int)
    K4 = np.array([i for i in K if i not in set(top)], int)
    print(f"[hybrid] {name}: f {f0}; {K.size} mergers with ESS < {ess_cut} at {hp}:{fp} ({K4.size} not in the top list); "
          f"top {top.tolist()} at {top_mult}x; {H.size} nodes", flush=True)
    cfg, src = make_cfg(name)
    N = int(cfg["nsamp"])
    cfg["limit_injections"] = 5000
    m = build(cfg, bind=False)
    S.install_capture()
    union = np.array(sorted(set(K4.tolist()) | set(top.tolist())), int)
    nblk = (top_mult if top.size else mult) * N // 2000
    td = time.time()
    raw, attrs = draw_rows(cfg, union, nblk)
    print(f"[hybrid] drew {nblk} blocks in {time.time() - td:.0f} s", flush=True)
    pos = {int(i): k for k, i in enumerate(union)}
    edge, om0 = S.EDGE[cfg["population_family"]], float(cfg["Om0"])
    res = {"H0": H, "f": f0, "K4": K4, "top": top, "meta": json.dumps({"run": name, "args_from": src, "ess_from": str(ess_from),
           "ess_point": [hp, fp], "ess_cut": ess_cut, "mult": mult, "top_mult": top_mult, "step": step, "N": N})}
    for grp, idx, mm in (("g4", K4, mult), ("top", top, top_mult)):
        if idx.size == 0:
            continue
        sel = [pos[int(i)] for i in idx]
        bind_rows(m, {k: v[sel] for k, v in raw.items()}, attrs, mm * N)
        print(f"[hybrid] {grp}: {m.n_events} mergers bound at {m.nsamp} samples", flush=True)
        rec = HW.scan(m, H, f0, N, edge, om0)
        for k in ("ev_M", "var_M", "ess_M", "ev_N", "var_N", "ess_N"):
            res[f"{grp}_{k}"] = rec[k]
        # check: the first N against the run's own terms at the nodes of the ess_from npz on this line
        for j, h in enumerate(H):
            jj = np.where(np.isclose(z["H0"], h) & np.isclose(z["f"], f0))[0]
            if jj.size:
                d = float(np.max(np.abs(rec["ev_N"][j] - z["ev"][jj[0]][idx])))
                print(f"[hybrid] {grp} check H0 {h:g}: first {N} vs the run's ln L_i max |diff| {d:.2e}", flush=True)
                res[f"{grp}_check_H0_{h:g}"] = d
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / f"hybrid_{slug(name)}{tag}.npz", **res)
    print(f"[hybrid] -> {OUT}/hybrid_{slug(name)}{tag}.npz ({time.time() - t0:.0f} s)", flush=True)


# ============================================================================ sel
SELCAP = {"blocks": {}}


def _selcb(start, ldw):
    SELCAP["blocks"][int(start)] = np.array(ldw, dtype=float)


def install_sel_capture():
    import jax
    import jax.numpy as jnp
    from jax import lax
    import darksirens.likelihood.hierarchical as Hh
    import darksirens.selection.gw as G
    if getattr(Hh, "_hd_scatter_sel_capture", False):
        return

    def patched(gw_sel, em_catalog_sel, log_weight_fn, Ndraw, nEvents, sel_batch_size=None, sky_log_weight_fn=None):
        bs = int(sel_batch_size)
        gp, _ = G.pad_gw_event_to_multiple(gw_sel, bs)
        nb = gp.dL.shape[0] // bs

        def body(_, bi):
            start = bi * bs
            sl = lambda a: lax.dynamic_slice_in_dim(a, start, bs)  # noqa: E731
            a = (sl(gp.m1det), sl(gp.q), sl(gp.dL), sl(gp.chieff), sl(gp.pixels), sl(gp.prior_wt), em_catalog_sel)
            ldw = log_weight_fn(*a) if gp.spin is None else log_weight_fn(*a, spin=sl(gp.spin))
            if sky_log_weight_fn is not None:
                ldw = ldw + sky_log_weight_fn(sl(gp.nx), sl(gp.ny), sl(gp.nz), sl(gp.dL))
            valid = sl(gp.valid) & (sl(gp.prior_wt) > 0.0)
            ldw = jnp.where(valid & jnp.isfinite(ldw), ldw, -jnp.inf)
            jax.debug.callback(_selcb, start, ldw)
            return None, (G.logsumexp_neginf_safe(ldw), G.logsumexp_neginf_safe(2.0 * ldw))

        _, (l1, l2) = lax.scan(body, None, jnp.arange(nb))
        return G._lse_to_log_mu_neff(G.logsumexp_neginf_safe(l1), G.logsumexp_neginf_safe(l2), Ndraw)

    Hh.compute_selection_term = patched
    Hh._hd_scatter_sel_capture = True


def sel(name, spec, inj=None, tag=""):
    t0 = time.time()
    pts, line = resolve_pts(name, spec)
    cfg, src = make_cfg(name)
    cfg["limit_events"] = 5
    if inj:
        cfg["injections_files"] = inj.split(",")
    m = build(cfg, bind=False)
    install_sel_capture()
    m.bind()
    n_det = int(m.inj_store.n_injections)
    print(f"[sel] {name}: injections {cfg['injections_files']}: {n_det} detected of {m.n_draw:g}; bound in {time.time() - t0:.0f} s", flush=True)
    rec = {"H0": np.array([p[0] for p in pts]), "f": np.array([p[1] for p in pts]), "n_draw": m.n_draw,
           "log_mu": np.full(len(pts), np.nan), "n_eff": np.full(len(pts), np.nan), "log_mu_numpy": np.full(len(pts), np.nan),
           "stored_log_mu": np.full(len(pts), np.nan), "stored_n_eff": np.full(len(pts), np.nan)}
    L = []
    for j, (h, f) in enumerate(pts):
        tc = time.time()
        SELCAP["blocks"] = {}
        r = m.evaluate({"H0": h, "f_agn": f})
        nb = max(SELCAP["blocks"]) + SELCAP["blocks"][max(SELCAP["blocks"])].size
        x = np.full(nb, -np.inf)
        for s_, v in SELCAP["blocks"].items():
            x[s_:s_ + v.size] = v
        x = x[:n_det]
        L.append(x.astype(np.float64))
        rec["log_mu"][j], rec["n_eff"][j] = r["log_mu"], r["n_eff"]
        rec["log_mu_numpy"][j] = S.lse(x) - np.log(m.n_draw)
        st = S.stored_node(name, h, f) if not inj else None
        if st:
            rec["stored_log_mu"][j], rec["stored_n_eff"][j] = st["log_mu"], st["n_eff"]
        print(f"[sel] H0 {h:g} f {f:g}: log_mu {r['log_mu']:.10f} (numpy {rec['log_mu_numpy'][j]:.10f}; stored "
              f"{rec['stored_log_mu'][j]:.10f}, diff {r['log_mu'] - rec['stored_log_mu'][j]:+.2e}) n_eff {r['n_eff']:.1f} "
              f"(stored {rec['stored_n_eff'][j]:.1f}) ({time.time() - tc:.0f} s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"sel_{slug(name)}{tag}.npz", **rec, ldw=np.stack(L),
             meta=json.dumps({"run": name, "injections_files": cfg["injections_files"], "n_detected": n_det}))
    print(f"[sel] -> {OUT}/sel_{slug(name)}{tag}.npz ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("nodes", "hybrid", "sel"))
    ap.add_argument("name")
    ap.add_argument("--pts", default="")
    ap.add_argument("--redraw", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--ess-from", default=None)
    ap.add_argument("--ess-point", default="64:fmax")
    ap.add_argument("--ess-cut", type=float, default=50.0)
    ap.add_argument("--mult", type=int, default=4)
    ap.add_argument("--top", default="")
    ap.add_argument("--top-mult", type=int, default=32)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--inj", default=None)
    a = ap.parse_args()
    if a.mode == "nodes":
        nodes(a.name, a.pts, a.redraw, a.tag)
    elif a.mode == "hybrid":
        top = [int(x) for x in a.top.split(",")] if a.top else []
        hybrid(a.name, a.ess_from, a.ess_point, a.ess_cut, a.mult, top, a.top_mult, a.step, a.tag)
    else:
        sel(a.name, a.pts, a.inj, a.tag)
