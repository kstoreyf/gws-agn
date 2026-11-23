#!/usr/bin/env python
"""Validation of --redraw-mixture support on CPU (the copies in tests/mixture_redraw; nothing in scripts/ or runs/ written).

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH= <core venv python> tests/mixture_redraw/mix_eval.py MODE ...

merger RUN IDX [--streams 8] [--nsamp N] [--window lo,hi] [--extra-h0 lo,hi,step]
    Merger IDX of RUN (its own model; hosts_known all-AGN runs as their AGN subset) bound alone, K = --streams independent
    redraws: stream k uses the run's redraw key with redraw_index k (k = 0 is the production key), drawn for the whole set
    per block as the production redraw, the merger's row kept; each stream once as the production redraw (off) and once
    with the mixture (on: the same production row, the restricted share refilled from stream k's own restricted stream).
    The 2K copies are bound together, evaluated at the converged reference's H0 nodes (tests/results/obslike_steps*):
    ln L_i(H0), effective sample number (captured per-sample weights), against the reference.
    -> results/merger_<slug>_i<IDX>_s<N>.npz / .json

line RUN [--nsamp N] [--all] [--tag T]
    The stored fine line of RUN at its f_AGN node (obslike_steps_halfwidth.stored_line). Default: the mergers the mixture
    refills (T) bound twice (production and mixture), few injections; ln L_mix(H0) = ln L_stored + sum_T [ev_on - ev_off],
    pe_var likewise (every other merger's samples are bitwise the production ones). --all: every merger on and off at
    --nsamp (for a sample number the run was not made with): ln L(H0) = sel_corr_stored + sum_i ev_i (core's ln L is
    exactly sum_i ev_i + sel_corr; checked on the stored runs). Half-variance corrected lines, Akima-refined as the run
    summaries; 90% interval, half-width; the steps (largest rise against a smooth fit); per-merger on - off differences.
    -> results/line_<slug><tag>.npz / .json

offcheck RUN [--nodes a,b]
    The copy with the option off, the run's full configuration and injections, at stored grid nodes: lnL, pe_var, n_eff,
    log_mu, sel_corr against the stored grid.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A16 = HERE.parents[1]
sys.path.insert(0, str(HERE))
import prod_common as P  # noqa: E402   the copies, imported first so that every later import gets them
import prod_model as PM  # noqa: E402
import prod_driver as PD  # noqa: E402
import redraw_mixture as RM  # noqa: E402

sys.path.insert(0, str(A16 / "tests"))
import obslike_steps_merger as S  # noqa: E402   (puts scripts/ on the path: the copies are already in sys.modules)
import obslike_steps_halfwidth as HW  # noqa: E402
import mc_redraw as M  # noqa: E402

for mod in (P, PM, PD, RM, S._P):
    if not str(mod.__file__).startswith(str(HERE)):
        raise SystemExit(f"[fatal] {mod.__name__} imported from {mod.__file__}, not the copies")
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)


def key_k(cfg, k):
    key = list(cfg["redraw_key"])
    key[-1] = int(k)            # P.redraw_key's last entry is the redraw index
    return key


def production_raw(obs, nobs, ns, key, rows=None):
    g = M.generator()
    bl = []
    for b in range(ns // P.REDRAW_BLOCK):
        rng = np.random.default_rng(np.random.SeedSequence(list(key) + [b]))
        p = g.posterior_samples_v3(rng, obs, P.REDRAW_BLOCK)
        bl.append({k: v.reshape(nobs, P.REDRAW_BLOCK) if rows is None else v.reshape(nobs, P.REDRAW_BLOCK)[rows]
                   for k, v in p.items()})
    return {k: np.concatenate([b[k] for b in bl], axis=1) for k in bl[0]}


def post_of(raw, nrows, ns, r=None):
    post = M.postprocess({k: np.ascontiguousarray(v).ravel() for k, v in raw.items()}, nrows, ns)
    if r is not None:
        post["p_pe"] = RM.corrected_p_pe(post["p_pe"], r, nrows, ns)
    return {k: v for k, v in post.items() if k in M.STORE_COLUMNS}


def model_for(name, nsamp=None, limit_inj=5000):
    cfg, src, check, note = HW.eval_cfg(name)
    if nsamp:
        cfg["nsamp"] = int(nsamp)
    if limit_inj:
        cfg["limit_injections"] = int(limit_inj)
    os.environ["DARKSIRENS_ZMAX"] = repr(float(cfg["zmax"]))
    m = PM.Model(cfg)
    m.import_core()
    import jax
    jax.config.update("jax_enable_x64", True)
    m.build()
    return m, cfg, note


def evaluate(m, H, f0, n, ns):
    ev, var, ess = (np.full((len(H), n), np.nan) for _ in range(3))
    sec = np.zeros(len(H))
    for j, h in enumerate(H):
        t0 = time.time()
        c = {"H0": float(h)} if f0 is None else {"H0": float(h), "f_agn": float(f0)}
        r = m.evaluate_events(c)
        ldw = S.captured(n * ns).reshape(n, ns)
        ev[j], var[j] = r["event_log_evidence"], r["event_mc_variance"]
        with np.errstate(invalid="ignore", over="ignore"):
            ess[j] = S.ess_of(ldw)
        sec[j] = time.time() - t0
    return ev, var, ess, sec


# ============================================================================ references
def reference(name, idx):
    """(H0 nodes, converged ln L_i, label) of a merger from tests/results/obslike_steps*."""
    if "realization_106" in name and idx == 691:
        an = name.split("/")[0]
        z = np.load(S.OUT / f"single_{an}_obslike_f0p30_s1_r106_i691.npz")
        return z["H0"], z["lnL_all256000"], "single npz all256000", {"sub8000": np.stack([z[f"lnL_sub8000_{k}"] for k in range(32)])}
    p = HW.RES / f"obslike_steps_halfwidth_{HW.slug(name)}_i{idx}.json"
    d = json.loads(p.read_text())
    e = d["ev_mergers"][str(idx)] if "ev_mergers" in d else d["ev_merger"]
    H = np.asarray(d["H0"]) if "H0" in d else np.load(S.OUT / f"halfwidth_{HW.slug(name)}.npz")["H0"]
    return H, np.asarray(e["all"]), f"{p.name} all ({d['samples']})", {"first_N": np.asarray(e["first_N"])}


def merger(name, idx, K, nsamp, window, tag):
    t0 = time.time()
    m, cfg, note = model_for(name, nsamp)
    ns = int(cfg["nsamp"])
    obs, attrs, nobs, _ = M.read_event_set(cfg["events_file"])
    g = M.generator()
    rows_off, rows_on, recs, rr = [], [], [], []
    for k in range(K):
        raw = production_raw(obs, nobs, ns, key_k(cfg, k), rows=[idx])
        rows_off.append(raw)
        ck = dict(cfg, redraw_key=key_k(cfg, k), redraw_mixture="support")
        raw_on, r, rec = RM.mixture_redraw(ck, g, obs, np.array([idx]), raw, ns)
        rows_on.append(raw_on)
        rr.append(np.ones((1, ns)) if r is None else r)
        recs.append(rec["mergers"].get(str(idx)))
    t_draw = time.time() - t0
    cat = {k: np.concatenate([x[k] for x in rows_off + rows_on]) for k in rows_off[0]}
    rfull = np.concatenate([np.ones((K, ns))] + rr)
    post = post_of(cat, 2 * K, ns, rfull)
    store = M.load_store(m.ds, post, attrs, 2 * K, ns)
    m.bind(events_store=store)
    S.install_capture()
    H, ref, rlab, rextra = reference(name, idx)
    line = stored_line(name)
    f0 = line["f"]
    ev, var, ess, sec = evaluate(m, H, f0, 2 * K, ns)
    lo, hi = window
    w = (H >= lo) & (H <= hi) & np.isfinite(ref)
    out = {"run": name, "merger": idx, "nsamp": ns, "streams": K, "f": f0, "reference": rlab, "window": [lo, hi],
           "n_nodes_window": int(w.sum()), "H0_step": float(np.median(np.diff(H))), "note": note,
           "mixture_records": recs, "seconds_draw": t_draw, "seconds_eval": float(sec.sum())}
    jc = int(np.argmin(np.abs(H - 0.5 * (lo + hi))))
    for lab, sl in (("off", slice(0, K)), ("on", slice(K, 2 * K))):
        e = ev[:, sl]
        d = e[w] - ref[w][:, None]
        d = d - d.mean(axis=0, keepdims=True)
        steps = np.abs(np.diff(e[w], axis=0))
        rsteps = np.abs(np.diff(d, axis=0))
        mean_curve = d.mean(axis=1)
        sd_curve = d.std(axis=1, ddof=1)
        D = e[w][-1] - e[w][0]
        Dref = ref[w][-1] - ref[w][0]
        out[lab] = {"ess_mid_median": float(np.median(ess[jc, sl])), "ess_mid_by_stream": ess[jc, sl].tolist(),
                    "ess_window_min_median": float(np.median(ess[w][:, sl].min(axis=0))),
                    "rms_resid_by_stream": np.sqrt((d ** 2).mean(axis=0)).tolist(),
                    "rms_resid_median": float(np.median(np.sqrt((d ** 2).mean(axis=0)))),
                    "max_abs_resid_median": float(np.median(np.abs(d).max(axis=0))),
                    "largest_step_median": float(np.median(steps.max(axis=0))),
                    "largest_step_max": float(steps.max()),
                    "largest_resid_step_median": float(np.median(rsteps.max(axis=0))),
                    "mean_resid_curve_max_abs": float(np.abs(mean_curve).max()),
                    "mean_resid_curve_max_abs_over_sem": float(np.max(np.abs(mean_curve) / np.maximum(sd_curve / np.sqrt(K), 1e-12))),
                    "D_end_minus_start": {"mean": float(D.mean()), "sd": float(D.std(ddof=1)), "ref": float(Dref)}}
    out["ref_largest_step"] = float(np.abs(np.diff(ref[w])).max())
    if "sub8000" in rextra and ns == 8000:
        s8 = rextra["sub8000"][:, w]
        d8 = s8 - ref[w][None]
        d8 = d8 - d8.mean(axis=1, keepdims=True)
        out["reference_sub8000_rms_resid_median"] = float(np.median(np.sqrt((d8 ** 2).mean(axis=1))))
    slug = HW.slug(name)
    np.savez_compressed(OUT / f"merger_{slug}_i{idx}_s{ns}{tag}.npz", H0=H, ev=ev, var=var, ess=ess, ref=ref, w=w)
    (OUT / f"merger_{slug}_i{idx}_s{ns}{tag}.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"[merger] {name} i{idx} S={ns} K={K} ({t_draw:.0f} s draws, {sec.sum():.0f} s eval, {len(H)} nodes); ref {rlab}")
    for lab in ("off", "on"):
        o = out[lab]
        print(f"   {lab:3s}: ESS mid {o['ess_mid_median']:.1f}, rms resid {o['rms_resid_median']:.3f}, max|resid| "
              f"{o['max_abs_resid_median']:.3f}, largest step {o['largest_step_median']:.3f} (max {o['largest_step_max']:.3f}), "
              f"mean-curve bias max {o['mean_resid_curve_max_abs']:.3f} ({o['mean_resid_curve_max_abs_over_sem']:.1f} sem); "
              f"D {o['D_end_minus_start']['mean']:+.3f} +- {o['D_end_minus_start']['sd']:.3f} (ref {o['D_end_minus_start']['ref']:+.3f})")
    print(f"   ref largest step {out['ref_largest_step']:.3f}; records {recs[0]}", flush=True)


# ============================================================================ line
def line_run(name, nsamp, all_mergers, tag, streams=1):
    t0 = time.time()
    m, cfg, note = model_for(name, nsamp)
    ns = int(cfg["nsamp"])
    obs, attrs, nobs, _ = M.read_event_set(cfg["events_file"])
    g = M.generator()
    raw = production_raw(obs, nobs, ns, cfg["redraw_key"])
    ck = dict(cfg, redraw_mixture="support")
    raw_on, r, rec = RM.mixture_redraw(ck, g, obs, np.arange(nobs), raw, ns)
    refilled = np.array(sorted(int(i) for i, x in rec["mergers"].items() if x["refilled"]), int)
    T = np.arange(nobs) if all_mergers else refilled
    nT = T.size
    extra = []
    for k in range(1, streams):            # further streams of the T mergers (production key k, its restricted stream)
        rk = production_raw(obs, nobs, ns, key_k(cfg, k), rows=T)
        cfk = dict(cfg, redraw_key=key_k(cfg, k), redraw_mixture="support")
        ro, rr_, _ = RM.mixture_redraw(cfk, g, obs, T, rk, ns)
        extra.append((ro, np.ones((nT, ns)) if rr_ is None else rr_))
    t_draw = time.time() - t0
    rfull = np.ones((nobs, ns)) if r is None else r
    parts = [{k: v[T] for k, v in raw.items()}, {k: v[T] for k, v in raw_on.items()}] + [e[0] for e in extra]
    rparts = [np.ones((nT, ns)), rfull[T]] + [e[1] for e in extra]
    nb = len(parts)
    cat = {k: np.concatenate([x[k] for x in parts]) for k in parts[0]}
    post = post_of(cat, nb * nT, ns, np.concatenate(rparts))
    store = M.load_store(m.ds, post, attrs, nb * nT, ns)
    m.bind(events_store=store)
    S.install_capture()
    t_bind = time.time() - t0 - t_draw
    line = stored_line(name)
    H = line["H0"]
    print(f"[line] {name}: S={ns}, {nT} mergers x {nb} copies bound ({t_draw:.0f} s draws, {t_bind:.0f} s bind); "
          f"{len(H)} nodes at f={line['f']}", flush=True)
    ev, var, ess, sec = evaluate(m, H, line["f"], nb * nT, ns)
    np.savez_compressed(OUT / f"line_{HW.slug(name)}{tag}.npz", H0=H, T=T, ev=ev, var=var, ess=ess, nb=nb,
                        meta=json.dumps({"rec_summary": rec["summary"], "region": rec["region"]}, default=float))
    report_line(name, tag, rec)
    print(f"[line] eval {sec.sum():.0f} s ({np.median(sec):.1f} s/node)", flush=True)


FORCE_F = None


def stored_line(name):
    """obslike_steps_halfwidth.stored_line, at --f when given (the f_AGN node of an earlier comparison)."""
    line = HW.stored_line(name)
    if FORCE_F is None or line["f"] is None:
        return line
    import h5py
    fp = HW.RUNS / name / "grid" / line["file"]
    with h5py.File(fp, "r") as F:
        af = np.asarray(F["axis_f_agn"][()], float)
        jf = int(np.argmin(np.abs(af - FORCE_F)))
        if abs(af[jf] - FORCE_F) > 1e-6:
            raise SystemExit(f"[fatal] f {FORCE_F} not a node of {fp}")
        for k in ("lnL", "pe_var", "n_eff", "sel_corr", "log_mu"):
            line[k] = np.asarray(F[k][()], float)[:, jf]
    line["f"], line["jf"] = float(af[jf]), jf
    return line


def smooth_steps(H, y, ipk):
    near = np.isfinite(y) & (y > np.nanmax(y) - 12)
    c = np.polyfit(H[near] - H[ipk], y[near], 4)
    res = np.diff(y - np.polyval(c, H - H[ipk]))
    res = np.where(near[1:] & near[:-1], res, np.nan)
    j = int(np.nanargmax(np.abs(res)))
    return {"largest_excess_step": float(res[j]), "between": [float(H[j]), float(H[j + 1])],
            "n_excess_steps_gt_0.2": int(np.nansum(np.abs(res) > 0.2)), "rms_resid_quartic": float(np.sqrt(np.nanmean(
                (y - np.polyval(c, H - H[ipk]))[near] ** 2)))}


def report_line(name, tag, rec):
    z = np.load(OUT / f"line_{HW.slug(name)}{tag}.npz")
    H, T, ev, var, ess, nb = z["H0"], z["T"], z["ev"], z["var"], z["ess"], int(z["nb"])
    nT = T.size
    line = stored_line(name)
    n = line["n_events"]
    all_m = nT == n
    off, on = slice(0, nT), slice(nT, 2 * nT)

    def corr(l, pv):
        return l + 0.5 * pv - 0.5 * n / line["n_eff"] if line["hvc"] else l

    lines = {}
    if all_m:
        lines["off"] = (line["sel_corr"] + ev[:, off].sum(1), var[:, off].sum(1))
        for b in range(1, nb):
            sl = slice(b * nT, (b + 1) * nT)
            lines["on" if b == 1 else f"on_stream{b - 1}"] = (line["sel_corr"] + ev[:, sl].sum(1), var[:, sl].sum(1))
        lines["stored"] = (line["lnL"], line["pe_var"])
    else:
        lines["stored"] = (line["lnL"], line["pe_var"])
        lines["off_check"] = (line["lnL"], line["pe_var"])
        for b in range(1, nb):
            sl = slice(b * nT, (b + 1) * nT)
            lines["on" if b == 1 else f"on_stream{b - 1}"] = (line["lnL"] + (ev[:, sl] - ev[:, off]).sum(1),
                                                            line["pe_var"] + (var[:, sl] - var[:, off]).sum(1))
    lc = corr(*lines["stored"])
    ipk = int(np.nanargmax(lc))
    out = {"run": name, "f": line["f"], "n_events": n, "mergers_bound": int(nT), "all_mergers": bool(all_m),
           "refill_summary": rec["summary"] if rec else json.loads(str(z["meta"]))["rec_summary"], "lines": {}}
    for lab, (l, pv) in lines.items():
        c = corr(l, pv)
        s = HW.line_summary(H, c)
        s["steps"] = smooth_steps(H, c, ipk)
        s["lnL_corrected_minus_max"] = (c - np.nanmax(c)).tolist()
        su = HW.line_summary(H, l)
        s["uncorrected"] = {"median": su["median"], "ci90": su["ci90"], "hw90": su["hw90"],
                            "steps": smooth_steps(H, l, ipk)}
        out["lines"][lab] = s
    # hybrid lines of tests/results (the converged comparison)
    hy = {}
    for p in sorted(HW.RES.glob(f"obslike_steps_halfwidth_{HW.slug(name)}*.json")):
        d = json.loads(p.read_text())
        for k, v in d.items():
            if isinstance(v, dict) and "hw90" in v and ("hybrid" in k or k in ("stored",)):
                hy[f"{p.name}:{k}"] = {kk: v[kk] for kk in ("median", "ci90", "hw90")}
            if k == "corrected":
                hy[f"{p.name}:corrected.hybrid"] = {kk: v["hybrid"][kk] for kk in ("median", "ci90", "hw90")}
                hy[f"{p.name}:corrected.stored"] = {kk: v["stored"][kk] for kk in ("median", "ci90", "hw90")}
    an = name.split("/")[0]
    hp = HW.RES / f"obslike_steps_hybrid_{an}_r{line_real(name)}.json"
    if hp.exists() and "obslike_f0p30_s1" in name:
        d = json.loads(hp.read_text())
        for k in ("posterior_along_H0_8000", "posterior_along_H0_hybrid"):
            v = d[k]
            hy[f"{hp.name}:{k} (uncorrected)"] = {"median": v["median"], "ci90": v["ci90"], "hw90": 0.5 * v["w90"]}
    out["hybrid_references"] = hy
    # per merger on - off (item 3): at the peak node, half-variance corrected ev_i + var_i / 2 as well
    d = ev[ipk, on] - ev[ipk, off]
    dc = (ev[ipk, on] + 0.5 * var[ipk, on]) - (ev[ipk, off] + 0.5 * var[ipk, off])
    out["per_merger_on_minus_off_at_peak"] = {"H0": float(H[ipk]), "n": int(nT), "mean": float(d.mean()),
                                              "sem": float(d.std(ddof=1) / np.sqrt(nT)), "rms": float(np.sqrt((d ** 2).mean())),
                                              "mean_corrected": float(dc.mean()), "sem_corrected": float(dc.std(ddof=1) / np.sqrt(nT))}
    out["ess_census_at_peak"] = {lab: {f"lt_{c}": int((ess[ipk, sl] < c).sum()) for c in (5, 20, 100)}
                                 for lab, sl in (("off", off), ("on", on))}
    out["ess_min_at_peak"] = {lab: float(np.nanmin(ess[ipk, sl])) for lab, sl in (("off", off), ("on", on))}
    p = OUT / f"line_{HW.slug(name)}{tag}.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    print(f"[line] {name} f={line['f']} ({nT} mergers bound):")
    for lab, s in out["lines"].items():
        print(f"   {lab:12s} median {s['median']:.3f} 90% [{s['ci90'][0]:.3f}, {s['ci90'][1]:.3f}] hw {s['hw90']:.3f}; "
              f"largest excess step {s['steps']['largest_excess_step']:+.3f} at {s['steps']['between']}, "
              f"steps > 0.2: {s['steps']['n_excess_steps_gt_0.2']}, rms vs quartic {s['steps']['rms_resid_quartic']:.3f} | "
              f"uncorrected [{s['uncorrected']['ci90'][0]:.3f}, {s['uncorrected']['ci90'][1]:.3f}] hw {s['uncorrected']['hw90']:.3f}")
    for k, v in hy.items():
        print(f"   ref {k}: median {v['median']:.3f} 90% [{v['ci90'][0]:.3f}, {v['ci90'][1]:.3f}] hw {v['hw90']:.3f}")
    print(f"   per merger on-off at {H[ipk]:g}: {out['per_merger_on_minus_off_at_peak']}")
    print(f"   ESS census at peak: {out['ess_census_at_peak']}, min {out['ess_min_at_peak']}")
    print("wrote", p, flush=True)


def line_real(name):
    return int(name.split("realization_")[1][:3])


# ============================================================================ off check
def offcheck(name, nodes):
    cfg, src, check = S.make_cfg(name)
    if cfg.get("redraw_mixture"):
        raise SystemExit("[fatal] option on")
    t0 = time.time()
    m = PD.setup_model(cfg)
    rows = []
    for h in nodes:
        line = stored_line(name)
        c = {"H0": float(h)} if line["f"] is None else {"H0": float(h), "f_agn": float(line["f"])}
        r = m.evaluate(c)
        st = S.stored_node(name, h, line["f"] if line["f"] is not None else 0.0)
        row = {"coords": c, "copy_off": {k: r[k] for k in ("lnL", "pe_var", "n_eff", "log_mu", "sel_corr")}, "stored": st,
               "diff": {k: r[k] - st[k] for k in ("lnL", "pe_var", "n_eff", "log_mu", "sel_corr")} if st else None}
        rows.append(row)
        print(f"[offcheck] {name} {c}: " + (", ".join(f"{k} {v:+.2e}" for k, v in row["diff"].items()) if st else "no stored node"), flush=True)
    out = {"run": name, "config_check": check, "modules": {mm.__name__: mm.__file__ for mm in (P, PM, PD, RM)},
           "rows": rows, "seconds": time.time() - t0, "store_info": m.info.get("events_samples")}
    (OUT / f"offcheck_{HW.slug(name)}.json").write_text(json.dumps(out, indent=1, default=float))


def timing(name, reps=6):
    """Per-call cost with the option off and on: the run's model bound to the full set (production redraw, then the
    mixture store), reps calls each at the same nodes after one warm-up call."""
    out = {"run": name}
    line = stored_line(name)
    H = line["H0"][len(line["H0"]) // 2 - reps // 2:][:reps]
    for mode in ("off", "support"):
        m, cfg, note = model_for(name)
        if mode != "off":
            m.cfg["redraw_mixture"] = mode
        t0 = time.time()
        m.bind()
        tb = time.time() - t0
        c = (lambda h: {"H0": float(h)}) if line["f"] is None else (lambda h: {"H0": float(h), "f_agn": float(line["f"])})
        m.evaluate(c(H[0]))
        ts = []
        for h in H:
            t1 = time.time()
            m.evaluate(c(h))
            ts.append(time.time() - t1)
        out[mode] = {"bind_seconds": tb, "events_load_seconds": m.info.get("events_load_seconds"), "call_seconds": ts,
                     "median_call_seconds": float(np.median(ts)), "mixture": m.info.get("redraw_mixture"),
                     "store_shape": [int(m.n_events), int(m.nsamp)]}
        print(f"[timing] {name} {mode}: bind {tb:.1f} s (events load {out[mode]['events_load_seconds']:.1f} s), "
              f"median call {np.median(ts):.3f} s over {len(ts)}", flush=True)
    (OUT / f"timing_{HW.slug(name)}.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("merger", "line", "offcheck", "timing", "report"))
    ap.add_argument("name")
    ap.add_argument("idx", nargs="?", type=int)
    ap.add_argument("--streams", type=int, default=None)
    ap.add_argument("--nsamp", type=int, default=None)
    ap.add_argument("--window", default="66,72")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--nodes", default="")
    ap.add_argument("--f", type=float, default=None, help="line at this f_AGN node (default: the node of largest weight)")
    a = ap.parse_args()
    FORCE_F = a.f
    if a.mode == "merger":
        merger(a.name, a.idx, a.streams or 8, a.nsamp, [float(x) for x in a.window.split(",")], a.tag)
    elif a.mode == "line":
        line_run(a.name, a.nsamp, a.all, a.tag, a.streams or 1)
    elif a.mode == "timing":
        timing(a.name)
    elif a.mode == "report":
        report_line(a.name, a.tag, None)
    else:
        offcheck(a.name, [float(x) for x in a.nodes.split(",")])
