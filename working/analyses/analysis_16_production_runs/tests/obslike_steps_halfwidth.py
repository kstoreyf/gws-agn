#!/usr/bin/env python
"""The 90% half-width of H0 along a stored grid line with the poorly resolved mergers re-evaluated at more samples (CPU only).

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH= <core venv python> tests/obslike_steps_halfwidth.py run RUN \
        [--check-h0 a,b,c | all] [--mult 4] [--ess-cut 50] [--limit-inj 5000] [--tag T]
    ... report RUN [--tag T]          (numpy only, from the npz of run)

RUN is a run directory under runs/ (any analysis; hosts_known runs whose mergers are all AGN-hosted are evaluated as their AGN
subset, the only one present). The line is the fine pass's H0 axis at the f_AGN node of largest posterior weight (two-parameter
runs; marginal over the fine H0 nodes of the half-variance corrected ln L).

run: (1) the run's model at its own sample number N with the full injection set, at the check nodes (always including the
peak node of the line): every merger's ln L_i, sigma_i^2, effective sample number, and the total ln L, pe_var, n_eff compared
with the stored grid; (2) the mergers with effective sample number < ess-cut at the peak node re-bound alone at mult x N
samples (the production redraw, blocks of 2000 with the run's key, so the first N are the run's) and a few injections (the
per-merger terms do not involve them), at every node of the line: ln L_i and sigma_i^2 from all mult x N samples and from the
first N (p_pe renormalised over the subset as core does), the latter checked against (1) at the shared nodes.
-> tests/results/obslike_steps/halfwidth_<slug><tag>.npz

report: lnL_hyb = lnL_stored + sum_K [ln L_i(mult N) - ln L_i(N)], pe_var likewise, n_eff (selection) as stored; the half-
variance corrected stored and hybrid lines, Akima-refined 16x as the run summaries, prod_common.summarize_1d: median, 90%
interval, half-width; the merger behind each step (largest jump of ln L_i(N) between neighbouring nodes).
-> tests/results/obslike_steps_halfwidth_<slug><tag>.json

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
import obslike_steps_merger as S  # noqa: E402  (adds scripts/ to the path, analysis 14's mc_redraw)

A16 = S.A16
RUNS = S.RUNS
OUT = S.OUT
RES = A16 / "tests" / "results"


def slug(name):
    p = name.split("/")
    return f"{p[0]}_{p[1]}_{p[3]}_{p[4]}"


def stored_line(name):
    """The fine pass's line: H0 axis, f node of largest posterior weight, stored lnL, pe_var, n_eff, sel_corr there."""
    import h5py
    cfg = json.loads((RUNS / name / "config.json").read_text())
    g = RUNS / name / "grid"
    fp = g / "fine_widened.h5" if (g / "fine_widened.h5").exists() else g / "fine.h5"
    with h5py.File(fp, "r") as F:
        ax = np.asarray(F["axis_H0"][()], float)
        n = int(F.attrs["n_events"])
        d = {k: np.asarray(F[k][()], float) for k in ("lnL", "pe_var", "n_eff", "sel_corr", "log_mu")}
        af = np.asarray(F["axis_f_agn"][()], float) if "axis_f_agn" in F else None
    lc = d["lnL"] + 0.5 * d["pe_var"] - 0.5 * n / d["n_eff"] if cfg.get("half_variance_correction") else d["lnL"]
    jf, f0 = None, None
    if af is not None:
        ok = np.isfinite(lc)
        w = np.where(ok, np.exp(lc - np.max(lc[ok])), 0.0).sum(0)
        jf = int(np.argmax(w))
        f0 = float(af[jf])
        d = {k: v[:, jf] for k, v in d.items()}
    return {"file": fp.name, "H0": ax, "f": f0, "jf": jf, "n_events": n, "hvc": bool(cfg.get("half_variance_correction")), **d}


def eval_cfg(name):
    cfg, src, check = S.make_cfg(name)
    note = None
    if cfg["analysis"] == "hosts_known":
        import prod_model as PM
        ht = PM.event_host_types(cfg["events_file"], cfg.get("limit_events"))
        if not (ht == 1).all():
            raise SystemExit("[fatal] hosts_known run with galaxy-hosted mergers: not handled here")
        cfg = PM.HostsKnownModel(cfg).sub_cfg("agn")
        note = "hosts_known: every merger AGN-hosted, evaluated as the AGN subset (its own nsamp and injections)"
    return cfg, src, check, note


def per_merger(ldw, pw, N):
    """ln L_i, sigma_i^2, ESS from all columns and from the first N (p_pe renormalised over them)."""
    n, M = ldw.shape
    out = {}
    lnS_all = np.log(pw.sum(1))
    for tag, cols, nn in (("M", slice(None), M), ("N", slice(0, N), N)):
        x = ldw[:, cols]
        if nn != M:
            x = x + (np.log(pw[:, cols].sum(1)) - lnS_all)[:, None]
        ev = S.lse(x) - np.log(nn)
        with np.errstate(invalid="ignore", over="ignore"):
            ess = S.ess_of(x)
        var = np.where(np.isfinite(ev), 1.0 / ess - 1.0 / nn, 0.0)
        out[f"ev_{tag}"], out[f"var_{tag}"], out[f"ess_{tag}"] = ev, var, ess
    return out


def scan(m, H, f0, N, edge, om0, keep_idx=None):
    n, ns = m.n_events, m.nsamp
    pw = np.asarray(m.b.gw_pe.prior_wt, dtype=float).reshape(n, ns)
    gw = m.b.gw_pe
    m1det = np.asarray(gw.m1det, dtype=float).reshape(n, ns)[:, :N]
    dL = np.asarray(gw.dL, dtype=float).reshape(n, ns)[:, :N]
    rec = {k: np.full((len(H), n), np.nan) for k in ("ev_M", "var_M", "ess_M", "ev_N", "var_N", "ess_N")}
    rec["n_in_N"] = np.zeros((len(H), n), int)
    for k in ("lnL", "pe_var", "n_eff", "sel_corr", "log_mu", "seconds", "core_minus_numpy"):
        rec[k] = np.full(len(H), np.nan)
    for j, h in enumerate(H):
        tc = time.time()
        coords = {"H0": float(h)} if f0 is None else {"H0": float(h), "f_agn": float(f0)}
        r = m.evaluate_events(coords)
        ldw = S.captured(n * ns).reshape(n, ns)
        pm = per_merger(ldw, pw, N)
        for k, v in pm.items():
            rec[k][j] = v
        fin = np.isfinite(r["event_log_evidence"])
        rec["core_minus_numpy"][j] = float(np.max(np.abs(pm["ev_M"][fin] - r["event_log_evidence"][fin]))) if fin.any() else 0.0
        ms, _ = S.m1src_at(m1det.ravel(), dL.ravel(), h, om0)
        rec["n_in_N"][j] = (ms < edge).reshape(n, N).sum(1)
        for k in ("lnL", "pe_var", "n_eff", "sel_corr", "log_mu"):
            rec[k][j] = r[k]
        rec["seconds"][j] = time.time() - tc
        print(f"[scan] H0 {h:g}: lnL {r['lnL']:.6f} pe_var {r['pe_var']:.5f} ({rec['seconds'][j]:.0f} s; core vs numpy "
              f"{rec['core_minus_numpy'][j]:.1e})", flush=True)
    return rec


def run(name, check_spec, mult, ess_cut, limit_inj, tag, stage1_only=False):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    t0 = time.time()
    line = stored_line(name)
    cfg, src, check, note = eval_cfg(name)
    N = int(cfg["nsamp"])
    os.environ["DARKSIRENS_ZMAX"] = repr(float(cfg["zmax"]))
    import prod_model as PM
    m = PM.Model(cfg)
    m.import_core()
    import jax
    jax.config.update("jax_enable_x64", True)
    m.build()
    m.bind()
    S.install_capture()
    t_setup = time.time() - t0
    print(f"[run] {name}: built + bound in {t_setup:.0f} s; n_events {m.n_events}; nsamp {m.nsamp}; f {line['f']}; {note or ''}", flush=True)
    edge, om0 = S.EDGE[cfg["population_family"]], float(cfg["Om0"])
    H = line["H0"]
    lc = line["lnL"] + 0.5 * line["pe_var"] - 0.5 * line["n_events"] / line["n_eff"] if line["hvc"] else line["lnL"]
    ipk = int(np.nanargmax(lc))
    if check_spec == "all":
        jc = np.arange(len(H))
    else:
        want = [float(x) for x in check_spec.split(",")] if check_spec else []
        jc = sorted({ipk} | {int(np.argmin(np.abs(H - w))) for w in want})
        jc = np.asarray(jc, int)
    bp = OUT / f"halfwidth_{slug(name)}{tag}_stage1.npz"
    if bp.exists() and np.array_equal(np.load(bp)["jc"], jc):
        zb = np.load(bp)
        base = {k: zb[k] for k in zb.files if k != "jc"}
        print(f"[run] stage 1 read from {bp.name}", flush=True)
    else:
        base = scan(m, H[jc], line["f"], N, edge, om0)
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(bp, jc=jc, **base)
    dev ={k: (base[k] - line[k][jc]).tolist() for k in ("lnL", "pe_var", "n_eff", "sel_corr")}
    print(f"[run] check vs stored ({line['file']}): max |lnL diff| {np.nanmax(np.abs(dev['lnL'])):.2e}, "
          f"max |pe_var diff| {np.nanmax(np.abs(dev['pe_var'])):.2e} over {len(jc)} nodes", flush=True)
    kpk = int(np.where(jc == ipk)[0][0])
    K = np.where(base["ess_N"][kpk] < ess_cut)[0]
    print(f"[run] {K.size} mergers with ESS < {ess_cut} at the peak H0 {H[ipk]:g}: "
          + ", ".join(f"{i}:{base['ess_N'][kpk, i]:.1f}" for i in K), flush=True)
    if stage1_only:
        return
    # stage 2: the K mergers alone at mult x N samples, few injections, every node of the line
    if K.size == 0:
        raise SystemExit("[run] no merger below the cut: the hybrid line is the stored line")
    t1 = time.time()
    cfg["nsamp"] = N * mult
    cfg["limit_injections"] = int(limit_inj)
    sel = np.zeros(m.n_events, bool)
    sel[K] = True
    m.bind(select=sel)
    t_bind2 = time.time() - t1
    print(f"[run] re-bound {m.n_events} mergers at {m.nsamp} samples in {t_bind2:.0f} s", flush=True)
    hyb = scan(m, H, line["f"], N, edge, om0)
    # check: the first N of the mult x N redraw against the run's own N-sample terms at the check nodes
    chk = float(np.max(np.abs(hyb["ev_N"][jc] - base["ev_N"][:, K]))) if K.size else 0.0
    chk_var = float(np.max(np.abs(hyb["var_N"][jc] - base["var_N"][:, K]))) if K.size else 0.0
    print(f"[run] first {N} of {N * mult} vs the run's own terms: max |d ev| {chk:.2e}, max |d var| {chk_var:.2e}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"halfwidth_{slug(name)}{tag}"
    meta = {"run": name, "args_from": src, "config_check": check, "note": note, "nsamp_run": N, "mult": mult, "ess_cut": ess_cut,
            "limit_inj_stage2": limit_inj, "f": line["f"], "stored_file": line["file"], "peak_H0": float(H[ipk]),
            "check_nodes": H[jc].tolist(), "check_minus_stored": dev, "mergers": K.tolist(),
            "first_N_vs_run_ev_max_abs": chk, "first_N_vs_run_var_max_abs": chk_var,
            "seconds_setup": t_setup, "seconds_bind2": t_bind2, "seconds_total": time.time() - t0}
    np.savez_compressed(OUT / f"{stem}.npz", H0=H, jc=jc, K=K, meta=json.dumps(meta, default=float),
                        **{f"base_{k}": v for k, v in base.items()}, **{f"hyb_{k}": v for k, v in hyb.items()})
    print(f"[run] -> {OUT / stem}.npz ({time.time() - t0:.0f} s)", flush=True)
    report(name, tag)


def line_summary(H, y):
    from scipy.interpolate import Akima1DInterpolator
    import prod_common as P
    ok = np.isfinite(y)
    y = np.where(ok, y, np.min(y[ok]) - 300.0)
    lat = np.linspace(H[0], H[-1], (len(H) - 1) * 16 + 1)
    sm = P.summarize_1d(lat, Akima1DInterpolator(H, y)(lat))
    return {"median": sm["median"], "ci90": sm["ci90"], "hw90": 0.5 * (sm["ci90"][1] - sm["ci90"][0]), "map": sm["map"]}


def report(name, tag=""):
    stem = f"halfwidth_{slug(name)}{tag}"
    z = np.load(OUT / f"{stem}.npz")
    meta = json.loads(str(z["meta"]))
    line = stored_line(name)
    H, K, jc = z["H0"], z["K"], z["jc"]
    n = line["n_events"]
    d_ev = (z["hyb_ev_M"] - z["hyb_ev_N"]).sum(1)
    d_var = (z["hyb_var_M"] - z["hyb_var_N"]).sum(1)
    lnL_s, pv_s = line["lnL"], line["pe_var"]
    lnL_h, pv_h = lnL_s + d_ev, pv_s + d_var

    def corr(l, pv):
        return l + 0.5 * pv - 0.5 * n / line["n_eff"] if line["hvc"] else l

    out = {"run": name, "f": line["f"], "nsamp_run": meta["nsamp_run"], "nsamp_hybrid": meta["nsamp_run"] * meta["mult"],
           "ess_cut": meta["ess_cut"], "peak_H0": meta["peak_H0"], "mergers_replaced": K.tolist()}
    out["check"] = {"nodes": meta["check_nodes"], "max_abs_lnL_minus_stored": float(np.nanmax(np.abs(meta["check_minus_stored"]["lnL"]))),
                    "max_abs_pe_var_minus_stored": float(np.nanmax(np.abs(meta["check_minus_stored"]["pe_var"]))),
                    "first_N_vs_run_ev_max_abs": meta["first_N_vs_run_ev_max_abs"], "note": meta["note"]}
    kpk = int(np.where(H[jc] == meta["peak_H0"])[0][0])
    ipk = int(np.argmin(np.abs(H - meta["peak_H0"])))
    out["mergers"] = {str(int(i)): {"ess_N_peak": float(z["base_ess_N"][kpk, i]), "ess_M_peak": float(z["hyb_ess_M"][ipk, k]),
                                    "n_in_edge_N_peak": int(z["hyb_n_in_N"][ipk, k]),
                                    "d_ev_M_minus_N_peak": float(z["hyb_ev_M"][ipk, k] - z["hyb_ev_N"][ipk, k])}
                      for k, i in enumerate(K)}
    # steps: the largest jumps of the stored line against a smooth fit, and the merger behind each
    sl = corr(lnL_s, pv_s)
    near = np.isfinite(sl) & (sl > np.nanmax(sl) - 12)
    c = np.polyfit(H[near] - H[ipk], sl[near], 4)
    res = np.diff(sl - np.polyval(c, H - H[ipk]))
    dN = np.diff(z["hyb_ev_N"], axis=0)                     # (nH-1, K)
    steps = []
    for j in np.argsort(-np.where(np.isfinite(res), res, -np.inf))[:5]:
        if not near[j] or res[j] < 0.2:
            continue
        o = np.argsort(-np.abs(dN[j]))[:2] if K.size else []
        steps.append({"between": [float(H[j]), float(H[j + 1])], "excess_rise_nats": float(res[j]), "d_lnL_stored": float(sl[j + 1] - sl[j]),
                      "top_mergers": [{"i": int(K[k]), "d_ev_N": float(dN[j, k]), "ess_N": [float(z["hyb_ess_N"][j, k]), float(z["hyb_ess_N"][j + 1, k])],
                                       "n_in_N": [int(z["hyb_n_in_N"][j, k]), int(z["hyb_n_in_N"][j + 1, k])]} for k in o],
                      "d_ev_N_sum_K": float(dN[j].sum()) if K.size else 0.0})
    out["steps"] = sorted(steps, key=lambda s: s["between"][0])
    s_st, s_hy = line_summary(H, corr(lnL_s, pv_s)), line_summary(H, corr(lnL_h, pv_h))
    s_st_u, s_hy_u = line_summary(H, lnL_s), line_summary(H, lnL_h)
    out["corrected"] = {"stored": s_st, "hybrid": s_hy, "d_hw90": s_hy["hw90"] - s_st["hw90"], "d_hw90_over_5": (s_hy["hw90"] - s_st["hw90"]) / 5}
    out["uncorrected"] = {"stored": s_st_u, "hybrid": s_hy_u, "d_hw90": s_hy_u["hw90"] - s_st_u["hw90"]}
    summ = json.loads((RUNS / name / "summary.json").read_text())
    out["summary_posterior_H0"] = summ["posterior"]["H0"]
    out["lines"] = {"H0": H.tolist(), "stored_corrected_minus_max": (corr(lnL_s, pv_s) - np.nanmax(corr(lnL_s, pv_s))).tolist(),
                    "hybrid_corrected_minus_max": (corr(lnL_h, pv_h) - np.nanmax(corr(lnL_h, pv_h))).tolist(),
                    "d_ev_sum": d_ev.tolist(), "d_var_sum": d_var.tolist()}
    p = RES / f"obslike_steps_halfwidth_{slug(name)}{tag}.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    print(f"[report] {name} f={line['f']}: K={K.tolist()}")
    for k, v in out["mergers"].items():
        print(f"   merger {k}: {v}")
    for s in out["steps"]:
        print(f"   step {s['between']}: excess {s['excess_rise_nats']:+.2f}; " + "; ".join(f"i={t['i']} d={t['d_ev_N']:+.2f} ess {t['ess_N'][0]:.1f}->{t['ess_N'][1]:.1f} n_in {t['n_in_N']}" for t in s["top_mergers"]))
    print(f"   check: {out['check']}")
    print(f"   corrected stored {s_st['ci90']} hw {s_st['hw90']:.3f} | hybrid {s_hy['ci90']} hw {s_hy['hw90']:.3f} | d hw {s_hy['hw90'] - s_st['hw90']:+.3f} (/5 {(s_hy['hw90'] - s_st['hw90']) / 5:+.3f})")
    print(f"   summary posterior H0 {out['summary_posterior_H0']['ci90']}")
    print("wrote", p)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "report"))
    ap.add_argument("name")
    ap.add_argument("--check-h0", default="")
    ap.add_argument("--mult", type=int, default=4)
    ap.add_argument("--ess-cut", type=float, default=50.0)
    ap.add_argument("--limit-inj", type=int, default=5000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--stage1-only", action="store_true", help="the check nodes only (all mergers at the run's N; npz _stage1)")
    a = ap.parse_args()
    if a.mode == "run":
        run(a.name, a.check_h0, a.mult, a.ess_cut, a.limit_inj, a.tag, a.stage1_only)
    else:
        report(a.name, a.tag)
