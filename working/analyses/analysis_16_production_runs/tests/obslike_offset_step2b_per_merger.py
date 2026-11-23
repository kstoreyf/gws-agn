#!/usr/bin/env python
"""Step 2 (b) of the obslike_f0p30_s1 H0-offset diagnosis: the AGN-hosted factor of hosts_known merger by merger, on CPU.

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH= <core venv python> tests/obslike_offset_step2b_per_merger.py compute SET R [R ...]
    <core venv or jax env python> tests/obslike_offset_step2b_per_merger.py analyze SET [SET ...]
    ... compute SET R [R ...] --redraw 1      (an independent production redraw: Monte Carlo noise of the scores)
    <core venv or jax env python> ... mc SET [SET ...]
    JAX_PLATFORMS=cpu ... grid SET R [R ...] [--classes agn|both] [--h0 lo,hi,step]   (hosts_known 1-D grid on CPU, any R
                                                                                        whose events and injections exist)
    ... combine SET R [R ...] [--classes agn|both]       (the grid outputs combined over realizations, level of the input)

compute: the hosts_known run of SET (realization R, ..._agn_s32000_inj1) is rebuilt from its run-list arguments exactly as the
driver builds it (prod_driver.parse_args/resolve/setup_model; the resolved configuration is compared with the stored
config.json), and its AGN-hosted subset (prod_model.HostsKnownModel.subs['agn']: AGN catalog, AGN-class population, 32000
samples per merger, the realization's injection lane) is evaluated with evaluate_events at the fine-grid nodes H0 = 67.2,
67.7, 68.2: every merger's ln evidence ln L_i and Monte Carlo variance, and the subset's ln L, pe_var, n_eff, log_mu. The
subset's ln L is checked against the stored grid's lnL_agn at the same nodes. -> tests/results/obslike_offset_step2b/<SET>_r<R>.npz
(resumable). Nothing in scripts/ or runs/ is written.

analyze: per merger, the contribution l_i(H0) = ln L_i - ln mu_agn to the AGN-class log likelihood, its score s_i (central
difference over 67.2..68.2, at 67.7) and curvature h_i; per realization and pooled: sum s_i (the score whose deficit the step-2
split found), the information-matrix comparison sum s_i^2 vs -sum h_i (equal in expectation when the model is right: White's
test), the share of the summed score carried by the k most negative mergers, and the correlation of s_i with the merger's
properties (true z, m1, snr_obs, PE fraction below m2 = 5, fraction of the AGN class's mergers ...). -> ..._analysis.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

A16 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(A16 / "scripts"))
RUNS = A16 / "runs"
OUT = A16 / "tests" / "results" / "obslike_offset_step2b"
NODES = (67.2, 67.7, 68.2)
KNOWN = "hosts_known/both_catalogs/complete/{set}/realization_{R:03d}_agn_s32000_inj1"


def runlist_args(name):
    import yaml
    for f in ("RUNLIST_experiment.yaml", "RUNLIST.yaml"):
        doc = yaml.safe_load((RUNS / f).read_text())
        for st in doc["stages"]:
            for r in st["runs"]:
                if r["name"] == name:
                    return list(r["args"]), f
    raise SystemExit(f"[fatal] {name} is in neither run list")


def compute(label, R, redraw=0):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import h5py
    import prod_driver as PD
    name = KNOWN.format(set=label, R=R)
    args, src = runlist_args(name)
    if redraw:
        args = args + ["--redraw-index", str(redraw)]
    cfg = PD.resolve(PD.parse_args(args))
    if cfg["run_name"] != name:
        raise SystemExit(f"[fatal] resolved run name {cfg['run_name']} != {name}")
    sc = json.loads((RUNS / name / "config.json").read_text())
    if redraw:
        exp_key = list(sc["redraw_key"][:-1]) + [int(redraw)]
        if cfg["redraw_key"] != exp_key:
            raise SystemExit(f"[fatal] redraw key {cfg['redraw_key']} != {exp_key}")
    keys = ("events_file", "injections_files", "surveys", "nsamp",) + (() if redraw else ("redraw_key",)) + ( "free", "population_values", "class_populations",
            "half_variance_correction", "max_var", "pe_event_block", "sel_batch", "catalog_evaluation", "zmax", "Om0", "agn_subset")
    diff = {k: (sc.get(k), cfg.get(k)) for k in keys if sc.get(k) != cfg.get(k)}
    if diff:
        raise SystemExit(f"[fatal] resolved configuration differs from the stored run's: {diff}")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / (f"{label}_r{R}.npz" if not redraw else f"{label}_r{R}_rd{redraw}.npz")
    t0 = time.time()
    m = PD.setup_model(cfg)
    sub = m.subs["agn"]
    n = sub.n_events
    H = np.asarray(NODES)
    rec = {"H0": H, "ev": np.full((len(H), n), np.nan), "var": np.full((len(H), n), np.nan), "seconds": np.full(len(H), np.nan),
           "agn_mask": m.masks["agn"]}
    scal = ("lnL", "pe_var", "n_eff", "log_mu")
    for k in scal:
        rec[k] = np.full(len(H), np.nan)
    if path.exists():
        z = np.load(path)
        if np.array_equal(z["H0"], H):
            rec = {k: np.array(z[k]) for k in rec}
    t_setup = time.time() - t0
    with h5py.File(RUNS / name / "grid" / "fine.h5", "r") as f:
        ax = np.asarray(f["axis_H0"][()])
        stored = {k: np.asarray(f[k][()]) for k in ("lnL_agn", "log_mu_agn", "pe_var_agn", "n_eff_agn")}
    for j, h in enumerate(H):
        if np.isfinite(rec["seconds"][j]):
            continue
        tc = time.time()
        r = sub.evaluate_events({"H0": float(h)})
        rec["seconds"][j] = time.time() - tc
        rec["ev"][j], rec["var"][j] = r["event_log_evidence"], r["event_mc_variance"]
        for k in scal:
            rec[k][j] = r[k]
        i = int(np.argmin(np.abs(ax - h)))
        info = {"run": name, "args_from": src, "nsamp_agn_subset": sub.nsamp, "n_agn_hosted": n, "setup_seconds": t_setup}
        np.savez(path, **rec, info=json.dumps(info))
        print(f"[step2b] {label} r{R} H0 {h:g}: lnL_agn {r['lnL']:.6f} (stored node {ax[i]:.2f}: {stored['lnL_agn'][i]:.6f}, "
              f"diff {r['lnL'] - stored['lnL_agn'][i]:+.2e}) log_mu {r['log_mu']:.8f} (stored {stored['log_mu_agn'][i]:.8f}) "
              f"({rec['seconds'][j]:.0f} s)", flush=True)
    print(f"[step2b] -> {path} ({time.time() - t0:.0f} s, setup {t_setup:.0f} s)", flush=True)


def analyze(labels):
    import h5py
    DATA = A16.parents[1] / "data"
    EV = {"obslike_f0p30_s1": "events_marked_dmu0p10_dmuG5_obslike.h5", "obslike_f0p30_s0": "events_obslike.h5",
          "peak_f0p30_s1": "events_marked_dmu0p10_dmuG5_corepop.h5", "obslike_f1p00_s0": "events_pureagn_obslike.h5"}
    res = {}
    for label in labels:
        S, Hh, rows, per = [], [], [], {}
        for p in sorted(OUT.glob(f"{label}_r[0-9][0-9][0-9].npz")):
            z = np.load(p)
            if not np.all(np.isfinite(z["seconds"])):
                continue
            R = int(p.stem.split("_r")[-1])
            H = z["H0"]
            n = z["ev"].shape[1]
            lmu = z["log_mu"]
            l = z["ev"] - lmu[:, None]                   # per-merger contribution to the AGN-class ln L
            d = H[2] - H[0]
            s = (l[2] - l[0]) / d
            h = (l[2] - 2 * l[1] + l[0]) / (0.5 * d) ** 2
            s_ev = (z["ev"][2] - z["ev"][0]) / d
            sel = -(lmu[2] - lmu[0]) / d
            with h5py.File(DATA / f"seed{R}" / "events" / EV[label], "r") as f:
                ht = f["truth/host_type"][:]
                T = {k: f["truth/" + k][:][ht == 1] for k in ("z", "m1src", "snr_obs", "q", "chieff")}
                nsmp = int(f.attrs["nsamp"])
                m2 = np.asarray(f["m2src"][:]).reshape(-1, nsmp)[ht == 1]
            T["frac_m2_lt5"] = (m2 < 5.0).mean(axis=1)
            assert np.array_equal(z["agn_mask"].astype(bool), ht == 1) and T["z"].size == n
            S.append(s)
            Hh.append(h)
            rows.append({k: v for k, v in T.items()})
            o = np.argsort(s)
            per[str(R)] = {"n": int(n), "sum_score": float(s.sum()), "sum_event_score": float(s_ev.sum()),
                           "selection_score": float(n * sel), "sum_s2": float((s ** 2).sum()), "minus_sum_h": float(-h.sum()),
                           "z_score": float(s.sum() / np.sqrt(-h.sum())), "im_ratio": float((s ** 2).sum() / -h.sum()),
                           "lnL_minus_stored": None, "most_negative": [(int(i), float(s[i])) for i in o[:5]],
                           "most_positive": [(int(i), float(s[i])) for i in o[::-1][:5]]}
        if not S:
            continue
        s = np.concatenate(S)
        h = np.concatenate(Hh)
        T = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
        mean_s = float(s.mean())
        o = np.sort(s)
        share = {str(k): float(o[:k].sum() - k * mean_s) for k in (1, 3, 10, 30)}
        cor = {k: float(np.corrcoef(s, v)[0, 1]) for k, v in T.items()}
        # trimmed: drop the k most extreme |s - median| mergers and recompute the z of the sum
        def zsum(mask):
            return float(s[mask].sum() / np.sqrt(-h[mask].sum()))
        dev = np.abs(s - np.median(s))
        trims = {}
        for k in (1, 5, 10, 20):
            keep = np.ones(s.size, bool)
            keep[np.argsort(dev)[::-1][:k]] = False
            trims[str(k)] = zsum(keep)
        res[label] = {"per_realization": per, "pooled": {
            "n": int(s.size), "sum_score": float(s.sum()), "minus_sum_h": float(-h.sum()), "z_score": float(s.sum() / np.sqrt(-h.sum())),
            "sum_s2": float((s ** 2).sum()), "im_ratio_sum_s2_over_minus_sum_h": float((s ** 2).sum() / -h.sum()),
            "im_ratio_centered": float(((s - mean_s) ** 2).sum() / -h.sum()),
            "score_sd": float(s.std()), "score_quantiles": [float(x) for x in np.quantile(s, [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999])],
            "excess_from_k_most_negative": share, "z_after_dropping_k_most_extreme": trims, "corr_score_with": cor}}
        P_ = res[label]["pooled"]
        print(f"=== {label}: n={P_['n']} sum score {P_['sum_score']:+.2f} info {P_['minus_sum_h']:.2f} z {P_['z_score']:+.2f} | "
              f"IM ratio {P_['im_ratio_sum_s2_over_minus_sum_h']:.3f} (centered {P_['im_ratio_centered']:.3f}) | "
              f"z after dropping k most extreme {trims} | corr {({k: round(v, 3) for k, v in cor.items()})}")
        for R, v in per.items():
            print(f"   r{R}: n={v['n']} score {v['sum_score']:+.2f} (event {v['sum_event_score']:+.2f}, selection {v['selection_score']:+.2f}) "
                  f"z {v['z_score']:+.2f} IM ratio {v['im_ratio']:.3f} most negative {[(i, round(x, 3)) for i, x in v['most_negative'][:3]]}")
    p = A16 / "tests" / "results" / "obslike_offset_step2b_analysis.json"
    p.write_text(json.dumps(res, indent=1))
    print("wrote", p)


def grid(label, R, classes="both", h0="62.0,74.0,0.2"):
    """grid: the hosts_known 1-D H0 grid of SET at realization R on CPU, for realizations with or without a stored run.
    The run-list arguments of the set's realization-100 run with --realization R (run name ..._R_agn_s32000_inj1), built and
    bound as the driver does; per node HostsKnownModel.evaluate (classes both) or subs['agn'].evaluate (classes agn): ln L,
    pe_var, n_eff, log_mu per class. Summary per class and in total as the run summaries make it: the half-variance corrected
    ln L, Akima-interpolated 16x finer, prod_common.summarize_1d. When a stored run exists its summary is compared.
    -> tests/results/obslike_offset_step2b/grid_<SET>_r<R>_<classes>.npz/.json (resumable). Nothing in runs/ is written."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import prod_common as P
    import prod_driver as PD
    from scipy.interpolate import Akima1DInterpolator
    tmpl, src = runlist_args(KNOWN.format(set=label, R=100))
    args = list(tmpl)
    args[args.index("--realization") + 1] = str(R)
    cfg = PD.resolve(PD.parse_args(args))
    name = KNOWN.format(set=label, R=R)
    if cfg["run_name"] != name:
        raise SystemExit(f"[fatal] resolved run name {cfg['run_name']} != {name}")
    for k in ("events_file", "injections_files", "surveys"):
        for f in (cfg[k].values() if isinstance(cfg[k], dict) else (cfg[k] if isinstance(cfg[k], list) else [cfg[k]])):
            if not Path(f).exists():
                raise SystemExit(f"[fatal] missing {k}: {f}")
    OUT.mkdir(parents=True, exist_ok=True)
    lo, hi, st = (float(x) for x in h0.split(","))
    H = np.round(lo + st * np.arange(int(round((hi - lo) / st)) + 1), 10)
    path = OUT / f"grid_{label}_r{R}_{classes}.npz"
    cls_list = ("agn", "gal") if classes == "both" else ("agn",)
    rec = {"H0": H, "seconds": np.full(len(H), np.nan)}
    for c in cls_list:
        for k in ("lnL", "pe_var", "n_eff", "log_mu"):
            rec[f"{k}_{c}"] = np.full(len(H), np.nan)
    if path.exists():
        z = np.load(path)
        if np.array_equal(z["H0"], H):
            rec = {k: np.array(z[k]) for k in rec}
    t0 = time.time()
    m = PD.setup_model(cfg)
    n_by = {c: int(m.subs[c].n_events) for c in m.subs}
    print(f"[step2b grid] {label} r{R}: built in {time.time() - t0:.0f} s; mergers by class {n_by}; redraw key {cfg['redraw_key']}", flush=True)
    for j, h in enumerate(H):
        if np.isfinite(rec["seconds"][j]):
            continue
        tc = time.time()
        if classes == "both":
            r = m.evaluate({"H0": float(h)})
            for c in cls_list:
                for k in ("lnL", "pe_var", "n_eff", "log_mu"):
                    rec[f"{k}_{c}"][j] = r[f"{k}_{c}"]
        else:
            r = m.subs["agn"].evaluate({"H0": float(h)})
            for k in ("lnL", "pe_var", "n_eff", "log_mu"):
                rec[f"{k}_agn"][j] = r[k]
        rec["seconds"][j] = time.time() - tc
        np.savez(path, **rec)
        if j % 10 == 0:
            print(f"[step2b grid] {label} r{R} H0 {h:g}: {rec['seconds'][j]:.1f} s", flush=True)
    res = {"run": name, "args_from": src + " (realization-100 entry, --realization replaced)", "classes": classes,
           "h0_nodes": [lo, hi, st], "n_by_class": n_by, "seconds_per_node_median": float(np.median(rec["seconds"])),
           "events_file": cfg["events_file"], "injections_files": cfg["injections_files"], "redraw_key": cfg["redraw_key"]}
    lat = np.linspace(H[0], H[-1], (len(H) - 1) * 16 + 1)
    tot = np.zeros(len(H))
    for c in cls_list:
        lc = P.half_variance_corrected(rec[f"lnL_{c}"], rec[f"pe_var_{c}"], rec[f"n_eff_{c}"], n_by[c])
        tot = tot + lc
        res[c] = P.summarize_1d(lat, Akima1DInterpolator(H, lc)(lat))
    if classes == "both":
        res["total"] = P.summarize_1d(lat, Akima1DInterpolator(H, tot)(lat))
    sj = RUNS / name / "summary.json"
    res["stored_summary"] = json.loads(sj.read_text())["posterior"]["H0"] if sj.exists() else None
    (OUT / f"grid_{label}_r{R}_{classes}.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"[step2b grid] {label} r{R}: " + " | ".join(f"{c} {res[c]['median']:.3f} [{res[c]['ci90'][0]:.3f}, {res[c]['ci90'][1]:.3f}]"
          for c in (list(cls_list) + (["total"] if classes == "both" else []))) + f" | stored total {res['stored_summary']}"
          + f" ({time.time() - t0:.0f} s)", flush=True)


def combine(label, reals, classes="both"):
    """combine: the grid-mode outputs of SET over the listed realizations, ln L summed on the common H0 nodes (per class
    and in total, half-variance corrected as the runs), Akima-refined, summarised (prod_common.summarize_1d) with the level of
    the input 67.74 (|2F - 1|). -> tests/results/obslike_offset_step2b/combined_<SET>_<classes>_<first>-<last>.json"""
    import prod_common as P
    from scipy.interpolate import Akima1DInterpolator
    tot, H, used = {}, None, []
    for R in reals:
        p = OUT / f"grid_{label}_r{R}_{classes}.npz"
        if not p.exists():
            print(f"[combine] missing {p.name}")
            continue
        z = np.load(p)
        if not np.all(np.isfinite(z["seconds"])):
            print(f"[combine] unfinished {p.name}")
            continue
        info = json.loads((OUT / f"grid_{label}_r{R}_{classes}.json").read_text())
        if H is None:
            H = z["H0"]
        elif not np.array_equal(H, z["H0"]):
            raise SystemExit("[fatal] different H0 nodes")
        for c in (("agn", "gal") if classes == "both" else ("agn",)):
            lc = P.half_variance_corrected(z[f"lnL_{c}"], z[f"pe_var_{c}"], z[f"n_eff_{c}"], info["n_by_class"][c])
            tot[c] = tot.get(c, 0.0) + lc
        used.append(R)
    if not used:
        return
    if classes == "both":
        tot["total"] = tot["agn"] + tot["gal"]
    lat = np.linspace(H[0], H[-1], (len(H) - 1) * 16 + 1)
    res = {"set": label, "realizations": used, "classes": classes}
    for c, lc in tot.items():
        lv = Akima1DInterpolator(H, lc)(lat)
        sm = P.summarize_1d(lat, lv)
        w = np.exp(lv - lv.max())
        cdf = np.concatenate([[0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(lat))])
        cdf /= cdf[-1]
        F = float(np.interp(67.74, lat, cdf))
        sm.update({"cdf_at_input": F, "level_of_input": abs(2 * F - 1)})
        res[c] = sm
        print(f"[combine] {label} {c:5s} over {used}: median {sm['median']:.3f} 90% [{sm['ci90'][0]:.3f}, {sm['ci90'][1]:.3f}] "
              f"sd {sm['sd']:.3f}; input at the {100 * sm['level_of_input']:.1f}% level", flush=True)
    (OUT / f"combined_{label}_{classes}_{used[0]}-{used[-1]}.json").write_text(json.dumps(res, indent=1, default=float))


def analyze_mc(labels):
    """Monte Carlo noise of the per-merger scores: the same mergers evaluated with an independent production redraw
    (--redraw-index 1, 32000 new samples per AGN-hosted merger): Var_MC(s_i) = Var(s_i,0 - s_i,1) / 2, the summed score and
    its z with the second redraw, and the information-matrix ratio with the Monte Carlo part removed."""
    res = {}
    for label in labels:
        rows = []
        for p in sorted(OUT.glob(f"{label}_r[0-9][0-9][0-9]_rd1.npz")):
            R = int(p.stem.split("_r")[1])
            q = OUT / f"{label}_r{R}.npz"
            a, b = np.load(q), np.load(p)
            if not (np.all(np.isfinite(a["seconds"])) and np.all(np.isfinite(b["seconds"]))):
                continue
            d = a["H0"][2] - a["H0"][0]
            out = {}
            for tag, z in (("rd0", a), ("rd1", b)):
                l = z["ev"] - z["log_mu"][:, None]
                out[tag] = ((l[2] - l[0]) / d, (l[2] - 2 * l[1] + l[0]) / (0.5 * d) ** 2, z["lnL"], z["log_mu"])
            s0, h0, L0, m0 = out["rd0"]
            s1, h1, L1, m1 = out["rd1"]
            rows.append({"R": R, "s0": s0, "s1": s1, "h0": h0, "h1": h1, "dlogmu": float(m1[1] - m0[1]),
                         "sum0": float(s0.sum()), "sum1": float(s1.sum()), "z0": float(s0.sum() / np.sqrt(-h0.sum())),
                         "z1": float(s1.sum() / np.sqrt(-h1.sum()))})
        if not rows:
            continue
        s0 = np.concatenate([r["s0"] for r in rows]); s1 = np.concatenate([r["s1"] for r in rows])
        h0 = np.concatenate([r["h0"] for r in rows]); h1 = np.concatenate([r["h1"] for r in rows])
        var_mc = float(np.var(s0 - s1) / 2)
        info = float(-0.5 * (h0.sum() + h1.sum()))
        sm = 0.5 * (s0 + s1)
        R_ = {"realizations": [r["R"] for r in rows], "n": int(s0.size),
              "sum_score_rd0": float(s0.sum()), "sum_score_rd1": float(s1.sum()), "sum_score_mean_of_two": float(sm.sum()),
              "z_rd0": float(s0.sum() / np.sqrt(-h0.sum())), "z_rd1": float(s1.sum() / np.sqrt(-h1.sum())),
              "z_mean_of_two": float(sm.sum() / np.sqrt(info)),
              "per_merger_score_var_total": float(np.mean(s0 ** 2)), "per_merger_score_var_mc": var_mc,
              "per_merger_info": info / s0.size,
              "im_ratio_rd0": float((s0 ** 2).sum() / -h0.sum()),
              "im_ratio_mc_removed": float(((s0 ** 2).sum() - s0.size * var_mc) / -h0.sum()),
              "sd_of_summed_score_from_mc": float(np.sqrt(s0.size * var_mc)),
              "per_realization": {str(r["R"]): {k: r[k] for k in ("sum0", "sum1", "z0", "z1", "dlogmu")} for r in rows}}
        res[label] = R_
        print(f"=== {label} (redraw 0 vs 1, realizations {R_['realizations']}): summed score {R_['sum_score_rd0']:+.2f} vs "
              f"{R_['sum_score_rd1']:+.2f} (z {R_['z_rd0']:+.2f} vs {R_['z_rd1']:+.2f}; mean of two {R_['z_mean_of_two']:+.2f}) | per-merger "
              f"score var {R_['per_merger_score_var_total']:.5f}, of which MC {var_mc:.5f}; info/merger {R_['per_merger_info']:.5f} | "
              f"IM ratio {R_['im_ratio_rd0']:.3f} -> {R_['im_ratio_mc_removed']:.3f} without MC | MC sd of the sum {R_['sd_of_summed_score_from_mc']:.2f}")
        for k, v in R_["per_realization"].items():
            print(f"   r{k}: sum {v['sum0']:+.2f} vs {v['sum1']:+.2f}  z {v['z0']:+.2f} vs {v['z1']:+.2f}  d log_mu(67.7) {v['dlogmu']:+.2e}")
    p = A16 / "tests" / "results" / "obslike_offset_step2b_mc_redraw.json"
    p.write_text(json.dumps(res, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    if sys.argv[1] == "compute":
        rd = 0
        argv = sys.argv[3:]
        if "--redraw" in argv:
            i = argv.index("--redraw")
            rd = int(argv[i + 1])
            argv = argv[:i] + argv[i + 2:]
        for R in argv:
            compute(sys.argv[2], int(R), rd)
    elif sys.argv[1] == "grid":
        kw = {}
        argv = sys.argv[3:]
        for opt in ("--classes", "--h0"):
            if opt in argv:
                i = argv.index(opt)
                kw[opt[2:]] = argv[i + 1]
                argv = argv[:i] + argv[i + 2:]
        for R in argv:
            grid(sys.argv[2], int(R), **kw)
    elif sys.argv[1] == "combine":
        argv = sys.argv[3:]
        cl = "both"
        if "--classes" in argv:
            i = argv.index("--classes")
            cl = argv[i + 1]
            argv = argv[:i] + argv[i + 2:]
        combine(sys.argv[2], [int(x) for x in argv], cl)
    elif sys.argv[1] == "mc":
        analyze_mc(sys.argv[2:])
    else:
        analyze(sys.argv[2:])
