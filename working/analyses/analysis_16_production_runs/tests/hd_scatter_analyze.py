#!/usr/bin/env python
"""Numpy analysis of the hd_scatter_eval.py outputs (peak_f0p30_s1 two-population H0 scatter, 25 Nov). CPU only.

    <core venv python> tests/hd_scatter_analyze.py t1 R [R ...]          per-merger decomposition of ln L(lo) - ln L(hi)
    ... t2census R [R ...]                                                 effective-sample census at each run's peak node
    ... t2hybrid R [--tag T]                                               stored vs hybrid line (hybrid_<slug><tag>.npz)
    ... t2redraw R --tags _rd1[,_rd2]                                      stored vs redrawn line at the redraw's nodes
    ... t3 R [R ...]                                                       the selection term (sel_<slug>*.npz + stored grid)
    ... t4 RUN [RUN ...]                                                   information equality at the input (3 nodes, f 0.30)
-> tests/results/hd_scatter_<mode>.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import obslike_steps_merger as S  # noqa: E402
import obslike_steps_halfwidth as HW  # noqa: E402
import hd_scatter_eval as E  # noqa: E402

A16, RUNS, OUT = S.A16, S.RUNS, E.OUT
RES = A16 / "tests" / "results"
BASE = "two_tracers_two_populations/both_catalogs/complete/peak_f0p30_s1/realization_{R:03d}"
H_IN = 67.74


def load_nodes(name, tag=""):
    z = np.load(OUT / f"nodes_{E.slug(name)}{tag}.npz")
    d = {k: z[k] for k in z.files if k != "meta"}
    d["meta"] = json.loads(str(z["meta"]))
    return d


def pt(d, h, f):
    j = np.where(np.isclose(d["H0"], h) & np.isclose(d["f"], f))[0]
    return int(j[0]) if j.size else None


def truth(events_file):
    import h5py
    with h5py.File(events_file, "r") as f:
        t = f["truth"]
        return {k: np.asarray(t[k][:]) for k in ("host_type", "z", "m1src", "m2src", "q", "chieff", "snr_obs", "dl")}


def check_rows(d):
    rows = []
    for j in range(len(d["H0"])):
        rows.append({"H0": float(d["H0"][j]), "f": float(d["f"][j]),
                     "sum_lnLi_minus_stored": float(d["sum_ev"][j] - d["stored_lnL_minus_selcorr"][j]),
                     "pe_var_minus_stored": float(d["pe_var"][j] - d["stored_pe_var"][j])})
    return rows


# ============================================================================ T1
def t1(reals):
    out = {}
    for R in reals:
        name = BASE.format(R=R)
        d = load_nodes(name)
        T = truth(d["meta"]["events_file"])
        ht = T["host_type"]
        N = ht.size
        fmax = d["meta"]["f_max"]
        # the pair: (64, f) against (the node nearest the input, f), f = f_max when both are evaluated there, else 0.30
        pairs = []
        for f in (fmax, 0.30):
            jl = pt(d, 64.0, f)
            cand = [j for j in range(len(d["H0"])) if np.isclose(d["f"][j], f) and abs(d["H0"][j] - H_IN) < 0.15]
            if jl is not None and cand:
                pairs.append((jl, min(cand, key=lambda j: abs(d["H0"][j] - H_IN)), f))
                break
        jl, jh, f = pairs[0]
        extra = []
        jx = pt(d, 70.0, f)
        if jx is not None:
            extra.append((jx, jh, f))
        res = {"run": name, "f": f, "f_max": fmax, "check": check_rows(d), "n_gal": int((ht == 0).sum()), "n_agn": int((ht == 1).sum())}
        for (a, b, ff) in [(jl, jh, f)] + extra:
            ha, hb = float(d["H0"][a]), float(d["H0"][b])
            key = f"lnL({ha:g})-lnL({hb:g})"
            D = d["ev"][a] - d["ev"][b]
            dsel = float(d["stored_sel_corr"][a] - d["stored_sel_corr"][b])
            # half-variance correction difference (the posterior is built from the corrected ln L)
            dhv = float(0.5 * (d["stored_pe_var"][a] - d["stored_pe_var"][b]) - 0.5 * N * (1 / d["stored_n_eff"][a] - 1 / d["stored_n_eff"][b]))
            tot_st = float(d["stored_lnL_minus_selcorr"][a] + d["stored_sel_corr"][a] - d["stored_lnL_minus_selcorr"][b] - d["stored_sel_corr"][b])
            r = {"total_stored": tot_st, "sum_events": float(D.sum()), "selection": dsel, "half_variance_corr": dhv,
                 "check_events_plus_selection_minus_stored": float(D.sum() + dsel - tot_st)}
            mc = d.get("mcvar_pair")
            for cname, msk in (("gal", ht == 0), ("agn", ht == 1)):
                x = D[msk]
                o = np.argsort(-x)
                xs = x + dsel / N                  # each merger with its share of the selection term (-ln mu per merger)
                c = {"n": int(msk.sum()), "sum": float(x.sum()), "sum_with_selection_share": float(xs.sum()),
                     "mean_with_selection_share": float(xs.mean()), "median_with_selection_share": float(np.median(xs)),
                     "sum_with_selection_share_without_10_largest": float(xs[o[10:]].sum()),
                     "sum_with_selection_share_without_50_largest": float(xs[o[50:]].sum()),
                     "mean": float(x.mean()), "median": float(np.median(x)),
                     "sd": float(x.std(ddof=1)), "sum_of_10_largest": float(x[o[:10]].sum()), "sum_of_50_largest": float(x[o[:50]].sum()),
                     "sum_without_10_largest": float(x[o[10:]].sum()), "sum_without_50_largest": float(x[o[50:]].sum()),
                     "n_positive": int((x > 0).sum()), "sum_of_10_most_negative": float(np.sort(x)[:10].sum())}
                if mc is not None and np.isfinite(mc[a, b]).all():
                    c["mc_sd_of_sum"] = float(np.sqrt(mc[a, b][msk].sum()))
                r[cname] = c
            o = np.argsort(-D)
            r["share_top10_of_events"] = float(D[o[:10]].sum() / D.sum()) if D.sum() != 0 else None
            r["share_top50_of_events"] = float(D[o[:50]].sum() / D.sum()) if D.sum() != 0 else None
            if mc is not None and np.isfinite(mc[a, b]).all():
                r["mc_sd_of_sum_events"] = float(np.sqrt(mc[a, b].sum()))
            top = []
            for i in o[:10]:
                row = {"i": int(i), "D": float(D[i]), "host": "agn" if ht[i] == 1 else "gal", "ess_lo": float(d["ess"][a, i]),
                       "ess_hi": float(d["ess"][b, i])}
                row.update({k: float(T[k][i]) for k in ("z", "m1src", "m2src", "chieff", "snr_obs", "dl")})
                if mc is not None and np.isfinite(mc[a, b, i]):
                    row["mc_sd_D"] = float(np.sqrt(mc[a, b, i]))
                top.append(row)
            r["top10"] = top
            r["top10_indices"] = [int(i) for i in o[:10]]
            # distribution comparison: quantiles of D by class
            r["quantiles_D_all"] = [float(q) for q in np.quantile(D, [0.01, 0.1, 0.5, 0.9, 0.99])]
            res[key] = r
            print(f"=== r{R} {key} at f {ff}: stored total {tot_st:+.3f} = events {D.sum():+.3f} + selection {dsel:+.3f} "
                  f"(check {r['check_events_plus_selection_minus_stored']:+.1e}); half-var corr {dhv:+.3f}")
            for cname in ("gal", "agn"):
                c = r[cname]
                print(f"   {cname} with selection share: sum {c['sum_with_selection_share']:+.3f}, without 10 largest "
                      f"{c['sum_with_selection_share_without_10_largest']:+.3f}, without 50 largest {c['sum_with_selection_share_without_50_largest']:+.3f}; "
                      f"mean {c['mean_with_selection_share']:+.4f} median {c['median_with_selection_share']:+.4f}")
                print(f"   {cname} n={c['n']} sum {c['sum']:+.3f} mean {c['mean']:+.4f} median {c['median']:+.4f} sd {c['sd']:.3f} "
                      f"| top10 {c['sum_of_10_largest']:+.3f} top50 {c['sum_of_50_largest']:+.3f} rest(after 50) {c['sum_without_50_largest']:+.3f} "
                      f"| n>0 {c['n_positive']}" + (f" | MC sd of sum {c['mc_sd_of_sum']:.3f}" if "mc_sd_of_sum" in c else ""))
            for t_ in top:
                print("     ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in t_.items()})
        out[str(R)] = res
    p = RES / "hd_scatter_t1.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(out)
    p.write_text(json.dumps(old, indent=1))
    print("wrote", p)


# ============================================================================ T2
def t2census(reals):
    out = {}
    for R in reals:
        name = BASE.format(R=R)
        d = load_nodes(name)
        summ = json.loads((RUNS / name / "summary.json").read_text())["posterior"]["H0"]
        j = int(np.argmin(np.abs(d["H0"] - summ["median"]) + 10 * np.abs(d["f"] - d["meta"]["f_max"])))
        e = d["ess"][j]
        out[str(R)] = {"node": [float(d["H0"][j]), float(d["f"][j])], **{f"n_ess_lt_{c}": int((e < c).sum()) for c in (5, 20, 50, 100)},
                       "ess_quantiles_1_5_50pct": [float(x) for x in np.quantile(e, [0.01, 0.05, 0.5])],
                       "pe_var_sum": float(d["var"][j].sum()), "lowest": [(int(i), float(e[i])) for i in np.argsort(e)[:6]]}
        print(f"[census] r{R} at {out[str(R)]['node']}: " + ", ".join(f"<{c}: {out[str(R)][f'n_ess_lt_{c}']}" for c in (5, 20, 50, 100))
              + f"; pe_var sum {out[str(R)]['pe_var_sum']:.3f}; lowest {out[str(R)]['lowest'][:4]}")
    (RES / "hd_scatter_t2census.json").write_text(json.dumps(out, indent=1))


def corrected(lnL, pv, neff, N):
    return lnL + 0.5 * pv - 0.5 * N / neff


def t2hybrid(R, tag=""):
    name = BASE.format(R=R)
    z = np.load(OUT / f"hybrid_{E.slug(name)}{tag}.npz")
    meta = json.loads(str(z["meta"]))
    line = HW.stored_line(name)
    N = line["n_events"]
    H = z["H0"]
    ix = [int(np.argmin(np.abs(line["H0"] - h))) for h in H]
    lnL, pv, ne = line["lnL"][ix], line["pe_var"][ix], line["n_eff"][ix]
    d_ev = np.zeros(len(H))
    d_var = np.zeros(len(H))
    parts = {}
    for grp in ("g4", "top"):
        if f"{grp}_ev_M" in z.files:
            de = (z[f"{grp}_ev_M"] - z[f"{grp}_ev_N"]).sum(1)
            dv = (z[f"{grp}_var_M"] - z[f"{grp}_var_N"]).sum(1)
            d_ev += de
            d_var += dv
            parts[grp] = {"d_ev": de.tolist(), "d_var": dv.tolist()}
    s_full = HW.line_summary(line["H0"], corrected(line["lnL"], line["pe_var"], line["n_eff"], N))
    s_st = HW.line_summary(H, corrected(lnL, pv, ne, N))
    s_hy = HW.line_summary(H, corrected(lnL + d_ev, pv + d_var, ne, N))
    s_hy_g4 = None
    if "g4" in parts:
        s_hy_g4 = HW.line_summary(H, corrected(lnL + np.array(parts["g4"]["d_ev"]), pv + np.array(parts["g4"]["d_var"]), ne, N))
    out = {"run": name, "f": float(z["f"]), "meta": meta, "K4": z["K4"].tolist(), "top": z["top"].tolist(),
           "stored_all_nodes": s_full, "stored_at_hybrid_nodes": s_st, "hybrid": s_hy, "hybrid_g4_only": s_hy_g4,
           "summary_marginal": json.loads((RUNS / name / "summary.json").read_text())["posterior"]["H0"],
           "H0": H.tolist(), "d_ev_sum": d_ev.tolist(), "d_var_sum": d_var.tolist(), "parts": parts,
           "checks": {k: float(z[k]) for k in z.files if "_check_" in k}}
    print(f"[hybrid] r{R} f {out['f']}: {len(out['K4'])} mergers at {meta['mult']}x, {len(out['top'])} at {meta['top_mult']}x")
    for k in ("stored_all_nodes", "stored_at_hybrid_nodes", "hybrid", "hybrid_g4_only"):
        if out[k]:
            print(f"   {k:24s} median {out[k]['median']:.3f} 90% [{out[k]['ci90'][0]:.3f}, {out[k]['ci90'][1]:.3f}] hw {out[k]['hw90']:.3f}")
    print("   checks", out["checks"])
    (RES / f"hd_scatter_t2hybrid_r{R}{tag}.json").write_text(json.dumps(out, indent=1, default=float))


def t2redraw(R, tags):
    name = BASE.format(R=R)
    line = HW.stored_line(name)
    N = line["n_events"]
    out = {"run": name, "f": line["f"]}
    meds = {}
    for tag in tags:
        d = load_nodes(name, tag)
        m = np.isclose(d["f"], line["f"])
        o = np.argsort(d["H0"][m])
        H = d["H0"][m][o]
        sev, pv = d["sum_ev"][m][o], d["pe_var"][m][o]
        ix = [int(np.argmin(np.abs(line["H0"] - h))) for h in H]
        if max(abs(line["H0"][i] - h) for i, h in zip(ix, H)) > 1e-6:
            raise SystemExit("[fatal] redraw nodes not on the stored line")
        selc, ne = line["sel_corr"][ix], line["n_eff"][ix]
        lnL_st, pv_st = line["lnL"][ix], line["pe_var"][ix]
        s_st = HW.line_summary(H, corrected(lnL_st, pv_st, ne, N))
        s_rd = HW.line_summary(H, corrected(sev + selc, pv, ne, N))
        s_rd_u = HW.line_summary(H, sev + selc)
        s_st_u = HW.line_summary(H, lnL_st)
        sd_post = (s_st["ci90"][1] - s_st["ci90"][0]) / (2 * 1.645)
        out[tag] = {"H0": H.tolist(), "stored": s_st, "redraw": s_rd, "stored_uncorrected": s_st_u, "redraw_uncorrected": s_rd_u,
                    "median_shift": s_rd["median"] - s_st["median"], "shift_over_posterior_sd": (s_rd["median"] - s_st["median"]) / sd_post,
                    "lnL_redraw_minus_stored": (sev + selc - lnL_st).tolist(), "pe_var_redraw": pv.tolist(), "pe_var_stored": pv_st.tolist()}
        meds[tag] = s_rd["median"]
        print(f"[redraw] r{R} {tag} f {line['f']} nodes {H.tolist()}: stored median {s_st['median']:.3f} {s_st['ci90']} | redraw "
              f"{s_rd['median']:.3f} {s_rd['ci90']} | shift {out[tag]['median_shift']:+.3f} = {out[tag]['shift_over_posterior_sd']:+.2f} sd")
        print("   lnL_redraw - lnL_stored (rel. to the 64 node):", np.round(np.array(out[tag]["lnL_redraw_minus_stored"]) - out[tag]["lnL_redraw_minus_stored"][int(np.argmin(np.abs(H - 64)))], 3).tolist())
    (RES / f"hd_scatter_t2redraw_r{R}.json").write_text(json.dumps(out, indent=1, default=float))


# ============================================================================ T3
def t3(reals, nboot=200, seed=7):
    import h5py
    rng = np.random.default_rng(seed)
    out = {}
    for R in reals:
        name = BASE.format(R=R)
        line = HW.stored_line(name)
        N = line["n_events"]
        H = line["H0"]
        res = {"run": name, "f": line["f"]}
        # along H0 at f_max: the selection's part of ln L and N^2/N_eff (stored)
        res["along_H0"] = {"H0": H.tolist(), "sel_corr": line["sel_corr"].tolist(), "N2_over_Neff": (N ** 2 / line["n_eff"]).tolist()}
        for fn in sorted(OUT.glob(f"sel_{E.slug(name)}*.npz")):
            z = np.load(fn)
            meta = json.loads(str(z["meta"]))
            ldw = z["ldw"]
            ja, jb = pt(z, 64.0, line["f"]), int(np.argmin(np.abs(z["H0"] - H_IN)))
            if ja is None:
                continue
            nd = float(z["n_draw"])
            la, lb = S.lse(ldw[ja]), S.lse(ldw[jb])
            wa, wb = np.exp(ldw[ja] - la), np.exp(ldw[jb] - lb)
            delta = -N * (la - lb)               # selection's part of ln L(64) - ln L(input node)
            sd_delta = N * np.sqrt(np.sum((wa - wb) ** 2))
            # Poisson bootstrap over the detected injections (the undetected carry zero weight)
            bs = []
            for _ in range(nboot):
                c = rng.poisson(1.0, size=wa.size).astype(float)
                bs.append(-N * (np.log(np.sum(c * wa)) - np.log(np.sum(c * wb))))
            bs = np.array(bs)
            # ten independent tenths (random partition)
            part = rng.permutation(wa.size) % 10
            tenth = np.array([-N * (np.log(np.sum(wa[part == k])) - np.log(np.sum(wb[part == k]))) for k in range(10)])
            key = fn.stem
            res[key] = {"injections": meta["injections_files"], "n_detected": meta["n_detected"], "n_draw": nd,
                        "nodes": [float(z["H0"][ja]), float(z["H0"][jb])], "f": float(z["f"][ja]),
                        "log_mu": z["log_mu"].tolist(), "log_mu_minus_stored": (z["log_mu"] - z["stored_log_mu"]).tolist(),
                        "n_eff": z["n_eff"].tolist(), "N2_over_Neff": (N ** 2 / z["n_eff"]).tolist(),
                        "sel_part_lnL64_minus_lnLin": float(delta), "sd_delta_method": float(sd_delta),
                        "sd_bootstrap": float(bs.std(ddof=1)), "bootstrap_mean_minus_value": float(bs.mean() - delta),
                        "tenths_values": tenth.tolist(), "sd_from_tenths_for_full_set": float(tenth.std(ddof=1) / np.sqrt(10)),
                        "sd_single_node_N_over_sqrtNeff": [float(N / np.sqrt(x)) for x in z["n_eff"]]}
            r = res[key]
            print(f"[sel] r{R} {key} ({meta['n_detected']} detected): selection part of lnL(64)-lnL({r['nodes'][1]:g}) at f {r['f']}: "
                  f"{delta:+.3f}; MC sd: delta {sd_delta:.3f}, bootstrap {r['sd_bootstrap']:.3f}, tenths {r['sd_from_tenths_for_full_set']:.3f}; "
                  f"N^2/N_eff {np.round(r['N2_over_Neff'], 3).tolist()}; log_mu - stored {np.array(r['log_mu_minus_stored'])}")
        jj = [int(np.argmin(np.abs(H - h))) for h in (60, 62, 64, 66, 67.74, 69)]
        print(f"[sel] r{R} stored along H0 at f {line['f']}: " + "; ".join(f"{H[j]:g}: sel {line['sel_corr'][j]:.2f}, N2/Neff {N ** 2 / line['n_eff'][j]:.3f}" for j in jj))
        out[str(R)] = res
    (RES / "hd_scatter_t3.json").write_text(json.dumps(out, indent=1, default=float))


# ============================================================================ T4
def t4(names, nboot=2000, seed=11):
    rng = np.random.default_rng(seed)
    p = RES / "hd_scatter_t4.json"
    out = json.loads(p.read_text()) if p.exists() else {}
    for name in names:
        d = load_nodes(name)
        T = truth(d["meta"]["events_file"])
        ht = T["host_type"]
        N = ht.size
        m = np.where(np.isclose(d["f"], 0.30))[0]
        m = m[np.argsort(d["H0"][m])]
        c3 = [j for j in m if abs(d["H0"][j] - H_IN) < 0.75]
        jc = min(c3, key=lambda j: abs(d["H0"][j] - H_IN))
        k = list(m).index(jc)
        jm, jp = m[k - 1], m[k + 1]
        hm, h0, hp = d["H0"][jm], d["H0"][jc], d["H0"][jp]
        if not np.isclose(hp - h0, h0 - hm):
            raise SystemExit("[fatal] unequal spacing")
        dh = h0 - hm
        ev = d["ev"]
        s = (ev[jp] - ev[jm]) / (2 * dh)
        c = (ev[jp] - 2 * ev[jc] + ev[jm]) / dh ** 2
        lm = d["stored_log_mu"]
        mu1 = (lm[jp] - lm[jm]) / (2 * dh)
        mu2 = (lm[jp] - 2 * lm[jc] + lm[jm]) / dh ** 2
        mcv = d["mcvar_pair"][jp, jm] / (2 * dh) ** 2 if "mcvar_pair" in d and np.isfinite(d["mcvar_pair"][jp, jm]).all() else None
        res = {"run": name, "nodes": [float(hm), float(h0), float(hp)], "f": 0.30, "dlogmu_dH0": float(mu1), "d2logmu_dH02": float(mu2),
               "check": check_rows(d)}

        def ratio(msk, with_sel):
            sc = s[msk] - (mu1 if with_sel else 0.0)
            num_c = float(((s[msk] - s[msk].mean()) ** 2).sum())
            num_u = float((sc ** 2).sum())
            info = float(-c[msk].sum() + (msk.sum() * mu2 if with_sel else 0.0))
            bs, bnum, bden = [], [], []
            idx = np.where(msk)[0]
            for _ in range(nboot):
                b = rng.choice(idx, idx.size)
                sb = s[b] - (mu1 if with_sel else 0.0)
                nu = (sb ** 2).sum() if with_sel else ((sb - sb.mean()) ** 2).sum()
                de = -c[b].sum() + (b.size * mu2 if with_sel else 0.0)
                bs.append(nu / de)
                bnum.append(nu)
                bden.append(de)
            q = np.percentile(bs, [5, 16, 50, 84, 95])
            o = {"n": int(msk.sum()), "sum_score": float(sc.sum()), "var_sum_centered": num_c, "sum_sq_uncentered": num_u,
                 "info": info, "ratio_centered": num_c / info, "ratio_uncentered": num_u / info, "ratio_sd_bootstrap": float(np.std(bs, ddof=1)),
                 "ratio_bootstrap_5_16_50_84_95": [float(x) for x in q], "numerator_sd_bootstrap": float(np.std(bnum, ddof=1)),
                 "info_sd_bootstrap": float(np.std(bden, ddof=1)), "z_of_summed_score": float(sc.sum() / np.sqrt(info)),
                 "curvature_sd_per_merger": float(c[msk].std()), "curvature_quantiles_0.1_50_99.9": [float(x) for x in np.quantile(c[msk], [0.001, 0.5, 0.999])]}
            if mcv is not None:
                o["mc_var_of_scores_sum"] = float(mcv[msk].sum())
                o["ratio_centered_mc_removed"] = (num_c - float(mcv[msk].sum())) / info
            return o

        res["events_gal"] = ratio(ht == 0, False)
        res["events_agn"] = ratio(ht == 1, False)
        res["events_all"] = ratio(np.ones(N, bool), False)
        res["with_selection_gal"] = ratio(ht == 0, True)
        res["with_selection_agn"] = ratio(ht == 1, True)
        res["with_selection_all"] = ratio(np.ones(N, bool), True)
        # the stored grid's own curvature along H0 at f 0.30 (quadratic over +-1.05 of the node), events and selection
        import h5py
        line = HW.stored_line(name)
        with h5py.File(RUNS / name / "grid" / line["file"], "r") as F:
            ax, af = np.asarray(F["axis_H0"][()]), np.asarray(F["axis_f_agn"][()])
            jf = int(np.argmin(np.abs(af - 0.30)))
            lnL_g, sc_g = np.asarray(F["lnL"][()])[:, jf], np.asarray(F["sel_corr"][()])[:, jf]
        near = np.abs(ax - h0) <= 1.05
        cq = np.polyfit(ax[near] - h0, lnL_g[near], 2)
        cs = np.polyfit(ax[near] - h0, (lnL_g - sc_g)[near], 2)
        res["stored_quadratic"] = {"info_total": float(-2 * cq[0]), "info_events": float(-2 * cs[0]), "slope_total": float(cq[1]),
                                   "n_nodes": int(near.sum())}
        res["with_selection_all"]["ratio_with_stored_info"] = res["with_selection_all"]["var_sum_centered"] / res["stored_quadratic"]["info_total"]
        res["model_sd_H0"] = float(1 / np.sqrt(res["with_selection_all"]["info"]))
        res["sandwich_sd_H0"] = float(np.sqrt(res["with_selection_all"]["var_sum_centered"]) / res["with_selection_all"]["info"])
        out[name] = res
        print(f"[t4] {name} nodes {res['nodes']} f 0.30: dlnmu/dH0 {mu1:+.5f} d2 {mu2:+.6f}")
        for k in ("events_gal", "events_agn", "events_all", "with_selection_gal", "with_selection_agn", "with_selection_all"):
            o = res[k]
            q = o["ratio_bootstrap_5_16_50_84_95"]
            print(f"   {k:20s} n={o['n']:4d} sum s {o['sum_score']:+8.3f} sum(s-sbar)^2 {o['var_sum_centered']:7.3f} (+-{o['numerator_sd_bootstrap']:.3f}) "
                  f"info {o['info']:7.3f} (+-{o['info_sd_bootstrap']:.3f}) ratio c {o['ratio_centered']:.3f} u {o['ratio_uncentered']:.3f} "
                  f"boot16-84 [{q[1]:.2f}, {q[3]:.2f}] curv sd/merger {o['curvature_sd_per_merger']:.4f}"
                  + (f" MC-removed {o['ratio_centered_mc_removed']:.3f}" if "ratio_centered_mc_removed" in o else "") + f" z {o['z_of_summed_score']:+.2f}")
        print(f"   model sd {res['model_sd_H0']:.3f}, sandwich sd {res['sandwich_sd_H0']:.3f}; stored quadratic info {res['stored_quadratic']['info_total']:.3f} "
              f"(events {res['stored_quadratic']['info_events']:.3f}) -> ratio with it {res['with_selection_all']['ratio_with_stored_info']:.3f}")
    p.write_text(json.dumps(out, indent=1, default=float))
    print("wrote", p)
    # pooled over the runs of each family
    fam = {}
    for nm, r in out.items():
        k = "control_equal_populations" if nm.startswith("known_population") else "peak_f0p30_s1_two_populations"
        fam.setdefault(k, []).append(nm)
    pooled = {}
    for k, nms in fam.items():
        w = [out[n]["with_selection_all"] for n in nms]
        num = sum(x["var_sum_centered"] for x in w)
        den = sum(x["info"] for x in w)
        den_st = sum(out[n]["stored_quadratic"]["info_total"] for n in nms)
        sd_num = np.sqrt(sum(x["numerator_sd_bootstrap"] ** 2 for x in w))
        sd_den = np.sqrt(sum(x["info_sd_bootstrap"] ** 2 for x in w))
        mc = [x.get("mc_var_of_scores_sum") for x in w]
        pr = {"runs": [n.split("/")[-2] + "/" + n.split("/")[-1] for n in nms], "sum_var_scores": num, "sum_info_per_merger_curvature": den,
              "ratio": num / den, "ratio_sd_approx": float(num / den * np.sqrt((sd_num / num) ** 2 + (sd_den / den) ** 2)),
              "sum_info_stored_quadratic": den_st, "ratio_with_stored_info": num / den_st,
              "ratio_with_stored_info_sd_approx": float(sd_num / den_st),
              "sum_z2_of_summed_score": float(sum(x["z_of_summed_score"] ** 2 for x in w)), "n_runs": len(nms)}
        if all(m is not None for m in mc):
            pr["ratio_mc_removed"] = (num - sum(mc)) / den
            pr["ratio_with_stored_info_mc_removed"] = (num - sum(mc)) / den_st
            pr["sum_mc_var"] = float(sum(mc))
        sub = [i for i, m in enumerate(mc) if m is not None]
        if sub and len(sub) < len(mc):            # the runs with the Monte Carlo variance of the scores, alone
            n_s = sum(w[i]["var_sum_centered"] for i in sub)
            d_s = sum(w[i]["info"] for i in sub)
            d_st = sum(out[nms[i]]["stored_quadratic"]["info_total"] for i in sub)
            m_s = sum(mc[i] for i in sub)
            pr["subset_with_mc"] = {"runs": [nms[i].split("/")[-1] for i in sub], "ratio": n_s / d_s, "ratio_mc_removed": (n_s - m_s) / d_s,
                                    "ratio_with_stored_info": n_s / d_st, "ratio_with_stored_info_mc_removed": (n_s - m_s) / d_st,
                                    "ratio_sd_approx": float(n_s / d_s * np.sqrt(sum(w[i]["numerator_sd_bootstrap"] ** 2 for i in sub) / n_s ** 2
                                                                                + sum(w[i]["info_sd_bootstrap"] ** 2 for i in sub) / d_s ** 2)),
                                    "mc_share_of_numerator": m_s / n_s}
            print(f"   subset with MC variance {pr['subset_with_mc']['runs']}: ratio {n_s / d_s:.3f} +- {pr['subset_with_mc']['ratio_sd_approx']:.3f}, "
                  f"MC removed {(n_s - m_s) / d_s:.3f}; with stored curvature {n_s / d_st:.3f} -> {(n_s - m_s) / d_st:.3f}; MC share {m_s / n_s:.2f}")
        pooled[k] = pr
        print(f"[t4 pooled] {k} ({len(nms)} runs): ratio {pr['ratio']:.3f} +- {pr['ratio_sd_approx']:.3f} (per-merger curvature); with stored "
              f"curvature {pr['ratio_with_stored_info']:.3f} +- {pr['ratio_with_stored_info_sd_approx']:.3f}"
              + (f"; MC removed {pr['ratio_mc_removed']:.3f} / {pr['ratio_with_stored_info_mc_removed']:.3f} (sum MC var {pr['sum_mc_var']:.3f})" if "ratio_mc_removed" in pr else "")
              + f"; sum z^2 of the summed score {pr['sum_z2_of_summed_score']:.2f} on {len(nms)}")
    (RES / "hd_scatter_t4_pooled.json").write_text(json.dumps(pooled, indent=1, default=float))


def t1bins(specs):
    """ln L_i(64) - ln L_i(input node) + (selection part)/N summed in bins of true redshift by host class; specs R:Hin:f."""
    out = {}
    for sp in specs:
        R, hin, f = sp.split(":")
        d = load_nodes(BASE.format(R=int(R)))
        T = truth(d["meta"]["events_file"])
        a, b = pt(d, 64.0, float(f)), pt(d, float(hin), float(f))
        Ds = d["ev"][a] - d["ev"][b] + (d["stored_sel_corr"][a] - d["stored_sel_corr"][b]) / T["z"].size
        rows = []
        for lo, hi in ((0, 0.06), (0.06, 0.1), (0.1, 0.15), (0.15, 0.25), (0.25, 1.2)):
            for c in (0, 1):
                m = (T["z"] >= lo) & (T["z"] < hi) & (T["host_type"] == c)
                rows.append({"z": [lo, hi], "host": "gal" if c == 0 else "agn", "n": int(m.sum()), "sum": float(Ds[m].sum())})
        out[R] = rows
        print(f"[t1bins] r{R}: " + "; ".join(f"z{r['z'][0]}-{r['z'][1]} {r['host']} n{r['n']} {r['sum']:+.2f}" for r in rows))
    (RES / "hd_scatter_t1bins.json").write_text(json.dumps(out, indent=1))


def t2pair(R=109, f=0.34):
    """Per-merger change of D = ln L_i(64) - ln L_i(67.75) between the stored samples and redraw 1 (67.75 interpolated
    linearly between the redraw's 67 and 68 nodes): how the Monte Carlo change of the summed D spreads over mergers."""
    n = BASE.format(R=R)
    d0, d1 = load_nodes(n), load_nodes(n, "_rd1")
    D0 = d0["ev"][pt(d0, 64, f)] - d0["ev"][pt(d0, 67.75, f)]
    D1 = d1["ev"][pt(d1, 64, f)] - (0.25 * d1["ev"][pt(d1, 67, f)] + 0.75 * d1["ev"][pt(d1, 68, f)])
    dD = D1 - D0
    e = np.minimum(d0["ess"][pt(d0, 64, f)], d1["ess"][pt(d1, 64, f)])
    res = {"sum_dD": float(dD.sum()), "sum_dD2": float((dD ** 2).sum()), "mc_sd_of_sum_D_one_draw": float(np.sqrt((dD ** 2).sum() / 2)),
           "by_ess": {str(c): {"n": int((e < c).sum()), "sum_dD2": float((dD[e < c] ** 2).sum())} for c in (20, 50, 100, 300)}}
    print(f"[t2pair] r{R}: sum dD {res['sum_dD']:+.3f}; sum dD^2 {res['sum_dD2']:.3f} -> MC sd of sum D per draw {res['mc_sd_of_sum_D_one_draw']:.3f}; "
          + "; ".join(f"ESS<{k}: n {v['n']} sum dD^2 {v['sum_dD2']:.3f}" for k, v in res["by_ess"].items()))
    (RES / "hd_scatter_t2pair.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    mode = sys.argv[1]
    args = sys.argv[2:]
    if mode == "t1bins":
        t1bins(args)
    elif mode == "t2pair":
        t2pair()
    elif mode == "t1":
        t1([int(x) for x in args])
    elif mode == "t2census":
        t2census([int(x) for x in args])
    elif mode == "t2hybrid":
        tag = args[args.index("--tag") + 1] if "--tag" in args else ""
        t2hybrid(int(args[0]), tag)
    elif mode == "t2redraw":
        t2redraw(int(args[0]), args[args.index("--tags") + 1].split(","))
    elif mode == "t3":
        t3([int(x) for x in args])
    elif mode == "t4":
        t4([BASE.format(R=int(x)) if x.isdigit() else x for x in args])
