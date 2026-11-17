#!/usr/bin/env python
"""Analysis 16: turn finished runs into small JSON summaries (no figures).

    <core venv python> scripts/prod_summary.py RUN_NAME [RUN_NAME ...]   -> runs/<run>/summary.json
    <core venv python> scripts/prod_summary.py --stage N | --all         (every run whose parts are finished)

Per run: the configuration that matters (analysis, catalogs, flux limit and densities, events,
realization, free parameters, truth), posterior medians and 90% intervals (with the half-variance
correction, and without it by reweighting), ln evidence and its error, the variance diagnostics,
likelihood calls and GPU hours (the sum of the wall times of every process that worked on the run).
Nested runs with two independent parts are merged with dynesty.utils.merge_runs (posterior and
evidence); each part's own ln Z is listed too, and their difference in units of the combined error
is a convergence check. Grid runs report the fine-grid posterior, the grid ln Z and the coarse-pass
check that the mass outside the fine grid is below 1e-4. The grid posterior is recomputed here from the
stored ln L on a lattice 16 times denser than the nodes (refined_grid_summary). The driver's own summary
(grid/result.json, kept here as "posterior_nodes") takes the density as linear between the stored nodes
and reads the quantiles off a CDF interpolated linearly between them; that moves each 90% bound outward,
by about 0.04 sd at an H0 step of a third of the sd, 0.1 sd at half the sd, and up to 0.7 sd for f_AGN
where its step exceeds the sd.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prod_common as P  # noqa: E402


def truth(cfg):
    t = {"H0": 67.74}
    if "f_agn" in cfg["free"]:
        t["f_agn"] = cfg["planted_f_agn"]
    pv = cfg["population_values"]
    for k in P.POP_FREE:
        t[k] = pv[k]
    t.update(cfg.get("agn_offsets_truth", {}))
    for tr, p in (("gal", "log10n_gal"), ("agn", "log10n_agn")):
        if p in cfg["free"]:
            t[p] = (-3.0 if tr == "gal" else -5.0) + 3 * np.log10(100 / 67.74)  # mock input in h^3 Mpc^-3
    return {k: float(t[k]) for k in cfg["free"] if k in t}


def compact(s, names):
    return {p: {"median": s[p]["median"], "ci90": s[p]["ci90"]} for p in names if p in s}


def sessions_of(part_dir):
    p = Path(part_dir) / "sessions.json"
    return json.loads(p.read_text()) if p.exists() else []


def selection_variance_grid(run_dir, cfg, r):
    """N^2/N_eff of the selection term (its contribution to the variance of ln L) over the final fine
    grid: at the posterior peak, its posterior mean, and its largest value where the posterior is above
    1e-3 of its peak; hosts_known: the sum over the two subsets, sum_c N_c^2/N_eff,c, and each subset's."""
    import h5py
    g = run_dir / "grid"
    fp = g / ("fine_widened.h5" if (r.get("outside_mass_check") or {}).get("widened") else "fine.h5")
    if not fp.exists():
        return None
    n = (r.get("model_info") or {}).get("n_events")
    with h5py.File(fp, "r") as f:
        d = {k: np.asarray(f[k]) for k in f.keys() if not k.startswith("axis_")}
        n = n or int(f.attrs.get("n_events"))
    s2 = d["sigma2_sel"] if "sigma2_sel" in d else n ** 2 / d["n_eff"]
    lnc = d["lnL"] + 0.5 * d["pe_var"] - 0.5 * n / d["n_eff"] if cfg.get("half_variance_correction") else d["lnL"]
    ok = np.isfinite(lnc)
    w = np.where(ok, np.exp(lnc - np.max(lnc[ok])), 0.0)
    pk = np.unravel_index(int(np.argmax(w)), w.shape)
    out = {"at_peak": float(s2[pk]), "posterior_mean": float(np.sum(w * s2) / np.sum(w)),
           "max_where_posterior_gt_1e-3": float(np.max(s2[w > 1e-3])),
           "pe_var_at_peak": float(d["pe_var"][pk]), "n_events": int(n), "max_var": cfg.get("max_var")}
    for c in ("agn", "gal"):
        if f"n_eff_{c}" in d and np.isfinite(d[f"n_eff_{c}"][pk]):
            nc = (r.get("model_info") or {}).get("n_events_by_class", {}).get(c)
            if nc:
                out[f"{c}_subset_at_peak"] = float(nc ** 2 / d[f"n_eff_{c}"][pk])
    return out


REFINE = 16   # lattice points per stored grid step in refined_grid_summary (16 -> 32 moves no bound by 0.005 sd)


def _to_lattice(x, y, xn, axis):
    """Akima interpolation of y (nodes x along `axis`) to xn (linear below five nodes)."""
    from scipy.interpolate import Akima1DInterpolator, interp1d
    if len(x) >= 5:
        return Akima1DInterpolator(x, y, axis=axis)(xn)
    return interp1d(x, y, axis=axis, kind="linear", assume_sorted=True)(xn)


def refined_grid_summary(run_dir, cfg, r):
    """Posterior summaries of a grid run's final fine pass (fine_widened.h5 when it was widened) from its
    stored ln L, with and without the half-variance correction: ln L interpolated along each axis onto a
    lattice REFINE times denser than the nodes, then prod_common.summarize_1d / summarize_2d on that
    lattice (trapezoid marginals, CDF, quantiles, correlation). The interpolant is Akima's (piecewise cubic,
    local): on smooth surfaces it agrees with a not-a-knot cubic spline to 0.01 sd, and it does not ring
    where ln L jumps by tens of nats between two nodes (a cubic spline overshoots there by up to 10 nats;
    seen in catalog-free 1-D grids). Non-finite cells are set 300 nats below the lowest finite one.
    -> (summary, summary_uncorrected, info), or None when the pass is not on disk."""
    import h5py
    g = run_dir / "grid"
    fp = g / ("fine_widened.h5" if (r.get("outside_mass_check") or {}).get("widened") else "fine.h5")
    if not fp.exists():
        return None
    with h5py.File(fp, "r") as f:
        names = json.loads(f.attrs["axes"])
        axes = [np.asarray(f[f"axis_{n}"][()], float) for n in names]
        n = int(f.attrs["n_events"])
        raw = np.asarray(f["lnL"][()], float)
        lnc = P.half_variance_corrected(raw, f["pe_var"][()], f["n_eff"][()], n) \
            if cfg.get("half_variance_correction") else raw
    lat = [np.linspace(a[0], a[-1], (len(a) - 1) * REFINE + 1) if len(a) > 1 else a for a in axes]
    out, info = [], {}
    for tag, l in (("corrected", lnc), ("uncorrected", raw)):
        ok = np.isfinite(l)
        if not ok.any():
            return None
        lf = np.where(ok, l, np.min(l[ok]) - 300.0)
        for i, (a, an) in enumerate(zip(axes, lat)):
            if len(a) > 1:
                lf = _to_lattice(a, lf, an, i)
        out.append({names[0]: P.summarize_1d(lat[0], lf)} if len(names) == 1 else
                   P.summarize_2d(lat[0], lat[1], lf, names=tuple(names)))
        if tag == "corrected":
            info = {"file": fp.name, "refine": REFINE, "interpolation": "akima, ln L",
                    "n_nonfinite_cells": int((~ok).sum()),
                    "lattice_max_minus_node_max_nats": float(np.max(lf) - np.max(l[ok]))}
    return out[0], out[1], info


def summarize_grid(run_dir, cfg):
    r = json.loads((run_dir / "grid" / "result.json").read_text())
    ses = sessions_of(run_dir / "grid")
    extra = {}
    ref = refined_grid_summary(run_dir, cfg, r)
    summ, summ_raw, how = ref if ref is not None else (r["summary"], r["summary_uncorrected"],
                                                         {"interpolation": "none: the driver's node summary"})
    try:
        extra["selection_N2_over_Neff"] = selection_variance_grid(run_dir, cfg, r)
    except Exception as e:  # noqa: BLE001
        extra["selection_N2_over_Neff"] = {"error": repr(e)}
    if r.get("extra_at_peak"):
        extra["at_peak"] = r["extra_at_peak"]
    if r.get("membership"):
        m = r["membership"]
        extra["membership"] = {k: m.get(k) for k in ("ok", "H0_median", "f_agn_median", "mean_r",
                                                      "f_median_mu_agn_over_mu", "true_agn_fraction",
                                                      "mean_r_true_agn", "mean_r_true_gal", "identity_ml",
                                                      "identity_event_evidence_max_abs", "file", "error")}
    if cfg.get("analysis") == "hosts_known":
        extra["n_events_by_class"] = (r.get("model_info") or {}).get("n_events_by_class")
    return {**extra, "posterior": compact(summ, cfg["free"]),
            "posterior_uncorrected": compact(summ_raw, cfg["free"]),
            "rho_H0_f": summ.get("rho"),
            "posterior_nodes": compact(r["summary"], cfg["free"]),
            "ln_evidence": {"value": r.get("ln_evidence_grid"), "error": None, "how": "grid (fine + coarse outside)"},
            "grid": {"fine_axes": r["fine_axes"], "coarse_axes": r["coarse_axes"],
                     "outside_mass_fraction": (r.get("outside_mass_check") or {}).get("mass_outside_fraction"),
                     "outside_mass_ok": r.get("outside_mass_ok"),
                     "mass_at_fine_edges": {p: r["summary"][p]["mass_at_edges"] for p in cfg["free"]},
                     "posterior_lattice": how},
            "variance": r["variance"], "calls": int(sum(s.get("calls", 0) for s in ses)),
            "gpu_hours": sum(s.get("wall_seconds", 0.0) for s in ses) / 3600,
            "seconds_per_call": r.get("seconds_per_call_median")}


def summarize_nested(run_dir, cfg, parts):
    from dynesty.utils import merge_runs
    res, recs, w_raw_all, s_all = [], [], [], []
    calls = 0
    gpu_h = 0.0
    var_rows = []
    for p in parts:
        d = run_dir / p
        rec = json.loads((d / "result.json").read_text())
        recs.append(rec)
        with open(d / "results.pkl", "rb") as fh:
            res.append(pickle.load(fh))
        z = np.load(d / "samples.npz")
        s_all.append(z["samples"])
        w_raw_all.append(z["weights_uncorrected"])
        w = z["weights"]
        blob = z["blob"]
        n_ev = rec["model_info"]["n_events"]
        tot = blob[:, 1] + n_ev ** 2 / blob[:, 2]
        o = np.argsort(tot)
        c = np.cumsum(w[o])
        var_rows.append({"pe_var_mean": float(np.sum(w * blob[:, 1])),
                         "sigma2_sel_mean": float(np.sum(w * n_ev ** 2 / blob[:, 2])),
                         "sigma2_total_mean": float(np.sum(w * tot)),
                         "sigma2_total_q99": float(tot[o][np.searchsorted(c, 0.99)]),
                         "posterior_mass_sigma2_total_gt_1": float(np.sum(w[tot > 1])),
                         "n_neginf_last_session": rec.get("n_neginf_this_session")})
        ses = sessions_of(d)
        calls += int(sum(s.get("calls", 0) for s in ses))
        gpu_h += sum(s.get("wall_seconds", 0.0) for s in ses) / 3600
    names = cfg["free"]
    if len(res) > 1:
        m = merge_runs(res)
        logwt = np.asarray(m["logwt"])
        w = np.exp(logwt - logwt.max())
        w /= w.sum()
        post = P.summarize_samples(np.asarray(m["samples"]), names, weights=w)
        lz, lze = float(m["logz"][-1]), float(m["logzerr"][-1])
        # without the correction: each part reweighted, the two normalised posteriors averaged
        S = np.concatenate(s_all)
        W = np.concatenate([x / x.sum() / len(w_raw_all) for x in w_raw_all])
        post_raw = P.summarize_samples(S, names, weights=W)
        dz = recs[0]["logZ"] - recs[1]["logZ"]
        conv = {"dlogZ_parts": dz, "dlogZ_over_err": dz / np.hypot(recs[0]["logZerr"], recs[1]["logZerr"]),
                "d_median_over_sd": {p: (recs[0]["summary"][p]["median"] - recs[1]["summary"][p]["median"])
                                     / max(post[p]["sd"], 1e-300) for p in names}}
    else:
        rec = recs[0]
        post, post_raw = rec["summary"], rec["summary_uncorrected"]
        lz, lze = rec["logZ"], rec["logZerr"]
        conv = None
    sel = {"posterior_mean_per_part": [v["sigma2_sel_mean"] for v in var_rows],
           "note": "N^2/N_eff of the selection term, posterior mean over each part's dead points"}
    return {"selection_N2_over_Neff": sel,
            "posterior": compact(post, names), "posterior_uncorrected": compact(post_raw, names),
            "ln_evidence": {"value": lz, "error": lze, "how": "dynesty" + (" (merged)" if len(res) > 1 else "")},
            "parts": [{"part": p, "logZ": r["logZ"], "logZerr": r["logZerr"], "ncall": r["ncall_total"],
                       "niter": r["niter"], "sampler": r["sampler"]["method"]} for p, r in zip(parts, recs)],
            "convergence": conv, "variance": var_rows, "calls": calls, "gpu_hours": gpu_h}


def summarize(name, root=None):
    run_dir = Path(root or P.RUNS) / name
    cfg = json.loads((run_dir / "config.json").read_text())
    if cfg["mode"] == "grid":
        if not (run_dir / "grid" / "result.json").exists():
            return None
        body = summarize_grid(run_dir, cfg)
    else:
        parts = sorted(p.name for p in run_dir.glob("nested_[0-9]") if (p / "result.json").exists()
                       and json.loads((p / "result.json").read_text()).get("finished"))
        if not parts:
            return None
        body = summarize_nested(run_dir, cfg, parts)
    out = {"run_name": name, "analysis": cfg["analysis"], "catalogs": cfg["catalogs"], "flux": cfg["flux"],
           "set_label": cfg.get("set_label"), "population_family": cfg.get("population_family"),
           "planted": {k: (cfg.get("events_truth") or {}).get(k) for k in
                       ("planted_f_agn", "truth_f_agn", "dmu_G_agn", "dmu_chi_agn")},
           "injections_file": cfg.get("injections_file"),
           "density_mode": cfg.get("density_mode"),
           "densities": {t: {k: d.get(k) for k in ("mode", "log10n_used", "z_complete", "n_below_z_complete",
                                                   "factor", "units")} for t, d in cfg.get("densities", {}).items()},
           "events_kind": cfg["events_kind"], "events_file": cfg["events_file"], "realization": cfg["realization"],
           "free": cfg["free"], "truth": truth(cfg), "nsamp": cfg["nsamp"], "max_var": cfg["max_var"],
           "half_variance_correction": cfg["half_variance_correction"], "zmax": cfg["zmax"], "core": cfg["core"],
           **body, "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (run_dir / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--stage", help="1-4 or E1, E2, E3")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--root", default=str(P.RUNS), help="runs directory (tests use their own)")
    a = ap.parse_args()
    names = list(a.names)
    if a.stage or a.all:
        import yaml
        for f in ("RUNLIST.yaml", "RUNLIST_experiment.yaml"):
            if not (P.RUNS / f).exists():
                continue
            doc = yaml.safe_load((P.RUNS / f).read_text())
            names += [r["name"] for st in doc["stages"] for r in st["runs"]
                      if a.all or str(st["stage"]).upper() == str(a.stage).upper()]
    for n in names:
        if not (Path(a.root) / n / "config.json").exists():
            continue
        s = summarize(n, a.root)
        if s is None:
            continue
        post = "  ".join(f"{p} {v['median']:.4g} [{v['ci90'][0]:.4g}, {v['ci90'][1]:.4g}]" for p, v in s["posterior"].items())
        lz = s["ln_evidence"]
        sel = s.get("selection_N2_over_Neff") or {}
        sv = sel.get("at_peak", sel.get("posterior_mean_per_part"))
        print(f"{n}: {post}  lnZ {lz['value']}  N2/Neff {sv}  calls {s['calls']}  GPU-h {s['gpu_hours']:.2f}")


if __name__ == "__main__":
    main()
