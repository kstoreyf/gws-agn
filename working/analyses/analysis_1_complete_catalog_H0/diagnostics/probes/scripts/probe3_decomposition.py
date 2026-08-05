#!/usr/bin/env python3
"""PROBE 3 -- numerator / selection decomposition of the matched-host scans (GPU).

The hierarchical log-likelihood at a fixed mixture weight splits EXACTLY into

    logL(H0) = SUM_i ln Z_i(H0)                          <- per-event numerator
             + [ -N_obs ln mu(H0) + N_obs(N_obs+3)/(2 N_eff) ]   <- selection term

``darksirens.likelihood.selection.selection_log_correction`` returns the bracket,
so an import-level pass-through spy on it (installed BEFORE ``make_likelihood``,
exactly as ``scan_h0f.py``'s guard record does -- darksirens itself is NOT
modified) recovers both pieces from the same evaluations.

This runs the decomposition for the MATCHED-HOST configuration of record --
``dark_sirens_complete``, K = 1, field weighting, targeted injections, W = 4096
for GAL, the campaign guard convention -- on both tracers and every available
realisation, and reports per seed:

  * the peak of the full logL          (must reproduce the stored ctrl_* scan)
  * the peak of the NUMERATOR ALONE    (the catalog + PE + mass channel)
  * the peak of the SELECTION TERM
  * d ln mu / dH0 at truth and the implied peak shift the selection slope buys
  * the numerator's H0 curvature at truth, per event

The GAL-vs-AGN contrast in the numerator's H0 curvature is the exhibit: if the
GAL numerator is flat/damped while the AGN numerator is sharply peaked at truth,
the dense catalog is carrying no localising redshift information and whatever
residual slope it has is not anchored.

Per-run output: results/probe3_decomp_{tracer}_s{seed}.json
Aggregate (--aggregate): results/probe3_decomposition.json + figs/probe3_*.png
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data")
BULK = Path("/hildafs/projects/phy220048p/magana/gws-agn-data/derived/"
            "analysis_1_complete_catalog_H0")

OM0_FID = 0.3075
H0_TRUE = 67.74
NUISANCE_DEFAULTS = {"delta": 0.0, "b_miss": 1.0, "sigma_kde": 0.0,
                     "delta_c2": 0.0, "b_miss_c2": 1.0, "sigma_kde_c2": 0.0}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--tracer", choices=["gal", "agn"], default="gal")
    ap.add_argument("--h0_grid", nargs=3, type=float, default=[50.0, 100.0, 201])
    ap.add_argument("--kde_window", type=int, default=None,
                    help="4096 for GAL (the analysis of record); unset for AGN "
                         "(its 178-column block is never windowed).")
    ap.add_argument("--kde_window_nsigma", type=float, default=8.0)
    ap.add_argument("--sel_batch_size", type=int, default=50000)
    ap.add_argument("--pe_event_block", type=int, default=25)
    ap.add_argument("--max_likelihood_variance", type=float, default=1e6)
    ap.add_argument("--survey_override", default=None,
                    help="Analyse a different survey file against the same events "
                         "and injections -- used to run this decomposition on "
                         "probe 4's synthetic continuum surveys.")
    ap.add_argument("--tag_suffix", default="",
                    help="Appended to the output tag when --survey_override is used.")
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    ap.add_argument("--aggregate", action="store_true",
                    help="Do not run; collect existing per-run JSONs into "
                         "results/probe3_decomposition.json and make the figures.")
    ap.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102, 103, 105])
    return ap.parse_args(argv)


def event_path(seed, tracer):
    if seed == 100:
        return ROOT / "data_derived" / f"events_{tracer}_hosted.h5"
    return BULK / f"seed{seed}" / f"events_{tracer}_hosted.h5"


def quad_refine(x, y):
    """Sub-grid peak by a parabola through the argmax and its two neighbours."""
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    if not ok.any():
        return float("nan"), True
    i = int(np.nanargmax(np.where(ok, y, np.nan)))
    if i in (0, len(y) - 1) or not (ok[i - 1] and ok[i + 1]):
        return float(x[i]), True                     # railed / edge
    d = y[i - 1] - 2 * y[i] + y[i + 1]
    if d == 0 or not np.isfinite(d):
        return float(x[i]), False
    return float(x[i] - 0.5 * (y[i + 1] - y[i - 1]) / d * (x[1] - x[0])), False


def local_deriv(x, y, x0):
    """(first, second) derivative of y at x0 from a local quadratic fit over the
    +/- 5-cell neighbourhood -- robust to the grid's 0.25 spacing."""
    y = np.asarray(y, float)
    i = int(np.argmin(np.abs(x - x0)))
    lo, hi = max(0, i - 5), min(len(x), i + 6)
    m = np.isfinite(y[lo:hi])
    xs, ys = x[lo:hi][m], y[lo:hi][m]
    if xs.size < 3:
        return float("nan"), float("nan")
    c = np.polyfit(xs - x0, ys, 2)
    return float(c[1]), float(2.0 * c[0])


def run_one(args):
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["JAX_PLATFORMS"] = "cuda,cpu"      # debug callbacks need a CPU device

    import jax
    import jax.numpy as jnp
    import darksirens
    from darksirens.inference.data import load_all_data, validate_loaded_survey_shapes
    from darksirens.likelihood import core as ds_core
    from darksirens.likelihood.factory import make_likelihood
    from darksirens.gw.populations import get_fixed_population_params
    from darksirens.inference.prior import build_parameter_space
    from darksirens.likelihood.selection import (
        selection_log_correction as _true_slc, DEFAULT_MAX_LIKELIHOOD_VARIANCE)
    print(f"darksirens: {darksirens.__file__}   devices: {jax.devices()}")

    rec = []

    def _spy(log_mu, Neff, nEvents, soft_guard=False,
             max_likelihood_variance=DEFAULT_MAX_LIKELIHOOD_VARIANCE,
             pe_variance_sum=0.0):
        out = _true_slc(log_mu, Neff, nEvents, soft_guard=soft_guard,
                        max_likelihood_variance=max_likelihood_variance,
                        pe_variance_sum=pe_variance_sum)
        jax.debug.callback(
            lambda lm, ne, pv, sel: rec.append(
                {"log_mu": float(lm), "Neff": float(ne),
                 "pe_variance_sum": float(pv), "sel_term": float(sel)}),
            log_mu, Neff, pe_variance_sum, out)
        return out

    ds_core.selection_log_correction = _spy

    if args.kde_window is not None:
        from darksirens.redshift.catalog import configure_catalog_kde_window
        configure_catalog_kde_window(size=int(args.kde_window),
                                     n_sigma=float(args.kde_window_nsigma))
        print(f"[kde] window W={args.kde_window} n_sigma={args.kde_window_nsigma}")

    sd = DATA / f"seed{args.seed}"
    survey = (args.survey_override if args.survey_override
              else str(sd / "surveys" / f"survey_{args.tracer}_complete_ns32.h5"))
    gw = str(event_path(args.seed, args.tracer))
    inj = str(sd / "injections" / "injections_targeted.h5")

    opts = SimpleNamespace(
        universe_model="dark_sirens_complete",
        survey_path=survey, survey_paths=[survey], n_catalogs=1,
        gw_path=gw, gwselection_path=inj,
        use_LSS=False, lss_completion=None, lss_completions=[], lss_marginalize=False,
        counterpart=None, counterpart_nside=1, counterpart_dz=1e-4,
        bright_siren_sky_marginalized=False, drop_full_catalog=False,
        sky_model="isotropic", mark_model="none", marks=None, mark_names=(),
        sel_batch_size=args.sel_batch_size, pe_event_block=args.pe_event_block,
        redshift_prior_barrier="auto",
        selection_neff_guard="hard", selection_neff_soft_guard=False,
        sampler="tinyns", fix_population=True, fix_cosmology=False, fix_de=True,
        fix_survey=False, pop_model="powerlaw+peak", shared_beta=True,
        shared_spin=True, shared_gamma=True, complete_empty_pixel_policy="zero",
        catalog_sky_weighting="field",
        max_likelihood_variance=args.max_likelihood_variance,
    )
    fixed = {"Om0": OM0_FID}

    t0 = time.time()
    data = load_all_data(opts)
    validate_loaded_survey_shapes(data)
    nobs = int(data["nEvents"])
    print(f"load_all_data {time.time()-t0:.1f}s  nEvents={nobs} "
          f"nsamp={data['nsamp']} Ndraw={data['Ndraw']}")

    res = build_parameter_space(
        opts.pop_model, opts.fix_population, opts.fix_cosmology, opts.fix_survey,
        fix_de=opts.fix_de, prior_overrides={}, fixed_parameter_values=fixed,
        universe_model=opts.universe_model, shared_beta=opts.shared_beta,
        shared_spin=opts.shared_spin, shared_gamma=opts.shared_gamma,
        sky_model=opts.sky_model, mark_model=opts.mark_model,
        mark_names=opts.mark_names, n_catalogs=1,
        lss_completion_active=[False], use_lss=False, mark_names_by_catalog=None)
    labels = list(res[0])
    print(f"labels: {labels}")
    pop_fid = get_fixed_population_params(opts.pop_model, shared_beta=True,
                                          shared_spin=True, shared_gamma=True)
    like = make_likelihood(opts=opts, data=data, pop_params_fid=pop_fid,
                           fixed_parameter_values=fixed)

    point = dict(NUISANCE_DEFAULTS)
    point["H0"] = H0_TRUE
    missing = [l for l in labels if l not in point]
    if missing:
        raise SystemExit(f"[fatal] missing label values: {missing}")
    base = np.asarray([float(point[l]) for l in labels], float)
    ih = labels.index("H0")

    H0 = np.linspace(args.h0_grid[0], args.h0_grid[1], int(round(args.h0_grid[2])))
    total, sel, log_mu, neff, pevar = [], [], [], [], []
    t0 = time.time()
    for k, h in enumerate(H0):
        c = base.copy(); c[ih] = h
        rec.clear()
        ll = float(like(c))
        if not rec:
            raise SystemExit("[fatal] selection spy captured nothing")
        r = rec[-1]
        total.append(ll); sel.append(r["sel_term"]); log_mu.append(r["log_mu"])
        neff.append(r["Neff"]); pevar.append(r["pe_variance_sum"])
        if k % 25 == 0:
            print(f"  [{k:3d}/{len(H0)}] H0={h:6.2f}  logL={ll:.4f}  "
                  f"sel={r['sel_term']:.4f}  ({time.time()-t0:.0f}s)")
    total = np.asarray(total); sel = np.asarray(sel)
    log_mu = np.asarray(log_mu); neff = np.asarray(neff); pevar = np.asarray(pevar)
    numer = total - sel

    pk_tot, railed_tot = quad_refine(H0, total)
    pk_num, railed_num = quad_refine(H0, numer)
    pk_sel, railed_sel = quad_refine(H0, sel)
    d1_num, d2_num = local_deriv(H0, numer, H0_TRUE)
    d1_sel, d2_sel = local_deriv(H0, sel, H0_TRUE)
    d1_tot, d2_tot = local_deriv(H0, total, H0_TRUE)
    d1_lnmu, d2_lnmu = local_deriv(H0, log_mu, H0_TRUE)

    out = {
        "probe": 3, "name": "decomposition", "seed": args.seed,
        "tracer": args.tracer, "nobs": nobs, "H0_true": H0_TRUE,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {"survey": survey, "gw": gw, "injections": inj,
                   "kde_window": args.kde_window,
                   "sel_batch_size": args.sel_batch_size,
                   "pe_event_block": args.pe_event_block,
                   "max_likelihood_variance": args.max_likelihood_variance,
                   "h0_grid": args.h0_grid},
        "H0_grid": H0.tolist(),
        "logL_total": total.tolist(),
        "selection_term": sel.tolist(),
        "per_event_numerator": numer.tolist(),
        "log_mu": log_mu.tolist(),
        "Neff": neff.tolist(),
        "pe_variance_sum": pevar.tolist(),
        "n_nonfinite_cells": int((~np.isfinite(total)).sum()),
        "min_Neff": float(np.min(neff)),
        "peak_total": pk_tot, "peak_total_railed": bool(railed_tot),
        "peak_numerator": pk_num, "peak_numerator_railed": bool(railed_num),
        "peak_selection": pk_sel, "peak_selection_railed": bool(railed_sel),
        "offset_total": pk_tot - H0_TRUE,
        "offset_numerator": pk_num - H0_TRUE,
        "shift_from_selection_term": pk_tot - pk_num,
        "dlnmu_dH0_at_truth": d1_lnmu,
        "d2lnmu_dH02_at_truth": d2_lnmu,
        "lnmu_range": [float(log_mu.min()), float(log_mu.max())],
        "at_truth": {
            "dnumerator_dH0": d1_num, "d2numerator_dH02": d2_num,
            "dselection_dH0": d1_sel, "d2selection_dH02": d2_sel,
            "dtotal_dH0": d1_tot, "d2total_dH02": d2_tot,
            "d2numerator_dH02_per_event": d2_num / nobs,
            "d2total_dH02_per_event": d2_tot / nobs,
            # Newton step from truth: where the total would peak if the local
            # quadratic held -- the "implied peak shift".
            "implied_shift_total": (-d1_tot / d2_tot) if d2_tot else float("nan"),
            "implied_shift_numerator_only": (-d1_num / d2_num) if d2_num else float("nan"),
            "selection_slope_share": (d1_sel / d1_tot) if d1_tot else float("nan"),
        },
        "seconds": time.time() - t0,
    }
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"probe3_decomp_{args.tracer}_s{args.seed}{args.tag_suffix}.json"
    p.write_text(json.dumps(out, indent=2))

    print(f"\n=== probe3 {args.tracer.upper()} seed {args.seed} (N_obs={nobs}) ===")
    print(f"  peak FULL logL      : {pk_tot:8.3f}  (offset {pk_tot-H0_TRUE:+.3f})"
          f"{'  RAILED' if railed_tot else ''}")
    print(f"  peak NUMERATOR only : {pk_num:8.3f}  (offset {pk_num-H0_TRUE:+.3f})"
          f"{'  RAILED' if railed_num else ''}")
    print(f"  shift from selection: {pk_tot-pk_num:+.3f} km/s/Mpc")
    print(f"  d ln mu/dH0 @ truth : {d1_lnmu:+.6f}   (enters as -N_obs * this)")
    print(f"  numerator curvature @ truth: {d2_num:+.5f} "
          f"({d2_num/nobs:+.6f} per event)")
    print(f"Wrote {p}")
    return out


def aggregate(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(args.outdir)
    agg = {"probe": 3, "name": "decomposition", "H0_true": H0_TRUE,
           "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "runs": {}, "table": []}
    curves = {}
    for tr in ("gal", "agn"):
        for s in args.seeds:
            p = outdir / f"probe3_decomp_{tr}_s{s}.json"
            if not p.exists():
                print(f"[miss] {p}")
                continue
            d = json.loads(p.read_text())
            curves[(tr, s)] = d
            agg["runs"][f"{tr}_s{s}"] = {
                k: d[k] for k in
                ("seed", "tracer", "nobs", "peak_total", "peak_total_railed",
                 "peak_numerator", "peak_numerator_railed", "peak_selection",
                 "offset_total", "offset_numerator", "shift_from_selection_term",
                 "dlnmu_dH0_at_truth", "lnmu_range", "at_truth",
                 "n_nonfinite_cells", "min_Neff")}
            # Score identity: for a correctly normalised hierarchical likelihood
            # the expected per-event numerator slope at the truth equals
            # d ln mu/dH0 (the selection term enters as -N_obs ln mu).  Their
            # difference, times N_obs, IS the total slope at truth; dividing by
            # -d2 total gives the peak shift it buys.
            per_ev_num_slope = d["at_truth"]["dnumerator_dH0"] / d["nobs"]
            agg["table"].append({
                "tracer": tr, "seed": s, "nobs": d["nobs"],
                "peak_total": d["peak_total"], "offset_total": d["offset_total"],
                "peak_numerator": d["peak_numerator"],
                "offset_numerator": d["offset_numerator"],
                "shift_from_selection": d["shift_from_selection_term"],
                "dlnmu_dH0": d["dlnmu_dH0_at_truth"],
                "per_event_numerator_slope": per_ev_num_slope,
                "score_residual_per_event": per_ev_num_slope - d["dlnmu_dH0_at_truth"],
                "score_residual_relative": (
                    per_ev_num_slope / d["dlnmu_dH0_at_truth"] - 1.0
                    if d["dlnmu_dH0_at_truth"] else float("nan")),
                "d2num_per_event": d["at_truth"]["d2numerator_dH02_per_event"],
                "implied_shift_total": d["at_truth"]["implied_shift_total"],
                "railed_numerator": d["peak_numerator_railed"],
            })

    for tr in ("gal", "agn"):
        rows = [r for r in agg["table"] if r["tracer"] == tr]
        if not rows:
            continue
        on = np.array([r["offset_numerator"] for r in rows])
        ot = np.array([r["offset_total"] for r in rows])
        sh = np.array([r["shift_from_selection"] for r in rows])
        cu = np.array([r["d2num_per_event"] for r in rows])
        agg[f"summary_{tr}"] = {
            "n_seeds": len(rows),
            "mean_offset_total": float(ot.mean()),
            "sem_offset_total": float(ot.std(ddof=1) / np.sqrt(len(rows))) if len(rows) > 1 else None,
            "mean_offset_numerator": float(on.mean()),
            "sem_offset_numerator": float(on.std(ddof=1) / np.sqrt(len(rows))) if len(rows) > 1 else None,
            "mean_shift_from_selection": float(sh.mean()),
            "mean_d2numerator_per_event": float(cu.mean()),
            "n_numerator_railed": int(sum(r["railed_numerator"] for r in rows)),
            "mean_dlnmu_dH0": float(np.mean([r["dlnmu_dH0"] for r in rows])),
            "mean_per_event_numerator_slope": float(
                np.mean([r["per_event_numerator_slope"] for r in rows])),
            "mean_score_residual_per_event": float(
                np.mean([r["score_residual_per_event"] for r in rows])),
            "mean_score_residual_relative": float(
                np.mean([r["score_residual_relative"] for r in rows])),
            "mean_implied_shift_total": float(
                np.mean([r["implied_shift_total"] for r in rows])),
        }
    if "summary_gal" in agg and "summary_agn" in agg:
        agg["gal_vs_agn_numerator_curvature_ratio"] = (
            agg["summary_gal"]["mean_d2numerator_per_event"]
            / agg["summary_agn"]["mean_d2numerator_per_event"])

    p = outdir / "probe3_decomposition.json"
    p.write_text(json.dumps(agg, indent=2))
    print(json.dumps({k: v for k, v in agg.items()
                      if k.startswith("summary") or k == "table"
                      or k.startswith("gal_vs")}, indent=2))
    print(f"Wrote {p}")

    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True)
    for row, tr in enumerate(("gal", "agn")):
        for col, (key, lab) in enumerate((
                ("per_event_numerator", r"numerator $\sum_i \ln Z_i$"),
                ("selection_term", r"selection $-N\ln\mu + \ldots$"),
                ("logL_total", r"total $\log\mathcal{L}$"))):
            ax = axes[row, col]
            for s in args.seeds:
                d = curves.get((tr, s))
                if d is None:
                    continue
                H0 = np.asarray(d["H0_grid"]); y = np.asarray(d[key])
                ax.plot(H0, y - np.nanmax(y), lw=1.2, label=f"seed {s}")
            ax.axvline(H0_TRUE, color="k", ls=":", lw=1)
            ax.set_ylim(-40, 2)
            if row == 1:
                ax.set_xlabel(r"$H_0$")
            if col == 0:
                ax.set_ylabel(f"{tr.upper()}\n" + r"curve $-$ max")
            ax.set_title(lab if row == 0 else "", fontsize=10)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Probe 3 — matched-host decomposition: numerator vs selection, "
                 "GAL (top) vs AGN (bottom)", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figs" / f"probe3_decomposition.{ext}", dpi=150)
    print("Wrote figs/probe3_decomposition.{png,pdf}")

    # peak-shift summary panel
    fig2, ax = plt.subplots(figsize=(7.0, 4.2))
    for tr, c, m in (("gal", "tab:blue", "o"), ("agn", "tab:orange", "s")):
        rows = [r for r in agg["table"] if r["tracer"] == tr]
        if not rows:
            continue
        x = np.arange(len(rows))
        ax.errorbar(x - 0.08 + 0.16 * (tr == "agn"),
                    [r["offset_total"] for r in rows], fmt=m, color=c,
                    label=f"{tr.upper()} total")
        ax.scatter(x - 0.08 + 0.16 * (tr == "agn"),
                   [r["offset_numerator"] for r in rows], marker="x", color=c,
                   label=f"{tr.upper()} numerator only")
        ax.set_xticks(x)
        ax.set_xticklabels([f"seed {r['seed']}" for r in rows])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel(r"peak $-$ 67.74  [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_title("Probe 3 — where each piece peaks")
    ax.legend(fontsize=8)
    fig2.tight_layout()
    for ext in ("png", "pdf"):
        fig2.savefig(ROOT / "figs" / f"probe3_peaks.{ext}", dpi=150)
    print("Wrote figs/probe3_peaks.{png,pdf}")


def main(argv=None):
    args = parse_args(argv)
    if args.aggregate:
        aggregate(args)
    else:
        run_one(args)


if __name__ == "__main__":
    main()
