#!/usr/bin/env python3
"""PROBE 2 -- catalog-KDE window sweep (GPU).

Question: is the windowed catalog-KDE evaluator's truncation contributing to the
matched-GAL H0 offset?  The seed-100 GAL survey block is (12288, 14569) and the
analysis ran at W = 4096.  This rescans the seed-100 matched-GAL configuration
at W = 4096, 8192 and 14569 (= N_max, i.e. every window covers the whole row:
``start`` clips to ``n_max - window`` = 0) with EVERYTHING else identical --
same events, same targeted injections, same H0 grid [50, 100] x 201, same guard
convention, same reduction blocking -- and compares the per-cell log-likelihood
bitwise.

The reduction blocking is held at sel_batch_size / pe_event_block SMALLER than
the production scan's (a W = 14569 pass at the production blocking does not fit),
so all three windows are run at the SAME reduced blocking and the W = 4096 arm
doubles as a blocking-invariance check against the stored ``ctrl_gal_matched``.

Writes results/probe2_kde_window.json and figs/probe2_kde_window.png.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--windows", nargs="+", type=int, default=[4096, 8192, 14569])
    ap.add_argument("--sel_batch_size", type=int, default=50000)
    ap.add_argument("--pe_event_block", type=int, default=25)
    ap.add_argument("--h0_grid", nargs=3, type=float, default=[50.0, 100.0, 201])
    ap.add_argument("--skip_scans", action="store_true",
                    help="Analyse existing probe2_* outputs without rerunning.")
    return ap.parse_args(argv)


def run_scan(args, W):
    tag = f"probe2_gal_W{W}"
    h5 = ROOT / "results" / f"{tag}.h5"
    if args.skip_scans and h5.exists():
        print(f"[skip] {tag} exists")
        return tag
    sd = DATA / f"seed{args.seed}"
    cmd = [
        sys.executable, str(HERE / "scan_h0f.py"),
        "--universe_model", "dark_sirens_complete",
        "--catalog_sky_weighting", "field",
        "--scan", "h0",
        "--h0_grid", *[str(x) for x in args.h0_grid],
        "--h0_true", "67.74",
        "--survey_path", str(sd / "surveys" / "survey_gal_complete_ns32.h5"),
        "--gw_path", str(ROOT / "data_derived" / "events_gal_hosted.h5"),
        "--gwselection_path", str(sd / "injections" / "injections_targeted.h5"),
        "--selection_neff_guard", "hard",
        "--max_likelihood_variance", "1e6",
        "--sel_batch_size", str(args.sel_batch_size),
        "--pe_event_block", str(args.pe_event_block),
        "--kde_window", str(W),
        "--kde_window_nsigma", "8.0",
        "--outdir", str(ROOT / "results"),
        "--out_tag", tag,
    ]
    log = ROOT / "logs" / f"{tag}.log"
    print(f"[{time.strftime('%H:%M:%S')}] scanning W={W} -> {tag}")
    t0 = time.time()
    with open(log, "w") as fh:
        subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.STDOUT,
                       cwd=str(ROOT))
    print(f"[{time.strftime('%H:%M:%S')}] W={W} done in {time.time()-t0:.0f}s")
    return tag


def summarise(H0, logL):
    """Flat-prior trapezoid posterior summary (the analysis' own convention)."""
    ok = np.isfinite(logL)
    ll = np.where(ok, logL, -np.inf)
    p = np.exp(ll - np.nanmax(ll[ok]))
    p = np.where(ok, p, 0.0)
    norm = np.trapz(p, H0)
    p = p / norm
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(H0))])
    cdf /= cdf[-1]
    q = lambda t: float(np.interp(t, cdf, H0))            # noqa: E731
    return {"map": float(H0[int(np.argmax(ll))]),
            "median": q(0.5), "ci68": [q(0.16), q(0.84)],
            "ci90": [q(0.05), q(0.95)]}


def main(argv=None):
    args = parse_args(argv)
    import h5py

    tags = [run_scan(args, W) for W in args.windows]

    res = {"probe": 2, "name": "kde_window_sweep", "seed": args.seed,
           "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "config": {"windows": args.windows, "N_max_row": 14569,
                      "sel_batch_size": args.sel_batch_size,
                      "pe_event_block": args.pe_event_block,
                      "h0_grid": args.h0_grid,
                      "note": "all arms share the reduced reduction blocking so "
                              "the only difference between them is W"},
           "arms": {}}

    curves = {}
    for W, tag in zip(args.windows, tags):
        with h5py.File(ROOT / "results" / f"{tag}.h5", "r") as f:
            H0 = np.asarray(f["H0_grid"])
            ll = np.asarray(f["log_likelihood"])
            neff = np.asarray(f["guard/Neff"])
            rej = np.asarray(f["guard/rejected"])
            secs = float(f.attrs["steady_state_median_seconds"])
        curves[W] = (H0, ll)
        res["arms"][str(W)] = {
            "tag": tag, "n_rejected": int(rej.sum()),
            "min_Neff": float(neff.min()),
            "seconds_per_eval": secs,
            **summarise(H0, ll),
        }
        res["arms"][str(W)]["offset"] = res["arms"][str(W)]["median"] - 67.74

    # bitwise / numeric comparison against the FULL-row arm
    Wref = max(args.windows)
    H0, ref = curves[Wref]
    res["reference_window"] = Wref
    res["comparisons"] = {}
    for W in args.windows:
        if W == Wref:
            continue
        _, ll = curves[W]
        d = ll - ref
        # extremes of the grid get their own line: that is where truncation,
        # if it acted, would act hardest.
        res["comparisons"][f"W{W}_vs_W{Wref}"] = {
            "bitwise_identical": bool(np.array_equal(ll, ref)),
            "max_abs_dlogL": float(np.nanmax(np.abs(d))),
            "max_abs_dlogL_at_H0": float(H0[int(np.nanargmax(np.abs(d)))]),
            "dlogL_at_H0_min": float(d[0]),
            "dlogL_at_H0_max": float(d[-1]),
            "dlogL_at_truth": float(d[int(np.argmin(np.abs(H0 - 67.74)))]),
            "d_median_kmsMpc": (res["arms"][str(W)]["median"]
                                - res["arms"][str(Wref)]["median"]),
            # shape-only: the likelihood is defined up to a constant, so the
            # inference-relevant quantity is the SPREAD of d, not its offset
            "dlogL_minus_mean_max_abs": float(np.nanmax(np.abs(d - np.nanmean(d)))),
        }

    # the production scan, for the blocking-invariance cross-check
    prod = ROOT / "results" / "ctrl_gal_matched.h5"
    if prod.exists():
        with h5py.File(prod, "r") as f:
            llp = np.asarray(f["log_likelihood"])
            H0p = np.asarray(f["H0_grid"])
        if 4096 in curves and np.array_equal(H0p, curves[4096][0]):
            d = curves[4096][1] - llp
            res["blocking_invariance_W4096_vs_production"] = {
                "production_blocking": {"sel_batch_size": 200000, "pe_event_block": 100},
                "bitwise_identical": bool(np.array_equal(curves[4096][1], llp)),
                "max_abs_dlogL": float(np.nanmax(np.abs(d))),
                "dlogL_minus_mean_max_abs": float(np.nanmax(np.abs(d - np.nanmean(d)))),
                "production_median": 62.785390760139784,
                "probe_median": res["arms"]["4096"]["median"],
            }

    worst = max((v["dlogL_minus_mean_max_abs"]
                 for v in res["comparisons"].values()), default=0.0)
    worst_med = max((abs(v["d_median_kmsMpc"])
                     for v in res["comparisons"].values()), default=0.0)
    res["verdict"] = {
        "max_shape_dlogL_vs_full_row": worst,
        "max_median_shift_kmsMpc": worst_med,
        "summary": ("WINDOW TRUNCATION DOES NOT CONTRIBUTE"
                    if worst_med < 0.01 else
                    "WINDOW TRUNCATION CONTRIBUTES -- characterise"),
    }

    p = ROOT / "results" / "probe2_kde_window.json"
    p.write_text(json.dumps(res, indent=2))
    print(json.dumps(res["arms"], indent=2))
    print(json.dumps(res["comparisons"], indent=2))
    print("VERDICT:", res["verdict"]["summary"])
    print(f"Wrote {p}")

    # ---- figure ----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})
    for W in args.windows:
        H0, ll = curves[W]
        ax[0].plot(H0, ll - np.nanmax(ll), lw=1.6,
                   label=f"W = {W}" + (" (full row)" if W == 14569 else ""))
    ax[0].axvline(67.74, color="k", ls=":", lw=1, label="truth 67.74")
    ax[0].set_ylabel(r"$\log \mathcal{L} - \max$")
    ax[0].set_ylim(-40, 2)
    ax[0].legend(fontsize=8)
    ax[0].set_title("Probe 2 — matched-GAL seed 100: catalog-KDE window sweep")
    for W in args.windows:
        if W == Wref:
            continue
        H0, ll = curves[W]
        d = ll - curves[Wref][1]
        ax[1].plot(H0, d - np.nanmean(d), lw=1.4, label=f"W={W} − W={Wref}")
    ax[1].axhline(0, color="k", lw=0.7)
    ax[1].axvline(67.74, color="k", ls=":", lw=1)
    ax[1].set_xlabel(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]")
    ax[1].set_ylabel(r"$\Delta\log\mathcal{L}$ (mean-removed)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figs" / f"probe2_kde_window.{ext}", dpi=150)
    print("Wrote figs/probe2_kde_window.{png,pdf}")


if __name__ == "__main__":
    main()
