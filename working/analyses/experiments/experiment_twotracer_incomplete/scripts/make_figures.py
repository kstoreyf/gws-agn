#!/usr/bin/env python3
"""How fast does f_AGN lose precision as the host survey becomes incomplete?

  figs/fig_incomplete_widths.{pdf,png}   -- the posteriors, and the degradation
  figs/fig_incomplete_joint.{pdf,png}    -- the (H0, f_AGN) plane along the ladder
  results/summary.json

The headline is the WIDTH, not the centre: this mock inherits an unresolved
absolute offset from ../experiment_matched_mock (see DESIGN.md), so every
statement here is differential against the complete-catalog rung.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from scipy.ndimage import binary_dilation

BASE = Path(__file__).resolve().parent.parent
RESULTS, FIGS, DD = BASE / "results", BASE / "figs", BASE / "data_derived"
FIGS.mkdir(parents=True, exist_ok=True)

LEVELS = ["complete", "m21.0", "m20.0", "m19.0", "m18.0"]
LABELS = {"complete": "complete", "m21.0": "$m<21$", "m20.0": "$m<20$",
          "m19.0": "$m<19$", "m18.0": "$m<18$"}
MAGLIM = {"complete": None, "m21.0": 21.0, "m20.0": 20.0, "m19.0": 19.0, "m18.0": 18.0}
Z_REF = 0.30
TRUTH_F, TRUTH_H0 = 0.30, 67.74

BLUE, AQUA, YELLOW, RED, INK, INK2, INK3 = ("#2a78d6", "#1baf7a", "#eda100",
                                            "#e34948", "#0b0b0b", "#52514e", "#898781")
GRIDCOL = "#e1e0d9"
RAMP = ["#0b0b0b", "#2a78d6", "#1baf7a", "#eda100", "#e34948"]
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.family": "serif",
    "font.serif": ["DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 7.6,
    "axes.edgecolor": INK2, "axes.linewidth": 0.7, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "grid.color": GRIDCOL, "grid.linewidth": 0.6,
    "legend.frameon": False, "lines.solid_capstyle": "round",
})


def completeness_table():
    """C(z <= Z_REF) and host counts per tracer per rung, from the catalogs."""
    cat = {}
    for t in ("gal", "agn"):
        with h5py.File(DD / f"catalog_{t}_complete.h5", "r") as f:
            cat[t] = (np.asarray(f["z"][:]), np.asarray(f["app_mag"][:]))
    out = {}
    for lev in LEVELS:
        ml = MAGLIM[lev]
        rec = {}
        for t in ("gal", "agn"):
            z, m = cat[t]
            keep = np.ones_like(z, dtype=bool) if ml is None else (m < ml)
            inh = z <= Z_REF
            with h5py.File(DD / f"survey_{t}_{lev}_ns32.h5", "r") as f:
                ng = np.asarray(f["ngals"][:])
            rec[t] = {
                "n_hosts": int(keep.sum()),
                "n_hosts_within_horizon": int((keep & inh).sum()),
                "completeness_within_horizon": float((keep & inh).sum() / inh.sum()),
                "empty_pixel_fraction": float(1.0 - (ng > 0).mean()),
            }
        out[lev] = rec
    return out


def posterior_1d(h5, key="f_grid"):
    with h5py.File(h5, "r") as f:
        x, ll = f[key][:], f["log_likelihood"][:]
    ok = np.isfinite(ll)
    if not ok.any():
        return x, np.zeros_like(x), ok
    p = np.where(ok, np.exp(ll - np.nanmax(ll[ok])), 0.0)
    return x, p / np.trapz(p, x), ok


def joint_stats(h5):
    with h5py.File(h5, "r") as f:
        H, F, ll = f["H0_grid"][:], f["f_grid"][:], f["log_likelihood"][:]
    ok = np.isfinite(ll)
    p = np.where(ok, np.exp(ll - np.nanmax(ll[ok])), 0.0)
    cell = np.outer(np.gradient(H), np.gradient(F))
    pw = p * cell
    pw = pw / pw.sum()
    edge = binary_dilation(~ok) & ok
    Hm, Fm = np.meshgrid(H, F, indexing="ij")
    mH, mF = (Hm * pw).sum(), (Fm * pw).sum()
    sH = float(np.sqrt(((Hm - mH) ** 2 * pw).sum()))
    sF = float(np.sqrt(((Fm - mF) ** 2 * pw).sum()))
    rho = float(((Hm - mH) * (Fm - mF) * pw).sum() / (sH * sF)) if sH * sF > 0 else 0.0
    return {"H0_mean": float(mH), "f_mean": float(mF), "H0_sd": sH, "f_sd": sF,
            "rho": rho, "n_rejected": int((~ok).sum()), "n_evals": int(ok.size),
            "posterior_mass_adjacent_to_rejected": float(pw[edge].sum()),
            "grids": (H, F), "pw": pw, "ok": ok}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="", help="_fix reads *_fix results and writes summary_fix.json / *_fix figures")
    sfx = ap.parse_args().suffix
    S = {"z_ref": Z_REF, "truth": {"f_AGN": TRUTH_F, "H0": TRUTH_H0},
         "completeness": completeness_table(), "levels": {}}

    for lev in LEVELS:
        rec = {"label": LABELS[lev], "mag_limit": MAGLIM[lev],
               **{f"{t}_{k}": v for t, d in S["completeness"][lev].items()
                  for k, v in d.items()}}
        gp = RESULTS / f"guard_{lev}{sfx}.json"
        if gp.exists():
            g = json.loads(gp.read_text())["guard_records"][0]
            rec["Neff"] = g["Neff"]
            rec["guard_threshold"] = g["threshold"]
            rec["passes_guard"] = g["passes_legacy_floor"]
            rec["pe_variance_sum"] = g["pe_variance_sum"]
        fp = RESULTS / f"fscan_{lev}{sfx}.json"
        if fp.exists():
            j = json.loads(fp.read_text())
            fb = j["f"]
            rec["fscan"] = {
                "median": fb["median"], "ci68": fb["ci68"], "ci90": fb["ci90"],
                "half_width68": 0.5 * (fb["ci68"][1] - fb["ci68"][0]),
                "argmax": fb["argmax"], "truth_in_ci68": fb.get("truth_in_ci68"),
                "n_rejected": j["n_neginf_cells"], "n_evals": j["n_evals"]}
        jp = RESULTS / f"joint_{lev}{sfx}.json"
        if jp.exists():
            j = json.loads(jp.read_text())
            st = joint_stats(RESULTS / f"joint_{lev}{sfx}.h5")
            rec["joint"] = {
                "map": j["map"],
                "H0_median": j["H0"]["median"], "H0_ci68": j["H0"]["ci68"],
                "H0_half_width68": 0.5 * (j["H0"]["ci68"][1] - j["H0"]["ci68"][0]),
                "f_median": j["f"]["median"], "f_ci68": j["f"]["ci68"],
                "f_half_width68": 0.5 * (j["f"]["ci68"][1] - j["f"]["ci68"][0]),
                "rho": st["rho"], "H0_sd": st["H0_sd"], "f_sd": st["f_sd"],
                "n_rejected": st["n_rejected"], "n_evals": st["n_evals"],
                "posterior_mass_adjacent_to_rejected":
                    st["posterior_mass_adjacent_to_rejected"]}
        S["levels"][lev] = rec

    # Sky-shuffle null: the same scan with each event's distance paired to another
    # event's sky patch.  Width that survives the permutation was never
    # host-association information (see shuffle_event_sky.py).
    for lev in LEVELS:
        np_ = RESULTS / f"fscan_null_{lev}{sfx}.json"
        if not np_.exists():
            continue
        nb = json.loads(np_.read_text())["f"]
        r = S["levels"][lev]
        hw_null = 0.5 * (nb["ci68"][1] - nb["ci68"][0])
        rec = {"median": nb["median"], "ci68": nb["ci68"], "half_width68": hw_null}
        if "fscan" in r:
            hw = r["fscan"]["half_width68"]
            rec["width_ratio_null_over_real"] = hw_null / hw
            rec["peak_displacement"] = r["fscan"]["median"] - nb["median"]
            rec["displacement_in_widths"] = (r["fscan"]["median"] - nb["median"]) / hw
        r["sky_shuffle_null"] = rec

    # degradation factors relative to the complete rung
    base = S["levels"]["complete"]
    for lev in LEVELS:
        r = S["levels"][lev]
        d = {}
        if "fscan" in r and "fscan" in base:
            d["fscan_f"] = r["fscan"]["half_width68"] / base["fscan"]["half_width68"]
        if "joint" in r and "joint" in base:
            d["joint_f"] = r["joint"]["f_half_width68"] / base["joint"]["f_half_width68"]
            d["joint_H0"] = r["joint"]["H0_half_width68"] / base["joint"]["H0_half_width68"]
        r["width_degradation_vs_complete"] = d

    # ------------------------------------------------------------------ fig 1
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.5, 3.2), dpi=300)
    for i, lev in enumerate(LEVELS):
        h5 = RESULTS / f"fscan_{lev}{sfx}.h5"
        if not h5.exists():
            continue
        x, p, ok = posterior_1d(h5)
        c = RAMP[i]
        cw = S["completeness"][lev]["agn"]["completeness_within_horizon"]
        axL.plot(x, p, color=c, lw=1.6, zorder=4,
                 label=f"{LABELS[lev]}  ($C\\simeq{cw:.2f}$)")
        if (~ok).any():
            axL.axvspan(float(x[ok].max()), float(x.max()), color=c, alpha=0.06,
                        lw=0, zorder=1)
        nh5 = RESULTS / f"fscan_null_{lev}{sfx}.h5"
        if nh5.exists():
            xn, pn, _ = posterior_1d(nh5)
            axL.plot(xn, pn, color=c, lw=1.0, ls=(0, (2.5, 1.8)), alpha=0.75, zorder=3)
    axL.axvline(TRUTH_F, color=INK2, lw=0.9, ls=(0, (1, 2)), zorder=2)
    axL.annotate("planted", xy=(TRUTH_F - 0.01, 0.94), xycoords=("data", "axes fraction"),
                 fontsize=7.0, color=INK2, ha="right", va="top")
    axL.set_xlim(0, 0.8)
    axL.set_ylim(bottom=0)
    axL.set_xlabel(r"AGN-hosted fraction  $f_{\rm AGN}$")
    axL.set_ylabel("posterior density")
    axL.set_title("Solid: the data.  Dashed: sky-shuffled null\n"
                  "(the separation is what the hosts buy)", fontsize=8.6)
    axL.grid(True, alpha=0.55)
    axL.set_axisbelow(True)
    axL.legend(loc="upper right")

    C = [S["completeness"][l]["agn"]["completeness_within_horizon"] for l in LEVELS]
    for key, col, mark, lab in (
            ("joint_H0", RED, "s", r"$\sigma(H_0)$"),
            ("joint_f", BLUE, "o", r"$\sigma(f_{\rm AGN})$")):
        y = [S["levels"][l]["width_degradation_vs_complete"].get(key) for l in LEVELS]
        pts = [(c, v) for c, v in zip(C, y) if v is not None]
        if pts:
            axR.plot(*zip(*pts), color=col, lw=1.7, marker=mark, ms=4.2, zorder=4,
                     label=lab)
    # The statistic that IS host-association information: how far the data move
    # the f peak away from the sky-shuffled null, in units of the width.
    pts = [(S["completeness"][l]["agn"]["completeness_within_horizon"],
            S["levels"][l]["sky_shuffle_null"]["displacement_in_widths"])
           for l in LEVELS if "sky_shuffle_null" in S["levels"][l]]
    if pts:
        base_d = pts[0][1]
        axR.plot([c for c, _ in pts], [base_d / v for _, v in pts], color=AQUA,
                 lw=1.7, marker="^", ms=4.6, zorder=4,
                 label="loss of AGN-detection significance\n"
                       r"(peak$-$null separation, in widths)")
    axR.axhline(1.0, color=INK2, lw=0.9, ls=(0, (1, 2)), zorder=2)
    axR.set_xlim(1.03, 0.10)          # thinning to the right
    axR.set_xlabel(r"completeness within the horizon  $C(z\leq0.30)$")
    axR.set_ylabel("degradation  /  complete-catalog value")
    axR.set_title("How much precision degrades")
    axR.grid(True, alpha=0.55)
    axR.set_axisbelow(True)
    axR.legend(loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_incomplete_widths{sfx}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote figs/fig_incomplete_widths.{pdf,png}")

    # ------------------------------------------------------------------ fig 2
    have = [l for l in LEVELS if (RESULTS / f"joint_{l}{sfx}.h5").exists()]
    if have:
        fig, axes = plt.subplots(1, len(have), figsize=(2.05 * len(have), 2.65),
                                 dpi=300, sharey=True)
        axes = np.atleast_1d(axes)
        for ax, lev, col in zip(axes, have, RAMP):
            st = joint_stats(RESULTS / f"joint_{lev}{sfx}.h5")
            H, F = st["grids"]
            pw, ok = st["pw"], st["ok"]
            Hm, Fm = np.meshgrid(H, F, indexing="ij")
            w = pw.ravel()
            order = np.argsort(w)[::-1]
            cs = np.cumsum(w[order])
            lv = [float(pw.ravel()[order[min(np.searchsorted(cs, fr), order.size - 1)]])
                  for fr in (0.68, 0.90)]
            ax.contourf(Hm, Fm, np.where(ok, np.nan, 1.0), levels=[0.5, 1.5],
                        colors=[RED], alpha=0.08, zorder=1)
            ax.contourf(Hm, Fm, pw, levels=[lv[1], lv[0], pw.max()],
                        colors=[col, col], alpha=0.22, zorder=3)
            ax.contour(Hm, Fm, pw, levels=sorted(lv), colors=[col], linewidths=1.0,
                       zorder=4)
            ax.plot([TRUTH_H0], [TRUTH_F], marker="*", ms=9, color=YELLOW, mec=INK,
                    mew=0.5, ls="none", zorder=6)
            ax.set_xlim(58, 78)
            ax.set_ylim(0, 1)
            ax.set_xticks([60, 68, 76])
            ax.set_title(LABELS[lev])
            ax.set_xlabel(r"$H_0$")
            ax.grid(True, alpha=0.4)
            ax.set_axisbelow(True)
        axes[0].set_ylabel(r"$f_{\rm AGN}$")
        fig.suptitle("Two-tracer plane along the completeness ladder "
                     "(star = truth, shaded red = inadmissible)", fontsize=9)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(FIGS / f"fig_incomplete_joint{sfx}.{ext}", bbox_inches="tight")
        plt.close(fig)
        print("wrote figs/fig_incomplete_joint.{pdf,png}")

    (RESULTS / f"summary{sfx}.json").write_text(json.dumps(
        {k: v for k, v in S.items()}, indent=2, default=float))
    print("wrote results/summary.json\n")
    hdr = (f"{'level':>9} {'C_agn':>6} {'AGN in horiz':>12} {'Neff':>9} "
           f"{'f (fscan)':>18} {'sig_f':>7} {'x':>5} | {'H0 (joint)':>16} "
           f"{'sig_H0':>7} {'x':>5}")
    print(hdr)
    for lev in LEVELS:
        r = S["levels"][lev]
        c = S["completeness"][lev]["agn"]
        f_ = r.get("fscan", {})
        j_ = r.get("joint", {})
        d_ = r["width_degradation_vs_complete"]
        print(f"{lev:>9} {c['completeness_within_horizon']:6.3f} "
              f"{c['n_hosts_within_horizon']:12,} {r.get('Neff', float('nan')):9.0f} "
              f"{f_.get('median', float('nan')):8.4f} "
              f"[{f_.get('ci68',[float('nan')]*2)[0]:.3f},{f_.get('ci68',[float('nan')]*2)[1]:.3f}] "
              f"{f_.get('half_width68', float('nan')):7.4f} "
              f"{d_.get('fscan_f', float('nan')):5.2f} | "
              f"{j_.get('H0_median', float('nan')):8.3f} "
              f"{j_.get('H0_half_width68', float('nan')):7.3f} "
              f"{d_.get('joint_H0', float('nan')):5.2f}")


if __name__ == "__main__":
    main()
