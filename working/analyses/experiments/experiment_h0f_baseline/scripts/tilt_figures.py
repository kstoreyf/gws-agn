#!/usr/bin/env python3
"""Figures for the H0-tilt mechanism diagnosis.

figs/tilt_decomposition.{png,pdf}   N, S, total (relative to truth) vs H0
figs/tilt_counterfactuals.{png,pdf} counterfactual-total peak shifts (budget)
figs/tilt_selection_model.{png,pdf} measured ln mu vs analytic dL-cut model
figs/tilt_leak.{png,pdf}            PE mass beyond the detection horizon vs H0
"""
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "figs"
H0_TRUE = 67.74
NOBS = 1000

# restrained categorical palette (Okabe-Ito subset), fixed assignment
C_NUM = "#0072B2"    # numerator
C_SEL = "#D55E00"    # selection
C_TOT = "#000000"    # total
C_CF1 = "#009E73"    # counterfactual (z-cut)
C_CF2 = "#CC79A7"    # counterfactual (mc-corrected)
C_MOD = "#56B4E9"    # analytic model

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "lines.linewidth": 1.6,
})


def quad_peak(x, y):
    y = np.asarray(y, float)
    i = int(np.nanargmax(y))
    if i in (0, len(y) - 1):
        return float(x[i])
    d = y[i - 1] - 2 * y[i] + y[i + 1]
    return float(x[i] - 0.5 * (y[i + 1] - y[i - 1]) / d * (x[1] - x[0])) \
        if d else float(x[i])


def load(tag):
    with h5py.File(ROOT / "results" / f"tilt_terms_{tag}.h5", "r") as fh:
        d = dict(
            H0=fh["H0_grid"][:],
            num={k: fh[f"numerator/{k}"][:] for k in fh["numerator"]},
            lnmu={k: fh[f"lnmu/{k}"][:] for k in fh["lnmu"]},
            neff=fh["sel_neff"][:],
            s2=fh["sigma2_ev"][:].sum(axis=0),
            fb={k: fh[f"frac_beyond/{k}"][:].mean(axis=0)
                for k in fh["frac_beyond"]},
        )
    d["S"] = -NOBS * d["lnmu"]["full"] + NOBS * (NOBS + 3.0) / (2 * d["neff"])
    d["total"] = d["num"]["full"] + d["S"]
    return d


def rel(y, x):
    return y - np.interp(H0_TRUE, x, y)


def fig_decomposition(data):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, (tag, f) in zip(axes, (("fagn0.3", 0.307), ("fagn0.7", 0.703))):
        d = data[tag]
        H = d["H0"]
        ax.plot(H, rel(d["num"]["full"], H), color=C_NUM, label="per-event numerator")
        ax.plot(H, rel(d["S"], H), color=C_SEL, label="selection term")
        ax.plot(H, rel(d["total"], H), color=C_TOT, label="total log L")
        for y, c in ((d["num"]["full"], C_NUM), (d["S"], C_SEL),
                     (d["total"], C_TOT)):
            p = quad_peak(H, y)
            if H[0] < p < H[-1]:
                ax.axvline(p, color=c, lw=0.8, ls=":", alpha=0.8)
        ax.axvline(H0_TRUE, color="0.4", lw=0.8, ls="--")
        ax.set_xlim(58, 78)
        ax.set_ylim(-120, 30)
        ax.set_xlabel(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]")
        ax.set_title(f"planted $f_{{\\rm AGN}}$ = {f}", fontsize=9)
    axes[0].set_ylabel(r"$\Delta \log L$ (rel. to truth) [nats]")
    axes[0].legend(frameon=False, fontsize=8, loc="lower center")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"tilt_decomposition.{ext}")
    plt.close(fig)


def fig_counterfactuals(data, budget):
    labels = [
        ("full model", "total_offset"),
        ("flat $\\beta$ (correct selection)", "numer_only"),
        ("catalog masked to $z \\leq 1$ (numerator)", "zcut_num"),
        ("catalog masked to $z \\leq 1$ (num.+sel.)", "zcut_both"),
        ("PE MC-bias corrected", "mc"),
        ("repaired: $z \\leq 1$ prior + flat $\\beta$", "repaired"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    y = np.arange(len(labels))[::-1]
    for j, (tag, f, off) in enumerate((("fagn0.3", 0.307, -0.18),
                                       ("fagn0.7", 0.703, 0.18))):
        d = data[tag]
        H = d["H0"]
        p0 = quad_peak(H, d["total"])
        vals = {
            "total_offset": p0 - H0_TRUE,
            "numer_only": quad_peak(H, d["num"]["full"]) - H0_TRUE,
            "zcut_num": quad_peak(H, d["num"]["zcut_1"] + d["S"]) - H0_TRUE,
            "zcut_both": quad_peak(
                H, d["num"]["zcut_1"] - NOBS * d["lnmu"]["zcut_1"]
                + NOBS * (NOBS + 3.0) / (2 * d["neff"])) - H0_TRUE,
            "mc": quad_peak(H, d["total"] + 0.5 * d["s2"]) - H0_TRUE,
            "repaired": quad_peak(
                H[np.isfinite(d["num"]["zcut_1"])],
                d["num"]["zcut_1"][np.isfinite(d["num"]["zcut_1"])])
            - H0_TRUE,
        }
        col = C_NUM if j == 0 else C_SEL
        ax.barh(y + off, [vals[k] for _, k in labels], height=0.32,
                color=col, label=f"$f_{{\\rm AGN}}$ = {f}")
        for yi, (_, k) in zip(y, labels):
            v = vals[k]
            ax.annotate(f"{v:+.2f}", (v, yi + off),
                        xytext=(4 if v >= 0 else -4, 0),
                        textcoords="offset points", va="center",
                        ha="left" if v >= 0 else "right", fontsize=7.5,
                        color="0.2")
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([l for l, _ in labels], fontsize=8.5)
    ax.set_xlabel(r"$H_0$ peak $-$ truth [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_xlim(-6.2, 4.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"tilt_counterfactuals.{ext}")
    plt.close(fig)


def fig_selection_model(data):
    model = json.loads((ROOT / "results" / "tilt_selection_model.json").read_text())
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    Hm = np.array(model["H0_grid"])
    for tag, f, c in (("fagn0.3", 0.307, C_SEL), ("fagn0.7", 0.703, C_NUM)):
        d = data[tag]
        H = d["H0"]
        ax.plot(H, rel(d["lnmu"]["full"], H), color=c,
                label=f"measured, $f$={f}")
        lm = np.array(model[tag]["lnmu_model"])
        ax.plot(Hm, lm - np.interp(H0_TRUE, Hm, lm), color=c, ls="--", lw=1.2,
                label=f"$dL$-cut model, $f$={f}")
    ax.axvline(H0_TRUE, color="0.4", lw=0.8, ls="--")
    ax.set_xlabel(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_ylabel(r"$\ln\mu$ (rel. to truth) [nats]")
    ax.set_xlim(58, 78)
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"tilt_selection_model.{ext}")
    plt.close(fig)


def fig_leak(data):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))
    ax = axes[0]
    for tag, f, c in (("fagn0.3", 0.307, C_NUM), ("fagn0.7", 0.703, C_SEL)):
        d = data[tag]
        for zc, ls in (("1", "-"), ("1.2", "--")):
            if zc in d["fb"]:
                ax.plot(d["H0"], 100 * d["fb"][zc], color=c, ls=ls,
                        label=f"$z>{zc}$, $f$={f}")
    ax.axvline(H0_TRUE, color="0.4", lw=0.8, ls="--")
    ax.set_xlabel(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_ylabel("mean PE posterior mass [%]")
    ax.set_title("PE mass mapped beyond the detection horizon", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5)

    # catalog dN/dz + event true z
    ax = axes[1]
    with h5py.File(ROOT / "data" / "gal.h5", "r") as fh:
        zg = fh["zgals"][:]; ng = fh["ngals"][:]
        m = np.arange(zg.shape[1])[None, :] < ng[:, None]
        zgal = zg[m]
    with h5py.File(ROOT / "data" / "agn.h5", "r") as fh:
        zg = fh["zgals"][:]; ng = fh["ngals"][:]
        m = np.arange(zg.shape[1])[None, :] < ng[:, None]
        zagn = zg[m]
    bins = np.linspace(0, 1.6, 65)
    for z, c, lab in ((zgal, C_NUM, "galaxies (b=1.2)"),
                      (zagn, C_SEL, "AGN (b=2.0)")):
        h, e = np.histogram(z, bins=bins, density=True)
        ax.stairs(h, e, color=c, label=lab, lw=1.4)
    ax.axvline(1.0, color="0.2", lw=1.0, ls="--")
    ax.annotate("detection\nhorizon $z=1$", (1.0, ax.get_ylim()[1] * 0.18),
                xytext=(-6, 0), textcoords="offset points", ha="right",
                fontsize=7.5, color="0.2")
    ax.axvline(1.5, color="0.5", lw=1.0, ls=":")
    ax.annotate("grid /\ncatalog\nedge\n$z=1.5$", (1.505, ax.get_ylim()[1] * 0.30),
                xytext=(4, 0), textcoords="offset points", ha="left",
                fontsize=7.5, color="0.4")
    ax.set_xlabel("$z$")
    ax.set_ylabel("normalized $dN/dz$")
    ax.set_title("catalog redshift distributions", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"tilt_leak.{ext}")
    plt.close(fig)


def main():
    FIGS.mkdir(exist_ok=True)
    data = {t: load(t) for t in ("fagn0.3", "fagn0.7")}
    budget = json.loads((ROOT / "results" / "tilt_budget.json").read_text()) \
        if (ROOT / "results" / "tilt_budget.json").exists() else {}
    fig_decomposition(data)
    fig_counterfactuals(data, budget)
    fig_selection_model(data)
    fig_leak(data)
    print("wrote figs/tilt_*.png|pdf")


if __name__ == "__main__":
    main()
