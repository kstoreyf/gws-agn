#!/usr/bin/env python
"""Analysis 13 figures, deterministic, from results/ only.

    fig_13_marginals  the eight marginals of seed 1 (both shared widths free), against 11D
                      (widths fixed at 5 Msun and 0.1) where 11D samples the parameter;
                      90% intervals as strips under each panel, planted values in ink.
    fig_13_edge       the (Delta mu_chi, H0) plane: 90% HPD region of seed 1, with the
                      samples of the clump at the Delta mu_chi = 0.30 prior edge drawn as points.

Every drawn median / 90% end is diffed against its JSON (``check``). All plotted intervals
and regions are 90%. Colours from ../../paper/scripts/figstyle.py: Analysis 13 = blue,
Analysis 11D = aqua, edge clump = orange, planted values in ink. KDE bandwidths use the Kish
n_eff.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
A13 = HERE.parent
RESULTS, FIGS = A13 / "results", A13 / "figs"
A11_RESULTS = A13.parent / "analysis_11_free_common_population" / "results"
sys.path.insert(0, str(A13.parent.parent / "paper" / "scripts"))
import figstyle as fs  # noqa: E402

fs.use()
C13, C11, CEDGE = fs.C["blue"], fs.C["aqua"], fs.C["orange"]
RUN = "a13core_dynesty_n200_s1"
EDGE = 0.26          # Delta mu_chi above which a sample belongs to the edge clump (none in 0.23-0.28)
NAMES = ("H0", "f_agn", "mu_G", "dmu_G", "mu_chi", "dmu_chi", "sigma_G", "sigma_chi")
TRUTH = {"H0": fs.H0_TRUTH, "f_agn": 0.30, "mu_G": 35.0, "dmu_G": 5.0, "mu_chi": 0.0,
         "dmu_chi": 0.10, "sigma_G": 5.0, "sigma_chi": 0.10}
LABEL = {"H0": r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]", "f_agn": r"$f_{\rm AGN}$",
         "mu_G": r"$\mu_{\rm G}$  [$M_\odot$]", "dmu_G": r"$\Delta\mu_{\rm G}$  [$M_\odot$]",
         "mu_chi": r"$\mu_\chi$", "dmu_chi": r"$\Delta\mu_\chi$",
         "sigma_G": r"$\sigma_{\rm G}$  [$M_\odot$] (shared)", "sigma_chi": r"$\sigma_\chi$ (shared)"}


def check(label, got, want, tol=1e-9):
    d = abs(got - want)
    print(f"      {label:<28s} drawn {got:.10f}  json {want:.10f}  |d| {d:.2e}  "
          f"{'OK' if d <= tol else 'MISMATCH'}")
    if d > tol:
        raise SystemExit(f"[fatal] {label}: drawn value does not match the JSON")


def load(path):
    js = json.loads(Path(str(path) + ".json").read_text())
    Z = np.load(str(path) + ".npz")
    names = [str(n) for n in Z["names"]]
    w = np.exp(Z["logwt"] - Z["logwt"].max())
    col = {n: Z["samples"][:, names.index(n)] for n in names}
    return js, col, float(w.sum() ** 2 / (w ** 2).sum())


def kde1(x, grid, neff):
    p = gaussian_kde(x, bw_method=neff ** (-1 / 5))(grid)
    return p / np.trapz(p, grid)


def kde2(x, y, gx, gy, neff):
    k = gaussian_kde(np.vstack([x, y]), bw_method=neff ** (-1 / 6))
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    return k(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)


def hpd90(P):
    v = np.sort(P.ravel())[::-1]
    c = np.cumsum(v) / v.sum()
    return v[np.searchsorted(c, 0.90)]


def strip(ax, k, ymax, c, ss):
    y = -0.07 * ymax * (k + 1)
    ax.plot(ss["ci90"], [y, y], color=c, lw=2.4, solid_capstyle="butt", clip_on=False)
    ax.plot([ss["median"]], [y], marker="|", color=c, ms=7, mew=1.6, clip_on=False)


def marginals(js, col, neff, j11, c11, n11):
    s, box = js["summary"], {b[0]: (b[1], b[2]) for b in js["box"]}
    fig, axs = plt.subplots(2, 4, figsize=(fs.TWOCOL, 3.6))
    fig.subplots_adjust(left=0.03, right=0.99, bottom=0.13, top=0.9, wspace=0.18, hspace=0.62)
    for ax, n in zip(axs.ravel(), NAMES):
        q = np.quantile(col[n], [0.05, 0.5, 0.95])
        check(f"{n} median", float(q[1]), s[n]["median"])
        check(f"{n} 90% low", float(q[0]), s[n]["ci90"][0])
        check(f"{n} 90% high", float(q[2]), s[n]["ci90"][1])
        has11 = n in c11
        xs = [col[n]] + ([c11[n]] if has11 else [])
        lo = min(np.quantile(x, 0.0005) for x in xs); hi = max(np.quantile(x, 0.9995) for x in xs)
        lo, hi = min(lo, TRUTH[n]), max(hi, TRUTH[n])
        grid = np.linspace(lo - 0.12 * (hi - lo), hi + 0.12 * (hi - lo), 500)
        inside = (grid >= box[n][0]) & (grid <= box[n][1])
        ymax = 0.0
        for x, c, ne, lw in ([(c11[n], C11, n11, 1.4)] if has11 else []) + [(col[n], C13, neff, 1.5)]:
            p = kde1(x, grid, ne); p[~inside] = 0.0; p /= np.trapz(p, grid)
            ax.plot(grid, p, color=c, lw=lw); ymax = max(ymax, p.max())
        strip(ax, 0, ymax, C13, s[n])
        if has11:
            strip(ax, 1, ymax, C11, j11["summary"][n])
        ax.set_ylim(-0.07 * ymax * 2.8, ymax * 1.25)
        ax.axvline(TRUTH[n], color=fs.TRUTH, lw=0.8, ls=(0, (3, 2)), alpha=0.75)
        for e in box[n]:
            if grid[0] < e < grid[-1] or e == grid[-1]:
                ax.axvline(e, color=fs.MUTED, lw=0.9)
        ax.set_xlim(grid[0], grid[-1]); ax.set_yticks([]); ax.grid(False, axis="y")
        ax.set_xlabel(LABEL[n], fontsize=7.5)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(labelsize=6.5)
    fig.legend(handles=[Line2D([], [], color=C13, lw=1.5, label="Analysis 13, seed 1: both widths free"),
                        Line2D([], [], color=C11, lw=1.4, label="11D: widths fixed"),
                        Line2D([], [], color=fs.TRUTH, lw=0.8, ls=(0, (3, 2)), label="planted"),
                        Line2D([], [], color=fs.MUTED, lw=0.9, label="prior edge")],
               loc="upper center", ncol=4, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    return fig


def edge_plane(js, col, neff):
    box = {b[0]: (b[1], b[2]) for b in js["box"]}
    x, y = col["dmu_chi"], col["H0"]
    clump = x > EDGE
    frac = float(clump.mean())
    print(f"      edge clump: {clump.sum()} of {len(x)} equal-weight samples ({frac:.4f})")
    fig, ax = plt.subplots(figsize=(fs.ONECOL, 2.7))
    gx = np.linspace(box["dmu_chi"][0], box["dmu_chi"][1], 200)
    gy = np.linspace(60.0, 76.0, 200)
    P = kde2(x, y, gx, gy, neff)
    l90 = hpd90(P)
    ax.contourf(gx, gy, P.T, levels=[l90, P.max() * 1.01], colors=[C13], alpha=0.22)
    ax.contour(gx, gy, P.T, levels=[l90], colors=C13, linewidths=1.3)
    ax.scatter(x[clump], y[clump], s=3, color=CEDGE, lw=0, alpha=0.6, zorder=3)
    ax.axvline(TRUTH["dmu_chi"], color=fs.TRUTH, lw=0.8, ls=(0, (3, 2)), alpha=0.75)
    ax.axhline(TRUTH["H0"], color=fs.TRUTH, lw=0.8, ls=(0, (3, 2)), alpha=0.75)
    ax.axvline(box["dmu_chi"][1], color=fs.MUTED, lw=0.9)
    ax.set_xlim(0.0, box["dmu_chi"][1] + 0.01); ax.set_ylim(61.0, 75.0)
    ax.set_xlabel(LABEL["dmu_chi"]); ax.set_ylabel(LABEL["H0"])
    ax.legend(handles=[Line2D([], [], color=C13, lw=1.3, label="90% region"),
                       Line2D([], [], color=CEDGE, marker="o", ls="", ms=2.5,
                              label=f"edge clump ({100 * frac:.1f}% of samples)"),
                       Line2D([], [], color=fs.TRUTH, lw=0.8, ls=(0, (3, 2)), label="planted")],
              fontsize=5.8, frameon=True, framealpha=1.0, edgecolor="none", loc="upper left")
    return fig


def main():
    if not (RESULTS / f"{RUN}.json").exists():
        print("[skip] seed 1 not finished")
        return
    js, col, neff = load(RESULTS / RUN)
    j11, c11, n11 = load(A11_RESULTS / "a11_11D")
    print(f"[{RUN}]  Kish n_eff {neff:.0f}")
    FIGS.mkdir(exist_ok=True)
    for name, fig in (("fig_13_marginals", marginals(js, col, neff, j11, c11, n11)),
                      ("fig_13_edge", edge_plane(js, col, neff))):
        for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
            fig.savefig(FIGS / f"{name}.{ext}", **kw)
            print(f"    wrote figs/{name}.{ext}")
        plt.close(fig)


if __name__ == "__main__":
    main()
