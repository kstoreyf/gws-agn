#!/usr/bin/env python
"""Analysis 12 figures, deterministic, from results/ only.

    fig_12   one row per arm. Left: the (shared width, offset) plane, 90% HPD region
             with the width free. Right: p(offset) with the width free against 11D
             (width pinned at its fiducial), 90% intervals as strips.
             Row 1 12M: (sigma_G, dmu_G). Row 2 12chi: (sigma_chi, dmu_chi).

Every drawn median / 90% end is diffed against its JSON (``check``). All plotted
intervals and regions are 90%. Colours from ../../paper/scripts/figstyle.py: Analysis 12
= blue, Analysis 11D = aqua, planted values in ink. KDE bandwidths use the Kish n_eff.
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
A12 = HERE.parent
RESULTS, FIGS = A12 / "results", A12 / "figs"
A11_RESULTS = A12.parent / "analysis_11_free_common_population" / "results"
sys.path.insert(0, str(A12.parent.parent / "paper" / "scripts"))
import figstyle as fs  # noqa: E402

fs.use()
C12, C11 = fs.C["blue"], fs.C["aqua"]
ARMS = (("12M", "sigma_G", "dmu_G", 5.0, 5.0),
        ("12chi", "sigma_chi", "dmu_chi", 0.10, 0.10))
LABEL = {"sigma_G": r"$\sigma_{\rm G}$  [$M_\odot$] (shared)",
         "dmu_G": r"$\Delta\mu_{\rm G}$  [$M_\odot$]",
         "sigma_chi": r"$\sigma_\chi$ (shared)", "dmu_chi": r"$\Delta\mu_\chi$"}


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


def main():
    j11, c11, n11 = load(A11_RESULTS / "a11_11D")
    rows = [a for a in ARMS if (RESULTS / f"a12_{a[0]}.json").exists()]
    if not rows:
        print("[skip] nothing merged yet")
        return
    fig, axs = plt.subplots(len(rows), 2, figsize=(fs.TWOCOL, 2.5 * len(rows)), squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.11, top=0.97, wspace=0.3, hspace=0.5)
    for r, (arm, w, mark, w_true, m_true) in enumerate(rows):
        js, col, neff = load(RESULTS / f"a12_{arm}")
        s = js["summary"]
        box = {b[0]: (b[1], b[2]) for b in js["box"]}
        print(f"\n[{arm}]  Kish n_eff {neff:.0f}")
        for n in (w, mark):
            q = np.quantile(col[n], [0.05, 0.5, 0.95])
            check(f"{arm} {n} median", float(q[1]), s[n]["median"])
            check(f"{arm} {n} 90% low", float(q[0]), s[n]["ci90"][0])
            check(f"{arm} {n} 90% high", float(q[2]), s[n]["ci90"][1])
        # (width, offset) plane
        ax = axs[r][0]
        lw, hw = np.quantile(col[w], [0.001, 0.999]); lm, hm = np.quantile(col[mark], [0.001, 0.999])
        gw = np.linspace(lw - 0.3 * (hw - lw), hw + 0.3 * (hw - lw), 120)
        gm = np.linspace(lm - 0.3 * (hm - lm), hm + 0.3 * (hm - lm), 120)
        gw = gw[(gw >= box[w][0]) & (gw <= box[w][1])]
        gm = gm[(gm >= box[mark][0]) & (gm <= box[mark][1])]
        P = kde2(col[w], col[mark], gw, gm, neff)       # the region is cut at the prior box
        _, l90 = fs.hpd_levels_2d(gw, gm, P)
        ax.contourf(gw, gm, P.T, levels=[l90, P.max() * 1.01], colors=[C12], alpha=0.22)
        ax.contour(gw, gm, P.T, levels=[l90], colors=C12, linewidths=1.3)
        ax.axvline(w_true, color=fs.TRUTH, lw=0.8, ls=(0, (3, 2)), alpha=0.75)
        ax.axhline(m_true, color=fs.TRUTH, lw=0.8, ls=(0, (3, 2)), alpha=0.75)
        cut = gm[-1] >= box[mark][1] - 0.01 * (gm[-1] - gm[0])   # the grid reaches the prior box
        ax.set_xlim(gw[0], gw[-1]); ax.set_ylim(gm[0], gm[-1] + (0.12 if cut else 0.0) * (gm[-1] - gm[0]))
        if cut:
            ax.axhline(box[mark][1], color=fs.MUTED, lw=0.9)
            ax.annotate("prior edge", (0.97, box[mark][1]), xycoords=("axes fraction", "data"),
                        fontsize=5.6, ha="right", va="bottom", color=fs.MUTED)
        ax.set_xlabel(LABEL[w]); ax.set_ylabel(LABEL[mark])
        rho = s["correlations"].get(f"{mark}|{w}", s["correlations"].get(f"{w}|{mark}"))
        ax.annotate(rf"$\rho$ = {rho:+.2f}", (0.05, 0.06), xycoords="axes fraction",
                    fontsize=6.5, va="bottom")
        # p(offset): width free against 11D
        ax = axs[r][1]
        lo = min(np.quantile(col[mark], 0.0005), np.quantile(c11[mark], 0.0005))
        hi = max(np.quantile(col[mark], 0.9995), np.quantile(c11[mark], 0.9995))
        grid = np.linspace(lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo), 400)
        p12, p11 = kde1(col[mark], grid, neff), kde1(c11[mark], grid, n11)
        # no density outside the prior box (both runs share it); renormalise inside
        inside = (grid >= box[mark][0]) & (grid <= box[mark][1])
        for p_ in (p12, p11):
            p_[~inside] = 0.0
            p_ /= np.trapz(p_, grid)
        ax.plot(grid, p11, color=C11, lw=1.4)
        ax.plot(grid, p12, color=C12, lw=1.5)
        ymax = max(p11.max(), p12.max())
        for k, (c, ss) in enumerate(((C12, s[mark]), (C11, j11["summary"][mark]))):
            y = -0.07 * ymax * (k + 1)
            ax.plot(ss["ci90"], [y, y], color=c, lw=2.4, solid_capstyle="butt", clip_on=False)
            ax.plot([ss["median"]], [y], marker="|", color=c, ms=7, mew=1.6, clip_on=False)
        ax.set_ylim(-0.07 * ymax * 2.8, ymax * 1.3)
        ax.axvline(m_true, color=fs.TRUTH, lw=0.8, ls=(0, (3, 2)), alpha=0.75)
        if grid[-1] > box[mark][1]:
            ax.axvline(box[mark][1], color=fs.MUTED, lw=0.9)
            ax.annotate("prior edge", (box[mark][1], 0.97), xycoords=("data", "axes fraction"),
                        fontsize=5.6, ha="right", va="top", rotation=90, color=fs.MUTED)
        ax.set_xlim(grid[0], grid[-1]); ax.set_yticks([])
        ax.set_xlabel(LABEL[mark]); ax.set_ylabel("posterior density")
        ax.legend(handles=[Line2D([], [], color=C12, lw=1.5, label=f"{arm}: shared width free"),
                           Line2D([], [], color=C11, lw=1.4, label="11D: width fixed"),
                           Line2D([], [], color=fs.TRUTH, lw=0.8, ls=(0, (3, 2)), label="planted")],
                  fontsize=5.6, frameon=True, framealpha=1.0, edgecolor="none", loc="upper left")
        for a in axs[r]:
            a.xaxis.set_major_locator(MaxNLocator(5))
    FIGS.mkdir(exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(FIGS / f"fig_12.{ext}", **kw)
        print(f"    wrote figs/fig_12.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
