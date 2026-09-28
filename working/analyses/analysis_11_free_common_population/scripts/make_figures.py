#!/usr/bin/env python
"""Analysis 11 figures, deterministic, from results/ only.

    fig_11A   (mu_G, dmu_G) and (f, dmu_G) 90% HPD regions; p(dmu_G), Analysis 11A
              (reference mass scale free) against Analysis 10's A10-J (reference
              pinned at 35 Msun, spin offset free).
    fig_11B   the same for the spin sector: (mu_chi, dmu_chi), (f, dmu_chi), p(dmu_chi).

Every drawn median / 90% end is diffed against its JSON (``check``).  All plotted
intervals and regions are 90%; 68% numbers are printed only.  Colours from
../../paper/scripts/figstyle.py: Analysis 11 = blue, Analysis 10 = aqua, planted
values in ink.
"""
import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
A11 = HERE.parent
RESULTS, FIGS = A11 / "results", A11 / "figs"
A10_RESULTS = A11.parent / "analysis_10_mass_spin_marked_multitracer" / "results"
sys.path.insert(0, str(A11.parent.parent / "paper" / "scripts"))
import figstyle as fs  # noqa: E402

fs.use()
C11, C10 = fs.C["blue"], fs.C["aqua"]
TRUTH = {"mu_G": 35.0, "dmu_G": 5.0, "mu_chi": 0.0, "dmu_chi": 0.10, "f_agn": 0.30}
LABEL = {"mu_G": r"$\mu_{\rm G}$  [$M_\odot$]", "dmu_G": r"$\Delta\mu_{\rm G}$  [$M_\odot$]",
         "mu_chi": r"$\mu_\chi$", "dmu_chi": r"$\Delta\mu_\chi$", "f_agn": r"$f_{\rm AGN}$"}


def summarise(x, p):
    p = np.asarray(p, float)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
    norm = cdf[-1]
    q = lambda t: float(np.interp(t, cdf / norm, x))
    return {"median": q(0.5), "ci68": [q(0.16), q(0.84)], "ci90": [q(0.05), q(0.95)],
            "p": p / norm}


def check(label, got, want, tol=1e-9):
    d = abs(got - want)
    print(f"      {label:<28s} drawn {got:.10f}  json {want:.10f}  |d| {d:.2e}  "
          f"{'OK' if d <= tol else 'MISMATCH'}")
    if d > tol:
        raise SystemExit(f"[fatal] {label}: drawn value does not match the JSON")


def hpd(ax, x, y, P, color, fill=True):
    _, l90 = fs.hpd_levels_2d(x, y, P)
    if fill:
        ax.contourf(x, y, P.T, levels=[l90, P.max() * 1.01], colors=[color], alpha=0.22)
    ax.contour(x, y, P.T, levels=[l90], colors=color, linewidths=1.3)


def save(fig, name):
    FIGS.mkdir(exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(FIGS / f"{name}.{ext}", **kw)
        print(f"    wrote figs/{name}.{ext}")
    plt.close(fig)


def fig_sector(arm, mu, dmu, a10_key, a10_marg_key, a10_axis_key):
    jp, hp = RESULTS / f"a11_{arm}.json", RESULTS / f"a11_{arm}.h5"
    if not (jp.exists() and hp.exists()):
        print(f"[skip] {arm}: results not assembled yet")
        return
    js = json.loads(jp.read_text())
    print(f"\n[{arm}]")
    with h5py.File(hp, "r") as h:
        f, xm, xd = h["axis/f_agn"][:], h[f"axis/{mu}"][:], h[f"axis/{dmu}"][:]
        Pmd = h[f"marginal_2d/{mu}__{dmu}"][:]
        Pfd = h[f"marginal_2d/f_agn__{dmu}"][:]
        md = h[f"marginal/{dmu}"][:]
    with h5py.File(A10_RESULTS / "a10_arm_J.h5", "r") as h:
        x10 = h[a10_axis_key][:]
        m10 = h[a10_marg_key][:]
    j10 = json.loads((A10_RESULTS / "a10_arm_J.json").read_text())

    fig, axs = plt.subplots(1, 3, figsize=(fs.TWOCOL, 2.45))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.2, top=0.93, wspace=0.42)
    hpd(axs[0], xm, xd, Pmd, C11)
    axs[0].axvline(TRUTH[mu], color=fs.TRUTH, lw=0.8, ls=(0, (3, 2)), alpha=0.75)
    axs[0].axhline(TRUTH[dmu], color=fs.TRUTH, lw=0.8, ls=(0, (3, 2)), alpha=0.75)
    axs[0].axhline(0.0, color=fs.MUTED, lw=0.7, alpha=0.8)
    axs[0].set_xlabel(LABEL[mu]); axs[0].set_ylabel(LABEL[dmu])
    axs[0].annotate(rf"$\rho$ = {js['correlations'][f'{mu}|{dmu}']:+.2f}", (0.05, 0.92),
                    xycoords="axes fraction", fontsize=6.5, va="top")
    hpd(axs[1], f, xd, Pfd, C11)
    axs[1].axvline(TRUTH["f_agn"], color=fs.TRUTH, lw=0.8, ls=(0, (3, 2)), alpha=0.75)
    axs[1].axhline(TRUTH[dmu], color=fs.TRUTH, lw=0.8, ls=(0, (3, 2)), alpha=0.75)
    axs[1].axhline(0.0, color=fs.MUTED, lw=0.7, alpha=0.8)
    axs[1].set_xlabel(LABEL["f_agn"]); axs[1].set_ylabel(LABEL[dmu])
    axs[1].annotate(rf"$\rho$ = {js['correlations'][f'f_agn|{dmu}']:+.2f}", (0.05, 0.92),
                    xycoords="axes fraction", fontsize=6.5, va="top")
    # zoom both planes onto the 90% region with margin
    lo, hi = js[dmu]["ci90"]
    w = hi - lo
    for ax in axs[:2]:
        ax.set_ylim(min(lo - 1.2 * w, -0.1 * w), hi + 1.2 * w)
    l0, h0 = js[mu]["ci90"]; ww = h0 - l0
    axs[0].set_xlim(l0 - 1.2 * ww, h0 + 1.2 * ww)
    lf, hf = js["f_agn"]["ci90"]; wf = hf - lf
    axs[1].set_xlim(max(0.0, lf - 1.2 * wf), hf + 1.2 * wf)

    s11 = summarise(xd, md)
    check(f"A11 {dmu} median", s11["median"], js[dmu]["median"])
    check(f"A11 {dmu} 90% low", s11["ci90"][0], js[dmu]["ci90"][0])
    check(f"A11 {dmu} 90% high", s11["ci90"][1], js[dmu]["ci90"][1])
    s10 = summarise(x10, m10)
    print(f"      A10-J {dmu}: median {s10['median']:.4f} 90% {np.round(s10['ci90'], 4).tolist()} "
          f"(json median {j10[a10_key]['median']:.4f})")
    print(f"      A11 {dmu} 68% {np.round(s11['ci68'], 4).tolist()}  A10-J 68% "
          f"{np.round(s10['ci68'], 4).tolist()} (printed only)")
    ax = axs[2]
    ax.plot(x10, s10["p"], color=C10, lw=1.4)
    ax.plot(xd, s11["p"], color=C11, lw=1.5)
    ymax = max(s10["p"].max(), s11["p"].max())
    for k, (c, s) in enumerate(((C11, s11), (C10, s10))):
        y = -0.07 * ymax * (k + 1)
        ax.plot(s["ci90"], [y, y], color=c, lw=2.4, solid_capstyle="butt", clip_on=False)
        ax.plot([s["median"]], [y], marker="|", color=c, ms=7, mew=1.6, clip_on=False)
    ax.set_ylim(-0.07 * ymax * 2.8, ymax * 1.25)
    ax.axvline(TRUTH[dmu], color=fs.TRUTH, lw=0.8, ls=(0, (3, 2)), alpha=0.75)
    ax.set_xlim(axs[0].get_ylim())
    ax.set_xlabel(LABEL[dmu]); ax.set_ylabel("posterior density")
    ax.legend(handles=[Line2D([], [], color=C11, lw=1.5, label="A11: reference free"),
                       Line2D([], [], color=C10, lw=1.4, label="A10-J: reference pinned"),
                       Line2D([], [], color=fs.TRUTH, lw=0.8, ls=(0, (3, 2)), label="planted")],
              fontsize=5.6, frameon=False, loc="upper left")
    from matplotlib.ticker import MaxNLocator
    for a in axs:
        a.xaxis.set_major_locator(MaxNLocator(5))
    save(fig, f"fig_{arm}")



C11C = fs.C["orange"]


def _kde1(x, grid):
    from scipy.stats import gaussian_kde
    k = gaussian_kde(x)
    p = k(grid)
    return p / np.trapz(p, grid)


def _kde2(x, y, gx, gy):
    from scipy.stats import gaussian_kde
    k = gaussian_kde(np.vstack([x, y]))
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    return k(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)


def fig_11C():
    jp, npz = RESULTS / "a11_11C.json", RESULTS / "a11_11C.npz"
    if not jp.exists():
        print("[skip] 11C: not merged yet")
        return
    js = json.loads(jp.read_text())
    S = np.load(npz)["samples"]
    names = list(js["names"])
    col = {n: S[:, names.index(n)] for n in names}
    print("\n[11C]")
    for n in ("dmu_G", "dmu_chi", "f_agn"):
        q = np.quantile(col[n], [0.05, 0.5, 0.95])
        check(f"11C {n} median", float(q[1]), js["summary"][n]["median"])
        check(f"11C {n} 90% low", float(q[0]), js["summary"][n]["ci90"][0])
        check(f"11C {n} 90% high", float(q[2]), js["summary"][n]["ci90"][1])
    a10 = h5py.File(A10_RESULTS / "a10_arm_J.h5", "r")
    gA = h5py.File(RESULTS / "a11_11A.h5", "r")
    gB = h5py.File(RESULTS / "a11_11B.h5", "r")
    fig, axs = plt.subplots(2, 3, figsize=(fs.TWOCOL, 4.6))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.1, top=0.95, wspace=0.42, hspace=0.45)
    rows = (("dmu_G", a10["dmu_G_grid"][:], a10["marginal/dmu_G"][:], gA, "11A"),
            ("dmu_chi", a10["dmu_chi_grid"][:], a10["marginal/dmu_chi"][:], gB, "11B"),
            ("f_agn", a10["f_grid"][:], a10["marginal/f"][:], None, None))
    for ax, (n, x10, m10, g, gname) in zip(axs[0], rows):
        lo, hi = np.quantile(col[n], [0.0005, 0.9995])
        w = hi - lo
        grid = np.linspace(lo - 0.3 * w, hi + 0.3 * w, 400)
        s10 = summarise(x10, m10)
        ax.plot(x10, s10["p"], color=C10, lw=1.3, label="A10-J: reference pinned")
        strips = [(C10, s10)]
        if g is not None:
            sg = summarise(g[f"axis/{n}"][:], g[f"marginal/{n}"][:])
            ax.plot(g[f"axis/{n}"][:], sg["p"], color=C11, lw=1.3,
                    label="11A / 11B: that sector's reference free")
            strips.append((C11, sg))
        p = _kde1(col[n], grid)
        ax.plot(grid, p, color=C11C, lw=1.5, label="11C: both references free")
        s11c = {"median": js["summary"][n]["median"], "ci90": js["summary"][n]["ci90"]}
        strips.append((C11C, s11c))
        ymax = max(p.max(), s10["p"].max())
        for k, (c, s_) in enumerate(strips):
            y = -0.07 * ymax * (k + 1)
            ax.plot(s_["ci90"], [y, y], color=c, lw=2.2, solid_capstyle="butt", clip_on=False)
            ax.plot([s_["median"]], [y], marker="|", color=c, ms=6.5, mew=1.5, clip_on=False)
        ax.set_ylim(-0.07 * ymax * (len(strips) + 0.8), ymax * 1.3)
        ax.axvline(TRUTH[n], color=fs.TRUTH, lw=0.8, ls=(0, (3, 2)), alpha=0.75)
        ax.set_xlim(grid[0], grid[-1])
        ax.set_xlabel(LABEL[n]); ax.set_yticks([])
    axs[0][0].legend(fontsize=5.2, frameon=False, loc="upper left")
    pairs = (("mu_G", "dmu_G", gA, "mu_G__dmu_G"), ("mu_chi", "dmu_chi", gB, "mu_chi__dmu_chi"),
             ("dmu_G", "dmu_chi", None, None))
    for ax, (a, b, g, key) in zip(axs[1], pairs):
        la, ha = np.quantile(col[a], [0.001, 0.999]); lb, hb = np.quantile(col[b], [0.001, 0.999])
        ga = np.linspace(la - 0.3 * (ha - la), ha + 0.3 * (ha - la), 120)
        gb = np.linspace(lb - 0.3 * (hb - lb), hb + 0.3 * (hb - lb), 120)
        if g is not None:
            hpd(ax, g[f"axis/{a}"][:], g[f"axis/{b}"][:], g[f"marginal_2d/{key}"][:], C11, fill=False)
        hpd(ax, ga, gb, _kde2(col[a], col[b], ga, gb), C11C)
        ax.axvline(TRUTH[a], color=fs.TRUTH, lw=0.8, ls=(0, (3, 2)), alpha=0.75)
        ax.axhline(TRUTH[b], color=fs.TRUTH, lw=0.8, ls=(0, (3, 2)), alpha=0.75)
        ax.set_xlim(ga[0], ga[-1]); ax.set_ylim(gb[0], gb[-1])
        ax.set_xlabel(LABEL[a]); ax.set_ylabel(LABEL[b])
        r = js["summary"]["correlations"].get(f"{a}|{b}")
        ax.annotate(rf"11C $\rho$ = {r:+.2f}", (0.05, 0.92), xycoords="axes fraction",
                    fontsize=6.2, va="top")
    from matplotlib.ticker import MaxNLocator
    for a in axs.ravel():
        a.xaxis.set_major_locator(MaxNLocator(5))
    save(fig, "fig_11C")

def main():
    fig_sector("11A", "mu_G", "dmu_G", "dmu_G", "marginal/dmu_G", "dmu_G_grid")
    fig_sector("11B", "mu_chi", "dmu_chi", "dmu_chi", "marginal/dmu_chi", "dmu_chi_grid")
    fig_11C()
    print("\ndone.")


if __name__ == "__main__":
    main()
