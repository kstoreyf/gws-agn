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
    save(fig, f"fig_{arm}")


def main():
    fig_sector("11A", "mu_G", "dmu_G", "dmu_G", "marginal/dmu_G", "dmu_G_grid")
    fig_sector("11B", "mu_chi", "dmu_chi", "dmu_chi", "marginal/dmu_chi", "dmu_chi_grid")
    print("\ndone.")


if __name__ == "__main__":
    main()
