#!/usr/bin/env python3
"""Figures for the score-residual attribution -> figs/attr_attribution.{png,pdf}."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIGS = ROOT / "figs"
FIGS.mkdir(exist_ok=True)

COL = {"gal": "#1f77b4", "agn": "#d62728"}


def main():
    terms = {t: json.load(open(RES / f"attr_terms_{t}_s100.json")) for t in ("gal", "agn")}
    arms = {t: json.load(open(RES / f"attr_mass_pe_{t}_s100.json")) for t in ("gal", "agn")}
    npz = {t: np.load(RES / f"attr_mass_pe_{t}_s100.npz") for t in ("gal", "agn")}

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    # -- (a) term decomposition -------------------------------------------------
    ax = axes[0]
    keys = ["mass", "rate", "pz", "jac"]
    lbl = [r"$p_{\rm pop}$: mass" + "\n" + r"$m_{1,\rm src}=m_{1,\rm det}/(1+z)$",
           r"$p_{\rm pop}$: rate" + "\n" + r"$(1+z)^{\gamma-1}$",
           r"catalog $p_z(z|{\rm pix})$", "Jacobian"]
    x = np.arange(len(keys))
    for i, t in enumerate(("gal", "agn")):
        v = [arms[t]["r_table"]["none"][k] for k in keys]
        ax.bar(x + (i - 0.5) * 0.38, np.array(v) * 1e3, width=0.36,
               color=COL[t], label=f"matched {t.upper()}"
                                   f"  (r = {terms[t]['r_total']*1e3:+.2f})")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(lbl, fontsize=8)
    ax.set_ylabel(r"$r_{\rm term}\ \times 10^{3}$  (per event)")
    ax.set_title("(a) where the score residual lives", fontsize=11)
    ax.legend(fontsize=8, frameon=False)

    # -- (b) PE arms -------------------------------------------------------------
    ax = axes[1]
    order = ["none", "m1obs", "m1", "m1m2"]
    names = ["stored PE\n(mock)", r"width from $obs$" + "\n(PR#335 style)",
             r"exact $p(m_1|obs)$", r"exact $p(m_1,m_2|obs)$"]
    xx = np.arange(len(order))
    for i, t in enumerate(("gal", "agn")):
        y = [arms[t]["r_table"][a]["CmA_mass"] * 1e3 for a in order]
        e = [arms[t]["r_table"][a]["CmA_mass_sem"] * 1e3 for a in order]
        ax.errorbar(xx + (i - 0.5) * 0.12, y, yerr=e, fmt="o-", color=COL[t],
                    capsize=3, label=f"matched {t.upper()}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xx); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel(r"$(C-A)_{\rm mass}\ \times 10^{3}$")
    ax.set_title("(b) repairing the mass measurement model", fontsize=11)
    ax.legend(fontsize=8, frameon=False)

    # -- (c) posterior-mean mass bias vs true mass -------------------------------
    ax = axes[2]
    for t in ("gal", "agn"):
        d = npz[t]
        m = d["true_m1src"]
        for arm, ls, lab in (("none", "-", "stored"), ("m1m2", "--", "exact")):
            dm = d[f"ev_m1srcbar_{arm}"] - m
            qs = np.quantile(m, np.linspace(0, 1, 7))
            c, y, e = [], [], []
            for i in range(6):
                s = (m >= qs[i]) & (m < qs[i + 1] if i < 5 else m <= qs[i + 1])
                if s.sum() < 10:
                    continue
                c.append(m[s].mean())
                y.append(100 * dm[s].mean() / m[s].mean())
                e.append(100 * dm[s].std(ddof=1) / np.sqrt(s.sum()) / m[s].mean())
            ax.errorbar(c, y, yerr=e, fmt="o" + ls, color=COL[t], alpha=1.0 if arm == "none" else 0.5,
                        capsize=2, label=f"{t.upper()} {lab}")
    ax.axhline(0, color="k", lw=0.8)
    txt = []
    for t in ("gal", "agn"):
        d = npz[t]; m = d["true_m1src"]; n = m.size
        for arm, lab in (("none", "stored"), ("m1m2", "exact")):
            dm = d[f"ev_m1srcbar_{arm}"] - m
            txt.append(f"{t.upper()} {lab}: ensemble mean "
                       f"{100*dm.mean()/m.mean():+.2f}% "
                       f"({abs(dm.mean())/(dm.std(ddof=1)/np.sqrt(n)):.1f}$\\sigma$)")
    ax.text(0.02, 0.03, "\n".join(txt), transform=ax.transAxes, fontsize=7,
            va="bottom", ha="left")
    ax.set_xlabel(r"true $m_{1,\rm src}\ [M_\odot]$")
    ax.set_ylabel(r"$\langle E_{\rm post}[m_{1,\rm src}] - m_{1,\rm src}^{\rm true}\rangle$  [%]")
    ax.set_title("(c) per-event shrinkage; the ENSEMBLE mean must vanish and does not",
                 fontsize=9.5)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper right")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"attr_attribution.{ext}", dpi=160, bbox_inches="tight")
    print(f"wrote {FIGS/'attr_attribution.png'}")


if __name__ == "__main__":
    main()
