#!/usr/bin/env python
"""figs/corepop_population.{pdf,png}: the core-population draw against core's density.

    python make_figure.py        (after sampler_check.py and detected_gof.py; deterministic)

Rows: the two host classes of the host-dependent set (GAL-hosted: mu_G 35, mu_chi 0;
AGN-hosted: mu_G 40, mu_chi 0.10).  Columns: m1, q, m2, chi_eff (source frame).
In each panel
  grey fill      4e6 proposals drawn by the generator's core route (sampler_check.py)
  black line     core's population density (core_eval.py tables)
  dashed grey    the legacy draw (q and m2 only: where the two codes differ)
  coloured step  the detected events of the five corepop host-dependent sets, pooled
  coloured line  core's density times the detection probability (detected_gof.py)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CP = Path(__file__).resolve().parents[1]
# reference categorical palette, slots 1 and 2 (dataviz skill references/palette.md)
COL = {"gal": "#2a78d6", "agn": "#eb6834"}
INK, INK2, FILL = "#222222", "#6b6b6b", "#d9d9d6"
ROW = {"gal": r"GAL-hosted   ($\mu_G = 35$, $\mu_\chi = 0$)",
       "agn": r"AGN-hosted   ($\mu_G = 40$, $\mu_\chi = 0.10$)"}
COLS = [("m1", "m1src", r"$m_1$ [$M_\odot$]", np.linspace(5, 80, 31)),
        ("q", "q", r"$q = m_2/m_1$", np.linspace(0.05, 1, 20)),
        ("m2", "m2src", r"$m_2$ [$M_\odot$]", np.linspace(0, 60, 31)),
        ("chi", "chieff", r"$\chi_{\rm eff}$", np.linspace(-0.45, 0.6, 22))]


def main():
    P = np.load(CP / "cache" / "figure_proposals.npz")
    D = np.load(CP / "cache" / "figure_detected.npz")
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK,
                         "legend.frameon": False})
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.2))
    for r, c in enumerate(("gal", "agn")):
        for j, (k, dk, lab, bins) in enumerate(COLS):
            ax = axes[r, j]
            e = P[f"bins_{k}"]
            h = P[f"core_{c}_{k}"].astype(float)
            ax.stairs(h / (h.sum() * np.diff(e)), e, fill=True, color=FILL, lw=0,
                      label=r"core route, $4\times10^6$ proposals")
            ax.plot(P[f"model_{c}_{k}_grid"], P[f"model_{c}_{k}_dens"], color=INK, lw=1.0,
                    label="core density")
            if k in ("q", "m2"):
                hl = P[f"legacy_{c}_{k}"].astype(float)
                ax.stairs(hl / (hl.sum() * np.diff(e)), e, color=INK2, lw=0.9, ls=(0, (3, 2)),
                          label="legacy draw")
            x = D[f"det_{c}_{dk}"]
            hd, _ = np.histogram(x, bins=bins)
            ax.stairs(hd / (x.size * np.diff(bins)), bins, color=COL[c], lw=1.4,
                      label=f"detected, 5 sets ({x.size} events)")
            if k == "m2":
                # the prediction is cell mass binned at 0.01 Msun (exact in its cumulative,
                # which the KS uses); for display it is summed into 0.5 Msun bins
                me = D["pred_m2_edges"][::50]
                mass = D[f"pred_{c}_m2_mass"].reshape(-1, 50).sum(axis=1)
                ax.plot(0.5 * (me[1:] + me[:-1]), mass / np.diff(me), color=COL[c], lw=1.0,
                        alpha=0.9, label="core density x detection")
                ax.axvline(5.0, color=INK2, lw=0.6, ls=":")
                ax.text(5.4, 0.97, r"$m_{\min} = 5$", transform=ax.get_xaxis_transform(),
                        va="top", fontsize=7.5, color=INK2)
            elif k == "chi":
                # chi_eff does not enter detection: the detected density IS core's
                ax.plot(P[f"model_{c}_chi_grid"], P[f"model_{c}_chi_dens"], color=COL[c],
                        lw=1.0, alpha=0.9, label="core density x detection")
            else:
                ax.plot(D[f"pred_{c}_{k}_grid"], D[f"pred_{c}_{k}_dens"], color=COL[c], lw=1.0,
                        alpha=0.9, label="core density x detection")
            ax.set_xlim(bins[0], bins[-1])
            ax.set_ylim(bottom=0)
            ax.set_xlabel(lab)
            ax.tick_params(labelsize=7.5, length=2.5)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if j == 0:
                ax.set_ylabel("density")
                ax.set_title(ROW[c], loc="left", fontsize=9.5, color=INK)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    n = {c: D[f"det_{c}_m1src"].size for c in ("gal", "agn")}
    handles = [Patch(color=FILL, label=r"core route, $4\times10^6$ proposals"),
               Line2D([], [], color=INK, lw=1.0, label="core population density"),
               Line2D([], [], color=INK2, lw=0.9, ls=(0, (3, 2)), label="legacy draw (q, m2)"),
               Line2D([], [], color="none", label=" "),          # column spacer
               Line2D([], [], color=COL["gal"], lw=1.4, drawstyle="steps-mid",
                      label=f"detected GAL-hosted, 5 sets ({n['gal']})"),
               Line2D([], [], color=COL["gal"], lw=1.0, label="GAL: core density x detection"),
               Line2D([], [], color=COL["agn"], lw=1.4, drawstyle="steps-mid",
                      label=f"detected AGN-hosted, 5 sets ({n['agn']})"),
               Line2D([], [], color=COL["agn"], lw=1.0, label="AGN: core density x detection")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, handlelength=2.0,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(CP / "figs" / f"corepop_population.{ext}", dpi=170)
    print("wrote", CP / "figs" / "corepop_population.pdf", CP / "figs" / "corepop_population.png")


if __name__ == "__main__":
    main()
