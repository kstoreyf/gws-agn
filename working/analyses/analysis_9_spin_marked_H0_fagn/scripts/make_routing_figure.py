#!/usr/bin/env python3
"""Analysis 9 mechanism figure: what the mark moves, and where the H0 information
changes.

    python scripts/make_routing_figure.py     # writes fig_mechanism_event_routing

Deterministic and read-only: the only inputs are
``diagnostics/a9_event_routing.h5`` and ``diagnostics/a9_event_routing.json``,
both written by ``scripts/a9_event_routing.py``.  Nothing is recomputed from the
likelihood, and every number drawn is printed to stdout beside the value the
JSON records, so the panels can be diffed without opening either file.

  LEFT   P_i(AGN | spatial) against P_i(AGN | spatial + mark), one point per
         event, with the two 0.5 decision lines and the identity.  A point off
         the diagonal is an event the mark re-routes; a point in an off-diagonal
         quadrant is an event that changes side of 0.5.  Points are coloured by
         the TRUE host label, which is a property of the mock and never an input
         to either probability -- it is here so the reader can see which way the
         re-routing goes.

  RIGHT  the same events, |dP_i| against dI_i = I_i^J - I_i^S, the change in the
         event's own H0 curvature -[d2 lnZ_i / dH0^2] at 69.0 on the 0.5
         stencil.  The horizontal bars are the mean dI_i within bins of |dP_i|,
         each spanning its own bin and drawn in the same units as the cloud, so
         the aggregate is read off the same axis as the events.  This is the
         link the diagnostic tests: if the mark's H0 effect lived in the routed
         events, dI_i would grow with |dP_i|.

House conventions, from ../../../paper/scripts/figstyle.py: that module is the
visual system and nothing here invents a colour.  The Analysis-8/9 arm identity
marked = blue is kept for the one mark that is a marked-minus-spatial contrast,
the binned mean dI_i; it is not spent on the true host label, which is not an
arm, so the label split uses the categorical orange against neutral grey.  No
interval is drawn on either panel; every interval this analysis quotes
elsewhere is 90 %.
"""
from __future__ import annotations

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
A9 = HERE.parent
DIAG = A9 / "diagnostics"
FIGS = A9 / "figs"
PAPER_SCRIPTS = A9.parent.parent / "paper" / "scripts"

sys.path.insert(0, str(PAPER_SCRIPTS))
import figstyle as fs  # noqa: E402

C_MARKED, C_SPATIAL = fs.C["blue"], fs.C["aqua"]   # the Analysis-8/9 arm identity
C_TRUE_AGN, C_TRUE_GAL = fs.C["orange"], fs.MUTED  # a mock label, not an arm
H5 = DIAG / "a9_event_routing.h5"
JSON = DIAG / "a9_event_routing.json"
NAME = "fig_mechanism_event_routing"


def check(label, drawn, recorded, tol=1e-9):
    d = abs(float(drawn) - float(recorded))
    print(f"    {label:<44s} drawn {float(drawn): .6f}   json {float(recorded): .6f}"
          f"   |d| {d:.2e} {'OK' if d <= tol else 'MISMATCH'}")


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        out = FIGS / f"{name}.{ext}"
        fig.savefig(out, **kw)
        print(f"    wrote {out}")
    plt.close(fig)


def main():
    fs.use()
    print(f"reading {H5}")
    with h5py.File(H5, "r") as h:
        P_s = h["P_AGN_spatial"][:]
        P_m = h["P_AGN_marked"][:]
        dP = h["delta_P"][:]
        dI = h["delta_I"][:]
        I_m = h["I_marked"][:]
        I_s = h["I_spatial"][:]
        host = h["true_host_type"][:].astype(int)
        H0_rep = float(h.attrs["H0_rep"])
        f_agn = float(h.attrs["f_agn"])
        mu_c2 = float(h.attrs["mu_chi_c2"])
    print(f"reading {JSON}")
    S = json.loads(JSON.read_text())
    R = S["routing_at_H0_rep"]
    H = S["h0_information"]

    is_agn = host == 1
    absdP = np.abs(dP)
    cross_up = (P_s <= 0.5) & (P_m > 0.5)
    cross_dn = (P_s > 0.5) & (P_m <= 0.5)

    print("\n  panel A -- the routing plane, against the recorded summary")
    check("sum P_i(AGN | spatial)", P_s.sum(), R["sum_P_spatial"], 1e-6)
    check("sum P_i(AGN | spatial + mark)", P_m.sum(), R["sum_P_marked"], 1e-6)
    check("median |dP_i|", np.median(absdP), R["abs_delta_P_median"], 1e-9)
    check("rms |dP_i|", np.sqrt(np.mean(dP ** 2)), R["abs_delta_P_rms"], 1e-9)
    check("N(|dP_i| > 0.1)", (absdP > 0.1).sum(), R["n_abs_delta_P_gt_0p1"], 0)
    check("N(|dP_i| > 0.2)", (absdP > 0.2).sum(), R["n_abs_delta_P_gt_0p2"], 0)
    check("N crossing 0.5, GAL -> AGN", cross_up.sum(),
          R["n_crossing_GAL_to_AGN"], 0)
    check("N crossing 0.5, AGN -> GAL", cross_dn.sum(),
          R["n_crossing_AGN_to_GAL"], 0)
    check("N called AGN, spatial", (P_s > 0.5).sum(), R["n_called_AGN_spatial"], 0)
    check("N called AGN, marked", (P_m > 0.5).sum(), R["n_called_AGN_marked"], 0)

    print("\n  panel B -- the H0 information, against the recorded summary")
    check("sum_i I_i^J", I_m.sum(), H["sum_I_marked"], 1e-6)
    check("sum_i I_i^S", I_s.sum(), H["sum_I_spatial"], 1e-6)
    check("sum_i dI_i", dI.sum(), H["sum_delta_I"], 1e-9)
    check("sum_i |dI_i|", np.abs(dI).sum(), H["sum_abs_delta_I"], 1e-9)
    frac01 = next(f for f in H["fractions"] if f["selector"] == "|dP_i| > 0.1")
    check("|dI| share of the |dP|>0.1 events",
          np.abs(dI[absdP > 0.1]).sum() / np.abs(dI).sum(),
          frac01["fraction_of_sum_abs_delta_I"], 1e-9)
    rho = next(c for c in H["correlations"] if c["pair"] == "|dP_i| vs |dI_i|")
    print(f"    |dP_i| vs |dI_i|: Spearman {rho['spearman_rho']:+.4f} "
          f"(p {rho['spearman_p']:.2e}), Pearson {rho['pearson_r']:+.4f}")

    # ------------------------------------------------------------------ draw --
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(fs.TWOCOL, 2.7))

    # --- LEFT: the routing plane ------------------------------------------- #
    ax.plot([0, 1], [0, 1], color=fs.AXIS, lw=0.8, zorder=1)
    ax.axhline(0.5, color=fs.TRUTH, lw=0.8, ls=(0, (3, 2)), alpha=0.7, zorder=1.5)
    ax.axvline(0.5, color=fs.TRUTH, lw=0.8, ls=(0, (3, 2)), alpha=0.7, zorder=1.5)
    ax.scatter(P_s[~is_agn], P_m[~is_agn], s=5.0, lw=0, alpha=0.55,
               color=C_TRUE_GAL, zorder=3)
    ax.scatter(P_s[is_agn], P_m[is_agn], s=5.0, lw=0, alpha=0.6,
               color=C_TRUE_AGN, zorder=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$P_i(\mathrm{AGN}\,|\,\mathrm{spatial})$")
    ax.set_ylabel(r"$P_i(\mathrm{AGN}\,|\,\mathrm{spatial}+\mathrm{mark})$")
    ax.set_title("branch probability, with and without the mark")
    ax.set_aspect("equal", adjustable="box")
    ax.annotate(
        f"{R['n_crossing_GAL_to_AGN']} cross up\n"
        f"{R['n_crossing_AGN_to_GAL']} cross down\n"
        f"median $|\\Delta P|$ {R['abs_delta_P_median']:.4f}",
        (0.04, 0.96), xycoords="axes fraction", ha="left", va="top",
        fontsize=6.6, color=fs.INK2, zorder=6)
    ax.legend(handles=[
        Line2D([], [], ls="none", marker="o", ms=3.2, color=C_TRUE_AGN,
               label="true AGN host"),
        Line2D([], [], ls="none", marker="o", ms=3.2, color=C_TRUE_GAL,
               label="true GAL host")],
        loc="lower right", fontsize=6.6)

    # --- RIGHT: routing against the change in H0 information ---------------- #
    bx.axhline(0.0, color=fs.AXIS, lw=0.8, zorder=1)
    bx.scatter(absdP, dI, s=5.0, lw=0, alpha=0.5, color=fs.MUTED, zorder=2)
    edges = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, max(0.3001, absdP.max())])
    which = np.digitize(absdP, edges[1:-1], right=False)
    xs, ys, ns, los, his = [], [], [], [], []
    for b in range(len(edges) - 1):
        m = which == b
        if not m.any():
            continue
        xs.append(0.5 * (edges[b] + edges[b + 1]))
        ys.append(float(dI[m].mean()))
        ns.append(int(m.sum()))
        los.append(float(edges[b]))
        his.append(float(edges[b + 1]))
    bx.hlines(ys, los, his, color=C_MARKED, lw=1.6, zorder=4)
    bx.plot(xs, ys, ls="none", marker="o", ms=3.4, color=C_MARKED, zorder=5)
    print("\n    binned mean dI_i against |dP_i| (the horizontal bars)")
    for lo, hi, y, n in zip(los, his, ys, ns):
        print(f"      |dP| in [{lo:6.4f}, {hi:6.4f})   N {n:>4d}   "
              f"mean dI {y:+.6f}")
    bx.set_xlabel(r"$|\Delta P_i|$")
    bx.set_ylabel(r"$\Delta I_i = I_i^{\rm marked} - I_i^{\rm spatial}$")
    bx.set_title(r"routing against the change in $H_0$ curvature")
    bx.annotate(
        f"Spearman $\\rho$ {rho['spearman_rho']:+.3f}\n"
        f"{frac01['fraction_of_sum_abs_delta_I'] * 100:.1f}% of "
        f"$\\sum_i|\\Delta I_i|$ in the\n"
        f"{frac01['n_events']} events with $|\\Delta P_i| > 0.1$",
        (0.96, 0.96), xycoords="axes fraction", ha="right", va="top",
        fontsize=6.6, color=fs.INK2, zorder=6)
    bx.legend(handles=[
        Line2D([], [], ls="none", marker="o", ms=3.2, color=fs.MUTED,
               label="one event"),
        Line2D([], [], color=C_MARKED, lw=1.6, marker="o", ms=3.4,
               label=r"mean $\Delta I_i$ per $|\Delta P_i|$ bin")],
        loc="lower right", fontsize=6.6)

    fig.text(0.5, 1.045,
             rf"seed 100, $f_{{\rm AGN}} = {f_agn:g}$, "
             rf"$\Delta\mu_\chi = {mu_c2:+.4f}$, $H_0 = {H0_rep:g}$"
             r" km s$^{-1}$ Mpc$^{-1}$",
             ha="center", va="top", fontsize=6.8, color=fs.INK2)
    fig.tight_layout(w_pad=1.6)
    save(fig, NAME)

    print("\n  caption text (the figure is drawn from these facts):")
    print(
        "    Left: each of the 1000 seed-100 events at the shared S9/J9 MAP\n"
        f"    node (f_AGN = {f_agn:g}, dmu_chi = {mu_c2:+.4f}) and "
        f"H0 = {H0_rep:g}. The abscissa is the\n"
        "    event's AGN-branch probability from the spatial factor alone, the\n"
        "    ordinate the same probability once the effective-spin mark is\n"
        "    added; the dashed lines are the 0.5 decision boundary and the grey\n"
        f"    line the identity. {R['n_crossing_GAL_to_AGN']} events cross from "
        f"GAL to AGN and\n"
        f"    {R['n_crossing_AGN_to_GAL']} the other way, while the expected AGN "
        f"count moves from\n"
        f"    {R['sum_P_spatial']:.1f} to {R['sum_P_marked']:.1f} against "
        f"{R['realised_AGN_count']} realised. Colour is the TRUE host label,\n"
        "    a property of the mock that enters neither probability; it is shown\n"
        "    only so the direction of the re-routing is visible.\n"
        "    Right: the same events, the size of that re-routing against the\n"
        "    change it makes to the event's own H0 curvature at 69.0,\n"
        "    dI_i = I_i^J - I_i^S with I_i = -d2 lnZ_i/dH0^2 on the 0.5 stencil.\n"
        f"    The Spearman correlation is {rho['spearman_rho']:+.3f}, and the "
        f"{frac01['n_events']} events with\n"
        f"    |dP_i| > 0.1 carry "
        f"{frac01['fraction_of_sum_abs_delta_I'] * 100:.1f}% of the total "
        f"|dI|.")
    print(f"\n  inputs read: {H5}\n               {JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
