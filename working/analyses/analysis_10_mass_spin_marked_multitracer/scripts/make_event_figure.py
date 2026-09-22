#!/usr/bin/env python3
"""Analysis 10: the two marks, event by event.

    python scripts/make_event_figure.py [--tag prov_f0p350_mu0p1075_G5]

Deterministic and read-only: the only inputs are
``diagnostics/a10_event_decomposition_<tag>.h5`` and the matching ``.json``,
both written by ``scripts/a10_event_decomposition.py``.  Nothing is recomputed
from the likelihood, and every number drawn is printed to stdout beside the
value the JSON records, so the panels can be diffed without opening either file.

  LEFT   the routing plane of the two marks: what the mass mark does to each
         event's AGN-branch probability against what the spin mark does to the
         same event, both measured against the same mark-free spatial model.  A
         point on the diagonal is an event the two marks move together; a point
         on an axis is an event only one mark moves; a point in an off-diagonal
         quadrant is an event the two marks move in OPPOSITE directions.  Colour
         is the event's branch probability under the full two-mark model, an
         inference quantity, not the true host label.

  RIGHT  the same events in evidence rather than probability: log BF_i,mass
         against log BF_i,spin, each the AGN-over-GAL log ratio the mark adds on
         top of the spatial factor.  This is the plane in which the two marks
         are additive or not; the interaction term the JSON reports is the
         departure of log BF_joint from the sum of these two coordinates.

House conventions, from ../../../paper/scripts/figstyle.py: that module is the
visual system and nothing here invents a colour.  The sequential ramp is the
house single-hue blue, used for the one ordered magnitude on the page; the
categorical slots are spent on the mass mark (orange) and the spin mark (aqua)
where those two are named.  No interval is drawn on either panel; every interval
this analysis quotes elsewhere is 90 %.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
A10 = HERE.parent
DIAG = A10 / "diagnostics"
FIGS = A10 / "figs"
PAPER_SCRIPTS = A10.parent.parent / "paper" / "scripts"

sys.path.insert(0, str(PAPER_SCRIPTS))
import figstyle as fs  # noqa: E402

DEFAULT_TAG = "prov_f0p350_mu0p1075_G5"
C_MASS, C_SPIN = fs.C["orange"], fs.C["aqua"]


def check(label, drawn, recorded, tol=1e-9):
    d = abs(float(drawn) - float(recorded))
    ok = "OK" if d <= tol else "MISMATCH"
    print(f"    {label:<52s} drawn {float(drawn): .6f}   json {float(recorded): .6f}"
          f"   |d| {d:.2e} {ok}")
    return d <= tol


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        out = FIGS / f"{name}.{ext}"
        fig.savefig(out, **kw)
        print(f"    wrote {out}")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=DEFAULT_TAG)
    args = ap.parse_args(argv)
    tag = args.tag
    h5_path = DIAG / f"a10_event_decomposition_{tag}.h5"
    json_path = DIAG / f"a10_event_decomposition_{tag}.json"
    name = f"fig_event_mass_vs_spin_{tag}"

    fs.use()
    print(f"reading {h5_path}")
    with h5py.File(h5_path, "r") as h:
        dP_spin = h["delta_P_spin"][:]
        dP_mass = h["delta_P_mass"][:]
        dP_joint = h["delta_P_joint"][:]
        P_joint = h["P_AGN_joint"][:]
        P_spatial = h["P_AGN_spatial"][:]
        bf_spin = h["log_BF_spin"][:]
        bf_mass = h["log_BF_mass"][:]
        bf_joint = h["log_BF_joint"][:]
        inter = h["interaction"][:]
        host = h["true_host_type"][:].astype(int)
        f_agn = float(h.attrs["f_agn"])
        dmu_chi = float(h.attrs["dmu_chi"])
        dmu_G = float(h.attrs["dmu_G"])
        H0 = float(h.attrs["H0"])
    print(f"reading {json_path}")
    S = json.loads(json_path.read_text())
    ST, CO, IN = S["statistics"], S["complementarity"], S["interaction"]

    print("\n  panel A -- the routing plane of the two marks")
    check("median |dP^spin|", np.median(np.abs(dP_spin)), ST["spin"]["median_abs"])
    check("median |dP^mass|", np.median(np.abs(dP_mass)), ST["mass"]["median_abs"])
    check("rms |dP^spin|", np.sqrt(np.mean(dP_spin ** 2)), ST["spin"]["rms_abs"])
    check("rms |dP^mass|", np.sqrt(np.mean(dP_mass ** 2)), ST["mass"]["rms_abs"])
    check("N(|dP^spin| > 0.1)", (np.abs(dP_spin) > 0.1).sum(),
          ST["spin"]["n_abs_gt"]["0.1"], 0)
    check("N(|dP^mass| > 0.1)", (np.abs(dP_mass) > 0.1).sum(),
          ST["mass"]["n_abs_gt"]["0.1"], 0)
    check("Pearson r(dP^mass, dP^spin)",
          np.corrcoef(dP_mass, dP_spin)[0, 1], CO["dP_mass_vs_dP_spin"]["pearson_r"],
          1e-9)
    both = (np.abs(dP_mass) > 0.05) & (np.abs(dP_spin) > 0.05)
    q = CO["quadrants_both_abs_gt_0p05"]
    check("N both |dP| > 0.05", both.sum(), q["n_events"], 0)
    check("  of which same sign",
          (both & (np.sign(dP_mass) * np.sign(dP_spin) > 0)).sum(), q["same_sign"], 0)
    a, b = np.abs(dP_mass) > 0.1, np.abs(dP_spin) > 0.1
    check("Jaccard of the two |dP| > 0.1 sets",
          (a & b).sum() / max((a | b).sum(), 1), CO["sets_abs_gt_0p1"]["jaccard"])
    check("min P(AGN | full)", P_joint.min(), ST["joint"]["P_range"][0])
    check("max P(AGN | full)", P_joint.max(), ST["joint"]["P_range"][1])

    print("\n  panel B -- the evidence plane")
    check("Pearson r(logBF_mass, logBF_spin)", np.corrcoef(bf_mass, bf_spin)[0, 1],
          CO["logBF_mass_vs_logBF_spin"]["pearson_r"])
    check("median |logBF_spin|", np.median(np.abs(bf_spin)),
          IN["scale_comparison"]["median_abs_logBF_spin"])
    check("median |logBF_mass|", np.median(np.abs(bf_mass)),
          IN["scale_comparison"]["median_abs_logBF_mass"])
    check("median |logBF_joint|", np.median(np.abs(bf_joint)),
          IN["scale_comparison"]["median_abs_logBF_joint"])
    check("median |I_i|", np.median(np.abs(inter)), IN["median_abs"])
    check("max |I_i|", np.abs(inter).max(), IN["max_abs"])

    rho_P = CO["dP_mass_vs_dP_spin"]
    rho_B = CO["logBF_mass_vs_logBF_spin"]
    jac = CO["sets_abs_gt_0p1"]
    reg = CO["regression_dP_joint_on_dP_spin_and_dP_mass"]
    n_joint_only = CO["crossings_0p5"]["n_crossing_under_joint_but_neither_single_mark"]

    # ------------------------------------------------------------------ draw --
    cmap = LinearSegmentedColormap.from_list("house_blue", fs.RAMP)
    vlo = float(np.floor(P_joint.min() * 20.0) / 20.0)
    vhi = float(np.ceil(P_joint.max() * 20.0) / 20.0)
    print(f"\n    colour scale: P(AGN | full) in [{vlo:.2f}, {vhi:.2f}] "
          f"(data range {P_joint.min():.4f} to {P_joint.max():.4f})")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(fs.TWOCOL, 2.9))

    # --- LEFT: the two marks' routing, coloured by the full-model probability --
    lim = 1.05 * max(np.abs(dP_spin).max(), np.abs(dP_mass).max())
    ax.axhline(0.0, color=fs.AXIS, lw=0.8, zorder=1)
    ax.axvline(0.0, color=fs.AXIS, lw=0.8, zorder=1)
    ax.plot([-lim, lim], [-lim, lim], color=fs.GRID, lw=0.8, zorder=1)
    sc = ax.scatter(dP_spin, dP_mass, c=P_joint, cmap=cmap, vmin=vlo, vmax=vhi,
                    s=6.0, lw=0, alpha=0.85, zorder=3)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\Delta P_i^{\rm spin}$")
    ax.set_ylabel(r"$\Delta P_i^{\rm mass}$")
    ax.set_title("what each mark does to the branch probability")
    ax.annotate(
        f"Pearson $r$ {rho_P['pearson_r']:+.3f}\n"
        f"Spearman $\\rho$ {rho_P['spearman_rho']:+.3f}\n"
        f"$|\\Delta P|>0.1$: {jac['n_spin']} spin, {jac['n_mass']} mass,\n"
        f"{jac['n_intersection']} both (Jaccard {jac['jaccard']:.2f})",
        (0.03, 0.97), xycoords="axes fraction", ha="left", va="top",
        fontsize=6.4, color=fs.INK2, zorder=6)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label(r"$P_i(\mathrm{AGN}\,|\,\mathrm{spatial+mass+spin})$", fontsize=7.0)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_visible(False)

    # --- RIGHT: the evidence plane -------------------------------------------- #
    bx.axhline(0.0, color=fs.AXIS, lw=0.8, zorder=1)
    bx.axvline(0.0, color=fs.AXIS, lw=0.8, zorder=1)
    bx.scatter(bf_spin, bf_mass, s=6.0, lw=0, alpha=0.6, color=fs.MUTED, zorder=3)
    bx.set_xlabel(r"$\log \mathrm{BF}_{i,\rm spin}$")
    bx.set_ylabel(r"$\log \mathrm{BF}_{i,\rm mass}$")
    bx.set_title("the same events in evidence")
    bx.annotate(
        f"Pearson $r$ {rho_B['pearson_r']:+.3f}\n"
        f"Spearman $\\rho$ {rho_B['spearman_rho']:+.3f}\n"
        f"median $|\\log\\mathrm{{BF}}|$: spin "
        f"{IN['scale_comparison']['median_abs_logBF_spin']:.3f}, mass "
        f"{IN['scale_comparison']['median_abs_logBF_mass']:.3f}\n"
        f"median $|I_i|$ {IN['median_abs']:.4f}, max {IN['max_abs']:.3f}",
        (0.03, 0.97), xycoords="axes fraction", ha="left", va="top",
        fontsize=6.4, color=fs.INK2, zorder=6)
    bx.legend(handles=[
        Line2D([], [], ls="none", marker="o", ms=3.2, color=fs.MUTED,
               label="one event")], loc="lower right", fontsize=6.6)

    fig.text(0.5, 1.04,
             rf"seed 100, $H_0 = {H0:g}$ km s$^{{-1}}$ Mpc$^{{-1}}$ (fixed), "
             rf"$f_{{\rm AGN}} = {f_agn:.3f}$, $\Delta\mu_\chi = {dmu_chi:+.4f}$, "
             rf"$\Delta\mu_{{\rm G}} = {dmu_G:+.1f}\,M_\odot$",
             ha="center", va="top", fontsize=6.8, color=fs.INK2)
    fig.tight_layout(w_pad=1.6)
    save(fig, name)

    print("\n  caption text (the figure is drawn from these facts):")
    print(
        f"    Each of the 1000 seed-100 two-mark events at the provisional point\n"
        f"    f_AGN = {f_agn:.3f}, dmu_chi = {dmu_chi:+.4f}, dmu_G = {dmu_G:+.1f} "
        f"Msun, H0 = {H0:g} fixed.\n"
        f"    Left: the change the mass mark makes to the event's AGN-branch\n"
        f"    probability against the change the spin mark makes, each measured\n"
        f"    against the same mark-free spatial model, so the axes are\n"
        f"    dP_i^mass and dP_i^spin. The grey diagonal is where the two marks\n"
        f"    agree. COLOUR is P_i(AGN | spatial+mass+spin), the event's branch\n"
        f"    probability under the full two-mark model -- an inference quantity,\n"
        f"    not the true host label, which enters nothing drawn here.\n"
        f"    The two marks correlate at Pearson {rho_P['pearson_r']:+.3f} "
        f"(Spearman {rho_P['spearman_rho']:+.3f});\n"
        f"    {jac['n_spin']} events move by more than 0.1 under the spin mark and "
        f"{jac['n_mass']} under\n"
        f"    the mass mark, {jac['n_intersection']} under both (Jaccard "
        f"{jac['jaccard']:.2f}), and {n_joint_only} events cross the 0.5\n"
        f"    decision line under the two marks together but under neither alone.\n"
        f"    Regressing dP^joint on the two gives coefficients "
        f"{reg['coef_dP_spin']:+.3f} (spin) and\n"
        f"    {reg['coef_dP_mass']:+.3f} (mass) with R2 = {reg['r_squared']:.4f}.\n"
        f"    Right: the same events in evidence, log BF_i,mass against\n"
        f"    log BF_i,spin, the AGN-over-GAL log ratio each mark adds on top of\n"
        f"    the spatial factor. Their correlation is {rho_B['pearson_r']:+.3f}; the\n"
        f"    interaction I_i = log BF_joint - log BF_spin - log BF_mass has\n"
        f"    median |I_i| = {IN['median_abs']:.4f} against median |log BF_joint| = "
        f"{IN['scale_comparison']['median_abs_logBF_joint']:.4f}.\n"
        f"    No interval is drawn; every interval this analysis quotes is 90%.")
    n_agn = int((host == 1).sum())
    print(f"\n  (the true host label is read here only to state that the file "
          f"carries {n_agn} AGN-hosted events; it is drawn nowhere)")
    print(f"\n  inputs read: {h5_path}\n               {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
