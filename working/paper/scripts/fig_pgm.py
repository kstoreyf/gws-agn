"""Figure 1 -- probabilistic graphical model of the generative process (v3).

Plate notation, drawn by hand in matplotlib (no external PGM package).  The
figure is a diagram, not a data chart, so it carries no categorical palette:
node *type* is encoded by fill lightness and by stroke count, which survives
both colour-vision deficiency and greyscale printing, and the only hue on the
page labels the plates.

Node types
    observed        shaded fill, single ink ring
    latent          page-white fill, single ink ring
    deterministic   shaded fill, double ink ring
    hyperparameter  small filled ink square

Source of truth for the graph: working/data/DESIGN_PE.md (the v3 measurement
family) and working/data/seed100/META.json.  What this version says that the
previous one did not:

  * the recorded datum of an event is the OBSERVED SIGNAL-TO-NOISE RATIO,
    rho_obs = rho_opt(theta) + N(0, sigma_rho).  The optimal SNR is an exact
    function of the source parameters, rho_opt ~ (Mc_det)^(5/6)/dL, so the SNR
    *is* the distance observable and there is no separate distance measurement
    node anywhere in the graph;
  * detection is a deterministic function of that datum, D_i = [rho_obs >= 8];
  * every measurement width -- masses, mass ratio, spin, sky -- is
    a_x (8/rho_obs), a deterministic function of the same datum.  Detection and
    all the widths therefore sit DOWNSTREAM of rho_obs, not beside theta_i;
  * the catalog redshifts carry realised photo-z scatter,
    z_obs = z + N(0, sigma_z (1+z)) with sigma_z = 3e-3.  The host is drawn on
    the true redshift; the survey, and hence the likelihood, sees z_obs only.

Layout rule kept from the previous version: no arrow crosses another and no
arrow crosses a node.  That is what fixes the placement of the two long edges
(the cosmology and the population into the source parameters, routed as arcs
through the empty band below the event chain, and the source parameters into
their own noisy measurement, routed below the whole chain).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

import figstyle as fs

# ---- canvas -----------------------------------------------------------------
XMAX, YMAX = 16.6, 10.0
FIGW = fs.TWOCOL
FIGH = FIGW * YMAX / XMAX
PT_PER_UNIT = FIGW * 72.0 / XMAX

R = 0.45          # node radius, data units
RSQ = 0.16        # hyperparameter half-side
DOUBLE = 0.075    # inner-ring inset for deterministic nodes

SHADE = "#d8d7ce"     # "observed" fill: one lightness step below the page
PLATE_EDGE = fs.AXIS
PLATE_LBL = fs.C["blue"]

SUBSIZE = 7.0     # every glyph in the figure is >= 7 pt at final size
LBLSIZE = 8.0
NODESIZE = 8.5

NODES: dict[str, dict] = {}


def node(ax, key, x, y, label, kind="latent", sub=None, subpos="below",
         lab="below"):
    """Draw one graph node and remember where its boundary is."""
    NODES[key] = {"x": x, "y": y, "r": RSQ * 1.3 if kind == "hyper" else R}
    if kind == "hyper":
        ax.add_patch(Rectangle((x - RSQ, y - RSQ), 2 * RSQ, 2 * RSQ,
                               facecolor=fs.INK, edgecolor="none", zorder=3))
        off = RSQ + 0.16
        px, py, ha, va = {
            "below": (x, y - off, "center", "top"),
            "above": (x, y + off, "center", "bottom"),
            "right": (x + off, y, "left", "center"),
            "left": (x - off, y, "right", "center"),
        }[lab]
        ax.text(px, py, label, ha=ha, va=va, fontsize=LBLSIZE, color=fs.INK,
                zorder=4)
        return
    face = SHADE if kind in ("observed", "det") else "white"
    ax.add_patch(Circle((x, y), R, facecolor=face, edgecolor=fs.INK,
                        linewidth=1.0, zorder=3))
    if kind == "det":
        ax.add_patch(Circle((x, y), R - DOUBLE, facecolor="none",
                            edgecolor=fs.INK, linewidth=0.8, zorder=4))
    ax.text(x, y, label, ha="center", va="center", fontsize=NODESIZE,
            color=fs.INK, zorder=5)
    if sub:
        dy, va = (R + 0.20, "top") if subpos == "below" else (-R - 0.20, "bottom")
        ax.text(x, y - dy, sub, ha="center", va=va, fontsize=SUBSIZE,
                color=fs.INK2, zorder=4, linespacing=1.25)


def edge(ax, a, b, rad=0.0):
    """Arrow from node a to node b, trimmed to both node boundaries."""
    na, nb = NODES[a], NODES[b]
    ax.add_patch(FancyArrowPatch(
        (na["x"], na["y"]), (nb["x"], nb["y"]),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", mutation_scale=8.0,
        shrinkA=na["r"] * PT_PER_UNIT + 1.0,
        shrinkB=nb["r"] * PT_PER_UNIT + 2.5,
        linewidth=0.85, color=fs.INK2, zorder=2, joinstyle="round"))


def plate(ax, x0, y0, x1, y1, label):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0,rounding_size=0.22",
        facecolor="none", edgecolor=PLATE_EDGE, linewidth=0.9, zorder=1))
    ax.text(x1 - 0.22, y0 + 0.18, label, ha="right", va="bottom",
            fontsize=SUBSIZE + 0.2, color=PLATE_LBL, zorder=2)


def legend(ax, x, y, dy=0.62):
    """Vertical key, parked in the empty top-right corner."""
    items = [("observed", "observed"), ("latent", "latent"),
             ("det", "deterministic"), ("hyper", "hyperparameter")]
    for kind, text in items:
        if kind == "hyper":
            ax.add_patch(Rectangle((x - 0.14, y - 0.14), 0.28, 0.28,
                                   facecolor=fs.INK, edgecolor="none", zorder=3))
        else:
            face = SHADE if kind in ("observed", "det") else "white"
            ax.add_patch(Circle((x, y), 0.25, facecolor=face, edgecolor=fs.INK,
                                linewidth=1.0, zorder=3))
            if kind == "det":
                ax.add_patch(Circle((x, y), 0.20, facecolor="none",
                                    edgecolor=fs.INK, linewidth=0.8, zorder=4))
        ax.text(x + 0.45, y, text, ha="left", va="center", fontsize=SUBSIZE,
                color=fs.INK2, zorder=4)
        y -= dy


def build():
    fs.use()
    fig = plt.figure(figsize=(FIGW, FIGH))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, YMAX)
    ax.set_aspect("equal")
    ax.axis("off")

    plate(ax, 3.45, 6.80, 10.05, 9.75, r"tracers  $k = 1 \dots K$")
    plate(ax, 3.35, 0.75, 16.25, 6.55, r"events  $i = 1 \dots N_{\rm obs}$")

    # ---- hyperparameters ----------------------------------------------------
    node(ax, "Lcos", 0.95, 3.30, r"$\Lambda_{\rm cos}$", "hyper", lab="below")
    node(ax, "Lpop", 2.30, 1.05, r"$\Lambda_{\rm pop}$", "hyper", lab="below")
    node(ax, "fagn", 2.30, 4.60, r"$f_{\rm AGN}$", "hyper", lab="above")
    node(ax, "bk", 4.30, 9.20, r"$b_k$", "hyper", lab="right")
    node(ax, "nk", 4.30, 7.40, r"$\bar{n}_k$", "hyper", lab="right")
    node(ax, "sigz", 10.85, 9.20, r"$\sigma_z$", "hyper", lab="below")
    node(ax, "mlim", 10.85, 7.40, r"$m_{\rm lim}$", "hyper", lab="below")
    # labels sit ABOVE these two squares: their arrows leave downward, and a
    # label below would be struck through by its own edge.
    node(ax, "Lsnr", 9.55, 5.45, r"$\sigma_\rho,\ \rho_{\rm th}$", "hyper",
         lab="above")
    node(ax, "Lwid", 13.45, 5.45,
         r"$a_{M_c},\ a_q,\ a_\chi,\ a_\Omega$", "hyper", lab="above")

    # ---- structure ----------------------------------------------------------
    node(ax, "delta", 2.30, 8.30, r"$\delta$", "latent",
         sub="lognormal\ndensity field", subpos="above")
    node(ax, "Ck", 6.20, 8.30, r"$C_k$", "latent",
         sub="complete catalog\n$(z_j,\\ \\hat{n}_j,\\ m_j)$", subpos="above")
    node(ax, "Chk", 8.75, 8.30, r"$\hat{C}_k$", "observed",
         sub="survey catalog\n$(z^{\\rm obs}_j,\\ \\hat{n}_j)$", subpos="above")

    # ---- events: the chain --------------------------------------------------
    node(ax, "ti", 4.20, 3.10, r"$t_i$", "latent")
    node(ax, "hi", 6.05, 3.10, r"$h_i$", "latent")
    node(ax, "thi", 7.90, 3.10, r"$\theta_i$", "latent",
         sub="source\nparameters", subpos="above")
    node(ax, "rho", 9.75, 3.10, r"$\rho_i$", "det",
         sub="optimal SNR\n$\\propto M_c^{5/6} / d_L$")
    node(ax, "rhat", 11.60, 3.10, r"$\hat{\rho}_i$", "observed",
         sub="observed SNR\n$\\rho_i + \\mathcal{N}(0, \\sigma_\\rho)$")
    node(ax, "sig", 13.45, 3.10, r"$\sigma_{x,i}$", "det",
         sub="every width\n$a_x\\,(8/\\hat{\\rho}_i)$")
    node(ax, "di", 15.30, 3.10, r"$d_i$", "observed",
         sub="$\\ln \\hat{M}_c,\\ \\ln \\hat{q}$\n"
             "$\\hat{\\chi}_{\\rm eff},\\ \\hat{n}_i$")
    node(ax, "Di", 11.60, 5.45, r"$D_i$", "det",
         sub="detected: $\\hat{\\rho}_i \\geq \\rho_{\\rm th}$", subpos="above")

    # ---- edges --------------------------------------------------------------
    edge(ax, "Lcos", "delta")
    edge(ax, "delta", "Ck")
    edge(ax, "bk", "Ck")
    edge(ax, "nk", "Ck")
    edge(ax, "Ck", "Chk")
    edge(ax, "sigz", "Chk", rad=-0.12)
    edge(ax, "mlim", "Chk", rad=0.12)
    edge(ax, "Ck", "hi", rad=-0.06)

    edge(ax, "fagn", "ti")
    edge(ax, "ti", "hi")
    edge(ax, "hi", "thi")
    edge(ax, "Lcos", "thi", rad=0.36)
    edge(ax, "Lpop", "thi", rad=0.18)

    edge(ax, "thi", "rho")
    edge(ax, "rho", "rhat")
    edge(ax, "rhat", "sig")
    edge(ax, "sig", "di")
    edge(ax, "thi", "di", rad=0.56)
    edge(ax, "rhat", "Di")
    edge(ax, "Lsnr", "rhat")
    edge(ax, "Lsnr", "Di")
    edge(ax, "Lwid", "sig")

    # ---- the one statement the graph cannot draw ----------------------------
    ax.text(3.65, 6.35,
            "No distance channel: the SNR is the\n"
            "distance coordinate, and $d_L$ is recovered\n"
            r"from $(\hat{M}_c, \hat{\rho}_i)$ rather than measured.",
            ha="left", va="top", fontsize=SUBSIZE, color=fs.INK2,
            linespacing=1.35, zorder=4)

    legend(ax, 13.30, 9.50)
    return fig


def main():
    fig = build()
    fs.save(fig, "fig_pgm")
    plt.close(fig)


if __name__ == "__main__":
    main()
