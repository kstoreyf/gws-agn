"""Figure style of the PRD paper (REVTeX two-column).

Sizes are the printed sizes: a single-column figure is COL = 3.375 in wide, a full-width
(figure*) one PAGE = 7.0 in. Every figure with more than one panel is full width.

Colours (few, each with one meaning everywhere in the paper)
    GAL    blue         galaxies, the galaxy catalog, galaxy-hosted mergers, an analysis with
                        the galaxy catalog alone
    AGN    vermilion    AGN, the AGN catalog, AGN-hosted mergers, the AGN number density
    BOTH   bluish green an analysis with both catalogs (the two-population model)
    INPUT  black        input values of the simulated catalogs (lines or markers)
    PRIOR  grey         prior densities and range edges (dotted)
    COMP   four purples completeness of the flux-limited catalogs, light (most complete)
                        to dark (least complete), always with the same marker per level

Line styles for the population model (Figs. 4 and 5)
    solid    host-dependent model, reference population fixed at its input
    dashdot  host-dependent model, reference peak location and spin mean free
    dotted   host-dependent model, reference population and both widths free
    dashed   common population (one mass and spin distribution for both host classes)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL, PAGE = 3.375, 7.0

GAL = "#0072B2"
AGN = "#D55E00"
BOTH = "#009E73"
INPUT = "#000000"
PRIOR = "#8C8C8C"
LIGHTGREY = "#BDBDBD"
# completeness inside the horizon, most complete first (flux limits m < 21, 20, 19, 18)
COMP = ["#B79BD0", "#8C62AA", "#62327F", "#33114A"]
COMP_MARKERS = ["o", "s", "^", "D"]

LS_FIXED, LS_FREE, LS_EIGHT, LS_COMMON = "-", "-.", ":", "--"

LW = 1.6          # primary curves (>= 1.5 pt)
LW_THIN = 0.6     # ensemble members (one curve per simulated catalog)
LW_CONTOUR = 1.4  # contour outlines (>= 1.3 pt)
LW_INPUT = 0.9    # input lines
MS = 5.0          # markers (>= 5 pt)

FS_LABEL, FS_TICK, FS_TEXT, FS_PANEL = 9.0, 8.0, 8.0, 9.0


def configure():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": FS_TEXT,
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_TEXT,
        "legend.frameon": False,
        "axes.linewidth": 0.7,
        "lines.linewidth": LW,
        "lines.markersize": MS,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.size": 3.2, "ytick.major.size": 3.2,
        "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.top": True, "ytick.right": True,
        "axes.labelpad": 2.5,
        "figure.constrained_layout.use": False,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
        "path.simplify": False,
        "image.interpolation": "nearest",
    })


def panel_label(ax, text, x=0.03, y=0.96, ha="left", va="top", **kw):
    """Panel letter or simulated-catalog letter inside the axes corner."""
    kw.setdefault("fontsize", FS_PANEL)
    return ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, **kw)


def label(ax, x, y, text, color, ha="left", va="center", transform=None, **kw):
    """Direct label next to a curve, in the curve's colour."""
    kw.setdefault("fontsize", FS_TEXT)
    return ax.text(x, y, text, color=color, ha=ha, va=va,
                   transform=transform if transform is not None else ax.transData, **kw)


def input_vline(ax, x, **kw):
    kw.setdefault("color", INPUT)
    kw.setdefault("lw", LW_INPUT)
    kw.setdefault("ls", "--")
    kw.setdefault("zorder", 1)
    return ax.axvline(x, **kw)


def input_hline(ax, y, **kw):
    kw.setdefault("color", INPUT)
    kw.setdefault("lw", LW_INPUT)
    kw.setdefault("ls", "--")
    kw.setdefault("zorder", 1)
    return ax.axhline(y, **kw)


H0_LABEL = r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]"
H0_DENS = r"$p(H_0)$ [km$^{-1}$ s Mpc]"
F_LABEL = r"$f_{\rm AGN}$"


def save(fig, outdir: Path, stem: str):
    """PDF and PNG with fixed metadata, so that two runs give identical files."""
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.pdf", metadata={"CreationDate": None, "ModDate": None,
                                                    "Producer": None, "Creator": None})
    fig.savefig(outdir / f"{stem}.png", metadata={"Software": None})
    plt.close(fig)
    print(f"wrote figures/{stem}.pdf, .png")

