"""One visual system for every figure in the paper.

Palette and rules follow the project data-viz standard: colours are assigned by
the job they do (identity / ordered magnitude / polarity), the categorical order
is fixed and never cycled, sequential ramps are single-hue, and every slot pair
used together was checked for colour-vision separation with the palette
validator rather than by eye.  Slots below 3:1 contrast against the page carry
direct labels or markers, never colour alone.

Measured on the adjacent pairlist used by lines and bars (OKLab dE x100, floor
8 for colour-vision deficiency, 15 for normal vision): worst adjacent CVD
separation 9.1, worst normal-vision separation 19.6 over the six slots below.
The ordered blue ramp clears the ordinal gates (adjacent dL >= 0.093, lightest
step 2.06:1 against the page).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np

PAPER = Path(__file__).resolve().parent.parent
FIGDIR = PAPER / "figures"
EXP = (PAPER.parent / "analyses" / "experiments").resolve()

# the analyses of record, and the dataset they were run on (read-only)
ANALYSES = (PAPER.parent / "analyses").resolve()
A0 = ANALYSES / "analysis_0_pure_tracer_H0" / "results"
A1 = ANALYSES / "analysis_1_complete_catalog_H0" / "results"
A2 = ANALYSES / "analysis_2_complete_catalog_H0_fagn" / "results"
DATA = (PAPER.parent / "data").resolve()

SEEDS = [100, 101, 102, 103, 105]
REF_SEED = 100
H0_TRUTH = 67.74
F_PLANTED = 0.30

# ---- identity: fixed categorical order, never cycled -----------------------
C = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
}
SERIES = [C["blue"], C["orange"], C["aqua"], C["yellow"], C["magenta"], C["green"]]

# ---- ordered magnitude: one hue, light to dark -----------------------------
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

# ---- polarity: two poles, neutral middle ----------------------------------
DIVERGING = ("#2a78d6", "#f0efec", "#d03b3b")

# ---- ink -------------------------------------------------------------------
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
TRUTH = "#0b0b0b"          # truth lines are ink, not a series colour
BAD = "#d03b3b"            # reserved status colour, only for "inadmissible"
OTHER = "#a9a8a2"          # the folded "every other realisation" neutral

# Only the first two slots are ever used together in one panel, so the
# all-pairs gate applies and both clear it against the white page:
#   worst all-pairs CVD dE 24.7 (protan), normal vision 33.6, contrast >= 3:1
# (validate_palette.py "#2a78d6,#eb6834" --mode light --surface "#ffffff"
#  --pairs all -> OK).  Sets of five realisations are folded to one neutral,
# the documented "fold to Other" pattern, rather than cycling five hues.

ONECOL = 3.35              # AASTeX single column, inches
TWOCOL = 6.9

RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8.0,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8.5,
    "axes.labelcolor": INK,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.7,
    "axes.titlelocation": "left",
    "axes.titlepad": 4.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "legend.labelcolor": INK2,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.6,
    "legend.borderaxespad": 0.3,
    "lines.linewidth": 1.6,
    "lines.markersize": 4.0,
    "lines.solid_capstyle": "round",
    "errorbar.capsize": 0.0,
    "text.color": INK,
    "mathtext.fontset": "dejavusans",
    # journal requirement: real embedded fonts, never Type 3
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


def use():
    mpl.use("Agg")
    mpl.rcParams.update(RC)


def save(fig, name: str):
    """Write one figure as a vector PDF plus a raster preview.

    Every figure is drawn at its final printed size (ONECOL or TWOCOL), so the
    PNG preview is the acceptance test for legibility: what is unreadable at
    300 dpi here is unreadable on paper.
    """
    FIGDIR.mkdir(exist_ok=True)
    out = FIGDIR / f"{name}.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"wrote {out} + {out.with_suffix('.png').name}")
    return out


def label_line(ax, x, y, text, color, *, dx=0.0, dy=0.0, ha="left", va="center",
               size=7.0):
    """Direct label in ink, with the mark's colour carried by a leading dot."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va=va, fontsize=size, color=INK2, zorder=6)


def truth_line(ax, value, axis="y", label=None, dashes=(3, 2), pos=None):
    """Reference line at a known value.  `pos` places the label along the line."""
    fn = ax.axhline if axis == "y" else ax.axvline
    fn(value, color=TRUTH, lw=0.9, ls=(0, dashes), zorder=1.5, alpha=0.75)
    if label:
        if axis == "y":
            ax.annotate(label, (pos if pos is not None else 0.995, value),
                        xycoords=("axes fraction", "data"),
                        xytext=(0, 3), textcoords="offset points", ha="right",
                        va="bottom", fontsize=6.8, color=INK2)
        else:
            ax.annotate(label, (value, pos if pos is not None else 0.99),
                        xycoords=("data", "axes fraction"),
                        xytext=(3, 0), textcoords="offset points", ha="left",
                        va="top", fontsize=6.8, color=INK2)


# ---------------------------------------------------------------------------
# posterior helpers, shared by the figure scripts
# ---------------------------------------------------------------------------
def posterior_1d(grid, logl):
    """Flat-prior posterior on a 1-D scan grid, normalised."""
    grid = np.asarray(grid, float)
    logl = np.asarray(logl, float)
    ok = np.isfinite(logl)
    p = np.zeros_like(logl)
    p[ok] = np.exp(logl[ok] - logl[ok].max())
    norm = np.trapz(p, grid)
    return p / norm if norm > 0 else p


def quantiles_1d(grid, logl, qs=(0.16, 0.5, 0.84)):
    p = posterior_1d(grid, logl)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(grid))])
    cdf /= cdf[-1]
    return np.interp(qs, cdf, grid)


def ci_band(x, y, lo, hi):
    """Curve restricted to [lo, hi], with the two ends interpolated.

    Masking the grid instead would shade to the nearest grid point, which on a
    coarse scan draws an interval up to one cell wider or narrower than the one
    quoted in the text.
    """
    x = np.asarray(x, float)
    inner = x[(x > lo) & (x < hi)]
    xs = np.concatenate([[lo], inner, [hi]])
    return xs, np.interp(xs, x, y)


def scan_1d(path, grid_key):
    """(grid, log_likelihood) from one darksirens scan file."""
    import h5py
    with h5py.File(path, "r") as f:
        return (np.asarray(f[grid_key][:], float),
                np.asarray(f["log_likelihood"][:], float))


def joint_grid(path):
    """(H0, f, normalised 2-D posterior) from a joint scan file.

    Flat priors on both axes, so the posterior is the exponentiated
    log-likelihood.  Same reduction as the analysis's own figure script,
    analyses/analysis_2_complete_catalog_H0_fagn/scripts/make_figures.py.
    """
    import h5py
    with h5py.File(path, "r") as fh:
        H0 = np.asarray(fh["H0_grid"][:], float)
        fv = np.asarray(fh["f_grid"][:], float)
        ll = np.asarray(fh["log_likelihood"][:], float)
    fin = np.isfinite(ll)
    P = np.where(fin, np.exp(ll - ll[fin].max()), 0.0)
    P /= np.trapz(np.trapz(P, fv, axis=1), H0, axis=0)
    return H0, fv, P


def hpd_levels_2d(H0, fv, P, fracs=(0.68, 0.90)):
    """Density thresholds enclosing the requested posterior mass.

    Cells are sorted by density, their probability mass accumulated, and the
    density at which the running total first reaches each level is returned:
    the standard highest-posterior-density construction.  Adapted from
    analyses/analysis_2_complete_catalog_H0_fagn/scripts/make_figures.py.
    """
    dA = np.outer(np.gradient(H0), np.gradient(fv))
    flat = P.ravel()
    w = flat * dA.ravel()
    order = np.argsort(flat)[::-1]
    cum = np.cumsum(w[order]) / w.sum()
    out = []
    for lev in fracs:
        k = min(int(np.searchsorted(cum, lev)), flat.size - 1)
        out.append(float(flat[order][k]))
    return out


def hpd_levels(logl, fracs=(0.68, 0.90)):
    """log-likelihood contour levels enclosing the given posterior mass."""
    lp = np.asarray(logl, float)
    p = np.where(np.isfinite(lp), np.exp(lp - np.nanmax(lp)), 0.0)
    flat = np.sort(p.ravel())[::-1]
    csum = np.cumsum(flat)
    csum /= csum[-1]
    return [float(flat[np.searchsorted(csum, f)]) for f in fracs], p / p.max()
