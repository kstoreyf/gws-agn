"""The fixed (H0, log10n0) cells every arm evaluates, and the shared inputs.

Two grids on one K = 1 GAL catalogue (seed 100, m < 19, conditional per-pixel
completeness):

    g1   1-D: H0 = 60, 60.5, ..., 76 (33 nodes) at log10n0 = -3.0 (the mock's truth)
    g2   2-D: H0 = 60, 62, ..., 76 (9) x log10n0 = -3.6, -3.4, ..., -2.4 (7)

The KDE nearest-galaxy window never binds on this catalogue (largest pixel: 457
galaxies against a 4096 window), so legacy's windowed sum and core's sum over
every galaxy in the pixel are the same sum.
"""
from pathlib import Path

import numpy as np

DATA = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data/seed100")
SURVEY = str(DATA / "surveys" / "survey_gal_m19_ns32.h5")
EVENTS = str(DATA / "events" / "events.h5")
INJECTIONS = str(DATA / "injections" / "injections_targeted.h5")
OM0 = 0.3075
FIXED_SURVEY = {"delta": 0.0, "sigma_kde": 0.0}   # pinned explicitly in every arm
MAX_LIKELIHOOD_VARIANCE = 1e6
SEL_BATCH_SIZE = 50000
PE_EVENT_BLOCK = 25


def cells():
    out = []
    for h in np.round(np.arange(60.0, 76.0001, 0.5), 10):
        out.append({"grid": "g1", "H0": float(h), "log10n0": -3.0})
    for h in np.round(np.arange(60.0, 76.0001, 2.0), 10):
        for n in np.round(np.arange(-3.6, -2.39, 0.2), 10):
            out.append({"grid": "g2", "H0": float(h), "log10n0": float(n)})
    return out


def bisect_cells():
    """Three fixed cells for the legacy-history bisection (log10n0 at the truth)."""
    return [{"grid": "bisect", "H0": h, "log10n0": -3.0} for h in (60.0, 67.0, 75.0)]
