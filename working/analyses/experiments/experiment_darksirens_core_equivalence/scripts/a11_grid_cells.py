"""Posterior-level grids for the A11 likelihood, core vs af896ca (around the 11D posterior).

    gH   1-D H0 = 60, 60.5, ..., 76 (33) at the 11D medians
    gF   1-D f_AGN = 0.10, 0.125, ..., 0.50 (17) at the 11D medians
    gM   2-D (mu_G, dmu_G): 34.6..36.6 step 0.25 (9) x 1.5..6.5 step 0.625 (9)
    gC   2-D (mu_chi, dmu_chi): -0.025..0.0175 step 0.0053125 (9) x 0.08..0.18 step 0.0125 (9)
Fixed coordinates sit at the 11D medians (H0 67.16, f 0.266, mu_G 35.61, dmu_G 3.94,
mu_chi -0.0035, dmu_chi 0.131).
"""
import numpy as np

MED = {"H0": 67.16, "fcat_2": 0.266, "mu_G": 35.61, "dmu_G": 3.94, "mu_chi": -0.0035, "dmu_chi": 0.131}


def cells():
    out = []
    for h in np.round(np.arange(60.0, 76.0001, 0.5), 10):
        out.append({**MED, "name": "gH", "H0": float(h)})
    for f in np.round(np.arange(0.10, 0.5001, 0.025), 10):
        out.append({**MED, "name": "gF", "fcat_2": float(f)})
    for a in np.round(np.linspace(34.6, 36.6, 9), 10):
        for d in np.round(np.linspace(1.5, 6.5, 9), 10):
            out.append({**MED, "name": "gM", "mu_G": float(a), "dmu_G": float(d)})
    for a in np.round(np.linspace(-0.025, 0.0175, 9), 10):
        for d in np.round(np.linspace(0.08, 0.18, 9), 10):
            out.append({**MED, "name": "gC", "mu_chi": float(a), "dmu_chi": float(d)})
    return out
