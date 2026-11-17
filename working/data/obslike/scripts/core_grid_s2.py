#!/usr/bin/env python
"""Core's ln p(m1, q) of the peak family's AGN class at s = 2 (mu_G 45, mu_chi 0.20) on corepop's
2-D grid (corepop/cache/core_tables.npz m1d, qd; chi at its mode, z = 0), for the
selection-corrected V5 of the s = 2 sets.  core venv.  -> cache/core_lnp2d_peak_s2.npz"""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core_family as CF  # noqa: E402
T = np.load(HERE.parent.parent / "corepop" / "cache" / "core_tables.npz")
lp, info = CF.setup("historical", "mock", s_mult=2.0)
chig = np.linspace(-1, 1, 20001)
lpc = lp(35.0 * np.ones_like(chig), 0.8 * np.ones_like(chig), chig, 0 * chig, "agn")
i0 = int(np.argmax(lpc))
r = np.exp(lpc - lpc[i0]); pchi = r / np.trapz(r, chig)
M1, Q = np.meshgrid(T["m1d"], T["qd"], indexing="ij")
l2 = lp(M1, Q, 0 * M1 + chig[i0], 0 * M1, "agn") - np.log(pchi[i0])
np.savez_compressed(HERE.parent / "cache" / "core_lnp2d_peak_s2.npz", lnp2d_agn_s2=l2, m1d=T["m1d"], qd=T["qd"],
                    theta=np.array(info["theta_agn"]))
print("theta_agn", info["theta_agn"], "chi mode", chig[i0], "integral", float(np.trapz(np.trapz(np.exp(l2), T["qd"], axis=1), T["m1d"])))
