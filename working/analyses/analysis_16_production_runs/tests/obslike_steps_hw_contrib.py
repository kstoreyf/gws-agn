#!/usr/bin/env python
"""Which replaced mergers shape the hybrid-minus-stored line: per merger, [ln L_i(M) - ln L_i(N)] at the 90% ends of the
stored line minus its value at the peak (numpy only; reads tests/results/obslike_steps/halfwidth_<slug>.npz and the report json).
    python tests/obslike_steps_hw_contrib.py <slug>"""
import json
import sys
from pathlib import Path

import numpy as np

A16 = Path(__file__).resolve().parents[1]
slug = sys.argv[1]
z = np.load(A16 / "tests/results/obslike_steps" / f"halfwidth_{slug}.npz")
r = json.loads((A16 / "tests/results" / f"obslike_steps_halfwidth_{slug}.json").read_text())
H, K = z["H0"], z["K"]
lo, hi = r["corrected"]["stored"]["ci90"]
ipk = int(np.argmin(np.abs(H - r["peak_H0"])))
il, ih = int(np.argmin(np.abs(H - lo))), int(np.argmin(np.abs(H - hi)))
dd = z["hyb_ev_M"] - z["hyb_ev_N"]
dv = 0.5 * (z["hyb_var_M"] - z["hyb_var_N"])
tot = dd + dv
print(f"{slug}: peak {H[ipk]:g}, stored 90% ends ~{H[il]:g}, {H[ih]:g}; total shape change (lo-pk, hi-pk): "
      f"{tot[il].sum() - tot[ipk].sum():+.2f}, {tot[ih].sum() - tot[ipk].sum():+.2f} nats")
o = np.argsort(-(np.abs(tot[il] - tot[ipk]) + np.abs(tot[ih] - tot[ipk])))[:8]
for k in o:
    print(f"   merger {int(K[k])}: ess_N(pk) {z['hyb_ess_N'][ipk, k]:.1f} ess_M(pk) {z['hyb_ess_M'][ipk, k]:.1f} n_in_N(pk) {int(z['hyb_n_in_N'][ipk, k])}"
          f"  (lo-pk) {tot[il, k] - tot[ipk, k]:+.2f}  (hi-pk) {tot[ih, k] - tot[ipk, k]:+.2f}")
