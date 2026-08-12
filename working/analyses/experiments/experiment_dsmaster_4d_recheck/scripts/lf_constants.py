#!/usr/bin/env python
"""The mock's LF constants, in the h-scaled convention darksirens fits in.

Derived from the seed's own recorded ``glass_field_meta.json`` rather than
retyped, because three of these numbers must agree to many digits or the fit
support silently shifts:

  --m_faint_offset = -2.5 log10(x_cut)          [a magnitude DIFFERENCE, no h]
  --m_faint_cut    = M_B_faint_limit - 5 log10 h   [h-scaled Mhat]
  truth Mstar_hat  = M_B_star        - 5 log10 h   [h-scaled Mhat]

and they satisfy ``m_faint_cut - Mstar_hat == m_faint_offset`` identically,
which this module asserts.  ``Mhat = m - DM(z; H0=100)``, so the h-scaling is
``M0 = M0hat + 5 log10 h`` (darksirens ``redshift/selection.py:142``).

    python lf_constants.py --seed 100          # human-readable
    python lf_constants.py --seed 100 --emit shell   # eval-able assignments
"""
import argparse
import json
import math
from pathlib import Path

DATA_ROOT = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data")
H0_FID = 67.74          # the mock's true H0; the only h in play


def constants(seed=100, data_root=DATA_ROOT, H0=H0_FID):
    meta = json.loads(
        (Path(data_root) / f"seed{seed}" / "catalogs" /
         "glass_field_meta.json").read_text())["magnitude_model"]
    x_cut = float(meta["x_cut_L_over_Lstar"])
    M_faint = float(meta["M_B_faint_limit"])
    M_star = float(meta["schechter"]["M_B_star"])
    alpha = float(meta["schechter"]["alpha"])

    five_log_h = 5.0 * math.log10(H0 / 100.0)
    out = {
        "x_cut": x_cut,
        "m_faint_offset": -2.5 * math.log10(x_cut),
        "m_faint_cut": M_faint - five_log_h,
        "Mstar_hat_truth": M_star - five_log_h,
        "alpha_truth": alpha,
        "M_B_star": M_star,
        "M_B_faint_limit": M_faint,
        "five_log_h": five_log_h,
        "H0_true": H0,
    }
    # The mock builds the truncation and M* from the same Schechter, so this is
    # an identity, not a fit: a mismatch means the metadata and the h-scaling
    # disagree and the cut would land off the population's edge.
    resid = (out["m_faint_cut"] - out["Mstar_hat_truth"]) - out["m_faint_offset"]
    if abs(resid) > 1e-12:
        raise SystemExit(
            f"inconsistent LF constants: (m_faint_cut - Mstar_hat) - "
            f"m_faint_offset = {resid:.3e}, must be 0")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--data_root", default=str(DATA_ROOT))
    ap.add_argument("--emit", default="human", choices=["human", "shell", "json"])
    a = ap.parse_args()
    c = constants(a.seed, Path(a.data_root))

    if a.emit == "json":
        print(json.dumps(c, indent=2))
    elif a.emit == "shell":
        print(f"M_FAINT_OFFSET={c['m_faint_offset']:.10f}")
        print(f"M_FAINT_CUT={c['m_faint_cut']:.10f}")
        print(f"MSTAR_HAT_TRUTH={c['Mstar_hat_truth']:.10f}")
        print(f"ALPHA_TRUTH={c['alpha_truth']:.10f}")
    else:
        print(f"  x_cut            {c['x_cut']!r}")
        print(f"  5 log10 h        {c['five_log_h']:.6f}   (h = {c['H0_true']/100})")
        print(f"  --m_faint_offset {c['m_faint_offset']:.6f}")
        print(f"  --m_faint_cut    {c['m_faint_cut']:.6f}   (h-scaled)")
        print(f"  truth Mstar_hat  {c['Mstar_hat_truth']:.6f}")
        print(f"  truth alpha      {c['alpha_truth']:.6f}")


if __name__ == "__main__":
    main()
