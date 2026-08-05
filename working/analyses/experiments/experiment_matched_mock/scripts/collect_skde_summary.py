#!/usr/bin/env python3
"""Assemble the full sigma_kde broadening ladder into results/skde_summary.json.

Globs every results/skde_*.json rung (realisation b: skde_<width>.json;
realisation s4102: skde_s4102_<width>.json), records median/offset/CI per rung,
and computes the reference scale the ladder has to be compared against: the
per-event PE redshift width sigma_z = sigma_dL * dL / (d dL/dz) implied by the
10% fractional distance uncertainty, at the median event of ev_obs_b.

The effective catalog kernel at rung sigma_kde is sqrt(dzgals^2 + sigma_kde^2)
with dzgals = 3e-3.
"""
import json
import re
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
H0_TRUTH = 67.74
DZ_GALS = 3e-3
SIGMA_DL = 0.10


def pe_redshift_width():
    """Median over events of sigma_dL * dL / (d dL/dz) at the true redshift."""
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u
    cosmo = FlatLambdaCDM(H0=H0_TRUTH, Om0=0.3075)
    with h5py.File(ROOT / "data_derived/obsdet/ev_obs_b.h5", "r") as f:
        z = f["truth/z"][:]
    zg = np.linspace(1e-4, 2.5, 4000)
    dl = cosmo.luminosity_distance(zg).to_value(u.Mpc)
    ddl = np.gradient(dl, zg)
    sig_z = SIGMA_DL * np.interp(z, zg, dl) / np.interp(z, zg, ddl)
    return {"median": float(np.median(sig_z)),
            "p16": float(np.percentile(sig_z, 16)),
            "p84": float(np.percentile(sig_z, 84)),
            "n_events": int(z.size),
            "definition": "sigma_dL * dL / (d dL/dz) at the true event redshift"}


def collect(pattern, key_re):
    rungs = {}
    for p in sorted(RES.glob(pattern)):
        m = re.match(key_re, p.name)
        if not m:
            continue
        d = json.loads(p.read_text())
        h = d["H0"]
        rungs[f"{float(m.group(1)):.3f}"] = {
            "median": h["median"],
            "offset": h["median"] - H0_TRUTH,
            "ci68": h["ci68"],
            "map": h["map"],
            "n_rejected": d.get("n_neginf_cells", 0),
            "file": f"results/{p.name}",
        }
    return rungs


def main():
    b = collect("skde_0.*.json", r"skde_(0\.\d+)\.json")
    s = collect("skde_s4102_*.json", r"skde_s4102_(0\.\d+)\.json")
    out = {
        "description": ("sigma_kde broadening ladder; effective kernel = "
                        "sqrt(dzgals^2 + sigma_kde^2), dzgals = 3e-3; obs-mode "
                        "PE, 2k samples"),
        "dzgals": DZ_GALS,
        "sigma_dL": SIGMA_DL,
        "pe_redshift_width": pe_redshift_width(),
        "baseline_no_broadening_offset": b["0.000"]["offset"],
        "realisations": {"b": {"rungs": b}, "s4102": {"rungs": s}},
        # kept for backward compatibility with earlier readers
        "rungs": b,
    }
    p = RES / "skde_summary.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"Wrote {p}")
    print(f"  PE z width: median {out['pe_redshift_width']['median']:.4f}")
    for tag, r in (("b", b), ("s4102", s)):
        for k in sorted(r, key=float):
            print(f"  {tag} sigma_kde={k}: offset {r[k]['offset']:+.2f}")


if __name__ == "__main__":
    main()
