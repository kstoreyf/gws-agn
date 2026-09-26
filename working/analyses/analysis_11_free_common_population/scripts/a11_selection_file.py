#!/usr/bin/env python
"""Analysis 11, selection-support gate, part (b): injection weight tails.

POPULATION-ONLY PROXY (as Analysis 10 Gate B, part b).  For each branch
population at the ABSOLUTE means the free-baseline domains reach, the weight of
detected injection j is

    w_j = p_pop(theta_j | mu_G_abs, mu_chi_abs) / pdraw_j

with p_pop the generator's density in the SAME canonical coordinates as the
stored pdraw (checked: at the fiducial it reproduces the file's
pdraw_population to < 1e-10).  The numerator carries the smooth
uniform-in-comoving-volume redshift prior, not the catalogue one the dark-siren
selection term uses, so this diagnoses each branch population's demand on the
proposal; the real selection term is measured live in ``a11_closure.py``.

Reported per branch point and for the f = 0.3 mixture of each (GAL, AGN) pair:
N_eff = (sum w)^2 / sum w^2 against the guard's 5 N_obs = 5000, the largest
normalised weight, and the top-10/100/1000 shares.  No injections are generated.

    JAX_PLATFORMS=cpu python scripts/a11_selection_file.py    (CPU node)
"""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.dont_write_bytecode = True

import numpy as np

HERE = Path(__file__).resolve().parent
A11 = HERE.parent
DATA_CODE = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data")
DARKSIRENS_A8 = Path("/hildafs/projects/phy230014p/magana/src/darksirens-a8")
INJ = DATA_CODE / "seed100" / "injections" / "injections_targeted.h5"
OUT = A11 / "diagnostics" / "a11_selection_file.json"

# (mu_G_GAL, dmu_G) mass points and (mu_chi_GAL, dmu_chi) spin points (brief s.10)
MASS_POINTS = [(31, -4), (31, 10), (35, 5), (39, -4), (39, 10)]
SPIN_POINTS = [(-0.10, -0.05), (-0.10, 0.20), (0.0, 0.10), (0.10, -0.05), (0.10, 0.20)]
F_MIX = 0.30
GUARD = 5000.0


def main():
    import h5py
    sys.path.insert(0, str(DATA_CODE))
    import generate_dataset as gd
    gmd = gd.import_gmd(DARKSIRENS_A8)

    t_start = time.time()
    with h5py.File(INJ, "r") as f:
        Ndraw = float(f.attrs["Ndraw"])
        zmax = float(f.attrs["zmax_proposal"])
        D = {k: f[k][:] for k in ("chieff", "pdraw", "pdraw_population",
                                  "m1src", "m2src", "z")}
    n = D["chieff"].size
    q = D["m2src"] / D["m1src"]
    cosmo = gmd._build_cosmology(gd.H0_FID, gd.OM0_FID, gd.W0_FID, gd.WA_FID)
    grids = gmd._cosmology_grids(cosmo, zmax)
    pc0 = gmd.PopulationConfig(gamma=gd.GAMMA)

    def density(mu_G_abs, mu_chi_abs, chunk=200_000):
        pop = gmd.PopulationConfig(gamma=gd.GAMMA, chi_mu=float(mu_chi_abs),
                                   peak_mu=float(mu_G_abs))
        o = np.empty(n)
        for i in range(0, n, chunk):
            s = slice(i, min(i + chunk, n))
            o[s] = gmd._selection_pdraw("population", D["m1src"][s], q[s],
                                        D["chieff"][s], D["z"][s], grids, pop)
        return o

    p_fid = density(pc0.peak_mu, pc0.chi_mu)
    rel = np.abs(p_fid - D["pdraw_population"]) / np.abs(D["pdraw_population"])
    conv = {"max_rel_diff": float(rel.max()), "passes": bool(rel.max() < 1e-10),
            "fiducial": {"peak_mu": pc0.peak_mu, "chi_mu": pc0.chi_mu}}
    if not conv["passes"]:
        raise SystemExit(f"[fatal] convention check failed ({rel.max():.3e})")

    cache = {}

    def w_of(mg, mc):
        key = (round(float(mg), 6), round(float(mc), 6))
        if key not in cache:
            cache[key] = density(mg, mc) / D["pdraw"]
        return cache[key]

    def tails(w, tag, **meta):
        s1, s2 = float(w.sum()), float(np.square(w).sum())
        ne = s1 * s1 / s2
        order = np.argsort(w)[::-1]
        return {"tag": tag, **meta, "Pdet_proxy": s1 / Ndraw, "Neff": ne,
                "Neff_over_5Nobs": ne / GUARD,
                "max_normalised_weight": float(w.max() / s1),
                "share_top": {str(k): float(w[order[:k]].sum() / s1)
                              for k in (10, 100, 1000)}}

    rows = []
    for (mg, dg) in MASS_POINTS:
        gal = w_of(mg, 0.0)
        agn = w_of(mg + dg, 0.10)
        rows.append(tails(gal, f"mass GAL mu_G={mg}", sector="mass", branch="GAL",
                          mu_G_abs=mg, mu_chi_abs=0.0, point=[mg, dg]))
        rows.append(tails(agn, f"mass AGN mu_G+dmu_G={mg + dg}", sector="mass",
                          branch="AGN", mu_G_abs=mg + dg, mu_chi_abs=0.10, point=[mg, dg]))
        # the f = 0.3 mixture of the two NORMALISED branch densities' weights
        rows.append(tails((1 - F_MIX) * gal + F_MIX * agn, f"mass mix f=0.3 ({mg},{dg})",
                          sector="mass", branch="mixture", point=[mg, dg]))
    for (mc, dc) in SPIN_POINTS:
        gal = w_of(35.0, mc)
        agn = w_of(40.0, mc + dc)
        rows.append(tails(gal, f"spin GAL mu_chi={mc}", sector="spin", branch="GAL",
                          mu_G_abs=35.0, mu_chi_abs=mc, point=[mc, dc]))
        rows.append(tails(agn, f"spin AGN mu_chi+dmu_chi={mc + dc:.2f}", sector="spin",
                          branch="AGN", mu_G_abs=40.0, mu_chi_abs=mc + dc, point=[mc, dc]))
        rows.append(tails((1 - F_MIX) * gal + F_MIX * agn, f"spin mix f=0.3 ({mc},{dc})",
                          sector="spin", branch="mixture", point=[mc, dc]))

    out = {
        "what": __doc__.split("Reported")[0].strip(),
        "provenance": {
            "gws_agn_sha": subprocess.run(["git", "-C", str(A11), "rev-parse", "HEAD"],
                                          capture_output=True, text=True).stdout.strip(),
            "darksirens_sha": subprocess.run(["git", "-C", str(DARKSIRENS_A8), "rev-parse",
                                              "HEAD"], capture_output=True,
                                             text=True).stdout.strip(),
            "host": socket.gethostname(), "injections": str(INJ), "n_detected": int(n),
            "Ndraw": Ndraw},
        "convention_check": conv,
        "rows": rows,
        "min_Neff_over_5Nobs": min(r["Neff_over_5Nobs"] for r in rows),
        "seconds": time.time() - t_start,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")
    print(f"convention check max rel diff {conv['max_rel_diff']:.2e}")
    for r in rows:
        print(f"  {r['tag']:<36s} Neff {r['Neff']:11.1f}  /5000 {r['Neff_over_5Nobs']:7.2f}  "
              f"w_max/sum {r['max_normalised_weight']:.2e}  top1000 {r['share_top']['1000']:.4f}")


if __name__ == "__main__":
    main()
