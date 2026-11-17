#!/usr/bin/env python
"""Selection-corrected V5 for the peak-family sets with 0 < f < 1 (jax env + legacy pin).

    python v5_selection_corrected.py

The detected AGN share of a set with planted f is f E_A / (f E_A + (1 - f) E_G), with
E_t = int p_core,t(m1, q) A_t(Mc) and A_t the detection acceptance built from the seed's own
complete catalog of tracer t (corepop/scripts/detected_gof.py, imported).  Equal-population
sets: both classes at the galaxy-class population (core tables lnp2d_gal); s = 1 sets: the
AGN class at mu_G 40, mu_chi 0.10 (lnp2d_agn).  Even at s = 0 the two catalogs need not have
the same efficiency (their own redshift structure inside the horizon).
Writes results/v5_selection_corrected.json.
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

OB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(OB.parent / "corepop" / "scripts"))
import detected_gof as DG  # noqa: E402

G = DG.G
SEEDS = (100, 101, 102, 103, 105)
SETS = {"_fagn0p10_corepop": (0.1, 0), "_corepop": (0.3, 0), "_fagn0p50_corepop": (0.5, 0),
        "_marked_dmu0p10_dmuG5_fagn0p10_corepop": (0.1, 1), "_marked_dmu0p10_dmuG5_corepop": (0.3, 1),
        "_marked_dmu0p10_dmuG5_fagn0p50_corepop": (0.5, 1),
        "_marked_dmu0p20_dmuG10_fagn0p10_corepop": (0.1, 2), "_marked_dmu0p20_dmuG10_corepop": (0.3, 2),
        "_marked_dmu0p20_dmuG10_fagn0p50_corepop": (0.5, 2)}


def main():
    gmd = G.import_gmd(DG.PIN)
    grids = gmd._cosmology_grids(gmd._build_cosmology(G.H0_FID, G.OM0_FID, G.W0_FID, G.WA_FID), G.ZMAX_GRID)
    T = np.load(OB.parent / "corepop" / "cache" / "core_tables.npz")
    T2 = np.load(OB / "cache" / "core_lnp2d_peak_s2.npz")     # AGN class at s = 2 (core_grid_s2.py)
    rows = []
    for s in SEEDS:
        A = {t: DG.acceptance_curve(s, t, grids) for t in ("gal", "agn")}
        E = {"G_gal": DG.pred_marginals(T["lnp2d_gal"], T["m1d"], T["qd"], A["gal"])["E"],
             "G_agn": DG.pred_marginals(T["lnp2d_gal"], T["m1d"], T["qd"], A["agn"])["E"],
             "A_agn": DG.pred_marginals(T["lnp2d_agn"], T["m1d"], T["qd"], A["agn"])["E"],
             "A2_agn": DG.pred_marginals(T2["lnp2d_agn_s2"], T["m1d"], T["qd"], A["agn"])["E"]}
        for sfx, (f, sz) in SETS.items():
            with h5py.File(DG.DATA / f"seed{s}" / "events" / f"events{sfx}.h5", "r") as fh:
                n_agn = int(np.asarray(fh["truth"]["host_type"]).sum())
            ea = {0: E["G_agn"], 1: E["A_agn"], 2: E["A2_agn"]}[sz]
            share = f * ea / (f * ea + (1 - f) * E["G_gal"])
            pull = (n_agn - 1000 * share) / np.sqrt(1000 * share * (1 - share))
            rows.append({"seed": s, "suffix": sfx, "f": f, "s": sz, "efficiency_ratio_AGN_over_GAL": ea / E["G_gal"],
                         "expected_share": float(share), "realised_share": n_agn / 1000, "pull": float(pull)})
            print(s, sfx, f"ratio {ea / E['G_gal']:.3f} expected {share:.3f} realised {n_agn / 1000:.3f} pull {pull:+.2f}",
                  flush=True)
    by = {}
    for r in rows:
        by.setdefault(r["suffix"], []).append(r["pull"])
    summ = {k: {"pulls": v, "chi2": float(np.sum(np.square(v))), "dof": len(v)} for k, v in by.items()}
    (OB / "results" / "v5_selection_corrected.json").write_text(json.dumps({"rows": rows, "summary": summ}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
