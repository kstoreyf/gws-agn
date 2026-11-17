#!/usr/bin/env python
"""Can the existing injection sets serve the obslike family?  Seed 100, both lanes.

    sbatch --job-name=injneff run_cpu.sbatch jax injection_neff.py     (jax env + legacy pin;
                                                    calls core_family.py points in the core venv)

The selection term is mu = (1/N_draw) sum_det w_k with w_k = pi(theta_k)/pdraw_k.  The
injections were drawn once, from a fixed proposal (0.65 legacy-mock population + 0.10
uniform + 0.25 AGN-targeted for the targeted lane; 0.9/0.1 for popuni), and store pdraw
exactly.  Changing the target population changes only the mass/spin factor of pi.  The
proxy below takes the population branch's redshift prior (uniform in comoving volume,
corepop/scripts/injection_check.py convention), i.e.

    w_k = p_F(m1, q, chi) * zfac(z) / pdraw_k,   zfac = p_Vc(z) / ((1+z) dL'(z)) / (4 pi),

with p_F core's powerlaw+peak density of family F (core venv, its (1+z)^-1 rate factor
removed).  For each family and host class: the fraction of detected injections with
nonzero weight, N_eff = (sum w)^2 / sum w^2, and the selection-term variance for N = 1000
events, N^2 (1/N_eff - 1/N_draw) ~ N^2/N_eff.  The proxy omits the catalog's redshift
structure; Analysis 14 measured the production term on the present family (legacy mock
population, seed 100, targeted lane): N_eff 7.5e5 (mixed set, catalog GAL+AGN) and 3.8e5
(all-AGN set).  The obslike estimate scales those by the proxy ratio obslike/legacy-mock.
Writes results/injection_neff.json.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
OB = HERE.parent
sys.path.insert(0, str(OB.parent))
import generate_dataset as G  # noqa: E402

PIN = Path("/hildafs/projects/phy230014p/magana/src/darksirens-2b86a2d")
DATA = Path("/hildafs/projects/phy230054p/magana/gws-agn-data-v3")
CPY = "/hildafs/projects/phy230014p/magana/envs/darksirens-core-e7c3007/bin/python"
SEED = 100
N_EV = 1000
BRANCH_NAMES = {"targeted": ["population", "uniform", "targeted_agn"], "popuni": ["population", "uniform"]}
A14_PRODUCTION = {"mixed_catalog_gal_agn": {"n_eff_ref": 7.52e5, "n_eff_range": [5.53e5, 8.17e5],
                                            "file": "analysis_14 results/scans/b_inj1_n8000_r00.json"},
                  "all_agn_catalog_agn": {"n_eff_ref": 3.77e5, "n_eff_range": [3.59e5, 4.03e5],
                                          "file": "analysis_14 results/repro_a.json"}}


def chunked(fn, n, chunk=200_000):
    return np.concatenate([fn(slice(i, min(i + chunk, n))) for i in range(0, n, chunk)])


def stats(w, ndraw, br, names, m1, m2):
    pos = w > 0
    s1, s2 = w.sum(), (w ** 2).sum()
    neff = float(s1 ** 2 / s2)
    var_n = float(N_EV ** 2 * (s2 / s1 ** 2 - 1.0 / ndraw))
    order = np.argsort(w)[::-1]
    top = order[:10]
    by = {}
    for b, nm in enumerate(names):
        sel = br == b
        by[nm] = {"n_detected": int(sel.sum()), "share_of_sum_w": float(w[sel].sum() / s1),
                  "share_of_sum_w2": float((w[sel] ** 2).sum() / s2)}
    cum2 = np.cumsum(w[order] ** 2) / s2
    return {"frac_detected_with_nonzero_weight": float(pos.mean()),
            "n_detected_with_nonzero_weight": int(pos.sum()),
            "n_eff": neff, "N2_over_neff": float(N_EV ** 2 / neff),
            "selection_variance_N1000": var_n,
            "mu_proxy": float(s1 / ndraw), "max_w": float(w.max()),
            "n_injections_carrying_half_of_sum_w2": int(np.searchsorted(cum2, 0.5) + 1),
            "top10_w": [{"w": float(w[i]), "m1src": float(m1[i]), "m2src": float(m2[i]),
                         "branch": names[int(br[i])]} for i in top],
            "by_branch": by,
            "m1src_quantiles_weighted": dict(zip(("0.1", "0.5", "0.9"),
                np.interp([0.1, 0.5, 0.9], np.cumsum(w[np.argsort(m1)]) / s1, np.sort(m1)).tolist()))}


def main():
    gmd = G.import_gmd(PIN)
    pop = gmd.PopulationConfig(gamma=G.GAMMA)
    grids = gmd._cosmology_grids(gmd._build_cosmology(G.H0_FID, G.OM0_FID, G.W0_FID, G.WA_FID), G.ZMAX_GRID)
    ddldz = np.gradient(grids["dl"], grids["z"])
    out = {"what": __doc__.split("\n\n")[1].strip(), "seed": SEED, "n_events": N_EV,
           "a14_production_present_family": A14_PRODUCTION, "lanes": {}}
    for lane in ("targeted", "popuni"):
        p = DATA / f"seed{SEED}" / "injections" / f"injections_{lane}.h5"
        with h5py.File(p, "r") as f:
            d = {k: np.asarray(f[k][:]) for k in ("m1src", "m2src", "chieff", "z", "pdraw",
                                                  "pdraw_population", "branch")}
            ndraw = int(f.attrs["Ndraw"])
        n = d["pdraw"].size
        q = d["m2src"] / d["m1src"]
        br = d["branch"].astype(int)
        pz = np.interp(d["z"], grids["z"], grids["dvc_dz"]) / np.trapz(grids["dvc_dz"], grids["z"])
        jac = np.interp(d["z"], grids["z"], ddldz) * (1.0 + d["z"])
        zfac = pz / np.maximum(jac, 1e-300) / (4.0 * np.pi)
        p_leg = chunked(lambda s: gmd._mass_spin_pdf(d["m1src"][s], q[s], d["chieff"][s], pop), n)
        rec = {"n_detected": n, "ndraw": ndraw,
               "zfac_check_max_rel_err_vs_pdraw_population":
                   float(np.max(np.abs(p_leg * zfac / d["pdraw_population"] - 1.0)[p_leg > 0])),
               "families": {}}
        ws = {"legacy_mock": p_leg * zfac / d["pdraw"]}
        tmp_in = OB / "cache" / f"inj_points_{lane}.npz"
        np.savez(tmp_in, m1src=d["m1src"], q=q, chieff=d["chieff"], z=d["z"])
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "NPY_DISABLE_CPU_FEATURES")}
        env["JAX_PLATFORMS"] = "cpu"
        for fam in ("mock", "obslike"):
            tmp_out = OB / "cache" / f"inj_lnp_{fam}_{lane}.npz"
            subprocess.run([CPY, str(HERE / "core_family.py"), "points", fam, str(tmp_in), str(tmp_out)],
                           check=True, env=env)
            L = np.load(tmp_out)
            for cls in ("gal", "agn"):
                p_core = np.exp(L[f"lnp_core_{cls}"] + np.log1p(d["z"]))   # drop core's (1+z)^-1
                ws[f"core_{fam}_{cls}"] = p_core * zfac / d["pdraw"]
            tmp_out.unlink()
        tmp_in.unlink()
        for k, w in ws.items():
            rec["families"][k] = stats(w, ndraw, br, BRANCH_NAMES[lane], d["m1src"], d["m2src"])
            r = rec["families"][k]
            print(f"[{lane} {k}] nonzero {r['frac_detected_with_nonzero_weight']:.4f}  N_eff {r['n_eff']:.4g}  "
                  f"N^2/N_eff {r['N2_over_neff']:.3f}  max w {r['max_w']:.3g}  half of sum w^2 in "
                  f"{r['n_injections_carrying_half_of_sum_w2']} injections", flush=True)
        base = rec["families"]["legacy_mock"]["n_eff"]
        rec["proxy_ratio_to_legacy_mock"] = {k: v["n_eff"] / base for k, v in rec["families"].items()}
        out["lanes"][lane] = rec
    # production estimate for the obslike family (targeted lane is what production reads)
    rt = out["lanes"]["targeted"]["proxy_ratio_to_legacy_mock"]
    est = {}
    for key, cls in (("mixed_catalog_gal_agn", "gal"), ("all_agn_catalog_agn", "gal")):
        for fam in ("mock", "obslike"):
            for c in ("gal", "agn"):
                ne = A14_PRODUCTION[key]["n_eff_ref"] * rt[f"core_{fam}_{c}"]
                est[f"{key}__core_{fam}_{c}"] = {"n_eff": ne, "N2_over_neff": N_EV ** 2 / ne}
    out["production_estimate_targeted"] = est
    (OB / "results" / "injection_neff.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(est, indent=1))
    print("wrote", OB / "results" / "injection_neff.json")


if __name__ == "__main__":
    main()
