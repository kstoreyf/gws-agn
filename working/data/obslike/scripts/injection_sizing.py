#!/usr/bin/env python
"""How large would a dedicated obslike injection set have to be?  (jax env; calls the core venv)

    sbatch --job-name=injsize run_cpu.sbatch jax injection_sizing.py

A dedicated set keeps the targeted lane's design and swaps the population branch's proposal
for the obslike population:  q_new = 0.65 p_obs + 0.10 p_unif + 0.25 p_targ  (p_obs = core's
obslike density x the population branch's redshift factor; p_unif, p_targ = the stored
component densities).  Its effective number per proposal for a target p is
    N_eff / N_draw = S1^2 / S2,   S1 = E_new[1_det p/q_new],  S2 = E_new[1_det (p/q_new)^2],
and both expectations are estimated from the EXISTING detected injections by reweighting,
E_new[g] = (1/N_old) sum_det g q_new/q_old, so S1 = mu and S2 = (1/N_old) sum_det p^2/(q_new q_old).
The estimate of S2 is itself noisy where the old proposal is thin; it is reported with the
number of injections carrying half of the sum.  Also for a proposal of pure obslike population
(w_pop = 1) and for 0.90 obslike + 0.10 uniform (the popuni design).
Seed 100, targeted lane, both classes (AGN: mu_G + 5, mu_chi + 0.10).
Writes results/injection_sizing.json.
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
N_EV = 1000


def main():
    gmd = G.import_gmd(PIN)
    grids = gmd._cosmology_grids(gmd._build_cosmology(G.H0_FID, G.OM0_FID, G.W0_FID, G.WA_FID), G.ZMAX_GRID)
    ddldz = np.gradient(grids["dl"], grids["z"])
    p = DATA / "seed100" / "injections" / "injections_targeted.h5"
    with h5py.File(p, "r") as f:
        d = {k: np.asarray(f[k][:]) for k in ("m1src", "m2src", "chieff", "z", "pdraw", "pdraw_population",
                                              "pdraw_uniform", "pdraw_targeted_agn", "branch")}
        nold = int(f.attrs["Ndraw"])
        meta = json.loads(f.attrs["metadata_json"])
    wts = meta.get("mixture_weights", [0.65, 0.10, 0.25])
    recon = wts[0] * d["pdraw_population"] + wts[1] * d["pdraw_uniform"] + wts[2] * d["pdraw_targeted_agn"]
    out = {"what": __doc__.split("\n\n")[1].strip(), "ndraw_old": nold, "n_detected_old": int(d["z"].size),
           "mixture_weights_old": wts,
           "pdraw_equals_weighted_components_max_rel": float(np.max(np.abs(recon / d["pdraw"] - 1.0)))}
    q = d["m2src"] / d["m1src"]
    pz = np.interp(d["z"], grids["z"], grids["dvc_dz"]) / np.trapz(grids["dvc_dz"], grids["z"])
    zfac = pz / np.maximum(np.interp(d["z"], grids["z"], ddldz) * (1.0 + d["z"]), 1e-300) / (4.0 * np.pi)
    tmp_in, tmp_out = OB / "cache" / "sizing_points.npz", OB / "cache" / "sizing_lnp.npz"
    np.savez(tmp_in, m1src=d["m1src"], q=q, chieff=d["chieff"], z=d["z"])
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "NPY_DISABLE_CPU_FEATURES")}
    env["JAX_PLATFORMS"] = "cpu"
    subprocess.run([CPY, str(HERE / "core_family.py"), "points", "obslike", str(tmp_in), str(tmp_out)],
                   check=True, env=env)
    L = np.load(tmp_out)
    tmp_in.unlink()
    tmp_out.unlink()
    p_obs = {c: np.exp(L[f"lnp_core_{c}"] + np.log1p(d["z"])) * zfac for c in ("gal", "agn")}
    designs = {"targeted_design_0.65pop_0.10unif_0.25targ": (0.65, 0.10, 0.25),
               "popuni_design_0.90pop_0.10unif": (0.90, 0.10, 0.0),
               "pure_population": (1.0, 0.0, 0.0)}
    old = {"current_set": {}}
    for c in ("gal", "agn"):
        w = p_obs[c] / d["pdraw"]
        old["current_set"][c] = {"n_eff": float(w.sum() ** 2 / (w ** 2).sum()),
                                 "n_eff_per_proposal": float(w.sum() ** 2 / (w ** 2).sum() / nold)}
    out["current"] = old["current_set"]
    out["designs"] = {}
    for name, (a, b, cc) in designs.items():
        rec = {}
        for c in ("gal", "agn"):
            # the new population branch proposes the GAL-class obslike population for both classes
            qn = sum(wt * comp for wt, comp in ((a, p_obs["gal"]), (b, d["pdraw_uniform"]),
                                                (cc, d["pdraw_targeted_agn"])) if wt > 0)
            s1 = (p_obs[c] / d["pdraw"]).sum() / nold
            with np.errstate(divide="ignore", invalid="ignore"):   # p = 0 rows (m2 < 5) contribute 0
                terms = np.where(p_obs[c] > 0, (p_obs[c] / qn) * (p_obs[c] / d["pdraw"]), 0.0)
            s2 = terms.sum() / nold
            per = s1 ** 2 / s2
            srt = np.sort(terms)[::-1]
            half = int(np.searchsorted(np.cumsum(srt) / srt.sum(), 0.5) + 1)
            rec[c] = {"n_eff_per_proposal": float(per),
                      "ndraw_for_N2_over_Neff_2": float(N_EV ** 2 / 2 / per),
                      "ndraw_for_N2_over_Neff_1": float(N_EV ** 2 / per),
                      "n_old_injections_carrying_half_of_S2": half}
            print(name, c, rec[c], flush=True)
        out["designs"][name] = rec
    (OB / "results" / "injection_sizing.json").write_text(json.dumps(out, indent=1))
    print("wrote", OB / "results" / "injection_sizing.json")


if __name__ == "__main__":
    main()
