#!/usr/bin/env python3
"""TASK 3 -- the HOST-ACCEPTANCE convention: the mock's host prior vs darksirens'.

WHAT THE GENERATOR DOES (``generate_dataset.stage_events``, verbatim):

    is_agn   = rng.uniform(ntry) < F_AGN                  # tracer choice
    i_gal    = rng.integers(0, n_gal, ntry)               # UNIFORM over rows
    i_agn    = rng.integers(0, n_agn, ntry)
    rate_gmax = max(1.0, (1 + z_grid[-1])**(gamma - 1))   # = 1 for gamma = 0
    acc      = rng.uniform(ntry) < (1 + z)**(gamma - 1) / rate_gmax
    det      = det_snr & acc

so the realised host prior is  p(host = g) proportional to  w_g (1+z_g)^(gamma-1)
with w_g = 1 for every catalog row, and gamma = GAMMA = 0.

WHAT DARKSIRENS DOES (``redshift/prior.py``, ``redshift/catalog.py``, field mode,
``volume_weighted = False``, ``z_depth = None``, ``sigma_kde = 0``):

    p_z(z|pix) = [N_obs(pix) p_cat(z|pix) + dN_miss] / Z_global
    p_cat(z|p) = g(z) SUM_{g in p} kw_g N(z; z_g, sig_g)
    kw_g       = w_g / (SUM_{g' in p} w_g') / Z(z_g),  Z(z_g) = INT N(z;z_g,sig) g(z) dz

and the population factor carries (1+z)^(gamma_fid - 1).  In the zero-bandwidth
limit Z(z_g) -> g(z_g), the front g(z) cancels the kernel normalisation and

    p_z -> [N_obs(p)/Z_global] SUM_g (w_g / SUM w) delta(z - z_g)
        =  w_g / Z_global                    since N_obs(p) = SUM_{g in p} w_g

i.e. UNIFORM over catalog rows with weight w_g, times (1+z)^(gamma_fid - 1) from
p_pop.  The two conventions therefore AGREE, provided

    (a) gamma_fid == GAMMA,                      (b) rate_gmax == 1,
    (c) w_g == 1 for every row,                  (d) N_obs(p) == SUM_{g in p} w_g,
    (e) Z_global == SUM_p N_obs(p).

All five are checked here against the live objects.  What does NOT cancel at
finite bandwidth is the ratio ``g(z_g)/Z(z_g) = 1 + O(sig^2)``; that residue and
the rate factor are then measured as PAIRED one-term substitutions on BOTH sides
of the score identity:

    Delta r = [numerator substitution]  -  [selection substitution]

numerator from ``attr_sky_oracle.py --host_prior_arms`` (arms ``delta_host_unif``
and ``delta_host_norate``), selection from ``attr_selmu_oracle.py`` (arms
``unif`` and ``norate`` against ``delta``).

Outputs: results/attr_hostw.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
sys.path.insert(0, str(HERE))
GEN = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data")
sys.path.insert(0, str(GEN))
import generate_dataset as G                                        # noqa: E402

H0_FID = 67.74


def jload(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--tracers", nargs="+", default=["gal", "agn"])
    ap.add_argument("--outdir", default=str(RES))
    args = ap.parse_args(argv)
    od = Path(args.outdir)
    t0 = time.time()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    import jax.numpy as jnp
    import attr_ds_bridge as bridge
    from darksirens.redshift.prior import prepare_redshift_prior_state

    out = {"name": "attr_hostw",
           "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "generator": {}, "darksirens": {}, "match": {}, "tracers": {}}

    # ---- the generator's own constants, read from the module -------------------
    import inspect
    gmd = G.import_gmd(G.DARKSIRENS_REPO)
    pop = gmd.PopulationConfig(gamma=G.GAMMA)
    cosmo = gmd._build_cosmology(G.H0_FID, G.OM0_FID, G.W0_FID, G.WA_FID)
    grids = gmd._cosmology_grids(cosmo, G.ZMAX_GRID)
    rate_gmax = max(1.0, (1.0 + float(grids["z"][-1])) ** (pop.gamma - 1.0))
    src = inspect.getsource(G.stage_events)
    out["generator"] = {
        "GAMMA": float(G.GAMMA), "gamma_in_PopulationConfig": float(pop.gamma),
        "rate_gmax": float(rate_gmax),
        "host_draw": "rng.integers(0, n_tracer, ntry)  -- UNIFORM over catalog rows",
        "host_draw_source_present": "rng.integers(0, n_gal, ntry)" in src,
        "acceptance_source_present":
            "(1.0 + z) ** (pop.gamma - 1.0) / rate_gmax" in src,
        "survey_weight_column": "np.ones_like(sub['z'])  (w_g = 1)",
        "F_AGN": float(G.F_AGN), "DZ_SCALE": float(G.DZ_SCALE)}

    for tr in args.tracers:
        kw = dict(kde_window=4096) if tr == "gal" else {}
        B = bridge.build(tracer=tr, seed=args.seed, h0=H0_FID, **kw)
        st = prepare_redshift_prior_state(
            "dark_sirens", B.cosmo0, B.survey_p, B.cat_pe, mark_model="none",
            mark_params=None, mark_names=(), materialize_state=True,
            catalog_sky_weighting="field")
        ngals = np.asarray(B.cat_pe.ngals)
        wg = np.asarray(B.cat_pe.wgals)
        occ = np.arange(wg.shape[0])[ngals > 0]
        rng = np.random.default_rng(11)
        rows = rng.choice(occ, size=min(3000, occ.size), replace=False)
        wmin, wmax = 1e30, -1e30
        for r in rows:
            n = int(ngals[r])
            wmin = min(wmin, float(wg[r, :n].min()))
            wmax = max(wmax, float(wg[r, :n].max()))
        Nobs = np.exp(np.asarray(st.log_Nobs))
        chk = {
            "gamma_fid": float(B.gamma_fid),
            "gamma_matches_generator": bool(
                abs(float(B.gamma_fid) - float(G.GAMMA)) == 0.0),
            "w_g_min": wmin, "w_g_max": wmax,
            "w_g_is_unity": bool(wmin == 1.0 and wmax == 1.0),
            "N_obs_minus_ngals_maxabs": float(
                np.nanmax(np.abs(Nobs[ngals > 0] - ngals[ngals > 0]))),
            "Z_global_minus_total_abs": float(
                abs(np.exp(float(np.asarray(st.log_Z_global))) - ngals.sum())),
            "volume_weighted": bool(st.kernels.volume_weighted),
            "z_depth": (None if st.kernels.z_depth is None
                        else float(st.kernels.z_depth)),
            "sigma_kde": float(np.asarray(B.survey_p.sigma_kde)),
            "max_dN_miss": float(np.asarray(st.dN_miss).max()),
        }
        out["tracers"][tr] = {"conventions": chk}
        print(f"[{tr}] gamma_fid={chk['gamma_fid']} (generator GAMMA={G.GAMMA}); "
              f"w_g in [{wmin}, {wmax}]; |N_obs-ngals|max="
              f"{chk['N_obs_minus_ngals_maxabs']:.2e}; "
              f"|Z_global-total|={chk['Z_global_minus_total_abs']:.2e}; "
              f"volume_weighted={chk['volume_weighted']}", flush=True)
        del B, st

    # ---- the two paired substitutions -----------------------------------------
    for tr in args.tracers:
        O = jload(RES / f"attr_selmu_{tr}.json")
        K = jload(RES / f"attr_sky_oracle_{tr}_hostw.json")
        rec = out["tracers"].setdefault(tr, {})
        if O:
            d = O["dlnmu_at_truth"]
            rec["selection_side"] = {
                "dlnmu_delta": d["delta"], "dlnmu_unif": d["unif"],
                "dlnmu_norate": d["norate"], "dlnmu_kde": d["kde"],
                "unif_minus_delta": d["unif"] - d["delta"],
                "norate_minus_delta": d["norate"] - d["delta"]}
        if K:
            s = K["substitutions"]
            rec["numerator_side"] = {
                "uniform_minus_darksirens":
                    s.get("hostprior__uniform_minus_darksirens__delta_host"),
                "norate_minus_darksirens":
                    s.get("hostprior__norate_minus_darksirens__delta_host"),
                "n_events": K["n_events"],
                "anchors": K.get("anchors", {}),
                "dlnmu_dH0": K.get("dlnmu_dH0"),
                "arms_r": {a: v["r"]["mean"] for a, v in K["arms"].items()}}
        if O and K:
            du = rec["numerator_side"]["uniform_minus_darksirens"]
            dn = rec["numerator_side"]["norate_minus_darksirens"]
            rec["delta_r"] = {
                "uniform_host_prior": {
                    "mean": du["mean"] - rec["selection_side"]["unif_minus_delta"],
                    "sem": du["sem"]},
                "no_rate_factor": {
                    "mean": dn["mean"] - rec["selection_side"]["norate_minus_delta"],
                    "sem": dn["sem"]}}
            print(f"[{tr}] Delta r (uniform host prior) = "
                  f"{rec['delta_r']['uniform_host_prior']['mean']:+.4e} "
                  f"+- {du['sem']:.2e};   Delta r (drop the rate factor) = "
                  f"{rec['delta_r']['no_rate_factor']['mean']:+.4e} "
                  f"+- {dn['sem']:.2e}", flush=True)

    (od / "attr_hostw.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {od/'attr_hostw.json'}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
