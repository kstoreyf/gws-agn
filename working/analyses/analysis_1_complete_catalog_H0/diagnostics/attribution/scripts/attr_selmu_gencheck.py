#!/usr/bin/env python3
"""TASK 1 -- the GENERATIVE cross-check of the selection oracle at truth.

``attr_selmu_oracle.py`` predicts, per catalog galaxy,

    <P_det>       = mu_norate / Ntot        (the SNR cut alone)
    <acc * P_det> = mu_unif   / Ntot        (with the host acceptance (1+z)^(gamma-1))

which is EXACTLY the quantity ``generate_dataset.stage_events`` realises when it
proposes an event: a host drawn uniformly from the tracer's catalog, masses and
spin from ``gmd``'s own samplers, ``observe()``, ``detect_from_observation()``
and the ``acc`` acceptance.  This script re-runs that loop verbatim -- the
GENERATOR's own functions, the generator's own ``gmd._interp_dl``, the real
catalogs -- for N draws per tracer and compares.

The mock's own ``events_meta.json::realised`` numbers come from only 200,000
proposals per seed, so they carry ~2e-4 of shot noise; this replication is 50x
larger and is the sharper test.

Outputs: results/attr_selmu_gencheck.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GEN = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data")
sys.path.insert(0, str(GEN))
import generate_dataset as G                                        # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--ndraw", type=float, default=1.0e7)
    ap.add_argument("--rng", type=int, default=987654321)
    ap.add_argument("--chunk", type=int, default=500_000)
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    args = ap.parse_args(argv)
    N = int(args.ndraw)
    t0 = time.time()

    gmd = G.import_gmd(G.DARKSIRENS_REPO)
    cosmo = gmd._build_cosmology(G.H0_FID, G.OM0_FID, G.W0_FID, G.WA_FID)
    grids = gmd._cosmology_grids(cosmo, G.ZMAX_GRID)
    pop = gmd.PopulationConfig(gamma=G.GAMMA)
    rate_gmax = max(1.0, (1.0 + float(grids["z"][-1])) ** (pop.gamma - 1.0))
    print(f"rate_gmax = {rate_gmax}  (gamma = {pop.gamma})", flush=True)

    rng = np.random.default_rng(args.rng)
    out = {"name": "attr_selmu_gencheck", "seed": args.seed, "n_draw": N,
           "rate_gmax": float(rate_gmax), "gamma": float(pop.gamma),
           "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "tracers": {}}
    for tr in ("gal", "agn"):
        with h5py.File(GEN / f"seed{args.seed}" / "catalogs"
                       / f"catalog_{tr}_complete.h5", "r") as f:
            zc = f["z"][:]
        ndet = nsnr = n = 0
        for i0 in range(0, N, args.chunk):
            m = min(args.chunk, N - i0)
            zz = zc[rng.integers(0, zc.size, m)]
            dl = gmd._interp_dl(zz, grids)
            m1, up = gmd._sample_powerlaw_peak_m1(rng, m, pop, return_component=True)
            q = gmd._sample_q(rng, m1, pop, use_peak=up)
            chi = gmd._sample_chieff(rng, m, pop)
            obs = G.observe(rng, m1 * (1 + zz), q * m1 * (1 + zz), chi, dl,
                            None, None, need_sky=False)
            det, _ = G.detect_from_observation(obs)
            acc = rng.uniform(size=m) < (1.0 + zz) ** (pop.gamma - 1.0) / rate_gmax
            nsnr += int(det.sum())
            ndet += int((det & acc).sum())
            n += m
        out["tracers"][tr] = {
            "n_catalog": int(zc.size), "n_draw": int(n),
            "acc_Pdet": ndet / n, "acc_Pdet_sigma": float(np.sqrt(ndet) / n),
            "Pdet": nsnr / n, "Pdet_sigma": float(np.sqrt(nsnr) / n)}
        print(f"{tr}: <acc*Pdet> = {ndet/n:.6e} +- {np.sqrt(ndet)/n:.2e}   "
              f"<Pdet> = {nsnr/n:.6e} +- {np.sqrt(nsnr)/n:.2e}  (N={n:,})", flush=True)

    # the oracle's prediction and the mock's own realised numbers
    fa = float(G.F_AGN)
    ora = {}
    for tr in ("gal", "agn"):
        p = Path(args.outdir) / f"attr_selmu_{tr}.json"
        if p.exists():
            ora[tr] = json.loads(p.read_text())[
                "per_galaxy_detection_probability_at_truth"]
    if set(ora) == {"gal", "agn"}:
        mix_o = (1 - fa) * ora["gal"]["unif"] + fa * ora["agn"]["unif"]
        mix_s = (1 - fa) * ora["gal"]["norate"] + fa * ora["agn"]["norate"]
        mix_b = ((1 - fa) * out["tracers"]["gal"]["acc_Pdet"]
                 + fa * out["tracers"]["agn"]["acc_Pdet"])
        mix_bs = ((1 - fa) * out["tracers"]["gal"]["Pdet"]
                  + fa * out["tracers"]["agn"]["Pdet"])
        sig = np.sqrt(((1 - fa) ** 2 * out["tracers"]["gal"]["acc_Pdet_sigma"] ** 2
                       + fa ** 2 * out["tracers"]["agn"]["acc_Pdet_sigma"] ** 2))
        sigs = np.sqrt(((1 - fa) ** 2 * out["tracers"]["gal"]["Pdet_sigma"] ** 2
                        + fa ** 2 * out["tracers"]["agn"]["Pdet_sigma"] ** 2))
        ev = json.loads((GEN / f"seed{args.seed}" / "events"
                         / "events_meta.json").read_text())["realised"]
        out["comparison"] = {
            "f_AGN": fa,
            "oracle": {"acc_Pdet": mix_o, "Pdet": mix_s},
            "brute_force": {"acc_Pdet": mix_b, "sigma": float(sig),
                            "Pdet": mix_bs, "Pdet_sigma": float(sigs)},
            "oracle_minus_brute_force_sigma": {
                "acc_Pdet": float((mix_o - mix_b) / sig),
                "Pdet": float((mix_s - mix_bs) / sigs)},
            "per_tracer_pull": {
                tr: {"acc_Pdet": float((ora[tr]["unif"]
                                        - out["tracers"][tr]["acc_Pdet"])
                                       / out["tracers"][tr]["acc_Pdet_sigma"]),
                     "Pdet": float((ora[tr]["norate"] - out["tracers"][tr]["Pdet"])
                                   / out["tracers"][tr]["Pdet_sigma"])}
                for tr in ("gal", "agn")},
            "mock_realised_seed": {
                "n_proposed": ev["n_proposed"],
                "detected_fraction": ev["detected_fraction"],
                "detected_fraction_sigma": float(
                    np.sqrt(ev["detected_fraction"] / ev["n_proposed"])),
                "detected_fraction_snr_only": ev["detected_fraction_snr_only"],
                "pull_vs_oracle": float(
                    (ev["detected_fraction"] - mix_o)
                    / np.sqrt(ev["detected_fraction"] / ev["n_proposed"])),
                "note": "the mock's own 200,000-proposal number is a 2e-4 shot-noise "
                        "estimate of the same quantity; the brute force above is the "
                        "sharper comparison"},
            "rate_factor_ratio": {
                "oracle": mix_o / mix_s, "brute_force": mix_b / mix_bs,
                "mock_realised": ev["detected_fraction"]
                / ev["detected_fraction_snr_only"]},
        }
        c = out["comparison"]
        print(f"\nMIXED (f_AGN = {fa}):")
        print(f"  <acc*Pdet>  oracle {mix_o:.6e}   brute force {mix_b:.6e} "
              f"+- {sig:.2e}   ({c['oracle_minus_brute_force_sigma']['acc_Pdet']:+.2f} sigma)")
        print(f"  <Pdet>      oracle {mix_s:.6e}   brute force {mix_bs:.6e} "
              f"+- {sigs:.2e}   ({c['oracle_minus_brute_force_sigma']['Pdet']:+.2f} sigma)")
        print(f"  rate ratio  oracle {c['rate_factor_ratio']['oracle']:.5f}  "
              f"brute {c['rate_factor_ratio']['brute_force']:.5f}  "
              f"mock {c['rate_factor_ratio']['mock_realised']:.5f}")
        print(f"  the mock's own realised fraction {ev['detected_fraction']:.5e} is "
              f"{c['mock_realised_seed']['pull_vs_oracle']:+.2f} sigma of its own "
              f"200,000-proposal shot noise")
    Path(args.outdir, "attr_selmu_gencheck.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {Path(args.outdir)/'attr_selmu_gencheck.json'} "
          f"({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
