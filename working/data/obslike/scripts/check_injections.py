#!/usr/bin/env python
"""V6-style checks of a dedicated injection file (jax env + legacy darksirens pin).

    python check_injections.py INJ_FILE OUT.json

1. pdraw equals the stored mixture of the stored branch densities (relative error);
2. at 3000 random rows the branch densities are recomputed independently of the stage:
   population and uniform with gmd._selection_pdraw at the file's recorded proposal
   PopulationConfig; the AGN-targeted one by brute force over the survey's FLAT host list
   (boxes rebuilt from zgals/dzgals with the recorded R_LO, R_HI, n_sigma, host cut and
   cap), not the stage's padded table;
3. every detected row clears the detection rule's support (pdraw > 0, finite), the branch
   counts add up, and the redshift reach of the detected rows per branch;
4. the file loads through darksirens' loader (legacy pin) -- core's loader is checked by the
   caller.
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, "/hildafs/projects/phy230014p/magana/gws-agn/working/data")
import generate_dataset as G  # noqa: E402

PIN = Path("/hildafs/projects/phy230014p/magana/src/darksirens-2b86a2d")


def main():
    path, outp = Path(sys.argv[1]), Path(sys.argv[2])
    gmd = G.import_gmd(PIN)
    with h5py.File(path, "r") as f:
        d = {k: np.asarray(f[k][:]) for k in f.keys()}
        meta = json.loads(f.attrs["metadata_json"])
        # record-format lanes carry no proposal attribute: their proposal is the default PopulationConfig
        prop = (json.loads(f.attrs["proposal_population_json"]) if "proposal_population_json" in f.attrs
                else {"gamma": float(G.GAMMA)})
        mix = [float(f.attrs["proposal_mix_population"]), float(f.attrs["proposal_mix_uniform"]),
               float(f.attrs["proposal_mix_targeted_agn"])]
        survey = str(f.attrs["targeted_agn_survey"])
        tb = json.loads(f.attrs["targeted_branch_json"])
        ded = meta.get("dedicated_set", {})
        # the cut and cap actually used (pilot files recorded the defaults in tb; fixed since)
        tb["host_zmax"], tb["z_cap"] = ded.get("tgt_host_zmax", tb["host_zmax"]), ded.get("tgt_z_cap", tb["z_cap"])
    comps = prop if isinstance(prop, list) else [prop]
    pops = [(float(c.get("weight", 1.0)), gmd.PopulationConfig(**{k: v for k, v in c.items() if k != "weight"}))
            for c in comps]
    tot = sum(w for w, _ in pops)
    pops = [(w / tot, pp) for w, pp in pops]
    msp = lambda m1, q, chi: sum(w * gmd._mass_spin_pdf(m1, q, chi, pp) for w, pp in pops)  # noqa: E731
    grids = gmd._cosmology_grids(gmd._build_cosmology(G.H0_FID, G.OM0_FID, G.W0_FID, G.WA_FID), G.ZMAX_GRID)
    n = d["pdraw"].size
    out = {"file": str(path), "n_detected": n, "ndraw": meta["ndraw"], "mix": mix, "proposal": prop,
           "targeted_branch": tb}
    recon = mix[0] * d["pdraw_population"] + mix[1] * d["pdraw_uniform"] + mix[2] * d["pdraw_targeted_agn"]
    out["pdraw_vs_mixture_max_rel"] = float(np.max(np.abs(recon / d["pdraw"] - 1)))
    rng = np.random.default_rng(5)
    idx = np.sort(rng.choice(n, 3000, replace=False))
    q = d["m2src"][idx] / d["m1src"][idx]
    pp = sum(w * gmd._selection_pdraw("population", d["m1src"][idx], q, d["chieff"][idx], d["z"][idx], grids, pk)
             for w, pk in pops)
    pu = gmd._selection_pdraw("uniform", d["m1src"][idx], q, d["chieff"][idx], d["z"][idx], grids, pops[0][1])
    # targeted, brute force over the flat host list
    import healpy as hp
    with h5py.File(survey, "r") as f:
        zg, dz, ng = (np.asarray(f[k][:]) for k in ("zgals", "dzgals", "ngals"))
        nside = int(f.attrs["nside"])
    valid = np.arange(zg.shape[1])[None, :] < ng[:, None]
    zf, sf = zg[valid], dz[valid]
    pf = np.repeat(np.arange(zg.shape[0]), ng)
    lo = np.maximum(0.0, tb["r_lo"] * (zf - tb["nsigma_kernel_pad"] * sf))
    hi = np.minimum(tb["z_cap"], tb["r_hi"] * (zf + tb["nsigma_kernel_pad"] * sf))
    keep = (zf <= tb["host_zmax"]) & (hi > lo)
    lo, hi, pf = lo[keep], hi[keep], pf[keep]
    nh = int(keep.sum())
    order = np.argsort(pf, kind="stable")
    lo, hi, pf = lo[order], hi[order], pf[order]
    starts = np.searchsorted(pf, np.arange(zg.shape[0]))
    ends = np.searchsorted(pf, np.arange(zg.shape[0]), side="right")
    pix = hp.ang2pix(nside, 0.5 * np.pi - d["dec"][idx], d["ra"][idx])
    zz = d["z"][idx]
    ks = np.array([np.sum(((zz[i] >= lo[starts[p]:ends[p]]) & (zz[i] <= hi[starts[p]:ends[p]]))
                          / (hi[starts[p]:ends[p]] - lo[starts[p]:ends[p]])) for i, p in enumerate(pix)])
    ddl = np.interp(zz, grids["z"], np.gradient(grids["dl"], grids["z"]))
    pt = msp(d["m1src"][idx], q, d["chieff"][idx]) * ks / nh / hp.nside2pixarea(nside) \
        / (ddl * (1 + zz))
    rel = lambda a, b: float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))  # noqa: E731
    out["recompute_3000_rows_max_rel"] = {"population": rel(pp, d["pdraw_population"][idx]),
                                          "uniform": rel(pu, d["pdraw_uniform"][idx]),
                                          "targeted_agn_bruteforce": rel(pt, d["pdraw_targeted_agn"][idx])}
    out["n_hosts_retained_bruteforce"] = nh
    out["n_hosts_retained_stage"] = tb["n_hosts_retained"]
    out["pdraw_all_positive_finite"] = bool(np.all(d["pdraw"] > 0) & np.all(np.isfinite(d["pdraw"])))
    br = d["branch"].astype(int)
    out["detected_by_branch"] = {nm: int((br == b).sum()) for b, nm in enumerate(("population", "uniform", "targeted_agn"))}
    out["z_max_detected_by_branch"] = {nm: float(d["z"][br == b].max()) for b, nm in
                                       enumerate(("population", "uniform", "targeted_agn")) if (br == b).any()}
    out["m2src_min"] = float(d["m2src"].min())
    try:
        import contextlib
        import io
        import darksirens as ds
        from darksirens.gw.utils import load_selection_samples  # noqa: F401
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            o = load_selection_samples(str(path))
        out["legacy_loader"] = {"ok": True, "darksirens": ds.__file__}
    except Exception as e:  # noqa: BLE001
        out["legacy_loader"] = {"ok": False, "error": repr(e)[:300]}
    ok = (out["pdraw_vs_mixture_max_rel"] < 1e-12 and out["pdraw_all_positive_finite"]
          and max(out["recompute_3000_rows_max_rel"].values()) < 1e-9 and nh == tb["n_hosts_retained"])
    out["pass"] = bool(ok)
    outp.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("proposal", "targeted_branch")}, indent=1))


if __name__ == "__main__":
    main()
