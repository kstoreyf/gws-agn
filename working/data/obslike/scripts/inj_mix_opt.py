#!/usr/bin/env python
"""Choose the branch weights of the dedicated lane by reweighting a pilot.  (core venv)

    python inj_mix_opt.py PILOT_FILE OUT.json [ROWS_TARGET]

Every pilot row stores its three branch densities, so a lane with weights (a, b, c) has
q_new = a p_pop + b p_unif + c p_tgt, and from the pilot (proposal q_old, N_old draws):
    rows per proposal  r = (1/N_old) sum_det q_new/q_old
    N_eff per proposal e = S1^2 / S2,   S1 = (1/N_old) sum_det t/q_old,
                                        S2 = (1/N_old) sum_det t^2/(q_new q_old)
for each target t of inj_stress.py (same points and combinations).  At a fixed stored size
ROWS_TARGET (default 3e6) the lane needs N_draw = ROWS_TARGET / r proposals and reaches
N_eff = e N_draw.  Writes N^2/N_eff (N = 1000) per mixture, point and combination.
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core_family as CF  # noqa: E402
import inj_stress as IS  # noqa: E402

MIXES = [(0.72, 0.03, 0.25), (0.65, 0.10, 0.25), (0.62, 0.03, 0.35), (0.52, 0.03, 0.45),
         (0.42, 0.03, 0.55), (0.80, 0.03, 0.17), (0.57, 0.08, 0.35)]


def main():
    inj, outp = Path(sys.argv[1]), Path(sys.argv[2])
    rows_target = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0e6
    with h5py.File(inj, "r") as f:
        d = {k: np.asarray(f[k][:]) for k in ("m1det", "m2det", "dL", "chieff", "ra", "dec", "pdraw",
                                              "pdraw_population", "pdraw_uniform", "pdraw_targeted_agn")}
        nold = int(f.attrs["Ndraw"])
        mix_old = [float(f.attrs[k]) for k in ("proposal_mix_population", "proposal_mix_uniform",
                                               "proposal_mix_targeted_agn")]
        seed = int(json.loads(f.attrs["metadata_json"])["seed"]) // 1000
    import healpy as hp
    q = d["m2det"] / d["m1det"]
    zg, dlg, dvcg = IS.zgrids()
    ddl = np.gradient(dlg, zg)
    sel = zg <= IS.ZMAX_T
    vol_norm = np.trapz((dvcg * (1 + zg) ** -1.0)[sel], zg[sel]) * 4 * np.pi
    pix = hp.ang2pix(32, 0.5 * np.pi - d["dec"], d["ra"])
    base = CF.FAMILIES["obslike"]
    targets = {}
    fns = {}
    for name, pt in IS.POINTS.items():
        h0 = pt.get("H0", IS.H0_FID)
        r = h0 / IS.H0_FID
        zp = np.interp(d["dL"] * r, dlg, zg)
        jac = (1 + zp) * np.interp(zp, zg, ddl) / r
        p_vol = np.interp(zp, zg, dvcg) / (1 + zp) / vol_norm * (zp <= IS.ZMAX_T)
        ka, _ = IS.agn_kernel(seed, zp, pix)
        p_agn = ka / (1 + zp) * (zp <= IS.ZMAX_T)
        m1s = d["m1det"] / (1 + zp)
        thG = dict(base)
        thG.update({k: v for k, v in pt.items() if k != "H0"})
        thA = dict(thG)
        thA["mu_G"] += CF.AGN_SHIFT["mu_G"]
        thA["mu_chi"] += CF.AGN_SHIFT["mu_chi"]
        m = {}
        for cls, th in (("G", thG), ("A", thA)):
            key = tuple(th.values())
            if key not in fns:
                CF.FAMILIES["_tmp"] = th
                fns[key] = CF.setup("historical", "_tmp")[0]
            m[cls] = np.exp(fns[key](m1s, q, d["chieff"], zp, "gal") + np.log1p(zp)) / jac
        F = IS.F
        targets[name] = {"GAL_class_vol": m["G"] * p_vol, "AGN_class_agncat": m["A"] * p_agn,
                         "host_dependent_mix": (1 - F) * m["G"] * p_vol + F * m["A"] * p_agn,
                         "equal_population_mix": (1 - F) * m["G"] * p_vol + F * m["G"] * p_agn,
                         "all_agn_hosted": m["G"] * p_agn}
        print("targets", name, flush=True)
    out = {"pilot": str(inj), "ndraw_pilot": nold, "mix_pilot": mix_old, "rows_target": rows_target,
           "what": __doc__.split("\n\n")[1].strip(), "mixtures": {}}
    for a, b, c in MIXES:
        qn = a * d["pdraw_population"] + b * d["pdraw_uniform"] + c * d["pdraw_targeted_agn"]
        rpp = float((qn / d["pdraw"]).sum() / nold)
        ndraw = rows_target / rpp
        rec = {"rows_per_proposal": rpp, "ndraw_for_rows_target": ndraw, "points": {}}
        for name, T in targets.items():
            rec["points"][name] = {}
            for k, t in T.items():
                s1 = (t / d["pdraw"]).sum() / nold
                s2 = ((t / qn) * (t / d["pdraw"])).sum() / nold
                neff = s1 ** 2 / s2 * ndraw
                rec["points"][name][k] = {"n_eff": float(neff), "N2_over_neff": float(1e6 / neff)}
        rec["worst"] = {k: max(rec["points"][p][k]["N2_over_neff"] for p in IS.POINTS)
                        for k in targets["design"]}
        out["mixtures"][f"{a:.2f}/{b:.2f}/{c:.2f}"] = rec
        print(f"mix {a:.2f}/{b:.2f}/{c:.2f}: ndraw {ndraw:.3g}  worst " +
              "  ".join(f"{k} {v:.2f}" for k, v in rec["worst"].items()) +
              "  | design " + "  ".join(f"{k} {v['N2_over_neff']:.2f}" for k, v in rec["points"]["design"].items()),
              flush=True)
    outp.write_text(json.dumps(out, indent=1))
    print("wrote", outp)


if __name__ == "__main__":
    main()
