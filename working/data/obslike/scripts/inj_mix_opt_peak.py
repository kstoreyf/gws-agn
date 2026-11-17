#!/usr/bin/env python
"""Choose the dedicated peak-family lane by reweighting a pilot.  (core venv; legacy gmd for the
proposal's mass/spin density)

    python inj_mix_opt_peak.py PILOT_FILE OUT.json [ROWS_TARGET]

The pilot's proposal is a mixture of legacy PopulationConfigs (components k) inside the three
branches.  A candidate keeps the components and changes their weights (u_k) and the branch
weights (a, b, c).  Its branch densities at a pilot row follow exactly from the stored ones:
    p_pop' = p_pop * M'/M,  p_tgt' = p_tgt * M'/M,  p_unif' = p_unif,
with M = sum_k w_k msp_k (pilot) and M' = sum_k u_k msp_k, msp_k = legacy _mass_spin_pdf.  Then,
as in inj_mix_opt.py, rows per proposal, N_eff per proposal, and at ROWS_TARGET rows the
N^2/N_eff (N = 1000) of every target of inj_stress.py for the peak family (POINTS_MOCK), for
s = 0 (equal-population mixtures), s = 1 and s = 2 (host-dependent mixtures, AGN class at
+5 s Msun, +0.10 s), each at f = 0.1, 0.3, 0.5, plus the per-class numbers.
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
import core_family as CF  # noqa: E402
import inj_stress as IS  # noqa: E402

PIN = Path("/hildafs/projects/phy230014p/magana/src/darksirens-2b86a2d")
BRANCH = [(0.62, 0.03, 0.35), (0.65, 0.10, 0.25), (0.72, 0.03, 0.25), (0.52, 0.03, 0.45)]
COMPW = [(0.35, 0.20, 0.20, 0.25), (0.50, 0.15, 0.15, 0.20), (0.40, 0.20, 0.20, 0.20),
         (0.30, 0.25, 0.25, 0.20), (0.45, 0.15, 0.20, 0.20), (0.40, 0.15, 0.25, 0.20),
         (0.55, 0.10, 0.15, 0.20), (0.40, 0.20, 0.25, 0.15)]
FS = (0.1, 0.3, 0.5)


def main():
    inj, outp = Path(sys.argv[1]), Path(sys.argv[2])
    rows_target = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0e6
    global BRANCH, COMPW
    if len(sys.argv) > 4:          # optional grid override: {"branch": [[a,b,c],...], "compw": [[...],...]}
        g = json.loads(sys.argv[4])
        BRANCH = [tuple(x) for x in g.get("branch", BRANCH)]
        COMPW = [tuple(x) for x in g.get("compw", COMPW)]
    import generate_dataset as G
    gmd = G.import_gmd(PIN)
    with h5py.File(inj, "r") as f:
        d = {k: np.asarray(f[k][:]) for k in ("m1det", "m2det", "dL", "chieff", "ra", "dec", "pdraw", "m1src",
                                              "m2src", "pdraw_population", "pdraw_uniform", "pdraw_targeted_agn")}
        nold = int(f.attrs["Ndraw"])
        comps = json.loads(f.attrs["proposal_population_json"])
        seed = int(json.loads(f.attrs["metadata_json"])["seed"]) // 1000
    comps = comps if isinstance(comps, list) else [comps]
    pops = [(float(c["weight"]), gmd.PopulationConfig(**{k: v for k, v in c.items() if k != "weight"})) for c in comps]
    qsrc = d["m2src"] / d["m1src"]
    msp = np.stack([G._chunked_pdraw(lambda s, pp=pp: gmd._mass_spin_pdf(d["m1src"][s], qsrc[s], d["chieff"][s], pp),
                                     d["pdraw"].size) for _, pp in pops])
    w_pilot = np.array([w for w, _ in pops])
    M = (w_pilot[:, None] * msp).sum(0)
    import healpy as hp
    q = d["m2det"] / d["m1det"]
    zg, dlg, dvcg = IS.zgrids()
    ddl = np.gradient(dlg, zg)
    sel = zg <= IS.ZMAX_T
    vol_norm = np.trapz((dvcg / (1 + zg))[sel], zg[sel]) * 4 * np.pi
    pix = hp.ang2pix(32, 0.5 * np.pi - d["dec"], d["ra"])
    base = CF.FAMILIES["mock"]
    fns, targets, kz = {}, {}, {}
    for name, pt in IS.POINTS_MOCK.items():
        h0 = pt.get("H0", IS.H0_FID)
        r = h0 / IS.H0_FID
        zp = np.interp(d["dL"] * r, dlg, zg)
        jac = (1 + zp) * np.interp(zp, zg, ddl) / r
        p_vol = np.interp(zp, zg, dvcg) / (1 + zp) / vol_norm * (zp <= IS.ZMAX_T)
        if h0 not in kz:
            kz[h0] = IS.agn_kernel(seed, zp, pix)[0] / (1 + zp) * (zp <= IS.ZMAX_T)
        p_agn = kz[h0]
        m1s = d["m1det"] / (1 + zp)
        mass = {}
        for s_ in (0, 1, 2):
            th = dict(base)
            th.update({k: v for k, v in pt.items() if k != "H0"})
            th["mu_G"] += s_ * CF.AGN_SHIFT["mu_G"]
            th["mu_chi"] += s_ * CF.AGN_SHIFT["mu_chi"]
            key = tuple(th.values())
            if key not in fns:
                CF.FAMILIES["_tmp"] = th
                fns[key] = CF.setup("historical", "_tmp")[0]
            mass[s_] = np.exp(fns[key](m1s, q, d["chieff"], zp, "gal") + np.log1p(zp)) / jac
        T = {"GAL_class": mass[0] * p_vol, "AGN_class_s1": mass[1] * p_agn, "AGN_class_s2": mass[2] * p_agn,
             "all_agn_s0": mass[0] * p_agn}
        for f in FS:
            T[f"s0_f{f}"] = (1 - f) * mass[0] * p_vol + f * mass[0] * p_agn
            T[f"s1_f{f}"] = (1 - f) * mass[0] * p_vol + f * mass[1] * p_agn
            T[f"s2_f{f}"] = (1 - f) * mass[0] * p_vol + f * mass[2] * p_agn
        targets[name] = T
        print("targets", name, flush=True)
    out = {"pilot": str(inj), "ndraw_pilot": nold, "components": comps, "rows_target": rows_target,
           "what": __doc__.split("\n\n")[1].strip(), "candidates": {}}
    mix_keys = [k for k in targets["design"] if k.startswith("s")]
    for (a, b, c) in BRANCH:
        for u in COMPW:
            Mp = (np.array(u)[:, None] * msp).sum(0)
            ratio = np.where(M > 0, Mp / M, 0.0)
            qn = a * d["pdraw_population"] * ratio + b * d["pdraw_uniform"] + c * d["pdraw_targeted_agn"] * ratio
            rpp = float((qn / d["pdraw"]).sum() / nold)
            ndraw = rows_target / rpp
            rec = {"branch": [a, b, c], "component_weights": list(u), "rows_per_proposal": rpp,
                   "ndraw_for_rows_target": ndraw, "points": {}}
            for name, T in targets.items():
                rec["points"][name] = {}
                for k, t in T.items():
                    s1 = (t / d["pdraw"]).sum() / nold
                    with np.errstate(divide="ignore", invalid="ignore"):
                        s2 = np.where(t > 0, (t / qn) * (t / d["pdraw"]), 0.0).sum() / nold
                    rec["points"][name][k] = float(1e6 / (s1 ** 2 / s2 * ndraw))
            rec["design_max_mix"] = max(rec["points"]["design"][k] for k in mix_keys)
            rec["stressed_max_mix"] = max(rec["points"][p][k] for p in targets for k in mix_keys)
            rec["design_classes"] = {k: rec["points"]["design"][k] for k in ("GAL_class", "AGN_class_s1",
                                                                           "AGN_class_s2", "all_agn_s0")}
            key = f"{a:.2f}/{b:.2f}/{c:.2f} | " + "/".join(f"{x:.2f}" for x in u)
            out["candidates"][key] = rec
            print(f"{key}: ndraw {ndraw:.3g} design max mix {rec['design_max_mix']:.2f} stressed max mix "
                  f"{rec['stressed_max_mix']:.2f} classes " + " ".join(f"{k} {v:.2f}" for k, v in
                                                                         rec["design_classes"].items()), flush=True)
    outp.write_text(json.dumps(out, indent=1))
    print("wrote", outp)


if __name__ == "__main__":
    main()
