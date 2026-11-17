#!/usr/bin/env python
"""The detected samples of the two mock families, side by side (any env with scipy/h5py/astropy).

    python describe_detected.py

For each realisation (100, 101, 102, 103, 105) and each of the equal-population and
host-dependent sets of the present family (``_corepop``) and the observed-like family
(``_obslike``):

  * fraction of detected mergers in the Gaussian peak: the mean over detected events of the
    membership probability w_G G(m1) / p(m1) under the event's own population (exact in
    expectation: the component flag and detection are independent given m1, since both
    components pair identically in core); for obslike files also the realised flag
    (truth/pop_component_peak);
  * redshift: median, 90th percentile, maximum; detected fraction per proposal;
  * median detected source-frame m1, m2, chirp mass and detector-frame chirp mass; median SNR;
  * the largest redshift any PE sample reaches at H0 = 100 (the largest trial H0), against
    the catalog edge z = 1.0 (constant-density plateau to 0.923) and darksirens' production
    grid end z = 1.10.

Writes results/detected_sample.json (all rows) and, for the paper's macros,
results/obslike_paper_values.json (design + realisation-100 table + 5-realisation spread).
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
OB = HERE.parent
sys.path.insert(0, str(OB.parent / "corepop" / "scripts"))
sys.path.insert(0, str(HERE))
import core_eval as CE  # noqa: E402
from core_family import FAMILIES, thetas  # noqa: E402  (pure-python constants; no darksirens import at module level)

DATA = Path("/hildafs/projects/phy230054p/magana/gws-agn-data-v3")
SEEDS = (100, 101, 102, 103, 105)
SETS = {("mock", "equal"): "events_corepop.h5",
        ("mock", "host_dependent"): "events_marked_dmu0p10_dmuG5_corepop.h5",
        ("obslike", "equal"): "events_obslike.h5",
        ("obslike", "host_dependent"): "events_marked_dmu0p10_dmuG5_obslike.h5"}
H0_MAX, OM0 = 100.0, 0.3075


def membership(cf, P, m1):
    pl = (CE.s_low(m1, P["m_min"], P["dm_min"]) * CE.s_high(m1, P["m_max"], P["dm_max"])
          * m1 ** (-P["alpha"]) / cf.n_pl)
    g = np.where((m1 >= 1) & (m1 <= 200), np.exp(-0.5 * ((m1 - cf.mu_G) / P["sigma_G"]) ** 2) / cf.n_g, 0.0)
    w_pl, w_g = P["v1"], 1.0 - P["v1"]
    return w_g * g / (w_pl * pl + w_g * g)


def z_of_dl_table(h0):
    from astropy.cosmology import FlatLambdaCDM
    c = FlatLambdaCDM(H0=h0, Om0=OM0)
    zg = np.linspace(0, 3.0, 30001)
    return zg, c.luminosity_distance(zg).value


def main():
    cfs = {fam: {cls: CE.ClosedForm(P["mu_G"], P["mu_chi"], P=P) for cls, P in thetas(fam).items()}
           for fam in FAMILIES}
    zg, dlg = z_of_dl_table(H0_MAX)
    rows = []
    for s in SEEDS:
        for (fam, kind), fn in SETS.items():
            p = DATA / f"seed{s}" / "events" / fn
            with h5py.File(p, "r") as f:
                t = {k: np.asarray(f["truth"][k]) for k in f["truth"].keys()}
                meta = json.loads(f.attrs["metadata_json"])
                dl_max = float(np.max(f["dL"][:]))
            m1, m2, z = t["m1src"], t["m2src"], t["z"]
            ht = t["host_type"]
            mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
            P = thetas(fam)
            pm = np.where((ht == 1) & (kind == "host_dependent"),
                          membership(cfs[fam]["agn"], P["agn"], m1), membership(cfs[fam]["gal"], P["gal"], m1))
            row = {"seed": s, "family": fam, "kind": kind, "file": fn, "n_events": int(m1.size),
                   "peak_share_detected_expected": float(pm.mean()),
                   "peak_share_detected_expected_sd": float(np.sqrt((pm * (1 - pm)).sum()) / m1.size),
                   "peak_share_detected_realised": (float(t["pop_component_peak"].mean())
                                                    if "pop_component_peak" in t else None),
                   "peak_share_population": 1.0 - FAMILIES[fam]["v1"],
                   "z_median": float(np.median(z)), "z_p90": float(np.quantile(z, 0.9)),
                   "z_max": float(z.max()),
                   "detected_fraction_per_proposal": float(meta["realised"]["detected_fraction"]),
                   "m1src_median": float(np.median(m1)), "m2src_median": float(np.median(m2)),
                   "mc_src_median": float(np.median(mc)), "mc_det_median": float(np.median(mc * (1 + z))),
                   "m1src_p10_p90": [float(np.quantile(m1, 0.1)), float(np.quantile(m1, 0.9))],
                   "m1src_max": float(m1.max()), "m2src_min": float(m2.min()),
                   "q_median": float(np.median(m2 / m1)),
                   "snr_obs_median": float(np.median(t["snr_obs"])),
                   "realised_f_agn": float(ht.mean()),
                   "pe_z_max_at_H0_100": float(np.interp(dl_max, dlg, zg))}
            if kind == "host_dependent":
                for c, nm in ((0, "gal"), (1, "agn")):
                    sel = ht == c
                    row[f"m1src_median_{nm}"] = float(np.median(m1[sel]))
                    row[f"chieff_mean_{nm}"] = float(np.mean(t["chieff"][sel]))
                    row[f"peak_share_expected_{nm}"] = float(pm[sel].mean())
            rows.append(row)
            print(s, fam, kind, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                                 if k in ("peak_share_detected_expected", "peak_share_detected_realised", "z_median",
                                          "z_max", "m1src_median", "m2src_median", "pe_z_max_at_H0_100")}, flush=True)
    out = {"what": __doc__.split("\n\n")[1].strip(), "rows": rows}
    (OB / "results" / "detected_sample.json").write_text(json.dumps(out, indent=1))

    keys = ("peak_share_detected_expected", "peak_share_detected_realised", "z_median", "z_p90", "z_max",
            "detected_fraction_per_proposal", "m1src_median", "m2src_median", "mc_src_median",
            "mc_det_median", "snr_obs_median", "pe_z_max_at_H0_100")
    table = {}
    for fam in FAMILIES:
        for kind in ("equal", "host_dependent"):
            rs = [r for r in rows if r["family"] == fam and r["kind"] == kind]
            r100 = next(r for r in rs if r["seed"] == 100)
            spread = {}
            for k in keys:
                v = [r[k] for r in rs if r[k] is not None]
                if v:
                    spread[k] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                                 "min": float(np.min(v)), "max": float(np.max(v))}
            table[f"{fam}_{kind}"] = {"realisation_100": {k: r100[k] for k in keys},
                                      "five_realisations": spread}
    paper = {
        "what": "values for the paper's table of simulated catalogs: the observed-like mass spectrum "
                "(design, GWTC-3 source) and the detected samples of both families",
        "source": "working/data/obslike/README.md; numbers from results/detected_sample.json",
        "design": json.loads((OB / "results" / "design_values.json").read_text()),
        "detected": table,
    }
    gof = OB / "results" / "detected_gof.json"
    if gof.exists():   # selection-model prediction of the detected peak share (obslike files)
        g = json.loads(gof.read_text())["files"]
        paper["peak_share_detected_predicted_obslike"] = {
            k: {str(r["seed"]): r["peak_share_predicted"] for r in g if r["kind"] == k}
            for k in ("equal", "host_dependent")}
    inj = OB / "results" / "injection_neff.json"
    if inj.exists():   # the existing injections under each family (seed 100, uniform-volume proxy)
        L = json.loads(inj.read_text())["lanes"]
        paper["existing_injections_N2_over_Neff_seed100_proxy"] = {
            lane: {k: v["N2_over_neff"] for k, v in L[lane]["families"].items()} for lane in L}
    (OB / "results" / "obslike_paper_values.json").write_text(json.dumps(paper, indent=1))
    print("wrote", OB / "results" / "detected_sample.json", OB / "results" / "obslike_paper_values.json")


if __name__ == "__main__":
    main()
