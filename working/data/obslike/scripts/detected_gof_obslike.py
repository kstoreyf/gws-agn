#!/usr/bin/env python
"""The detected events of every obslike file against core's density times selection. (jax env)

    sbatch --job-name=detgof run_cpu.sbatch jax detected_gof_obslike.py      (after core_family.py identity)

The prediction of corepop/scripts/detected_gof.py (imported, not copied): a proposal of source
chirp mass Mc at host j is detected with probability (1+z_j)^-1 Phi(rho_opt(Mc(1+z_j), dL(z_j)) - 8),
so the detected density of class t is p_core,t(m1, q) A_t(Mc), A_t built from the seed's own
complete catalog of tracer t.  Here p_core,t is core's obslike 2-D grid
(cache/core_tables_obslike.npz): GAL class (mu_G 34, mu_chi 0), AGN class of the host-dependent
set (mu_G 39, mu_chi 0.10).  Per file: KS of m1, q, m2, chi_eff (per class for the
host-dependent set), the predicted and realised share of detected events in the Gaussian peak,
and for the host-dependent set the selection-corrected AGN share (V5) and the class offsets.
Writes results/detected_gof.json.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
OB = HERE.parent
sys.path.insert(0, str(OB.parent / "corepop" / "scripts"))
sys.path.insert(0, str(HERE))
import detected_gof as DG  # noqa: E402  (corepop's functions; its main() is not called)
import core_eval as CE  # noqa: E402
from core_family import thetas  # noqa: E402

G = DG.G
SEEDS = (100, 101, 102, 103, 105)
FILES = {"equal": ("events_obslike.h5", 0.30), "host_dependent": ("events_marked_dmu0p10_dmuG5_obslike.h5", 0.30)}


def peak_prob(m1, cls):
    P = thetas("obslike")[cls]
    cf = CE.ClosedForm(P["mu_G"], P["mu_chi"], P=P)
    pl = (CE.s_low(m1, P["m_min"], P["dm_min"]) * CE.s_high(m1, P["m_max"], P["dm_max"])
          * m1 ** (-P["alpha"]) / cf.n_pl)
    g = np.exp(-0.5 * ((m1 - P["mu_G"]) / P["sigma_G"]) ** 2) / cf.n_g
    return (1 - P["v1"]) * g / (P["v1"] * pl + (1 - P["v1"]) * g)


def predicted_peak_share(lnp2d, m1d, qd, A, cls):
    M1, Q = np.meshgrid(m1d, qd, indexing="ij")
    p = np.exp(lnp2d) * np.interp(G.mc_of_m1q(M1, Q), DG.MC, A)
    pm1 = np.trapz(p, qd, axis=1)
    return float(np.trapz(pm1 * peak_prob(m1d, cls), m1d) / np.trapz(pm1, m1d))


def chi_cdf(T, pop):
    g = np.linspace(-1.0, 1.0, T[f"pchi_{pop}"].size)
    d = T[f"pchi_{pop}"]
    c = np.concatenate([[0.0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(g))])
    c /= c[-1]
    return lambda v: np.interp(v, g, c)


def main():
    gmd = G.import_gmd(DG.PIN)
    grids = gmd._cosmology_grids(gmd._build_cosmology(G.H0_FID, G.OM0_FID, G.W0_FID, G.WA_FID), G.ZMAX_GRID)
    T = np.load(OB / "cache" / "core_tables_obslike.npz")
    m1d, qd = T["m1d"], T["qd"]
    out = {"what": __doc__.split("\n\n")[1].strip(), "files": []}
    pooled = {"dchi": [], "dm1": [], "dm1_pred": []}
    for s in SEEDS:
        A = {t: DG.acceptance_curve(s, t, grids) for t in ("gal", "agn")}
        for kind, (fn, f) in FILES.items():
            tr, meta = DG.load_truth(DG.DATA / f"seed{s}" / "events" / fn)
            row = {"seed": s, "kind": kind, "file": fn}
            if kind == "equal":
                Am = (1 - f) * A["gal"] + f * A["agn"]
                pm = DG.pred_marginals(T["lnp2d_gal"], m1d, qd, Am)
                row["ks"], _ = DG.ks_block(tr, np.ones(tr["z"].size, bool), pm, chi_cdf(T, "gal"))
                row["peak_share_predicted"] = predicted_peak_share(T["lnp2d_gal"], m1d, qd, Am, "gal")
            else:
                pm = {"gal": DG.pred_marginals(T["lnp2d_gal"], m1d, qd, A["gal"]),
                      "agn": DG.pred_marginals(T["lnp2d_agn"], m1d, qd, A["agn"])}
                row["ks"] = {}
                for c, cls in ((0, "gal"), (1, "agn")):
                    row["ks"][cls], _ = DG.ks_block(tr, tr["host_type"] == c, pm[cls], chi_cdf(T, cls))
                share = f * pm["agn"]["E"] / (f * pm["agn"]["E"] + (1 - f) * pm["gal"]["E"])
                n_agn = int(tr["host_type"].sum())
                sd = np.sqrt(1000 * share * (1 - share))
                row["V5_selection_corrected"] = {"planted_f_agn": f, "expected_detected_agn_share": float(share),
                                                 "realised_detected_agn_share": n_agn / 1000,
                                                 "binomial_pull": float((n_agn - 1000 * share) / sd)}
                sh_g = predicted_peak_share(T["lnp2d_gal"], m1d, qd, A["gal"], "gal")
                sh_a = predicted_peak_share(T["lnp2d_agn"], m1d, qd, A["agn"], "agn")
                row["peak_share_predicted"] = float((1 - share) * sh_g + share * sh_a)
                ag, ga = tr["host_type"] == 1, tr["host_type"] == 0
                dchi = float(tr["chieff"][ag].mean() - tr["chieff"][ga].mean())
                dm1 = float(tr["m1src"][ag].mean() - tr["m1src"][ga].mean())
                row["offsets"] = {"dmean_chieff": dchi, "dmean_m1src": dm1,
                                  "dmean_m1src_predicted": pm["agn"]["mean_m1"] - pm["gal"]["mean_m1"]}
                pooled["dchi"].append(dchi)
                pooled["dm1"].append(dm1)
                pooled["dm1_pred"].append(row["offsets"]["dmean_m1src_predicted"])
            row["peak_share_realised"] = float(tr["pop_component_peak"].mean())
            out["files"].append(row)
            print(json.dumps(row), flush=True)
    pv = [v["pvalue"] for r in out["files"] for blk in ([r["ks"]] if r["kind"] == "equal" else r["ks"].values())
          for v in blk.values()]
    out["n_ks_tests"] = len(pv)
    out["ks_pvalues_min"] = float(min(pv))
    out["ks_frac_below_0p05"] = float(np.mean(np.array(pv) < 0.05))
    out["pooled_host_dependent"] = {k: float(np.mean(v)) for k, v in pooled.items()}
    (OB / "results" / "detected_gof.json").write_text(json.dumps(out, indent=1))
    print("min p", out["ks_pvalues_min"], "frac<0.05", out["ks_frac_below_0p05"], out["pooled_host_dependent"])


if __name__ == "__main__":
    main()
