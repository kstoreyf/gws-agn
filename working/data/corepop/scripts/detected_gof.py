#!/usr/bin/env python
"""The detected events of every corepop file against core's density times selection. (jax env)

    export PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-2b86a2d
    JAX_PLATFORMS=cpu python detected_gof.py          (after core_eval.py tables)

PREDICTION.  A proposal at (m1, q, chi, host j) is detected with probability
    (1 + z_j)^(gamma - 1) * Phi(rho_opt(Mc (1 + z_j), dL(z_j)) - 8)        (gamma = 0)
-- the generator's own rule (conventions (a), v3; rho_obs = rho_opt + N(0, 1)) -- which
depends on the masses only through the source chirp mass Mc and not at all on chi.  With
hosts uniform within tracer t of the catalog, the detected density of class t is
    p_det,t(m1, q) ~ p_core,t(m1, q) A_t(Mc(m1, q)),   A_t(Mc) = mean_j (1+z_j)^-1 Phi(...)
and the detected chi_eff of class t is core's truncated Gaussian itself.  A_t is built from
the seed's own complete catalog (its true-z histogram, 4000 bins), p_core,t is core's
2-D grid (cache/core_tables.npz).  Mixed and all-AGN/all-GAL sets: one population, the
planted host mixture of A_t.  Host-dependent set: GAL class under (mu_G 35, mu_chi 0),
AGN class under (mu_G 40, mu_chi 0.1), and the expected detected AGN share
f E_AGN / (f E_AGN + (1 - f) E_GAL), E_t = int p_core,t A_t -- V5 for that set.

Per file: KS p-values of m1, q, m2 and chi_eff (per host class for the host-dependent
set), the realised offsets between host classes, and the correspondence with the legacy
file of the same kind (events whose host, z, m1 and chi_eff are bit-identical).
Writes results/detected_gof.json and cache/figure_detected.npz.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import h5py
import numpy as np
from scipy import stats
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
CP = HERE.parent
sys.path.insert(0, str(CP.parent))
import generate_dataset as G  # noqa: E402

PIN = Path("/hildafs/projects/phy230014p/magana/src/darksirens-2b86a2d")
DATA = Path("/hildafs/projects/phy230054p/magana/gws-agn-data-v3")
SEEDS = (100, 101, 102, 103, 105)
KINDS = {"mixed": ("events_corepop.h5", "events.h5", 0.30),
         "pureagn": ("events_pureagn_corepop.h5", "events_pureagn.h5", 1.0),
         "puregal": ("events_puregal_corepop.h5", "events_puregal.h5", 0.0),
         "marked": ("events_marked_dmu0p10_dmuG5_corepop.h5",
                    "events_marked_dmu0p10_dmuG5.h5", 0.30)}
ZBINS = np.linspace(0.0, 1.02, 4001)
MC = np.geomspace(2.0, 120.0, 3000)


def acceptance_curve(seed, tracer, grids):
    """A_t(Mc) on MC: mean over the tracer's complete catalog of (1+z)^-1 Phi(rho_opt-8)."""
    with h5py.File(DATA / f"seed{seed}" / "catalogs" / f"catalog_{tracer}_complete.h5", "r") as f:
        z = np.asarray(f["z"][:], float)
    h, _ = np.histogram(z, bins=ZBINS)
    n_tot = z.size
    del z
    zc = 0.5 * (ZBINS[1:] + ZBINS[:-1])
    dl = np.interp(zc, grids["z"], grids["dl"])
    rho = G.rho_opt_of_mc_dl(MC[:, None] * (1.0 + zc[None, :]), dl[None, :])
    w = h * (1.0 + zc) ** (G.GAMMA - 1.0)
    return (ndtr((rho - G.SNR_THRESHOLD) / G.SIGMA_RHO) * w[None, :]).sum(axis=1) / n_tot


def pred_marginals(lnp2d, m1d, qd, A):
    """Detected (m1, q, m2) densities on the grid for core's p(m1, q) times A(Mc)."""
    M1, Q = np.meshgrid(m1d, qd, indexing="ij")
    mc = G.mc_of_m1q(M1, Q)
    p = np.exp(lnp2d) * np.interp(mc, MC, A)
    p_m1 = np.trapz(p, qd, axis=1)
    p_q = np.trapz(p, m1d, axis=0)
    E = float(np.trapz(p_m1, m1d))
    # m2 = q m1: cell masses binned into fine m2 bins
    wq = np.gradient(qd)
    wm = np.gradient(m1d)
    m2e = np.linspace(4.0, 100.0, 9601)
    h, _ = np.histogram((M1 * Q).ravel(), bins=m2e, weights=(p * wm[:, None] * wq[None, :]).ravel())
    return {"m1": (m1d, p_m1 / E), "q": (qd, p_q / E), "m2_edges": m2e, "m2_mass": h / h.sum(),
            "E": E, "mean_m1": float(np.trapz(m1d * p_m1, m1d) / E)}


def cdf_fn(kind, pm):
    if kind == "m2":
        e, w = pm["m2_edges"], pm["m2_mass"]
        c = np.concatenate([[0.0], np.cumsum(w)])
        return lambda v: np.interp(v, e, c)
    g, d = pm[kind]
    c = np.concatenate([[0.0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(g))])
    c /= c[-1]
    return lambda v: np.interp(v, g, c)


def chi_cdf(T, pop):
    g, d = T["chig"], T[f"pchi_historical_{pop}"]
    c = np.concatenate([[0.0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(g))])
    c /= c[-1]
    return lambda v: np.interp(v, g, c)


def ks_block(tr, sel, pm, chicdf):
    out = {}
    vals = {"m1": tr["m1src"][sel], "q": tr["q"][sel], "m2": tr["m2src"][sel],
            "chi": tr["chieff"][sel]}
    for k, v in vals.items():
        f = chicdf if k == "chi" else cdf_fn(k, pm)
        r = stats.kstest(v, f)
        out[k] = {"D": float(r.statistic), "pvalue": float(r.pvalue)}
    return out, int(sel.sum())


def correspondence(new, old):
    """Events of the corepop file that are bit-identical, in their latent draw, to an event
    of the legacy file (same host, z, m1src, chi_eff, q)."""
    def keys(t):
        return {(int(a), int(b), float(c), float(d), float(e), float(g)) for a, b, c, d, e, g in
                zip(t["host_type"], t["host_index"], t["z"], t["m1src"], t["chieff"], t["q"])}
    kn, ko = keys(new), keys(old)
    both = kn & ko
    # leading run of positionally identical events (shares the PE noise too)
    same = (new["host_index"] == old["host_index"][:new["host_index"].size]) & \
           (new["m1src"] == old["m1src"]) & (new["q"] == old["q"]) & (new["z"] == old["z"])
    lead = int(np.argmin(same)) if not same.all() else int(same.size)
    # same proposal, q moved by the core route (same host, z, m1, chi; different q)
    k3 = lambda t: {(int(a), int(b), float(c), float(d), float(e)) for a, b, c, d, e in  # noqa: E731
                    zip(t["host_type"], t["host_index"], t["z"], t["m1src"], t["chieff"])}
    return {"n_identical_latent_draw": len(both), "n_same_proposal_q_moved": len(k3(new) & k3(old)) - len(both),
            "n_only_in_corepop": len(kn) - len(k3(new) & k3(old)),
            "leading_events_positionally_identical": lead}


def load_truth(p):
    with h5py.File(p, "r") as f:
        t = {k: np.asarray(f["truth"][k]) for k in f["truth"].keys()}
        meta = json.loads(f.attrs["metadata_json"])
    return t, meta


def main():
    gmd = G.import_gmd(PIN)
    grids = gmd._cosmology_grids(gmd._build_cosmology(G.H0_FID, G.OM0_FID, G.W0_FID, G.WA_FID),
                                 G.ZMAX_GRID)
    T = np.load(CP / "cache" / "core_tables.npz")
    m1d, qd = T["m1d"], T["qd"]
    chicdf = {p: chi_cdf(T, p) for p in ("gal", "agn")}
    out = {"what": __doc__.split("\n\n")[1].strip(), "files": [], "marked_offsets": {}}
    fig = {}
    pooled = {"gal": {k: [] for k in ("m1src", "q", "m2src", "chieff")},
              "agn": {k: [] for k in ("m1src", "q", "m2src", "chieff")}}
    pred_pool = {"gal": [], "agn": []}
    for s in SEEDS:
        A = {t: acceptance_curve(s, t, grids) for t in ("gal", "agn")}
        print(f"[seed {s}] acceptance curves built", flush=True)
        for kind, (fn, fn_old, f_agn) in KINDS.items():
            p = DATA / f"seed{s}" / "events" / fn
            tr, meta = load_truth(p)
            row = {"seed": s, "kind": kind, "file": fn,
                   "population_code": meta.get("population_code", {}).get("code"),
                   "n_detected_with_q_redrawn": meta.get("population_code", {}).get(
                       "n_detected_with_q_redrawn"),
                   "core_route_tally": meta.get("population_code", {}).get("tally_all_proposals")}
            ht = tr["host_type"]
            if kind == "marked":
                pm = {c: pred_marginals(T[f"lnp2d_{c}"], m1d, qd, A[c]) for c in ("gal", "agn")}
                row["ks"] = {}
                for c, code in (("gal", 0), ("agn", 1)):
                    row["ks"][c], n_c = ks_block(tr, ht == code, pm[c], chicdf[c])
                    row["ks"][c]["n_events"] = n_c
                    for k, kk in (("m1src", "m1src"), ("q", "q"), ("m2src", "m2src"),
                                  ("chieff", "chieff")):
                        pooled[c][k].append(tr[kk][ht == code])
                    pred_pool[c].append((n_c, pm[c]))
                share = f_agn * pm["agn"]["E"] / (f_agn * pm["agn"]["E"] + (1 - f_agn) * pm["gal"]["E"])
                n_agn = int(ht.sum())
                sd = np.sqrt(1000 * share * (1 - share))
                row["V5_selection_corrected"] = {
                    "planted_f_agn": f_agn, "expected_detected_agn_share": float(share),
                    "realised_detected_agn_share": n_agn / 1000,
                    "binomial_pull": float((n_agn - 1000 * share) / sd),
                    "pass_abs_pull_lt_4": bool(abs((n_agn - 1000 * share) / sd) < 4)}
                g, a = ht == 0, ht == 1
                dchi = float(tr["chieff"][a].mean() - tr["chieff"][g].mean())
                dchi_se = float(np.sqrt(tr["chieff"][a].var(ddof=1) / a.sum()
                                        + tr["chieff"][g].var(ddof=1) / g.sum()))
                dm1 = float(tr["m1src"][a].mean() - tr["m1src"][g].mean())
                dm1_se = float(np.sqrt(tr["m1src"][a].var(ddof=1) / a.sum()
                                       + tr["m1src"][g].var(ddof=1) / g.sum()))
                row["offsets_agn_minus_gal"] = {
                    "mean_chieff": dchi, "mean_chieff_se": dchi_se,
                    "mean_chieff_expected": 0.10,
                    "mean_m1src": dm1, "mean_m1src_se": dm1_se,
                    "mean_m1src_expected_detected": pm["agn"]["mean_m1"] - pm["gal"]["mean_m1"],
                    "median_m1src": float(np.median(tr["m1src"][a]) - np.median(tr["m1src"][g]))}
            else:
                lp = T["lnp2d_gal"]
                Amix = (1 - f_agn) * A["gal"] + f_agn * A["agn"]
                pm = pred_marginals(lp, m1d, qd, Amix)
                row["ks"], n_c = ks_block(tr, np.ones(ht.size, bool), pm, chicdf["gal"])
                row["realised_f_agn"] = float(ht.mean())
            # the legacy file of the same kind, if it exists: correspondence + KS of its m2
            po = DATA / f"seed{s}" / "events" / fn_old
            if po.exists():
                to, _ = load_truth(po)
                row["correspondence_with_legacy"] = correspondence(tr, to)
                if kind != "marked":
                    r = stats.kstest(to["m2src"], cdf_fn("m2", pm))
                    row["legacy_file_m2_ks_vs_core_prediction"] = {"D": float(r.statistic),
                                                                    "pvalue": float(r.pvalue)}
            row["min_ks_pvalue"] = float(min(v["pvalue"] for blk in (
                row["ks"].values() if kind == "marked" else [row["ks"]])
                for k, v in blk.items() if k != "n_events"))
            out["files"].append(row)
            print(f"[seed {s} {kind}] min KS p {row['min_ks_pvalue']:.3g}"
                  + (f"  share exp {row['V5_selection_corrected']['expected_detected_agn_share']:.3f}"
                     f" real {row['V5_selection_corrected']['realised_detected_agn_share']:.3f}"
                     if kind == "marked" else ""), flush=True)
    # pooled host-dependent set (5 seeds): offsets + figure data
    for c in ("gal", "agn"):
        for k in pooled[c]:
            pooled[c][k] = np.concatenate(pooled[c][k])
    g, a = pooled["gal"], pooled["agn"]
    out["marked_offsets"]["pooled_5_seeds"] = {
        "n_gal": int(g["m1src"].size), "n_agn": int(a["m1src"].size),
        "mean_chieff_agn_minus_gal": float(a["chieff"].mean() - g["chieff"].mean()),
        "mean_chieff_se": float(np.sqrt(a["chieff"].var(ddof=1) / a["chieff"].size
                                        + g["chieff"].var(ddof=1) / g["chieff"].size)),
        "mean_chieff_expected": 0.10,
        "mean_m1src_agn_minus_gal": float(a["m1src"].mean() - g["m1src"].mean()),
        "mean_m1src_se": float(np.sqrt(a["m1src"].var(ddof=1) / a["m1src"].size
                                       + g["m1src"].var(ddof=1) / g["m1src"].size)),
        "mean_m1src_expected_detected": float(
            sum(n * pm["mean_m1"] for n, pm in pred_pool["agn"]) / sum(n for n, _ in pred_pool["agn"])
            - sum(n * pm["mean_m1"] for n, pm in pred_pool["gal"]) / sum(n for n, _ in pred_pool["gal"])),
        "note": "population-level offsets are exactly +5 in mu_G and +0.10 in mu_chi; the "
                "detected mean-m1 offset is smaller (the shared power law and selection "
                "damp it) and is compared with its own prediction, never with +5"}
    for c in ("gal", "agn"):
        ntot = sum(n for n, _ in pred_pool[c])
        for k in ("m1", "q"):
            gr = pred_pool[c][0][1][k][0]
            fig[f"pred_{c}_{k}_grid"] = gr
            fig[f"pred_{c}_{k}_dens"] = sum(n * pm[k][1] for n, pm in pred_pool[c]) / ntot
        e = pred_pool[c][0][1]["m2_edges"]
        fig["pred_m2_edges"] = e
        fig[f"pred_{c}_m2_mass"] = sum(n * pm["m2_mass"] for n, pm in pred_pool[c]) / ntot
        for k in ("m1src", "q", "m2src", "chieff"):
            fig[f"det_{c}_{k}"] = pooled[c][k]
    np.savez_compressed(CP / "cache" / "figure_detected.npz", **fig)
    (CP / "results" / "detected_gof.json").write_text(json.dumps(out, indent=1))
    print("wrote", CP / "results" / "detected_gof.json")


if __name__ == "__main__":
    main()
