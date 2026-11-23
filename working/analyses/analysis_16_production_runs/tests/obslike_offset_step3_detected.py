#!/usr/bin/env python
"""Step 3 of the obslike_f0p30_s1 H0-offset diagnosis: are the detected mergers of each host class what the class
population and the injection set predict? No catalog likelihood, input cosmology (H0 = 67.74). (core venv)

    JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 PYTHONPATH= <core venv python> tests/obslike_offset_step3_detected.py

Per set, realization and class, the prediction for detected mergers is the set's injection file reweighted to the class
population, with the weights of the selection term (data/obslike/scripts/inj_stress.py's convention, which reproduces the
production N_eff): in the canonical basis (m1det, q, dL, chi_eff, sky) the injections store,
    w_k = p_pop,c(m1src, q, chi | Lambda_c) p_z,c(z, Omega) / [(1+z) dL'(z)] / pdraw_k,
p_pop,c core's powerlaw+peak (data/obslike/scripts/core_family.setup, 'historical' = per-sample pairing norm, the runs'
setting) with its rate factor removed, p_z,c = (1+z)^-1 x
  galaxy class: the realization's complete galaxy survey's global n(z) (photo-z histogram smoothed by the kernel width;
               isotropic; GAL_PRIOR = "vol" gives uniform in comoving volume instead -- that version over-predicts mergers
               at z < 0.04, where both catalogs hold 0.36-0.73 of the uniform density: results/..._galvol.json),
  AGN class: the realization's complete AGN survey, sum_j N(z; z_j, dz_j) / Omega_pix in the injection's pixel (nside 32).
Detected quantities compared (true values, group `truth`): m1src, q, m2src, chi_eff, z, dL, snr_obs, snr_true; for an
injection, snr_true = rho_opt(Mc_det, dL) (generator's rho_opt_of_mc_dl) and snr_obs one draw of N(rho_opt, 1) truncated to
>= 8 (its exact law given detection). Pooled over realizations 100-103, 105: each realization's class weights normalised to
its number of class-c events, so the pooled reference is the event-count mixture of the per-realization predictions.
Statistics: event mean vs predicted mean (z = diff / (sd_pred / sqrt n)), KS (D, p with n = n_events) and Anderson-Darling
(A^2 on u = F_ref(x), asymptotic p, Marsaglia & Marsaglia 2004) against the weighted reference CDF; per realization the m1
and chi_eff mean pulls. Sets: obslike_f0p30_s1, obslike_f0p30_s0, peak_f0p30_s1.
Output: tests/results/obslike_offset_step3_detected.json
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import h5py
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import kstwo

A16 = Path(__file__).resolve().parents[1]
DATA = A16.parents[1] / "data"
sys.path.insert(0, str(DATA / "obslike" / "scripts"))
import core_family as CF  # noqa: E402

H0_FID, OM0 = 67.74, 0.3075
ZMAX_T = 1.10
SNR_REF = 6.278363879917771
THR = 8.0
REAL = [100, 101, 102, 103, 105]
SETS = {
    "obslike_f0p30_s1": dict(events="events_marked_dmu0p10_dmuG5_obslike.h5", inj="injections_targeted_obslike.h5",
                             family="obslike", agn_shift=1.0),
    "obslike_f0p30_s0": dict(events="events_obslike.h5", inj="injections_targeted_obslike.h5",
                             family="obslike", agn_shift=0.0),
    "peak_f0p30_s1": dict(events="events_marked_dmu0p10_dmuG5_corepop.h5", inj="injections_targeted_peak.h5",
                          family="mock", agn_shift=1.0),
}
GAL_PRIOR = "catalog"   # galaxy class's p(z): the galaxy survey's global n(z) ("catalog") or uniform in volume ("vol")
QUANT = ["m1src", "q", "m2src", "chieff", "z", "dL", "snr_obs", "snr_true", "mc_det"]


def zgrids():
    from astropy.cosmology import FlatLambdaCDM
    c = FlatLambdaCDM(H0=H0_FID, Om0=OM0)
    z = np.linspace(0.0, 3.0, 60001)
    return z, c.luminosity_distance(z).value, c.differential_comoving_volume(z).value


def agn_kernel(seed, z, pix, chunk=50_000):
    with h5py.File(DATA / f"seed{seed}" / "surveys" / "survey_agn_complete_ns32.h5", "r") as f:
        zg, dz, ng = (np.asarray(f[k][:]) for k in ("zgals", "dzgals", "ngals"))
        nside = int(f.attrs["nside"])
    import healpy as hp
    valid = np.arange(zg.shape[1])[None, :] < ng[:, None]
    norm = float(((1.0 + zg) ** -1.0)[valid].sum())
    apix = hp.nside2pixarea(nside)
    out = np.empty(z.size)
    for i in range(0, z.size, chunk):
        s = slice(i, min(i + chunk, z.size))
        r = pix[s]
        zz = z[s][:, None]
        k = np.exp(-0.5 * ((zz - zg[r]) / dz[r]) ** 2) / (np.sqrt(2 * np.pi) * dz[r])
        out[s] = np.where(valid[r], k, 0.0).sum(axis=1)
    return out / apix / norm


def gal_nz(seed, edges):
    """The realization's complete galaxy survey's redshift distribution (all pixels, unit weights): counts per bin of
    the catalog's photo-z, smoothed by the kernel width 0.003 (1 + z) -> density per unit z (unnormalised)."""
    from scipy.ndimage import gaussian_filter1d
    H = np.zeros(edges.size - 1)
    with h5py.File(DATA / f"seed{seed}" / "surveys" / "survey_gal_complete_ns32.h5", "r") as f:
        ng = f["ngals"][:]
        zg = f["zgals"]
        for i0 in range(0, ng.size, 512):
            blk = zg[i0:i0 + 512]
            m = np.arange(blk.shape[1])[None, :] < ng[i0:i0 + 512, None]
            H += np.histogram(blk[m], bins=edges)[0]
    dzb = edges[1] - edges[0]
    return gaussian_filter1d(H / dzb, 0.0033 / dzb, mode="constant")


def ad_pvalue(A2):
    """Asymptotic P(A^2 > A2) for the Anderson-Darling statistic (Marsaglia & Marsaglia 2004, ADinf)."""
    z = A2
    if z < 2:
        v = np.exp(-1.2337141 / z) / np.sqrt(z) * (2.00012 + (.247105 - (.0649821 - (.0347962 - (.011672 - .00168691 * z)
                                                                                    * z) * z) * z) * z)
    else:
        v = np.exp(-np.exp(1.0776 - (2.30695 - (.43424 - (.082433 - (.008056 - .0003146 * z) * z) * z) * z) * z))
    return float(1.0 - v)


def compare(x_ev, x_ref, w_ref):
    x_ev = np.sort(np.asarray(x_ev, float))
    o = np.argsort(x_ref)
    xr, wr = x_ref[o], w_ref[o]
    cw = np.cumsum(wr)
    cw /= cw[-1]
    F = lambda v: np.interp(v, xr, cw, left=0.0, right=1.0)  # noqa: E731
    n = x_ev.size
    u = np.clip(F(x_ev), 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1)
    D = float(max(np.max(i / n - u), np.max(u - (i - 1) / n)))
    A2 = float(-n - np.sum((2 * i - 1) / n * (np.log(u) + np.log(1 - u[::-1]))))
    wsum = w_ref.sum()
    mu = float(np.sum(w_ref * x_ref) / wsum)
    sd = float(np.sqrt(np.sum(w_ref * (x_ref - mu) ** 2) / wsum))
    neff = float(wsum ** 2 / np.sum(w_ref ** 2))
    m_ev = float(x_ev.mean())
    return {"n_events": int(n), "mean_events": m_ev, "mean_pred": mu, "sd_pred": sd,
            "z_mean": (m_ev - mu) / (sd / np.sqrt(n)), "ks_D": D, "ks_p": float(kstwo.sf(D, n)),
            "ad_A2": A2, "ad_p": ad_pvalue(A2), "median_events": float(np.median(x_ev)),
            "median_pred": float(np.interp(0.5, cw, xr)), "n_eff_reference": neff}


def main():
    t0 = time.time()
    zg, dlg, dvcg = zgrids()
    ddl = np.gradient(dlg, zg)
    rate = (1.0 + zg) ** -1.0
    sel = zg <= ZMAX_T
    vol_norm = np.trapz((dvcg * rate)[sel], zg[sel]) * 4 * np.pi
    import healpy as hp
    rng = np.random.default_rng(20261123)
    lp_cache = {}
    out = {"what": __doc__.split("\n\n")[1].strip(), "gal_prior": GAL_PRIOR, "sets": {}}
    groups = {}
    for label, S in SETS.items():
        groups.setdefault(S["inj"], []).append(label)
    for inj_name, labels in groups.items():
        for label in labels:
            S = SETS[label]
            key_lp = (S["family"], S["agn_shift"])
            if key_lp not in lp_cache:
                lp_cache[key_lp] = CF.setup("historical", S["family"], S["agn_shift"])
        ref = {q: [] for q in QUANT}
        pooled = {lab: {c: {"ev": {q: [] for q in QUANT}, "w": []} for c in ("gal", "agn")} for lab in labels}
        per_real = {lab: {} for lab in labels}
        for r in REAL:
            with h5py.File(DATA / f"seed{r}" / "injections" / inj_name, "r") as f:
                d = {k: np.asarray(f[k][:]) for k in ("m1det", "m2det", "dL", "chieff", "ra", "dec", "pdraw", "z")}
            z = d["z"]
            d["q"] = d["m2det"] / d["m1det"]
            d["m1src"] = d["m1det"] / (1 + z)
            d["m2src"] = d["m2det"] / (1 + z)
            d["mc_det"] = d["m1det"] * d["q"] ** 0.6 / (1 + d["q"]) ** 0.2
            d["snr_true"] = SNR_REF * (d["mc_det"] / 30.0) ** (5 / 6) * (1000.0 / d["dL"])
            a = ndtr(THR - d["snr_true"])
            u = rng.random(z.size)
            d["snr_obs"] = d["snr_true"] + ndtri(np.clip(a + u * (1 - a), 1e-300, 1 - 1e-16))
            jac = (1 + z) * np.interp(z, zg, ddl)
            p_vol = np.interp(z, zg, dvcg) * (1 + z) ** -1.0 / vol_norm * (z <= ZMAX_T)
            if GAL_PRIOR == "catalog":
                ed = np.linspace(0.0, 1.2, 1201)
                nz = gal_nz(r, ed)
                p_vol = np.interp(z, 0.5 * (ed[1:] + ed[:-1]), nz) * (1 + z) ** -1.0 * (z <= ZMAX_T)
            pix = hp.ang2pix(32, 0.5 * np.pi - d["dec"], d["ra"])
            p_agn = agn_kernel(r, z, pix) * (1 + z) ** -1.0 * (z <= ZMAX_T)
            base = 1.0 / d["pdraw"] / jac
            for q in QUANT:
                ref[q].append(d[q])
            print(f"[{time.time() - t0:.0f}s] loaded {inj_name} r{r}: {z.size} rows", flush=True)
            for label in labels:
                S = SETS[label]
                lp, info = lp_cache[(S["family"], S["agn_shift"])]
                ws = {}
                for c in ("gal", "agn"):
                    lm = lp(d["m1src"], d["q"], d["chieff"], z, c) + np.log1p(z)
                    pz = p_vol if c == "gal" else p_agn
                    ws[c] = np.where(np.isfinite(lm), np.exp(lm), 0.0) * pz * base
                with h5py.File(DATA / f"seed{r}" / "events" / S["events"], "r") as f:
                    T = {k: np.asarray(f["truth/" + k][:]) for k in
                         ("m1src", "m2src", "q", "chieff", "z", "dl", "snr_obs", "snr_true", "host_type", "m1det")}
                T["dL"] = T["dl"]
                T["mc_det"] = T["m1det"] * T["q"] ** 0.6 / (1 + T["q"]) ** 0.2
                row = {}
                for c, ht in (("gal", 0), ("agn", 1)):
                    m = T["host_type"] == ht
                    n = int(m.sum())
                    w = ws[c]
                    pooled[label][c]["w"].append(w / w.sum() * n)
                    for q in QUANT:
                        pooled[label][c]["ev"][q].append(T[q][m])
                    row[c] = {"n": n, "n_eff_weights": float(w.sum() ** 2 / (w ** 2).sum())}
                    for q in ("m1src", "chieff", "z", "snr_obs"):
                        cc = compare(T[q][m], d[q], w)
                        row[c][q] = {k: cc[k] for k in ("mean_events", "mean_pred", "z_mean", "ks_p")}
                per_real[label][str(r)] = row
                print(f"[{time.time() - t0:.0f}s] {label} r{r}: " + " | ".join(
                    f"{c} n={row[c]['n']} Neff={row[c]['n_eff_weights']:.0f} m1 z={row[c]['m1src']['z_mean']:+.2f} "
                    f"chi z={row[c]['chieff']['z_mean']:+.2f} zz={row[c]['z']['z_mean']:+.2f} "
                    f"snr z={row[c]['snr_obs']['z_mean']:+.2f}" for c in ("gal", "agn")), flush=True)
            del d
        R = {q: np.concatenate(ref[q]) for q in QUANT}
        del ref
        for label in labels:
            S = SETS[label]
            info = lp_cache[(S["family"], S["agn_shift"])][1]
            res = {"per_realization": per_real[label], "pooled": {},
                   "core_info": {k: v for k, v in info.items() if k != "darksirens_file"}}
            for c in ("gal", "agn"):
                W = np.concatenate(pooled[label][c]["w"])
                res["pooled"][c] = {q: compare(np.concatenate(pooled[label][c]["ev"][q]), R[q], W) for q in QUANT}
            out["sets"][label] = res
            print(f"=== {label} pooled")
            for c in ("gal", "agn"):
                for q in QUANT:
                    v = res["pooled"][c][q]
                    print(f"  {c:3s} {q:9s} n={v['n_events']:5d} mean ev {v['mean_events']:9.4f} pred {v['mean_pred']:9.4f} "
                          f"z {v['z_mean']:+6.2f}  KS p {v['ks_p']:.3f}  AD p {v['ad_p']:.3f}  Neff_ref {v['n_eff_reference']:.0f}",
                          flush=True)
        del R
    p = A16 / "tests" / "results" / "obslike_offset_step3_detected.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p, f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
