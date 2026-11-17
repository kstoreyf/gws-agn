#!/usr/bin/env python
"""The observed-like mass spectrum: GWTC-3's POWER LAW+PEAK medians and their mapping onto
darksirens-core's powerlaw+peak (any env with scipy).

    python design_values.py      -> results/design_values.json

Source: R. Abbott et al. (LVK), "Population of Merging Compact Binaries Inferred Using
Gravitational Waves through GWTC-3", Phys. Rev. X 13, 011048 (2023), arXiv:2111.03634
(text checked in v4 and the journal version; v1/v2 quoted different medians, listed below).

GWTC-3's model (App. B 1 b, Eqs. B4-B7):
    pi(m1) ~ [(1-lambda) P(m1; -alpha, m_min, m_max) + lambda G(m1; mu_m, sigma_m)] S(m1; m_min, delta_m)
    pi(q | m1) ~ q^beta_q S(q m1; m_min, delta_m)
with P a normalised power law cut SHARPLY at m_max, G a normalised Gaussian, S the Planck
taper of Eqs. B5-B6, and the product renormalised.  Core's powerlaw+peak:
    p(m1) = w_PL PL(m1) + w_G G(m1),   PL = m^-alpha S_low(m; m_min, dm_min) S_high(m; m_max, dm_max) / N
    p(q | m1) ~ q^beta S_low(q m1; m_min, dm_min)           (both components pair at the PL edge)
so S_low is GWTC-3's S (same Planck form), dm_min = delta_m, and the two densities coincide when
    w_G = lambda <S>_G / ((1-lambda) <S>_PL + lambda <S>_G),   dm_max -> 0,
with <S>_PL = int P S, <S>_G = int G S (= 1 to 1e-9 here: the peak sits 6.7 sigma above
m_min + delta_m, which is also why core's untapered Gaussian is immaterial).  The taper
removes 61% of P's mass, so lambda = 0.038 is w_G = 0.092: the peak's share of mergers in
GWTC-3's fitted density is 9.2%, not 3.8%.
"""
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

OB = Path(__file__).resolve().parent.parent

PAPER = {
    "citation": "Abbott et al. (LVK), Phys. Rev. X 13, 011048 (2023), arXiv:2111.03634",
    "model": "POWER LAW+PEAK, App. B 1 b, Eqs. (B4)-(B7); priors Table VI",
    "medians": {
        "alpha": {"value": 3.5, "interval": "+0.6 -0.56", "where": "Sec. VI A"},
        "mu_m": {"value": 34.0, "interval": "+2.6 -4.0", "where": "Sec. VI A"},
        "beta_q": {"value": 1.1, "interval": "+1.7 -1.3", "where": "Sec. VI A"},
        "lambda_peak": {"value": 0.038, "interval": "+0.058 -0.026", "where": "Sec. VI A"},
        "m_min": {"value": 5.0, "interval": "+0.86 -1.7",
                  "where": "Sec. VI E (fit without GW190814 and GW190917)"},
        "delta_m": {"value": 4.9, "interval": "+3.4 -3.2", "where": "Sec. VI E (same fit)"},
        "m_99": {"value": 44.0, "interval": "+9.2 -5.1", "where": "Sec. VI A (derived 99th percentile of m1)"},
    },
    "not_quoted_in_the_paper": {
        "sigma_m": {"used": 3.6, "note": "no median in any version's text or tables (the paper says only that "
                    "the Gaussian's mean and width agree with GWTC-2); 3.6 is the brief's value, said to be the "
                    "data release's median; not verified here"},
        "m_max": {"used": 87.0, "note": "no median quoted (the paper reports m_99 instead); 87 is the brief's "
                  "value; not verified here. With alpha 3.5 the power law above 77 Msun holds 3e-4 of it"},
    },
    "earlier_arxiv_versions": "v1/v2 (Nov 2021) quote alpha = 3.4 +0.58 -0.49, mu_m = 34 +2.3 -3.8, beta_q = 1.1 "
                              "+1.8 -1.3, m_99 = 45 +11 -6.2, and (without the two low-mass events) "
                              "m_min = 2.5 +0.67 -0.44, delta_m = 7.8 +1.9 -4.0; v3/v4 and the journal "
                              "version quote the values used here",
}
DESIGN = dict(alpha=3.5, m_min=5.0, m_max=87.0, delta_m=4.9, lambda_peak=0.038, mu_m=34.0, sigma_m=3.6, beta_q=1.1)


def S(m, mmin, dm):
    if m <= mmin:
        return 0.0
    if m >= mmin + dm:
        return 1.0
    x = m - mmin
    e = dm / x + dm / (x - dm)
    return 0.0 if e > 700 else 1.0 / (np.exp(e) + 1.0)


def main():
    a, lo, hi, dm = DESIGN["alpha"], DESIGN["m_min"], DESIGN["m_max"], DESIGN["delta_m"]
    lam, mu, sg = DESIGN["lambda_peak"], DESIGN["mu_m"], DESIGN["sigma_m"]
    zp = quad(lambda m: m ** -a, lo, hi)[0]
    s_pl = quad(lambda m: S(m, lo, dm) * m ** -a, lo, lo + dm, limit=400, epsrel=1e-12)[0] / zp \
        + quad(lambda m: m ** -a, lo + dm, hi, epsrel=1e-12)[0] / zp
    zg = norm.cdf(hi, mu, sg) - norm.cdf(lo, mu, sg)
    s_g = (quad(lambda m: S(m, lo, dm) * norm.pdf(m, mu, sg), lo, lo + dm, limit=400)[0]
           + norm.cdf(hi, mu, sg) - norm.cdf(lo + dm, mu, sg)) / zg
    w_g_exact = lam * s_g / ((1 - lam) * s_pl + lam * s_g)
    w_g = 0.092                                   # core's peak weight, rounded
    lam_implied = w_g * s_pl / (w_g * s_pl + (1 - w_g) * s_g)

    # cross-checks against the paper's derived 99th percentile (GWTC-3 density, Eq. B4)
    def m_quantiles(lmb, mmax=hi, sig=sg):
        zp_ = quad(lambda m: m ** -a, lo, mmax)[0]
        zg_ = norm.cdf(mmax, mu, sig) - norm.cdf(lo, mu, sig)
        ms = np.linspace(lo, 100.0, 400001)
        sv = np.array([S(m, lo, dm) for m in ms])
        p = ((1 - lmb) * ms ** -a / zp_ * (ms < mmax) + lmb * norm.pdf(ms, mu, sig) / zg_) * sv
        c = np.cumsum(p)
        c /= c[-1]
        return {"m1_q01": float(np.interp(0.01, c, ms)), "m1_median": float(np.interp(0.5, c, ms)),
                "m1_q99": float(np.interp(0.99, c, ms))}
    core = {"v1": round(1 - w_g, 6), "alpha": a, "m_min": lo, "m_max": hi, "dm_min": dm, "dm_max": 0.01,
            "mu_G": mu, "sigma_G": sg, "beta": DESIGN["beta_q"], "mu_chi": 0.0, "sigma_chi": 0.1, "gamma": 0.0}
    out = {
        "what": "the observed-like (obslike) mass spectrum: GWTC-3 POWER LAW+PEAK medians and their "
                "mapping onto darksirens-core powerlaw+peak; generator name --mass_spectrum gwtc3_pp",
        "paper": PAPER,
        "design_gwtc3_parameters": DESIGN,
        "mapping": {
            "S_avg_powerlaw": s_pl, "S_avg_gaussian": s_g,
            "w_G_exact_from_lambda": w_g_exact, "w_G_used": w_g, "v1_used": 1 - w_g,
            "lambda_implied_by_w_G_used": lam_implied,
            "w_G_if_lambda_taken_literally": lam,
            "dm_min": "delta_m (core's low-mass taper of the power law AND the pairing taper of m2 = q m1 for "
                      "both components, as GWTC-3's Eqs. B4 and B7)",
            "dm_max": "core's high-mass taper width of the power law; GWTC-3 cuts P sharply at m_max, so it is "
                      "set to core's prior floor 0.01 Msun (DM_MAX bounds [0.01, 20])",
            "structural_differences": "core leaves the Gaussian untapered and truncates it to [1, 200] "
                                      "(GWTC-3: tapered by S); <S>_G = 1 - 1e-9, so immaterial",
        },
        "core_theta": core,
        "core_theta_order": "v1 alpha m_min m_max dm_min dm_max mu_G sigma_G beta mu_chi sigma_chi gamma "
                            "(v1 = POWER-LAW weight)",
        "unchanged_from_present_family": {"mu_chi": 0.0, "sigma_chi": 0.1, "gamma": 0.0,
                                          "host_dependent_offsets": {"dmu_G": 5.0, "dmu_chi": 0.10},
                                          "f_agn": 0.30, "measurement_model": "v3", "detection": "rho_obs >= 8"},
        "present_family_core_theta": {"v1": 0.1, "alpha": 2.3, "m_min": 5.0, "m_max": 80.0, "dm_min": 3.0,
                                      "dm_max": 10.0, "mu_G": 35.0, "sigma_G": 5.0, "beta": 1.0,
                                      "mu_chi": 0.0, "sigma_chi": 0.1, "gamma": 0.0},
        "cross_check_m99": {
            "paper_m99": "44 +9.2 -5.1",
            "design_density": m_quantiles(lam),
            "same_with_lambda_0p038_but_no_renormalisation_ie_w_G_0p038": None,
        },
    }
    # the literal reading (core w_G = 0.038): its 99th percentile, for contrast
    a_ = a
    ms = np.linspace(lo, 100.0, 400001)
    sv = np.array([S(m, lo, dm) for m in ms])
    pl = ms ** -a_ * sv * (ms < hi)
    pl /= np.trapz(pl, ms)
    g = norm.pdf(ms, mu, sg)
    g /= np.trapz(g, ms)
    for wg_, key in ((0.038, "same_with_lambda_0p038_but_no_renormalisation_ie_w_G_0p038"),):
        p = (1 - wg_) * pl + wg_ * g
        c = np.cumsum(p)
        c /= c[-1]
        out["cross_check_m99"][key] = {"m1_q99": float(np.interp(0.99, c, ms)),
                                       "m1_median": float(np.interp(0.5, c, ms))}
    out["cross_check_m99"]["sensitivity_to_unverified"] = {
        "sigma_m_5p5_m_max_65 (core registry placeholders)": m_quantiles(lam, mmax=65.0, sig=5.5),
        "sigma_m_3p6_m_max_65": m_quantiles(lam, mmax=65.0, sig=3.6)}
    (OB / "results").mkdir(exist_ok=True)
    (OB / "results" / "design_values.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["mapping"], indent=1))
    print(json.dumps(out["cross_check_m99"], indent=1))


if __name__ == "__main__":
    main()
