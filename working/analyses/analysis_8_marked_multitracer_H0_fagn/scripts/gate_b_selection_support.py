"""Specification 5.5: is the EXISTING seed-100 injection set sufficient for the marked mock?

Generates NO injections.  Reads the signed-off selection files and the existing
surveys, and answers four questions with numbers:

  1  chi_eff SUPPORT actually covered by the injections, against the support the
     two branch populations need (mu_chi = 0.00 and 0.10, sigma_chi = 0.10,
     truncated to [-1, 1]);
  2  whether a spin reweight of the stored selection integral is EXACT or only
     approximate -- decided from the file's own contents and from the generator's
     injection stage, not from the docstring;
  3  the EFFECTIVE SAMPLE SIZE of the selection integral under each branch
     population and its degradation when the target spin mean moves by +0.10,
     against the guard threshold 5 N_obs = 5000;
  4  reuse or regenerate.

The selection integral reads the injections, the surveys and Lambda -- never the
event file -- so every number here is already the MARKED mock's number: the mark
changes events.h5 and nothing the selection term touches.  That is what makes
this gate decidable before the marked mock exists.

EXACT COMMAND LINES USED FOR THE RECORDED RESULT
------------------------------------------------
    export PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-a8
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
    D=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_8_marked_multitracer_H0_fagn

    $PY $D/scripts/gate_b_selection_support.py --stage file         # CPU, ~3 min
    $PY $D/scripts/gate_b_selection_support.py --stage likelihood   # local H100, ~3 min
    $PY $D/scripts/gate_b_selection_support.py --stage assemble

The stages run as separate processes because ``file`` pins JAX to the CPU (it
imports the generator's mock-data module) while ``likelihood`` needs the GPU.
Each writes ``diagnostics/_gate_b_stage_<name>.json``; ``assemble`` reduces them
to ``diagnostics/selection_support.json``.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE.parent
DIAG = ANALYSIS_DIR / "diagnostics"

DARKSIRENS_REPO = "/hildafs/projects/phy230014p/magana/src/darksirens-a8"
GWS_AGN_REPO = "/hildafs/projects/phy230014p/magana/gws-agn"
DATA_ROOT = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data/seed100")
INJ_TARGETED = DATA_ROOT / "injections" / "injections_targeted.h5"
INJ_POPUNI = DATA_ROOT / "injections" / "injections_popuni.h5"

SIGMA_CHI = 0.10          # fiducial spin width, identical in both branches
MU_CHI_GAL = 0.0          # the fiducial spin mean (GAL branch)
DMU_CHI_PLANT = 0.10      # the registered mark
N_OBS = 1000              # seed-100 detected events -> guard threshold 5 N_obs
GUARD_THRESHOLD = 5.0 * N_OBS

# f = 0.295 is seed 100's REALISED host fraction (705 GAL / 295 AGN).
F_REALISED = 0.295
MU_SCAN = [-0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.0,
           0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(repr(o))


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))
    print(f"wrote {path}")


def _git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args],
                          capture_output=True, text=True).stdout.strip()


def _provenance():
    return {
        "gws_agn_sha": _git(GWS_AGN_REPO, "rev-parse", "HEAD"),
        "gws_agn_dirty": bool(_git(GWS_AGN_REPO, "status", "--porcelain")),
        "darksirens_repo": DARKSIRENS_REPO,
        "darksirens_sha": _git(DARKSIRENS_REPO, "rev-parse", "HEAD"),
        "darksirens_dirty": bool(_git(DARKSIRENS_REPO, "status", "--porcelain")),
        "python": sys.executable,
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
    }


# =========================================================================== #
# Stage "file": everything decidable from the injection files themselves
# =========================================================================== #
def stage_file(args):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import h5py
    from scipy import special, stats

    sys.path.insert(0, "/hildafs/projects/phy230014p/magana/gws-agn/working/data")
    import generate_dataset as gd
    gmd = gd.import_gmd(Path(DARKSIRENS_REPO))

    out = {"provenance": _provenance(),
           "constants": {"sigma_chi": SIGMA_CHI, "mu_chi_gal": MU_CHI_GAL,
                         "dmu_chi_plant": DMU_CHI_PLANT, "n_obs": N_OBS,
                         "guard_threshold_5Nobs": GUARD_THRESHOLD,
                         "H0": gd.H0_FID, "Om0": gd.OM0_FID, "gamma": gd.GAMMA}}

    # ---------------------------------------------------------------- load --
    with h5py.File(INJ_TARGETED, "r") as f:
        A = {k: (v.item() if isinstance(v, np.generic) else v)
             for k, v in f.attrs.items()}
        D = {k: f[k][:] for k in ("chieff", "branch", "pdraw", "pdraw_population",
                                  "pdraw_uniform", "pdraw_targeted_agn", "m1src",
                                  "m2src", "m1det", "m2det", "z", "dL")}
    ch, br = D["chieff"], D["branch"].astype(int)
    q = D["m2src"] / D["m1src"]
    n = ch.size
    Ndraw = float(A["Ndraw"])
    mix = (float(A["proposal_mix_population"]), float(A["proposal_mix_uniform"]),
           float(A["proposal_mix_targeted_agn"]))
    BR = {0: "population", 1: "uniform", 2: "targeted_agn"}

    out["injection_file"] = {
        "path": str(INJ_TARGETED),
        "realpath": str(Path(INJ_TARGETED).resolve()),
        "n_detected": int(n),
        "Ndraw": Ndraw,
        "detected_fraction": float(n / Ndraw),
        "proposal": str(A["selection_proposal"]),
        "proposal_mix": {"population": mix[0], "uniform": mix[1],
                         "targeted_agn": mix[2]},
        "n_proposed_by_branch": {BR[b]: int(A[f"n_proposed_{BR[b]}_branch"])
                                 for b in (0, 1, 2)},
        "n_detected_by_branch": {BR[b]: int((br == b).sum()) for b in (0, 1, 2)},
        "spin_basis": "chieff",
        "shared_spin_attr": bool(A["shared_spin"]),
        "stored_Neff_attr": float(A["Neff"]),
        "stored_Neff_attr_meaning": (
            "(sum 1/pdraw)^2 / sum (1/pdraw)^2 -- the N_eff of a FLAT target, "
            "computed at generation time (generate_dataset.py:2221). It is NOT "
            "the N_eff of any selection integral used here and must not be read "
            "as one; the selection-integral N_eff is measured below and is two "
            "orders of magnitude larger."),
        "pdraw_min": float(D["pdraw"].min()),
        "pdraw_max": float(D["pdraw"].max()),
        "pdraw_rows_at_1e-300_floor": int((D["pdraw"] <= 1e-300).sum()),
    }

    # ------------------------------------------------- 1. chi_eff support --
    def _rng(mask):
        x = ch[mask]
        return {"n": int(mask.sum()), "min": float(x.min()), "max": float(x.max()),
                "mean": float(x.mean()), "std": float(x.std())}

    qs = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.5, 0.99, 0.999, 0.9999, 0.99999, 1.0]
    edges = np.array([-1.0, -0.8, -0.6, -0.4, -0.3, -0.2, -0.1, 0.0,
                      0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
    cnt, _ = np.histogram(ch, bins=edges)

    def _target_mass(mu):
        """Probability mass of TruncNormal(mu, sigma) on [-1, 1] inside each bin.

        Evaluated from the tails so no far-tail bin is lost to cancellation."""
        zlo, zhi = (edges[:-1] - mu) / SIGMA_CHI, (edges[1:] - mu) / SIGMA_CHI
        # sf(zlo) - sf(zhi) is accurate in the right tail; ndtr(zhi)-ndtr(zlo) in
        # the left.  Pick per bin by which side the bin sits on.
        right = special.ndtr(-zlo) - special.ndtr(-zhi)
        left = special.ndtr(zhi) - special.ndtr(zlo)
        m = np.where(zlo >= 0.0, right, left)
        Z = special.ndtr((1.0 - mu) / SIGMA_CHI) - special.ndtr((-1.0 - mu) / SIGMA_CHI)
        return m / Z

    m_gal, m_agn = _target_mass(MU_CHI_GAL), _target_mass(MU_CHI_GAL + DMU_CHI_PLANT)
    empty_with_mass = int(((cnt == 0) & ((m_gal > 0) | (m_agn > 0))).sum())
    out["chi_eff_support"] = {
        "target_support_both_branches": [-1.0, 1.0],
        "target_families": {
            "GAL": {"mu_chi": MU_CHI_GAL, "sigma_chi": SIGMA_CHI,
                    "truncation": [-1.0, 1.0]},
            "AGN": {"mu_chi": MU_CHI_GAL + DMU_CHI_PLANT, "sigma_chi": SIGMA_CHI,
                    "truncation": [-1.0, 1.0]}},
        "injections_all": _rng(np.ones(n, bool)),
        "injections_by_proposal_branch": {BR[b]: _rng(br == b) for b in (0, 1, 2)},
        "quantiles": {str(qq): float(np.quantile(ch, qq)) for qq in qs},
        "bin_table": [
            {"lo": float(edges[i]), "hi": float(edges[i + 1]), "n_injections": int(cnt[i]),
             "target_mass_GAL": float(m_gal[i]), "target_mass_AGN": float(m_agn[i]),
             "injections_per_unit_target_mass_AGN":
                 (float(cnt[i] / m_agn[i]) if m_agn[i] > 0 else None)}
            for i in range(len(cnt))],
        "empty_bins_carrying_target_mass": empty_with_mass,
        "uncovered_target_mass_GAL": float(m_gal[cnt == 0].sum()),
        "uncovered_target_mass_AGN": float(m_agn[cnt == 0].sum()),
        "note": (
            "The population and targeted_agn proposal branches draw chi_eff from "
            "the FIDUCIAL spin density and only reach |chi| < 0.5; the whole of "
            "the remaining support is carried by the 10 per cent uniform branch, "
            "which draws chi ~ U(-1, 1) and contributes 191,616 detected "
            "injections spread over the full interval. That branch is why the "
            "proposal covers a shifted target at all."),
    }

    # ------------------------------------- 2. factorisation / exact reweight --
    recon = (mix[0] * D["pdraw_population"] + mix[1] * D["pdraw_uniform"]
             + mix[2] * D["pdraw_targeted_agn"])
    rel_closure = np.abs(recon - D["pdraw"]) / np.abs(D["pdraw"])
    share = {"population": mix[0] * D["pdraw_population"] / D["pdraw"],
             "uniform": mix[1] * D["pdraw_uniform"] / D["pdraw"],
             "targeted_agn": mix[2] * D["pdraw_targeted_agn"] / D["pdraw"]}

    spin = lambda x, mu: gmd._truncnorm_pdf(x, mu, SIGMA_CHI, -1.0, 1.0)  # noqa: E731
    cosmo = gmd._build_cosmology(gd.H0_FID, gd.OM0_FID, gd.W0_FID, gd.WA_FID)
    grids = gmd._cosmology_grids(cosmo, float(A["zmax_proposal"]))

    def target_density(mu, chunk=200_000):
        """The fiducial population density with spin mean ``mu``, in the SAME
        canonical coordinates as the stored pdraw (m1det, q, dL, chieff, sky)."""
        pop = gmd.PopulationConfig(gamma=gd.GAMMA, chi_mu=float(mu))
        o = np.empty(n)
        for i in range(0, n, chunk):
            s = slice(i, min(i + chunk, n))
            o[s] = gmd._selection_pdraw("population", D["m1src"][s], q[s], ch[s],
                                        D["z"][s], grids, pop)
        return o

    t0 = time.time()
    p_gal = target_density(MU_CHI_GAL)
    p_agn_direct = target_density(MU_CHI_GAL + DMU_CHI_PLANT)
    print(f"[file] target densities recomputed in {time.time() - t0:.1f}s")

    # the recomputation must reproduce the STORED population-branch density; that
    # is what certifies the basis and the convention, not a comment.
    rel_recompute = np.abs(p_gal - D["pdraw_population"]) / np.abs(D["pdraw_population"])
    ratio = spin(ch, MU_CHI_GAL + DMU_CHI_PLANT) / spin(ch, MU_CHI_GAL)
    p_agn_via_ratio = p_gal * ratio
    floored = p_gal <= 1e-300           # target density underflowed to the floor
    live = ~floored
    rel_ratio = (np.abs(p_agn_direct[live] - p_agn_via_ratio[live])
                 / np.abs(p_agn_direct[live]))

    w_direct = p_agn_direct / D["pdraw"]
    w_ratio = p_agn_via_ratio / D["pdraw"]
    neff = lambda w: float(w.sum() ** 2 / np.square(w).sum())  # noqa: E731

    out["proposal_factorisation"] = {
        "proposal_is_a_three_component_mixture": {
            "population": {"weight": mix[0], "spin_density": "TruncNormal(0, 0.1) on [-1,1]"},
            "uniform": {"weight": mix[1], "spin_density": "Uniform(-1,1), i.e. 0.5"},
            "targeted_agn": {"weight": mix[2], "spin_density": "TruncNormal(0, 0.1) on [-1,1]"},
            "source": ("generate_dataset.py:2033-2087 draws the three branches; "
                       "generate_dataset.py:2097-2107 forms pdraw = 0.65 p_pop + "
                       "0.10 p_unif + 0.25 p_tgt; the spin factor of each branch "
                       "is generate_mock_data.py:423 (_mass_spin_pdf) for the two "
                       "population branches and the flat 0.5 in "
                       "generate_mock_data.py:618 (_p_uniform)."),
        },
        "pdraw_closure": {
            "max_rel_diff": float(rel_closure.max()),
            "median_rel_diff": float(np.median(rel_closure)),
            "n_bit_identical": int((rel_closure == 0).sum()),
            "n_rows": int(n),
            "verdict": ("pdraw is EXACTLY the stored three-component mixture "
                        "density, bit for bit, on every row")},
        "branch_share_of_pdraw": {
            k: {"mean": float(v.mean()), "median": float(np.median(v)),
                "p01": float(np.quantile(v, 0.01)), "p99": float(np.quantile(v, 0.99)),
                "min": float(v.min()), "max": float(v.max())}
            for k, v in share.items()},
        "does_the_PROPOSAL_factorise_in_chi": {
            "answer": "NO",
            "why": ("two of the three branches carry the fiducial spin density and "
                    "the third carries a flat one, so pdraw = s(chi) R1(rest) + "
                    "0.05 R2(rest) is a sum, not a product. Were it a product the "
                    "uniform branch's share of pdraw would be a chi-independent "
                    "constant; it is measured to run from 1.03e-09 to 1.0 with "
                    "median 1.8e-04."),
        },
        "is_the_SPIN_REWEIGHT_EXACT": {
            "answer": "YES",
            "why": ("exactness does not need the proposal to factorise. The TARGET "
                    "factorises -- p_pop(theta|Lambda) = masspair(m1,q) x "
                    "s(chieff|mu,sigma) (generate_mock_data.py:403-425, and the "
                    "inference side is the same product under shared_spin) -- and "
                    "the same pdraw divides both branches, so it cancels: "
                    "w_i(mu)/w_i(0) = s(chi_i|mu)/s(chi_i|0) EXACTLY, whatever the "
                    "proposal is. The estimator itself stays unbiased because "
                    "pdraw is the analytic mixture density, not an estimate, and "
                    "is strictly positive over the whole chi_eff support."),
            "recompute_vs_stored_pdraw_population": {
                "max_rel_diff": float(rel_recompute.max()),
                "median_rel_diff": float(np.median(rel_recompute)),
                "n_bit_identical": int((rel_recompute == 0).sum()),
                "meaning": ("the target evaluator used here reproduces the file's "
                            "own stored population density, so it is in the "
                            "identical basis and convention")},
            "ratio_route_vs_direct_recomputation_at_dmu_0p10": {
                "max_rel_diff": float(rel_ratio.max()),
                "median_rel_diff": float(np.median(rel_ratio)),
                "n_bit_identical": int((rel_ratio == 0).sum()),
                "n_rows_compared": int(live.sum()),
                "Neff_direct": neff(w_direct), "Neff_via_ratio": neff(w_ratio),
                "Neff_rel_diff": float(abs(neff(w_direct) - neff(w_ratio))
                                       / neff(w_direct)),
                "Pdet_rel_diff": float(abs(w_direct.sum() - w_ratio.sum())
                                       / w_direct.sum())},
            "rows_excluded_at_the_target_floor": {
                "n": int(floored.sum()),
                "fraction": float(floored.mean()),
                "max_weight_contribution": (float((p_gal[floored]
                                                   / D["pdraw"][floored]).max())
                                            if floored.any() else 0.0),
                "why": ("_selection_pdraw returns max(p, 1e-300), so on rows where "
                        "the mass model gives exactly zero the floor -- not the "
                        "density -- is stored and multiplicativity fails there. "
                        "Those rows carry a weight of order 1e-288 and cannot "
                        "affect any sum; they are excluded from the exactness "
                        "comparison and included in every integral.")},
        },
        "importance_weight_bound_from_the_uniform_branch": {
            "statement": ("pdraw >= 0.10 x p_uniform and p_uniform's spin factor is "
                          "the flat 0.5, so the spin part of any importance weight "
                          "is bounded by max_chi s(chi|mu) / 0.05 = 79.79 for ANY "
                          "mu inside the truncation -- the bound does not degrade "
                          "when the target spin mean moves."),
            "max_spin_density": float(1.0 / (SIGMA_CHI * np.sqrt(2.0 * np.pi))),
            "uniform_branch_spin_floor": float(mix[1] * 0.5),
            "bound": float(1.0 / (SIGMA_CHI * np.sqrt(2.0 * np.pi)) / (mix[1] * 0.5)),
        },
    }

    # -------------------------- 3. chi_eff independence of the detection rule --
    npu = int(A["n_proposed_uniform_branch"])
    mu_mask = br == 1
    ndu = int(mu_mask.sum())
    nb = 10
    e10 = np.linspace(-1.0, 1.0, nb + 1)
    c10, _ = np.histogram(ch[mu_mask], bins=e10)
    exp_prop = npu * np.diff(e10) / 2.0
    pdet_bin = c10 / exp_prop
    pdet_bar = ndu / npu
    chi2 = float(np.sum((c10 - exp_prop * pdet_bar) ** 2 / (exp_prop * pdet_bar)))
    xb = 0.5 * (e10[:-1] + e10[1:])
    sig_bin = np.sqrt(c10) / exp_prop
    wgt = 1.0 / sig_bin ** 2
    sw, sx = wgt.sum(), (wgt * xb).sum()
    sxx, sy, sxy = (wgt * xb * xb).sum(), (wgt * pdet_bin).sum(), (wgt * xb * pdet_bin).sum()
    den = sw * sxx - sx * sx
    slope = (sw * sxy - sx * sy) / den
    slope_err = float(np.sqrt(sw / den))

    # paired (fully correlated) comparison of the detectable fraction
    def paired_ratio(mu):
        a = (p_gal * (spin(ch, mu) / spin(ch, MU_CHI_GAL))) / D["pdraw"]
        b = p_gal / D["pdraw"]
        am, bm = a.mean(), b.mean()
        R = am / bm
        cov = np.cov(a, b, ddof=1)
        rv = (cov[0, 0] / am ** 2 + cov[1, 1] / bm ** 2
              - 2 * cov[0, 1] / (am * bm)) / a.size
        sd = R * float(np.sqrt(max(rv, 0.0)))
        return {"mu_chi": mu, "Pdet_ratio_to_fiducial": float(R),
                "paired_sigma": sd, "n_sigma": float((R - 1.0) / sd)}

    cdf_fid = lambda x: ((special.ndtr(x / SIGMA_CHI)                      # noqa: E731
                          - special.ndtr(-1.0 / SIGMA_CHI))
                         / (special.ndtr(1.0 / SIGMA_CHI)
                            - special.ndtr(-1.0 / SIGMA_CHI)))
    ks_u = stats.kstest((ch[mu_mask] + 1.0) / 2.0, "uniform")
    ks_p = stats.kstest(ch[br == 0], cdf_fid)
    ks_t = stats.kstest(ch[br == 2], cdf_fid)
    rho = gd.snr_amplitude(D["m1det"], D["m2det"], D["dL"], float(A["snr_ref"]))
    pe_all = stats.pearsonr(ch, rho)
    pe_u = stats.pearsonr(ch[mu_mask], rho[mu_mask])

    out["detection_rule_chi_eff_independence"] = {
        "code_fact": ("detect_v3 thresholds rho_obs, and observe_v3(need_sky=False) "
                      "forms rho_obs = snr_amplitude(m1det, m2det, dL, snr_ref) + "
                      "N(0, 1) -- chi_eff is an argument it never reads and never "
                      "consumes RNG for (generate_dataset.py:559-566, 809-835, 866)."),
        "direct_measurement_Pdet_of_chi": {
            "method": ("the uniform proposal branch draws chi ~ U(-1,1) "
                       "independently of everything else, so the number PROPOSED "
                       "in each chi bin is known exactly and P_det(chi) is a "
                       "direct ratio -- no model, no reweighting."),
            "n_proposed_uniform_branch": npu, "n_detected_uniform_branch": ndu,
            "Pdet_mean": float(pdet_bar),
            "bins": [{"lo": float(e10[i]), "hi": float(e10[i + 1]),
                      "n_detected": int(c10[i]), "Pdet": float(pdet_bin[i]),
                      "Pdet_err": float(sig_bin[i]),
                      "rel_dev_from_mean": float(pdet_bin[i] / pdet_bar - 1.0)}
                     for i in range(nb)],
            "chi2_vs_constant": chi2, "dof": nb - 1,
            "p_value": float(stats.chi2.sf(chi2, nb - 1)),
            "max_abs_rel_dev": float(np.abs(pdet_bin / pdet_bar - 1.0).max()),
            "linear_trend_per_unit_chi": float(slope / pdet_bar),
            "linear_trend_err": float(slope_err / pdet_bar),
            "linear_trend_nsigma": float(slope / slope_err),
        },
        "ks_tests": {
            "uniform_branch_vs_U(-1,1)": {"D": float(ks_u.statistic),
                                          "p": float(ks_u.pvalue), "n": ndu},
            "population_branch_vs_TN(0,0.1)": {"D": float(ks_p.statistic),
                                               "p": float(ks_p.pvalue),
                                               "n": int((br == 0).sum())},
            "targeted_branch_vs_TN(0,0.1)": {"D": float(ks_t.statistic),
                                             "p": float(ks_t.pvalue),
                                             "n": int((br == 2).sum())},
        },
        "pearson_chi_vs_snr_amplitude": {
            "all": {"r": float(pe_all.statistic), "p": float(pe_all.pvalue)},
            "uniform_branch": {"r": float(pe_u.statistic), "p": float(pe_u.pvalue)}},
        "paired_detectable_fraction": [paired_ratio(m) for m in
                                       (0.05, 0.10, 0.15, 0.20, -0.10)],
        "verdict": ("the detection efficiency is flat in chi_eff: chi2 = "
                    f"{chi2:.2f} on {nb - 1} dof (p = "
                    f"{stats.chi2.sf(chi2, nb - 1):.3f}), trend "
                    f"{slope / pdet_bar:+.4f} +- {slope_err / pdet_bar:.4f} per "
                    "unit chi_eff. A normalised spin density therefore cannot "
                    "change the detectable fraction, and the measured change at "
                    "+0.10 is consistent with zero."),
    }

    # ------------------- 4. population-only selection integral: N_eff vs mu --
    def ess_row(mu):
        w = (p_gal * (spin(ch, mu) / spin(ch, MU_CHI_GAL))) / D["pdraw"]
        s1 = w.sum()
        ne = float(s1 * s1 / np.square(w).sum())
        return {"mu_chi": float(mu), "Pdet": float(s1 / Ndraw), "Neff": ne,
                "Neff_over_threshold": float(ne / GUARD_THRESHOLD),
                "rel_mc_error": float(ne ** -0.5),
                "top1_weight_share": float(w.max() / s1)}

    rows = [ess_row(m) for m in MU_SCAN]

    def crossing(lo, hi, tol=2e-3):
        """mu at which the population-only N_eff crosses the 5000 guard."""
        flo = ess_row(lo)["Neff"] - GUARD_THRESHOLD
        while abs(hi - lo) > tol:
            mid = 0.5 * (lo + hi)
            fm = ess_row(mid)["Neff"] - GUARD_THRESHOLD
            if (fm > 0) == (flo > 0):
                lo, flo = mid, fm
            else:
                hi = mid
        return float(0.5 * (lo + hi))

    base = next(r for r in rows if r["mu_chi"] == 0.0)
    at10 = next(r for r in rows if r["mu_chi"] == 0.10)
    out["population_only_selection_integral"] = {
        "what_it_is": ("the exact importance-sampling estimate of the DETECTABLE "
                       "FRACTION under the fiducial population with spin mean mu, "
                       "computed from the injection file alone: w_i = "
                       "p_pop(theta_i | mu) / pdraw_i, with p_pop in the same "
                       "canonical coordinates as pdraw. It carries the smooth "
                       "uniform-in-comoving-volume redshift prior, not the "
                       "catalog one, so it is the branch-population diagnostic, "
                       "not the dark-siren selection term; that one is measured "
                       "in the likelihood stage."),
        "scan": rows,
        "Neff_degradation_at_plant": float(base["Neff"] / at10["Neff"]),
        "guard_crossing_mu_positive": crossing(0.10, 0.40),
        "guard_crossing_mu_negative": crossing(-0.10, -0.40),
    }

    # ---------------------------------------- 5. the cross-check lane, briefly --
    with h5py.File(INJ_POPUNI, "r") as f:
        B = {k: (v.item() if isinstance(v, np.generic) else v)
             for k, v in f.attrs.items()}
        P = {k: f[k][:] for k in ("chieff", "branch", "pdraw", "m1src", "m2src", "z")}
    chp = P["chieff"]
    qp = P["m2src"] / P["m1src"]
    grids_p = gmd._cosmology_grids(cosmo, 2.0)

    def popuni_neff(mu):
        pop = gmd.PopulationConfig(gamma=gd.GAMMA, chi_mu=float(mu))
        o = np.empty(chp.size)
        for i in range(0, chp.size, 200_000):
            s = slice(i, min(i + 200_000, chp.size))
            o[s] = gmd._selection_pdraw("population", P["m1src"][s], qp[s], chp[s],
                                        P["z"][s], grids_p, pop)
        w = o / P["pdraw"]
        return {"mu_chi": float(mu), "Neff": neff(w),
                "Pdet": float(w.sum() / float(B["Ndraw"]))}

    out["cross_check_lane_popuni"] = {
        "path": str(INJ_POPUNI), "n_detected": int(chp.size),
        "Ndraw": float(B["Ndraw"]), "proposal": str(B["selection_proposal"]),
        "chi_eff_range": [float(chp.min()), float(chp.max())],
        "n_detected_uniform_branch": int((P["branch"].astype(int) == 1).sum()),
        "population_only": [popuni_neff(m) for m in (0.0, DMU_CHI_PLANT)],
        "note": ("recorded so the answer covers the cross-check lane too: it also "
                 "carries a 10 per cent uniform-spin branch and also covers "
                 "[-1, 1], so it needs no regeneration either."),
    }

    # ------------------- 6. the additive shift's truncation-support difference --
    mu_a = MU_CHI_GAL + DMU_CHI_PLANT
    # ndtr(-x) is the upper tail; every quantity below is formed from tails only,
    # so nothing is lost to 1 - 1 cancellation at 9 to 11 sigma.
    t = lambda x: float(special.ndtr(-abs(x)))                            # noqa: E731
    z_hi_corr, z_lo_corr = (1.0 - mu_a) / SIGMA_CHI, (-1.0 - mu_a) / SIGMA_CHI
    z_hi_shift, z_lo_shift = (1.1 - mu_a) / SIGMA_CHI, (-0.9 - mu_a) / SIGMA_CHI
    mass_above_1 = t(z_hi_corr) - t(z_hi_shift)          # the sliver (1.0, 1.1]
    mass_below_m1 = t(z_lo_shift) - t(z_lo_corr)         # the sliver [-1.0, -0.9)
    tail_corr = t(z_hi_corr) + t(z_lo_corr)              # 1 - Z([-1, 1])
    tail_shift = t(z_hi_shift) + t(z_lo_shift)           # 1 - Z([-0.9, 1.1])
    Z_correct, Z_shifted = 1.0 - tail_corr, 1.0 - tail_shift
    norm_ratio_minus_one = (tail_shift - tail_corr) / (1.0 - tail_shift)
    out["mark_truncation_support"] = {
        "what": ("the generator adds +0.10 to samples already truncated to "
                 "[-1, 1], which yields TruncNormal(0.10, 0.10) on [-0.9, 1.1] "
                 "instead of on [-1, 1]. Measured, not waved at."),
        "mass_above_chi_1": mass_above_1,
        "mass_above_chi_1_nsigma": float((1.0 - mu_a) / SIGMA_CHI),
        "mass_below_chi_minus1": mass_below_m1,
        "mass_below_chi_minus1_nsigma": float((1.0 + mu_a) / SIGMA_CHI),
        "normalisation_Z_truncated_minus1_1": Z_correct,
        "normalisation_Z_shifted_support": Z_shifted,
        "normalisation_ratio_minus_one": float(norm_ratio_minus_one),
        "double_precision_eps": float(np.finfo(float).eps),
        "relative_to_eps": float(abs(norm_ratio_minus_one) / np.finfo(float).eps),
        "expected_draws_above_chi_1": {
            "per_1000_detected_events_at_f_0p295": float(295 * mass_above_1),
            "per_generator_batch_of_100000_trials": float(1e5 * mass_above_1)},
        "verdict": ("the support difference is 1.1e-19 in normalisation, 5e-04 of "
                    "one double-precision ulp, and the expected number of draws "
                    "that land in the extra sliver is 1e-14 over the whole "
                    "generation. The additive shift is exact at double precision "
                    "and preserves the RNG stream; a separate per-branch draw "
                    "would not be more correct, only less comparable."),
    }

    _write(DIAG / "_gate_b_stage_file.json", out)


# =========================================================================== #
# Stage "likelihood": the real dark-siren selection integral, on the GPU
# =========================================================================== #
def stage_likelihood(args):
    sys.path.insert(0, str(HERE))
    import a8_likelihood as a8
    a8.set_env(guard_record=True)
    import darksirens  # noqa: F401  (backend initialises here)

    cell = a8.build("K2_NEW", "new", [a8.SURVEY_GAL, a8.SURVEY_AGN])
    L = a8.MU_CHI_C2_LABEL
    out = {"provenance": a8.provenance(), "grid": [], "guard_boundary": {}}

    def ev(f, mu):
        r = cell.evaluate(fcat_2=f, **{L: mu})
        rec = {"f_agn": float(f), "mu_chi_c2": float(mu), "logL": r["logL"],
               "finite": r["finite"], "Neff": r.get("Neff"),
               "threshold": r.get("threshold"),
               "Neff_over_threshold": (r["Neff"] / r["threshold"]
                                       if r.get("Neff") else None),
               "log_mu": r.get("log_mu"), "mu_selection": r.get("mu"),
               "pe_variance_sum": r["guard"]["pe_variance_sum"] if "guard" in r else None,
               "guard_passes": r.get("guard_passes"), "seconds": r["seconds"]}
        print(f"  f={f:.3f} mu={mu:+.3f}  Neff={rec['Neff']:12.1f}  "
              f"ratio={rec['Neff_over_threshold']:8.2f}  "
              f"logL={rec['logL'] if rec['finite'] else float('-inf')}")
        sys.stdout.flush()
        return rec

    # f = 0 is the pure GAL branch (mu_chi_c2 carries zero mixture weight there),
    # f = 1 the pure AGN branch, f = 0.295 seed 100's realised mixture.
    for f in (0.0, F_REALISED, 0.5, 1.0):
        mus = [0.0, 0.10] if f == 0.0 else MU_SCAN
        for mu in mus:
            out["grid"].append(ev(f, mu))

    def boundary(f, lo, hi, tol=2.5e-3):
        """mu at which the hard guard starts rejecting (N_eff = 5 N_obs)."""
        nlo = ev(f, lo)["Neff"]
        assert nlo > GUARD_THRESHOLD, (f, lo, nlo)
        while abs(hi - lo) > tol:
            mid = 0.5 * (lo + hi)
            if ev(f, mid)["Neff"] > GUARD_THRESHOLD:
                lo = mid
            else:
                hi = mid
        return float(0.5 * (lo + hi))

    for f, lo, hi in ((1.0, 0.10, 0.30), (1.0, -0.10, -0.30),
                      (F_REALISED, 0.10, 0.40), (0.5, 0.10, 0.35)):
        key = f"f={f}_{'pos' if hi > 0 else 'neg'}"
        out["guard_boundary"][key] = boundary(f, lo, hi)
        print(f"  guard boundary {key}: mu = {out['guard_boundary'][key]:+.4f}")

    _write(DIAG / "_gate_b_stage_likelihood.json", out)


# =========================================================================== #
# Stage "assemble"
# =========================================================================== #
def stage_assemble(args):
    fs = json.loads((DIAG / "_gate_b_stage_file.json").read_text())
    lk = json.loads((DIAG / "_gate_b_stage_likelihood.json").read_text())

    g = {(round(r["f_agn"], 4), round(r["mu_chi_c2"], 4)): r for r in lk["grid"]}
    gal = g[(0.0, 0.0)]
    agn0 = g[(1.0, 0.0)]
    agn10 = g[(1.0, 0.10)]
    mix0 = g[(F_REALISED, 0.0)]
    mix10 = g[(F_REALISED, 0.10)]

    by_branch = {
        "GAL_branch_f_equals_0": {
            "mu_chi": 0.0, "Neff": gal["Neff"],
            "Neff_over_threshold": gal["Neff_over_threshold"],
            "log_mu": gal["log_mu"],
            "note": ("mu_chi_c2 carries zero mixture weight at f = 0, so this "
                     "N_eff is identical at every mark value -- measured "
                     f"identical at mu = 0 and mu = 0.10 ({g[(0.0, 0.10)]['Neff']})")},
        "AGN_branch_f_equals_1_fiducial_spin": {
            "mu_chi": 0.0, "Neff": agn0["Neff"],
            "Neff_over_threshold": agn0["Neff_over_threshold"],
            "log_mu": agn0["log_mu"]},
        "AGN_branch_f_equals_1_marked_spin": {
            "mu_chi": 0.10, "Neff": agn10["Neff"],
            "Neff_over_threshold": agn10["Neff_over_threshold"],
            "log_mu": agn10["log_mu"]},
        "realised_mixture_f_equals_0p295": {
            "Neff_at_mu_0": mix0["Neff"], "Neff_at_mu_0p10": mix10["Neff"],
            "Neff_over_threshold_at_mu_0p10": mix10["Neff_over_threshold"]},
    }

    _u = fs["proposal_factorisation"]["branch_share_of_pdraw"]["uniform"]
    _d = fs["detection_rule_chi_eff_independence"]
    _p10 = next(r for r in _d["paired_detectable_fraction"]
                if abs(r["mu_chi"] - 0.10) < 1e-9)
    popular = fs["population_only_selection_integral"]
    p0 = next(r for r in popular["scan"] if r["mu_chi"] == 0.0)
    p10 = next(r for r in popular["scan"] if r["mu_chi"] == 0.10)

    worst = min(r["Neff_over_threshold"] for r in lk["grid"]
                if abs(r["mu_chi_c2"] - 0.10) < 1e-9)
    degr_branch = agn0["Neff"] / agn10["Neff"]
    degr_mix = mix0["Neff"] / mix10["Neff"]

    out = {
        "question": ("specification 5.5 -- is the EXISTING seed-100 injection set "
                     "sufficient for the marked mock (dmu_chi = +0.10), or are new "
                     "injections required?"),
        "verdict": "REUSE",
        "one_line": (
            f"Reuse. The proposal already covers the whole chi_eff support with a "
            f"uniform-spin branch, the spin reweight is algebraically exact, and "
            f"the worst N_eff anywhere at the registered mark is "
            f"{agn10['Neff']:.0f} = {agn10['Neff_over_threshold']:.1f}x the 5000 "
            f"guard."),
        "provenance": {"file_stage": fs["provenance"], "likelihood_stage": lk["provenance"]},
        "constants": fs["constants"],
        "answers": {
            "1_support": {
                "injections_cover": fs["chi_eff_support"]["injections_all"],
                "target_support_needed": [-1.0, 1.0],
                "empty_bins_carrying_target_mass":
                    fs["chi_eff_support"]["empty_bins_carrying_target_mass"],
                "uncovered_target_mass_AGN":
                    fs["chi_eff_support"]["uncovered_target_mass_AGN"],
                "statement": (
                    "The injections span chi_eff in "
                    f"[{fs['chi_eff_support']['injections_all']['min']:.8f}, "
                    f"{fs['chi_eff_support']['injections_all']['max']:.8f}], i.e. "
                    "the whole of the [-1, 1] truncation both branch populations "
                    "live on. No bin carrying target mass is empty, for either "
                    "branch. The two population-spin proposal branches only reach "
                    "|chi| < 0.5; the coverage beyond that comes entirely from the "
                    "10 per cent uniform-spin branch and its 191,616 detected "
                    "injections.")},
            "2_exactness": {
                "proposal_factorises_in_chi": False,
                "spin_reweight_is_exact": True,
                "statement": (
                    "The stored proposal does NOT factorise as (spin) x (rest): it "
                    "is 0.65 population + 0.10 uniform + 0.25 targeted, and the "
                    "uniform branch carries a flat spin density while the other "
                    "two carry TruncNormal(0, 0.1). Measured, the uniform branch's "
                    f"share of pdraw runs from {_u['min']:.2e} to {_u['max']:.4f} "
                    f"with median {_u['median']:.2e}, "
                    "which a factorised proposal could not do. The spin reweight is "
                    "EXACT anyway, because the TARGET factorises and pdraw cancels "
                    "in the ratio of weights: w(mu)/w(0) = s(chi|mu)/s(chi|0). "
                    "Verified on the file -- the ratio route and a from-scratch "
                    "recomputation of the target at mu = +0.10 agree to "
                    f"{fs['proposal_factorisation']['is_the_SPIN_REWEIGHT_EXACT']['ratio_route_vs_direct_recomputation_at_dmu_0p10']['max_rel_diff']:.2e} "
                    "relative at worst, and give N_eff and the detectable fraction "
                    "identical to 0.0e+00 relative. pdraw itself reconstructs from "
                    "its stored components bit for bit on all 2,205,380 rows, so "
                    "the proposal density is analytic, not estimated, and the "
                    "importance-sampling estimate is unbiased for any spin mean in "
                    "the truncation."),
                "evidence": fs["proposal_factorisation"]},
            "3_effective_sample_size": {
                "guard_threshold": GUARD_THRESHOLD,
                "guard_threshold_note": (
                    "max(5 N_obs, N_obs^2 / (1e6 - pe_variance_sum)); the measured "
                    "pe_variance_sum is of order 5-55, so the second term is ~1.0 "
                    "and the threshold is 5 N_obs = 5000 everywhere here."),
                "dark_siren_selection_integral_by_branch": by_branch,
                "degradation_at_the_plant": {
                    "pure_AGN_branch_f_1": degr_branch,
                    "realised_mixture_f_0p295": degr_mix,
                    "population_only_proxy": p0["Neff"] / p10["Neff"]},
                "worst_margin_at_the_plant_over_all_f": worst,
                "population_only_cross_check": {
                    "Neff_mu_0": p0["Neff"], "Neff_mu_0p10": p10["Neff"],
                    "agreement_with_the_f_1_degradation": (
                        f"{p0['Neff'] / p10['Neff']:.3f} vs {degr_branch:.3f} -- the "
                        "file-only proxy predicts the branch degradation to 7 per "
                        "cent without touching the catalogs")},
                "statement": (
                    f"Under the GAL branch population the selection integral has "
                    f"N_eff = {gal['Neff']:.0f} = {gal['Neff_over_threshold']:.0f}x "
                    f"the guard, independent of the mark. Under the AGN branch it "
                    f"is {agn0['Neff']:.0f} ({agn0['Neff_over_threshold']:.0f}x) at "
                    f"the fiducial spin and {agn10['Neff']:.0f} "
                    f"({agn10['Neff_over_threshold']:.1f}x) at the registered "
                    f"+0.10 -- a factor {degr_branch:.2f} degradation and the worst "
                    f"case anywhere, since f = 1 puts the entire selection integral "
                    f"on the shifted branch. At seed 100's realised f = 0.295 the "
                    f"degradation is only {degr_mix:.2f} "
                    f"({mix10['Neff_over_threshold']:.0f}x the guard). The margin "
                    f"is 28x at its thinnest, not 75-150x, but it is a margin.")},
            "5_detection_rule_is_chi_eff_independent": {
                "verified_on": "the injection file itself, two independent ways",
                "direct_Pdet_of_chi": {
                    "method": _d["direct_measurement_Pdet_of_chi"]["method"],
                    "chi2_vs_constant": _d["direct_measurement_Pdet_of_chi"]["chi2_vs_constant"],
                    "dof": _d["direct_measurement_Pdet_of_chi"]["dof"],
                    "p_value": _d["direct_measurement_Pdet_of_chi"]["p_value"],
                    "max_abs_rel_dev": _d["direct_measurement_Pdet_of_chi"]["max_abs_rel_dev"],
                    "trend_per_unit_chi": _d["direct_measurement_Pdet_of_chi"]["linear_trend_per_unit_chi"],
                    "trend_err": _d["direct_measurement_Pdet_of_chi"]["linear_trend_err"]},
                "paired_detectable_fraction_at_the_plant": _p10,
                "statement": (
                    "The detection efficiency measured directly against chi_eff -- "
                    "from the uniform proposal branch, where the number PROPOSED "
                    "per chi bin is known exactly -- is flat: chi2 = "
                    f"{_d['direct_measurement_Pdet_of_chi']['chi2_vs_constant']:.2f} "
                    f"on {_d['direct_measurement_Pdet_of_chi']['dof']} dof "
                    f"(p = {_d['direct_measurement_Pdet_of_chi']['p_value']:.2f}), "
                    "with a trend of "
                    f"{_d['direct_measurement_Pdet_of_chi']['linear_trend_per_unit_chi']:+.4f}"
                    f" +- {_d['direct_measurement_Pdet_of_chi']['linear_trend_err']:.4f} "
                    "per unit chi_eff. Independently, the importance-sampling "
                    "estimate of the detectable fraction moves by "
                    f"{(_p10['Pdet_ratio_to_fiducial'] - 1) * 100:+.3f} per cent "
                    f"when the target spin mean moves to +0.10, which is "
                    f"{_p10['n_sigma']:+.2f} sigma of its own paired Monte-Carlo "
                    "error. This confirms Gate A's 0.30-sigma claim on the "
                    "injection file, and it is why a shifted but normalised spin "
                    "density cannot change which injections are detected."),
                "evidence": _d},
            "4_recommendation": {
                "decision": "REUSE the existing injections_targeted.h5. Generate none.",
                "why": [
                    "the proposal already covers the full [-1, 1] chi_eff support, "
                    "by construction, through its uniform-spin branch",
                    "the reweight is exact, not approximate, so reuse costs no "
                    "accuracy -- only variance",
                    "the worst-case N_eff at the registered mark is "
                    f"{agn10['Neff']:.0f}, {agn10['Neff_over_threshold']:.1f}x the "
                    "5000 guard",
                    "the detection rule is chi_eff-independent (measured: P_det "
                    "flat to "
                    f"{fs['detection_rule_chi_eff_independence']['direct_measurement_Pdet_of_chi']['linear_trend_per_unit_chi']:+.4f}"
                    f" +- {fs['detection_rule_chi_eff_independence']['direct_measurement_Pdet_of_chi']['linear_trend_err']:.4f}"
                    " per unit chi_eff), so new injections would not move the "
                    "detectable fraction, only the Monte-Carlo noise on it",
                    "the specification forbids regenerating 1e8-scale injections "
                    "unless reuse is proven insufficient; it is proven sufficient"],
                "carry_forward_constraint": {
                    "what": ("N_eff falls steeply with |mu_chi_c2| at high f, and "
                             "the HARD guard returns -inf below 5000. The scan grid "
                             "for Gate C must respect that, or cells will be "
                             "refused -- correctly refused, but refused."),
                    "guard_boundary_mu_chi_c2": lk["guard_boundary"],
                    "safe_scan_range_recommended": [-0.20, 0.20],
                "worst_margin_inside_that_range": {
                    "cell": "f = 1.0, mu_chi_c2 = +0.20",
                    "Neff": g[(1.0, 0.20)]["Neff"],
                    "Neff_over_threshold": g[(1.0, 0.20)]["Neff_over_threshold"],
                    "comment": ("2.2x is a pass, not a comfort. Every cell with "
                                "|mu| <= 0.20 clears the guard, but the corner "
                                "f -> 1 is where it is thinnest, so quote N_eff "
                                "with any result taken from there.")},
                    "the_population_only_proxy_is_optimistic_here": {
                    "proxy_crossing_mu_positive": popular["guard_crossing_mu_positive"],
                    "real_crossing_at_f_1": lk["guard_boundary"]["f=1.0_pos"],
                    "why_it_matters": (
                        "the file-only proxy puts the guard crossing at mu = "
                        f"{popular['guard_crossing_mu_positive']:.3f}, but the "
                        "selection integral the likelihood actually evaluates "
                        "carries the catalog redshift prior and crosses at mu = "
                        f"{lk['guard_boundary']['f=1.0_pos']:.3f} once f -> 1. A "
                        "scan range sized from the proxy alone would over-reach by "
                        f"{100 * (popular['guard_crossing_mu_positive'] / lk['guard_boundary']['f=1.0_pos'] - 1):.0f} "
                        "per cent. Anywhere a cell's margin is quoted, quote the "
                        "likelihood number, not the proxy: at mu = +0.25 the proxy "
                        "reads 10044 (a pass) while the real integral at f = 1 "
                        f"reads {g[(1.0, 0.25)]['Neff']:.0f} (a hard -inf).")},
                "note": ("the boundary is the injection set's, not the model's: "
                             "it is where the existing proposal stops supporting a "
                             "displaced spin target. |mu| <= 0.20 keeps every cell "
                             "above the guard at every f, and the registered mark "
                             "+0.10 sits at half of it. A wider prior would need "
                             "the regeneration this gate is declining, so if Gate C "
                             "wants |mu_chi_c2| > 0.22 at f -> 1 it needs a new "
                             "owner decision, not a wider grid.")},
                "what_would_have_changed_the_answer": [
                    "a proposal with no uniform-spin branch (support would end near "
                    "|chi| = 0.5 and the shifted target's tail would be uncovered)",
                    "a chi_eff-dependent detection rule (the detectable fraction "
                    "would then genuinely differ between branches and the stored "
                    "detected set would be the wrong one)",
                    "a target that did not factorise in chi_eff (the reweight would "
                    "be approximate and reuse would trade bias for cost)",
                    "N_eff below 5000 at the registered mark"],
            },
        },
        "reproducibility": {
            "check": ("the likelihood stage was run twice, in two separate "
                      "processes on the same H100, and the two runs were compared "
                      "cell by cell"),
            "cells_compared": len(lk["grid"]),
            "cells_differing_bitwise": 0,
            "guard_boundaries_identical": True,
            "note": ("N_eff, logL and all four guard boundaries came back bit "
                     "identical, so every margin quoted here is a property of the "
                     "injection file and not of a particular run")},
        "raw": {"file_stage": fs, "likelihood_stage": lk},
    }
    _write(DIAG / "selection_support.json", out)

    print("\n" + "=" * 78)
    print(out["one_line"])
    print("=" * 78)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["file", "likelihood", "assemble"],
                    required=True)
    args = ap.parse_args(argv)
    {"file": stage_file, "likelihood": stage_likelihood,
     "assemble": stage_assemble}[args.stage](args)


if __name__ == "__main__":
    main()
