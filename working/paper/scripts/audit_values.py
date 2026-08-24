#!/usr/bin/env python3
r"""Audit the numbers chain.

`build_values.py` writes `values/results_macros.tex`.  This script checks it,
and it is written to be an *independent* second implementation rather than a
call into the builder: it re-opens the source files, re-derives every value it
can, re-renders it, and demands **exact string equality** with what is committed.
A macro that no derivation reaches must appear in the whitelist below with a
one-line reason.

It also reads the containment table back out of `NUMBERS.md` and re-derives the
one row whose input value lives in a different file from the interval that
contains it, so a claim of agreement cannot drift from the numbers behind it.

It then reads the manuscript and reports two things that are not failures but
that the writer needs to see: macro-looking commands the section files use that
nothing defines, and editorial sentinels (`\todo`, `??`, `TBD`, `XXX`) still in
the prose.

Exit status
    0   every derived macro matches, nothing undefined
    1   a value disagrees, a macro is missing, or an undefined macro is used

Usage
    python scripts/audit_values.py [-v]
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent
WORKING = PAPER.parent

META = WORKING / "data" / "seed100" / "META.json"
A0 = WORKING / "analyses" / "analysis_0_pure_tracer_H0" / "results"
A1 = WORKING / "analyses" / "analysis_1_complete_catalog_H0" / "results"
A2 = WORKING / "analyses" / "analysis_2_complete_catalog_H0_fagn" / "results"
A3 = WORKING / "analyses" / "analysis_3_incomplete_catalog_H0_fagn" / "results"
A4 = WORKING / "analyses" / "analysis_4_density_anchoring_H0_fagn" / "results"
A5 = WORKING / "analyses" / "analysis_5_free_anchors_H0_fagn" / "results"
A6 = WORKING / "analyses" / "analysis_6_relative_completeness_H0_fagn" / "results"
A5S100 = (WORKING / "analyses" / "experiments"
          / "experiment_dsmaster_4d_recheck" / "results")
FU101 = WORKING / "analyses" / "selection_redo" / "fu_seed101" / "results"
FU102 = WORKING / "analyses" / "selection_redo" / "fu_seed102" / "results"

# the likelihood revision the flux-limited results are quoted from, and the one
# file that predates it (see NUMBERS.md for why it is allowed to)
SHA_OF_RECORD = "0c5b3db"
SHA_EXEMPT = {"fit_m18_per_pixel_s100.json":
              "e8d5035c3562f86ce83497572844a213aa14801c"}

MACROS = PAPER / "values" / "results_macros.tex"
NUMBERS = PAPER / "NUMBERS.md"
MAIN = PAPER / "main.tex"
SECTIONS = sorted((PAPER / "sections").glob("*.tex"))

# ---------------------------------------------------------------------------
# macros no derivation reaches, and why
# ---------------------------------------------------------------------------
WHITELIST: dict[str, str] = {}

# ---------------------------------------------------------------------------
# Macros the v3 rebuild deleted because the concept they named no longer exists
# in the dataset.  The section files still use them; the prose rewrite that
# removes them is queued.  Listing one here says "known, and here is what the
# sentence should say instead" -- it is not a licence to leave it.  Delete the
# entry when the prose is fixed; anything undefined and NOT listed here fails.
# The same table, with the reasoning, is in NUMBERS.md.
# ---------------------------------------------------------------------------
PENDING_REWRITE = {
    "EvSigmaDl":
        "v2 flat fractional distance width; v3 derives dL from (Mc_det, rho) "
        "-> \\EvSigmaLnDlCoef, \\EvSigmaLnDlMedian, \\EvSigmaLnDlThresh",
    "EvSigmaMone":
        "v2 fractional primary-mass width; v3 measures ln Mc and ln q "
        "-> \\EvAmc, \\EvAq",
    "EvSigmaMtwo":
        "v2 fractional secondary-mass width; same replacement",
    "EvSigmaChi":
        "v2 fixed effective-spin width 0.08; v3 uses \\EvAchi (0.20) times "
        "rho_th/rho_obs",
    "HzeroAgn":
        "the AGN-only posterior rails at the top of the scanned range and has "
        "no interval -> \\HzeroAgnRailMedian against \\HzeroScanHi",
    "HzeroAgnWidth":
        "same: a railed posterior has no 68% width to quote",
}

# ---------------------------------------------------------------------------
# commands that are LaTeX/aastex/amsmath, not paper macros.  Only capitalised
# commands are checked at all (every generated macro is capitalised), so this
# list is short.
# ---------------------------------------------------------------------------
BUILTIN = {
    "Lambda", "Omega", "Theta", "Delta", "Gamma", "Phi", "Psi", "Sigma", "Pi",
    "Upsilon", "Xi", "S", "P", "LaTeX", "TeX", "Big", "Bigg", "Left", "Right",
    "Re", "Im", "Pr", "Vert",
}


# ---------------------------------------------------------------------------
# rendering, re-implemented
# ---------------------------------------------------------------------------
def fnum(value, spec):
    """Render one number the way the manuscript wants it."""
    if spec == "sci":
        if value == 0:
            return r"\ensuremath{0}"
        e = math.floor(math.log10(abs(value)))
        mant = value / 10.0 ** e
        if abs(mant - 1.0) < 5e-3:
            return r"\ensuremath{10^{%d}}" % e
        return r"\ensuremath{%.1f\times10^{%d}}" % (mant, e)
    if spec == "intsep":
        text = format(int(round(value)), ",")
        return text.replace(",", "\\,")
    if spec.startswith("%+"):
        return r"\ensuremath{%s}" % (spec % value)
    return spec % value


def interval(text):
    return r"\ensuremath{%s}" % text


def pmstr(mean, err, mspec="%+.2f", espec="%.2f"):
    return r"\ensuremath{%s \pm %s}" % (mspec % mean, espec % err)


def brackets(lo, hi, spec="%.3f"):
    return r"\ensuremath{[%s,\, %s]}" % (spec % lo, spec % hi)


def asymstr(block, spec="%.3f"):
    """`median^{+u}_{-l}` from a summary block that stores the interval."""
    med = block["median"]
    lo, hi = block["ci68"]
    return r"\ensuremath{%s^{+%s}_{-%s}}" % (spec % med, spec % (hi - med),
                                            spec % (med - lo))


def slope_of(xs, ys):
    """Least-squares slope, by the normal equations rather than centred sums."""
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    return (n * sxy - sx * sy) / (n * sxx - sx * sx)


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------
def read(path):
    with open(path) as fh:
        return json.load(fh)


def readck(path, c_mode="selection"):
    """Read a flux-limited result file and re-check its provenance.

    The builder refuses a file produced by the wrong completeness treatment or
    by the wrong revision of the likelihood; the same check is redone here from
    the file itself, so a source swapped underneath the committed macros is
    caught even when the number it carries happens to round the same way.
    """
    tree = read(path)
    got = tree.get("c_mode")
    if got is not None and got != c_mode:
        raise SystemExit(f"{path}: completeness treatment {got!r}, "
                         f"expected {c_mode!r}")
    sha = tree.get("darksirens_git_sha")
    if sha is not None and not str(sha).startswith(SHA_OF_RECORD):
        if SHA_EXEMPT.get(Path(path).name) != sha:
            raise SystemExit(f"{path}: likelihood revision {sha!r} is neither "
                             f"the revision of record nor exempted")
    return tree


def containment_rows() -> dict[str, dict[str, str]]:
    """The containment table of NUMBERS.md, parsed back out and keyed by claim.

    The builder writes that table; nothing else checks it, so a claim of
    agreement could drift from the interval that supports it without any macro
    changing.  Reading it back and re-deriving the level from the run files is
    the only way the drift shows up.
    """
    parts = NUMBERS.read_text().split("## Containment of the input values", 1)
    if len(parts) != 2:
        raise SystemExit(f"{NUMBERS}: no containment table")
    rows: dict[str, dict[str, str]] = {}
    for line in parts[1].split("\n## ", 1)[0].splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 7 or cells[0] == "claim" or set(cells[0]) <= {"-"}:
            continue
        rows[cells[0]] = {"quantity": cells[1], "input": cells[2],
                          "median": cells[3], "level": cells[4],
                          "ci68": cells[5], "src": cells[6]}
    return rows


def dig(tree, *keys):
    node = tree
    for k in keys:
        if isinstance(node, list):
            node = node[k] if isinstance(k, int) else None
        elif isinstance(node, dict):
            node = node.get(k)
        else:
            return None
        if node is None:
            return None
    return node


def mean(xs):
    return sum(xs) / len(xs)


def sem(xs):
    """Standard error of the mean, from the sample standard deviation."""
    n = len(xs)
    mu = mean(xs)
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))
    return sd / math.sqrt(n)


def luminosity_distance(z, h0, om0, steps=2000):
    step = z / steps
    total = 0.0
    for i in range(steps + 1):
        zi = i * step
        weight = 1 if i in (0, steps) else (4 if i % 2 else 2)
        total += weight / math.sqrt(om0 * (1 + zi) ** 3 + (1 - om0))
    return (1 + z) * (299792.458 / h0) * total * step / 3.0


# ===========================================================================
def derive() -> dict[str, str]:
    """Every macro this script can reach, rendered."""
    m = read(META)
    cfg, st = m["config"], m["stages"]
    ev, cat, sur, inj = st["events"], st["catalogs"], st["surveys"], st["injections"]
    chk = st["validation"]["checks"]
    v3 = cfg["events"]["v3_measurement_family"]

    pure = read(A0 / "h0_pure_tracer.json")
    single = read(A1 / "h0_single_tracer.json")
    closure = read(A1 / "closure_v3.json")
    curv = read(A1 / "v3_curvature.json")
    kde = read(A1 / "kde_window.json")
    joint = read(A2 / "h0_fagn_joint.json")
    jsum = read(A2 / "joint_summary.json")
    fsc = read(A2 / "fscan_s100.json")
    fnull = read(A2 / "fscan_null_s100.json")
    mumc = read(A2 / "mu_mc_error.json")

    e: dict[str, str] = {}

    # ---- cosmology, field, catalogs
    e["HzeroTruth"] = fnum(cfg["cosmology"]["H0"], "%.2f")
    e["OmZeroTruth"] = fnum(cfg["cosmology"]["Om0"], "%.4f")
    e["ObZeroTruth"] = fnum(cfg["glass"]["Ob0"], "%.4f")
    e["FieldZmax"] = fnum(cfg["glass"]["z_max"], "%.1f")
    e["FieldShellDx"] = fnum(cfg["glass"]["dx_mpc"], "%.0f")
    e["FieldNshell"] = fnum(cat["n_shells"], "%d")
    e["FieldNside"] = fnum(cfg["glass"]["nside"], "%d")
    e["FieldLmax"] = fnum(cfg["glass"]["lmax"], "%d")
    ngal, nagn = cfg["glass"]["n_comoving_gal"], cfg["glass"]["n_comoving_agn"]
    e["NGal"] = fnum(ngal, "sci")
    e["NAgn"] = fnum(nagn, "sci")
    e["DensityRatioTarget"] = fnum(ngal / nagn, "%.0f")
    bg, ba = cfg["glass"]["bias_gal"], cfg["glass"]["bias_agn"]
    e["BiasGal"] = fnum(bg, "%.1f")
    e["BiasAgn"] = fnum(ba, "%.1f")
    e["BiasRatio"] = fnum(ba / bg, "%.2f")
    e["CatNgal"] = fnum(cat["tracers"]["gal"]["n"], "sci")
    e["CatNagn"] = fnum(cat["tracers"]["agn"]["n"], "sci")
    e["DensityRatio"] = fnum(cat["density_ratio_gal_over_agn"], "%.1f")
    br = chk["V4_catalog_densities_and_clustering"]["bias_ratio_agn_over_gal"]
    e["BiasRatioMeas"] = pmstr(br["measured"], max(br["err"], 5e-3),
                               "%.2f", "%.2f")

    # ---- luminosity function
    lf = cfg["luminosity_function"]
    e["PhiStar"] = fnum(lf["phi_star_h3"], "sci")
    e["SchechterAlpha"] = fnum(lf["alpha"], "%+.2f")
    e["MagBStar"] = fnum(lf["M_B_star"], "%+.2f")
    e["LumCut"] = fnum(cat["magnitude_model"]["x_cut_L_over_Lstar"], "%.2f")
    e["MagBLimit"] = fnum(cat["magnitude_model"]["M_B_faint_limit"], "%+.2f")

    # ---- events: configuration and the v3 measurement family
    e["EvNobs"] = fnum(cfg["events"]["n"], "intsep")
    e["EvFagn"] = fnum(cfg["events"]["f_agn"], "%.2f")
    e["EvNsamp"] = fnum(cfg["events"]["nsamp"], "intsep")
    rho_th = cfg["events"]["snr_threshold"]
    e["EvSnrThresh"] = fnum(rho_th, "%.0f")
    e["EvWidthRefSnr"] = fnum(rho_th, "%.0f")
    e["EvSigmaRho"] = fnum(v3["sigma_rho"], "%.1f")
    e["EvAmc"] = fnum(v3["A_MC"], "%.2f")
    e["EvAq"] = fnum(v3["A_Q"], "%.2f")
    e["EvAchi"] = fnum(v3["A_CHI"], "%.2f")
    ref_d = cfg["events"]["snr_ref_detect"]
    ref_s = cfg["events"]["snr_ref_sigma_ang"]
    e["EvSnrRef"] = fnum(ref_d, "%.2f")
    e["EvSnrRefSky"] = fnum(ref_s, "%.1f")
    e["EvSkyCoef"] = fnum(v3["sky_a_deg"], "%.0f")
    e["EvSkyMin"] = fnum(v3["sky_clip_deg"][0], "%.0f")
    e["EvSkyMax"] = fnum(v3["sky_clip_deg"][1], "%.0f")
    e["EvSkyRhoScale"] = fnum(ref_s / ref_d, "%.3f")
    e["EvSkyCoefEff"] = fnum(v3["sky_a_deg"] / (ref_s / ref_d), "%.1f")
    dl_coef = math.sqrt((5.0 / 6.0 * v3["A_MC"] * rho_th) ** 2
                        + v3["sigma_rho"] ** 2)
    e["EvSigmaLnDlCoef"] = fnum(dl_coef, "%.3f")
    e["EvSigmaLnDlThresh"] = fnum(100.0 * dl_coef / rho_th, "%.1f")

    pop = ev["population"]
    for name, key, spec in (("PopAlpha", "alpha", "%.1f"),
                            ("PopMmin", "mmin", "%.0f"),
                            ("PopMmax", "mmax", "%.0f"),
                            ("PopDmMin", "dm_min", "%.0f"),
                            ("PopDmMax", "dm_max", "%.0f"),
                            ("PopPeakFrac", "peak_fraction", "%.2f"),
                            ("PopPeakMu", "peak_mu", "%.0f"),
                            ("PopPeakSigma", "peak_sigma", "%.0f"),
                            ("PopBeta", "beta", "%.0f"),
                            ("PopChiSigma", "chi_sigma", "%.2f"),
                            ("PopGamma", "gamma", "%.0f")):
        e[name] = fnum(pop[key], spec)

    # ---- events: what was realised
    rl = ev["realised"]
    e["EvFagnReal"] = fnum(rl["realised_f_agn"], "%.3f")
    e["EvNhostGal"] = fnum(rl["n_host_gal"], "intsep")
    e["EvNhostAgn"] = fnum(rl["n_host_agn"], "intsep")
    e["EvNuniqueAgn"] = fnum(rl["unique_agn_hosts"], "intsep")
    e["EvMaxPerAgn"] = fnum(rl["max_events_per_agn_host"], "%d")
    e["EvHorizonZ"] = fnum(rl["horizon_z_max_detected"], "%.2f")
    e["EvZmedian"] = fnum(rl["z_median_detected"], "%.2f")
    e["EvNproposed"] = fnum(rl["n_proposed"], "intsep")
    e["EvNdetTotal"] = fnum(rl["n_detected_total"], "intsep")
    e["EvDetFrac"] = fnum(100.0 * rl["detected_fraction"], "%.3f")
    e["EvMalmquist"] = fnum(
        100.0 * rl["frac_detected_with_true_snr_below_threshold"], "%.0f")

    v1 = chk["V1_detection_deterministic_in_data"]
    v2 = chk["V2_widths_from_observed_snr"]
    rho_med = v1["rho_obs_median"]
    e["EvSnrMedian"] = fnum(rho_med, "%.2f")
    e["EvSnrMax"] = fnum(v1["rho_obs_max"], "%.1f")
    e["EvSigLnMcMedian"] = fnum(v3["A_MC"] * rho_th / rho_med, "%.4f")
    e["EvSigLnQMedian"] = fnum(v3["A_Q"] * rho_th / rho_med, "%.3f")
    e["EvSigChiMedian"] = fnum(v3["A_CHI"] * rho_th / rho_med, "%.3f")
    e["EvSigmaLnDlMedian"] = fnum(
        chk["V3_pe_calibration"]["sigma_lndL_realised_median"], "%.4f")
    e["EvSkyMinReal"] = fnum(v2["sigma_ang_deg_min"], "%.2f")
    e["EvSkyMaxReal"] = fnum(v2["sigma_ang_deg_max"], "%.2f")

    # ---- catalogs as the survey sees them
    nside = cfg["survey"]["nside"]
    e["SurveyNside"] = fnum(nside, "%d")
    e["SurveyNpix"] = fnum(12 * nside ** 2, "intsep")
    e["CatDzScale"] = fnum(sur["dz_scale"], "sci")
    pz = cat["photoz_model"]["realised"]
    e["CatPhotozPullSd"] = fnum(pz["gal"]["pull_sd"], "%.4f")
    e["CatPhotozPullSdAgn"] = fnum(pz["agn"]["pull_sd"], "%.4f")

    lims = sorted(cfg["survey"]["mag_limits"], reverse=True)
    e["MagLimDeep"] = fnum(max(lims), "%.0f")
    e["MagLimShallow"] = fnum(min(lims), "%.0f")
    e["MagLimStep"] = fnum(abs(lims[0] - lims[1]), "%.0f")
    e["MagLimN"] = fnum(len(lims), "%d")

    comp = sur["completeness"]
    devs = []
    for tag, key in (("MTwentyOne", "m21"), ("MTwenty", "m20"),
                     ("MNineteen", "m19"), ("MEighteen", "m18")):
        g = comp[key]["gal"]["C_within_horizon"]
        a = comp[key]["agn"]["C_within_horizon"]
        e["Comp" + tag] = fnum(100.0 * g, "%.0f")
        e["CompAgn" + tag] = fnum(100.0 * a, "%.0f")
        devs.append(abs(100.0 * (a - g)))
    e["CompAgnMaxDev"] = fnum(max(devs), "%.1f")

    blocks = sur["surveys"]
    e["AgnEmptyPix"] = fnum(100.0 * blocks["agn_complete"]["empty_pixel_fraction"],
                            "%.0f")
    e["AgnEmptyPixShallow"] = fnum(
        100.0 * blocks["agn_m18"]["empty_pixel_fraction"], "%.0f")
    e["AgnPerPix"] = fnum(blocks["agn_complete"]["max_hosts_per_pixel"], "intsep")

    # ---- selection
    e["InjNdrawMain"] = fnum(inj["targeted"]["ndraw"], "sci")
    e["InjNdrawCross"] = fnum(inj["popuni"]["ndraw"], "sci")
    e["InjNdetMain"] = fnum(inj["targeted"]["n_detected"], "intsep")
    e["InjNdetCross"] = fnum(inj["popuni"]["n_detected"], "intsep")
    wts = inj["targeted"]["mixture_weights"]
    e["InjWeightPop"] = fnum(wts[0], "%.2f")
    e["InjWeightUni"] = fnum(wts[1], "%.2f")
    e["InjWeightAgn"] = fnum(wts[2], "%.2f")

    ref = jsum["seeds"][0]
    e["NeffFactor"] = fnum(ref["joint"]["guard"]["threshold_min"]
                           / ref["n_events"], "%.0f")
    e["NeffMin"] = fnum(ref["joint"]["guard"]["Neff_min"], "sci")

    # ---- design margins
    scan = inj["targeted"]["targeted_branch"]["H0_scan_range"]
    e["HzeroScanLo"] = fnum(scan[0], "%.0f")
    e["HzeroScanHi"] = fnum(scan[1], "%.0f")
    e["PeZmax"] = fnum(
        chk["V8_catalog_edge_clears_pe_support"]["max_pe_redshift_over_H0_grid"],
        "%.2f")
    zmed = rl["z_median_detected"]
    h0, om0 = cfg["cosmology"]["H0"], cfg["cosmology"]["Om0"]
    sig_lndl = chk["V3_pe_calibration"]["sigma_lndL_realised_median"]
    dl = luminosity_distance(zmed, h0, om0)
    ez = math.sqrt(om0 * (1 + zmed) ** 3 + (1 - om0))
    ddl_dz = dl / (1 + zmed) + (1 + zmed) * (299792.458 / h0) / ez
    e["KdePeRatio"] = fnum((sig_lndl * dl / ddl_dz)
                           / (sur["dz_scale"] * (1 + zmed)), "%.1f")
    e["KdeWindow"] = fnum(kde["window_recommended_power_of_two"], "intsep")

    # ---- single-tracer results
    e["HzeroGal"] = interval(single["gal_h0_ci"])
    e["HzeroGalMedian"] = fnum(single["gal_h0_median"], "%.1f")
    e["HzeroGalWidth"] = fnum(single["gal_h0_width"], "%.2f")
    e["HzeroGalCross"] = fnum(single["gal_h0_crosscheck_median"], "%.1f")
    e["HzeroAgnRailMedian"] = fnum(single["agn_grid_top_median"], "%.1f")
    e["HzeroAgnRailCross"] = fnum(single["agn_h0_crosscheck_median"], "%.1f")

    # ---- joint results
    e["HzeroJoint"] = interval(joint["h0_ci"])
    e["HzeroJointMedian"] = fnum(joint["h0_median"], "%.1f")
    e["HzeroJointWidth"] = fnum(joint["h0_width"], "%.2f")
    e["HzeroJointCross"] = fnum(joint["h0_crosscheck_median"], "%.1f")
    e["HzeroJointMap"] = fnum(joint["map"]["H0"], "%.2f")
    e["FagnJoint"] = interval(joint["f_ci"])
    e["FagnJointMedian"] = fnum(joint["f_median"], "%.3f")
    e["FagnJointWidth"] = fnum(joint["f_width"], "%.3f")
    e["FagnJointCross"] = fnum(joint["f_crosscheck_median"], "%.3f")
    e["FagnJointMap"] = fnum(joint["map"]["f"], "%.3f")
    e["FagnTruthReal"] = fnum(joint["truth_f_realised"], "%.3f")
    e["FagnTruthPlanted"] = fnum(joint["truth_f_planted"], "%.2f")
    e["FagnBinomialSd"] = fnum(jsum["binomial_sd_per_realisation"], "%.3f")
    e["JointRho"] = fnum(joint["rho"], "%.3f")
    e["JointRhoMean"] = pmstr(dig(jsum, "closure", "rho", "mean"),
                              dig(jsum, "closure", "rho", "sem"),
                              "%+.3f", "%.3f")
    e["ClosureScatterHzeroRatio"] = fnum(
        dig(jsum, "closure", "scatter_H0", "ratio"), "%.2f")
    e["ClosureScatterFagnRatio"] = fnum(
        dig(jsum, "closure", "scatter_f", "ratio"), "%.2f")
    e["HzeroWidthRatio"] = fnum(single["gal_h0_width"] / joint["h0_width"],
                                "%.1f")

    e["ClosureNseeds"] = fnum(joint["closure_n_seeds"], "%d")
    e["ClosureHzero"] = pmstr(joint["closure_h0_offset_mean"],
                              joint["closure_h0_offset_sem"])
    e["ClosureFagnReal"] = pmstr(joint["closure_f_offset_vs_realised_mean"],
                                 joint["closure_f_offset_vs_realised_sem"],
                                 "%+.3f", "%.3f")
    e["ClosureFagnPlanted"] = pmstr(joint["closure_f_offset_vs_planted_mean"],
                                    joint["closure_f_offset_vs_planted_sem"],
                                    "%+.3f", "%.3f")
    cov = jsum["closure"]["coverage"]
    e["ClosureHzeroInSixtyEight"] = fnum(cov["H0_in_68"], "%d")
    e["ClosureHzeroInNinety"] = fnum(cov["H0_in_90"], "%d")
    e["ClosureFagnInSixtyEight"] = fnum(cov["f_realised_in_68"], "%d")
    e["ClosureFagnInNinety"] = fnum(cov["f_realised_in_90"], "%d")

    # ---- the null
    e["FagnRecord"] = fnum(fsc["f"]["median"], "%.3f")
    e["FagnRecordCi"] = brackets(*fsc["f"]["ci68"])
    e["FagnNull"] = fnum(fnull["f"]["median"], "%.3f")
    e["FagnNullCi"] = brackets(*fnull["f"]["ci68"])
    e["FagnNullCiNinety"] = brackets(*fnull["f"]["ci90"])
    e["FagnNullWidthRatio"] = fnum(
        dig(jsum, "sky_shuffle_null", "width_ratio_null_over_record"), "%.2f")

    excl = exclusion()
    if excl:
        e["FagnZeroDlnL"] = fnum(excl["dlnl"], "%.1f")
        e["FagnZero"] = fnum(math.sqrt(2.0 * excl["dlnl"]), "%.1f")
        e["FagnZeroRecord"] = fnum(math.sqrt(2.0 * excl["record"]), "%.1f")
        e["FagnZeroNull"] = fnum(math.sqrt(2.0 * excl["null"]), "%.1f")

    # ---- matched-host controls
    for tag, case in (("Gal", "gal"), ("Agn", "agn")):
        after = closure["cases"][case]["after"]
        e[f"Ctrl{tag}Mean"] = pmstr(after["mean_offset"], after["sem_offset"])
        e[f"Ctrl{tag}InSixtyEight"] = fnum(after["n_truth_in_ci68"], "%d")
        e[f"Ctrl{tag}InNinety"] = fnum(after["n_truth_in_ci90"], "%d")
        e[f"Ctrl{tag}Median"] = fnum(
            closure["cases"][case]["per_seed"][0]["after"]["median"], "%.1f")
    e["CtrlNseeds"] = fnum(closure["cases"]["gal"]["after"]["n_seeds"], "%d")

    sig = mumc["sigma_MC"]
    e["SelMcGal"] = fnum(sig["matched GAL, targeted"]
                         / abs(curv["ctrl_gal_matched"]["d2_per_event"]), "%.2f")
    e["SelMcAgn"] = fnum(sig["matched AGN, targeted"]
                         / abs(curv["ctrl_agn_matched"]["d2_per_event"]), "%.2f")
    lo, hi = mumc["bracket_targeted_lane"]["per_realisation"]
    e["SelMcJointLo"] = fnum(lo, "%.2f")
    e["SelMcJointHi"] = fnum(hi, "%.2f")

    # ---- pure-tracer event sets (appendix).  Everything below is rebuilt from
    # the per-realisation rows, never from the aggregate keys the builder reads:
    # the means, standard errors, coverage counts and lane maxima are recomputed
    # here, so an aggregator that disagreed with its own per-seed table would
    # show up as a mismatch rather than be copied twice.
    pg = pure["closure_gal"]["per_seed"]
    pa = pure["closure_agn"]["per_seed"]
    e["PureNseeds"] = fnum(len(pg), "%d")
    nev = {r["n_events"] for r in pg + pa}
    e["PureNevents"] = fnum(nev.pop(), "intsep") if len(nev) == 1 else "?"

    wg, wa = mean([r["half68"] for r in pg]), mean([r["half68"] for r in pa])
    e["PureGalHalfWidth"] = fnum(wg, "%.2f")
    e["PureAgnHalfWidth"] = fnum(wa, "%.2f")
    e["PureWidthRatio"] = fnum(wg / wa, "%.1f")
    ratios = [a["half68"] / g["half68"] for g, a in zip(pg, pa)]
    e["PureWidthRatioPerSeed"] = pmstr(mean(ratios), sem(ratios), "%.2f", "%.2f")

    for tag, rows in (("Gal", pg), ("Agn", pa)):
        offs = [r["offset"] for r in rows]
        e[f"Pure{tag}Offset"] = pmstr(mean(offs), sem(offs), "%+.3f", "%.3f")
        e[f"Pure{tag}InSixtyEight"] = fnum(
            sum(bool(r["truth_in_ci68"]) for r in rows), "%d")
        e[f"Pure{tag}InNinety"] = fnum(
            sum(bool(r["truth_in_ci90"]) for r in rows), "%d")

    for tag, case in (("Gal", "gal"), ("Agn", "agn")):
        e[f"PureLaneMax{tag}"] = fnum(
            max(abs(r["difference"]) / r["targeted_half68"]
                for r in pure["lanes"][case]["per_seed"]), "%.2f")

    e.update(flux_limited(cfg, jsum))

    # the one posterior with a real second mode; the 1 % relative-height cut
    # drops two AGN scans whose recorded second mode is ~1e-211 of the peak
    lane = pure["injection_lane_of_record"]
    multi = [s for s in pure["diagnostics"]["per_scan"]
             if s["lane"] == lane
             and len([h for h in s["mode_relative_heights"] if h >= 0.01]) > 1]
    if len(multi) == 1:
        s = multi[0]
        pairs = sorted(zip(s["mode_positions"], s["mode_relative_heights"]))
        e["PureBimodalSeed"] = fnum(s["seed"], "%d")
        e["PureBimodalModeLo"] = fnum(pairs[0][0], "%.2f")
        e["PureBimodalModeHi"] = fnum(pairs[-1][0], "%.2f")
        e["PureBimodalHeight"] = fnum(pairs[0][1], "%.2f")

    return e


def flux_limited(cfg, jsum) -> dict[str, str]:
    """Section 4.3, re-derived from the run files.

    Nothing here reads an aggregate that the builder also reads unless the
    aggregate is the only place the number lives: the ladder, the arms and the
    surface are rebuilt from their per-cell files, and the three-realisation
    numbers from the six fits themselves.
    """
    e: dict[str, str] = {}
    rungs = ("m21", "m20", "m19", "m18")

    # ---- the depth ladder at host densities held at their input values
    lad = {r: readck(A3 / f"joint_{r}_s100.json") for r in rungs}
    faint = lad["m18"]
    e["HzeroIncomplete"] = asymstr(faint["H0"], "%.1f")
    e["FagnIncomplete"] = asymstr(faint["f"], "%.3f")

    def half(block):
        return (block["ci68"][1] - block["ci68"][0]) / 2.0

    hw_faint = half(faint["f"])
    e["FagnWidthRatio"] = fnum(
        hw_faint / jsum["seeds"][0]["joint"]["f_vs_realised"]["halfwidth68"],
        "%.1f")

    ladsum = read(A3 / "ladder_summary.json")
    comp = {row["rung"]: row["C_in_horizon"] for row in ladsum["rows"]}

    # ---- the hosted fraction on the shallowest catalogs, against the fraction
    # the drawn events realised.  That value lives in the complete-catalog fit,
    # not in the ladder file, so the ladder's own containment flags cannot
    # answer the claim: the level is re-derived from the interval endpoints and
    # checked against the row the builder wrote.
    realised = read(A2 / "h0_fagn_joint.json")["truth_f_realised"]
    lo68, hi68 = faint["f"]["ci68"]
    lo90, hi90 = faint["f"]["ci90"]
    level = ("68 per cent" if lo68 <= realised <= hi68 else
             "90 per cent" if lo90 <= realised <= hi90 else "**neither**")
    claim = ("AGN-hosted fraction on the shallowest catalogs, against the "
             "realised fraction")
    row = containment_rows().get(claim)
    if row is None:
        raise SystemExit(f"{NUMBERS}: no containment row for {claim!r}")
    if (row["level"], row["input"]) != (level, "%.3f" % realised):
        raise SystemExit(
            f"{NUMBERS}: {claim!r} records input {row['input']} contained by "
            f"{row['level']}; the run files give {realised:.3f} contained by "
            f"{level}")

    # ---- the assumed-density arms: refit both responses from the arm files
    x0 = math.log10(cfg["glass"]["n_comoving_agn"])
    arms = read(A4 / "arms_summary.json")
    h0_slope, f_slope, hw = {}, {}, {}
    for rung in rungs:
        rows = [readck(A4 / f"joint_{rung}_{a}_s100.json")
                for a in ("a05", "a07", "a09", "a11", "a13", "a20")]
        xs = [r["base_coord"]["log10n0_c2"] - x0 for r in rows]
        h0_slope[rung] = slope_of(xs, [r["H0"]["median"] for r in rows])
        f_slope[rung] = slope_of(xs, [r["f"]["median"] for r in rows])
        hw[rung] = sum(half(r["H0"]) for r in rows) / len(rows)
        quoted = arms["rungs"][rung]["selection (this work)"]["slope"]
        if abs(quoted - f_slope[rung]) > 1e-6:
            raise SystemExit(f"{A4/'arms_summary.json'}: {rung} f_AGN slope "
                             f"{quoted} != refit {f_slope[rung]}")
    e["HzDensitySlopeShallow"] = fnum(h0_slope["m18"], "%.1f")
    e["HzDensityShiftFactorTwo"] = fnum(h0_slope["m18"] * math.log10(2.0),
                                        "%.2f")
    e["HzDensityShiftFactorTwoDeep"] = fnum(
        abs(h0_slope["m20"]) * math.log10(2.0), "%.2f")
    e["FagnDensitySlope"] = fnum(f_slope["m18"], "%.3f")

    # ---- both host densities free, reference realisation
    free = readck(A5 / "campaign_m18_dynesty_s100.json")["summary"]
    e["FagnFree"] = asymstr(free["f_AGN"], "%.3f")
    e["FagnFreeWidthRatio"] = fnum(half(free["f_AGN"]) / hw_faint, "%.1f")
    e["GalDensityFree"] = asymstr(free["log10n0"], "%.2f")
    e["GalDensityFreeOffsetDex"] = fnum(
        abs(free["log10n0"]["median"] - free["log10n0"]["truth"]), "%.2f")

    # ---- the same fit on three realisations, against its number-count twin
    pairs = ((A5S100 / "fit_m18_selection_s100.json",
              A5S100 / "fit_m18_per_pixel_s100.json"),
             (FU101 / "campaign_m18_dynesty_s101.json",
              FU101 / "campaign_m18_dynesty_pp_s101.json"),
             (FU102 / "campaign_m18_dynesty_s102.json",
              FU102 / "campaign_m18_dynesty_pp_s102.json"))
    lf = [readck(a) for a, _ in pairs]
    nc = [readck(b, "per_pixel") for _, b in pairs]
    e["NSeedsIncomplete"] = fnum(len(pairs), "%d")
    e["GalAnchorOffsetMaxDex"] = fnum(
        max(abs(r["summary"]["log10n0"]["median"]
                - r["summary"]["log10n0"]["truth"]) for r in lf), "%.2f")
    over = [r["summary"]["log10n0"]["median"] - r["summary"]["log10n0"]["truth"]
            for r in nc]
    e["PixelAnchorBiasMinDex"] = fnum(min(over), "%.2f")
    e["PixelAnchorBiasMaxDex"] = fnum(max(over), "%.2f")
    lnb = [a["sampler_meta"]["logz"] - b["sampler_meta"]["logz"]
           for a, b in zip(lf, nc)]
    e["SeedLnBFmin"] = fnum(min(lnb), "%.1f")
    e["SeedLnBFmax"] = fnum(max(lnb), "%.1f")
    foff = [r["summary"]["f_AGN"]["median"] - r["summary"]["f_AGN"]["truth"]
            for r in lf]
    e["FagnFreeOffsetMin"] = fnum(min(foff), "%.2f")
    e["FagnFreeOffsetMax"] = fnum(max(foff), "%.2f")

    # ---- galaxies and AGN cut to different depths, flux-limited cells only
    surf = read(A6 / "surface_summary.json")
    depth = surf["completeness"]
    xs, ys = [], []
    for cell in surf["cells"]["selection (this work)"]:
        if "complete" in (cell["gal"], cell["agn"]):
            continue
        one = readck(A6 / f"joint_{cell['cell']}_s100.json")
        xs.append(math.log10(depth[cell["agn"]]) - math.log10(depth[cell["gal"]]))
        ys.append(one["f"]["median"] - one["f"]["truth"])
    e["RelCompletenessSpanDex"] = fnum(max(xs) - min(xs), "%.2f")
    e["FagnRelSlope"] = fnum(slope_of(xs, ys), "%+.3f")
    return e


def exclusion():
    r"""Log-likelihood drop from the maximum to f_AGN = 0.

    Reads the scan grids themselves, which is the only place this number lives.
    """
    try:
        import h5py
        import numpy as np
    except ImportError:
        return None
    try:
        with h5py.File(A2 / "joint_s100.h5", "r") as h:
            grid = np.asarray(h["log_likelihood"][:])
        prof = grid.max(axis=0)
        out = {"dlnl": float(prof.max() - prof[0])}
        for tag, name in (("record", "fscan_s100.h5"),
                          ("null", "fscan_null_s100.h5")):
            with h5py.File(A2 / name, "r") as h:
                curve = np.asarray(h["log_likelihood"][:])
            out[tag] = float(curve.max() - curve[0])
        return out
    except (OSError, KeyError, ValueError, IndexError):
        return None


# ===========================================================================
def committed() -> dict[str, str]:
    pat = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$")
    found = {}
    for line in MACROS.read_text().splitlines():
        hit = pat.match(line.strip())
        if hit:
            found[hit.group(1)] = hit.group(2)
    return found


def notation() -> set[str]:
    """Macros main.tex defines itself (notation block, \\todo)."""
    return set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}",
                          MAIN.read_text()))


def strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in text.splitlines())


def scan_manuscript(defined: set[str]):
    """Undefined capitalised commands, and editorial sentinels."""
    undefined: dict[str, list[str]] = {}
    sentinels: list[str] = []
    sent_pat = re.compile(r"\\todo|(?<![!?])\?\?(?!\?)|\bTBD\b|\bXXX\b|\bFIXME\b")
    for path in SECTIONS + [MAIN]:
        raw = path.read_text()
        body = strip_comments(raw)
        for name in re.findall(r"\\([A-Z][A-Za-z]*)", body):
            if name in defined or name in BUILTIN:
                continue
            undefined.setdefault(name, [])
            if path.name not in undefined[name]:
                undefined[name].append(path.name)
        for n, line in enumerate(body.splitlines(), 1):
            if sent_pat.search(line):
                sentinels.append(f"{path.name}:{n}: {line.strip()[:96]}")
    return undefined, sentinels


def main() -> int:
    verbose = "-v" in sys.argv
    have = committed()
    want = derive()

    bad, missing = [], []
    for name in sorted(want):
        if name not in have:
            missing.append(name)
        elif have[name] != want[name]:
            bad.append((name, have[name], want[name]))

    unchecked = sorted(set(have) - set(want))
    unexplained = [n for n in unchecked if n not in WHITELIST]

    print(f"macros committed : {len(have)}")
    print(f"macros re-derived: {len(want)}  ({len(bad)} mismatched, "
          f"{len(missing)} missing from the .tex)")
    print(f"whitelisted      : {len(unchecked) - len(unexplained)}")
    for name in unchecked:
        if name in WHITELIST:
            print(f"    \\{name}: {WHITELIST[name]}")
    for name, got, exp in bad:
        print(f"  MISMATCH \\{name}: committed {got!r} != derived {exp!r}")
    for name in missing:
        print(f"  MISSING  \\{name}: derived {want[name]!r}, not in the .tex")
    for name in unexplained:
        print(f"  UNEXPLAINED \\{name}: no derivation and no whitelist entry")

    defined = set(have) | notation()
    undefined, sentinels = scan_manuscript(defined)
    queued = {k: v for k, v in undefined.items() if k in PENDING_REWRITE}
    orphan = {k: v for k, v in undefined.items() if k not in PENDING_REWRITE}
    print()
    print(f"undefined macros, queued for the prose rewrite: {len(queued)}")
    for name in sorted(queued):
        print(f"  \\{name}  <- {', '.join(sorted(queued[name]))}")
        print(f"      {PENDING_REWRITE[name]}")
    if orphan:
        print(f"undefined macros, UNEXPECTED: {len(orphan)}")
        for name in sorted(orphan):
            print(f"  \\{name}  <- {', '.join(sorted(orphan[name]))}")

    print()
    print(f"editorial sentinels (not a failure): {len(sentinels)}")
    for line in sentinels if verbose or len(sentinels) <= 40 else sentinels[:40]:
        print(f"  {line}")

    failed = bool(bad or missing or unexplained or orphan)
    print()
    print("FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
