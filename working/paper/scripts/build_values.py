#!/usr/bin/env python3
r"""Build every number the manuscript quotes.

Emits
  values/results_macros.tex   -- one \newcommand per quoted number
  NUMBERS.md                  -- macro -> source file audit trail

Nothing in the manuscript body is hand-typed.  Prose and captions reference
macros; the macros are registered here, and every macro carries the path of the
file that fixes its value.  Run this before latex.

There are three kinds of entry, and they are kept apart on purpose.

  CONFIGURATION  -- constants that *define* the simulated universe and the
                    estimator's fiducial setup.  Inputs, not measurements.

  DATASET        -- properties of the generated dataset, read out of its
                    metadata file.  These are measured on the realisation, not
                    chosen.

  RESULTS        -- numbers computed from an inference run's output files.

`add(..., None)` renders a loud \todo so a half-finished number cannot hide in
the PDF.  Every hook degrades to that when its source file is absent.

Record of 2026-08-02
--------------------
The dataset of record is **v3 + D3** (`working/data/seed<S>`, five seeds
{100, 101, 102, 103, 105}; reference realisation seed 100).  The v3 measurement
family replaced v2's fixed fractional widths: the recorded amplitude
`rho_obs = rho_opt(theta) + N(0, sigma_rho)` is the one datum, detection is
`rho_obs >= rho_th`, *every* other width is `A_x (rho_th / rho_obs)`, and the
luminosity distance is DERIVED from `(Mc_det, rho)` rather than measured on its
own.  The v2 macros for a flat fractional distance error and for fixed
fractional mass widths therefore no longer name anything in the dataset and are
gone; see the rename map in NUMBERS.md.

Results come from the two analyses of record:
  analysis_1_complete_catalog_H0        single-tracer K = 1 baselines + the
                                        matched-host controls (five seeds)
  analysis_2_complete_catalog_H0_fagn   the joint K = 2 (H0, f_AGN) measurement,
                                        its five-seed closure and its sky-shuffle
                                        null
and, for the appendix,
  analysis_0_pure_tracer_H0             two further 1000-event sets per
                                        realisation, one all-galaxy-hosted and
                                        one all-AGN-hosted, each measured
                                        against its own catalog (`\Pure*`)
Analysis 3 (incomplete catalogs) has not landed; its three macros are the only
ones that still render `\todo{pending}`.

Rounding convention
-------------------
* Where a source JSON quotes a credible interval as a **string**
  (`"69.2^{+1.0}_{-1.0}"`), that string is copied verbatim -- the analysis owns
  its own rounding.
* Otherwise: H0-like quantities (km/s/Mpc) get 1 decimal for locations and
  2 decimals for interval widths; f-like quantities (dimensionless fractions)
  get 3 decimals; ratios and sigma-counts get 1 decimal; percentages get 0
  decimals unless the value is below 1 per cent.
* Counts are exact.  Densities and injection counts are rendered in scientific
  notation with two significant figures.

Usage
    python scripts/build_values.py
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent
WORKING = PAPER.parent

# --- sources -----------------------------------------------------------------
# The dataset metadata file is the single source for everything in Section 3.
# It is written by the generator at the end of every run and carries the config
# it was run with, the realised catalog and event properties, and the outcome of
# every consistency check.  DESIGN_PE.md is its prose companion: it fixes the
# v3 measurement family and cites it.  Every constant quoted below is read from
# META.json, never from DESIGN_PE.md, so that the numbers track the data.
DATASET = WORKING / "data" / "seed100"
META = DATASET / "META.json"
DESIGN = WORKING / "data" / "DESIGN_PE.md"

# Inference runs.  One directory per measurement; each writes summary files that
# the results hooks below read.
ANALYSES = WORKING / "analyses"
A0 = ANALYSES / "analysis_0_pure_tracer_H0" / "results"
A1 = ANALYSES / "analysis_1_complete_catalog_H0" / "results"
A2 = ANALYSES / "analysis_2_complete_catalog_H0_fagn" / "results"

C_KM_S = 299792.458

# ---------------------------------------------------------------------------
# macro registry
# ---------------------------------------------------------------------------
REGISTRY: list[dict] = []
_SEEN: set[str] = set()


def add(name: str, value, fmt: str = "%.3f", *, src: str = "", note: str = "",
        kind: str = "configuration") -> str:
    """Register one macro.  `src` is the file that fixes the value."""
    if name in _SEEN:
        raise SystemExit(f"duplicate macro name: {name}")
    _SEEN.add(name)
    if value is None:
        body = r"\todo{pending}"
    elif isinstance(value, str):
        body = value
    elif fmt == "sci":
        body = rf"\ensuremath{{{_sci(value)}}}"
    elif fmt == "intsep":
        body = f"{int(round(value)):,}".replace(",", r"\,")
    elif fmt.startswith("%+"):
        # signed numbers need a real minus sign and must work in both modes
        body = rf"\ensuremath{{{fmt % value}}}"
    else:
        body = fmt % value
    REGISTRY.append({"name": name, "body": body, "src": str(src), "note": note,
                     "kind": kind, "raw": value})
    return name


def _sci(v: float, sig: int = 2) -> str:
    if v == 0:
        return "0"
    e = int(math.floor(math.log10(abs(v))))
    m = v / 10 ** e
    if abs(m - 1.0) < 5e-3:
        return rf"10^{{{e}}}"
    return rf"{m:.{sig - 1}f}\times10^{{{e}}}"


def ci(value: str | None) -> str | None:
    r"""Wrap a JSON-quoted interval string so it is safe outside math mode."""
    return None if value is None else rf"\ensuremath{{{value}}}"


def pm(mean, err, fmt="%+.2f", efmt="%.2f") -> str | None:
    """Render `mean +- err` as one math-mode body."""
    if mean is None or err is None:
        return None
    return rf"\ensuremath{{{fmt % mean} \pm {efmt % err}}}"


def bracket(lo, hi, fmt="%.3f") -> str | None:
    if lo is None or hi is None:
        return None
    return rf"\ensuremath{{[{fmt % lo},\, {fmt % hi}]}}"


# ---------------------------------------------------------------------------
# source access
# ---------------------------------------------------------------------------
def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return {}


def get(tree: dict, path: str):
    """Dotted lookup; None if any level is missing.  Integer keys index lists."""
    node = tree
    for key in path.split("."):
        if isinstance(node, list):
            try:
                node = node[int(key)]
            except (ValueError, IndexError):
                return None
        elif isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return None
    return node


def pct(v):
    return None if v is None else 100.0 * v


def rel(p) -> str:
    """Path as the campaign names it: relative to `working/`, symlinks intact.

    `data/seed<S>` is a symlink into the shared data root, so resolving it would
    print a path nobody uses.  Resolve only as a fallback.
    """
    p = Path(p)
    for cand in (p, p.resolve()):
        try:
            return str(cand.relative_to(WORKING))
        except ValueError:
            continue
    return str(p)


META_SRC = rel(META)
S_PURE = rel(A0 / "h0_pure_tracer.json")
S_SINGLE = rel(A1 / "h0_single_tracer.json")
S_CLOSURE = rel(A1 / "closure_v3.json")
S_CURV = rel(A1 / "v3_curvature.json")
S_KDE = rel(A1 / "kde_window.json")
S_JOINT = rel(A2 / "h0_fagn_joint.json")
S_JSUM = rel(A2 / "joint_summary.json")
S_FSCAN = rel(A2 / "fscan_s100.json")
S_FNULL = rel(A2 / "fscan_null_s100.json")
S_MUMC = rel(A2 / "mu_mc_error.json")


# ---------------------------------------------------------------------------
# a dependency-free flat-LCDM distance, for the one derived design number
# ---------------------------------------------------------------------------
def _lum_dist(z, H0, Om0, n=2000):
    if z <= 0:
        return 0.0
    h = z / n
    tot = 0.0
    for i in range(n + 1):
        zi = i * h
        w = 1 if i in (0, n) else (4 if i % 2 else 2)
        tot += w / math.sqrt(Om0 * (1 + zi) ** 3 + (1 - Om0))
    return (1 + z) * (C_KM_S / H0) * tot * h / 3.0


def _sigma_z_pe(z, H0, Om0, sigma_lndl):
    """Redshift width implied by a fractional distance uncertainty at z."""
    dl = _lum_dist(z, H0, Om0)
    ez = math.sqrt(Om0 * (1 + z) ** 3 + (1 - Om0))
    ddl_dz = dl / (1 + z) + (1 + z) * (C_KM_S / H0) / ez
    return sigma_lndl * dl / ddl_dz


# ===========================================================================
# 1. Configuration and realised properties of the simulated universe
# ===========================================================================
def sec_cosmology(m):
    add("HzeroTruth", get(m, "config.cosmology.H0"), "%.2f", src=META_SRC,
        note="fiducial expansion rate of the mock, km/s/Mpc")
    add("OmZeroTruth", get(m, "config.cosmology.Om0"), "%.4f", src=META_SRC,
        note="matter density, pinned in the inference")
    add("ObZeroTruth", get(m, "config.glass.Ob0"), "%.4f", src=META_SRC,
        note="baryon density used for the matter power spectrum")


def sec_field(m):
    """The shared lognormal density field and the two tracers sampled from it."""
    add("FieldZmax", get(m, "config.glass.z_max"), "%.1f", src=META_SRC,
        note="redshift extent of the density field and of the catalogs")
    add("FieldShellDx", get(m, "config.glass.dx_mpc"), "%.0f", src=META_SRC,
        note="comoving shell spacing, Mpc")
    add("FieldNshell", get(m, "stages.catalogs.n_shells"), "%d", src=META_SRC,
        note="number of radial shells")
    add("FieldNside", get(m, "config.glass.nside"), "%d", src=META_SRC,
        note="HEALPix resolution of the density field")
    add("FieldLmax", get(m, "config.glass.lmax"), "%d", src=META_SRC,
        note="highest multipole of the field")

    add("NGal", get(m, "config.glass.n_comoving_gal"), "sci", src=META_SRC,
        note="galaxy comoving number density, Mpc^-3")
    add("NAgn", get(m, "config.glass.n_comoving_agn"), "sci", src=META_SRC,
        note="AGN comoving number density, Mpc^-3")
    ngal = get(m, "config.glass.n_comoving_gal")
    nagn = get(m, "config.glass.n_comoving_agn")
    add("DensityRatioTarget", None if None in (ngal, nagn) else ngal / nagn,
        "%.0f", src=META_SRC, note="input galaxy/AGN number-density ratio")

    add("BiasGal", get(m, "config.glass.bias_gal"), "%.1f", src=META_SRC,
        note="linear bias of the galaxies")
    add("BiasAgn", get(m, "config.glass.bias_agn"), "%.1f", src=META_SRC,
        note="linear bias of the AGN")
    bg = get(m, "config.glass.bias_gal")
    ba = get(m, "config.glass.bias_agn")
    add("BiasRatio", None if None in (bg, ba) else ba / bg, "%.2f", src=META_SRC,
        note="input AGN/galaxy bias contrast")

    # --- realised on the catalogs
    add("CatNgal", get(m, "stages.catalogs.tracers.gal.n"), "sci",
        src=META_SRC, note="galaxies in the complete catalog", kind="dataset")
    add("CatNagn", get(m, "stages.catalogs.tracers.agn.n"), "sci",
        src=META_SRC, note="AGN in the complete catalog", kind="dataset")
    add("DensityRatio", get(m, "stages.catalogs.density_ratio_gal_over_agn"),
        "%.1f", src=META_SRC, note="realised number-density ratio",
        kind="dataset")

    v4 = "stages.validation.checks.V4_catalog_densities_and_clustering."
    bmeas = get(m, v4 + "bias_ratio_agn_over_gal.measured")
    berr = get(m, v4 + "bias_ratio_agn_over_gal.err")
    add("BiasRatioMeas",
        None if bmeas is None else
        rf"\ensuremath{{{bmeas:.2f} \pm {max(berr or 0.0, 5e-3):.2f}}}",
        src=META_SRC, kind="dataset",
        note="AGN/galaxy bias ratio recovered from the cross-correlation of the "
             "two catalogs")


def sec_luminosity(m):
    """The luminosity function the magnitudes and the flux limit act through."""
    add("PhiStar", get(m, "config.luminosity_function.phi_star_h3"), "sci",
        src=META_SRC, note="Schechter normalisation, h^3 Mpc^-3")
    add("SchechterAlpha", get(m, "config.luminosity_function.alpha"), "%+.2f",
        src=META_SRC, note="faint-end slope")
    add("MagBStar", get(m, "config.luminosity_function.M_B_star"), "%+.2f",
        src=META_SRC, note="B-band characteristic magnitude")
    add("LumCut", get(m, "stages.catalogs.magnitude_model.x_cut_L_over_Lstar"),
        "%.2f", src=META_SRC, kind="dataset",
        note="luminosity cut, in units of L*, that yields the adopted density")
    add("MagBLimit",
        get(m, "stages.catalogs.magnitude_model.M_B_faint_limit"), "%+.2f",
        src=META_SRC, kind="dataset", note="absolute-magnitude limit of the cut")


def sec_events(m):
    """Host assignment, source population, the v3 measurement family, detection."""
    add("EvNobs", get(m, "config.events.n"), "intsep", src=META_SRC,
        note="detected events analysed")
    add("EvFagn", get(m, "config.events.f_agn"), "%.2f", src=META_SRC,
        note="planted AGN-hosted fraction")
    add("EvNsamp", get(m, "config.events.nsamp"), "intsep", src=META_SRC,
        note="posterior samples per event")

    # ---- the v3 measurement family.  One recorded amplitude sets everything.
    v3 = "config.events.v3_measurement_family."
    rho_th = get(m, "config.events.snr_threshold")
    add("EvSnrThresh", rho_th, "%.0f", src=META_SRC,
        note="detection threshold on the recorded amplitude")
    add("EvSigmaRho", get(m, v3 + "sigma_rho"), "%.1f", src=META_SRC,
        note="width of the amplitude measurement, rho_obs = rho_opt + N(0, "
             "sigma_rho)")
    add("EvAmc", get(m, v3 + "A_MC"), "%.2f", src=META_SRC,
        note="log-chirp-mass width coefficient, sigma_lnMc = A_Mc rho_th/rho_obs")
    add("EvAq", get(m, v3 + "A_Q"), "%.2f", src=META_SRC,
        note="log-mass-ratio width coefficient")
    add("EvAchi", get(m, v3 + "A_CHI"), "%.2f", src=META_SRC,
        note="effective-spin width coefficient")
    add("EvWidthRefSnr", rho_th, "%.0f", src=META_SRC,
        note="reference amplitude of the width law; equal to the detection "
             "threshold, so every width is A_x at threshold")
    add("EvSnrRef", get(m, "config.events.snr_ref_detect"), "%.2f",
        src=META_SRC,
        note="amplitude of a 30 Msun detector-frame chirp mass at 1 Gpc")
    add("EvSnrRefSky", get(m, "config.events.snr_ref_sigma_ang"), "%.1f",
        src=META_SRC, note="amplitude scale the sky-width convention is written on")
    add("EvSkyCoef", get(m, v3 + "sky_a_deg"), "%.0f", src=META_SRC,
        note="sky-width numerator, degrees")
    add("EvSkyMin", get(m, v3 + "sky_clip_deg.0"), "%.0f", src=META_SRC,
        note="sky-width floor, degrees")
    add("EvSkyMax", get(m, v3 + "sky_clip_deg.1"), "%.0f", src=META_SRC,
        note="sky-width ceiling, degrees")

    ref_d = get(m, "config.events.snr_ref_detect")
    ref_s = get(m, "config.events.snr_ref_sigma_ang")
    scale = None if None in (ref_d, ref_s) else ref_s / ref_d
    add("EvSkyRhoScale", scale, "%.3f", src=META_SRC,
        note="rho_sigma = (snr_ref_sky / snr_ref_detect) rho_obs; the factor")
    a_deg = get(m, v3 + "sky_a_deg")
    add("EvSkyCoefEff", None if None in (a_deg, scale) else a_deg / scale,
        "%.1f", src=META_SRC,
        note="sky-width law folded onto the recorded amplitude, "
             "sigma_ang = coef/rho_obs degrees")

    # the distance is derived, not measured: sigma_ln dL follows from the two
    # channels that build it, ln dL = const + (5/6) ln Mc_det - ln rho.
    a_mc = get(m, v3 + "A_MC")
    s_rho = get(m, v3 + "sigma_rho")
    coef = None
    if None not in (a_mc, s_rho, rho_th):
        coef = math.hypot(5.0 / 6.0 * a_mc * rho_th, s_rho)
    add("EvSigmaLnDlCoef", coef, "%.3f", src=META_SRC,
        note="sigma_ln dL = coef / rho_obs, from the chirp-mass and amplitude "
             "channels alone")
    add("EvSigmaLnDlThresh", pct(None if coef is None else coef / rho_th),
        "%.1f", src=META_SRC,
        note="per cent distance precision of a threshold event")

    # population
    p = "stages.events.population."
    add("PopAlpha", get(m, p + "alpha"), "%.1f", src=META_SRC,
        note="primary-mass power-law index")
    add("PopMmin", get(m, p + "mmin"), "%.0f", src=META_SRC, note="Msun")
    add("PopMmax", get(m, p + "mmax"), "%.0f", src=META_SRC, note="Msun")
    add("PopDmMin", get(m, p + "dm_min"), "%.0f", src=META_SRC,
        note="low-mass taper width, Msun")
    add("PopDmMax", get(m, p + "dm_max"), "%.0f", src=META_SRC,
        note="high-mass taper width, Msun")
    add("PopPeakFrac", get(m, p + "peak_fraction"), "%.2f", src=META_SRC,
        note="weight of the Gaussian peak")
    add("PopPeakMu", get(m, p + "peak_mu"), "%.0f", src=META_SRC, note="Msun")
    add("PopPeakSigma", get(m, p + "peak_sigma"), "%.0f", src=META_SRC,
        note="Msun")
    add("PopBeta", get(m, p + "beta"), "%.0f", src=META_SRC,
        note="mass-ratio pairing index")
    add("PopChiSigma", get(m, p + "chi_sigma"), "%.2f", src=META_SRC,
        note="effective-spin width")
    add("PopGamma", get(m, p + "gamma"), "%.0f", src=META_SRC,
        note="merger-rate redshift index")

    # --- realised on the event set
    r = "stages.events.realised."
    add("EvFagnReal", get(m, r + "realised_f_agn"), "%.3f", src=META_SRC,
        kind="dataset", note="AGN-hosted fraction actually drawn")
    add("EvNhostGal", get(m, r + "n_host_gal"), "intsep", src=META_SRC,
        kind="dataset", note="galaxy-hosted events")
    add("EvNhostAgn", get(m, r + "n_host_agn"), "intsep", src=META_SRC,
        kind="dataset", note="AGN-hosted events")
    add("EvNuniqueAgn", get(m, r + "unique_agn_hosts"), "intsep",
        src=META_SRC, kind="dataset",
        note="distinct AGN hosting at least one event")
    add("EvMaxPerAgn", get(m, r + "max_events_per_agn_host"), "%d",
        src=META_SRC, kind="dataset", note="most events sharing one AGN host")
    add("EvHorizonZ", get(m, r + "horizon_z_max_detected"), "%.2f",
        src=META_SRC, kind="dataset", note="highest detected redshift")
    add("EvZmedian", get(m, r + "z_median_detected"), "%.2f", src=META_SRC,
        kind="dataset", note="median detected redshift")
    add("EvNproposed", get(m, r + "n_proposed"), "intsep", src=META_SRC,
        kind="dataset", note="sources proposed to reach the detected set")
    add("EvNdetTotal", get(m, r + "n_detected_total"), "intsep", src=META_SRC,
        kind="dataset",
        note="sources that passed the detection cut; the first EvNobs of them "
             "are the analysed set")
    add("EvDetFrac", pct(get(m, r + "detected_fraction")), "%.3f",
        src=META_SRC, kind="dataset",
        note="per cent of proposals that pass the detection cut")
    add("EvMalmquist", pct(get(m, r + "frac_detected_with_true_snr_below_"
                                   "threshold")), "%.0f",
        src=META_SRC, kind="dataset",
        note="per cent of detected events whose true amplitude is below "
             "threshold")

    # realised widths: the whole point of v3 is that these follow from rho_obs
    v1 = "stages.validation.checks.V1_detection_deterministic_in_data."
    v2 = "stages.validation.checks.V2_widths_from_observed_snr."
    rho_med = get(m, v1 + "rho_obs_median")
    add("EvSnrMedian", rho_med, "%.2f", src=META_SRC, kind="dataset",
        note="median recorded amplitude of the detected set")
    add("EvSnrMax", get(m, v1 + "rho_obs_max"), "%.1f", src=META_SRC,
        kind="dataset", note="largest recorded amplitude")
    for tag, key, f in (("EvSigLnMcMedian", "A_MC", "%.4f"),
                        ("EvSigLnQMedian", "A_Q", "%.3f"),
                        ("EvSigChiMedian", "A_CHI", "%.3f")):
        a = get(m, v3 + key)
        add(tag, None if None in (a, rho_med, rho_th) else a * rho_th / rho_med,
            f, src=META_SRC, kind="dataset",
            note=f"median realised width of the {key} channel")
    add("EvSigmaLnDlMedian",
        get(m, "stages.validation.checks.V3_pe_calibration."
               "sigma_lndL_realised_median"), "%.4f", src=META_SRC,
        kind="dataset", note="median realised log-distance width")
    add("EvSkyMinReal", get(m, v2 + "sigma_ang_deg_min"), "%.2f", src=META_SRC,
        kind="dataset", note="smallest realised sky width, degrees")
    add("EvSkyMaxReal", get(m, v2 + "sigma_ang_deg_max"), "%.2f", src=META_SRC,
        kind="dataset", note="largest realised sky width, degrees")


def sec_survey(m):
    """The photo-z the catalogs carry and the flux limits that thin them."""
    lims = sorted(get(m, "config.survey.mag_limits") or [], reverse=True)
    add("SurveyNside", get(m, "config.survey.nside"), "%d", src=META_SRC,
        note="HEALPix resolution of the pixelated catalogs")
    add("SurveyNpix", None if get(m, "config.survey.nside") is None else
        12 * get(m, "config.survey.nside") ** 2, "intsep", src=META_SRC,
        note="pixels in the pixelated catalogs")
    add("CatDzScale", get(m, "stages.surveys.dz_scale"), "sci", src=META_SRC,
        note="catalog redshift error, realised as z_obs = z + N(0, eps (1+z)) "
             "and declared to the likelihood as sigma_z = eps (1+z_obs)")
    pz = "stages.catalogs.photoz_model.realised."
    add("CatPhotozPullSd", get(m, pz + "gal.pull_sd"), "%.4f", src=META_SRC,
        kind="dataset",
        note="standard deviation of the realised galaxy photo-z pull")
    add("CatPhotozPullSdAgn", get(m, pz + "agn.pull_sd"), "%.4f", src=META_SRC,
        kind="dataset", note="the same for the AGN catalog")

    add("MagLimDeep", max(lims) if lims else None, "%.0f", src=META_SRC,
        note="deepest apparent-magnitude limit")
    add("MagLimShallow", min(lims) if lims else None, "%.0f", src=META_SRC,
        note="shallowest apparent-magnitude limit")
    add("MagLimStep", abs(lims[0] - lims[1]) if len(lims) > 1 else None, "%.0f",
        src=META_SRC, note="spacing of the limits, mag")
    add("MagLimN", len(lims) or None, "%d", src=META_SRC,
        note="number of flux-limited catalogs")

    # completeness inside the volume the events occupy
    c = "stages.surveys.completeness."
    for tag, lim in (("MTwentyOne", "m21"), ("MTwenty", "m20"),
                     ("MNineteen", "m19"), ("MEighteen", "m18")):
        add(f"Comp{tag}", pct(get(m, c + lim + ".gal.C_within_horizon")),
            "%.0f", src=META_SRC, kind="dataset",
            note=f"galaxy completeness within the horizon at m < {lim[1:]}")
        add(f"CompAgn{tag}", pct(get(m, c + lim + ".agn.C_within_horizon")),
            "%.0f", src=META_SRC, kind="dataset",
            note=f"AGN completeness within the horizon at m < {lim[1:]}")
    devs = []
    for lim in ("m21", "m20", "m19", "m18"):
        g = get(m, c + lim + ".gal.C_within_horizon")
        a = get(m, c + lim + ".agn.C_within_horizon")
        if None not in (g, a):
            devs.append(abs(100.0 * (a - g)))
    add("CompAgnMaxDev", max(devs) if devs else None, "%.1f", src=META_SRC,
        kind="dataset",
        note="largest difference between the AGN and galaxy completeness "
             "within the horizon, percentage points")

    add("AgnEmptyPix",
        pct(get(m, "stages.surveys.surveys.agn_complete.empty_pixel_"
                   "fraction")), "%.0f", src=META_SRC, kind="dataset",
        note="per cent of pixels with no AGN, complete catalog")
    add("AgnEmptyPixShallow",
        pct(get(m, "stages.surveys.surveys.agn_m18.empty_pixel_fraction")),
        "%.0f", src=META_SRC, kind="dataset",
        note="per cent of pixels with no AGN at the shallowest limit")
    add("AgnPerPix",
        get(m, "stages.surveys.surveys.agn_complete.max_hosts_per_pixel"),
        "intsep", src=META_SRC, kind="dataset",
        note="most AGN in a single pixel, complete catalog")


def sec_selection(m, jsum):
    """The injection sets behind the selection-function estimate."""
    add("InjNdrawMain", get(m, "stages.injections.targeted.ndraw"), "sci",
        src=META_SRC, kind="dataset",
        note="injections drawn against the catalogued AGN")
    add("InjNdrawCross", get(m, "stages.injections.popuni.ndraw"), "sci",
        src=META_SRC, kind="dataset",
        note="injections drawn from the population and volume")
    add("InjNdetMain", get(m, "stages.injections.targeted.n_detected"),
        "intsep", src=META_SRC, kind="dataset",
        note="detected injections, targeted lane")
    add("InjNdetCross", get(m, "stages.injections.popuni.n_detected"),
        "intsep", src=META_SRC, kind="dataset",
        note="detected injections, cross-check lane")
    w = get(m, "stages.injections.targeted.mixture_weights")
    add("InjWeightPop", None if not w else w[0], "%.2f", src=META_SRC,
        note="weight of the population branch")
    add("InjWeightUni", None if not w else w[1], "%.2f", src=META_SRC,
        note="weight of the uniform-in-volume branch")
    add("InjWeightAgn", None if not w else w[2], "%.2f", src=META_SRC,
        note="weight of the branch placed on catalogued AGN")

    # the validity criterion the scans enforce, read off the run itself
    thr = get(jsum, "seeds.0.joint.guard.threshold_min")
    nobs = get(jsum, "seeds.0.n_events")
    add("NeffFactor", None if None in (thr, nobs) else thr / nobs, "%.0f",
        src=S_JSUM, kind="result",
        note="selection-estimate validity criterion, Neff > factor x Nobs")
    add("NeffMin", get(jsum, "seeds.0.joint.guard.Neff_min"), "sci",
        src=S_JSUM, kind="result",
        note="smallest effective injection count anywhere on the joint grid")


def sec_design(m, kde):
    """Design margins the manuscript states as requirements."""
    v8 = "stages.validation.checks.V8_catalog_edge_clears_pe_support."
    add("HzeroScanLo",
        get(m, "stages.injections.targeted.targeted_branch.H0_scan_range.0"),
        "%.0f", src=META_SRC, note="low edge of the analysed H0 range")
    add("HzeroScanHi",
        get(m, "stages.injections.targeted.targeted_branch.H0_scan_range.1"),
        "%.0f", src=META_SRC, note="high edge of the analysed H0 range")
    add("PeZmax", get(m, v8 + "max_pe_redshift_over_H0_grid"), "%.2f",
        src=META_SRC, kind="dataset",
        note="highest redshift any posterior sample reaches, maximised over "
             "the analysed H0 range")

    z = get(m, "stages.events.realised.z_median_detected")
    H0 = get(m, "config.cosmology.H0")
    Om0 = get(m, "config.cosmology.Om0")
    sd = get(m, "stages.validation.checks.V3_pe_calibration."
                "sigma_lndL_realised_median")
    eps = get(m, "stages.surveys.dz_scale")
    ratio = None
    if None not in (z, H0, Om0, sd, eps):
        ratio = _sigma_z_pe(z, H0, Om0, sd) / (eps * (1 + z))
    add("KdePeRatio", ratio, "%.1f", src=META_SRC, kind="dataset",
        note="ratio of the distance posteriors' redshift width to the catalog "
             "kernel width, at the median detected redshift")
    add("KdeWindow", kde.get("window_recommended_power_of_two"), "intsep",
        src=S_KDE, kind="result",
        note="width of the per-pixel catalog window the likelihood evaluates")


# ===========================================================================
# 2. Results
# ===========================================================================
def sec_single():
    """Single-tracer K = 1 baselines on the reference realisation (seed 100)."""
    s = load_json(A1 / "h0_single_tracer.json")

    add("HzeroGal", ci(s.get("gal_h0_ci")), src=S_SINGLE, kind="result",
        note="H0 from the galaxy catalog alone, median and 68% interval")
    add("HzeroGalMedian", s.get("gal_h0_median"), "%.1f", src=S_SINGLE,
        kind="result", note="the same, median only")
    add("HzeroGalWidth", s.get("gal_h0_width"), "%.2f", src=S_SINGLE,
        kind="result", note="68% width of the galaxy-catalog H0 posterior, "
                            "km/s/Mpc")
    add("HzeroGalCross", s.get("gal_h0_crosscheck_median"), "%.1f",
        src=S_SINGLE, kind="result",
        note="the same measurement on the independent injection lane")

    # The AGN-only posterior is mis-specified against a mixed universe and rails
    # at the top of the scanned range: it has no interval to quote, and the
    # manuscript says so with these two numbers against \HzeroScanHi instead.
    add("HzeroAgnRailMedian", s.get("agn_grid_top_median"), "%.1f",
        src=S_SINGLE, kind="result",
        note="median of the railed AGN-only posterior, km/s/Mpc")
    add("HzeroAgnRailCross", s.get("agn_h0_crosscheck_median"), "%.1f",
        src=S_SINGLE, kind="result",
        note="the same on the independent injection lane, also railed")


def sec_joint(j, jsum, fs, fn, hj):
    """The joint (H0, f_AGN) measurement, its closure and its null."""
    add("HzeroJoint", ci(j.get("h0_ci")), src=S_JOINT, kind="result",
        note="H0 from the joint two-tracer fit, median and 68% interval")
    add("HzeroJointMedian", j.get("h0_median"), "%.1f", src=S_JOINT,
        kind="result", note="the same, median only")
    add("HzeroJointWidth", j.get("h0_width"), "%.2f", src=S_JOINT,
        kind="result", note="68% width of the joint H0 posterior, km/s/Mpc")
    add("HzeroJointCross", j.get("h0_crosscheck_median"), "%.1f", src=S_JOINT,
        kind="result", note="the same on the independent injection lane")
    add("HzeroJointMap", get(j, "map.H0"), "%.2f", src=S_JOINT, kind="result",
        note="H0 at the maximum of the joint likelihood")

    add("FagnJoint", ci(j.get("f_ci")), src=S_JOINT, kind="result",
        note="AGN-hosted fraction from the joint fit, median and 68% interval")
    add("FagnJointMedian", j.get("f_median"), "%.3f", src=S_JOINT,
        kind="result", note="the same, median only")
    add("FagnJointWidth", j.get("f_width"), "%.3f", src=S_JOINT, kind="result",
        note="68% width of the joint f_AGN posterior")
    add("FagnJointCross", j.get("f_crosscheck_median"), "%.3f", src=S_JOINT,
        kind="result", note="the same on the independent injection lane")
    add("FagnJointMap", get(j, "map.f"), "%.3f", src=S_JOINT, kind="result",
        note="f_AGN at the maximum of the joint likelihood")

    add("FagnTruthReal", j.get("truth_f_realised"), "%.3f", src=S_JOINT,
        kind="dataset", note="AGN-hosted fraction the drawn events actually "
                             "contain on the reference realisation")
    add("FagnTruthPlanted", j.get("truth_f_planted"), "%.2f", src=S_JOINT,
        note="AGN-hosted fraction the events were drawn with")
    add("FagnBinomialSd", jsum.get("binomial_sd_per_realisation"), "%.3f",
        src=S_JSUM, kind="dataset",
        note="binomial scatter of the realised fraction about the planted one, "
             "per realisation")

    add("JointRho", j.get("rho"), "%.3f", src=S_JOINT, kind="result",
        note="correlation of H0 and f_AGN in the joint posterior, reference "
             "realisation")
    add("JointRhoMean", pm(get(jsum, "closure.rho.mean"),
                           get(jsum, "closure.rho.sem"), "%+.3f", "%.3f"),
        src=S_JSUM, kind="result",
        note="the same, mean and standard error over the five realisations")
    add("ClosureScatterHzeroRatio", get(jsum, "closure.scatter_H0.ratio"),
        "%.2f", src=S_JSUM, kind="result",
        note="sd of the five H0 medians over the mean quoted 68% half-width")
    add("ClosureScatterFagnRatio", get(jsum, "closure.scatter_f.ratio"),
        "%.2f", src=S_JSUM, kind="result",
        note="the same for f_AGN")

    # --- the width the second tracer buys
    gw = load_json(A1 / "h0_single_tracer.json").get("gal_h0_width")
    jw = j.get("h0_width")
    add("HzeroWidthRatio", None if None in (gw, jw) or not jw else gw / jw,
        "%.1f", src=f"{S_SINGLE} + {S_JOINT}", kind="result",
        note="galaxy-only 68% H0 width divided by the joint 68% H0 width")

    # --- five-realisation closure
    add("ClosureNseeds", j.get("closure_n_seeds"), "%d", src=S_JOINT,
        kind="result", note="independent realisations of the whole mock")
    add("ClosureHzero", pm(j.get("closure_h0_offset_mean"),
                           j.get("closure_h0_offset_sem")),
        src=S_JOINT, kind="result",
        note="mean H0 offset from truth over the realisations, km/s/Mpc")
    add("ClosureFagnReal",
        pm(j.get("closure_f_offset_vs_realised_mean"),
           j.get("closure_f_offset_vs_realised_sem"), "%+.3f", "%.3f"),
        src=S_JOINT, kind="result",
        note="mean f_AGN offset from the realised host fraction")
    add("ClosureFagnPlanted",
        pm(j.get("closure_f_offset_vs_planted_mean"),
           j.get("closure_f_offset_vs_planted_sem"), "%+.3f", "%.3f"),
        src=S_JOINT, kind="result",
        note="mean f_AGN offset from the planted fraction")
    add("ClosureHzeroInSixtyEight", get(jsum, "closure.coverage.H0_in_68"),
        "%d", src=S_JSUM, kind="result",
        note="realisations whose 68% H0 interval contains truth")
    add("ClosureHzeroInNinety", get(jsum, "closure.coverage.H0_in_90"), "%d",
        src=S_JSUM, kind="result",
        note="realisations whose 90% H0 interval contains truth")
    add("ClosureFagnInSixtyEight",
        get(jsum, "closure.coverage.f_realised_in_68"), "%d", src=S_JSUM,
        kind="result",
        note="realisations whose 68% f_AGN interval contains the realised "
             "fraction")
    add("ClosureFagnInNinety", get(jsum, "closure.coverage.f_realised_in_90"),
        "%d", src=S_JSUM, kind="result", note="the same at 90%")

    # --- the sky-shuffle null
    add("FagnRecord", get(fs, "f.median"), "%.3f", src=S_FSCAN, kind="result",
        note="f_AGN with H0 held at truth, the record events")
    add("FagnRecordCi", bracket(get(fs, "f.ci68.0"), get(fs, "f.ci68.1")),
        src=S_FSCAN, kind="result", note="its 68% interval")
    add("FagnNull", get(fn, "f.median"), "%.3f", src=S_FNULL, kind="result",
        note="the same after the event sky positions are shuffled")
    add("FagnNullCi", bracket(get(fn, "f.ci68.0"), get(fn, "f.ci68.1")),
        src=S_FNULL, kind="result", note="its 68% interval")
    add("FagnNullCiNinety",
        bracket(get(fn, "f.ci90.0"), get(fn, "f.ci90.1")), src=S_FNULL,
        kind="result", note="its 90% interval")
    add("FagnNullWidthRatio",
        get(jsum, "sky_shuffle_null.width_ratio_null_over_record"), "%.2f",
        src=S_JSUM, kind="result",
        note="68% width of the shuffled-sky posterior over the record's")

    # f = 0 is excluded by the profile likelihood over the joint grid
    add("FagnZeroDlnL", hj.get("dlnl_joint"), "%.1f", src=hj.get("src", ""),
        kind="result",
        note="log-likelihood drop from the joint maximum to f_AGN = 0, "
             "maximised over H0 at each f")
    add("FagnZero", hj.get("sigma_joint"), "%.1f", src=hj.get("src", ""),
        kind="result",
        note="sqrt(2 x that drop): the significance with which f_AGN = 0 is "
             "excluded")
    add("FagnZeroNull", hj.get("sigma_null"), "%.1f", src=hj.get("src_null", ""),
        kind="result",
        note="the same statistic on the sky-shuffled events, at fixed H0")
    add("FagnZeroRecord", hj.get("sigma_record"), "%.1f",
        src=hj.get("src_record", ""), kind="result",
        note="the same statistic on the record events, at fixed H0 -- the "
             "matched comparison for the null")


def sec_controls(cl, curv, mumc):
    """Matched-host controls: each catalog handed only the events it hosts."""
    for tag, case in (("Gal", "gal"), ("Agn", "agn")):
        a = get(cl, f"cases.{case}.after")
        add(f"Ctrl{tag}Mean", pm(get(a, "mean_offset"), get(a, "sem_offset")),
            src=S_CLOSURE, kind="result",
            note=f"mean H0 offset from truth of the matched {case.upper()} "
                 f"control over five realisations, km/s/Mpc")
        add(f"Ctrl{tag}InSixtyEight", get(a, "n_truth_in_ci68"), "%d",
            src=S_CLOSURE, kind="result",
            note="realisations whose 68% interval contains truth")
        add(f"Ctrl{tag}InNinety", get(a, "n_truth_in_ci90"), "%d",
            src=S_CLOSURE, kind="result", note="the same at 90%")
        add(f"Ctrl{tag}Median",
            get(cl, f"cases.{case}.per_seed.0.after.median"), "%.1f",
            src=S_CLOSURE, kind="result",
            note="the control's median on the reference realisation, km/s/Mpc")
    add("CtrlNseeds", get(cl, "cases.gal.after.n_seeds"), "%d", src=S_CLOSURE,
        kind="result", note="realisations behind the control means")

    # what the selection estimator's own Monte-Carlo error costs, converted on
    # the curvature of the likelihood it is carried into
    sm = mumc.get("sigma_MC") or {}
    for tag, key, cur in (("Gal", "matched GAL, targeted", "ctrl_gal_matched"),
                          ("Agn", "matched AGN, targeted", "ctrl_agn_matched")):
        s = sm.get(key)
        d2 = get(curv, f"{cur}.d2_per_event")
        add(f"SelMc{tag}", None if None in (s, d2) or not d2 else s / abs(d2),
            "%.2f", src=f"{S_MUMC} + {S_CURV}", kind="result",
            note=f"selection Monte-Carlo error carried onto the matched "
                 f"{tag.upper()} control, km/s/Mpc per realisation")
    br = get(mumc, "bracket_targeted_lane.per_realisation") or []
    add("SelMcJointLo", br[0] if br else None, "%.2f", src=S_MUMC,
        kind="result",
        note="lower end of the same term on the joint fit (its f = 0 limit)")
    add("SelMcJointHi", br[1] if len(br) > 1 else None, "%.2f", src=S_MUMC,
        kind="result", note="upper end (its f = 1 limit)")


def sec_pure(pt):
    """Pure-tracer event sets: each catalog measured on its own 1000 events.

    Analysis 0.  For every realisation the generator drew two further event
    sets on the same catalogs -- one with every host a galaxy, one with every
    host an AGN, 1000 detected events each, noise streams independent of each
    other and of the mixture events of the main text.  Each set was analysed
    against its own catalog alone, so the two tracers' widths are a like-for-
    like comparison; the matched-host controls of analysis 1 are not, because
    they split one mixed event set into a 705/295 pair that shares its noise
    with the measurement of record.  Appendix material.

    The record lane is `targeted`; `popuni` is the cross-check whose largest
    disagreement is carried below.
    """
    gal = get(pt, "closure_gal") or {}
    agn = get(pt, "closure_agn") or {}

    add("PureNseeds", get(gal, "n_seeds"), "%d", src=S_PURE, kind="result",
        note="realisations behind the pure-tracer comparison")
    counts = {v for b in (gal, agn)
              for v in (get(b, "widths.n_events_per_seed") or {}).values()}
    add("PureNevents", counts.pop() if len(counts) == 1 else None, "intsep",
        src=S_PURE, kind="result",
        note="detected events in each pure-tracer set; the same for both "
             "tracers and every realisation, which is what makes the widths "
             "comparable")

    # ---- constraining power at matched N
    wg = get(gal, "widths.mean_half68")
    wa = get(agn, "widths.mean_half68")
    add("PureGalHalfWidth", wg, "%.2f", src=S_PURE, kind="result",
        note="mean 68% half-width of the H0 posterior from the galaxy catalog "
             "on 1000 galaxy-hosted events, km/s/Mpc")
    add("PureAgnHalfWidth", wa, "%.2f", src=S_PURE, kind="result",
        note="the same for the AGN catalog on 1000 AGN-hosted events")
    add("PureWidthRatio", None if None in (wg, wa) or not wa else wg / wa,
        "%.1f", src=S_PURE, kind="result",
        note="mean galaxy half-width over mean AGN half-width: how much "
             "tighter the sparse tracer is at equal event count")
    add("PureWidthRatioPerSeed",
        pm(get(pt, "constraining_power.mean_of_per_seed_ratios"),
           get(pt, "constraining_power.sem_of_per_seed_ratios"),
           "%.2f", "%.2f"),
        src=S_PURE, kind="result",
        note="the AGN/galaxy half-width ratio formed realisation by "
             "realisation, mean and standard error")

    # ---- recovery
    for tag, block in (("Gal", gal), ("Agn", agn)):
        add(f"Pure{tag}Offset",
            pm(get(block, "mean_offset"), get(block, "sem_offset"),
               "%+.3f", "%.3f"),
            src=S_PURE, kind="result",
            note=f"mean H0 offset from truth of the pure-{tag.upper()} sets "
                 f"over the realisations, km/s/Mpc")
        add(f"Pure{tag}InSixtyEight", get(block, "coverage.n_truth_in_ci68"),
            "%d", src=S_PURE, kind="result",
            note="realisations whose 68% interval contains truth")
        add(f"Pure{tag}InNinety", get(block, "coverage.n_truth_in_ci90"), "%d",
            src=S_PURE, kind="result", note="the same at 90%")

    # ---- the two injection lanes on the same events
    for tag, case in (("Gal", "gal"), ("Agn", "agn")):
        add(f"PureLaneMax{tag}",
            get(pt, f"lanes.{case}.max_abs_difference_over_half68"), "%.2f",
            src=S_PURE, kind="result",
            note=f"largest shift of the pure-{tag.upper()} median between the "
                 f"two injection lanes, in units of one 68% half-width")

    # ---- the one genuinely multimodal posterior, named rather than smoothed
    # Mode positions and heights are recorded per scan by the analysis's own
    # aggregator (`diagnostics.per_scan[].mode_{positions,relative_heights}`),
    # so nothing here is re-derived from the likelihood grids.  "Genuinely"
    # multimodal means a secondary mode above 1 per cent of the peak: two AGN
    # scans carry a second entry at ~1e-211 of the peak, which is numerical
    # dust, and the threshold removes them.
    lane = get(pt, "injection_lane_of_record")
    bimodal = [s for s in (get(pt, "diagnostics.per_scan") or [])
               if s.get("lane") == lane
               and sum(h >= 0.01 for h in s.get("mode_relative_heights") or []) > 1]
    one = bimodal[0] if len(bimodal) == 1 else {}
    heights = one.get("mode_relative_heights") or []
    modes = one.get("mode_positions") or []
    lo = min(range(len(modes)), key=lambda i: modes[i]) if modes else None
    add("PureBimodalSeed", one.get("seed"), "%d", src=S_PURE, kind="result",
        note="the one realisation whose galaxy-catalog posterior has two modes")
    add("PureBimodalModeLo", None if lo is None else min(modes), "%.2f",
        src=S_PURE, kind="result", note="its lower mode, km/s/Mpc")
    add("PureBimodalModeHi", None if lo is None else max(modes), "%.2f",
        src=S_PURE, kind="result", note="its upper mode, km/s/Mpc")
    add("PureBimodalHeight", None if lo is None else heights[lo], "%.2f",
        src=S_PURE, kind="result",
        note="height of the lower mode relative to the higher one")


def sec_incomplete():
    """Analysis 3.  Not landed; these are the only pending macros."""
    p = "(pending: analyses/analysis_3_incomplete_catalog_H0_fagn)"
    add("HzeroIncomplete", None, src=p, kind="result",
        note="H0 from the two-tracer fit on the shallowest catalogs")
    add("FagnIncomplete", None, src=p, kind="result",
        note="AGN-hosted fraction on the shallowest catalogs")
    add("FagnWidthRatio", None, src=p, kind="result",
        note="growth of the f_AGN interval across the magnitude limits")


# ---------------------------------------------------------------------------
# the one derivation that needs the likelihood grids themselves
# ---------------------------------------------------------------------------
def profile_exclusion() -> dict:
    """How far f_AGN = 0 sits below the maximum, in log-likelihood.

    The joint grid gives the profile over H0; the two f-scans at fixed H0 give
    the record-versus-shuffled comparison at matched conditions.
    """
    out = {"src": rel(A2 / "joint_s100.h5"),
           "src_record": rel(A2 / "fscan_s100.h5"),
           "src_null": rel(A2 / "fscan_null_s100.h5")}
    try:
        import h5py
        import numpy as np
    except ImportError:
        return out
    try:
        with h5py.File(A2 / "joint_s100.h5", "r") as h:
            prof = np.asarray(h["log_likelihood"][:]).max(axis=0)
        d = float(prof.max() - prof[0])
        out["dlnl_joint"] = d
        out["sigma_joint"] = math.sqrt(2.0 * d)
        for tag, name in (("record", "fscan_s100.h5"),
                          ("null", "fscan_null_s100.h5")):
            with h5py.File(A2 / name, "r") as h:
                ll = np.asarray(h["log_likelihood"][:])
            out[f"sigma_{tag}"] = math.sqrt(2.0 * float(ll.max() - ll[0]))
    except (OSError, KeyError, ValueError):
        pass
    return out


# ===========================================================================
def main():
    m = load_json(META)
    if not m:
        print(f"WARNING: {META} not found; dataset macros will be pending")

    jsum = load_json(A2 / "joint_summary.json")
    kde = load_json(A1 / "kde_window.json")

    sec_cosmology(m)
    sec_field(m)
    sec_luminosity(m)
    sec_events(m)
    sec_survey(m)
    sec_selection(m, jsum)
    sec_design(m, kde)

    sec_single()
    sec_joint(load_json(A2 / "h0_fagn_joint.json"), jsum,
              load_json(A2 / "fscan_s100.json"),
              load_json(A2 / "fscan_null_s100.json"),
              profile_exclusion())
    sec_controls(load_json(A1 / "closure_v3.json"),
                 load_json(A1 / "v3_curvature.json"),
                 load_json(A2 / "mu_mc_error.json"))
    sec_pure(load_json(A0 / "h0_pure_tracer.json"))
    sec_incomplete()

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    (PAPER / "values").mkdir(exist_ok=True)
    out = [
        "% results_macros.tex -- GENERATED, DO NOT EDIT BY HAND",
        f"% written {stamp} by scripts/build_values.py",
        f"% {len(REGISTRY)} macros; dataset of record v3 + D3, seed 100",
        "",
    ]
    for r in REGISTRY:
        out.append(rf"\newcommand{{\{r['name']}}}{{{r['body']}}}")
    (PAPER / "values" / "results_macros.tex").write_text("\n".join(out) + "\n")

    # ---- audit trail
    md = [
        "# NUMBERS.md -- macro audit trail",
        "",
        "GENERATED by `scripts/build_values.py`; do not edit by hand.",
        "",
        f"Written {stamp}. {len(REGISTRY)} macros.",
        "",
        "Every number that appears in the manuscript body or a caption is one of",
        "the macros below. Paths are relative to",
        f"`{WORKING}` unless absolute. Regenerate with",
        "",
        "```",
        "python scripts/build_values.py",
        "python scripts/audit_values.py     # re-derives and compares",
        "```",
        "",
        "## Sources of record",
        "",
        "| what | file |",
        "|---|---|",
        f"| dataset (v3 + D3, reference realisation) | `{META_SRC}` |",
        f"| measurement family, prose and citations | `{rel(DESIGN)}` |",
        f"| single-tracer baselines | `{S_SINGLE}` |",
        f"| matched-host controls, five seeds | `{S_CLOSURE}` |",
        f"| joint (H0, f_AGN) headline | `{S_JOINT}` |",
        f"| joint per-seed detail and closure | `{S_JSUM}` |",
        f"| pure-tracer event sets (appendix) | `{S_PURE}` |",
        f"| f-scan, record and sky-shuffled | `{S_FSCAN}`, `{S_FNULL}` |",
        f"| selection Monte-Carlo error | `{S_MUMC}`, `{S_CURV}` |",
        "",
        "## Rounding",
        "",
        "Interval strings quoted by a source JSON are copied verbatim. Otherwise",
        "H0-like locations carry 1 decimal and H0-like widths 2; f-like",
        "quantities carry 3 decimals; ratios and sigma-counts carry 1.",
        "",
        "## Caveats found in the sources",
        "",
        "Read before quoting anything not on this page.",
        "",
        "* `META.json`'s `config.injections.ndraw_targeted` and `ndraw_popuni`"
        " both read `1.2e8`, but the run drew `1.5e8` and `4.0e8`"
        " (`stages.injections.*.ndraw`, and `DESIGN_PE.md` §6 agrees). The"
        " realised numbers are the ones used here.",
        "* `config.events` still carries the v2 constants `sigma_dL = 0.10`,"
        " `sig_m1_frac = 0.08`, `sig_m2_frac = 0.10`, `sigma_chieff = 0.08`."
        " They are dead: nothing in the v3 generator reads them. The live"
        " constants are under `config.events.v3_measurement_family`.",
        "* The two analyses round to different precisions in their own summary"
        " files: the galaxy 68% H0 width is stored as `3.3`, the joint one as"
        " `1.94`. Both are rendered at two decimals here.",
        "* `\\JointRho` is the reference realisation's value (0.068); the"
        " five-realisation mean is `\\JointRhoMean` (+0.105 +- 0.035). They are"
        " different quantities and the draft should not mix them.",
        "* The sky-shuffle null's median is quoted with its 68% interval"
        " (`\\FagnNullCi`) and its 90% (`\\FagnNullCiNinety`); earlier internal"
        " notes quote the 90% next to a median, which reads as a 68%.",
        "* `mu_mc_error.json`'s `sigma_MC` block is transcribed from"
        " `analysis_1/CLOSURE.md` §16.6 rather than recomputed, so `\\SelMcGal`"
        " and `\\SelMcAgn` inherit that table's precision. The conversion to"
        " km/s/Mpc is redone here from `v3_curvature.json`.",
        "* The two pure-tracer offsets (`\\PureGalOffset`,"
        " `\\PureAgnOffset`) carry **three** decimals, not the two the"
        " H0 convention asks for: the AGN mean offset is -0.001 km/s/Mpc and"
        " rounds to `-0.00` at two, which reads as a typesetting accident and"
        " throws away the fact that the pure-AGN sets land on truth. The two"
        " numbers are quoted at the same precision as each other and as"
        " `analysis_0_pure_tracer_H0/README.md`.",
        "* `\\PureWidthRatio` (5.1) is the ratio of the two *mean* half-widths;"
        " `\\PureWidthRatioPerSeed` (0.22 +- 0.03) is the mean of the"
        " per-realisation AGN/galaxy ratios, and is the reciprocal-of-a-mean"
        " rather than the mean-of-reciprocals, so 1/0.22 != 5.1. They are"
        " different statistics; do not present one as the other's inverse.",
        "* `diagnostics.per_scan[].n_interior_modes` reads 2 for"
        " `h0_pureagn_targeted_s101` and `h0_pureagn_popuni_s101` as well, but"
        " those second modes sit at ~1e-211 of the peak. `\\PureBimodal*` is"
        " built with a 1 per cent relative-height threshold, which leaves"
        " exactly one genuinely bimodal scan, `h0_puregal_targeted_s105`.",
        "* Analysis 1's `h0_single_tracer.json` stores `agn_h0_ci` and"
        " `agn_h0_width` as `null` on purpose: the AGN-only posterior rails."
        " `agn_grid_top_median` is the number to quote, against the top of the"
        " scanned range.",
        "",
    ]
    for kind, title in (("configuration", "Configuration"),
                        ("dataset", "Realised dataset properties"),
                        ("result", "Results")):
        rows = [r for r in REGISTRY if r["kind"] == kind]
        md += [f"## {title}", ""]
        if not rows:
            md += ["None yet.", ""]
            continue
        md += ["| macro | value | source | meaning |", "|---|---|---|---|"]
        for r in sorted(rows, key=lambda x: x["name"]):
            body = r["body"].replace("|", r"\|")
            md.append(f"| `\\{r['name']}` | `{body}` | `{r['src']}` | {r['note']} |")
        md.append("")
    pend = [r["name"] for r in REGISTRY if r["raw"] is None]
    md += ["## Pending macros", ""]
    if pend:
        md += [f"{len(pend)} macros still resolve to `\\todo{{pending}}`:", ""]
        md += [f"- `\\{n}`" for n in pend]
    else:
        md.append("None: every macro resolves to a number.")
    md.append("")
    md += RENAME_MAP
    md += usage_report()
    (PAPER / "NUMBERS.md").write_text("\n".join(md))

    print(f"{len(REGISTRY)} macros -> values/results_macros.tex")
    print(f"{len(pend)} pending: {', '.join(pend) if pend else 'none'}")


def usage_report() -> list[str]:
    """Which macros the manuscript uses, which it does not, and what is missing.

    This is the working list for the prose rewrite: `defined but unused` is what
    the new sections have to spend, `used but undefined` is what they must stop
    saying.
    """
    import re as _re
    files = sorted((PAPER / "sections").glob("*.tex")) + [PAPER / "main.tex"]
    defined = {r["name"] for r in REGISTRY}
    own = set(_re.findall(r"\\newcommand\{\\([A-Za-z]+)\}",
                          (PAPER / "main.tex").read_text()))
    used: dict[str, set[str]] = {}
    unknown: dict[str, set[str]] = {}
    builtin = {"Lambda", "Omega", "Theta", "Delta", "Gamma", "Phi", "Psi",
               "Sigma", "Pi", "Upsilon", "Xi", "S", "P", "LaTeX", "TeX", "Big",
               "Bigg", "Left", "Right", "Re", "Im", "Pr", "Vert"}
    for f in files:
        body = "\n".join(_re.sub(r"(?<!\\)%.*$", "", ln)
                         for ln in f.read_text().splitlines())
        for name in _re.findall(r"\\([A-Z][A-Za-z]*)", body):
            if name in defined:
                used.setdefault(name, set()).add(f.name)
            elif name not in own and name not in builtin:
                unknown.setdefault(name, set()).add(f.name)

    out = ["## Manuscript usage", "",
           f"{len(used)} of {len(defined)} macros are used by the current draft.",
           ""]
    out += ["### Used but undefined -- the prose rewrite must remove these", ""]
    if unknown:
        out += ["| macro | appears in |", "|---|---|"]
        for name in sorted(unknown):
            out.append(f"| `\\{name}` | {', '.join(sorted(unknown[name]))} |")
    else:
        out.append("None.")
    out += ["",
            "### Defined but not yet used -- available to the rewrite", ""]
    spare = sorted(defined - set(used))
    if spare:
        out += ["| macro | value | meaning |", "|---|---|---|"]
        by_name = {r["name"]: r for r in REGISTRY}
        for name in spare:
            r = by_name[name]
            out.append(f"| `\\{name}` | `{r['body']}` | {r['note']} |")
    else:
        out.append("None.")
    out.append("")
    return out


# ---------------------------------------------------------------------------
# The v2 -> v3 rename map.  Kept here so that it is regenerated with the file
# it describes.  Every macro that the previous generation of this script
# defined and this one does not appears below, with what replaced it.
# ---------------------------------------------------------------------------
RENAME_MAP = [
    "## Rename map: what the v3 rebuild removed",
    "",
    "The dataset moved from the v2 measurement family (independent Gaussian",
    "component masses of fixed fractional width, an independent lognormal",
    "distance of fixed log-width, a sky width recomputed from the observed",
    "masses and distance) to v3, in which one recorded amplitude",
    "`rho_obs = rho_opt + N(0, sigma_rho)` carries the detection decision and",
    "sets every width as `A_x (rho_th/rho_obs)`, and the distance is derived",
    "from `(Mc_det, rho)` rather than measured. The macros below named v2-only",
    "concepts. **The section files that use them must be rewritten** -- this",
    "script deliberately does not define them, so the compile will flag every",
    "site.",
    "",
    "| old macro | v2 meaning | disposition | used in |",
    "|---|---|---|---|",
    "| `\\EvSigmaDl` | flat fractional distance width, 0.10 | REMOVED: v3 has no"
    " distance channel; the distance is derived from `(Mc_det, rho)` and its"
    " realised log-width is `\\EvSigmaLnDlCoef`/rho, i.e. `\\EvSigmaLnDlMedian`"
    " at the detected median | `sections/data.tex` |",
    "| `\\EvSigmaMone` | fractional width on the observed primary mass, 0.08 |"
    " REMOVED: v3 measures `ln Mc` and `ln q`, not the component masses ->"
    " `\\EvAmc`, `\\EvAq` | `sections/data.tex` |",
    "| `\\EvSigmaMtwo` | fractional width on the observed secondary mass, 0.10 |"
    " REMOVED: same | `sections/data.tex` |",
    "| `\\EvSigmaChi` | absolute effective-spin width, 0.08 | RENAMED ->"
    " `\\EvAchi` (0.20), now a coefficient of `rho_th/rho_obs` rather than a"
    " constant | `sections/data.tex` |",
    "| `\\EvSkyCoef`, `\\EvSkyMin`, `\\EvSkyMax` | sky-width law | KEPT, same"
    " values, but the law is now written on the recorded amplitude:"
    " `sigma_ang = clip(\\EvSkyCoefEff/rho_obs, ...)`, with"
    " `\\EvSkyRhoScale` the conversion | `sections/data.tex` |",
    "| `\\HzeroAgn` | AGN-only H0, median and 68% interval | REMOVED: on a mixed"
    " universe the AGN-only posterior rails at the top of the scanned range and"
    " has no interval. Quote `\\HzeroAgnRailMedian` against `\\HzeroScanHi`"
    " (and `\\HzeroAgnRailCross` for the second injection lane) and say that it"
    " rails | `main.tex`, `sections/results.tex` |",
    "| `\\HzeroAgnWidth` | 68% width of that posterior | REMOVED: same |"
    " `sections/results.tex` |",
    "| `\\EvSnrRefSky` | amplitude scale of the sky-width model | KEPT; v3 forms"
    " `rho_sigma` from `rho_obs` rather than recomputing it | `sections/data.tex`"
    " |",
    "",
    "Renamed or newly split, for completeness:",
    "",
    "| old | new |",
    "|---|---|",
    "| `\\InjNdrawMain`, `\\InjNdrawCross` (config values) | same names, now read"
    " from the realised `stages.injections.*.ndraw` -- the v3 `config` block"
    " still carries a stale `1.2e8` for both lanes |",
    "| `\\NeffFactor` (hand constant) | same name, now read off the run's own"
    " guard threshold |",
    "| `\\KdePeRatio` (used the flat v2 `sigma_dL`) | same name, now uses the"
    " realised median `sigma_ln dL` of v3 |",
    "",
    "New macros with no v2 ancestor: the v3 family constants (`\\EvSigmaRho`,",
    "`\\EvAmc`, `\\EvAq`, `\\EvAchi`, `\\EvWidthRefSnr`, `\\EvSkyRhoScale`,",
    "`\\EvSkyCoefEff`, `\\EvSigmaLnDlCoef`, `\\EvSigmaLnDlThresh`), the realised",
    "widths (`\\EvSnrMedian`, `\\EvSnrMax`, `\\EvSigLnMcMedian`,",
    "`\\EvSigLnQMedian`, `\\EvSigChiMedian`, `\\EvSigmaLnDlMedian`,",
    "`\\EvSkyMinReal`, `\\EvSkyMaxReal`), the realised photo-z",
    "(`\\CatPhotozPullSd`, `\\CatPhotozPullSdAgn`), the AGN railing statement,",
    "the whole joint block (`\\HzeroJoint*`, `\\Fagn*`), the closure block",
    "(`\\Closure*`, `\\Ctrl*`), the null (`\\FagnNull*`, `\\FagnZero*`), the",
    "selection Monte-Carlo terms (`\\SelMc*`) and `\\HzeroWidthRatio`.",
    "",
]

if __name__ == "__main__":
    main()
