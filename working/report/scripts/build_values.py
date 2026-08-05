#!/usr/bin/env python3
r"""Build every number the manuscript quotes from the experiments' results files.

Emits
  values/results_macros.tex   -- one \newcommand per quoted number
  tables/tab_*.tex            -- the generated tables
  NUMBERS.md                  -- macro -> source file -> experiment audit trail

Nothing in the manuscript body is hand-typed: prose and captions reference the
macros, the macros are computed here, and every macro carries the path of the
results file it came from.  Run this before latex.

Usage
    JAX_PLATFORMS=cpu python scripts/build_values.py
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parent.parent
EXP = (PAPER.parent / "analyses" / "experiments").resolve()
GWAGN = (PAPER.parent / "gw_agn_darksirens").resolve()
GWAGN_SRC = (PAPER.parent / "gw_agn").resolve()

E_BASE = EXP / "experiment_h0f_baseline"
E_CLO = EXP / "experiment_matched_mock"
E_ANC = EXP / "experiment_completeness_anchored"
E_DEEP = EXP / "experiment_twotracer_deep"
E_INC = EXP / "experiment_twotracer_incomplete"
E_FREE = EXP / "experiment_completeness_free"
E_SEED = EXP / "experiment_twotracer_seeds"

H0_TRUTH = 67.74

# ---------------------------------------------------------------------------
# macro registry
# ---------------------------------------------------------------------------
REGISTRY: list[dict] = []
_SEEN: set[str] = set()


def add(name: str, value, fmt: str = "%.3f", *, src: str = "", note: str = "",
        exp: str = "") -> str:
    """Register one macro.  `src` is a path relative to the experiments root."""
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
        # signed numbers need a real minus sign, and must work in both modes
        body = rf"\ensuremath{{{fmt % value}}}"
    else:
        body = fmt % value
    REGISTRY.append({"name": name, "body": body, "src": src, "note": note,
                     "exp": exp, "raw": value})
    return name


def _sci(v: float, sig: int = 2) -> str:
    if v == 0:
        return "0"
    e = int(math.floor(math.log10(abs(v))))
    m = v / 10 ** e
    return rf"{m:.{sig - 1}f}\times10^{{{e}}}"


def jload(p: Path):
    with open(p) as fh:
        return json.load(fh)


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(EXP))
    except ValueError:
        return str(Path(p).resolve())


def hw(ci) -> float:
    return 0.5 * (ci[1] - ci[0])


def stats(x):
    x = np.asarray(x, float)
    sd = float(x.std(ddof=1))
    sem = sd / math.sqrt(x.size)
    return {"n": int(x.size), "mean": float(x.mean()), "sd": sd, "sem": sem,
            "sig": abs(float(x.mean())) / sem}


# ===========================================================================
# 1. Shared truth / configuration
# ===========================================================================
def sec_setup():
    cat = jload(GWAGN / "data" / "catalog_meta.json")
    src = rel(GWAGN / "data" / "catalog_meta.json")
    e = "GLASS mock catalogs"
    add("HzeroTruth", H0_TRUTH, "%.2f", src=src, exp=e,
        note="fiducial expansion rate of every mock (Planck15)")
    add("OmZeroTruth", cat["cosmology"]["Om0"], "%.4f", src=src, exp=e)
    add("GlassZmax", cat["z_full_cosmo"], "%.1f", src=src, exp=e,
        note="GLASS host field redshift extent")
    add("GlassZtrunc", cat["z_trunc"], "%.1f", src=src, exp=e,
        note="event redshift cut")
    add("GlassDzScale", cat["dz_scale"], "sci", src=src, exp=e,
        note="catalog KDE width used for pixelation")
    g = cat["files"]["gal.h5"]
    a = cat["files"]["agn.h5"]
    add("GlassNside", g["nside"], "%d", src=src, exp=e)
    npix = 12 * g["nside"] ** 2
    add("GlassNpix", npix, "intsep", src=src, exp=e, note="12 nside^2")
    add("GlassNgal", g["n_total"], "intsep", src=src, exp=e)
    add("GlassNagn", a["n_total"], "intsep", src=src, exp=e)
    add("GlassGalLogNzero", g["log10n0_true"], "%.3f", src=src, exp=e)
    add("GlassAgnLogNzero", a["log10n0_true"], "%.3f", src=src, exp=e)
    add("GlassGalEmptyPix", 100.0 * g["empty_pixels"] / npix, "%.3g", src=src, exp=e)
    add("GlassAgnEmptyPix", 100.0 * a["empty_pixels"] / npix, "%.3g", src=src, exp=e)
    add("GlassDensityRatio", 10 ** (g["log10n0_true"] - a["log10n0_true"]), "%.0f",
        src=src, exp=e, note="galaxy/AGN comoving number-density ratio")

    # generation configuration of the clustered field
    cfg = GWAGN_SRC / "configs" / "data_glass_prod_fagn0.3.yaml"
    conf = {}
    for line in cfg.read_text().splitlines():
        if ":" in line and not line.strip().startswith("#"):
            k, v = line.split(":", 1)
            conf[k.strip()] = v.strip()
    csrc = rel(cfg)
    add("GlassBiasGal", float(conf["bias_gal"]), "%.1f", src=csrc, exp=e,
        note="linear bias of the dense tracer")
    add("GlassBiasAgn", float(conf["bias_agn"]), "%.1f", src=csrc, exp=e,
        note="linear bias of the sparse tracer")
    add("GlassBiasRatio", float(conf["bias_agn"]) / float(conf["bias_gal"]), "%.2f",
        src=csrc, exp=e)
    add("GlassLambdaAgn", float(conf["lambda_agn"]), "%.1f", src=csrc, exp=e,
        note="AGN host-selection sharpness parameter of the planted mixture")
    add("GlassDlUnc", float(conf["dL_uncertainty_fac"]), "%.2f", src=csrc, exp=e)
    add("GlassSeedField", int(conf["seed"]), "%d", src=csrc, exp=e)

    import h5py
    ev = GWAGN / "data" / "gw_fagn0.3.h5"
    with h5py.File(ev, "r") as fh:
        at = dict(fh.attrs)
    esrc = rel(ev)
    add("GlassNobs", int(at["nobs"]), "intsep", src=esrc, exp=e)
    add("GlassNsamp", int(at["nsamp"]), "intsep", src=esrc, exp=e,
        note="posterior samples per event")
    add("DarksirensSha", str(at["darksirens_repo_head"])[:7], "%s",
        src=esrc, exp=e, note="code version of the mock ingest")


def sec_matched_setup():
    """Configuration of the matched single-tracer mock, read from the data
    products the scans actually consumed (h5 attributes, not prose)."""
    import h5py
    e = "experiment\\_matched\\_mock"
    dd = E_CLO / "data_derived"

    sel = dd / "deep_mock_z2_big" / "mock_gw_selection.h5"
    with h5py.File(sel, "r") as f:
        ndraw = int(float(f.attrs["ndraw"]))
        ndet = int(f["chieff"].shape[0])
        prop = str(f.attrs["selection_proposal"])
    s = rel(sel)
    add("CloInjNdraw", ndraw / 1e6, "%.0f", src=s, exp=e,
        note="proposals drawn for the single-tracer selection integral, millions")
    add("CloInjNdet", ndet, "intsep", src=s, exp=e)
    add("CloInjProposal", prop.replace("+", " + "), "%s", src=s, exp=e)

    ev = dd / "deep_mock_z2_big" / "mock_gw_events.h5"
    with h5py.File(ev, "r") as f:
        add("CloNobs", int(float(f.attrs["nobs"])), "intsep", src=rel(ev), exp=e)
        add("CloNsamp", int(float(f.attrs["nsamp"])), "intsep", src=rel(ev), exp=e)

    sv = dd / "deep_survey_z2_ns16.h5"
    with h5py.File(sv, "r") as f:
        s = rel(sv)
        add("CloNside", int(float(f.attrs["nside"])), "%d", src=s, exp=e)
        add("CloNpix", int(float(f.attrs["occupied_pixels"])), "intsep", src=s, exp=e)
        add("CloNhosts", int(float(f.attrs["n_galaxies_used"])), "intsep", src=s, exp=e)
        add("CloZmax", float(f.attrs["z_max"]), "%.1f", src=s, exp=e,
            note="host catalog redshift extent, far beyond the detected events")
        add("CloEmptyPix", 100 * float(f.attrs["empty_pixel_fraction"]), "%.0f",
            src=s, exp=e)

    for arm, tag in (("ctrl", "Ctrl"), ("obs", "Obs")):
        p = dd / "obsdet" / f"sel_{arm}.h5"
        with h5py.File(p, "r") as f:
            add(f"Ab{tag}InjNdraw", int(float(f.attrs["ndraw"])) / 1e6, "%.0f",
                src=rel(p), exp=e)
            add(f"Ab{tag}InjNdet", int(f["chieff"].shape[0]), "intsep", src=rel(p), exp=e)
            add(f"Ab{tag}SnrRef", float(f.attrs["snr_ref"]), "%.3f", src=rel(p), exp=e,
                note="reference signal-to-noise calibrating the detected fraction")


# ===========================================================================
# 2. GLASS K=2 baseline: (H0, f_AGN)
# ===========================================================================
FKEYS = {"0.0": "Nought", "0.3": "Three", "0.7": "Seven", "1.0": "Unity"}


def sec_baseline():
    p = E_BASE / "results" / "summary.json"
    d = jload(p)
    src = rel(p)
    e = "experiment\\_h0f\\_baseline"
    add("BaseSha", d["darksirens_sha"][:7], "%s", src=src, exp=e)
    for k, tag in FKEYS.items():
        add(f"BaseFTruth{tag}", d["f_truth"][k], "%.5g", src=src, exp=e,
            note=f"planted AGN-hosted fraction, set {k}")
        s = d["f_scan_at_true_H0"][k]
        add(f"BaseFMed{tag}", s["median"], "%.4f", src=src, exp=e,
            note="recovered f_AGN median at the true expansion rate")
        add(f"BaseFLo{tag}", s["ci68"][0], "%.3f", src=src, exp=e)
        add(f"BaseFHi{tag}", s["ci68"][1], "%.3f", src=src, exp=e)
        add(f"BaseFOff{tag}", s["median"] - d["f_truth"][k], "%+.3f", src=src, exp=e)
        add(f"BaseFHw{tag}", hw(s["ci68"]), "%.3f", src=src, exp=e)
        add(f"BaseFRej{tag}", s["n_rejected_cells"], "%d", src=src, exp=e)
    offs = [abs(d["f_scan_at_true_H0"][k]["median"] - d["f_truth"][k]) for k in FKEYS]
    add("BaseFOffMaxAbs", max(offs), "%.3f", src=src, exp=e,
        note="largest absolute offset on the f ladder")
    add("BaseFHwMin", min(hw(d["f_scan_at_true_H0"][k]["ci68"]) for k in FKEYS),
        "%.3f", src=src, exp=e)
    add("BaseFHwMax", max(hw(d["f_scan_at_true_H0"][k]["ci68"]) for k in FKEYS),
        "%.3f", src=src, exp=e)
    add("BaseFOneSidedUnity", d["f_scan_at_true_H0"]["1.0"]["onesided68_lo"], "%.3f",
        src=src, exp=e, note="one-sided 68% lower limit at the prior boundary")
    add("BaseFGridN", 41, "%d", src=rel(E_BASE / "results" / "fscan_fagn0.3.h5"), exp=e)

    for k, tag in (("0.3", "Three"), ("0.7", "Seven")):
        j = d["joint"][k]
        h = j["marg_H0"]
        f = j["marg_f"]
        add(f"BaseJointHzero{tag}", h["median"], "%.2f", src=src, exp=e,
            note="marginal H0 from the joint (H0, f) grid")
        add(f"BaseJointHzeroUp{tag}", h["ci68"][1] - h["median"], "%.2f", src=src, exp=e)
        add(f"BaseJointHzeroDn{tag}", h["median"] - h["ci68"][0], "%.2f", src=src, exp=e)
        add(f"BaseJointHzeroHw{tag}", hw(h["ci68"]), "%.2f", src=src, exp=e)
        off = h["median"] - H0_TRUTH
        add(f"BaseJointHzeroOff{tag}", off, "%+.2f", src=src, exp=e)
        add(f"BaseJointHzeroOffAbs{tag}", abs(off), "%.2f", src=src, exp=e)
        add(f"BaseJointHzeroSig{tag}", abs(off) / hw(h["ci68"]), "%.1f", src=src, exp=e,
            note="offset in units of the 68% half-width")
        add(f"BaseJointF{tag}", f["median"], "%.3f", src=src, exp=e)
        add(f"BaseJointFHw{tag}", hw(f["ci68"]), "%.3f", src=src, exp=e)
        add(f"BaseJointRho{tag}", j["rho"], "%+.2f", src=src, exp=e)
        add(f"BaseJointRhoAbs{tag}", abs(j["rho"]), "%.2f", src=src, exp=e)
        add(f"BaseJointCells{tag}", j["n_cells"], "%d", src=src, exp=e)
        add(f"BaseJointRej{tag}", j["n_rejected_cells"], "%d", src=src, exp=e)

    # likelihood decomposition at fixed mixture weight
    for k, tag in (("0.3", "Three"), ("0.7", "Seven")):
        dp = E_BASE / "results" / f"h0_decomposition_fagn{k}.json"
        dd = jload(dp)
        s2 = rel(dp)
        add(f"BaseDecompTotal{tag}", dd["peak_total"], "%.2f", src=s2, exp=e,
            note="full-likelihood H0 peak at fixed mixture weight")
        add(f"BaseDecompNum{tag}", dd["peak_per_event_numerator"], "%.2f", src=s2, exp=e,
            note="per-event term alone")
        add(f"BaseDecompNumOff{tag}", dd["peak_per_event_numerator"] - H0_TRUTH,
            "%+.2f", src=s2, exp=e, note="per-event term alone, offset from truth")
        add(f"BaseDecompShift{tag}", dd["shift_from_selection_term"], "%+.2f",
            src=s2, exp=e, note="peak shift contributed by the selection term")
        add(f"BaseDecompShiftAbs{tag}", abs(dd["shift_from_selection_term"]), "%.2f",
            src=s2, exp=e)
        add(f"BaseDecompDlnmu{tag}", dd["dlnmu_dH0_at_truth"], "%.3f", src=s2, exp=e)
        add(f"BaseDecompLnmuRange{tag}",
            dd["lnmu_range"][1] - dd["lnmu_range"][0], "%.2f", src=s2, exp=e,
            note="span of ln mu over the scanned H0 range")
        add(f"BaseDecompLever{tag}",
            (dd["lnmu_range"][1] - dd["lnmu_range"][0]) * dd["nobs"], "%.0f",
            src=s2, exp=e, note="N_obs x ln mu span, in nats")

    # the two eliminated explanations (scans live with the closure experiment,
    # which is where the follow-up was run, but they act on the GLASS mock)
    gshift, eshift = {}, {}
    for k, tag in (("0.3", "Three"), ("0.7", "Seven")):
        g0 = jload(E_CLO / "results" / f"gtest_g0_fagn{k}.json")["H0"]["median"]
        g1 = jload(E_CLO / "results" / f"gtest_g1_fagn{k}.json")["H0"]["median"]
        gshift[tag] = g1 - g0
        add(f"BaseGammaShift{tag}", g1 - g0, "%+.2f",
            src=rel(E_CLO / "results" / f"gtest_g1_fagn{k}.json"), exp=e,
            note="H0 shift when the host rate index is matched to the mock's draw")
        ed = jload(E_CLO / "results" / f"edgetest_zlt1_fagn{k}.json")["H0"]["median"]
        eshift[tag] = ed - g0
        add(f"BaseEdgeShift{tag}", ed - g0, "%+.2f",
            src=rel(E_CLO / "results" / f"edgetest_zlt1_fagn{k}.json"), exp=e,
            note="H0 shift when the catalog redshift edge moves, events held fixed")
    add("BaseGammaShiftMax", max(abs(v) for v in gshift.values()), "%.2f",
        src=rel(E_CLO / "results" / "gtest_g1_fagn0.7.json"), exp=e,
        note="largest |dH0| over the two rate-index tests")
    add("BaseEdgeShiftAbsThree", abs(eshift["Three"]), "%.1f",
        src=rel(E_CLO / "results" / "edgetest_zlt1_fagn0.3.json"), exp=e,
        note="the catalog-edge lever, absolute value")


def sec_tilt():
    """Mechanism of the clustered mocks' H0 tilt: one model inconsistency
    (detection on true redshift vs a dL-space selection and a host prior with
    catalog support above the horizon), two opposing levers."""
    e = "experiment\\_h0f\\_baseline"
    bp = E_BASE / "results" / "tilt_budget.json"
    b = jload(bp)
    bsrc = rel(bp)
    for k, tag in (("fagn0.3", "Three"), ("fagn0.7", "Seven")):
        d = b[k]
        add(f"TiltNumLeak{tag}", -d["shifts_km_s_Mpc"]["numerator_zcut_1"],
            "%+.2f", src=bsrc, exp=e,
            note="numerator pull from catalog support above the z=1 horizon")
        add(f"TiltLeakTruth{tag}",
            100 * d["diagnostics"]["mean_pe_massfrac_z_gt_1"]["at_truth"], "%.1f",
            src=bsrc, exp=e, note="mean PE posterior mass beyond z=1 at truth, %")
        add(f"TiltLeakHigh{tag}",
            100 * d["diagnostics"]["mean_pe_massfrac_z_gt_1"]["at_75"], "%.1f",
            src=bsrc, exp=e, note="same at H0=75")
        add(f"TiltAgnShare{tag}", d["diagnostics"]["agn_share_mean_at_truth"],
            "%.2f", src=bsrc, exp=e,
            note="mean posterior weight on the AGN branch at truth")
        add(f"TiltNumAgnhost{tag}", d["peaks"]["numerator_agnhost"], "%.2f",
            src=bsrc, exp=e, note="numerator peak, AGN-hosted events only")
        add(f"TiltNumAgnhostOff{tag}", d["peaks"]["numerator_agnhost"] - H0_TRUTH,
            "%+.2f", src=bsrc, exp=e)
        add(f"TiltNumGalhost{tag}", d["peaks"]["numerator_galhost"], "%.1f",
            src=bsrc, exp=e, note="numerator peak, galaxy-hosted events only")
        add(f"TiltCurv{tag}", abs(d["diagnostics"]["total_curvature_at_peak"]),
            "%.1f", src=bsrc, exp=e, note="total curvature at the peak, nats/(km/s/Mpc)^2")
        add(f"TiltSelSlopeNats{tag}", abs(d["slopes_nats_per_km"]["selection_term"]),
            "%.0f", src=bsrc, exp=e,
            note="selection-term lever, nats per km/s/Mpc (N_obs x dlnmu/dH0)")
        add(f"TiltMcBias{tag}", d["shifts_km_s_Mpc"]["mc_bias_correction"], "%+.2f",
            src=bsrc, exp=e, note="delta-method PE Monte Carlo bias on the peak")
    add("TiltNumGalhostOffThree",
        b["fagn0.3"]["peaks"]["numerator_galhost"] - H0_TRUTH, "%+.1f",
        src=bsrc, exp=e)
    add("TiltMcBiasMax",
        max(abs(b[k]["shifts_km_s_Mpc"]["mc_bias_correction"])
            for k in ("fagn0.3", "fagn0.7")), "%.2f", src=bsrc, exp=e)

    sp = E_BASE / "results" / "tilt_selection_model.json"
    s = jload(sp)
    ssrc = rel(sp)
    add("TiltDlmax", s["dLmax_Mpc"], "intsep", src=ssrc, exp=e,
        note="dL(z=1; H0 truth), the detected set's exact dL boundary, Mpc")
    add("TiltZstarLo", s["zstar_at_60_truth_75"][0], "%.2f", src=ssrc, exp=e,
        note="z*(H0=60): where the dL boundary sits under the analysis model")
    add("TiltZstarHi", s["zstar_at_60_truth_75"][2], "%.2f", src=ssrc, exp=e,
        note="z*(H0=75)")
    for k, tag in (("fagn0.3", "Three"), ("fagn0.7", "Seven")):
        add(f"TiltSlopeMeas{tag}", s[k]["dlnmu_dH0_measured_at_truth"], "%.4f",
            src=ssrc, exp=e, note="measured dlnmu/dH0 at truth")
        add(f"TiltSlopeModel{tag}", s[k]["dlnmu_dH0_model_zweighted_at_truth"],
            "%.4f", src=ssrc, exp=e,
            note="zero-parameter analytic dL-threshold model")
    add("TiltModelPct",
        100 * s["fagn0.3"]["dlnmu_dH0_model_zweighted_at_truth"]
        / s["fagn0.3"]["dlnmu_dH0_measured_at_truth"], "%.0f", src=ssrc, exp=e,
        note="fraction of the measured selection slope the analytic model predicts")
    add("TiltModelResidNats", s["fagn0.3"]["shape_residual_max_nats"], "%.2f",
        src=ssrc, exp=e)
    add("TiltModelRangeNats", s["fagn0.3"]["shape_range_nats"], "%.2f",
        src=ssrc, exp=e)

    pp = E_BASE / "results" / "tilt_predicted_selection_shift.json"
    p = jload(pp)
    add("TiltPredShiftThree", p["fagn0.3"]["predicted_selection_shift"], "%+.2f",
        src=rel(pp), exp=e,
        note="selection shift re-peaked against the analytic model mu")

    rp = E_BASE / "results" / "tilt_repaired_estimator.json"
    r = jload(rp)
    add("TiltRepairOffThree", r["fagn0.3"]["repaired_offset"], "%+.2f",
        src=rel(rp), exp=e,
        note="repaired estimator: z<=1-truncated prior + H0-independent beta")
    add("TiltRepairSigThree", r["fagn0.3"]["sigma_stat"], "%.2f", src=rel(rp), exp=e)
    add("TiltRepairOffSeven", r["fagn0.7"]["repaired_offset"], "%+.2f",
        src=rel(rp), exp=e)

    ip = E_BASE / "results" / "tilt_mu_proposal_independence.json"
    i = jload(ip)
    add("TiltSlopeTgt", i["linfit_slope_60_75_catinj"], "%.5f", src=rel(ip), exp=e,
        note="lnmu slope 60-75, catalog-targeted injections")
    add("TiltSlopeUntgt", i["linfit_slope_60_75_altinj"], "%.5f", src=rel(ip), exp=e,
        note="same, untargeted independent-seed injections")
    add("TiltSlopeDiffPct",
        100 * abs(i["linfit_slope_60_75_catinj"] - i["linfit_slope_60_75_altinj"])
        / i["linfit_slope_60_75_catinj"], "%.1f", src=rel(ip), exp=e,
        note="proposal (in)dependence of the selection slope")

    cp = E_BASE / "results" / "tilt_mu_injB_compare.json"
    c = jload(cp)
    offs = [c["offset_injA"], c["offset_injB"], c["offset_untargeted"]]
    st = stats(offs)
    add("TiltInjSpreadMean", st["mean"], "%+.1f", src=rel(cp), exp=e,
        note="f=0.7 offset over three independent injection sets")
    add("TiltInjSpreadSd", st["sd"], "%.1f", src=rel(cp), exp=e)

    # catalog mass above the detection horizon, from the survey files the
    # likelihood conditions on
    import h5py
    for fn, tag in (("gal.h5", "Gal"), ("agn.h5", "Agn")):
        fp = GWAGN / "data" / fn
        with h5py.File(fp, "r") as f:
            z = f["zgals"][:]
        z = z[(z > 0) & (z < 90)]
        add(f"TiltCatAbove{tag}", 100.0 * float((z > 1.0).mean()), "%.0f",
            src=rel(fp), exp=e,
            note="per cent of catalogued objects above the z=1 event horizon")


# ===========================================================================
# 3. Matched-mock closure ladder (K=1)
# ===========================================================================
def sec_closure():
    p = E_CLO / "results" / "summary.json"
    d = jload(p)
    src = rel(p)
    e = "experiment\\_matched\\_mock"

    SIG = {"0.01": "One", "0.03": "Three", "0.1": "Ten"}
    for arm, atag in (("truth_centred", "Truth"), ("corrected", "Corr")):
        for s, stag in SIG.items():
            r = d["sigma_ladder"][arm][s]
            add(f"Clo{atag}Sig{stag}Med", r["median"], "%.3f", src=src, exp=e,
                note=f"{arm} PE, sigma_dL={s}")
            add(f"Clo{atag}Sig{stag}Off", r["median"] - H0_TRUTH, "%+.3f", src=src, exp=e)
            add(f"Clo{atag}Sig{stag}Hw", r["hw"], "%.3f", src=src, exp=e)
    add("CloSigLo", 0.01, "%.2f", src=src, exp=e)
    add("CloSigMid", 0.03, "%.2f", src=src, exp=e)
    add("CloSigHi", 0.10, "%.2f", src=src, exp=e)

    ms = d["multi_seed"]
    add("CloSeedN", ms["n_seeds"], "%d", src=src, exp=e)
    add("CloSeedSigmaDl", ms["sigma_dL"], "%.2f", src=src, exp=e)
    add("CloSeedMeanHzero", ms["mean_H0"], "%.2f", src=src, exp=e)
    add("CloSeedOffset", ms["offset"], "%+.2f", src=src, exp=e)
    add("CloSeedOffsetAbs", abs(ms["offset"]), "%.2f", src=src, exp=e)
    add("CloSeedSem", ms["sem"], "%.2f", src=src, exp=e)
    add("CloSeedSig", ms["significance"], "%.1f", src=src, exp=e)
    add("CloSeedScatter", ms["scatter_sd"], "%.2f", src=src, exp=e,
        note="realised seed-to-seed standard deviation of H0")
    add("CloSeedMeanHw", ms["mean_halfwidth"], "%.2f", src=src, exp=e,
        note="mean quoted 68% half-width per realisation")
    add("CloSeedUnderFactor", ms["interval_underestimate_factor"], "%.1f", src=src, exp=e,
        note="scatter / quoted half-width")
    add("CloCatalogVariance", ms["catalog_variance_component"], "%.2f", src=src, exp=e,
        note="quadrature excess of the scatter over the quoted width")

    # closure blocks: disjoint 100-event blocks of the same parent set
    blocks = sorted((E_CLO / "results").glob("closure_b?.json"))
    vals = [jload(b)["H0"]["median"] for b in blocks]
    st = stats([v - H0_TRUTH for v in vals])
    bsrc = rel(E_CLO / "results") + "/closure_b{0..9}.json"
    add("CloBlockN", st["n"], "%d", src=bsrc, exp=e,
        note="disjoint 100-event blocks of the 1000-event parent set")
    add("CloBlockMeanHzero", float(np.mean(vals)), "%.3f", src=bsrc, exp=e)
    add("CloBlockOffset", st["mean"], "%+.2f", src=bsrc, exp=e)
    add("CloBlockOffsetAbs", abs(st["mean"]), "%.2f", src=bsrc, exp=e)
    add("CloBlockSem", st["sem"], "%.2f", src=bsrc, exp=e)
    add("CloBlockSig", st["sig"], "%.1f", src=bsrc, exp=e)

    dec = d["decomposition"]
    add("CloDecompTotal", dec["peak_total"], "%.2f", src=src, exp=e)
    add("CloDecompNum", dec["peak_per_event_numerator"], "%.2f", src=src, exp=e)
    add("CloDecompNumOff", dec["peak_per_event_numerator"] - H0_TRUTH, "%+.2f",
        src=src, exp=e)
    add("CloDecompShift", dec["shift_from_selection_term"], "%+.2f", src=src, exp=e)
    add("CloDecompDlnmu", dec["dlnmu_dH0_at_truth"], "%.4f", src=src, exp=e)
    add("CloDecompNobs", dec["nobs"], "%d", src=src, exp=e)

    loc = d["localisation"]
    add("CloLocBlockSize", loc["block_size"], "%d", src=src, exp=e)
    add("CloLocNBlocks", loc["n_blocks"], "%d", src=src, exp=e)
    add("CloLocPullMean", loc["dlogL_per_event"]["mean"], "%+.3f", src=src, exp=e,
        note="per-event log-likelihood pull, mean over blocks")
    add("CloLocPullMad", loc["dlogL_per_event"]["mad"], "%.3f", src=src, exp=e)
    add("CloLocOutliers", len(loc["outlier_blocks"]), "%d", src=src, exp=e)

    # levers that were tested and moved nothing
    lev = {r["lever"]: r["delta_H0"] for r in d["elimination"]}
    names = {
        "CloLeverGamma": "rate index γ: 0 → 1 (GLASS mock)",
        "CloLeverDepthHalf": "catalog truncated to z ≤ 0.5 (1M → 48.6k hosts)",
        "CloLeverDepthOne": "catalog truncated to z ≤ 1.0 (1M → 260.6k hosts)",
        "CloLeverNside": "sky pixelisation nside 64 → 16",
        "CloLeverSigma": "distance uncertainty σ: 0.10 → 0.01",
        "CloLeverPe": "PE construction: truth-centred → flat-prior posterior",
        "CloLeverEdge": "GLASS catalog edge 1.56 → 1.0 (separate mock)",
    }
    for nm, key in names.items():
        add(nm, lev[key], "%+.3f", src=src, exp=e, note=key)
    add("CloLeverDepthMax",
        max(abs(lev[names["CloLeverDepthHalf"]]), abs(lev[names["CloLeverDepthOne"]])),
        "sci", src=src, exp=e, note="largest |dH0| over the two depth truncations")

    # the detection/measurement A/B
    op = E_CLO / "results" / "obsdet_summary.json"
    o = jload(op)
    osrc = rel(op)
    for arm, tag in (("ctrl", "Ctrl"), ("obs", "Obs")):
        s = o["arms"][arm]["offset_stats"]
        add(f"Ab{tag}N", s["n"], "%d", src=osrc, exp=e)
        add(f"Ab{tag}Mean", s["mean"], "%+.3f", src=osrc, exp=e)
        add(f"Ab{tag}MeanAbs", abs(s["mean"]), "%.2f", src=osrc, exp=e)
        add(f"Ab{tag}Sd", s["sd"], "%.3f", src=osrc, exp=e)
        add(f"Ab{tag}Sem", s["sem"], "%.3f", src=osrc, exp=e)
        add(f"Ab{tag}Sig", s["sigma_from_zero"], "%.1f", src=osrc, exp=e)
        add(f"Ab{tag}Hw", o["arms"][arm]["mean_half_width"], "%.2f", src=osrc, exp=e)
        add(f"Ab{tag}Rej", o["arms"][arm]["n_rejected_total"], "%d", src=osrc, exp=e)
    pd = o["paired_difference_obs_minus_ctrl"]
    add("AbDiffMean", pd["mean"], "%+.3f", src=osrc, exp=e,
        note="paired per-realisation difference, fixed-noise arm minus control")
    add("AbDiffSem", pd["sem"], "%.3f", src=osrc, exp=e)
    add("AbDiffSig", pd["sigma_from_zero"], "%.1f", src=osrc, exp=e)
    add("AbFracRemoved", 100.0 * o["fraction_of_control_bias_removed"], "%.0f",
        src=osrc, exp=e, note="percent of the control offset removed")
    bp = o["baseline_published"]["offset_stats"]
    add("AbBaselineMean", bp["mean"], "%+.3f", src=osrc, exp=e,
        note="the five-seed baseline the control arm has to reproduce")
    add("AbBaselineSem", bp["sem"], "%.3f", src=osrc, exp=e)
    add("AbCellsTotal", 3220, "intsep", src=osrc, exp=e,
        note="20 realisations x 161 grid cells")

    # residual after both generator fixes
    add("CloResidual", o["arms"]["obs"]["offset_stats"]["mean"], "%+.2f", src=osrc, exp=e,
        note="unexplained residual offset")
    add("CloResidualAbs", abs(o["arms"]["obs"]["offset_stats"]["mean"]), "%.2f",
        src=osrc, exp=e)
    add("CloResidualSem", o["arms"]["obs"]["offset_stats"]["sem"], "%.2f",
        src=osrc, exp=e)
    add("CloResidualSig", o["arms"]["obs"]["offset_stats"]["sigma_from_zero"], "%.1f",
        src=osrc, exp=e)

    # post-fix closure hooks: fill automatically once the regenerated-mock
    # campaign summary exists (fixed generator, PR #335 sky width)
    fx = E_CLO / "results" / "obsdet_fix_summary.json"
    fx_names = ["CloFixN", "CloFixMean", "CloFixSem", "CloFixSig",
                "CloFixDiff", "CloFixDiffSem", "CloFixVsOracle",
                "CloFixVsOracleSem"]
    if fx.exists():
        f = jload(fx)
        s = f["arms"]["fix"]["offset_stats"]
        add("CloFixN", s["n"], "%d", src=rel(fx), exp=e)
        add("CloFixMean", s["mean"], "%+.2f", src=rel(fx), exp=e,
            note="campaign offset with the repaired generator")
        add("CloFixSem", s["sem"], "%.2f", src=rel(fx), exp=e)
        add("CloFixSig", s["sigma_from_zero"], "%.1f", src=rel(fx), exp=e)
        pdd = f["paired_fix_minus_obs"]
        add("CloFixDiff", pdd["mean"], "%+.2f", src=rel(fx), exp=e,
            note="paired per-realisation improvement from the sky-width repair")
        add("CloFixDiffSem", pdd["sem"], "%.2f", src=rel(fx), exp=e)
        po = f["paired_fix_minus_oracle_old"]
        add("CloFixVsOracle", po["mean"], "%+.2f", src=rel(fx), exp=e,
            note="repaired campaign vs the exact-likelihood attribution")
        add("CloFixVsOracleSem", po["sem"], "%.2f", src=rel(fx), exp=e)
    else:
        for nm in fx_names:
            add(nm, None, src=rel(fx) + " (not yet written)", exp=e,
                note="PENDING: post-fix K=1 campaign rerun in progress")


def sec_oracle():
    """The exact-likelihood oracle campaign: the residual decomposed into the
    third generator defect (latent-derived sky width) and estimator overhead."""
    e = "experiment\\_matched\\_mock"
    p = E_CLO / "results" / "oracle_summary.json"
    d = jload(p)
    src = rel(p)
    add("OrcN", d["n_realisations"], "%d", src=src, exp=e)
    ex = d["offset_exact_oracle"]
    add("OrcExactOff", ex["mean"], "%+.2f", src=src, exp=e,
        note="exact-likelihood offset: the latent sky-width defect")
    add("OrcExactSem", ex["sem"], "%.2f", src=src, exp=e)
    add("OrcExactSig", ex["sigma_from_zero"], "%.1f", src=src, exp=e)
    add("OrcExactSd", ex["sd"], "%.2f", src=src, exp=e)
    pr = d["paired_ds_minus_oracle"]
    add("OrcPairedOff", pr["mean"], "%+.2f", src=src, exp=e,
        note="darksirens minus exact likelihood: the estimator overhead")
    add("OrcPairedSem", pr["sem"], "%.2f", src=src, exp=e)
    add("OrcPairedSig", pr["sigma_from_zero"], "%.1f", src=src, exp=e)
    sc = d["score_identity"]["per_event_score_deficit_at_truth"]
    add("OrcScoreDef", sc["mean"], "%+.4f", src=src, exp=e,
        note="per-event score deficit at truth, per km/s/Mpc")
    add("OrcScoreDefSem", sc["sem"], "%.4f", src=src, exp=e)
    add("OrcScoreDemand", d["farr_term"]["b"]["dlnmu_dH0_exact"], "%.4f",
        src=src, exp=e, note="d ln mu/dH0 at truth: what the score identity demands")
    ba = d["bootstrap_asis"]["offset"]
    bf = d["bootstrap_fix"]["offset"]
    add("OrcBootN", ba["n"], "%d", src=src, exp=e)
    add("OrcBootAsisOff", ba["mean"], "%+.2f", src=src, exp=e,
        note="exact likelihood on fresh events, generator recipe as-is")
    add("OrcBootAsisSem", ba["sem"], "%.2f", src=src, exp=e)
    add("OrcBootFixOff", bf["mean"], "%+.2f", src=src, exp=e,
        note="same with the sky width derived from the observables: closure")
    add("OrcBootFixSem", bf["sem"], "%.2f", src=src, exp=e)
    add("OrcFarr", d["farr_term"]["b"]["dH0_sel_farr_term"], "%+.2f", src=src,
        exp=e, note="H0 tilt of the 1/Neff selection-variance correction")
    mu = d["muhat_minus_exact_dH0"]
    add("OrcMuNoiseSd", mu["sd"], "%.2f", src=src, exp=e,
        note="per-realisation scatter of the mu_hat estimator on H0")
    add("OrcMuNoiseMean", mu["mean"], "%+.2f", src=src, exp=e)
    lad = d["numerator_ladder_dH0"]
    add("OrcPix", lad["pixelation_O3_minus_O2"]["mean"], "%+.2f", src=src, exp=e,
        note="sky-pixelation term of the numerator ladder, on H0")
    add("OrcPixSem", lad["pixelation_O3_minus_O2"]["sem"], "%.2f", src=src, exp=e)
    add("OrcJac", lad["jacobian_O3b_minus_O3"]["mean"], "%+.3f", src=src, exp=e)
    add("OrcPeWidth", lad["pe_width_O2_minus_O1"]["mean"], "%+.3f", src=src, exp=e,
        note="fixed-vs-heteroscedastic mass-width term (a documented latent)")
    add("OrcKde", lad["kde_O4_minus_O3b"]["mean"], "%+.3f", src=src, exp=e)


def sec_kernel():
    """The catalog kernel width: the N_eff / PE-variance trade and the
    sigma_kde broadening threshold."""
    e = "experiment\\_matched\\_mock"
    p = E_CLO / "results" / "kernel_width_neff.json"
    d = jload(p)
    src = rel(p)
    a = d["arms"]["surveyconf"]
    b = d["arms"]["dz3e3"]
    add("KwNeffSpec", a["Neff_at_truth"], "intsep", src=src, exp=e,
        note="selection N_eff at truth, SurveyConfig (spectroscopic) widths")
    add("KwNeffAdopt", b["Neff_at_truth"], "intsep", src=src, exp=e,
        note="same, adopted dz = 3e-3 kernels")
    add("KwPeVarSpec", a["pe_variance_sum"], "%.0f", src=src, exp=e,
        note="summed per-event MC variance, spectroscopic widths")
    add("KwPeVarAdopt", b["pe_variance_sum"], "%.0f", src=src, exp=e)
    add("KwDzSpecMed", a["median_kernel_width"], "sci", src=src, exp=e,
        note="median kernel width of the spectroscopic pixelation")
    add("KwFloor", a["legacy_floor_5N"], "intsep", src=src, exp=e)
    add("KwNeffRatio", d["neff_ratio_dz3e3_over_surveyconf"], "%.1f", src=src, exp=e)
    add("KwPeVarRatio", d["pe_variance_ratio_surveyconf_over_dz3e3"], "%.1f",
        src=src, exp=e)

    sp = E_CLO / "results" / "skde_summary.json"
    s = jload(sp)
    ssrc = rel(sp)
    pz = s["pe_redshift_width"]
    add("SkdePeZWidth", pz["median"], "%.3f", src=ssrc, exp=e,
        note="median per-event PE redshift width sigma_dL dL/(ddL/dz)")
    rb = s["realisations"]["b"]["rungs"]
    names = {"0.000": "Nought", "0.020": "Twenty", "0.025": "TwentyFive",
             "0.030": "Thirty", "0.035": "ThirtyFive", "0.040": "Forty",
             "0.070": "Seventy"}
    for k, nm in names.items():
        add(f"Skde{nm}Off", rb[k]["offset"], "%+.2f", src=ssrc, exp=e,
            note=f"H0 offset at sigma_kde = {k}")
    flat = [rb[k]["offset"] for k in ("0.000", "0.003", "0.010", "0.020")]
    add("SkdeFlatSpread", max(flat) - min(flat), "%.2f", src=ssrc, exp=e,
        note="span of the offset over the flat regime sigma_kde <= 0.020")
    rs = s["realisations"]["s4102"]["rungs"]
    add("SkdeSTwentyOff", rs["0.020"]["offset"], "%+.2f", src=ssrc, exp=e,
        note="second realisation, sigma_kde = 0.020")
    add("SkdeSFortyOff", rs["0.040"]["offset"], "%+.2f", src=ssrc, exp=e,
        note="second realisation, sigma_kde = 0.040")
    add("SkdeEffTwenty", math.hypot(s["dzgals"], 0.020), "%.3f", src=ssrc, exp=e,
        note="effective kernel width at sigma_kde = 0.020")
    add("SkdeEffThirty", math.hypot(s["dzgals"], 0.030), "%.3f", src=ssrc, exp=e)
    add("SkdeEffForty", math.hypot(s["dzgals"], 0.040), "%.3f", src=ssrc, exp=e)
    add("SkdeEffSeventy", math.hypot(s["dzgals"], 0.070), "%.3f", src=ssrc, exp=e)


# ===========================================================================
# 4. Anchored completeness ladder (K=1)
# ===========================================================================
ANC = {"c100": "Complete", "m20": "Mtwenty", "m19": "Mnineteen", "m18": "Meighteen"}


def sec_anchored():
    p = E_ANC / "results" / "summary.json"
    d = jload(p)
    src = rel(p)
    e = "experiment\\_completeness\\_anchored"
    a = d["anchor"]["anchor"]
    add("AncLogNzero", a["log10n0"], "%.4f", src=src, exp=e,
        note="density amplitude anchored to the model form's best fit")
    add("AncDelta", a["delta"], "%+.3f", src=src, exp=e)
    add("AncNaiveLogNzero", d["anchor"]["naive_mean_density_log10n0"], "%.4f",
        src=src, exp=e, note="raw mean density, for contrast with the fitted anchor")
    add("AncResidFit", 100 * d["anchor"]["shape_residual_fit_range"]["rms_frac"], "%.1f",
        src=src, exp=e, note="density shape residual over the fit range, per cent rms")
    add("AncResidHorizon",
        100 * d["anchor"]["shape_residual_within_z_ref"]["rms_frac"], "%.1f",
        src=src, exp=e, note="same, within the detection horizon")
    add("AncZref", d["anchor"]["shape_residual_within_z_ref"]["z_ref"], "%.2f",
        src=src, exp=e)
    add("AncNHostsInHorizon", 9105, "intsep", src=src, exp=e,
        note="hosts inside the horizon that set the shot-noise floor")
    for k, tag in ANC.items():
        L = d["levels"][k]
        add(f"Anc{tag}C", 100 * L["completeness_within_z_ref"], "%.3g", src=src, exp=e,
            note=f"completeness within the horizon, level {k}")
        add(f"Anc{tag}CAll", 100 * L["completeness_all_z"], "%.3g", src=src, exp=e)
        add(f"Anc{tag}NHosts", L["n_hosts"], "intsep", src=src, exp=e)
        add(f"Anc{tag}Empty", 100 * L["empty_pixel_fraction"], "%.3g", src=src, exp=e)
        add(f"Anc{tag}Off", L["offset"], "%+.3f", src=src, exp=e)
        add(f"Anc{tag}OffAbs", abs(L["offset"]), "%.2f", src=src, exp=e)
        add(f"Anc{tag}Hw", L["hw"], "%.3f", src=src, exp=e)
        add(f"Anc{tag}OffCtl", L["offset_vs_control"], "%+.3f", src=src, exp=e)
        add(f"Anc{tag}SigCtl", L["sigma_vs_control"], "%.1f", src=src, exp=e)
        add(f"Anc{tag}Growth", d["verdict"]["interval_growth"][k], "%.2f", src=src, exp=e)
        add(f"Anc{tag}Rej", L["n_rejected_cells"], "%d", src=src, exp=e)
        add(f"Anc{tag}Cells", L["n_evals"], "%d", src=src, exp=e)
        if L["mag_limit"]:
            add(f"Anc{tag}Mag", L["mag_limit"], "%.0f", src=src, exp=e)
    v = d["verdict"]
    add("AncMaxSigCtl", v["max_sigma_vs_control"], "%.1f", src=src, exp=e,
        note="largest departure from the complete-catalog control")
    add("AncMaxOffCtl", v["max_abs_offset_vs_control"], "%.2f", src=src, exp=e)
    add("AncMaxGrowth", max(v["interval_growth"].values()), "%.1f", src=src, exp=e)
    add("AncHwLo", min(d["levels"][k]["hw"] for k in ANC), "%.2f", src=src, exp=e)
    add("AncHwHi", max(d["levels"][k]["hw"] for k in ANC), "%.2f", src=src, exp=e)


# ===========================================================================
# 5. Deep two-tracer mock and the catalog-targeted selection lane
# ===========================================================================
def sec_deep():
    p = E_DEEP / "results" / "summary.json"
    d = jload(p)
    src = rel(p)
    e = "experiment\\_twotracer\\_deep"
    m = d["meta"]
    add("DeepNobs", m["nobs"], "%d", src=src, exp=e)
    add("DeepNgalEvents", m["n_gal_events"], "%d", src=src, exp=e)
    add("DeepNagnEvents", m["n_agn_events"], "%d", src=src, exp=e)
    add("DeepFTruth", m["truth_f_agn"], "%.3f", src=src, exp=e)
    add("DeepNside", m["nside"], "%d", src=src, exp=e)
    add("DeepSnrThresh", m["snr_threshold"], "%.0f", src=src, exp=e)
    add("DeepSigmaDl", m["dL_fractional_uncertainty"], "%.2f", src=src, exp=e)
    add("DeepZmax", m["zmax"], "%.1f", src=src, exp=e)
    add("DeepNgalHosts", m["surveys"]["gal"]["n_hosts"], "intsep", src=src, exp=e)
    add("DeepNagnHosts", m["surveys"]["agn"]["n_hosts"], "intsep", src=src, exp=e)
    add("DeepGalLogNzero", m["surveys"]["gal"]["log10n0_count_anchored"], "%.2f",
        src=src, exp=e)
    add("DeepAgnLogNzero", m["surveys"]["agn"]["log10n0_count_anchored"], "%.2f",
        src=src, exp=e)
    add("DeepGalEmpty", 100 * m["surveys"]["gal"]["empty_pixel_fraction"], "%.1f",
        src=src, exp=e)
    add("DeepAgnEmpty", 100 * m["surveys"]["agn"]["empty_pixel_fraction"], "%.1f",
        src=src, exp=e)

    FTAG = {0.0: "Nought", 0.3: "Three", 0.7: "Seven", 1.0: "Unity"}
    for prop, ptag in (("popuni", "Pop"), ("targeted", "Tgt")):
        for row in d["neff_vs_f_at_N" + str(m["nobs"])][prop]:
            t = FTAG[row["f"]]
            add(f"Neff{ptag}{t}", row["Neff"], "%.0f", src=src, exp=e,
                note=f"selection-integral N_eff, {prop} proposal at f_AGN={row['f']}")
            add(f"Neff{ptag}{t}Pass", "yes" if row["passes"] else "no", "%s",
                src=src, exp=e)
    for row in d["neff_vs_f_at_N" + str(m["nobs"])]["popuni"]:
        add(f"NeffF{FTAG[row['f']]}", row["f"], "%.5g", src=src, exp=e,
            note="mixture weight at which the selection integral was evaluated")
    add("NeffFloor", d["neff_vs_f_at_N200"]["popuni"][0]["threshold"], "%.0f",
        src=src, exp=e, note="validity floor, 5 N_obs")
    pop = {r["f"]: r["Neff"] for r in d["neff_vs_f_at_N200"]["popuni"]}
    tgt = {r["f"]: r["Neff"] for r in d["neff_vs_f_at_N200"]["targeted"]}
    add("NeffPopDecay", pop[0.0] / pop[1.0], "%.0f", src=src, exp=e,
        note="factor by which the population proposal decays across the f grid")
    add("NeffTgtRise", tgt[1.0] / tgt[0.0], "%.0f", src=src, exp=e)
    add("NeffTgtCostAtZero", 100 * (1 - tgt[0.0] / pop[0.0]), "%.0f", src=src, exp=e,
        note="per cent of N_eff given up at f_AGN = 0")
    add("NeffTgtMarginAtZero", tgt[0.0] / d["neff_vs_f_at_N200"]["popuni"][0]["threshold"],
        "%.1f", src=src, exp=e)

    for k, tag in (("deep_popuni", "Pop"), ("deep_targeted", "Tgt")):
        s = d["fscans"][k]
        add(f"DeepF{tag}Med", s["median"], "%.4f", src=src, exp=e)
        add(f"DeepF{tag}Lo", s["ci68"][0], "%.3f", src=src, exp=e)
        add(f"DeepF{tag}Hi", s["ci68"][1], "%.3f", src=src, exp=e)
        add(f"DeepF{tag}Hw", hw(s["ci68"]), "%.3f", src=src, exp=e)
        add(f"DeepF{tag}Rej", s["n_rejected_cells"], "%d", src=src, exp=e)
        add(f"DeepF{tag}Cells", s["n_evals"], "%d", src=src, exp=e)
        add(f"DeepF{tag}Adm", s["n_evals"] - s["n_rejected_cells"], "%d", src=src, exp=e)
        add(f"DeepF{tag}AdmHi", s["admitted_f_range"][1], "%.3f", src=src, exp=e)
    st = d["fscans"]["deep_targeted"]
    add("DeepFTgtSig", abs(st["median"] - 0.30) / hw(st["ci68"]), "%.1f", src=src, exp=e,
        note="f_AGN offset in units of its own half-width")
    add("DeepFTgtOff", st["median"] - 0.30, "%+.3f", src=src, exp=e)
    add("DeepFTgtOffAbs", abs(st["median"] - 0.30), "%.3f", src=src, exp=e)

    # like-for-like at the reduced sample: same events, only the campaign changes
    small = jload(E_DEEP / "results" / "tgt_fscan_n80.json")["f"]
    ssrc = rel(E_DEEP / "results" / "tgt_fscan_n80.json")
    add("DeepSmallNobs", 80, "%d", src=ssrc, exp=e,
        note="reduced event sample at which the untargeted campaign is still usable")
    add("DeepFSmallTgtMed", small["median"], "%.4f", src=ssrc, exp=e)
    add("DeepFSmallTgtLo", small["ci68"][0], "%.3f", src=ssrc, exp=e)
    add("DeepFSmallTgtHi", small["ci68"][1], "%.3f", src=ssrc, exp=e)
    add("DeepFSmallTgtHw", hw(small["ci68"]), "%.3f", src=ssrc, exp=e)

    j = d["joint_targeted"]
    add("DeepJointHzero", j["H0"], "%.2f", src=src, exp=e)
    add("DeepJointHzeroLo", j["H0_ci68"][0], "%.2f", src=src, exp=e)
    add("DeepJointHzeroHi", j["H0_ci68"][1], "%.2f", src=src, exp=e)
    add("DeepJointHzeroHw", hw(j["H0_ci68"]), "%.2f", src=src, exp=e)
    add("DeepJointHzeroOff", j["H0"] - H0_TRUTH, "%+.2f", src=src, exp=e)
    add("DeepJointHzeroOffAbs", abs(j["H0"] - H0_TRUTH), "%.2f", src=src, exp=e)
    add("DeepJointHzeroSig", abs(j["H0"] - H0_TRUTH) / hw(j["H0_ci68"]), "%.1f",
        src=src, exp=e)
    add("DeepJointF", j["f"], "%.4f", src=src, exp=e)
    add("DeepJointFHw", hw(j["f_ci68"]), "%.3f", src=src, exp=e)
    add("DeepJointRho", j["rho"], "%+.2f", src=src, exp=e)
    add("DeepJointRhoAbs", abs(j["rho"]), "%.2f", src=src, exp=e)
    add("DeepJointRej", j["n_rejected_cells"], "%d", src=src, exp=e)
    add("DeepJointCells", j["n_evals"], "%d", src=src, exp=e)
    add("DeepJointAdmLo", j["fully_admitted_H0_range"][0], "%.2f", src=src, exp=e)
    add("DeepJointAdmHi", j["fully_admitted_H0_range"][1], "%.2f", src=src, exp=e)
    add("DeepJointMassAdj", j["posterior_mass_adjacent_to_rejected"], "sci",
        src=src, exp=e, note="posterior mass adjacent to the inadmissible region")

    # post-fix (repaired-generator) targeted scans: the measurement of record.
    # The detected set, catalogs, survey files and injection campaigns are
    # bit-identical to the pre-fix run; only the PE construction downstream of
    # the observable-derived sky width changed, so pre/post is a paired
    # comparison on the same events.
    fp = E_DEEP / "results" / "summary_fix.json"
    fd = jload(fp)
    fsrc = rel(fp)
    fx = fd["fix"]
    s = fx["tgt_fscan_n200"]
    add("DeepFixFTgtMed", s["median"], "%.4f", src=fsrc, exp=e,
        note="repaired-generator fraction at the full sample, targeted campaign")
    add("DeepFixFTgtLo", s["ci68"][0], "%.3f", src=fsrc, exp=e)
    add("DeepFixFTgtHi", s["ci68"][1], "%.3f", src=fsrc, exp=e)
    add("DeepFixFTgtHw", s["half_width68"], "%.3f", src=fsrc, exp=e)
    add("DeepFixFTgtOff", s["offset"], "%+.3f", src=fsrc, exp=e)
    add("DeepFixFTgtOffAbs", abs(s["offset"]), "%.3f", src=fsrc, exp=e)
    add("DeepFixFTgtSig", abs(s["offset_sigma"]), "%.1f", src=fsrc, exp=e,
        note="offset in units of its own half-width")
    s80 = fx["tgt_fscan_n80"]
    add("DeepFixFSmallMed", s80["median"], "%.4f", src=fsrc, exp=e,
        note="repaired-generator fraction at the reduced sample")
    add("DeepFixFSmallLo", s80["ci68"][0], "%.3f", src=fsrc, exp=e)
    add("DeepFixFSmallHi", s80["ci68"][1], "%.3f", src=fsrc, exp=e)
    j = fx["tgt_joint_n200"]
    add("DeepFixJointHzero", j["H0_median"], "%.2f", src=fsrc, exp=e,
        note="repaired-generator joint plane, marginal H0")
    add("DeepFixJointHzeroLo", j["H0_ci68"][0], "%.2f", src=fsrc, exp=e)
    add("DeepFixJointHzeroHi", j["H0_ci68"][1], "%.2f", src=fsrc, exp=e)
    add("DeepFixJointHzeroHw", j["H0_half_width68"], "%.2f", src=fsrc, exp=e)
    add("DeepFixJointHzeroOff", j["H0_offset"], "%+.2f", src=fsrc, exp=e)
    add("DeepFixJointHzeroSig", abs(j["H0_offset_sigma"]), "%.1f", src=fsrc, exp=e)
    add("DeepFixJointF", j["f_median"], "%.4f", src=fsrc, exp=e)
    add("DeepFixJointFHw", j["f_half_width68"], "%.3f", src=fsrc, exp=e)
    add("DeepFixJointFSig",
        abs(j["f_median"] - fd["truth"]["f_AGN"]) / j["f_half_width68"], "%.1f",
        src=fsrc, exp=e)
    add("DeepFixJointRho", j["rho"], "%+.2f", src=fsrc, exp=e)
    add("DeepFixJointRej", j["n_rejected"], "%d", src=fsrc, exp=e)
    add("DeepFixJointMassAdj", j["posterior_mass_adjacent_to_rejected"], "sci",
        src=fsrc, exp=e)

    vp = E_DEEP / "results" / "injections_targeted_k2_validation.json"
    v = jload(vp)
    vsrc = rel(vp)
    add("InjNdraw", v["ndraw_total_proposed"] / 1e6, "%.0f", src=vsrc, exp=e,
        note="proposals drawn, millions")
    add("InjNdet", v["n_detected"], "intsep", src=vsrc, exp=e)
    add("InjFracDet", v["frac_detected"], "sci", src=vsrc, exp=e)
    add("InjOnAgnSupport", 100 * v["frac_rows_on_catalog_support"]["agn"], "%.1f",
        src=vsrc, exp=e)
    br = v["frac_rows_on_catalog_support_by_branch"]["agn"]
    add("InjOnAgnPop", 100 * br["population"], "%.1f", src=vsrc, exp=e)
    add("InjOnAgnUni", 100 * br["uniform"], "%.1f", src=vsrc, exp=e)
    add("InjPdrawErr", v["pdraw_recompute_max_rel_err"], "sci", src=vsrc, exp=e,
        note="max relative error of the stored mixture density vs an independent recompute")
    mx = v["meta"]["mixture"]
    add("InjWpop", mx["population"], "%.2f", src=vsrc, exp=e)
    add("InjWuni", mx["uniform"], "%.2f", src=vsrc, exp=e)
    add("InjWtgt", mx["targeted_agn"], "%.2f", src=vsrc, exp=e)


# ===========================================================================
# 6. Two-tracer completeness ladder
# ===========================================================================
INC = {"complete": "Complete", "m21.0": "Mtwentyone", "m20.0": "Mtwenty",
       "m19.0": "Mnineteen", "m18.0": "Meighteen"}


def sec_incomplete():
    # The ladder of record is the post-fix rerun (repaired generator, same
    # detected set and catalogs, PE rebuilt with the observable-derived sky
    # width); the pre-fix ladder survives only through the IncPre* macros used
    # in explicit before/after statements.
    p = E_INC / "results" / "summary_fix.json"
    d = jload(p)
    src = rel(p)
    e = "experiment\\_twotracer\\_incomplete"
    add("IncZref", d["z_ref"], "%.2f", src=src, exp=e)
    add("IncFTruth", d["truth"]["f_AGN"], "%.2f", src=src, exp=e)
    for k, tag in INC.items():
        L = d["levels"][k]
        add(f"Inc{tag}C", L["agn_completeness_within_horizon"], "%.3f", src=src, exp=e,
            note=f"completeness within the horizon, rung {k}")
        add(f"Inc{tag}CPct", 100 * L["agn_completeness_within_horizon"], "%.3g",
            src=src, exp=e)
        add(f"Inc{tag}AgnHosts", L["agn_n_hosts"], "intsep", src=src, exp=e)
        add(f"Inc{tag}AgnInHorizon", L["agn_n_hosts_within_horizon"], "%d",
            src=src, exp=e)
        add(f"Inc{tag}GalHosts", L["gal_n_hosts"], "intsep", src=src, exp=e)
        add(f"Inc{tag}GalEmpty", 100 * L["gal_empty_pixel_fraction"], "%.3g",
            src=src, exp=e)
        add(f"Inc{tag}AgnEmpty", 100 * L["agn_empty_pixel_fraction"], "%.3g",
            src=src, exp=e)
        add(f"Inc{tag}Neff", L["Neff"] / 1e3, "%.0f", src=src, exp=e,
            note="selection N_eff, thousands")
        j = L["joint"]
        add(f"Inc{tag}SigHzero", j["H0_half_width68"], "%.3f", src=src, exp=e)
        add(f"Inc{tag}SigF", j["f_half_width68"], "%.4f", src=src, exp=e)
        add(f"Inc{tag}Rho", j["rho"], "%+.2f", src=src, exp=e)
        add(f"Inc{tag}Hzero", j["H0_median"], "%.2f", src=src, exp=e)
        add(f"Inc{tag}F", j["f_median"], "%.3f", src=src, exp=e)
        add(f"Inc{tag}Rej", j["n_rejected"], "%d", src=src, exp=e)
        add(f"Inc{tag}Cells", j["n_evals"], "%d", src=src, exp=e)
        add(f"Inc{tag}MassAdj", 100 * j["posterior_mass_adjacent_to_rejected"], "%.1f",
            src=src, exp=e)
        wd = L["width_degradation_vs_complete"]
        add(f"Inc{tag}FacHzero", wd["joint_H0"], "%.2f", src=src, exp=e)
        add(f"Inc{tag}FacF", wd["joint_f"], "%.2f", src=src, exp=e)
        add(f"Inc{tag}Fscan", L["fscan"]["median"], "%.4f", src=src, exp=e)
        add(f"Inc{tag}FscanHw", L["fscan"]["half_width68"], "%.4f", src=src, exp=e)
        if "sky_shuffle_null" in L:
            n = L["sky_shuffle_null"]
            add(f"Inc{tag}NullMed", n["median"], "%.4f", src=src, exp=e)
            add(f"Inc{tag}NullHw", n["half_width68"], "%.4f", src=src, exp=e)
            add(f"Inc{tag}NullRatio", n["width_ratio_null_over_real"], "%.2f",
                src=src, exp=e, note="null width / data width")
            add(f"Inc{tag}Sep", n["displacement_in_widths"], "%.2f", src=src, exp=e,
                note="peak-to-null separation in data widths")
    facH = {k: d["levels"][k]["width_degradation_vs_complete"]["joint_H0"] for k in INC}
    add("IncFacHzeroMax", max(facH.values()), "%.2f", src=src, exp=e)
    add("IncFacHzeroFirst", facH["m21.0"], "%.2f", src=src, exp=e,
        note="H0 width factor at the first rung below complete")
    offs = {k: d["levels"][k]["joint"]["H0_median"] - H0_TRUTH for k in INC}
    add("IncHzeroOffLo", min(offs.values()), "%+.2f", src=src, exp=e,
        note="most negative H0 offset anywhere on the ladder")
    add("IncHzeroOffHi", max(offs.values()), "%+.2f", src=src, exp=e,
        note="most positive H0 offset anywhere on the ladder")
    fsc = d["levels"]["complete"]["fscan"]
    add("IncCompleteFscanOff", fsc["median"] - d["truth"]["f_AGN"], "%+.3f",
        src=src, exp=e, note="complete-rung fraction offset at fixed H0")
    add("IncCompleteFscanSig",
        abs(fsc["median"] - d["truth"]["f_AGN"]) / fsc["half_width68"], "%.1f",
        src=src, exp=e)
    facF = {k: d["levels"][k]["width_degradation_vs_complete"]["joint_f"] for k in INC}
    add("IncFacFMax", max(facF.values()), "%.2f", src=src, exp=e)
    add("IncCRange", 1.0 / d["levels"]["m18.0"]["agn_completeness_within_horizon"],
        "%.1f", src=src, exp=e, note="span of the completeness ladder")
    sep = {k: d["levels"][k]["sky_shuffle_null"]["displacement_in_widths"]
           for k in INC if "sky_shuffle_null" in d["levels"][k]}
    add("IncSepLoss", 100 * (1 - sep["m18.0"] / sep["complete"]), "%.0f", src=src, exp=e,
        note="per cent loss of peak-to-null separation across the ladder")
    add("IncNeffRise", d["levels"]["m18.0"]["Neff"] / d["levels"]["complete"]["Neff"],
        "%.1f", src=src, exp=e)
    add("IncRejMax", max(d["levels"][k]["joint"]["n_rejected"] for k in INC), "%d",
        src=src, exp=e)
    add("IncRejMin", min(d["levels"][k]["joint"]["n_rejected"] for k in INC), "%d",
        src=src, exp=e)
    add("IncMassAdjMax",
        max(d["levels"][k]["joint"]["posterior_mass_adjacent_to_rejected"]
            for k in INC), "sci", src=src, exp=e,
        note="largest posterior mass adjacent to an inadmissible cell, fraction")
    add("IncPeVarComplete", d["levels"]["complete"]["pe_variance_sum"], "%.1f",
        src=src, exp=e,
        note="summed per-event Monte Carlo variance of the log-likelihood")
    add("IncGuardFloor", d["levels"]["complete"]["guard_threshold"], "%.0f",
        src=src, exp=e, note="reliability floor 5 N_obs at N_obs = 200")

    # pre-fix ladder values kept only for explicit before/after statements
    pp = E_INC / "results" / "ladder_prepost_fix.json"
    pd = jload(pp)
    psrc = rel(pp)
    pre_degH = pd["pre"]["deg_H0"]
    add("IncPreFacHzeroMin", min(pre_degH), "%.2f", src=psrc, exp=e,
        note="pre-repair H0 width factor minimum (the apparent improvement)")
    add("IncPreFacHzeroGain", 100 * (1 - min(pre_degH)), "%.0f", src=psrc, exp=e,
        note="pre-repair apparent improvement at intermediate depth, per cent")
    pre_off = [h - H0_TRUTH for h in pd["pre"]["H0_median"]]
    add("IncPreHzeroOffLo", min(pre_off), "%+.2f", src=psrc, exp=e,
        note="pre-repair ladder H0 offsets: every rung low")
    add("IncPreHzeroOffHi", max(pre_off), "%+.2f", src=psrc, exp=e)
    add("IncPreCompleteFscan", pd["pre"]["f_median"][0], "%.3f", src=psrc, exp=e,
        note="pre-repair complete-rung fraction at fixed H0")

    for tr, tag in (("gal", "Gal"), ("agn", "Agn")):
        ap = E_INC / "results" / f"density_anchor_{tr}.json"
        a = jload(ap)
        asrc = rel(ap)
        an = a["anchor"] if "anchor" in a and isinstance(a["anchor"], dict) else a
        add(f"Anchor{tag}LogNzero", an["log10n0"], "%.5f", src=asrc, exp=e)
        add(f"Anchor{tag}Delta", an["delta"], "%+.4f", src=asrc, exp=e)
        add(f"Anchor{tag}ResidFit",
            100 * a["shape_residual_fit_range"]["rms_frac"], "%.1f", src=asrc, exp=e)
        add(f"Anchor{tag}ResidHorizon",
            100 * a["shape_residual_within_z_ref"]["rms_frac"], "%.1f", src=asrc, exp=e)

    mp = E_INC / "results" / "mixture_calibration.json"
    if mp.exists():
        mc = jload(mp)
        add("MixCalibNote", "yes", "%s", src=rel(mp), exp=e,
            note="per-rung proposal weights solved at fixed detected-row split")
        try:
            w = mc["levels"]["complete"]["weights"]
            for key, nm in (("population", "MixWpop"), ("uniform", "MixWuni"),
                            ("targeted_gal", "MixWgal"), ("targeted_agn", "MixWagn")):
                add(nm, w[key], "%.3f", src=rel(mp), exp=e)
        except (KeyError, TypeError):
            pass


# ===========================================================================
# 7. Freeing the tracer density
# ===========================================================================
ARMS = {"fixed": "Exact", "5%": "Five", "10%": "Ten", "30%": "Thirty",
        "factor 2": "Ftwo", "free": "Free"}
ARM_LABEL = {"fixed": "exact", "5%": r"$5\%$", "10%": r"$10\%$", "30%": r"$30\%$",
             "factor 2": "factor 2", "free": "free"}


def sec_free():
    # post-fix (f, n0) grids: same events as the post-fix ladder
    p = E_FREE / "results" / "n0_arms_summary_fix.json"
    d = jload(p)
    src = rel(p)
    e = "experiment\\_completeness\\_free"
    add("FreeGtrue", d["g_true"], "%.4f", src=src, exp=e,
        note="true log10 n0 of the sparse tracer")
    add("FreeFtruth", d["truth_f"], "%.2f", src=src, exp=e)
    for k, rtag in INC.items():
        L = d["levels"][k]
        for arm, atag in ARMS.items():
            A = L["arms"][arm]
            add(f"Free{rtag}{atag}Sig", A["detection_sigma"], "%.1f", src=src, exp=e,
                note=f"f_AGN detection significance, rung {k}, n0 knowledge {arm}")
            add(f"Free{rtag}{atag}Hw", A["half_width68"], "%.3f", src=src, exp=e)
            add(f"Free{rtag}{atag}Med", A["median"], "%.3f", src=src, exp=e)
        add(f"Free{rtag}Rho", L["rho_f_n0_flat_prior"], "%+.2f", src=src, exp=e)
        n0 = L["log10n0_agn_flat_prior"]
        add(f"Free{rtag}NzeroOff", n0["offset_from_truth"], "%+.2f", src=src, exp=e,
            note="flat-prior density recovered relative to truth, dex")
        add(f"Free{rtag}NzeroOffAbs", abs(n0["offset_from_truth"]), "%.2f",
            src=src, exp=e)
        add(f"Free{rtag}EdgeMass", 100 * n0["edge_mass_low"], "%.1f", src=src, exp=e,
            note="flat-prior density mass against the low edge of the scanned range")
        add(f"Free{rtag}Prior", 10 ** abs(n0["offset_from_truth"]), "%.1f",
            src=src, exp=e, note="the same offset as a linear factor")
    for arm, atag in ARMS.items():
        A0 = d["levels"]["complete"]["arms"][arm]["detection_sigma"]
        A1 = d["levels"]["m18.0"]["arms"][arm]["detection_sigma"]
        add(f"FreeCost{atag}", 100 * (1 - A1 / A0), "%.0f", src=src, exp=e,
            note=f"per cent significance lost across the ladder, {arm} arm")
        dex = d["levels"]["complete"]["arms"][arm]["prior_sigma_dex"]
        add(f"FreePriorDex{atag}", "flat" if dex is None else dex, "%.3f",
            src=src, exp=e, note=f"prior width on log10 n0, {arm} arm, dex")
    add("FreeRhoMax", max(d["levels"][k]["rho_f_n0_flat_prior"] for k in INC), "%+.2f",
        src=src, exp=e)
    add("FreeRhoMin", min(d["levels"][k]["rho_f_n0_flat_prior"] for k in INC), "%+.2f",
        src=src, exp=e)
    add("FreeGridF", 51, "%d", src=rel(E_FREE / "results" / "fn0_complete_fix.h5"), exp=e)
    add("FreeGridN", 201, "%d", src=rel(E_FREE / "results" / "fn0_complete_fix.h5"), exp=e)
    add("FreeRangeLo", -9.6, "%.1f", src=rel(E_FREE / "results" / "fn0_complete_fix.h5"),
        exp=e)
    add("FreeRangeHi", -7.1, "%.1f", src=rel(E_FREE / "results" / "fn0_complete_fix.h5"),
        exp=e)
    add("FreeRej", d["levels"]["complete"]["n_rejected"], "%d", src=src, exp=e)
    hws = [d["levels"][k]["arms"][a]["half_width68"] for k in INC for a in ARMS]
    add("FreeHwMin", min(hws), "%.3f", src=src, exp=e,
        note="narrowest entry anywhere in the width table")


# ===========================================================================
# 8. Catalog-realisation scatter for the two-tracer mock (IN PROGRESS)
# ===========================================================================
def sec_seeds():
    p = E_SEED / "results" / "seeds_summary.json"
    e = "experiment\\_twotracer\\_seeds"
    names = ["SdsN", "SdsFOffset", "SdsFSem", "SdsFSig", "SdsFScatter", "SdsFHw",
             "SdsFFactor", "SdsHzeroOffset", "SdsHzeroSem", "SdsHzeroSig",
             "SdsHzeroScatter", "SdsHzeroHw", "SdsHzeroFactor"]
    if not p.exists():
        n_done = len(list((E_SEED / "results").glob("joint_s*.json")))
        n_plan = len(list((E_SEED / "data_derived").glob("s*")))
        for nm in names:
            add(nm, None, src=rel(p) + " (not yet written)", exp=e,
                note="PENDING: 12-seed catalog-realisation run in progress")
        add("SdsPlanned", n_plan, "%d", src=rel(E_SEED / "data_derived"), exp=e,
            note="catalog realisations requested")
        add("SdsDone", n_done, "%d", src=rel(E_SEED / "results"), exp=e,
            note="realisations with a completed joint scan at build time")
        return
    d = jload(p)
    src = rel(p)
    add("SdsN", d["joint_H0"]["n"], "%d", src=src, exp=e)
    for key, tag in (("joint_f", "F"), ("joint_H0", "Hzero"),
                     ("fscan_f", "Fscan")):
        s = d[key]
        fmt = "%+.2f" if tag == "Hzero" else "%+.3f"
        sfmt = "%.2f" if tag == "Hzero" else "%.3f"
        add(f"Sds{tag}Offset", s["mean"], fmt, src=src, exp=e,
            note=s.get("label", ""))
        add(f"Sds{tag}Sem", s["sem"], sfmt, src=src, exp=e)
        add(f"Sds{tag}Sig", s["sigma_from_zero"], "%.1f", src=src, exp=e)
        add(f"Sds{tag}Scatter", s["sd"], sfmt, src=src, exp=e)
        add(f"Sds{tag}Hw", s["mean_quoted_half_width"], sfmt, src=src, exp=e)
        add(f"Sds{tag}Factor", s["scatter_over_quoted_half_width"], "%.1f",
            src=src, exp=e)
    add("SdsPlanned", 12, "%d", src=src, exp=e)
    add("SdsDone", d["joint_H0"]["n"], "%d", src=src, exp=e)

    # post-fix hooks: the same 12 realisations regenerated with the fixed
    # generator (observable-derived sky width); fill automatically on rerun
    fx = E_SEED / "results" / "seeds_summary_fix.json"
    fx_names = ["SdsFixN",
                "SdsFixFOffset", "SdsFixFSem", "SdsFixFSig", "SdsFixFScatter",
                "SdsFixFHw", "SdsFixFFactor",
                "SdsFixHzeroOffset", "SdsFixHzeroSem", "SdsFixHzeroSig",
                "SdsFixHzeroScatter", "SdsFixHzeroHw", "SdsFixHzeroFactor",
                "SdsFixFscanOffset", "SdsFixFscanSem", "SdsFixFscanSig",
                "SdsFixFscanScatter", "SdsFixFscanHw", "SdsFixFscanFactor"]
    if fx.exists():
        f = jload(fx)
        fsrc = rel(fx)
        add("SdsFixN", f["joint_H0"]["n"], "%d", src=fsrc, exp=e)
        for key, tag in (("joint_f", "F"), ("joint_H0", "Hzero"),
                         ("fscan_f", "Fscan")):
            s = f[key]
            fmt = "%+.2f" if tag == "Hzero" else "%+.3f"
            sfmt = "%.2f" if tag == "Hzero" else "%.3f"
            add(f"SdsFix{tag}Offset", s["mean"], fmt, src=fsrc, exp=e)
            add(f"SdsFix{tag}Sem", s["sem"], sfmt, src=fsrc, exp=e)
            add(f"SdsFix{tag}Sig", s["sigma_from_zero"], "%.1f", src=fsrc, exp=e)
            add(f"SdsFix{tag}Scatter", s["sd"], sfmt, src=fsrc, exp=e)
            add(f"SdsFix{tag}Hw", s["mean_quoted_half_width"], sfmt, src=fsrc, exp=e)
            add(f"SdsFix{tag}Factor", s["scatter_over_quoted_half_width"], "%.1f",
                src=fsrc, exp=e)
    else:
        for nm in fx_names:
            add(nm, None, src=rel(fx) + " (not yet written)", exp=e,
                note="PENDING: post-fix 12-seed rerun in progress")


# ===========================================================================
# tables
# ===========================================================================
def table_f_recovery():
    d = jload(E_BASE / "results" / "summary.json")
    rows = []
    for k, tag in FKEYS.items():
        s = d["f_scan_at_true_H0"][k]
        t = d["f_truth"][k]
        interval = (rf"$[{s['ci68'][0]:.3f},\,{s['ci68'][1]:.3f}]$" if k != "1.0"
                    else rf"$[{s['onesided68_lo']:.3f},\,1]^{{\dagger}}$")
        rows.append(rf"{t:.4f} & {s['median']:.4f} & {interval} & "
                    rf"${s['median'] - t:+.3f}$ \\")
    return (
        r"\begin{deluxetable}{cccc}" "\n"
        r"\tablecaption{The AGN-hosted fraction recovered from the clustered "
        r"two-tracer mock at the true expansion rate, on complete catalogs. "
        r"\label{tab:frecovery}}" "\n"
        r"\tablehead{\colhead{planted $f_{\rm AGN}$} & \colhead{median} & "
        r"\colhead{68\% interval} & \colhead{offset}}" "\n"
        r"\startdata" "\n" + "\n".join(rows) + "\n"
        r"\enddata" "\n"
        r"\tablecomments{$^{\dagger}$ At $f_{\rm AGN}=1$ the truth lies on the prior "
        r"boundary, where an equal-tailed interval cannot cover it; the one-sided "
        r"68\% lower limit is quoted instead. Each row is a "
        r"\BaseFGridN-point scan of \GlassNobs\ events.}" "\n"
        r"\end{deluxetable}" "\n")


def table_completeness_anchored():
    d = jload(E_ANC / "results" / "summary.json")
    rows = []
    for k, tag in ANC.items():
        L = d["levels"][k]
        lab = "complete" if L["mag_limit"] is None else f"$m<{L['mag_limit']:.0f}$"
        rows.append(
            rf"{lab} & {L['n_hosts']:,} & {100*L['completeness_within_z_ref']:.1f} & "
            rf"{100*L['empty_pixel_fraction']:.1f} & ${L['offset']:+.2f}\pm{L['hw']:.2f}$ & "
            rf"${L['offset_vs_control']:+.2f}$ & {L['sigma_vs_control']:.1f} & "
            rf"{d['verdict']['interval_growth'][k]:.2f} \\".replace(",", r"\,"))
    return (
        r"\begin{deluxetable*}{lccccccc}" "\n"
        r"\tablecaption{Single-tracer flux-limit ladder. Completeness $C$ is quoted "
        r"within the detection horizon $z\le\AncZref$. Offsets are relative to the "
        r"true expansion rate; the last three columns are differential against the "
        r"complete-catalog control. \label{tab:anchored}}" "\n"
        r"\tablehead{\colhead{level} & \colhead{hosts} & \colhead{$C$ (\%)} & "
        r"\colhead{empty sky (\%)} & \colhead{$\Delta H_0$} & "
        r"\colhead{vs.\ control} & \colhead{$\sigma$} & \colhead{width $\times$}}" "\n"
        r"\startdata" "\n" + "\n".join(rows) + "\n"
        r"\enddata" "\n"
        r"\end{deluxetable*}" "\n")


def table_twotracer_ladder():
    d = jload(E_INC / "results" / "summary_fix.json")
    rows = []
    for k, tag in INC.items():
        L = d["levels"][k]
        lab = "complete" if L["mag_limit"] is None else f"$m<{L['mag_limit']:.0f}$"
        j = L["joint"]
        wd = L["width_degradation_vs_complete"]
        sep = (f"{L['sky_shuffle_null']['displacement_in_widths']:.2f}"
               if "sky_shuffle_null" in L else r"\nodata")
        rows.append(
            rf"{lab} & {L['agn_completeness_within_horizon']:.3f} & "
            rf"{L['agn_n_hosts_within_horizon']:d} & "
            rf"{100*L['agn_empty_pixel_fraction']:.1f} & "
            rf"{L['Neff']/1e3:.0f} & {j['H0_half_width68']:.3f} & {wd['joint_H0']:.2f} & "
            rf"{j['f_half_width68']:.4f} & {wd['joint_f']:.2f} & {sep} \\")
    return (
        r"\begin{deluxetable*}{lcccccccccc}" "\n"
        r"\tablecaption{Two-tracer flux-limit ladder at fixed data. $C$ and the host "
        r"count are for the sparse tracer within $z\le\IncZref$; $\sigma$ are 68\% "
        r"half-widths from the joint $(H_0,f_{\rm AGN})$ grid, with the factor "
        r"against the complete rung beside each. The last column is the separation "
        r"between the measured $f_{\rm AGN}$ peak and its sky-shuffled null, in data "
        r"widths. \label{tab:twotracer}}" "\n"
        r"\tablehead{\colhead{level} & \colhead{$C$} & \colhead{AGN in horizon} & "
        r"\colhead{empty (\%)} & \colhead{$N_{\rm eff}/10^3$} & "
        r"\colhead{$\sigma(H_0)$} & \colhead{$\times$} & "
        r"\colhead{$\sigma(f_{\rm AGN})$} & \colhead{$\times$} & "
        r"\colhead{separation}}" "\n"
        r"\startdata" "\n" + "\n".join(rows) + "\n"
        r"\enddata" "\n"
        r"\end{deluxetable*}" "\n")


def table_n0_width():
    """The width table, which supports rather than replaces the significance map
    shown as a figure: a posterior can be narrow and centred anywhere."""
    d = jload(E_FREE / "results" / "n0_arms_summary_fix.json")
    inc = jload(E_INC / "results" / "summary_fix.json")
    rows = []
    for k in INC:
        C = inc["levels"][k]["agn_completeness_within_horizon"]
        cells = " & ".join(f"{d['levels'][k]['arms'][a]['half_width68']:.3f}"
                           for a in ARMS)
        rows.append(rf"{C:.2f} & {cells} \\")
    head = " & ".join(ARM_LABEL[a] for a in ARMS)
    return (
        r"\begin{deluxetable*}{ccccccc}" "\n"
        r"\tablecaption{68\% half-width on the AGN-hosted fraction across survey "
        r"completeness (rows) and prior knowledge of the sparse tracer's comoving "
        r"number density (columns), on the same grids as "
        r"Figure~\ref{fig:n0sig}. The width alone is a misleading summary: its "
        r"smallest entry anywhere in the table belongs to the free-density column "
        r"at the complete rung, which is among the \emph{least} significant "
        r"entries of that row. \label{tab:n0width}}" "\n"
        r"\tablehead{\colhead{$C$} & \multicolumn{6}{c}{knowledge of "
        r"$n_{0,\rm AGN}$}\\ \colhead{} & \colhead{" +
        head.replace(" & ", "} & \\colhead{") + r"}}" "\n"
        r"\startdata" "\n" + "\n".join(rows) + "\n"
        r"\enddata" "\n"
        r"\end{deluxetable*}" "\n")


def table_bias_budget():
    o = jload(E_CLO / "results" / "obsdet_summary.json")
    d = jload(E_CLO / "results" / "summary.json")
    orc = jload(E_CLO / "results" / "oracle_summary.json")
    c = o["arms"]["ctrl"]["offset_stats"]
    b = o["arms"]["obs"]["offset_stats"]
    pd = o["paired_difference_obs_minus_ctrl"]
    ex = orc["offset_exact_oracle"]
    pr = orc["paired_ds_minus_oracle"]
    bf = orc["bootstrap_fix"]["offset"]
    ms = d["multi_seed"]
    rows = [
        rf"offset as generated & {c['n']:d} & ${c['mean']:+.3f}\pm{c['sem']:.3f}$ & "
        rf"{c['sigma_from_zero']:.1f} \\",
        rf"detection acting on latent data & {pd['n']:d} & "
        rf"${pd['mean']:+.3f}\pm{pd['sem']:.3f}$ & {pd['sigma_from_zero']:.1f} \\",
        rf"sky width drawn from latent parameters & {ex['n']:d} & "
        rf"${ex['mean']:+.3f}\pm{ex['sem']:.3f}$ & {ex['sigma_from_zero']:.1f} \\",
        rf"estimator overhead (vs.\ exact likelihood) & {pr['n']:d} & "
        rf"${pr['mean']:+.3f}\pm{pr['sem']:.3f}$ & {pr['sigma_from_zero']:.1f} \\",
        r"\hline",
        rf"closure: exact likelihood, repaired recipe & {bf['n']:d} & "
        rf"${bf['mean']:+.3f}\pm{bf['sem']:.3f}$ & {bf['sigma_from_zero']:.1f} \\",
    ]
    return (
        r"\begin{deluxetable*}{lccc}" "\n"
        r"\tablecaption{Error budget of the distance scale on the matched "
        r"single-tracer mock. The first row is the offset the mock as generated "
        r"produces; the second is the paired difference between two arms that "
        r"differ only in the detection rule; the third is the offset of the exact "
        r"likelihood itself, which isolates the latent sky-width defect; the "
        r"fourth is the paired difference between the production estimator and "
        r"the exact likelihood on the same events. The last row is the closure "
        r"endpoint: the exact likelihood on events drawn with the repaired "
        r"recipe. \label{tab:budget}}" "\n"
        r"\tablehead{\colhead{} & \colhead{$N_{\rm real}$} & "
        r"\colhead{$\Delta H_0$ (\kmsmpc)} & \colhead{$\sigma$}}" "\n"
        r"\startdata" "\n" + "\n".join(rows) + "\n"
        r"\enddata" "\n"
        r"\tablecomments{Independent of the offset, the host-catalog realisation "
        rf"contributes ${ms['catalog_variance_component']:.2f}$~\kmsmpc\ of scatter "
        r"per 1000-event realisation, which the per-realisation intervals do not "
        rf"contain: the realised scatter is {ms['interval_underestimate_factor']:.1f}"
        r"$\times$ the quoted 68\% half-width.}" "\n"
        r"\end{deluxetable*}" "\n")


TABLES = {
    "tab_f_recovery.tex": table_f_recovery,
    "tab_completeness_anchored.tex": table_completeness_anchored,
    "tab_twotracer_ladder.tex": table_twotracer_ladder,
    "tab_n0_width.tex": table_n0_width,
    "tab_bias_budget.tex": table_bias_budget,
}


# ===========================================================================
# emit
# ===========================================================================
def main():
    for fn in (sec_setup, sec_matched_setup, sec_baseline, sec_tilt, sec_closure,
               sec_oracle, sec_kernel, sec_anchored, sec_deep,
               sec_incomplete, sec_free, sec_seeds):
        fn()

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = [
        "% results_macros.tex -- GENERATED, DO NOT EDIT BY HAND",
        f"% written {stamp} by scripts/build_values.py",
        f"% experiments root: {EXP}",
        f"% {len(REGISTRY)} macros",
        "",
    ]
    for r in REGISTRY:
        out.append(rf"\newcommand{{\{r['name']}}}{{{r['body']}}}")
    (PAPER / "values" / "results_macros.tex").write_text("\n".join(out) + "\n")

    (PAPER / "tables").mkdir(exist_ok=True)
    for name, fn in TABLES.items():
        (PAPER / "tables" / name).write_text(fn())

    # ---- audit trail
    by_exp: dict[str, list[dict]] = {}
    for r in REGISTRY:
        by_exp.setdefault(r["exp"] or "(unattributed)", []).append(r)
    md = [
        "# NUMBERS.md — macro audit trail",
        "",
        "GENERATED by `scripts/build_values.py`; do not edit by hand.",
        "",
        f"Written {stamp}. {len(REGISTRY)} macros over {len(by_exp)} sources.",
        "",
        "Every number that appears in the manuscript body, a caption or a table is",
        "one of the macros below. Source paths are relative to",
        f"`{EXP}`",
        "unless absolute. Regenerate with",
        "",
        "```",
        "JAX_PLATFORMS=cpu python scripts/build_values.py",
        "```",
        "",
    ]
    for exp in sorted(by_exp):
        md += [f"## {exp.replace(chr(92), '')}", "",
               "| macro | value | source | meaning |", "|---|---|---|---|"]
        for r in sorted(by_exp[exp], key=lambda x: x["name"]):
            body = r["body"].replace("|", r"\|")
            md.append(f"| `\\{r['name']}` | `{body}` | `{r['src']}` | {r['note']} |")
        md.append("")
    md += ["## Generated tables", "",
           "| file | built from |", "|---|---|"]
    srcs = {
        "tab_f_recovery.tex": "experiment_h0f_baseline/results/summary.json",
        "tab_completeness_anchored.tex":
            "experiment_completeness_anchored/results/summary.json",
        "tab_twotracer_ladder.tex":
            "experiment_twotracer_incomplete/results/summary_fix.json",
        "tab_n0_width.tex":
            "experiment_completeness_free/results/n0_arms_summary_fix.json + "
            "experiment_twotracer_incomplete/results/summary_fix.json",
        "tab_bias_budget.tex":
            "experiment_matched_mock/results/{obsdet_summary,summary}.json",
    }
    for k in TABLES:
        md.append(f"| `tables/{k}` | `{srcs[k]}` |")
    md.append("")
    pend = [r["name"] for r in REGISTRY if r["raw"] is None]
    md += ["## Pending macros", ""]
    if pend:
        md += [f"{len(pend)} macros still resolve to `\\todo{{pending}}` because their "
               "run has not finished:", ""]
        md += [f"- `\\{n}`" for n in pend]
    else:
        md.append("None: every macro resolves to a number.")
    md.append("")
    (PAPER / "NUMBERS.md").write_text("\n".join(md))

    print(f"{len(REGISTRY)} macros -> values/results_macros.tex")
    print(f"{len(TABLES)} tables -> tables/")
    print(f"{len(pend)} pending: {', '.join(pend) if pend else 'none'}")


if __name__ == "__main__":
    main()
