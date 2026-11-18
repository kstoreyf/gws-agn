#!/usr/bin/env python
"""The assumed AGN number density with flux-limited catalogs (Sec. IV C, `sec:flux`; Fig. 3(b)): f_AGN and H0 when the
AGN number density is assumed at half and at twice the value from the catalog counts. Analysis 16's stage-2 runs

    known_population/both_catalogs/flux_limit_<m>_densities_from_counts_agn_density_<half|double>/equal_populations/realization_<R>

(m = 20, 18; the galaxy density from its counts), each against its unit-factor partner of the same realization, the run of
key Flux<limit> (known_population/both_catalogs/flux_limit_<m>_densities_from_counts/equal_populations/realization_<R>).

    PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
    $PY working/paper_prd/scripts/density.py          # every macro: real or red, value

Used by build_values.py (the macros), production.py (--write-map writes the runs into values/production_map.json under
"density_response"; --status, --report), make_figures.py (Fig. 3(b)) and audit_values.py (L checks).

Macros (L = Twenty for m < 20, Eighteen for m < 18; X = Half or Double, the AGN density assumed at 0.5 or 2 times the
value from the catalog counts; claim 'density'):
  Flux<L><X><fAGN|Hzero>            realization A, median and 90% equal-tailed interval (the run's summary.json, block
                                    posterior); pieces Med, Lo, Hi, Wid as in the key scheme
  Flux<L><X>Delta<fAGN|Hzero>       realization A: the median at factor X minus the median of the unit-factor run of the same
                                    realization (key Flux<L>, its summary.json); signed
  Flux<L><X>Delta<fAGN|Hzero>Mean   the mean of the same shift over the realizations run at factor X, each against the
                                    unit-factor run of its own realization; ...Min, ...Max the smallest and largest (signed)
  Flux<L><X>Delta<fAGN|Hzero>Pred   the shift the normalization identity predicts (notes/LIKELIHOOD_MATH.md Sec. 8) from the
                                    unit-factor grid of realization A: ln L_x(H0, f) = ln L_1(H0, f'),
                                    logit f' = logit f + ln[F_AGN(x)/F_AGN(1)], F_AGN(x) = N_AGN/(N_AGN + x n_AGN V_miss,AGN);
                                    the median of the mapped posterior at x minus that at x = 1 (the same lattice). It maps
                                    the normaliser only: the hosts the model adds where the catalog is incomplete AND a
                                    merger or a detected injection lies also scale with the assumed density, and the identity
                                    leaves that part out (exact when no merger lies where the catalog is incomplete)
  NormFAGN<L>Counts                 F_AGN(1) of that map: N_AGN = the cataloged AGN of realization A's survey file (sum of
                                    ngals), n_AGN = the unit-factor run's density from the counts (summary densities.agn
                                    log10n_used, h^3 Mpc^-3), V_miss,AGN = 4 pi int_0^1.10 [1 - C_AGN(z)] dV_c/(dz dOmega) dz
                                    at H0 = 100 (h^-3 Mpc^3) with the run's AGN selection fit; n V does not depend on H0
  NrealFluxDensity                  the realizations run at each factor (a word)

A group (limit, factor) is real (black) when every summary.json it names exists: the factor-X runs of its realizations
and their unit-factor partners; the prediction and the share when the unit-factor run of realization A exists.
Validation and check stages of the run list (stage names starting with 'V' or 'X', build_values.production_stage) never
feed these macros.

The densities in Mpc^-3 (a check of stage X, read from RUNLIST.yaml by name like scripts/skyres.py, claim 'validation';
never a production run of any key): the m < 18 grid of realization A with both densities given in Mpc^-3 at the values
from the counts converted at H0 = 67.74 (--density given --density-units physical), the AGN density at that value and at
twice it,

    known_population/both_catalogs/flux_limit_18_densities_given/equal_populations/realization_100_units_physical
    known_population/both_catalogs/flux_limit_18_densities_given_agn_density_double/equal_populations/realization_100_units_physical

  FluxEighteenPhys<fAGN|Hzero>        the unit-factor run (Mpc^-3): median and 90% equal-tailed interval (summary.json, block
                                      posterior); pieces Med, Lo, Hi, Wid, HalfWid (the 90% width and half of it)
  FluxEighteenPhysDouble<fAGN|Hzero>  the same for the run with the AGN density doubled
  FluxEighteenPhysDoubleDelta<fAGN|Hzero>  median of the doubled run minus that of the unit-factor run (Mpc^-3), signed; the
                                      production pair (h^3 Mpc^-3) is FluxEighteenDoubleDelta<fAGN|Hzero>
Black when both summaries exist.
"""
from __future__ import annotations

import json
import math
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import build_values as bv  # noqa: E402

PRD = HERE.parent
PMAP = PRD / "values" / "production_map.json"
A16 = bv.AN / "analysis_16_production_runs"
RUNS = A16 / "runs"
RUNLIST = RUNS / "RUNLIST.yaml"
REAL_A = 100
LIMITS = [("m20", "Twenty", 20), ("m18", "Eighteen", 18)]
FACTORS = [(0.5, "Half", "half"), (2.0, "Double", "double")]
UNIT = "known_population/both_catalogs/flux_limit_{m}_densities_from_counts/equal_populations/realization_{{R}}"
ARM = "known_population/both_catalogs/flux_limit_{m}_densities_from_counts_agn_density_{v}/equal_populations/realization_{{R}}"
# the factors at which the identity's curve of Fig. 3(b) is evaluated (0.5, 1 and 2 exactly among them)
CURVE_X = sorted({round(float(v), 6) for v in np.geomspace(0.45, 2.2, 31)} | {0.5, 1.0, 2.0})
QW = {"fAGN": ("f_agn", "f_AGN (fraction of mergers hosted by AGN)", ""), "Hzero": ("H0", "H0", "km/s/Mpc")}
DIG = {"fAGN": 3, "Hzero": 2}          # decimals of the shifts (as NormDelta<f|Hzero>Two...)


def arm_word(word, xw):
    return f"{word}{xw}"


def limit_desc(mlim):
    return f"population known, both catalogs, m < {mlim}, hosts normalized to z = {bv.ZMAX_PROD:g}"


# ============================================================================================= #
# the run list -> the map block (production.py --write-map writes it into values/production_map.json)
# ============================================================================================= #
def _runlist_index():
    """run name -> (stage, args as a dict), RUNLIST.yaml, every stage except the validation and check stages (names starting
    with 'V' or 'X', bv.production_stage)."""
    import yaml
    out = {}
    if not RUNLIST.exists():
        return out
    d = yaml.safe_load(RUNLIST.read_text())
    for st in d["stages"]:
        if not bv.production_stage(st["stage"]):
            continue
        for r in st["runs"]:
            a = r["args"]
            kv = {a[i].lstrip("-"): a[i + 1] for i in range(len(a) - 1) if a[i].startswith("--") and not a[i + 1].startswith("--")}
            out[r["name"]] = (str(st["stage"]), kv)
    return out


def map_block(_runs=None):
    """{"arms": {<L><X>: {run, unit_run, unit_key, flux, factor, runlist, stage, realizations, realization_A}}, "not_in_run_lists"}."""
    idx = _runlist_index()
    arms, missing = {}, {}
    for rk, word, mlim in LIMITS:
        unit = UNIT.format(m=mlim)
        for x, xw, v in FACTORS:
            pat = ARM.format(m=mlim, v=v)
            rx = re.compile("^" + re.escape(pat).replace(re.escape("{R}"), r"(\d{3})") + "$")
            recs = {int(m.group(1)): idx[n] for n in idx if (m := rx.match(n))}
            recs = {r: s for r, s in recs.items() if abs(float(s[1].get("agn-density-factor", "nan")) - x) < 1e-12 and s[1].get("density") == "counts"}
            name = arm_word(word, xw)
            if not recs:
                missing[name] = f"no run named {pat} (--agn-density-factor {x:g}) in {RUNLIST.name} outside the validation stages"
                continue
            reals = sorted(recs)
            nounit = [r for r in reals if unit.format(R=f"{r:03d}") not in idx]
            if nounit:
                missing[name] = f"the unit-factor runs {unit} of realizations {nounit} are not in {RUNLIST.name}"
                continue
            arms[name] = {"run": pat, "unit_run": unit, "unit_key": f"Flux{word}", "flux": str(mlim), "factor": x, "runlist": RUNLIST.name,
                          "stage": "/".join(sorted({recs[r][0] for r in reals})), "realizations": reals,
                          "realization_A": REAL_A if REAL_A in reals else reals[0]}
    return {"_what": "the runs of the density-response macros (scripts/density.py): factor-X runs and their unit-factor partners "
                     "(the runs of key Flux<limit>) of the same realizations; validation stages excluded",
            "arms": arms, "not_in_run_lists": missing, "units_physical": phys_block()}


# ============================================================================================= #
# the densities in Mpc^-3 (stage X of RUNLIST.yaml, read by name; never a production run)
# ============================================================================================= #
PHYS_LIMIT = ("m18", "Eighteen", 18)
PHYS = {"": ("known_population/both_catalogs/flux_limit_{m}_densities_given/equal_populations/realization_{{R}}_units_physical", 1.0),
        "Double": ("known_population/both_catalogs/flux_limit_{m}_densities_given_agn_density_double/equal_populations/realization_{{R}}_units_physical", 2.0)}


def phys_block():
    """{"runs": {"" | "Double": {run, stage, factor, log10n_Mpc-3}}, "realization", "flux", "partners"} from RUNLIST.yaml by name: both runs in a
    check stage (X) with --density given --density-units physical, the same given densities, the AGN factor 1 and 2; or {"runs": None, "why"}."""
    import yaml
    if not RUNLIST.exists():
        return {"runs": None, "why": f"no {RUNLIST.name}"}
    d = yaml.safe_load(RUNLIST.read_text())
    idx = {r["name"]: (str(st["stage"]), r["args"]) for st in d["stages"] for r in st["runs"]}
    _, word, mlim = PHYS_LIMIT
    why, recs = [], {}
    for tag, (pat, x) in PHYS.items():
        p = pat.format(m=mlim)
        name = p.format(R=f"{REAL_A:03d}")
        if name not in idx:
            why.append(f"no run {name} in {RUNLIST.name}")
            continue
        st, a = idx[name]
        kv = {a[i].lstrip("-"): a[i + 1] for i in range(len(a) - 1) if a[i].startswith("--") and not a[i + 1].startswith("--")}
        if not (st.startswith("X") and kv.get("density") == "given" and kv.get("density-units") == "physical"
                and abs(float(kv.get("agn-density-factor", "1")) - x) < 1e-12 and kv.get("flux") == str(mlim)):
            why.append(f"{name}: stage {st}, args {kv} (expected stage X, --density given --density-units physical, factor {x:g})")
            continue
        recs[tag] = {"run": p, "stage": st, "factor": x, "log10n_Mpc-3": {"gal": float(kv["log10n-gal"]), "agn": float(kv["log10n-agn"])}}
    if not why and recs[""]["log10n_Mpc-3"] != recs["Double"]["log10n_Mpc-3"]:
        why.append("the two runs give different densities")
    if why:
        return {"runs": None, "why": "; ".join(why)}
    return {"_what": f"the m < {mlim} grid of realization A with both densities given in Mpc^-3 (stage X, a check; scripts/density.py "
                     f"FluxEighteenPhys...), the AGN density at the value from the counts and at twice it",
            "runs": recs, "realization": REAL_A, "flux": str(mlim),
            "partners": {"": UNIT.format(m=mlim), "Double": ARM.format(m=mlim, v="double")}}


@lru_cache(None)
def mapped():
    m = json.loads(PMAP.read_text()) if PMAP.exists() else {}
    blk = m.get("density_response")
    blk = dict(blk) if blk else map_block()
    if "units_physical" not in blk:             # a map written before 18 Nov: the run list by name
        blk["units_physical"] = phys_block()
    return blk


def phys_evaluate():
    """{"real", "missing", "rec", "runs": {tag: posterior}, "shift": {q}, "files"}."""
    rec = mapped()["units_physical"]
    if not rec.get("runs"):
        return {"real": False, "missing": [rec.get("why", "not in the map")], "rec": None}
    paths = {t: run_dir(r["run"], rec["realization"]) / "summary.json" for t, r in rec["runs"].items()}
    miss = [bv.rel(p) for p in paths.values() if not p.exists()]
    out = {"real": not miss, "missing": miss, "rec": rec, "files": [bv.rel(p) for p in paths.values()]}
    if not miss:
        S = {t: _jr(p) for t, p in paths.items()}
        for t, s in S.items():
            da = s["densities"]
            x = rec["runs"][t]["factor"]
            assert all(da[c]["units"] == "physical" and da[c]["mode"] == "given" for c in ("gal", "agn")), (t, da)
            assert abs(da["agn"]["factor"] - x) < 1e-12 and abs(da["gal"]["factor"] - 1.0) < 1e-12, (t, da)
        assert abs(S["Double"]["densities"]["agn"]["log10n_used"] - S[""]["densities"]["agn"]["log10n_used"] - math.log10(2.0)) < 1e-9
        for k in ("events_file", "injections_file", "nsamp", "half_variance_correction", "max_var", "zmax", "core", "realization"):
            assert S["Double"].get(k) == S[""].get(k), (k, S["Double"].get(k), S[""].get(k))
        out["runs"] = {t: s["posterior"] for t, s in S.items()}
        out["shift"] = {q: S["Double"]["posterior"][par]["median"] - S[""]["posterior"][par]["median"] for q, (par, _, _) in QW.items()}
    return out


def run_dir(pat, r):
    return RUNS / pat.format(R=f"{r:03d}")


def _jr(p):
    return json.loads(Path(p).read_text())


def needed(rec):
    """summary.json paths of an arm: its runs and their unit-factor partners, every realization."""
    return [run_dir(rec["run"], r) / "summary.json" for r in rec["realizations"]] + \
           [run_dir(rec["unit_run"], r) / "summary.json" for r in rec["realizations"]]


def status():
    """{arm name: (real, missing paths)}; arms not in the run list: (False, [why])."""
    blk = mapped()
    out = {}
    for rk, word, mlim in LIMITS:
        for x, xw, v in FACTORS:
            n = arm_word(word, xw)
            rec = blk["arms"].get(n)
            if rec is None:
                out[n] = (False, [blk["not_in_run_lists"].get(n, "not in the map")])
                continue
            miss = [bv.rel(p) for p in needed(rec) if not p.exists()]
            out[n] = (not miss, miss)
    return out


# ============================================================================================= #
# the share F_AGN and the identity
# ============================================================================================= #
def share(rk, unit_dir, r):
    """F_AGN(1) = N/(N + n V_miss) of the unit-factor run `unit_dir` (realization r), and its parts."""
    sm, cfg = _jr(unit_dir / "summary.json"), _jr(unit_dir / "config.json")
    NC = bv.norm_cfg(r)[rk]
    out = {}
    for cls in ("gal", "agn"):
        d = sm["densities"][cls]
        assert d["units"] == "h_scaled" and abs(d["factor"] - 1.0) < 1e-12, (unit_dir, d)
        rec = NC[cls]
        assert Path(cfg["surveys"][cls]).resolve() == (bv.W / rec["survey"]).resolve(), (cfg["surveys"][cls], rec["survey"])
        assert Path(cfg["selection_fits"][cls]["path"]).resolve() == (bv.W / rec["fit"]["file"]).resolve(), (cfg["selection_fits"][cls], rec["fit"])
        nV = 10 ** d["log10n_used"] * rec["Vp"]
        out[cls] = {"N": int(rec["N"]), "log10n_h3Mpc-3": float(d["log10n_used"]), "V_miss_h-3Mpc3": float(rec["Vp"]), "nV": float(nV),
                    "F": float(rec["N"] / (rec["N"] + nV)), "survey": rec["survey"], "fit": rec["fit"]["file"]}
    return out


def logit_shift(F, x):
    """ln[F_AGN(x)/F_AGN(1)] = -ln[F + x (1 - F)] (the AGN density assumed x times the unit-factor one)."""
    return -math.log(F + x * (1.0 - F))


@lru_cache(None)
def _unit_lattice(unit_dir, nf=4001):
    """Corrected ln L of a unit-factor grid run (run_posteriors: cubic spline of the fine pass, linear in the coarse pass) on
    H0 (the fine pass widened by 30%, 401 points) x f' in [0, 1] (nf points): (H, fp, L normalised to its maximum)."""
    import run_posteriors as RP
    parts = RP._grid_parts(Path(unit_dir))
    names = parts[0][0]
    assert names == ["H0", "f_agn"], names
    _, axes = RP._common_axes([parts])
    H = axes[0]
    fp = np.linspace(0.0, 1.0, nf)
    mesh = np.stack(np.meshgrid(H, fp, indexing="ij"), -1).reshape(-1, 2)
    lnL = RP._eval_parts_spline(parts, mesh).reshape(len(H), nf)
    L = np.exp(lnL - np.max(lnL[np.isfinite(lnL)]))
    L[~np.isfinite(L)] = 0.0
    return H, fp, L


def identity(unit_dir, F, xs):
    """The unit-factor posterior mapped through the identity at each factor x: {x: {"f": summary, "H0": summary, "dlogit"}}.
    Flat priors in (H0, f): p_x(f) = m_1(f'(f)) with m_1 the f' marginal of L_1; p_x(H0) = int L_1(H0, f') (df/df') df'."""
    import placeholder_results as phr
    H, fp, L = _unit_lattice(str(unit_dir))
    m1 = np.trapz(L, H, axis=0)
    f = fp
    out = {}
    for x in xs:
        d = logit_shift(F, x)
        with np.errstate(divide="ignore", over="ignore"):
            g = bv.expit(bv.logit(f) + d)               # f' of each f
            finv = bv.expit(bv.logit(fp) - d)           # f of each f'
        g[0], g[-1] = 0.0, 1.0
        finv[0], finv[-1] = 0.0, 1.0
        pf = np.interp(g, fp, m1)
        jac = np.empty_like(fp)
        inner = (fp > 0) & (fp < 1)
        jac[inner] = finv[inner] * (1 - finv[inner]) / (fp[inner] * (1 - fp[inner]))
        jac[0], jac[-1] = math.exp(-d), math.exp(d)
        pH = np.trapz(L * jac[None, :], fp, axis=1)
        sf = phr.summarize_xp(f, pf / np.trapz(pf, f))
        sH = phr.summarize_xp(H, pH / np.trapz(pH, H))
        out[float(x)] = {"dlogit": d, "f": {k: sf[k] for k in ("median", "ci90")}, "H0": {k: sH[k] for k in ("median", "ci90")}}
    return out


def a15_complete_curve(xs, nf=4001):
    """Analysis 15's complete catalogs normalized to z = 5 (scan p1_complete), the identity's curve for Fig. 3(b): its zero-density
    reference grid (ref_complete_n0m24) mapped row by row in H0 by Analysis 15's own map (aggregate_normalisation.predict: cubic spline
    of ln L in logit f, logit f' = logit f + ln(F_A,x/F_G) - ln(F_A,ref/F_G,ref)), with Z_AGN(x; H0) = N_AGN + x [Z_AGN(H0) - N_AGN]
    from p1_complete's normaliser rows (Z is linear in the density), evaluated on a lattice of nf points in f (instead of the scan's
    0.025 cells) and summarized by a15_common.summarize_2d. -> {"x", "median", "files"} (None when a scan file is missing)."""
    if str(bv.A15_COMMON.parent) not in sys.path:
        sys.path.insert(0, str(bv.A15_COMMON.parent))
    dwb, sys.dont_write_bytecode = sys.dont_write_bytecode, True        # read Analysis 15's code, write nothing into its directory
    try:
        import a15_common as C  # noqa: E402
        import aggregate_normalisation as AG  # noqa: E402
    finally:
        sys.dont_write_bytecode = dwb
    res = bv.A15_SUMMARY.parent
    ref, s1 = AG.load(res, "ref_complete_n0m24"), AG.load(res, "p1_complete")
    if ref is None or s1 is None:
        return None
    obs, Z1 = s1["obs"], s1["Z"]
    fs = np.linspace(0.0, 1.0, nf)
    med = []
    for x in xs:
        Zx = Z1.copy()
        Zx[:, 1] = obs[:, 1] + x * (Z1[:, 1] - obs[:, 1])
        s = dict(s1, lnF=np.log(obs / Zx), f_grid=fs, lnL=np.zeros((len(s1["H0_grid"]), nf)))
        with np.errstate(divide="ignore", invalid="ignore"):
            pred, _, _ = AG.predict(s, ref)
        sm = C.summarize_2d(s1["H0_grid"], fs, np.where(np.isfinite(pred), pred, -np.inf))
        med.append(float(sm["f"]["median"]))
    return {"x": [float(v) for v in xs], "median": med,
            "files": [bv.rel(res / "scans" / f"{n}.h5") for n in ("ref_complete_n0m24", "p1_complete")]}


# ============================================================================================= #
# the macros
# ============================================================================================= #
def evaluate():
    """Every number of the module: {limit word: {"share", "pred", "arms": {xw: {...}}}} (None where a run is missing)."""
    blk = mapped()
    st = status()
    out = {}
    for rk, word, mlim in LIMITS:
        L = {"rk": rk, "mlim": mlim, "share": None, "pred": None, "arms": {}}
        unitA = run_dir(UNIT.format(m=mlim), REAL_A)
        if (unitA / "summary.json").exists():
            S = share(rk, unitA, REAL_A)
            L["share"] = S
            L["pred"] = identity(unitA, S["agn"]["F"], CURVE_X)
            L["unit_A"] = _jr(unitA / "summary.json")["posterior"]
        for x, xw, v in FACTORS:
            n = arm_word(word, xw)
            rec = blk["arms"].get(n)
            real, miss = st[n]
            E = {"real": real, "missing": miss, "rec": rec, "factor": x}
            if real:
                per = {}
                for r in rec["realizations"]:
                    sa, su = _jr(run_dir(rec["run"], r) / "summary.json"), _jr(run_dir(rec["unit_run"], r) / "summary.json")
                    da, du = sa["densities"], su["densities"]
                    assert abs(da["agn"]["factor"] - x) < 1e-12 and abs(da["gal"]["factor"] - 1.0) < 1e-12, (n, r, da)
                    assert abs(da["agn"]["log10n_used"] - du["agn"]["log10n_used"] - math.log10(x)) < 1e-9, (n, r)
                    assert abs(da["gal"]["log10n_used"] - du["gal"]["log10n_used"]) < 1e-12, (n, r)
                    assert sa["events_file"] == su["events_file"] and sa["injections_file"] == su["injections_file"], (n, r)
                    per[r] = {"arm": sa["posterior"], "unit": su["posterior"],
                              "shift": {q: sa["posterior"][QW[q][0]]["median"] - su["posterior"][QW[q][0]]["median"] for q in QW}}
                E["per"] = per
                E["A"] = per[rec["realization_A"]]
                E["files"] = [bv.rel(p) for p in needed(rec)]
            L["arms"][xw] = E
        out[word] = L
    return out


EXPECTED = {"fAGN": r"\ph{0.XX^{+0.XX}_{-0.XX}}", "Hzero": r"\ph{XX.X^{+X.X}_{-X.X}}"}


def build(add, add_ph, fnum, fci, fsig, NUM_WORD):
    """Register the macros with build_values' registry (add, add_ph)."""
    ev = evaluate()
    blk = mapped()
    reals_all = sorted({tuple(rec["realizations"]) for rec in blk["arms"].values()})
    for rk, word, mlim in LIMITS:
        L = ev[word]
        desc = limit_desc(mlim)
        # the share F_AGN of the identity
        nmF = f"NormFAGN{word}Counts"
        mF = (f"cataloged share F_AGN = N_AGN/(N_AGN + n_AGN V_miss,AGN) of the m < {mlim} AGN catalog of realization A, hosts normalized to "
              f"z = {bv.ZMAX_PROD:g}, n_AGN the AGN density from the catalog counts (unit-factor run of key Flux{word}, h^3 Mpc^-3), V_miss,AGN at "
              f"H0 = 100 (h^-3 Mpc^3), so independent of H0: the share in the identity behind Flux{word}<Half|Double>DeltafAGNPred")
        if L["share"] is not None:
            S = L["share"]["agn"]
            add(nmF, "plain", fsig(S["F"], 2), S["F"], "", mF,
                {"files": [bv.rel(run_dir(UNIT.format(m=mlim), REAL_A) / "summary.json"), S["survey"], S["fit"]],
                 "computation": "N = sum of ngals of the survey file; n = 10^densities.agn.log10n_used; V = build_values.vmiss100(fit, 1.10)"},
                "density", extra={"agn": L["share"]["agn"], "gal": L["share"]["gal"]})
        else:
            add_ph(nmF, r"\ph{0.0X}", "", mF, f"the unit-factor run of key Flux{word}, realization {REAL_A}", "density")
        for x, xw, v in FACTORS:
            E = L["arms"][xw]
            rec = E["rec"]
            key = f"Flux{word}{xw}"
            fac = f"the AGN number density assumed at {x:g} times the value from the catalog counts"
            nreal = len(rec["realizations"]) if rec else 0
            src = {"files": E["files"]} if E["real"] else {"missing": "; ".join(E["missing"][:4]) + (" ..." if len(E["missing"]) > 4 else ""),
                                                          "placeholder_source": "none (template)"}
            run_s = f" (runs {rec['run']}, stage {rec['stage']} of {rec['runlist']})" if rec else ""
            for q, (par, qd, units) in QW.items():
                mean = f"{qd}, {desc}, {fac}{run_s}, realization A"
                if E["real"]:
                    s = E["A"]["arm"][par]
                    r_, d = fci(s["median"], s["ci90"][0], s["ci90"][1])
                    val = {"median": s["median"], "ci90": list(s["ci90"])}
                    srcA = {"file": bv.rel(run_dir(rec["run"], rec["realization_A"]) / "summary.json"), "key": f"posterior.{par}"}
                    add(f"{key}{q}", "plain", r_, val, units, mean + " (median and 90% equal-tailed interval)", srcA, "density", extra={"decimals": d})
                    for pc, vv, w in (("Med", s["median"], "median"), ("Lo", s["ci90"][0], "lower end of the 90% interval"), ("Hi", s["ci90"][1], "upper end of the 90% interval"),
                                      ("Wid", s["ci90"][1] - s["ci90"][0], "full width of the 90% interval")):
                        add(f"{key}{q}{pc}", "plain", fnum(vv, d), vv, units, f"{w} of {mean}", srcA, "density")
                else:
                    add_ph(f"{key}{q}", EXPECTED[q], units, mean + " (median and 90% equal-tailed interval)", src["missing"], "density")
                    for pc, w in (("Med", "median"), ("Lo", "lower end of the 90% interval"), ("Hi", "upper end of the 90% interval"), ("Wid", "full width of the 90% interval")):
                        add_ph(f"{key}{q}{pc}", r"\ph{X.XX}", units, f"{w} of {mean}", src["missing"], "density")
                # the shifts against the unit-factor runs
                dg = DIG[q]
                shift_of = f"shift of the {qd} median when {fac}: median of the run at factor {x:g} minus that of the unit-factor run (key Flux{word}) of the same realization, {desc}"
                if E["real"]:
                    per = {r: E["per"][r]["shift"][q] for r in rec["realizations"]}
                    vA = per[rec["realization_A"]]
                    mn = float(np.mean(list(per.values())))
                    lo_, hi_ = min(per.values()), max(per.values())
                    ex = {"per_realization": {str(r): per[r] for r in per}, "unit_median": {str(r): E["per"][r]["unit"][par]["median"] for r in per},
                          "arm_median": {str(r): E["per"][r]["arm"][par]["median"] for r in per}}
                    add(f"{key}Delta{q}", "plain", fnum(vA, dg), vA, units, f"{shift_of}; realization A (signed; set in math mode)", src, "density", extra=ex)
                    add(f"{key}Delta{q}Mean", "plain", fnum(mn, dg), mn, units, f"{shift_of}; mean over the {nreal} realizations {rec['realizations']} (signed)", src, "density", extra=ex)
                    add(f"{key}Delta{q}Min", "plain", fnum(lo_, dg), lo_, units, f"{shift_of}; the smallest (signed) over the {nreal} realizations {rec['realizations']}", src, "density", extra=ex)
                    add(f"{key}Delta{q}Max", "plain", fnum(hi_, dg), hi_, units, f"{shift_of}; the largest (signed) over the {nreal} realizations {rec['realizations']}", src, "density", extra=ex)
                else:
                    for suf, w in (("", "realization A"), ("Mean", "mean over the realizations"), ("Min", "smallest over the realizations"), ("Max", "largest over the realizations")):
                        add_ph(f"{key}Delta{q}{suf}", r"\ph{0.XX}", units, f"{shift_of}; {w}", src["missing"], "density")
                # the identity's prediction (realization A, the unit-factor grid)
                pm = (f"shift of the {qd} median predicted by the normalization identity (LIKELIHOOD_MATH Sec. 8) when {fac}: the corrected ln L of the "
                      f"unit-factor grid of realization A (key Flux{word}) mapped as ln L_x(H0, f) = ln L_1(H0, f'), logit f' = logit f + "
                      f"ln[F_AGN(x)/F_AGN(1)] = logit f - ln[F + x(1 - F)], F = NormFAGN{word}Counts; median at x = {x:g} minus median at x = 1 on the same "
                      f"lattice; the normaliser part only (the uncataloged hosts where mergers and detected injections lie also scale with the assumed "
                      f"density and are not in the map), {desc}")
                if L["pred"] is not None:
                    P = L["pred"]
                    pv = P[x][par if par == "H0" else "f"]["median"] - P[1.0][par if par == "H0" else "f"]["median"]
                    simple = None
                    if q == "fAGN":
                        f1 = L["unit_A"]["f_agn"]["median"]
                        simple = float(bv.expit(bv.logit(f1) - P[x]["dlogit"]) - f1)
                    ex = {"F_AGN": L["share"]["agn"]["F"], "dlogit": P[x]["dlogit"], "predicted_at_x": P[x], "predicted_at_1": P[1.0],
                          "median_map_only": simple,
                          "curve": {"x": CURVE_X, "shift": [P[xx][par if par == "H0" else "f"]["median"] - P[1.0][par if par == "H0" else "f"]["median"] for xx in CURVE_X],
                                    "what": "predicted median at x minus at 1 (Fig. 3(b) draws the unit-factor median of realization A plus this)"}}
                    add(f"{key}Delta{q}Pred", "plain", fnum(pv, dg), pv, units, pm,
                        {"files": [bv.rel(run_dir(UNIT.format(m=mlim), REAL_A) / "summary.json"), bv.rel(run_dir(UNIT.format(m=mlim), REAL_A) / "grid")],
                         "computation": "density.identity: lattice 401 (H0) x 4001 (f in [0, 1]); p_x(f) = m_1(f'), p_x(H0) = int L_1 (df/df') df'"},
                        "density", extra=ex)
                else:
                    add_ph(f"{key}Delta{q}Pred", r"\ph{0.XX}", units, pm, f"the unit-factor run of key Flux{word}, realization {REAL_A}", "density")
    build_phys(add, add_ph, fnum, fci)
    if len(reals_all) == 1:
        N = len(reals_all[0])
        add("NrealFluxDensity", "plain", NUM_WORD.get(N, str(N)), N, "realizations",
            f"realizations run at each assumed AGN density (0.5 and 2 times the counts), m < 20 and m < 18, realizations {list(reals_all[0])}",
            {"file": bv.rel(RUNLIST), "computation": "realizations of the runs of values/production_map.json 'density_response'"}, "design,realizations",
            extra={"realizations": list(reals_all[0])})
    else:
        add_ph("NrealFluxDensity", r"\ph{three}", "realizations", "realizations run at each assumed AGN density (the arms differ or are missing: "
               f"{reals_all})", "the density-response runs of RUNLIST.yaml", "design,realizations")


def build_phys(add, add_ph, fnum, fci):
    """The macros of the two stage-X runs with the densities given in Mpc^-3 (claim 'validation')."""
    E = phys_evaluate()
    _, word, mlim = PHYS_LIMIT
    rec = E["rec"]
    desc = f"population known, both catalogs, m < {mlim}, realization A, both number densities given in Mpc^-3 (the counts' values converted at H0 = 67.74)"
    src_miss = "; ".join(E["missing"]) if not E["real"] else None
    for tag, x in (("", 1.0), ("Double", 2.0)):
        fac = "the AGN density at that value" if x == 1.0 else "the AGN density at twice that value"
        run_s = f" (run {rec['runs'][tag]['run'].format(R=f'{REAL_A:03d}')}, stage X of RUNLIST.yaml)" if rec else ""
        for q, (par, qd, units) in QW.items():
            key = f"Flux{word}Phys{tag}{q}"
            mean = f"{qd}, {desc}, {fac}{run_s}"
            if E["real"]:
                s = E["runs"][tag][par]
                r_, d = fci(s["median"], s["ci90"][0], s["ci90"][1])
                srcA = {"file": bv.rel(run_dir(rec["runs"][tag]["run"], REAL_A) / "summary.json"), "key": f"posterior.{par}"}
                add(key, "plain", r_, {"median": s["median"], "ci90": list(s["ci90"])}, units, mean + " (median and 90% equal-tailed interval)", srcA, "validation",
                    extra={"decimals": d})
                for pc, vv, w in (("Med", s["median"], "median"), ("Lo", s["ci90"][0], "lower end of the 90% interval"),
                                  ("Hi", s["ci90"][1], "upper end of the 90% interval"), ("Wid", s["ci90"][1] - s["ci90"][0], "full width of the 90% interval"),
                                  ("HalfWid", 0.5 * (s["ci90"][1] - s["ci90"][0]), "half of the full 90% width")):
                    add(key + pc, "plain", fnum(vv, d), vv, units, f"{w} of {mean}", srcA, "validation")
            else:
                add_ph(key, EXPECTED[q], units, mean + " (median and 90% equal-tailed interval)", src_miss, "validation")
                for pc, w in (("Med", "median"), ("Lo", "lower end of the 90% interval"), ("Hi", "upper end of the 90% interval"), ("Wid", "full width of the 90% interval"),
                              ("HalfWid", "half of the full 90% width")):
                    add_ph(key + pc, r"\ph{X.XX}", units, f"{w} of {mean}", src_miss, "validation")
    for q, (par, qd, units) in QW.items():
        name = f"Flux{word}PhysDoubleDelta{q}"
        mean = (f"shift of the {qd} median when the AGN density is doubled with both densities given in Mpc^-3: median of Flux{word}PhysDouble{q} minus "
                f"that of Flux{word}Phys{q}, {desc}; signed (set in math mode); the production pair in h^3 Mpc^-3 is Flux{word}DoubleDelta{q}")
        if E["real"]:
            v = float(E["shift"][q])
            add(name, "plain", fnum(v, DIG[q]), v, units, mean, {"files": E["files"], "computation": f"posterior.{par}.median of the doubled run minus that of the unit run"},
                "validation", extra={"decimals": DIG[q], "medians": {"unit": E["runs"][""][par]["median"], "double": E["runs"]["Double"][par]["median"]}})
        else:
            add_ph(name, r"\ph{0.XX}", units, mean, src_miss, "validation")


DEFINITIONS = """## The assumed AGN number density with flux-limited catalogs (Sec. IV C, Fig. 3(b); scripts/density.py)

Analysis 16's stage-2 runs with the AGN number density assumed at half (`Half`) and at twice (`Double`) the value from the
catalog counts (the galaxy density from its counts), m < 20 (`Twenty`) and m < 18 (`Eighteen`), hosts normalized to z = 1.10,
population known, both catalogs; \\NrealFluxDensity{} realizations each (realization A among them). Their unit-factor partners
are the runs of keys `FluxTwenty` and `FluxEighteen`, whose 90% half-widths of realization A (`\\FluxTwentyfAGNHalfWid`,
`\\FluxTwentyHzeroHalfWid`, `\\FluxEighteenfAGNHalfWid`, `\\FluxEighteenHzeroHalfWid`) are the precision beside which the shifts
read. Runs: `values/production_map.json`, block `density_response`. Black when every summary of the group exists.

- **`\\Flux<L><X><fAGN|Hzero>`**, pieces `Med`, `Lo`, `Hi`, `Wid`: realization A, median and 90% interval (its summary.json).
- **`\\Flux<L><X>Delta<fAGN|Hzero>`**: realization A, median at factor X minus the median of the unit-factor run of the same
  realization. **`...Mean`**, **`...Min`**, **`...Max`**: the mean, smallest and largest of that shift over the realizations
  run at factor X, each against its own unit-factor run. Signed (negative when halving lowers the median): set them in math mode.
  f_AGN shifts to three decimals, H0 shifts to two (as `\\NormDeltafTwoEighteenProd`, `\\NormDeltaHzeroTwoEighteenProd`).
- **`\\Flux<L><X>Delta<fAGN|Hzero>Pred`**: the shift the normalization identity (Eq. rates, LIKELIHOOD_MATH Sec. 8) predicts
  from the unit-factor grid of realization A: ln L_x(H0, f) = ln L_1(H0, f') with logit f' = logit f - ln[F + x(1 - F)],
  F = `\\NormFAGN<L>Counts`; the median of the mapped posterior at x minus that at x = 1. Because the densities are in h^3
  Mpc^-3 and the missing volume at H0 = 100, F does not depend on H0 and the map is one shift of logit f for every H0; the H0
  prediction differs from zero only through the correlation of H0 and f_AGN. The map carries the normaliser only: where a
  catalog is incomplete and mergers or detected injections lie (m < 18: half of the detected mergers), the hosts the model adds
  also scale with the assumed density, and that part (which moves H0) is not in the prediction.
- **`\\NormFAGN<L>Counts`**: F_AGN(1) = N_AGN/(N_AGN + n_AGN V_miss,AGN): N_AGN the cataloged AGN of realization A's survey file,
  n_AGN the unit-factor run's density from the counts (h^3 Mpc^-3), V_miss,AGN = 4 pi int_0^1.10 [1 - C_AGN(z)] dV_c/(dz dOmega) dz
  at H0 = 100 with the run's AGN selection fit. `\\NormFAGN<L>Prod` is the same share at the input density (1e-5 Mpc^-3).
- **`\\NrealFluxDensity`** (a word): the realizations run at each factor.
- The densities in Mpc^-3 (a check, stage X of RUNLIST.yaml, never a production run of any key; map block
  `density_response.units_physical`): the m < 18 grid of realization A with both densities given in Mpc^-3 (the counts'
  values converted at H0 = 67.74, `--density given --density-units physical`), the AGN density at that value and at twice it.
  **`\\FluxEighteenPhys<fAGN|Hzero>`** (AGN density at the counts' value) and **`\\FluxEighteenPhysDouble<fAGN|Hzero>`** (doubled):
  median and 90% interval of the run's summary.json; pieces `Med`, `Lo`, `Hi`, `Wid`, `HalfWid` (the full 90% width and half of it).
  **`\\FluxEighteenPhysDoubleDelta<fAGN|Hzero>`**: median of the doubled run minus that of the unit run, signed, to three and two
  decimals; beside `\\FluxEighteenDoubleDelta<fAGN|Hzero>`, the same shift with the densities in h^3 Mpc^-3 (production).
  Black when both summaries exist.
"""


def main():
    ev = evaluate()
    for word, L in ev.items():
        S = L["share"]
        print(f"m < {L['mlim']}: F_AGN = {S['agn']['F']:.5g} (N {S['agn']['N']}, log10 n {S['agn']['log10n_h3Mpc-3']:.4f}, V {S['agn']['V_miss_h-3Mpc3']:.4g})" if S else f"m < {L['mlim']}: no unit run")
        for xw, E in L["arms"].items():
            if not E["real"]:
                print(f"  {xw}: RED, missing {E['missing'][:3]}")
                continue
            for q, (par, _, _) in QW.items():
                sh = [E["per"][r]["shift"][q] for r in E["rec"]["realizations"]]
                pr = L["pred"][E["factor"]][par if par == "H0" else "f"]["median"] - L["pred"][1.0][par if par == "H0" else "f"]["median"]
                print(f"  {xw:6s} {q:5s} A {E['A']['arm'][par]['median']:.4f} shift A {sh[0]:+.4f} mean {np.mean(sh):+.4f} [{min(sh):+.4f}, {max(sh):+.4f}] pred {pr:+.4f}")


if __name__ == "__main__":
    main()
