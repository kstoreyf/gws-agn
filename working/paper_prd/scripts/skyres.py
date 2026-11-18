#!/usr/bin/env python
"""The sky resolution of the catalog term (Sec. V, last paragraph of the validation): how far the H0 and f_AGN medians of
realization A move when both catalogs are pixelized at HEALPix nside 64 instead of 32 (pixels half as wide), and how the
Monte Carlo quality of the two grids compares. Two pairs of Analysis 16 grids of realization A, identical but for the
survey files:

  pair (the production Monte Carlo settings: 8000 samples per merger, the realization's own injection set)
    known_population/both_catalogs/complete/equal_populations/realization_100            (key KnownBoth, stage 1, nside 32)
    known_population/both_catalogs/complete/equal_populations/realization_100_nside64    (stage X of RUNLIST.yaml, --nside 64)
  mc_pair (Analysis 14's settings of the all-AGN analyses: 32000 samples per merger, the six injection sets of the
           realization pooled; both in stage X)
    known_population/both_catalogs/complete/equal_populations/realization_100_s32000_inj6           (nside 32)
    known_population/both_catalogs/complete/equal_populations/realization_100_nside64_s32000_inj6   (--nside 64)

    PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
    $PY working/paper_prd/scripts/skyres.py           # the macros: real or red, value

Used by build_values.py (the macros), production.py (--write-map writes the pairs into values/production_map.json under
"sky_resolution"; --status) and audit_values.py (S checks).

Macros (claim 'validation'; <P> = "" for the pair, "MC" for mc_pair):
  SkyRes<P>ShiftHzero   median of H0 of the nside-64 run minus that of the nside-32 run (realization A), km/s/Mpc, signed,
                        two decimals
  SkyRes<P>ShiftfAGN    the same for f_AGN, signed, three decimals
  SkyRes<P>VarPE<SixtyFour|ThirtyTwo>   variance of the posterior-sample sums of ln L at the grid's peak (summary.json
                        variance.pe_var_at_peak: the sum over mergers of the Monte Carlo variance of each merger's ln L), two
                        significant figures
  SkyRes<P>VarSel<SixtyFour|ThirtyTwo>  the selection term's N^2/N_eff at the peak (variance.sigma2_sel_at_peak, N = 1000 mergers)
  SkyRes<P>VarPERatio, SkyRes<P>VarSelRatio   nside 64 over nside 32
Black when both summary.json of the pair exist (block posterior, block variance). Red values while a pair waits:
  pair     the earlier invented values (shifts 0.1 and 0.005; variances 2 and 1.5, ratios 1)
  mc_pair  computed from the pair when it is real, else the invented ones: the shifts = the pair's own shifts (the value the
           cleaner test replaces), the variances = the pair's scaled by Analysis 14's Monte Carlo settings (posterior-sample
           variance x 8000/32000, N^2/N_eff / 6 for six pooled sets of the same size), so the ratios are the pair's
The pairs are read from RUNLIST.yaml by name: the nside-64 runs must carry --nside 64 and sit in stage X (a check, never a
production stage); the pair's partner must be the KnownBoth run of the same realization in a production stage; mc_pair's
nside-32 run must sit in stage X too, with --nsamp 32000 and five --pool-injections. The selectors of the paper's keys
(placeholder_results.py, classes.py, density.py) skip the stages whose names start with V or X.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

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
NSIDE = 64
MC_NSAMP, MC_SETS = 32000, 6
PARTNER = "known_population/both_catalogs/complete/equal_populations/realization_{R}"
RUN = PARTNER + f"_nside{NSIDE}"
MC_PARTNER = PARTNER + f"_s{MC_NSAMP}_inj{MC_SETS}"
MC_RUN = PARTNER + f"_nside{NSIDE}_s{MC_NSAMP}_inj{MC_SETS}"
QW = {"Hzero": ("H0", "H0", "km/s/Mpc", 2, 0.1), "fAGN": ("f_agn", "f_AGN", "", 3, 0.005)}
# Monte Carlo quality at the grid's peak (summary.json, block variance)
VW = {"PE": ("pe_var_at_peak", "variance of the posterior-sample sums of ln L at the grid's peak (sum over the mergers of the Monte "
                               "Carlo variance of each merger's ln L; summary.json variance.pe_var_at_peak)", 2.0),
      "Sel": ("sigma2_sel_at_peak", "the selection term's N^2/N_eff at the grid's peak (N = 1000 mergers, N_eff the effective number of "
                                    "detected injections; summary.json variance.sigma2_sel_at_peak)", 1.5)}
NS_WORD = {64: "SixtyFour", 32: "ThirtyTwo"}
PAIRS = {"pair": "", "mc_pair": "MC"}


def _args(a):
    return {a[i].lstrip("-"): a[i + 1] for i in range(len(a) - 1) if a[i].startswith("--") and not a[i + 1].startswith("--")}


def map_block(_runs=None):
    """{"pair": {run, partner, nside, realization, stage, partner_stage, runlist}, "mc_pair": {... nsamp, n_injection_sets}}; a pair
    not in the run list is None with its reason under "not_in_run_lists"."""
    import yaml
    if not RUNLIST.exists():
        return {"pair": None, "mc_pair": None, "not_in_run_lists": f"no {RUNLIST.name}"}
    d = yaml.safe_load(RUNLIST.read_text())
    idx = {r["name"]: (str(st["stage"]), r) for st in d["stages"] for r in st["runs"]}
    out = {"_what": "the runs of the SkyRes... macros (scripts/skyres.py), realization A: pair = the nside-64 run and its nside-32 partner (key "
                    "KnownBoth) at the production Monte Carlo settings; mc_pair = the same two grids at Analysis 14's settings (32000 samples, "
                    "six pooled injection sets), both in stage X"}
    why_all = {}
    for which, (run_p, partner_p) in (("pair", (RUN, PARTNER)), ("mc_pair", (MC_RUN, MC_PARTNER))):
        run, partner = run_p.format(R=f"{REAL_A:03d}"), partner_p.format(R=f"{REAL_A:03d}")
        why = []
        if run not in idx:
            why.append(f"no run {run} in {RUNLIST.name}")
        else:
            st, r = idx[run]
            if _args(r["args"]).get("nside") != str(NSIDE) or not st.startswith("X"):
                why.append(f"{run}: stage {st}, --nside {_args(r['args']).get('nside')} (expected stage X, --nside {NSIDE})")
        if which == "pair":
            if partner not in idx or not bv.production_stage(idx[partner][0]):
                why.append(f"no production run {partner} in {RUNLIST.name}")
        else:
            for n in (run, partner):
                if n not in idx:
                    if n == partner:
                        why.append(f"no run {n} in {RUNLIST.name}")
                    continue
                st, r = idx[n]
                a = r["args"]
                if not (st.startswith("X") and _args(a).get("nsamp") == str(MC_NSAMP) and a.count("--pool-injections") == MC_SETS - 1):
                    why.append(f"{n}: stage {st}, --nsamp {_args(a).get('nsamp')}, {a.count('--pool-injections')} pooled sets "
                               f"(expected stage X, --nsamp {MC_NSAMP}, {MC_SETS - 1} pooled)")
        if why:
            out[which] = None
            why_all[which] = "; ".join(why)
            continue
        out[which] = {"run": run_p, "partner": partner_p, "nside": NSIDE, "realization": REAL_A, "stage": idx[run][0],
                      "partner_stage": idx[partner][0], "runlist": RUNLIST.name}
        if which == "mc_pair":
            out[which].update({"nsamp": MC_NSAMP, "n_injection_sets": MC_SETS})
    if why_all:
        out["not_in_run_lists"] = why_all
    return out


@lru_cache(None)
def mapped():
    m = json.loads(PMAP.read_text()) if PMAP.exists() else {}
    blk = m.get("sky_resolution")
    if not blk:
        return map_block()
    if "mc_pair" not in blk:                    # a map written before 18 Nov: the second pair from the run list by name
        fresh = map_block()
        blk = dict(blk, mc_pair=fresh.get("mc_pair"))
        if (fresh.get("not_in_run_lists") or {}).get("mc_pair"):
            blk["not_in_run_lists"] = dict(blk.get("not_in_run_lists") or {}, mc_pair=fresh["not_in_run_lists"]["mc_pair"])
    return blk


def summaries(rec):
    r = f"{rec['realization']:03d}"
    return RUNS / rec["run"].format(R=r) / "summary.json", RUNS / rec["partner"].format(R=r) / "summary.json"


def status(which="pair"):
    """(real, missing): real when both summaries of the pair exist."""
    blk = mapped()
    rec = blk.get(which)
    if rec is None:
        nl = blk.get("not_in_run_lists")
        return False, [(nl.get(which) if isinstance(nl, dict) else nl) or "not in the map"]
    miss = [bv.rel(p) for p in summaries(rec) if not p.exists()]
    return not miss, miss


def evaluate(which="pair"):
    """{"real", "missing", "rec", "shift": {q: value}, "var": {v: {64, 32, ratio}}, "runs": {nside: posterior}} (None while red)."""
    real, miss = status(which)
    rec = mapped().get(which)
    out = {"real": real, "missing": miss, "rec": rec, "shift": None, "var": None}
    if real:
        p64, p32 = summaries(rec)
        s64, s32 = json.loads(p64.read_text()), json.loads(p32.read_text())
        for k in ("events_file", "injections_file", "nsamp", "half_variance_correction", "max_var", "zmax", "core", "realization"):
            assert s64.get(k) == s32.get(k), (k, s64.get(k), s32.get(k))
        if which == "mc_pair":
            assert s64["nsamp"] == MC_NSAMP, s64["nsamp"]
        out["shift"] = {q: s64["posterior"][par]["median"] - s32["posterior"][par]["median"] for q, (par, *_r) in QW.items()}
        out["var"] = {v: {NSIDE: float(s64["variance"][f]), 32: float(s32["variance"][f]), "ratio": float(s64["variance"][f]) / float(s32["variance"][f])}
                      for v, (f, *_r) in VW.items()}
        out["runs"] = {NSIDE: s64["posterior"], 32: s32["posterior"]}
        out["files"] = [bv.rel(p64), bv.rel(p32)]
    return out


def render(x, d):
    """Signed number valid in text and math mode."""
    s = bv.fnum(x, d)
    return f"\\ensuremath{{{s}}}" if s.startswith("-") else s


def red_values(which, E0):
    """The placeholder values of a pair that waits: {"shift": {q}, "var": {v: {64, 32, ratio}}, "how"} (see the module header)."""
    if which == "mc_pair" and E0["real"]:
        scale = {"PE": 8000.0 / MC_NSAMP, "Sel": 1.0 / MC_SETS}
        return {"shift": dict(E0["shift"]),
                "var": {v: {NSIDE: E0["var"][v][NSIDE] * scale[v], 32: E0["var"][v][32] * scale[v], "ratio": E0["var"][v]["ratio"]} for v in VW},
                "how": ("from the 8000-sample pair (SkyResShift..., SkyResVar...): the shifts as they are (the value this cleaner test replaces); the "
                        f"variances scaled by the Monte Carlo settings (posterior-sample sums x 8000/{MC_NSAMP}, N^2/N_eff / {MC_SETS} for "
                        f"{MC_SETS} pooled sets of the same size), the ratios unchanged")}
    return {"shift": {q: ph for q, (*_r, ph) in QW.items()}, "var": {v: {NSIDE: ph, 32: ph, "ratio": 1.0} for v, (*_r, ph) in VW.items()},
            "how": "invented (expectation: shifts below a tenth of the 90% half-width of realization A; Monte Carlo variances of order one, ratio one)"}


def build(add, add_ph=None):
    """Register the macros of both pairs with build_values' registry."""
    E0 = evaluate("pair")
    for which, P in PAIRS.items():
        E = E0 if which == "pair" else evaluate(which)
        rec = E["rec"]
        sett = ("at the production Monte Carlo settings (8000 samples per merger, the realization's own injection set)" if which == "pair" else
                f"at Analysis 14's Monte Carlo settings ({MC_NSAMP} samples per merger, {MC_SETS} pooled injection sets; both runs in stage X)")
        runs = [rec["run"].format(R=f"{rec['realization']:03d}"), rec["partner"].format(R=f"{rec['realization']:03d}")] if rec else []
        red = None if E["real"] else red_values(which, E0)
        srcR = None if E["real"] else {"missing": "; ".join(E["missing"]) or "the runs", "placeholder_source": red["how"]}
        for q, (par, qd, units, d, ph) in QW.items():
            name = f"SkyRes{P}Shift{q}"
            meaning = (f"change of the {qd} median of realization A (population known, both catalogs complete) when the sky pixels are half as "
                       f"wide (nside 64 against 32), {sett}: median of the nside-64 run minus that of the nside-32 run, signed")
            if E["real"]:
                v = float(E["shift"][q])
                add(name, "plain", render(v, d), v, units, meaning,
                    {"files": E["files"], "computation": f"posterior.{par}.median of the nside-64 summary minus that of the nside-32 summary"},
                    "validation", extra={"decimals": d, "nside64": E["runs"][NSIDE][par], "nside32": E["runs"][32][par], "runs": runs})
            else:
                v = float(red["shift"][q])
                add(name, "ph", f"\\ph{{{render(v, d) if which == 'mc_pair' else bv.fnum(v, 1 if units else 3)}}}", v, units, meaning, srcR, "validation")
        for v_, (field, what, _ph) in VW.items():
            for part in (NSIDE, 32, "ratio"):
                name = f"SkyRes{P}Var{v_}{NS_WORD[part] if part != 'ratio' else 'Ratio'}"
                if part == "ratio":
                    meaning = f"ratio of the {what}, nside 64 over nside 32, realization A, {sett}"
                else:
                    meaning = f"{what}, the nside-{part} grid of realization A (population known, both catalogs complete), {sett}"
                if E["real"]:
                    x = float(E["var"][v_][part])
                    add(name, "plain", bv.fsig(x, 2), x, "", meaning,
                        {"files": E["files"] if part == "ratio" else [E["files"][0 if part == NSIDE else 1]],
                         "computation": f"variance.{field}" + (" of the nside-64 summary over that of the nside-32 summary" if part == "ratio" else "")},
                        "validation", extra={"nside64": E["var"][v_][NSIDE], "nside32": E["var"][v_][32], "runs": runs})
                else:
                    x = float(red["var"][v_][part])
                    add(name, "ph", f"\\ph{{{bv.fsig(x, 2)}}}", x, "", meaning, srcR, "validation")


DEFINITIONS = """## The sky resolution of the catalog term (Sec. V; scripts/skyres.py)

Grids of realization A, population known, both catalogs complete, identical (events, redraw, injections, settings) but
for the survey files: the production pixelization (HEALPix nside 32) and pixels half as wide (nside 64;
`data/seed100/surveys_ns64/`, made by `analysis_16_production_runs/scripts/make_survey_nside.py` the way the generator made the
nside-32 files). Two pairs (`values/production_map.json`, block `sky_resolution`):
`pair` at the production Monte Carlo settings (the run of key `KnownBoth` and `.../realization_100_nside64`, stage X of
RUNLIST.yaml); `mc_pair` at Analysis 14's settings of the all-AGN analyses, 32000 samples per merger and the realization's six
injection sets pooled (`.../realization_100_s32000_inj6` and `.../realization_100_nside64_s32000_inj6`, both stage X). Each pair
is black when both its summaries exist.

- **`\\SkyResShiftHzero`**, **`\\SkyResShiftfAGN`** (`pair`) and **`\\SkyResMCShiftHzero`**, **`\\SkyResMCShiftfAGN`** (`mc_pair`):
  median at nside 64 minus median at nside 32 (H0 in km/s/Mpc to two decimals, f_AGN to three), signed (a negative value is
  wrapped in \\ensuremath, so it reads right in text and in math).
- **`\\SkyRes[MC]VarPE<SixtyFour|ThirtyTwo>`**: the variance of the posterior-sample sums of ln L at the grid's peak (summary.json
  `variance.pe_var_at_peak`); **`\\SkyRes[MC]VarSel<SixtyFour|ThirtyTwo>`**: the selection term's N^2/N_eff at the peak
  (`variance.sigma2_sel_at_peak`); **`...Ratio`**: nside 64 over nside 32. Two significant figures.
- Red values of `mc_pair` while it waits: the shifts of `pair`, and its variances scaled by the Monte Carlo settings (x 8000/32000
  for the posterior samples, / 6 for six pooled injection sets of the same size), so the ratios are those of `pair`.
"""


def main():
    for which in PAIRS:
        E = evaluate(which)
        if not E["real"]:
            print(f"{which}: RED: {E['missing']}")
            continue
        for q, (par, *_r) in QW.items():
            print(f"{which} {q}: nside 64 {E['runs'][NSIDE][par]['median']:.4f}, nside 32 {E['runs'][32][par]['median']:.4f}, shift {E['shift'][q]:+.4f}")
        for v in VW:
            print(f"{which} Var{v}: nside 64 {E['var'][v][NSIDE]:.4f}, nside 32 {E['var'][v][32]:.4f}, ratio {E['var'][v]['ratio']:.4f}")


if __name__ == "__main__":
    main()
