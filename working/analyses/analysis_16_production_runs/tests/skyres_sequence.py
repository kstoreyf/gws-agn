#!/usr/bin/env python
"""The sky-resolution sequence of stage X (19 Nov): which part of the likelihood carries the shift of KnownBoth's H0 at nside 64,
and whether it converges with finer pixels. CPU only; reads each run's summary.json and its stored grids (through the paper
chain's combination, which also reads grid/result.json and config.json for the free parameters, the widened fine pass and the
half-variance correction), and config.json's survey_nside to confirm each run's pixelization.

    PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= $PY tests/skyres_sequence.py      # -> tests/results/skyres_sequence.json

Analyses (paper keys of paper_prd/values/production_map.json), realizations 100, 101, 102, 103, 105, nside 16, 32, 64, 128:
  AGNhosted  every merger AGN-hosted, AGN catalog alone (32000 samples per merger, six pooled injection sets, as stage 1)
  Galhosted  every merger galaxy-hosted, galaxy catalog alone (8000 samples, the realization's own set)
  KnownGal   mixed mergers, galaxy catalog alone (8000 samples, own set)
  KnownBoth  mixed mergers, both catalogs, at Analysis 14's Monte Carlo settings (32000 samples, six pooled sets)
Run names: the stage-1 run (KnownBoth: stage X's realization_NNN_s32000_inj6) at nside 32, with _nside<N> before the settings
tag otherwise (make_runlist.py, build_checks (e)).

For every analysis and nside: per realization the H0 median, 90% interval and half-width from summary.json (posterior.H0; f_AGN
the same where it is a parameter), the Monte Carlo variances at the peak (variance.pe_var_at_peak, variance.sigma2_sel_at_peak),
the paired difference of each median to the realization's nside-32 run with its mean, standard deviation (ddof 1) and standard
error over the realizations, and the ratio of the half-widths. Where all five summaries exist: the posterior combined over the
realizations (product of the likelihoods = sum of the corrected ln L, paper_prd/scripts/run_posteriors.combined, the function of
the paper's <Key>...Comb macros of grid runs), its median, 90% interval and the credible level |2F - 1| of the input (H0 = 67.74;
f_AGN from the runs' truth blocks, checked equal over the realizations). A run without a summary is reported as missing.

Block KnownBoth_production (21 Nov): KnownBoth at the production Monte Carlo settings (8000 samples per merger, the realization's
own injection set) on realizations 100, 101, 102, 103, 105, 106, 107, 108, 109, 110 at nside 32 (stage 1's realization_NNN) and 64
(stage X's realization_NNN_nside64; make_runlist.py build_checks (a), (d), (f)). As above per realization and for the paired
differences over all realizations present; in addition, at each nside, the posterior combined over the realizations present there
(their list under "combined" "realizations"), and per group (100-105, 106-110) the paired differences and the combined posterior
over the group's runs present.

21 Nov, the mixture's offset (make_runlist.py build_checks (g)), realizations 100, 101, 102, 103, 105, nside 32 and 64, each with
the same outputs (per realization, paired to nside 32, combined over the realizations present):
  HostsKnown_<set>    host class known, hosts_known .../<set>/realization_NNN[_nside64]_agn_s32000_inj1 (D1)
  HostsUnknown_<set>  host class unknown, two_tracers_one_population .../<set>/realization_NNN[_nside64] (D2)
  for <set> = peak_f0p30_s0, peak_f0p50_s0 (nside 32: E1's runs; nside 64: stage X's).
  KnownBoth_fheld     KnownBoth at the production settings with f_AGN held at 0.30 (D3): the driver cannot hold f_AGN with both
                      catalogs, so the block reads the line f = 0.30 of KnownBoth_production's stored 2-D grids (the node, or the
                      cubic spline across f where 0.30 is not a node of the fine f axis), summarised as the run summaries are
                      (Akima, 16 times denser) and combined as the paper combines grids; with a check of the interpolation.
  comparisons         unknown minus known on the same set at nside 32 and 64 (and that difference's change), nside 64 minus 32
                      for the known and the unknown analysis, f held minus f free (KnownBoth_production) at each nside.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A16 = HERE.parent
RUNS = A16 / "runs"
WORK = A16.parents[1]
PAPER_SCRIPTS = WORK / "paper_prd" / "scripts"
OUT = HERE / "results" / "skyres_sequence.json"

H0_IN = 67.74
REALIZATIONS = (100, 101, 102, 103, 105)
NSIDES = (16, 32, 64, 128)
REF_NSIDE = 32
ANALYSES = {
    "AGNhosted": {"what": "every merger AGN-hosted, AGN catalog alone (32000 samples per merger, six pooled injection sets)",
                  "base": "known_population/agn_catalog/complete/agn_hosted/realization_{R}", "settings": "_s32000_inj6"},
    "Galhosted": {"what": "every merger galaxy-hosted, galaxy catalog alone (8000 samples, the realization's own injection set)",
                  "base": "known_population/galaxy_catalog/complete/galaxy_hosted/realization_{R}", "settings": ""},
    "KnownGal": {"what": "mixed mergers, galaxy catalog alone (8000 samples, own injection set)",
                 "base": "known_population/galaxy_catalog/complete/equal_populations/realization_{R}", "settings": ""},
    "KnownBoth": {"what": "mixed mergers, both catalogs (32000 samples per merger, six pooled injection sets)",
                  "base": "known_population/both_catalogs/complete/equal_populations/realization_{R}", "settings": "_s32000_inj6"},
    # 21 Nov: KnownBoth at the production settings on ten realizations (make_runlist.py build_checks (a), (d), (f)): nside 32 = the
    # stage-1 run realization_NNN, nside 64 = stage X's realization_NNN_nside64; combined over the realizations present, and per group
    "KnownBoth_production": {
        "what": "mixed mergers, both catalogs, at the production Monte Carlo settings (8000 samples per merger, the realization's own "
                "injection set): nside 32 = stage 1's realization_NNN, nside 64 = stage X's realization_NNN_nside64",
        "base": "known_population/both_catalogs/complete/equal_populations/realization_{R}", "settings": "",
        "realizations": (100, 101, 102, 103, 105, 106, 107, 108, 109, 110), "nsides": (32, 64), "combine_present": True,
        "groups": {"100-105": (100, 101, 102, 103, 105), "106-110": (106, 107, 108, 109, 110)}},
}
# 21 Nov (make_runlist.py build_checks (g)): the mixture's offset. The experiment's grids of the sets peak_f0p30_s0 and
# peak_f0p50_s0 (E1 at nside 32, stage X at nside 64): host class known (hosts_known, D1; its AGN-hosted subset at 32000 samples
# per merger) and unknown (two_tracers_one_population, D2, f_AGN free), on realizations 100, 101, 102, 103, 105.
MIX_SETS = ("peak_f0p30_s0", "peak_f0p50_s0")
for _s in MIX_SETS:
    ANALYSES[f"HostsKnown_{_s}"] = {
        "what": f"host class known (hosts_known), set {_s}: AGN-hosted mergers with the AGN catalog (32000 samples per merger), "
                "galaxy-hosted with the galaxy catalog (8000), the set's injection lane; nside 32 = E1, nside 64 = stage X (D1)",
        "base": f"hosts_known/both_catalogs/complete/{_s}/realization_{{R}}", "settings": "_agn_s32000_inj1",
        "nsides": (32, 64), "combine_present": True}
    ANALYSES[f"HostsUnknown_{_s}"] = {
        "what": f"host class unknown (two_tracers_one_population, f_AGN free), set {_s}: both catalogs, 8000 samples per merger, "
                "the set's injection lane; nside 32 = E1, nside 64 = stage X (D2)",
        "base": f"two_tracers_one_population/both_catalogs/complete/{_s}/realization_{{R}}", "settings": "",
        "nsides": (32, 64), "combine_present": True}
# D3: KnownBoth at the production settings with f_AGN held at its input. The driver cannot hold f_AGN with both catalogs, so
# this block reads the line f = F_HELD of the stored 2-D grids of KnownBoth_production's runs (fine pass, coarse outside it): the
# ln L values a one-dimensional run would evaluate at those H0 nodes. Where F_HELD is not a node of the fine f axis the line is
# the cubic spline of the fine pass across f (run_posteriors._spline_parts, the paper's combination), and the error of that
# interpolation is measured on the runs that have the node (the same spline with every other f column, F_HELD left out).
F_HELD = 0.30
FHELD_KEY = "KnownBoth_fheld"
FHELD_OF = "KnownBoth_production"
PARAMS = ("H0", "f_agn")


def run_name(key, real, nside):
    a = ANALYSES[key]
    return a["base"].format(R=f"{real:03d}") + ("" if nside == REF_NSIDE else f"_nside{nside}") + a["settings"]


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def one_run(name, nside):
    """Per-run numbers from summary.json (None when missing)."""
    s = read(RUNS / name / "summary.json")
    if s is None:
        return None
    cfg = read(RUNS / name / "config.json") or {}
    row = {"run": name, "survey_nside": int(cfg.get("survey_nside", REF_NSIDE)), "nsamp": s.get("nsamp"),
           "pe_var_at_peak": s["variance"]["pe_var_at_peak"], "sigma2_sel_at_peak": s["variance"]["sigma2_sel_at_peak"],
           "truth": s.get("truth", {})}
    row["survey_nside_ok"] = row["survey_nside"] == nside
    for p in PARAMS:
        if p in s.get("free", []):
            b = s["posterior"][p]
            row[p] = {"median": b["median"], "ci90": list(b["ci90"]), "half_width": 0.5 * (b["ci90"][1] - b["ci90"][0])}
    return row


def mean_sd_se(v):
    v = [float(x) for x in v]
    n = len(v)
    m = statistics.fmean(v) if n else None
    sd = statistics.stdev(v) if n > 1 else None
    return {"n": n, "mean": m, "sd": sd, "se": sd / math.sqrt(n) if sd is not None else None}


def combiner():
    """paper_prd/scripts/run_posteriors.combined (imports build_values and, when called, placeholder_results: both only read
    files at import)."""
    if str(PAPER_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(PAPER_SCRIPTS))
    import run_posteriors as rp
    return rp


def combined(rp, names, param, x0):
    s, _, _, x0 = rp.combined([RUNS / n for n in names], param, x0=x0)
    lo, hi = s["ci90"]
    return {"median": s["median"], "ci90": [lo, hi], "minus": s["median"] - lo, "plus": hi - s["median"],
            "half_width": 0.5 * (hi - lo), "input": x0, "level_of_input": s["level_of_input"]}


def input_of(p, have):
    """(input value, the differing truth values or None): H0 = 67.74; f_AGN from the runs' truth blocks."""
    if p == "H0":
        return H0_IN, None
    tv = sorted({float(r["truth"][p]) for r in have.values() if p in r["truth"]})
    return (tv[0] if tv else None), (tv if len(tv) > 1 else None)


def combined_over(rp, have, params):
    """The posterior of each parameter combined over the runs `have` ({R: row}), with the realizations it used."""
    comb = {"realizations": sorted(have)}
    for p in params:
        x0, differ = input_of(p, have)
        if differ:
            comb[f"{p}_truth_values_differ"] = differ
        comb[p] = combined(rp, [have[R]["run"] for R in sorted(have)], p, x0)
    return comb


def groups_block(rp, key, rows, ns, params):
    """Per group of realizations (ANALYSES[key]["groups"]): the paired differences to nside 32 (mean, sd ddof 1, se) and the
    posterior combined over the group's runs present at this nside."""
    out = {}
    ref = rows[REF_NSIDE]
    for g, reals in ANALYSES[key]["groups"].items():
        have = {R: rows[ns][R] for R in reals if rows[ns].get(R) is not None}
        blk = {"realizations": list(reals), "present": sorted(have),
               "missing": [run_name(key, R, ns) for R in reals if rows[ns].get(R) is None]}
        if ns != REF_NSIDE:
            blk["paired_to_nside32"] = {p: {"stats": mean_sd_se([r[p]["median"] - ref[R][p]["median"]
                                                                 for R, r in have.items() if ref.get(R)])} for p in params}
        if have:
            blk["combined"] = combined_over(rp, have, params)
        out[g] = blk
    return out


def analysis_block(rp, key):
    a = ANALYSES[key]
    reals, nsides = a.get("realizations", REALIZATIONS), a.get("nsides", NSIDES)
    out = {"what": a["what"], "nsides": {}}
    if "realizations" in a:
        out["realizations"] = list(reals)
    rows = {ns: {R: one_run(run_name(key, R, ns), ns) for R in reals} for ns in nsides}
    ref = rows[REF_NSIDE]
    for ns in nsides:
        have = {R: r for R, r in rows[ns].items() if r is not None}
        missing = [run_name(key, R, ns) for R, r in rows[ns].items() if r is None]
        blk = {"runs": {str(R): r for R, r in have.items()}, "missing": missing, "complete": not missing}
        if not have:
            if "groups" in a:
                blk["groups"] = {g: {"realizations": list(rs), "present": [], "missing": [run_name(key, R, ns) for R in rs]}
                                 for g, rs in a["groups"].items()}
            out["nsides"][str(ns)] = blk
            continue
        params = [p for p in PARAMS if p in next(iter(have.values()))]
        blk["params"] = params
        if ns != REF_NSIDE:
            paired = {}
            for p in params:
                d = {str(R): r[p]["median"] - ref[R][p]["median"] for R, r in have.items() if ref.get(R)}
                q = {str(R): r[p]["half_width"] / ref[R][p]["half_width"] for R, r in have.items() if ref.get(R)}
                paired[p] = {"median_minus_nside32": d, "stats": mean_sd_se(d.values()),
                             "half_width_over_nside32": q, "half_width_ratio_mean": mean_sd_se(q.values())["mean"]}
            blk["paired_to_nside32"] = paired
        blk["variance_at_peak_mean"] = {v: statistics.fmean(r[v] for r in have.values())
                                        for v in ("pe_var_at_peak", "sigma2_sel_at_peak")}
        blk["survey_nside_ok"] = all(r["survey_nside_ok"] for r in have.values())
        if a.get("combine_present"):
            # every realization present at this nside (the list is in the block), and per group
            blk["combined"] = combined_over(rp, have, params)
            if "groups" in a:
                blk["groups"] = groups_block(rp, key, rows, ns, params)
        elif not missing:
            comb = {}
            for p in params:
                if p == "H0":
                    x0 = H0_IN
                else:
                    tv = sorted({float(r["truth"][p]) for r in have.values() if p in r["truth"]})
                    x0 = tv[0] if tv else None
                    if len(tv) > 1:
                        comb[f"{p}_truth_values_differ"] = tv
                comb[p] = combined(rp, [r["run"] for r in have.values()], p, x0)
            blk["combined"] = comb
        out["nsides"][str(ns)] = blk
    return out


# ---------------------------------------------------------------- D3: the f = F_HELD line of KnownBoth's 2-D grids
def _slice_parts(rp, run_dir, f0=F_HELD, drop_node=False):
    """The corrected ln L of a 2-D grid run on the line f = f0, as one-dimensional parts [(["H0"], [axis], lnc)] (fine pass
    first, coarse outside it), and how the fine line was obtained. drop_node: the fine pass's f columns of one parity only (every
    other column, the parity without f0), then the spline (the interpolation check)."""
    parts2 = rp._grid_parts(run_dir)
    out, how = [], {}
    for i, (names, axes, lnc) in enumerate(parts2):
        if names != ["H0", "f_agn"]:
            raise ValueError(f"{run_dir}: free parameters {names}")
        h, fa = axes
        k = np.where(np.abs(fa - f0) < 1e-9)[0]
        if i == 0:
            if drop_node:
                if not k.size:
                    return None, None
                keep = (np.arange(len(fa)) % 2 != k[0] % 2) & np.all(np.isfinite(lnc), axis=0)
                sp = rp._spline_parts([(names, [h, fa[keep]], lnc[:, keep])])
                line = sp[0](h, np.full_like(h, f0), grid=False)
                how = {"fine": "cubic spline across f of every other column (f0 left out)", "f_step": float(fa[2] - fa[0])}
            elif k.size:
                line = lnc[:, k[0]]
                how = {"fine": "node", "f_step": float(fa[1] - fa[0])}
            else:
                # the spline of the fine pass; where some cells are guarded (-inf, at large f), of the columns that are finite
                # at every H0 (they must bracket f0)
                cols = np.all(np.isfinite(lnc), axis=0)
                if not (cols[fa < f0].any() and cols[fa > f0].any()):
                    raise ValueError(f"{run_dir}: no finite fine columns on both sides of f = {f0}")
                sp = rp._spline_parts([(names, [h, fa[cols]], lnc[:, cols])])
                line = sp[0](h, np.full_like(h, f0), grid=False)
                how = {"fine": "cubic spline across f (f0 is not a node)", "f_step": float(fa[1] - fa[0]),
                       "nearest_nodes": [float(fa[fa < f0].max()), float(fa[fa > f0].min())],
                       "columns_used": f"{int(cols.sum())} of {len(fa)} (the others hold guarded cells)" if not cols.all() else "all"}
        else:
            if not k.size:
                continue          # the coarse pass only outside the fine one; f step 0.1 holds 0.30
            line = lnc[:, k[0]]
        out.append((["H0"], [h], np.asarray(line, float)))
    return out, how


def _line_summary(parts):
    """Median and 90% interval of the fine line, as the run summaries do (prod_summary.refined_grid_summary: Akima onto a
    lattice 16 times denser than the nodes, prod_common.summarize_1d)."""
    from scipy.interpolate import Akima1DInterpolator
    import prod_common as PC
    names, (h,), l = parts[0]
    ok = np.isfinite(l)
    lf = np.where(ok, l, np.min(l[ok]) - 300.0)
    lat = np.linspace(h[0], h[-1], (len(h) - 1) * 16 + 1)
    s = PC.summarize_1d(lat, Akima1DInterpolator(h, lf)(lat))
    return {"median": s["median"], "ci90": list(s["ci90"]), "half_width": 0.5 * (s["ci90"][1] - s["ci90"][0]),
            "mass_at_edges": s["mass_at_edges"]}


def _peak_variances(run_dir, f0=F_HELD):
    """pe_var and N^2/N_eff of the fine pass at the line's peak (at the f node nearest f0)."""
    import h5py
    rp_fine = "fine_widened" if ((read(Path(run_dir) / "grid" / "result.json") or {}).get("outside_mass_check") or {}).get(
        "widened") else "fine"
    with h5py.File(Path(run_dir) / "grid" / f"{rp_fine}.h5", "r") as f:
        fa = np.asarray(f["axis_f_agn"])
        k = int(np.argmin(np.abs(fa - f0)))
        lnL, pv, ne = (np.asarray(f[x])[:, k] for x in ("lnL", "pe_var", "n_eff"))
        n = int(f.attrs["n_events"])
    lc = lnL + 0.5 * pv - 0.5 * n / ne
    j = int(np.nanargmax(lc))
    return float(pv[j]), float(n ** 2 / ne[j]), float(fa[k])


def combined_lines(rp, all_parts, x0=H0_IN):
    """The H0 posterior from lines summed over realizations, as run_posteriors.grid_marginal sums grids (common mesh, cubic
    spline of each fine line, linear in the coarse line), summarised by placeholder_results.summarize_xp."""
    import placeholder_results as phr
    names, axes = rp._common_axes(all_parts)
    mesh = axes[0][:, None]
    tot = sum(rp._eval_parts_spline(p, mesh) for p in all_parts)
    w = np.exp(tot - np.max(tot[np.isfinite(tot)]))
    x = axes[0]
    s = phr.summarize_xp(x, w / np.trapz(w, x), x0)
    lo, hi = s["ci90"]
    return {"median": s["median"], "ci90": [lo, hi], "minus": s["median"] - lo, "plus": hi - s["median"],
            "half_width": 0.5 * (hi - lo), "input": x0, "level_of_input": s["level_of_input"]}


def fheld_block(rp):
    """D3: per realization and nside the f = F_HELD line of KnownBoth_production's grid; paired nside 64 minus 32; combined over
    the realizations present (and per group); the interpolation check."""
    src = ANALYSES[FHELD_OF]
    reals, nsides = src["realizations"], src["nsides"]
    out = {"what": f"KnownBoth at the production settings with f_AGN held at {F_HELD}: the line f = {F_HELD} of the stored 2-D grids "
                   f"of {FHELD_OF} (nside 32 = stage 1's realization_NNN, nside 64 = stage X's realization_NNN_nside64); fine line at "
                   "the f node, else the cubic spline across f", "f_held": F_HELD, "realizations": list(reals), "nsides": {}}
    parts, rows = {}, {}
    checks = []
    for ns in nsides:
        rows[ns], parts[ns] = {}, {}
        for R in reals:
            name = run_name(FHELD_OF, R, ns)
            d = RUNS / name
            if read(d / "summary.json") is None or not (d / "grid" / "result.json").exists():
                rows[ns][R] = None
                continue
            pp, how = _slice_parts(rp, d)
            cfg = read(d / "config.json") or {}
            pv, s2, fk = _peak_variances(d)
            rows[ns][R] = {"run": name, "survey_nside": int(cfg.get("survey_nside", REF_NSIDE)), "line": how,
                           "H0": _line_summary(pp), "pe_var_at_peak": pv, "sigma2_sel_at_peak": s2,
                           "variances_at_f_node": fk}
            rows[ns][R]["survey_nside_ok"] = rows[ns][R]["survey_nside"] == ns
            parts[ns][R] = pp
            if how["fine"] == "node":
                pd, hd = _slice_parts(rp, d, drop_node=True)
                l0, l1 = pp[0][2], pd[0][2]
                near = l0 > np.nanmax(l0) - 10.0
                checks.append({"run": name, "f_step_of_check": hd["f_step"],
                               "max_abs_dlnL_where_within_10_nats": float(np.max(np.abs(l1 - l0)[near])),
                               "median_shift": _line_summary(pd)["median"] - rows[ns][R]["H0"]["median"],
                               "half_width_ratio": _line_summary(pd)["half_width"] / rows[ns][R]["H0"]["half_width"]})
    ref = rows[REF_NSIDE]
    for ns in nsides:
        have = {R: r for R, r in rows[ns].items() if r is not None}
        blk = {"runs": {str(R): r for R, r in have.items()}, "missing": [run_name(FHELD_OF, R, ns) for R in reals if rows[ns][R] is None],
               "params": ["H0"], "complete": len(have) == len(reals)}
        blk["interpolated"] = [R for R, r in have.items() if r["line"]["fine"] != "node"]
        if ns != REF_NSIDE:
            d = {str(R): r["H0"]["median"] - ref[R]["H0"]["median"] for R, r in have.items() if ref.get(R)}
            q = {str(R): r["H0"]["half_width"] / ref[R]["H0"]["half_width"] for R, r in have.items() if ref.get(R)}
            blk["paired_to_nside32"] = {"H0": {"median_minus_nside32": d, "stats": mean_sd_se(d.values()),
                                               "half_width_over_nside32": q, "half_width_ratio_mean": mean_sd_se(q.values())["mean"]}}
        blk["variance_at_peak_mean"] = {v: statistics.fmean(r[v] for r in have.values()) for v in ("pe_var_at_peak", "sigma2_sel_at_peak")} if have else {}
        blk["survey_nside_ok"] = all(r["survey_nside_ok"] for r in have.values())
        if have:
            blk["combined"] = {"realizations": sorted(have), "H0": combined_lines(rp, [parts[ns][R] for R in sorted(have)])}
        grp = {}
        for g, gr in src["groups"].items():
            gh = {R: have[R] for R in gr if R in have}
            gb = {"realizations": list(gr), "present": sorted(gh), "missing": [run_name(FHELD_OF, R, ns) for R in gr if R not in gh]}
            if ns != REF_NSIDE:
                gb["paired_to_nside32"] = {"H0": {"stats": mean_sd_se([r["H0"]["median"] - ref[R]["H0"]["median"]
                                                                       for R, r in gh.items() if ref.get(R)])}}
            if gh:
                gb["combined"] = {"H0": combined_lines(rp, [parts[ns][R] for R in sorted(gh)])}
            grp[g] = gb
        blk["groups"] = grp
        out["nsides"][str(ns)] = blk
    out["interpolation_check"] = {
        "what": "on every run whose fine f axis has the node f0: the line from the spline of every other f column (f0 left out; "
                "f step 0.04-0.06) against the node's own line",
        "n_runs": len(checks), "max_abs_dlnL": max((c["max_abs_dlnL_where_within_10_nats"] for c in checks), default=None),
        "max_abs_median_shift": max((abs(c["median_shift"]) for c in checks), default=None), "runs": checks}
    return out, rows


# ---------------------------------------------------------------- paired comparisons of the mixture's offset
def _medians(blocks, key, ns):
    b = blocks.get(key, {}).get("nsides", {}).get(str(ns), {})
    return {int(R): r["H0"]["median"] for R, r in (b.get("runs") or {}).items()}, (b.get("combined") or {}).get("H0")


def paired(a, b):
    """a minus b over the realizations in both: per realization, mean/sd/se."""
    d = {str(R): a[R] - b[R] for R in sorted(set(a) & set(b))}
    return {"per_realization": d, "stats": mean_sd_se(d.values())}


def comparisons(blocks):
    """Unknown minus known on the same set (nside 32, 64, and the change of that difference); nside 64 minus 32 for the known
    and the unknown analysis; f held minus f free (stage 1's KnownBoth) at each nside."""
    out = {"unknown_minus_known": {}, "nside64_minus_32": {}, "fheld_minus_ffree": {}}
    for s in MIX_SETS:
        ku, kk = f"HostsUnknown_{s}", f"HostsKnown_{s}"
        blk = {}
        for ns in (32, 64):
            mu, cu = _medians(blocks, ku, ns)
            mk, ck = _medians(blocks, kk, ns)
            p = paired(mu, mk)
            if cu and ck:
                p["combined_unknown"], p["combined_known"] = cu["median"], ck["median"]
                p["combined_unknown_minus_known"] = cu["median"] - ck["median"]
            blk[str(ns)] = p
        d32, d64 = blk["32"]["per_realization"], blk["64"]["per_realization"]
        dd = {R: d64[R] - d32[R] for R in d64 if R in d32}
        blk["64_minus_32_of_the_difference"] = {"per_realization": dd, "stats": mean_sd_se(dd.values())}
        out["unknown_minus_known"][s] = blk
        for tag, key in (("known", kk), ("unknown", ku)):
            m32, c32 = _medians(blocks, key, 32)
            m64, c64 = _medians(blocks, key, 64)
            p = paired(m64, m32)
            if c32 and c64:
                p["combined_64_minus_32"] = c64["median"] - c32["median"]
            out["nside64_minus_32"].setdefault(tag, {})[s] = p
    for ns in (32, 64):
        mh, ch = _medians(blocks, FHELD_KEY, ns)
        mf, cf = _medians(blocks, FHELD_OF, ns)
        p = paired(mh, mf)
        if ch and cf:
            p["combined_held"], p["combined_free"] = ch["median"], cf["median"]
            p["combined_held_minus_free"] = ch["median"] - cf["median"]
        out["fheld_minus_ffree"][str(ns)] = p
    return out


def print_comparisons(c):
    def line(lbl, p):
        s = p["stats"]
        txt = f"    {lbl}: " + (" ".join(f"{R} {v:+.3f}" for R, v in p["per_realization"].items()) or "no pair present yet")
        if s["n"]:
            txt += f"  | n {s['n']} mean {s['mean']:+.3f}" + (f" sd {s['sd']:.3f} se {s['se']:.3f}" if s["sd"] is not None else "")
        print(txt)
    print("\n=== comparisons (H0 medians)")
    for s, blk in c["unknown_minus_known"].items():
        for ns in ("32", "64"):
            p = blk[ns]
            line(f"unknown minus known, {s}, nside {ns}", p)
            if "combined_unknown_minus_known" in p:
                print(f"      combined: unknown {p['combined_unknown']:.3f} known {p['combined_known']:.3f} difference "
                      f"{p['combined_unknown_minus_known']:+.3f}")
        line(f"(unknown minus known) nside 64 minus 32, {s}", blk["64_minus_32_of_the_difference"])
    for tag, d in c["nside64_minus_32"].items():
        for s, p in d.items():
            line(f"nside 64 minus 32, {tag}, {s}", p)
            if "combined_64_minus_32" in p:
                print(f"      combined 64 minus 32: {p['combined_64_minus_32']:+.3f}")
    for ns, p in c["fheld_minus_ffree"].items():
        line(f"f held at {F_HELD} minus f free (KnownBoth, production settings), nside {ns}", p)
        if "combined_held_minus_free" in p:
            print(f"      combined: held {p['combined_held']:.3f} free {p['combined_free']:.3f} difference {p['combined_held_minus_free']:+.3f}")


def fmt_iv(b):
    return f"{b['median']:.3f} [{b['ci90'][0]:.3f}, {b['ci90'][1]:.3f}] hw {b['half_width']:.3f}"


def print_combined(p, c, indent="    "):
    if isinstance(c, dict):
        print(f"{indent}combined {p}: {c['median']:.3f} (+{c['plus']:.3f}, -{c['minus']:.3f}) 90%, input {c['input']} at the "
              f"{100 * c['level_of_input']:.1f}% credible level")
    elif p != "realizations":
        print(f"{indent}combined {p}: truth values differ over the realizations {c}")


def print_block(key, blk):
    print(f"\n=== {key}: {blk['what']}")
    n_real = len(blk.get("realizations", REALIZATIONS))
    for ns, b in blk["nsides"].items():
        n_have = len(b["runs"])
        print(f"  nside {ns}: {n_have} of {n_real} summaries"
              + (f"; missing {len(b['missing'])}" if b["missing"] else "")
              + ("" if b.get("survey_nside_ok", True) else "  [config survey_nside differs from the name!]"))
        if not n_have:
            continue
        for R, r in b["runs"].items():
            line = f"    {R}  H0 {fmt_iv(r['H0'])}"
            pr = b.get("paired_to_nside32", {})
            if "H0" in pr and R in pr["H0"]["median_minus_nside32"]:
                line += f"  dH0 {pr['H0']['median_minus_nside32'][R]:+.3f}"
            if "f_agn" in r:
                line += f"  f_AGN {fmt_iv(r['f_agn'])}"
                if "f_agn" in pr and R in pr["f_agn"]["median_minus_nside32"]:
                    line += f"  df {pr['f_agn']['median_minus_nside32'][R]:+.4f}"
            line += f"  pe_var {r['pe_var_at_peak']:.3f}  sigma2_sel {r['sigma2_sel_at_peak']:.3f}"
            print(line)
        for p, pr in b.get("paired_to_nside32", {}).items():
            s = pr["stats"]
            if s["n"]:
                print(f"    paired {p} median minus nside 32 over {s['n']}: mean {s['mean']:+.4f}"
                      + (f", sd {s['sd']:.4f}, se {s['se']:.4f}" if s["sd"] is not None else "")
                      + f"; half-width ratio mean {pr['half_width_ratio_mean']:.3f}")
        comb = b.get("combined") or {}
        if "realizations" in comb:
            print(f"    combined over the {len(comb['realizations'])} realizations present ({', '.join(map(str, comb['realizations']))}):")
        for p, c in comb.items():
            print_combined(p, c)
        for g, gb in (b.get("groups") or {}).items():
            print(f"    group {g}: {len(gb['present'])} of {len(gb['realizations'])} present"
                  + (f" (missing {', '.join(str(R) for R in gb['realizations'] if R not in gb['present'])})" if gb["missing"] else ""))
            for p, pr in (gb.get("paired_to_nside32") or {}).items():
                s = pr["stats"]
                if s["n"]:
                    print(f"      paired {p} median minus nside 32 over {s['n']}: mean {s['mean']:+.4f}"
                          + (f", sd {s['sd']:.4f}, se {s['se']:.4f}" if s["sd"] is not None else ""))
            for p, c in (gb.get("combined") or {}).items():
                print_combined(p, c, indent="      ")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    rp = combiner()
    doc = {"_what": "tests/skyres_sequence.py: the sky-resolution sequence of stage X (make_runlist.py build_checks (c)-(e))",
           "realizations": list(REALIZATIONS), "nsides": list(NSIDES), "reference_nside": REF_NSIDE, "H0_input": H0_IN,
           "combination": "paper_prd/scripts/run_posteriors.combined: corrected ln L summed over the realizations on a common mesh "
                          "(cubic splines of each fine pass, linear in the coarse pass), placeholder_results.summarize_xp "
                          "(median, equal-tailed 90%, level_of_input = |2 F(input) - 1|)",
           "analyses": {}}
    for key in ANALYSES:
        doc["analyses"][key] = analysis_block(rp, key)
        print_block(key, doc["analyses"][key])
    doc["analyses"][FHELD_KEY], _ = fheld_block(rp)
    print_block(FHELD_KEY, doc["analyses"][FHELD_KEY])
    ic = doc["analyses"][FHELD_KEY]["interpolation_check"]
    print(f"    interpolation check ({ic['n_runs']} runs with the node; spline of every other f column): max |d lnL| "
          f"{ic['max_abs_dlnL']:.2e} within 10 nats of the peak, max |median shift| {ic['max_abs_median_shift']:.2e}")
    doc["comparisons"] = comparisons(doc["analyses"])
    print_comparisons(doc["comparisons"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
