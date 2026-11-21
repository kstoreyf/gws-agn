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
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

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
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
