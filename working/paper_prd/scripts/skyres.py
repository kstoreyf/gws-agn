#!/usr/bin/env python
"""The sky resolution of the catalog term (Sec. V, last paragraph of the validation): how far the H0 and f_AGN medians of
realization A move when both catalogs are pixelized at HEALPix nside 64 instead of 32 (pixels half as wide). Two
Analysis 16 grids of realization A, identical but for the survey files:

    known_population/both_catalogs/complete/equal_populations/realization_100            (key KnownBoth, stage 1, nside 32)
    known_population/both_catalogs/complete/equal_populations/realization_100_nside64    (stage X of RUNLIST.yaml, --nside 64)

    PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
    $PY working/paper_prd/scripts/skyres.py           # the two macros: real or red, value

Used by build_values.py (the macros), production.py (--write-map writes the pair into values/production_map.json under
"sky_resolution"; --status) and audit_values.py (S checks).

Macros (claim 'validation'):
  SkyResShiftHzero   median of H0 of the nside-64 run minus that of the nside-32 run (realization A), km/s/Mpc, signed,
                     two decimals
  SkyResShiftfAGN    the same for f_AGN, signed, three decimals
Both black when both summary.json exist (block posterior); otherwise the earlier red values (0.1 and 0.005). The pair is
read from RUNLIST.yaml by name: the nside-64 run must carry --nside 64 and sit in stage X (a check, never a production
stage), its partner must be the KnownBoth run of the same realization in a production stage. The selectors of the paper's
keys (placeholder_results.py, classes.py, density.py) skip the stages whose names start with V or X.
"""
from __future__ import annotations

import json
import math
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
PARTNER = "known_population/both_catalogs/complete/equal_populations/realization_{R}"
RUN = PARTNER + f"_nside{NSIDE}"
QW = {"Hzero": ("H0", "H0", "km/s/Mpc", 2, 0.1), "fAGN": ("f_agn", "f_AGN", "", 3, 0.005)}


def _args(a):
    return {a[i].lstrip("-"): a[i + 1] for i in range(len(a) - 1) if a[i].startswith("--") and not a[i + 1].startswith("--")}


def map_block(_runs=None):
    """{"pair": {run, partner, nside, realization, stage, partner_stage, runlist}} or {"not_in_run_lists": why}."""
    import yaml
    if not RUNLIST.exists():
        return {"pair": None, "not_in_run_lists": f"no {RUNLIST.name}"}
    d = yaml.safe_load(RUNLIST.read_text())
    idx = {r["name"]: (str(st["stage"]), r) for st in d["stages"] for r in st["runs"]}
    run, partner = RUN.format(R=f"{REAL_A:03d}"), PARTNER.format(R=f"{REAL_A:03d}")
    why = []
    if run not in idx:
        why.append(f"no run {run} in {RUNLIST.name}")
    else:
        st, r = idx[run]
        if _args(r["args"]).get("nside") != str(NSIDE) or not st.startswith("X"):
            why.append(f"{run}: stage {st}, --nside {_args(r['args']).get('nside')} (expected stage X, --nside {NSIDE})")
    if partner not in idx or not bv.production_stage(idx[partner][0]):
        why.append(f"no production run {partner} in {RUNLIST.name}")
    if why:
        return {"pair": None, "not_in_run_lists": "; ".join(why)}
    return {"_what": "the runs of SkyResShift<Hzero|fAGN> (scripts/skyres.py): the nside-64 run and its nside-32 partner (key "
                     "KnownBoth), realization A",
            "pair": {"run": RUN, "partner": PARTNER, "nside": NSIDE, "realization": REAL_A, "stage": idx[run][0],
                     "partner_stage": idx[partner][0], "runlist": RUNLIST.name}}


@lru_cache(None)
def mapped():
    m = json.loads(PMAP.read_text()) if PMAP.exists() else {}
    blk = m.get("sky_resolution")
    return blk if blk else map_block()


def summaries(rec):
    r = f"{rec['realization']:03d}"
    return RUNS / rec["run"].format(R=r) / "summary.json", RUNS / rec["partner"].format(R=r) / "summary.json"


def status():
    """(real, missing): real when both summaries exist."""
    blk = mapped()
    rec = blk.get("pair")
    if rec is None:
        return False, [blk.get("not_in_run_lists", "not in the map")]
    miss = [bv.rel(p) for p in summaries(rec) if not p.exists()]
    return not miss, miss


def evaluate():
    """{"real", "missing", "rec", "shift": {q: value}, "runs": {nside: posterior}} (shift None while red)."""
    real, miss = status()
    rec = mapped().get("pair")
    out = {"real": real, "missing": miss, "rec": rec, "shift": None}
    if real:
        p64, p32 = summaries(rec)
        s64, s32 = json.loads(p64.read_text()), json.loads(p32.read_text())
        for k in ("events_file", "injections_file", "nsamp", "half_variance_correction", "max_var", "zmax", "core", "realization"):
            assert s64.get(k) == s32.get(k), (k, s64.get(k), s32.get(k))
        out["shift"] = {q: s64["posterior"][par]["median"] - s32["posterior"][par]["median"] for q, (par, *_r) in QW.items()}
        out["runs"] = {NSIDE: s64["posterior"], 32: s32["posterior"]}
        out["files"] = [bv.rel(p64), bv.rel(p32)]
    return out


def render(x, d):
    """Signed number valid in text and math mode."""
    s = bv.fnum(x, d)
    return f"\\ensuremath{{{s}}}" if s.startswith("-") else s


def build(add, add_ph=None):
    """Register the two macros with build_values' registry."""
    E = evaluate()
    rec = E["rec"]
    for q, (par, qd, units, d, ph) in QW.items():
        name = f"SkyResShift{q}"
        meaning = (f"change of the {qd} median of KnownBoth, realization A, when the sky pixels are half as wide (nside 64 against 32): "
                   f"median of the nside-64 run minus that of the nside-32 run, signed")
        if E["real"]:
            v = float(E["shift"][q])
            add(name, "plain", render(v, d), v, units, meaning,
                {"files": E["files"], "computation": f"posterior.{par}.median of the nside-64 summary minus that of the nside-32 summary"},
                "validation", extra={"decimals": d, "nside64": E["runs"][NSIDE][par], "nside32": E["runs"][32][par],
                                     "runs": [rec["run"].format(R=f"{rec['realization']:03d}"), rec["partner"].format(R=f"{rec['realization']:03d}")]})
        else:
            add(name, "ph", f"\\ph{{{bv.fnum(ph, 1 if units else 3)}}}", ph, units, meaning,
                {"missing": "; ".join(E["missing"]) or "the runs",
                 "placeholder_source": "invented (expectation: below a tenth of the 90% half-width of realization A)"}, "validation")


DEFINITIONS = """## The sky resolution of the catalog term (Sec. V; scripts/skyres.py)

Two grids of realization A, population known, both catalogs complete, identical (events, redraw, injections, settings) but
for the survey files: the production pixelization (HEALPix nside 32, the run of key `KnownBoth`) and pixels half as wide
(nside 64; `data/seed100/surveys_ns64/`, made by `analysis_16_production_runs/scripts/make_survey_nside.py` the way the
generator made the nside-32 files; run `.../realization_100_nside64`, stage X of RUNLIST.yaml). Runs: `values/production_map.json`,
block `sky_resolution`. Black when both summaries exist.

- **`\\SkyResShiftHzero`**, **`\\SkyResShiftfAGN`**: median at nside 64 minus median at nside 32 (H0 in km/s/Mpc to two decimals,
  f_AGN to three), signed (a negative value is wrapped in \\ensuremath, so it reads right in text and in math).
"""


def main():
    E = evaluate()
    if not E["real"]:
        print(f"RED: {E['missing']}")
        return
    for q, (par, *_r) in QW.items():
        print(f"{q}: nside 64 {E['runs'][NSIDE][par]['median']:.4f}, nside 32 {E['runs'][32][par]['median']:.4f}, shift {E['shift'][q]:+.4f}")


if __name__ == "__main__":
    main()
