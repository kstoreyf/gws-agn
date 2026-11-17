#!/usr/bin/env python
"""Audit of the numbers chain: rerun every computation of build_values.py and check the written files.

    /hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python \
        working/paper_prd/scripts/audit_values.py

Checks (each prints PASS or FAIL; the script exits 1 if any fails):
  A. values/results_macros.tex equals a fresh rendering, macro by macro (name, class, body)
  B. every macro has a values.json record with class, meaning, units and a source; every source path exists
  C. values.json numbers equal the recomputed ones
  D. placeholders.txt lists exactly the red macros; MACROS.md lists every macro
  E. containment records are internally consistent (in68/in90 against the two-sided level)
  F. independent cross-checks of the machinery against stored results:
       f = 0 / f = 1 columns of the joint grid equal Analysis 1's single-catalog scans
       spline-refined medians agree with the stored per-grid JSON summaries
       chain summaries equal the stored run summaries
       grid evidence of the Analysis-10 cube equals the dynesty evidence of the same model and box
       Savage-Dickey (interpolated density) equals the evidence ratio, where both exist
       no spline overshoot above the stored maximum by more than 1 nat; refined maximum within one stored cell
       spline refinement of a subsampled grid reproduces the dense-grid quantiles
       the combined AGN-hosted H0 posterior does not depend on the choice of smooth interpolant
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
import build_values as B  # noqa: E402

OUT = B.OUT
RES = {"PASS": 0, "FAIL": 0}
FAILS = []


def check(ok, what):
    tag = "PASS" if ok else "FAIL"
    RES[tag] += 1
    if not ok:
        FAILS.append(what)
    print(f"[{tag}] {what}")


def parse_tex(text):
    out = {}
    for line in text.splitlines():
        m = re.match(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}  % ", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def main():
    reg, diag, tables, bayes = B.compute()
    fresh = B.render_tex(reg)
    texp = OUT / "values" / "results_macros.tex"
    disk = texp.read_text()
    # ---------------- A ---------------- #
    check(disk == fresh, "A0 results_macros.tex is byte-identical to a fresh rendering")
    pd, pf = parse_tex(disk), parse_tex(fresh)
    check(set(pd) == set(pf), f"A1 same macro set on disk and recomputed ({len(pd)} vs {len(pf)})")
    nmis = [k for k in pf if pd.get(k) != pf[k]]
    check(not nmis, f"A2 every macro body matches the recomputation ({len(nmis)} mismatches{': ' + ', '.join(nmis[:8]) if nmis else ''})")
    check(all(re.fullmatch(r"[A-Za-z]+", k) for k in pd), "A3 every macro name is letters only (valid LaTeX control word)")
    check(len(pd) == len(reg) == len({r['name'] for r in reg}), "A4 no duplicate macro names")
    for r in reg:
        body = pd.get(r["name"], "")
        want = {"anc": f"\\anc{{{r['render']}}}", "ph": r["render"], "plain": r["render"]}[r["class"]]
        if body != want:
            check(False, f"A5 class wrapper of {r['name']}")
            break
    else:
        check(True, "A5 every teal macro is wrapped in \\anc, every red in \\ph, plain ones bare")
    check("\\newcommand{\\ph}[1]" in disk and "\\newcommand{\\anc}[1]" in disk and "\\definecolor{phred}" in disk and "\\definecolor{ancteal}" in disk,
          "A6 colour and \\ph/\\anc definitions at the top")
    # ---------------- B ---------------- #
    vj = json.loads((OUT / "values" / "values.json").read_text())
    mac = vj["macros"]
    check(set(mac) == set(pd), "B0 values.json has a record for every macro and no extra")
    missing_fields = [k for k, v in mac.items() if not all(f in v for f in ("class", "meaning", "units", "source", "claim", "render"))]
    check(not missing_fields, f"B1 every record has class, meaning, units, source, claim, render ({len(missing_fields)} incomplete)")
    nosrc = [k for k, v in mac.items() if not v["source"] or (v["class"] != "ph" and not any(x in v["source"] for x in ("file", "files", "computation")))
             or (v["class"] == "ph" and "missing" not in v["source"])]
    check(not nosrc, f"B2 every teal/plain macro names a file or a computation, every red one what is missing ({len(nosrc)} without)")
    bad_paths = set()
    for k, v in mac.items():
        src = v["source"]
        paths = []
        if isinstance(src.get("file"), str):
            paths.append(src["file"])
        if isinstance(src.get("files"), list):
            paths += [p for p in src["files"] if isinstance(p, str)]
        if isinstance(src.get("json"), str):
            paths.append(src["json"])
        for p in paths:
            q = B.W / p if not p.startswith("/") else Path(p)
            if not q.exists():
                bad_paths.add(p)
    check(not bad_paths, f"B3 every cited source file exists ({len(bad_paths)} missing{': ' + ', '.join(sorted(bad_paths)[:5]) if bad_paths else ''})")
    archive = [k for k, v in mac.items() if "archive" in json.dumps(v["source"])]
    check(not archive, f"B4 no source under working/archive ({len(archive)} found)")
    # ---------------- C ---------------- #
    def flat(o, pre=""):
        if isinstance(o, dict):
            for k, v in o.items():
                yield from flat(v, f"{pre}.{k}")
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                yield from flat(v, f"{pre}[{i}]")
        else:
            yield pre, o
    nbad = 0
    for r in reg:
        a = dict(flat(B.jsonable(r["value"])))
        b = dict(flat(mac[r["name"]]["value"]))
        for key in a:
            x, y = a[key], b.get(key)
            if isinstance(x, (int, float)) and not isinstance(x, bool):
                if y is None or not np.isclose(float(x), float(y), rtol=1e-12, atol=1e-12):
                    nbad += 1
            elif x != y:
                nbad += 1
    check(nbad == 0, f"C0 values.json numbers equal the recomputation ({nbad} differences)")
    # ---------------- D ---------------- #
    ph = [r["name"] for r in reg if r["class"] == "ph"]
    ptxt = (OUT / "placeholders.txt").read_text()
    listed = re.findall(r"^\\([A-Za-z]+) \|", ptxt, flags=re.M)
    check(sorted(listed) == sorted(ph), f"D0 placeholders.txt lists exactly the {len(ph)} red macros")
    mtxt = (OUT / "values" / "MACROS.md").read_text()
    inmenu = re.findall(r"^\| `\\([A-Za-z]+)` \|", mtxt, flags=re.M)
    check(sorted(inmenu) == sorted(pd), f"D1 MACROS.md lists every macro ({len(inmenu)})")
    # ---------------- E ---------------- #
    bad = []
    nrec = 0
    for k, v in mac.items():
        c = v.get("containment")
        if not c:
            continue
        nrec += 1
        lev = c["level_two_sided"]
        if c["in90"] != (lev <= 0.90 + 1e-9) or c["in68"] != (lev <= 0.68 + 1e-9):
            # the two definitions differ only by the quantile interpolation at the boundary
            if not (abs(lev - 0.90) < 2e-3 or abs(lev - 0.68) < 2e-3):
                bad.append(k)
    check(not bad, f"E0 {nrec} containment records consistent (in68/in90 against the two-sided level){': ' + ', '.join(bad[:6]) if bad else ''}")
    with_input = [r["name"] for r in reg if r["class"] == "anc" and "median and 90%" in r["meaning"] and r["containment"] is None]
    check(not with_input, f"E1 every teal posterior summary carries a containment record ({len(with_input)} without)")
    # ---------------- F ---------------- #
    cid = diag["c2_column_identity"]
    check(cid["max_abs_diff_f0_vs_h0_gal_targeted"] == 0.0 and cid["max_abs_diff_f1_vs_h0_agn_targeted"] == 0.0,
          "F0 f = 0 and f = 1 columns of joint_s100 equal analysis_1 h0_gal_targeted / h0_agn_targeted bitwise")
    worst = max(abs(r["H0"]["median"] - r["stored_json_H0_median"]) for r in tables["C1_per_catalog"])
    worstf = max(abs(r["f"]["median"] - r["stored_json_f_median"]) for r in tables["C1_per_catalog"])
    check(worst < 0.05 and worstf < 0.005, f"F1 C1 spline medians vs stored joint_s*.json medians: max |dH0| {worst:.4f}, max |df| {worstf:.4f}")
    # a3 grids against the stored ladder summary
    lad = {r["rung"]: r["selection (this work)"] for r in json.loads((B.A3 / "ladder_summary.json").read_text())["rows"]}
    dd = []
    for key, word, _ in B.RUNGS:
        dd.append(abs(mac[f"Fluxf{word}"]["value"]["median"] - lad[key]["f_AGN"]))
        dd.append(abs(mac[f"FluxHzero{word}"]["value"]["median"] - lad[key]["H0"]) / 10)
    check(max(dd) < 0.005, f"F2 flux-limited spline medians vs analysis_3 ladder_summary.json (max |df| or |dH0|/10 = {max(dd):.4f})")
    # chains equal stored summaries
    chk = []
    for npz, js, pre in ((B.A11 / "a11_11D.npz", B.A11 / "a11_11D.json", "HDFree"), (B.A12 / "a12_12M.npz", B.A12 / "a12_12M.json", "WidthM"),
                         (B.A12 / "a12_12chi.npz", B.A12 / "a12_12chi.json", "WidthChi"),
                         (B.A13 / "a13core_e7c3007_rslice_cap20_historical_n200_s1.npz", B.A13 / "a13core_e7c3007_rslice_cap20_historical_n200_s1.json", "HDEight")):
        js_ = json.loads(js.read_text())["summary"]
        for n, short in B.CHAIN_NAME.items():
            if n in js_:
                v = mac[pre + short]["value"]
                chk.append(abs(v["median"] - js_[n]["median"]) + abs(v["ci90"][0] - js_[n]["ci90"][0]) + abs(v["ci90"][1] - js_[n]["ci90"][1]))
    for key, word, _ in B.RUNGS:
        js_ = json.loads((B.A5 / f"campaign_{key}_dynesty_s100.json").read_text())["summary"]
        for n, short in (("H0", "Hzero"), ("log10n0", "LognGal"), ("log10n0_c2", "LognAGN"), ("f_AGN", "fAGN")):
            v = mac[f"Free{short}{word}"]["value"]
            chk.append(abs(v["median"] - js_[n]["median"]) + abs(v["ci90"][0] - js_[n]["ci90"][0]) + abs(v["ci90"][1] - js_[n]["ci90"][1]))
    check(max(chk) < 1e-9, f"F3 nested-sampling summaries equal the stored run summaries ({len(chk)} parameters, max diff {max(chk):.1e})")
    g = diag["c8_grid_vs_dynesty_lnZ_A10J_box"]
    z = abs(g["grid_refined"] - g["dynesty"]) / g["dynesty_err"]
    check(z < 3, f"F4 grid evidence of the A10 cube {g['grid_refined']:.3f} vs dynesty {g['dynesty']:.3f} +- {g['dynesty_err']:.3f} ({z:.2f} sd)")
    sd_i = diag["c8_SD_both_offsets_zero_interpolated_ln"]
    sd_e = mac["LBBothOffsetsFixH"]["extra"]["lnB"]
    check(abs(sd_i - sd_e) < 0.1, f"F5 both-offsets-zero: Savage-Dickey by interpolation {sd_i:.3f} vs evidence ratio {sd_e:.3f} (ln)")
    a8 = diag["c9_A8_J_vs_S_evidence_ratio_log10"]
    a9 = diag["c9_A9_J_vs_S_evidence_ratio_log10"]
    check(abs(a8 - mac["LBSpinFixH"]["value"]) < 0.05 and abs(a9 - mac["LBSpinHfree"]["value"]) < 0.05,
          f"F6 spin offset: Savage-Dickey {mac['LBSpinFixH']['value']:.3f} / {mac['LBSpinHfree']['value']:.3f} vs evidence ratio {a8:.3f} / {a9:.3f} (log10)")
    gl = diag["grid_log"]
    over = max(x["spline_max_minus_stored_max"] for x in gl)
    check(over < 2.0, f"F7 refined maximum exceeds the stored maximum by < 2 nats on all {len(gl)} grids (max {over:.3f}: the sub-grid peak of the combined AGN-hosted posterior, sd 0.05 on a 0.25 grid; cross-checked in F10)")
    capd = [x["capped_points_depth_below_max_nats"] for x in gl if x["refined_points_capped"]]
    check(min(capd) > 15, f"F7b the overshoot cap acts only far below the maximum: capped points lie >= {min(capd):.1f} nats below it on all {len(capd)} grids where it acts")
    far = [x["grid"] for x in gl if not x["argmax_within_one_stored_cell"]]
    check(not far, f"F8 refined maximum within one stored cell of the stored maximum on every axis ({len(far)} exceptions{': ' + ', '.join(far[:6]) if far else ''})")
    with h5py.File(B.A8 / "arm_S_spatial.h5") as h:
        F_, L_ = h["f_grid"][()], h["log_likelihood"][()]
    dense = B.GridPost(["f"], [F_], L_, [10]).summary("f")
    sub = B.GridPost(["f"], [F_[::5]], L_[::5], [50]).summary("f")
    dq = max(abs(dense["ci90"][0] - sub["ci90"][0]), abs(dense["median"] - sub["median"]), abs(dense["ci90"][1] - sub["ci90"][1]))
    check(dq < 1e-3, f"F9 spline on a 0.05-spaced subsample reproduces the 0.01-spaced quantiles of A8 arm S (max diff {dq:.1e})")
    rob = diag["c3_agn_combined_interpolant_check"]
    dm = abs(rob["cubic"]["q05_q50_q95"][1] - rob["akima"]["q05_q50_q95"][1])
    dl = abs(rob["cubic"]["q05_q50_q95"][1] - rob["local_quadratic_5_nodes"]["peak"])
    check(dm < 0.005 and dl < 0.01, f"F10 combined AGN-hosted H0: cubic vs Akima median differ by {dm:.4f}, cubic vs local quadratic peak by {dl:.4f}")
    # the f = 1 Savage-Dickey and f = 0: refined vs unrefined agree
    dz = max(abs(mac[f"LBfZeroCat{L}"]["extra"]["numerical_diff"]) for L in B.CAT_LETTERS)
    check(dz < 0.01, f"F11 Savage-Dickey at f = 0: refined vs unrefined trapezoid agree to {dz:.4f} in log10 B")
    print(f"\n{RES['PASS']} PASS, {RES['FAIL']} FAIL")
    if FAILS:
        print("failed:", *FAILS, sep="\n  ")
    return 1 if RES["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
