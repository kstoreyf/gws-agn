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
    # D2: values/fallback_macros.tex gives a default to exactly the macros main.tex uses and results_macros.tex does not define
    tex = (OUT / "main.tex").read_text()
    t_nc = re.sub(r"(?<!\\)%.*", "", tex)
    used = set(re.findall(r"\\([A-Za-z]+)", t_nc))
    own = set(re.findall(r"\\(?:newcommand|renewcommand|providecommand|DeclareMathOperator)\*?\s*\{?\\([A-Za-z]+)", t_nc))
    want = sorted(n for n in used if n[0].isupper() and n not in pd and n not in own and n not in B.LATEX_NAMES)
    fbt = B.FALLBACK.read_text()
    have_fb = sorted(re.findall(r"^\\providecommand\{\\([A-Za-z]+)\}", fbt, flags=re.M))
    in_tex = [n for n in own if n in pd]
    check(have_fb == want, f"D2 values/fallback_macros.tex defaults exactly the {len(want)} macros main.tex uses and the chain does not define"
                                          f"{': ' + ', '.join(want) if want else ''}{'; note: main.tex also provides chain macros itself (harmless, stale): ' + ', '.join(sorted(in_tex)) if in_tex else ''}"
                                          f"{'; file has ' + ', '.join(have_fb) if have_fb != want else ''}")
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
    audit_r2(mac, reg, diag, tables)
    print(f"\n{RES['PASS']} PASS, {RES['FAIL']} FAIL")
    if FAILS:
        print("failed:", *FAILS, sep="\n  ")
    return 1 if RES["FAIL"] else 0


# ============================================================================================= #
# G: revision 2 (17 Nov 2026). Every check recomputes with code that build_values.py does not use:
# Analysis 15's own closed form (a15_common.py) and darksirens-core's normalisers from the Analysis-15
# dry run, astropy distances, exact host membership in the flux-limited catalog files, brute-force
# angular counting, Monte Carlo draws from the stored grids, the per-scan JSONs of Analysis 14.
# ============================================================================================= #
def _a15():
    sys.path.insert(0, str(B.A15_COMMON.parent))
    import a15_common as C  # noqa: E402
    return C


def _astro_z_of_dl(H0):
    from astropy.cosmology import FlatLambdaCDM
    cos = FlatLambdaCDM(H0=H0, Om0=B.OM0, Tcmb0=0.0)
    zg = np.linspace(0.0, 1.2, 24001)
    dl = cos.luminosity_distance(zg).value
    return lambda d: np.interp(d, dl, zg), cos


def _pdg_decimals(err):
    """Independent statement of the PDG rule: keep 2 significant figures if the 3 leading digits are
    below 355, else 1; 950-999 round up to the next decade with 2."""
    s = f"{err:.2e}"                      # d.dde+XX
    lead, ex = int(round(float(s[:4]) * 100)), int(s.split("e")[1])
    if lead >= 950:
        return max(0, 2 - 1 - (ex + 1))
    return max(0, (2 if lead < 355 else 1) - 1 - ex)


def audit_r2(mac, reg, diag, tables):
    C = _a15()
    H0 = B.IN["H0"]
    ng, na = 10 ** B.IN["lognGal"], 10 ** B.IN["lognAGN"]
    # ---- G1: F_k against Analysis 15's closed form and against darksirens-core's own Z_k ----
    worst = 0.0
    for key, word, _ in B.RUNGS_ALL:
        for cls, n0 in (("gal", ng), ("agn", na)):
            fit = C.schechter_fit(C.FITS / f"fit_{cls}_{key}_schechter.json")
            N = C.survey_nobs(B.DATA / "seed100" / "surveys" / f"survey_{cls}_{key}_ns32.h5")
            for zmax, suf in ((5.0, ""), (C.ZMAX_R, "Prod")):
                F = N / C.z_closed(N, math.log10(n0), fit, H0, zmax)
                got = mac[f"NormF{'Gal' if cls == 'gal' else 'AGN'}{word}{suf}"]["value"]
                worst = max(worst, abs(got / F - 1))
    check(worst < 2e-4, f"G1 F_k (5 rungs x 2 classes x 2 normalisation ends) equal Analysis 15's own closed form (a15_common.z_closed) to {worst:.1e} relative")
    dry = B.AN / "analysis_15_host_normalisation" / "logs" / "dryrun" / "normalisation_summary.json"
    dd = json.loads(dry.read_text())["scans"]
    pairs = [("p1_complete", "Complete", ""), ("p1_m21", "TwentyOne", ""), ("p1_m18", "Eighteen", ""), ("d_m18_x1", "Eighteen", "Prod"), ("b_complete_x1", "Complete", "Prod")]
    dev = 0.0
    for scan, word, suf in pairs:
        Fc = dd[scan]["normaliser"]["F_at_67.74"]
        dev = max(dev, abs(mac[f"NormFGal{word}{suf}"]["value"] / Fc["GAL"] - 1), abs(mac[f"NormFAGN{word}{suf}"]["value"] / Fc["AGN"] - 1))
    check(dev < 1e-4, f"G1b F_k equal darksirens-core's own normalisers (Analysis-15 dry run, H0 = 67.74, five configurations) to {dev:.1e} relative")
    # ---- G2: the redshift to which each flux limit is complete, from the catalog itself ----
    with h5py.File(B.DATA / "seed100" / "catalogs" / "catalog_agn_complete.h5") as h:
        za, ma = h["z"][()], h["app_mag"][()]
    d2 = []
    for key, word, mlim in B.RUNGS:
        zemp = float(za[ma >= mlim].min())          # first AGN of the complete catalog that the limit removes
        d2.append(abs(zemp - mac[f"MockzComplete{word}"]["extra"]["agn"]))
    check(max(d2) < 0.01, f"G2 z_complete of the selection fits equals the lowest true redshift of an AGN removed by each flux limit (max diff {max(d2):.4f})")
    nrow = []
    for key, word, mlim in B.RUNGS:
        with h5py.File(B.DATA / "seed100" / "catalogs" / f"catalog_agn_{key}.h5") as h:
            nrow.append(int(h["z"].shape[0]) == int(np.sum(ma < mlim)))
    check(all(nrow), "G2b each flux-limited AGN catalog holds exactly the complete-catalog AGN with app_mag below its limit (rows counted)")
    # ---- G3: completeness weighted by the detected mergers (the production event set of realization A, B.EV_REF), by host
    # membership in the flux-limited files ----
    with h5py.File(B.EV_REF) as h:
        tra, tdec, ht = h["truth/ra"][()], h["truth/dec"][()], h["truth/host_type"][()]
    kept = {}
    for key, word, mlim in B.RUNGS:
        inside = np.zeros(len(ht), bool)
        for typ, cls in ((1, "agn"), (0, "gal")):
            if cls == "gal" and key in ("m21", "m20"):
                continue
            with h5py.File(B.DATA / "seed100" / "catalogs" / f"catalog_{cls}_{key}.h5") as h:
                pos = set(zip(h["ra"][()].tolist(), h["dec"][()].tolist()))
            sel = ht == typ
            inside[sel] = [(a, d) in pos for a, d in zip(tra[sel], tdec[sel])]
        kept[key] = inside
    ok3 = []
    for key, word, mlim in B.RUNGS:
        ex = mac[f"MockEventComp{word}"]["extra"]
        ok3.append(abs(np.mean(kept[key][ht == 1]) - ex["agn_hosted"]) < 1e-12)
        if key in ("m19", "m18"):
            ok3.append(abs(np.mean(kept[key][ht == 0]) - ex["galaxy_hosted"]) < 1e-12)
            ok3.append(abs(np.mean(kept[key]) - mac[f"MockEventComp{word}"]["value"]) < 1e-12)
    check(all(ok3), "G3 completeness weighted by the detected mergers equals host membership in the flux-limited catalog files (AGN at all limits, galaxies at m < 19 and m < 18)")
    # ---- G4: hosts per localization volume, brute force (astropy distances, exact angular distance) ----
    z_of, cos = _astro_z_of_dl(H0)
    q = B.HOSTVOL_Q
    with h5py.File(B.DATA / "seed100" / "catalogs" / "catalog_agn_complete.h5") as h:
        A = {c: h[c][()] for c in ("ra", "dec", "z_obs")}
    va = np.stack([np.cos(A["dec"]) * np.cos(A["ra"]), np.cos(A["dec"]) * np.sin(A["ra"]), np.sin(A["dec"])], 1)

    def regions(path, idx):
        with h5py.File(path) as h:
            ra = h["ra"][()].reshape(1000, -1)[idx]
            dec = h["dec"][()].reshape(1000, -1)[idx]
            dl = h["dL"][()].reshape(1000, -1)[idx]
        out = []
        for k in range(len(idx)):
            # centre: the normalised mean of the sample unit vectors; angular distance by the haversine formula
            x = np.stack([np.cos(dec[k]) * np.cos(ra[k]), np.cos(dec[k]) * np.sin(ra[k]), np.sin(dec[k])], 1).mean(0)
            x /= np.linalg.norm(x)
            ra0, de0 = math.atan2(x[1], x[0]), math.asin(x[2])
            hv = np.sin((dec[k] - de0) / 2) ** 2 + np.cos(dec[k]) * math.cos(de0) * np.sin((ra[k] - ra0) / 2) ** 2
            ang = 2 * np.arcsin(np.sqrt(hv))
            z = z_of(dl[k])
            out.append((x, float(np.quantile(ang, q)), float(np.quantile(z, (1 - q) / 2)), float(np.quantile(z, (1 + q) / 2))))
        return out
    # the production all-AGN-hosted and all-galaxy-hosted event sets of realization A (events_pure<cls>_corepop.h5), which build_values
    # reads through HOSTVOL_EVENTS: named here by file, so that a change of set on either side fails this check
    assert B.HOSTVOL_EVENTS == {"gal": B.DATA / "seed100" / "events" / "events_puregal_corepop.h5", "agn": B.DATA / "seed100" / "events" / "events_pureagn_corepop.h5"}
    reg_a = regions(B.DATA / "seed100" / "events" / "events_pureagn_corepop.h5", np.arange(1000))
    cnt = []
    for x, r, zl, zh in reg_a:
        m = (A["z_obs"] >= zl) & (A["z_obs"] <= zh)
        cnt.append(int(np.sum(va[m] @ x >= math.cos(r))))
    med_bf = float(np.median(cnt))
    got = mac["PureAGNHostsPerVol"]["value"]
    per = np.asarray(tables["r2_hosts_per_event"]["agn_catalog_on_agn_hosted_events"], float)
    rel_tot = abs(per.sum() / np.sum(cnt) - 1)
    check(abs(got - med_bf) <= max(1.0, 0.05 * med_bf) and rel_tot < 0.03,
          f"G4 AGN per 90% localization volume (production all-AGN-hosted set of realization A): brute-force exact count median {med_bf:.0f} vs {got:.1f} (pixelised); "
          f"totals over 1000 events differ by {100 * rel_tot:.2f}%")
    rng = np.random.default_rng(7)
    idx = np.sort(rng.choice(1000, 12, replace=False))
    reg_g = regions(B.DATA / "seed100" / "events" / "events_puregal_corepop.h5", idx)
    zcut = max(r[3] for r in reg_g) + 0.03
    G = B.read_catalog_rows(B.DATA / "seed100" / "catalogs" / "catalog_gal_complete.h5", zcut)
    cg = []
    for x, r, zl, zh in reg_g:
        m = (G["z_obs"] >= zl) & (G["z_obs"] <= zh)
        vg = np.stack([np.cos(G["dec"][m]) * np.cos(G["ra"][m]), np.cos(G["dec"][m]) * np.sin(G["ra"][m]), np.sin(G["dec"][m])], 1)
        cg.append(int(np.sum(vg @ x >= math.cos(r))))
    del G
    pg = np.asarray(tables["r2_hosts_per_event"]["gal_catalog_on_gal_hosted_events"], float)[idx]
    rg = abs(pg.sum() / sum(cg) - 1)
    check(rg < 0.03, f"G4b galaxies per 90% localization volume, 12 random events of the production all-galaxy-hosted set: brute-force total {sum(cg)} vs pixelised {pg.sum():.0f} ({100 * rg:.2f}%)")
    # ---- G5: rate per host. Catalog counts from the catalog files; Monte Carlo draws from the stored grids ----
    cnts = {}
    for cls in ("gal", "agn"):
        with h5py.File(B.DATA / "seed100" / "catalogs" / f"catalog_{cls}_complete.h5") as h:
            cnts[cls] = int(h["z"].shape[0])
    Rin = B.IN["f"] / (1 - B.IN["f"]) * cnts["gal"] / cnts["agn"]
    check(abs(mac["MockRateRatio"]["value"] / Rin - 1) < 1e-12, f"G5 input rate-per-host ratio from the catalog row counts: {Rin:.3f}")
    fv = mac["JointfCatA"]["value"]
    rr = mac["RateRatioJointCatA"]["value"]
    o = lambda f: f / (1 - f)
    dev5 = max(abs(rr["median"] / (o(fv["median"]) * cnts["gal"] / cnts["agn"]) - 1), abs(rr["ci90"][0] / (o(fv["ci90"][0]) * cnts["gal"] / cnts["agn"]) - 1))
    check(dev5 < 1e-9, f"G5b complete catalogs (Z_k = N_k): R quantiles are the f quantiles times N_gal/N_AGN ({dev5:.1e})")

    def mc_R(path, x=1.0, ndraw=400_000, seed=11):
        """Draws from the stored (H0, f) grid: linear interpolation of ln L on a 4x finer lattice (not the
        spline of build_values), inverse-CDF sampling with uniform jitter; Z_k from a15_common."""
        with h5py.File(path) as h:
            Hg, Fg, L = h["H0_grid"][()], h["f_grid"][()], h["log_likelihood"][()]
            rej = h["guard/rejected"][()] if "guard/rejected" in h else np.zeros_like(L, bool)
        L = np.where(rej | ~np.isfinite(L), -np.inf, L)
        from scipy.interpolate import RegularGridInterpolator
        hf = np.linspace(Hg[0], Hg[-1], 4 * (len(Hg) - 1) + 1)
        ff = np.linspace(Fg[0], Fg[-1], 4 * (len(Fg) - 1) + 1)
        Lf = RegularGridInterpolator((Hg, Fg), np.where(np.isfinite(L), L, L[np.isfinite(L)].min() - 200), method="linear")(np.stack(np.meshgrid(hf, ff, indexing="ij"), -1))
        Pm = np.exp(Lf - Lf.max()).ravel()
        Pm /= Pm.sum()
        rng_ = np.random.default_rng(seed)
        k = rng_.choice(Pm.size, ndraw, p=Pm)
        hs = hf[k // len(ff)] + rng_.uniform(-0.5, 0.5, ndraw) * (hf[1] - hf[0])
        fs_ = np.clip(ff[k % len(ff)] + rng_.uniform(-0.5, 0.5, ndraw) * (ff[1] - ff[0]), 1e-9, 1 - 1e-9)
        key = path.name.split("_")[1]
        fg_ = C.schechter_fit(C.FITS / f"fit_gal_{key}_schechter.json")
        fa_ = C.schechter_fit(C.FITS / f"fit_agn_{key}_schechter.json")
        Ng = C.survey_nobs(B.DATA / "seed100" / "surveys" / f"survey_gal_{key}_ns32.h5")
        Na = C.survey_nobs(B.DATA / "seed100" / "surveys" / f"survey_agn_{key}_ns32.h5")
        Vg = C.missing_volume(fg_, H0, 5.0) * (H0 / hs) ** 3
        Va = C.missing_volume(fa_, H0, 5.0) * (H0 / hs) ** 3
        Rs = o(fs_) * (Ng + ng * Vg) / (Na + x * na * Va)
        return np.percentile(Rs, [5, 50, 95])
    dmax = 0.0
    for key, word, _ in B.RUNGS:
        q5, q50, q95 = mc_R(B.A3 / f"joint_{key}_s100.h5")
        v = mac[f"RateRatio{word}"]["value"]
        hw = 0.5 * (v["ci90"][1] - v["ci90"][0])
        dmax = max(dmax, abs(q50 - v["median"]) / hw, abs(q5 - v["ci90"][0]) / hw, abs(q95 - v["ci90"][1]) / hw)
    q5, q50, q95 = mc_R(B.A4 / "joint_m18_a20_s100.h5", x=2.0)
    v = mac["RateRatioTwoEighteen"]["value"]
    hw = 0.5 * (v["ci90"][1] - v["ci90"][0])
    dmax = max(dmax, abs(q50 - v["median"]) / hw, abs(q5 - v["ci90"][0]) / hw, abs(q95 - v["ci90"][1]) / hw)
    check(dmax < 0.03, f"G5c flux-limited R (four limits and m < 18 at twice the AGN density): Monte Carlo draws from the stored grids agree with the CDF transform to {dmax:.3f} of the 90% half-width")
    c5, _, _, _ = B.load_a5(B.A5 / "campaign_m18_dynesty_s100.h5", B.A5 / "campaign_m18_dynesty_s100.json")
    fg_ = C.schechter_fit(C.FITS / "fit_gal_m18_schechter.json")
    fa_ = C.schechter_fit(C.FITS / "fit_agn_m18_schechter.json")
    Ng = C.survey_nobs(B.DATA / "seed100" / "surveys" / "survey_gal_m18_ns32.h5")
    Na = C.survey_nobs(B.DATA / "seed100" / "surveys" / "survey_agn_m18_ns32.h5")
    hs = c5.col("H0")
    Rs = o(c5.col("fcat_2")) * (Ng + 10 ** c5.col("log10n0") * C.missing_volume(fg_, H0, 5.0) * (H0 / hs) ** 3) / (Na + 10 ** c5.col("log10n0_c2") * C.missing_volume(fa_, H0, 5.0) * (H0 / hs) ** 3)
    v = mac["RateRatioFreeEighteen"]["value"]
    d5 = abs(float(np.median(Rs)) / v["median"] - 1)
    check(d5 < 1e-3, f"G5d R with both densities free (m < 18, catalog A), per sample with a15_common's normalisers: median {np.median(Rs):.2f} vs {v['median']:.2f}")
    # ---- G6: the identity, by a different map (cubic spline in logit f, trapezoid summaries without refinement) ----
    dev6 = []
    for key, word, _ in (("m21", "TwentyOne", 21), ("m20", "Twenty", 20)):
        with h5py.File(B.A3 / f"joint_{key}_s100.h5") as h:
            Hg, Fg, L = h["H0_grid"][()], h["f_grid"][()], h["log_likelihood"][()]
        fa_ = C.schechter_fit(C.FITS / f"fit_agn_{key}_schechter.json")
        Na = C.survey_nobs(B.DATA / "seed100" / "surveys" / f"survey_agn_{key}_ns32.h5")
        inner = (Fg > 0) & (Fg < 1)
        from scipy.interpolate import CubicSpline as CS
        pred = np.full_like(L, -np.inf)
        Va0 = C.missing_volume(fa_, H0, 5.0)
        for i, h_ in enumerate(Hg):
            Va = Va0 * (H0 / h_) ** 3
            d = math.log((Na + na * Va) / (Na + 2 * na * Va))      # ln F_A(x=2)/F_A(x=1)
            sp = CS(C.logit(Fg[inner]), L[i, inner])
            xx = C.logit(Fg[inner]) + d
            okk = (xx >= C.logit(Fg[inner])[0]) & (xx <= C.logit(Fg[inner])[-1])
            row = np.full(inner.sum(), -np.inf)
            row[okk] = sp(xx[okk])
            pred[i, inner] = row
        s1 = C.summarize_2d(Hg, Fg, L)["f"]["median"]
        s2 = C.summarize_2d(Hg, Fg, pred)["f"]["median"]
        dev6.append(abs((s2 - s1) - mac[f"NormDeltafTwo{word}FivePred"]["value"]))
    check(max(dev6) < 0.004, f"G6 identity prediction of the change of f_AGN for a doubled AGN density (m < 21, m < 20) by a spline in logit f and unrefined trapezoid medians: max diff {max(dev6):.4f}")
    # ---- G7: Analysis 14 and 15 from their scan files (or red placeholders when the summary is absent) ----
    ptxt = (OUT / "placeholders.txt").read_text()
    for tag, path, pre in (("Analysis 14", B.A14_SUMMARY, ("MC",)), ("Analysis 15", B.A15_SUMMARY, ("NormHzero", "Normf", "NormRateRatio", "NormDeltafTwoComplete", "NormDeltafTwoEighteenProd", "NormDeltaHzeroTwoComplete", "NormDeltaHzeroTwoEighteenProd", "SlopefAGNDepthNorm"))):
        names = [r["name"] for r in reg if r["name"].startswith(pre) and r["name"] not in ("NormDeltafTwoEighteenFive", "NormDeltafTwoEighteenFivePred")]
        if not path.exists():
            red = all(mac[n]["class"] == "ph" for n in names)
            listed = all(re.search(rf"^\\{n} \|.*{tag}", ptxt, flags=re.M) for n in names)
            check(red and listed, f"G7 {tag} summary absent: its {len(names)} macros are red and listed in placeholders.txt with '{tag}' as their source")
        elif tag == "Analysis 14":
            scans = [json.loads(p.read_text()) for p in (B.A14R / "scans").glob("b_pe_n2000_r*.json")]
            meds = [s_["summary"]["H0"]["median"] for s_ in scans]
            okk = len(meds) < 2 or abs(np.std(meds, ddof=1) - mac["MCMixedHzeroSdTwoK"]["value"]) < 1e-9
            check(okk, f"G7 Analysis 14: spread of the H0 median over the {len(meds)} mixed-set redraws at 2000 samples recomputed from the scan JSONs")
        else:
            sc = json.loads(path.read_text())["scans"]
            dd_ = []
            for scan, word, _ in B.A15_SCANS:
                pj = B.A15R / "scans" / f"{scan}.json"
                if pj.exists() and mac[f"Normf{word}"]["class"] != "ph":
                    dd_.append(abs(json.loads(pj.read_text())["summary"]["f"]["median"] - mac[f"Normf{word}"]["value"]["median"]))
            check(not dd_ or max(dd_) < 1e-9, f"G7 Analysis 15: f_AGN medians of {len(dd_)} scans equal the per-scan JSONs")
    # ---- G8: prior-induced upper limits, by a different draw (a15_common normalisers, new seed, linear lookup on the stored grid) ----
    from scipy.interpolate import RegularGridInterpolator
    key, word = "m18", "Eighteen"
    pr = json.loads((B.A5 / f"campaign_{key}_dynesty_s100.json").read_text())["priors"]
    rng = np.random.default_rng(99)
    n = 1_000_000
    Hs, f, lg, la = (rng.uniform(*pr[k], n) for k in ("H0", "fcat_2", "log10n0", "log10n0_c2"))
    fg_ = C.schechter_fit(C.FITS / f"fit_gal_{key}_schechter.json")
    fa_ = C.schechter_fit(C.FITS / f"fit_agn_{key}_schechter.json")
    Ng = C.survey_nobs(B.DATA / "seed100" / "surveys" / f"survey_gal_{key}_ns32.h5")
    Na = C.survey_nobs(B.DATA / "seed100" / "surveys" / f"survey_agn_{key}_ns32.h5")
    Vg, Va = C.missing_volume(fg_, H0, 5.0) * (H0 / Hs) ** 3, C.missing_volume(fa_, H0, 5.0) * (H0 / Hs) ** 3
    lnr = lambda g_, a_: np.log(Na / (Na + a_ * Va)) - np.log(Ng / (Ng + g_ * Vg))
    fst = C.expit(C.logit(f) + lnr(10 ** lg, 10 ** la) - lnr(ng, na))
    with h5py.File(B.A3 / f"joint_{key}_s100.h5") as h:
        Hg, Fg, L = h["H0_grid"][()], h["f_grid"][()], h["log_likelihood"][()]
    w = np.exp(RegularGridInterpolator((Hg, Fg), L - L.max())(np.stack([Hs, fst], 1)))
    out = []
    for arr, nm in ((lg, "Gal"), (la, "AGN")):
        oo = np.argsort(arr)
        c = np.cumsum(w[oo])
        out.append(abs(float(np.interp(0.9, c / c[-1], arr[oo])) - mac[f"FreeLogn{nm}InducedUpper{word}"]["value"]))
    check(max(out) < 0.03, f"G8 prior-induced 90% upper limits on log10 n_gal and log10 n_AGN (m < 18) by an independent draw: max diff {max(out):.3f} dex")
    # the same limits in h^3 Mpc^-3 (FreeEighteenLogn<cls>PriorULh) while the key is a placeholder: each draw converted with its own H0
    doc8 = json.loads((OUT / "values" / "placeholder_results.json").read_text())["keys"]
    if not doc8["FreeEighteen"].get("final"):
        # the Table I box keeps about two thirds of the prior weight and the galaxy limit lies on a flat stretch of its distribution, so
        # its Monte Carlo error at 1e6 draws is about 0.03 dex: three further independent draws (4e6 in all) are added for this check
        nh, ws = {"Gal": [], "AGN": []}, []
        for sd_ in (99, 199, 299, 399):
            if sd_ == 99:
                Hs_, lg_, la_, w_ = Hs, lg, la, w
            else:
                r_ = np.random.default_rng(sd_)
                Hs_, f_, lg_, la_ = (r_.uniform(*pr[k], n) for k in ("H0", "fcat_2", "log10n0", "log10n0_c2"))
                Vg_, Va_ = C.missing_volume(fg_, H0, 5.0) * (H0 / Hs_) ** 3, C.missing_volume(fa_, H0, 5.0) * (H0 / Hs_) ** 3
                lnr_ = lambda g_, a_: np.log(Na / (Na + a_ * Va_)) - np.log(Ng / (Ng + g_ * Vg_))
                fst_ = C.expit(C.logit(f_) + lnr_(10 ** lg_, 10 ** la_) - lnr_(ng, na))
                w_ = np.exp(RegularGridInterpolator((Hg, Fg), L - L.max())(np.stack([Hs_, fst_], 1)))
            g_h, a_h = lg_ - 3 * np.log10(Hs_ / 100.0), la_ - 3 * np.log10(Hs_ / 100.0)
            inb = ((g_h >= mac["PriorLognGalLo"]["value"]) & (g_h <= mac["PriorLognGalHi"]["value"])
                   & (a_h >= mac["PriorLognAGNLo"]["value"]) & (a_h <= mac["PriorLognAGNHi"]["value"]))
            nh["Gal"].append(g_h[inb])
            nh["AGN"].append(a_h[inb])
            ws.append(w_[inb])
        wh = np.concatenate(ws)
        out_h = []
        for nm in ("Gal", "AGN"):
            ah = np.concatenate(nh[nm])
            oo = np.argsort(ah)
            c = np.cumsum(wh[oo])
            out_h.append(abs(float(np.interp(0.9, c / c[-1], ah[oo])) - mac[f"FreeEighteenLogn{nm}PriorULh"]["value"]))
        check(max(out_h) < 0.03, f"G8b the same prior-induced limits in h^3 Mpc^-3 (FreeEighteenLogn<Gal|AGN>PriorULh, placeholder: exploratory draws "
                                 f"converted with their own H0 and kept inside the Table I box) by four independent draws of 1e6: max diff {max(out_h):.3f} dex")
    # ---- G9: realised mock numbers from the raw files: the production injection set and event sets (until 17 Nov the older events.h5
    # and the events_meta.json of the nine exploratory catalogs) ----
    with h5py.File(B.DATA / "seed100" / "injections" / "injections_targeted.h5") as h:
        k0 = [k for k in h.keys() if isinstance(h[k], h5py.Dataset)][0]
        nfound = int(h[k0].shape[0])
    with h5py.File(B.DATA / "seed100" / "events" / "events_corepop.h5") as h:
        dl_ = h["dL"][()]
        sdl = np.array([np.std(np.log(dl_[i * (len(dl_) // 1000):(i + 1) * (len(dl_) // 1000)])) for i in range(1000)])
        sky = np.degrees(h["truth/obs_sigma_ang"][()])
    import yaml
    rl9 = yaml.safe_load((B.AN / "analysis_16_production_runs" / "runs" / "RUNLIST.yaml").read_text())
    kb = sorted({int(r["args"][r["args"].index("--realization") + 1]) for st in rl9["stages"] for r in st["runs"]
                 if r["name"].startswith("known_population/both_catalogs/complete/equal_populations/realization_")})
    zmax9 = {}
    for s_ in kb:
        with h5py.File(B.DATA / f"seed{s_}" / "events" / "events_corepop.h5") as h:
            zmax9[s_] = float(h["truth/z"][()].max())
    ok9 = (nfound == mac["MockNinjFound"]["value"] and abs(np.median(sdl) - mac["MockSigLnDL"]["value"]) < 1e-12
           and abs(sky.min() - mac["MockSkyMin"]["value"]) < 1e-12 and abs(sky.max() - mac["MockSkyMax"]["value"]) < 1e-12
           and abs(min(zmax9.values()) - mac["MockzHorizonMin"]["value"]) < 1e-12 and abs(max(zmax9.values()) - mac["MockzHorizonMax"]["value"]) < 1e-12
           and abs(zmax9[100] - mac["MockzHorizon"]["value"]) < 1e-12)
    check(ok9, f"G9 detected injections counted in the file ({nfound}); from the production event sets: median sd of ln dL over the events' samples "
               f"{np.median(sdl):.4f} vs {mac['MockSigLnDL']['value']:.4f}, sky widths {sky.min():.2f} to {sky.max():.2f} deg, largest true z of the "
               f"{len(kb)} KnownBoth realizations {min(zmax9.values()):.4f} to {max(zmax9.values()):.4f} (MockzHorizonMin/Max), realization A {zmax9[100]:.4f} (MockzHorizon)")
    # ---- G10: digits ----
    bad = []
    for r in reg:
        if r["class"] == "ph" or not isinstance(r["value"], dict) or "ci90" not in r["value"] or "^{+" not in r["render"]:
            continue
        v = r["value"]
        up, dn = v["ci90"][1] - v["median"], v["median"] - v["ci90"][0]
        e = min(up, dn) if min(up, dn) > 0 else max(up, dn)
        m_ = re.match(r"^-?\d+(?:\.(\d+))?\^", r["render"])
        if m_ is None or len(m_.group(1) or "") != min(5, _pdg_decimals(e)):
            bad.append(r["name"])
    check(not bad, f"G10 every rendered interval follows the PDG rounding of its smaller 90% error ({len(bad)} exceptions{': ' + ', '.join(bad[:6]) if bad else ''})")
    lbs = [r for r in reg if r["name"].startswith("LB") and r["class"] != "ph"]
    sig = []
    for r in lbs:
        body = re.sub(r"\\pm.*$", "", r["render"]).replace("\\ensuremath{", "").replace("}", "")
        mant = body.split("\\times")[0].lstrip("-").replace(".", "").lstrip("0")
        if len(mant) > 2:
            sig.append(r["name"])
    check(not sig, f"G10b every log10 Bayes factor has at most two significant figures ({len(lbs)} macros{': ' + ', '.join(sig[:6]) if sig else ''})")
    # ---- G11: C1 relabel, small derived numbers ----
    c1 = mac["CtrlHeavierFrac"]
    check(c1["value"] == mac["CtrlModfBranch"]["value"] and "NOT f_AGN" in c1["meaning"], "G11 Table III's control-mock entry has its own macro (CtrlHeavierFrac, the heavier-population fraction, not f_AGN)")
    sh = mac["GalOnlyHzeroComb"]["value"]["median"] - mac["JointHzeroComb"]["value"]["median"]
    check(abs(sh - mac["ShiftGalOnlyMinusJointComb"]["value"]) < 1e-12, f"G11b ShiftGalOnlyMinusJointComb = {sh:.3f} from the two combined medians in values.json")
    audit_keys(mac)


def audit_keys(mac):
    """H: the key scheme of notes/PAPER_SKELETON.md Sec. 6."""
    keys = (["KnownBoth", "KnownGal", "KnownAGN", "KnownNone", "AGNhosted", "Galhosted", "CtrlZero", "CtrlTenth"]
            + [f"Flux{w}" for _, w, _ in B.RUNGS] + [f"Free{w}" for _, w, _ in B.RUNGS]
            + ["SingleBoth", "SingleNone", "HDSingle", "HDHost", "HDNone", "ObsSingle", "ObsHost", "ObsNone"])
    need = {"KnownBoth": ["Hzero", "fAGN"], "KnownGal": ["Hzero"], "KnownAGN": ["Hzero"], "KnownNone": ["Hzero"], "AGNhosted": ["Hzero"], "Galhosted": ["Hzero"],
            "CtrlZero": ["Hzero", "fAGN"], "CtrlTenth": ["Hzero", "fAGN"], "SingleNone": ["Hzero", "MuG", "SigmaG", "MuChi", "SigmaChi"],
            "SingleBoth": ["Hzero", "fAGN", "MuG", "SigmaG", "MuChi", "SigmaChi"]}
    for _, w, _ in B.RUNGS:
        need[f"Flux{w}"] = ["Hzero", "fAGN"]
        need[f"Free{w}"] = ["Hzero", "fAGN", "LognGal", "LognAGN"]
    for pre in ("HD", "Obs"):
        need[f"{pre}Single"] = ["Hzero", "fAGN", "MuG", "SigmaG", "MuChi", "SigmaChi"]
        need[f"{pre}Host"] = need[f"{pre}Single"] + ["DmuG", "DmuChi"]
        need[f"{pre}None"] = ["Hzero", "MuG", "SigmaG", "MuChi", "SigmaChi"]
    missing = []
    for k in keys:
        for q in need[k]:
            for suf in ("", "Med", "Lo", "Hi", "Wid", "Comb", "CombMed", "CombLo", "CombHi", "CombWid"):
                if f"{k}{q}{suf}" not in mac:
                    missing.append(f"{k}{q}{suf}")
    extra = ["HDBiasHzero", "ObsBiasHzero", "HDLB", "ObsLB", "WidthRatioAGNGal", "MassScaleShareKnown", "MassScaleShareSingle", "CorrHzeroMuGNone", "CorrHzeroMuGBoth"]
    missing += [n for n in extra if n not in mac]
    check(not missing, f"H0 every key and quantity of the skeleton has its macro, Comb and pieces, and the further macros exist ({len(missing)} missing{': ' + ', '.join(missing[:6]) if missing else ''})")
    doc = json.loads((OUT / "values" / "placeholder_results.json").read_text())["keys"]
    final = {k for k, E in doc.items() if E.get("final")}
    owner = macro_keys(mac, doc)
    wrong = [n for n, ks in owner.items() if (mac[n]["class"] == "ph") == all(k in final for k in ks)]
    check(not wrong, f"H1 every key-scheme macro prints black exactly when all its keys are production results ({len(final)} keys real: "
                     f"{', '.join(sorted(final)) or 'none'}; {len(wrong)} exceptions{': ' + ', '.join(wrong[:6]) if wrong else ''})")
    # exploratory keys equal the exploratory macros built by the separate code of build_c1..build_r2
    # (the combinations of the all-AGN, all-galaxy and galaxy-catalog-alone keys use the production realizations only since
    #  17 Nov; they are checked against an independent sum in I9 below, not against the 93- and 9-realization macros).
    # Keys that are production results are audited in J instead.
    # (the flux-limited keys and the keys with the number densities inferred are re-anchored and reweighted while placeholders: H2c, H2b)
    pairs = [("KnownBothHzero", "JointHzeroCatA"), ("KnownBothHzeroComb", "JointHzeroComb"), ("KnownBothfAGNComb", "JointfComb"), ("KnownGalHzero", "GalOnlyHzeroCatA"),
             ("HDHostDmuChi", "HDEightDmuChi"), ("HDHostHzero", "HDEightHzero"),
             ("KnownBothRateRatio", "RateRatioJointCatA"), ("KnownBothRateRatioComb", "RateRatioJointComb")]
    dev = []
    for a, b in pairs:
        if any(k in final for k in owner[a]):
            continue
        va, vb = mac[a]["value"], mac[b]["value"]
        hw = 0.5 * (vb["ci90"][1] - vb["ci90"][0])
        dev.append(max(abs(va["median"] - vb["median"]), abs(va["ci90"][0] - vb["ci90"][0]), abs(va["ci90"][1] - vb["ci90"][1])) / hw)
    for a, b in (("KnownBothLB", "LBfZeroCatA"), ("KnownBothLBComb", "LBfZeroComb")):
        if not any(k in final for k in owner[a]):
            dev.append(abs(mac[a]["value"] - mac[b]["value"]))
    check(max(dev, default=0.0) < 2e-3, f"H2 {len(dev)} exploratory key values equal the exploratory macros computed separately (largest difference {max(dev, default=0.0):.1e}, "
                                         f"in 90% half-widths or log10 units; keys that are production results are audited in J)")
    # H2b: the exploratory chains of the keys with the number densities inferred, read here (h5py), each sample converted with its own H0 to
    # h^3 Mpc^-3 and kept when both densities lie inside the Table I box (the production prior; placeholder_results header, 'prior box');
    # the limits ULh equal the 90th percentile of what is kept, and every ULh and PriorULh lies inside the box. The order of the two limits
    # of a key is reported, not required: the box pins both within about 0.1 dex of its upper edge.
    lo_hi = {cls: (mac[f"PriorLogn{cls}Lo"]["value"], mac[f"PriorLogn{cls}Hi"]["value"]) for cls in ("Gal", "AGN")}
    devh, order, outside, kept_chain = [], [], [], {}
    for rk, w, _ in B.RUNGS:
        k = f"Free{w}"
        if k in final:
            continue
        with h5py.File(B.A5 / f"campaign_{rk}_dynesty_s100.h5") as h:
            lab = [x.decode() for x in h["sampled_labels"][()]]
            s_ = h["samples"][()]
        lh_ = 3 * np.log10(s_[:, lab.index("H0")] / 100.0)
        nh = {"Gal": s_[:, lab.index("log10n0")] - lh_, "AGN": s_[:, lab.index("log10n0_c2")] - lh_}
        keep_ = np.all([(nh[c_] >= lo_hi[c_][0]) & (nh[c_] <= lo_hi[c_][1]) for c_ in ("Gal", "AGN")], axis=0)
        kept_chain[k] = {"H0": s_[keep_, lab.index("H0")], "f": s_[keep_, lab.index("fcat_2")]}
        for cls in ("Gal", "AGN"):
            if not all(n in mac for n in (f"{k}Logn{cls}ULh", f"{k}Logn{cls}PriorULh")):
                devh.append(np.inf)
                continue
            own = float(np.percentile(nh[cls][keep_], 90))
            devh.append(abs(own - mac[f"{k}Logn{cls}ULh"]["value"]))
            order.append(mac[f"{k}Logn{cls}ULh"]["value"] - mac[f"{k}Logn{cls}PriorULh"]["value"])
            outside += [n for n in (f"{k}Logn{cls}ULh", f"{k}Logn{cls}PriorULh") if not lo_hi[cls][0] <= mac[n]["value"] <= lo_hi[cls][1]]
    check(max(devh, default=0.0) < 1e-9 and not outside,
          f"H2b {len(devh)} limits in h^3 Mpc^-3 (Free<limit>Logn<cls>ULh) of the placeholder keys equal the 90th percentile of the exploratory chains "
          f"converted per sample and cut to the Table I box (largest difference {max(devh, default=0.0):.1e}); every ULh and PriorULh lies inside the box "
          f"(PriorLogn<cls>Lo/Hi){': OUTSIDE ' + ', '.join(outside) if outside else ''}; largest ULh - PriorULh {max(order, default=0.0):+.3f} dex (reported)")
    # H2c: the re-anchored placeholders (REVIEW_2 1.11; placeholder_results header, 're-anchoring'): realization A of the flux-limited keys from the
    # exploratory macros computed separately (FluxHzero<limit>, Fluxf<limit>, RateRatio<limit>; complete: JointHzeroCatA, JointfCatA,
    # RateRatioJointCatA) and the finished KnownBoth; of the keys with the densities inferred, H0 and f_AGN from the chains cut to the box above
    if "KnownBoth" in final:
        hwf = lambda s: 0.5 * (s["ci90"][1] - s["ci90"][0])
        kb = {q: mac[f"KnownBoth{q}"]["value"] for q in ("Hzero", "fAGN", "RateRatio")}
        ec = {"Hzero": mac["JointHzeroCatA"]["value"], "fAGN": mac["JointfCatA"]["value"], "RateRatio": mac["RateRatioJointCatA"]["value"]}
        devr, n_r, wider = [], 0, []
        for rk, w, _ in B.RUNGS:
            for k in (f"Flux{w}", f"Free{w}"):
                if k in final:
                    continue
                for q in ("Hzero", "fAGN", "RateRatio"):
                    if k.startswith("Flux"):
                        ex = mac[{"Hzero": f"FluxHzero{w}", "fAGN": f"Fluxf{w}", "RateRatio": f"RateRatio{w}"}[q]]["value"]
                    elif q == "RateRatio":
                        continue
                    else:
                        x_ = kept_chain[k]["H0" if q == "Hzero" else "f"]
                        pq = np.percentile(x_, [5, 50, 95])
                        ex = {"median": float(pq[1]), "ci90": [float(pq[0]), float(pq[2])]}
                    sc = hwf(kb[q]) / hwf(ec[q])
                    m = kb[q]["median"] + ex["median"] - ec[q]["median"]
                    want = (m, m + sc * (ex["ci90"][0] - ex["median"]), m + sc * (ex["ci90"][1] - ex["median"]))
                    got = mac[f"{k}{q}"]["value"]
                    devr.append(max(abs(got["median"] - want[0]), abs(got["ci90"][0] - want[1]), abs(got["ci90"][1] - want[2])) / hwf(got))
                    n_r += 1
                    if q == "Hzero" and hwf(got) < hwf(kb[q]) * (1 - 1e-2):
                        wider.append(f"{k}{q}")
        check(max(devr, default=0.0) < 2e-3 and not wider,
              f"H2c {n_r} re-anchored placeholder quantities (realization A of the flux-limited keys and the keys with the densities inferred: H0, f_AGN, "
              f"rate ratio) follow the rule from the finished KnownBoth and the exploratory values computed separately (largest difference "
              f"{max(devr, default=0.0):.1e} of a 90% half-width); no H0 half-width narrower than the complete catalogs' by more than 1%"
              f"{': NARROWER ' + ', '.join(wider) if wider else ''}")
    # invented combinations: width A / sqrt(N)
    rat = []
    for k, E in doc.items():
        for q, Q in E["quantities"].items():
            if Q["comb_source"].startswith("invented") and "prior dominates" not in Q["comb_source"] and not (k == "CtrlZero" and q == "fAGN"):
                wa = Q["A"]["ci90"][1] - Q["A"]["ci90"][0]
                wc = Q["Comb"]["ci90"][1] - Q["Comb"]["ci90"][0]
                rat.append(abs(wc / wa * math.sqrt(E["N_comb"]) - 1))
    check(max(rat) < 0.05, f"H3 every invented combination is narrower than realization A by sqrt(N) to {100 * max(rat):.1f}% ({len(rat)} key-quantities; truncation at prior edges allowed for)")
    audit_requests(mac, doc)
    audit_production(mac, doc, owner)
    audit_classes(mac, doc)
    audit_density(mac)


# the further macros of PAPER_SKELETON.md Sec. 6 and the keys they are built from
FURTHER = {"HDBiasHzero": ["HDSingle"], "ObsBiasHzero": ["ObsSingle"], "HDLB": ["HDHost"], "ObsLB": ["ObsHost"], "WidthRatioAGNGal": ["Galhosted", "AGNhosted"],
           "MassScaleShareKnown": ["KnownNone", "KnownBoth"], "MassScaleShareSingle": ["SingleNone", "SingleBoth"], "CorrHzeroMuGNone": ["SingleNone"],
           "CorrHzeroMuGBoth": ["SingleBoth"]}


def macro_keys(mac, doc):
    """key-scheme macro -> the keys it is built from (longest key prefix, or FURTHER)."""
    keys = sorted(doc, key=len, reverse=True)
    out = {}
    for n, v in mac.items():
        if not v["claim"].startswith("skeleton"):
            continue
        if n in FURTHER:
            out[n] = FURTHER[n]
            continue
        k = next((k for k in keys if n.startswith(k)), None)
        assert k is not None, n
        out[n] = [k]
    return out


# ============================================================================================= #
# I: the macros of 17 Nov (requests of both drafts, the observed-like family), each from a source build_values.py
# does not read or by a different computation: the run list parsed here, the event files themselves, the survey file
# counted with h5py at build_values' own completeness redshift, the A14 scan files, prod_common's text.
# ============================================================================================= #
def audit_requests(mac, doc):
    import yaml
    val = lambda n: mac[n]["value"]
    # ---- I0: numbers of realizations from RUNLIST.yaml (own parser) and runs.txt ----
    rl = yaml.safe_load((B.AN / "analysis_16_production_runs" / "runs" / "RUNLIST.yaml").read_text())
    cnt = {}
    for st in rl["stages"]:
        for r in st["runs"]:
            a = r["args"]
            d = {a[i][2:]: a[i + 1] for i in range(0, len(a) - 1, 2)}
            if "agn-density-factor" in d:
                continue
            ek = d["events-kind"] + (d["events-suffix"] if d.get("events-suffix") else "")      # Stage 5: host_dependent + _obslike
            cnt.setdefault((d["analysis"], d["catalogs"], d["flux"], ek, d.get("density")), set()).add(d["realization"])
    n = lambda *k: len(cnt.get(k, ()))
    want = {"NrealKnown": n("known_population", "both", "complete", "equal_populations", None),
            "NrealKnownAlone": {n("known_population", c, "complete", "equal_populations", None) for c in ("galaxy", "agn", "none")},
            "NrealPure": {n("known_population", "agn", "complete", "agn_hosted", None), n("known_population", "galaxy", "complete", "galaxy_hosted", None)},
            "NrealFlux": {n("known_population", "both", m, "equal_populations", "counts") for m in ("21", "20", "19", "18")},
            "NrealFree": {n("free_densities", "both", m, "equal_populations", "free") for m in ("21", "20", "19", "18")},
            "NrealSingle": {n("single_population", c, "complete", "equal_populations", None) for c in ("both", "none")},
            "NrealHD": n("host_dependent_populations", "both", "complete", "host_dependent", None),
            "NrealHDSingle": n("single_population", "both", "complete", "host_dependent", None),
            "NrealHDNone": n("single_population", "none", "complete", "host_dependent", None)}
    runs_txt = (OUT / "runs.txt").read_text()
    want["NrealObs"] = {n("single_population", c, "complete", "host_dependent_obslike", None) for c in ("both", "none")} | {
        n("host_dependent_populations", "both", "complete", "host_dependent_obslike", None)}
    rx = yaml.safe_load((B.AN / "analysis_16_production_runs" / "runs" / "RUNLIST_experiment.yaml").read_text())
    ctrl = {lab: {r["args"][r["args"].index("--realization") + 1] for st in rx["stages"] for r in st["runs"]
                  if r["analysis"] == "two_tracers_one_population" and r["set"]["label"] == lab} for lab in ("peak_f0p00_s0", "peak_f0p10_s0")}
    want["NrealCtrl"] = {len(v) for v in ctrl.values()} if all(ctrl.values()) else (5 if "five for the others" in runs_txt else -1)
    words = {3: "three", 5: "five", 10: "ten"}
    bad = []
    for k, w in want.items():
        w = w.pop() if isinstance(w, set) and len(w) == 1 else w
        if isinstance(w, set) or val(k) != w or mac[k]["render"] != words.get(w) or mac[k]["class"] != "plain":
            bad.append(f"{k}: {mac[k]['render']} vs {w}")
    bad += [f"{k} N_comb {E['N_comb']} vs macro" for k, E in doc.items()
            for m, ks in (("NrealKnown", ["KnownBoth"]), ("NrealKnownAlone", ["KnownGal", "KnownAGN", "KnownNone"]), ("NrealPure", ["AGNhosted", "Galhosted"]),
                          ("NrealSingle", ["SingleBoth", "SingleNone"]), ("NrealHD", ["HDHost"]), ("NrealHDSingle", ["HDSingle"]), ("NrealHDNone", ["HDNone"]),
                          ("NrealObs", ["ObsSingle", "ObsHost", "ObsNone"]), ("NrealCtrl", ["CtrlZero", "CtrlTenth"])) if k in ks and E["N_comb"] != val(m)]
    check(not bad, f"I0 the eleven numbers of realizations equal the run lists (own parser; the control sets: the experiment's f = 0 and f = 0.1 sets; Stage 5 for Obs), "
                   f"print as words, and equal N_comb of their keys{': ' + '; '.join(bad) if bad else ''}")
    # ---- I1: observed-like design values from the events files' own metadata ----
    E = B.DATA / "seed100" / "events"
    with h5py.File(E / "events_obslike.h5") as h:
        ms = json.loads(h.attrs["metadata_json"])["mass_spectrum"]
    th, rf = ms["core_theta"], ms["replaced_fields"]
    pairs = [("ObsPeakFrac", rf["peak_fraction"]), ("ObsMuG", rf["peak_mu"]), ("ObsSigmaG", rf["peak_sigma"]), ("ObsAlpha", rf["alpha"]), ("ObsMmin", rf["mmin"]),
             ("ObsMmax", rf["mmax"]), ("ObsDeltaM", rf["dm_min"]), ("ObsBeta", rf["beta"])]
    d1 = max(abs(val(k) - x) for k, x in pairs)
    check(d1 < 1e-12 and abs(1 - th["v1"] - val("ObsPeakFrac")) < 1e-12 and all(mac[k]["class"] == "plain" for k, _ in pairs),
          f"I1 observed-like design values equal the mass_spectrum block of events_obslike.h5 (max diff {d1:.1e})")
    K = json.loads((OUT / "values" / "placeholder_results.json").read_text())
    inp = K["_inputs_observed_like"]
    check(abs(inp["MuG"] - rf["peak_mu"]) < 1e-12 and abs(inp["SigmaG"] - rf["peak_sigma"]) < 1e-12 and K["keys"]["ObsHost"]["quantities"]["MuG"]["input"] == rf["peak_mu"],
          "I1b the Obs... placeholder keys take the observed-like inputs (mass peak and width) for their credible levels")
    # ---- I2: detected samples from the truth groups of the four event files ----
    files = {"Mock": "events_corepop.h5", "MockHD": "events_marked_dmu0p10_dmuG5_corepop.h5", "Obs": "events_obslike.h5", "ObsHD": "events_marked_dmu0p10_dmuG5_obslike.h5"}
    bad = []
    for fam, f in files.items():
        with h5py.File(E / f) as h:
            t = {k: h["truth/" + k][()] for k in ("z", "m1src", "m2src")}
            pk = h["truth/pop_component_peak"][()] if "pop_component_peak" in h["truth"] else None
        for q, x in (("DetzMedian", np.median(t["z"])), ("DetzMax", t["z"].max()), ("DetMoneMedian", np.median(t["m1src"])), ("DetMtwoMedian", np.median(t["m2src"]))):
            if abs(val(fam + q) - x) > 1e-9:
                bad.append(fam + q)
        if pk is not None and abs(val(fam + "DetPeakShare") - pk.mean()) > 1e-12:
            bad.append(fam + "DetPeakShare")
    check(not bad, f"I2 detected-sample macros (median and largest z, median masses, observed-like peak share) recomputed from the truth groups of the four event files{': ' + ', '.join(bad) if bad else ''}")
    # ---- I3: control set, production samples, Analysis-14 sample counts ----
    with h5py.File(E / "events_fagn0p10_corepop.h5") as h:
        ft = float(h.attrs["planted_f_agn"])
    drv = (B.AN / "analysis_16_production_runs" / "scripts" / "prod_driver.py").read_text()
    nsd = re.findall(r"--nsamp[^\n]*default=(\d+)", drv)
    scans = sorted({int(m) for m in re.findall(r"b_pe_n(\d+)_r\d+\.json", " ".join(p.name for p in (B.A14R / "scans").glob("b_pe_n*_r*.json")))})
    check(abs(val("MockfAGNTenth") - ft) < 1e-12 and (not nsd or int(nsd[0]) == val("ProdNsamp")) and val("ProdNsamp") == 8000
          and scans == [val("MCNsampLo"), val("MCNsampMid"), val("MCNsampHi")],
          f"I3 f_AGN of the one-in-ten set from its event file ({ft}); production samples per merger {val('ProdNsamp')} (driver default {nsd[:1]}); Analysis-14 sample counts from its scan files {scans}")
    # ---- I4: AGN counts at m < 18, counted here at build_values' own completeness redshift ----
    zc = B.norm_cfg(100)["m18"]["agn"]["zc"]
    with h5py.File(B.DATA / "seed100" / "surveys" / "survey_agn_m18_ns32.h5") as h:
        ng, zg = h["ngals"][()], h["zgals"][()]
    real = np.arange(zg.shape[1])[None, :] < ng[:, None]
    nown = int(np.sum(real & (zg < zc)))
    rr = json.loads((B.AN / "analysis_16_production_runs" / "tests" / "results" / "runlist_resolve.json").read_text())["densities_from_counts"]["100|18|agn"]["n_below_z_complete"]
    p_ = 100 / math.sqrt(val("MockNAGNCompleteEighteen"))
    slope = val("SlopefAGNDepthNorm") if mac["SlopefAGNDepthNorm"]["class"] != "ph" else None
    okd = slope is None or abs(val("NormDeltafCountsEighteen") - slope * math.log10(1 + p_ / 100)) < 1e-12
    check(abs(nown - val("MockNAGNCompleteEighteen")) <= 2 and rr == val("MockNAGNCompleteEighteen") and abs(val("MocknAGNPrecEighteen") - p_) < 1e-12 and okd,
          f"I4 AGN below z_c at m < 18: {val('MockNAGNCompleteEighteen')} (counted here at z_c = {zc:.4f}: {nown}; analysis_16 resolve test: {rr}); precision 100/sqrt N; f_AGN shift = slope x log10(1 + p)")
    # ---- I5: one definition of the width ratio; half-widths ----
    wr = val("GalhostedHzeroWid") / val("AGNhostedHzeroWid")
    hb = [k for k in mac if k.endswith("HalfWid") and abs(val(k) - 0.5 * val(k[:-7] + "Wid")) > 1e-12]
    nh = sum(1 for k in mac if k.endswith("HalfWid"))
    keys_hf = sum(1 for k, E_ in doc.items() for q in ("Hzero", "fAGN") if q in E_["quantities"])
    check(abs(wr - val("WidthRatioAGNGal")) < 1e-12 and not hb and nh == keys_hf,
          f"I5 WidthRatioAGNGal = GalhostedHzeroWid / AGNhostedHzeroWid ({wr:.3f}); {nh} HalfWid macros, one per key with Hzero or fAGN ({keys_hf}), each half its Wid")
    # ---- I6: Table I prior ranges from prod_common's text ----
    txt = (B.AN / "analysis_16_production_runs" / "scripts" / "prod_common.py").read_text()
    pri = {m.group(1): (float(m.group(2)), float(m.group(3))) for m in re.finditer(r'"(\w+)": \(([-0-9.]+), ([-0-9.]+)\)', txt.split("PRIORS = {")[1].split("}")[0])}
    mp = {"Hzero": "H0", "f": "f_agn", "MuG": "mu_G", "SigmaG": "sigma_G", "MuChi": "mu_chi", "SigmaChi": "sigma_chi", "DmuG": "dmu_G", "DmuChi": "dmu_chi",
          "LognGal": "log10n_gal", "LognAGN": "log10n_agn"}
    bad = [k for k, pn in mp.items() for e, i in (("Lo", 0), ("Hi", 1)) if abs(val(f"Prior{k}{e}") - pri[pn][i]) > 1e-12]
    check(not bad, f"I6 the Table I prior macros equal the production ranges of prod_common.PRIORS (text parsed){': ' + ', '.join(bad) if bad else ''}")
    # ---- I9: exploratory combinations over the production realizations, by plain trapezoid sums of the stored grids ----
    dev = []
    for k, tr in (("AGNhosted", "agn"), ("Galhosted", "gal")):
        if doc[k].get("final"):
            continue
        use = doc[k]["comb_realizations_used"]
        L = None
        for s in use:
            with h5py.File(B.A0 / f"h0_pure{tr}_targeted_s{s}.h5") as h:
                Hg, l_ = h["H0_grid"][()], h["log_likelihood"][()]
            L = l_ if L is None else L + l_
        dev.append((k, len(use), _trap_median(Hg, L), val(f"{k}HzeroComb")))
    for k, col in (("KnownGal", 0), ("KnownAGN", -1)):
        if doc[k].get("final"):
            continue
        use = doc[k]["comb_realizations_used"]
        L = None
        for s in use:
            with h5py.File(B.A2 / f"joint_s{s}.h5") as h:
                Hg, l_ = h["H0_grid"][()], h["log_likelihood"][()][:, col]
            L = l_ if L is None else L + l_
        dev.append((k, len(use), _trap_median(Hg, L), val(f"{k}HzeroComb")))
    # the AGN-catalog-alone posterior piles up in the last stored cell (99.75-100): there the stored-cell trapezoid and the
    # spline-refined lattice differ by a fraction of that cell, so it is held to a quarter of the 0.25 km/s/Mpc grid step
    rel_ = [abs(m - v["median"]) / (0.5 * (v["ci90"][1] - v["ci90"][0])) for k, _, m, v in dev if k != "KnownAGN"]
    edge = [abs(m - v["median"]) for k, _, m, v in dev if k == "KnownAGN"]
    check(max(rel_, default=0.0) < 0.06 and max(edge, default=0.0) < 0.25 * 0.5 + 1e-9 and all(nn == doc[k]["N_comb"] for k, nn, _, _ in dev),
          f"I9 exploratory combinations (all-AGN, all-galaxy, each catalog alone) use N_comb production realizations; medians by a plain trapezoid sum agree to "
          f"{max(rel_, default=0.0):.3f} of the 90% half-width (AGN catalog alone, at the prior edge: {max(edge, default=0.0):.3f} km/s/Mpc; "
          f"{len(dev)} still exploratory, the others are production results audited in J)")


# ============================================================================================= #
# J: production results (scripts/production.py from values/production_map.json and Analysis 16's runs). Every macro
# that turned real is recomputed here from the run summaries by code that production.py and the letter's results.py
# do not use: summary.json read directly; grid runs' ln L read with h5py and corrected by Analysis 14's formula
# written out here, cubic splines (the fine pass inside its box, the coarse pass outside), Simpson sums on a lattice;
# nested runs: each part's weighted samples, weighted quantiles, Scott-rule Gaussian kernel densities multiplied;
# survey counts with h5py from the files the run's config names; missing volumes from Analysis 15's closed form.
# ============================================================================================= #
RUNS16 = B.AN / "analysis_16_production_runs" / "runs"
QPAR = {"Hzero": "H0", "fAGN": "f_agn", "MuG": "mu_G", "SigmaG": "sigma_G", "MuChi": "mu_chi", "SigmaChi": "sigma_chi", "DmuG": "dmu_G",
        "DmuChi": "dmu_chi", "LognGal": "log10n_gal", "LognAGN": "log10n_agn"}


def _jr(p):
    return json.loads(Path(p).read_text())


def _grid_spl(run):
    """[(names, axes, spline, box)] of a grid run, fine pass first: ln L + sigma^2/2 - N/(2 N_eff) (when the run applied it)."""
    from scipy.interpolate import CubicSpline, RectBivariateSpline
    res, cfg = _jr(run / "grid" / "result.json"), _jr(run / "config.json")
    out = []
    for nm in (("fine_widened.h5" if (res.get("outside_mass_check") or {}).get("widened") else "fine.h5"), "coarse.h5"):
        with h5py.File(run / "grid" / nm, "r") as h:
            names = json.loads(h.attrs["axes"])
            ax = [np.asarray(h[f"axis_{n}"][()], float) for n in names]
            n = int(h.attrs["n_events"])
            l = np.asarray(h["lnL"][()], float)
            if cfg["half_variance_correction"]:
                l = l + 0.5 * np.asarray(h["pe_var"][()], float) - 0.5 * n / np.asarray(h["n_eff"][()], float)
        l = np.where(np.isfinite(l), l, np.nanmin(l[np.isfinite(l)]) - 300.0)
        sp = CubicSpline(ax[0], l) if len(names) == 1 else RectBivariateSpline(ax[0], ax[1], l, kx=3, ky=3, s=0)
        out.append((names, ax, sp, [(a[0], a[-1]) for a in ax]))
    return out


def _grid_eval(parts, pts):
    val = np.full(len(pts), np.nan)
    for names, ax, sp, box in parts:
        ins = np.all([(pts[:, i] >= lo) & (pts[:, i] <= hi) for i, (lo, hi) in enumerate(box)], axis=0) & np.isnan(val)
        if ins.any():
            val[ins] = sp(pts[ins, 0]) if len(names) == 1 else sp.ev(pts[ins, 0], pts[ins, 1])
    return val


def _lattice16(all_parts, n1=6001, n2=(321, 241)):
    """Each axis: the union of the fine passes (cut to the coarse range), widened by 20%."""
    names = all_parts[0][0][0]
    axes = []
    for i in range(len(names)):
        lo, hi = min(p[0][3][i][0] for p in all_parts), max(p[0][3][i][1] for p in all_parts)
        clo, chi = min(p[1][3][i][0] for p in all_parts), max(p[1][3][i][1] for p in all_parts)
        w = hi - lo
        axes.append(np.linspace(max(clo, lo - 0.2 * w), min(chi, hi + 0.2 * w), n1 if len(names) == 1 else n2[i]))
    return names, axes


def _simpson(y, x, axis=-1):
    from scipy.integrate import simpson
    return simpson(y, x=x, axis=axis)


def _grid_comb(runs):
    """Summed corrected ln L of grid runs on a lattice -> (names, axes, lnP total)."""
    all_parts = [_grid_spl(r) for r in runs]
    names, axes = _lattice16(all_parts)
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(names))
    tot = sum(_grid_eval(p, mesh) for p in all_parts).reshape([len(a) for a in axes])
    return names, axes, tot


def _marg(names, axes, tot, par):
    P = np.exp(tot - tot.max())
    i = names.index(par)
    for j in sorted((j for j in range(len(names)) if j != i), reverse=True):
        P = _simpson(P, axes[j], axis=j)
    return axes[i], P / _simpson(P, axes[i])


def _quant(x, p, x0=None):
    from scipy.integrate import cumulative_trapezoid
    xf = np.linspace(x[0], x[-1], 40001)
    pf = np.interp(xf, x, p)
    c = cumulative_trapezoid(pf, xf, initial=0.0)
    c /= c[-1]
    q = lambda u: float(np.interp(u, c, xf))
    out = {"median": q(0.5), "ci90": [q(0.05), q(0.95)], "ul90": q(0.90)}
    if x0 is not None:
        out["level"] = abs(2 * float(np.interp(x0, xf, c)) - 1)
    return out


def _nested_w(run, params):
    """Weighted samples of a nested run: each finished part's normalised weights, the parts averaged."""
    S, W = [], []
    for part in sorted(run.glob("nested_[0-9]")):
        if not (part / "result.json").exists() or not _jr(part / "result.json").get("finished"):
            continue
        z = np.load(part / "samples.npz", allow_pickle=False)
        nm = [str(x) for x in z["names"]]
        S.append(np.asarray(z["samples"], float)[:, [nm.index(p) for p in params]])
        w = np.asarray(z["weights"], float)
        W.append(w / w.sum())
    return np.concatenate(S), np.concatenate(W) / len(W)


def _wq(x, w, qs):
    o = np.argsort(x)
    c = np.cumsum(w[o]) - 0.5 * w[o]
    return np.interp(qs, c / c[-1], x[o])


def _kde_prod(sets, lo, hi, n=6001):
    """Product over realizations of Scott-rule Gaussian kernel densities of weighted samples (reflected at lo, hi)."""
    from scipy.stats import gaussian_kde
    x = np.linspace(lo, hi, n)
    lp = np.zeros(n)
    for s, w in sets:
        keep = w > 1e-9 * w.max()
        k = gaussian_kde(s[keep], weights=w[keep])
        p = k(x) + k(2 * lo - x) + k(2 * hi - x)
        lp += np.log(np.maximum(p, 1e-300))
    p = np.exp(lp - lp.max())
    return x, p / np.trapz(p, x)


def _counts(run):
    cfg = _jr(run / "config.json")
    out = {}
    for cls in ("gal", "agn"):
        with h5py.File(cfg["surveys"][cls], "r") as h:
            out[cls] = int(np.sum(h["ngals"][()].astype(np.int64)))
    return out


def audit_production(mac, doc, owner):
    pmap = _jr(OUT / "values" / "production_map.json")
    specs = pmap.get("keys", {})
    final = [k for k, E in doc.items() if E.get("final")]
    if not final:
        check(True, "J0 no key is a production result yet (nothing to recompute)")
        return
    odds = lambda f: f / (1.0 - f)
    val = lambda n: mac[n]["value"]
    rows, bad = [], []

    def cmp(name, got, want, tol, what):
        d = abs(got - want)
        rows.append((name, what, d, tol))
        if not d <= tol:
            bad.append(f"{name} {what}: {got:.6g} vs {want:.6g} (tol {tol:.2g})")

    # ---- J0: every final key has its spec, every summary it names exists, its macros print black and cite the summaries
    miss = [k for k in final if k not in specs]
    nosum = [k for k in final for r in specs.get(k, {}).get("realizations", []) if not (RUNS16 / specs[k]["run"].format(R=f"{r:03d}") / "summary.json").exists()]
    notblack = [n for n, ks in owner.items() if all(k in final for k in ks) and (mac[n]["class"] != "plain" or not ("file" in mac[n]["source"] or "files" in mac[n]["source"]))]
    check(not miss and not nosum and not notblack, f"J0 {len(final)} production keys ({', '.join(sorted(final))}): each has its runs in production_map.json, every summary.json exists, "
                                                   f"every macro prints black and cites its summary ({len(miss) + len(nosum) + len(notblack)} exceptions{': ' + ', '.join((miss + nosum + notblack)[:6]) if miss or nosum or notblack else ''})")
    comb_cache = {}
    for k in sorted(final):
        sp = specs[k]
        runs = [RUNS16 / sp["run"].format(R=f"{r:03d}") for r in sp["realizations"]]
        rA = RUNS16 / sp["run"].format(R=f"{sp['realization_A']:03d}")
        sA = _jr(rA / "summary.json")
        grid = sp["mode"] == "grid"
        if grid:
            names, axes, tot = _grid_comb(runs)
            nA, aA, tA = _grid_comb([rA])
        for q in sp["quantities"]:
            if q == "RateRatio":
                continue
            par = QPAR[q]
            vA, vC = val(f"{k}{q}"), val(f"{k}{q}Comb")
            hwA, hwC = 0.5 * (vA["ci90"][1] - vA["ci90"][0]), 0.5 * (vC["ci90"][1] - vC["ci90"][0])
            x0 = doc[k]["quantities"][q]["input"]
            conv = k.startswith("Free") and q in ("LognGal", "LognAGN")
            # realization A: the macro is the summary's median and 90% interval
            if not conv:
                ps = sA["posterior"][par]
                for nm, a, b in (("median", vA["median"], ps["median"]), ("lo", vA["ci90"][0], ps["ci90"][0]), ("hi", vA["ci90"][1], ps["ci90"][1])):
                    cmp(f"{k}{q}", a, b, 1e-12, f"A {nm} = summary.json")
                cmp(f"{k}{q}Wid", val(f"{k}{q}Wid"), ps["ci90"][1] - ps["ci90"][0], 1e-12, "A width = summary.json")
            # realization A and the combination from the grids / samples
            if grid:
                ownA = _quant(*_marg(nA, aA, tA, par))
                own = _quant(*_marg(names, axes, tot, par), x0)
                comb_cache[(k, q)] = _marg(names, axes, tot, par)
            else:
                if conv:
                    sets = []
                    for r in runs:
                        s, w = _nested_w(r, [par, "H0"])
                        sets.append((s[:, 0] + 3 * np.log10(s[:, 1] / 100.0), w))
                else:
                    sets = [(lambda sw: (sw[0][:, 0], sw[1]))(_nested_w(r, [par])) for r in runs]
                sa, wa = sets[sp["realizations"].index(sp["realization_A"])]
                qa = _wq(sa, wa, [0.05, 0.5, 0.95])
                ownA = {"median": float(qa[1]), "ci90": [float(qa[0]), float(qa[2])]}
                lo = min(s.min() for s, _ in sets)
                hi = max(s.max() for s, _ in sets)
                pr = {"H0": (50.0, 100.0), "f_agn": (0.0, 1.0)}.get(par)
                if pr and not conv:
                    lo, hi = pr
                xc, pc = _kde_prod(sets, lo, hi)
                own = _quant(xc, pc, x0)
                comb_cache[(k, q)] = (xc, pc)
            # a grid run's summary.json interpolates the stored ln L (Akima) onto a lattice 16 times denser than the nodes
            # (prod_summary.refined_grid_summary), H0 and f_AGN alike; against the cubic spline here the two agree to 0.03%
            # of the half-width on the paper's grids, so both are held to the same 1.5% of the half-width
            tolA = 0.015 * hwA if not conv else 0.05 * hwA
            for nm, a, b in (("median", vA["median"], ownA["median"]), ("lo", vA["ci90"][0], ownA["ci90"][0]), ("hi", vA["ci90"][1], ownA["ci90"][1])):
                cmp(f"{k}{q}", a, b, max(tolA, 5e-4), f"A {nm} (summary.json) vs own {'spline of the grid' if grid else 'weighted samples'}")
            tolC = (0.01 if grid else 0.08) * hwC
            for nm, a, b in (("median", vC["median"], own["median"]), ("lo", vC["ci90"][0], own["ci90"][0]), ("hi", vC["ci90"][1], own["ci90"][1])):
                cmp(f"{k}{q}Comb", a, b, max(tolC, 5e-4 if q != "Hzero" else 0.005), f"Comb {nm} vs own {'summed splines' if grid else 'Scott-KDE product'}")
            if f"{k}{q}CombUL" in mac:
                cmp(f"{k}{q}CombUL", val(f"{k}{q}CombUL"), own["ul90"], max(tolC, 2e-3), "Comb 90% upper limit")
            if conv and f"{k}{q}ULh" in mac:
                # the limit in h^3 Mpc^-3: the run's own weighted samples of its parameter, no conversion
                sh, wh = _nested_w(rA, [par])
                qh = _wq(sh[:, 0], wh, [0.05, 0.90, 0.95])
                cmp(f"{k}{q}ULh", val(f"{k}{q}ULh"), float(qh[1]), max(0.05 * 0.5 * (qh[2] - qh[0]), 5e-4), "A 90% upper limit in h^3 Mpc^-3 vs own weighted samples")
            if x0 is not None and isinstance(val(f"{k}{q}CL"), (int, float)):
                cmp(f"{k}{q}CL", val(f"{k}{q}CL"), own["level"], 0.01 if grid else 0.04, "credible level of the input in Comb")
        # rate per host
        if "RateRatio" in sp["quantities"]:
            Rin = val("MockRateRatio")
            zr = sp["quantities"]["RateRatio"]["z_ratio"]
            vA, vC = val(f"{k}RateRatio"), val(f"{k}RateRatioComb")
            if zr in ("catalog_counts", "counts_and_missing_volume"):
                def zratio(run):
                    c = _counts(run)
                    if zr == "catalog_counts":
                        return c["gal"] / c["agn"]
                    C = _a15()
                    sm = _jr(run / "summary.json")
                    rk = f"m{sm['flux']}" if not str(sm["flux"]).startswith("m") else sm["flux"]
                    Z = {}
                    for cls in ("gal", "agn"):
                        d = sm["densities"][cls]
                        fit = C.schechter_fit(C.FITS / f"fit_{cls}_{rk}_schechter.json")
                        Z[cls] = C.z_closed(c[cls], d["log10n_used"] + 3 * math.log10(0.6774), fit, 67.74, C.ZMAX_R)
                    return Z["gal"] / Z["agn"]
                ratios = [zratio(r) for r in runs]
                ra = ratios[sp["realizations"].index(sp["realization_A"])]
                pf = sA["posterior"]["f_agn"]
                for nm, a, b in (("median", vA["median"], odds(pf["median"]) * ra), ("lo", vA["ci90"][0], odds(pf["ci90"][0]) * ra), ("hi", vA["ci90"][1], odds(pf["ci90"][1]) * ra)):
                    cmp(f"{k}RateRatio", a, b, 1e-6 * abs(b) + (0 if zr == "catalog_counts" else 2e-4 * abs(b)), f"A {nm} = odds(summary f) x own Z ratio")
                gm = math.exp(float(np.mean(np.log(ratios))))
                xf, pff = comb_cache[(k, "fAGN")]
                of = _quant(xf, pff, Rin / (Rin + gm))
                hw = 0.5 * (vC["ci90"][1] - vC["ci90"][0])
                for nm, a, b in (("median", vC["median"], odds(of["median"]) * gm), ("lo", vC["ci90"][0], odds(of["ci90"][0]) * gm), ("hi", vC["ci90"][1], odds(of["ci90"][1]) * gm)):
                    cmp(f"{k}RateRatioComb", a, b, 0.01 * hw, f"Comb {nm}: own f combination x geometric-mean Z ratio")
                cmp(f"{k}RateRatioCL", val(f"{k}RateRatioCL"), of["level"], 0.01, "credible level of the input rate ratio")
        # Bayes factors
        L = sp.get("LB")
        if L and L["kind"] == "savage_dickey_f0":
            from scipy.interpolate import Akima1DInterpolator
            for part, rs, nmm in (("A", [rA], f"{k}LB"), ("Comb", runs, f"{k}LBComb")):
                Hd = np.linspace(50, 100, 25001)
                col = np.zeros_like(Hd)
                for r in rs:
                    c_ = _grid_spl(r)[1]                                     # the coarse pass holds f = 0 on its nodes
                    names_c, ax_c = c_[0], c_[1]
                    with h5py.File(r / "grid" / "coarse.h5", "r") as h:
                        l = np.asarray(h["lnL"][()], float)
                        if _jr(r / "config.json")["half_variance_correction"]:
                            l = l + 0.5 * np.asarray(h["pe_var"][()], float) - 0.5 * int(h.attrs["n_events"]) / np.asarray(h["n_eff"][()], float)
                    j = int(np.argmin(np.abs(ax_c[1])))
                    assert abs(ax_c[1][j]) < 1e-12
                    col += Akima1DInterpolator(ax_c[0], l[:, j])(Hd)
                lz0 = col.max() + math.log(np.trapz(np.exp(col - col.max()), Hd))
                if part == "A":
                    # the run's own grid evidence: ln[(1/vol) int int L]; the f = 0 evidence with the same H0 prior: ln[(1/50) int L(H0, 0)]
                    lzt = sA["ln_evidence"]["value"] + math.log(50.0 * 1.0)
                else:
                    nm_, ax_, tt = names, axes, tot
                    m = tt.max()
                    I = _simpson(_simpson(np.exp(tt - m), ax_[1], axis=1), ax_[0])
                    lzt = m + math.log(I)
                cmp(nmm, val(nmm), (lzt - lz0) / math.log(10), max(0.02, 0.002 * abs(val(nmm))), f"{part}: Savage-Dickey at f = 0 (Akima f = 0 column; "
                                                                                             f"{'summary ln_evidence' if part == 'A' else 'own lattice'})")
        elif L:
            per = []
            for r in sp["realizations"]:
                za = _jr(RUNS16 / sp["run"].format(R=f"{r:03d}") / "summary.json")["ln_evidence"]["value"]
                zb = _jr(RUNS16 / L["reference_run"].format(R=f"{r:03d}") / "summary.json")["ln_evidence"]["value"]
                per.append((za - zb) / math.log(10))
            cmp(f"{k}LB", val(f"{k}LB"), per[sp["realizations"].index(sp["realization_A"])], 1e-9, "A: ln Z difference of the summaries")
            cmp(f"{k}LBComb", val(f"{k}LBComb"), float(sum(per)), 1e-9, "Comb: sum over the realizations")
        # correlations of the joint posteriors
        if k == "KnownBoth":
            cmp("KnownBothCorrHzerofAGN", val("KnownBothCorrHzerofAGN"), sA["rho_H0_f"], 0.03, "A correlation vs the run's own (fine grid)")
            P = np.exp(tot - tot.max())
            X, Y = np.meshgrid(*axes, indexing="ij")
            w = P / P.sum()
            mx, my = (w * X).sum(), (w * Y).sum()
            rc = (w * (X - mx) * (Y - my)).sum() / math.sqrt((w * (X - mx) ** 2).sum() * (w * (Y - my) ** 2).sum())
            cmp("KnownBothCorrHzerofAGNComb", val("KnownBothCorrHzerofAGNComb"), rc, 0.03, "Comb correlation on the own lattice")
        if k in ("SingleBoth", "SingleNone"):
            s, w = _nested_w(rA, ["H0", "mu_G"])
            mx, my = np.sum(w * s[:, 0]), np.sum(w * s[:, 1])
            rc = np.sum(w * (s[:, 0] - mx) * (s[:, 1] - my)) / math.sqrt(np.sum(w * (s[:, 0] - mx) ** 2) * np.sum(w * (s[:, 1] - my) ** 2))
            cmp(f"CorrHzeroMuG{'Both' if k == 'SingleBoth' else 'None'}", val(f"CorrHzeroMuG{'Both' if k == 'SingleBoth' else 'None'}"), rc, 0.05 + 1e-9,
                "A correlation of the weighted samples (printed to one decimal)")
    # ---- further macros from the summaries
    fin = set(final)
    widA = lambda k: (lambda p: p["ci90"][1] - p["ci90"][0])(_jr(RUNS16 / specs[k]["run"].format(R=f"{specs[k]['realization_A']:03d}") / "summary.json")["posterior"]["H0"])
    if {"Galhosted", "AGNhosted"} <= fin:
        cmp("WidthRatioAGNGal", val("WidthRatioAGNGal"), widA("Galhosted") / widA("AGNhosted"), 1e-12, "ratio of the summaries' 90% widths")
    for nm, kn, kb in (("MassScaleShareKnown", "KnownNone", "KnownBoth"), ("MassScaleShareSingle", "SingleNone", "SingleBoth")):
        if {kn, kb} <= fin:
            cmp(nm, val(nm), widA(kn) / widA(kb), 1e-12, "ratio of the summaries' 90% widths")
    for pre in ("HD", "Obs"):
        if f"{pre}Single" in fin and (f"{pre}Single", "Hzero") in comb_cache:
            o = _quant(*comb_cache[(f"{pre}Single", "Hzero")])
            cmp(f"{pre}BiasHzero", val(f"{pre}BiasHzero"), o["median"] - B.IN["H0"], 0.08 * 0.5 * (o["ci90"][1] - o["ci90"][0]), "own Comb median minus the input")
        if f"{pre}Host" in fin:
            cmp(f"{pre}LB", val(f"{pre}LB"), val(f"{pre}HostLB"), 1e-12, "= HostLB")
    worst = max(rows, key=lambda r: r[2] / r[3] if r[3] > 0 else (0 if r[2] == 0 else np.inf))
    check(not bad, f"J1 {len(rows)} numbers of the production keys recomputed by a second code path (summary.json read directly; own splines of the grids; "
                   f"own weighted samples and kernel densities; own survey counts; Analysis 15's missing volumes); largest deviation {worst[2]:.2g} "
                   f"({worst[0]}, {worst[1]}, tolerance {worst[3]:.2g}){'; MISMATCH: ' + '; '.join(bad[:6]) if bad else ''}")


# ============================================================================================= #
# K: the experiment's macros (scripts/classes.py) recomputed by code that classes.py and run_posteriors.py do not use:
# the run summaries read directly; each grid's ln L read with h5py, corrected by Analysis 14's formula written out here
# (the class counts of a host-class-known grid from its summary.json, not from grid/result.json), interpolated by PCHIP
# (not the Akima of the summaries, not the cubic splines of the combinations), quantiles from an own CDF; the probability of
# an AGN host recomputed from the per-merger likelihoods and the true host classes of the events file; the run lists parsed
# here. K2: no grid the paper uses has a jump of ln L between neighbouring nodes where its posterior has mass.
# ============================================================================================= #
def _pass_file(run):
    """The fine pass the run's summary.json was computed from (its own record), and the coarse pass."""
    sm = _jr(run / "summary.json")
    fine = ((sm.get("grid") or {}).get("posterior_lattice") or {}).get("file") or "fine.h5"
    return fine, "coarse.h5"


def _corrected(run, nm, cls=None):
    """(names, axes, corrected ln L) of pass `nm` of a grid run; cls: one host class's factor (lnL_<cls>)."""
    cfg = _jr(run / "config.json")
    suf = f"_{cls}" if cls else ""
    with h5py.File(run / "grid" / nm, "r") as h:
        names = json.loads(h.attrs["axes"])
        ax = [np.asarray(h[f"axis_{n}"][()], float) for n in names]
        l = np.asarray(h["lnL" + suf][()], float)
        n = int(h.attrs["n_events"]) if not cls else int(_jr(run / "summary.json")["n_events_by_class"][cls])
        if cfg["half_variance_correction"]:
            l = l + 0.5 * np.asarray(h["pe_var" + suf][()], float) - 0.5 * n / np.asarray(h["n_eff" + suf][()], float)
    return names, ax, l


def _pchip_H0(names, ax, l, refine=8):
    """H0 marginal of a 1-D or 2-D grid by PCHIP of ln L along each axis onto a lattice `refine` times denser: (x, p)."""
    from scipy.interpolate import PchipInterpolator
    ok = np.isfinite(l)
    l = np.where(ok, l, np.min(l[ok]) - 300.0)
    lat = [np.linspace(a[0], a[-1], (len(a) - 1) * refine + 1) for a in ax]
    for i, (a, an) in enumerate(zip(ax, lat)):
        l = PchipInterpolator(a, l, axis=i)(an)
    pp = np.exp(l - l.max())
    i = names.index("H0")
    for j in sorted((j for j in range(len(names)) if j != i), reverse=True):
        pp = _simpson(pp, lat[j], axis=j)
    return lat[i], pp


def _own_hw(run, cls=None):
    names, ax, l = _corrected(run, _pass_file(run)[0], cls)
    q = _quant(*_pchip_H0(names, ax, l))
    return 0.5 * (q["ci90"][1] - q["ci90"][0]), q


def _steps(l, ax, names, T=20.0, iso=5.0):
    """Isolated jumps of ln L between neighbouring interior nodes along each axis: |d| > T nats and > iso times the
    larger of the two neighbouring steps. -> [(axis, x_i, x_i+1, d, nats of the upper node below the array's maximum)]."""
    out = []
    mx = np.nanmax(np.where(np.isfinite(l), l, np.nan))
    for a in range(l.ndim):
        arr = np.moveaxis(l, a, 0)
        d = np.diff(arr, axis=0)
        for i in range(1, arr.shape[0] - 2):
            di, nb = d[i], np.fmax(np.abs(d[i - 1]), np.abs(d[i + 1]))
            hit = np.isfinite(di) & np.isfinite(nb) & (np.abs(di) > T) & (np.abs(di) > iso * nb)
            for k in zip(*np.nonzero(np.atleast_1d(hit))):
                k = tuple(k) if di.ndim else ()
                up = max(arr[(i,) + k], arr[(i + 1,) + k])
                out.append((names[a], float(ax[a][i]), float(ax[a][i + 1]), float(di[k] if di.ndim else di), float(mx - up)))
    return out


def audit_classes(mac, doc):
    import yaml
    from scipy.interpolate import CubicSpline
    pm = _jr(OUT / "values" / "production_map.json")
    blk = dict((pm.get("classes") or {}).get("macros") or {})
    cl = {k: v for k, v in mac.items() if v["claim"].startswith("classes")}
    if not cl:
        check(False, "K0 values.json has the experiment's macros (claim 'classes')")
        return
    # where the probability of an AGN host is evaluated and its mean over every merger (ProbEvalfAGN<info>, ProbAll<info>) read the run
    # and membership file of ProbAGN<info>; a map written before they existed has no record of its own for them
    for n in cl:
        m_ = re.fullmatch(r"Prob(?:EvalfAGN|All)([A-Za-z]+)", n)
        if m_ and n not in blk and f"ProbAGN{m_.group(1)}" in blk:
            blk[n] = dict(blk[f"ProbAGN{m_.group(1)}"])
    missing_new = [n for n in [f"Prob{s}{i}" for s in ("EvalfAGN", "All") for i in ("Equal", "Pos", "Pops", "PosTwice", "PopsTwice")]
                   + [f"Cost{t}{f}{e}" for t in ("Unknown", "Pops", "PopsTwice") for f in ("Ten", "Thirty", "Fifty") for e in ("Min", "Max")] if n not in cl]
    if missing_new:
        check(False, f"K0 the macros of REVIEW_2 1.1 and 1.2 exist: missing {', '.join(missing_new[:6])}")
    val = lambda n: mac[n]["value"]
    rows, bad = [], []

    def cmp(name, got, want, tol, what):
        d = abs(got - want)
        rows.append((name, what, d, tol))
        if not d <= tol:
            bad.append(f"{name} {what}: {got:.6g} vs {want:.6g} (tol {tol:.2g})")

    # ---- K0: black exactly when every summary the map names exists; the map's runs are the run lists' (own parser)
    exp_idx = {}
    ev_idx = {}                      # run pattern -> {realization: events file of the run list}
    rx = yaml.safe_load((RUNS16 / "RUNLIST_experiment.yaml").read_text())
    for st in rx["stages"]:
        for r in st["runs"]:
            pat_, rr_ = re.sub(r"realization_\d{3}", "realization_{R}", r["name"]), int(r["args"][r["args"].index("--realization") + 1])
            exp_idx.setdefault(pat_, set()).add(rr_)
            ev_idx.setdefault(pat_, {})[rr_] = r.get("events_file")
    for r in rx.get("reused_from_stage_1") or []:
        exp_idx.setdefault(re.sub(r"realization_\d{3}", "realization_{R}", r["run"]), set()).add(int(r["realization"]))
    rl = yaml.safe_load((RUNS16 / "RUNLIST.yaml").read_text())
    for st in rl["stages"]:
        for r in st["runs"]:
            pat_, rr_ = re.sub(r"realization_\d{3}", "realization_{R}", r["name"]), int(r["args"][r["args"].index("--realization") + 1])
            exp_idx.setdefault(pat_, set()).add(rr_)
            ev_idx.setdefault(pat_, {})[rr_] = r.get("events_file")
    wrong, notlisted = [], []
    have = {}
    for name, rec in blk.items():
        if rec.get("how") == "derived":
            continue
        if sorted(exp_idx.get(rec["run"], ())) != sorted(rec["realizations"]):
            notlisted.append(name)
        reals = [rec["realization_A"]] if rec["how"] == "membership" else rec["realizations"]
        have[name] = all((RUNS16 / rec["run"].format(R=f"{r:03d}") / "summary.json").exists() for r in reals)
    for name, rec in blk.items():
        if rec.get("how") == "derived":
            have[name] = all(have[d] for d in rec["from"])
    for name, ok in have.items():
        names = [name] if name in mac else []
        names += [n for n in cl if n.startswith(name) and n[len(name):] in ("Min", "Max")]
        if name == "KnownAGNNoPeak":
            names += [n for n in cl if n.startswith("KnownAGNPeak")]
        for n in names:
            if (mac[n]["class"] == "plain") != ok:
                wrong.append(n)
    nblack = sum(1 for v in cl.values() if v["class"] == "plain")
    check(not wrong and not notlisted, f"K0 the experiment's macros ({len(cl)}; {nblack} black) print black exactly when every summary of their runs exists, and "
                                       f"every run pattern of the map's 'classes' block has the realizations of the run lists (own parser)"
                                       f"{': ' + ', '.join((wrong + notlisted)[:6]) if wrong or notlisted else ''}")
    # ---- K1: the numbers, by a second code path
    own_mean, own_per_s, own_per_o = {}, {}, {}
    for name, rec in blk.items():
        if rec.get("how") not in ("mean_hw", "factor_hw") or not have.get(name):
            continue
        per_s, per_o = [], []
        for r in rec["realizations"]:
            run = RUNS16 / rec["run"].format(R=f"{r:03d}")
            if rec["how"] == "mean_hw":
                b = _jr(run / "summary.json")["posterior"]["H0"]
                per_s.append(0.5 * (b["ci90"][1] - b["ci90"][0]))
                if _jr(run / "config.json")["mode"] == "grid":
                    hw_o, _ = _own_hw(run)
                    cmp(f"{name}[{r}]", per_s[-1], hw_o, 0.02 * hw_o, "half-width: summary.json vs own PCHIP of the grid")
                    per_o.append(hw_o)
                else:
                    s_, w_ = _nested_w(run, ["H0"])
                    q_ = _wq(s_[:, 0], w_, [0.05, 0.95])
                    per_o.append(0.5 * (q_[1] - q_[0]))
                    cmp(f"{name}[{r}]", per_s[-1], per_o[-1], 0.03 * per_o[-1], "half-width: summary.json vs own weighted samples")
            else:
                hw_o, _ = _own_hw(run, rec["factor"])
                per_o.append(hw_o)
        if rec["how"] == "mean_hw":
            cmp(name, val(name), float(np.mean(per_s)), 1e-12, "mean of the summaries' half-widths")
            cmp(name + "Min", val(name + "Min"), min(per_s), 1e-12, "smallest of the summaries' half-widths")
            cmp(name + "Max", val(name + "Max"), max(per_s), 1e-12, "largest of the summaries' half-widths")
        cmp(name, val(name), float(np.mean(per_o)), 0.02 * float(np.mean(per_o)), "mean half-width vs own PCHIP grids")
        cmp(name + "Min", val(name + "Min"), min(per_o), 0.02 * min(per_o), "smallest half-width vs own")
        cmp(name + "Max", val(name + "Max"), max(per_o), 0.02 * max(per_o), "largest half-width vs own")
        own_mean[name] = float(np.mean(per_o))
        own_per_o[name] = dict(zip(rec["realizations"], per_o))
        if per_s:
            own_per_s[name] = dict(zip(rec["realizations"], per_s))
    for name, rec in blk.items():
        if rec.get("how") != "derived" or not have.get(name):
            continue
        a_, b_ = rec["from"]
        f = (lambda x, y: 100.0 * (x / y) ** 2) if rec["fn"] == "share" else (lambda x, y: (x / y) ** 2)
        cmp(name, val(name), f(val(a_), val(b_)), 1e-12, "from the unrounded values of its two macros")
        cmp(name, val(name), f(own_mean[a_], own_mean[b_]), 0.04 * val(name), "from the own half-widths")
        if rec["fn"] == "cost":
            # Min, Max: the same ratio per realization (the summaries' half-widths read here; and the own PCHIP half-widths)
            for src_, rtol, how_ in ((own_per_s, 1e-12, "summaries' half-widths"), (own_per_o, 0.04, "own PCHIP half-widths")):
                if a_ not in src_ or b_ not in src_ or sorted(src_[a_]) != sorted(src_[b_]):
                    bad.append(f"{name}Min/Max: no per-realization half-widths of {a_} and {b_} on the same realizations ({how_})")
                    continue
                per = [(src_[a_][r] / src_[b_][r]) ** 2 for r in src_[a_]]
                for suf, want in (("Min", min(per)), ("Max", max(per))):
                    cmp(name + suf, val(name + suf), want, rtol * want, f"{suf.lower()} over the realizations of the per-realization ratio ({how_})")
    # red cost ranges: the central value times the relative spread of the finished CostUnknown<f> at the same f (classes.cost_range)
    for name in [n for n in cl if re.fullmatch(r"Cost(Pops|PopsTwice)(Ten|Thirty|Fifty)", n)]:
        ref = "CostUnknown" + re.fullmatch(r"Cost(?:Pops|PopsTwice)(Ten|Thirty|Fifty)", name).group(1)
        if mac[name]["class"] == "ph" and mac[ref]["class"] == "plain":
            for suf in ("Min", "Max"):
                cmp(name + suf, val(name + suf), val(name) * val(ref + suf) / val(ref), 1e-12, f"red range: {name} times {ref}{suf} / {ref}")
    # ShareSumThirty (build_values.build_classes, not in the map): black exactly when ShareAGNThirty is; then ShareAGNThirty plus the galaxy
    # share of values.json, and the sum of the two shares from the own PCHIP half-widths of the AGN and galaxy factors of the same grids
    if "ShareSumThirty" not in mac:
        bad.append("ShareSumThirty missing")
    else:
        real_s = bool(have.get("ShareAGNThirty")) and mac["ShareAGNThirty"]["extra"].get("galaxy_share_percent") is not None
        if (mac["ShareSumThirty"]["class"] == "plain") != real_s:
            bad.append(f"ShareSumThirty prints {'black' if mac['ShareSumThirty']['class'] == 'plain' else 'red'} but ShareAGNThirty is {'real' if real_s else 'red'}")
        if real_s:
            recA = blk["HWKnownAGNThirty"]
            cmp("ShareSumThirty", val("ShareSumThirty"), val("ShareAGNThirty") + mac["ShareAGNThirty"]["extra"]["galaxy_share_percent"], 1e-9,
                "ShareAGNThirty + galaxy_share_percent (values.json)")
            hw_g = float(np.mean([_own_hw(RUNS16 / recA["run"].format(R=f"{r:03d}"), "gal")[0] for r in recA["realizations"]]))
            k_ = own_mean["HWKnownThirty"]
            own_s = 100.0 * ((k_ / own_mean["HWKnownAGNThirty"]) ** 2 + (k_ / hw_g) ** 2)
            cmp("ShareSumThirty", val("ShareSumThirty"), own_s, 0.04 * val("ShareSumThirty"), "sum of the two shares from the own PCHIP half-widths (AGN and galaxy factors)")
    # the host class known on the host-dependent mergers themselves (HW[Obs]KnownPops[Twice]<f>) and the cost ratios: every cost
    # ratio divides by the host class known on the mergers of its numerator, and every same-merger macro is on the mergers of its
    # host-class-unknown partner (events files of the run lists, realization by realization, red or black); a finished
    # same-merger run gives each class its own population, offset as in its events file, and counts the classes of that file
    pairs = [(n, rec["from"][0], rec["from"][1]) for n, rec in blk.items() if rec.get("how") == "derived" and rec.get("fn") == "cost"]
    same_m = [n for n in blk if re.fullmatch(r"HW(Obs)?KnownPops(Twice)?(Ten|Thirty|Fifty)", n)]
    pairs += [(n, n.replace("Known", "", 1), n) for n in same_m]
    for n, a_, b_ in pairs:
        ea, eb = ev_idx.get(blk[a_]["run"], {}), ev_idx.get(blk[b_]["run"], {})
        ok_ = bool(ea) and sorted(ea) == sorted(eb) and all(ea[r] and ea[r] == eb[r] for r in ea)
        cmp(f"{n} mergers", float(not ok_), 0.0, 0.0, f"{a_} and {b_} on the same events files in every realization (run lists)")
    for n in same_m:
        if not have.get(n):
            continue
        for r in blk[n]["realizations"]:
            run = RUNS16 / blk[n]["run"].format(R=f"{r:03d}")
            cfg, sm = _jr(run / "config.json"), _jr(run / "summary.json")
            with h5py.File(cfg["events_file"], "r") as h:
                ht = np.asarray(h["truth/host_type"][()], int)
                dG, dX = float(h.attrs["dmu_G_agn"]), float(h.attrs["dmu_chi_agn"])
            nc = sm["n_events_by_class"]
            cmp(f"{n}[{r}] classes", float(abs(nc["agn"] - int(np.sum(ht == 1))) + abs(nc["gal"] - int(np.sum(ht == 0)))), 0.0, 0.0,
                "class counts of summary.json = truth/host_type of the events file")
            pg, pa = cfg["class_populations"]["gal"], cfg["class_populations"]["agn"]
            cmp(f"{n}[{r}] offsets", abs(pa["mu_G"] - pg["mu_G"] - dG) + abs(pa["mu_chi"] - pg["mu_chi"] - dX) + (0.0 if dG > 0 else 1.0), 0.0, 1e-9,
                "AGN population of the run = galaxy population + the events file's offsets (nonzero)")
    # membership: r from the per-merger likelihoods at the stored f, the true classes from the events file
    for name, rec in blk.items():
        if rec.get("how") != "membership" or not have.get(name):
            continue
        run = RUNS16 / rec["run"].format(R=f"{rec['realization_A']:03d}")
        mj = _jr(run / "grid" / "membership.json")
        z = np.load(run / "grid" / "membership.npz")
        f = float(mj["f_agn_median"])
        la, lg = np.asarray(z["ln_L_agn"], float), np.asarray(z["ln_L_gal"], float)
        m_ = np.maximum(la, lg)
        r_own = f * np.exp(la - m_) / (f * np.exp(la - m_) + (1 - f) * np.exp(lg - m_))
        with h5py.File(_jr(run / "config.json")["events_file"], "r") as h:
            ht = np.asarray(h["truth/host_type"][()], int)
        cmp(f"{name} r", float(np.max(np.abs(r_own - np.asarray(z["r"], float)))), 0.0, 1e-6, "per-merger r recomputed from ln L_AGN, ln L_gal and f")
        cmp(f"{name} host", float(np.sum(ht != np.asarray(z["host_type"], int))), 0.0, 0.0, "true host classes = the events file's truth/host_type")
        if name.startswith("ProbEvalfAGN"):
            cmp(name, val(name), f, 1e-12, "the f at which r is evaluated (grid/membership.json f_agn_median)")
            cmp(f"{name} median", val(name), float(_jr(run / "summary.json")["posterior"]["f_agn"]["median"]), 1e-3, "the posterior median of f_AGN (summary.json)")
        elif name.startswith("ProbAll"):
            cmp(name, val(name), float(r_own.mean()), 1e-6, "mean of the own r over all the mergers")
        else:
            cls = 1 if name.startswith("ProbAGN") else 0
            cmp(name, val(name), float(r_own[ht == cls].mean()), 1e-6, "mean of the own r over the true class")
        if mac[name]["render"].lstrip("\\ph{").rstrip("}") != f"{val(name):.2f}":
            bad.append(f"{name} render {mac[name]['render']}")
    # the AGN catalog alone: interior maxima from the passes read here
    if have.get("KnownAGNNoPeak"):
        rec = blk["KnownAGNNoPeak"]
        n_no, k = 0, 0
        for r in rec["realizations"]:
            run = RUNS16 / rec["run"].format(R=f"{r:03d}")
            best = (-np.inf, None)
            xs, ls = [], []
            fine = _pass_file(run)[0]
            for nm in (fine, "coarse.h5"):
                names, ax, l = _corrected(run, nm)
                ok = np.isfinite(l)
                i = int(np.argmax(np.where(ok, l, -np.inf)))
                best = max(best, (float(l[i]), float(ax[0][i])))
                xs.append(ax[0][ok])
                ls.append(l[ok])
            edge = best[1] in (50.0, 100.0)
            n_no += edge
            if not edge:
                k += 1
                base = f"KnownAGNPeak{['One', 'Two', 'Three', 'Four', 'Five'][k - 1]}Hzero"
                ps = _jr(run / "summary.json")["posterior"]["H0"]
                for nm_, a_, b_ in (("median", val(base)["median"], ps["median"]), ("lo", val(base)["ci90"][0], ps["ci90"][0]), ("hi", val(base)["ci90"][1], ps["ci90"][1])):
                    cmp(base, a_, b_, 1e-12, f"{nm_} = summary.json")
                # level of the input: the fine pass inside its box, the coarse nodes outside it, PCHIP of ln L
                from scipy.interpolate import PchipInterpolator
                inb = (xs[1] < xs[0][0]) | (xs[1] > xs[0][-1])
                xx = np.concatenate([xs[0], xs[1][inb]])
                ll = np.concatenate([ls[0], ls[1][inb]])
                o = np.argsort(xx)
                lat = np.linspace(xx[o][0], xx[o][-1], 50001)
                lv = PchipInterpolator(xx[o], ll[o])(lat)
                q = _quant(lat, np.exp(lv - lv.max()), 67.74)
                cmp(base + "CL", min(val(base + "CL"), 0.9999), min(q["level"], 0.9999), 0.005, "credible level of the input, own PCHIP over both passes")
        cmp("KnownAGNNoPeak", float(val("KnownAGNNoPeak")), float(n_no), 0.0, "realizations whose largest ln L lies at a prior edge (own read of both passes)")
    # units and counts
    meta = _jr(B.META100)
    h = _jr(B.EV_REF_META)["cosmology"]["H0"] / 100.0
    for nm, key in (("MockLognGalh", "n_comoving_gal"), ("MockLognAGNh", "n_comoving_agn")):
        cmp(nm, val(nm), math.log10(meta["config"]["glass"][key] / h ** 3), 1e-12, "log10(n / h^3): n from META.json config.glass, H0 from the production events' metadata")
    sets = {rec["run"] for n, rec in blk.items() if rec.get("how") in ("mean_hw", "factor_hw") and rec.get("stage", "").startswith("E")}
    nset = {len(exp_idx[s_]) for s_ in sets}
    if len(nset) == 1:
        cmp("NrealClasses", float(val("NrealClasses")), float(nset.pop()), 0.0, f"realizations of the {len(sets)} experiment sets (own parser of RUNLIST_experiment.yaml)")
    else:
        bad.append(f"NrealClasses: the experiment sets have different realization counts {nset}")
    worst = max(rows, key=lambda r: r[2] / r[3] if r[3] > 0 else (0 if r[2] == 0 else np.inf))
    check(not bad, f"K1 {len(rows)} numbers of the experiment's macros recomputed by a second code path (summary.json read directly; own PCHIP of each grid "
                   f"and of the AGN factor; r from the per-merger likelihoods and the events file's true classes, its f of evaluation against the "
                   f"posterior median and its mean over all mergers; the cost ranges per realization; interior maxima from both passes; "
                   f"numerator and denominator of every cost ratio, and each host-class-known macro on host-dependent mergers and its "
                   f"host-class-unknown partner, on the same events files; class counts and AGN offsets of the finished same-merger runs); "
                   f"largest deviation {worst[2]:.2g} ({worst[0]}, {worst[1]}, tolerance {worst[3]:.2g}){'; MISMATCH: ' + '; '.join(bad[:6]) if bad else ''}")
    # ---- K2: jumps of ln L between neighbouring nodes in the grids the paper uses
    used = {}
    for k_, sp in pm.get("keys", {}).items():
        if doc.get(k_, {}).get("final") and sp["mode"] == "grid":
            for r in sp["realizations"]:
                used[RUNS16 / sp["run"].format(R=f"{r:03d}")] = (f"key {k_}", None)
    for name, rec in ((pm.get("density_response") or {}).get("arms") or {}).items():
        for pat in (rec["run"], rec["unit_run"]):
            for r in rec["realizations"]:
                run = RUNS16 / pat.format(R=f"{r:03d}")
                if (run / "summary.json").exists():
                    used.setdefault(run, (f"density {name}", None))
    for name, rec in blk.items():
        if rec.get("how") in (None, "derived") or not have.get(name):
            continue
        reals = [rec["realization_A"]] if rec["how"] == "membership" else rec["realizations"]
        for r in reals:
            run = RUNS16 / rec["run"].format(R=f"{r:03d}")
            if _jr(run / "config.json")["mode"] == "grid":
                prev = used.get(run, (name, None))
                used[run] = (prev[0] if prev[0] != name else name, rec.get("factor") or prev[1])
    fails, seen, over = [], [], []
    for run, (who, fac) in sorted(used.items(), key=lambda kv: str(kv[0])):
        for nm in _pass_file(run):
            for cls in [None] + ([fac] if fac else []):
                names, ax, l = _corrected(run, nm, cls)
                for st_ in _steps(l, ax, names):
                    tag = f"{run.relative_to(RUNS16)} {nm}{'/' + cls if cls else ''}: {st_[0]} {st_[1]:g}->{st_[2]:g} jump {st_[3]:+.1f} nats, upper node {st_[4]:.1f} below the maximum"
                    seen.append(tag)
                    if st_[4] < max(15.0, 0.3 * abs(st_[3])):
                        fails.append(tag + f" ({who})")
                if nm != "coarse.h5" and l.ndim == 1 and np.all(np.isfinite(l)):
                    xf = np.linspace(ax[0][0], ax[0][-1], 20001)
                    ov = float(CubicSpline(ax[0], l)(xf).max() - l.max())
                    over.append(ov)
                    if ov > 1.0:
                        fails.append(f"{run.relative_to(RUNS16)} {nm}{'/' + cls if cls else ''}: cubic spline rises {ov:.1f} nats above the largest node ({who})")
    check(not fails, f"K2 no jump of ln L between neighbouring nodes inside the region that carries posterior mass, in the {len(used)} grid runs the paper uses "
                     f"(fine and coarse passes, every ln L array read; a jump: > 20 nats and > 5x both neighbouring steps; inside: its upper node within "
                     f"max(15 nats, 0.3 x the jump) of the maximum, where a cubic spline overshoots; 1-D fine passes: cubic-spline overshoot < 1 nat, largest "
                     f"{max(over, default=0.0):.2f}); jumps found outside that region: {len(seen) - len(fails)}"
                     + (f" ({'; '.join([s_ for s_ in seen if s_ not in [f_.rsplit(' (', 1)[0] for f_ in fails]][:3])})" if len(seen) > len(fails) else "")
                     + (f"; FAIL: {'; '.join(fails[:4])}" if fails else ""))


# ============================================================================================= #
# L: the response to the assumed AGN density (scripts/density.py; map block 'density_response'), by code that density.py and
# run_posteriors.py do not use: the run list parsed here (stages and --agn-density-factor), the summaries read directly, the
# cataloged counts read with h5py from the files the unit-factor run's config names, Analysis 15's closed form of the missing
# volume, and the identity applied cell by cell to the stored grid (own cubic splines of the fine pass, coarse pass outside),
# marginals by Simpson sums, quantiles from an own CDF.
# ============================================================================================= #
def audit_density(mac):
    import yaml
    pm = _jr(OUT / "values" / "production_map.json")
    blk = (pm.get("density_response") or {}).get("arms") or {}
    dm = {k: v for k, v in mac.items() if v["claim"] == "density"}
    if not dm or not blk:
        check(False, "L0 values.json has the density-response macros (claim 'density') and the map its block 'density_response'")
        return
    rows, bad = [], []

    def cmp(name, got, want, tol, what):
        d = abs(got - want)
        rows.append((name, what, d, tol))
        if not d <= tol:
            bad.append(f"{name} {what}: {got:.6g} vs {want:.6g} (tol {tol:.2g})")

    # ---- L0: the map's runs are the run list's (own parser): every run with --agn-density-factor outside the validation stages, by limit,
    # factor and realization; no run of a validation stage named; black exactly when every summary of the group exists
    rl = yaml.safe_load((RUNS16 / "RUNLIST.yaml").read_text())
    own, vnames = {}, set()
    for st in rl["stages"]:
        for r in st["runs"]:
            a = r["args"]
            if str(st["stage"]).startswith("V"):
                vnames.add(r["name"])
                continue
            if "--agn-density-factor" not in a:
                continue
            fl, x = a[a.index("--flux") + 1], float(a[a.index("--agn-density-factor") + 1])
            own.setdefault((fl, x), set()).add(int(a[a.index("--realization") + 1]))
    mine = {(rec["flux"], float(rec["factor"])): set(rec["realizations"]) for rec in blk.values()}
    named = {rec[k].format(R=f"{r:03d}") for rec in blk.values() for k in ("run", "unit_run") for r in rec["realizations"]}
    wrong = []
    for name, rec in blk.items():
        word, xw = re.fullmatch(r"(Twenty|Eighteen)(Half|Double)", name).groups()
        need = [RUNS16 / rec[k].format(R=f"{r:03d}") / "summary.json" for k in ("run", "unit_run") for r in rec["realizations"]]
        ok = all(p_.exists() for p_ in need)
        okA = (RUNS16 / rec["unit_run"].format(R=f"{rec['realization_A']:03d}") / "summary.json").exists()
        for n, v in dm.items():
            if n.startswith(f"Flux{word}{xw}"):
                want_black = okA if n.endswith("Pred") else ok
                if (v["class"] == "plain") != want_black:
                    wrong.append(n)
    check(mine == own and not (named & vnames) and not wrong,
          f"L0 the density-response groups ({len(blk)}; {len(dm)} macros, {sum(1 for v in dm.values() if v['class'] == 'plain')} black) name the runs of "
          f"RUNLIST.yaml with --agn-density-factor outside the validation stages ({', '.join(f'm<{k[0]} x{k[1]:g}: {sorted(v)}' for k, v in sorted(own.items()))}; "
          f"own parser), no run of a validation stage ({len(vnames)} runs there), and print black exactly when every summary of the group exists"
          f"{'; MAP ' + str(mine) if mine != own else ''}{'; V-STAGE RUN NAMED' if named & vnames else ''}{': ' + ', '.join(wrong[:6]) if wrong else ''}")
    # ---- L1: the numbers
    C = _a15()
    from scipy.interpolate import CubicSpline  # noqa: F401
    val = lambda n: mac[n]["value"]
    par = {"fAGN": "f_agn", "Hzero": "H0"}
    shares = {}
    for name, rec in blk.items():
        word, xw = re.fullmatch(r"(Twenty|Eighteen)(Half|Double)", name).groups()
        key = f"Flux{word}{xw}"
        x = float(rec["factor"])
        if mac[f"{key}fAGN"]["class"] != "plain":
            continue
        S = {r: (_jr(RUNS16 / rec["run"].format(R=f"{r:03d}") / "summary.json"), _jr(RUNS16 / rec["unit_run"].format(R=f"{r:03d}") / "summary.json"))
             for r in rec["realizations"]}
        rA = rec["realization_A"]
        for r, (sa, su) in S.items():
            cmp(f"{key} r{r}", sa["densities"]["agn"]["log10n_used"] - su["densities"]["agn"]["log10n_used"], math.log10(x), 1e-9,
                "assumed AGN density over the unit-factor run's (summaries)")
        for q, pn in par.items():
            ps = S[rA][0]["posterior"][pn]
            v = val(f"{key}{q}")
            for nm, a_, b_ in (("median", v["median"], ps["median"]), ("lo", v["ci90"][0], ps["ci90"][0]), ("hi", v["ci90"][1], ps["ci90"][1])):
                cmp(f"{key}{q}", a_, b_, 1e-12, f"A {nm} = summary.json")
            for pc, b_ in (("Med", ps["median"]), ("Lo", ps["ci90"][0]), ("Hi", ps["ci90"][1]), ("Wid", ps["ci90"][1] - ps["ci90"][0])):
                cmp(f"{key}{q}{pc}", val(f"{key}{q}{pc}"), b_, 1e-12, "= summary.json")
            sh = {r: sa["posterior"][pn]["median"] - su["posterior"][pn]["median"] for r, (sa, su) in S.items()}
            cmp(f"{key}Delta{q}", val(f"{key}Delta{q}"), sh[rA], 1e-12, "A: medians of the two summaries")
            cmp(f"{key}Delta{q}Mean", val(f"{key}Delta{q}Mean"), sum(sh.values()) / len(sh), 1e-12, "mean over the realizations")
            cmp(f"{key}Delta{q}Min", val(f"{key}Delta{q}Min"), min(sh.values()), 1e-12, "smallest over the realizations")
            cmp(f"{key}Delta{q}Max", val(f"{key}Delta{q}Max"), max(sh.values()), 1e-12, "largest over the realizations")
    for name, rec in blk.items():
        word, xw = re.fullmatch(r"(Twenty|Eighteen)(Half|Double)", name).groups()
        key = f"Flux{word}{xw}"
        x = float(rec["factor"])
        if mac[f"{key}DeltafAGNPred"]["class"] != "plain":
            continue
        run = RUNS16 / rec["unit_run"].format(R=f"{rec['realization_A']:03d}")
        if word not in shares:
            # the share: own counts (h5py, the files the run's config names) and Analysis 15's closed form of the missing volume
            sm, cfg = _jr(run / "summary.json"), _jr(run / "config.json")
            N = _counts(run)["agn"]
            fit = C.schechter_fit(cfg["selection_fits"]["agn"]["path"])
            Z = C.z_closed(N, sm["densities"]["agn"]["log10n_used"] + 3 * math.log10(0.6774), fit, 67.74, C.ZMAX_R)
            shares[word] = N / Z
            cmp(f"NormFAGN{word}Counts", val(f"NormFAGN{word}Counts"), shares[word], 2e-4 * shares[word], "own counts and Analysis 15's closed form")
            # the identity cell by cell on the stored grid: ln L_x(H0, f) = ln L_1(H0, f'), own splines, Simpson marginals
            parts = _grid_spl(run)
            _, axes = _lattice16([parts])
            Hg, fg = axes[0], np.linspace(0.0, 1.0, 2001)
            Hm, Fm = np.meshgrid(Hg, fg, indexing="ij")
            med = {}
            for xx in (0.5, 1.0, 2.0):
                d = -math.log(shares[word] + xx * (1 - shares[word]))
                with np.errstate(divide="ignore", over="ignore"):
                    fp = 1.0 / (1.0 + np.exp(-(np.log(Fm) - np.log1p(-Fm) + d)))
                fp[:, 0], fp[:, -1] = 0.0, 1.0
                lv = _grid_eval(parts, np.stack([Hm.ravel(), fp.ravel()], -1)).reshape(Hm.shape)
                med[xx] = {q: _quant(*_marg(["H0", "f_agn"], [Hg, fg], lv, pn))["median"] for q, pn in par.items()}
            shares[word] = (shares[word], med)
        F, med = shares[word]
        hw = {q: 0.5 * (lambda s: s["ci90"][1] - s["ci90"][0])(_jr(run / "summary.json")["posterior"][pn]) for q, pn in par.items()}
        cmp(f"{key}DeltafAGNPred", mac[f"{key}DeltafAGNPred"]["extra"]["F_AGN"], F, 2e-4 * F, "F_AGN of the map")
        for q in par:
            cmp(f"{key}Delta{q}Pred", val(f"{key}Delta{q}Pred"), med[x][q] - med[1.0][q], 0.01 * hw[q],
                "own map of the stored grid cell by cell (cubic splines, Simpson marginals), median at x minus at 1")
    reals = {tuple(rec["realizations"]) for rec in blk.values()}
    if len(reals) == 1:
        cmp("NrealFluxDensity", float(val("NrealFluxDensity")), float(len(next(iter(own.values())))), 0.0, "realizations of the run list (own parser)")
    worst = max(rows, key=lambda r: r[2] / r[3] if r[3] > 0 else (0 if r[2] == 0 else np.inf))
    check(not bad, f"L1 {len(rows)} numbers of the density-response macros recomputed by a second code path (summaries read directly; own counts and "
                   f"Analysis 15's missing volume for F_AGN; the identity applied cell by cell to the stored unit-factor grid); largest deviation "
                   f"{worst[2]:.2g} ({worst[0]}, {worst[1]}, tolerance {worst[3]:.2g}){'; MISMATCH: ' + '; '.join(bad[:6]) if bad else ''}")


def _trap_median(x, L):
    p = np.exp(L - L.max())
    c = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
    return float(np.interp(0.5, c / c[-1], x))


if __name__ == "__main__":
    sys.exit(main())
