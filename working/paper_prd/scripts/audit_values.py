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
                 if not str(st["stage"]).startswith(("V", "X"))
                 and re.fullmatch(r"known_population/both_catalogs/complete/equal_populations/realization_\d{3}", r["name"])}
                | set(_own_kb_stage_x()[1]))        # and KnownBoth's deliberate stage-X realizations 111-120 (21 Nov)
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
    kdoc = json.loads((OUT / "values" / "placeholder_results.json").read_text())["keys"]
    for k in keys:
        for q in need[k]:
            # a number density the data do not bound on both sides (Comb 'limit_only', production.edge_bounds; recomputed in J1): its
            # combination has the 90% upper limit only, and no combined interval, width or credible level may exist
            lim = bool(((kdoc.get(k, {}).get("quantities", {}).get(q) or {}).get("Comb") or {}).get("limit_only"))
            sufs = ("", "Med", "Lo", "Hi", "Wid") + (("CombUL",) if lim else ("Comb", "CombMed", "CombLo", "CombHi", "CombWid"))
            for suf in sufs:
                if f"{k}{q}{suf}" not in mac:
                    missing.append(f"{k}{q}{suf}")
            if lim:
                missing += [f"{k}{q}{s} (must not exist)" for s in ("Comb", "CombMed", "CombLo", "CombHi", "CombWid", "CL") if f"{k}{q}{s}" in mac]
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
    audit_density_phys(mac)
    audit_skyres(mac)
    audit_skyres_all(mac)
    audit_skyres_sets(mac)
    audit_classes_comb(mac)


# the further macros of PAPER_SKELETON.md Sec. 6 and the keys they are built from
FURTHER = {"HDBiasHzero": ["HDSingle"], "ObsBiasHzero": ["ObsSingle"], "HDLB": ["HDHost"], "ObsLB": ["ObsHost"], "WidthRatioAGNGal": ["Galhosted", "AGNhosted"],
           "MassScaleShareKnown": ["KnownNone", "KnownBoth"], "MassScaleShareSingle": ["SingleNone", "SingleBoth"], "CorrHzeroMuGNone": ["SingleNone"],
           "CorrHzeroMuGBoth": ["SingleBoth"],
           # 23 Nov (Sec. IV E on three steps): the shift of the assumed population against the two populations at their inputs needs both keys;
           # the per-realization Bayes factors are those of the host-dependent key (its reference runs are part of it)
           **{f"{pre}AssumedShiftHzero{s}": [f"{pre}Assumed", f"{pre}Pops"] for pre in ("HD", "Obs") for s in ("", "Min", "Max")},
           **{f"{pre}LB{s}": [f"{pre}Host"] for pre in ("HD", "Obs") for s in ("Min", "Max")}}


def macro_keys(mac, doc):
    """key-scheme macro -> the keys it is built from (longest key prefix, or FURTHER)."""
    keys = sorted(doc, key=len, reverse=True)
    out = {}
    for n, v in mac.items():
        # 23 Nov: macros over a subset of a key's realizations (claim 'skeleton,subsets', Obs<Pops|Assumed><First|Later>...) follow their
        # subset's summaries, not the key's state (audit J6)
        if not v["claim"].startswith("skeleton") or v["claim"] == "skeleton,subsets":
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
        if str(st["stage"]).startswith(("V", "X")):       # validation and check stages are not production runs
            continue
        for r in st["runs"]:
            a = r["args"]
            d = {a[i][2:]: a[i + 1] for i in range(0, len(a) - 1, 2)}
            if "agn-density-factor" in d:
                continue
            ek = d["events-kind"] + (d["events-suffix"] if d.get("events-suffix") else "")      # Stage 5: host_dependent + _obslike
            cnt.setdefault((d["analysis"], d["catalogs"], d["flux"], ek, d.get("density")), set()).add(d["realization"])
    n = lambda *k: len(cnt.get(k, ()))
    # KnownBoth: stage 1's realizations and the deliberate stage-X runs realization_NNN_nside32 (own parser, _own_kb_stage_x)
    _s1, _xr = _own_kb_stage_x()
    want = {"NrealKnown": n("known_population", "both", "complete", "equal_populations", None) + len(_xr),
            "NrealKnownAlone": {n("known_population", c, "complete", "equal_populations", None) for c in ("galaxy", "agn", "none")},
            "NrealPure": {n("known_population", "agn", "complete", "agn_hosted", None), n("known_population", "galaxy", "complete", "galaxy_hosted", None)},
            "NrealFlux": {n("known_population", "both", m, "equal_populations", "counts") for m in ("21", "20", "19", "18")},
            "NrealFree": {n("free_densities", "both", m, "equal_populations", "free") for m in ("21", "20", "19", "18")},
            "NrealSingle": {n("single_population", c, "complete", "equal_populations", None) for c in ("both", "none")},
            # 23 Nov: key HDHost's runs are stage V8's realization_NNN_unif_then_rslice (own parser; build_values.STAGE_V_KEY_RUNS)
            "NrealHD": len(_own_hd_stage_v()[1]),
            "NrealHDSingle": n("single_population", "both", "complete", "host_dependent", None),
            "NrealHDNone": n("single_population", "none", "complete", "host_dependent", None)}
    runs_txt = (OUT / "runs.txt").read_text()
    want["NrealObs"] = {n("single_population", c, "complete", "host_dependent_obslike", None) for c in ("both", "none")} | {
        n("host_dependent_populations", "both", "complete", "host_dependent_obslike", None)}
    rx = yaml.safe_load((B.AN / "analysis_16_production_runs" / "runs" / "RUNLIST_experiment.yaml").read_text())
    ctrl = {lab: {r["args"][r["args"].index("--realization") + 1] for st in rx["stages"] for r in st["runs"]
                  if r["analysis"] == "two_tracers_one_population" and r["set"]["label"] == lab} for lab in ("peak_f0p00_s0", "peak_f0p10_s0")}
    want["NrealCtrl"] = {len(v) for v in ctrl.values()} if all(ctrl.values()) else (5 if "five for the others" in runs_txt else -1)
    # 23 Nov: the experiment's keys of Sec. IV E (own parser of RUNLIST_experiment.yaml: analysis and set label of every run)
    n_exp = lambda an, lab: len({r["args"][r["args"].index("--realization") + 1] for st in rx["stages"] for r in st["runs"]
                                 if r["analysis"] == an and r["set"]["label"] == lab})
    want["NrealHDAssumed"] = {n_exp("two_tracers_one_population", "peak_f0p30_s1"), n_exp("two_tracers_one_population", "peak_f0p30_s2"),
                              n_exp("two_tracers_two_populations", "peak_f0p30_s1")}
    want["NrealHDSingleTwice"] = n_exp("single_population", "peak_f0p30_s2")
    want["NrealObsAssumed"] = {n_exp("two_tracers_one_population", "obslike_f0p30_s1"), n_exp("two_tracers_two_populations", "obslike_f0p30_s1")}
    words = {3: "three", 5: "five", 10: "ten", 15: "fifteen", 20: "twenty"}     # (15: the observed-like keys of Sec. IV F since 23 Nov)
    bad = []
    for k, w in want.items():
        w = w.pop() if isinstance(w, set) and len(w) == 1 else w
        if isinstance(w, set) or val(k) != w or mac[k]["render"] != words.get(w) or mac[k]["class"] != "plain":
            bad.append(f"{k}: {mac[k]['render']} vs {w}")
    bad += [f"{k} N_comb {E['N_comb']} vs macro" for k, E in doc.items()
            for m, ks in (("NrealKnown", ["KnownBoth"]), ("NrealKnownAlone", ["KnownGal", "KnownAGN", "KnownNone"]), ("NrealPure", ["AGNhosted", "Galhosted"]),
                          ("NrealSingle", ["SingleBoth", "SingleNone"]), ("NrealHD", ["HDHost"]), ("NrealHDSingle", ["HDSingle"]), ("NrealHDNone", ["HDNone"]),
                          ("NrealObs", ["ObsSingle", "ObsHost", "ObsNone"]), ("NrealCtrl", ["CtrlZero", "CtrlTenth"]),
                          ("NrealHDAssumed", ["HDAssumed", "HDAssumedTwice", "HDPops"]), ("NrealHDSingleTwice", ["HDSingleTwice"]),
                          ("NrealObsAssumed", ["ObsAssumed", "ObsPops"])) if k in ks and E["N_comb"] != val(m)]
    check(not bad, f"I0 the {len(want)} numbers of realizations equal the run lists (own parser; the control sets: the experiment's f = 0 and f = 0.1 sets; Stage 5 for Obs; "
                   f"the keys of Sec. IV E on three steps: RUNLIST_experiment.yaml's runs on the host-dependent sets; "
                   f"KnownBoth: stage 1 and the {len(_xr)} stage-X runs realization_NNN_nside32 {sorted(_xr)}; HDHost: stage V8's "
                   f"realization_NNN_unif_then_rslice {sorted(_own_hd_stage_v()[1])}), "
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
    # SkyResAllShiftPctHalfWid (21 Nov) is a percentage of a half-width, not a half-width: checked in S12, not here
    not_hw = {"SkyResAllShiftPctHalfWid"}
    hb = [k for k in mac if k.endswith("HalfWid") and k not in not_hw and abs(val(k) - 0.5 * val(k[:-7] + "Wid")) > 1e-12]
    nh = sum(1 for k in mac if k.endswith("HalfWid") and mac[k]["claim"].startswith("skeleton"))
    nh_other = sum(1 for k in mac if k.endswith("HalfWid") and k not in not_hw and not mac[k]["claim"].startswith("skeleton"))
    # (a key that prints only a few macros, build_values.KEY_MACROS_ONLY, has no HalfWid)
    keys_hf = sum(1 for k, E_ in doc.items() if k not in B.KEY_MACROS_ONLY for q in ("Hzero", "fAGN") if q in E_["quantities"])
    check(abs(wr - val("WidthRatioAGNGal")) < 1e-12 and not hb and nh == keys_hf,
          f"I5 WidthRatioAGNGal = GalhostedHzeroWid / AGNhostedHzeroWid ({wr:.3f}); {nh} HalfWid macros of the key scheme, one per key with Hzero or "
          f"fAGN ({keys_hf}), and {nh_other} others (the stage-X checks), each half its Wid{': NOT ' + ', '.join(hb[:4]) if hb else ''}")
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


# The one deliberate exception to "stages V and X never feed a key" (build_values.STAGE_X_KEY_RUNS), read here by own code: key
# KnownBoth's realizations 111-120 exist only as stage-X runs .../realization_NNN_nside32, the same analysis at the same settings.
_KB_RX = r"known_population/both_catalogs/complete/equal_populations/realization_(\d{3})"
SEQ_JSON = B.AN / "analysis_16_production_runs" / "tests" / "results" / "skyres_sequence.json"


def _own_flags(a):
    """{flag: value or True} of a run's argument list, and the number of tokens it accounts for."""
    d, i = {}, 0
    while i < len(a):
        if a[i].startswith("--") and i + 1 < len(a) and not a[i + 1].startswith("--"):
            d[a[i]] = a[i + 1]
            i += 2
        else:
            d[a[i]] = True
            i += 1
    return d


def _own_runlist_idx():
    import yaml
    rl = yaml.safe_load((RUNS16 / "RUNLIST.yaml").read_text())
    return {r["name"]: (str(st["stage"]), list(r["args"])) for st in rl["stages"] for r in st["runs"]}


def _own_kb_stage_x(idx=None):
    """(stage-1 realizations of KnownBoth, {realization: run name} of its deliberate stage-X runs): a stage-X run named
    realization_NNN_nside32 joins when NNN is in 111-120 and has no stage-1 run, and its flags are, one by one, those every stage-1
    KnownBoth run shares (--membership and the realization set aside) with its own realization and --variant nside32, nothing else
    (no --nside, no density factor), each flag given once."""
    idx = idx or _own_runlist_idx()
    s1 = {int(m.group(1)): _own_flags(a) for n, (st, a) in idx.items() if st == "1" for m in [re.fullmatch(_KB_RX, n)] if m}
    common = [{k: v for k, v in f.items() if k not in ("--membership", "--realization")} for f in s1.values()]
    if not common or any(c != common[0] for c in common):
        return sorted(s1), {}
    out = {}
    for n, (st, a) in idx.items():
        m = re.fullmatch(_KB_RX + "_nside32", n)
        if not m or st != "X":
            continue
        R = int(m.group(1))
        want = dict(common[0], **{"--realization": str(R), "--variant": "nside32"})
        if 111 <= R <= 120 and R not in s1 and _own_flags(a) == want and len(a) == 2 * len(want):
            out[R] = n
    return sorted(s1), dict(sorted(out.items()))


# The second deliberate exception (build_values.STAGE_V_KEY_RUNS, 23 Nov), read here by own code: key HDHost's runs are stage V8's
# .../realization_NNN_unif_then_rslice (the eight-parameter model with the mixed sampler: unif, then rslice) for every realization.
_HD_RX = r"host_dependent_populations/both_catalogs/complete/host_dependent/realization_(\d{3})"
_HD_V8 = "host_dependent_populations/both_catalogs/complete/host_dependent/realization_{R}_unif_then_rslice"
_OBS_HOST = "host_dependent_populations/both_catalogs/complete/host_dependent_obslike/realization_{R}_unif_then_rslice"
_OWN_SAMPLER_FLAGS = ("--sample", "--auto-switch-sample", "--auto-switch-cpi", "--init-from", "--switch-sample")


def _own_hd_stage_v(idx=None):
    """(stage 4's realizations of the pure-rslice runs .../host_dependent/realization_NNN, {realization: run name} of key HDHost's runs):
    a stage-V8 run named realization_NNN_unif_then_rslice joins when NNN is one of 100, 101, 102, 103, 105 and its flags, the sampler's
    set aside (_OWN_SAMPLER_FLAGS), are, one by one, those every stage-4 run of the pattern shares (the realization set aside) with its
    own realization and --variant unif_then_rslice, nothing else, each flag given once with its value."""
    idx = idx or _own_runlist_idx()
    s4 = {int(m.group(1)): _own_flags(a) for n, (st, a) in idx.items() if st == "4" for m in [re.fullmatch(_HD_RX, n)] if m}
    common = [{k: v for k, v in f.items() if k != "--realization"} for f in s4.values()]
    if not common or any(c != common[0] for c in common):
        return sorted(s4), {}
    out = {}
    for n, (st, a) in idx.items():
        m = re.fullmatch(_HD_RX + "_unif_then_rslice", n)
        if not m or st != "V8":
            continue
        R, f = int(m.group(1)), _own_flags(a)
        want = dict(common[0], **{"--realization": str(R), "--variant": "unif_then_rslice"})
        if R in (100, 101, 102, 103, 105) and {k: v for k, v in f.items() if k not in _OWN_SAMPLER_FLAGS} == want and len(a) == 2 * len(f):
            out[R] = n
    return sorted(s4), dict(sorted(out.items()))


def _key_run(sp, r):
    """A key's run of realization r as the map names it (runs_by_realization, else the pattern)."""
    return (sp.get("runs_by_realization") or {}).get(str(r)) or sp["run"].format(R=f"{r:03d}")


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


def _edge_ratios(raw, prior, n=4001):
    """Own test of the support: per realization, the weighted Scott-rule Gaussian kernel density of the sampled parameter
    (h^3 Mpc^-3; scipy gaussian_kde, reflected at the prior edges) at each prior edge over its maximum on the box."""
    from scipy.stats import gaussian_kde
    lo, hi = prior
    g = np.linspace(lo, hi, n)
    out = {"lower": [], "upper": []}
    for xh, _H, w in raw:
        keep = w > 1e-9 * w.max()
        k = gaussian_kde(xh[keep], weights=w[keep])
        p = k(g) + k(2 * lo - g) + k(2 * hi - g)
        out["lower"].append(float(p[0] / p.max()))
        out["upper"].append(float(p[-1] / p.max()))
    return out


def _kde_prod_boxed(raw, prior, n=6001, chunk=300):
    """Own combination of log10 n in Mpc^-3 (y = x + 3 log10 h per weighted sample) that respects each sample's own support:
    Scott-rule bandwidth of the weighted y (scipy), each kernel reflected at and truncated to the prior box of x shifted by the
    sample's 3 log10 h; product over the realizations."""
    from scipy.stats import gaussian_kde
    lo, hi = prior
    sh = [3 * np.log10(H / 100.0) for _x, H, _w in raw]
    y = np.linspace(lo + min(s.min() for s in sh), hi + max(s.max() for s in sh), n)
    lp = np.zeros(n)
    for (xh, H, w), s in zip(raw, sh):
        keep = w > 1e-9 * w.max()
        xk, sk, wk = xh[keep], s[keep], w[keep] / w[keep].sum()
        yk = xk + sk
        bw = float(np.sqrt(gaussian_kde(yk, weights=wk).covariance[0, 0]))
        p = np.zeros(n)
        for c in np.array_split(np.arange(len(yk)), max(1, len(yk) // chunk)):
            el, eh = lo + sk[c], hi + sk[c]
            kk = sum(np.exp(-0.5 * ((y[:, None] - m[None, :]) / bw) ** 2) for m in (yk[c], 2 * el - yk[c], 2 * eh - yk[c]))
            kk = kk * ((y[:, None] >= el[None, :]) & (y[:, None] <= eh[None, :]))
            p += kk @ wk[c]
        lp += np.log(np.maximum(p, 1e-300))
    p = np.exp(lp - lp.max())
    return y, p / np.trapz(p, y)


def _prior_limits_own(spec, n=4_000_000, seed=424242):
    """Own prior-induced 90% upper limits of the two densities (realization A of a Free key): uniform draws from the free-density
    run's priors (its config.json), weighted by the known-density grid of the same realization and flux limit (own splines of its
    stored ln L) at f* = expit(logit f + ln[F_AGN(n_AGN)/F_gal(n_gal)] - ln[F_AGN/F_gal at the densities of that run]),
    F_k = N_k/(N_k + n_k V_miss,k): N_k counted with h5py in the survey files the known-density run names, n_k V_miss,k from
    Analysis 15's closed form at H0 = 67.74 (n in Mpc^-3 = 10^(log10 n_h) h^3, so n V does not depend on H0). -> {cls: (UL h^3 Mpc^-3, UL Mpc^-3)}."""
    C = _a15()
    fdir = RUNS16 / spec["prior_induced_from"]
    sm, cfg = _jr(fdir / "summary.json"), _jr(fdir / "config.json")
    pri = _jr(RUNS16 / spec["run"].format(R=f"{spec['realization_A']:03d}") / "config.json")["priors"]
    N = _counts(fdir)
    lh = 3 * math.log10(0.6774)
    Vp = {c: C.z_closed(0, lh, C.schechter_fit(cfg["selection_fits"][c]["path"]), 67.74, C.ZMAX_R) for c in ("gal", "agn")}
    F = lambda c, l: N[c] / (N[c] + 10.0 ** np.asarray(l, float) * Vp[c])
    used = {c: sm["densities"][c]["log10n_used"] for c in ("gal", "agn")}
    rng = np.random.default_rng(seed)
    H, f, lg, la = (rng.uniform(*pri[p], n) for p in ("H0", "f_agn", "log10n_gal", "log10n_agn"))
    f = np.clip(f, 1e-12, 1 - 1e-12)
    lgt = np.log(f) - np.log1p(-f) + np.log(F("agn", la) / F("gal", lg)) - math.log(F("agn", used["agn"]) / F("gal", used["gal"]))
    fs = 1.0 / (1.0 + np.exp(-lgt))
    lv = _grid_eval(_grid_spl(fdir), np.stack([H, fs], 1))
    lv = np.where(np.isfinite(lv), lv, -np.inf)
    w = np.exp(lv - lv.max())
    conv = 3 * np.log10(H / 100.0)
    neff = float(w.sum() ** 2 / np.sum(w ** 2))
    out = {"n_eff": neff}
    for c, x in (("Gal", lg), ("AGN", la)):
        out[c] = []
        for y in (x, x + conv):
            q88, q90, q92 = _wq(y, w, [0.88, 0.90, 0.92])
            # Monte Carlo error of a weighted 90% quantile per unit 1/sqrt(n_eff): sqrt(0.9 x 0.1) / (density at the quantile)
            out[c].append((float(q90), float(math.sqrt(0.09) * (q92 - q88) / 0.04)))
    return out


def audit_knownboth_twenty(mac, doc, specs):
    """J2 (21 Nov): key KnownBoth over twenty realizations, by own code: the run list parsed here (_own_kb_stage_x: stage 1's
    realizations and the deliberate stage-X runs realization_NNN_nside32, flag by flag), each stage-X run's config against a stage-1
    config (only the realization's own files differ; survey files at nside 32 by h5py), the map and placeholder_results.json naming
    exactly those runs, NrealKnown and N_comb twenty, realization A's macros the stage-1 run's summary; the combination recomputed
    by _grid_comb_safe (own summed splines, PCHIP where a pass has non-finite cells) and compared with the reference computation of
    analysis_16 (tests/results/skyres_sequence.json, block KnownBoth_production_111_120, nside 32, combined over all twenty)."""
    rows, bad = [], []

    def cmp(name, got, want, tol, what):
        d = abs(got - want)
        rows.append((name, what, d, tol))
        if not d <= tol:
            bad.append(f"{name} {what}: {got:.6g} vs {want:.6g} (tol {tol:.2g})")
    idx = _own_runlist_idx()
    s1, xr = _own_kb_stage_x(idx)
    sp = specs.get("KnownBoth") or {}
    want_reals = sorted(set(s1) | set(xr))
    why = []
    if sp.get("realizations") != want_reals:
        why.append(f"map realizations {sp.get('realizations')} vs own {want_reals}")
    if {int(k): v for k, v in (sp.get("runs_by_realization") or {}).items()} != xr:
        why.append("map runs_by_realization differ from the own parse")
    others = [k for k, s in specs.items() if k != "KnownBoth" and s.get("runs_by_realization")]
    if others:
        why.append(f"other keys name runs per realization: {others}")
    E = doc.get("KnownBoth") or {}
    if E.get("N_comb") != len(want_reals) or E.get("realizations") != want_reals:
        why.append(f"placeholder_results.json KnownBoth N_comb {E.get('N_comb')} realizations {E.get('realizations')}")
    nr = mac["NrealKnown"]
    if nr["value"] != len(want_reals) or nr["render"] != _OWN_WORDS[len(want_reals)] or nr["class"] != "plain":
        why.append(f"NrealKnown '{nr['render']}' ({nr['value']})")
    # configs: a stage-X run differs from a stage-1 run only by the realization's own files
    c1 = _jr(RUNS16 / sp["run"].format(R=f"{s1[-1]:03d}") / "config.json")
    own_files = {"events_file", "events_truth", "injections_file", "injections_files", "realization", "redraw_key", "run_name", "surveys", "variant", "membership"}
    for R, n in xr.items():
        c = _jr(RUNS16 / n / "config.json")
        for k in sorted((set(c) | set(c1)) - own_files):
            if c.get(k) != c1.get(k):
                why.append(f"{n}: config {k} {str(c.get(k))[:40]} vs {str(c1.get(k))[:40]}")
        rk, rk1 = list(c["redraw_key"]), list(c1["redraw_key"])
        if rk[:1] + rk[2:] != rk1[:1] + rk1[2:] or rk[1] != R or c["realization"] != R or c.get("variant") != "nside32":
            why.append(f"{n}: redraw key {rk}, realization {c['realization']}, variant {c.get('variant')}")
        if str(c["events_file"]) != str(B.DATA / f"seed{R}" / "events" / "events_corepop.h5") or \
                [str(Path(f)) for f in c["injections_files"]] != [str(B.DATA / f"seed{R}" / "injections" / "injections_targeted.h5")]:
            why.append(f"{n}: events or injections not the realization's own")
        for t, path in c["surveys"].items():
            with h5py.File(path, "r") as h:
                if int(h.attrs["nside"]) != 32 or f"seed{R}/" not in str(path):
                    why.append(f"{n}: survey {t} {path}")
        if c.get("membership"):
            why.append(f"{n}: membership requested")
    # the combination, own and the reference computation
    runs = [RUNS16 / _key_run(sp, r) for r in want_reals]
    names, axes, tot = _grid_comb_safe(runs)
    seq = _jr(SEQ_JSON)["analyses"]["KnownBoth_production_111_120"]["nsides"]["32"]
    seqc = seq.get("combined") or {}
    if seqc.get("realizations") != want_reals:
        why.append(f"skyres_sequence.json combined over {seqc.get('realizations')}")
    if not all(seq["runs"][str(R)]["run"] == _key_run(sp, R) for R in want_reals if str(R) in seq["runs"]):
        why.append("skyres_sequence.json reads other runs")
    for q, par in (("Hzero", "H0"), ("fAGN", "f_agn")):
        v = mac[f"KnownBoth{q}Comb"]["value"]
        x0 = doc["KnownBoth"]["quantities"][q]["input"]
        own = _quant(*_marg(names, axes, tot, par), x0)
        hw = 0.5 * (v["ci90"][1] - v["ci90"][0])
        tol = max(0.01 * hw, 0.005 if par == "H0" else 5e-4)
        for nm, a, b in (("median", v["median"], own["median"]), ("lo", v["ci90"][0], own["ci90"][0]), ("hi", v["ci90"][1], own["ci90"][1])):
            cmp(f"KnownBoth{q}Comb", a, b, tol, f"{nm} vs own summed splines (_grid_comb_safe)")
        cmp(f"KnownBoth{q}CL", mac[f"KnownBoth{q}CL"]["value"], own["level"], 0.01, "credible level vs own")
        r = seqc.get(par) or {}
        if r:
            for nm, a, b in (("median", v["median"], r["median"]), ("lo", v["ci90"][0], r["median"] - r["minus"]), ("hi", v["ci90"][1], r["median"] + r["plus"])):
                cmp(f"KnownBoth{q}Comb", a, b, 1e-6, f"{nm} vs skyres_sequence.json")
            cmp(f"KnownBoth{q}CL", mac[f"KnownBoth{q}CL"]["value"], r["level_of_input"], 1e-6, "credible level vs skyres_sequence.json")
            cmp(f"KnownBoth{q}Comb", x0, r["input"], 1e-12, "input = skyres_sequence.json's")
        else:
            why.append(f"skyres_sequence.json has no combined {par}")
    # realization A unchanged: the stage-1 run's own summary
    sA = _jr(RUNS16 / sp["run"].format(R="100") / "summary.json")
    for q, par in (("Hzero", "H0"), ("fAGN", "f_agn")):
        v = mac[f"KnownBoth{q}"]["value"]
        cmp(f"KnownBoth{q}", v["median"], sA["posterior"][par]["median"], 1e-12, "A = stage 1's realization 100")
    worst = max(rows, key=lambda r: r[2] / r[3] if r[3] > 0 else (0 if r[2] == 0 else np.inf))
    check(not why and not bad,
          f"J2 key KnownBoth over {len(want_reals)} realizations (own parser: stage 1 {s1}, stage X realization_NNN_nside32 {sorted(xr)} with stage 1's "
          f"flags plus --variant nside32 only): the map, placeholder_results.json and NrealKnown ('{nr['render']}') name exactly those runs; each "
          f"stage-X config as stage 1's but for the realization's own events, injections and nside-32 survey files (h5py); no other key names runs "
          f"per realization; {len(rows)} numbers of the combination against the own summed splines and against tests/results/skyres_sequence.json "
          f"(KnownBoth_production_111_120, nside 32: H0 {seqc.get('H0', {}).get('median', float('nan')):.3f}, f_AGN {seqc.get('f_agn', {}).get('median', float('nan')):.4f}); "
          f"largest deviation {worst[2]:.2g} ({worst[0]}, {worst[1]}, tolerance {worst[3]:.2g})"
          + (f"; FAIL: {'; '.join((why + bad)[:5])}" if why or bad else ""))


def audit_hdhost_mixed(mac, doc, specs):
    """J4 (23 Nov): key HDHost reads stage V8's runs realization_NNN_unif_then_rslice (the second deliberate exception,
    build_values.STAGE_V_KEY_RUNS), by own code: the run list parsed here (_own_hd_stage_v: stage 4's flags plus the sampler's and
    --variant unif_then_rslice only); the map's key HDHost (pattern, realizations, stage, no per-realization names), the classes block's
    HWHost, placeholder_results.json and NrealHD name exactly those runs; ObsHost and HWObsHost name stage 5's own mixed-sampler runs;
    stage 4's pure-rslice run of realization 100 is named by no key or macro; each V8 config is stage 4's realization-100 config but for
    the realization's own files and the sampler (nested.sample, nested.auto_switch); while the key is real, realization A's macros
    are the V8 run's summary, and HWHost, once black, is the mean half-width of the V8 summaries."""
    import statistics
    why = []
    idx = _own_runlist_idx()
    s4, vr = _own_hd_stage_v(idx)
    reals = sorted(vr)
    sp = specs.get("HDHost") or {}
    if sp.get("run") != _HD_V8 or sp.get("realizations") != reals or sp.get("stage") != "V8" or sp.get("runs_by_realization"):
        why.append(f"map HDHost run {sp.get('run')} realizations {sp.get('realizations')} stage {sp.get('stage')} vs own {_HD_V8} {reals} V8")
    if (sp.get("LB") or {}).get("reference_run") != "single_population/both_catalogs/complete/host_dependent/realization_{R}":
        why.append(f"map HDHost Bayes-factor reference {(sp.get('LB') or {}).get('reference_run')}")
    so = specs.get("ObsHost") or {}
    obs5 = sorted(int(m.group(1)) for n, (st, _a) in idx.items() if st == "5"
                  for m in [re.fullmatch(re.escape(_OBS_HOST).replace(re.escape("{R}"), r"(\d{3})"), n)] if m)
    if so.get("run") != _OBS_HOST or so.get("realizations") != obs5 or so.get("stage") != "5":
        why.append(f"map ObsHost run {so.get('run')} realizations {so.get('realizations')} stage {so.get('stage')} vs stage 5's {obs5}")
    blk = ((_jr(OUT / "values" / "production_map.json").get("classes") or {}).get("macros") or {})
    hw, hwo = blk.get("HWHost") or {}, blk.get("HWObsHost") or {}
    if hw.get("run") != _HD_V8 or hw.get("realizations") != reals or hw.get("stage") != "V8":
        why.append(f"classes HWHost {hw.get('run')} {hw.get('realizations')} {hw.get('stage')}")
    if hwo.get("run") != _OBS_HOST or hwo.get("realizations") != obs5:
        why.append(f"classes HWObsHost {hwo.get('run')} {hwo.get('realizations')}")
    E = doc.get("HDHost") or {}
    if E.get("N_comb") != len(reals) or E.get("realizations") != reals:
        why.append(f"placeholder_results.json HDHost N_comb {E.get('N_comb')} realizations {E.get('realizations')}")
    nr = mac["NrealHD"]
    if nr["value"] != len(reals) or nr["render"] != _OWN_WORDS[len(reals)] or nr["class"] != "plain":
        why.append(f"NrealHD '{nr['render']}' ({nr['value']})")
    # stage 4's pure-rslice run of realization 100 (the sampler reference) feeds nothing
    ref = _HD_V8.replace("_unif_then_rslice", "").format(R="100")
    named ={_key_run(s, R) for s in specs.values() if isinstance(s, dict) and "run" in s for R in s.get("realizations", [])}
    named |= {r["run"].format(R=f"{R:03d}") for r in blk.values() if isinstance(r, dict) and "run" in r for R in r.get("realizations", [])}
    if ref in named:
        why.append(f"{ref} is named by the map")
    # configs: stage 4's realization-100 config but for the realization's own files and the sampler
    own_files = {"events_file", "events_truth", "injections_file", "injections_files", "realization", "redraw_key", "run_name", "surveys", "variant"}
    c4p = RUNS16 / ref / "config.json"
    ncfg = 0
    if c4p.exists():
        c4 = _jr(c4p)
        for R, n in vr.items():
            p = RUNS16 / n / "config.json"
            if not p.exists():
                continue
            ncfg += 1
            c = _jr(p)
            for k in sorted((set(c) | set(c4)) - own_files - {"nested"}):
                if c.get(k) != c4.get(k):
                    why.append(f"{n}: config {k} {str(c.get(k))[:40]} vs {str(c4.get(k))[:40]}")
            nd = {k for k in set(c["nested"]) | set(c4["nested"]) if c["nested"].get(k) != c4["nested"].get(k)} - {"sample", "auto_switch"}
            if nd:
                why.append(f"{n}: nested {sorted(nd)} differ from stage 4's")
            rk, rk4 = list(c["redraw_key"]), list(c4["redraw_key"])
            if rk[:1] + rk[2:] != rk4[:1] + rk4[2:] or rk[1] != R or c["realization"] != R or c.get("variant") != "unif_then_rslice":
                why.append(f"{n}: redraw key {rk}, realization {c['realization']}, variant {c.get('variant')}")
    else:
        why.append(f"no config of {ref} (stage 4's sampler reference)")
    # numbers: realization A from the V8 run's summary; HWHost the mean half-width of the V8 summaries
    sums = {R: (RUNS16 / n / "summary.json") for R, n in vr.items()}
    have = all(p.exists() for p in sums.values()) and bool(sums)
    rows = []
    if E.get("final"):
        sA = _jr(sums[100])
        for q, par in (("Hzero", "H0"), ("DmuG", "dmu_G"), ("DmuChi", "dmu_chi")):
            d = abs(mac[f"HDHost{q}"]["value"]["median"] - sA["posterior"][par]["median"])
            rows.append(d)
            if d > 1e-12:
                why.append(f"HDHost{q} median vs the V8 summary of realization 100: {d:.2g}")
    if (mac["HWHost"]["class"] == "plain") != have:
        why.append(f"HWHost class {mac['HWHost']['class']} with {sum(p.exists() for p in sums.values())} of {len(sums)} V8 summaries")
    if have and mac["HWHost"]["class"] == "plain":
        own = statistics.fmean(0.5 * (_jr(p)["posterior"]["H0"]["ci90"][1] - _jr(p)["posterior"]["H0"]["ci90"][0]) for p in sums.values())
        rows.append(abs(mac["HWHost"]["value"] - own))
        if abs(mac["HWHost"]["value"] - own) > 1e-12:
            why.append(f"HWHost {mac['HWHost']['value']:.6g} vs own mean half-width {own:.6g}")
    check(not why,
          f"J4 key HDHost over stage V8's mixed-sampler runs (own parser: stage 4's pure-rslice realizations {s4}, V8's "
          f"realization_NNN_unif_then_rslice {reals} with stage 4's flags plus the sampler's and the variant only): the map's key and its "
          f"Bayes-factor reference, the classes block's HWHost, placeholder_results.json and NrealHD ('{nr['render']}') name exactly those runs; "
          f"ObsHost and HWObsHost stage 5's own {obs5}; stage 4's realization-100 run named by nothing; {ncfg} V8 configs equal stage 4's but for "
          f"the realization's files and the sampler; HDHost {'real' if E.get('final') else 'not yet real'}, HWHost {mac['HWHost']['class']} "
          f"({sum(p.exists() for p in sums.values())} of {len(sums)} summaries)" + (f"; FAIL: {'; '.join(why[:5])}" if why else ""))


# J5 (23 Nov): the keys and further macros of Sec. IV E on three steps (notes/MACRO_REQUESTS.md G1, G2), by own code: RUNLIST_experiment.yaml
# parsed here (key -> analysis, set label, stage), summaries read directly, the grids' corrected ln L read with h5py (_corrected) for the
# interior maxima and the 90% upper limit of HDAssumedTwice (own PCHIP marginal), weighted samples for the correlation, the events files'
# truth/host_type for the detected AGN shares. The combined posteriors of the new keys are J1's (own summed splines, own levels of the input).
_EXP_KEYS = {"HDAssumed": ("two_tracers_one_population", "peak_f0p30_s1", "E1"), "HDAssumedTwice": ("two_tracers_one_population", "peak_f0p30_s2", "E1"),
             "HDPops": ("two_tracers_two_populations", "peak_f0p30_s1", "E1"), "HDSingleTwice": ("single_population", "peak_f0p30_s2", "E3"),
             "ObsAssumed": ("two_tracers_one_population", "obslike_f0p30_s1", "E2"), "ObsPops": ("two_tracers_two_populations", "obslike_f0p30_s1", "E2")}
_SAME_MERGERS = (("HDAssumed", "HDPops"), ("HDAssumedTwice", "HDSingleTwice"), ("ObsAssumed", "ObsPops"))


def _own_pchip_marg(names, ax, l, par, refine=8):
    """Marginal of `par` of a 1-D or 2-D grid by PCHIP of ln L along each axis onto a lattice `refine` times denser: (x, p)."""
    from scipy.interpolate import PchipInterpolator
    ok = np.isfinite(l)
    l = np.where(ok, l, np.min(l[ok]) - 300.0)
    lat = [np.linspace(a[0], a[-1], (len(a) - 1) * refine + 1) for a in ax]
    for i, (a, an) in enumerate(zip(ax, lat)):
        l = PchipInterpolator(a, l, axis=i)(an)
    pp = np.exp(l - l.max())
    i = names.index(par)
    for j in sorted((j for j in range(len(names)) if j != i), reverse=True):
        pp = _simpson(pp, lat[j], axis=j)
    return lat[i], pp


def audit_three_steps(mac, doc, specs, comb_cache):
    import statistics
    import yaml
    rows, bad, why, out = [], [], [], []
    val = lambda n: mac[n]["value"]

    def cmp(name, got, want, tol, what):
        d = abs(got - want)
        rows.append((name, what, d, tol))
        if not d <= tol:
            bad.append(f"{name} {what}: {got:.6g} vs {want:.6g} (tol {tol:.2g})")
    fin = {k for k, E in doc.items() if E.get("final")}
    # (a) the keys' runs: RUNLIST_experiment.yaml parsed here; the map, placeholder_results.json and the stages; the same mergers of each pair
    rxl = yaml.safe_load((RUNS16 / "RUNLIST_experiment.yaml").read_text())
    own, evf = {}, {}
    for k, (an, lab, stg) in _EXP_KEYS.items():
        got = {int(r["args"][r["args"].index("--realization") + 1]): (str(st["stage"]), r["name"], r.get("events_file"))
               for st in rxl["stages"] for r in st["runs"] if r["analysis"] == an and r["set"]["label"] == lab}
        own[k], evf[k] = got, {R: v[2] for R, v in got.items()}
        sp, E = specs.get(k) or {}, doc.get(k) or {}
        pats = {re.sub(r"realization_\d{3}", "realization_{R}", v[1]) for v in got.values()}
        if pats != {sp.get("run")} or sp.get("realizations") != sorted(got) or sp.get("stage") != stg or {v[0] for v in got.values()} != {stg}:
            why.append(f"{k}: map {sp.get('run')} {sp.get('realizations')} stage {sp.get('stage')} vs own {sorted(pats)} {sorted(got)} {stg}")
        if E.get("N_comb") != len(got):
            why.append(f"{k}: N_comb {E.get('N_comb')} vs {len(got)}")
    for a, b in _SAME_MERGERS:
        if sorted(evf[a]) != sorted(evf[b]) or any(not evf[a][R] or evf[a][R] != evf[b][R] for R in evf[a]):
            why.append(f"{a} and {b} not on the same events files")
    # (b) HDAssumedTwice: only realization A's median of H0 and 90% upper limit of f_AGN (and its NoPeak)
    k = "HDAssumedTwice"
    extra_m = [n for n in mac if n.startswith(k) and n not in (f"{k}HzeroMed", f"{k}fAGNUL", f"{k}NoPeak")]
    if extra_m:
        why.append(f"{k} has macros beyond its three: {extra_m[:4]}")
    if k in fin:
        rA = RUNS16 / _key_run(specs[k], specs[k]["realization_A"])
        cmp(f"{k}HzeroMed", val(f"{k}HzeroMed"), _jr(rA / "summary.json")["posterior"]["H0"]["median"], 1e-12, "realization A's median = summary.json")
        nm_, ax_, l_ = _corrected(rA, _pass_file(rA)[0])
        qf = _quant(*_own_pchip_marg(nm_, ax_, l_, "f_agn"))
        cmp(f"{k}fAGNUL", val(f"{k}fAGNUL"), qf["ul90"], max(0.05 * 0.5 * (qf["ci90"][1] - qf["ci90"][0]), 2e-3),
            "realization A's 90th percentile of f_AGN vs own PCHIP marginal of the fine pass")
        out.append(f"{k}: H0 median {val(k + 'HzeroMed'):.3f}, f_AGN < {val(k + 'fAGNUL'):.4f} (own {qf['ul90']:.4f})")
    # (c) no interior maximum: the node with the largest corrected ln L over the fine and coarse passes (own reading) at 50 or 100
    for k in ("HDAssumed", "HDAssumedTwice", "ObsAssumed"):
        if k not in fin:
            continue
        per = {}
        for R in specs[k]["realizations"]:
            run = RUNS16 / _key_run(specs[k], R)
            best = (-np.inf, None)
            for nm in (_pass_file(run)[0], "coarse.h5"):
                names_, ax_, l_ = _corrected(run, nm)
                i = np.unravel_index(int(np.argmax(np.where(np.isfinite(l_), l_, -np.inf))), l_.shape)
                if l_[i] > best[0]:
                    best = (float(l_[i]), float(ax_[names_.index("H0")][i[names_.index("H0")]]))
            per[R] = best[1]
        n_no = sum(1 for v in per.values() if v in (50.0, 100.0))
        cmp(f"{k}NoPeak", float(val(f"{k}NoPeak")), float(n_no), 0.0, "count of realizations whose largest ln L node lies at H0 = 50 or 100 (own)")
        if mac[f"{k}NoPeak"]["render"] != _OWN_WORDS[n_no]:
            bad.append(f"{k}NoPeak render {mac[k + 'NoPeak']['render']}")
        out.append(f"{k}NoPeak {n_no} (argmax H0 by realization: {', '.join(f'{R}: {v:g}' for R, v in per.items())})")
    # (d) the shift of the H0 median of the assumed population against the two populations at their inputs, per realization (summaries)
    for ka, kp in (("HDAssumed", "HDPops"), ("ObsAssumed", "ObsPops")):
        if not {ka, kp} <= fin:
            continue
        rs = [R for R in specs[ka]["realizations"] if R in specs[kp]["realizations"]]
        sh = [_jr(RUNS16 / _key_run(specs[ka], R) / "summary.json")["posterior"]["H0"]["median"] -
              _jr(RUNS16 / _key_run(specs[kp], R) / "summary.json")["posterior"]["H0"]["median"] for R in rs]
        for suf, v in (("", statistics.median(sh)), ("Min", min(sh)), ("Max", max(sh))):
            cmp(f"{ka}ShiftHzero{suf}", val(f"{ka}ShiftHzero{suf}"), v, 1e-12, "median / smallest / largest of the summaries' paired shifts")
            if mac[f"{ka}ShiftHzero{suf}"]["render"] != _own_fixed(v, 1):
                bad.append(f"{ka}ShiftHzero{suf} render {mac[ka + 'ShiftHzero' + suf]['render']} vs {_own_fixed(v, 1)}")
        out.append(f"{ka} - {kp}: {', '.join(f'{x:+.2f}' for x in sh)}")
    # (e) the biases of the combined medians: J1's own combination (summed splines / Scott-KDE product) minus the input
    for k in ("HDAssumed", "ObsAssumed", "HDSingleTwice"):
        if k in fin and (k, "Hzero") in comb_cache:
            o = _quant(*comb_cache[(k, "Hzero")])
            cmp(f"{k}BiasHzero", val(f"{k}BiasHzero"), o["median"] - B.IN["H0"], 0.01 * 0.5 * (o["ci90"][1] - o["ci90"][0]) + 1e-9,
                "own Comb median minus the input")
        cmp(f"{k}BiasHzero", val(f"{k}BiasHzero"), val(f"{k}HzeroComb")["median"] - B.IN["H0"], 1e-12, f"= {k}HzeroComb's median minus the input")
    # (f) the per-realization Bayes factors of the host-dependent analyses: own from the summaries' ln_evidence (red: the realizations whose two
    # summaries exist, the red rule)
    for pre in ("HD", "Obs"):
        sp = specs.get(f"{pre}Host") or {}
        ref = (sp.get("LB") or {}).get("reference_run")
        per = []
        for R in sp.get("realizations", []):
            pa, pb = RUNS16 / _key_run(sp, R) / "summary.json", RUNS16 / ref.format(R=f"{R:03d}") / "summary.json"
            if pa.exists() and pb.exists():
                per.append((_jr(pa)["ln_evidence"]["value"] - _jr(pb)["ln_evidence"]["value"]) / math.log(10))
        if per:
            cmp(f"{pre}LBMin", val(f"{pre}LBMin"), min(per), 1e-9, f"smallest per-realization log10 B of the summaries ({len(per)} realizations)")
            cmp(f"{pre}LBMax", val(f"{pre}LBMax"), max(per), 1e-9, f"largest per-realization log10 B of the summaries ({len(per)} realizations)")
            out.append(f"{pre}LB per realization ({len(per)}): {', '.join(f'{x:.2f}' for x in per)}")
        if (mac[f"{pre}LBMin"]["class"] == "plain") != (f"{pre}Host" in fin):
            bad.append(f"{pre}LBMin black/red does not follow key {pre}Host")
    # (g) the largest correlation of H0 with another parameter, realization A of key HDHost: own weighted samples
    sp = specs.get("HDHost") or {}
    rA = RUNS16 / _key_run(sp, sp["realization_A"])
    if (rA / "nested_1" / "samples.npz").exists():
        pars = ["H0", "f_agn", "mu_G", "sigma_G", "mu_chi", "sigma_chi", "dmu_G", "dmu_chi"]
        s, w = _nested_w(rA, pars)
        m = np.sum(w[:, None] * s, axis=0)
        C = np.einsum("n,ni,nj->ij", w, s - m, s - m)
        r = C[0, 1:] / np.sqrt(C[0, 0] * np.diag(C)[1:])
        cmp("HDHostMaxAbsCorrHzero", val("HDHostMaxAbsCorrHzero"), float(np.max(np.abs(r))), 0.02, "largest |correlation| of H0, own weighted samples")
        out.append(f"HDHost A: |rho(H0, .)| max {float(np.max(np.abs(r))):.3f} ({pars[1 + int(np.argmax(np.abs(r)))]})")
    # (h) the AGN-hosted share of the detected mergers: own from the events files' truth/host_type, and the summaries' planted.truth_f_agn
    for nm, key, stem in (("MockHDfDetMean", "HDSingle", "marked_dmu0p10_dmuG5_corepop"), ("ObsHDfDetMean", "ObsSingle", "marked_dmu0p10_dmuG5_obslike")):
        sp = specs.get(key) or {}
        fr = []
        for R in sp.get("realizations", []):
            with h5py.File(B.DATA / f"seed{R}" / "events" / f"events{('_' + stem)}.h5", "r") as h:
                ht = np.asarray(h["truth/host_type"][()], int)
            fr.append(float(np.mean(ht == 1)))
            ps = RUNS16 / _key_run(sp, R) / "summary.json"
            if ps.exists():
                cmp(f"{nm}[{R}]", float(_jr(ps)["planted"]["truth_f_agn"]), fr[-1], 1e-9, "summary planted.truth_f_agn = events file truth/host_type")
        cmp(nm, val(nm), float(np.mean(fr)), 1e-9, f"mean over the {len(fr)} realizations of key {key} of the events files' AGN-hosted share")
        out.append(f"{nm} {np.mean(fr):.4f} over {len(fr)}")
    worst = max(rows, key=lambda r: r[2] / r[3] if r[3] > 0 else (0 if r[2] == 0 else np.inf)) if rows else ("-", "-", 0.0, 1.0)
    check(not why and not bad,
          f"J5 Sec. IV E on three steps: the keys {', '.join(_EXP_KEYS)} name RUNLIST_experiment.yaml's runs (own parser; stages E1-E3; the same "
          f"events files within each pair); {len(rows)} numbers by own code (HDAssumedTwice's median and upper limit, the interior maxima, the "
          f"shifts, the biases, the per-realization Bayes factors, the correlation of H0, the detected AGN shares); largest deviation "
          f"{worst[2]:.2g} ({worst[0]}, tolerance {worst[3]:.2g}); {'; '.join(out)}" + (f"; FAIL: {'; '.join((why + bad)[:5])}" if why or bad else ""))


# J6 (23 Nov): the observed-like keys ObsPops and ObsAssumed over two subsets of their realizations (build_values.build_obslike_subsets), by
# own code: the run lists parsed here (the two grids' realizations on obslike_f0p30_s1 in RUNLIST_experiment.yaml; stage 5's realizations in
# RUNLIST.yaml), the summaries read directly, every combination by _grid_comb_safe (own summed splines, PCHIP where a pass has non-finite
# cells) with own quantiles and levels, own formatting.
_OBS_SUB = {"ObsPops": "two_tracers_two_populations/both_catalogs/complete/obslike_f0p30_s1/realization_{R}",
            "ObsAssumed": "two_tracers_one_population/both_catalogs/complete/obslike_f0p30_s1/realization_{R}"}


def audit_obslike_subsets(mac, specs):
    import statistics
    import yaml
    rows, bad, out = [], [], []
    val = lambda n: mac[n]["value"]

    def cmp(name, got, want, tol, what):
        d = abs(got - want)
        rows.append((name, what, d, tol))
        if not d <= tol:
            bad.append(f"{name} {what}: {got:.6g} vs {want:.6g} (tol {tol:.2g})")
    names = [f"{k}{t}Hzero{p}" for k in _OBS_SUB for t in ("First", "Later") for p in ("Comb", "CL")] + ["NrealObsFirst", "NrealObsLater", "ObsAssumedFirstShiftHzero"]
    absent = [n for n in names if n not in mac]
    if absent:
        check(False, f"J6 values.json has the {len(names)} macros of the observed-like subsets ({', '.join(absent[:6])} missing)")
        return
    rxl = yaml.safe_load((RUNS16 / "RUNLIST_experiment.yaml").read_text())
    own = {k: sorted(int(m.group(1)) for st in rxl["stages"] for r in st["runs"]
                     for m in [re.fullmatch(re.escape(p).replace(re.escape("{R}"), r"(\d{3})"), r["name"])] if m) for k, p in _OBS_SUB.items()}
    rl = yaml.safe_load((RUNS16 / "RUNLIST.yaml").read_text())
    s5 = {int(m.group(1)) for st in rl["stages"] if str(st["stage"]) == "5" for r in st["runs"]
          for m in [re.fullmatch(r"single_population/both_catalogs/complete/host_dependent_obslike/realization_(\d{3})", r["name"])] if m}
    common = sorted(set(own["ObsPops"]) & set(own["ObsAssumed"]))
    sub = {"First": [r for r in common if r in s5], "Later": [r for r in common if r not in s5]}
    for nm, t in (("NrealObsFirst", "First"), ("NrealObsLater", "Later")):
        n_ = len(sub[t])
        if val(nm) != n_ or mac[nm]["render"] != _OWN_WORDS[n_] or mac[nm]["class"] != "plain":
            bad.append(f"{nm} '{mac[nm]['render']}' vs own {n_}")
    for k in _OBS_SUB:
        if (specs.get(k) or {}).get("realizations") != own[k]:
            bad.append(f"map {k} realizations {(specs.get(k) or {}).get('realizations')} vs own {own[k]}")
    states = []
    for k, pat in _OBS_SUB.items():
        for t, rs in sub.items():
            nc, nl = f"{k}{t}HzeroComb", f"{k}{t}HzeroCL"
            runs = [RUNS16 / pat.format(R=f"{r:03d}") for r in rs]
            have = bool(rs) and all((r_ / "summary.json").exists() for r_ in runs)
            states.append((nc, (mac[nc]["class"] == "plain") == have and (mac[nl]["class"] == "plain") == have))
            if (mac[nc].get("extra") or {}).get("realizations") != rs:
                bad.append(f"{nc} realizations {(mac[nc].get('extra') or {}).get('realizations')} vs own {rs}")
            if not have:
                out.append(f"{nc} red ({sum((r_ / 'summary.json').exists() for r_ in runs)} of {len(rs)} summaries)")
                continue
            x0 = {float(_jr(r_ / "summary.json")["truth"]["H0"]) for r_ in runs}
            cmp(nc, float(len(x0) != 1), 0.0, 0.0, "one input H0 in every summary's truth")
            x0 = x0.pop()
            nm_, ax_, tot_ = _grid_comb_safe(runs)
            xo, po = _marg(nm_, ax_, tot_, "H0")
            o = _quant(xo, po, x0)
            v = val(nc)
            hw = 0.5 * (v["ci90"][1] - v["ci90"][0])
            tol = max(0.01 * hw, 0.005)
            for w, a, b in (("median", v["median"], o["median"]), ("lo", v["ci90"][0], o["ci90"][0]), ("hi", v["ci90"][1], o["ci90"][1])):
                cmp(nc, a, b, tol, f"{w} vs own summed splines (_grid_comb_safe)")
            cmp(nl, val(nl), o["level"], max(0.01, 2.0 * float(np.interp(x0, xo, po)) * tol), "level of the input vs own")
            if mac[nc]["render"] != _own_interval(v):
                bad.append(f"{nc} render {mac[nc]['render']} vs {_own_interval(v)}")
            pc = _own_pct(val(nl))
            if (pc is not None and mac[nl]["render"] != pc) or (pc is None and "99" not in mac[nl]["render"]):
                bad.append(f"{nl} render {mac[nl]['render']} vs {pc}")
            out.append(f"{nc} {o['median']:.2f} [{o['ci90'][0]:.2f}, {o['ci90'][1]:.2f}], input at {100 * o['level']:.1f}% (own; {len(rs)} realizations)")
    # the paired shift over the first realizations: the summaries' medians
    rs = sub["First"]
    pa = [(RUNS16 / _OBS_SUB["ObsAssumed"].format(R=f"{r:03d}") / "summary.json", RUNS16 / _OBS_SUB["ObsPops"].format(R=f"{r:03d}") / "summary.json")
          for r in rs]
    have = bool(rs) and all(a.exists() and b.exists() for a, b in pa)
    states.append(("ObsAssumedFirstShiftHzero", (mac["ObsAssumedFirstShiftHzero"]["class"] == "plain") == have))
    if have:
        sh = [_jr(a)["posterior"]["H0"]["median"] - _jr(b)["posterior"]["H0"]["median"] for a, b in pa]
        cmp("ObsAssumedFirstShiftHzero", val("ObsAssumedFirstShiftHzero"), statistics.median(sh), 1e-12, "median of the summaries' paired shifts")
        if mac["ObsAssumedFirstShiftHzero"]["render"] != _own_fixed(statistics.median(sh), 1):
            bad.append("ObsAssumedFirstShiftHzero render")
        out.append(f"first-five shift {statistics.median(sh):+.2f}")
    bad += [f"{n} black/red does not follow its subset's summaries" for n, ok in states if not ok]
    worst = max(rows, key=lambda r: r[2] / r[3] if r[3] > 0 else (0 if r[2] == 0 else np.inf)) if rows else ("-", "-", 0.0, 1.0)
    check(not bad,
          f"J6 the observed-like keys ObsPops and ObsAssumed over two subsets (own parser: realizations {common} of both grids on "
          f"obslike_f0p30_s1, stage 5's {sorted(s5)}: First {sub['First']}, Later {sub['Later']}); black exactly when every summary of the "
          f"subset exists ({sum(ok for _n, ok in states)} of {len(states)}); {len(rows)} numbers against the own summed splines and the "
          f"summaries, largest deviation {worst[2]:.2g} ({worst[0]}, tolerance {worst[3]:.2g}); {'; '.join(out)}"
          + (f"; FAIL: {'; '.join(bad[:5])}" if bad else ""))


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
    nosum = [k for k in final for r in specs.get(k, {}).get("realizations", []) if not (RUNS16 / _key_run(specs[k], r) / "summary.json").exists()]
    notblack = [n for n, ks in owner.items() if all(k in final for k in ks) and (mac[n]["class"] != "plain" or not ("file" in mac[n]["source"] or "files" in mac[n]["source"]))]
    check(not miss and not nosum and not notblack, f"J0 {len(final)} production keys ({', '.join(sorted(final))}): each has its runs in production_map.json, every summary.json exists, "
                                                   f"every macro prints black and cites its summary ({len(miss) + len(nosum) + len(notblack)} exceptions{': ' + ', '.join((miss + nosum + notblack)[:6]) if miss or nosum or notblack else ''})")
    comb_cache = {}
    edge_rows, prior_own = [], {}
    for k in sorted(final):
        if k in B.KEY_MACROS_ONLY:           # 23 Nov: a key that prints only a few macros (HDAssumedTwice): checked in J5
            continue
        sp = specs[k]
        runs = [RUNS16 / _key_run(sp, r) for r in sp["realizations"]]
        rA = RUNS16 / _key_run(sp, sp["realization_A"])
        sA = _jr(rA / "summary.json")
        grid = sp["mode"] == "grid"
        if grid:
            names, axes, tot = _grid_comb(runs)
            nA, aA, tA = _grid_comb([rA])
        for q in sp["quantities"]:
            if q == "RateRatio":
                continue
            par = QPAR[q]
            x0 = doc[k]["quantities"][q]["input"]
            conv = k.startswith("Free") and q in ("LognGal", "LognAGN")
            # a number density the data do not bound on both sides has no combined interval, width or credible level (checked below)
            lim_only = bool(doc[k]["quantities"][q]["Comb"].get("limit_only"))
            vA = val(f"{k}{q}")
            hwA = 0.5 * (vA["ci90"][1] - vA["ci90"][0])
            if not lim_only:
                vC = val(f"{k}{q}Comb")
                hwC = 0.5 * (vC["ci90"][1] - vC["ci90"][0])
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
                    sets, raw = [], []
                    for r in runs:
                        s, w = _nested_w(r, [par, "H0"])
                        sets.append((s[:, 0] + 3 * np.log10(s[:, 1] / 100.0), w))
                        raw.append((s[:, 0], s[:, 1], w))
                    pri_h = {tuple(_jr(r / "config.json")["priors"][par]) for r in runs}
                    assert len(pri_h) == 1, (k, q, pri_h)
                    pri_h = pri_h.pop()
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
            if conv:
                # the support: own edge ratios (weighted Scott kernel, reflected at the prior edges of the sampled parameter); a side is
                # bounded when the ratio is below 1% in every realization; limit only exactly when a side is not bounded
                er = _edge_ratios(raw, pri_h)
                own_lim = not (max(er["lower"]) < 0.01 and max(er["upper"]) < 0.01)
                edge_rows.append(f"{k}{q} lower {max(er['lower']):.3f} upper {max(er['upper']):.3f} -> {'limit only' if own_lim else 'interval'}")
                cmp(f"{k}{q}Comb", float(lim_only), float(own_lim), 0.0, "limit only (no combined interval) by the own edge ratios")
                # the limits: realization A in Mpc^-3 (converted samples) and in h^3 Mpc^-3 (the sampled parameter), own weighted samples
                qa_ = _wq(sa, wa, [0.05, 0.90, 0.95])
                cmp(f"{k}{q}UL", val(f"{k}{q}UL"), float(qa_[1]), max(0.05 * 0.5 * (qa_[2] - qa_[0]), 5e-4), "A 90% upper limit in Mpc^-3 vs own weighted samples")
            if lim_only:
                gone = [n for n in (f"{k}{q}Comb", f"{k}{q}CombMed", f"{k}{q}CombLo", f"{k}{q}CombHi", f"{k}{q}CombWid", f"{k}{q}CL") if n in mac]
                cmp(f"{k}{q}Comb", float(len(gone)), 0.0, 0.0, f"no combined interval, width or credible level{': ' + ', '.join(gone) if gone else ''}")
                xb, pb = _kde_prod_boxed(raw, pri_h)
                ob = _quant(xb, pb)
                cmp(f"{k}{q}CombUL", val(f"{k}{q}CombUL"), ob["ul90"], max(0.08 * 0.5 * (ob["ci90"][1] - ob["ci90"][0]), 2e-3),
                    "Comb 90% upper limit vs own Scott-KDE product respecting each sample's support")
            else:
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
            if conv and sp.get("prior_induced_from") and f"{k}{q}PriorULh" in mac:
                # the limits the priors allow once the rate per host is measured: own draws, counts, closed form and splines
                if k not in prior_own:
                    prior_own[k] = _prior_limits_own(sp)
                po = prior_own[k]
                ne_p = (doc[k].get("joint", {}).get("prior_induced") or {}).get("n_eff") or 0.0
                for nm_, (qv, s1) in zip(("PriorULh", "PriorUL"), po[q[4:]]):
                    # tolerance: four times the combined Monte Carlo error of the two weighted quantiles (own draws and the build's, each
                    # sqrt(0.09 / n_eff) over the density at the quantile)
                    tol_ = 4.0 * s1 * math.sqrt(1.0 / po["n_eff"] + (1.0 / ne_p if ne_p > 0 else 1.0))
                    cmp(f"{k}{q}{nm_}", val(f"{k}{q}{nm_}"), qv, tol_, f"A prior-induced 90% upper limit ({'h^3 Mpc^-3' if nm_.endswith('h') else 'Mpc^-3'}) vs own "
                                                                         f"4e6 prior draws (n_eff {po['n_eff']:.0f}; the build's {ne_p:.0f})")
            if x0 is not None and not lim_only and isinstance(val(f"{k}{q}CL"), (int, float)):
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
                za = _jr(RUNS16 / _key_run(sp, r) / "summary.json")["ln_evidence"]["value"]
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
    widA = lambda k: (lambda p: p["ci90"][1] - p["ci90"][0])(_jr(RUNS16 / _key_run(specs[k], specs[k]["realization_A"]) / "summary.json")["posterior"]["H0"])
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
                   f"({worst[0]}, {worst[1]}, tolerance {worst[3]:.2g}); number densities, largest posterior density at a prior edge over its maximum "
                   f"(own weighted kernel; a limit only unless both < 0.01): {'; '.join(edge_rows) or 'none final'}"
                   f"{'; MISMATCH: ' + '; '.join(bad[:6]) if bad else ''}")
    audit_knownboth_twenty(mac, doc, specs)
    audit_hdhost_mixed(mac, doc, specs)
    audit_three_steps(mac, doc, specs, comb_cache)
    audit_obslike_subsets(mac, specs)
    # ---- J3 (21 Nov): HDNoneHzeroCombUL, the 90% upper limit of the combined H0 posterior with no catalog on host-dependent mergers (it
    # leans on the lower prior edge): own Scott-KDE product of the weighted samples (J1's), its 90th percentile, mode and density at the edge
    if "HDNone" in fin and ("HDNone", "Hzero") in comb_cache and "HDNoneHzeroCombUL" in mac:
        xh, ph_ = comb_cache[("HDNone", "Hzero")]
        oh = _quant(xh, ph_)
        ul = mac["HDNoneHzeroCombUL"]
        rc = mac["HDNoneHzeroComb"]["render"]
        d_ = len(rc.split("^")[0].split(".")[1]) if "." in rc.split("^")[0] else 0
        hwh = 0.5 * (mac["HDNoneHzeroComb"]["value"]["ci90"][1] - mac["HDNoneHzeroComb"]["value"]["ci90"][0])
        okr = ul["render"] == f"{ul['value']:.{d_}f}" and ul["class"] == "plain"
        mode_, edge_ = float(xh[np.argmax(ph_)]), float(ph_[0] / ph_.max())
        check(okr and abs(ul["value"] - oh["ul90"]) <= 0.08 * hwh and xh[0] == 50.0,
              f"J3 HDNoneHzeroCombUL = '{ul['render']}' ({ul['value']:.3f}): the 90th percentile of the combined H0 posterior (single population, no "
              f"catalog, host-dependent mergers, {len(specs['HDNone']['realizations'])} realizations), printed at the decimals of HDNoneHzeroComb "
              f"({d_}); own Scott-KDE product (reflected at 50 and 100): 90th percentile {oh['ul90']:.3f} (tolerance {0.08 * hwh:.2f}), mode "
              f"{mode_:.2f}, density at the lower prior edge (50) over the peak {edge_:.3f}")


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
    # 23 Nov: the experiment's macros are on the experiment's realizations (the header's 'realizations'); a set's further realizations
    # (make_runlist.EXPERIMENT_EXTRA) feed the paper's keys only
    xr_ = {int(x) for x in (rx.get("realizations") or [])}
    for st in rx["stages"]:
        for r in st["runs"]:
            pat_, rr_ = re.sub(r"realization_\d{3}", "realization_{R}", r["name"]), int(r["args"][r["args"].index("--realization") + 1])
            if xr_ and rr_ not in xr_:
                continue
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
        if rec.get("realizations_of"):
            # 23 Nov (HW<pre>Single): the realizations of another macro's record, all of which the run list holds for this run
            ref_ = blk.get(rec["realizations_of"]) or {}
            if rec["realizations"] != ref_.get("realizations") or not set(rec["realizations"]) <= set(exp_idx.get(rec["run"], ())):
                notlisted.append(name)
        elif sorted(exp_idx.get(rec["run"], ())) != sorted(rec["realizations"]):
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
        # the paired differences of 21 Nov (fn paired_...) are checked in K3
        if rec.get("how") != "derived" or not have.get(name) or rec.get("fn") not in ("share", "cost"):
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
        # each record's own realizations (23 Nov: HW<pre>Single holds those of HWHost / HWObsHost only)
        ea = {r: v for r, v in ev_idx.get(blk[a_]["run"], {}).items() if r in blk[a_]["realizations"]}
        eb = {r: v for r, v in ev_idx.get(blk[b_]["run"], {}).items() if r in blk[b_]["realizations"]}
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
                used[RUNS16 / _key_run(sp, r)] = (f"key {k_}", None)
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
# S: the sky resolution (scripts/skyres.py; map block 'sky_resolution'), by own code: the run list parsed here, the two
# summaries and configs read directly, the survey files' nside and object counts read with h5py.
# ============================================================================================= #
def audit_skyres(mac):
    import yaml
    names = ("SkyResShiftHzero", "SkyResShiftfAGN")
    if not all(n in mac for n in names):
        check(False, "S0 values.json has SkyResShiftHzero and SkyResShiftfAGN")
        return
    rl = yaml.safe_load((RUNS16 / "RUNLIST.yaml").read_text())
    idx = {r["name"]: (str(st["stage"]), r["args"]) for st in rl["stages"] for r in st["runs"]}
    run, partner = (f"known_population/both_catalogs/complete/equal_populations/realization_100{s}" for s in ("_nside64", ""))
    ok_list = run in idx and partner in idx and idx[run][0].startswith("X") and idx[partner][0] == "1"
    if ok_list:
        a64, a32 = list(idx[run][1]), list(idx[partner][1])
        i = a64.index("--nside")
        ok_list = a64[i + 1] == "64" and a64[a64.index("--variant") + 1] == "nside64"
        rest64 = [x for j, x in enumerate(a64) if j not in (i, i + 1, a64.index("--variant"), a64.index("--variant") + 1)]
        rest32 = [x for x in a32 if x != "--membership"]
        ok_list = ok_list and rest64 == rest32
    pm = _jr(OUT / "values" / "production_map.json")
    blk = (pm.get("sky_resolution") or {}).get("pair") or {}
    ok_map = blk.get("run", "").format(R="100") == run and blk.get("partner", "").format(R="100") == partner
    have = all((RUNS16 / n / "summary.json").exists() for n in (run, partner))
    black = [mac[n]["class"] == "plain" for n in names]
    check(ok_list and ok_map and all(b == have for b in black),
          f"S0 the sky-resolution pair is RUNLIST.yaml's (own parser): {run} in a check stage (X) with --nside 64 and otherwise stage 1's "
          f"arguments of {partner} (without --membership); the map names the same pair; black exactly when both summaries exist ({have})")
    if not have:
        return
    s64, s32 = (_jr(RUNS16 / n / "summary.json") for n in (run, partner))
    c64, c32 = (_jr(RUNS16 / n / "config.json") for n in (run, partner))
    # the injection set each likelihood used: the newer driver records the list (injections_files, with
    # n_injection_sets and the pooled lanes); the older one only its file (injections_file), which prod_model takes
    # when the list is absent; with neither, the driver's default for the realization and family. The three keys
    # only the newer driver writes are compared through this set, not one by one.
    inj_keys = ("injections_files", "n_injection_sets", "injection_lanes_pooled")

    def inj_set(c):
        if c.get("injections_files"):
            files = list(c["injections_files"])
        elif c.get("injections_file"):
            files = [c["injections_file"]]
        else:
            import importlib.util
            spec = importlib.util.spec_from_file_location("_pc16", RUNS16.parent / "scripts" / "prod_common.py")
            pc = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(pc)
            files = [str(pc.injections_path(c["realization"], c.get("population_family", "corepop"), c.get("injections_lane")))]
        ok = (c.get("n_injection_sets") in (None, len(files))
              and (c.get("injection_lanes_pooled") is None or len(c["injection_lanes_pooled"]) == len(files) - 1))
        return [str(Path(f).resolve()) for f in files], ok
    (i64, ok64), (i32, ok32) = inj_set(c64), inj_set(c32)
    ok_inj = ok64 and ok32 and i64 == i32
    # membership (--membership) writes each merger's AGN-host probability after the grid is summarized
    # (prod_driver.run_grid: membership_safe(m, cfg, summ, gdir) from the finished summary); the model that
    # evaluates the grid (prod_model.py) never reads the key, so it may differ between the two runs
    drv = (RUNS16.parent / "scripts" / "prod_driver.py").read_text()
    mod = (RUNS16.parent / "scripts" / "prod_model.py").read_text()
    calls = re.findall(r"(?<!def )membership_safe\(m, cfg, ([^,]+), gdir\)", drv)
    ok_mem = "membership" not in mod and calls == ['rec["summary"]', "summ"] and drv.index("summ = grid_summary(") < drv.rindex("membership_safe(m, cfg, summ")
    diff = sorted(k for k in set(c64) | set(c32) if c64.get(k) != c32.get(k) and k not in inj_keys)
    want_diff = ["membership", "run_name", "survey_nside", "surveys", "variant"]
    nside, nobj = {}, {}
    for tag, c in (("64", c64), ("32", c32)):
        for t, path in c["surveys"].items():
            with h5py.File(path, "r") as h:
                nside[(tag, t)] = int(h.attrs["nside"])
                nobj[(tag, t)] = int(np.asarray(h["ngals"]).sum())
    ok_files = all(nside[("64", t)] == 64 and nside[("32", t)] == 32 and nobj[("64", t)] == nobj[("32", t)] for t in ("gal", "agn"))
    dev = []
    for n, par in zip(names, ("H0", "f_agn")):
        want = s64["posterior"][par]["median"] - s32["posterior"][par]["median"]
        dev.append(abs(mac[n]["value"] - want))
    # the Monte Carlo quality at the peak: the macros equal the summaries' variance block, and that block equals an own reading of
    # each grid's final fine pass (h5py: argmax of the corrected ln L, the posterior-sample variance and N^2/N_eff there)
    dvar, vrows = _skyres_var_dev(mac, "", {64: (run, s64), 32: (partner, s32)})
    dev += dvar
    check(set(diff) <= set(want_diff) and ok_files and ok_inj and ok_mem and max(dev) < 1e-9,
          f"S1 the two runs' configs differ only in {diff} (allowed {want_diff}; membership output written after the grid, never read by "
          f"prod_model: {ok_mem}); the same injection set {[Path(p).name for p in i64]} (nside 64) and {[Path(p).name for p in i32]} "
          f"(nside 32: {'its injections_files' if c32.get('injections_files') else 'its injections_file, no list recorded'}), set counts consistent "
          f"{ok64 and ok32}; survey files at nside 64 and 32 with the same objects "
          f"({ {t: nobj[('64', t)] for t in ('gal', 'agn')} }); SkyResShift<Hzero|fAGN> = difference of the summaries' medians, "
          f"SkyResVar<PE|Sel><SixtyFour|ThirtyTwo|Ratio> = the summaries' variance at the peak and the own reading of each fine grid "
          f"({vrows}), to {max(dev):.1g}")
    audit_skyres_mc(mac, idx, s64, s32, c32)


def _skyres_var_dev(mac, P, runs):
    """|macro - summary| and |summary - own grid reading| of the SkyRes<P>Var... macros; runs {nside: (run name, summary)}."""
    dev, own = [], {}
    for ns, (name, sm) in runs.items():
        run = RUNS16 / name
        res, cfg = _jr(run / "grid" / "result.json"), _jr(run / "config.json")
        fn = "fine_widened.h5" if (res.get("outside_mass_check") or {}).get("widened") else "fine.h5"
        with h5py.File(run / "grid" / fn, "r") as h:
            l, pv, ne = (np.asarray(h[k][()], float) for k in ("lnL", "pe_var", "n_eff"))
            n = int(h.attrs["n_events"])
        lc = l + 0.5 * pv - 0.5 * n / ne if cfg["half_variance_correction"] else l
        i = np.unravel_index(int(np.argmax(np.where(np.isfinite(l), lc, -np.inf))), lc.shape)
        own[ns] = {"PE": float(pv[i]), "Sel": float(n ** 2 / ne[i])}
        for v, f in (("PE", "pe_var_at_peak"), ("Sel", "sigma2_sel_at_peak")):
            dev.append(abs(sm["variance"][f] - own[ns][v]) / own[ns][v])
            dev.append(abs(mac[f"SkyRes{P}Var{v}{ {64: 'SixtyFour', 32: 'ThirtyTwo'}[ns] }"]["value"] - sm["variance"][f]))
    for v, f in (("PE", "pe_var_at_peak"), ("Sel", "sigma2_sel_at_peak")):
        dev.append(abs(mac[f"SkyRes{P}Var{v}Ratio"]["value"] - runs[64][1]["variance"][f] / runs[32][1]["variance"][f]))
    rows = "; ".join(f"{v} {own[64][v]:.3g}/{own[32][v]:.3g} = {own[64][v] / own[32][v]:.3g}" for v in ("PE", "Sel"))
    return dev, rows


def audit_skyres_mc(mac, idx, s64_8k, s32_8k, c32_8k):
    """S2/S3: the sky-resolution pair at Analysis 14's Monte Carlo settings (skyres 'mc_pair'), by own code."""
    P = "MC"
    names = [f"SkyRes{P}Shift{q}" for q in ("Hzero", "fAGN")] + [f"SkyRes{P}Var{v}{w}" for v in ("PE", "Sel") for w in ("SixtyFour", "ThirtyTwo", "Ratio")]
    if not all(n in mac for n in names):
        check(False, f"S2 values.json has the {len(names)} SkyResMC... macros ({', '.join(n for n in names if n not in mac)} missing)")
        return
    base = "known_population/both_catalogs/complete/equal_populations/realization_100"
    run, partner = base + "_nside64_s32000_inj6", base + "_s32000_inj6"
    ok_list, why = run in idx and partner in idx, []
    if ok_list:
        (st64, a64), (st32, a32), (_, a1) = idx[run], idx[partner], idx[base]
        sets = json.loads((B.W / "data" / "experiment_sets.json").read_text())["extra_record_lane_sets"]["100"]["files"]
        stage1 = [x for x in a1 if x != "--membership"]
        want32 = stage1 + ["--nsamp", "32000", "--variant", "s32000_inj6"] + [x for f in sets for x in ("--pool-injections", f)]
        want64 = stage1 + ["--nside", "64", "--nsamp", "32000", "--variant", "nside64_s32000_inj6"] + [x for f in sets for x in ("--pool-injections", f)]
        ok_list = st64.startswith("X") and st32.startswith("X") and list(a32) == want32 and list(a64) == want64
        if not ok_list:
            why.append(f"stages {st64}/{st32}; args 32: {a32 == want32}, 64: {a64 == want64}")
    pm = _jr(OUT / "values" / "production_map.json")
    blk = (pm.get("sky_resolution") or {}).get("mc_pair") or {}
    ok_map = blk.get("run", "").format(R="100") == run and blk.get("partner", "").format(R="100") == partner
    have = all((RUNS16 / n / "summary.json").exists() for n in (run, partner))
    black = [mac[n]["class"] == "plain" for n in names]
    # while red: the shifts of the 8000-sample pair, its variances scaled by the Monte Carlo settings, its ratios
    red_dev = 0.0
    if not have:
        for q, par in (("Hzero", "H0"), ("fAGN", "f_agn")):
            red_dev = max(red_dev, abs(mac[f"SkyRes{P}Shift{q}"]["value"] - (s64_8k["posterior"][par]["median"] - s32_8k["posterior"][par]["median"])))
        for v, f, sc in (("PE", "pe_var_at_peak", 8000 / 32000), ("Sel", "sigma2_sel_at_peak", 1 / 6)):
            red_dev = max(red_dev, abs(mac[f"SkyRes{P}Var{v}SixtyFour"]["value"] - sc * s64_8k["variance"][f]),
                          abs(mac[f"SkyRes{P}Var{v}ThirtyTwo"]["value"] - sc * s32_8k["variance"][f]),
                          abs(mac[f"SkyRes{P}Var{v}Ratio"]["value"] - s64_8k["variance"][f] / s32_8k["variance"][f]))
    check(ok_list and ok_map and all(b == have for b in black) and red_dev < 1e-9,
          f"S2 the sky-resolution pair at 32000 samples and six injection sets is RUNLIST.yaml's (own parser): {partner} and {run} in a check stage "
          f"(X) with stage 1's arguments of {base} (without --membership), --nsamp 32000, the five further sets of data/experiment_sets.json "
          f"(extra_record_lane_sets 100) pooled, the nside-64 one with --nside 64; the map names the pair; black exactly when both summaries "
          f"exist ({have}){'' if have else f'; red values = the 8000-sample pair scaled (shifts as they are, posterior-sample variance x 1/4, N^2/N_eff x 1/6; to {red_dev:.1g})'}"
          f"{'; ' + '; '.join(why) if why else ''}")
    if not have:
        return
    s64, s32 = (_jr(RUNS16 / n / "summary.json") for n in (run, partner))
    c64, c32 = (_jr(RUNS16 / n / "config.json") for n in (run, partner))
    diff = sorted(k for k in set(c64) | set(c32) if c64.get(k) != c32.get(k))
    want_diff = ["run_name", "survey_nside", "surveys", "variant"]
    inj = [str(Path(f).resolve()) for f in c32["injections_files"]]
    ok_inj = (c32["nsamp"] == 32000 and len(set(inj)) == 6 and inj[0] == str(Path(c32_8k["injections_file"]).resolve())
              and c32["redraw_key"] == c32_8k["redraw_key"] and c32["events_file"] == c32_8k["events_file"])
    nside = {}
    for tag, c in (("64", c64), ("32", c32)):
        for t, path in c["surveys"].items():
            with h5py.File(path, "r") as h:
                nside[(tag, t)] = int(h.attrs["nside"])
    ok_files = all(nside[("64", t)] == 64 and nside[("32", t)] == 32 for t in ("gal", "agn"))
    dev = [abs(mac[f"SkyRes{P}Shift{q}"]["value"] - (s64["posterior"][par]["median"] - s32["posterior"][par]["median"])) for q, par in (("Hzero", "H0"), ("fAGN", "f_agn"))]
    dvar, vrows = _skyres_var_dev(mac, P, {64: (run, s64), 32: (partner, s32)})
    check(set(diff) <= set(want_diff) and ok_inj and ok_files and max(dev + dvar) < 1e-9,
          f"S3 the two runs' configs differ only in {diff} (allowed {want_diff}); 32000 samples, six distinct injection sets (the first the record "
          f"lane of the 8000-sample pair), the same redraw and events; survey files at nside 64 and 32; SkyResMC... = the summaries "
          f"(shifts of the medians; variance at the peak, own reading of the fine grids {vrows}) to {max(dev + dvar):.1g}")


# S4-S8: the sky resolution over the realizations (19 Nov; skyres SkyRes<P>Mean..., SkyRes<P>Sd..., SkyResNreal, SkyResMCNsets), by own
# code: the run list parsed here, the realizations with six injection sets from data/experiment_sets.json and the files on disk, the
# summaries read directly, mean and standard deviation by the statistics module, numbers formatted and written as words by own code.
_OWN_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
              "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty")


def _own_fixed(x, d):
    """x to d decimals, '-0' as '0', a negative value wrapped in \\ensuremath (as the shift macros print)."""
    s = f"{x:.{d}f}"
    if s.startswith("-") and float(s) == 0.0:
        s = s[1:]
    return "\\ensuremath{" + s + "}" if s.startswith("-") else s


def _own_sig2(x):
    """x to two significant figures (values between 0.01 and 99)."""
    e = math.floor(math.log10(abs(x)))
    assert -2 <= e <= 1, x
    return f"{x:.{1 - e}f}"


def audit_skyres_all(mac):
    import statistics
    import yaml
    new = ["SkyResNreal", "SkyResMCNsets"] + [f"SkyRes{P}{m}" for P in ("", "MC") for m in
                                               ("MeanShiftHzero", "SdShiftHzero", "MeanShiftfAGN", "SdShiftfAGN", "MeanVarPERatio")]
    if not all(n in mac for n in new):
        check(False, f"S4 values.json has the {len(new)} sky-resolution macros over the realizations ({', '.join(n for n in new if n not in mac)} missing)")
        return
    rl = yaml.safe_load((RUNS16 / "RUNLIST.yaml").read_text())
    idx = {r["name"]: (str(st["stage"]), list(r["args"])) for st in rl["stages"] for r in st["runs"]}
    base = "known_population/both_catalogs/complete/equal_populations/realization_{R}"
    # the realizations with six injection sets: the record lane and the five further sets of experiment_sets.json, all on disk
    ex = json.loads((B.W / "data" / "experiment_sets.json").read_text())["extra_record_lane_sets"]
    six = sorted(int(R) for R, v in ex.items() if len(set(v["files"])) == 5
                 and all((B.W / "data" / f"seed{R}" / f).exists() for f in list(v["files"]) + ["injections/injections_targeted.h5"]))
    # the realizations with an nside-64 grid of the sky resolution at 32000 samples and six sets in a check stage (X); 21 Nov: stage X
    # also holds nside-64 grids at 8000 samples on realizations with one set (106-110), which no macro reads
    pat = re.compile(re.escape(base.replace("{R}", "@")).replace("@", r"(\d{3})") + "_nside64_s32000_inj6")
    xr = sorted(int(m.group(1)) for n, (s, _) in idx.items() if s.startswith("X") for m in [pat.fullmatch(n)] if m)
    bad = []
    for R in six:
        b = base.format(R=f"{R:03d}")
        if b not in idx or idx[b][0] != "1":
            bad.append(f"{b}: no stage-1 run")
            continue
        stage1 = [x for x in idx[b][1] if x != "--membership"]
        pool = [x for f in ex[str(R)]["files"] for x in ("--pool-injections", f)]
        want = {b + "_nside64": stage1 + ["--nside", "64", "--variant", "nside64"],
                b + "_s32000_inj6": stage1 + ["--nsamp", "32000", "--variant", "s32000_inj6"] + pool,
                b + "_nside64_s32000_inj6": stage1 + ["--nside", "64", "--nsamp", "32000", "--variant", "nside64_s32000_inj6"] + pool}
        for n, a in want.items():
            if n not in idx:
                bad.append(f"{n}: not in the run list")
            elif not idx[n][0].startswith("X") or idx[n][1] != a:
                bad.append(f"{n}: stage {idx[n][0]}, arguments {'as expected' if idx[n][1] == a else 'differ'}")
    pm = _jr(OUT / "values" / "production_map.json")
    sk = pm.get("sky_resolution") or {}
    maps = {w: (sk.get(w) or {}).get("realizations") for w in ("pair", "mc_pair")}
    check(six == xr and len(six) >= 2 and not bad and all(v == six for v in maps.values()),
          f"S4 the sky-resolution grids over the realizations are RUNLIST.yaml's (own parser): the realizations with six injection sets "
          f"(experiment_sets.json extra_record_lane_sets, files on disk) {six} = those with an nside-64 grid at 32000 samples and six sets in "
          f"stage X {xr}; each has its "
          f"three stage-X grids with stage 1's arguments of its KnownBoth run (without --membership) plus --nside 64 / --nsamp 32000 and its "
          f"five further sets pooled; the map lists the same realizations for both pairs ({maps})" + (f"; {'; '.join(bad[:4])}" if bad else ""))
    # S5: the two counts, words of the run list
    nr = mac["SkyResNreal"]
    mc_runs = [base.format(R=f"{R:03d}") + s for R in six for s in ("_s32000_inj6", "_nside64_s32000_inj6") if base.format(R=f"{R:03d}") + s in idx]
    nsets = sorted({1 + idx[n][1].count("--pool-injections") for n in mc_runs})
    used = {}
    for n in mc_runs:
        cfg = RUNS16 / n / "config.json"
        if cfg.exists():
            used[n] = len({str(Path(f).resolve()) for f in _jr(cfg).get("injections_files") or []})
    ns = mac["SkyResMCNsets"]
    ok_n = nr["class"] == "plain" and nr["value"] == len(six) and nr["render"] == _OWN_WORDS[len(six)]
    ok_s = (len(nsets) == 1 and ns["class"] == "plain" and ns["value"] == nsets[0] and ns["render"] == _OWN_WORDS[nsets[0]]
            and all(v == nsets[0] for v in used.values()))
    check(ok_n and ok_s,
          f"S5 SkyResNreal = '{nr['render']}' ({nr['class']}) = the {len(six)} realizations; SkyResMCNsets = '{ns['render']}' ({ns['class']}) = 1 + the "
          f"--pool-injections of each of the {len(mc_runs)} high-accuracy runs in the run list ({nsets}), and the distinct injection files of "
          f"the {len(used)} of them that ran (config.json injections_files: {sorted(set(used.values()))})")
    # S6 (8000 samples), S7 (32000 samples, six sets): means and standard deviations over the realizations, or the red rule
    for tag, P, (suf64, suf32) in (("S6", "", ("_nside64", "")), ("S7", "MC", ("_nside64_s32000_inj6", "_s32000_inj6"))):
        sums = {}
        for R in six:
            b = base.format(R=f"{R:03d}")
            ss = [_jr(p) if p.exists() else {} for p in (RUNS16 / (b + suf64) / "summary.json", RUNS16 / (b + suf32) / "summary.json")]
            ok = all("posterior" in s and "pe_var_at_peak" in (s.get("variance") or {}) for s in ss)
            sums[R] = ss if ok else None
        have = all(sums.values())
        names = [f"SkyRes{P}{m}" for m in ("MeanShiftHzero", "SdShiftHzero", "MeanShiftfAGN", "SdShiftfAGN", "MeanVarPERatio")]
        black = [mac[n]["class"] == "plain" for n in names]
        dev, rend, rows = [], [], []
        if have:
            for q, par, d in (("Hzero", "H0", 2), ("fAGN", "f_agn", 3)):
                xs = [sums[R][0]["posterior"][par]["median"] - sums[R][1]["posterior"][par]["median"] for R in six]
                m_, s_ = statistics.fmean(xs), statistics.stdev(xs)
                dev += [abs(mac[f"SkyRes{P}MeanShift{q}"]["value"] - m_), abs(mac[f"SkyRes{P}SdShift{q}"]["value"] - s_)]
                rend += [mac[f"SkyRes{P}MeanShift{q}"]["render"] == _own_fixed(m_, d), mac[f"SkyRes{P}SdShift{q}"]["render"] == _own_fixed(s_, d)]
                rows.append(f"{q} {m_:+.4f} sd {s_:.4f}")
            rs = [sums[R][0]["variance"]["pe_var_at_peak"] / sums[R][1]["variance"]["pe_var_at_peak"] for R in six]
            dev.append(abs(mac[f"SkyRes{P}MeanVarPERatio"]["value"] - statistics.fmean(rs)))
            rend.append(mac[f"SkyRes{P}MeanVarPERatio"]["render"] == _own_sig2(statistics.fmean(rs)))
            rows.append(f"PE ratio {statistics.fmean(rs):.4f}")
        else:
            # red: the means are realization A's own values of the setting (from its summaries when its pair has finished, else the red
            # value of its own macro), the standard deviations the invented 0.3 (H0) and 0.010 (f_AGN)
            sA = sums.get(100)
            for q, par, d, sd in (("Hzero", "H0", 2, 0.3), ("fAGN", "f_agn", 3, 0.010)):
                a = (sA[0]["posterior"][par]["median"] - sA[1]["posterior"][par]["median"]) if sA else mac[f"SkyRes{P}Shift{q}"]["value"]
                dev += [abs(mac[f"SkyRes{P}MeanShift{q}"]["value"] - a), abs(mac[f"SkyRes{P}SdShift{q}"]["value"] - sd)]
                rend += [mac[f"SkyRes{P}MeanShift{q}"]["render"] == "\\ph{" + _own_fixed(a, d) + "}",
                         mac[f"SkyRes{P}SdShift{q}"]["render"] == "\\ph{" + _own_fixed(sd, d) + "}"]
                rows.append(f"{q} mean {a:+.4f} (realization A{'' if sA else ', red'}), sd {sd} (invented)")
            ra = (sA[0]["variance"]["pe_var_at_peak"] / sA[1]["variance"]["pe_var_at_peak"]) if sA else mac[f"SkyRes{P}VarPERatio"]["value"]
            dev.append(abs(mac[f"SkyRes{P}MeanVarPERatio"]["value"] - ra))
            rend.append(mac[f"SkyRes{P}MeanVarPERatio"]["render"] == "\\ph{" + _own_sig2(ra) + "}")
            rows.append(f"PE ratio {ra:.4f} (realization A)")
        n_have = sum(1 for v in sums.values() if v)
        check(all(b == have for b in black) and max(dev) < 1e-9 and all(rend),
              f"{tag} SkyRes{P}Mean.../Sd... over the realizations: black exactly when all {len(six)} pairs have both summaries with posterior and "
              f"variance blocks ({n_have} of {len(six)}: {'black' if have else 'red'}); {'; '.join(rows)}; values to {max(dev):.1g}, renders "
              f"(decimals, sign, \\ph) as own formatting: {sum(rend)} of {len(rend)}")
    # S8: stage X feeds no paper key: no run the map names for a key, an experiment macro or a density arm is a stage-X run
    xnames = {n for n, (s, _) in idx.items() if s.startswith("X")}
    named = set()
    for E in (pm.get("keys") or {}).values():
        if isinstance(E, dict) and "run" in E:
            named |= {_key_run(E, R) for R in E.get("realizations", [])}
    # the one deliberate exception (build_values.STAGE_X_KEY_RUNS): KnownBoth's stage-X realization_NNN_nside32 runs, as the own parser finds them
    exc8 = set(_own_kb_stage_x({n: (s, a) for n, (s, a) in idx.items()})[1].values())
    named_exc = named & exc8
    named -= exc8
    for E in ((pm.get("classes") or {}).get("macros") or {}).values():
        if isinstance(E, dict) and "run" in E:
            named |= {E["run"].format(R=f"{R:03d}") for R in E.get("realizations", [])}
    for E in ((pm.get("density_response") or {}).get("arms") or {}).values():
        if isinstance(E, dict):
            named |= {E[k].format(R=f"{R:03d}") for k in ("run", "unit_run") if k in E for R in E.get("realizations", [])}
    feed = sorted(named & xnames)
    # 23 Nov: stage V likewise, but for the second deliberate exception (build_values.STAGE_V_KEY_RUNS): key HDHost's (and so HWHost's) stage-V8
    # runs realization_NNN_unif_then_rslice, as the own parser finds them
    vnames = {n for n, (s, _) in idx.items() if s.startswith("V")}
    excv = set(_own_hd_stage_v({n: (s, a) for n, (s, a) in idx.items()})[1].values())
    feed_v = sorted((named & vnames) - excv)
    hd_named = {_key_run(pm["keys"]["HDHost"], R) for R in pm["keys"]["HDHost"].get("realizations", [])} if "HDHost" in (pm.get("keys") or {}) else set()
    check(not feed and not feed_v and len(named) > 100 and named_exc == exc8 and hd_named == excv,
          f"S8 no stage-X run ({len(xnames)} in RUNLIST.yaml) and no stage-V run ({len(vnames)}) is among the {len(named)} runs that "
          f"production_map.json names for the keys, the experiment's macros and the density arms (stage X reaches the paper only through "
          f"sky_resolution and density_response.units_physical), but for the two deliberate exceptions: key KnownBoth's {len(named_exc)} stage-X "
          f"runs realization_NNN_nside32 (the same analysis at the same settings, own parser: {len(exc8)}) and key HDHost's {len(hd_named)} "
          f"stage-V8 runs realization_NNN_unif_then_rslice (the mixed sampler, own parser: {len(excv)})"
          + (f"; FEEDS: {(feed + feed_v)[:3]}" if feed or feed_v else ""))


# S9-S11 (21 Nov): the sky-resolution pairs combined over their realizations (skyres SETS: mc_pair, agn_pair; macros
# SkyResMCHzero<Comb|CL><ThirtyTwo|SixtyFour>, SkyResAGN...; 23 Nov: SkyResMC<Min|Max>ShiftHzero, SkyResMCNarrowPct and pair_all's
# SkyResTen..., SkyResOffsetHzero... removed with S10), by own code: the run list parsed here (realizations, stages, arguments), configs and survey files read directly
# (h5py: the nside of each survey file), the summaries read directly, statistics by the statistics module, every combination by
# _grid_comb_safe (not run_posteriors; the Simpson marginals and quantiles of the J checks), numbers formatted by own code (PDG decimals
# _pdg_decimals, percent, signed decimals, words).
def _grid_comb_safe(runs, n1=6001, n2=(321, 241)):
    """Summed corrected ln L of grid runs on a lattice -> (names, axes, lnP total), safe where a pass has non-finite cells: inside each
    run's final fine pass a cubic spline of ln L when the pass is finite everywhere, otherwise PCHIP along each axis; outside it the
    coarse pass by PCHIP along each axis (non-finite cells 300 nats below the lowest finite one). _grid_comb's cubic spline of the coarse
    pass rings where the nside-64 coarse passes hold non-finite cells (65-72 of 286 at the variance cap) and puts spurious maxima on the
    lattice; PCHIP never rises above the neighbouring nodes. Lattice as _lattice16."""
    from scipy.interpolate import CubicSpline, PchipInterpolator, RectBivariateSpline
    allp = []
    for run in runs:
        res, cfg = _jr(run / "grid" / "result.json"), _jr(run / "config.json")
        ps = []
        for nm in (("fine_widened.h5" if (res.get("outside_mass_check") or {}).get("widened") else "fine.h5"), "coarse.h5"):
            with h5py.File(run / "grid" / nm, "r") as h:
                names = json.loads(h.attrs["axes"])
                ax = [np.asarray(h[f"axis_{n}"][()], float) for n in names]
                n = int(h.attrs["n_events"])
                l = np.asarray(h["lnL"][()], float)
                if cfg["half_variance_correction"]:
                    l = l + 0.5 * np.asarray(h["pe_var"][()], float) - 0.5 * n / np.asarray(h["n_eff"][()], float)
            ps.append((names, ax, l))
        allp.append(ps)
    names = allp[0][0][0]
    lat = []
    for i in range(len(names)):
        lo, hi = min(p[0][1][i][0] for p in allp), max(p[0][1][i][-1] for p in allp)
        clo, chi = min(p[1][1][i][0] for p in allp), max(p[1][1][i][-1] for p in allp)
        w = hi - lo
        lat.append(np.linspace(max(clo, lo - 0.2 * w), min(chi, hi + 0.2 * w), n1 if len(names) == 1 else n2[i]))
    tot = np.zeros([len(g) for g in lat])
    for (_nf, axf, lf), (_nc, axc, lc) in allp:
        val = np.full(tot.shape, np.nan)
        for ax, l, cubic in ((axc, lc, False), (axf, lf, True)):          # the coarse pass first, the fine pass over it inside its box
            ok = np.isfinite(l)
            l = np.where(ok, l, np.min(l[ok]) - 300.0)
            ins = [(g >= a[0]) & (g <= a[-1]) for g, a in zip(lat, ax)]
            sub = [g[m] for g, m in zip(lat, ins)]
            if cubic and ok.all():
                v = CubicSpline(ax[0], l)(sub[0]) if len(ax) == 1 else RectBivariateSpline(ax[0], ax[1], l, kx=3, ky=3, s=0)(sub[0], sub[1])
            else:
                v = l
                for i, (a, s) in enumerate(zip(ax, sub)):
                    v = PchipInterpolator(a, v, axis=i)(s)
            val[np.ix_(*[np.nonzero(m)[0] for m in ins])] = v
        tot = tot + val
    return names, lat, tot


def _own_interval(v):
    """Own rendering of a median with its 90% interval: the PDG decimals of the smaller side (as the key scheme prints a combination)."""
    med, lo, hi = v["median"], v["ci90"][0], v["ci90"][1]
    up, dn = hi - med, med - lo
    d = _pdg_decimals(min(up, dn) if min(up, dn) > 0 else max(up, dn))
    return f"{med:.{d}f}^{{+{up:.{d}f}}}_{{-{dn:.{d}f}}}"


def _own_pct(lev):
    """Own rendering of a credible level in percent: an integer from 10% to 99.5%, one decimal below 10% (None above 99.5%)."""
    v = 100.0 * lev
    if v >= 99.5:
        return None
    return f"{v:.0f}" if v >= 10 else f"{v:.1f}"


def _own_pair_configs(pairs):
    """For [(nside-64 run, nside-32 run)]: the reasons the two configs differ in anything but the pixelization (events file, the
    injection files resolved, samples, redraw key, correction) or their survey files are not at nside 64 and 32 (h5py attrs)."""
    bad = []
    for r64, r32 in pairs:
        c64, c32 = _jr(RUNS16 / r64 / "config.json"), _jr(RUNS16 / r32 / "config.json")
        inj = lambda c: [str(Path(f).resolve()) for f in (c.get("injections_files") or [c["injections_file"]])]
        for k in ("events_file", "nsamp", "redraw_key", "half_variance_correction", "max_var", "free", "catalogs", "realization"):
            if c64.get(k) != c32.get(k):
                bad.append(f"{r64}: config {k} differs")
        if inj(c64) != inj(c32):
            bad.append(f"{r64}: injection files differ")
        for c, want in ((c64, 64), (c32, 32)):
            for t, path in c["surveys"].items():
                with h5py.File(path, "r") as h:
                    if int(h.attrs["nside"]) != want:
                        bad.append(f"{c['run_name']}: survey {t} at nside {int(h.attrs['nside'])}")
    return bad


def _own_comb_rows(mac, cmp, rend, name_c, name_l, runs, par, x0):
    """Own combination of `par` over `runs` (_grid_comb_safe, Simpson marginal) against the macros name_c (median and 90% interval)
    and name_l (level of the input x0); renders of both from their stored values by own formatting."""
    names, axes, tot = _grid_comb_safe(runs)
    xo, po = _marg(names, axes, tot, par)
    own = _quant(xo, po, x0)
    v = mac[name_c]["value"]
    hw = 0.5 * (v["ci90"][1] - v["ci90"][0])
    tol = max(0.01 * hw, 0.005 if par == "H0" else 5e-4)
    for nm, a, b in (("median", v["median"], own["median"]), ("lo", v["ci90"][0], own["ci90"][0]), ("hi", v["ci90"][1], own["ci90"][1])):
        cmp(name_c, a, b, tol, f"{nm} vs own summed splines (_grid_comb_safe)")
    # the level is |2 F(input) - 1|: a quantile held to `tol` moves it by up to 2 p(input) tol, so the level is held to the same accuracy
    # (at least 0.01). 21 Nov: with the input next to the median (f_AGN over the twenty nside-64 grids, four of them with non-finite cells
    # in the fine pass, where _grid_comb_safe uses PCHIP) a median that agrees to half its tolerance moves the level by 0.012
    tol_l = max(0.01, 2.0 * float(np.interp(x0, xo, po)) * tol)
    cmp(name_l, mac[name_l]["value"], own["level"], tol_l, "credible level of the input vs own (the quantiles' tolerance mapped through 2 p(input))")
    cmp(name_l, mac[name_l]["value"], v["level_of_input"], 1e-12, f"= level_of_input of {name_c}")
    rend.append((name_c, mac[name_c]["render"] == _own_interval(v)))
    rend.append((name_l, mac[name_l]["render"] == _own_pct(mac[name_l]["value"])))
    return own


def audit_skyres_sets(mac):
    import statistics
    import yaml
    rl = yaml.safe_load((RUNS16 / "RUNLIST.yaml").read_text())
    idx = {r["name"]: (str(st["stage"]), list(r["args"])) for st in rl["stages"] for r in st["runs"]}
    pm = _jr(OUT / "values" / "production_map.json")
    sk = pm.get("sky_resolution") or {}
    base = "known_population/both_catalogs/complete/equal_populations/realization_{R}"
    agn = "known_population/agn_catalog/complete/agn_hosted/realization_{R}"
    rx = lambda pat: re.compile(re.escape(pat.replace("{R}", "@")).replace("@", r"(\d{3})"))
    reals_of = lambda pat, stage_ok: sorted(int(m.group(1)) for n, (s, _a) in idx.items() if stage_ok(s) for m in [rx(pat).fullmatch(n)] if m)
    H0in = 67.74

    def strip(a, drop):
        """Arguments without --membership and without the flags in `drop` (each with its value)."""
        out, i = [], 0
        while i < len(a):
            if a[i] == "--membership":
                i += 1
            elif a[i] in drop:
                i += 2
            else:
                out.append(a[i])
                i += 1
        return out

    def flag(a, f):
        return a[a.index(f) + 1] if f in a else None

    def pairs_ok(reals, pat64, pat32, variant64):
        """Run-list check of each pair: the nside-64 run in stage X with --nside 64, --variant `variant64` and otherwise the arguments
        of its partner in stage 1 (without --membership)."""
        bad = []
        for R in reals:
            n64, n32 = pat64.format(R=f"{R:03d}"), pat32.format(R=f"{R:03d}")
            if n64 not in idx or n32 not in idx:
                bad.append(f"{n64 if n64 not in idx else n32}: not in the run list")
                continue
            (s64, a64), (s32, a32) = idx[n64], idx[n32]
            ok = (s64 == "X" and s32 == "1" and flag(a64, "--nside") == "64" and flag(a64, "--variant") == variant64
                  and strip(a64, ("--nside", "--variant")) == strip(a32, ("--variant",)))
            if not ok:
                bad.append(f"{n64}: stage {s64}/{s32}, arguments differ from the partner's by more than --nside 64 and the variant")
        return bad

    def summ(pat, R):
        p = RUNS16 / pat.format(R=f"{R:03d}") / "summary.json"
        return _jr(p) if p.exists() else None

    def shifts(reals, pat64, pat32, par="H0"):
        s = {R: (summ(pat64, R), summ(pat32, R)) for R in reals}
        if not all(a and b for a, b in s.values()):
            return None, s
        return {R: a["posterior"][par]["median"] - b["posterior"][par]["median"] for R, (a, b) in s.items()}, s

    def truth(sums, par):
        t = {float(x["truth"][par]) for x in sums}
        return t.pop() if len(t) == 1 else None

    # ---- S9: mc_pair (32000 samples, six sets; the realizations with an nside-64 grid at those settings in stage X, as S4)
    rows, bad, rend = [], [], []

    def cmp(name, got, want, tol, what):
        d = abs(got - want)
        rows.append((name, what, d, tol))
        if not d <= tol:
            bad.append(f"{name} {what}: {got:.6g} vs {want:.6g} (tol {tol:.2g})")

    # 23 Nov: SkyResMCMinShiftHzero, SkyResMCMaxShiftHzero, SkyResMCNarrowPct and pair_all's SkyResTen..., SkyResOffsetHzero... (S10) are
    # removed (main.tex used none): they must not exist
    names9 = [f"SkyResMCHzero{p}{w}" for w in ("ThirtyTwo", "SixtyFour") for p in ("Comb", "CL")]
    names11 = ["SkyResAGNHzeroCombSixtyFour", "SkyResAGNHzeroCLSixtyFour", "SkyResAGNMeanShiftHzero", "SkyResAGNSdShiftHzero"]
    gone = [n for n in mac if n in ("SkyResMCMinShiftHzero", "SkyResMCMaxShiftHzero", "SkyResMCNarrowPct") or n.startswith(("SkyResTen", "SkyResOffset"))]
    absent = [n for n in names9 + names11 if n not in mac]
    if absent or gone:
        check(False, f"S9 values.json has the {len(names9 + names11)} sky-resolution macros combined over the realizations "
                     f"({', '.join(absent[:6])} missing) and none of the removed ones ({', '.join(gone[:6])} present)")
        return
    p64, p32 = base + "_nside64_s32000_inj6", base + "_s32000_inj6"
    r9 = reals_of(p64, lambda s: s.startswith("X"))
    sh, sums = shifts(r9, p64, p32)
    have = sh is not None and len(r9) >= 2
    map9 = (sk.get("mc_pair") or {}).get("realizations")
    if have:
        x0 = truth([s for ab in sums.values() for s in ab], "H0")
        cmp("SkyResMCHzeroComb", float(x0 is None), 0.0, 0.0, "one input H0 in every summary's truth")
        for w, pat in (("ThirtyTwo", p32), ("SixtyFour", p64)):
            _own_comb_rows(mac, cmp, rend, f"SkyResMCHzeroComb{w}", f"SkyResMCHzeroCL{w}", [RUNS16 / pat.format(R=f"{R:03d}") for R in r9], "H0", x0 or H0in)
    black = all(mac[n]["class"] == "plain" for n in names9)
    nb = len(bad) + sum(1 for _n, ok in rend if not ok)
    check(black == have and map9 == r9 and not bad and all(ok for _n, ok in rend),
          f"S9 SkyResMCHzeroComb/CL at nside 32 and 64 (the removed Min/Max shift, NarrowPct and SkyResTen..., SkyResOffset... absent) over the realizations with an nside-64 grid at 32000 samples "
          f"and six sets in stage X {r9} (the map's mc_pair: {map9}): black exactly when every summary exists ({'black' if black else 'red'}, "
          f"summaries {'all there' if have else 'missing'}); {len(rows)} numbers against the summaries and the own summed splines (_grid_comb_safe)"
          + (f", largest deviation over tolerance {max(r[2] / r[3] if r[3] else (0 if r[2] == 0 else np.inf) for r in rows):.2g}" if rows else "")
          + f"; renders by own formatting {sum(ok for _n, ok in rend)} of {len(rend)}" + (f"; MISMATCH ({nb}): {'; '.join(bad[:4])}"
                                                                                           + f" {[n for n, ok in rend if not ok][:4]}" if nb else ""))
    # ---- S10 (pair_all: SkyResTen..., SkyResOffsetHzero...) removed 23 Nov with its macros; S9 checks that none of them exists
    # ---- S11: agn_pair (key AGNhosted's stage-1 runs, 32000 samples and six sets, against stage X's nside-64 runs)
    rows.clear(), bad.clear(), rend.clear()
    p32, p64 = agn + "_s32000_inj6", agn + "_nside64_s32000_inj6"
    r11 = reals_of(p32, lambda s: s == "1")
    lb = pairs_ok(r11, p64, p32, "nside64_s32000_inj6")
    sh, sums = shifts(r11, p64, p32)
    have = sh is not None and len(r11) >= 2 and not lb
    m11 = sk.get("agn_pair") or {}
    ka_reals = ((pm.get("keys") or {}).get("AGNhosted") or {}).get("realizations")
    ok_map = m11.get("realizations") == r11 and m11.get("realizations_wanted") == r11 and ka_reals == r11
    if have:
        lb += _own_pair_configs([(p64.format(R=f"{R:03d}"), p32.format(R=f"{R:03d}")) for R in r11])
        xs = [sh[R] for R in r11]
        m_, s_ = statistics.fmean(xs), statistics.stdev(xs)
        cmp("SkyResAGNMeanShiftHzero", mac["SkyResAGNMeanShiftHzero"]["value"], m_, 1e-12, "mean shift of the summaries' medians")
        cmp("SkyResAGNSdShiftHzero", mac["SkyResAGNSdShiftHzero"]["value"], s_, 1e-12, "standard deviation (ddof 1) of the shifts")
        rend += [("SkyResAGNMeanShiftHzero", mac["SkyResAGNMeanShiftHzero"]["render"] == _own_fixed(m_, 2)),
                 ("SkyResAGNSdShiftHzero", mac["SkyResAGNSdShiftHzero"]["render"] == _own_fixed(s_, 2))]
        x0 = truth([a for a, _b in sums.values()], "H0")
        _own_comb_rows(mac, cmp, rend, "SkyResAGNHzeroCombSixtyFour", "SkyResAGNHzeroCLSixtyFour",
                       [RUNS16 / p64.format(R=f"{R:03d}") for R in r11], "H0", x0 or H0in)
    black = all(mac[n]["class"] == "plain" for n in names11)
    nb = len(bad) + len(lb) + sum(1 for _n, ok in rend if not ok)
    check(black == have and ok_map and not lb and not bad and all(ok for _n, ok in rend),
          f"S11 SkyResAGN... over the realizations of key AGNhosted in stage 1 {r11} (own parser; the map's agn_pair {m11.get('realizations')}, "
          f"key AGNhosted {ka_reals}): each nside-64 run in stage X with the stage-1 arguments plus --nside 64 (32000 samples, the same six "
          f"sets), configs equal but for the pixelization and survey files at nside 64 and 32 (h5py); black exactly when every summary exists "
          f"({'black' if black else 'red'}); {len(rows)} numbers against the summaries and the own summed splines (_grid_comb_safe)"
          + (f", largest deviation over tolerance {max(r[2] / r[3] if r[3] else (0 if r[2] == 0 else np.inf) for r in rows):.2g}" if rows else "")
          + f"; renders by own formatting {sum(ok for _n, ok in rend)} of {len(rend)}"
          + (f"; MISMATCH ({nb}): {'; '.join((lb + bad)[:4])} {[n for n, ok in rend if not ok][:4]}" if nb else ""))
    audit_skyres_twenty(mac, idx, sk, strip, flag, truth, H0in)


# S12-S13 (21 Nov, later): the pair over the twenty realizations of key KnownBoth (skyres pair_twenty, SkyResAll...) and the paired shift
# of two other sets (pair_half, SkyResHalf...; agn_pair's SkyResAGN<Mean|Sd>ShiftHzero), by own code: the run lists parsed here (the
# deliberate stage-X runs of KnownBoth by _own_kb_stage_x; every nside-64 run's flags against its partner's), configs and survey files read
# directly (h5py), the summaries read directly, statistics by the statistics module, combinations by _grid_comb_safe, own formatting; and
# every number that analysis_16's reference computation has (tests/results/skyres_sequence.json: KnownBoth_production_111_120,
# HostsUnknown_peak_f0p50_s0, AGNhosted at nside 64) against it.
def audit_skyres_twenty(mac, idx, sk, strip, flag, truth, H0in):
    import statistics
    import yaml
    rows, bad, rend = [], [], []

    def cmp(name, got, want, tol, what):
        d = abs(got - want)
        rows.append((name, what, d, tol))
        if not d <= tol:
            bad.append(f"{name} {what}: {got:.6g} vs {want:.6g} (tol {tol:.2g})")
    names12 = ["SkyResAllNreal", "SkyResAllHzeroComb", "SkyResAllHzeroCL", "SkyResAllfAGNComb", "SkyResAllfAGNCL", "SkyResAllMeanShiftHzero",
               "SkyResAllSdShiftHzero", "SkyResAllMeanShiftfAGN", "SkyResAllSdShiftfAGN", "SkyResAllNarrowPct", "SkyResAllShiftPctHalfWid"]
    names13 = ["SkyResHalfMeanShiftHzero", "SkyResHalfSdShiftHzero", "SkyResAGNMeanShiftHzero", "SkyResAGNSdShiftHzero"]
    absent = [n for n in names12 + names13 if n not in mac]
    if absent:
        check(False, f"S12 values.json has the {len(names12 + names13)} macros of the pairs over the twenty realizations and the other two sets "
                     f"({', '.join(absent[:6])} missing)")
        return
    seq = _jr(SEQ_JSON)["analyses"]
    base = "known_population/both_catalogs/complete/equal_populations/realization_{R}"
    # ---- S12: pair_twenty
    s1, kx = _own_kb_stage_x(idx)
    r12 = sorted(set(s1) | set(kx))
    n32 = {R: kx.get(R) or base.format(R=f"{R:03d}") for R in r12}
    n64 = {R: (base + "_nside64").format(R=f"{R:03d}") for R in r12}
    lb = []
    for R in r12:
        if n64[R] not in idx or n32[R] not in idx:
            lb.append(f"{n64[R]}: pair not in the run list")
            continue
        (s64, a64), (s32, a32) = idx[n64[R]], idx[n32[R]]
        if not (s64 == "X" and s32 == ("X" if R in kx else "1") and flag(a64, "--nside") == "64" and flag(a64, "--variant") == "nside64"
                and strip(a64, ("--nside", "--variant")) == strip(a32, ("--variant",))):
            lb.append(f"{n64[R]}: stage {s64}/{s32}, flags differ from its partner's by more than --nside 64 and the variant")
    m12 = sk.get("pair_twenty") or {}
    ok_map = (m12.get("realizations") == r12 and m12.get("realizations_wanted") == r12
              and {int(k): v for k, v in (m12.get("partners") or {}).items()} == kx and m12.get("partner_key") == "KnownBoth")
    S = {R: tuple((lambda p: _jr(p) if p.exists() else None)(RUNS16 / n / "summary.json") for n in (n64[R], n32[R])) for R in r12}
    have = all(a and b for a, b in S.values()) and len(r12) >= 2 and not lb
    nr = mac["SkyResAllNreal"]
    ok_n = nr["value"] == len(r12) and nr["render"] == _OWN_WORDS[len(r12)] and (nr["class"] == "plain") == (not lb)
    sq = (seq.get("KnownBoth_production_111_120") or {}).get("nsides", {}).get("64") or {}
    if have:
        lb += _own_pair_configs([(n64[R], n32[R]) for R in r12])
        hwd = lambda s, par: 0.5 * (s["posterior"][par]["ci90"][1] - s["posterior"][par]["ci90"][0])
        sh = {}
        for par, q, d in (("H0", "Hzero", 2), ("f_agn", "fAGN", 3)):
            xs = [S[R][0]["posterior"][par]["median"] - S[R][1]["posterior"][par]["median"] for R in r12]
            sh[par] = xs
            m_, s_ = statistics.fmean(xs), statistics.stdev(xs)
            cmp(f"SkyResAllMeanShift{q}", mac[f"SkyResAllMeanShift{q}"]["value"], m_, 1e-12, "mean shift of the summaries' medians")
            cmp(f"SkyResAllSdShift{q}", mac[f"SkyResAllSdShift{q}"]["value"], s_, 1e-12, "standard deviation (ddof 1) of the shifts")
            rend += [(f"SkyResAllMeanShift{q}", mac[f"SkyResAllMeanShift{q}"]["render"] == _own_fixed(m_, d)),
                     (f"SkyResAllSdShift{q}", mac[f"SkyResAllSdShift{q}"]["render"] == _own_fixed(s_, d))]
            ps = ((sq.get("paired_to_nside32") or {}).get(par) or {}).get("stats") or {}
            if ps.get("n") == len(r12):
                cmp(f"SkyResAllMeanShift{q}", mac[f"SkyResAllMeanShift{q}"]["value"], ps["mean"], 1e-9, "mean vs skyres_sequence.json")
                cmp(f"SkyResAllSdShift{q}", mac[f"SkyResAllSdShift{q}"]["value"], ps["sd"], 1e-9, "sd vs skyres_sequence.json")
            else:
                bad.append(f"skyres_sequence.json KnownBoth_production_111_120 nside 64 has {ps.get('n')} paired {par} shifts")
        rat = [hwd(S[R][0], "H0") / hwd(S[R][1], "H0") for R in r12]
        nar = 100.0 * (1.0 - statistics.fmean(rat))
        cmp("SkyResAllNarrowPct", mac["SkyResAllNarrowPct"]["value"], nar, 1e-9, "100 (1 - mean half-width ratio) of the summaries")
        hr = ((sq.get("paired_to_nside32") or {}).get("H0") or {}).get("half_width_ratio_mean")
        if hr is not None:
            cmp("SkyResAllNarrowPct", mac["SkyResAllNarrowPct"]["value"], 100.0 * (1.0 - hr), 1e-9, "100 (1 - half_width_ratio_mean of skyres_sequence.json)")
        hwm = statistics.fmean(hwd(S[R][1], "H0") for R in r12)
        pct = 100.0 * abs(statistics.fmean(sh["H0"])) / hwm
        cmp("SkyResAllShiftPctHalfWid", mac["SkyResAllShiftPctHalfWid"]["value"], pct, 1e-9, "100 |mean H0 shift| / mean nside-32 half-width of the summaries")
        rend += [("SkyResAllNarrowPct", mac["SkyResAllNarrowPct"]["render"] == _own_fixed(nar, 0)),
                 ("SkyResAllShiftPctHalfWid", mac["SkyResAllShiftPctHalfWid"]["render"] == _own_fixed(pct, 0))]
        x0 = truth([a for a, _b in S.values()], "H0")
        f0 = truth([a for a, _b in S.values()], "f_agn")
        cmp("SkyResAllHzeroComb", float(x0 is None or f0 is None), 0.0, 0.0, "one input H0 and f_AGN in every summary's truth")
        runs64 = [RUNS16 / n64[R] for R in r12]
        _own_comb_rows(mac, cmp, rend, "SkyResAllHzeroComb", "SkyResAllHzeroCL", runs64, "H0", x0 or H0in)
        _own_comb_rows(mac, cmp, rend, "SkyResAllfAGNComb", "SkyResAllfAGNCL", runs64, "f_agn", f0 if f0 is not None else 0.3)
        sc = sq.get("combined") or {}
        if sc.get("realizations") == r12 and all(sq["runs"][str(R)]["run"] == n64[R] for R in r12):
            for q, par in (("Hzero", "H0"), ("fAGN", "f_agn")):
                v, r_ = mac[f"SkyResAll{q}Comb"]["value"], sc[par]
                for nm, a, b in (("median", v["median"], r_["median"]), ("lo", v["ci90"][0], r_["median"] - r_["minus"]), ("hi", v["ci90"][1], r_["median"] + r_["plus"])):
                    cmp(f"SkyResAll{q}Comb", a, b, 1e-6, f"{nm} vs skyres_sequence.json")
                cmp(f"SkyResAll{q}CL", mac[f"SkyResAll{q}CL"]["value"], r_["level_of_input"], 1e-6, "credible level vs skyres_sequence.json")
        else:
            bad.append(f"skyres_sequence.json KnownBoth_production_111_120 nside 64 combined over {sc.get('realizations')} or other runs")
    black = all(mac[n]["class"] == "plain" for n in names12)
    nb = len(bad) + len(lb) + sum(1 for _n, ok in rend if not ok)
    check(black == have and ok_map and ok_n and not lb and not bad and all(ok for _n, ok in rend),
          f"S12 SkyResAll... over the {len(r12)} realizations of key KnownBoth (own parser: stage 1 {s1}, stage X realization_NNN_nside32 {sorted(kx)}; "
          f"the map's pair_twenty {m12.get('realizations')}): each nside-64 run in stage X with its partner's flags plus --nside 64 and the variant, "
          f"configs equal but for the pixelization and survey files at nside 64 and 32 (h5py); SkyResAllNreal '{nr['render']}' ({nr['class']}); black "
          f"exactly when every summary exists ({'black' if black else 'red'}); {len(rows)} numbers against the summaries, the own summed splines "
          f"(_grid_comb_safe) and tests/results/skyres_sequence.json (KnownBoth_production_111_120, nside 64)"
          + (f", largest deviation over tolerance {max(r[2] / r[3] if r[3] else (0 if r[2] == 0 else np.inf) for r in rows):.2g}" if rows else "")
          + f"; renders by own formatting {sum(ok for _n, ok in rend)} of {len(rend)}"
          + (f"; MISMATCH ({nb}): {'; '.join((lb + bad)[:4])} {[n for n, ok in rend if not ok][:4]}" if nb else ""))
    # ---- S13: the paired shift of H0 with half of the mergers in AGN (host class unknown) and with every merger in AGN (AGN catalog alone)
    rows.clear(), bad.clear(), rend.clear()
    rx_ = yaml.safe_load((RUNS16 / "RUNLIST_experiment.yaml").read_text())
    idx_e = {r["name"]: (str(st["stage"]), list(r["args"])) for st in rx_["stages"] for r in st["runs"]}
    hb = "two_tracers_one_population/both_catalogs/complete/peak_f0p50_s0/realization_{R}"
    r13 = sorted(int(m.group(1)) for n, (s, _a) in idx_e.items() if s.startswith("E") for m in [re.fullmatch(hb.replace("{R}", r"(\d{3})"), n)] if m)
    lb = []
    for R in r13:
        a, b = (hb + "_nside64").format(R=f"{R:03d}"), hb.format(R=f"{R:03d}")
        if a not in idx:
            lb.append(f"{a}: not in RUNLIST.yaml")
            continue
        (s64, a64), (_s32, a32) = idx[a], idx_e[b]
        if not (s64 == "X" and flag(a64, "--nside") == "64" and strip(a64, ("--nside", "--variant")) == strip(a32, ())):
            lb.append(f"{a}: stage {s64}, flags differ from its partner's by more than --nside 64 and the variant")
    m13 = sk.get("pair_half") or {}
    ok_map = m13.get("realizations") == r13 and m13.get("realizations_wanted") == r13
    Sh = {R: tuple((lambda p: _jr(p) if p.exists() else None)(RUNS16 / n / "summary.json") for n in ((hb + "_nside64").format(R=f"{R:03d}"), hb.format(R=f"{R:03d}")))
          for R in r13}
    have_h = all(a and b for a, b in Sh.values()) and len(r13) >= 2 and not lb
    out = []
    if have_h:
        lb += _own_pair_configs([((hb + "_nside64").format(R=f"{R:03d}"), hb.format(R=f"{R:03d}")) for R in r13])
        xs = [Sh[R][0]["posterior"]["H0"]["median"] - Sh[R][1]["posterior"]["H0"]["median"] for R in r13]
        m_, s_ = statistics.fmean(xs), statistics.stdev(xs)
        cmp("SkyResHalfMeanShiftHzero", mac["SkyResHalfMeanShiftHzero"]["value"], m_, 1e-12, "mean shift of the summaries' medians")
        cmp("SkyResHalfSdShiftHzero", mac["SkyResHalfSdShiftHzero"]["value"], s_, 1e-12, "standard deviation (ddof 1) of the shifts")
        rend += [("SkyResHalfMeanShiftHzero", mac["SkyResHalfMeanShiftHzero"]["render"] == _own_fixed(m_, 2)),
                 ("SkyResHalfSdShiftHzero", mac["SkyResHalfSdShiftHzero"]["render"] == _own_fixed(s_, 2))]
        out.append(f"half of the mergers in AGN {m_:+.4f} (sd {s_:.4f}, {len(r13)} realizations)")
    for tag, blk_, mm, ms in (("HostsUnknown_peak_f0p50_s0", seq.get("HostsUnknown_peak_f0p50_s0"), "SkyResHalfMeanShiftHzero", "SkyResHalfSdShiftHzero"),
                              ("AGNhosted", seq.get("AGNhosted"), "SkyResAGNMeanShiftHzero", "SkyResAGNSdShiftHzero")):
        ps = ((((blk_ or {}).get("nsides") or {}).get("64") or {}).get("paired_to_nside32") or {}).get("H0", {}).get("stats") or {}
        if not ps.get("n"):
            bad.append(f"skyres_sequence.json {tag}: no paired H0 shifts at nside 64")
            continue
        want_n = len(r13) if tag.startswith("Hosts") else len((sk.get("agn_pair") or {}).get("realizations") or [])
        cmp(mm, float(ps["n"]), float(want_n), 0.0, f"number of pairs in skyres_sequence.json {tag}")
        cmp(mm, mac[mm]["value"], ps["mean"], 1e-9, f"mean vs skyres_sequence.json {tag}")
        cmp(ms, mac[ms]["value"], ps["sd"], 1e-9, f"sd vs skyres_sequence.json {tag}")
        out.append(f"{tag}: {ps['mean']:+.4f} (sd {ps['sd']:.4f})")
    black = all(mac[n]["class"] == "plain" for n in ("SkyResHalfMeanShiftHzero", "SkyResHalfSdShiftHzero"))
    nb = len(bad) + len(lb) + sum(1 for _n, ok in rend if not ok)
    check(black == have_h and ok_map and not lb and not bad and all(ok for _n, ok in rend),
          f"S13 the paired shift of the H0 median, nside 64 minus 32, of two other sets: half of the mergers in AGN, host class unknown "
          f"(SkyResHalf..., RUNLIST_experiment.yaml stage E1 {r13} against RUNLIST.yaml stage X, flags and configs as S12; the map's pair_half "
          f"{m13.get('realizations')}) and every merger in AGN with the AGN catalog alone (SkyResAGN..., S11's runs); {len(rows)} numbers against the "
          f"summaries and tests/results/skyres_sequence.json: {'; '.join(out)}; renders by own formatting {sum(ok for _n, ok in rend)} of {len(rend)}"
          + (f"; MISMATCH ({nb}): {'; '.join((lb + bad)[:4])} {[n for n, ok in rend if not ok][:4]}" if nb else ""))


# K3 (21 Nov): H0 with the host class unknown and known on the same mergers (classes ClsDiff<f>Hzero<Mean|Sd>, ClsDiff<f>Npos; 23 Nov:
# the combinations Cls<Unknown|Known><f>Hzero<Comb|CL> are removed, main.tex used none, and must not exist), by own code:
# RUNLIST_experiment.yaml parsed here (realizations and events files of each run), the summaries read directly, statistics by the
# statistics module.
def audit_classes_comb(mac):
    import statistics
    import yaml
    rx_ = yaml.safe_load((RUNS16 / "RUNLIST_experiment.yaml").read_text())
    ev = {}
    for st in rx_["stages"]:
        for r in st["runs"]:
            ev[r["name"]] = r.get("events_file")
    blk = ((_jr(OUT / "values" / "production_map.json").get("classes") or {}).get("macros") or {})
    rows, bad, rend, states = [], [], [], []

    def cmp(name, got, want, tol, what):
        d = abs(got - want)
        rows.append((name, what, d, tol))
        if not d <= tol:
            bad.append(f"{name} {what}: {got:.6g} vs {want:.6g} (tol {tol:.2g})")

    allnames = [f"ClsDiff{w}{s}" for w in ("Thirty", "Fifty") for s in ("HzeroMean", "HzeroSd", "Npos")]
    gone = [n for n in mac if re.fullmatch(r"Cls(?:Unknown|Known)(?:Thirty|Fifty)Hzero(?:Comb|CL)", n)]
    absent = [n for n in allnames if n not in mac]
    if absent or gone:
        check(False, f"K3 values.json has the {len(allnames)} macros of H0 with the host class unknown and known ({', '.join(absent[:6])} missing) "
                     f"and none of the removed combinations ({', '.join(gone[:6])} present)")
        return
    for w, lab in (("Thirty", "f0p30"), ("Fifty", "f0p50")):
        pu = f"two_tracers_one_population/both_catalogs/complete/peak_{lab}_s0/realization_{{R}}"
        pk = f"hosts_known/both_catalogs/complete/peak_{lab}_s0/realization_{{R}}_agn_s32000_inj1"
        rr = {p: sorted(int(m.group(1)) for n in ev for m in [re.fullmatch(re.escape(p.replace("{R}", "@")).replace("@", r"(\d{3})"), n)] if m)
              for p in (pu, pk)}
        reals = rr[pu]
        # the map's records: ClsDiff from HWUnknown<f> and HWKnown<f>, whose runs are these, at the run lists' realizations; no record of the
        # removed combinations; the same events file per realization
        ok_map = all((blk.get(f"HW{t}{w}") or {}).get("run") == pat and (blk.get(f"HW{t}{w}") or {}).get("realizations") == reals
                     for t, pat in (("Unknown", pu), ("Known", pk)))
        ok_map = ok_map and all((blk.get(f"ClsDiff{w}{s}") or {}).get("from") == [f"HWUnknown{w}", f"HWKnown{w}"]
                                for s in ("HzeroMean", "HzeroSd", "Npos"))
        ok_map = ok_map and not any(f"Cls{t}{w}Hzero{p}" in blk for t in ("Unknown", "Known") for p in ("Comb", "CL"))
        same_ev = rr[pu] == rr[pk] and len(reals) >= 2 and all(ev[pu.format(R=f"{R:03d}")] and ev[pu.format(R=f"{R:03d}")] == ev[pk.format(R=f"{R:03d}")]
                                                                for R in reals)
        su = {R: (lambda p: _jr(p) if p.exists() else None)(RUNS16 / pu.format(R=f"{R:03d}") / "summary.json") for R in reals}
        sk_ = {R: (lambda p: _jr(p) if p.exists() else None)(RUNS16 / pk.format(R=f"{R:03d}") / "summary.json") for R in reals}
        hu, hk = all(su.values()), all(sk_.values())
        names_w = [n for n in allnames if w in n]
        for n in names_w:
            states.append((n, (mac[n]["class"] == "plain") == (hu and hk)))
        cmp(f"ClsDiff{w}", float(not (ok_map and same_ev)), 0.0, 0.0, f"map records = the runs of HWUnknown{w}/HWKnown{w} on the same events files {reals}")
        if not (hu and hk):
            continue
        d = [su[R]["posterior"]["H0"]["median"] - sk_[R]["posterior"]["H0"]["median"] for R in reals]
        m_, s_, n_ = statistics.fmean(d), statistics.stdev(d), sum(1 for x in d if x > 0)
        cmp(f"ClsDiff{w}HzeroMean", mac[f"ClsDiff{w}HzeroMean"]["value"], m_, 1e-12, "mean of the paired differences of the summaries' medians")
        cmp(f"ClsDiff{w}HzeroSd", mac[f"ClsDiff{w}HzeroSd"]["value"], s_, 1e-12, "standard deviation (ddof 1) of the paired differences")
        cmp(f"ClsDiff{w}Npos", float(mac[f"ClsDiff{w}Npos"]["value"]), float(n_), 0.0, "realizations with a positive difference")
        rend += [(f"ClsDiff{w}HzeroMean", mac[f"ClsDiff{w}HzeroMean"]["render"] == _own_fixed(m_, 2)),
                 (f"ClsDiff{w}HzeroSd", mac[f"ClsDiff{w}HzeroSd"]["render"] == _own_fixed(s_, 2)),
                 (f"ClsDiff{w}Npos", mac[f"ClsDiff{w}Npos"]["render"] == _OWN_WORDS[n_])]
    nb = len(bad) + sum(1 for _n, ok in rend + states if not ok)
    check(not bad and all(ok for _n, ok in rend + states),
          f"K3 H0 with the host class unknown and known on the same mergers ({len(allnames)} macros ClsDiff...; the combinations Cls...Hzero<Comb|CL> "
          f"absent; f = 0.3 and 0.5, the runs of RUNLIST_experiment.yaml parsed here, the map's ClsDiff records from HWUnknown<f> and HWKnown<f> "
          f"and their runs these, the same events file in every realization): black exactly when every summary exists "
          f"({sum(ok for _n, ok in states)} of {len(states)}); {len(rows)} numbers against the summaries"
          + (f", largest deviation over tolerance {max(r[2] / r[3] if r[3] else (0 if r[2] == 0 else np.inf) for r in rows):.2g}" if rows else "")
          + f"; renders by own formatting {sum(ok for _n, ok in rend)} of {len(rend)}"
          + (f"; MISMATCH ({nb}): {'; '.join(bad[:4])} {[n for n, ok in rend + states if not ok][:4]}" if nb else ""))


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
            if str(st["stage"]).startswith(("V", "X")):
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


def audit_density_phys(mac):
    """L2/L3: the two stage-X runs with both densities given in Mpc^-3 (density.py FluxEighteenPhys...), by own code: the run list parsed
    here, the summaries read directly, the given densities against the production partner's densities from the counts converted at
    H0 = 67.74; and no run of a check or validation stage (X, V) is a production run of any key, experiment macro or density arm."""
    import yaml
    rl = yaml.safe_load((RUNS16 / "RUNLIST.yaml").read_text())
    idx = {r["name"]: (str(st["stage"]), r["args"]) for st in rl["stages"] for r in st["runs"]}
    kv = lambda a: {a[i].lstrip("-"): a[i + 1] for i in range(len(a) - 1) if a[i].startswith("--") and not a[i + 1].startswith("--")}
    base = "known_population/both_catalogs/flux_limit_18_densities_given{}/equal_populations/realization_100_units_physical"
    runs = {"": base.format(""), "Double": base.format("_agn_density_double")}
    partner = "known_population/both_catalogs/flux_limit_18_densities_from_counts/equal_populations/realization_100"
    names = [f"FluxEighteenPhys{t}{q}{pc}" for t in runs for q in ("fAGN", "Hzero") for pc in ("", "Med", "Lo", "Hi", "Wid", "HalfWid")] + \
            [f"FluxEighteenPhysDoubleDelta{q}" for q in ("fAGN", "Hzero")]
    if not all(n in mac for n in names):
        check(False, f"L2 values.json has the {len(names)} FluxEighteenPhys... macros ({', '.join(n for n in names if n not in mac)[:200]} missing)")
        return
    why = []
    ps = _jr(RUNS16 / partner / "summary.json") if (RUNS16 / partner / "summary.json").exists() else None
    for t, n in runs.items():
        if n not in idx:
            why.append(f"no {n}")
            continue
        st, a = idx[n]
        k = kv(a)
        if not (st.startswith("X") and k.get("density") == "given" and k.get("density-units") == "physical" and k.get("flux") == "18"
                and abs(float(k.get("agn-density-factor", "1")) - (2.0 if t else 1.0)) < 1e-12):
            why.append(f"{n}: stage {st}, {k}")
        if ps is not None:
            for c, f in (("gal", "log10n-gal"), ("agn", "log10n-agn")):
                if abs(float(k[f]) - (ps["densities"][c]["log10n_used"] + 3 * math.log10(0.6774))) > 1e-6:
                    why.append(f"{n} --{f} {k[f]} is not the partner's density from the counts converted at H0 = 67.74")
    have = all((RUNS16 / n / "summary.json").exists() for n in runs.values())
    black = [mac[n]["class"] == "plain" for n in names]
    dev = []
    if have:
        S = {t: _jr(RUNS16 / n / "summary.json") for t, n in runs.items()}
        for t, s in S.items():
            if not all(s["densities"][c]["units"] == "physical" for c in ("gal", "agn")):
                why.append(f"{runs[t]}: summary densities not in physical units")
            for q, par in (("fAGN", "f_agn"), ("Hzero", "H0")):
                p = s["posterior"][par]
                v = mac[f"FluxEighteenPhys{t}{q}"]["value"]
                dev += [abs(v["median"] - p["median"]), abs(v["ci90"][0] - p["ci90"][0]), abs(v["ci90"][1] - p["ci90"][1])]
                for pc, want in (("Med", p["median"]), ("Lo", p["ci90"][0]), ("Hi", p["ci90"][1]), ("Wid", p["ci90"][1] - p["ci90"][0]),
                                 ("HalfWid", 0.5 * (p["ci90"][1] - p["ci90"][0]))):
                    dev.append(abs(mac[f"FluxEighteenPhys{t}{q}{pc}"]["value"] - want))
        for q, par in (("fAGN", "f_agn"), ("Hzero", "H0")):
            dev.append(abs(mac[f"FluxEighteenPhysDoubleDelta{q}"]["value"] - (S["Double"]["posterior"][par]["median"] - S[""]["posterior"][par]["median"])))
    sh = {q: mac[f"FluxEighteenPhysDoubleDelta{q}"]["value"] for q in ("Hzero", "fAGN")}
    prod = {q: mac.get(f"FluxEighteenDoubleDelta{q}", {}).get("value") for q in ("Hzero", "fAGN")}
    sh = {q: (v if isinstance(v, (int, float)) else float("nan")) for q, v in sh.items()}
    prod = {q: (v if isinstance(v, (int, float)) else float("nan")) for q, v in prod.items()}
    check(not why and all(b == have for b in black) and max(dev, default=0.0) < 1e-12,
          f"L2 the two runs with both densities given in Mpc^-3 are RUNLIST.yaml's (own parser): stage X, --density given --density-units physical, "
          f"m < 18, AGN factor 1 and 2, the given densities = the production partner's from the counts + 3 log10 0.6774; the {len(names)} "
          f"FluxEighteenPhys... macros print black exactly when both summaries exist ({have}) and equal the summaries to {max(dev, default=0.0):.1g} "
          f"(shift under the doubling: H0 {sh['Hzero']:+.2f}, f_AGN {sh['fAGN']:+.3f}; production, h^3 Mpc^-3: {prod['Hzero']:+.2f}, {prod['fAGN']:+.3f})"
          f"{'; ' + '; '.join(why) if why else ''}")
    # L3: check and validation stages never feed a key, an experiment macro or a density arm (own regex of every pattern of the map)
    pm = _jr(OUT / "values" / "production_map.json")
    pats = [sp["run"] for sp in pm.get("keys", {}).values()] + [r["run"] for r in (pm.get("classes") or {}).get("macros", {}).values() if r.get("run")]
    pats += [rec[k] for rec in ((pm.get("density_response") or {}).get("arms") or {}).values() for k in ("run", "unit_run")]
    rx = [re.compile("^" + re.escape(p).replace(re.escape("{R}"), r"\d{3}") + "$") for p in pats]
    nonprod = [n for n, (st, _a) in idx.items() if st.startswith(("X", "V"))]
    # the second deliberate exception (build_values.STAGE_V_KEY_RUNS, 23 Nov): the pattern of key HDHost and of its macro HWHost is stage V8's
    # realization_NNN_unif_then_rslice (_HD_V8); it may match exactly the V8 runs the own parser finds (_own_hd_stage_v), nothing else may
    allowed_v = set(_own_hd_stage_v({n: (st, a) for n, (st, a) in idx.items()})[1].values())
    hit = [n for n in nonprod if any(r.match(n) and not (p == _HD_V8 and n in allowed_v) for p, r in zip(pats, rx))]
    exc_v = sorted(n for n in nonprod if n in allowed_v and any(p == _HD_V8 and r.match(n) for p, r in zip(pats, rx)))
    n_v_pats = sum(1 for p in pats if p == _HD_V8)
    # the runs the map names per realization (runs_by_realization): only the deliberate exception, KnownBoth's stage-X realization_NNN_nside32
    # runs as the own parser finds them (_own_kb_stage_x)
    named = {k: set((sp.get("runs_by_realization") or {}).values()) for k, sp in pm.get("keys", {}).items()}
    allowed = set(_own_kb_stage_x({n: (st, a) for n, (st, a) in idx.items()})[1].values())
    hit += sorted(n for k, ns in named.items() for n in ns if n in nonprod and not (k == "KnownBoth" and n in allowed))
    exc = sorted(n for n in named.get("KnownBoth", ()) if n in allowed)
    check(not hit and set(exc) == allowed and set(exc_v) == allowed_v and n_v_pats == 2,
          f"L3 none of the {len(nonprod)} runs of the check and validation stages (X, V) of RUNLIST.yaml matches a run pattern of the "
          f"production map ({len(pats)} patterns: keys, experiment macros, density arms) or is named by it per realization, but for the two "
          f"deliberate exceptions: key KnownBoth's {len(exc)} stage-X runs realization_NNN_nside32 (stage 1's flags plus --variant nside32, own "
          f"parser) and key HDHost's {len(exc_v)} stage-V8 runs realization_NNN_unif_then_rslice (stage 4's flags plus the sampler's and the "
          f"variant, own parser), matched only by the pattern of key HDHost and of HWHost ({n_v_pats} patterns)"
          f"{': ' + ', '.join(hit[:4]) if hit else ''}{'; the map lacks some of them' if set(exc) != allowed or set(exc_v) != allowed_v else ''}")


def _trap_median(x, L):
    p = np.exp(L - L.max())
    c = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
    return float(np.interp(0.5, c / c[-1], x))


if __name__ == "__main__":
    sys.exit(main())
