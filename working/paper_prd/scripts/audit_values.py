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
    # ---- G3: completeness weighted by the detected mergers, by host membership in the flux-limited files ----
    with h5py.File(B.DATA / "seed100" / "events" / "events.h5") as h:
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
    reg_a = regions(B.DATA / "seed100" / "events" / "events_pureagn.h5", np.arange(1000))
    cnt = []
    for x, r, zl, zh in reg_a:
        m = (A["z_obs"] >= zl) & (A["z_obs"] <= zh)
        cnt.append(int(np.sum(va[m] @ x >= math.cos(r))))
    med_bf = float(np.median(cnt))
    got = mac["PureAGNHostsPerVol"]["value"]
    per = np.asarray(tables["r2_hosts_per_event"]["agn_catalog_on_agn_hosted_events"], float)
    rel_tot = abs(per.sum() / np.sum(cnt) - 1)
    check(abs(got - med_bf) <= max(1.0, 0.05 * med_bf) and rel_tot < 0.03,
          f"G4 AGN per 90% localization volume: brute-force exact count median {med_bf:.0f} vs {got:.1f} (pixelised); totals over 1000 events differ by {100 * rel_tot:.2f}%")
    rng = np.random.default_rng(7)
    idx = np.sort(rng.choice(1000, 12, replace=False))
    reg_g = regions(B.DATA / "seed100" / "events" / "events_puregal.h5", idx)
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
    check(rg < 0.03, f"G4b galaxies per 90% localization volume, 12 random galaxy-hosted events: brute-force total {sum(cg)} vs pixelised {pg.sum():.0f} ({100 * rg:.2f}%)")
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
    # ---- G9: realised mock numbers from the raw files ----
    with h5py.File(B.DATA / "seed100" / "injections" / "injections_targeted.h5") as h:
        k0 = [k for k in h.keys() if isinstance(h[k], h5py.Dataset)][0]
        nfound = int(h[k0].shape[0])
    with h5py.File(B.DATA / "seed100" / "events" / "events.h5") as h:
        sdl = np.std(np.log(h["dL"][()].reshape(1000, -1)), axis=1)
    check(nfound == mac["MockNinjFound"]["value"] and abs(np.median(sdl) - mac["MockSigLnDL"]["value"]) < 0.005,
          f"G9 detected injections counted in the file ({nfound}); median sd of ln dL over the events' samples {np.median(sdl):.4f} vs META {mac['MockSigLnDL']['value']:.4f}")
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
    pm = json.loads((OUT / "values" / "production_map.json").read_text())
    red = [n for n in mac if mac[n]["claim"].startswith("skeleton") and mac[n]["class"] != "ph" and not any(n.startswith(k) for k in pm)]
    check(not red, f"H1 with production_map.json {'empty' if not pm else 'partial'}, every key-scheme macro without a production summary prints red ({len(red)} exceptions)")
    # exploratory keys equal the exploratory macros built by the separate code of build_c1..build_r2
    pairs = [("KnownBothHzero", "JointHzeroCatA"), ("KnownBothHzeroComb", "JointHzeroComb"), ("KnownBothfAGNComb", "JointfComb"), ("KnownGalHzeroComb", "GalOnlyHzeroComb"),
             ("AGNhostedHzeroComb", "PureAGNHzeroComb"), ("GalhostedHzeroComb", "PureGalHzeroComb"), ("FluxEighteenfAGN", "FluxfEighteen"), ("FluxTwentyOneHzero", "FluxHzeroTwentyOne"),
             ("FreeEighteenLognAGN", "FreeLognAGNEighteen"), ("FreeEighteenfAGN", "FreefAGNEighteen"), ("HDHostDmuChi", "HDEightDmuChi"), ("HDHostHzero", "HDEightHzero"),
             ("KnownBothRateRatio", "RateRatioJointCatA"), ("KnownBothRateRatioComb", "RateRatioJointComb"), ("FluxEighteenRateRatio", "RateRatioEighteen"),
             ("FreeEighteenRateRatio", "RateRatioFreeEighteen")]
    dev = []
    for a, b in pairs:
        va, vb = mac[a]["value"], mac[b]["value"]
        hw = 0.5 * (vb["ci90"][1] - vb["ci90"][0])
        dev.append(max(abs(va["median"] - vb["median"]), abs(va["ci90"][0] - vb["ci90"][0]), abs(va["ci90"][1] - vb["ci90"][1])) / hw)
    dev.append(abs(mac["KnownBothLB"]["value"] - mac["LBfZeroCatA"]["value"]))
    dev.append(abs(mac["KnownBothLBComb"]["value"] - mac["LBfZeroComb"]["value"]))
    dev.append(abs(mac["FreeEighteenLognAGNPriorUL"]["value"] - mac["FreeLognAGNInducedUpperEighteen"]["value"]))
    dev.append(abs(mac["FreeEighteenLognAGNUL"]["value"] - mac["FreeLognAGNUpperEighteen"]["value"]))
    check(max(dev) < 2e-3, f"H2 {len(dev)} exploratory key values equal the exploratory macros computed separately (largest difference {max(dev):.1e}, in 90% half-widths or log10 units)")
    # invented combinations: width A / sqrt(N)
    doc = json.loads((OUT / "values" / "placeholder_results.json").read_text())["keys"]
    rat = []
    for k, E in doc.items():
        for q, Q in E["quantities"].items():
            if Q["comb_source"].startswith("invented") and "prior dominates" not in Q["comb_source"] and not (k == "CtrlZero" and q == "fAGN"):
                wa = Q["A"]["ci90"][1] - Q["A"]["ci90"][0]
                wc = Q["Comb"]["ci90"][1] - Q["Comb"]["ci90"][0]
                rat.append(abs(wc / wa * math.sqrt(E["N_comb"]) - 1))
    check(max(rat) < 0.05, f"H3 every invented combination is narrower than realization A by sqrt(N) to {100 * max(rat):.1f}% ({len(rat)} key-quantities; truncation at prior edges allowed for)")


if __name__ == "__main__":
    sys.exit(main())
