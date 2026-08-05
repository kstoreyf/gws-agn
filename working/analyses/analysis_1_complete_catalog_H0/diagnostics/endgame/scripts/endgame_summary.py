#!/usr/bin/env python3
"""ENDGAME -- one JSON for CLOSURE.md 15: the (A - B) / (C - A) split, the
truncation test, and the declared-photo-z-kernel scan.

Consumes
  results/abc_{tracer}_s{seed}.{json,npz}       A, C, B_inj per realisation
  results/attr_selmu_{tracer}[_s{seed}].json    the EXACT oracle B (kde arm)
  results/abc_{tracer}_mega[_s103].npz          A on the untruncated replays
  results/abc_{tracer}_mega_dz*.npz             A at rescaled declared kernels
  results/attr_selmu_{tracer}_dz*.json          the exact B on the same blocks
  results/trunc_diag.json                       the proposal-stream diagnostics

Writes results/endgame_summary.json and prints the three tables.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
KEYS = ("pop", "rate", "mass", "pz", "jac", "tot")
SEEDS = (100, 101, 102, 103, 105)


def sem(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(x.size))


def bex(tracer, seed=100, tag=None):
    name = (f"attr_selmu_{tracer}_{tag}.json" if tag else
            f"attr_selmu_{tracer}.json" if seed == 100 else
            f"attr_selmu_{tracer}_s{seed}.json")
    p = RES / name
    return json.loads(p.read_text())["dlnmu_at_truth"]["kde"] if p.exists() else None


def group_stats(npz, B):
    Z = np.load(npz)
    S = Z["X_tot"]; R = Z["Xg_rank"]
    out = {}
    for nm, m in (("head_kept", R < 1000), ("tail_withheld", R >= 1000),
                  ("full", np.ones_like(R, bool))):
        x = S[m]
        out[nm] = {"n": int(m.sum()), "A": float(x.mean()), "sem": sem(x),
                   "A_minus_B": float(x.mean() - B), "sigma": float((x.mean() - B) / sem(x))}
    h, t = S[R < 1000], S[R >= 1000]
    d = float(h.mean() - t.mean())
    dse = float(np.sqrt(h.var(ddof=1) / h.size + t.var(ddof=1) / t.size))
    out["head_minus_tail"] = {"value": d, "sem": dse, "sigma": d / dse}
    out["B_exact_kde"] = B
    out["per_term_head"] = {k: {"A": float(Z[f"X_{k}"][R < 1000].mean()),
                                "sem": sem(Z[f"X_{k}"][R < 1000])} for k in KEYS}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RES / "endgame_summary.json"))
    args = ap.parse_args(argv)
    out = {"name": "endgame_summary",
           "B_convention": "B == d ln mu/dH0 EXACTLY; the oracle's kde arm is the "
                           "measure darksirens conditions on, so the injection "
                           "estimator's +-1.2e-4 (GAL) common mode drops out.",
           "score_convention": "term-sum central difference, dh = 0.5, at H0 = 67.74"}

    for tr in ("gal", "agn"):
        T = {}
        # ---- the five realisations ----------------------------------------
        per, Aall, Call, Ball = {}, [], [], []
        for s in SEEDS:
            Z = np.load(RES / f"abc_{tr}_s{s}.npz")
            J = json.loads((RES / f"abc_{tr}_s{s}.json").read_text())
            B = bex(tr, s)
            A, C = Z["A_tot"], Z["C_tot"]
            per[str(s)] = {
                "n_events": int(A.size), "B_exact_kde": B,
                "B_inj_minus_B_exact": J["B_inj"]["tot"] - B,
                "anchor_log_mu_absdiff": J["anchor_log_mu_absdiff"],
                "A": float(A.mean()), "A_sem": sem(A),
                "A_minus_B": float(A.mean() - B), "A_minus_B_sem": sem(A),
                "C_minus_A": float((C - A).mean()), "C_minus_A_sem": sem(C - A),
                "r": float(C.mean() - B), "r_sem": sem(C)}
            Aall.append(A - B); Call.append(C - B)
        aB, cB = np.concatenate(Aall), np.concatenate(Call)
        T["five_realisations"] = {
            "per_seed": per,
            "pooled": {"n_events": int(aB.size),
                       "A_minus_B": float(aB.mean()), "A_minus_B_sem": sem(aB),
                       "C_minus_A": float((cB - aB).mean()), "C_minus_A_sem": sem(cB - aB),
                       "r": float(cB.mean()), "r_sem": sem(cB)},
            "by_seed_mean": {
                k: {"mean": float(np.mean([per[str(s)][k] for s in SEEDS])),
                    "sem": float(np.std([per[str(s)][k] for s in SEEDS], ddof=1)
                                 / np.sqrt(len(SEEDS)))}
                for k in ("A_minus_B", "C_minus_A", "r")}}

        # ---- per-term (A - B) [replays], (C - A) and r [five realisations] --
        Zm = np.load(RES / f"abc_{tr}_mega.npz"); Rm = Zm["Xg_rank"]; hd = Rm < 1000
        Binj = json.loads((RES / f"abc_{tr}_s100.json").read_text())["B_inj"]
        terms = {}
        for k in KEYS:
            B = bex(tr, 100) if k == "tot" else Binj[k]
            x = Zm[f"X_{k}"][hd]
            ca, rr = [], []
            for s in SEEDS:
                Z = np.load(RES / f"abc_{tr}_s{s}.npz")
                bi = json.loads((RES / f"abc_{tr}_s{s}.json").read_text())["B_inj"]
                ca.append(Z[f"C_{k}"] - Z[f"A_{k}"])
                rr.append(Z[f"C_{k}"] - (bex(tr, s) if k == "tot" else bi[k]))
            ca, rr = np.concatenate(ca), np.concatenate(rr)
            terms[k] = {
                "A_minus_B_replays": float(x.mean() - B), "A_minus_B_sem": sem(x),
                "B_used": ("exact oracle" if k == "tot" else "injection estimator"),
                "C_minus_A": float(ca.mean()), "C_minus_A_sem": sem(ca),
                "r": float(rr.mean()), "r_sem": sem(rr)}
        T["by_term"] = terms

        # ---- the truncation test, both catalogs ----------------------------
        trunc = {"seed100_1500_replays": group_stats(RES / f"abc_{tr}_mega.npz",
                                                     bex(tr, 100))}
        p103 = RES / f"abc_{tr}_mega_s103.npz"
        if p103.exists():
            trunc["seed103_500_replays"] = group_stats(p103, bex(tr, 103))
            v = [trunc[k]["head_minus_tail"] for k in trunc]
            w = np.array([1.0 / q["sem"] ** 2 for q in v])
            m = float(np.sum(w * np.array([q["value"] for q in v])) / w.sum())
            s_ = float(1.0 / np.sqrt(w.sum()))
            trunc["head_minus_tail_combined"] = {"value": m, "sem": s_,
                                                 "sigma": m / s_}
        T["truncation_test"] = trunc

        # ---- the declared-kernel scan --------------------------------------
        scan = {}
        for tag, scale in (("dzx0p5", 0.5), ("dzx2", 2.0), ("dzx3", 3.0)):
            np_ = RES / f"abc_{tr}_mega_{tag}.npz"
            if not np_.exists():
                continue
            B = bex(tr, tag=tag)
            Z = np.load(np_); S = Z["X_tot"]; R = Z["Xg_rank"]
            x = S[R < 1000]
            scan[f"x{scale:g}"] = {"dz_scale": 3.0e-3 * scale, "B_exact_kde": B,
                                   "A_head": float(x.mean()), "sem": sem(x),
                                   "A_minus_B_head": float(x.mean() - B)}
        b1 = bex(tr, 100)
        x1 = Zm["X_tot"][hd]
        scan["x1"] = {"dz_scale": 3.0e-3, "B_exact_kde": b1,
                      "A_head": float(x1.mean()), "sem": sem(x1),
                      "A_minus_B_head": float(x1.mean() - b1)}
        T["declared_kernel_scan"] = scan
        out[tr] = T

    # ---- the declared kernel against the local per-pixel redshift spacing ----
    # This is what makes the two catalogs respond to the kernel scan differently:
    # GAL's kernel already spans ~10 galaxies, AGN's spans 0.1.
    import h5py
    DATA = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data")
    z0, DZS = 0.132, 3.0e-3
    ks = {}
    for tr in ("gal", "agn"):
        sp = DATA / "seed100" / "surveys" / f"survey_{tr}_complete_ns32.h5"
        if not sp.exists():
            continue
        with h5py.File(sp, "r") as f:
            ng = f["ngals"][:]
            rows = np.sort(np.random.default_rng(0).choice(
                np.flatnonzero(ng > 0), 200, replace=False))
            dz = DZS * (1.0 + z0)
            cnt = np.array([np.sum(np.abs(f["zgals"][r, :ng[r]] - z0) < dz)
                            for r in rows], float)
        ks[tr] = {"z0": z0, "kernel_width": dz,
                  "galaxies_per_pixel_mean": float(ng[ng > 0].mean()),
                  "within_one_kernel_width_mean": float(cnt.mean()),
                  "local_spacing": float(2 * dz / max(cnt.mean(), 1e-12)),
                  "kernel_over_spacing": float(cnt.mean() / 2.0),
                  "n_pixels_sampled": int(rows.size)}
    out["kernel_vs_spacing"] = ks
    print("\n--- the declared kernel against the local per-pixel z spacing "
          f"(z = {z0}, 200 pixels) ---")
    for tr, q in ks.items():
        print(f"  {tr}: {q['galaxies_per_pixel_mean']:.0f} gal/pix, "
              f"{q['within_one_kernel_width_mean']:.2f} within +-1 kernel width, "
              f"spacing {q['local_spacing']:.2e}, kernel/spacing = "
              f"{q['kernel_over_spacing']:.2f}")

    td = RES / "trunc_diag.json"
    if td.exists():
        d = json.loads(td.read_text())
        out["proposal_stream"] = {k: d[k] for k in ("record_seed100", "replays",
                                                    "stream") if k in d}
        out["score_vs_rank"] = d.get("score_vs_rank", {})

    Path(args.out).write_text(json.dumps(out, indent=2))

    # ------------------------------------------------------------- tables ----
    for tr in ("gal", "agn"):
        T = out[tr]
        print(f"\n================================ {tr.upper()} "
              "================================")
        print("--- the split, per term.  (A-B): 1500 replays of the event stage on "
              "seed 100's\n    catalog, the KEPT 1000 (the record's truncation).  "
              "(C-A), r: five realisations. ---")
        print(f"{'term':>5} {'(A-B)':>13} {'sem':>10} {'(C-A)':>13} {'sem':>10} "
              f"{'r':>13} {'sem':>10}")
        for k in KEYS:
            q = T["by_term"][k]
            print(f"{k:>5} {q['A_minus_B_replays']:13.4e} {q['A_minus_B_sem']:10.2e} "
                  f"{q['C_minus_A']:13.4e} {q['C_minus_A_sem']:10.2e} "
                  f"{q['r']:13.4e} {q['r_sem']:10.2e}")
        f = T["five_realisations"]
        print("\n--- the same split from the five realisations ALONE (no replays) ---")
        print(f"{'seed':>6} {'n':>5} {'A-B':>13} {'sem':>10} {'C-A':>13} {'sem':>10} "
              f"{'r':>13}")
        for s in SEEDS:
            p = f["per_seed"][str(s)]
            print(f"{s:>6} {p['n_events']:>5} {p['A_minus_B']:13.4e} "
                  f"{p['A_minus_B_sem']:10.2e} {p['C_minus_A']:13.4e} "
                  f"{p['C_minus_A_sem']:10.2e} {p['r']:13.4e}")
        p = f["pooled"]
        print(f"{'POOL':>6} {p['n_events']:>5} {p['A_minus_B']:13.4e} "
              f"{p['A_minus_B_sem']:10.2e} {p['C_minus_A']:13.4e} "
              f"{p['C_minus_A_sem']:10.2e} {p['r']:13.4e} {p['r_sem']:10.2e}")

        print("\n--- the truncation test ---")
        for cat, g in T["truncation_test"].items():
            if cat == "head_minus_tail_combined":
                continue
            print(f"  {cat}   B_exact = {g['B_exact_kde']:.8e}")
            for nm in ("head_kept", "tail_withheld", "full"):
                q = g[nm]
                print(f"    {nm:>14} n={q['n']:>9}  A={q['A']:.7e}  "
                      f"A-B={q['A_minus_B']:+.4e} +- {q['sem']:.2e} "
                      f"({q['sigma']:+.2f} sigma)")
            q = g["head_minus_tail"]
            print(f"    {'head - tail':>14} {q['value']:+.4e} +- {q['sem']:.3e} "
                  f"({q['sigma']:+.2f} sigma)")
        if "head_minus_tail_combined" in T["truncation_test"]:
            q = T["truncation_test"]["head_minus_tail_combined"]
            print(f"    COMBINED head - tail  {q['value']:+.4e} +- {q['sem']:.3e} "
                  f"({q['sigma']:+.2f} sigma)")

        print("\n--- (A-B) vs the survey block's DECLARED photo-z kernel ---")
        for kk in sorted(T["declared_kernel_scan"],
                         key=lambda z: T["declared_kernel_scan"][z]["dz_scale"]):
            q = T["declared_kernel_scan"][kk]
            print(f"  dz = {q['dz_scale']:.4f} (1+z)  [{kk:>5}]  "
                  f"A-B = {q['A_minus_B_head']:+.4e} +- {q['sem']:.2e}")

    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
