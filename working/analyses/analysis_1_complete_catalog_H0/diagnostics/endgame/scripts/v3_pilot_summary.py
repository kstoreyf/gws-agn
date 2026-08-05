#!/usr/bin/env python3
"""THE v3 CLOSURE GATE -- collect the pilot's (A - B) / (C - A) split and rule.

Reads whatever of the pilot's products exist and prints the one table the owner's
gate is defined on:

    r = <d ln Z_i/dH0> - d ln mu/dH0 = (C - A) + (A - B)

  * ``(C - A)`` is a PAIRED per-event statistic and needs no selection integral at
    all -- it is the measurement-model term, and it is the one CLOSURE.md 15
    measured at -1.274e-3 +- 0.113e-3 (11.3 sigma) in the mass channel under v2.
  * ``(A - B)`` uses B from the EXACT selection oracle where available (so the
    injection estimator's own +-1.2e-4 common-mode Monte-Carlo error is removed
    from the comparison), and from the replay campaign for A when
    ``abc_<tracer>_v3_mega*.json`` exists.

GATE (owner, 2026-08-01): both (A - B) and (C - A) must be consistent with zero at
the pilot's precision before the full regeneration proceeds.

Outputs: results/v3_pilot_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
KEYS = ("pop", "rate", "mass", "pz", "jac", "tot")

# the v2 reference values these are to be compared against (CLOSURE.md 15.3)
V2 = {"gal": {"CmA_pop": -1.2738e-3, "CmA_pop_sem": 1.13e-4,
              "AmB_tot": +5.836e-4, "AmB_tot_sem": 0.844e-4,
              "r_tot": -1.1874e-3, "r_tot_sem": 2.54e-4},
      "agn": {"CmA_pop": -1.7354e-3, "CmA_pop_sem": 1.72e-4,
              "AmB_tot": +4.942e-4, "AmB_tot_sem": 1.292e-4,
              "r_tot": +1.494e-3, "r_tot_sem": 1.57e-3}}


def _load(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--suffix", default="_v3_s100")
    ap.add_argument("--mega_suffix", default="_v3_mega")
    ap.add_argument("--out", default=str(RES / "v3_pilot_summary.json"))
    args = ap.parse_args(argv)

    doc = {"name": "v3_pilot_summary", "seed": args.seed,
           "gate": ("both (A - B) and (C - A) consistent with zero at the pilot's "
                    "precision"),
           "v2_reference": V2, "tracers": {}}
    verdict_lines = []
    for tr in ("gal", "agn"):
        abc = _load(RES / f"abc_{tr}{args.suffix}.json")
        sel = _load(RES / f"attr_selmu_{tr}{args.suffix}.json")
        mega = _load(RES / f"abc_{tr}{args.mega_suffix}.json")
        if abc is None:
            print(f"[skip] no results/abc_{tr}{args.suffix}.json")
            continue
        S = abc["split_inj"]
        d = {"anchor_log_mu_absdiff": abc.get("anchor_log_mu_absdiff"),
             "nobs": abc["nobs"], "nsamp": abc["nsamp"],
             "B_injections_fd": abc.get("B_inj_fd"),
             "Neff_injections": abc.get("Neff_inj"),
             "per_term": {}}
        for k in KEYS:
            s = S[k]
            d["per_term"][k] = {
                "A": s["A"], "A_sem": s["A_sem"], "B_inj": s["B"],
                "A_minus_B_inj": s["A_minus_B"],
                "C": s["C"], "C_sem": s["C_sem"],
                "C_minus_A": s["C_minus_A"], "C_minus_A_sem": s["C_minus_A_sem"],
                "r_inj": s["C"] - s["B"]}
        if sel is not None:
            Bex = float(sel["dlnmu_at_truth"]["kde"])
            d["B_exact_oracle"] = Bex
            d["B_inj_minus_B_exact"] = float(abc["B_inj"]["tot"] - Bex)
            d["r_exact_B"] = float(abc["C"]["tot"]["mean"] - Bex)
            d["A_minus_B_exact"] = float(abc["A"]["tot"]["mean"] - Bex)
            d["A_minus_B_exact_sem"] = float(abc["A"]["tot"]["sem"])
            d["pe_model_oracle"] = sel.get("pe_model")
        if mega is not None:
            g = mega["extra_truth"]["groups"]
            d["replay_campaign"] = {
                "n_rows": mega["extra_truth"]["n_rows"],
                "A_kept": g["head_kept"]["tot"]["mean"],
                "A_kept_sem": g["head_kept"]["tot"]["sem"],
                "A_full": g["full"]["tot"]["mean"],
                "A_full_sem": g["full"]["tot"]["sem"],
                "head_minus_tail": (g["head_kept"]["tot"]["mean"]
                                    - g["tail_withheld"]["tot"]["mean"]),
                "per_term_A_kept": {k: g["head_kept"][k]["mean"] for k in KEYS},
                "per_term_A_kept_sem": {k: g["head_kept"][k]["sem"] for k in KEYS}}
            if sel is not None:
                Bex = d["B_exact_oracle"]
                d["replay_campaign"]["A_minus_B_exact"] = g["head_kept"]["tot"]["mean"] - Bex
                d["replay_campaign"]["A_minus_B_exact_sem"] = g["head_kept"]["tot"]["sem"]
                d["replay_campaign"]["A_minus_B_exact_sigma"] = (
                    (g["head_kept"]["tot"]["mean"] - Bex)
                    / max(g["head_kept"]["tot"]["sem"], 1e-30))
                # per term against the injection estimator's split of B (as in 15.3)
                d["replay_campaign"]["A_minus_B_inj_per_term"] = {
                    k: g["head_kept"][k]["mean"] - S[k]["B"] for k in KEYS}
        doc["tracers"][tr] = d

        print(f"\n================  {tr.upper()}  (v3, seed {args.seed})  "
              f"================")
        print(f"anchor |Delta log mu| = {d['anchor_log_mu_absdiff']}")
        if sel is not None:
            print(f"B exact oracle        = {d['B_exact_oracle']:+.8e}")
            print(f"B injections          = {abc['B_inj']['tot']:+.8e}  "
                  f"(inj - exact = {d['B_inj_minus_B_exact']:+.3e})")
            print(f"r  (exact B)          = {d['r_exact_B']:+.5e}")
        print(f"{'term':>6} {'A':>12} {'B_inj':>12} {'A-B':>12} {'(sem)':>10} "
              f"{'C':>12} {'C-A':>12} {'(sem)':>10} {'sigma':>7}")
        for k in KEYS:
            t = d["per_term"][k]
            sg = t["C_minus_A"] / max(t["C_minus_A_sem"], 1e-30)
            print(f"{k:>6} {t['A']:12.4e} {t['B_inj']:12.4e} "
                  f"{t['A_minus_B_inj']:12.4e} {t['A_sem']:10.2e} "
                  f"{t['C']:12.4e} {t['C_minus_A']:12.4e} "
                  f"{t['C_minus_A_sem']:10.2e} {sg:7.2f}")
        cm = d["per_term"]["pop"]
        sg = cm["C_minus_A"] / max(cm["C_minus_A_sem"], 1e-30)
        v2 = V2[tr]
        verdict_lines.append(
            f"{tr.upper()}  (C-A)_pop = {cm['C_minus_A']:+.4e} +- "
            f"{cm['C_minus_A_sem']:.2e}  ({sg:+.2f} sigma)   "
            f"[v2: {v2['CmA_pop']:+.4e} +- {v2['CmA_pop_sem']:.2e}]")
        if mega is not None and sel is not None:
            rc = d["replay_campaign"]
            verdict_lines.append(
                f"{tr.upper()}  (A-B)_tot = {rc['A_minus_B_exact']:+.4e} +- "
                f"{rc['A_minus_B_exact_sem']:.2e}  "
                f"({rc['A_minus_B_exact_sigma']:+.2f} sigma, "
                f"{rc['n_rows']:,} replayed truths)   "
                f"[v2: {v2['AmB_tot']:+.4e} +- {v2['AmB_tot_sem']:.2e}]")
        else:
            t = d["per_term"]["tot"]
            verdict_lines.append(
                f"{tr.upper()}  (A-B)_tot = {t['A_minus_B_inj']:+.4e} +- "
                f"{t['A_sem']:.2e}  (record events only -- the replay campaign "
                f"has not been run)")

    print("\n=====================  THE GATE  =====================")
    for l in verdict_lines:
        print("  " + l)
    doc["verdict_lines"] = verdict_lines
    Path(args.out).write_text(json.dumps(doc, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
