#!/usr/bin/env python3
"""ENDGAME -- collect the (A - B) / (C - A) split and the truncation test.

Reads
  results/abc_{tracer}_s{seed}.{json,npz}          per-realisation A, B_inj, C
  results/attr_selmu_{tracer}[_s{seed}].json       the EXACT oracle B (kde arm)
  results/abc_{tracer}_mega.npz                    A on the untruncated replay
  <scratch>/events_notrunc_full_s100.h5            the seed-100 batch structure

and writes results/abc_summary.json.

B convention.  B == d ln mu/dH0 EXACTLY, and the oracle's ``kde`` arm is the
measure darksirens actually conditions on, so ``B_exact = dlnmu_at_truth["kde"]``
of that seed's own catalog.  Using it in place of the injection estimate removes
the +-1.2e-4 (GAL) common-mode Monte-Carlo error CLOSURE.md 14.2 identified.

Pooling.  B is exact and seed-specific, so the five realisations pool at the EVENT
level: the pooled statistic is the mean over all events of (varsigma_i - B_{seed(i)})
with the ordinary sem of that per-event array.  The seed-level pooling (mean of the
five per-seed values, sem over 5) is reported beside it as a consistency check.
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
SCRATCH = Path("/hildafs/projects/phy220048p/magana/gws-agn-data/scratch_truncation_test")
KEYS = ("pop", "rate", "mass", "pz", "jac", "tot")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102, 103, 105])
    ap.add_argument("--tracers", nargs="+", default=["gal", "agn"])
    ap.add_argument("--out", default=str(RES / "abc_summary.json"))
    return ap.parse_args(argv)


def sem(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else float("nan")


def b_exact(tracer, seed):
    p = RES / (f"attr_selmu_{tracer}.json" if seed == 100
               else f"attr_selmu_{tracer}_s{seed}.json")
    if not p.exists():
        return None, str(p)
    d = json.loads(p.read_text())
    return d["dlnmu_at_truth"], str(p)


def main(argv=None):
    args = parse_args(argv)
    out = {"name": "abc_summary", "seeds": args.seeds,
           "B_convention": "exact oracle d ln mu/dH0, kde arm (the measure "
                           "darksirens conditions on)"}

    for tr in args.tracers:
        per_seed, pool = {}, {"A": [], "C": [], "Bmap": [], "seed": []}
        for s in args.seeds:
            jp = RES / f"abc_{tr}_s{s}.json"
            npp = RES / f"abc_{tr}_s{s}.npz"
            if not (jp.exists() and npp.exists()):
                print(f"[warn] missing {jp.name}")
                continue
            J = json.loads(jp.read_text())
            Z = np.load(npp)
            Bx, bp = b_exact(tr, s)
            if Bx is None:
                print(f"[warn] missing exact oracle for {tr} s{s} ({bp})")
                continue
            A = {k: Z[f"A_{k}"] for k in KEYS}
            C = {k: Z[f"C_{k}"] for k in KEYS}
            n = A["tot"].size
            row = {"n_events": int(n), "B_exact_kde": Bx["kde"],
                   "B_exact_delta": Bx["delta"], "B_exact_unif": Bx["unif"],
                   "B_inj_tot": J["B_inj"]["tot"],
                   "B_inj_minus_B_exact": J["B_inj"]["tot"] - Bx["kde"],
                   "anchor_log_mu_absdiff": J["anchor_log_mu_absdiff"],
                   "oracle_path": bp}
            for k in KEYS:
                a, c = A[k], C[k]
                row[k] = {
                    "A": float(a.mean()), "A_sem": sem(a),
                    "C": float(c.mean()), "C_sem": sem(c),
                    "C_minus_A": float((c - a).mean()), "C_minus_A_sem": sem(c - a),
                }
            # only the TOTAL has an exact B
            row["tot"]["B_exact"] = Bx["kde"]
            row["tot"]["A_minus_B"] = float(A["tot"].mean() - Bx["kde"])
            row["tot"]["A_minus_B_sem"] = sem(A["tot"])
            row["tot"]["r"] = float(C["tot"].mean() - Bx["kde"])
            row["tot"]["r_sem"] = sem(C["tot"])
            row["tot"]["r_with_B_inj"] = float(C["tot"].mean() - J["B_inj"]["tot"])
            per_seed[str(s)] = row
            pool["A"].append(A["tot"] - Bx["kde"])
            pool["C"].append(C["tot"] - Bx["kde"])
            pool["seed"].append(np.full(n, s))
        if not per_seed:
            continue
        aB = np.concatenate(pool["A"])          # per-event (varsigma_i - B)
        cB = np.concatenate(pool["C"])          # per-event (dlnZ_i - B)
        cA = cB - aB                            # paired (C - A)
        seedv = np.concatenate(pool["seed"])
        ss = sorted(per_seed)
        pooled = {
            "n_events": int(aB.size),
            "A_minus_B": float(aB.mean()), "A_minus_B_sem": sem(aB),
            "C_minus_A": float(cA.mean()), "C_minus_A_sem": sem(cA),
            "r": float(cB.mean()), "r_sem": sem(cB),
            "identity_residual": float(cB.mean() - (aB.mean() + cA.mean())),
            "by_seed_mean": {
                "A_minus_B": float(np.mean([per_seed[k]["tot"]["A_minus_B"] for k in ss])),
                "A_minus_B_sem": float(np.std([per_seed[k]["tot"]["A_minus_B"]
                                               for k in ss], ddof=1) / np.sqrt(len(ss))),
                "C_minus_A": float(np.mean([per_seed[k]["tot"]["C_minus_A"] for k in ss])),
                "C_minus_A_sem": float(np.std([per_seed[k]["tot"]["C_minus_A"]
                                               for k in ss], ddof=1) / np.sqrt(len(ss))),
                "r": float(np.mean([per_seed[k]["tot"]["r"] for k in ss])),
                "r_sem": float(np.std([per_seed[k]["tot"]["r"] for k in ss],
                                      ddof=1) / np.sqrt(len(ss))),
            },
        }
        # per-term pooled (C - A): paired, so it is the sharp one
        pooled["by_term_C_minus_A"] = {}
        for k in KEYS:
            v = []
            for s in args.seeds:
                npp = RES / f"abc_{tr}_s{s}.npz"
                if not npp.exists():
                    continue
                Z = np.load(npp)
                v.append(Z[f"C_{k}"] - Z[f"A_{k}"])
            v = np.concatenate(v)
            pooled["by_term_C_minus_A"][k] = {"mean": float(v.mean()), "sem": sem(v)}
        out[tr] = {"per_seed": per_seed, "pooled": pooled}

        print(f"\n================ {tr.upper()} ================")
        print(f"{'seed':>6} {'n':>5} {'A':>12} {'B_exact':>12} {'A-B':>12} {'sem':>10} "
              f"{'C-A':>12} {'sem':>10} {'r=C-B':>12} {'B_inj-B_ex':>11}")
        for s in ss:
            R = per_seed[s]; t = R["tot"]
            print(f"{s:>6} {R['n_events']:>5} {t['A']:12.5e} {t['B_exact']:12.5e} "
                  f"{t['A_minus_B']:12.5e} {t['A_minus_B_sem']:10.2e} "
                  f"{t['C_minus_A']:12.5e} {t['C_minus_A_sem']:10.2e} "
                  f"{t['r']:12.5e} {R['B_inj_minus_B_exact']:11.2e}")
        p = pooled
        print(f"{'POOL':>6} {p['n_events']:>5} {'':>12} {'':>12} "
              f"{p['A_minus_B']:12.5e} {p['A_minus_B_sem']:10.2e} "
              f"{p['C_minus_A']:12.5e} {p['C_minus_A_sem']:10.2e} "
              f"{p['r']:12.5e}")
        b = p["by_seed_mean"]
        print(f"{'(seed)':>6} {5:>5} {'':>12} {'':>12} "
              f"{b['A_minus_B']:12.5e} {b['A_minus_B_sem']:10.2e} "
              f"{b['C_minus_A']:12.5e} {b['C_minus_A_sem']:10.2e} "
              f"{b['r']:12.5e} {b['r_sem']:11.2e}")

    # ------------------------------------------------ the truncation test ------
    trunc = {}
    for tr in args.tracers:
        mp = RES / f"abc_{tr}_mega.npz"
        mj = RES / f"abc_{tr}_mega.json"
        if not mp.exists():
            continue
        Z = np.load(mp)
        J = json.loads(mj.read_text()) if mj.exists() else {}
        Bx, _ = b_exact(tr, 100)
        B = Bx["kde"]
        rank = Z["Xg_rank"]; rep = Z["Xg_replica"]; batch = Z["Xg_batch"]
        S = Z["X_tot"]
        head = rank < 1000
        g = {}
        for name, m in (("head_kept", head), ("tail_withheld", ~head),
                        ("full", np.ones_like(head))):
            x = S[m]
            g[name] = {"n": int(m.sum()), "A": float(x.mean()), "A_sem": sem(x),
                       "A_minus_B": float(x.mean() - B),
                       "A_minus_B_sem": sem(x),
                       "sigma": float((x.mean() - B) / sem(x))}
        d = S[head].mean() - S[~head].mean()
        dse = np.sqrt(S[head].var(ddof=1) / head.sum()
                      + S[~head].var(ddof=1) / (~head).sum())
        # per-replica paired head/tail difference -- removes any replica-level
        # common mode, and its sem is an independent check on the unpaired one
        nrep = int(rep.max()) + 1
        dh = []
        for k in range(nrep):
            m = rep == k
            hk, tk = m & head, m & ~head
            if hk.sum() > 1 and tk.sum() > 1:
                dh.append(S[hk].mean() - S[tk].mean())
        dh = np.asarray(dh)
        trunc[tr] = {
            "B_exact_kde": B, "n_replicas": nrep,
            "source": J.get("extra_truth", {}).get("source_info", {}),
            "groups": g,
            "head_minus_tail": float(d), "head_minus_tail_sem": float(dse),
            "head_minus_tail_sigma": float(d / dse),
            "head_minus_tail_paired": float(dh.mean()),
            "head_minus_tail_paired_sem": sem(dh),
            "per_term_head": {k: {"mean": float(Z[f"X_{k}"][head].mean()),
                                  "sem": sem(Z[f"X_{k}"][head])} for k in KEYS},
            "per_term_full": {k: {"mean": float(Z[f"X_{k}"][~np.zeros_like(head)].mean()),
                                  "sem": sem(Z[f"X_{k}"])} for k in KEYS},
            "batch_split": {str(int(b)): {"n": int((batch == b).sum()),
                                          "A": float(S[batch == b].mean()),
                                          "sem": sem(S[batch == b])}
                            for b in np.unique(batch)},
        }
        print(f"\n---------------- TRUNCATION TEST, {tr.upper()} "
              f"({nrep} replays of the event stage on seed 100's catalog) ------")
        print(f"B_exact = {B:.8e}")
        print(f"{'group':>15} {'n':>9} {'A':>14} {'sem':>11} {'A-B':>13} {'sigma':>7}")
        for name in ("head_kept", "tail_withheld", "full"):
            q = g[name]
            print(f"{name:>15} {q['n']:9d} {q['A']:14.7e} {q['A_sem']:11.3e} "
                  f"{q['A_minus_B']:13.5e} {q['sigma']:7.2f}")
        print(f"head - tail = {d:+.5e} +- {dse:.3e}  ({d/dse:+.2f} sigma); "
              f"paired {dh.mean():+.5e} +- {sem(dh):.3e}")
    if trunc:
        out["truncation_test"] = trunc

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
