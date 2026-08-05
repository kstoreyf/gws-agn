#!/usr/bin/env python3
"""ENDGAME -- exchangeability of the generator's proposal stream.

``stage_events`` draws fixed batches of ``ntry = 100_000`` i.i.d. proposals until
1000 detections have accumulated and then keeps ``[:1000]`` of them.  The kept set
is unbiased IF position in the stream carries no information.  This script tests
that on the 1500-replay campaign (``regen_events_notrunc.py --replicas``):

  * the accepted-index GAP sequence: is it geometric, and is it autocorrelated?
  * rank / within-batch-slot vs the event's own (z, m1src, snr_obs) -- pooled
    Spearman over every replay and every batch, so the null sem is ~1/sqrt(3000)
    of a single realisation's;
  * the SCORE itself, ``varsigma(theta_true)``, in bins of stream rank -- the only
    version of the question that feeds r directly.

Outputs results/trunc_diag.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
SCRATCH = Path("/hildafs/projects/phy220048p/magana/gws-agn-data/scratch_truncation_test")


def gap_chi2(gaps, p, n_bins=20):
    """chi^2 of the accepted-index gaps against Geometric(p) over quantile bins."""
    g = stats.geom(p)
    q = g.ppf(np.linspace(1.0 / n_bins, 1.0 - 1.0 / n_bins, n_bins - 1))
    edges = np.unique(np.concatenate([[0.5], q + 0.5, [np.inf]]))
    obs = np.histogram(gaps, bins=edges)[0].astype(float)
    cdf = g.cdf(np.floor(edges[1:] - 0.5))
    exp = np.diff(np.concatenate([[0.0], cdf])) * gaps.size
    keep = exp > 0
    chi2 = float((((obs - exp) ** 2 / exp)[keep]).sum())
    dof = int(keep.sum() - 1)
    return {"p_hat": float(p), "n_bins": int(keep.sum()), "chi2": chi2, "dof": dof,
            "chi2_per_dof": chi2 / dof, "pvalue": float(stats.chi2.sf(chi2, dof))}


def acf(x, k):
    x = np.asarray(x, float) - np.asarray(x, float).mean()
    d = np.dot(x, x)
    return float(np.dot(x[:-k], x[k:]) / d) if d > 0 else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replicas", default=str(
        SCRATCH / "events_notrunc_replicas_s100_n1500.h5"))
    ap.add_argument("--single", default=str(SCRATCH / "events_notrunc_full_s100.h5"))
    ap.add_argument("--out", default=str(RES / "trunc_diag.json"))
    args = ap.parse_args(argv)

    import h5py
    out = {"name": "trunc_diag"}

    # ------------------------------------------------ the record realisation ---
    with h5py.File(args.single, "r") as f:
        info = json.loads(f.attrs["info_json"])
        rk = f["rank"][:]; bt = f["batch"][:]; sl = f["slot"][:]
        z1 = f["z"][:]
    out["record_seed100"] = {
        "verify": info.get("verify", {}),
        "n_detected_total": int(rk.size),
        "n_batches": int(bt.max()) + 1,
        "kept_from_batch": {str(int(b)): int(((bt == b) & (rk < 1000)).sum())
                            for b in np.unique(bt)},
        "withheld_from_batch": {str(int(b)): int(((bt == b) & (rk >= 1000)).sum())
                                for b in np.unique(bt)},
        "last_kept_batch": int(bt[999]), "last_kept_slot": int(sl[999]),
    }

    # ------------------------------------------------------- the 1500 replays --
    with h5py.File(args.replicas, "r") as f:
        rep = f["replica"][:]; rk = f["rank"][:]; bt = f["batch"][:]; sl = f["slot"][:]
        z = f["z"][:]; m1 = f["m1src"][:]; snr = f["snr_obs"][:]; ht = f["host_type"][:]
        rinfo = json.loads(f.attrs["info_json"])
    nrep = int(rep.max()) + 1
    ntry = int(rinfo["ntry"])
    nd = np.array([x["n_det"] for x in rinfo["per_replica"]])
    out["replays"] = {
        "n_replicas": nrep, "ntry_per_batch": ntry,
        "n_detected_per_replay": {"mean": float(nd.mean()), "sd": float(nd.std(ddof=1)),
                                  "min": int(nd.min()), "max": int(nd.max())},
        "record_seed100_n_detected": int(out["record_seed100"]["n_detected_total"]),
        "record_pull_vs_replays": float(
            (out["record_seed100"]["n_detected_total"] - nd.mean()) / nd.std(ddof=1)),
        "n_batches_used": sorted(set(int(x["n_tried"]) // ntry
                                     for x in rinfo["per_replica"])),
    }

    # gap sequence + autocorrelation, per replay, pooled
    glob = bt.astype(np.int64) * ntry + sl
    lags = (1, 2, 3, 5, 10, 25)
    gap_ac = {k: [] for k in lags}
    z_ac = {k: [] for k in lags}
    sp_slot_z, sp_slot_snr, sp_rank_z = [], [], []
    gaps_all = []
    for k in range(nrep):
        m = rep == k
        g = np.diff(glob[m])
        gaps_all.append(g)
        for L in lags:
            gap_ac[L].append(acf(g, L))
            z_ac[L].append(acf(z[m], L))
        sp_rank_z.append(stats.spearmanr(rk[m], z[m]).statistic)
        for b in np.unique(bt[m]):
            mb = m & (bt == b)
            sp_slot_z.append(stats.spearmanr(sl[mb], z[mb]).statistic)
            sp_slot_snr.append(stats.spearmanr(sl[mb], snr[mb]).statistic)
    gaps = np.concatenate(gaps_all)
    p_hat = nd.sum() / (nd.size * 2 * ntry)

    def mstat(v):
        v = np.asarray(v, float)
        return {"mean": float(v.mean()), "sem": float(v.std(ddof=1) / np.sqrt(v.size)),
                "n": int(v.size)}

    out["stream"] = {
        "gap_mean": float(gaps.mean()), "gap_sd": float(gaps.std(ddof=1)),
        "geometric_expectation_1_over_p": float(1.0 / p_hat),
        "geometric_expectation_sd": float(np.sqrt(1.0 - p_hat) / p_hat),
        # A KS test against a DISCRETE law is not valid here: scipy's continuous
        # kstest returns exactly the largest point mass P(gap=1) = p as its
        # statistic, an artifact of the step CDF.  A chi^2 over geometric
        # quantile bins is the right test and is what is reported.
        "gap_chi2_vs_geometric": gap_chi2(gaps, p_hat),
        "gap_autocorrelation": {str(L): mstat(gap_ac[L]) for L in lags},
        "z_autocorrelation": {str(L): mstat(z_ac[L]) for L in lags},
        "spearman_slot_z_within_batch": mstat(sp_slot_z),
        "spearman_slot_snr_within_batch": mstat(sp_slot_snr),
        "spearman_rank_z_within_replay": mstat(sp_rank_z),
    }
    print("=== proposal-stream exchangeability, "
          f"{nrep} replays x {len(np.unique(bt))} batches ===")
    s = out["stream"]
    g2 = s["gap_chi2_vs_geometric"]
    print(f"gap mean {s['gap_mean']:.2f} (geometric 1/p = "
          f"{s['geometric_expectation_1_over_p']:.2f})  sd {s['gap_sd']:.2f} "
          f"(geometric {s['geometric_expectation_sd']:.2f});  "
          f"chi2/dof = {g2['chi2_per_dof']:.3f} on {g2['dof']} dof, p = "
          f"{g2['pvalue']:.3f}")
    for nm in ("gap_autocorrelation", "z_autocorrelation"):
        print(f"{nm}: " + "  ".join(
            f"L{L}={s[nm][str(L)]['mean']:+.5f}+-{s[nm][str(L)]['sem']:.5f}"
            for L in lags))
    for nm in ("spearman_slot_z_within_batch", "spearman_slot_snr_within_batch",
               "spearman_rank_z_within_replay"):
        q = s[nm]
        print(f"{nm}: {q['mean']:+.5f} +- {q['sem']:.5f}  (n={q['n']}, "
              f"{q['mean']/q['sem']:+.2f} sigma)")

    # ------------------------------------------- the score vs stream position --
    out["score_vs_rank"] = {}
    for tr, want in (("gal", 0), ("agn", 1)):
        p = RES / f"abc_{tr}_mega.npz"
        if not p.exists():
            continue
        Z = np.load(p)
        S = Z["X_tot"]; R = Z["Xg_rank"]
        edges = np.array([0, 200, 400, 600, 800, 1000, 1200, 1400, 100000])
        rows = []
        for i in range(len(edges) - 1):
            m = (R >= edges[i]) & (R < edges[i + 1])
            if m.sum() < 50:
                continue
            x = S[m]
            rows.append({"rank_lo": int(edges[i]), "rank_hi": int(edges[i + 1]),
                         "n": int(m.sum()), "A": float(x.mean()),
                         "sem": float(x.std(ddof=1) / np.sqrt(x.size))})
        # linear trend of the score against rank, on the kept range only
        keep = R < 1000
        sl_, ic_, rv, pv, se = stats.linregress(R[keep].astype(float), S[keep])
        out["score_vs_rank"][tr] = {
            "bins": rows,
            "linregress_over_kept_1000": {"slope_per_rank": float(sl_),
                                          "stderr": float(se), "pvalue": float(pv),
                                          "slope_x1000": float(sl_ * 1000.0)},
        }
        print(f"\n--- {tr}: varsigma(theta_true) in bins of stream rank ---")
        for r in rows:
            print(f"  rank [{r['rank_lo']:>5},{r['rank_hi']:>6})  n={r['n']:>8}  "
                  f"A={r['A']:.6e} +- {r['sem']:.2e}")
        q = out["score_vs_rank"][tr]["linregress_over_kept_1000"]
        print(f"  slope over the kept 1000: {q['slope_x1000']:+.3e} per 1000 ranks "
              f"(+-{se*1000:.2e}, p={q['pvalue']:.3f})")

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
