#!/usr/bin/env python
"""Test (c) report: dynesty settings for the eight-parameter model (CPU, seconds).

    JAX_PLATFORMS=cpu <core venv python> tests/sampler_report.py [--root tests/sampler_runs/...]
    -> tests/results/sampler_test.json

From the dynesty states of Phase A (unif, enlarge 1.25, bootstrap off, from scratch) and of the
Phase B branches (100 iterations each from Phase A's last state copy):
- calls per iteration (Phase A in blocks of 250 iterations; each branch over its own iterations);
- the insertion-index test (Fowlie, Handley & Su 2020): the rank of each new point's ln L among the
  live points it joins is uniform on 0..nlive-1 when the proposal samples the constrained prior
  correctly; a bound that misses part of the contour pushes the ranks up. KS p-value per block /
  branch (ranks randomised within their unit bin);
- ln L* reached and ln Z gained over the same 100 iterations from the same state; their expected
  scatter from the shrinkage alone, sd(ln L*) ~ |d lnL/d lnX| sqrt(K)/nlive, with the slope from
  Phase A's last 200 iterations;
- projected calls for the whole run: N_it = 4645 iterations (Analysis 13's 3898 + 747 for the wider
  production priors), rslice and rwalk at their measured rate throughout, unif from Phase A's actual
  calls to the branch point plus the branch rate after it (rates usually keep falling or flatten after
  the first half, so this is an estimate, not a bound); GPU hours at 0.44 s/call (8,000 samples).
Recommendation rule: the cheapest setting whose insertion-index p-values are >= 0.01 (Phase A blocks
and its branch) and whose ln L* after 100 iterations is within 3 sd of rslice's; otherwise rslice.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

A16 = Path(__file__).resolve().parent.parent
ROOT = A16 / "tests" / "sampler_runs" / "host_dependent_populations/both_catalogs/complete/host_dependent/realization_100"
N_IT_PROJECTED = 4645
SPC_PROD = 0.44
BRANCHES = ("phaseB_rslice", "phaseB_rwalk", "phaseB_unif_b0", "phaseB_unif_boot5")


def load(d):
    p = Path(d) / "state.save"
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        D = pickle.load(fh)
    s = D["sampler"] if isinstance(D, dict) and "sampler" in D else D
    sr = s.saved_run
    return {"logl": np.asarray(sr["logl"], float), "nc": np.asarray(sr["nc"], float), "it": np.asarray(sr["it"], int),
            "logz": np.asarray(sr["logz"], float), "logvol": np.asarray(sr["logvol"], float),
            "live_logl": np.asarray(s.live_logl, float), "live_it": np.asarray(s.live_it, int),
            "nlive": int(s.nlive), "ncall": int(s.ncall), "niter": int(s.it - 1), "method": s.method,
            "bootstrap": int(s.bootstrap), "enlarge": float(s.enlarge)}


def insertion_ranks(st):
    """rank of the point born at iteration T (1-based) among the points alive after T's removal."""
    n_dead = len(st["logl"])
    born = np.concatenate([st["it"], st["live_it"]])
    died = np.concatenate([np.arange(1, n_dead + 1), np.full(len(st["live_logl"]), np.iinfo(np.int64).max)])
    ll = np.concatenate([st["logl"], st["live_logl"]])
    ranks = np.full(n_dead + 1, -1)
    where_born = {}
    for i, b in enumerate(born):
        if b > 0:
            where_born[int(b)] = i
    for T in range(1, n_dead + 1):
        i = where_born.get(T)
        if i is None:
            continue
        alive = (born < T) & (died > T)
        ranks[T] = int(np.sum(ll[alive] < ll[i]))
    return ranks  # index by iteration; -1 where unknown


def ks_uniform(ranks, nlive, seed=0):
    from scipy.stats import kstest
    r = np.asarray([x for x in ranks if x >= 0], float)
    if r.size < 10:
        return {"n": int(r.size), "ks_p": None, "mean_rank_over_nlive": None}
    u = (r + np.random.default_rng(seed).random(r.size)) / nlive
    return {"n": int(r.size), "ks_p": float(kstest(u, "uniform").pvalue),
            "mean_rank_over_nlive": float(np.mean(r) / nlive)}


def sessions_spc(d):
    p = Path(d) / "sessions.json"
    if not p.exists():
        return None
    rows = json.loads(p.read_text())
    c = sum(r.get("calls", 0) for r in rows)
    return sum(r.get("seconds_in_likelihood", 0.0) for r in rows) / max(c, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args()
    root = Path(a.root)
    A = load(root / "nested_1_phaseA_unif_b0")
    out = {"root": str(root), "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if A is None:
        out["error"] = "no Phase A state"
        (A16 / "tests/results/sampler_test.json").write_text(json.dumps(out, indent=1))
        print(out)
        return
    nl = A["nlive"]
    rA = insertion_ranks(A)
    blocks = []
    for a0 in range(0, A["niter"], 250):
        b0 = min(a0 + 250, A["niter"])
        sl = slice(a0, b0)
        blocks.append({"iterations": [a0 + 1, b0], "calls_per_iteration": float(A["nc"][sl].mean()),
                       "cum_calls_end": float(A["nc"][:b0].sum() + nl), "logl_end": float(A["logl"][b0 - 1]),
                       "logz_end": float(A["logz"][b0 - 1]), **ks_uniform(rA[a0 + 1:b0 + 1], nl, seed=a0)})
    tail = slice(max(0, A["niter"] - 200), A["niter"])
    slope = float(np.polyfit(A["logvol"][tail], A["logl"][tail], 1)[0]) if A["niter"] > 50 else float("nan")
    out["phase_A"] = {"setting": {"method": A["method"], "bootstrap": A["bootstrap"], "enlarge": A["enlarge"]},
                      "niter": A["niter"], "ncall": A["ncall"], "seconds_per_call": sessions_spc(root / "nested_1_phaseA_unif_b0"),
                      "insertion_all": ks_uniform(rA, nl), "blocks": blocks, "dlogl_dlnX_last200": slope}
    br = {}
    for b in BRANCHES:
        S = load(root / f"nested_1_{b}")
        if S is None:
            br[b] = {"available": False}
            continue
        init = json.loads((root / f"nested_1_{b}" / "init_from.json").read_text())
        it0, nc0 = init["it"] - 1, init["ncall"]
        k = S["niter"] - it0
        r = insertion_ranks(S)
        sl = slice(it0, S["niter"])
        br[b] = {"available": True, "setting": {"method": S["method"], "bootstrap": S["bootstrap"], "enlarge": S["enlarge"]},
                 "switch": init.get("switch"), "start_it": it0, "iterations_done": int(k),
                 "calls": int(S["ncall"] - nc0), "calls_per_iteration": float((S["ncall"] - nc0) / max(k, 1)),
                 "logl_star_end": float(S["logl"][-1]), "dlogz": float(S["logz"][-1] - S["logz"][it0 - 1]),
                 "insertion": ks_uniform(r[it0 + 1:], nl, seed=it0),
                 "seconds_per_call": sessions_spc(root / f"nested_1_{b}")}
    out["phase_B"] = br
    # expected scatter of ln L* after K iterations from the same state, and the comparison with rslice
    rs = br.get("phaseB_rslice", {})
    K = rs.get("iterations_done") or 100
    sd = abs(slope) * np.sqrt(K) / nl if np.isfinite(slope) else float("nan")
    for b, v in br.items():
        if v.get("available") and rs.get("available"):
            v["dlogl_star_vs_rslice"] = v["logl_star_end"] - rs["logl_star_end"]
            v["z_vs_rslice"] = v["dlogl_star_vs_rslice"] / (np.sqrt(2) * sd) if sd > 0 else None
    out["expected_sd_logl_star_after_K"] = sd
    # projections
    proj = {}
    itS = br.get("phaseB_rslice", {}).get("start_it") or A["niter"]
    pre = float(A["nc"][:itS].sum() + nl)
    for b, v in br.items():
        if not v.get("available"):
            continue
        rate = v["calls_per_iteration"]
        if v["setting"]["method"] == "unif":
            calls = pre + rate * max(N_IT_PROJECTED - itS, 0)
            how = f"Phase A calls to iteration {itS} + branch rate after"
        else:
            calls = nl + rate * N_IT_PROJECTED
            how = "branch rate for every iteration"
        proj[b] = {"calls": calls, "gpu_hours_at_0.44": calls * SPC_PROD / 3600, "how": how}
    out["projection"] = {"n_iterations": N_IT_PROJECTED, "per_setting": proj,
                         "a13_record": {"calls": 227836, "niter": 3898, "calls_per_iteration": 58.4}}
    # recommendation
    pa_ok = all((b["ks_p"] is None or b["ks_p"] >= 0.01) for b in blocks)
    cands = []
    for b, v in br.items():
        if not v.get("available") or b not in proj:
            continue
        ins_ok = v["insertion"]["ks_p"] is None or v["insertion"]["ks_p"] >= 0.01
        if v["setting"]["method"] == "unif" and v["setting"]["bootstrap"] == 0:
            ins_ok &= pa_ok
        z_ok = v.get("z_vs_rslice") is None or abs(v["z_vs_rslice"]) <= 3
        v["admissible"] = bool(ins_ok and z_ok)
        if v["admissible"]:
            cands.append((proj[b]["calls"], b))
    if cands and rs.get("available"):
        best = min(cands)[1]
        save = 1 - proj[best]["calls"] / proj["phaseB_rslice"]["calls"]
        out["recommendation"] = {"setting": best, "projected_calls": proj[best]["calls"],
                                 "rslice_projected_calls": proj["phaseB_rslice"]["calls"],
                                 "saving_fraction": save}
    else:
        out["recommendation"] = {"setting": "phaseB_rslice (Analysis 13)", "reason": "no admissible cheaper setting"}
    (A16 / "tests/results/sampler_test.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"phase_A": {k: out["phase_A"][k] for k in ("niter", "ncall", "insertion_all")},
                      "blocks": [(b["iterations"], round(b["calls_per_iteration"], 1), b["ks_p"]) for b in blocks],
                      "phase_B": {b: {k: v.get(k) for k in ("calls_per_iteration", "insertion", "z_vs_rslice", "dlogz")}
                                  for b, v in br.items()},
                      "recommendation": out["recommendation"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
