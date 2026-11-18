#!/usr/bin/env python
"""Test (c) report: dynesty settings for the eight-parameter model (CPU, seconds).

    JAX_PLATFORMS=cpu <core venv python> tests/sampler_report.py [--root tests/sampler_runs/...] [--out JSON]
    -> tests/results/sampler_test.json (default)

From the dynesty states of Phase A (unif, enlarge 1.25, bootstrap off, from scratch) and of the
Phase B branches (100 iterations each from Phase A's last state copy):
- calls per iteration (Phase A in blocks of 250 iterations; each branch over its own iterations);
- the insertion-index test (Fowlie, Handley & Su 2020), as corrected on 18 Nov (tests/nested_diagnostics.py):
  the rank of each new point among the alive points strictly above L* (the points tied at L*, here the
  driver's floor -1e300 for core's -inf, are outside the constrained prior and no longer counted), only
  the first niter dead points of a state (a converged state carries the final live points twice), u =
  (rank + V)/(n_ref + 1); KS p-value per block / branch. The first version (18 Nov, 05:27 UTC) counted the
  floor points (Phase A's first block: 198 of 200 initial points on the floor, mean rank 0.71) and the
  duplicated final live points (Phase A's last two blocks, 0.60 and 0.79);
- the geometry of the run (nested_diagnostics.geometry): the log-volume of the bound in use against ln X,
  and the flatness of the live set (smallest over largest singular value in the unit cube). Test c's
  Phase A built its first bound at 400 calls from the 3 live points above the floor: a degenerate
  ellipsoid (ln V = -79.7) that confined every later point to a plane; the test is then void;
- ln L* reached and ln Z gained over the same 100 iterations from the same state; the highest ln L of
  each branch's live points; expected scatter of ln L* from the shrinkage alone,
  sd(ln L*) ~ |d lnL/d lnX| sqrt(K)/nlive, with the slope from Phase A's last 200 iterations;
- projected calls for the whole run: N_it = 4645 iterations (Analysis 13's 3898 + 747 for the wider
  production priors), rslice and rwalk at their measured rate throughout, unif from Phase A's actual
  calls to the branch point plus the branch rate after it (an estimate, not a bound; void when Phase A
  is flat); GPU hours at 0.44 s/call (8,000 samples).
Recommendation rule: none when Phase A's geometry is flagged (flat live set or bound below the contour
volume); else the cheapest setting whose insertion-index p-values are >= 0.01 (Phase A blocks and its
branch) and whose ln L* after 100 iterations is within 3 sd of rslice's; otherwise rslice.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

A16 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nested_diagnostics as N  # noqa: E402

ROOT = A16 / "tests" / "sampler_runs" / "host_dependent_populations/both_catalogs/complete/host_dependent/realization_100"
N_IT_PROJECTED = 4645
SPC_PROD = 0.44
BRANCHES = ("phaseB_rslice", "phaseB_rwalk", "phaseB_unif_b0", "phaseB_unif_boot5")


def load(d):
    p = Path(d) / "state.save"
    if not p.exists():
        return None
    return N.load_run(p)


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
    ap.add_argument("--out", default=str(A16 / "tests/results/sampler_test.json"))
    a = ap.parse_args()
    root = Path(a.root)
    out_path = Path(a.out)
    A = load(root / "nested_1_phaseA_unif_b0")
    out = {"root": str(root), "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "diagnostic": "tests/nested_diagnostics.py (corrected insertion ranks, 18 Nov)"}
    if A is None:
        out["error"] = "no Phase A state"
        out_path.write_text(json.dumps(out, indent=1))
        print(out)
        return
    nl = A["nlive"]
    uA, nrefA, platA, _ = N.insertion_ranks(A, seed=0)
    old = N.insertion_ranks_old(A)
    blocks = []
    for a0 in range(0, A["niter"], 250):
        b0 = min(a0 + 250, A["niter"])
        sl = slice(a0, b0)
        blocks.append({"iterations": [a0 + 1, b0], "calls_per_iteration": float(A["dead_nc"][sl].mean()),
                       "cum_calls_end": float(A["dead_nc"][:b0].sum() + nl), "logl_end": float(A["dead_logl"][b0 - 1]),
                       "n_plateau_iterations": int(platA[sl].sum()), **N.ks_summary(uA[sl]),
                       "mean_rank_first_version_without_duplicates": float(np.nanmean(old[sl]))})
    tail = slice(max(0, A["niter"] - 200), A["niter"])
    slope = float(np.polyfit(A["dead_logvol"][tail], A["dead_logl"][tail], 1)[0]) if A["niter"] > 50 else float("nan")
    geoA = N.geometry(A, block=250)
    insA = N.insertion_test(A, block=250)
    out["phase_A"] = {"setting": A["settings"], "niter": A["niter"], "ncall": A["ncall"],
                      "state_carried_final_live_points": A["added_live_in_state"],
                      "seconds_per_call": sessions_spc(root / "nested_1_phaseA_unif_b0"),
                      "n_initial_at_floor": insA["n_initial_at_floor"], "n_plateau_iterations": insA["n_plateau_iterations"],
                      "insertion_all": insA["whole"], "insertion_after_plateau": insA["after_plateau"], "blocks": blocks,
                      "dlogl_dlnX_last200": slope, "geometry": geoA, "flags": N.verdict(insA, geoA, "unif"),
                      "max_live_logl": float(np.max(A["live_logl"]))}
    br = {}
    for b in BRANCHES:
        S = load(root / f"nested_1_{b}")
        if S is None:
            br[b] = {"available": False}
            continue
        init = json.loads((root / f"nested_1_{b}" / "init_from.json").read_text())
        it0, nc0 = init["it"] - 1, init["ncall"]
        k = S["niter"] - it0
        u, _, _, _ = N.insertion_ranks(S, seed=it0)
        geo = N.geometry(S, block=100)
        br[b] = {"available": True, "setting": S["settings"],
                 "switch": init.get("switch"), "start_it": it0, "iterations_done": int(k),
                 "calls": int(S["ncall"] - nc0), "calls_per_iteration": float((S["ncall"] - nc0) / max(k, 1)),
                 "logl_star_end": float(S["dead_logl"][-1]), "max_live_logl": float(np.max(S["live_logl"])),
                 "dlogz": float(S["dead_logz"][-1] - S["dead_logz"][it0 - 1]),
                 "insertion": N.ks_summary(u[it0:]),
                 "flatness_end": geo["flatness"][-1].get("sv_min_over_max"),
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
    pre = float(A["dead_nc"][:itS].sum() + nl)
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
    if out["phase_A"]["flags"] and any(("subspace" in f) or ("bound volume" in f) for f in out["phase_A"]["flags"]):
        out["recommendation"] = {"setting": None, "reason": "test void: Phase A's live set is confined to a subspace "
                                 "(its first bound was built from the few live points above the floor), so every branch "
                                 "starts from a state that is not a sample of the eight-dimensional contour; the unif "
                                 "projections are the cost of sampling that subspace", "phase_A_flags": out["phase_A"]["flags"]}
    else:
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
    out_path.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"phase_A": {k: out["phase_A"][k] for k in ("niter", "ncall", "n_initial_at_floor", "insertion_all",
                                                                 "insertion_after_plateau", "flags")},
                      "blocks": [(b["iterations"], round(b["calls_per_iteration"], 1), b["ks_p"], round(b["mean_u"], 3),
                                  round(b["mean_rank_first_version_without_duplicates"], 3)) for b in blocks],
                      "phase_B": {b: {k: v.get(k) for k in ("calls_per_iteration", "insertion", "max_live_logl",
                                                             "flatness_end", "z_vs_rslice")}
                                  for b, v in br.items()},
                      "recommendation": out["recommendation"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
