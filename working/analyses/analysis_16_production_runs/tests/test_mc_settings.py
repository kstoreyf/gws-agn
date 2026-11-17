#!/usr/bin/env python
"""CPU checks of the Monte Carlo settings of the all-AGN-hosted analyses (Analysis 14: 32000 samples
per merger, the realization's injection sets pooled), on the first N events and M detected injections
of each set (default 24 and 30,000):

  P  pooling: ln mu of the pool equals ln[sum_k ndraw_k mu_k / sum_k ndraw_k] of the single sets, and
     the pooled N_eff equals mu^2 / [(1/N^2) sum_j w_j^2 - mu^2/N] over the concatenated rows (core's own
     estimate), at three H0 (realization 100: its lane + Analysis 14's five sets)
  Q  hosts_known on the all-AGN set with the AGN-hosted subset at 32000 samples and the pool == the AGN
     catalog alone at 32000 samples with the pool (the stage-1 run it replaces)
  R  hosts_known on a mixed set: the galaxy-hosted subset is unchanged by the AGN subset's settings
     (= the 8000-sample run's subset), and the AGN-hosted subset equals the AGN-class model at 32000
     samples with the pool on all events, restricted to the subset, with its own selection term
  S  the 32000-sample redraw contains the 8000-sample one: the first 8000 samples of every merger are
     the 8000-sample run's (ln L of the 8000-sample run reproduced from a 32000 redraw cut to 8000)

    <core venv python> tests/test_mc_settings.py -> tests/results/mc_settings_checks.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

A16 = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.environ.get("A16_SCRIPTS", str(A16 / "scripts")))
import prod_common as P  # noqa: E402
import prod_driver as PD  # noqa: E402

H0S = (60.0, 67.74, 76.0)


def model(argv, ne, ni):
    cfg = PD.resolve(PD.parse_args(argv + ["--mode", "grid", "--realization", "100", "--limit-events", str(ne),
                                           "--limit-injections", str(ni)]))
    return PD.setup_model(cfg), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=24)
    ap.add_argument("--injections", type=int, default=30000)
    ap.add_argument("--out", default=str(A16 / "tests" / "results" / "mc_settings_checks.json"))
    ap.add_argument("--only", default="PQRS")
    a = ap.parse_args()
    ne, ni = a.events, a.injections
    pool = P.extra_injection_lanes(100)
    pool_args = [x for ln in pool for x in ("--pool-injections", ln)]
    sub_pool_args = [x for ln in pool for x in ("--agn-subset-pool", ln)]
    res = {"events": ne, "injections_per_set": ni, "pool": pool, "checks": {}}
    C = res["checks"]
    h1 = [{"H0": h} for h in H0S]

    # P: pooling identity
    if "P" in a.only:
        mp, cp = model(["--analysis", "known_population", "--catalogs", "agn", "--events-kind", "agn_hosted"] + pool_args,
                       ne, ni)
        singles = []
        for lane in [P.default_injection_lane()] + pool:
            ms, _ = model(["--analysis", "known_population", "--catalogs", "agn", "--events-kind", "agn_hosted",
                           "--injections-lane", lane], ne, ni)
            singles.append(ms)
        rows, ok = [], True
        for p in h1:
            rp = mp.evaluate(p)
            lm = np.array([m.evaluate(p)["log_mu"] for m in singles])
            nd = np.array([float(m.inj_store.ndraw) for m in singles])
            expect = float(np.log(np.sum(nd * np.exp(lm)) / nd.sum()))
            rows.append({"H0": p["H0"], "log_mu_pool": rp["log_mu"], "log_mu_from_sets": expect,
                         "diff": rp["log_mu"] - expect, "n_eff_pool": rp["n_eff"],
                         "n_eff_sets": [m.evaluate(p)["n_eff"] for m in singles]})
            ok &= abs(rp["log_mu"] - expect) < 1e-10
        C["P_pooling"] = {"rows": rows, "n_sets": len(singles), "ndraw_pool": int(mp.inj_store.ndraw),
                          "n_detected_pool": int(mp.inj_store.n_injections), "pass": bool(ok)}
        print("P", json.dumps(C["P_pooling"])[:700], flush=True)

    # Q: hosts_known on the all-AGN set at the AGN-only settings == AGN catalog alone at the same settings
    if "Q" in a.only:
        mh, _ = model(["--analysis", "hosts_known", "--catalogs", "both", "--events-kind", "agn_hosted",
                       "--agn-subset-nsamp", "32000"] + sub_pool_args, ne, ni)
        mk, _ = model(["--analysis", "known_population", "--catalogs", "agn", "--events-kind", "agn_hosted",
                       "--nsamp", "32000"] + pool_args, ne, ni)
        x = np.array([mh.evaluate(p)["lnL"] for p in h1])
        y = np.array([mk.evaluate(p)["lnL"] for p in h1])
        C["Q_all_agn"] = {"hosts_known": x.tolist(), "agn_catalog_alone": y.tolist(), "max_abs_diff": float(np.max(np.abs(x - y))),
                          "nsamp_agn_subset": mh.subs["agn"].nsamp, "pass": bool(np.max(np.abs(x - y)) <= 1e-9)}
        print("Q", C["Q_all_agn"], flush=True)

    # R: mixed set
    if "R" in a.only:
        m_new, cfg = model(["--analysis", "hosts_known", "--catalogs", "both", "--events-kind", "equal_populations",
                            "--agn-subset-nsamp", "32000"] + sub_pool_args, ne, ni)
        m_old, _ = model(["--analysis", "hosts_known", "--catalogs", "both", "--events-kind", "equal_populations"], ne, ni)
        mf, _ = model(["--analysis", "known_population", "--catalogs", "agn", "--events-kind", "equal_populations",
                       "--nsamp", "32000"] + pool_args, ne, ni)
        mask = m_new.host_types == 1
        nc = int(mask.sum())
        rows, ok = [], True
        for p in h1:
            a_new, a_old = m_new.evaluate(p), m_old.evaluate(p)
            full = mf.evaluate_events(p)
            expect = float(np.sum(full["event_log_evidence"][mask]) - nc * full["log_mu"]
                           + nc * (nc + 3) / (2.0 * full["n_eff"]))
            rows.append({"H0": p["H0"], "gal_subset_new": a_new["lnL_gal"], "gal_subset_8000": a_old["lnL_gal"],
                         "agn_subset": a_new["lnL_agn"], "agn_from_full_model": expect,
                         "agn_subset_8000_one_set": a_old["lnL_agn"]})
            ok &= a_new["lnL_gal"] == a_old["lnL_gal"] and abs(a_new["lnL_agn"] - expect) <= 1e-8 * max(1, abs(expect))
        C["R_mixed"] = {"rows": rows, "N_agn": nc, "nsamp": {"agn": m_new.subs["agn"].nsamp, "gal": m_new.subs["gal"].nsamp},
                        "n_injection_rows": {"agn": int(m_new.subs["agn"].inj_store.n_injections),
                                             "gal": int(m_new.subs["gal"].inj_store.n_injections)},
                        "pass": bool(ok)}
        print("R", json.dumps(C["R_mixed"])[:700], flush=True)

    # S: the 32000 redraw nests the 8000 one
    if "S" in a.only:
        # the 8000-sample run of the mixed set and a hosts_known model of the same set (same redraw key)
        m8, _ = model(["--analysis", "known_population", "--catalogs", "agn", "--events-kind", "equal_populations"],
                      ne, ni)
        hk, _ = model(["--analysis", "hosts_known", "--catalogs", "both", "--events-kind", "equal_populations"], ne, ni)
        raw = hk._redraw_all(32000)
        store = hk._subset_store(m8, raw, np.ones(ne, bool), 8000)
        import prod_model as PM
        m8b = PM.Model(m8.cfg)
        m8b.ds = m8.ds
        m8b.build()
        m8b.bind(events_store=store, inj_store=m8.inj_store)
        x = np.array([m8.evaluate(p)["lnL"] for p in h1])
        y = np.array([m8b.evaluate(p)["lnL"] for p in h1])
        C["S_nested_redraw"] = {"redraw_8000": x.tolist(), "cut_from_32000": y.tolist(),
                                "max_abs_diff": float(np.max(np.abs(x - y))), "pass": bool(np.max(np.abs(x - y)) == 0.0)}
        print("S", C["S_nested_redraw"], flush=True)
    res["pass"] = all(v["pass"] for v in C.values())
    res["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(f"PASS={res['pass']} -> {a.out}")


if __name__ == "__main__":
    main()
