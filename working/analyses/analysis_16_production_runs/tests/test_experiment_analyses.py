#!/usr/bin/env python
"""CPU checks of the short paper's analysis types (driver + prod_model), on the first N events of a
set and the first M detected injections (default 40 and 40,000), 8,000 redrawn samples per event:

  A  hosts_known on the all-AGN set   == known_population with the AGN catalog alone (same samples)
  B  hosts_known on the all-galaxy set == known_population with the galaxy catalog alone
  C  hosts_known on the host-dependent set: each subset's ln L equals
         sum_{i in subset} ln L_i,c - N_c ln mu_c + N_c (N_c + 3) / (2 N_eff,c)
     (core's selection term with its Monte Carlo correction, Farr 2019 eq. 11, for N_c events), with
     ln L_i,c, ln mu_c and N_eff,c from the class-c single-catalog model (class-c population) bound to
     ALL events: the subset's selection term is that of its own class, for its own number of events
  D  two_tracers_two_populations with zero offsets == two_tracers_one_population (equal populations)
  E  two_tracers_two_populations at f = 0 / f = 1 == galaxy catalog alone with the galaxy-class /
     AGN catalog alone with the AGN-class population (host-dependent set: the AGN branch carries the
     AGN-class population)
  F  mass_spectrum_alone with two populations: at f = 0 / 1 core's spectral likelihood of either class
     (checked at every model start), and with the two populations made equal it is the one-population
     spectral likelihood at any f
  G  membership: per-event identities, and at the maximum-likelihood f the probabilities average to
     f mu_agn / mu (= f when the classes' detection fractions are equal)

    <core venv python> tests/test_experiment_analyses.py [--events 40] [--injections 40000]
        -> tests/results/experiment_checks.json
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

A16 = Path(__file__).resolve().parent.parent
SCRIPTS = Path(os.environ.get("A16_SCRIPTS", A16 / "scripts"))
sys.path.insert(0, str(SCRIPTS))
import prod_common as P  # noqa: E402
import prod_driver as PD  # noqa: E402

H0S = (60.0, 67.74, 76.0)
HF = ((60.0, 0.1), (67.74, 0.3), (76.0, 0.8))


def cfg_of(argv, ne, ni):
    a = PD.parse_args(argv + ["--mode", "grid", "--realization", "100", "--limit-events", str(ne),
                              "--limit-injections", str(ni)])
    return PD.resolve(a)


def model(argv, ne, ni):
    cfg = cfg_of(argv, ne, ni)
    return PD.setup_model(cfg), cfg


def lnls(m, pts):
    return np.array([m.evaluate(p)["lnL"] for p in pts])


def rel(a, b):
    return float(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b))))


def check_c(C, ne, ni, h1):
    mh, cfg = model(["--analysis", "hosts_known", "--catalogs", "both", "--events-kind", "host_dependent"], ne, ni)
    ht = mh.host_types
    rows = {}
    ok = True
    for cls, cat in (("agn", "agn"), ("gal", "galaxy")):
        pop = cfg["class_populations"][cls]
        mf, _ = model(["--analysis", "known_population", "--catalogs", cat, "--events-kind", "host_dependent",
                       "--pop", f"mu_G={pop['mu_G']}", "--pop", f"mu_chi={pop['mu_chi']}"], ne, ni)
        mask = ht == (1 if cls == "agn" else 0)
        nc = int(mask.sum())
        d = []
        for p in h1:
            full = mf.evaluate_events(p)
            sub = mh.subs[cls].evaluate(p)
            sel = -nc * full["log_mu"] + nc * (nc + 3) / (2.0 * full["n_eff"])
            expect = float(np.sum(full["event_log_evidence"][mask]) + sel)
            d.append({"H0": p["H0"], "subset_lnL": sub["lnL"], "from_full_model": expect,
                      "diff": sub["lnL"] - expect, "subset_selection_term": sub["sel_corr"],
                      "expected_selection_term": sel, "log_mu_subset": sub["log_mu"], "log_mu_full": full["log_mu"],
                      "n_eff_subset": sub["n_eff"], "n_eff_full": full["n_eff"], "N_c": nc})
            ok &= abs(sub["lnL"] - expect) <= 1e-8 * max(1.0, abs(expect)) and sub["log_mu"] == full["log_mu"]
        rows[cls] = d
    tot = [mh.evaluate(p) for p in h1]
    sums = [abs(t["lnL"] - t["lnL_agn"] - t["lnL_gal"]) for t in tot]
    C["C_subset_selection"] = {"rows": rows, "composite_equals_sum": max(sums), "pass": bool(ok and max(sums) < 1e-9),
                               "mu_agn_over_mu_gal_at_H0": [float(np.exp(t["log_mu_agn"] - t["log_mu_gal"])) for t in tot]}
    print("C", json.dumps(C["C_subset_selection"])[:900], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=40)
    ap.add_argument("--injections", type=int, default=40000)
    ap.add_argument("--out", default=str(A16 / "tests" / "results" / "experiment_checks.json"))
    ap.add_argument("--only", default="ABCDEFG", help="letters of the checks to run")
    a = ap.parse_args()
    ne, ni = a.events, a.injections
    res = {"events": ne, "injections": ni, "nsamp": 8000, "checks": {}}
    C = res["checks"]
    h1 = [{"H0": h} for h in H0S]

    # A, B: hosts_known on single-class sets equals the one-catalog run
    for tag, kind, cat in (("A_all_agn", "agn_hosted", "agn"), ("B_all_galaxy", "galaxy_hosted", "galaxy")):
        if tag[0] not in a.only:
            continue
        mh, _ = model(["--analysis", "hosts_known", "--catalogs", "both", "--events-kind", kind], ne, ni)
        mk, _ = model(["--analysis", "known_population", "--catalogs", cat, "--events-kind", kind], ne, ni)
        x, y = lnls(mh, h1), lnls(mk, h1)
        C[tag] = {"hosts_known": x.tolist(), "one_catalog": y.tolist(), "max_abs_diff": float(np.max(np.abs(x - y))),
                  "n_by_class": mh.info["n_events_by_class"], "pass": bool(np.max(np.abs(x - y)) <= 1e-9)}
        print(tag, C[tag], flush=True)

    # C: subset selection term on the host-dependent set
    if "C" in a.only:
        check_c(C, ne, ni, h1)

    # D: two populations with zero offsets == one population (equal-population set)
    if "D" in a.only:
        m2, _ = model(["--analysis", "two_tracers_two_populations", "--catalogs", "both",
                       "--events-kind", "equal_populations"], ne, ni)
        m1, _ = model(["--analysis", "two_tracers_one_population", "--catalogs", "both",
                       "--events-kind", "equal_populations"], ne, ni)
        pts = [{"H0": h, "f_agn": f} for h, f in HF]
        x, y = lnls(m2, pts), lnls(m1, pts)
        C["D_zero_offsets"] = {"two_populations": x.tolist(), "one_population": y.tolist(), "max_rel_diff": rel(x, y),
                               "fixed_offsets": m2.cfg["fixed_offsets"], "pass": bool(rel(x, y) <= 1e-9)}
        print("D", C["D_zero_offsets"], flush=True)

    # E: endpoints of the two-population mixture on the host-dependent set
    if "E" in a.only:
        m2, cfg2 = model(["--analysis", "two_tracers_two_populations", "--catalogs", "both",
                          "--events-kind", "host_dependent"], ne, ni)
        pa, pg = cfg2["class_populations"]["agn"], cfg2["class_populations"]["gal"]
        mg, _ = model(["--analysis", "known_population", "--catalogs", "galaxy", "--events-kind", "host_dependent"], ne, ni)
        ma, _ = model(["--analysis", "known_population", "--catalogs", "agn", "--events-kind", "host_dependent",
                       "--pop", f"mu_G={pa['mu_G']}", "--pop", f"mu_chi={pa['mu_chi']}"], ne, ni)
        x0 = lnls(m2, [{"H0": h, "f_agn": 0.0} for h in H0S])
        x1 = lnls(m2, [{"H0": h, "f_agn": 1.0} for h in H0S])
        y0, y1 = lnls(mg, h1), lnls(ma, h1)
        C["E_endpoints"] = {"fixed_offsets": cfg2["fixed_offsets"], "agn_population": {k: pa[k] for k in ("mu_G", "mu_chi")},
                            "f0_mixture": x0.tolist(), "galaxy_alone": y0.tolist(), "f1_mixture": x1.tolist(),
                            "agn_alone_agn_population": y1.tolist(), "max_rel_diff_f0": rel(x0, y0),
                            "max_rel_diff_f1": rel(x1, y1), "pass": bool(max(rel(x0, y0), rel(x1, y1)) <= 1e-9)}
        print("E", C["E_endpoints"], flush=True)

    # F: mass_spectrum_alone
    if "F" in a.only:
        ms, cfgs = model(["--analysis", "mass_spectrum_alone", "--catalogs", "none", "--events-kind", "host_dependent"],
                         ne, ni)
        fchk = ms.check_diag_equals_bound([{"H0": h} for h in H0S])
        cfg_eq = dict(cfgs)
        cfg_eq["class_populations"] = {"gal": cfgs["class_populations"]["gal"], "agn": cfgs["class_populations"]["gal"]}
        import prod_model as PM
        me = PM.SpectralMixtureModel(cfg_eq)
        me.import_core()
        me.build()
        me.bind()
        x = lnls(me, [{"H0": h, "f_agn": f} for h, f in HF])
        ms0, _ = model(["--analysis", "mass_spectrum_alone", "--catalogs", "none", "--events-kind", "equal_populations"],
                       ne, ni)
        # same events: the equal-population-equivalent check uses the host-dependent set's own events
        mk, _ = model(["--analysis", "known_population", "--catalogs", "none", "--events-kind", "host_dependent"], ne, ni)
        y = lnls(mk, [{"H0": h} for h, _ in HF])
        C["F_mass_spectrum_alone"] = {
            "free_on_host_dependent": cfgs["free"], "free_on_equal_populations": ms0.cfg["free"] if hasattr(ms0, "cfg") else None,
            "endpoints_vs_core_bound": [{k: r[k] for k in ("coords", "bound", "diag", "abs_diff", "population")} for r in fchk],
            "equal_populations_mixture": x.tolist(), "one_population_spectral": y.tolist(), "max_rel_diff_equal": rel(x, y),
            "pass": bool(rel(x, y) <= 1e-9 and all(r["abs_diff"] <= 1e-8 * max(1, abs(r["bound"])) for r in fchk))}
        print("F", C["F_mass_spectrum_alone"], flush=True)

    # G: membership identities (host-dependent set, two populations; equal set, one population)
    if "G" in a.only:
        G = {}
        for tag, argv in (("two_populations_host_dependent", ["--analysis", "two_tracers_two_populations",
                                                              "--catalogs", "both", "--events-kind", "host_dependent"]),
                          ("one_population_equal", ["--analysis", "two_tracers_one_population", "--catalogs", "both",
                                                    "--events-kind", "equal_populations"])):
            mm, cm = model(argv + ["--membership"], ne, ni)
            with tempfile.TemporaryDirectory() as td:
                out = PD.membership(mm, cm, {"H0": {"median": 67.74}, "f_agn": {"median": 0.3}}, Path(td))
            G[tag] = out
            print("G", tag, out, flush=True)
        okG = all(v["identity_event_evidence_max_abs"] < 1e-9 and v["identity_log_mu_abs"] < 1e-9 and
                  (v["identity_ml"] is None or abs(v["identity_ml"]["difference"]) < 1e-9) for v in G.values())
        G["pass"] = bool(okG)
        C["G_membership"] = G
    res["pass"] = all(v["pass"] for v in C.values())
    res["only"] = a.only
    res["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(f"PASS={res['pass']} -> {a.out}")


if __name__ == "__main__":
    main()
