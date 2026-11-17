#!/usr/bin/env python
"""One row per new core-population event set (2026-11-17, second batch): seeds 106-110 x the
four kinds, and the f_AGN = 0.10 control on seeds 100-105.  Any env with h5py/numpy.

    python summarize_more.py      (after more_seeds_job.sh, obslike/scripts/make_sets.sh fagn0p10
                                   and core_family.py truth -> results/more_core_logp_truth.json)

Per file: the events-only validation verdict, 1000 events, min true m2 and the largest
per-event fraction of PE samples with m2 < 5 (from the truth check), core ln p finite at
every truth (both pairing settings) and core's loader, detected events with a redrawn q,
the realised AGN share, and the number of events bit-identical (host, z, m1, q, chi_eff and
every recorded observation) to an event of the legacy file of the same kind.  For the
host-dependent sets also the class offsets of detected chi_eff and m1.
Writes results/more_summary.json.
"""
import json
from pathlib import Path

import h5py
import numpy as np

CP = Path(__file__).resolve().parent.parent
DATA = Path("/hildafs/projects/phy230054p/magana/gws-agn-data-v3")
KINDS = {"mixed": ("_corepop", ""), "pureagn": ("_pureagn_corepop", "_pureagn"),
         "puregal": ("_puregal_corepop", "_puregal"),
         "marked": ("_marked_dmu0p10_dmuG5_corepop", "_marked_dmu0p10_dmuG5")}
ROWS = [(s, k) for s in (106, 107, 108, 109, 110) for k in KINDS] + \
       [(s, "fagn0p10") for s in (100, 101, 102, 103, 105)]
TRUTH_KEYS = ("host_type", "host_index", "z", "m1src", "q", "chieff", "obs_rho", "obs_lnmc", "obs_lnq",
              "obs_chieff", "obs_dec", "obs_ra")


def truth(p):
    with h5py.File(p, "r") as f:
        t = {k: np.asarray(f["truth"][k]) for k in f["truth"].keys()}
        a = {k: f.attrs[k] for k in ("planted_f_agn", "seed_events", "nobs", "nsamp")}
    return t, a


def main():
    tj = json.loads((CP / "results" / "more_core_logp_truth.json").read_text())
    tby = {Path(r["file"]).name + "|" + Path(r["file"]).parent.parent.name: r for r in tj["files"]}
    rows = []
    for s, kind in ROWS:
        sfx, leg = KINDS.get(kind, ("_fagn0p10_corepop", None))
        p = DATA / f"seed{s}" / "events" / f"events{sfx}.h5"
        v = json.loads((DATA / f"seed{s}" / "validation" / f"validation_events{sfx}.json").read_text())
        t, a = truth(p)
        tr = tby[p.name + f"|seed{s}"]
        row = {"seed": s, "kind": kind, "file": p.name, "size_MB": p.stat().st_size / 1e6,
               "seed_events": int(a["seed_events"]), "planted_f_agn": float(a["planted_f_agn"]),
               "n_events": int(a["nobs"]), "nsamp": int(a["nsamp"]),
               "validation_checks_passed": sorted(k.split("_")[0] for k, c in v["checks"].items() if c["pass"]),
               "validation_n_failed": v["n_failed"],
               "min_true_m2": tr["min_true_m2src"], "n_true_m2_lt_5": tr["n_true_m2_lt_5"],
               "max_frac_samples_m2_lt_5": tr["max_frac_samples_m2_lt_5"],
               "core_lnp_finite_historical": tr["historical"]["all_finite"],
               "core_lnp_finite_default": tr["default"]["all_finite"],
               "core_loader_ok": tr["core_loader"]["ok"],
               "n_q_redrawn": int(t["core_q_redrawn"].sum()),
               "realised_f_agn": float(t["host_type"].mean())}
        if leg is not None and (DATA / f"seed{s}" / "events" / f"events{leg}.h5").exists():
            tl, _ = truth(DATA / f"seed{s}" / "events" / f"events{leg}.h5")
            key = lambda d, i: tuple(float(d[k][i]) for k in TRUTH_KEYS)  # noqa: E731
            legset = {key(tl, i) for i in range(tl["z"].size)}
            row["events_shared_with_legacy"] = int(sum(key(t, i) in legset for i in range(t["z"].size)))
        else:
            row["events_shared_with_legacy"] = None
        if kind == "fagn0p10":
            n, f = row["n_events"], row["planted_f_agn"]
            row["binomial_pull_agn_share"] = float((t["host_type"].sum() - n * f) / np.sqrt(n * f * (1 - f)))
        if kind == "marked":
            g, ag = t["host_type"] == 0, t["host_type"] == 1
            row["dmean_chieff_agn_minus_gal"] = [float(t["chieff"][ag].mean() - t["chieff"][g].mean()),
                                                 float(np.hypot(t["chieff"][ag].std() / np.sqrt(ag.sum()),
                                                                t["chieff"][g].std() / np.sqrt(g.sum())))]
            row["dmean_m1_agn_minus_gal"] = [float(t["m1src"][ag].mean() - t["m1src"][g].mean()),
                                             float(np.hypot(t["m1src"][ag].std() / np.sqrt(ag.sum()),
                                                            t["m1src"][g].std() / np.sqrt(g.sum())))]
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("seed", "kind", "validation_n_failed", "min_true_m2",
                                               "core_lnp_finite_historical", "events_shared_with_legacy",
                                               "realised_f_agn")}))
    ok = all(r["validation_n_failed"] == 0 and r["n_events"] == 1000 and r["n_true_m2_lt_5"] == 0
             and r["core_lnp_finite_historical"] and r["core_lnp_finite_default"] and r["core_loader_ok"]
             for r in rows)
    out = {"what": __doc__.split("\n\n")[1].strip(), "all_pass": ok, "n_files": len(rows),
           "total_MB": float(sum(r["size_MB"] for r in rows)), "rows": rows}
    (CP / "results" / "more_summary.json").write_text(json.dumps(out, indent=1))
    print("all_pass", ok, "files", len(rows), "MB", out["total_MB"])


if __name__ == "__main__":
    main()
