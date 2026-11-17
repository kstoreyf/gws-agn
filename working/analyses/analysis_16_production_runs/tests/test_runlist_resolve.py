#!/usr/bin/env python
"""CPU check of the run lists (RUNLIST.yaml and RUNLIST_experiment.yaml): every run's arguments parse and resolve, the resolved run name is the
run list's (the launcher finds results by it), the sampled parameters and priors are the analysis
class's, every file exists (or the run is marked as waiting for data), and the number densities from
catalog counts are recorded with the redshift to which each catalog is complete.

    python tests/test_runlist_resolve.py      -> tests/results/runlist_resolve.json
"""
import functools
import json
import os
import sys
import time
from pathlib import Path

A16 = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.environ.get("A16_SCRIPTS", str(A16 / "scripts")))
import prod_common as P  # noqa: E402

P.density_from_counts = functools.lru_cache(maxsize=None)(P.density_from_counts)
import prod_driver as PD  # noqa: E402


def main():
    import yaml
    runs_dir = Path(os.environ.get("A16_RUNLIST_DIR", str(P.RUNS)))
    docs = [yaml.safe_load((runs_dir / f).read_text()) for f in ("RUNLIST.yaml", "RUNLIST_experiment.yaml")
            if (runs_dir / f).exists()]
    rows, bad, dens, names = [], [], {}, {}
    awaiting = []
    for doc in docs:
      for st in doc["stages"]:
        for r in st["runs"]:
            if r["name"] in names:
                bad.append({"run": r["name"], "error": "duplicate run name"})
            names[r["name"]] = 1
            if not r.get("args"):
                awaiting.append(r["name"])
                continue
            for part in r["parts"]:
                argv = list(r["args"]) + (["--nested-index", part.split("_")[1]] if part.startswith("nested") else [])
                try:
                    cfg = PD.resolve(PD.parse_args(argv))
                except SystemExit as e:
                    bad.append({"run": r["name"], "part": part, "error": str(e)})
                    continue
                # (a set not generated yet: the driver reads the class offsets from the file, so the
                # sampled parameters of mass_spectrum_alone are only known once it exists)
                ok = cfg["run_name"] == r["name"] and (cfg["free"] == r["free"] or not Path(cfg["events_file"]).exists())
                ok &= all(cfg["priors"][p] == list(P.PRIORS[p]) for p in cfg["free"])
                files = [cfg["injections_file"], *cfg["surveys"].values(),
                         *[f["path"] for f in cfg["selection_fits"].values()]]
                ev_ok = Path(cfg["events_file"]).exists()
                inj_ok = P.injections_ready(cfg["injections_file"])
                missing = [f for f in files if not Path(f).exists() and f != cfg["injections_file"]]
                ok &= cfg["injections_file"] == r.get("injections_file", cfg["injections_file"])
                if missing or ((ev_ok and inj_ok) != r["data_exists"]):
                    ok = False
                if ev_ok and r.get("set"):
                    t = cfg["events_truth"]
                    ok &= abs(t["planted_f_agn"] - r["set"]["f"]) < 1e-9
                    ok &= abs(t["dmu_G_agn"] - 5.0 * r["set"]["s"]) < 1e-9 and abs(t["dmu_chi_agn"] - 0.1 * r["set"]["s"]) < 1e-9
                row = {"run": r["name"], "part": part, "ok": bool(ok), "free": cfg["free"],
                       "injections_ready": inj_ok, "planted_f_agn": cfg["planted_f_agn"],
                       "fixed_offsets": cfg.get("fixed_offsets"), "membership": cfg.get("membership"),
                       "sampler": cfg.get("nested", {}).get("sample"), "seed": cfg.get("nested", {}).get("seed"),
                       "events_exists": ev_ok, "missing_files": missing, "redraw_key": cfg["redraw_key"]}
                for t, d in cfg["densities"].items():
                    row[f"log10n_{t}"] = d.get("log10n_used")
                    if "z_complete" in d:
                        dens[f"{cfg['realization']}|{cfg['flux']}|{t}"] = {
                            k: d[k] for k in ("z_complete", "n_below_z_complete", "log10n_h_scaled", "n_Mpc-3_at_H0_67.74")}
                rows.append(row)
                if not ok:
                    bad.append(row)
    res = {"pass": not bad, "n_tasks": len(rows), "n_bad": len(bad), "bad": bad, "awaiting_sets": awaiting,
           "densities_from_counts": dens, "rows": rows,
           "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    Path(os.environ.get("A16_RESOLVE_OUT", str(A16 / "tests" / "results" / "runlist_resolve.json"))).write_text(json.dumps(res, indent=1))
    print(f"tasks {len(rows)}, bad {len(bad)}; densities from counts for {len(dens)} (realization, limit, class)")
    for k, v in sorted(dens.items()):
        print(f"  {k:16s} z_c {v['z_complete']:.4f}  N {v['n_below_z_complete']:>9d}  log10 n [h^3 Mpc^-3] "
              f"{v['log10n_h_scaled']:.4f}  n(67.74) {v['n_Mpc-3_at_H0_67.74']:.4e}")
    for b in bad[:40]:
        print("BAD", b)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
