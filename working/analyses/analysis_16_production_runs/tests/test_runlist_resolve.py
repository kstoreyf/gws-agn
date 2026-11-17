#!/usr/bin/env python
"""CPU check of the run list: every run's arguments parse and resolve, the resolved run name is the
run list's (the launcher finds results by it), the sampled parameters and priors are the analysis
class's, every file exists (or the run is marked as waiting for data), and the number densities from
catalog counts are recorded with the redshift to which each catalog is complete.

    python tests/test_runlist_resolve.py      -> tests/results/runlist_resolve.json
"""
import functools
import json
import sys
import time
from pathlib import Path

A16 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(A16 / "scripts"))
import prod_common as P  # noqa: E402

P.density_from_counts = functools.lru_cache(maxsize=None)(P.density_from_counts)
import prod_driver as PD  # noqa: E402


def main():
    import yaml
    doc = yaml.safe_load((P.RUNS / "RUNLIST.yaml").read_text())
    rows, bad, dens = [], [], {}
    for st in doc["stages"]:
        for r in st["runs"]:
            for part in r["parts"]:
                argv = list(r["args"]) + (["--nested-index", part.split("_")[1]] if part.startswith("nested") else [])
                try:
                    cfg = PD.resolve(PD.parse_args(argv))
                except SystemExit as e:
                    bad.append({"run": r["name"], "part": part, "error": str(e)})
                    continue
                ok = cfg["run_name"] == r["name"] and cfg["free"] == r["free"]
                ok &= all(cfg["priors"][p] == list(P.PRIORS[p]) for p in cfg["free"])
                files = [cfg["injections_file"], *cfg["surveys"].values(),
                         *[f["path"] for f in cfg["selection_fits"].values()]]
                missing = [f for f in files if not Path(f).exists()]
                ev_ok = Path(cfg["events_file"]).exists()
                if missing or (ev_ok != r["data_exists"]):
                    ok = False
                row = {"run": r["name"], "part": part, "ok": bool(ok), "free": cfg["free"],
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
    res = {"pass": not bad, "n_tasks": len(rows), "n_bad": len(bad), "bad": bad,
           "densities_from_counts": dens, "rows": rows,
           "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (A16 / "tests" / "results" / "runlist_resolve.json").write_text(json.dumps(res, indent=1))
    print(f"tasks {len(rows)}, bad {len(bad)}; densities from counts for {len(dens)} (realization, limit, class)")
    for k, v in sorted(dens.items()):
        print(f"  {k:16s} z_c {v['z_complete']:.4f}  N {v['n_below_z_complete']:>9d}  log10 n [h^3 Mpc^-3] "
              f"{v['log10n_h_scaled']:.4f}  n(67.74) {v['n_Mpc-3_at_H0_67.74']:.4e}")
    for b in bad[:10]:
        print("BAD", b)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
