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


def production_planted_ok(cfg, require=False):
    """A production run's events file states what its kind and family say: planted f_AGN, the AGN
    offsets (the family's AGN_OFFSET_INPUTS for host-dependent sets, else none), the galaxy class's
    mu_G, mu_chi and mass spectrum of the population family. Attributes the file lacks are skipped
    unless require (stage 5: every attribute must be there)."""
    t = cfg["events_truth"]
    fam = cfg["population_family"]
    pv = P.POPULATION_INPUTS[fam]
    off = P.AGN_OFFSET_INPUTS[fam] if cfg["events_kind"] == "host_dependent" else {"dmu_G": 0.0, "dmu_chi": 0.0}
    want = {"planted_f_agn": P.EVENT_KINDS[cfg["events_kind"]][2], "dmu_G_agn": off["dmu_G"],
            "dmu_chi_agn": off["dmu_chi"], "mu_G_gal": pv["mu_G"], "mu_chi_gal": pv["mu_chi"],
            "mass_spectrum": "gwtc3_pp" if fam == "obslike" else "mock"}
    got, bad = {}, {}
    for k, w in want.items():
        v = t.get(k)
        if v is None and k == "mass_spectrum" and fam != "obslike" and t.get("mu_G_gal") is not None:
            v = "mock"            # the record family's files carry no mass_spectrum attribute
        got[k] = v
        if v is None:
            if require:
                bad[k] = [None, w]
            continue
        if (abs(float(v) - w) > 1e-9) if not isinstance(w, str) else (v != w):
            bad[k] = [v, w]
    return {"ok": not bad, "want": want, "got": got, "bad": bad}


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
                planted = None
                if ev_ok and not r.get("set"):
                    planted = production_planted_ok(cfg, require=st["stage"] == 5)
                    ok &= planted["ok"]
                if st["stage"] == 5:
                    ok &= ev_ok and inj_ok           # stage 5's files all exist today
                # every sampled parameter's prior contains the value the events were drawn with
                truth = PD.truth_coords(cfg)
                outside = {p: [truth[p], cfg["priors"][p]] for p in cfg["free"]
                           if not cfg["priors"][p][0] <= truth[p] <= cfg["priors"][p][1]}
                ok &= not outside
                row = {"run": r["name"], "part": part, "stage": st["stage"], "ok": bool(ok), "free": cfg["free"],
                       "truth": truth, "truth_outside_prior": outside, "planted_check": planted,
                       "events_file": cfg["events_file"], "injections_file": cfg["injections_file"],
                       "population_family": cfg["population_family"],
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
    by_stage = {}
    for row in rows:
        b = by_stage.setdefault(str(row["stage"]), [0, 0, 0, 0])
        b[0] += 1
        b[1] += row["ok"]
        b[2] += bool(row["planted_check"] and row["planted_check"]["ok"])
        b[3] += row["events_exists"] and row["injections_ready"]
    for s, (n, nok, npl, nf) in by_stage.items():
        print(f"  stage {s}: {n} tasks, {nok} ok, {npl} with the planted values checked against the file, "
              f"{nf} with events and injections on disk")
    for row in rows:
        if str(row["stage"]) == "5":
            pc = row["planted_check"]
            print(f"  [5] {row['run']}|{row['part']} ok={row['ok']} sampler {row['sampler']} seed {row['seed']} "
                  f"redraw {row['redraw_key']}\n        events {Path(row['events_file']).name}  injections "
                  f"{'/'.join(Path(row['injections_file']).parts[-3:])}  family {row['population_family']}\n"
                  f"        planted {pc['got'] if pc else None}\n        inputs (inside priors) {row['truth']}")
    for k, v in sorted(dens.items()):
        print(f"  {k:16s} z_c {v['z_complete']:.4f}  N {v['n_below_z_complete']:>9d}  log10 n [h^3 Mpc^-3] "
              f"{v['log10n_h_scaled']:.4f}  n(67.74) {v['n_Mpc-3_at_H0_67.74']:.4e}")
    for b in bad[:40]:
        print("BAD", b)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
