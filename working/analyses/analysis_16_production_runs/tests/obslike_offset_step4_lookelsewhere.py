#!/usr/bin/env python
"""Step 4 of the obslike_f0p30_s1 H0-offset diagnosis: how unusual is a five-realization combination at the level of the
input found for hosts_known on obslike_f0p30_s1, given how many comparable five-realization combinations exist?

Every (analysis, catalogs, flux, set) directory under runs/ whose realizations 100, 101, 102, 103 and 105 all have a finished
grid run (summary.json + grid/) is combined with the paper chain's function (run_posteriors.combined: corrected ln L summed);
nested runs are skipped. Per combination: median, 90% CI, cdf of the input, level |2F - 1|, and the equivalent two-sided
Gaussian z. Combinations on the same event sets are strongly correlated (different analyses of the same mergers), so the
number of independent trials is counted by distinct event sets (set label x catalogs x flux) as well.
Output: tests/results/obslike_offset_step4_lookelsewhere.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

A16 = Path(__file__).resolve().parents[1]
RUNS = A16 / "runs"
sys.path.insert(0, str(A16.parents[1] / "paper_prd" / "scripts"))
import run_posteriors as rp  # noqa: E402

REAL = [100, 101, 102, 103, 105]


def find(base):
    out = []
    for r in REAL:
        c = [d for d in sorted(list(base.glob(f"realization_{r}_*")) + list(base.glob(f"realization_{r}")))
             if (d / "summary.json").exists() and (d / "grid" / "result.json").exists()
             and (d / "config.json").exists()]
        if len(c) == 0:
            return None
        # prefer the plain directory, else the first variant (hosts_known: agn_s32000_inj1)
        plain = [d for d in c if d.name == f"realization_{r}"]
        out.append(plain[0] if plain else c[0])
    return out


def main():
    rows = {}
    for an in sorted(p for p in RUNS.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for cat in sorted(p for p in an.iterdir() if p.is_dir()):
            for flux in sorted(p for p in cat.iterdir() if p.is_dir()):
                for st in sorted(p for p in flux.iterdir() if p.is_dir()):
                    dirs = find(st)
                    if dirs is None:
                        continue
                    try:
                        if not all(rp.read_json(d / "config.json")["mode"] == "grid" for d in dirs):
                            continue
                        variants = sorted({d.name.split("_", 2)[2] if d.name.count("_") >= 2 else "" for d in dirs})
                        s, x, p, x0 = rp.combined(dirs, "H0")
                    except Exception as e:  # noqa: BLE001
                        print("skip", st.relative_to(RUNS), type(e).__name__, e)
                        continue
                    c = np.concatenate([[0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
                    c /= c[-1]
                    F = float(np.interp(x0, x, c))
                    key = str(st.relative_to(RUNS))
                    rows[key] = {"analysis": an.name, "catalogs": cat.name, "flux": flux.name, "set": st.name,
                                 "variants": variants, "median": s["median"], "ci90": s["ci90"], "cdf_at_input": F,
                                 "level_of_input": s["level_of_input"],
                                 "z_two_sided": float(norm.isf((1 - s["level_of_input"]) / 2)) if s["level_of_input"] < 1 else 99.0,
                                 "signed_z": float(norm.ppf(1 - F)) if 0 < F < 1 else (99.0 if F == 0 else -99.0)}
                    print(f"{key:75s} {s['median']:7.3f} [{s['ci90'][0]:.2f},{s['ci90'][1]:.2f}] level {s['level_of_input']:.4f} "
                          f"z {rows[key]['signed_z']:+.2f}", flush=True)
    levels = np.array([v["level_of_input"] for v in rows.values()])
    sets = sorted({(v["set"], v["catalogs"], v["flux"]) for v in rows.values()})
    hk = {k: v for k, v in rows.items() if v["analysis"] == "hosts_known"}
    summary = {"n_combinations": len(rows), "n_distinct_event_sets": len(sets),
               "n_hosts_known_combinations": len(hk),
               "n_combinations_level_ge_0p999": int((levels >= 0.999).sum()),
               "n_combinations_level_ge_0p99": int((levels >= 0.99).sum()),
               "hosts_known_levels": {k: v["level_of_input"] for k, v in hk.items()}}
    out = A16 / "tests" / "results" / "obslike_offset_step4_lookelsewhere.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
