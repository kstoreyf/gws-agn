#!/usr/bin/env python
"""Step 1 of the obslike_f0p30_s1 H0-offset diagnosis: the combined H0 posterior of the five realizations of
each set, computed with the paper chain's own combination function (working/paper_prd/scripts/run_posteriors.py,
`combined`: corrected ln L of each grid run summed on a common mesh, cubic splines of the fine pass), and the
credible level of the input H0 = 67.74 (|2F - 1|, placeholder_results.summarize_xp).

Read-only use of the paper scripts (run with PYTHONDONTWRITEBYTECODE=1 so nothing is written there).
Output: tests/results/obslike_offset_step1_combined.json
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

ROWS = [
    ("hosts_known", "obslike_f0p30_s1"),
    ("two_tracers_two_populations", "obslike_f0p30_s1"),
    ("hosts_known", "obslike_f0p30_s0"),
    ("two_tracers_one_population", "obslike_f0p30_s0"),
    ("hosts_known", "obslike_f1p00_s0"),
    ("hosts_known", "peak_f0p30_s1"),
    ("two_tracers_one_population", "obslike_f0p30_s1"),
    ("two_tracers_two_populations", "peak_f0p30_s1"),
    ("two_tracers_one_population", "peak_f0p30_s1"),
]
REAL = [100, 101, 102, 103, 105]


def run_dirs(analysis, label):
    base = RUNS / analysis / "both_catalogs" / "complete" / label
    out = []
    for r in REAL:
        c = sorted(list(base.glob(f"realization_{r}_*")) + list(base.glob(f"realization_{r}")))
        c = [d for d in c if (d / "summary.json").exists()]
        if len(c) != 1:
            return None, f"realization {r}: {len(c)} run dirs with summary.json ({[d.name for d in c]})"
        out.append(c[0])
    return out, None


def main():
    res = {}
    for analysis, label in ROWS:
        dirs, err = run_dirs(analysis, label)
        key = f"{analysis}/{label}"
        if dirs is None:
            res[key] = {"error": err}
            print(key, "SKIP", err)
            continue
        meds = []
        for d in dirs:
            s = json.loads((d / "summary.json").read_text())
            meds.append(s["posterior"]["H0"]["median"])
        mode = rp.read_json(dirs[0] / "config.json")["mode"]
        free = rp.read_json(dirs[0] / "config.json")["free"]
        s, x, p, x0 = rp.combined(dirs, "H0")
        F = float(np.interp(x0, x, np.cumsum(p) / np.sum(p)))
        # cdf of the combined posterior at the input, and the equivalent one-sided Gaussian z
        c = np.concatenate([[0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
        c /= c[-1]
        F = float(np.interp(x0, x, c))
        mean = float(np.trapz(x * p, x))
        sd = float(np.sqrt(np.trapz((x - mean) ** 2 * p, x)))
        z_tail = float(norm.ppf(F))
        out = {"runs": [str(d.relative_to(A16)) for d in dirs], "mode": mode, "free": free,
               "per_realization_median": meds, "median": s["median"], "ci68": s["ci68"], "ci90": s["ci90"],
               "mean": mean, "sd": sd, "input": x0, "cdf_at_input": F, "level_of_input": s["level_of_input"],
               "z_equiv_from_cdf": z_tail, "z_mean_sd": (mean - x0) / sd}
        res[key] = out
        print(f"{key:55s} mode={mode} free={free} med={s['median']:.3f} ci90=[{s['ci90'][0]:.3f},{s['ci90'][1]:.3f}] "
              f"mean={mean:.3f} sd={sd:.3f} F(in)={F:.4f} level={s['level_of_input']:.4f} z={z_tail:+.2f}")
    out = A16 / "tests" / "results" / "obslike_offset_step1_combined.json"
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
