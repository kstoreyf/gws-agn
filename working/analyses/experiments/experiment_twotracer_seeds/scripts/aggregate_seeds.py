#!/usr/bin/env python3
"""Catalog-realisation error bars for the deep two-tracer mock.

The headline two-tracer numbers -- f_AGN = 0.2353 (1.8 sigma low) and
H0 = 66.40 (2.7 sigma low) -- come from ONE draw.  Their quoted intervals are
conditional on that catalog, and on the K=1 side the realisation term was
measured at 2.2x the quoted width.  This aggregates the per-seed reruns and
reports, for f_AGN and H0:

  * the mean offset from truth and its standard error over realisations -- the
    statistic that actually says whether the offset is real;
  * the seed-to-seed scatter against the mean per-seed 68% half-width, i.e. the
    factor by which single-realisation intervals understate the uncertainty.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
TRUTH_F, TRUTH_H0 = 0.30, 67.74


def stats(x):
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return None
    sd = float(x.std(ddof=1)) if n > 1 else float("nan")
    sem = sd / np.sqrt(n) if n > 1 else float("nan")
    return {"n": int(n), "mean": float(x.mean()), "sd": sd, "sem": sem,
            "sigma_from_zero": (abs(float(x.mean())) / sem) if sem and sem > 0 else None}


def main():
    rows = []
    for jp in sorted(RESULTS.glob("fscan_s*.json")):
        seed = jp.stem.split("_s")[1]
        d = json.loads(jp.read_text())
        fb = d["f"]
        rec = {"seed": seed,
               "fscan_f": fb["median"],
               "fscan_f_hw": 0.5 * (fb["ci68"][1] - fb["ci68"][0]),
               "fscan_rejected": d["n_neginf_cells"]}
        jj = RESULTS / f"joint_s{seed}.json"
        if jj.exists():
            e = json.loads(jj.read_text())
            rec.update({
                "joint_H0": e["H0"]["median"],
                "joint_H0_hw": 0.5 * (e["H0"]["ci68"][1] - e["H0"]["ci68"][0]),
                "joint_f": e["f"]["median"],
                "joint_f_hw": 0.5 * (e["f"]["ci68"][1] - e["f"]["ci68"][0]),
                "joint_rho": e.get("rho"),
                "joint_rejected": e["n_neginf_cells"]})
        gp = RESULTS / f"guard_s{seed}.json"
        if gp.exists():
            g = json.loads(gp.read_text())["guard_records"][0]
            rec["Neff"] = g["Neff"]
            rec["passes_guard"] = g["passes_legacy_floor"]
        rows.append(rec)

    if not rows:
        raise SystemExit("no per-seed results found")

    out = {"truth": {"f_AGN": TRUTH_F, "H0": TRUTH_H0}, "per_seed": rows}
    for key, truth, label in (("fscan_f", TRUTH_F, "f_AGN (f-scan, H0 fixed)"),
                              ("joint_f", TRUTH_F, "f_AGN (joint)"),
                              ("joint_H0", TRUTH_H0, "H0 (joint)")):
        vals = [r[key] for r in rows if key in r]
        if not vals:
            continue
        hws = [r[key.replace("_f", "_f_hw").replace("_H0", "_H0_hw")]
               for r in rows if key in r]
        st = stats([v - truth for v in vals])
        st["label"] = label
        st["mean_quoted_half_width"] = float(np.mean(hws))
        st["scatter_over_quoted_half_width"] = (
            st["sd"] / st["mean_quoted_half_width"]
            if st["mean_quoted_half_width"] > 0 else None)
        out[key] = st

    (RESULTS / "seeds_summary.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {RESULTS / 'seeds_summary.json'}  ({len(rows)} realisations)\n")
    for key in ("fscan_f", "joint_f", "joint_H0"):
        st = out.get(key)
        if not st:
            continue
        print(f"{st['label']}")
        print(f"   offset {st['mean']:+.4f} +- {st['sem']:.4f} (sd {st['sd']:.4f}, "
              f"n={st['n']})  => {st['sigma_from_zero']:.1f} sigma from zero"
              if st["sigma_from_zero"] else "")
        print(f"   seed scatter / mean quoted half-width = "
              f"{st['scatter_over_quoted_half_width']:.2f}"
              f"   (quoted hw {st['mean_quoted_half_width']:.4f})\n")


if __name__ == "__main__":
    main()
