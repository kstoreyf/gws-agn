#!/usr/bin/env python
"""Setup cost of --redraw-mixture support (numpy only): per realization the production redraw at S samples, then the
mixture refill (redraw_mixture.mixture_redraw) with the run's stored configuration; per refilled merger P_A, raw draws,
cap; seconds of the production redraw and of the refill.

    <core venv python> tests/mixture_redraw/mix_scan.py RUN_TEMPLATE R[,R...] [--nsamp S] [--tag T]

RUN_TEMPLATE: a run path under runs/ with {R:03d} for the realization (its config.json gives events file, key, population).
-> tests/mixture_redraw/results/scan_<tag>.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prod_common as P  # noqa: E402  (the copy)
import redraw_mixture as RM  # noqa: E402

P.ensure_paths()
import mc_redraw as M  # noqa: E402


def production_raw(cfg, obs, nobs, ns):
    g = M.generator()
    blocks = []
    for b in range(ns // P.REDRAW_BLOCK):
        rng = np.random.default_rng(np.random.SeedSequence(list(cfg["redraw_key"]) + [b]))
        p = g.posterior_samples_v3(rng, obs, P.REDRAW_BLOCK)
        blocks.append({k: v.reshape(nobs, P.REDRAW_BLOCK) for k, v in p.items()})
    return {k: np.concatenate([bl[k] for bl in blocks], axis=1) for k in blocks[0]}


def n_inside(raw, edge, h0, om0):
    z = RM.z_of_dl(raw["dL"], h0, om0)
    return (raw["m1det"] / (1 + z) < edge).sum(1)


def main(template, reals, ns, tag):
    out = {"template": template, "nsamp": ns, "realizations": {}}
    for R in reals:
        name = template.format(R=R)
        cfg = json.loads((P.RUNS / name / "config.json").read_text())
        cfg["redraw_mixture"] = "support"
        obs, attrs, nobs, _ = M.read_event_set(cfg["events_file"])
        t0 = time.time()
        raw = production_raw(cfg, obs, nobs, ns)
        t_prod = time.time() - t0
        nin_off = n_inside(raw, cfg["population_values"]["m_max"], 67.74, cfg["Om0"])
        t1 = time.time()
        raw2, r, rec = RM.mixture_redraw(cfg, M.generator(), obs, np.arange(nobs), raw, ns)
        t_mix = time.time() - t1
        nin_on = n_inside(raw2, cfg["population_values"]["m_max"], 67.74, cfg["Om0"])
        same = [i for i in range(nobs) if not rec["mergers"].get(str(i), {}).get("refilled")]
        ident = all(np.array_equal(raw2[k][same], raw[k][same, :ns]) for k in raw)
        ms = {k: m for k, m in rec["mergers"].items() if m["refilled"]}
        rows = sorted(((int(i), m) for i, m in ms.items()), key=lambda x: x[1]["P_A"])
        s = rec["summary"]
        out["realizations"][str(R)] = {
            "run": name, "events_file": cfg["events_file"], "region": rec["region"], "summary": s,
            "seconds_production_redraw": t_prod, "seconds_refill": t_mix, "untouched_bitwise_identical": bool(ident),
            "n_refilled": len(ms), "n_triggered": len(rec["mergers"]), "n_P_A_lt": {str(c): int(sum(m["P_A"] < c for m in ms.values())) for c in (0.01, 0.1, 0.5, 0.9, 0.99)},
            "n_inside_67.74_lt_100": {"off": int((nin_off < 100).sum()), "on": int((nin_on < 100).sum())},
            "lowest_P_A": [{"i": i, "P_A": m["P_A"], "raw_draws": m["raw_draws"], "a": m["a"], "capped": m["capped"],
                            "production_in_A": m["production_in_A"], "n_inside_67.74_off": int(nin_off[i]),
                            "n_inside_67.74_on": int(nin_on[i]), "seconds": m["seconds"]} for i, m in rows[:12]],
            "per_merger": {str(i): {"P_A": m["P_A"], "raw_draws": m["raw_draws"], "a": m["a"], "capped": m["capped"]}
                           for i, m in rows}}
        print(f"[scan] {name}: triggered {len(rec['mergers'])}, refilled {len(ms)}, capped {s['n_capped']}, raw draws total {s.get('raw_draws_total', 0):.3g} "
              f"(max {s.get('raw_draws_max', 0):.3g}), P_A min {s.get('P_A_min', 1.0):.3g}; region active {s['active']}; production redraw {t_prod:.1f} s, refill "
              f"{t_mix:.1f} s; untouched identical {ident}; lowest P_A "
              + ", ".join(f"{x['i']}:{x['P_A']:.2e}({x['n_inside_67.74_off']}->{x['n_inside_67.74_on']})"
                          for x in out["realizations"][str(R)]["lowest_P_A"][:5]), flush=True)
    (HERE / "results").mkdir(exist_ok=True)
    p = HERE / "results" / f"scan_{tag}.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    print("wrote", p)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("template")
    ap.add_argument("reals")
    ap.add_argument("--nsamp", type=int, default=8000)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    main(a.template, [int(x) for x in a.reals.split(",")], a.nsamp, a.tag)
