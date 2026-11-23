#!/usr/bin/env python
"""Hybrid line of a two-catalog run on a pure-AGN set at f_AGN = 1, with the replaced mergers' terms from the AGN-only model.

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH= <core venv python> tests/obslike_steps_hw_endpoint.py RUN_BOTH RUN_AGN [--mult 4]

At f_AGN = 1 the galaxy branch of the two-catalog mixture has weight zero, so every merger's term equals the AGN-catalog-only
term of the same samples (the hosts_known run on the same set and redraw key). The mergers to replace (effective sample number
< 50 at RUN_BOTH's peak node, RUN_BOTH's own sample number) are read from RUN_AGN's stage-1 scan at that node
(tests/results/obslike_steps/halfwidth_<slug(RUN_AGN)>_stage1.npz); the replaced mergers are re-bound in the AGN-only model at
mult x N samples and evaluated on RUN_BOTH's fine H0 nodes. The identity (per-merger terms of RUN_BOTH at f = 1 against
RUN_AGN's at the same node) is checked separately by `obslike_steps_halfwidth.py run RUN_BOTH --stage1-only` (compare its
_stage1.npz with RUN_AGN's at the shared node). Writes halfwidth_<slug(RUN_BOTH)>.npz in the layout report() reads, then report.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import obslike_steps_halfwidth as HW  # noqa: E402

S = HW.S


def main(name_both, name_agn, mult=4, ess_cut=50.0, limit_inj=5000):
    t0 = time.time()
    line = HW.stored_line(name_both)
    if line["f"] is None or abs(line["f"] - 1.0) > 1e-9:
        raise SystemExit(f"[fatal] the line of {name_both} is at f = {line['f']}, not 1")
    H = line["H0"]
    lc = line["lnL"] + 0.5 * line["pe_var"] - 0.5 * line["n_events"] / line["n_eff"] if line["hvc"] else line["lnL"]
    ipk = int(np.nanargmax(lc))
    la = HW.stored_line(name_agn)
    s1 = np.load(HW.OUT / f"halfwidth_{HW.slug(name_agn)}_stage1.npz")
    ja = [int(j) for j in s1["jc"] if abs(la["H0"][j] - H[ipk]) < 1e-9]
    if not ja:
        raise SystemExit(f"[fatal] {name_agn}'s stage 1 has no node at {H[ipk]}")
    ka = int(np.where(s1["jc"] == ja[0])[0][0])
    cfg, src, check, note = HW.eval_cfg(name_agn)
    cfg_b, _, _ = S.make_cfg(name_both)
    for k in ("events_file", "redraw_key", "nsamp", "population_values", "Om0", "zmax"):
        vb = cfg_b[k] if k != "population_values" else cfg_b["class_populations"]["agn"]
        if vb != cfg[k]:
            raise SystemExit(f"[fatal] {k} differs: {vb} vs {cfg[k]}")
    N = int(cfg["nsamp"])
    K = np.where(s1["ess_N"][ka] < ess_cut)[0]
    print(f"[endpoint] {name_both}: peak {H[ipk]:g}; {K.size} mergers with ESS < {ess_cut} there (from {name_agn}'s stage 1)", flush=True)
    os.environ["DARKSIRENS_ZMAX"] = repr(float(cfg["zmax"]))
    import prod_model as PM
    m = PM.Model(cfg)
    m.import_core()
    import jax
    jax.config.update("jax_enable_x64", True)
    m.build()
    cfg["nsamp"] = N * mult
    cfg["limit_injections"] = int(limit_inj)
    sel = np.zeros(line["n_events"], bool)
    sel[K] = True
    m.bind(select=sel)
    S.install_capture()
    hyb = HW.scan(m, H, None, N, S.EDGE[cfg["population_family"]], float(cfg["Om0"]))
    chk = float(np.max(np.abs(hyb["ev_N"][ipk] - s1["ev_N"][ka][K])))
    print(f"[endpoint] first {N} of {N * mult} vs {name_agn}'s stage-1 terms at {H[ipk]:g}: max |d ev| {chk:.2e}", flush=True)
    base = {k: s1[k][ka:ka + 1] for k in s1.files if k != "jc"}
    nan = [float("nan")]
    meta = {"run": name_both, "args_from": src, "config_check": check,
            "note": f"per-merger terms at f = 1 from the AGN-only model of {name_agn} (same set, key, samples); K from its stage 1 at the peak",
            "nsamp_run": N, "mult": mult, "ess_cut": ess_cut, "limit_inj_stage2": limit_inj, "f": line["f"], "stored_file": line["file"],
            "peak_H0": float(H[ipk]), "check_nodes": [float(H[ipk])],
            "check_minus_stored": {"lnL": nan, "pe_var": nan, "n_eff": nan, "sel_corr": nan}, "mergers": K.tolist(),
            "first_N_vs_run_ev_max_abs": chk, "first_N_vs_run_var_max_abs": None, "seconds_total": time.time() - t0}
    stem = f"halfwidth_{HW.slug(name_both)}"
    np.savez_compressed(HW.OUT / f"{stem}.npz", H0=H, jc=np.array([ipk]), K=K, meta=json.dumps(meta, default=float),
                        **{f"base_{k}": v for k, v in base.items()}, **{f"hyb_{k}": v for k, v in hyb.items()})
    print(f"[endpoint] -> {HW.OUT / stem}.npz ({time.time() - t0:.0f} s)", flush=True)
    HW.report(name_both)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("both")
    ap.add_argument("agn")
    ap.add_argument("--mult", type=int, default=4)
    a = ap.parse_args()
    main(a.both, a.agn, a.mult)
