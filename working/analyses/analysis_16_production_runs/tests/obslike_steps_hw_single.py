#!/usr/bin/env python
"""One merger at many samples in the AGN-only model, then the hybrid lines of the runs that replaced it at 4x with it at B x 2000.

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH= <core venv python> tests/obslike_steps_hw_single.py RUN_AGN IDX[,IDX...] BLOCKS RUN [RUN ...]

The merger's samples of blocks 0..BLOCKS-1 of the production redraw (obslike_steps_merger.redraw_one: the whole set drawn per block
with RUN_AGN's key, the merger's row kept), bound alone in RUN_AGN's model (a hosts_known run whose mergers are all AGN-hosted:
its AGN subset; any other run: its own model, RUN_AGN then naming that run) with few injections, evaluated on the fine H0 nodes
of every RUN at the RUN line's f_AGN node. For each RUN (whose halfwidth_<slug>.npz
replaced IDX at mult x N samples): the merger's first mult x N samples are checked against that npz, and the hybrid line is
recomputed with IDX at all BLOCKS x 2000 samples (the other replaced mergers as in the npz).
-> tests/results/obslike_steps_halfwidth_<slug(RUN)>_i<IDX>.json
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


def main(name_agn, idxs, blocks, runs):
    import mc_redraw as M
    t0 = time.time()
    cfg, _, _, note = HW.eval_cfg(name_agn)
    cfg["limit_injections"] = 5000
    raws = []
    for idx in idxs:                    # IDX may be a comma list: each merger drawn as one_merger draws it, bound together
        raw, attrs = S.redraw_one(cfg, idx, blocks)
        raws.append(raw)
    raw = {k: np.concatenate([r[k] for r in raws]) for k in raws[0]}
    nI = len(idxs)
    Ntot = raw["dL"].size // nI
    post = M.postprocess({k: v.copy() for k, v in raw.items()}, nI, Ntot)
    sub = {k: v for k, v in post.items() if k in M.STORE_COLUMNS}
    t_draw = time.time() - t0
    os.environ["DARKSIRENS_ZMAX"] = repr(float(cfg["zmax"]))
    import prod_model as PM
    m = PM.Model(cfg)
    m.import_core()
    import jax
    jax.config.update("jax_enable_x64", True)
    m.build()
    cfg["nsamp"] = Ntot
    store = M.load_store(m.ds, sub, attrs, nI, Ntot)
    m.bind(events_store=store)
    S.install_capture()
    print(f"[single] mergers {idxs}: {Ntot} samples drawn in {t_draw:.0f} s; bound in {time.time() - t0 - t_draw:.0f} s", flush=True)
    for name in runs:
        stem = f"halfwidth_{HW.slug(name)}"
        z = np.load(HW.OUT / f"{stem}.npz")
        meta = json.loads(str(z["meta"]))
        H, K = z["H0"], z["K"]
        ks = [int(np.where(K == idx)[0][0]) for idx in idxs]
        n4 = meta["nsamp_run"] * meta["mult"]
        line = HW.stored_line(name)
        rec = HW.scan(m, H, line["f"], n4, S.EDGE[cfg["population_family"]], float(cfg["Om0"]))
        chk = float(np.max(np.abs(rec["ev_N"] - z["hyb_ev_M"][:, ks])))
        n = line["n_events"]
        d_ev = (z["hyb_ev_M"] - z["hyb_ev_N"]).sum(1)
        d_var = (z["hyb_var_M"] - z["hyb_var_N"]).sum(1)
        d_ev2 = d_ev + (rec["ev_M"] - z["hyb_ev_M"][:, ks]).sum(1)
        d_var2 = d_var + (rec["var_M"] - z["hyb_var_M"][:, ks]).sum(1)

        def corr(lnl, pv):
            return lnl + 0.5 * pv - 0.5 * n / line["n_eff"] if line["hvc"] else lnl

        s0 = HW.line_summary(H, corr(line["lnL"], line["pe_var"]))
        s4 = HW.line_summary(H, corr(line["lnL"] + d_ev, line["pe_var"] + d_var))
        sB = HW.line_summary(H, corr(line["lnL"] + d_ev2, line["pe_var"] + d_var2))
        ipk = int(np.argmin(np.abs(H - meta["peak_H0"])))
        tagI = "-".join(str(i) for i in idxs)
        out = {"run": name, "mergers": list(idxs), "samples": Ntot, "note": note, "check_first_4x_vs_npz_max_abs": chk,
               "ess_at_peak": {str(i): {"N": float(z["hyb_ess_N"][ipk, k]), "4x": float(rec["ess_N"][ipk, c]), "all": float(rec["ess_M"][ipk, c])}
                               for c, (i, k) in enumerate(zip(idxs, ks))},
               "stored": s0, "hybrid_4x": s4, f"hybrid_4x_with_{tagI}_at_{Ntot}": sB,
               "d_hw90_4x": s4["hw90"] - s0["hw90"], "d_hw90_with_mergers_at_all": sB["hw90"] - s0["hw90"],
               "ev_mergers": {str(i): {"first_N": z["hyb_ev_N"][:, k].tolist(), "first_4x": rec["ev_N"][:, c].tolist(), "all": rec["ev_M"][:, c].tolist()}
                              for c, (i, k) in enumerate(zip(idxs, ks))}, "H0": H.tolist()}
        p = HW.RES / f"obslike_steps_halfwidth_{HW.slug(name)}_i{tagI}.json"
        p.write_text(json.dumps(out, indent=1, default=float))
        print(f"[single] {name}: check first {n4} vs npz {chk:.1e}; ESS at peak {out['ess_at_peak']}; hw stored {s0['hw90']:.3f}, "
              f"4x {s4['hw90']:.3f}, with {tagI} at {Ntot}: {sB['hw90']:.3f} {sB['ci90']} -> {p.name}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2].split(",")], int(sys.argv[3]), sys.argv[4:])
