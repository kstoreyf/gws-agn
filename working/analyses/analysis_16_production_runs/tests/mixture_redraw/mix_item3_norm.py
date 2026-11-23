#!/usr/bin/env python
"""Item 3 check of the per-merger constant: core divides each merger's samples by p_pe normalised over its own samples,
so ln L_i = ln sum_j pi_j / (p_PE,j r_j) + ln sum_k p_PE,k r_k + const; the second term differs between the production
(r = 1) and the mixture sample sets but not with Lambda. c_i = ln mean_k(p_PE r)_on - ln mean_k(p_PE)_off from the raw
PE-prior density (generator's p_pe_v3); then mean over non-edge mergers of d_i - c_i at the peak node.
    python mix_item3_norm.py LINE_STEM SCAN_JSON REALIZATION RUN"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prod_common as P
import redraw_mixture as RM
P.ensure_paths()
import mc_redraw as M
stem, scan, real, run = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
R = HERE / "results"
z = np.load(R / f"{stem}.npz"); lj = json.loads((R / f"{stem}.json").read_text())
H, T, ev = z["H0"], z["T"], z["ev"]; nT = T.size
jpk = int(np.argmin(np.abs(H - lj["per_merger_on_minus_off_at_peak"]["H0"])))
cfg = json.loads((P.RUNS / run / "config.json").read_text()); cfg["redraw_mixture"] = "support"
ns = int(cfg["nsamp"]) if not cfg.get("agn_subset") else int(cfg["agn_subset"]["nsamp"])
obs, attrs, nobs, _ = M.read_event_set(cfg["events_file"])
g = M.generator()
bl = []
for b in range(ns // 2000):
    p = g.posterior_samples_v3(np.random.default_rng(np.random.SeedSequence(list(cfg["redraw_key"]) + [b])), obs, 2000)
    bl.append({k: v.reshape(nobs, 2000) for k, v in p.items()})
raw = {k: np.concatenate([x[k] for x in bl], axis=1) for k in bl[0]}
on, r, rec = RM.mixture_redraw(cfg, g, obs, np.arange(nobs), raw, ns)
def ppe(d, rows): return g.p_pe_v3(d["m1det"][rows], d["q"][rows], d["dL"][rows], mc_det=d["mc_det"][rows])
c = np.log((ppe(on, T) * r[T]).mean(1)) - np.log(ppe(raw, T)[:, :ns].mean(1))
pa = json.loads((R / scan).read_text())["realizations"][str(real)]["per_merger"]
ne = np.array([pa[str(int(i))]["P_A"] >= 0.5 for i in T])
d = ev[jpk, nT:2 * nT] - ev[jpk, :nT]
x = (d - c)[ne]
out = {"line": stem, "n_non_edge": int(ne.sum()), "H0": float(H[jpk]),
       "d_mean": float(d[ne].mean()), "d_sem": float(d[ne].std(ddof=1) / np.sqrt(ne.sum())),
       "c_mean": float(c[ne].mean()), "d_minus_c_mean": float(x.mean()), "d_minus_c_sem": float(x.std(ddof=1) / np.sqrt(x.size)),
       "d_minus_c_rms": float(np.sqrt((x ** 2).mean()))}
(R / f"item3norm_{stem}.json").write_text(json.dumps(out, indent=1))
print(out)
