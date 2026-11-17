#!/usr/bin/env python
"""Analysis 15 CPU check of the normalisation identity in darksirens-core, cell by cell.

    [DARKSIRENS_ZMAX=1.10] <core venv python> identity_check_cpu.py --rung complete --out results/identity_check_cpu_complete_grid5.json

On a subset (20 events, 20,000 injections; the identity is per sample, so it holds on any subset),
the bound core likelihood at densities n0 and f is compared with the same likelihood at the
reference densities (--ref-n0, default zero density) and f' = expit(logit f + ln(F_A/F_G)), F_k = N_obs,k/Z_k from
core's own Z_k at that H0. Exact (to rounding) wherever no PE sample or injection lies where a
catalog is incomplete: the complete rung on either grid. On m<18 the in-data missing hosts
change with n0, so the reference there is the input densities (--ref-n0=-3,-5) and the
difference is that part (against zero density the identity does not apply at m<18 at all).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a15_common as C  # noqa: E402
import a15_host_norm as A  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="complete")
    ap.add_argument("--out", required=True)
    ap.add_argument("--events", type=int, default=20)
    ap.add_argument("--injections", type=int, default=20000)
    ap.add_argument("--ref-n0", default="-24,-24",
                    help="reference densities log10n0,log10n0_c2 (zero density by default)")
    args = ap.parse_args()
    REF = tuple(float(x) for x in args.ref_n0.split(","))
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax.numpy as jnp
    import darksirens as ds
    from darksirens.cosmology import _grid
    from darksirens.runtime_binding import bind_analysis
    t0 = time.time()
    ev = A.events_subset(ds, args.events)
    inj = A.injections_subset(ds.load_injections(str(C.INJECTIONS)), args.injections)
    an, _, _, info = A.build_model(ds, args.rung)
    b = bind_analysis(an, events=ev, injections=inj, max_likelihood_variance=A.MAXVAR,
                      sel_batch_size=A.SEL_BATCH, pe_event_block=A.PEB)
    labels = list(b.labels)
    zk = A.make_zk(an)

    def th(h, f, n0):
        v = {"H0": h, "fcat_2": f, "log10n0": n0[0], "log10n0_c2": n0[1]}
        return jnp.asarray([v[l] for l in labels], dtype=jnp.float64)

    rows = []
    for n0 in ((-3.0, -5.0), (-3.0, -5.0 + C.LOG2), (-3.0, -5.0 - C.LOG2)):
        for h in (64.0, 67.75, 74.0):
            o, m = zk(th(h, 0.3, n0), b.model_operands)
            o, m = np.asarray(o, float), np.asarray(m, float)
            lnF = np.log(o / (o + m))
            d = float(lnF[1] - lnF[0])
            o0, m0 = zk(th(h, 0.3, REF), b.model_operands)
            lnF0 = np.log(np.asarray(o0, float) / (np.asarray(o0, float) + np.asarray(m0, float)))
            d0 = float(lnF0[1] - lnF0[0])
            for f in (0.15, 0.30, 0.45):
                fp = float(C.expit(C.logit(f) + d - d0))
                v = float(b(th(h, f, n0)))
                v0 = float(b(th(h, fp, REF)))
                vnaive = float(b(th(h, f, REF)))
                rows.append({"log10n0": n0, "H0": h, "f": f, "logit_shift": d - d0, "f_mapped": fp,
                             "lnL": v, "lnL_reference_at_f_mapped": v0, "difference": v - v0,
                             "lnL_reference_at_f": vnaive, "difference_unmapped": v - vnaive,
                             "F": np.exp(lnF).tolist()})
                print(f"n0 {n0} H0 {h} f {f}: shift {d - d0:+.5f} f' {fp:.5f}  lnL {v:.10f}  "
                      f"reference(f') {v0:.10f}  diff {v - v0:+.3e}  (unmapped {v - vnaive:+.3e})", flush=True)
    out = {"rung": args.rung, "reference_log10n0": REF, "zmax_grid": float(_grid.zMax), "n_events": int(b.n_events),
           "n_injections": int(inj.n_injections), "labels": labels, "N_obs": info["n_obs"],
           "max_abs_difference": float(max(abs(r["difference"]) for r in rows)),
           "max_abs_difference_unmapped": float(max(abs(r["difference_unmapped"]) for r in rows)),
           "rows": rows, "seconds": time.time() - t0}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("rung", "zmax_grid", "max_abs_difference", "max_abs_difference_unmapped")}))


if __name__ == "__main__":
    main()
