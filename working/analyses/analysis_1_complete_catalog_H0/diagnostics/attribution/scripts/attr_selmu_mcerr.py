#!/usr/bin/env python3
"""The injection estimator's OWN Monte-Carlo error on ``d ln mu/dH0`` -- the
common-mode term CLOSURE.md 14.2 named and 10 item 7 asks to be carried.

``r`` subtracts ONE number from EVERY event, so whatever Monte-Carlo error the
selection estimator's slope carries is a **common-mode** shift of ``r`` that the
per-event ``sem`` cannot see and that does NOT average down with ``nEvents``.

This is the lean version of ``attr_selmu_inj.py``: it makes a SINGLE selection pass
at ``(H0-dh, H0, H0+dh)`` -- exactly the pass ``attr_ds_bridge.sel_pass`` already
performs and anchors -- and then Poisson-bootstraps the finite-difference estimator

    d ln mu/dH0  =  [ lse(ldw_+) - lse(ldw_-) ] / 2dh

over the injections.  A Poisson bootstrap (weights ~ Poisson(1)) is the right
resampling for a sum over a Poisson-sized detected set at fixed ``Ndraw``; the
delta-method influence function of the same statistic is reported alongside as an
independent estimate, and the two agreed to 5 % in the v2 campaign.

Outputs: results/attr_selmu_mcerr_<tag>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

H0_FID = 67.74


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracer", choices=["gal", "agn"], default="gal")
    ap.add_argument("--injections", choices=["targeted", "popuni"],
                    default="targeted")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--dataroot", default=None)
    ap.add_argument("--events", default=None)
    ap.add_argument("--dh", type=float, default=0.5)
    ap.add_argument("--sel_batch", type=int, default=50000)
    ap.add_argument("--n_boot", type=int, default=400)
    ap.add_argument("--boot_seed", type=int, default=20260801)
    ap.add_argument("--exact", type=float, default=None,
                    help="The exact oracle's d ln mu/dH0, for the comparison line.")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    args = ap.parse_args(argv)
    tag = args.tag or f"{args.tracer}_{args.injections}_s{args.seed}"
    t0 = time.time()

    import attr_ds_bridge as bridge
    kw = dict(kde_window=4096) if args.tracer == "gal" else {}
    if args.dataroot:
        kw["dataroot"] = args.dataroot
    if args.events:
        kw["events"] = args.events
    B = bridge.build(tracer=args.tracer, seed=args.seed, h0=H0_FID,
                     injections=args.injections, sel_batch=args.sel_batch, **kw)
    S = bridge.sel_pass(B, dh=args.dh, sel_batch=args.sel_batch, want_q=False)
    print(f"[anchor] log_mu mine {S['log_mu']:.10f} vs darksirens "
          f"{B.spy0['log_mu']:.10f}  diff {S['log_mu']-B.spy0['log_mu']:.3e}",
          flush=True)

    lm, lp = np.asarray(S["ldw_m"]), np.asarray(S["ldw_p"])
    fin = np.isfinite(lm) & np.isfinite(lp)
    lm, lp = lm[fin], lp[fin]
    mm, mp = lm.max(), lp.max()
    wm, wp = np.exp(lm - mm), np.exp(lp - mp)
    twodh = 2.0 * args.dh
    d_hat = float((mp + np.log(wp.sum()) - mm - np.log(wm.sum())) / twodh)

    rng = np.random.default_rng(args.boot_seed)
    vals = np.empty(args.n_boot)
    n = wm.size
    for b in range(args.n_boot):
        k = rng.poisson(1.0, n)
        sm, sp = float(np.dot(k, wm)), float(np.dot(k, wp))
        vals[b] = (mp + np.log(sp) - mm - np.log(sm)) / twodh
    boot_sd = float(vals.std(ddof=1))

    # delta method: the influence function of (log sum wp - log sum wm)/2dh
    infl = (wp / wp.sum() - wm / wm.sum()) / twodh
    delta_sd = float(np.sqrt(np.sum(infl ** 2)))

    out = {"name": "attr_selmu_mcerr", "tag": tag, "tracer": args.tracer,
           "injections": args.injections, "seed": args.seed, "dh": args.dh,
           "n_injections": int(n), "Ndraw": float(B.Ndraw),
           "anchor_log_mu_absdiff": float(abs(S["log_mu"] - B.spy0["log_mu"])),
           "dlnmu_fd": d_hat,
           "sigma_MC_bootstrap": boot_sd, "sigma_MC_delta_method": delta_sd,
           "n_bootstrap": int(args.n_boot),
           "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if args.exact is not None:
        out["dlnmu_exact_oracle"] = args.exact
        out["inj_minus_exact"] = d_hat - args.exact
        out["inj_minus_exact_sigma"] = (d_hat - args.exact) / max(boot_sd, 1e-30)
    print(f"[mcerr] {tag}: d ln mu/dH0 = {d_hat:+.8e}  "
          f"sigma_MC = {boot_sd:.3e} (bootstrap) / {delta_sd:.3e} (delta method)")
    if args.exact is not None:
        print(f"        exact = {args.exact:+.8e}   inj - exact = "
              f"{out['inj_minus_exact']:+.3e} = {out['inj_minus_exact_sigma']:+.2f} "
              f"sigma of the estimator's own error")
    od = Path(args.outdir)
    (od / f"attr_selmu_mcerr_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {od / f'attr_selmu_mcerr_{tag}.json'}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
