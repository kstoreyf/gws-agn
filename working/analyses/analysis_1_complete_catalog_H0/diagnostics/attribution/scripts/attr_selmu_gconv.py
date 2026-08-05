#!/usr/bin/env python3
"""Convergence battery for the selection kernel G(b) of ``attr_selmu_oracle.py``.

The quantity that has to be converged is ``d ln mu/dH0``, not ``G`` in a tail the
host measure never reaches, so each knob is moved and the oracle's own host
measures (saved by ``attr_selmu_oracle.py`` into ``attr_selmu_<tracer>.npz``) are
re-integrated against the rebuilt kernel.  Nothing here touches darksirens or a
GPU: it is a pure-quadrature re-run.

  python scripts/attr_selmu_gconv.py --variant n_ghx2
  python scripts/attr_selmu_gconv.py --collect

Outputs: results/attr_selmu_gconv_<variant>.json, then results/attr_selmu_gconv.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
sys.path.insert(0, str(HERE))

H0_FID = 67.74
BASE = dict(n_m1=2800, n_q=1400, n_gh=32, dv=1.0e-3)
VARIANTS = {
    "base": {},
    "n_m1x2": dict(n_m1=5600),
    "n_qx2": dict(n_q=2800),
    "n_ghx2": dict(n_gh=64),
    "dv_half": dict(dv=5.0e-4),
    "range_wide": dict(m1_lo=0.5, m1_hi=190.0),
}
ARMS = ("kde", "delta", "unif", "norate")


def measures(tracer):
    d = np.load(RES / f"attr_selmu_{tracer}.npz")
    out = {"kde": (d["b_zk"], d["nu_kde"])}
    for a in ("delta", "unif", "norate"):
        out[a] = (d["bL"], d[f"w_{a}"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="base", choices=list(VARIANTS))
    ap.add_argument("--tracers", nargs="+", default=["gal", "agn"])
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--outdir", default=str(RES))
    args = ap.parse_args(argv)
    od = Path(args.outdir)

    if args.collect:
        out = {"name": "attr_selmu_gconv", "base": BASE, "variants": {}}
        base = json.loads((od / "attr_selmu_gconv_base.json").read_text())
        out["base_dlnmu"] = base["dlnmu"]
        for v in VARIANTS:
            p = od / f"attr_selmu_gconv_{v}.json"
            if not p.exists():
                continue
            r = json.loads(p.read_text())
            out["variants"][v] = {
                "settings": r["settings"], "dlnmu": r["dlnmu"],
                "abs_change": {t: {a: r["dlnmu"][t][a] - base["dlnmu"][t][a]
                                   for a in r["dlnmu"][t]} for t in r["dlnmu"]},
                "mass_norm": r["mass_norm"]}
        (od / "attr_selmu_gconv.json").write_text(json.dumps(out, indent=2))
        for v, r in out["variants"].items():
            print(f"{v:>12}  " + "  ".join(
                f"{t}/{a}:{r['abs_change'][t][a]:+.2e}"
                for t in r["abs_change"] for a in ARMS))
        print(f"Wrote {od/'attr_selmu_gconv.json'}")
        return 0

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax.numpy as jnp
    import attr_selmu_oracle as AO
    from darksirens.gw.populations.registry import get_model
    from darksirens.gw.populations import get_fixed_population_params

    kk = dict(BASE)
    kk.update(VARIANTS[args.variant])
    model = get_model("powerlaw+peak", shared_beta=True, shared_spin=True,
                      shared_gamma=True)
    pf = np.asarray(get_fixed_population_params(
        "powerlaw+peak", shared_beta=True, shared_spin=True, shared_gamma=True))
    th = jnp.asarray(pf[:model.mixture.n_params])
    t0 = time.time()
    bg, G, Gp, diag = AO.build_G(model, th, **kk)
    Gf = AO.Gfun(bg, G, Gp)
    print(f"[{args.variant}] built in {time.time()-t0:.0f}s  {diag}", flush=True)

    res = {}
    for tr in args.tracers:
        if not (RES / f"attr_selmu_{tr}.npz").exists():
            continue
        M = measures(tr)
        res[tr] = {}
        for a in ARMS:
            b, w = M[a]
            res[tr][a] = float(np.dot(w, Gf.deriv(b)) / np.dot(w, Gf(b))
                               / (AO.S_DL * H0_FID))
        print(f"  {tr}: " + "  ".join(f"{a}:{res[tr][a]:+.8e}" for a in ARMS),
              flush=True)
    out = {"variant": args.variant, "settings": kk, "dlnmu": res,
           "mass_norm": diag["mass_norm"], "diag": diag,
           "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (od / f"attr_selmu_gconv_{args.variant}.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {od/f'attr_selmu_gconv_{args.variant}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
