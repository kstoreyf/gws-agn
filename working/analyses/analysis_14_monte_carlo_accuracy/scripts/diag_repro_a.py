#!/usr/bin/env python
"""Diagnose the failed reproduction of the stored all-AGN H0 scan (Analysis 0) on core (CPU).

    JAX_PLATFORMS=cpu /hildafs/projects/phy230014p/magana/envs/darksirens-core-e7c3007/bin/python \\
        diag_repro_a.py [--events pureagn|mixed] [--arms field1,cond1]

Arms (all core e7c3007, historical settings, gamma = 0, log10n0 = -24, Om0 0.3075, hard guard 1e6):
  field1  K = 1, AGN complete catalog only, catalog_sky_weighting="field"
  cond1   K = 1, AGN complete catalog only, catalog_sky_weighting="conditional" (core's default)
Evaluated at a few H0 cells of the stored grid, against
  * the stored legacy 2b86a2d values (A0 h0_pureagn_targeted_s100.h5 for pureagn),
  * the GPU K = 2 f = 1 values of results/scans/repro_a.h5 (same cells).
Writes results/diag_repro_a_<events>.json.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a14_mc_accuracy as A  # noqa: E402

CELLS = (66.0, 67.0, 67.75, 68.5, 69.5, 70.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="pureagn", choices=["pureagn", "mixed"])
    ap.add_argument("--arms", default="field1,cond1")
    args = ap.parse_args()
    import h5py
    import jax.numpy as jnp
    import darksirens as ds
    from darksirens.catalog import settings as cs
    from darksirens.population.utils import configure_normalization_grids
    from darksirens.runtime_binding import bind_analysis
    cs.configure_catalog_evaluation(kernel_window=1e-10, kernel_layout="padded", missing_density="grid")
    configure_normalization_grids(pairing_norm="per_sample", pairing_scale="analytic")
    cos = ds.Cosmology(H0=(50.0, 100.0), Om0=0.3075)
    probe = ds.model(cosmology=cos, population=ds.Population("powerlaw+peak", fixed=True)).parameters
    fv = dict(zip(probe.population_labels, probe.fixed_population))
    fv[[k for k in fv if "gamma" in k.lower()][0]] = 0.0
    cat = ds.load_catalog(str(A.D / "surveys" / "survey_agn_complete_ns32.h5"))
    ev_path = A.EVENTS["a" if args.events == "pureagn" else "b"]
    events = ds.load_events(str(ev_path))
    inj = ds.load_injections(str(A.INJ_RECORD))
    out = {"events": str(ev_path), "cells": list(CELLS), "arms": {}}
    # references
    if args.events == "pureagn":
        with h5py.File(A.REF["a"], "r") as f:
            H, L = np.asarray(f["H0_grid"]), np.asarray(f["log_likelihood"])
            pv, ne = np.asarray(f["guard/pe_variance_sum"]), np.asarray(f["guard/Neff"])
        idx = [int(np.argmin(abs(H - h))) for h in CELLS]
        out["stored_legacy"] = {"lnL": L[idx].tolist(), "pe_var": pv[idx].tolist(), "n_eff": ne[idx].tolist()}
        rp = A.A14 / "results" / "scans" / "repro_a.h5"
        if rp.exists():
            with h5py.File(rp, "r") as f:
                Hc = np.asarray(f["H0_grid"])
                j = [int(np.argmin(abs(Hc - h))) for h in CELLS]
                out["core_K2_f1_gpu"] = {k: np.asarray(f[k])[j].tolist()
                                         for k in ("lnL", "pe_var", "n_eff", "log_mu", "sel_corr")}
    for arm in args.arms.split(","):
        t0 = time.time()
        sky = {"field1": "field", "cond1": "conditional"}[arm]
        kw = dict(catalog_sky_weighting=sky) if sky == "field" else {}
        an = ds.model(cosmology=cos, population=ds.Population("powerlaw+peak", fixed=fv),
                      catalog=cat, completeness=None,
                      fixed_survey={"log10n0": -24.0, "delta": 0.0, "sigma_kde": 0.0},
                      allow_out_of_prior=True, **kw)
        b = bind_analysis(an, events=events, injections=inj, max_likelihood_variance=1e6,
                          sel_batch_size=5000, pe_event_block=5)
        labels = list(b.labels)
        rows = []
        diag = None
        if sky == "field":
            diag = A.make_diag(an, int(b.n_events), int(b.nsamp), float(b.n_draw), 5)
        for h in CELLS:
            th = jnp.asarray([h if l == "H0" else -24.0 for l in labels], dtype=jnp.float64)
            v = float(b(th))
            r = {"H0": h, "lnL": v}
            if diag is not None:
                d = diag(th, b.gw_pe, b.gw_selection, b.model_operands)
                r.update(lnL_diag=float(d[0]), pe_var=float(d[1]), n_eff=float(d[2]), log_mu=float(d[3]),
                         sel_corr=float(d[4]), sum_event_ln_Z=float(np.sum(np.asarray(d[5]))))
            rows.append(r)
            print(f"[diag {arm}] H0 {h:6.2f} lnL {v:.4f} {r}", flush=True)
        out["arms"][arm] = {"labels": labels, "redshift_type": type(an.redshift).__name__,
                            "rows": rows, "seconds": time.time() - t0}
    # summaries: differences relative to the stored values, after removing the value at 67.75
    def rel(x):
        x = np.asarray(x, float)
        return (x - x[CELLS.index(67.75)]).tolist()
    out["relative_to_67p75"] = {}
    if "stored_legacy" in out:
        out["relative_to_67p75"]["stored_legacy"] = rel(out["stored_legacy"]["lnL"])
    if "core_K2_f1_gpu" in out:
        out["relative_to_67p75"]["core_K2_f1_gpu"] = rel(out["core_K2_f1_gpu"]["lnL"])
    for arm, a in out["arms"].items():
        out["relative_to_67p75"][arm] = rel([r["lnL"] for r in a["rows"]])
    dst = A.A14 / "results" / f"diag_repro_a_{args.events}.json"
    dst.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["relative_to_67p75"], indent=1))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
