#!/usr/bin/env python
"""Validation 1 (CPU): the redraw sampler reproduces the stored posterior samples statistically.

    JAX_PLATFORMS=cpu /hildafs/projects/phy230014p/magana/envs/darksirens-core-e7c3007/bin/python \\
        validate_redraw.py

For both event sets (all-AGN and mixed, seed 100):
  1. the stored file's derived columns are reproduced by the post-processing this analysis
     imports (m1src/m2src from the generator's z(dL) grid; p_pe from p_pe_v3, mean-1);
  2. redraw r = 0 at N_s = 2000 against the stored samples: per-event two-sample KS in every
     stored column for 12 events (lowest/median/highest SNR and 9 random), plus pooled KS;
  3. the prior-weight column of the redraw and of the stored file obey the same closed form,
     p_pe = c_i * rho/(dL m1det q) with one constant per event, and rho is the bijection's
     rho_opt(Mc_det, dL);
  4. common random numbers: redraw 0 at 8000 contains redraw 0 at 2000 as its first 2000 samples;
  5. darksirens' loader on the in-memory store reproduces ds.load_events(path) bitwise for the
     stored samples, and accepts the redraw.
Writes results/validation_redraw.json; exits 1 on failure.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mc_redraw as M  # noqa: E402

A14 = HERE.parent
D = M.DATA / "seed100" / "events"
SETS = {"pureagn": D / "events_pureagn.h5", "mixed": D / "events.h5"}
KS_COLS = ("ra", "dec", "dL", "m1det", "m2det", "chieff", "m1src", "m2src", "p_pe", "mc_det", "q", "rho")


def closed_form_ratio(c, nobs, nsamp):
    g = M.generator()
    rho = g.rho_opt_of_mc_dl(c["mc_det"].ravel(), c["dL"].ravel())
    cf = (rho / (c["dL"].ravel() * c["m1det"].ravel() * c["q"].ravel())).reshape(nobs, nsamp)
    ratio = c["p_pe"].reshape(nobs, nsamp) / cf
    rel_spread = float(np.max(np.std(ratio, axis=1) / np.mean(ratio, axis=1)))
    rho_rel = float(np.max(np.abs(rho - c["rho"].ravel()) / c["rho"].ravel()))
    return rel_spread, rho_rel


def main():
    import darksirens as ds
    if "darksirens-core-e7c3007" not in ds.__file__:
        sys.exit(f"[fatal] darksirens from {ds.__file__}")
    out = {"sets": {}, "pass": True}
    fails = []
    for name, path in SETS.items():
        t0 = time.time()
        obs, attrs, nobs, nsamp = M.read_event_set(path)
        st = M.read_stored_samples(path)
        rec = {"file": str(path), "nobs": nobs, "nsamp_stored": nsamp}
        # 1. derived columns of the stored file
        post = {k: st[k].ravel().copy() for k in ("m1det", "m2det", "dL", "mc_det", "q")}
        post = M.postprocess(post, nobs, nsamp)
        rec["stored_rederived_max_rel"] = {
            k: float(np.max(np.abs(post[k] - st[k].ravel()) / np.abs(st[k].ravel())))
            for k in ("m1src", "m2src", "p_pe")}
        # 2. redraw 0 at 2000
        code = M.SET_CODES[name]
        rd = M.redraw(obs, 2000, code, 0)
        rdr = {k: v.reshape(nobs, 2000) for k, v in rd.items()}
        rng = np.random.default_rng(7)
        order = np.argsort(obs["rho"])
        ev = sorted(set([int(order[0]), int(order[nobs // 2]), int(order[-1])]
                        + rng.choice(nobs, 9, replace=False).tolist()))
        per = []
        for i in ev:
            row = {"event": i, "rho_obs": float(obs["rho"][i])}
            for k in KS_COLS:
                a, b = st[k][i], rdr[k][i]
                if k == "ra":   # wrapped channel: compare on a cut away from the mode
                    cut = (obs["ra"][i] + np.pi) % (2 * np.pi)
                    a, b = (a - cut) % (2 * np.pi), (b - cut) % (2 * np.pi)
                row[k] = float(ks_2samp(a, b).pvalue)
            per.append(row)
        pvals = np.array([[r[k] for k in KS_COLS] for r in per])
        rec["per_event_ks"] = per
        rec["per_event_ks_min_p"] = float(pvals.min())
        rec["per_event_ks_n_tests"] = int(pvals.size)
        rec["per_event_ks_frac_below_0p05"] = float((pvals < 0.05).mean())
        # pooled over all events: the per-event standardised value, so the pool is one law
        pooled = {}
        for k in ("mc_det", "q", "rho", "chieff", "dec"):
            a, b = st[k], rdr[k]
            mu, sd = a.mean(axis=1, keepdims=True), a.std(axis=1, keepdims=True)
            pooled[k] = float(ks_2samp(((a - mu) / sd).ravel()[::10], ((b - mu) / sd).ravel()[::10]).pvalue)
        rec["pooled_standardised_ks_p"] = pooled
        # 3. closed form of p_pe, stored and redrawn
        rec["p_pe_closed_form_stored"] = dict(zip(("per_event_ratio_rel_spread", "rho_bijection_rel"),
                                                  closed_form_ratio(st, nobs, nsamp)))
        rec["p_pe_closed_form_redraw"] = dict(zip(("per_event_ratio_rel_spread", "rho_bijection_rel"),
                                                  closed_form_ratio(rdr, nobs, 2000)))
        # 4. CRN nesting
        rd8 = M.redraw(obs, 8000, code, 0)
        nest = {}
        for k in ("ra", "dec", "dL", "m1det", "m2det", "chieff", "m1src", "m2src"):
            nest[k] = bool(np.array_equal(rd8[k].reshape(nobs, 8000)[:, :2000], rdr[k]))
        p8 = rd8["p_pe"].reshape(nobs, 8000)[:, :2000]
        nest["p_pe_up_to_event_constant"] = float(np.max(np.std(p8 / rdr["p_pe"], axis=1)
                                                         / np.mean(p8 / rdr["p_pe"], axis=1)))
        rec["crn_nesting_8000_contains_2000"] = nest
        # 5. loader on memory
        ref = ds.load_events(str(path))
        stcols = {k: st[k].ravel() for k in M.STORE_COLUMNS}
        mem = M.load_store(ds, stcols, attrs, nobs, nsamp)
        rec["loader_memory_vs_file_bitwise"] = bool(
            all(np.array_equal(np.asarray(ref.columns[k]), np.asarray(mem.columns[k])) for k in M.STORE_COLUMNS)
            and np.array_equal(ref.prior_wt, mem.prior_wt))
        mrd = M.load_store(ds, rd, attrs, nobs, 2000)
        rec["loader_accepts_redraw"] = bool(mrd.n_events == nobs and mrd.nsamp == 2000)
        rec["seconds"] = time.time() - t0
        # verdicts
        checks = {
            "stored_rederived": max(rec["stored_rederived_max_rel"].values()) < 1e-10,
            "per_event_ks": rec["per_event_ks_min_p"] > 1e-4,
            "pooled_ks": min(pooled.values()) > 1e-3,
            "closed_form": (rec["p_pe_closed_form_stored"]["per_event_ratio_rel_spread"] < 1e-10
                            and rec["p_pe_closed_form_redraw"]["per_event_ratio_rel_spread"] < 1e-10
                            and rec["p_pe_closed_form_redraw"]["rho_bijection_rel"] < 1e-10),
            "crn": all(v for k, v in nest.items() if k != "p_pe_up_to_event_constant")
                   and nest["p_pe_up_to_event_constant"] < 1e-12,
            "loader": rec["loader_memory_vs_file_bitwise"] and rec["loader_accepts_redraw"],
        }
        rec["checks"] = checks
        for k, v in checks.items():
            if not v:
                fails.append(f"{name}:{k}")
        out["sets"][name] = rec
        print(f"[val1] {name}: {json.dumps(checks)}  KS min p {rec['per_event_ks_min_p']:.3g} "
              f"({rec['per_event_ks_n_tests']} tests, {rec['per_event_ks_frac_below_0p05']:.2f} < 0.05)  "
              f"pooled {pooled}  [{rec['seconds']:.0f}s]", flush=True)
    out["pass"] = not fails
    out["fails"] = fails
    (A14 / "results").mkdir(exist_ok=True)
    (A14 / "results" / "validation_redraw.json").write_text(json.dumps(out, indent=1))
    print(f"[val1] {'PASS' if not fails else 'FAIL ' + str(fails)}")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
