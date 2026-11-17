#!/usr/bin/env python
"""Analysis 14's redraw validation, and Analysis 16's production redraw, on an obslike set.

    sbatch --job-name=redraw run_cpu.sbatch core validate_redraw_obslike.py [SEED] [SUFFIX]
    (core venv; defaults: seed 100, _obslike = events_obslike.h5)

Part A is analysis_14_monte_carlo_accuracy/scripts/validate_redraw.py's five checks, made
with that script's own helpers (mc_redraw, closed_form_ratio) on the new file instead of its
two hard-coded seed-100 files, written here instead of into Analysis 14's results:
  1. the stored m1src/m2src/p_pe are re-derived by the imported post-processing (< 1e-10);
  2. redraw r = 0 at 2000 samples against the stored samples: per-event two-sample KS in
     every column for 12 events (lowest/median/highest SNR + 9 random) and pooled KS;
  3. p_pe = c_i rho/(dL m1det q) for stored and redrawn samples (one constant per event);
  4. common random numbers: the redraw at 8000 per event contains the one at 2000;
  5. darksirens-core's loader on the in-memory store reproduces load_events(path) bitwise
     and accepts the redraw.
Part B redraws 8000 samples per event exactly as prod_model.Model.load_events does
(redraw_key from prod_common for this suffix, blocks of 2000), loads them through core's
loader and reports the PE Monte Carlo summary (per-event effective sample size of the
p_pe-weighted samples under the population is not computed here; the loader's row sums and
the sample count are).  Writes results/validation_redraw_<file stem>.json; exits 1 on failure.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

os.environ.setdefault("JAX_PLATFORMS", "cpu")
WORK = Path("/hildafs/projects/phy230014p/magana/gws-agn/working")
A14S = WORK / "analyses" / "analysis_14_monte_carlo_accuracy" / "scripts"
A16S = WORK / "analyses" / "analysis_16_production_runs" / "scripts"
sys.path.insert(0, str(A14S))
sys.path.insert(0, str(A16S))
import mc_redraw as M  # noqa: E402
import validate_redraw as V  # noqa: E402  (its helpers; its main() is not called)
import prod_common as P  # noqa: E402

OB = WORK / "data" / "obslike"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SFX = sys.argv[2] if len(sys.argv) > 2 else "_obslike"
KIND = "host_dependent" if "marked" in SFX else "equal_populations"
PATH = P.events_path(SEED, KIND, SFX.replace("_marked_dmu0p10_dmuG5", "")) if "marked" in SFX \
    else P.events_path(SEED, KIND, SFX)
SET_CODE = 2   # A14's code for an equal-population set; with seed != 100 block_rng appends the seed


def main():
    import darksirens as ds
    if "darksirens-core-e7c3007" not in ds.__file__:
        sys.exit(f"[fatal] darksirens from {ds.__file__}")
    t0 = time.time()
    obs, attrs, nobs, nsamp = M.read_event_set(PATH)
    st = M.read_stored_samples(PATH)
    rec = {"file": str(PATH), "seed": SEED, "suffix": SFX, "nobs": nobs, "nsamp_stored": nsamp,
           "darksirens_file": ds.__file__}
    post = {k: st[k].ravel().copy() for k in ("m1det", "m2det", "dL", "mc_det", "q")}
    post = M.postprocess(post, nobs, nsamp)
    rec["stored_rederived_max_rel"] = {k: float(np.max(np.abs(post[k] - st[k].ravel()) / np.abs(st[k].ravel())))
                                       for k in ("m1src", "m2src", "p_pe")}
    rd = M.redraw(obs, 2000, SET_CODE, 0, seed=SEED + 1000)   # a stream no A14 run used
    rdr = {k: v.reshape(nobs, 2000) for k, v in rd.items()}
    rng = np.random.default_rng(7)
    order = np.argsort(obs["rho"])
    ev = sorted(set([int(order[0]), int(order[nobs // 2]), int(order[-1])]
                    + rng.choice(nobs, 9, replace=False).tolist()))
    per = []
    for i in ev:
        row = {"event": i, "rho_obs": float(obs["rho"][i])}
        for k in V.KS_COLS:
            a, b = st[k][i], rdr[k][i]
            if k == "ra":
                cut = (obs["ra"][i] + np.pi) % (2 * np.pi)
                a, b = (a - cut) % (2 * np.pi), (b - cut) % (2 * np.pi)
            row[k] = float(ks_2samp(a, b).pvalue)
        per.append(row)
    pv = np.array([[r[k] for k in V.KS_COLS] for r in per])
    rec.update({"per_event_ks": per, "per_event_ks_min_p": float(pv.min()), "per_event_ks_n_tests": int(pv.size),
                "per_event_ks_frac_below_0p05": float((pv < 0.05).mean())})
    pooled = {}
    for k in ("mc_det", "q", "rho", "chieff", "dec"):
        a, b = st[k], rdr[k]
        mu, sd = a.mean(axis=1, keepdims=True), a.std(axis=1, keepdims=True)
        pooled[k] = float(ks_2samp(((a - mu) / sd).ravel()[::10], ((b - mu) / sd).ravel()[::10]).pvalue)
    rec["pooled_standardised_ks_p"] = pooled
    rec["p_pe_closed_form_stored"] = dict(zip(("per_event_ratio_rel_spread", "rho_bijection_rel"),
                                              V.closed_form_ratio(st, nobs, nsamp)))
    rec["p_pe_closed_form_redraw"] = dict(zip(("per_event_ratio_rel_spread", "rho_bijection_rel"),
                                              V.closed_form_ratio(rdr, nobs, 2000)))
    t1 = time.time()
    rd8 = M.redraw(obs, 8000, SET_CODE, 0, seed=SEED + 1000)
    rec["redraw_8000_seconds"] = time.time() - t1
    nest = {k: bool(np.array_equal(rd8[k].reshape(nobs, 8000)[:, :2000], rdr[k]))
            for k in ("ra", "dec", "dL", "m1det", "m2det", "chieff", "m1src", "m2src")}
    p8 = rd8["p_pe"].reshape(nobs, 8000)[:, :2000]
    nest["p_pe_up_to_event_constant"] = float(np.max(np.std(p8 / rdr["p_pe"], axis=1) / np.mean(p8 / rdr["p_pe"], axis=1)))
    rec["crn_nesting_8000_contains_2000"] = nest
    rec["redraw_8000_m2src_min"] = float(rd8["m2src"].min())
    ref = ds.load_events(str(PATH))
    mem = M.load_store(ds, {k: st[k].ravel() for k in M.STORE_COLUMNS}, attrs, nobs, nsamp)
    rec["loader_memory_vs_file_bitwise"] = bool(
        all(np.array_equal(np.asarray(ref.columns[k]), np.asarray(mem.columns[k])) for k in M.STORE_COLUMNS)
        and np.array_equal(ref.prior_wt, mem.prior_wt))
    m8 = M.load_store(ds, rd8, attrs, nobs, 8000)
    rs = np.asarray(m8.prior_wt).reshape(nobs, 8000).sum(axis=1)
    rec["loader_accepts_redraw_8000"] = bool(m8.n_events == nobs and m8.nsamp == 8000)
    rec["loader_redraw_8000_p_pe_row_sum_max_abs_dev"] = float(np.abs(rs - 1).max())
    del rd8, m8
    # Part B: the production redraw (prod_model.Model.load_events, verbatim recipe)
    t2 = time.time()
    key = P.redraw_key(SEED, KIND, SFX if KIND == "equal_populations" else SFX.replace("_marked_dmu0p10_dmuG5", ""), 0)
    g = M.generator()
    blocks = []
    for b in range(8000 // P.REDRAW_BLOCK):
        rng_b = np.random.default_rng(np.random.SeedSequence(list(key) + [b]))
        p = g.posterior_samples_v3(rng_b, obs, P.REDRAW_BLOCK)
        blocks.append({k: v.reshape(nobs, P.REDRAW_BLOCK) for k, v in p.items()})
    postp = {k: np.concatenate([bl[k] for bl in blocks], axis=1).ravel() for k in blocks[0]}
    postp = M.postprocess(postp, nobs, 8000)
    store = M.load_store(ds, postp, attrs, nobs, 8000)
    rsp = np.asarray(store.prior_wt).reshape(nobs, 8000).sum(axis=1)
    rec["production_redraw"] = {"redraw_key": key, "n_events": int(store.n_events), "nsamp": int(store.nsamp),
                                "p_pe_row_sum_max_abs_dev": float(np.abs(rsp - 1).max()),
                                "seconds": time.time() - t2,
                                "m2src_min": float(postp["m2src"].min()),
                                "frac_samples_m2src_lt_5": float((postp["m2src"] < 5).mean())}
    rec["seconds"] = time.time() - t0
    checks = {
        "stored_rederived": max(rec["stored_rederived_max_rel"].values()) < 1e-10,
        "per_event_ks": rec["per_event_ks_min_p"] > 1e-4,
        "pooled_ks": min(pooled.values()) > 1e-3,
        "closed_form": (rec["p_pe_closed_form_stored"]["per_event_ratio_rel_spread"] < 1e-10
                        and rec["p_pe_closed_form_redraw"]["per_event_ratio_rel_spread"] < 1e-10
                        and rec["p_pe_closed_form_redraw"]["rho_bijection_rel"] < 1e-10),
        "crn": all(v for k, v in nest.items() if k != "p_pe_up_to_event_constant")
               and nest["p_pe_up_to_event_constant"] < 1e-12,
        "loader": rec["loader_memory_vs_file_bitwise"] and rec["loader_accepts_redraw_8000"]
                  and rec["loader_redraw_8000_p_pe_row_sum_max_abs_dev"] < 1e-9,
        "production_redraw_8000": rec["production_redraw"]["n_events"] == nobs
                                  and rec["production_redraw"]["nsamp"] == 8000
                                  and rec["production_redraw"]["p_pe_row_sum_max_abs_dev"] < 1e-9,
    }
    rec["checks"] = checks
    rec["pass"] = all(checks.values())
    print(json.dumps({k: v for k, v in rec.items() if k not in ("per_event_ks",)}, indent=1, default=str))
    out = OB / "results" / f"validation_redraw_{PATH.stem}_s{SEED}.json"
    out.write_text(json.dumps(rec, indent=1, default=str))
    print("wrote", out, "PASS" if rec["pass"] else "FAIL")
    sys.exit(0 if rec["pass"] else 1)


if __name__ == "__main__":
    main()
