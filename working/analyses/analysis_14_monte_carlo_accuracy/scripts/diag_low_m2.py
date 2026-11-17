#!/usr/bin/env python
"""Core and the mocks disagree below the minimum mass: the numbers behind the failed seed-100 (a) gate.

    python diag_low_m2.py            (jax env; runs the two density arms as subprocesses)

1. The fiducial population density (powerlaw+peak at the mock's input, gamma = 0) at the true
   (m1src, q, chi_eff, z) of seed 100's all-AGN event 495, and at its posterior samples,
   evaluated by darksirens-core e7c3007 and by legacy darksirens 2b86a2d (the commit that
   generated the mock and ran Analyses 0-2). Arms: --arm core | --arm legacy (each prints JSON).
2. For every seed on disk and each of events.h5, events_pureagn.h5, events_puregal.h5: the
   smallest true m2src, the number of events with true m2src < 5 Msun (the population's m_min),
   and the number with more than 95% of their posterior samples at m2src < 5.
3. Event 495's per-event ln evidence and variance at the reference cell of the GPU K = 2, f = 1
   reproduction scan (results/scans/repro_a.h5).
Writes results/diag_low_m2.json.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A14 = HERE.parent
DATA = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data")
CORE_PY = "/hildafs/projects/phy230014p/magana/envs/darksirens-core-e7c3007/bin/python"
JAX_PY = "/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python"
LEGACY_SRC = "/hildafs/projects/phy230014p/magana/src/darksirens-2b86a2d"
EVENT = 495
M_MIN = 5.0
FILES = ("events.h5", "events_pureagn.h5", "events_puregal.h5")


def event_points(seed=100, fname="events_pureagn.h5", i=EVENT):
    import h5py
    with h5py.File(DATA / f"seed{seed}" / "events" / fname, "r") as f:
        t = f["truth"]
        nobs, nsamp = int(f.attrs["nobs"]), int(f.attrs["nsamp"])
        truth = {k: float(np.asarray(t[k])[i]) for k in ("m1src", "m2src", "chieff", "z", "q")}
        sl = slice(i * nsamp, (i + 1) * nsamp)
        smp = {k: np.asarray(f[k][sl]) for k in ("m1src", "m2src", "chieff", "dL")}
    return truth, smp


def arm(which):
    """Population log density at the event's truth and samples, in one code base."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    truth, smp = event_points()
    # z of each sample at the fiducial cosmology (only enters the rate factor (1+z)^(gamma-1))
    from astropy.cosmology import FlatLambdaCDM, z_at_value  # noqa: F401
    zs = np.interp(smp["dL"], *_dl_table())
    if which == "core":
        import darksirens as ds
        from darksirens.population import get_model
        assert "darksirens-core-e7c3007" in ds.__file__, ds.__file__
        probe = ds.model(cosmology=ds.Cosmology(H0=(50.0, 100.0), Om0=0.3075),
                         population=ds.Population("powerlaw+peak", fixed=True)).parameters
        labels = list(probe.population_labels)
        theta = np.asarray(probe.fixed_population, float)
        theta[[i for i, l in enumerate(labels) if "gamma" in l.lower()][0]] = 0.0
        model = get_model("powerlaw+peak", shared_beta=True, shared_spin=True, shared_gamma=True)
        src = ds.__file__
    else:
        import darksirens
        from darksirens.gw.populations import get_fixed_population_params, get_model
        assert "darksirens-2b86a2d" in darksirens.__file__, darksirens.__file__
        theta = np.asarray(get_fixed_population_params("powerlaw+peak", shared_beta=True,
                                                       shared_spin=True, shared_gamma=True), float)
        labels = None
        model = get_model("powerlaw+peak", shared_beta=True, shared_spin=True, shared_gamma=True)
        src = darksirens.__file__
    th = jnp.asarray(theta)

    def lp(m1, m2, chi, z):
        m1 = jnp.atleast_1d(jnp.asarray(m1, float))
        q = jnp.atleast_1d(jnp.asarray(m2, float)) / m1
        return np.asarray(model.log_p_pop(m1, q, jnp.atleast_1d(jnp.asarray(z, float)),
                                          jnp.atleast_1d(jnp.asarray(chi, float)), th))

    out = {"code": which, "darksirens_file": src, "theta": theta.tolist(), "labels": labels,
           "truth": truth,
           "log_p_pop_at_truth": float(lp(truth["m1src"], truth["m2src"], truth["chieff"], truth["z"])[0])}
    # the same point with m2 moved to 4.9, 5.1 and 6.0 (q at fixed m1), and a typical binary
    for m2 in (4.9, 5.1, 6.0):
        out[f"log_p_pop_m2_{m2}"] = float(lp(truth["m1src"], m2, truth["chieff"], truth["z"])[0])
    out["log_p_pop_typical_35_30"] = float(lp(35.0, 30.0, 0.0, 0.1)[0])
    v = lp(smp["m1src"], smp["m2src"], smp["chieff"], zs)
    fin = np.isfinite(v)
    out["samples"] = {"n": int(v.size), "n_finite": int(fin.sum()),
                      "n_m2src_below_5": int((smp["m2src"] < M_MIN).sum()),
                      "max_m2src": float(smp["m2src"].max()),
                      "log_mean_p_pop_over_samples": float(np.log(np.mean(np.exp(np.where(fin, v, -np.inf)))))
                      if fin.any() else None,
                      "log_p_pop_quantiles_finite": (np.quantile(v[fin], [0.0, 0.5, 1.0]).tolist()
                                                     if fin.any() else None)}
    return out


def _dl_table():
    from astropy.cosmology import FlatLambdaCDM
    c = FlatLambdaCDM(H0=67.74, Om0=0.3075)
    z = np.linspace(0, 2, 20001)
    return c.luminosity_distance(z).value, z


def table():
    import h5py
    rows = []
    for sd in sorted(DATA.glob("seed[0-9]*")):
        seed = int(sd.name[4:])
        for fn in FILES:
            p = sd / "events" / fn
            if not p.exists():
                continue
            try:
                with h5py.File(p, "r") as f:
                    nobs, nsamp = int(f.attrs["nobs"]), int(f.attrs["nsamp"])
                    m2t = np.asarray(f["truth"]["m2src"])
                    m2s = np.asarray(f["m2src"]).reshape(nobs, nsamp)
            except Exception as e:  # noqa: BLE001
                rows.append({"seed": seed, "file": fn, "error": str(e)})
                continue
            frac = (m2s < M_MIN).mean(axis=1)
            rows.append({"seed": seed, "file": fn, "min_true_m2src": float(m2t.min()),
                         "n_true_m2src_below_5": int((m2t < M_MIN).sum()),
                         "n_events_frac_samples_below_5_gt_0p95": int((frac > 0.95).sum()),
                         "max_frac_samples_below_5": float(frac.max()),
                         "worst_event": int(np.argmax(frac))})
        print(f"[table] seed {seed} done", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["core", "legacy"])
    args = ap.parse_args()
    if args.arm:
        print("JSON:" + json.dumps(arm(args.arm)))
        return
    out = {"event": {"seed": 100, "file": "events_pureagn.h5", "index": EVENT}, "m_min": M_MIN}
    for which, py, env in (("core", CORE_PY, {"PYTHONPATH": ""}),
                           ("legacy", JAX_PY, {"PYTHONPATH": LEGACY_SRC})):
        r = subprocess.run([py, str(Path(__file__).resolve()), "--arm", which], capture_output=True, text=True,
                           env={**os.environ, "JAX_PLATFORMS": "cpu", **env})
        line = [l for l in r.stdout.splitlines() if l.startswith("JSON:")]
        out[f"density_{which}"] = json.loads(line[0][5:]) if line else {"error": r.stderr[-3000:]}
        print(f"[density {which}] {json.dumps(out[f'density_{which}'])[:600]}", flush=True)
    rp = A14 / "results" / "scans" / "repro_a.h5"
    if rp.exists():
        import h5py
        with h5py.File(rp, "r") as f:
            ev = np.asarray(f["ref_cell/event_log_evidence"])
            va = np.asarray(f["ref_cell/event_mc_variance"])
            meta = json.loads(f.attrs["meta_json"])
        others = np.delete(ev, EVENT)
        out["repro_a_ref_cell"] = {"ref_cell": meta.get("ref_cell") or [67.75, 1.0],
                                   "event_ln_evidence": float(ev[EVENT]), "event_mc_variance": float(va[EVENT]),
                                   "median_ln_evidence_other_events": float(np.median(others)),
                                   "rank_lowest": int(np.argsort(ev)[0]), "second_lowest": float(np.sort(ev)[1])}
    rows = table()
    out["event_sets"] = rows
    ok = [r for r in rows if "error" not in r]
    out["event_sets_summary"] = {
        "n_sets": len(ok),
        "n_sets_with_true_m2src_below_5": int(sum(r["n_true_m2src_below_5"] > 0 for r in ok)),
        "n_sets_with_an_event_gt_0p95_samples_below_5": int(sum(r["n_events_frac_samples_below_5_gt_0p95"] > 0
                                                                for r in ok)),
        "by_file": {fn: {"n_sets": sum(r["file"] == fn for r in ok),
                         "n_with_event_gt_0p95_below_5": sum(r["file"] == fn and
                                                             r["n_events_frac_samples_below_5_gt_0p95"] > 0
                                                             for r in ok)} for fn in FILES}}
    dst = A14 / "results" / "diag_low_m2.json"
    dst.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "event_sets"}, indent=1)[:4000])
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
