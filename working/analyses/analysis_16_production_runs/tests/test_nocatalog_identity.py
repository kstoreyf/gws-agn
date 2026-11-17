#!/usr/bin/env python
"""The no-catalog limit: core's catalog-free likelihood is "hosts uniform in comoving volume".

    JAX_PLATFORMS=cpu <core venv python> tests/test_nocatalog_identity.py [--events 50 --injections 300000]

The driver's --catalogs none builds ds.model(cosmology, population) with no catalog, which core
evaluates with its spectral-siren likelihood. This test recomputes that likelihood by a direct sum
over the same posterior samples and detected injections, with an independent numpy cosmology:

    w_ij = p_pop(m1det/(1+z), q, z, chi | Lambda) * p_V(z) / [ (1+z) dd_L/dz * p_pe,ij ],
    p_V(z) = (dV_c/dz) / int_0^zmax dV_c/dz dz        (hosts uniform in comoving volume),
    z = z(d_L; H0, Om0) by inverting d_L = (1+z) chi(z) (fine quadrature, numpy),
    ln Z_i = ln mean_j w_ij,   sigma_i^2 = sum w^2 / (sum w)^2 - 1/n,
    mu = sum_k w_k / N_draw (injections, p_draw in place of p_pe),  1/N_eff = sum w^2/(sum w)^2 - 1/N_draw,
    ln L = sum_i ln Z_i - N ln mu + N (N + 3) / (2 N_eff)   [the guard: -inf unless N_eff > max(5N,
           N^2 / max(V_max - sum sigma_i^2, 1e-12))],

with samples or injections above the grid end zmax given zero weight (as core masks d_L beyond its
table). Only p_pop (core's population density; the same function in both) and the data columns
(core's loaded stores, including core's per-event p_pe normalisation) are shared.

Two comparisons per point: (A) the direct sum with core's own z(d_L) and dd_L/dz tables (tests the
structure: expected agreement ~1e-9 nats), (B) the fully independent cosmology (adds core's table
interpolation error). Points: H0 = 52, 60, 67.74, 80, 95; population at the inputs and moved.
Output: tests/results/nocatalog_identity.json. Pass: (A) max |d lnL| < 1e-6, (B) max |d lnL| < 1e-3
per 50 events, and every per-event ln Z_i, sigma_i^2, ln mu, N_eff within the same tolerances.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

A16 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(A16 / "scripts"))
import prod_common as P  # noqa: E402

C = P.C_KMS


def cosmology_numpy(H0, Om0, zmax, n=400001):
    """chi(z), dL(z), ddL/dz and dV_c/dz on a fine grid to zmax (+ margin), flat LCDM."""
    z = np.expm1(np.linspace(0.0, np.log1p(zmax * 1.5 + 0.5), n))
    E = np.sqrt(Om0 * (1 + z) ** 3 + 1 - Om0)
    chi = np.concatenate([[0.0], np.cumsum(0.5 * (1 / E[1:] + 1 / E[:-1]) * np.diff(z))]) * C / H0
    dl = (1 + z) * chi
    ddl = chi + (1 + z) * C / (H0 * E)
    dvc = C / H0 * chi ** 2 / E
    m = z <= zmax
    norm = np.trapz(dvc[m], z[m])
    return z, dl, ddl, dvc / norm


def direct_sum(m, coords, mode):
    """The direct-sum ln L and its pieces; mode 'A' (core's distance tables) or 'B' (numpy)."""
    import jax.numpy as jnp
    from darksirens.population import pop_model_parser
    from darksirens.cosmology import distances as D
    cfg = m.cfg
    H0 = float(coords["H0"])
    zmax = float(cfg["zmax"])
    pv = dict(cfg["population_values"])
    pv.update({k: v for k, v in coords.items() if k in pv})
    labels = list(m.an.parameters.population_labels)
    inv = {v: k for k, v in P.POP_LABELS.items()}
    pop_vec = jnp.asarray([pv[inv[l]] for l in labels], dtype=jnp.float64)
    pop = m.an.population
    logp_pop = pop_model_parser(pop.model_name, shared_beta=pop.shared_beta, shared_spin=pop.shared_spin,
                                shared_gamma=pop.shared_gamma)
    zg, dlg, ddlg, pv_z = cosmology_numpy(H0, P.OM0, zmax)

    def ldw(ev):
        dL = np.asarray(ev.dL, float)
        if mode == "A":
            from darksirens.cosmology.parameters import CosmologyParameters
            from darksirens.cosmology.volume import log_comoving_volume_prior
            dL_grid = D.dL_of_z(D.zgrid, H0, P.OM0, -1.0, 0.0)
            z = np.asarray(D.z_of_dL_precomputed(jnp.asarray(dL), dL_grid))
            ddl = np.asarray(D.ddL_of_z(jnp.asarray(z), jnp.asarray(dL), H0, P.OM0, -1.0, 0.0))
            dl_hi = float(dL_grid[-1])
            cos = CosmologyParameters(H0, P.OM0, -1.0, 0.0)
            lp_z = np.asarray(log_comoving_volume_prior(jnp.asarray(z), cos))
        else:
            z = np.interp(dL, dlg, zg)
            ddl = np.interp(z, zg, ddlg)
            dl_hi = float(np.interp(zmax, zg, dlg))
            lp_z = np.log(np.interp(z, zg, pv_z))
        m1src = np.asarray(ev.m1det, float) / (1 + z)
        lpp = np.asarray(logp_pop(jnp.asarray(m1src), jnp.asarray(ev.q), jnp.asarray(z), jnp.asarray(ev.chieff),
                                  pop_vec))
        pw = np.asarray(ev.prior_wt, float)
        out = lpp + lp_z - np.log(ddl) - np.log1p(z) - np.log(pw)
        ok = np.asarray(ev.valid, bool) & (pw > 0) & (dL <= dl_hi) & np.isfinite(out)
        return np.where(ok, out, -np.inf)

    from scipy.special import logsumexp
    N, n = m.n_events, m.nsamp
    lw = ldw(m.b.gw_pe).reshape(N, n)
    lse = logsumexp(lw, axis=1)
    lse2 = logsumexp(2 * lw, axis=1)
    lnZ = lse - np.log(n)
    var = np.maximum(np.exp(lse2 - 2 * lse) - 1.0 / n, 0.0)
    ls = ldw(m.b.gw_selection)
    L1, L2 = logsumexp(ls), logsumexp(2 * ls)
    log_mu = L1 - np.log(m.n_draw)
    n_eff = 1.0 / max(np.exp(L2 - 2 * L1) - 1.0 / m.n_draw, 1e-300)
    thr = max(5 * N, N ** 2 / max(float(cfg["max_var"]) - var.sum(), 1e-12))
    lnL = lnZ.sum() - N * log_mu + N * (N + 3) / (2 * n_eff) if n_eff > thr else -np.inf
    return {"lnL": float(lnL), "lnZ": lnZ, "var": var, "log_mu": float(log_mu), "n_eff": float(n_eff)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=50)
    ap.add_argument("--injections", type=int, default=300000)
    ap.add_argument("--nsamp", type=int, default=2000)
    args = ap.parse_args()
    os.environ["DARKSIRENS_ZMAX"] = repr(P.ZMAX_PRODUCTION)
    import prod_driver as PD
    cfg = PD.resolve(PD.parse_args(["--mode", "nested", "--analysis", "single_population", "--catalogs", "none",
                                    "--events-kind", "equal_populations", "--realization", "100",
                                    "--nsamp", str(args.nsamp), "--limit-events", str(args.events),
                                    "--limit-injections", str(args.injections), "--out-root", "/dev/null"]))
    m = PD.setup_model(cfg)
    import jax.numpy as jnp
    pts = []
    for H0 in (52.0, 60.0, 67.74, 80.0, 95.0):
        pts.append({"H0": H0, "mu_G": 35.0, "sigma_G": 5.0, "mu_chi": 0.0, "sigma_chi": 0.1})
    pts.append({"H0": 67.74, "mu_G": 30.0, "sigma_G": 3.0, "mu_chi": 0.05, "sigma_chi": 0.2})
    pts.append({"H0": 72.0, "mu_G": 41.0, "sigma_G": 7.5, "mu_chi": -0.1, "sigma_chi": 0.05})
    rows = []
    ok = True
    for c in pts:
        r = m.evaluate(c)
        b = m.bound_call(c)
        # core's per-event pieces through the diagnostics program
        out = m.diag(m.theta(c), *m.ops)
        ev_lnZ, ev_var = np.asarray(out[5]), np.asarray(out[6])
        row = {"coords": c, "core_lnL": r["lnL"], "core_bound_lnL": b, "core_log_mu": r["log_mu"],
               "core_n_eff": r["n_eff"]}
        for mode, tol in (("A", 1e-6), ("B", 1e-3 * args.events / 50)):
            d = direct_sum(m, c, mode)
            dl = abs(d["lnL"] - r["lnL"]) if np.isfinite(r["lnL"]) else (0.0 if d["lnL"] == r["lnL"] else np.inf)
            de = float(np.max(np.abs(d["lnZ"] - ev_lnZ)))
            dv = float(np.max(np.abs(d["var"] - ev_var)))
            dmu = abs(d["log_mu"] - r["log_mu"])
            dne = abs(d["n_eff"] / r["n_eff"] - 1)
            passed = bool(dl < tol and de < tol and dv < tol and dmu < tol and dne < tol)
            ok &= passed
            row[mode] = {"direct_lnL": d["lnL"], "abs_dlnL": dl, "max_abs_d_event_lnZ": de,
                         "max_abs_d_event_var": dv, "abs_d_log_mu": dmu, "rel_d_n_eff": dne,
                         "tolerance": tol, "pass": passed}
        print(f"{c}: core {r['lnL']:.8f}  A {row['A']['direct_lnL']:.8f} (|d| {row['A']['abs_dlnL']:.2e})  "
              f"B {row['B']['direct_lnL']:.8f} (|d| {row['B']['abs_dlnL']:.2e})", flush=True)
        rows.append(row)
    # core's tabulated redshift prior against the numpy volume element itself
    from darksirens.cosmology.parameters import CosmologyParameters
    from darksirens.cosmology.volume import log_comoving_volume_prior
    pv_rows = []
    for H0 in (52.0, 67.74, 95.0):
        zg, _, _, pvz = cosmology_numpy(H0, P.OM0, float(cfg["zmax"]))
        zt = np.linspace(0.01, float(cfg["zmax"]) - 1e-3, 2000)
        core_lp = np.asarray(log_comoving_volume_prior(jnp.asarray(zt), CosmologyParameters(H0, P.OM0, -1.0, 0.0)))
        d = core_lp - np.log(np.interp(zt, zg, pvz))
        pv_rows.append({"H0": H0, "max_abs_d_ln_pV_0.01_to_zmax": float(np.max(np.abs(d))),
                        "max_abs_d_ln_pV_z_below_0.1": float(np.max(np.abs(d[zt < 0.1]))),
                        "median_abs_d": float(np.median(np.abs(d)))})
    print(f"core ln p_V(z) vs numpy dV_c/dz normalised: {pv_rows}")
    res = {"test": "no-catalog limit = hosts uniform in comoving volume (direct sum)", "pass": bool(ok),
           "core_volume_table_vs_numpy": pv_rows,
           "n_events": m.n_events, "nsamp": m.nsamp, "n_detected_injections": m.info["n_detected_injections"],
           "n_draw": m.n_draw, "zmax": cfg["zmax"], "max_var": cfg["max_var"], "rows": rows,
           "model_info": {k: v for k, v in m.info.items() if k != "catalog_model"},
           "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = A16 / "tests" / "results" / "nocatalog_identity.json"
    out.write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"PASS={ok} -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
