#!/usr/bin/env python
"""darksirens-core e7c3007 on CPU, in its own venv, for either mock family.

    PY=/hildafs/projects/phy230014p/magana/envs/darksirens-core-e7c3007/bin/python
    JAX_PLATFORMS=cpu PYTHONPATH= $PY core_family.py identity            # obslike: core vs closed form
    JAX_PLATFORMS=cpu PYTHONPATH= $PY core_family.py truth OUT.json FAMILY FILE...   # ln p at truths
    JAX_PLATFORMS=cpu PYTHONPATH= $PY core_family.py points FAMILY IN.npz OUT.npz    # ln p at points

FAMILY is ``mock`` (the present family, corepop/README.md) or ``obslike`` (the GWTC-3
power law + peak mapped onto core, obslike/README.md).  Both are core's powerlaw+peak with
shared beta/spin/gamma; the theta vectors are below in core's order
(v1 alpha m_min m_max dm_min dm_max mu_G sigma_G beta mu_chi sigma_chi gamma), v1 being
the POWER-LAW weight.  The host-dependent set's AGN class is the same vector with
mu_G + 5 and mu_chi + 0.10.

``identity`` compares core's ln p with the closed form the generator's core route samples
(corepop/scripts/core_eval.py ClosedForm, scipy quadrature, independent of both codes) at
the obslike theta, checks that core's p(m1, q) integrates to one, and tabulates the 1-D
marginals and a 2-D (m1, q) grid used by the sampler and detected-set checks.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np

HERE = Path(__file__).resolve().parent
OB = HERE.parent
sys.path.insert(0, str(OB.parent / "corepop" / "scripts"))
import core_eval as CE  # noqa: E402  (closed form, s_low/s_high; imports nothing from darksirens)

CORE_SRC = "/hildafs/projects/phy230014p/magana/src/darksirens-core-e7c3007"
NAMES = ["v1", "alpha", "m_min", "m_max", "dm_min", "dm_max", "mu_G", "sigma_G", "beta",
         "mu_chi", "sigma_chi", "gamma"]
FAMILIES = {
    "mock": dict(v1=0.1, alpha=2.3, m_min=5.0, m_max=80.0, dm_min=3.0, dm_max=10.0,
                 mu_G=35.0, sigma_G=5.0, beta=1.0, mu_chi=0.0, sigma_chi=0.1, gamma=0.0),
    "obslike": dict(v1=0.908, alpha=3.5, m_min=5.0, m_max=87.0, dm_min=4.9, dm_max=0.01,
                    mu_G=34.0, sigma_G=3.6, beta=1.1, mu_chi=0.0, sigma_chi=0.1, gamma=0.0),
}
AGN_SHIFT = dict(mu_G=5.0, mu_chi=0.10)


def thetas(family, s_mult=1.0):
    gal = dict(FAMILIES[family])
    agn = dict(gal)
    agn["mu_G"] += s_mult * AGN_SHIFT["mu_G"]
    agn["mu_chi"] += s_mult * AGN_SHIFT["mu_chi"]
    return {"gal": gal, "agn": agn}


def setup(settings, family, s_mult=1.0):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import darksirens as ds
    from darksirens.population import get_model
    from darksirens.population.utils import configure_normalization_grids, normalization_grid_settings
    assert CORE_SRC in ds.__file__, ds.__file__
    for v in ("DARKSIRENS_GW_PAIRING_NORM", "DARKSIRENS_GW_PAIRING_SCALE"):
        assert v not in os.environ, v
    configure_normalization_grids(pairing_norm="per_sample" if settings == "historical" else "auto",
                                  pairing_scale="analytic")
    ng = normalization_grid_settings()
    probe = ds.model(cosmology=ds.Cosmology(H0=(50.0, 100.0), Om0=0.3075),
                     population=ds.Population("powerlaw+peak", fixed=True)).parameters
    labels = list(probe.population_labels)
    assert len(labels) == len(NAMES), labels
    th = {k: np.array([v[n] for n in NAMES], float) for k, v in thetas(family, s_mult).items()}
    model = get_model("powerlaw+peak", shared_beta=True, shared_spin=True, shared_gamma=True)
    fn = jax.jit(lambda m1, q, z, chi, t: model.log_p_pop(m1, q, z, chi, t))

    def lp(m1, q, chi, z, pop, chunk=1_000_000):
        m1, q, chi, z = (np.broadcast_to(np.asarray(a, float), np.broadcast(m1, q, chi, z).shape)
                         for a in (m1, q, chi, z))
        shp = m1.shape
        m1, q, chi, z = (a.ravel() for a in (m1, q, chi, z))
        out = np.empty(m1.size)
        t = jnp.asarray(th[pop])
        for i in range(0, m1.size, chunk):
            s = slice(i, min(i + chunk, m1.size))
            n = s.stop - s.start
            pad = chunk - n if m1.size > chunk else 0
            a = [np.concatenate([x[s], np.full(pad, y)]) for x, y in
                 ((m1, 30.0), (q, 0.8), (z, 0.1), (chi, 0.0))]
            out[s] = np.asarray(fn(*[jnp.asarray(x) for x in a], t))[:n]
        return out.reshape(shp)

    info = {"darksirens_file": ds.__file__, "settings": settings, "family": family,
            "labels": labels, "names": NAMES, "pairing_norm": ng.pairing_norm,
            "theta_gal": th["gal"].tolist(), "theta_agn": th["agn"].tolist()}
    return lp, info


def closed_forms(family):
    out = {}
    for pop, P in thetas(family).items():
        cf = CE.ClosedForm(P["mu_G"], P["mu_chi"], P=P)
        out[pop] = cf
    return out


# ---------------------------------------------------------------------- identity
def cmd_identity(a):
    fam = "obslike"
    P = FAMILIES[fam]
    rng = np.random.default_rng(20261118)
    n = 60_000
    lo, d1, hi = P["m_min"], P["dm_min"], P["m_max"]
    m1 = np.concatenate([rng.uniform(lo, 100.0, n), rng.uniform(lo, lo + 3 * d1, n // 3),
                         rng.uniform(25.0, 45.0, n // 3), rng.uniform(hi - 1.0, hi + 1.0, n // 6)])
    qlo = lo / m1
    k = n
    q = np.concatenate([qlo[:k] + (1 - qlo[:k]) * rng.uniform(size=k),
                        qlo[k:] + (np.minimum((lo + d1) / m1[k:], 1.0) - qlo[k:]) * rng.uniform(size=m1.size - k)])
    chi = rng.uniform(-0.6, 0.7, m1.size)
    z = rng.uniform(0.0, 0.6, m1.size)
    m2 = q * m1
    # where S_low < e^-130 (core's exponent cap; closed form uses the same cap)
    floor_zone = (m2 - lo) < d1 * 2.0 / (132.0 + math.sqrt(130.0 ** 2 + 4.0))
    out = {"what": "core powerlaw+peak at the obslike theta vs the closed form the generator's "
                   "core route samples; normalisation; marginals",
           "family": fam, "theta_order": NAMES, "thetas": thetas(fam)}
    cfs = closed_forms(fam)
    npz = {}
    for settings in ("historical", "default"):
        lp, info = setup(settings, fam)
        out.setdefault("core", {})[settings] = info
        for pop in ("gal", "agn"):
            A = lp(m1, q, chi, z, pop)
            B = cfs[pop].lnp(m1, q, chi, z)
            fin = np.isfinite(A) & np.isfinite(B)
            d = np.abs(A - B)
            rec = {"n_points": int(m1.size), "n_both_finite": int(fin.sum()),
                   "n_finite_mismatch": int((np.isfinite(A) != np.isfinite(B)).sum()),
                   "max_abs_dlnp_outside_floor_zone": float(d[fin & ~floor_zone].max()),
                   "max_abs_dlnp_all": float(d[fin].max()),
                   "max_abs_dlnp_m1_near_mmax": float(d[fin & (np.abs(m1 - hi) < 1)].max()),
                   "max_abs_dlnp_taper_band": float(d[fin & ~floor_zone & (m2 < lo + d1)].max()),
                   "median_abs_dlnp": float(np.median(d[fin]))}
            out.setdefault("identity", {}).setdefault(settings, {})[pop] = rec
            print(f"[identity {settings} {pop}] {rec}", flush=True)
            if settings != "historical":
                continue
            # normalisation and marginals at z = 0 (rate factor 1); the chi factor is
            # divided out at its mode
            P_ = thetas(fam)[pop]
            muchi = P_["mu_chi"]
            chig = np.linspace(-1.0, 1.0, 20001)
            i0 = int(np.argmin(np.abs(chig - muchi)))
            one = np.ones(chig.size)
            lpc = lp(35.0 * one, 0.8 * one, chig, 0.0 * one, pop)
            r = np.exp(lpc - lpc[i0])
            p_chi = r / np.trapz(r, chig)
            lpchi0 = float(np.log(p_chi[i0]))
            m1g = np.concatenate([np.linspace(5.0, 10.0, 2001)[:-1], np.linspace(10.0, 100.0, 3601)])
            qg = np.linspace(0.04, 1.0, 3841)
            M1, Q = np.meshgrid(m1g, qg, indexing="ij")
            p_mq = np.exp(lp(M1, Q, 0.0 * M1 + chig[i0], 0.0 * M1, pop) - lpchi0)
            pm1 = np.trapz(p_mq, qg, axis=1)
            pq = np.trapz(p_mq, m1g, axis=0)
            total = float(np.trapz(pm1, m1g))
            cq = np.concatenate([np.zeros((m1g.size, 1)),
                                 np.cumsum(0.5 * (p_mq[:, 1:] + p_mq[:, :-1]) * np.diff(qg)[None, :], axis=1)],
                                axis=1)
            m2g = np.linspace(4.0, 100.0, 4801)
            inner = np.empty((m1g.size, m2g.size))
            for i, m in enumerate(m1g):
                inner[i] = np.interp(m2g / m, qg, cq[i], left=0.0, right=cq[i, -1])
            F_m2 = np.trapz(inner, m1g, axis=0)
            # peak share and quantiles of the primary
            cdf1 = np.concatenate([[0.0], np.cumsum(0.5 * (pm1[1:] + pm1[:-1]) * np.diff(m1g))])
            cdf1 /= cdf1[-1]
            out.setdefault("marginals", {})[pop] = {
                "integral_p_m1_q": total,
                "P_m2_lt_5": float(np.interp(5.0, m2g, F_m2) / F_m2[-1]),
                "P_m2_lt_6": float(np.interp(6.0, m2g, F_m2) / F_m2[-1]),
                "P_m2_lt_8": float(np.interp(8.0, m2g, F_m2) / F_m2[-1]),
                "m1_median": float(np.interp(0.5, cdf1, m1g)),
                "m1_q01": float(np.interp(0.01, cdf1, m1g)),
                "m1_q99": float(np.interp(0.99, cdf1, m1g)),
                "chi_mean": float(np.trapz(chig * p_chi, chig))}
            print(f"[marginals {pop}] {out['marginals'][pop]}", flush=True)
            npz.update({"m1g": m1g, "qg": qg, "m2g": m2g, f"pm1_{pop}": pm1, f"pq_{pop}": pq,
                        f"Fm2_{pop}": F_m2, f"pchi_{pop}": p_chi})
            m1d = np.concatenate([np.linspace(5.0, 10.0, 501)[:-1], np.linspace(10.0, 100.0, 1801)])
            qd = np.linspace(0.04, 1.0, 961)
            M1d, Qd = np.meshgrid(m1d, qd, indexing="ij")
            npz.update({"m1d": m1d, "qd": qd,
                        f"lnp2d_{pop}": lp(M1d, Qd, 0.0 * M1d + chig[i0], 0.0 * M1d, pop) - lpchi0})
            del M1, Q, p_mq, cq, inner
    (OB / "cache").mkdir(exist_ok=True)
    np.savez_compressed(OB / "cache" / "core_tables_obslike.npz", **npz)
    (OB / "results" / "density_identity.json").write_text(json.dumps(out, indent=1))
    print("wrote", OB / "results" / "density_identity.json")


# ------------------------------------------------------------------------- truth
def cmd_truth(a):
    import h5py
    lps_by = {}
    from darksirens.gw.samples import load_gw_store
    res = {"what": f"core ln p_pop at every event's TRUE parameters, family {a.family} "
                   "(host-dependent set: each event under its host class's population); "
                   "core's own loader on each file", "family": a.family,
           "thetas": thetas(a.family), "files": []}
    for p in a.files:
        with h5py.File(p, "r") as f:
            t = f["truth"]
            m1, q, z, chi = (np.asarray(t[k]) for k in ("m1src", "q", "z", "chieff"))
            ht = np.asarray(t["host_type"])
            m2 = np.asarray(t["m2src"])
            marked = float(f.attrs.get("dmu_G_agn", 0.0)) != 0.0
            s_mult = float(f.attrs.get("dmu_G_agn", 0.0)) / AGN_SHIFT["mu_G"] if marked else 1.0
            if marked and abs(float(f.attrs["dmu_chi_agn"]) - s_mult * AGN_SHIFT["mu_chi"]) > 1e-12:
                raise SystemExit(f"{p}: offsets not in the 5 Msun : 0.10 ratio")
            m2s = np.asarray(f["m2src"]).reshape(int(f.attrs["nobs"]), int(f.attrs["nsamp"]))
        if s_mult not in lps_by:
            lps_by[s_mult] = {st: setup(st, a.family, s_mult)[0] for st in ("historical", "default")}
        lps = lps_by[s_mult]
        row = {"file": str(p), "n_events": int(m1.size), "host_dependent": marked, "offset_s": s_mult if marked else 0,
               "min_true_m2src": float(m2.min()), "n_true_m2_lt_5": int((m2 < 5).sum()),
               "max_frac_samples_m2_lt_5": float((m2s < 5.0).mean(axis=1).max())}
        for st, lp in lps.items():
            v = (np.where(ht == 1, lp(m1, q, chi, z, "agn"), lp(m1, q, chi, z, "gal"))
                 if marked else lp(m1, q, chi, z, "gal"))
            fin = np.isfinite(v)
            row[st] = {"all_finite": bool(fin.all()), "n_finite": int(fin.sum()),
                       "min_lnp": float(v.min()), "sum_lnp": float(v[fin].sum())}
        try:
            import contextlib
            import io
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                st = load_gw_store(str(p))
            rs = np.asarray(st.prior_wt).reshape(int(st.n_events), int(st.nsamp)).sum(axis=1)
            row["core_loader"] = {"ok": True, "n_events": int(st.n_events), "nsamp": int(st.nsamp),
                                  "p_pe_row_sum_max_abs_dev": float(np.abs(rs - 1).max())}
        except Exception as e:  # noqa: BLE001
            row["core_loader"] = {"ok": False, "error": repr(e)[:400]}
        res["files"].append(row)
        print(Path(p).name, row["min_true_m2src"], row["historical"]["all_finite"],
              row["default"]["all_finite"], row["core_loader"]["ok"], flush=True)
    res["all_pass"] = bool(all(r["historical"]["all_finite"] and r["default"]["all_finite"]
                               and r["n_true_m2_lt_5"] == 0 and r["core_loader"]["ok"]
                               and r["n_events"] == 1000 for r in res["files"]))
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote", a.out, "all_pass", res["all_pass"])


def cmd_points(a):
    d = np.load(a.inp)
    lp, info = setup("historical", a.family)
    np.savez_compressed(a.out, lnp_core_gal=lp(d["m1src"], d["q"], d["chieff"], d["z"], "gal"),
                        lnp_core_agn=lp(d["m1src"], d["q"], d["chieff"], d["z"], "agn"))
    print("wrote", a.out, info["family"], info["settings"])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("identity")
    t = sub.add_parser("truth")
    t.add_argument("out")
    t.add_argument("family", choices=list(FAMILIES))
    t.add_argument("files", nargs="+")
    p = sub.add_parser("points")
    p.add_argument("family", choices=list(FAMILIES))
    p.add_argument("inp")
    p.add_argument("out")
    a = ap.parse_args()
    {"identity": cmd_identity, "truth": cmd_truth, "points": cmd_points}[a.cmd](a)


if __name__ == "__main__":
    main()
