#!/usr/bin/env python
"""darksirens-core e7c3007, evaluated on CPU in its own venv: everything corepop needs
from core itself.  Never imports legacy darksirens.

    PY=/hildafs/projects/phy230014p/magana/envs/darksirens-core-e7c3007/bin/python
    JAX_PLATFORMS=cpu $PY core_eval.py tables     # density identity + marginals + 2-D grid
    JAX_PLATFORMS=cpu $PY core_eval.py truth      # sum ln p_pop over every set's truths; core loader
    JAX_PLATFORMS=cpu $PY core_eval.py points IN.npz OUT.npz   # ln p_pop at given points

THE DENSITY.  powerlaw+peak, shared beta/spin/gamma, at the mock's input values with
gamma = 0 (core's fixed=True vector has gamma = 2.5, so gamma is set by hand):
theta = [v1 0.1, alpha 2.3, m_min 5, m_max 80, dm_min 3, dm_max 10, mu_G 35, sigma_G 5,
beta 1, mu_chi 0, sigma_chi 0.1, gamma 0].  The AGN branch of the host-dependent set is
the same vector with mu_G 40 and mu_chi 0.1 (core's mu_G,c2 / mu_chi,c2).

Two evaluation settings: "historical" (pairing_norm per_sample, pairing_scale analytic:
what Analyses 13/14 run) and "default" (core's own defaults: pairing_norm auto =
per_point).  Both are core's density; they differ by the pairing quadrature only.

`tables` also re-derives the density in closed form with scipy (independent of both
codes) and reports max |ln p_core - ln p_closed| -- the closed form being exactly the
density the generator's --population_code core route samples (legacy primary-mass
mixture x q**beta S_low(q m1; 5, 3)/N(m1) x truncated Gaussian x (1+z)^-1).
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
CP = HERE.parent
DATA = Path("/hildafs/projects/phy230054p/magana/gws-agn-data-v3")
CORE_SRC = "/hildafs/projects/phy230014p/magana/src/darksirens-core-e7c3007"
SEEDS = (100, 101, 102, 103, 105)
KINDS = {"mixed": ("events_corepop.h5", "events.h5"),
         "pureagn": ("events_pureagn_corepop.h5", "events_pureagn.h5"),
         "puregal": ("events_puregal_corepop.h5", "events_puregal.h5"),
         "marked": ("events_marked_dmu0p10_dmuG5_corepop.h5", "events_marked_dmu0p10_dmuG5.h5")}
MOCK = dict(v1=0.1, alpha=2.3, m_min=5.0, m_max=80.0, dm_min=3.0, dm_max=10.0,
            mu_G=35.0, sigma_G=5.0, beta=1.0, mu_chi=0.0, sigma_chi=0.1, gamma=0.0)
AGN_SHIFT = dict(mu_G=5.0, mu_chi=0.10)


# ------------------------------------------------------------------ core handles
def core_setup(settings):
    import jax
    jax.config.update("jax_enable_x64", True)
    import darksirens as ds
    from darksirens.population import get_model
    from darksirens.population.utils import configure_normalization_grids
    assert CORE_SRC in ds.__file__, ds.__file__
    # explicit in both arms: an argument left out keeps the CURRENT setting
    for v in ("DARKSIRENS_GW_PAIRING_NORM", "DARKSIRENS_GW_PAIRING_SCALE"):
        assert v not in os.environ, v
    if settings == "historical":
        configure_normalization_grids(pairing_norm="per_sample", pairing_scale="analytic")
    else:
        configure_normalization_grids(pairing_norm="auto", pairing_scale="analytic")
    from darksirens.population.utils import normalization_grid_settings
    _ng = normalization_grid_settings()
    probe = ds.model(cosmology=ds.Cosmology(H0=(50.0, 100.0), Om0=0.3075),
                     population=ds.Population("powerlaw+peak", fixed=True)).parameters
    labels = list(probe.population_labels)
    theta = np.asarray(probe.fixed_population, float)
    names = ["v1", "alpha", "m_min", "m_max", "dm_min", "dm_max", "mu_G", "sigma_G",
             "beta", "mu_chi", "sigma_chi", "gamma"]
    assert len(labels) == len(names), labels
    fixed = dict(zip(names, theta.tolist()))
    for k, v in MOCK.items():
        if k != "gamma":
            assert abs(fixed[k] - v) < 1e-12, (k, fixed[k], v)
    th = {"gal": np.array([MOCK[k] for k in names], float)}
    agn = dict(MOCK)
    agn["mu_G"] += AGN_SHIFT["mu_G"]
    agn["mu_chi"] += AGN_SHIFT["mu_chi"]
    th["agn"] = np.array([agn[k] for k in names], float)
    model = get_model("powerlaw+peak", shared_beta=True, shared_spin=True, shared_gamma=True)
    import jax.numpy as jnp
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
            pad = chunk - n if m1.size > chunk else 0      # one compiled shape
            a = [np.concatenate([x[s], np.full(pad, y)]) for x, y in
                 ((m1, 30.0), (q, 0.8), (z, 0.1), (chi, 0.0))]
            out[s] = np.asarray(fn(*[jnp.asarray(x) for x in a], t))[:n]
        return out.reshape(shp)

    info = {"darksirens_file": ds.__file__, "settings": settings, "labels": labels,
            "pairing_norm": _ng.pairing_norm, "pairing_scale": _ng.pairing_scale,
            "per_point_pairing_norm": bool(_ng.per_point_pairing_norm()),
            "theta_gal": th["gal"].tolist(), "theta_agn": th["agn"].tolist()}
    return lp, info


# ---------------------------------------------------------- closed form (scipy)
CAP, FLOOR = 130.0, 1e-140


def s_low(m, m_min, dm, cap=CAP):
    m = np.asarray(m, float)
    d = m - m_min
    sd = np.where(d > 0, np.maximum(d, FLOOR), 1.0)
    x = sd - dm
    xf = np.where(x < 0, -np.maximum(np.abs(x), FLOOR), np.maximum(np.abs(x), FLOOR))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        e = np.clip(dm / sd + dm / xf, -cap, cap)
        s = 1.0 / (np.exp(e) + 1.0)
    s = np.where(m <= m_min, 0.0, s)
    return np.where(m >= m_min + dm, 1.0, s)


def s_high(m, m_max, dm, cap=CAP):
    m = np.asarray(m, float)
    d = m_max - m
    sd = np.where(d > 0, np.maximum(d, FLOOR), 1.0)
    x = sd - dm
    xf = np.where(x < 0, -np.maximum(np.abs(x), FLOOR), np.maximum(np.abs(x), FLOOR))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        e = np.clip(dm / sd + dm / xf, -cap, cap)
        s = 1.0 / (np.exp(e) + 1.0)
    s = np.where(m >= m_max, 0.0, s)
    return np.where(m <= m_max - dm, 1.0, s)


class ClosedForm:
    """The route's sampling density, normalised by adaptive quadrature."""

    def __init__(self, mu_G, mu_chi, P=MOCK):
        from scipy.integrate import quad
        from scipy.special import ndtr
        self.P, self.mu_G, self.mu_chi = P, mu_G, mu_chi
        a, lo, hi, d1, d2 = P["alpha"], P["m_min"], P["m_max"], P["dm_min"], P["dm_max"]
        f = lambda m: float(s_low(m, lo, d1) * s_high(m, hi, d2)) * m ** (-a)
        pts = [lo + d1, hi - d2]
        self.n_pl = sum(quad(f, x, y, epsabs=0, epsrel=1e-13, limit=500)[0]
                        for x, y in zip([lo] + pts, pts + [hi]))
        s = P["sigma_G"]
        self.n_g = s * math.sqrt(2 * math.pi) * (ndtr((200 - mu_G) / s) - ndtr((1 - mu_G) / s))
        sc = P["sigma_chi"]
        self.n_chi = sc * math.sqrt(2 * math.pi) * (ndtr((1 - mu_chi) / sc) - ndtr((-1 - mu_chi) / sc))
        b = P["beta"]
        g = lambda m2: float(s_low(m2, lo, d1)) * m2 ** b
        self.h_shoulder = quad(g, lo, lo + d1, epsabs=0, epsrel=1e-13, limit=500)[0]
        self._quad, self._g = quad, g

    def h(self, x):
        """H(x) = int_{m_min}^{x} m2^beta S_low(m2) dm2; N(m1) = H(m1) / m1^(1+beta)."""
        P = self.P
        lo, d1, b = P["m_min"], P["dm_min"], P["beta"]
        out = np.empty(np.size(x))
        for i, xi in enumerate(np.atleast_1d(x)):
            if xi <= lo:
                out[i] = 0.0
            elif xi < lo + d1:
                out[i] = self._quad(self._g, lo, xi, epsabs=0, epsrel=1e-13, limit=500)[0]
            else:
                out[i] = self.h_shoulder + (xi ** (b + 1) - (lo + d1) ** (b + 1)) / (b + 1)
        return out

    def lnp(self, m1, q, chi, z):
        P = self.P
        m1, q, chi, z = (np.asarray(a, float) for a in (m1, q, chi, z))
        w_pl, w_g = P["v1"], 1 - P["v1"]
        pl = (s_low(m1, P["m_min"], P["dm_min"]) * s_high(m1, P["m_max"], P["dm_max"])
              * m1 ** (-P["alpha"]) / self.n_pl)
        g = np.where((m1 >= 1) & (m1 <= 200),
                     np.exp(-0.5 * ((m1 - self.mu_G) / P["sigma_G"]) ** 2) / self.n_g, 0.0)
        hh = self.h(m1)
        with np.errstate(divide="ignore", invalid="ignore"):
            pair = np.where((q > 0) & (q <= 1) & (hh > 0),
                            q ** P["beta"] * s_low(q * m1, P["m_min"], P["dm_min"])
                            * m1 ** (1 + P["beta"]) / np.where(hh > 0, hh, 1.0), 0.0)
            chi_p = np.where(np.abs(chi) <= 1,
                             np.exp(-0.5 * ((chi - self.mu_chi) / P["sigma_chi"]) ** 2) / self.n_chi,
                             0.0)
            return (np.log(w_pl * pl + w_g * g) + np.log(pair) + np.log(chi_p)
                    + (P["gamma"] - 1.0) * np.log1p(z))


# ---------------------------------------------------------------------- tables
def cmd_tables(args):
    out = {"what": "core's powerlaw+peak density at the mock's input (gamma 0): identity with "
                   "the closed form the generator's core route samples, normalisation, and the "
                   "1-D marginals the sampler and detected-set checks use",
           "theta_order": "v1 alpha m_min m_max dm_min dm_max mu_G sigma_G beta mu_chi sigma_chi gamma"}
    rng = np.random.default_rng(20261117)
    npz = {}
    # identity points: uniform over the support, plus the taper band, plus the edges
    n = 60_000
    m1 = np.concatenate([rng.uniform(5.0, 100.0, n), rng.uniform(5.0, 12.0, n // 3),
                         rng.uniform(25.0, 55.0, n // 3)])
    qlo = 5.0 / m1
    q = np.concatenate([qlo[:n] + (1 - qlo[:n]) * rng.uniform(size=n),
                        qlo[n:] + (np.minimum(8.0 / m1[n:], 1.0) - qlo[n:])
                        * rng.uniform(size=m1.size - n)])
    chi = rng.uniform(-0.6, 0.7, m1.size)
    z = rng.uniform(0.0, 0.5, m1.size)
    m2 = q * m1
    floor_zone = (m2 - 5.0) < 3.0 * 2.0 / (132.0 + math.sqrt(130.0 ** 2 + 4.0))  # S at cap
    for settings in ("historical", "default"):
        lp, info = core_setup(settings)
        out.setdefault("core", {})[settings] = info
        for pop, (mug, muchi) in (("gal", (35.0, 0.0)), ("agn", (40.0, 0.1))):
            cf = ClosedForm(mug, muchi)
            a = lp(m1, q, chi, z, pop)
            b = cf.lnp(m1, q, chi, z)
            fin = np.isfinite(a) & np.isfinite(b)
            d = np.abs(a - b)[fin]
            taper = (m2 > 5.0) & (m2 < 8.0) & ~floor_zone
            plateau = m2 >= 8.0
            iw = int(np.flatnonzero(fin)[np.argmax(d)])
            rec = {"n_points": int(m1.size), "n_both_finite": int(fin.sum()),
                   "n_finite_mismatch": int((np.isfinite(a) != np.isfinite(b)).sum()),
                   "max_abs_dlnp_all": float(d.max()),
                   "argmax_all": {"m1": float(m1[iw]), "m2": float(m2[iw]),
                                  "in_floor_zone": bool(floor_zone[iw]),
                                  "core": float(a[iw]), "closed_form": float(b[iw])},
                   "max_abs_dlnp_outside_floor_zone": float(np.abs(a - b)[fin & ~floor_zone].max()),
                   "max_abs_dlnp_m2_ge_8": float(np.abs(a - b)[fin & plateau].max()),
                   "max_abs_dlnp_taper_5_to_8": float(np.abs(a - b)[fin & taper].max()),
                   "median_abs_dlnp": float(np.median(d)),
                   "n_points_in_floor_zone": int(floor_zone.sum()),
                   "core_lnp_below_m2_5": "-inf by construction (no point there)"}
            # explicit edge points: m2 just below / above 5, and m1 <= 5
            e_m1 = np.array([33.7173, 33.7173, 33.7173, 33.7173, 4.9, 35.0])
            e_q = np.array([3.3651 / 33.7173, 4.9 / 33.7173, 5.1 / 33.7173, 6.0 / 33.7173, 0.9,
                            30.0 / 35.0])
            e_chi = np.array([0.0723, 0.0723, 0.0723, 0.0723, 0.0, 0.0])
            e_z = np.array([0.0627, 0.0627, 0.0627, 0.0627, 0.1, 0.1])
            rec["edge_points"] = {"m1": e_m1.tolist(), "m2": (e_m1 * e_q).tolist(),
                                  "core": lp(e_m1, e_q, e_chi, e_z, pop).tolist(),
                                  "closed_form": cf.lnp(e_m1, e_q, e_chi, e_z).tolist()}
            out.setdefault("identity", {}).setdefault(settings, {})[pop] = rec
            print(f"[identity {settings} {pop}] max|dlnp| all {rec['max_abs_dlnp_all']:.3e} "
                  f"plateau {rec['max_abs_dlnp_m2_ge_8']:.3e} taper {rec['max_abs_dlnp_taper_5_to_8']:.3e}",
                  flush=True)

            # 1-D marginals of core's density on fine grids, at z = 0 (rate factor 1).
            # core's p = p(m1, q) p(chi) (shared spin): the chi factor is read off a
            # chi scan at fixed (m1, q), normalised, and divided out of the mass grid.
            chig = np.linspace(-1.0, 1.0, 20001)
            i0 = int(np.argmin(np.abs(chig - muchi)))
            chi0 = float(chig[i0])
            one = np.ones(chig.size)
            lpc = lp(35.0 * one, 0.8 * one, chig, 0.0 * one, pop)
            r = np.exp(lpc - lpc[i0])                       # p_chi(chi) / p_chi(chi0)
            p_chi = r / np.trapz(r, chig)
            lpchi0 = float(np.log(p_chi[i0]))
            m1g = np.linspace(5.0, 100.0, 4751)
            qg = np.linspace(0.04, 1.0, 3841)
            M1, Q = np.meshgrid(m1g, qg, indexing="ij")
            p_mq = np.exp(lp(M1, Q, 0.0 * M1 + chi0, 0.0 * M1, pop) - lpchi0)
            pm1 = np.trapz(p_mq, qg, axis=1)
            pq = np.trapz(p_mq, m1g, axis=0)
            total = float(np.trapz(pm1, m1g))
            # m2 = q m1: CDF F(t) = int dm1 int_{q <= t/m1} p(m1, q) dq, from each row's
            # cumulative q-integral (smooth in q), then trapezoid over m1
            cq = np.concatenate([np.zeros((m1g.size, 1)),
                                 np.cumsum(0.5 * (p_mq[:, 1:] + p_mq[:, :-1]) * np.diff(qg)[None, :],
                                           axis=1)], axis=1)
            m2g = np.linspace(4.0, 100.0, 4801)
            inner = np.empty((m1g.size, m2g.size))
            for i, m in enumerate(m1g):
                inner[i] = np.interp(m2g / m, qg, cq[i], left=0.0, right=cq[i, -1])
            F_m2 = np.trapz(inner, m1g, axis=0)
            pm2 = np.gradient(F_m2, m2g)
            del M1, Q, p_mq, cq, inner
            key = f"{settings}_{pop}"
            npz.update({f"m1g": m1g, f"qg": qg, f"m2g": m2g, f"chig": chig,
                        f"pm1_{key}": pm1, f"pq_{key}": pq, f"pm2_{key}": pm2,
                        f"Fm2_{key}": F_m2, f"pchi_{key}": p_chi})
            cdf_m2 = F_m2
            out.setdefault("marginals", {}).setdefault(settings, {})[pop] = {
                "integral_p_m1_q": total,
                "integral_p_m2": float(cdf_m2[-1]),
                "P_m2_lt_5": float(np.interp(5.0, m2g, cdf_m2) / cdf_m2[-1]),
                "P_m2_lt_6": float(np.interp(6.0, m2g, cdf_m2) / cdf_m2[-1]),
                "P_m2_lt_8": float(np.interp(8.0, m2g, cdf_m2) / cdf_m2[-1]),
                "chi_mean": float(np.trapz(chig * p_chi, chig)),
                "chi_sd": float(np.sqrt(np.trapz((chig - np.trapz(chig * p_chi, chig)) ** 2 * p_chi,
                                                 chig)))}
            print(f"[marginals {settings} {pop}] {out['marginals'][settings][pop]}", flush=True)
            if settings == "historical":
                # 2-D grid for the detected-set prediction (ln p at chi mode, z = 0)
                m1d = np.linspace(5.0, 100.0, 1901)
                qd = np.linspace(0.04, 1.0, 961)
                M1d, Qd = np.meshgrid(m1d, qd, indexing="ij")
                l2 = lp(M1d, Qd, 0.0 * M1d + chi0, 0.0 * M1d, pop) - lpchi0
                npz.update({"m1d": m1d, "qd": qd, f"lnp2d_{pop}": l2.astype(np.float64)})
    np.savez_compressed(CP / "cache" / "core_tables.npz", **npz)
    (CP / "results" / "density_identity.json").write_text(json.dumps(out, indent=1))
    print("wrote", CP / "cache" / "core_tables.npz", CP / "results" / "density_identity.json")


# ----------------------------------------------------------------------- truth
def cmd_truth(args):
    import h5py
    res = {"what": "core ln p_pop(m1src, q, z, chi_eff) at every event's TRUE parameters "
                   "(host-dependent set: each event under its host class's population); "
                   "finite for every event of every corepop file is the requirement",
           "files": []}
    lps = {s: core_setup(s)[0] for s in ("historical", "default")}
    from darksirens.gw.samples import load_gw_store
    for kind, (new, old) in KINDS.items():
        for s in SEEDS:
            for tag, fn in (("corepop", new), ("legacy", old)):
                p = DATA / f"seed{s}" / "events" / fn
                if not p.exists():
                    continue
                with h5py.File(p, "r") as f:
                    t = f["truth"]
                    m1, q, z, chi = (np.asarray(t[k]) for k in ("m1src", "q", "z", "chieff"))
                    ht = np.asarray(t["host_type"])
                    m2 = np.asarray(t["m2src"])
                row = {"seed": s, "kind": kind, "population_code": tag, "file": fn,
                       "n_events": int(m1.size), "min_true_m2src": float(m2.min())}
                for st, lp in lps.items():
                    v = np.where(ht == 1, lp(m1, q, chi, z, "agn"), lp(m1, q, chi, z, "gal")) \
                        if kind == "marked" else lp(m1, q, chi, z, "gal")
                    fin = np.isfinite(v)
                    row[st] = {"n_finite": int(fin.sum()), "sum_lnp_finite": float(v[fin].sum()),
                               "min_lnp": float(v.min()), "argmin": int(np.argmin(v)),
                               "all_finite": bool(fin.all())}
                if tag == "corepop":
                    try:
                        import contextlib
                        import io
                        buf = io.StringIO()
                        with contextlib.redirect_stdout(buf):
                            st = load_gw_store(str(p))
                        ppe = np.asarray(st.prior_wt)        # the loader's normalised p_pe
                        nob, nsa = int(st.n_events), int(st.nsamp)
                        rs = ppe.reshape(nob, nsa).sum(axis=1)
                        row["core_loader"] = {"ok": True, "n_events": nob, "nsamp": nsa,
                                              "p_pe_row_sum_max_abs_dev": float(np.abs(rs - 1).max()),
                                              "stdout": buf.getvalue().strip()[:300]}
                    except Exception as e:  # noqa: BLE001
                        row["core_loader"] = {"ok": False, "error": repr(e)[:500]}
                res["files"].append(row)
                print(json.dumps({k: row[k] for k in ("seed", "kind", "population_code")}),
                      row["historical"]["all_finite"], row.get("core_loader", {}).get("ok"),
                      flush=True)
    (CP / "results" / "core_logp_truth.json").write_text(json.dumps(res, indent=1))
    print("wrote", CP / "results" / "core_logp_truth.json")


def cmd_points(args):
    d = np.load(args.inp)
    lp, info = core_setup("historical")
    out = {"lnp_core_gal": lp(d["m1src"], d["q"], d["chieff"], d["z"], "gal")}
    np.savez_compressed(args.out, **out)
    print("wrote", args.out, info["settings"])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tables")
    sub.add_parser("truth")
    pp = sub.add_parser("points")
    pp.add_argument("inp")
    pp.add_argument("out")
    a = ap.parse_args()
    {"tables": cmd_tables, "truth": cmd_truth, "points": cmd_points}[a.cmd](a)


if __name__ == "__main__":
    main()
