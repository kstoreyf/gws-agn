#!/usr/bin/env python3
"""ATTRIBUTION follow-up, TASK 1 -- sampler-vs-pdf ratio test.

The events' TRUE source-frame masses were drawn by ``gmd``'s own samplers
(``_sample_powerlaw_peak_m1`` + ``_sample_q``, called from
``generate_dataset.py::stage_events``).  The likelihood evaluates darksirens'
ANALYTIC ``powerlaw+peak`` density.  If the two differ anywhere the score is
sensitive, the detected-truth mean ``A`` and the model's ``B`` differ and the
difference lands in ``r`` directly.  ``attr_mass_pe.py`` measured that channel
on 720 events -- ``(A - B)_mass = -6.4e-5 +- 4.5e-4`` -- Poisson-limited.  This
script replaces the 720-event Monte Carlo by

  (a) >= 1e8 draws from the SAME sampler code path, binned against the analytic
      density on a grid that is tight where ``d ln p/dm`` is steep (the
      ``mmin + dm_min`` and ``mmax - dm_max`` tapers, the Gaussian peak wings);

  (b) a SEMI-ANALYTIC sampler density.  Both gmd samplers are exact rejection
      samplers, so their density is known in closed form up to normalisation:

        q_s(m1,q) = SUM_c w_c [u_c(m1)/U_c^exact] [q^beta S_low(q m1)/V_c^exact(m1)]

      with U, V computed by composite Gauss-Legendre to machine precision.  The
      binned draws in (a) VALIDATE q_s; (b) then gives R = q_s/p_analytic with
      no Monte-Carlo error at all;

  (c) the direct prediction of this channel's contribution to ``r``:

        delta = E_q[varsigma] - E_p[varsigma]

      over the DETECTED set, obtained by importance-reweighting the stored
      injections -- darksirens' own selection weights (which already carry
      pdraw, the detection decision, p_z and the Jacobian) multiplied by
      R(m1src, q).  Nothing but the mass channel moves.

Outputs: results/attr_sampler_ratio.json, results/attr_sampler_ratio.npz
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
GEN = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data")

BETA = 1.0
GL_N = 24                     # Gauss-Legendre nodes per subinterval
GL_SUB = 600                  # subintervals for the 1-D normalisations


# ----------------------------------------------------------------------------
# the sampler, called exactly as generate_dataset.py::stage_events calls it
# ----------------------------------------------------------------------------
def _worker(task):
    seed, n, m1_edges, q_edges, chi_edges = task
    # gmd imports jax at module scope; keep the 24 fork children off the GPU and
    # single-threaded (the draw itself is pure numpy).
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    sys.path.insert(0, str(GEN))
    import generate_dataset as gd
    gmd = gd.import_gmd()
    pop = gmd.PopulationConfig(gamma=gd.GAMMA)
    rng = np.random.default_rng(seed)
    H2 = np.zeros((len(m1_edges) - 1, len(q_edges) - 1), dtype=np.int64)
    Hc = np.zeros(len(chi_edges) - 1, dtype=np.int64)
    Hm1 = np.zeros(len(m1_edges) - 1, dtype=np.int64)
    acc = {"n": 0, "sum_m1": 0.0, "sum_m2": 0.0, "n_peak": 0}
    chunk = 2_000_000
    done = 0
    while done < n:
        k = min(chunk, n - done)
        # --- the stage_events call sequence, verbatim ---------------------
        m1, use_peak = gmd._sample_powerlaw_peak_m1(rng, k, pop, return_component=True)
        q = gmd._sample_q(rng, m1, pop, use_peak=use_peak)
        chi = gmd._sample_chieff(rng, k, pop)
        # ------------------------------------------------------------------
        H2 += np.histogram2d(m1, q, bins=[m1_edges, q_edges])[0].astype(np.int64)
        Hm1 += np.histogram(m1, bins=m1_edges)[0].astype(np.int64)
        Hc += np.histogram(chi, bins=chi_edges)[0].astype(np.int64)
        acc["n"] += k
        acc["sum_m1"] += float(m1.sum())
        acc["sum_m2"] += float((q * m1).sum())
        acc["n_peak"] += int(use_peak.sum())
        done += k
    return H2, Hm1, Hc, acc


def build_edges():
    """Non-uniform m1 edges: tight at the tapers and across the peak."""
    segs = [
        (4.0, 5.0, 10),        # below mmin: sampler support ends, peak tail only
        (5.0, 8.0, 120),       # LOW TAPER  [mmin, mmin+dm_min]
        (8.0, 20.0, 60),
        (20.0, 30.0, 100),     # peak low wing
        (30.0, 40.0, 200),     # peak core
        (40.0, 50.0, 100),     # peak high wing
        (50.0, 70.0, 80),
        (70.0, 80.0, 200),     # HIGH TAPER [mmax-dm_max, mmax]
        (80.0, 100.0, 20),     # above mmax: peak tail only
    ]
    e = [np.linspace(a, b, n + 1)[:-1] for a, b, n in segs]
    e.append(np.array([segs[-1][1]]))
    m1_edges = np.concatenate(e)
    q_edges = np.linspace(0.0, 1.0, 101)
    chi_edges = np.linspace(-0.5, 0.5, 201)
    return m1_edges, q_edges, chi_edges


# ----------------------------------------------------------------------------
# semi-analytic sampler density
# ----------------------------------------------------------------------------
def _gl_nodes(a, b, nsub, n=GL_N):
    x, w = np.polynomial.legendre.leggauss(n)
    edges = np.linspace(a, b, nsub + 1)
    lo, hi = edges[:-1, None], edges[1:, None]
    nodes = 0.5 * (hi - lo) * x[None, :] + 0.5 * (hi + lo)
    wts = 0.5 * (hi - lo) * w[None, :]
    return nodes, wts, edges


class SamplerDensity:
    """Exact density of gmd's (m1, q) rejection samplers, machine-precision norm."""

    def __init__(self, gmd, pop):
        self.gmd, self.pop = gmd, pop
        self.w = np.array([1.0 - pop.peak_fraction, pop.peak_fraction])
        self.mgrid_lo = float(gmd._MASS_NORM_GRID[0])
        self.mgrid_hi = float(gmd._MASS_NORM_GRID[-1])
        self.pair = [(pop.mmin, pop.dm_min), (gmd._PAIR_M_LO, gmd._PAIR_DM)]
        # ---- U_c : exact primary-mass normalisations -----------------------
        n, w_, _ = _gl_nodes(pop.mmin, pop.mmax, GL_SUB)
        self.U_pl = float((w_ * self._u_pl(n)).sum())
        n, w_, _ = _gl_nodes(self.mgrid_lo, self.mgrid_hi, GL_SUB)
        self.U_g = float((w_ * self._u_g(n)).sum())
        # ---- G_c(x) = int_0^x m2^beta S_low(m2) dm2, exact cumulative -------
        self._G = []
        for (mmin, dm) in self.pair:
            nodes, wts, edges = _gl_nodes(0.0, 220.0, 22000)
            f = nodes ** BETA * gmd._sfilter_low(nodes, mmin, dm)
            seg = (wts * f).sum(axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            self._G.append((edges, cum, mmin, dm))

    def _u_pl(self, m):
        p = self.pop
        return (self.gmd._sfilter_low(m, p.mmin, p.dm_min)
                * self.gmd._sfilter_high(m, p.mmax, p.dm_max)
                * np.power(np.maximum(m, 1e-30), -p.alpha))

    def _u_g(self, m):
        p = self.pop
        out = np.exp(-0.5 * ((m - p.peak_mu) / p.peak_sigma) ** 2)
        return np.where((m >= self.mgrid_lo) & (m <= self.mgrid_hi), out, 0.0)

    def _Gfun(self, x, c):
        """Exact int_0^x m2^beta S_low dm2 (GL-exact cell + GL-exact remainder)."""
        edges, cum, mmin, dm = self._G[c]
        h = edges[1] - edges[0]
        k = np.clip(np.floor(x / h).astype(int), 0, len(edges) - 2)
        base = cum[k]
        a = edges[k]
        xx, ww = np.polynomial.legendre.leggauss(GL_N)
        nodes = 0.5 * (x - a)[:, None] * xx[None, :] + 0.5 * (x + a)[:, None]
        wts = 0.5 * (x - a)[:, None] * ww[None, :]
        f = nodes ** BETA * self.gmd._sfilter_low(nodes, mmin, dm)
        return base + (wts * f).sum(axis=1)

    def V(self, m1, c):
        """int_0^1 q^beta S_low(q m1) dq = m1^-(beta+1) G_c(m1)."""
        m1 = np.asarray(m1, float)
        return np.power(np.maximum(m1, 1e-30), -(BETA + 1.0)) * self._Gfun(m1, c)

    def __call__(self, m1, q):
        m1 = np.asarray(m1, float); q = np.asarray(q, float)
        m1f, qf = np.broadcast_arrays(m1, q)
        shp = m1f.shape
        m1f = m1f.ravel(); qf = qf.ravel()
        out = np.zeros_like(m1f)
        for c, (u, U) in enumerate(((self._u_pl, self.U_pl), (self._u_g, self.U_g))):
            mmin, dm = self.pair[c]
            V = self.V(m1f, c)
            pair = (np.power(np.maximum(qf, 0.0), BETA)
                    * self.gmd._sfilter_low(qf * m1f, mmin, dm))
            out += self.w[c] * (u(m1f) / U) * np.where(V > 0, pair / np.maximum(V, 1e-300), 0.0)
        return out.reshape(shp)


# ----------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ndraw", type=float, default=1.2e8)
    ap.add_argument("--nproc", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--tracers", default="gal,agn")
    ap.add_argument("--dh", type=float, default=0.5)
    ap.add_argument("--sel_batch", type=int, default=50000)
    ap.add_argument("--skip_draws", action="store_true")
    ap.add_argument("--skip_predict", action="store_true")
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    npz_path = outdir / "attr_sampler_ratio.npz"
    m1_edges, q_edges, chi_edges = build_edges()
    summary = {"name": "attr_sampler_ratio",
               "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # ------------------------------------------------------------------ (a)
    ndraw = int(args.ndraw)
    if not args.skip_draws:
        import multiprocessing as mp
        nper = int(np.ceil(ndraw / args.nproc))
        ss = np.random.SeedSequence(args.seed)
        seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(args.nproc)]
        tasks = [(sd, nper, m1_edges, q_edges, chi_edges) for sd in seeds]
        t0 = time.time()
        with mp.get_context("fork").Pool(args.nproc) as pool:
            res = pool.map(_worker, tasks)
        H2 = sum(r[0] for r in res); Hm1 = sum(r[1] for r in res)
        Hc = sum(r[2] for r in res)
        acc = {k: sum(r[3][k] for r in res) for k in res[0][3]}
        print(f"[draws] {acc['n']:,} pairs in {time.time()-t0:.0f}s "
              f"({args.nproc} procs); peak fraction {acc['n_peak']/acc['n']:.6f}")
        np.savez_compressed(outdir / "attr_sampler_draws.npz", H2=H2, Hm1=Hm1, Hc=Hc,
                            m1_edges=m1_edges, q_edges=q_edges, chi_edges=chi_edges,
                            n=acc["n"], sum_m1=acc["sum_m1"], sum_m2=acc["sum_m2"],
                            n_peak=acc["n_peak"], seeds=np.asarray(seeds))
    d = np.load(outdir / "attr_sampler_draws.npz")
    H2 = d["H2"]; Hm1 = d["Hm1"]; Hc = d["Hc"]; N = int(d["n"])
    print(f"[draws] loaded N = {N:,}")

    # ------------------------------------------------------------------ (b)
    sys.path.insert(0, str(GEN))
    import generate_dataset as gd
    gmd = gd.import_gmd()
    pop = gmd.PopulationConfig(gamma=gd.GAMMA)
    t0 = time.time()
    SD = SamplerDensity(gmd, pop)
    print(f"[semi-analytic] U_pl={SD.U_pl:.12e}  U_g={SD.U_g:.12e}  "
          f"({time.time()-t0:.1f}s)")

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    import jax.numpy as jnp
    from darksirens.gw.populations import get_fixed_population_params
    from darksirens.gw.populations.registry import get_model
    model = get_model("powerlaw+peak", shared_beta=True, shared_spin=True,
                      shared_gamma=True)
    th = np.asarray(get_fixed_population_params("powerlaw+peak", shared_beta=True,
                                                shared_spin=True, shared_gamma=True))
    th_mix = jnp.asarray(th[:model.mixture.n_params])

    def p_analytic(m1, q, chunk=2_000_000):
        m1 = np.asarray(m1, float).ravel(); q = np.asarray(q, float).ravel()
        out = np.empty(m1.size)
        for i in range(0, m1.size, chunk):
            out[i:i + chunk] = np.asarray(model.mixture.mass_q_density(
                jnp.asarray(m1[i:i + chunk]), jnp.asarray(q[i:i + chunk]), th_mix))
        return out

    # bin masses of the analytic density by tensor Gauss-Legendre inside each bin
    gx, gw = np.polynomial.legendre.leggauss(6)
    def _bin_nodes(edges):
        lo, hi = edges[:-1, None], edges[1:, None]
        return (0.5 * (hi - lo) * gx[None, :] + 0.5 * (hi + lo),
                0.5 * (hi - lo) * gw[None, :])
    mn, mw = _bin_nodes(m1_edges)          # (nm, 6)
    qn, qw = _bin_nodes(q_edges)           # (nq, 6)
    nm, nq = mn.shape[0], qn.shape[0]
    M = (mn[:, :, None, None] * np.ones((1, 1, nq, 6))).ravel()
    Q = (qn[None, None, :, :] * np.ones((nm, 6, 1, 1))).ravel()
    Wt = (mw[:, :, None, None] * qw[None, None, :, :]).ravel()
    t0 = time.time()
    P_bin = (p_analytic(M, Q) * Wt).reshape(nm, 6, nq, 6).sum(axis=(1, 3))
    Ps_bin = (SD(M, Q) * Wt).reshape(nm, 6, nq, 6).sum(axis=(1, 3))
    print(f"[bins] analytic bin masses in {time.time()-t0:.1f}s; "
          f"total analytic mass over grid = {P_bin.sum():.10f}, "
          f"semi-analytic sampler mass = {Ps_bin.sum():.10f}")
    P_m1 = P_bin.sum(axis=1); Ps_m1 = Ps_bin.sum(axis=1)

    obs = H2 / N
    with np.errstate(divide="ignore", invalid="ignore"):
        lr_mc = np.log(obs / P_bin)
        lr_semi = np.log(Ps_bin / P_bin)
        sig_lr = 1.0 / np.sqrt(np.maximum(H2, 1))
    ok = (H2 >= 50) & (P_bin > 0)
    dev = (lr_mc - lr_semi)[ok] / sig_lr[ok]
    print(f"[validate] bins with >=50 counts: {ok.sum()}/{ok.size}; "
          f"(MC - semi)/sigma: mean {dev.mean():+.4f} sd {dev.std():.4f} "
          f"max|.| {np.abs(dev).max():.2f}")
    ok1 = (Hm1 >= 50) & (P_m1 > 0)
    lr1_mc = np.log((Hm1 / N) / np.where(P_m1 > 0, P_m1, 1.0))
    lr1_semi = np.log(np.where(P_m1 > 0, Ps_m1 / np.where(P_m1 > 0, P_m1, 1), 1.0))
    sig1 = 1.0 / np.sqrt(np.maximum(Hm1, 1))
    dev1 = (lr1_mc - lr1_semi)[ok1] / sig1[ok1]
    print(f"[validate] m1 marginal: mean dev {dev1.mean():+.4f} sd {dev1.std():.4f} "
          f"max|.| {np.abs(dev1).max():.2f}")
    summary["draws"] = {
        "n_draws": N, "n_procs": args.nproc, "seed": args.seed,
        "peak_fraction_realised": float(d["n_peak"]) / N,
        "peak_fraction_config": pop.peak_fraction,
        "mean_m1src_sampler": float(d["sum_m1"]) / N,
        "mean_m2src_sampler": float(d["sum_m2"]) / N,
        "analytic_mass_on_grid": float(P_bin.sum()),
        "semianalytic_mass_on_grid": float(Ps_bin.sum()),
        "validation_semi_vs_mc": {
            "n_bins_used": int(ok.sum()),
            "pull_mean": float(dev.mean()), "pull_sd": float(dev.std()),
            "pull_absmax": float(np.abs(dev).max()),
            "m1_marginal_pull_mean": float(dev1.mean()),
            "m1_marginal_pull_sd": float(dev1.std()),
            "m1_marginal_pull_absmax": float(np.abs(dev1).max()),
        },
        "max_abs_log_ratio_semi": float(np.abs(lr_semi[P_bin > 0]).max()),
        "weighted_rms_log_ratio_semi": float(
            np.sqrt((P_bin * lr_semi ** 2)[(P_bin > 0)].sum() / P_bin.sum())),
    }
    # analytic-model mean m1src (no detection): direct check of the 0.02 % claim
    mean_ana = float((P_bin * ((m1_edges[:-1] + m1_edges[1:]) / 2)[:, None]).sum()
                     / P_bin.sum())
    summary["draws"]["mean_m1src_analytic_binned"] = mean_ana

    # chieff marginal (H0-independent and SNR-independent; a pure cross-check)
    from darksirens.gw.populations.registry import get_model as _gm
    spin = model.mixture.spin_components[0]
    _, _, _, ts = model.mixture._split_theta(th_mix)
    cn, cw = _bin_nodes(chi_edges)
    pc = np.asarray(spin(jnp.asarray(cn.ravel()), ts[0])).reshape(cn.shape)
    Pc = (pc * cw).sum(axis=1)
    okc = Hc >= 50
    devc = (np.log((Hc / N)[okc] / Pc[okc])) * np.sqrt(Hc[okc])
    summary["draws"]["chieff_pull_mean"] = float(devc.mean())
    summary["draws"]["chieff_pull_sd"] = float(devc.std())
    summary["draws"]["chieff_mass_in_window"] = float(Pc.sum())
    print(f"[chieff] pull mean {devc.mean():+.3f} sd {devc.std():.3f} "
          f"(mass in +-0.5 window {Pc.sum():.6f})")

    np.savez_compressed(npz_path, H2=H2, Hm1=Hm1, Hc=Hc, N=N,
                        m1_edges=m1_edges, q_edges=q_edges, chi_edges=chi_edges,
                        P_bin=P_bin, Ps_bin=Ps_bin, P_m1=P_m1, Ps_m1=Ps_m1,
                        lr_mc=lr_mc, lr_semi=lr_semi, sig_lr=sig_lr,
                        Pc=Pc)

    # ------------------------------------------------------------------ (c)
    if not args.skip_predict:
        sys.path.insert(0, str(HERE))
        import attr_ds_bridge as bridge
        summary["prediction"] = {}
        for tracer in args.tracers.split(","):
            tracer = tracer.strip()
            kw = dict(kde_window=4096) if tracer == "gal" else {}
            B = bridge.build(tracer=tracer, sel_batch=args.sel_batch, **kw)
            S = bridge.sel_pass(B, dh=args.dh, sel_batch=args.sel_batch)
            anchor = abs(float(S["log_mu"]) - B.spy0["log_mu"])
            print(f"[{tracer}] ANCHOR |log_mu diff| = {anchor:.3e}")
            w = S["w"]; m1s = S["m1src"]; qq = S["q"]

            pa = p_analytic(m1s, qq)
            ps = SD(m1s, qq)
            good = (pa > 0) & np.isfinite(pa) & np.isfinite(ps)
            R = np.where(good, ps / np.where(pa > 0, pa, 1.0), 1.0)
            wmass_bad = float(w[~good].sum())

            # histogram-based R, as an independent (MC-limited) estimate
            im = np.clip(np.searchsorted(m1_edges, m1s) - 1, 0, nm - 1)
            iq = np.clip(np.searchsorted(q_edges, qq) - 1, 0, nq - 1)
            inside = (m1s >= m1_edges[0]) & (m1s <= m1_edges[-1])
            Rh_map = np.where(P_bin > 0, obs / np.where(P_bin > 0, P_bin, 1.0), 1.0)
            Rh = np.where(inside, Rh_map[im, iq], 1.0)
            Rh = np.where((H2[im, iq] >= 20) & inside, Rh, R)   # fall back where empty

            terms = {k: S[k] for k in ("mass", "rate", "pz", "jac")}
            terms["tot"] = S["pop"] + S["pz"] + S["jac"]
            out = {"anchor_log_mu_absdiff": anchor,
                   "n_injections": int(w.size),
                   "weight_mass_outside_support": wmass_bad,
                   "R_weighted_mean": float((w * R).sum()),
                   "R_weighted_sd": float(np.sqrt((w * (R - (w * R).sum()) ** 2).sum())),
                   "R_min": float(R.min()), "R_max": float(R.max())}
            for k, s in terms.items():
                W = w.sum(); T = float((w * s).sum())
                WR = float((w * R).sum()); TR = float((w * R * s).sum())
                Bp = T / W; Bq = TR / WR
                psi = (w * R * (s - Bq)) / WR - (w * (s - Bp)) / W
                var = float((psi ** 2).sum())
                WRh = float((w * Rh).sum()); TRh = float((w * Rh * s).sum())
                out[k] = {"B_analytic": Bp, "B_sampler": Bq,
                          "delta": Bq - Bp, "delta_sem_injections": float(np.sqrt(var)),
                          "delta_histogram_R": TRh / WRh - Bp}
                print(f"[{tracer}] {k:>5}: B_p={Bp:.6e}  B_q={Bq:.6e}  "
                      f"delta={Bq-Bp:+.4e} +- {np.sqrt(var):.2e}  "
                      f"(hist R: {TRh/WRh-Bp:+.4e})")
            summary["prediction"][tracer] = out
            del B, S

    (outdir / "attr_sampler_ratio.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {outdir/'attr_sampler_ratio.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
