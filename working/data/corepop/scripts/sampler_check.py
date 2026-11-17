#!/usr/bin/env python
"""Does the generator's --population_code core draw follow core's density?  (jax env)

    export PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-2b86a2d
    export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL"
    JAX_PLATFORMS=cpu python sampler_check.py          (after core_eval.py tables)

N = 4,000,000 proposals per population branch are drawn through
generate_dataset.draw_source_population -- the events stage's own function, not a copy --
in core mode and, for contrast, in legacy mode:

  gal: pop at the mock's input (mu_G 35, mu_chi 0)            -- every set's GAL hosts
  agn: the host-dependent set's AGN branch (mu_G 40 via pop_agn on the child mass stream,
       chi shifted by +0.10)

and each of m1, q, m2 and chi_eff is compared with core's own marginal
(cache/core_tables.npz, "historical" settings) by a one-sample KS test.  A KS distance
of D at N = 4e6 resolves CDF errors of ~1.4/sqrt(N) = 7e-4; 1000 events resolve ~0.043.
Also: the fraction below 5, 6, 8 Msun in m2 against core's, the core route's tally, and
the one approximation the route inherits from the legacy power-law lanes (the
10,000-iteration fallback of gmd._sample_q near m1 = m_min), computed in closed form.

Writes results/sampler_gof.json and the proposal histograms into cache/figure_proposals.npz.
"""
import dataclasses
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
from scipy import stats
from scipy.integrate import quad

HERE = Path(__file__).resolve().parent
CP = HERE.parent
sys.path.insert(0, str(CP.parent))
import generate_dataset as G  # noqa: E402

PIN = Path("/hildafs/projects/phy230014p/magana/src/darksirens-2b86a2d")
N = 4_000_000
CHUNK = 500_000
ROOT = 20261117
BINS = {"m1": np.linspace(5, 80, 151), "q": np.linspace(0.05, 1, 96),
        "m2": np.linspace(0, 60, 241), "chi": np.linspace(-0.5, 0.6, 111)}


def draw(gmd, branch, mode):
    pop = gmd.PopulationConfig(gamma=G.GAMMA)
    agn = branch == "agn"
    pop_agn = dataclasses.replace(pop, peak_mu=pop.peak_mu + 5.0) if agn else None
    bidx = {"gal": 0, "agn": 1}[branch]
    rng = np.random.default_rng(np.random.SeedSequence(ROOT, spawn_key=(bidx, 0)))
    rng_m = np.random.default_rng(np.random.SeedSequence(ROOT, spawn_key=(bidx, 1)))
    rng_c = (np.random.default_rng(np.random.SeedSequence(ROOT, spawn_key=(bidx, 2)))
             if mode == "core" else None)
    tally = G.new_core_tally() if mode == "core" else None
    cols = {k: [] for k in ("m1", "q", "m2", "chi")}
    n_drop = 0
    for i in range(0, N, CHUNK):
        n = min(CHUNK, N - i)
        is_mark = np.full(n, agn)
        s = G.draw_source_population(gmd, rng, n, pop, is_mark,
                                     dmu_chi=(0.10 if agn else 0.0), pop_agn=pop_agn,
                                     rng_mass_agn=rng_m, rng_pair_core=rng_c, tally=tally)
        keep = s["in_core_support"] if s["in_core_support"] is not None else np.ones(n, bool)
        n_drop += int((~keep).sum())
        for k in cols:
            cols[k].append(s[k][keep])
    return {k: np.concatenate(v) for k, v in cols.items()}, tally, n_drop


def cdf_from(grid, dens):
    c = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(grid))])
    return c / c[-1]


def main():
    gmd = G.import_gmd(PIN)
    T = np.load(CP / "cache" / "core_tables.npz")
    dens_id = json.loads((CP / "results" / "density_identity.json").read_text())
    out = {"what": __doc__.split("\n\n")[1].strip(), "n_per_branch": N, "seed_root": ROOT,
           "ks_resolution_95pct_N": float(1.358 / np.sqrt(N)),
           "ks_resolution_95pct_1000_events": float(1.358 / np.sqrt(1000)),
           "branches": {}}
    hist = {}
    for branch in ("gal", "agn"):
        key = f"historical_{branch}"
        model = {"m1": (T["m1g"], cdf_from(T["m1g"], T[f"pm1_{key}"])),
                 "q": (T["qg"], cdf_from(T["qg"], T[f"pq_{key}"])),
                 "m2": (T["m2g"], T[f"Fm2_{key}"] / T[f"Fm2_{key}"][-1]),
                 "chi": (T["chig"], cdf_from(T["chig"], T[f"pchi_{key}"]))}
        for k, (g, d) in (("m1", (T["m1g"], T[f"pm1_{key}"])), ("q", (T["qg"], T[f"pq_{key}"])),
                          ("m2", (T["m2g"], T[f"pm2_{key}"])), ("chi", (T["chig"], T[f"pchi_{key}"]))):
            hist[f"model_{branch}_{k}_grid"] = g
            hist[f"model_{branch}_{k}_dens"] = d / np.trapz(d, g)
        marg = dens_id["marginals"]["historical"][branch]
        res = {}
        for mode in ("core", "legacy"):
            x, tally, n_drop = draw(gmd, branch, mode)
            r = {"n_drawn": N, "n_dropped_outside_core_support": n_drop, "tally": tally,
                 "ks": {}}
            for k, (g, c) in model.items():
                ks = stats.kstest(x[k], lambda v, g=g, c=c: np.interp(v, g, c))
                r["ks"][k] = {"D": float(ks.statistic), "pvalue": float(ks.pvalue)}
                hist[f"{mode}_{branch}_{k}"] = np.histogram(x[k], bins=BINS[k])[0]
            r["fraction_m2_below"] = {
                str(t): {"drawn": float((x["m2"] < t).mean()),
                         "core": marg[f"P_m2_lt_{t}"],
                         "binomial_sd": float(np.sqrt(max(marg[f"P_m2_lt_{t}"], 1.0 / N)
                                                      * (1 - marg[f"P_m2_lt_{t}"]) / N))}
                for t in (5, 6, 8)}
            r["min_m2"] = float(x["m2"].min())
            r["chi_mean_drawn"] = float(x["chi"].mean())
            r["chi_mean_core"] = marg["chi_mean"]
            r["chi_sd_drawn"] = float(x["chi"].std())
            r["chi_sd_core"] = marg["chi_sd"]
            r["m1_mean_drawn"] = float(x["m1"].mean())
            res[mode] = r
            print(f"[{branch} {mode}] " + "  ".join(
                f"{k} D={v['D']:.2e} p={v['pvalue']:.3g}" for k, v in r["ks"].items())
                + f"  P(m2<5)={r['fraction_m2_below']['5']['drawn']:.4g}", flush=True)
        out["branches"][branch] = res
    for k, b in BINS.items():
        hist[f"bins_{k}"] = b
    # the legacy power-law fallback (gmd._sample_q gives up after 10,000 rejections and
    # returns a hard q**beta draw): probability per proposal, closed form
    pop = gmd.PopulationConfig(gamma=G.GAMMA)
    lo, d1, b = pop.mmin, pop.dm_min, pop.beta
    S = lambda m: float(G.core_sfilter_low(m, lo, d1))
    pl = lambda m: float(G.core_sfilter_low(m, lo, d1)) * m ** (-pop.alpha)  # S_high = 1 here
    n_pl = sum(quad(lambda m: float(gmd._powerlaw_unnorm(m, pop.alpha, pop.mmin, pop.mmax,
                                                          pop.dm_min, pop.dm_max)), x, y,
                    epsabs=0, epsrel=1e-12, limit=400)[0]
               for x, y in ((5, 8), (8, 70), (70, 80)))

    def accept(m1):  # acceptance of one rejection round: N_core(m1) / N_hard(m1)
        h = quad(lambda m2: S(m2) * m2 ** b, lo, m1, epsabs=0, epsrel=1e-12, limit=400)[0]
        hard = (m1 ** (b + 1) - lo ** (b + 1)) / (b + 1)
        return h / hard
    m1s = np.linspace(5.0 + 1e-6, 8.0, 3001)
    p_fail = np.array([(1 - accept(m)) ** G.CORE_PAIR_MAX_ITER if m > lo else 1.0 for m in m1s])
    dens = np.array([pl(m) for m in m1s]) / n_pl
    p_fb = float((1 - pop.peak_fraction) * np.trapz(dens * p_fail, m1s))
    out["legacy_powerlaw_fallback"] = {
        "what": "probability per proposal that a power-law lane exhausts gmd._sample_q's "
                "10,000 rejection rounds and gets a hard q**beta draw instead of the "
                "tapered one (the same in legacy and core mode: power-law lanes are "
                "never touched by the core route)",
        "probability_per_proposal": p_fb,
        "m1_where_it_can_happen": [float(m1s[p_fail > 1e-3].min()),
                                   float(m1s[p_fail > 1e-3].max())] if (p_fail > 1e-3).any() else None,
        "note": "those binaries have m2 in (5, m1) with m1 < ~5.5, so core gives them finite "
                "density; they differ from core only in the shape of q within that sliver"}
    print("[fallback]", out["legacy_powerlaw_fallback"]["probability_per_proposal"],
          out["legacy_powerlaw_fallback"]["m1_where_it_can_happen"], flush=True)
    np.savez_compressed(CP / "cache" / "figure_proposals.npz", **hist)
    (CP / "results" / "sampler_gof.json").write_text(json.dumps(out, indent=1))
    print("wrote", CP / "results" / "sampler_gof.json")


if __name__ == "__main__":
    main()
