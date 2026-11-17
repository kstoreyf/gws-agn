#!/usr/bin/env python
"""Does the generator's draw at --mass_spectrum gwtc3_pp follow core's density?  (jax env)

    sbatch --job-name=sampler run_cpu.sbatch jax sampler_check.py     (after core_family.py identity)

N = 4,000,000 proposals per population branch through generate_dataset.draw_source_population
(the events stage's own function) in core mode at the obslike population:

  gal: the obslike vector (mu_G 34, mu_chi 0)
  agn: the host-dependent set's AGN branch (mu_G 39 via pop_agn on the child mass stream,
       chi_eff shifted by +0.10)

m1, q, m2 and chi_eff are compared with core's own marginals (cache/core_tables_obslike.npz)
by one-sample KS.  Also: the peak-component share against w_G = 0.092, m2 below 5/6/8 Msun
against core, the route's tally, and the probability that a power-law lane exhausts
gmd._sample_q's 10,000 rejection rounds (the one approximation the route inherits).
Writes results/sampler_gof.json.
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
OB = HERE.parent
sys.path.insert(0, str(OB.parent))
import generate_dataset as G  # noqa: E402

PIN = Path("/hildafs/projects/phy230014p/magana/src/darksirens-2b86a2d")
N = 4_000_000
CHUNK = 500_000
ROOT = 20261118


def obs_pop(gmd):
    return dataclasses.replace(gmd.PopulationConfig(gamma=G.GAMMA), **G.MASS_SPECTRA["gwtc3_pp"])


def draw(gmd, branch):
    pop = obs_pop(gmd)
    agn = branch == "agn"
    pop_agn = dataclasses.replace(pop, peak_mu=pop.peak_mu + 5.0) if agn else None
    bidx = {"gal": 0, "agn": 1}[branch]
    rng = np.random.default_rng(np.random.SeedSequence(ROOT, spawn_key=(bidx, 0)))
    rng_m = np.random.default_rng(np.random.SeedSequence(ROOT, spawn_key=(bidx, 1)))
    rng_c = np.random.default_rng(np.random.SeedSequence(ROOT, spawn_key=(bidx, 2)))
    tally = G.new_core_tally()
    cols = {k: [] for k in ("m1", "q", "m2", "chi", "use_peak")}
    n_drop = 0
    for i in range(0, N, CHUNK):
        n = min(CHUNK, N - i)
        s = G.draw_source_population(gmd, rng, n, pop, np.full(n, agn),
                                     dmu_chi=(0.10 if agn else 0.0), pop_agn=pop_agn,
                                     rng_mass_agn=rng_m, rng_pair_core=rng_c, tally=tally)
        keep = s["in_core_support"]
        n_drop += int((~keep).sum())
        for k in cols:
            cols[k].append(s[k][keep])
    return {k: np.concatenate(v) for k, v in cols.items()}, tally, n_drop


def cdf_from(grid, dens):
    c = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(grid))])
    return c / c[-1]


def main():
    gmd = G.import_gmd(PIN)
    T = np.load(OB / "cache" / "core_tables_obslike.npz")
    marg_all = json.loads((OB / "results" / "density_identity.json").read_text())["marginals"]
    out = {"what": __doc__.split("\n\n")[1].strip(), "n_per_branch": N, "seed_root": ROOT,
           "population": dataclasses.asdict(obs_pop(gmd)),
           "ks_resolution_95pct_N": float(1.358 / np.sqrt(N)),
           "ks_resolution_95pct_1000_events": float(1.358 / np.sqrt(1000)), "branches": {}}
    for branch in ("gal", "agn"):
        model = {"m1": (T["m1g"], cdf_from(T["m1g"], T[f"pm1_{branch}"])),
                 "q": (T["qg"], cdf_from(T["qg"], T[f"pq_{branch}"])),
                 "m2": (T["m2g"], T[f"Fm2_{branch}"] / T[f"Fm2_{branch}"][-1]),
                 "chi": (np.linspace(-1, 1, T[f"pchi_{branch}"].size),
                         cdf_from(np.linspace(-1, 1, T[f"pchi_{branch}"].size), T[f"pchi_{branch}"]))}
        marg = marg_all[branch]
        x, tally, n_drop = draw(gmd, branch)
        r = {"n_drawn": N, "n_dropped_outside_core_support": n_drop, "tally": tally, "ks": {}}
        for k, (g, c) in model.items():
            ks = stats.kstest(x[k], lambda v, g=g, c=c: np.interp(v, g, c))
            r["ks"][k] = {"D": float(ks.statistic), "pvalue": float(ks.pvalue)}
        wg = G.MASS_SPECTRA["gwtc3_pp"]["peak_fraction"]
        r["peak_share"] = {"drawn": float(x["use_peak"].mean()), "w_G": wg,
                           "binomial_sd": float(np.sqrt(wg * (1 - wg) / x["m1"].size))}
        r["fraction_m2_below"] = {
            str(t): {"drawn": float((x["m2"] < t).mean()), "core": marg[f"P_m2_lt_{t}"]} for t in (5, 6, 8)}
        r["min_m2"] = float(x["m2"].min())
        r["m1_median_drawn"] = float(np.median(x["m1"]))
        r["m1_median_core"] = marg["m1_median"]
        r["m1_q99_drawn"] = float(np.quantile(x["m1"], 0.99))
        r["m1_q99_core"] = marg["m1_q99"]
        r["chi_mean_drawn"] = float(x["chi"].mean())
        r["chi_mean_core"] = marg["chi_mean"]
        out["branches"][branch] = r
        print(f"[{branch}] " + "  ".join(f"{k} D={v['D']:.2e} p={v['pvalue']:.3g}" for k, v in r["ks"].items())
              + f"  peak {r['peak_share']}  P(m2<5)={r['fraction_m2_below']['5']['drawn']:.3g}  tally {tally}",
              flush=True)
    # the legacy power-law fallback near m1 = m_min, closed form
    pop = obs_pop(gmd)
    lo, d1, b = pop.mmin, pop.dm_min, pop.beta
    S = lambda m: float(G.core_sfilter_low(m, lo, d1))
    n_pl = sum(quad(lambda m: float(gmd._powerlaw_unnorm(m, pop.alpha, pop.mmin, pop.mmax,
                                                          pop.dm_min, pop.dm_max)), x0, x1,
                    epsabs=0, epsrel=1e-12, limit=400)[0]
               for x0, x1 in ((lo, lo + d1), (lo + d1, pop.mmax - pop.dm_max), (pop.mmax - pop.dm_max, pop.mmax)))

    def accept(m1):
        h = quad(lambda m2: S(m2) * m2 ** b, lo, m1, epsabs=0, epsrel=1e-12, limit=400)[0]
        return h / ((m1 ** (b + 1) - lo ** (b + 1)) / (b + 1))
    m1s = np.linspace(lo + 1e-6, lo + d1, 4001)
    p_fail = np.array([(1 - accept(m)) ** 10_000 for m in m1s])
    dens = np.array([S(m) * m ** (-pop.alpha) for m in m1s]) / n_pl
    p_fb = float((1 - pop.peak_fraction) * np.trapz(dens * p_fail, m1s))
    out["legacy_powerlaw_fallback"] = {
        "what": "probability per proposal that a power-law lane exhausts gmd._sample_q's 10,000 "
                "rejection rounds and gets a hard q**beta draw on [m_min/m1, 1] instead of the "
                "tapered one (m2 >= 5 either way; core gives those binaries finite density)",
        "probability_per_proposal": p_fb,
        "m1_where_p_fail_gt_1e-3": ([float(m1s[p_fail > 1e-3].min()), float(m1s[p_fail > 1e-3].max())]
                                    if (p_fail > 1e-3).any() else None)}
    print("[fallback]", out["legacy_powerlaw_fallback"], flush=True)
    (OB / "results" / "sampler_gof.json").write_text(json.dumps(out, indent=1))
    print("wrote", OB / "results" / "sampler_gof.json")


if __name__ == "__main__":
    main()
