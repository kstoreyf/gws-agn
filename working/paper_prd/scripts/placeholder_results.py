#!/usr/bin/env python
"""Placeholder results of the PRD paper (notes/PAPER_SKELETON.md, Secs. 0, 6, 7): one entry per key and
quantity, with the median, the 68% and 90% equal-tailed intervals, a posterior shape for plotting, and its
source. Writes values/placeholder_results.json. Deterministic (no random draws; everything is a stored
posterior or an analytic shape).

    /hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python \
        working/paper_prd/scripts/placeholder_results.py

Sources
    exploratory  an earlier run of the same analysis under the old settings (the file is named); its
                 posterior is used as it is (grid or equal-weight samples)
    invented     no run exists: the expectation of PAPER_SKELETON.md Sec. 7, a Gaussian (truncated to the
                 prior range; half-normal for an upper limit) with the stated median and 90% half-width
    production   a production summary named in values/production_map.json exists and overrides the entry

Realization A ("A") and the combination over realizations ("Comb"). N, the number of realizations the
production combines for each key, is read from analyses/analysis_16_production_runs/runs/RUNLIST.yaml (the
runs of the key's analysis, catalogs, flux limit and event set); the keys the run list does not hold yet take
it from runs.txt (observed-like spectrum: Stage 5, five; the two control sets: run R4, "five for the others").
Where no combination exists on disk the combination is invented from realization A as realization A combined
with N - 1 realizations whose posteriors have A's width and sit at their expectation (input + bias): narrower
by sqrt(N) and centred at
    med_Comb = input + bias + (med_A - input - bias) / N,
where `bias` is the systematic offset the skeleton expects to survive the combination (the single-population
analysis of host-dependent events) and zero otherwise. (Until 17 Nov the shift was divided by sqrt(N), which
kept the input at realization A's credible level and left it outside the combined region wherever it lay
outside A's.) Exploratory combinations use the exploratory realizations that are also production
realizations of the key where those exist (all-AGN and all-galaxy sets, each catalog alone: 100, 101, 102,
103, 105), and all nine exploratory realizations for both catalogs.

Production summaries (contract): values/production_map.json maps a key to a JSON file with the same schema
as one entry here ({"quantities": {Q: {"A": summary, "Comb": summary, ...}}, "joint": ..., "LB": ...}); when
the file exists it replaces the entry and the entry is marked final.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.special import ndtr, ndtri

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_values as bv  # noqa: E402

PRD = HERE.parent
OUTJ = PRD / "values" / "placeholder_results.json"
PMAP = PRD / "values" / "production_map.json"
IN = bv.IN
Z90 = 1.6448536269514722

# production runs: analysis_16's run list (one selector per key) and realizations combined
RUNLIST = bv.AN / "analysis_16_production_runs" / "runs" / "RUNLIST.yaml"
KEY_SELECTOR = {   # key -> the run-list arguments of its runs (realization excluded)
    "KnownBoth": dict(analysis="known_population", catalogs="both", flux="complete", events="equal_populations"),
    "KnownGal": dict(analysis="known_population", catalogs="galaxy", flux="complete", events="equal_populations"),
    "KnownAGN": dict(analysis="known_population", catalogs="agn", flux="complete", events="equal_populations"),
    "KnownNone": dict(analysis="known_population", catalogs="none", flux="complete", events="equal_populations"),
    "AGNhosted": dict(analysis="known_population", catalogs="agn", flux="complete", events="agn_hosted"),
    "Galhosted": dict(analysis="known_population", catalogs="galaxy", flux="complete", events="galaxy_hosted"),
    "SingleBoth": dict(analysis="single_population", catalogs="both", flux="complete", events="equal_populations"),
    "SingleNone": dict(analysis="single_population", catalogs="none", flux="complete", events="equal_populations"),
    "HDSingle": dict(analysis="single_population", catalogs="both", flux="complete", events="host_dependent"),
    "HDHost": dict(analysis="host_dependent_populations", catalogs="both", flux="complete", events="host_dependent"),
    "HDNone": dict(analysis="single_population", catalogs="none", flux="complete", events="host_dependent"),
}
for _k, _w, _m in bv.RUNGS:
    KEY_SELECTOR[f"Flux{_w}"] = dict(analysis="known_population", catalogs="both", flux=str(_m), events="equal_populations", density="counts")
    KEY_SELECTOR[f"Free{_w}"] = dict(analysis="free_densities", catalogs="both", flux=str(_m), events="equal_populations", density="free")
# keys whose runs are not in the run list yet: runs.txt, block "REVISION OF 17 NOV" and "APPROVED BY THE OWNER"
NOT_IN_RUNLIST = {
    "CtrlZero": (5, "run R4 of runs.txt (no AGN-hosted mergers), not yet in the run list or the approved stages; 'five for the others' (revision of 17 Nov)"),
    "CtrlTenth": (5, "run R4 of runs.txt (one merger in ten AGN-hosted), not yet in the run list or the approved stages; 'five for the others' (revision of 17 Nov)"),
    "ObsSingle": (5, "Stage 5 of runs.txt (observed-like spectrum, 5 realizations), not yet in the run list"),
    "ObsHost": (5, "Stage 5 of runs.txt (observed-like spectrum, 5 realizations), not yet in the run list"),
    "ObsNone": (5, "Stage 5 of runs.txt (observed-like spectrum, 5 realizations), not yet in the run list"),
}


def _runlist_realizations():
    """key -> sorted list of the realizations of its runs in RUNLIST.yaml (only runs without an AGN-density factor)."""
    import yaml
    doc = yaml.safe_load(RUNLIST.read_text())
    out = {}
    for st in doc["stages"]:
        for r in st["runs"]:
            a = r["args"]
            kv = {a[i].lstrip("-"): a[i + 1] for i in range(len(a) - 1) if a[i].startswith("--") and not a[i + 1].startswith("--")}
            if "agn-density-factor" in kv:
                continue
            sel = dict(analysis=kv["analysis"], catalogs=kv["catalogs"], flux=kv["flux"], events=kv["events-kind"])
            if "density" in kv:
                sel["density"] = kv["density"]
            for key, want in KEY_SELECTOR.items():
                if want == sel:
                    out.setdefault(key, set()).add(int(kv["realization"]))
    return {k: sorted(v) for k, v in out.items()}


RUNREAL = _runlist_realizations()
NREAL = {k: (len(v), f"RUNLIST.yaml: {len(v)} runs {KEY_SELECTOR[k]}, realizations {v}") for k, v in RUNREAL.items()}
NREAL.update({k: v for k, v in NOT_IN_RUNLIST.items()})
assert set(NREAL) == set(KEY_SELECTOR) | set(NOT_IN_RUNLIST), sorted(set(KEY_SELECTOR) - set(NREAL))
STAGE_TITLE = {"known": "Stage 1: population known, complete catalogs", "pure": "Stage 1: all-AGN and all-galaxy event sets",
               "ctrl": "run R4 of runs.txt (not yet in the run list)", "flux": "Stage 2: flux-limited catalogs, densities from catalog counts",
               "free": "Stage 2: number densities inferred", "single": "Stage 3: single population, equal populations",
               "hdsingle": "Stage 4: host-dependent events, single population", "hdhost": "Stage 4: host-dependent events, host-dependent populations",
               "hdnone": "Stage 4: host-dependent events, no catalog", "obs": "Stage 5: observed-like spectrum, host-dependent events (not yet in the run list)"}
PRIOR = {"Hzero": (50.0, 100.0), "fAGN": (0.0, 1.0), "MuG": (20.0, 50.0), "SigmaG": (1.0, 10.0), "MuChi": (-0.2, 0.3), "SigmaChi": (0.01, 1.0),
         "DmuG": (-10.0, 10.0), "DmuChi": (-0.05, 0.30), "LognGal": (-4.0, -1.0), "LognAGN": (-6.0, -4.0), "RateRatio": (0.0, 400.0)}
UNIT = {"Hzero": "km/s/Mpc", "fAGN": "", "MuG": "Msun", "SigmaG": "Msun", "MuChi": "", "SigmaChi": "", "DmuG": "Msun", "DmuChi": "",
        "LognGal": "log10 Mpc^-3", "LognAGN": "log10 Mpc^-3", "RateRatio": ""}
CHAIN_Q = {"H0": "Hzero", "f_agn": "fAGN", "mu_G": "MuG", "sigma_G": "SigmaG", "mu_chi": "MuChi", "sigma_chi": "SigmaChi", "dmu_G": "DmuG", "dmu_chi": "DmuChi"}


def _ratio100():
    c = bv.survey_counts(100)
    return c["gal"] / c["agn"]


RIN = IN["f"] / (1 - IN["f"]) * _ratio100()
INPUT = {"Hzero": IN["H0"], "fAGN": IN["f"], "MuG": IN["muG"], "SigmaG": IN["sigmaG"], "MuChi": IN["muChi"], "SigmaChi": IN["sigmaChi"],
         "DmuG": IN["dmuG"], "DmuChi": IN["dmuChi"], "LognGal": IN["lognGal"], "LognAGN": IN["lognAGN"], "RateRatio": RIN}
# the observed-like family (data/obslike/README.md): only the mass spectrum differs; offsets, spin, f_AGN as above
OBSJ = bv.DATA / "obslike" / "results" / "obslike_paper_values.json"
_obs = bv.jload(OBSJ)["design"]["core_theta"]
_obs_un = bv.jload(OBSJ)["design"]["unchanged_from_present_family"]
assert abs(_obs_un["host_dependent_offsets"]["dmu_G"] - IN["dmuG"]) < 1e-12 and abs(_obs_un["host_dependent_offsets"]["dmu_chi"] - IN["dmuChi"]) < 1e-12
assert abs(_obs["mu_chi"] - IN["muChi"]) < 1e-12 and abs(_obs["sigma_chi"] - IN["sigmaChi"]) < 1e-12 and abs(_obs_un["f_agn"] - IN["f"]) < 1e-12
INPUT_OBS = dict(INPUT, MuG=float(_obs["mu_G"]), SigmaG=float(_obs["sigma_G"]))


# ---------------------------------------------------------------- shapes and summaries
def _cdf(x, p):
    c = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
    return c / c[-1]


def summarize_xp(x, p, x0=None, n=400):
    """Median, 68%/90% equal-tailed intervals, the level of the input and a compact plotting grid."""
    x = np.asarray(x, float)
    p = np.maximum(np.asarray(p, float), 0.0)
    c = _cdf(x, p)
    q = lambda u: float(np.interp(u, c, x))
    lo, hi = q(1e-6), q(1 - 1e-6)
    pad = 0.15 * (hi - lo)
    lo, hi = max(x[0], lo - pad), min(x[-1], hi + pad)
    xs = np.linspace(lo, hi, n)
    ps = np.interp(xs, x, p)
    ps /= np.trapz(ps, xs)
    out = {"median": q(0.5), "ci68": [q(0.16), q(0.84)], "ci90": [q(0.05), q(0.95)], "ul90": q(0.90),
           "x": [float(f"{v:.7g}") for v in xs], "p": [float(f"{v:.6g}") for v in ps]}
    if x0 is not None:
        F = float(np.interp(x0, x, c)) if x[0] <= x0 <= x[-1] else (0.0 if x0 < x[0] else 1.0)
        out["level_of_input"] = abs(2 * F - 1)
        out["input"] = x0
    return out


def summarize_samples(s, lo, hi, x0=None):
    q = np.percentile(s, [5, 16, 50, 84, 90, 95])
    x, p, bw = bv.kde1(s, lo, hi)
    out = summarize_xp(x, p, x0)
    out.update({"median": float(q[2]), "ci68": [float(q[1]), float(q[3])], "ci90": [float(q[0]), float(q[5])], "ul90": float(q[4]),
                "kde_bandwidth": bw, "n_samples": int(len(s))})
    if x0 is not None:
        F = float(np.mean(np.asarray(s) < x0))
        out["level_of_input"] = abs(2 * F - 1)
    return out


def gauss_xp(med, hw, rng, x0=None, kind="normal"):
    """A Gaussian with median `med` and 90% half-width `hw`, truncated to the prior range `rng`; kind
    'halfnormal': density exp(-x^2/2s^2) on [0, ...) with its 90% upper limit at `hw`."""
    lo, hi = rng
    if kind == "halfnormal":
        s = hw / Z90
        x = np.linspace(lo, min(hi, lo + 8 * s), 4001)
        p = np.exp(-0.5 * ((x - lo) / s) ** 2)
    else:
        s = hw / Z90
        a, b = max(lo, med - 8 * s), min(hi, med + 8 * s)
        x = np.linspace(a, b, 4001)
        p = np.exp(-0.5 * ((x - med) / s) ** 2)
    return summarize_xp(x, p, x0)


def joint_grid(x, y, P, n=110):
    """Downsample a 2-D density to at most n x n points (linear interpolation), normalised."""
    from scipy.interpolate import RegularGridInterpolator
    xs = np.linspace(x[0], x[-1], min(n, len(x)))
    ys = np.linspace(y[0], y[-1], min(n, len(y)))
    Q = RegularGridInterpolator((x, y), P)(np.stack(np.meshgrid(xs, ys, indexing="ij"), -1))
    Q = np.maximum(Q, 0)
    Q /= np.trapz(np.trapz(Q, ys, axis=1), xs)
    return {"x": [float(f"{v:.7g}") for v in xs], "y": [float(f"{v:.7g}") for v in ys], "p": [[float(f"{v:.5g}") for v in row] for row in Q]}


def joint_gauss(qx, qy, sx, sy, rho, n=261):
    """2-D Gaussian density from two 1-D summaries (median, 90% half-width) and a correlation, on a lattice
    reaching 5 sd along each axis (cut at the prior edges): the 90% region (2.15 sd) lies well inside it, and the
    thin tips of strongly correlated regions are resolved."""
    mx, my = qx["median"], qy["median"]
    hx = 0.5 * (qx["ci90"][1] - qx["ci90"][0])
    hy = 0.5 * (qy["ci90"][1] - qy["ci90"][0])
    s1, s2 = hx / Z90, hy / Z90
    xs = np.linspace(max(PRIOR[sx][0], mx - 5 * s1), min(PRIOR[sx][1], mx + 5 * s1), n)
    ys = np.linspace(max(PRIOR[sy][0], my - 5 * s2), min(PRIOR[sy][1], my + 5 * s2), n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    u, v = (X - mx) / s1, (Y - my) / s2
    P = np.exp(-0.5 * (u * u - 2 * rho * u * v + v * v) / (1 - rho * rho))
    return joint_grid(xs, ys, P, n) | {"params": [sx, sy], "correlation": rho, "shape": "Gaussian"}


def joint_samples(a, b, ra, rb, sx, sy, nb=60, smooth=1.4):
    from scipy.ndimage import gaussian_filter
    H, xe, ye = np.histogram2d(a, b, bins=nb, range=[ra, rb])
    H = gaussian_filter(H.astype(float), smooth, mode="reflect")
    xs, ys = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    return joint_grid(xs, ys, H, nb) | {"params": [sx, sy], "correlation": float(np.corrcoef(a, b)[0, 1]),
                                        "shape": f"{nb}x{nb} histogram on the prior box, Gaussian smoothing {smooth} bins"}


def invented_comb(sA, N, x0, bias=0.0, rng=None, kind="normal"):
    """Combination over N realizations from realization A (header rule): A combined with N - 1 realizations of
    A's width centred at their expectation x0 + bias."""
    if x0 is None:
        x0 = sA["median"]
    med = x0 + bias + (sA["median"] - x0 - bias) / N
    hw = 0.5 * (sA["ci90"][1] - sA["ci90"][0]) / math.sqrt(N)
    if kind == "halfnormal":
        return gauss_xp(0.0, sA["ul90"] / math.sqrt(N), rng, x0, "halfnormal")
    return gauss_xp(med, hw, rng, x0)


# ---------------------------------------------------------------- the entries
def entry(events, analysis, catalogs, free, stage, key):
    N, why = NREAL[key]
    return {"events": events, "analysis": analysis, "catalogs": catalogs, "free": free, "stage": f"{STAGE_TITLE[stage]}, {N} realizations",
            "N_comb": N, "N_comb_source": why, "realizations": RUNREAL.get(key),
            "final": False, "quantities": {}, "joint": {}, "LB": None}


def q_put(E, q, A, comb, srcA, srcC, fileA=None, fileC=None, x0="default", note=None):
    x0 = INPUT.get(q) if x0 == "default" else x0
    E["quantities"][q] = {"A": A, "Comb": comb, "source": srcA, "comb_source": srcC, "file": fileA, "comb_file": fileC,
                          "input": x0, "units": UNIT[q], "note": note}


def rel(p):
    return bv.rel(p)


def build():
    K = {}
    EXT = {"H0": [IN["H0"]], "f": [IN["f"]]}
    # ======== population known, complete catalogs (Analysis 2; f = 0 and f = 1 columns) ========
    Ls = []
    for s in bv.C1_SEEDS:
        H, F, L, _ = bv.load_hf(bv.A2 / f"joint_s{s}.h5")
        Ls.append(L)
    gA = bv.GridPost(["H0", "f"], [H, F], Ls[0], [4, 10], extra=dict(EXT), tag="ph_joint_s100")
    gC = bv.GridPost(["H0", "f"], [H, F], np.sum(Ls, axis=0), [4, 10], extra=dict(EXT), tag="ph_joint_comb")
    fa = rel(bv.A2 / "joint_s100.h5")
    fc = f"{rel(bv.A2)}/joint_s<seed>.h5, the {len(bv.C1_SEEDS)} realizations {bv.C1_SEEDS} (log-likelihoods summed)"
    E = entry("equal populations", "population known", "both catalogs, complete", "H0, f_AGN", "known", "KnownBoth")
    E["comb_realizations_used"] = list(bv.C1_SEEDS)
    for q, n in (("Hzero", "H0"), ("fAGN", "f")):
        q_put(E, q, summarize_xp(*gA.marg1(n), INPUT[q]), summarize_xp(*gC.marg1(n), INPUT[q]), "exploratory", "exploratory", fa, fc)
    lr = [math.log(bv.survey_counts(s_)["gal"] / bv.survey_counts(s_)["agn"]) for s_ in bv.C1_SEEDS]
    for g, key in ((gA, "A"), (gC, "Comb")):
        ratio = _ratio100() if key == "A" else math.exp(float(np.mean(lr)))   # as RateRatioJointCatA / RateRatioJointComb
        x, p = g.marg1("f")
        m = (x > 1e-9) & (x < 1 - 1e-9)
        r = x[m] / (1 - x[m]) * ratio
        pr = p[m] * (1 - x[m]) ** 2 / ratio          # density in R
        o = np.argsort(r)
        rr = np.linspace(r.min(), min(r.max(), 400), 20001)
        sm = summarize_xp(rr, np.interp(rr, r[o], pr[o]), RIN)
        Rx = bv.rate_ratio_constant(g.summary("f"), ratio)          # exact: quantiles of f mapped one to one
        sm.update({k: Rx[k] for k in ("median", "ci68", "ci90")})
        sm["ul90"] = float(np.interp(0.9, Rx["_cdf"][1], Rx["_cdf"][0]))
        sm["level_of_input"] = bv.level_from_cdf(Rx["_cdf"], RIN)[0]
        E.setdefault("_rr", {})[key] = sm
    q_put(E, "RateRatio", E["_rr"]["A"], E["_rr"]["Comb"], "exploratory", "exploratory", fa, fc, note="R = [f/(1-f)] N_gal/N_AGN (Z_k = N_k)")
    del E["_rr"]
    for key, g in (("A", gA), ("Comb", gC)):
        x1, x2, m = g.marg2("H0", "f")
        sel1 = (x1 > 62) & (x1 < 76)
        sel2 = x2 < 0.65
        E["joint"][key] = joint_grid(x1[sel1], x2[sel2], m[np.ix_(sel1, sel2)], 120) | {"params": ["Hzero", "fAGN"], "correlation": g.corr("H0", "f"), "shape": "stored grid"}
    E["LB"] = {"reference": "galaxy hosts only (f_AGN = 0), Savage-Dickey on the same grid, f_AGN prior [0, 1]", "source": "exploratory", "file": fa,
               "A": None, "Comb": None}
    K["KnownBoth"] = E
    for key, col, cat, desc in (("KnownGal", 0, "galaxy catalog alone", "f_AGN = 0 column"), ("KnownAGN", -1, "AGN catalog alone", "f_AGN = 1 column")):
        E = entry("equal populations", "population known", cat, "H0", "known", key)
        use = [s_ for s_ in bv.C1_SEEDS if s_ in RUNREAL[key]]          # the production realizations among the exploratory ones
        E["comb_realizations_used"] = use
        g1 = bv.GridPost(["H0"], [H], Ls[0][:, col], [8], extra={"H0": [IN["H0"]]}, tag=f"ph_{key}_A")
        g2 = bv.GridPost(["H0"], [H], np.sum([Ls[bv.C1_SEEDS.index(s_)] for s_ in use], axis=0)[:, col], [8], extra={"H0": [IN["H0"]]}, tag=f"ph_{key}_C")
        q_put(E, "Hzero", summarize_xp(*g1.marg1("H0"), IN["H0"]), summarize_xp(*g2.marg1("H0"), IN["H0"]), "exploratory", "exploratory",
              fa + f" ({desc})", f"{rel(bv.A2)}/joint_s<seed>.h5, the {len(use)} realizations {use} (log-likelihoods summed; {desc})",
              note="the posterior rises to the upper end of the prior" if col == -1 else None)
        K[key] = E
    E = entry("equal populations", "population known", "no catalog", "H0", "known", "KnownNone")
    A = gauss_xp(68.6, 3.2, PRIOR["Hzero"], IN["H0"])
    q_put(E, "Hzero", A, invented_comb(A, E["N_comb"], IN["H0"], rng=PRIOR["Hzero"]), "invented", "invented")
    K["KnownNone"] = E
    # ======== all-AGN-hosted and all-galaxy-hosted sets (Analysis 0) ========
    ens = bv.jload(bv.A0 / "h0_pure_tracer_ens93.json")
    seeds = list(ens["seeds_requested"])
    for key, tr, cat in (("AGNhosted", "agn", "AGN catalog"), ("Galhosted", "gal", "galaxy catalog")):
        E = entry("all AGN-hosted" if tr == "agn" else "all galaxy-hosted", "population known", cat, "H0", "pure", key)
        use = [s_ for s_ in seeds if s_ in RUNREAL[key]]               # the production realizations among the 93 exploratory ones
        assert len(use) == E["N_comb"], (key, use)
        E["comb_realizations_used"] = use
        acc = None
        for s in use:
            with h5py.File(bv.A0 / f"h0_pure{tr}_targeted_s{s}.h5") as h:
                Hp, L = h["H0_grid"][()], h["log_likelihood"][()]
            if s == 100:
                g1 = bv.GridPost(["H0"], [Hp], L, [8], extra={"H0": [IN["H0"]]}, tag=f"ph_pure{tr}_A")
            acc = L if acc is None else acc + L
        g2 = bv.GridPost(["H0"], [Hp], acc, [16], extra={"H0": [IN["H0"]]}, tag=f"ph_pure{tr}_C")
        q_put(E, "Hzero", summarize_xp(*g1.marg1("H0"), IN["H0"]), summarize_xp(*g2.marg1("H0"), IN["H0"]), "exploratory", "exploratory",
              rel(bv.A0 / f"h0_pure{tr}_targeted_s100.h5"), f"{rel(bv.A0)}/h0_pure{tr}_targeted_s<seed>.h5, the {len(use)} realizations {use} summed")
        K[key] = E
    # ======== no AGN-hosted mergers; one in ten (invented) ========
    for key, f0, fmed, fhw, lb in (("CtrlZero", 0.0, None, 0.05, -1.0), ("CtrlTenth", 0.10, 0.11, 0.05, 2.5)):
        E = entry(f"mergers with {'no AGN hosts' if f0 == 0 else 'one in ten hosted by AGN'}", "population known", "both catalogs, complete", "H0, f_AGN", "ctrl", key)
        A = gauss_xp(67.9, 1.7, PRIOR["Hzero"], IN["H0"])
        q_put(E, "Hzero", A, invented_comb(A, E["N_comb"], IN["H0"], rng=PRIOR["Hzero"]), "invented", "invented",
              note="H0 width as for population known with both catalogs")
        if f0 == 0:
            A = gauss_xp(0.0, fhw, PRIOR["fAGN"], 0.0, "halfnormal")
            C = invented_comb(A, E["N_comb"], 0.0, rng=PRIOR["fAGN"], kind="halfnormal")
            note = "upper limit: 90% upper limit 0.05 (half-normal)"
        else:
            A = gauss_xp(fmed, fhw, PRIOR["fAGN"], f0)
            C = invented_comb(A, E["N_comb"], f0, rng=PRIOR["fAGN"])
            note = None
        q_put(E, "fAGN", A, C, "invented", "invented", x0=f0, note=note)
        E["LB"] = {"reference": "galaxy hosts only (f_AGN = 0), Savage-Dickey, f_AGN prior [0, 1]", "source": "invented", "file": None,
                   "A": lb, "Comb": -1.5 if lb < 0 else lb * E["N_comb"]}
        K[key] = E
    # ======== flux-limited catalogs, densities known (Analysis 3) ========
    NC = bv.norm_cfg(100)
    ng, na = 10 ** IN["lognGal"], 10 ** IN["lognAGN"]
    for rk, word, mlim in bv.RUNGS:
        E = entry("equal populations", "population known, number densities from the catalog counts", f"both catalogs, m < {mlim}", "H0, f_AGN", "flux", f"Flux{word}")
        p = bv.A3 / f"joint_{rk}_s100.h5"
        g = bv.hf_post(p, f"ph_a3_{rk}")
        for q, n in (("Hzero", "H0"), ("fAGN", "f")):
            A = summarize_xp(*g.marg1(n), INPUT[q])
            q_put(E, q, A, invented_comb(A, E["N_comb"], INPUT[q], rng=PRIOR[q]), "exploratory", "invented", rel(p))
        rg, ra = NC[rk]["gal"], NC[rk]["agn"]
        R = bv.rate_ratio_grid(g, lambda h, rg=rg, ra=ra: bv.Zk(rg, ng, h) / bv.Zk(ra, na, h))
        rr, F = R["_cdf"]
        pr = np.gradient(F, rr)
        A = summarize_xp(rr, pr, RIN)
        A.update({k: R[k] for k in ("median", "ci68", "ci90")})
        q_put(E, "RateRatio", A, invented_comb(A, E["N_comb"], RIN, rng=PRIOR["RateRatio"]), "exploratory", "invented", rel(p),
              note="hosts normalised to z = 5 (stored run); the production convention normalises to z = 1.10")
        K[f"Flux{word}"] = E
    # ======== number densities inferred (Analysis 5) ========
    for rk, word, mlim in bv.RUNGS:
        E = entry("equal populations", "number densities inferred", f"both catalogs, m < {mlim}", "H0, f_AGN, log10 n_gal, log10 n_AGN", "free", f"Free{word}")
        ph5, pj = bv.A5 / f"campaign_{rk}_dynesty_s100.h5", bv.A5 / f"campaign_{rk}_dynesty_s100.json"
        c, _, _, j = bv.load_a5(ph5, pj)
        pri = j["priors"]
        cnt = bv.survey_counts(100, rk)
        rg, ra = dict(NC[rk]["gal"], N=cnt["gal"]), dict(NC[rk]["agn"], N=cnt["agn"])
        Rs = c.col("fcat_2") / (1 - c.col("fcat_2")) * bv.Zk(rg, 10 ** c.col("log10n0"), c.col("H0")) / bv.Zk(ra, 10 ** c.col("log10n0_c2"), c.col("H0"))
        cols = {"Hzero": (c.col("H0"), pri["H0"]), "fAGN": (c.col("fcat_2"), pri["fcat_2"]), "LognGal": (c.col("log10n0"), pri["log10n0"]),
                "LognAGN": (c.col("log10n0_c2"), pri["log10n0_c2"]), "RateRatio": (Rs, (0.0, float(np.percentile(Rs, 99.9)) * 1.3))}
        I = bv.induced_density(rk)
        for q, (s, rng) in cols.items():
            A = summarize_samples(s, *rng, x0=INPUT[q])
            if q in ("Hzero", "RateRatio"):
                C, cs = invented_comb(A, E["N_comb"], INPUT[q], rng=PRIOR[q]), "invented"
                note = None
            else:
                C, cs = A, "invented (the prior dominates: the combination is taken equal to realization A)"
                note = "follows its prior along the curve of constant rate per host"
            q_put(E, q, A, C, "exploratory", cs, rel(ph5), note=note)
            if q in ("LognGal", "LognAGN"):
                E["quantities"][q]["A"]["prior_ul90"] = float(bv.wquant(I["log10n0" if q == "LognGal" else "log10n0_c2"], I["w"], 0.90))
        E["joint"]["A"] = joint_samples(c.col("fcat_2"), c.col("log10n0_c2"), tuple(pri["fcat_2"]), tuple(pri["log10n0_c2"]), "fAGN", "LognAGN", nb=50)
        # the region the priors allow once the rate per host is measured (prior-induced density, Task 1)
        Hh, xe, ye = np.histogram2d(I["f"], I["log10n0_c2"], bins=50, range=[tuple(pri["fcat_2"]), tuple(pri["log10n0_c2"])], weights=I["w"])
        from scipy.ndimage import gaussian_filter
        Hh = gaussian_filter(Hh, 1.0, mode="reflect")
        E["joint"]["prior_induced"] = joint_grid(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), Hh, 50) | {
            "params": ["fAGN", "LognAGN"], "shape": "2e6 prior draws weighted by the stored known-density likelihood mapped through the identity (build_values.induced_density)",
            "file": rel(bv.A3 / f"joint_{rk}_s100.h5")}
        K[f"Free{word}"] = E
    # ======== single population, equal populations (invented) ========
    E = entry("equal populations", "single population", "both catalogs, complete", "H0, f_AGN, mass-peak location and width, spin mean and width", "single", "SingleBoth")
    spec = {"Hzero": (68.0, 2.6), "fAGN": (0.29, 0.09), "MuG": (34.9, 0.6), "SigmaG": (5.0, 0.4), "MuChi": (0.000, 0.010), "SigmaChi": (0.100, 0.008)}
    for q, (m, hw) in spec.items():
        A = gauss_xp(m, hw, PRIOR[q], INPUT[q])
        q_put(E, q, A, invented_comb(A, E["N_comb"], INPUT[q], rng=PRIOR[q]), "invented", "invented")
    E["joint"]["A"] = joint_gauss(E["quantities"]["Hzero"]["A"], E["quantities"]["MuG"]["A"], "Hzero", "MuG", -0.5)
    E["joint"]["Comb"] = joint_gauss(E["quantities"]["Hzero"]["Comb"], E["quantities"]["MuG"]["Comb"], "Hzero", "MuG", -0.5)
    K["SingleBoth"] = E
    E = entry("equal populations", "single population", "no catalog", "H0, mass-peak location and width, spin mean and width", "single", "SingleNone")
    spec = {"Hzero": (69.0, 9.0), "MuG": (34.6, 1.6), "SigmaG": (5.0, 0.5), "MuChi": (0.000, 0.012), "SigmaChi": (0.100, 0.010)}
    for q, (m, hw) in spec.items():
        A = gauss_xp(m, hw, PRIOR[q], INPUT[q])
        q_put(E, q, A, invented_comb(A, E["N_comb"], INPUT[q], rng=PRIOR[q]), "invented", "invented",
              note="mass-peak width, spin mean and width not stated in the skeleton: invented at the inputs" if q in ("SigmaG", "MuChi", "SigmaChi") else None)
    E["joint"]["A"] = joint_gauss(E["quantities"]["Hzero"]["A"], E["quantities"]["MuG"]["A"], "Hzero", "MuG", -0.9)
    E["joint"]["Comb"] = joint_gauss(E["quantities"]["Hzero"]["Comb"], E["quantities"]["MuG"]["Comb"], "Hzero", "MuG", -0.9)
    K["SingleNone"] = E
    # ======== host-dependent events, peak-dominated and observed-like spectra ========
    p13 = bv.A13 / "a13core_e7c3007_rslice_cap20_historical_n200_s1.npz"
    c13, j13 = bv.load_chain_npz(p13, bv.A13 / "a13core_e7c3007_rslice_cap20_historical_n200_s1.json")
    box13 = {b[0]: (float(b[1]), float(b[2])) for b in j13["box"]}
    rho13 = c13.corr("dmu_G", "dmu_chi")
    for pre, stage_s, stage_n, label in (("HD", "hdsingle", "hdnone", "peak-dominated"), ("Obs", "obs", "obs", "observed-like")):
        ev = "host-dependent populations" + ("" if pre == "HD" else ", observed-like spectrum")
        INP = INPUT if pre == "HD" else INPUT_OBS
        inv_note = None if pre == "HD" else (f"observed-like inputs (data/obslike): mass peak {INP['MuG']:g} Msun, width {INP['SigmaG']:g} Msun; "
                                             "the invented medians keep their offsets from the inputs of the earlier placeholder")
        # single population (biased); the observed-like medians sit at the same offsets from the observed-like inputs
        E = entry(ev, "single population", "both catalogs, complete", "six", stage_s, f"{pre}Single")
        if pre == "HD":
            spec = {"Hzero": (70.9, 2.6, 3.0), "fAGN": (0.24, 0.09, -0.05), "MuG": (36.4, 0.6, 1.4), "SigmaG": (5.6, 0.4, 0.6), "MuChi": (0.030, 0.010, 0.03), "SigmaChi": (0.110, 0.008, 0.01)}
        else:
            spec = {"Hzero": (69.0, 5.0, 1.0), "fAGN": (0.27, 0.10, -0.03), "MuG": (INP["MuG"] + 0.4, 1.2, 0.4), "SigmaG": (INP["SigmaG"] + 0.3, 0.8, 0.3),
                    "MuChi": (0.030, 0.014, 0.03), "SigmaChi": (0.110, 0.012, 0.01)}
        for q, (m, hw, b) in spec.items():
            x0 = INP[q] if q in ("Hzero", "fAGN") else None
            A = gauss_xp(m, hw, PRIOR[q], x0)
            ref = INP[q]
            q_put(E, q, A, invented_comb(A, E["N_comb"], ref, bias=b, rng=PRIOR[q]), "invented", "invented (the bias survives the combination)", x0=x0,
                  note=("one population for a mixture: no single input" if x0 is None else None) if inv_note is None else inv_note)
        K[f"{pre}Single"] = E
        # host-dependent populations
        E = entry(ev, "host-dependent populations", "both catalogs, complete", "eight", "hdhost" if pre == "HD" else "obs", f"{pre}Host")
        if pre == "HD":
            for cn, q in CHAIN_Q.items():
                A = summarize_samples(c13.col(cn), *box13[cn], x0=INPUT[q])
                q_put(E, q, A, invented_comb(A, E["N_comb"], INPUT[q], rng=PRIOR[q]), "exploratory", "invented", rel(p13),
                      note="H0 prior [60, 76] in the exploratory run" if q == "Hzero" else None)
            E["joint"]["A"] = joint_samples(c13.col("dmu_G"), c13.col("dmu_chi"), box13["dmu_G"], box13["dmu_chi"], "DmuG", "DmuChi", nb=50) | {"file": rel(p13)}
        else:
            spec = {"Hzero": (67.9, 5.2), "fAGN": (0.30, 0.08), "MuG": (INP["MuG"], 1.2), "SigmaG": (INP["SigmaG"], 0.8), "MuChi": (0.000, 0.014), "SigmaChi": (0.100, 0.012),
                    "DmuG": (4.5, 3.5), "DmuChi": (0.10, 0.04)}
            for q, (m, hw) in spec.items():
                A = gauss_xp(m, hw, PRIOR[q], INP[q])
                q_put(E, q, A, invented_comb(A, E["N_comb"], INP[q], rng=PRIOR[q]), "invented", "invented", x0=INP[q],
                      note=inv_note if q in ("MuG", "SigmaG") else None)
            E["joint"]["A"] = joint_gauss(E["quantities"]["DmuG"]["A"], E["quantities"]["DmuChi"]["A"], "DmuG", "DmuChi", rho13)
        E["joint"]["Comb"] = joint_gauss(E["quantities"]["DmuG"]["Comb"], E["quantities"]["DmuChi"]["Comb"], "DmuG", "DmuChi", rho13) | {
            "correlation_from": rel(p13)}
        lb = 9.0 if pre == "HD" else 3.0
        E["LB"] = {"reference": "the single-population analysis of the same events, same priors (Table I)", "source": "invented", "file": None,
                   "A": lb, "Comb": lb * E["N_comb"]}
        K[f"{pre}Host"] = E
        # no catalog
        E = entry(ev, "single population", "no catalog", "five", stage_n, f"{pre}None")
        if pre == "HD":
            spec = {"Hzero": (73.0, 9.0, 5.0), "MuG": (36.4, 1.6, 1.4), "SigmaG": (5.6, 0.5, 0.6), "MuChi": (0.030, 0.012, 0.03), "SigmaChi": (0.110, 0.010, 0.01)}
        else:
            spec = {"Hzero": (70.0, 14.0, 2.0), "MuG": (INP["MuG"] + 0.4, 2.4, 0.4), "SigmaG": (INP["SigmaG"] + 0.3, 1.0, 0.3), "MuChi": (0.030, 0.016, 0.03),
                    "SigmaChi": (0.110, 0.014, 0.01)}
        for q, (m, hw, b) in spec.items():
            x0 = INP[q] if q == "Hzero" else None
            A = gauss_xp(m, hw, PRIOR[q], x0)
            q_put(E, q, A, invented_comb(A, E["N_comb"], INP[q], bias=b, rng=PRIOR[q]), "invented", "invented (the bias survives the combination)", x0=x0,
                  note=inv_note)
        K[f"{pre}None"] = E
    # Bayes factors of the anchor from the stored grids (exploratory)
    vals = {}
    for g, key in ((gA, "A"), (gC, "Comb")):
        tot = g.log_int()[0]
        col = bv.GridPost(["H0"], [H], (Ls[0] if key == "A" else np.sum(Ls, axis=0))[:, 0], [8], extra={"H0": [IN["H0"]]}, tag=f"ph_lb_{key}").log_int()[0]
        vals[key] = (tot - col) / bv.LN10
    K["KnownBoth"]["LB"].update(vals)
    return K


def production_override(K):
    pm = json.loads(PMAP.read_text()) if PMAP.exists() else {}
    for key, path in pm.items():
        p = Path(path) if Path(path).is_absolute() else bv.W / path
        if key in K and p.exists():
            E = json.loads(p.read_text())
            E.update({"final": True, "production_file": str(path)})
            for q in E.get("quantities", {}).values():
                q["source"] = q["comb_source"] = "production"
            K[key] = E
    return pm


def main():
    K = build()
    if not PMAP.exists():
        PMAP.write_text(json.dumps({}, indent=1) + "\n")
    pm = production_override(K)
    doc = {"_what": "placeholder results of the PRD paper (notes/PAPER_SKELETON.md Secs. 6 and 7), one entry per key; written by scripts/placeholder_results.py",
           "_conventions": __doc__, "_inputs": {**INPUT, "RateRatio_note": "[0.30/0.70] N_gal/N_AGN of the reference catalog pair"},
           "_inputs_observed_like": {**INPUT_OBS, "file": bv.rel(OBSJ)},
           "_realizations_combined": {k: {"N": v[0], "source": v[1], "realizations": RUNREAL.get(k)} for k, v in NREAL.items()},
           "_production_map": pm, "keys": bv.jsonable(K)}
    OUTJ.write_text(json.dumps(doc, indent=0))
    n = sum(len(E["quantities"]) for E in K.values())
    print(f"wrote {OUTJ.relative_to(PRD)}: {len(K)} keys, {n} key-quantities")
    return doc


if __name__ == "__main__":
    main()
