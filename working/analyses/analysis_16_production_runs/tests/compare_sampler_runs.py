#!/usr/bin/env python
"""Compare finished nested runs of one model (CPU): are they draws of the same posterior and evidence?

    JAX_PLATFORMS=cpu <core venv python> tests/compare_sampler_runs.py RUN RUN [RUN ...] [--out JSON]
        [--names H0,f_agn,...] [--n-null 200] [--n-draw 2000] [--merge] [--seed 0]

RUN: a part directory (runs/<run>/nested_k: results.pkl, result.json, state.save), a run directory (each
finished nested_k part becomes one RUN; --merge adds the parts merged with dynesty.utils.merge_runs, as
prod_summary.py merges them), or a dynesty results.pkl. LABEL=PATH names a RUN.

Per RUN: the sampler setting (state.save or result.json), calls, iterations, ln Z and its error, per parameter
the weighted median and 90% interval (5% and 95% quantiles of the importance-weighted dead and final live
points, the weights the driver uses: the corrected likelihood), and the diagnostics of
tests/nested_diagnostics.py over the whole run (the corrected insertion-index test, the bound volume against
the contour volume, the flatness of the live set, flags).

Per pair of RUNs:
  - per parameter: difference of medians in units of the 90% half-width (the mean of the two half-widths),
    and the two-sample Kolmogorov distance between the weighted marginals;
  - joint: the energy distance between n_draw equal-weight draws of each posterior, coordinates scaled by
    the pooled weighted standard deviations;
  - their expectation for two independent correct runs, by resampling: each single run is replaced by a
    simulated repeat of itself (dynesty.utils.resample_run, bootstrap over its live-point threads; Higson et
    al. 2018), and the statistics are computed between two independent repeats of the same run, n_null times
    for each run of the pair (pooled). Reported: the mean and 95% point of that distribution and the fraction
    of it at or above the observed value (p). Not jitter_run: the driver weights the samples with dynesty's
    expected volumes, and jittering them halves the Kish ESS of a repeat, which inflates the null (on Analysis
    11D's pair: Kolmogorov distance between repeats 0.051 with jitter, 0.037 without, 0.034 expected for two
    independent runs of ESS 1,300, 0.029 observed);
  - ln Z difference in units of the combined error sqrt(err1^2 + err2^2).
A merged RUN contributes no repeats to the null (resample_run does not apply to merged runs): its pairs
take the null of the single run they are compared with, and a merged run has no insertion diagnostics.
"""
import argparse
import itertools
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nested_diagnostics as N  # noqa: E402

warnings.filterwarnings("ignore", message=".*sampling was stopped short.*")


# ============================================================================ inputs
def _names_of(part_dir):
    for f in ("result.json", "config.json"):
        p = Path(part_dir) / f
        if p.exists():
            d = json.loads(p.read_text())
            if d.get("free"):
                return list(d["free"])
    p = Path(part_dir).parent / "config.json"
    if p.exists():
        return list(json.loads(p.read_text()).get("free") or [])
    return None


def expand(spec, merge=False):
    """LABEL=PATH -> [items]: {label, path, res (dynesty Results), names, part_dir, merged}."""
    lab, _, path = spec.rpartition("=") if "=" in spec else ("", "", spec)
    p = Path(path)
    items = []
    if p.is_dir() and not (p / "results.pkl").exists():
        parts = sorted(q for q in p.glob("nested_[0-9]*") if (q / "results.pkl").exists()
                       and json.loads((q / "result.json").read_text()).get("finished"))
        if not parts:
            raise SystemExit(f"[fatal] {p}: no finished nested part")
        for q in parts:
            items += expand(f"{(lab or p.name)}:{q.name}={q}")
        if merge and len(parts) > 1:
            from dynesty.utils import merge_runs
            res = merge_runs([it["res"] for it in items])
            items.append({"label": f"{lab or p.name}:merged", "path": str(p), "res": res, "names": items[0]["names"],
                          "part_dir": None, "merged": True})
        return items
    if p.is_dir():
        rec = json.loads((p / "result.json").read_text()) if (p / "result.json").exists() else {}
        if rec and not rec.get("finished"):
            print(f"[warning] {p}: result.json says finished = {rec.get('finished')} (dry run?)")
        with open(p / "results.pkl", "rb") as fh:
            res = pickle.load(fh)
        return [{"label": lab or str(p), "path": str(p), "res": res, "names": _names_of(p), "part_dir": str(p),
                 "merged": False}]
    with open(p, "rb") as fh:
        res = pickle.load(fh)
    return [{"label": lab or p.name, "path": str(p), "res": res, "names": None, "part_dir": None, "merged": False}]


# ============================================================================ statistics
def weights(res):
    lw = np.asarray(res["logwt"], float)
    w = np.exp(lw - lw.max())
    return w / w.sum()


def wquantile(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    c = (c - 0.5 * w[o]) / c[-1]
    return np.interp(q, c, x[o])


def summary(res, names):
    X = np.asarray(res["samples"], float)
    w = weights(res)
    out = {}
    for i, n in enumerate(names):
        q05, q50, q95 = wquantile(X[:, i], w, [0.05, 0.5, 0.95])
        mu = float(np.sum(w * X[:, i]))
        out[n] = {"median": float(q50), "ci90": [float(q05), float(q95)], "halfwidth90": float(0.5 * (q95 - q05)),
                  "mean": mu, "sd": float(np.sqrt(np.sum(w * (X[:, i] - mu) ** 2)))}
    return out


def wks(x, w, y, v):
    """Two-sample Kolmogorov distance between the weighted empirical distributions of x and y."""
    z = np.concatenate([x, y])
    o = np.argsort(z, kind="mergesort")
    cw = np.concatenate([w, np.zeros(len(y))])[o].cumsum()
    cv = np.concatenate([np.zeros(len(x)), v])[o].cumsum()
    return float(np.max(np.abs(cw - cv)))


def draws(res, n, rng):
    """n equal-weight draws (systematic resampling of the importance weights)."""
    w = weights(res)
    pos = (rng.random() + np.arange(n)) / n
    idx = np.minimum(np.searchsorted(np.cumsum(w), pos), len(w) - 1)
    return np.asarray(res["samples"], float)[idx]


def energy(Xa, Xb, scale):
    from scipy.spatial.distance import cdist
    A, B = Xa / scale, Xb / scale
    return float(2 * cdist(A, B).mean() - cdist(A, A).mean() - cdist(B, B).mean())


def pair_stats(ra, rb, names, scale, n_draw, rng):
    Xa, Xb = np.asarray(ra["samples"], float), np.asarray(rb["samples"], float)
    wa, wb = weights(ra), weights(rb)
    ks = {n: wks(Xa[:, i], wa, Xb[:, i], wb) for i, n in enumerate(names)}
    sa, sb = summary(ra, names), summary(rb, names)
    dmed = {n: (sa[n]["median"] - sb[n]["median"]) / max(0.5 * (sa[n]["halfwidth90"] + sb[n]["halfwidth90"]), 1e-300)
            for n in names}
    e = energy(draws(ra, n_draw, rng), draws(rb, n_draw, rng), scale)
    return ks, dmed, e


def simulate(res, rng):
    from dynesty.utils import resample_run
    return resample_run(res, rstate=rng)


def null_stats(items_pair, names, scale, n_draw, n_null, rng):
    """Statistics between two independent simulated repeats of the same run, n_null times per run."""
    rows = []
    for it in items_pair:
        if it["merged"]:
            continue
        for _ in range(n_null):
            a, b = simulate(it["res"], rng), simulate(it["res"], rng)
            rows.append(pair_stats(a, b, names, scale, n_draw, rng))
    return rows


def null_summary(obs, vals):
    v = np.asarray(vals, float)
    if v.size == 0:
        return {"observed": obs, "null_mean": None, "null_95": None, "p": None, "n_null": 0}
    return {"observed": obs, "null_mean": float(v.mean()), "null_95": float(np.quantile(v, 0.95)),
            "p": float((1 + np.sum(v >= obs)) / (1 + v.size)), "n_null": int(v.size)}


# ============================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--names", default=None, help="comma list of parameter names (default: the runs' records)")
    ap.add_argument("--n-null", type=int, default=200, help="simulated repeats per run for the null (default 200)")
    ap.add_argument("--n-draw", type=int, default=2000, help="equal-weight draws per posterior for the energy distance")
    ap.add_argument("--merge", action="store_true", help="also compare the merged parts of each run directory")
    ap.add_argument("--no-diagnostics", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    items = [it for spec in a.runs for it in expand(spec, merge=a.merge)]
    if len(items) < 2:
        raise SystemExit("[fatal] need two or more runs")
    names = a.names.split(",") if a.names else (items[0]["names"] or
                                                [f"x{i}" for i in range(np.asarray(items[0]["res"]["samples"]).shape[1])])
    for it in items:
        d = np.asarray(it["res"]["samples"]).shape[1]
        if d != len(names) or (it["names"] and it["names"] != names):
            raise SystemExit(f"[fatal] {it['label']}: parameters {it['names'] or d} differ from {names}")
    out = {"written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "names": names,
           "settings": {"n_null": a.n_null, "n_draw": a.n_draw, "seed": a.seed}, "runs": [], "pairs": []}
    # pooled weighted sd per parameter: the scale of the energy distance
    sds = np.array([[summary(it["res"], names)[n]["sd"] for n in names] for it in items])
    scale = np.sqrt(np.mean(sds ** 2, axis=0))
    for it in items:
        r = it["res"]
        row = {"label": it["label"], "path": it["path"], "merged": it["merged"],
               "logZ": float(r["logz"][-1]), "logZerr": float(r["logzerr"][-1]),
               "ncall": int(np.sum(r["ncall"])), "niter": int(r["niter"]),
               "ess_kish": float(1.0 / np.sum(weights(r) ** 2)), "posterior": summary(r, names)}
        if not it["merged"] and not a.no_diagnostics:
            try:
                d = N.diagnose(it["part_dir"] or it["path"], label=it["label"])
                row["settings"] = d["settings"]
                row["diagnostics"] = {"insertion_whole": d["insertion"]["whole"],
                                      "insertion_after_plateau": d["insertion"]["after_plateau"],
                                      "n_initial_at_floor": d["insertion"]["n_initial_at_floor"],
                                      "blocks_p_lt_0.01": [d["insertion"]["n_blocks_p_lt_0.01"], d["insertion"]["n_blocks"]],
                                      "bound_minus_lnX_min": (d["geometry"].get("bound_minus_lnX") or {}).get("min"),
                                      "flatness": d["geometry"]["min_sv_ratio"], "flags": d["flags"]}
            except Exception as e:  # noqa: BLE001
                row["diagnostics"] = {"error": f"{type(e).__name__}: {e}"}
        out["runs"].append(row)
    for i, j in itertools.combinations(range(len(items)), 2):
        A, B = items[i], items[j]
        ks, dmed, e = pair_stats(A["res"], B["res"], names, scale, a.n_draw, rng)
        nul = null_stats((A, B), names, scale, a.n_draw, a.n_null, rng) if a.n_null > 0 else []
        za, zb = out["runs"][i], out["runs"][j]
        dz = za["logZ"] - zb["logZ"]
        pr = {"a": A["label"], "b": B["label"],
              "dlogZ": dz, "dlogZ_over_err": dz / np.hypot(za["logZerr"], zb["logZerr"]),
              "per_parameter": {n: {"dmedian_over_halfwidth90": dmed[n],
                                    "ks": null_summary(ks[n], [r[0][n] for r in nul]),
                                    "abs_dmedian_over_halfwidth90": null_summary(abs(dmed[n]), [abs(r[1][n]) for r in nul])}
                                for n in names},
              "energy": null_summary(e, [r[2] for r in nul])}
        out["pairs"].append(pr)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    # table
    print(f"parameters: {', '.join(names)}")
    for r in out["runs"]:
        s = r.get("settings") or {}
        dg = r.get("diagnostics") or {}
        ins = dg.get("insertion_whole") or {}
        print(f"  {r['label']}: {s.get('method', '?')} boot {s.get('bootstrap', '?')}; ln Z {r['logZ']:.3f} +- {r['logZerr']:.3f}; "
              f"calls {r['ncall']:,}, it {r['niter']:,}, ESS {r['ess_kish']:.0f}; insertion KS p "
              f"{N._f(ins.get('ks_p'), '.3g')} (mean u {N._f(ins.get('mean_u'), '.3f')}), blocks p<0.01 "
              f"{dg.get('blocks_p_lt_0.01')}, bound-lnX min {N._f(dg.get('bound_minus_lnX_min'), '.2f')}, flatness "
              f"{N._f(dg.get('flatness'), '.2g')}" + (f"; FLAGS {dg['flags']}" if dg.get("flags") else ""))
        print("      " + "  ".join(f"{n} {v['median']:.4g} [{v['ci90'][0]:.4g}, {v['ci90'][1]:.4g}]"
                                   for n, v in r["posterior"].items()))
    for pr in out["pairs"]:
        e = pr["energy"]
        print(f"  {pr['a']}  vs  {pr['b']}: dlnZ {pr['dlogZ']:+.3f} = {pr['dlogZ_over_err']:+.2f} sd; energy "
              f"{e['observed']:.2e} (null mean {N._f(e['null_mean'], '.2e')}, 95% {N._f(e['null_95'], '.2e')}, "
              f"p {N._f(e['p'], '.3f')})")
        print(f"      {'param':10s} {'dmed/hw90':>9s} {'KS':>7s} {'KS null mean':>12s} {'KS null 95%':>11s} {'p(KS)':>6s} "
              f"{'p(|dmed|)':>9s}")
        for n, v in pr["per_parameter"].items():
            k, dm = v["ks"], v["abs_dmedian_over_halfwidth90"]
            print(f"      {n:10s} {v['dmedian_over_halfwidth90']:+9.3f} {k['observed']:7.4f} {N._f(k['null_mean'], '12.4f')} "
                  f"{N._f(k['null_95'], '11.4f')} {N._f(k['p'], '6.3f')} {N._f(dm['p'], '9.3f')}")
    if a.out:
        print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
