#!/usr/bin/env python3
"""The five-realisation closure table for the joint (H0, f_AGN) measurement.

Reads, per seed, `results/joint_s<seed>.json`, `results/fscan_s<seed>.json` and
`results/h0scan_s<seed>.json`, plus the seed's own `META.json` for the REALISED
host fraction, and writes

  results/joint_summary.json   the full table: per-seed medians, 68/90% intervals,
                               offsets against both f references, coverage flags,
                               rho, the guard/N_eff behaviour, lane agreement
  results/h0_fagn_joint.json   the clean paper-hook schema, mirroring
                               analysis_1's results/h0_single_tracer.json

TWO TRUTH REFERENCES FOR f, and both are reported.

  * the REALISED host fraction (n_AGN / 1000 for that seed).  This is the closure
    reference.  The mixture weight is estimated from the events that were actually
    drawn, and with perfect host identification the maximum-likelihood estimate of
    a mixture weight from N events IS the realised fraction; the difference between
    the realised fraction and the planted value is a property of the mock's own
    binomial draw, not of the estimator, and including it would charge the
    estimator for noise it cannot see.
  * the PLANTED value 0.30, the population parameter the mock was built from.  An
    offset against this reference carries the extra binomial term
    sqrt(0.3 x 0.7 / 1000) = 0.0145 per realisation (0.0065 on the five-seed mean).

Both columns appear in the table and in the JSON; the README quotes both.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_ROOT = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data")
H0_TRUTH = 67.74
F_PLANTED = 0.30


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=[100, 101, 102, 103, 105])
    ap.add_argument("--resdir", default=str(ROOT / "results"))
    ap.add_argument("--data_root", default=str(DATA_ROOT))
    ap.add_argument("--lane_seed", type=int, default=100,
                    help="seed carrying the popuni cross-check lane")
    return ap.parse_args(argv)


def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def realised_f(data_root, seed):
    m = json.loads((Path(data_root) / f"seed{seed}" / "META.json").read_text())
    r = m["stages"]["events"]["realised"]
    n_agn, n_gal = int(r["n_host_agn"]), int(r["n_host_gal"])
    return {"n_host_agn": n_agn, "n_host_gal": n_gal,
            "n_events": n_agn + n_gal,
            "f_realised": n_agn / (n_agn + n_gal)}


def block(b, truth):
    """median / CI / half-widths / offset for one marginal block."""
    if b is None:
        return None
    lo68, hi68 = b["ci68"]
    lo90, hi90 = b["ci90"]
    med = b["median"]
    out = {
        "median": med, "map": b.get("map"),
        "ci68": [lo68, hi68], "ci90": [lo90, hi90],
        "minus68": med - lo68, "plus68": hi68 - med,
        "halfwidth68": 0.5 * (hi68 - lo68),
        "width68": hi68 - lo68, "width90": hi90 - lo90,
    }
    if truth is not None:
        out["truth"] = truth
        out["offset"] = med - truth
        out["truth_in_ci68"] = bool(lo68 <= truth <= hi68)
        out["truth_in_ci90"] = bool(lo90 <= truth <= hi90)
        out["pull"] = ((med - truth) / out["halfwidth68"]
                       if out["halfwidth68"] > 0 else float("nan"))
    return out


def mean_sem(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"n": 0}
    m = float(v.mean())
    sd = float(v.std(ddof=1)) if v.size > 1 else float("nan")
    sem = sd / math.sqrt(v.size) if v.size > 1 else float("nan")
    t = m / sem if sem and np.isfinite(sem) and sem > 0 else float("nan")
    return {"n": int(v.size), "mean": m, "sd": sd, "sem": sem, "t": t}


def fmt_ci(med, minus, plus, nd=1):
    return f"{med:.{nd}f}^{{+{plus:.{nd}f}}}_{{-{minus:.{nd}f}}}"


def main(argv=None):
    a = parse_args(argv)
    R = Path(a.resdir)
    rows = []
    for s in a.seeds:
        rf = realised_f(a.data_root, s)
        j = load(R / f"joint_s{s}.json")
        fs = load(R / f"fscan_s{s}.json")
        h0s = load(R / f"h0scan_s{s}.json")
        row = {"seed": s, **rf}
        if j:
            row["joint"] = {
                "H0": block(j.get("H0"), H0_TRUTH),
                "f_vs_realised": block(j.get("f"), rf["f_realised"]),
                "f_vs_planted": block(j.get("f"), F_PLANTED),
                "rho": j.get("rho"),
                "map": j.get("map"),
                "moments": j.get("moments"),
                "n_rejected": j.get("n_rejected"),
                "guard": {k: v for k, v in (j.get("guard", {}).get("summary") or {}).items()},
                "neff_vs_f_at_truth_H0": j.get("guard", {}).get("neff_vs_f_at_truth_H0"),
            }
        if fs:
            row["fscan"] = {"vs_realised": block(fs.get("f"), rf["f_realised"]),
                            "vs_planted": block(fs.get("f"), F_PLANTED),
                            "h0_fixed": fs.get("h0_fixed"),
                            "n_rejected": fs.get("n_rejected"),
                            "guard": (fs.get("guard", {}).get("summary") or {})}
        if h0s:
            row["h0scan"] = {"H0": block(h0s.get("H0"), H0_TRUTH),
                             "f_fixed": h0s.get("f_fixed"),
                             "n_rejected": h0s.get("n_rejected"),
                             "guard": (h0s.get("guard", {}).get("summary") or {})}
        rows.append(row)

    have = [r for r in rows if "joint" in r]
    closure = {}
    if have:
        closure["joint_H0"] = mean_sem([r["joint"]["H0"]["offset"] for r in have])
        closure["joint_f_vs_realised"] = mean_sem(
            [r["joint"]["f_vs_realised"]["offset"] for r in have])
        closure["joint_f_vs_planted"] = mean_sem(
            [r["joint"]["f_vs_planted"]["offset"] for r in have])
        closure["rho"] = mean_sem([r["joint"]["rho"] for r in have])
        # scatter of the medians against the quoted widths
        for key, blk in (("H0", "H0"), ("f", "f_vs_realised")):
            meds = np.array([r["joint"][blk]["median"] for r in have])
            hw = np.array([r["joint"][blk]["halfwidth68"] for r in have])
            closure[f"scatter_{key}"] = {
                "sd_of_medians": float(meds.std(ddof=1)) if meds.size > 1 else float("nan"),
                "mean_halfwidth68": float(hw.mean()),
                "ratio": (float(meds.std(ddof=1) / hw.mean())
                          if meds.size > 1 and hw.mean() > 0 else float("nan")),
            }
        closure["coverage"] = {
            "H0_in_68": sum(r["joint"]["H0"]["truth_in_ci68"] for r in have),
            "H0_in_90": sum(r["joint"]["H0"]["truth_in_ci90"] for r in have),
            "f_realised_in_68": sum(r["joint"]["f_vs_realised"]["truth_in_ci68"] for r in have),
            "f_realised_in_90": sum(r["joint"]["f_vs_realised"]["truth_in_ci90"] for r in have),
            "f_planted_in_68": sum(r["joint"]["f_vs_planted"]["truth_in_ci68"] for r in have),
            "f_planted_in_90": sum(r["joint"]["f_vs_planted"]["truth_in_ci90"] for r in have),
            "n": len(have),
        }
    havef = [r for r in rows if "fscan" in r]
    if havef:
        closure["fscan_f_vs_realised"] = mean_sem(
            [r["fscan"]["vs_realised"]["offset"] for r in havef])
        closure["fscan_f_vs_planted"] = mean_sem(
            [r["fscan"]["vs_planted"]["offset"] for r in havef])
    haveh = [r for r in rows if "h0scan" in r]
    if haveh:
        closure["h0scan_H0"] = mean_sem([r["h0scan"]["H0"]["offset"] for r in haveh])

    # ---- lane agreement (targeted = record, popuni = cross-check) --------------
    lanes = {}
    s = a.lane_seed
    for tag, key in (("joint", "joint"), ("fscan", "fscan"), ("h0scan", "h0scan")):
        rec = load(R / f"{tag}_s{s}.json")
        xch = load(R / f"{tag}_s{s}_popuni.json")
        if not (rec and xch):
            continue
        d = {}
        if tag == "joint":
            for p in ("H0", "f"):
                d[p] = {"targeted": rec[p]["median"], "popuni": xch[p]["median"],
                        "delta": xch[p]["median"] - rec[p]["median"],
                        "delta_over_halfwidth68":
                            (xch[p]["median"] - rec[p]["median"])
                            / (0.5 * (rec[p]["ci68"][1] - rec[p]["ci68"][0]))}
            d["rho"] = {"targeted": rec.get("rho"), "popuni": xch.get("rho")}
        else:
            p = "f" if tag == "fscan" else "H0"
            d[p] = {"targeted": rec[p]["median"], "popuni": xch[p]["median"],
                    "delta": xch[p]["median"] - rec[p]["median"],
                    "delta_over_halfwidth68":
                        (xch[p]["median"] - rec[p]["median"])
                        / (0.5 * (rec[p]["ci68"][1] - rec[p]["ci68"][0]))}
        d["Neff_min"] = {"targeted": (rec.get("guard", {}).get("summary") or {}).get("Neff_min"),
                         "popuni": (xch.get("guard", {}).get("summary") or {}).get("Neff_min")}
        lanes[tag] = d

    # ---- the sky-shuffle null --------------------------------------------------
    null = None
    nrec = load(R / f"fscan_null_s{s}.json")
    frec = load(R / f"fscan_s{s}.json")
    if nrec:
        nb = nrec["f"]
        null = {"seed": s,
                "median": nb["median"],
                "ci68": nb["ci68"], "ci90": nb["ci90"],
                "halfwidth68": 0.5 * (nb["ci68"][1] - nb["ci68"][0]),
                "map": nb.get("map")}
        if frec:
            fb = frec["f"]
            null["record_median"] = fb["median"]
            null["record_halfwidth68"] = 0.5 * (fb["ci68"][1] - fb["ci68"][0])
            null["width_ratio_null_over_record"] = (
                null["halfwidth68"] / null["record_halfwidth68"]
                if null["record_halfwidth68"] > 0 else float("nan"))

    out = {
        "_what": "the joint (H0, f_AGN) measurement on the complete GAL + AGN "
                 "catalogs; dark_sirens K = 2 mixture at log10n0 = log10n0_c2 = -24 "
                 "(the complete-catalog limit), field sky weighting, survey order "
                 "[GAL, AGN] so fcat_2 = f_AGN; all population parameters fixed at "
                 "the mock fiducial, Om0 pinned; free parameters H0 and fcat_2 only",
        "truth": {"H0": H0_TRUTH, "f_planted": F_PLANTED,
                  "f_reference_note":
                      "closure is quoted against the REALISED per-seed host "
                      "fraction (the value the drawn events actually contain) AND "
                      "against the planted 0.30; the two differ by the mock's own "
                      "binomial draw, sd = sqrt(0.3*0.7/1000) = 0.0145 per "
                      "realisation"},
        "binomial_sd_per_realisation": math.sqrt(F_PLANTED * (1 - F_PLANTED) / 1000),
        "seeds": rows,
        "closure": closure,
        "lane_agreement": lanes,
        "sky_shuffle_null": null,
    }
    (R / "joint_summary.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {R / 'joint_summary.json'}")

    # ---- the paper hook --------------------------------------------------------
    if have:
        s100 = next((r for r in have if r["seed"] == 100), have[0])
        jH0, jf = s100["joint"]["H0"], s100["joint"]["f_vs_realised"]
        hook = {
            "_provenance":
                "built from results/joint_s<seed>.json (measurement, targeted "
                "injection lane) with results/*_popuni.json as the cross-check; "
                "estimator dark_sirens K = 2 at log10n0 = log10n0_c2 = -24, the "
                "complete-catalog limit, field sky weighting, survey order "
                "[GAL, AGN] so fcat_2 = f_AGN; H0 and fcat_2 free, everything else "
                "fixed.  Same conventions as analysis_1/results/h0_single_tracer.json.",
            "h0_ci": fmt_ci(jH0["median"], jH0["minus68"], jH0["plus68"], 1),
            "h0_median": jH0["median"],
            "h0_width": round(jH0["width68"], 2),
            "f_ci": fmt_ci(jf["median"], jf["minus68"], jf["plus68"], 3),
            "f_median": jf["median"],
            "f_width": round(jf["width68"], 3),
            "rho": s100["joint"]["rho"],
            "map": s100["joint"]["map"],
            "reference_seed": s100["seed"],
            "h0_crosscheck_median":
                lanes.get("joint", {}).get("H0", {}).get("popuni"),
            "f_crosscheck_median":
                lanes.get("joint", {}).get("f", {}).get("popuni"),
            "truth_h0": H0_TRUTH,
            "truth_f_planted": F_PLANTED,
            "truth_f_realised": s100["f_realised"],
            "closure_h0_offset_mean": closure["joint_H0"].get("mean"),
            "closure_h0_offset_sem": closure["joint_H0"].get("sem"),
            "closure_f_offset_vs_realised_mean": closure["joint_f_vs_realised"].get("mean"),
            "closure_f_offset_vs_realised_sem": closure["joint_f_vs_realised"].get("sem"),
            "closure_f_offset_vs_planted_mean": closure["joint_f_vs_planted"].get("mean"),
            "closure_f_offset_vs_planted_sem": closure["joint_f_vs_planted"].get("sem"),
            "closure_n_seeds": closure["coverage"]["n"],
            "rho_mean": closure["rho"].get("mean"),
        }
        (R / "h0_fagn_joint.json").write_text(json.dumps(hook, indent=2))
        print(f"Wrote {R / 'h0_fagn_joint.json'}")

    # ---- console table ---------------------------------------------------------
    print("\n=== five-realisation closure, joint (H0, f_AGN) ===")
    print(f"{'seed':>5} {'N_AGN':>6} {'f_real':>7} | {'H0 med':>7} {'+-68':>6} "
          f"{'off':>7} {'68/90':>6} | {'f med':>6} {'+-68':>6} {'off(real)':>9} "
          f"{'off(0.30)':>9} {'68/90':>6} | {'rho':>6}")
    for r in have:
        H, F = r["joint"]["H0"], r["joint"]["f_vs_realised"]
        Fp = r["joint"]["f_vs_planted"]
        print(f"{r['seed']:>5} {r['n_host_agn']:>6} {r['f_realised']:>7.3f} | "
              f"{H['median']:>7.2f} {H['halfwidth68']:>6.2f} {H['offset']:>+7.2f} "
              f"{int(H['truth_in_ci68'])}/{int(H['truth_in_ci90'])}    | "
              f"{F['median']:>6.3f} {F['halfwidth68']:>6.3f} {F['offset']:>+9.3f} "
              f"{Fp['offset']:>+9.3f} "
              f"{int(F['truth_in_ci68'])}/{int(F['truth_in_ci90'])}    | "
              f"{r['joint']['rho']:>+6.3f}")
    for k in ("joint_H0", "joint_f_vs_realised", "joint_f_vs_planted", "rho"):
        if k in closure:
            c = closure[k]
            print(f"{k:>24}: {c.get('mean', float('nan')):+.4f} +- "
                  f"{c.get('sem', float('nan')):.4f}  (t = {c.get('t', float('nan')):+.2f}, "
                  f"n = {c.get('n')})")
    for k in ("scatter_H0", "scatter_f"):
        if k in closure:
            c = closure[k]
            print(f"{k:>24}: sd(medians) = {c['sd_of_medians']:.4f}, "
                  f"mean half-width = {c['mean_halfwidth68']:.4f}, "
                  f"ratio = {c['ratio']:.2f}")
    if closure.get("coverage"):
        print(f"{'coverage':>24}: {closure['coverage']}")
    if lanes:
        print("\n=== lane agreement (seed %d) ===" % s)
        print(json.dumps(lanes, indent=2))
    if null:
        print("\n=== sky-shuffle null (fscan, seed %d) ===" % s)
        print(json.dumps(null, indent=2))


if __name__ == "__main__":
    main()
