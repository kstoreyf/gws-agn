#!/usr/bin/env python3
"""How much does f_AGN cost when the AGN density is not known a priori?

`../experiment_twotracer_incomplete` found sigma(f_AGN) almost independent of
completeness -- but only because n0 was pinned at the true best fit of the
density model.  That is the most favourable possible case: the completion
converts a *known* n0 into a missing-host budget, and the budget is what keeps
the two tracer priors distinguishable once their observed hosts are gone.  If
n0 is uncertain, a missing AGN host and an AGN-hosted event explain the same
observation, and the two parameters trade against each other.

Each rung's ``fn0`` grid is a 2-D likelihood L(f, g) with g = log10 n0_AGN, so
every level of prior knowledge about g is a REWEIGHTING of one grid, not a
separate run:

    p(f) propto int dg  L(f, g) pi(g)

with pi a Gaussian of width sigma_g dex about the truth (sigma_g = 0 recovers
the fixed-n0 slice exactly) or flat over the scanned range.  A fractional
uncertainty eps on n0 is sigma_g = log10(1 + eps) dex.

Reported per (rung, prior width): the marginal f_AGN posterior, its 68%
half-width, and the detection significance median/sigma -- the number that says
whether an AGN-hosted fraction was measured at all.  The n0 marginal's edge mass
is reported too: where the flat-prior arm rails against the scanned range, its
width is a statement about the range, not about the data, and is flagged.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
INC = BASE.parent / "experiment_twotracer_incomplete"

LEVELS = ["complete", "m21.0", "m20.0", "m19.0", "m18.0"]
LABELS = {"complete": "complete", "m21.0": "m<21", "m20.0": "m<20",
          "m19.0": "m<19", "m18.0": "m<18"}
G_TRUE = -7.720033
TRUTH_F = 0.30

# (label, fractional uncertainty on n0 -> dex).  None = flat over the grid.
ARMS = [("fixed", 0.0), ("5%", 0.05), ("10%", 0.10), ("30%", 0.30),
        ("factor 2", 1.0), ("free", None)]


def dex(frac):
    return None if frac is None else (0.0 if frac == 0 else float(np.log10(1.0 + frac)))


def ci_from(x, p):
    """Median, equal-tailed 68/90% CIs and moments of a 1-D density on x."""
    norm = np.trapz(p, x)
    if not np.isfinite(norm) or norm <= 0:
        return None
    p = p / norm
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
    cdf /= cdf[-1]
    q = lambda t: float(np.interp(t, cdf, x))  # noqa: E731
    mean = float(np.trapz(x * p, x))
    sd = float(np.sqrt(max(np.trapz((x - mean) ** 2 * p, x), 0.0)))
    return {"median": q(0.5), "ci68": [q(0.16), q(0.84)], "ci90": [q(0.05), q(0.95)],
            "half_width68": 0.5 * (q(0.84) - q(0.16)), "mean": mean, "sd": sd,
            "argmax": float(x[int(np.argmax(p))])}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="",
                    help="Read fn0_<lev><suffix>.h5 and write "
                         "n0_arms_summary<suffix>.json (e.g. _fix for the "
                         "post-PR#335 sigma_ang-fixed events).")
    sfx = ap.parse_args().suffix

    out = {"g_true": G_TRUE, "truth_f": TRUTH_F, "arms": [a for a, _ in ARMS],
           "levels": {}}

    for lev in LEVELS:
        h5 = RESULTS / f"fn0_{lev}{sfx}.h5"
        if not h5.exists():
            continue
        with h5py.File(h5, "r") as f:
            fv, gv, ll = f["f_grid"][:], f["n0c2_grid"][:], f["log_likelihood"][:]
        ok = np.isfinite(ll)
        L = np.where(ok, np.exp(ll - np.nanmax(ll[ok])), 0.0)

        rec = {"label": LABELS[lev], "n_rejected": int((~ok).sum()),
               "n_evals": int(ok.size), "arms": {}}

        # n0 marginal under a flat prior, and whether it rails on the range
        pg = np.trapz(L, fv, axis=0)
        gblock = ci_from(gv, pg)
        if gblock:
            pgn = pg / np.trapz(pg, gv)
            span = gv[-1] - gv[0]
            lo_mask, hi_mask = gv <= gv[0] + 0.05 * span, gv >= gv[-1] - 0.05 * span
            gblock["edge_mass_low"] = float(np.trapz(pgn[lo_mask], gv[lo_mask]))
            gblock["edge_mass_high"] = float(np.trapz(pgn[hi_mask], gv[hi_mask]))
            gblock["grid"] = [float(gv[0]), float(gv[-1])]
            gblock["offset_from_truth"] = gblock["median"] - G_TRUE
        rec["log10n0_agn_flat_prior"] = gblock

        # correlation of the flat-prior 2-D posterior
        Z = L / np.trapz(np.trapz(L, gv, axis=1), fv)
        pf_ = np.trapz(Z, gv, axis=1)
        pg_ = np.trapz(Z, fv, axis=0)
        Ef, Eg = np.trapz(fv * pf_, fv), np.trapz(gv * pg_, gv)
        Vf = np.trapz((fv - Ef) ** 2 * pf_, fv)
        Vg = np.trapz((gv - Eg) ** 2 * pg_, gv)
        Fg, Gg = np.meshgrid(fv, gv, indexing="ij")
        cov = np.trapz(np.trapz((Fg - Ef) * (Gg - Eg) * Z, gv, axis=1), fv)
        rec["rho_f_n0_flat_prior"] = (float(cov / np.sqrt(Vf * Vg))
                                      if Vf > 0 and Vg > 0 else float("nan"))

        for name, frac in ARMS:
            sg = dex(frac)
            if sg is None:
                pf = np.trapz(L, gv, axis=1)                      # flat prior
            elif sg == 0.0:
                pf = L[:, int(np.argmin(np.abs(gv - G_TRUE)))]    # exact slice
            else:
                w = np.exp(-0.5 * ((gv - G_TRUE) / sg) ** 2)
                pf = np.trapz(L * w[None, :], gv, axis=1)
            b = ci_from(fv, pf)
            if b is None:
                rec["arms"][name] = None
                continue
            b["prior_sigma_dex"] = sg
            b["truth_in_ci68"] = bool(b["ci68"][0] <= TRUTH_F <= b["ci68"][1])
            b["truth_in_ci90"] = bool(b["ci90"][0] <= TRUTH_F <= b["ci90"][1])
            # detection significance of a nonzero AGN-hosted fraction
            b["detection_sigma"] = (b["median"] / b["sd"]) if b["sd"] > 0 else None
            rec["arms"][name] = b
        out["levels"][lev] = rec

    # degradation of each arm relative to the fixed-n0 complete rung
    ref = out["levels"].get("complete", {}).get("arms", {}).get("fixed")
    if ref:
        for lev, rec in out["levels"].items():
            for name, b in rec["arms"].items():
                if b:
                    b["half_width_vs_fixed_complete"] = (
                        b["half_width68"] / ref["half_width68"])

    (RESULTS / f"n0_arms_summary{sfx}.json").write_text(json.dumps(out, indent=2,
                                                        default=float))
    print(f"wrote results/n0_arms_summary{sfx}.json\n")

    names = [a for a, _ in ARMS]
    print("sigma(f_AGN)  [68% half-width]")
    print(f"{'level':>9} " + "".join(f"{n:>10}" for n in names))
    for lev in LEVELS:
        if lev not in out["levels"]:
            continue
        r = out["levels"][lev]["arms"]
        print(f"{LABELS[lev]:>9} " + "".join(
            f"{r[n]['half_width68']:10.4f}" if r.get(n) else f"{'--':>10}"
            for n in names))
    print("\nf_AGN median   (planted 0.300)")
    print(f"{'level':>9} " + "".join(f"{n:>10}" for n in names))
    for lev in LEVELS:
        if lev not in out["levels"]:
            continue
        r = out["levels"][lev]["arms"]
        print(f"{LABELS[lev]:>9} " + "".join(
            f"{r[n]['median']:10.4f}" if r.get(n) else f"{'--':>10}" for n in names))
    print("\ndetection significance  median/sigma")
    print(f"{'level':>9} " + "".join(f"{n:>10}" for n in names))
    for lev in LEVELS:
        if lev not in out["levels"]:
            continue
        r = out["levels"][lev]["arms"]
        print(f"{LABELS[lev]:>9} " + "".join(
            f"{r[n]['detection_sigma']:10.1f}" if r.get(n) else f"{'--':>10}"
            for n in names))
    print("\nlog10 n0_AGN under a flat prior (truth -7.7200)")
    for lev in LEVELS:
        if lev not in out["levels"]:
            continue
        g = out["levels"][lev]["log10n0_agn_flat_prior"]
        print(f"{LABELS[lev]:>9}  median {g['median']:+.3f} "
              f"({g['offset_from_truth']:+.3f})  hw {g['half_width68']:.3f}  "
              f"edge mass lo/hi {g['edge_mass_low']:.3f}/{g['edge_mass_high']:.3f}  "
              f"rho(f,n0) {out['levels'][lev]['rho_f_n0_flat_prior']:+.3f}")


if __name__ == "__main__":
    main()
