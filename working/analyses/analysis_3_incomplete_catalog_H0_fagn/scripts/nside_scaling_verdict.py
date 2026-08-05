#!/usr/bin/env python3
"""Does the complete-catalog f_AGN shift scale like per-pixel Poisson noise?

THE CLAIM.  On a complete catalog the true-n0 completion still manufactures a
missing-AGN budget, because it is evaluated per HEALPix pixel and an nside-32 pixel
holds a mean of 7.0 AGN inside the horizon.  Wherever a pixel's count fluctuates low
the model reads C < 1 and invents AGN; where it fluctuates high the budget clips at
zero.  The error is one-sided, so it does not average away, and its size should
track the per-pixel fractional Poisson error 1/sqrt(N_per_pixel).

THE PREDICTION, fixed before the answer is looked at.  Going from nside 32 to
nside 16 quadruples the solid angle per pixel and therefore the hosts per pixel,
so 1/sqrt(N) halves.  The shift between the two configurations should fall by
about the same factor:

    shift(nside 16) / shift(nside 32)  ~  sqrt(N32 / N16)  =  sqrt(1/4)  =  0.5

THE CONTROL.  Both arms are re-run at nside 16 -- the true-n0 one and the
log10n0 = -24 one -- so the comparison is shift-against-shift at fixed
pixelisation, and the coarser sky's own effect on how well f_AGN is measured
cancels out of the ratio instead of contaminating it.

Reads the four f-scans and writes results/nside_scaling.json.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
A2 = ROOT.parent / "analysis_2_complete_catalog_H0_fagn" / "results"

# (label, true-n0 scan, log10n0=-24 scan)
ARMS = {
    "nside32": (RES / "fscan_complete_s100.json", A2 / "fscan_s100.json"),
    "nside16": (
        RES / "fscan_complete_ns16_truen0_s100.json",
        RES / "fscan_complete_ns16_n24_s100.json",
    ),
}


def blk(p: Path):
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    b = d["f"]
    lo, hi = b["ci68"]
    return {
        "path": str(p),
        "median": b["median"],
        "ci68": [lo, hi],
        "halfwidth68": 0.5 * (hi - lo),
        "n_rejected": d.get("n_rejected"),
        "kde_window": (d.get("guard", {}) or {}).get("kde_window"),
        "Neff_min": ((d.get("guard", {}) or {}).get("summary") or {}).get("Neff_min"),
    }


def main() -> None:
    out = {"_what": __doc__.split("\n")[0], "arms": {}}
    for name, (p_true, p_n24) in ARMS.items():
        a, b = blk(p_true), blk(p_n24)
        if not (a and b):
            print(f"[skip] {name}: missing scan(s)")
            continue
        shift = a["median"] - b["median"]
        out["arms"][name] = {
            "true_n0": a,
            "n0_minus24": b,
            "shift_f": shift,
            "shift_in_n24_halfwidths": shift / b["halfwidth68"],
            "halfwidth_ratio_true_over_n24": a["halfwidth68"] / b["halfwidth68"],
        }
        print(
            f"[{name}]  f(true n0) = {a['median']:.4f} +- {a['halfwidth68']:.4f}   "
            f"f(-24) = {b['median']:.4f} +- {b['halfwidth68']:.4f}\n"
            f"          shift = {shift:+.4f}  "
            f"({shift / b['halfwidth68']:+.3f} of the -24 half-width)   "
            f"width ratio {a['halfwidth68'] / b['halfwidth68']:.3f}"
        )

    # per-pixel occupancy, measured, for the predicted ratio
    diag = RES / "continuity_failure_diag.json"
    n32 = ramp_frac = None
    if diag.exists():
        d = json.loads(diag.read_text())
        s100 = (d.get("seeds") or {}).get("100") or {}
        occ_all = (s100.get("per_pixel_occupancy_in_horizon")
                   or d.get("per_pixel_occupancy_in_horizon_seed100") or {})
        occ = occ_all.get("agn")
        if occ:
            n32 = occ["mean_per_pixel"]
        agn = (s100.get("tracers") or {}).get("agn") or {}
        ramp_frac = agn.get("spurious_fraction_from_ramp")
    if n32:
        out["agn_hosts_per_pixel"] = {"nside32": n32, "nside16": 4.0 * n32}
        out["predicted_shift_ratio"] = 0.5
        if ramp_frac is not None:
            # Not all of the spurious budget is per-pixel shot noise: the GLASS
            # low-z shell ramp contributes a piece that is a property of the
            # redshift distribution, not of the pixelisation, so it does not
            # shrink when pixels are merged.  Splitting the budget that way gives
            # a shallower expectation than pure 1/sqrt(N).  This is an
            # AFTER-THE-FACT refinement, recorded beside the pre-registered 0.50,
            # not in place of it.
            out["refined_prediction_shift_ratio"] = float(
                ramp_frac + (1.0 - ramp_frac) * 0.5)
            out["refined_prediction_basis"] = (
                f"{100*ramp_frac:.0f} % of the AGN spurious budget comes from the "
                "GLASS low-z ramp, which is pixelisation-independent; the remaining "
                f"{100*(1-ramp_frac):.0f} % is per-pixel Poisson and halves")
        out["prediction_basis"] = (
            f"AGN per nside-32 pixel inside the horizon = {n32:.1f} "
            f"(1/sqrt(N) = {100 / n32 ** 0.5:.0f} %); at nside 16 it is "
            f"{4 * n32:.1f} (1/sqrt(N) = {100 / (4 * n32) ** 0.5:.0f} %), so the "
            "one-sided per-pixel budget -- and with it the shift -- should halve"
        )

    if "nside32" in out["arms"] and "nside16" in out["arms"]:
        r32 = out["arms"]["nside32"]["shift_f"]
        r16 = out["arms"]["nside16"]["shift_f"]
        ratio = r16 / r32 if r32 else float("nan")
        out["observed_shift_ratio"] = ratio
        # "roughly as 1/sqrt(N)" -> within a factor 1.5 of the predicted 0.5
        out["verdict"] = {
            "predicted": 0.5,
            "observed": ratio,
            "consistent_with_poisson_scaling": bool(0.33 <= ratio <= 0.75),
            "criterion": "observed shift ratio within [0.33, 0.75] of the "
            "predicted 0.50 (a factor 1.5 either way)",
        }
        print(
            f"\nshift(nside16)/shift(nside32) = {ratio:.3f}   "
            f"predicted {0.5:.2f} from 1/sqrt(N_per_pixel)\n"
            f"VERDICT: {'CONSISTENT' if out['verdict']['consistent_with_poisson_scaling'] else 'NOT CONSISTENT'}"
            " with the per-pixel Poisson mechanism"
        )

    (RES / "nside_scaling.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RES / 'nside_scaling.json'}")


if __name__ == "__main__":
    main()
