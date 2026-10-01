#!/usr/bin/env python3
"""Choose each rung's proposal weights so the ladder is comparable where it counts.

A fixed set of mixture WEIGHTS is not a fixed allocation of statistical
resource.  The targeted branches place injections at catalog hosts, and a flux
limit leaves only bright, nearby hosts -- which are almost all detected.
Measured on this ladder (4M-draw pilots, weights 0.55/0.10/0.15/0.20):

    level      population   uniform    GAL-targeted   AGN-targeted
    complete    1.90e-3     1.18e-2      1.84e-3        2.27e-3
    m < 18      1.95e-3     1.19e-2      3.85e-1        4.55e-1

so the same 20% AGN-targeted weight yields 0.6% of detected rows at the complete
rung and 60% at m < 18.  Holding the weights fixed would therefore (a) produce
~18M detected rows at the faint end, most of them carrying negligible importance
weight because their proposal density is enormous, and (b) starve the population
branch, which is the ONLY branch covering the out-of-catalog part of the target
-- precisely the part that grows as the survey empties.

So the invariant held fixed across the ladder is the DETECTED-ROW SPLIT, not the
weights: the targeted branches supply a set fraction of detected injections
(default 10% GAL + 15% AGN), and the total detected count is held at a target.
The proposal is a nuisance -- ``pdraw`` is exact for whatever mixture is drawn,
so the estimator is unbiased for any choice -- and equalising where the samples
LAND is what makes N_eff comparable between rungs.

Given per-branch detection efficiencies e_p, e_u, e_g, e_a and a fixed uniform
weight w_u, requiring w_g e_g = f_g D and w_a e_a = f_a D with
D = sum_b w_b e_b gives, with K = f_g/e_g + f_a/e_a,

    D   = ((1 - w_u) e_p + w_u e_u) / (1 - f_g - f_a + K e_p)
    w_g = f_g D / e_g,   w_a = f_a D / e_a,   w_p = 1 - w_u - w_g - w_a

and ``ndraw = target_detected / D``.  At the complete rung this reproduces
0.545/0.10/0.160/0.195 -- i.e. essentially the hand-picked weights the deep
two-tracer experiment used -- so the scheme is a generalisation of that choice,
not a departure from it.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

EXP_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "bti", Path(__file__).resolve().parent / "build_targeted_injections_k2.py")
bti = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bti)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--levels", nargs="+",
                   default=["complete", "m21.0", "m20.0", "m19.0", "m18.0"])
    p.add_argument("--survey_fmt", default="data_derived/survey_{tracer}_{level}_ns32.h5")
    p.add_argument("--pilot_ndraw", type=int, default=4_000_000)
    p.add_argument("--target_detected", type=int, default=350_000)
    p.add_argument("--mix_uniform", type=float, default=0.10)
    p.add_argument("--frac_detected_gal", type=float, default=0.10)
    p.add_argument("--frac_detected_agn", type=float, default=0.15)
    p.add_argument("--pilot_weights", nargs=4, type=float,
                   default=[0.55, 0.10, 0.15, 0.20],
                   help="pop unif gal agn, used ONLY to measure efficiencies.")
    p.add_argument("--seed", type=int, default=74001)
    p.add_argument("--worktree", default=bti.DEFAULT_WORKTREE)
    p.add_argument("--out_json", default="results/mixture_calibration.json")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    sys.path.insert(0, str(Path(a.worktree) / "scripts/mock_dark_sirens"))
    import generate_mock_data as gmd  # noqa: E402

    cosmo = gmd._build_cosmology(bti.H0_FID, bti.OM0_FID, bti.W0_FID, bti.WA_FID)
    grids = gmd._cosmology_grids(cosmo, zmax=bti.ZMAX)
    ddldz = np.gradient(grids["dl"], grids["z"])
    pop = gmd.PopulationConfig()

    out = {"pilot_ndraw": a.pilot_ndraw, "target_detected": a.target_detected,
           "frac_detected": {"gal": a.frac_detected_gal, "agn": a.frac_detected_agn},
           "mix_uniform": a.mix_uniform, "levels": {}}
    wp0, wu0, wg0, wa0 = a.pilot_weights

    for lev in a.levels:
        pmaps = [bti.SurveyPixelMap(
                    EXP_ROOT / a.survey_fmt.format(tracer=t, level=lev), bti.ZMAX)
                 for t in ("gal", "agn")]
        sel = bti.draw_injections(
            gmd, a.pilot_ndraw, a.seed, grids, ddldz, pop, pmaps,
            wp0, wu0, [wg0, wa0], bti.SNR_THRESHOLD, a.pilot_ndraw, bti.ZMAX,
            verbose=False)
        nd = sel["n_detected_branch"]
        eff = [nd[0] / (wp0 * a.pilot_ndraw), nd[1] / (wu0 * a.pilot_ndraw),
               nd[2] / (wg0 * a.pilot_ndraw), nd[3] / (wa0 * a.pilot_ndraw)]
        e_p, e_u, e_g, e_a = eff
        f_g, f_a, w_u = a.frac_detected_gal, a.frac_detected_agn, a.mix_uniform
        if min(e_g, e_a, e_p) <= 0:
            raise SystemExit(f"{lev}: a branch detected nothing in the pilot")
        K = f_g / e_g + f_a / e_a
        D = ((1.0 - w_u) * e_p + w_u * e_u) / (1.0 - f_g - f_a + K * e_p)
        w_g, w_a = f_g * D / e_g, f_a * D / e_a
        w_p = 1.0 - w_u - w_g - w_a
        if w_p <= 0:
            raise SystemExit(f"{lev}: no weight left for the population branch")
        ndraw = int(round(a.target_detected / D))
        out["levels"][lev] = {
            "pilot_detected_by_branch": {"population": nd[0], "uniform": nd[1],
                                         "targeted_gal": nd[2], "targeted_agn": nd[3]},
            "detection_efficiency": {"population": e_p, "uniform": e_u,
                                     "targeted_gal": e_g, "targeted_agn": e_a},
            "weights": {"population": w_p, "uniform": w_u,
                        "targeted_gal": w_g, "targeted_agn": w_a},
            "predicted_detected_fraction_per_draw": D,
            "ndraw": ndraw,
            "predicted_detected": int(round(D * ndraw)),
        }
        print(f"{lev:>9}  eff(pop/unif/gal/agn) = "
              f"{e_p:.3e} {e_u:.3e} {e_g:.3e} {e_a:.3e}")
        print(f"{'':>9}  weights = {w_p:.6f} {w_u:.3f} {w_g:.6f} {w_a:.6f}   "
              f"ndraw = {ndraw:,}  (predicted {int(round(D*ndraw)):,} detected)",
              flush=True)

    p = EXP_ROOT / a.out_json
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
