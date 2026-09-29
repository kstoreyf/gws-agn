#!/usr/bin/env python
"""Analysis 11 -- the one mechanism diagnostic for H0 (brief section 21).

At a representative 11D posterior point, the likelihood as a function of H0 alone:

    full     (f, mu_G, dmu_G, mu_chi, dmu_chi) at the 11D posterior medians;
    pinned   the same offsets and f, with the references pinned at (mu_G, mu_chi)
             = (35, 0), Analysis 10's values.

The difference between the two isolates what the free reference zero point does to
H0 at fixed environmental offsets. Each node records logL and its PE and selection
parts. H0 nodes: 60 to 76 in steps of 0.25, plus 67.74 and the 11D median.

    sbatch --job-name=a11_s21 --time=01:00:00 \\
        --export=ALL,A11_SCRIPT=scripts/a11_h0_profile.py scripts/submit_a11_gpu.sbatch
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
A11 = HERE.parent
sys.path.insert(0, str(HERE))

import a11_likelihood as L                                  # noqa: E402

OUT = A11 / "diagnostics" / "a11_h0_profile.json"


def summarise(h, logL):
    """Normalised exp(logL) on the H0 nodes: median, 68/90% ends, peak, curvature sd."""
    from scipy.interpolate import CubicSpline
    xf = np.linspace(h[0], h[-1], 20001)
    lf = CubicSpline(h, logL)(xf)
    p = np.exp(lf - lf.max())
    c = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(xf))])
    c /= c[-1]
    q = lambda t: float(np.interp(t, c, xf))
    i = int(np.argmax(lf))
    k = (np.abs(h - xf[i]) <= 2.5)
    a2 = np.polyfit(h[k] - xf[i], logL[k], 2)[0]
    return {"median": q(0.5), "ci68": [q(0.16), q(0.84)], "ci90": [q(0.05), q(0.95)],
            "peak": float(xf[i]), "logL_peak": float(lf[i]),
            "curvature_sd": float(np.sqrt(-0.5 / a2)) if a2 < 0 else None}


def main():
    t0 = time.time()
    d = json.loads((A11 / "results" / "a11_11D.json").read_text())["summary"]
    med = {k: d[k]["median"] for k in ("H0", "f_agn", "mu_G", "dmu_G", "mu_chi", "dmu_chi")}
    points = {
        "full": {k: med[k] for k in ("mu_G", "dmu_G", "mu_chi", "dmu_chi")},
        "pinned": {"mu_G": 35.0, "dmu_G": med["dmu_G"], "mu_chi": 0.0,
                   "dmu_chi": med["dmu_chi"]},
    }
    H = np.unique(np.round(np.concatenate([np.arange(60.0, 76.0001, 0.25),
                                           [L.H0_FID, med["H0"]]]), 10))
    cell = L.build_a11("A11_S21")
    out = {"what": __doc__.split("    sbatch")[0].strip(), "f_agn": med["f_agn"],
           "points": points, "H0_nodes": H.tolist(), "D11_H0_median": med["H0"],
           "provenance": L.A10.a8.provenance(gw_path=L.GW_PATH_A11,
                                             survey_paths=[L.SURVEY_GAL, L.SURVEY_AGN]),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "curves": {}}
    for name, pt in points.items():
        rows = []
        for h in H:
            r = cell.evaluate_at(H0=float(h), fcat_2=med["f_agn"], **pt)
            rows.append({k: r.get(k) for k in ("logL", "logL_pe", "logL_selection", "Neff",
                                               "threshold", "finite", "seconds")})
            print(f"[{name}] H0 {h:7.3f} logL {r['logL']:.6f}", flush=True)
        logL = np.array([x["logL"] for x in rows], float)
        ok = np.isfinite(logL)
        cur = {"rows": rows, "n_nonfinite": int((~ok).sum()),
               "summary": summarise(H[ok], logL[ok])}
        for part in ("logL_pe", "logL_selection"):
            v = np.array([x[part] for x in rows], float)
            # d(part)/dH0 at the 11D median, by central difference on the 0.25 grid
            j = int(np.argmin(np.abs(H - med["H0"])))
            cur[f"d_{part}_dH0_at_11D_median"] = float((v[j + 1] - v[j - 1]) / (H[j + 1] - H[j - 1]))
        out["curves"][name] = cur
    f, p = out["curves"]["full"]["summary"], out["curves"]["pinned"]["summary"]
    out["pinned_minus_full"] = {"median": p["median"] - f["median"], "peak": p["peak"] - f["peak"],
                                "W68_ratio": (p["ci68"][1] - p["ci68"][0]) / (f["ci68"][1] - f["ci68"][0]),
                                "W90_ratio": (p["ci90"][1] - p["ci90"][0]) / (f["ci90"][1] - f["ci90"][0])}
    out["wall_seconds"] = time.time() - t0
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: out["curves"][k]["summary"] for k in out["curves"]}, indent=1))
    print(json.dumps(out["pinned_minus_full"], indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
