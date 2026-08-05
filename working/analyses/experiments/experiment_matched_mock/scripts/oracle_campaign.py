#!/usr/bin/env python3
"""Campaign-level paired comparison: archived darksirens peaks vs exact-oracle
peaks for every realisation with an oracle_num/oracle_mu pair.

darksirens side: results/obsdet_obs_<tag>.h5 (compiled scans, all 20).
oracle side:     ln L = sum_i ln O1_i(H0) - N ln mu_exact(H0).
Also reports the oracle ladder terms per realisation (slope@truth / curvature).
Writes results/oracle_campaign.json.
"""
import json
from pathlib import Path

import h5py
import numpy as np

EXP = Path(__file__).resolve().parents[1]
H0_TRUE = 67.74

TAGS = (["b"] + [f"s{k}" for k in (4102, 4103, 4104, 4105)]
        + [f"n{k}" for k in range(4201, 4216)])


def quad_peak(x, y):
    i = int(np.argmax(y))
    if i in (0, len(y) - 1):
        return float(x[i])
    d = y[i - 1] - 2 * y[i] + y[i + 1]
    return float(x[i] - 0.5 * (y[i + 1] - y[i - 1]) / d * (x[1] - x[0]))


def post_median(x, y):
    """Median of the flat-prior posterior exp(y) (summarize_grid convention)."""
    p = np.exp(y - np.max(y))
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
    cdf /= cdf[-1]
    return float(np.interp(0.5, cdf, x))


def main():
    rows = []
    for tag in TAGS:
        fn = EXP / "results" / f"oracle_num_{tag}.npz"
        fm = EXP / "results" / f"oracle_mu_{tag}.npz"
        fd = EXP / "results" / f"obsdet_obs_{tag}.h5"
        if not (fn.exists() and fm.exists() and fd.exists()):
            continue
        num = np.load(fn); mu = np.load(fm)
        with h5py.File(fd) as f:
            ds_ll = np.asarray(f["log_likelihood"]); H0 = np.asarray(f["H0_grid"])
        assert np.allclose(H0, num["H0"])
        N = num["ln_O1"].shape[0]
        i0 = int(np.argmin(np.abs(H0 - H0_TRUE))); h = H0[1] - H0[0]
        or_tot = num["ln_O1"].sum(0) - N * mu["ln_mu"]
        row = {"tag": tag,
               "peak_ds": quad_peak(H0, ds_ll),
               "peak_oracle": quad_peak(H0, or_tot),
               "med_ds": post_median(H0, ds_ll),
               "med_oracle": post_median(H0, or_tot)}
        row["offset_ds"] = row["med_ds"] - H0_TRUE
        row["offset_oracle"] = row["med_oracle"] - H0_TRUE
        row["paired"] = row["med_ds"] - row["med_oracle"]
        row["paired_peak"] = row["peak_ds"] - row["peak_oracle"]
        for a, b in [("O2", "O1"), ("O3", "O2"), ("O3b", "O3"), ("O4", "O3b")]:
            dd = (num[f"ln_{a}"] - num[f"ln_{b}"]).sum(0)
            row[f"slope_{a}_minus_{b}"] = float((dd[i0 + 2] - dd[i0 - 2]) / (4 * h))
        rows.append(row)
        print(f"{tag:6s} ds {row['offset_ds']:+.3f}  oracle {row['offset_oracle']:+.3f}"
              f"  paired {row['paired']:+.3f}")

    if rows:
        for key in ("offset_ds", "offset_oracle", "paired"):
            v = np.array([r[key] for r in rows])
            print(f"{key}: mean {v.mean():+.3f} +- {v.std(ddof=1)/np.sqrt(len(v)):.3f} "
                  f"(sd {v.std(ddof=1):.3f}, n={len(v)})")
        for key in ("slope_O2_minus_O1", "slope_O3_minus_O2",
                    "slope_O3b_minus_O3", "slope_O4_minus_O3b"):
            v = np.array([r[key] for r in rows])
            print(f"{key}: mean {v.mean():+.3f} +- {v.std(ddof=1)/np.sqrt(len(v)):.3f} nats/km")
    (EXP / "results" / "oracle_campaign.json").write_text(json.dumps(rows, indent=2))
    print("wrote results/oracle_campaign.json")


if __name__ == "__main__":
    main()
