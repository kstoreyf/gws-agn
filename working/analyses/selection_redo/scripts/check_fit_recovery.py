#!/usr/bin/env python
"""Acceptance test for the selection fits: do they recover the mock's own LF?

The mock's magnitudes ARE a Schechter truncated at ``x_cut``, so with the
matching ``--m_faint_offset`` / ``--m_faint_cut`` the fitted model is the
generative process exactly.  The fit must therefore return the mock's
``Mstar_hat`` and ``alpha``, and at n = 821k the galaxy error bars are ~1e-3
mag -- so a pull beyond 3 sigma is a systematic (h-scaling of the cut, the
z >= 0.01 floor drops, K-correction leakage), not sampling noise.

Run this BEFORE spending GPU hours on the K=2 inference: a fit that misses
truth here would anchor the mixture's theta priors in the wrong place.
"""
import argparse
import json
from pathlib import Path

from lf_constants import constants

PULL_FAIL = 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", required=True, help="directory of fit JSONs")
    ap.add_argument("--rung", default="m18")
    ap.add_argument("--family", default="schechter")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--tracers", default="gal,agn")
    a = ap.parse_args()

    c = constants(a.seed)
    truth = {"Mstar_hat": c["Mstar_hat_truth"], "alpha": c["alpha_truth"]}

    print(f"\n{'=' * 74}\nLF RECOVERY vs the mock's generative truth"
          f"   (Mstar_hat {truth['Mstar_hat']:.4f}, alpha {truth['alpha']:.4f})"
          f"\n{'=' * 74}")
    print(f"{'tracer':<8}{'param':<12}{'fitted':>12}{'sigma':>10}"
          f"{'truth':>12}{'pull':>9}")

    worst, rows = 0.0, []
    for t in a.tracers.split(","):
        p = Path(a.fits) / f"fit_{t}_{a.rung}_{a.family}.json"
        if not p.exists():
            raise SystemExit(f"missing fit: {p}")
        rec = json.loads(p.read_text())
        # single-stratum fits store one entry; tolerate either layout
        s = rec["strata"][0] if "strata" in rec else rec
        # selection-fit-1.0 carries the 2x2 (Mstar_hat, alpha) covariance
        cov = s.get("cov")
        sds = {"Mstar_hat": cov[0][0] ** 0.5, "alpha": cov[1][1] ** 0.5} if cov \
            else {}
        for k, tv in truth.items():
            val = float(s[k])
            sd = float(sds.get(k, float("nan")))
            pull = (val - tv) / sd if sd == sd and sd > 0 else float("nan")
            worst = max(worst, abs(pull) if pull == pull else 0.0)
            rows.append((t, k, val, sd, tv, pull))
            print(f"{t:<8}{k:<12}{val:>12.5f}{sd:>10.5f}{tv:>12.5f}"
                  f"{pull:>+9.2f}")

    ok = worst <= PULL_FAIL
    print(f"\n  worst |pull| = {worst:.2f}   "
          f"{'PASS' if ok else 'FAIL'} (threshold {PULL_FAIL})")
    if not ok:
        print("  At this N the bars are ~1e-3 mag, so this is a systematic, "
              "not noise:\n    check the h-scaling of --m_faint_cut, the "
              "z >= 0.01 floor drops, K-correction leakage.")

    out = Path(a.fits) / f"recovery_{a.rung}_{a.family}.json"
    out.write_text(json.dumps(
        {"truth": truth, "threshold": PULL_FAIL, "worst_abs_pull": worst,
         "pass": bool(ok),
         "rows": [{"tracer": r[0], "param": r[1], "fitted": r[2], "sigma": r[3],
                   "truth": r[4], "pull": r[5]} for r in rows]}, indent=2))
    print(f"  wrote {out}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
