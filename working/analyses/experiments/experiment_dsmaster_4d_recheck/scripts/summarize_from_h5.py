#!/usr/bin/env python
"""Rebuild an arm's summary JSON from its posterior h5.

The selection arm's 4 h 32 m run converged and wrote its h5, then died in the
JSON writer on a provenance bug of mine (`opts` is local to build_closure and
the writer lives in main).  Resuming is not an option -- dynesty declines to
re-enter a finished static run -- but nothing needs re-running: the h5 carries
the equal-weight samples, the raw sampler products and every meta_* attribute
the JSON reports.

This reproduces the driver's output contract from that h5, reusing
sample_4d.summarize so the numbers are computed by the same code that would
have written them, not a re-implementation that could drift.

    python summarize_from_h5.py --h5 results/fit_m18_selection_s100.h5
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sample_4d import summarize  # noqa: E402  (same code path as the driver)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out", default=None, help="default: <h5>.json")
    a = ap.parse_args()

    h5 = Path(a.h5)
    out = Path(a.out) if a.out else h5.with_suffix(".json")

    with h5py.File(h5, "r") as f:
        a_ = dict(f.attrs)
        samples = np.asarray(f["samples"])
        sampled = [s.decode() if isinstance(s, bytes) else str(s)
                   for s in np.asarray(f["sampled_labels"])]

    # summarize() indexes truths POSITIONALLY, in its own
    # (H0, log10n0, log10n0_c2, f_AGN) column order -- same order as `samples`.
    truths = [float(a_["arg_h0_true"]), float(a_["arg_n0_true"]),
              float(a_["arg_n0c2_true"]), float(a_["arg_f_true"])]
    summary = summarize(samples, truths)

    doc = {
        "_what": ("Summary rebuilt from the posterior h5 after the driver's "
                  "JSON writer raised on a provenance bug. The sampling itself "
                  "completed and converged; nothing here is re-derived from a "
                  "shorter or restarted run."),
        "_rebuilt_from": str(h5),
        "out_tag": str(a_["arg_out_tag"]),
        "sampled_labels": sampled,
        "sampler_meta": {
            "sampler": str(a_["meta_sampler"]),
            "nlive": int(a_["meta_nlive"]),
            "dlogz_target": float(a_["meta_dlogz_target"]),
            "maxcall": int(a_["meta_maxcall"]),
            "rstate_seed": int(a_["meta_rstate_seed"]),
            "ncall_total": int(a_["meta_ncall_total"]),
            "niter": int(a_["meta_niter"]),
            "logz": float(a_["meta_logz"]),
            "logzerr": float(a_["meta_logzerr"]),
            "dlogz_reached": float(a_["meta_dlogz_reached"]),
            "eff_percent": float(a_["meta_eff_percent"]),
            "stopped_by_maxcall": bool(a_["meta_stopped_by_maxcall"]),
        },
        "summary": summary,
        "priors": {"H0": json.loads(str(a_["arg_h0_prior"])),
                   "log10n0": json.loads(str(a_["arg_n0_prior"])),
                   "log10n0_c2": json.loads(str(a_["arg_n0c2_prior"])),
                   "fcat_2": json.loads(str(a_["arg_f_prior"]))},
        "truths": {"H0": truths[0], "log10n0": truths[1],
                   "log10n0_c2": truths[2], "f_realised": truths[3]},
        "darksirens_git_sha": str(a_["darksirens_git_sha"]),
        "c_mode": str(a_["c_mode"]),
    }

    if str(a_["c_mode"]) == "selection":
        paths = [p for p in str(a_["arg_selection_fit"]).split(",") if p]
        labels = json.loads(str(a_["labels"]))
        base = np.asarray(a_["base_coord"], dtype=float)
        pinned = {lbl: float(base[labels.index(lbl)])
                  for lbl in ("Mstar_hat", "alpha", "Mstar_hat_c2", "alpha_c2")
                  if lbl in labels}
        doc["selection"] = {
            "family": "schechter",
            "fit_paths": paths,
            "fit_records": [json.loads(Path(p).read_text()) for p in paths
                            if Path(p).exists()],
            "theta_treatment": (
                "PINNED at the anchored prior centres (the offline fits), not "
                "sampled -- so all three arms sample the same four parameters "
                "and the difference is the estimator, not a wider "
                "marginalisation. Same treatment delta/sigma_kde get in every "
                "arm."),
            "theta_pinned_at": pinned,
        }

    out.write_text(json.dumps(doc, indent=1))
    print(f"wrote {out}")
    for k in ("H0", "log10n0", "log10n0_c2", "f_AGN"):
        v = summary[k]
        print(f"  {k:<12} {v['median']:>10.4f} +/- {v['sd']:.4f}   "
              f"truth {v['truth']:>8.3f}   offset {v['offset']:+.4f}   "
              f"pull {v['pull']:+.2f}")


if __name__ == "__main__":
    main()
