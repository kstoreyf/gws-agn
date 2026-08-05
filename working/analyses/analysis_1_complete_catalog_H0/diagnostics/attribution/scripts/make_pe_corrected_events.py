#!/usr/bin/env python3
"""ATTRIBUTION follow-up, TASK 2 -- write PE-mass-corrected events files.

The generator's mass measurement model is ``obs ~ N(m, f m)`` with ``f``
constant (``generate_dataset.py::observe``), but the stored PE mass samples are
a FIXED-width Gaussian about the observation whose width was computed from the
LATENT true mass (``posterior_samples``).  The exact flat-prior posterior of
that measurement model is

    p_ex(m | obs)  propto  (1/(f m)) exp[ -(obs - m)^2 / (2 f^2 m^2) ]

so the stored samples are reweighted from the fixed-width proposal to ``p_ex``:

    rho(m) = p_ex(m | obs) / N(m; obs, sig_stored)

This is EXACTLY the weight ``attr_mass_pe.py`` builds (its ``log_pex`` and
``log_ptilde`` are imported here, not re-derived).  It is H0-INDEPENDENT by
construction, so it cannot manufacture or hide an H0 slope.

Applying it inside darksirens without touching darksirens: the likelihood forms
the per-sample weight ``p_target/p_pe``, so writing

    p_pe_new = p_pe_old / rho

reproduces the reweighted evidence exactly.  ``load_gw_samples`` renormalises
``p_pe`` to sum 1 per event, so the estimator becomes

    Zhat' = [ SUM_k rho_k p_target_k / p_pe_k ] * [ SUM_j p_pe_j / rho_j ] / nsamp

which is the self-normalised reweighted estimator
``SUM_k rho_k p_target_k/p_pe_k / SUM_k rho_k`` times a per-event constant that
does NOT depend on H0.  The H0 posterior is therefore exactly the reweighted
one; only an irrelevant additive constant in log L differs.

The selection integral is untouched (injections carry TRUE parameters), so
``d ln mu/dH0`` is identical in every arm.

Writes: data_derived/events_<tracer>_hosted_pefix_<arm>.h5   (copies; the
originals and working/data are never modified).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from attr_mass_pe import log_pex, log_ptilde, SIG_M1_FRAC, SIG_M2_FRAC   # noqa: E402

LOG_FLOOR = -200.0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracers", default="gal,agn")
    ap.add_argument("--arms", default="m1m2,m1")
    ap.add_argument("--indir", default=str(ROOT / "data_derived"))
    ap.add_argument("--outdir", default=str(ROOT / "data_derived"))
    ap.add_argument("--report", default=str(ROOT / "results" / "pe_corrected_events.json"))
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = {"name": "make_pe_corrected_events",
              "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "weight": "rho = p_ex(m|obs) / N(m; obs, sig_stored), "
                        "p_ex propto (1/(f m)) exp[-(obs-m)^2/(2 f^2 m^2)]; "
                        "p_pe_new = p_pe_old / rho (H0-independent)",
              "files": {}}
    for tracer in [t.strip() for t in args.tracers.split(",")]:
        src = Path(args.indir) / f"events_{tracer}_hosted.h5"
        with h5py.File(src, "r") as f:
            nobs = int(f.attrs["nobs"]); nsamp = int(f.attrs["nsamp"])
            m1 = np.asarray(f["m1det"][:], float)
            m2 = np.asarray(f["m2det"][:], float)
            p_pe = np.asarray(f["p_pe"][:], float)
            obs_m1 = np.asarray(f["truth/obs_m1det"][:], float)
            obs_m2 = np.asarray(f["truth/obs_m2det"][:], float)
            sig_m1 = np.asarray(f["truth/obs_sig_m1"][:], float)
            sig_m2 = np.asarray(f["truth/obs_sig_m2"][:], float)
            t_m1 = np.asarray(f["truth/m1det"][:], float)
            t_m2 = np.asarray(f["truth/m2det"][:], float)
        # the named defect, verified on this file
        dev1 = float(np.max(np.abs(sig_m1 / (SIG_M1_FRAC * t_m1) - 1.0)))
        dev2 = float(np.max(np.abs(sig_m2 / (SIG_M2_FRAC * t_m2) - 1.0)))
        ev = np.repeat(np.arange(nobs), nsamp)
        lr1 = (log_pex(m1, obs_m1[ev], SIG_M1_FRAC)
               - log_ptilde(m1, obs_m1[ev], sig_m1[ev]))
        lr2 = (log_pex(m2, obs_m2[ev], SIG_M2_FRAC)
               - log_ptilde(m2, obs_m2[ev], sig_m2[ev]))
        for arm in [a.strip() for a in args.arms.split(",")]:
            lr = {"m1": lr1, "m1m2": lr1 + lr2}[arm].reshape(nobs, nsamp).copy()
            lr -= lr.max(axis=1, keepdims=True)
            lr = np.maximum(lr, LOG_FLOOR)
            rho = np.exp(lr)
            ess = 1.0 / ((rho / rho.sum(axis=1, keepdims=True)) ** 2).sum(axis=1)
            new = p_pe.reshape(nobs, nsamp) / rho
            new = new / new.mean(axis=1, keepdims=True)
            assert np.all(np.isfinite(new)) and np.all(new > 0)
            dst = Path(args.outdir) / f"events_{tracer}_hosted_pefix_{arm}.h5"
            shutil.copyfile(src, dst)
            with h5py.File(dst, "r+") as f:
                f["p_pe"][:] = new.ravel()
                f.attrs["pe_mass_correction"] = arm
                f.attrs["pe_mass_correction_note"] = (
                    "p_pe divided by rho = p_ex(m|obs)/N(m;obs,sig_stored) so the "
                    "self-normalised evidence equals the exact-flat-prior-posterior "
                    "reweighting of the stored samples; H0-independent.")
                f.attrs["pe_mass_correction_source"] = str(src)
            report["files"][f"{tracer}_{arm}"] = {
                "path": str(dst), "nobs": nobs, "nsamp": nsamp,
                "sig_m1_over_f_m1true_maxdev": dev1,
                "sig_m2_over_f_m2true_maxdev": dev2,
                "ess_mean": float(ess.mean()), "ess_min": float(ess.min()),
                "log_rho_range": [float(lr.min()), float(lr.max())],
                "p_pe_new_min": float(new.min()), "p_pe_new_max": float(new.max()),
            }
            print(f"[{tracer}/{arm}] -> {dst.name}  ESS {ess.mean():.0f} "
                  f"(min {ess.min():.0f})  ln rho in [{lr.min():.2f}, 0]")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
