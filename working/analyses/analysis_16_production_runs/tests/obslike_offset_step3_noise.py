#!/usr/bin/env python
"""Step 3 (d) of the obslike_f0p30_s1 H0-offset diagnosis: the measurement-noise draws of the detected mergers, per class.

For a merger whose host is (nearly) identified, the H0 information is the recorded distance against the host's redshift, so a
coherent offset of ln dL_obs - ln dL_true over the AGN-hosted mergers would show up as an H0 offset. The recorded data
(group `truth`, v3 measurement model) are rho_obs = rho_opt + N(0, 1) (the detection statistic) and ln Mc_obs = ln Mc_det +
N(0, sig_lnmc); the point distance is dL_obs = 1000 snr_ref (Mc_obs/30)^(5/6) / rho_obs. Per set and class, pooled over
realizations 100-103, 105:
  * pull_lnmc = (obs_lnmc - ln Mc_det,true) / obs_sig_lnmc: N(0, 1), untouched by selection (detection reads rho only);
  * d_rho = obs_rho - rho_true: given detection a N(0, 1) truncated to rho_obs >= 8, whose mean is predicted per merger from
    its own rho_true (phi(a) / (1 - Phi(a)), a = 8 - rho_true) -- the event-by-event prediction, no injections needed;
  * ln(dL_obs / dL_true) and its prediction under the same truncation (Monte Carlo over the rho noise given rho_true,
    the ln Mc noise N(0, sig) at the recorded width);
  * pull_chi = (obs_chieff - chieff_true) / obs_sig_chieff, and the sky pulls, as further checks of the noise draws.
Output: tests/results/obslike_offset_step3_noise.json
"""
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import norm

A16 = Path(__file__).resolve().parents[1]
DATA = A16.parents[1] / "data"
REAL = [100, 101, 102, 103, 105]
SETS = {"obslike_f0p30_s1": "events_marked_dmu0p10_dmuG5_obslike.h5", "obslike_f0p30_s0": "events_obslike.h5",
        "obslike_f1p00_s0": "events_pureagn_obslike.h5", "peak_f0p30_s1": "events_marked_dmu0p10_dmuG5_corepop.h5"}
SNR_REF = 6.278363879917771
KEYS = ("host_type", "obs_rho", "obs_rho_true", "snr_true", "obs_lnmc", "obs_sig_lnmc", "m1det", "m2det", "dl",
        "obs_chieff", "obs_sig_chieff", "chieff", "obs_dec", "dec", "obs_sigma_ang")


def main():
    rng = np.random.default_rng(7)
    res = {}
    for label, fn in SETS.items():
        acc = {}
        for r in REAL:
            with h5py.File(DATA / f"seed{r}" / "events" / fn, "r") as f:
                T = {k: np.asarray(f["truth/" + k][:]) for k in KEYS}
            q = T["m2det"] / T["m1det"]
            mc = T["m1det"] * q ** 0.6 / (1 + q) ** 0.2
            rho_t = T["obs_rho_true"]
            pull_mc = (T["obs_lnmc"] - np.log(mc)) / T["obs_sig_lnmc"]
            d_rho = T["obs_rho"] - rho_t
            a = 8.0 - rho_t
            d_rho_pred = norm.pdf(a) / np.maximum(1 - ndtr(a), 1e-300)
            dl_obs = 1000.0 * SNR_REF * (np.exp(T["obs_lnmc"]) / 30.0) ** (5 / 6) / T["obs_rho"]
            lndl = np.log(dl_obs / T["dl"])
            # prediction of E[ln dL_obs/dL_true | det, theta]: rho noise truncated, Mc noise at the recorded width
            M = 4000
            u = rng.random((M, rho_t.size))
            A = ndtr(a)[None, :]
            rho_o = rho_t[None, :] + ndtri(np.clip(A + u * (1 - A), 1e-300, 1 - 1e-16))
            e_mc = T["obs_sig_lnmc"][None, :] * rng.standard_normal((M, rho_t.size))
            lndl_pred = ((5 / 6) * e_mc - np.log(rho_o / rho_t[None, :])).mean(axis=0)
            pull_chi = (T["obs_chieff"] - T["chieff"]) / T["obs_sig_chieff"]
            pull_dec = (T["obs_dec"] - T["dec"]) / T["obs_sigma_ang"]
            ok_rho = np.allclose(rho_t, T["snr_true"])
            for c, ht in (("gal", 0), ("agn", 1)):
                m = T["host_type"] == ht
                if not m.any():
                    continue
                A_ = acc.setdefault(c, {k: [] for k in ("pull_mc", "d_rho", "d_rho_pred", "lndl", "lndl_pred", "pull_chi",
                                                        "pull_dec", "real")})
                for k, v in (("pull_mc", pull_mc), ("d_rho", d_rho), ("d_rho_pred", d_rho_pred), ("lndl", lndl),
                             ("lndl_pred", lndl_pred), ("pull_chi", pull_chi), ("pull_dec", pull_dec)):
                    A_[k].append(v[m])
                A_["real"].append(np.full(int(m.sum()), r))
            res.setdefault(label, {})["rho_true_equals_snr_true"] = res.get(label, {}).get("rho_true_equals_snr_true", True) and bool(ok_rho)
        for c, A_ in acc.items():
            V = {k: np.concatenate(v) for k, v in A_.items()}
            n = V["pull_mc"].size
            dd = V["d_rho"] - V["d_rho_pred"]
            dl = V["lndl"] - V["lndl_pred"]
            out = {"n": int(n),
                   "pull_lnmc_mean": float(V["pull_mc"].mean()), "pull_lnmc_z": float(V["pull_mc"].mean() * np.sqrt(n)),
                   "pull_lnmc_sd": float(V["pull_mc"].std()),
                   "d_rho_mean": float(V["d_rho"].mean()), "d_rho_pred_mean": float(V["d_rho_pred"].mean()),
                   "d_rho_excess_z": float(dd.mean() / (dd.std() / np.sqrt(n))),
                   "lndl_mean": float(V["lndl"].mean()), "lndl_pred_mean": float(V["lndl_pred"].mean()),
                   "lndl_excess": float(dl.mean()), "lndl_excess_z": float(dl.mean() / (dl.std() / np.sqrt(n))),
                   "pull_chi_mean_z": float(V["pull_chi"].mean() * np.sqrt(n)),
                   "pull_dec_mean_z": float(V["pull_dec"].mean() * np.sqrt(n)),
                   "per_realization_lndl_excess_z": {}}
            for r in REAL:
                m = V["real"] == r
                if m.sum() > 1:
                    out["per_realization_lndl_excess_z"][str(r)] = float(dl[m].mean() / (dl[m].std() / np.sqrt(m.sum())))
            res[label][c] = out
            print(f"{label:18s} {c}: n={n} pull lnMc {out['pull_lnmc_mean']:+.4f} (z {out['pull_lnmc_z']:+.2f}, sd {out['pull_lnmc_sd']:.3f}) | "
                  f"d_rho {out['d_rho_mean']:.4f} pred {out['d_rho_pred_mean']:.4f} (z {out['d_rho_excess_z']:+.2f}) | "
                  f"ln dL_obs/dL_true {out['lndl_mean']:+.5f} pred {out['lndl_pred_mean']:+.5f} excess {out['lndl_excess']:+.5f} "
                  f"(z {out['lndl_excess_z']:+.2f}) | chi z {out['pull_chi_mean_z']:+.2f} dec z {out['pull_dec_mean_z']:+.2f} | per-real lndl z "
                  + " ".join(f"{v:+.2f}" for v in out["per_realization_lndl_excess_z"].values()))
    p = A16 / "tests" / "results" / "obslike_offset_step3_noise.json"
    p.write_text(json.dumps(res, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
