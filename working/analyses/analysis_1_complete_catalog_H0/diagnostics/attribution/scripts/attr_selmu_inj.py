#!/usr/bin/env python3
"""TASK 1b -- darksirens' INJECTION-BASED selection integral on an H0 grid.

Companion to ``attr_selmu_oracle.py``.  Evaluates darksirens' own

    log mu_hat(H0) = logsumexp_j [ ldw_j(H0) ] - log Ndraw,
    ldw = log p_pop + log p_z(z|pix) - log(ddL/dz) - log(1+z) - log pdraw

through ``attr_ds_bridge`` -- the SAME loader, the SAME operands and the SAME
``|Delta log mu| = 0`` anchor every attribution run has used -- at every H0 on a
grid, so the estimator's ``d ln mu/dH0`` can be compared to the closed-form
oracle across the scanned range instead of only at truth.

Also records, per H0: N_eff, the term-summed analytic score
``<d ln p_target/dH0>_injections`` (the convention ``attr_score_terms`` reports),
and the branch decomposition of the weight, so a discrepancy can be localised to
a proposal branch.

Finally it extracts the generator's own population-branch detection bookkeeping
(``injections_*_meta.json::pdet_z_grid``), which is a DIRECT empirical
measurement of ``F(z)`` -- the oracle's mass-integrated detection probability --
on ~1e8 proposals, independent of everything in the likelihood.

Outputs: results/attr_selmu_inj_<tracer>_<lane>.{json,npz}
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

DATA = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data")
H0_FID = 67.74

H0_GRID = [50.0, 55.0, 60.0, 65.0, 67.74, 70.0, 75.0, 80.0, 90.0, 100.0]
FD_STEPS = [1.0, 0.5, 0.25, 0.125]


def lse(x):
    f = np.isfinite(x)
    if not f.any():
        return -np.inf
    m = x[f].max()
    return float(m + np.log(np.exp(np.where(f, x - m, -np.inf)).sum()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracer", choices=["gal", "agn"], default="gal")
    ap.add_argument("--injections", choices=["targeted", "popuni"],
                    default="targeted")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--dataroot", default=None,
                    help="Root holding seed<N>/ (default: working/data).")
    ap.add_argument("--events", default=None,
                    help="Events file handed to the bridge (the anchor only).")
    ap.add_argument("--sel_batch", type=int, default=50000)
    ap.add_argument("--grid_only", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    args = ap.parse_args(argv)
    tag = args.tag or f"{args.tracer}_{args.injections}"
    od = Path(args.outdir)
    t00 = time.time()

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    import jax
    import jax.numpy as jnp
    import attr_ds_bridge as bridge

    kw = dict(kde_window=4096) if args.tracer == "gal" else {}
    B = bridge.build(tracer=args.tracer, seed=args.seed, h0=H0_FID,
                     injections=args.injections, sel_batch=args.sel_batch, **kw,
                     **({"dataroot": args.dataroot} if args.dataroot else {}),
                     **({"events": args.events} if args.events else {}))
    nsel = int(B.gw_sel.dL.shape[0])
    print(f"[inj] {args.tracer}/{args.injections}: {nsel:,} detected injections, "
          f"Ndraw={B.Ndraw:.3e}", flush=True)

    import h5py
    with h5py.File(B.paths["injections"], "r") as f:
        branch = (np.asarray(f["branch"][:], int) if "branch" in f
                  else np.zeros(nsel, int))
        pdraw_mix = np.asarray(f["pdraw"][:], float)
        pdraw_b = {}
        for nm, key in (("population", "pdraw_population"),
                        ("uniform", "pdraw_uniform"),
                        ("targeted_agn", "pdraw_targeted_agn")):
            if key in f:
                pdraw_b[nm] = np.asarray(f[key][:], float)
    nb_br = int(branch.max()) + 1
    # darksirens pads the selection block up to a round batch multiple; the pad
    # rows carry valid = False and are already excluded by ``pieces``.  Pad the
    # file-side arrays to match and mark the pad with branch = -1.
    n_file = int(pdraw_mix.size)
    n_pad = nsel - n_file
    if n_pad > 0:
        pdraw_mix = np.concatenate([pdraw_mix, np.zeros(n_pad)])
        branch = np.concatenate([branch, np.full(n_pad, -1, int)])
        for k in list(pdraw_b):
            pdraw_b[k] = np.concatenate([pdraw_b[k], np.zeros(n_pad)])
    print(f"[inj] file rows {n_file:,}, darksirens block {nsel:,} "
          f"({n_pad} padded)", flush=True)
    # darksirens' prior_wt must BE the stored pdraw -- check it, then the
    # single-branch estimators below can be built by shifting ldw.
    pw_ds = np.asarray(B.gw_sel.prior_wt, float)
    pdraw_absdiff = float(np.max(np.abs(pw_ds[:n_file] - pdraw_mix[:n_file])))
    print(f"[check] |darksirens prior_wt - file pdraw|max = {pdraw_absdiff:.3e}",
          flush=True)
    meta0 = json.loads((DATA / f"seed{args.seed}" / "injections"
                        / f"injections_{args.injections}_meta.json").read_text())
    npro_b = meta0.get("n_proposed_branch", {})

    h0_list = sorted(set([float(h) for h in H0_GRID]
                         + ([] if args.grid_only else
                            [H0_FID + s * d for d in FD_STEPS for s in (-1, 1)])
                         + [h + s * 0.5 for h in H0_GRID for s in (-1, 1)]))
    print(f"[inj] {len(h0_list)} H0 evaluations", flush=True)

    def pass_at(h):
        f = B.make_pieces(h)
        acc_ldw = []
        acc_pop, acc_pz, acc_jac = [], [], []
        for j0 in range(0, nsel, args.sel_batch):
            j1 = min(j0 + args.sel_batch, nsel)
            sl = lambda a: jnp.asarray(a)[j0:j1]
            out = f(sl(B.gw_sel.m1det), sl(B.gw_sel.q), sl(B.gw_sel.dL),
                    sl(B.gw_sel.chieff), sl(B.gw_sel.pixels),
                    sl(B.gw_sel.prior_wt), sl(B.gw_sel.valid))
            acc_ldw.append(np.asarray(out[0]))
            acc_pop.append(np.asarray(out[1]))
            acc_pz.append(np.asarray(out[2]))
            acc_jac.append(np.asarray(out[3]))
        del f
        gc.collect()
        return (np.concatenate(acc_ldw), np.concatenate(acc_pop),
                np.concatenate(acc_pz), np.concatenate(acc_jac))

    rec = {}
    ldw_store = {}
    for k, h in enumerate(h0_list):
        t1 = time.time()
        ldw, lpop, lpz, ljac = pass_at(h)
        lm = lse(ldw) - np.log(B.Ndraw)
        fin = np.isfinite(ldw)
        mx = ldw[fin].max()
        w = np.where(fin, np.exp(ldw - mx), 0.0)
        sw = w.sum()
        neff = float(sw ** 2 / (w ** 2).sum())
        wn = w / sw
        br = [float(wn[branch == b].sum()) for b in range(nb_br)]
        # SINGLE-BRANCH estimators.  Each proposal branch is a valid importance
        # sampler on its own, with its OWN declared density, so
        #   mu_b = (1/N_b) SUM_{j in branch b, detected} p_target(j)/q_b(j)
        # estimates the SAME mu.  Disagreement between branches localises a
        # mis-declared proposal density (pdraw), which the mixture estimator
        # alone cannot separate from a wrong target.
        bmu = {}
        names = meta0.get("branch_names", [])
        for bi, nm in enumerate(names):
            if nm not in pdraw_b or nm not in npro_b:
                continue
            sel_b = branch == bi
            n_b = float(npro_b.get(nm, 0.0))
            if not sel_b.any() or n_b <= 0:
                continue
            shift = np.log(np.maximum(pdraw_mix[sel_b], 1e-300)) \
                - np.log(np.maximum(pdraw_b[nm][sel_b], 1e-300))
            bmu[nm] = lse(ldw[sel_b] + shift) - np.log(n_b)
        rec[h] = {"log_mu": lm, "Neff": neff, "branch_weight": br,
                  "n_finite": int(fin.sum()), "log_mu_branch": bmu}
        ldw_store[h] = ((lpop, lpz, ljac, wn, ldw.copy())
                        if abs(h - H0_FID) <= 0.55 else None)
        print(f"  H0={h:8.4f}  log_mu={lm:+.10f}  Neff={neff:.4e}  "
              f"({time.time()-t1:.0f}s)", flush=True)
        del ldw, lpop, lpz, ljac, w, wn
        gc.collect()

    # --- derivatives -----------------------------------------------------------
    fd_truth = {}
    for d in FD_STEPS:
        a, b = H0_FID - d, H0_FID + d
        if a in rec and b in rec:
            fd_truth[str(d)] = (rec[b]["log_mu"] - rec[a]["log_mu"]) / (2.0 * d)
    grid_fd = {}
    for h in H0_GRID:
        a, b = h - 0.5, h + 0.5
        if a in rec and b in rec:
            grid_fd[str(h)] = (rec[b]["log_mu"] - rec[a]["log_mu"]) / 1.0
    branch_fd = {}
    a, b = H0_FID - 0.5, H0_FID + 0.5
    if a in rec and b in rec:
        for nm in rec[a].get("log_mu_branch", {}):
            if nm in rec[b]["log_mu_branch"]:
                branch_fd[nm] = (rec[b]["log_mu_branch"][nm]
                                 - rec[a]["log_mu_branch"][nm]) / 1.0
        branch_fd["_log_mu_at_truth"] = rec[H0_FID].get("log_mu_branch", {})
        print("[branch] d ln mu/dH0 per proposal branch: "
              + "  ".join(f"{k}:{v:+.6e}" for k, v in branch_fd.items()
                          if not k.startswith("_")), flush=True)

    # --- the MONTE-CARLO error of the injection estimator's own d ln mu/dH0 -----
    # r = <d ln Z_i/dH0> - d ln mu/dH0 subtracts ONE number from every event, so
    # whatever error mu_hat's slope carries is a COMMON-MODE shift of r that the
    # per-event sem cannot see and that does not average down with nEvents.
    # Two independent estimates: a Poisson bootstrap over injections, and the
    # delta-method influence function of the same statistic.
    mcerr = {}
    for dstep in (0.5, 0.25):
        a, b = H0_FID - dstep, H0_FID + dstep
        if ldw_store.get(a) is None or ldw_store.get(b) is None:
            continue
        lm, lp = ldw_store[a][4], ldw_store[b][4]
        fin = np.isfinite(lm) & np.isfinite(lp)
        mx = max(lm[fin].max(), lp[fin].max())
        em = np.where(fin, np.exp(lm - mx), 0.0)
        ep = np.where(fin, np.exp(lp - mx), 0.0)
        u = ep / ep.sum()
        v = em / em.sum()
        psi = u - v
        delta_sd = float(np.sqrt((psi ** 2).sum()) / (2.0 * dstep))
        rb = np.random.default_rng(12345)
        vals = []
        for _ in range(200):
            m = rb.poisson(1.0, size=lm.size).astype(np.float64)
            sp, sm = float((ep * m).sum()), float((em * m).sum())
            if sp <= 0 or sm <= 0:
                continue
            vals.append((np.log(sp) - np.log(sm)) / (2.0 * dstep))
        vals = np.asarray(vals)
        mcerr[str(dstep)] = {
            "fd": float((rec[b]["log_mu"] - rec[a]["log_mu"]) / (2.0 * dstep)),
            "bootstrap_sd": float(vals.std(ddof=1)),
            "bootstrap_mean": float(vals.mean()),
            "delta_method_sd": delta_sd,
            "n_bootstrap": int(vals.size)}
        print(f"[mc-error] dh={dstep}: d ln mu/dH0 = {mcerr[str(dstep)]['fd']:+.8e} "
              f"+- {mcerr[str(dstep)]['bootstrap_sd']:.3e} (bootstrap) "
              f"/ {delta_sd:.3e} (delta method)", flush=True)

    # term-summed analytic score at truth (attr_score_terms' convention)
    term = {}
    if (H0_FID - 0.5) in rec and (H0_FID + 0.5) in rec:
        pm = ldw_store[H0_FID - 0.5]
        pp = ldw_store[H0_FID + 0.5]
        w0 = ldw_store[H0_FID][3]
        d = lambda i: np.nan_to_num((pp[i] - pm[i]) / 1.0)
        term = {"pop": float((w0 * d(0)).sum()), "pz": float((w0 * d(1)).sum()),
                "jac": float(-(w0 * d(2)).sum())}
        term["total"] = term["pop"] + term["pz"] + term["jac"]
        print(f"[term-sum] d ln mu/dH0 = {term['total']:+.8e} "
              f"(pop {term['pop']:+.5e}, pz {term['pz']:+.5e}, "
              f"jac {term['jac']:+.5e})", flush=True)
    print("[fd] " + "  ".join(f"dh={k}:{v:+.8e}" for k, v in fd_truth.items()),
          flush=True)

    # --- the generator's own population-branch P_det(z) ------------------------
    mp = (DATA / f"seed{args.seed}" / "injections"
          / f"injections_{args.injections}_meta.json")
    meta = json.loads(mp.read_text())
    pz = meta.get("pdet_z_grid", {})
    pdet_emp = {}
    if pz:
        e = np.asarray(pz["edges"], float)
        npro = np.asarray(pz["n_proposed_population"], float)
        ndet = np.asarray(pz["n_detected_population"], float)
        pdet_emp = {"edges": e.tolist(), "n_proposed": npro.tolist(),
                    "n_detected": ndet.tolist()}

    out = {"name": "attr_selmu_inj", "tracer": args.tracer,
           "injections": args.injections, "seed": args.seed, "tag": tag,
           "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "config": B.paths, "Ndraw": float(B.Ndraw), "n_injections": nsel,
           "anchor_log_mu_darksirens": B.spy0["log_mu"],
           "anchor_log_mu_absdiff": abs(rec[H0_FID]["log_mu"]
                                        - B.spy0["log_mu"]),
           "branch_names": meta.get("branch_names", []),
           "per_H0": {str(h): rec[h] for h in h0_list},
           "prior_wt_minus_file_pdraw_maxabs": pdraw_absdiff,
           "n_file_rows": n_file, "n_padded": int(n_pad),
           "n_proposed_branch": npro_b,
           "fd_at_truth": fd_truth, "grid_fd_dh0p5": grid_fd,
           "mc_error_of_dlnmu": mcerr,
           "branch_only_fd_at_truth": branch_fd,
           "term_sum_at_truth": term,
           "pdet_z_empirical": pdet_emp}
    print(f"[anchor] |log_mu(mine) - log_mu(darksirens)| = "
          f"{out['anchor_log_mu_absdiff']:.3e}", flush=True)
    (od / f"attr_selmu_inj_{tag}.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(od / f"attr_selmu_inj_{tag}.npz",
                        h0=np.array(h0_list),
                        log_mu=np.array([rec[h]["log_mu"] for h in h0_list]),
                        neff=np.array([rec[h]["Neff"] for h in h0_list]))
    print(f"Wrote {od/f'attr_selmu_inj_{tag}.json'}  ({time.time()-t00:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
