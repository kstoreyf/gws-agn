#!/usr/bin/env python
"""Analysis 15 GPU driver: the host-catalog normalisation volume in the two-catalog likelihood.

    [DARKSIRENS_ZMAX=1.10] <core venv python> a15_host_norm.py --scans p1_complete,p1_m21,...
    (--dry-run on CPU: 20 events, 20000 injections, 2x2 cells per scan, output to --out)

Likelihood: darksirens-core e7c3007 with the "historical" catalogue evaluation of Analyses 13/14
(kernel_window 1e-10, padded, missing_density grid, pairing per_sample/analytic). K = 2 GAL + AGN
seed-100 catalogs at nside 32, field weighting, completeness="selection" with the true-z
Schechter fits of the legacy runs (m_lim, Mstar_hat, alpha, M_faint_offset as stored; the
selection nuisances are fixed at the fits), field normaliser "auto" (= moments), delta =
sigma_kde = 0, Om0 0.3075, n0 physical (Mpc^-3), one shared powerlaw+peak population fixed at
core's fiducials with gamma = 0 (the mock's). log10n0 and log10n0_c2 are coordinates (uniform
(-25, -1) so the bound program is the same for every density) and are set per scan; H0 and
fcat_2 = f_AGN are scanned. max_likelihood_variance 1e6 (inert, as every stored run), sel_batch
5000, pe_event_block 5. Events: seed 100 events.h5 (mixed, input f_AGN 0.30); injections:
injections_targeted.h5. Nothing on disk is modified.

The redshift grid end is a process constant of darksirens (DARKSIRENS_ZMAX, read at import);
every scan declares the grid it needs ("5" or "R" = a15_common.ZMAX_R) and the driver refuses a
scan whose grid differs from the imported one. The sbatch runs one process per grid block.

Per scan: ln L, PE variance sum, N_eff, ln mu and the selection term at every cell (core's
return_diagnostics path, checked equal to the bound call), each catalog's N_obs, missing total
and Z_k at every H0 row and at H0 = 67.74 (core's own normaliser code: likelihood/mixture.py
_observed + _missing_total) next to the closed form, and a posterior summary ->
results/scans/<name>.{h5,json}. Gate scans (Part 1 and the zero-density reference) are compared
with the stored legacy grids -> results/gate_<name>.json; scans that depend on a failed or
missing gate are not run. A resubmission skips every scan whose .json exists.
"""
import argparse
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a15_common as C  # noqa: E402

MAXVAR = 1e6
SEL_BATCH = 5000
PEB = 5
N0_PRIOR = (-25.0, -1.0)
REF_CELL = (67.75, 0.30)


def log(msg):
    print(f"[a15 {time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)


# ----------------------------------------------------------------------------- model
def schechter(path):
    from darksirens.selection.catalog import SchechterMagnitudeSelection
    s = C.schechter_fit(path)
    return SchechterMagnitudeSelection(m_lim=s["m_lim"], Mstar_hat=s["Mstar_hat"], alpha=s["alpha"],
                                       M_faint_offset=s["M_faint_offset"])


def build_model(ds, rung):
    from darksirens.catalog import settings as cs
    from darksirens.population.utils import configure_normalization_grids
    if f"darksirens-core-{C.CORE}" not in ds.__file__:
        sys.exit(f"[fatal] darksirens imported from {ds.__file__}, not core {C.CORE}")
    cs.configure_catalog_evaluation(kernel_window=1e-10, kernel_layout="padded", missing_density="grid")
    ng = configure_normalization_grids(pairing_norm="per_sample", pairing_scale="analytic")
    st = cs.catalog_evaluation_settings()
    settings = {k: str(getattr(st, k)) for k in st.__dataclass_fields__}
    settings.update({"pairing_norm": str(getattr(ng, "pairing_norm", "per_sample")),
                     "pairing_scale": str(getattr(ng, "pairing_scale", "analytic"))})
    cos = ds.Cosmology(H0=(50.0, 100.0), Om0=C.OM0)
    probe = ds.model(cosmology=cos, population=ds.Population("powerlaw+peak", fixed=True)).parameters
    fv = dict(zip(probe.population_labels, probe.fixed_population))
    g = [k for k in fv if "gamma" in k.lower()]
    assert len(g) == 1, fv
    fv[g[0]] = 0.0
    R = C.RUNGS[rung]
    cats = [ds.load_catalog(str(p)) for p in R["surveys"]]
    for p, c in zip(R["surveys"], cats):
        if c.z_depth is not None:
            sys.exit(f"[fatal] {p} carries z_depth {c.z_depth}; this analysis runs without a survey depth")
    sel = [schechter(p) for p in R["fits"]]
    an = ds.model(cosmology=cos, population=ds.Population("powerlaw+peak", fixed=fv),
                  catalog=cats, catalog_sky_weighting="field", completeness="selection",
                  selection=sel, field_normalizer="auto",
                  survey_priors={"log10n0": N0_PRIOR, "log10n0_c2": N0_PRIOR},
                  fixed_survey={"delta": 0.0, "sigma_kde": 0.0, "delta_c2": 0.0, "sigma_kde_c2": 0.0})
    info = {"surveys": [str(p) for p in R["surveys"]], "selection_fits": [str(p) for p in R["fits"]],
            "selection": [C.schechter_fit(p) for p in R["fits"]],
            "completeness": list(an.redshift.catalog_completeness),
            "normalizer": list(an.redshift.catalog_normalizers),
            "n_obs": [int(np.sum(np.asarray(c.catalog.ngals, dtype=np.int64))) for c in cats],
            "n_rows": [int(np.shape(c.catalog.ngals)[0]) for c in cats]}
    return an, settings, {k: float(v) for k, v in fv.items()}, info


def make_diag(an, n_events, nsamp, n_draw):
    """The bound call's program with return_diagnostics=True (as Analysis 14's driver)."""
    import jax
    import jax.numpy as jnp
    from darksirens.cosmology.distances import threads_distance_table
    from darksirens.likelihood.mixture import field_mixture_log_likelihood
    from darksirens.runtime_binding import _decode_theta
    pop = an.population

    @threads_distance_table()
    def diag(theta, gw_pe, gw_sel, model_operands, distance_table=None):
        cosmology, population, catalog_params, angular = _decode_theta(an, theta, z_depth=None)
        d = field_mixture_log_likelihood(
            cosmology, population, catalog_params, gw_pe, gw_sel, model_operands,
            n_events, nsamp, n_draw,
            completeness=an.redshift.completeness, normalizer=an.redshift.normalizer,
            pop_model=pop.model_name, shared_beta=pop.shared_beta, shared_spin=pop.shared_spin,
            shared_gamma=pop.shared_gamma, sel_batch_size=SEL_BATCH,
            selection_neff_soft_guard=False, max_likelihood_variance=MAXVAR,
            pe_event_block=PEB, angular_model=an.angular_model, angular_params=angular,
            return_diagnostics=True)
        return (d.log_likelihood, jnp.sum(d.event_mc_variance), d.n_eff, d.log_mu,
                d.selection_log_correction, d.event_log_evidence, d.event_mc_variance)
    return jax.jit(diag)


def make_zk(an):
    """Each catalog's (sum_p N_obs,p m_p, sum_p N_miss,p) by the likelihood's own normaliser code."""
    import jax
    import jax.numpy as jnp
    from darksirens.cosmology.distances import threads_distance_table
    from darksirens.likelihood.mixture import _missing_total, _observed
    from darksirens.runtime_binding import _decode_theta
    comp = tuple(an.redshift.catalog_completeness)
    norm = tuple(an.redshift.catalog_normalizers)

    @threads_distance_table()
    def zk(theta, model_operands, distance_table=None):
        cosmology, _, catalog_params, _ = _decode_theta(an, theta, z_depth=None)
        obs, miss = [], []
        for k, (params, component) in enumerate(zip(catalog_params.components, model_operands.components)):
            o, _ = _observed(cosmology, params, component)
            m = _missing_total(k, cosmology, params, component, comp[k], norm[k],
                               None, None, None, None, o)
            obs.append(o)
            miss.append(m)
        return jnp.stack(obs), jnp.stack(miss)
    return jax.jit(zk)


# ----------------------------------------------------------------------------- data
EV_COLUMNS = ("ra", "dec", "dL", "m1det", "m2det", "chieff", "p_pe", "m1src", "m2src")


def events_subset(ds, n_ev):
    """The first n_ev events of events.h5 as an in-memory store (dry run only)."""
    import h5py
    bio = io.BytesIO()
    with h5py.File(C.EVENTS, "r") as src, h5py.File(bio, "w") as dst:
        nsamp = int(src.attrs["nsamp"])
        for k, v in src.attrs.items():
            if k != "metadata_json":
                dst.attrs[k] = v
        dst.attrs["nobs"] = int(n_ev)
        for k in EV_COLUMNS:
            dst.create_dataset(k, data=np.asarray(src[k][: n_ev * nsamp], dtype=np.float64))
    bio.seek(0)
    return ds.load_events(bio)


def injections_subset(store, n):
    import dataclasses
    cols = {k: np.asarray(v)[:n] for k, v in store.columns.items()}
    return dataclasses.replace(store, columns=cols, n_injections=int(n), prior_wt=np.asarray(store.prior_wt)[:n])


# ----------------------------------------------------------------------------- gate
def compare(spec, axes, lnL, pe_var, n_eff, outdir):
    """Core against the stored legacy grid on the same cells (Analysis 14's gate)."""
    import h5py
    st = C.stored_on_cells(spec["stored"], axes["H0"], axes["f"])
    with h5py.File(spec["stored"], "r") as f:
        H = np.asarray(f["H0_grid"])
        F = np.asarray(f["f_grid"])
        ih = [int(np.argmin(np.abs(H - h))) for h in axes["H0"]]
        jf = [int(np.argmin(np.abs(F - x))) for x in axes["f"]]
        pv = np.asarray(f["guard/pe_variance_sum"])[np.ix_(ih, jf)]
        ne = np.asarray(f["guard/Neff"])[np.ix_(ih, jf)]
        a = {k: str(f.attrs[k]) for k in ("arg_log10n0", "arg_log10n0_c2", "darksirens_git_sha", "c_mode",
                                          "arg_selection_fit") if k in f.attrs}
    Ls = st["lnL"]
    mine = C.summarize_2d(axes["H0"], axes["f"], lnL)
    ref = C.summarize_2d(axes["H0"], axes["f"], Ls)
    w = np.exp(Ls - Ls.max()).ravel()
    sel = w > 1e-3
    d = (lnL - Ls).ravel()
    rows, ok = {}, True
    for p in ("H0", "f"):
        x, y = mine[p], ref[p]
        r = {"core_median": x["median"], "stored_median": y["median"],
             "median_shift_over_hw68": (x["median"] - y["median"]) / y["hw68"],
             "w68_ratio": x["w68"] / y["w68"], "w90_ratio": x["w90"] / y["w90"],
             "core_ci68": x["ci68"], "stored_ci68": y["ci68"], "core_ci90": x["ci90"], "stored_ci90": y["ci90"]}
        r["pass"] = bool(abs(r["median_shift_over_hw68"]) <= C.GATE["dmed_hw"]
                         and C.GATE["wratio_lo"] <= r["w68_ratio"] <= C.GATE["wratio_hi"]
                         and C.GATE["wratio_lo"] <= r["w90_ratio"] <= C.GATE["wratio_hi"])
        ok &= r["pass"]
        rows[p] = r
    out = {"scan": spec["name"], "reference": str(spec["stored"]), "reference_attrs": a, "gate": C.GATE,
           "pass": bool(ok), "params": rows, "rho_core": mine["rho"], "rho_stored": ref["rho"],
           "dlnL_core_minus_stored": {"mean_in_bulk": float(np.mean(d[sel])), "sd_in_bulk": float(np.std(d[sel])),
                                      "min": float(np.min(d)), "max": float(np.max(d)), "n_bulk_cells": int(sel.sum())},
           "pe_var_ratio_core_over_stored_bulk": [float(np.min((pe_var.ravel() / pv.ravel())[sel])),
                                                  float(np.max((pe_var.ravel() / pv.ravel())[sel]))],
           "n_eff_ratio_core_over_stored_bulk": [float(np.min((n_eff.ravel() / ne.ravel())[sel])),
                                                 float(np.max((n_eff.ravel() / ne.ravel())[sel]))],
           "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (outdir / f"gate_{spec['name']}.json").write_text(json.dumps(out, indent=1))
    return out


def gate_state(outdir, name):
    p = outdir / f"gate_{name}.json"
    if not p.exists():
        return None
    return bool(json.loads(p.read_text())["pass"])


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", required=True, help="comma list of scan names (a15_common.SCANS), run in order")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(C.A15 / "results"))
    ap.add_argument("--dry-events", type=int, default=20)
    ap.add_argument("--dry-injections", type=int, default=20000)
    args = ap.parse_args()
    names = [s for s in args.scans.split(",") if s]
    for n in names:
        if n not in C.BY_NAME:
            sys.exit(f"[fatal] unknown scan {n!r}")
    dry = args.dry_run
    outdir = Path(args.out)
    (outdir / "scans").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    deadline = float(os.environ.get("A15_DEADLINE", time.time() + 1e7))
    import jax
    import jax.numpy as jnp
    import darksirens as ds
    from darksirens.cosmology import _grid, distances
    zmax_now = float(_grid.zMax)
    if float(distances.zMax) != zmax_now:
        sys.exit(f"[fatal] grid zMax {zmax_now} != distance-table zMax {distances.zMax}")
    for n in names:
        want = C.zmax_of(C.BY_NAME[n]["zmax"])
        if abs(want - zmax_now) > 1e-12:
            sys.exit(f"[fatal] scan {n} needs the redshift grid to {want}, this process has {zmax_now} "
                     f"(DARKSIRENS_ZMAX={os.environ.get('DARKSIRENS_ZMAX')})")
    log(f"host {os.uname().nodename} job {os.environ.get('SLURM_JOB_ID')} devices {jax.devices()} "
        f"darksirens {ds.__file__} zMax {zmax_now} (grid nodes {int(_grid.zgrid.shape[0])}, distance nodes "
        f"{int(distances.zgrid.shape[0])}) dry={dry} scans {names}")
    from darksirens.runtime_binding import bind_analysis

    ev_store = events_subset(ds, args.dry_events) if dry else ds.load_events(str(C.EVENTS))
    inj_full = ds.load_injections(str(C.INJECTIONS))
    inj_store = injections_subset(inj_full, args.dry_injections) if dry else inj_full
    built = {}

    def get(rung):
        if rung in built:
            return built[rung]
        built.clear()
        t0 = time.time()
        an, settings, popfixed, info = build_model(ds, rung)
        b = bind_analysis(an, events=ev_store, injections=inj_store, max_likelihood_variance=MAXVAR,
                          sel_batch_size=SEL_BATCH, pe_event_block=PEB)
        labels = list(b.labels)
        if sorted(labels) != sorted(["H0", "log10n0", "log10n0_c2", "fcat_2"]):
            sys.exit(f"[fatal] core labels {labels}")
        diag = make_diag(an, int(b.n_events), int(b.nsamp), float(b.n_draw))
        zk = make_zk(an)
        ctx = {"an": an, "b": b, "labels": labels, "diag": diag, "zk": zk, "settings": settings,
               "popfixed": popfixed, "info": info, "rung": rung}

        def theta(h, f, n0):
            v = {"H0": h, "fcat_2": f, "log10n0": n0[0], "log10n0_c2": n0[1]}
            return jnp.asarray([v[l] for l in labels], dtype=jnp.float64)
        ctx["theta"] = theta
        ops = (b.gw_pe, b.gw_selection, b.model_operands)
        ctx["ops"] = ops
        # the diagnostics program must equal the bound call
        for (h, f, n0) in ((67.75, 0.30, (-3.0, -5.0)), (69.0, 0.9, (-24.0, -24.0))):
            vb = float(b(theta(h, f, n0)))
            vd = float(diag(theta(h, f, n0), *ops)[0])
            log(f"[{rung}] diag vs bound at ({h}, {f}, {n0}): {vd:.10f} vs {vb:.10f}")
            if not abs(vd - vb) <= 1e-8 * max(1.0, abs(vb)):
                sys.exit(f"[fatal] the diagnostics program differs from the bound call by {vd - vb:g}")
        log(f"[{rung}] built + bound in {time.time() - t0:.0f}s; labels {labels}; n_events {b.n_events}; "
            f"nsamp {b.nsamp}; n_draw {b.n_draw:g}; n_inj {inj_store.n_injections}; info {json.dumps(info)[:600]}")
        built[rung] = ctx
        return ctx

    common = {"code": f"darksirens-core {C.CORE}", "darksirens_file": ds.__file__, "zmax_grid": zmax_now,
              "grid_nodes": int(_grid.zgrid.shape[0]), "distance_table_nodes": int(distances.zgrid.shape[0]),
              "DARKSIRENS_ZMAX": os.environ.get("DARKSIRENS_ZMAX"), "max_likelihood_variance": MAXVAR,
              "sel_batch_size": SEL_BATCH, "pe_event_block": PEB, "Om0": C.OM0, "dry_run": dry,
              "events_file": str(C.EVENTS), "injection_file": str(C.INJECTIONS),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "n0_units": "physical (Mpc^-3)"}

    for name in names:
        spec = C.BY_NAME[name]
        tag = f"[{name}]"
        if (outdir / "scans" / f"{name}.json").exists():
            log(f"{tag} on record; skipped")
            continue
        if spec["needs"] and not dry:
            g = gate_state(outdir, spec["needs"])
            if g is not True:
                log(f"{tag} [STOP] not run: the gate it depends on ({spec['needs']}) "
                    f"{'FAILED' if g is False else 'is not on record'}; see {outdir}/gate_{spec['needs']}.json")
                continue
        try:
            ctx = None
            ctx = get(spec["rung"])
            # dry run: the H0 grid ends (where the data reach their lowest / highest z) and the
            # lowest f cell next to a middle one
            hs = C.H0_GRID[[0, -1]] if dry else C.H0_GRID
            fs = C.f_grid(*spec["f"])
            if dry:
                fs = fs[[0, len(fs) // 2]]
            cells = [(h, f) for h in hs for f in fs]
            n0 = spec["n0"]
            # Z_k by core's normaliser code at every H0 row and at 67.74, with the closed form
            zk_rows = []
            for h in list(hs) + [C.H0_REF]:
                o, m = ctx["zk"](ctx["theta"](float(h), 0.3, n0), ctx["b"].model_operands)
                zk_rows.append((np.asarray(o, float), np.asarray(m, float)))
            obs = np.array([r[0] for r in zk_rows])
            miss = np.array([r[1] for r in zk_rows])
            Z = obs + miss
            fits = ctx["info"]["selection"]
            zc = np.array([[C.z_closed(ctx["info"]["n_obs"][k], n0[k], fits[k], float(h), zmax_now, "core")
                            for k in (0, 1)] for h in list(hs) + [C.H0_REF]])
            zrel = float(np.max(np.abs(Z / zc - 1.0)))
            log(f"{tag} Z_k at 67.74: GAL {Z[-1, 0]:.6e} (F {obs[-1, 0] / Z[-1, 0]:.6g}), AGN {Z[-1, 1]:.6e} "
                f"(F {obs[-1, 1] / Z[-1, 1]:.6g}); max |core/closed - 1| over H0 rows {zrel:.2e}")
            # the scan
            est = None
            res = {k: np.full(len(cells), np.nan) for k in ("lnL", "pe_var", "n_eff", "log_mu", "sel_corr", "seconds")}
            ri = int(np.argmin([abs(c[0] - REF_CELL[0]) + abs(c[1] - REF_CELL[1]) for c in cells]))
            ref_ev = None
            t1 = time.time()
            for i, (h, f) in enumerate(cells):
                if i == 3 and not dry:
                    spc = float(np.median(res["seconds"][1:3]))
                    est = spc * len(cells)
                    if time.time() + est > deadline:
                        log(f"{tag} [STOP] estimated {est / 60:.1f} min exceeds the time left; left for a resubmission")
                        break
                tc = time.time()
                r = ctx["diag"](ctx["theta"](float(h), float(f), n0), *ctx["ops"])
                vals = [float(x) for x in r[:5]]
                res["seconds"][i] = time.time() - tc
                for k, v in zip(("lnL", "pe_var", "n_eff", "log_mu", "sel_corr"), vals):
                    res[k][i] = v
                if i == ri:
                    ref_ev = (np.asarray(r[5]), np.asarray(r[6]))
            if np.isnan(res["seconds"]).any():
                continue
            shape = (len(hs), len(fs))
            axes = {"H0": np.asarray(hs), "f": np.asarray(fs)}
            L2 = res["lnL"].reshape(shape)
            summ = C.summarize_2d(axes["H0"], axes["f"], L2)
            meta = {**common, "scan": name, "part": spec["part"], "rung": spec["rung"], "zmax_code": spec["zmax"],
                    "label": spec["label"], "log10n0": n0[0], "log10n0_c2": n0[1],
                    "agn_density_factor": float(10 ** (n0[1] + 5.0)) if n0[1] > -10 else 0.0,
                    "ref_scan": spec["ref"], "needs_gate": spec["needs"], "stored": str(spec["stored"]),
                    "catalog_evaluation": ctx["settings"], "population_fixed": ctx["popfixed"], "model": ctx["info"],
                    "labels": ctx["labels"], "n_events": int(ctx["b"].n_events), "nsamp": int(ctx["b"].nsamp),
                    "n_draw": float(ctx["b"].n_draw), "n_injections": int(inj_store.n_injections),
                    "ref_cell": REF_CELL, "ref_cell_index": ri, "wall_seconds": time.time() - t1}
            import h5py
            with h5py.File(outdir / "scans" / f"{name}.h5", "w") as f:
                f.create_dataset("H0_grid", data=axes["H0"])
                f.create_dataset("f_grid", data=axes["f"])
                for k in ("lnL", "pe_var", "n_eff", "log_mu", "sel_corr"):
                    f.create_dataset(k, data=res[k].reshape(shape))
                f.create_dataset("seconds", data=res["seconds"])
                g = f.create_group("normaliser")
                g.create_dataset("H0", data=np.asarray(list(hs) + [C.H0_REF]))
                g.create_dataset("observed_total", data=obs)
                g.create_dataset("missing_total", data=miss)
                g.create_dataset("Z", data=Z)
                g.create_dataset("Z_closed_form", data=zc)
                if ref_ev is not None:
                    g2 = f.create_group("ref_cell")
                    g2.create_dataset("event_log_evidence", data=ref_ev[0])
                    g2.create_dataset("event_mc_variance", data=ref_ev[1])
                f.attrs["meta_json"] = json.dumps(meta)
            out = {**meta, "summary": summ,
                   "Z_at_67.74": {"GAL": float(Z[-1, 0]), "AGN": float(Z[-1, 1])},
                   "N_obs": {"GAL": float(obs[-1, 0]), "AGN": float(obs[-1, 1])},
                   "F_at_67.74": {"GAL": float(obs[-1, 0] / Z[-1, 0]), "AGN": float(obs[-1, 1] / Z[-1, 1])},
                   "ln_FA_over_FG_at_67.74": float(np.log(obs[-1, 1] / Z[-1, 1]) - np.log(obs[-1, 0] / Z[-1, 0])),
                   "Z_core_over_closed_max_abs_rel_dev": zrel,
                   "lnL_max": float(np.nanmax(res["lnL"])), "n_nonfinite": int((~np.isfinite(res["lnL"])).sum()),
                   "pe_var_range": [float(np.nanmin(res["pe_var"])), float(np.nanmax(res["pe_var"]))],
                   "n_eff_range": [float(np.nanmin(res["n_eff"])), float(np.nanmax(res["n_eff"]))],
                   "at_ref_cell": {"lnL": float(res["lnL"][ri]), "pe_var": float(res["pe_var"][ri]),
                                   "n_eff": float(res["n_eff"][ri])},
                   "seconds_per_call_median": float(np.median(res["seconds"][1:])) if len(cells) > 1
                   else float(res["seconds"][0]),
                   "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            (outdir / "scans" / f"{name}.json").write_text(json.dumps(out, indent=1))
            log(f"{tag} {len(cells)} cells in {time.time() - t1:.0f}s ({out['seconds_per_call_median']:.3f} s/call): "
                f"H0 {summ['H0']['median']:.3f} [{summ['H0']['ci90'][0]:.2f}, {summ['H0']['ci90'][1]:.2f}]  "
                f"f {summ['f']['median']:.4f} [{summ['f']['ci90'][0]:.3f}, {summ['f']['ci90'][1]:.3f}]  "
                f"edges(f) {summ['f']['mass_at_edges']}  nonfinite {out['n_nonfinite']}")
            if spec["gate"]:
                rep = compare(spec, axes, L2, res["pe_var"].reshape(shape), res["n_eff"].reshape(shape), outdir)
                log(f"{tag} GATE {'PASS' if rep['pass'] else 'FAIL'}: {json.dumps(rep['params'])}  "
                    f"dlnL {rep['dlnL_core_minus_stored']}")
                if not rep["pass"]:
                    log(f"{tag} [STOP] reproduction FAILED against {spec['stored']}; every scan that depends on "
                        f"{name} will be skipped")
        except Exception as e:  # noqa: BLE001
            log(f"{tag} ERROR {type(e).__name__}: {e}")
            traceback.print_exc()
    done = sorted(p.stem for p in (outdir / "scans").glob("*.json"))
    log(f"finished this block; scans on record: {done}")


if __name__ == "__main__":
    main()
