#!/usr/bin/env python
"""Analysis 14 GPU driver: how much the Monte Carlo noise of ln L moves the posteriors.

    /hildafs/projects/phy230014p/magana/envs/darksirens-core-e7c3007/bin/python a14_mc_accuracy.py
    (--dry-run on CPU: AGN catalog twice, 20 events, 2 cells per scan, output to --out)

Likelihood: darksirens-core e7c3007, "historical" catalogue evaluation set explicitly
(kernel_window 1e-10, kernel_layout padded, missing_density grid, pairing_norm per_sample,
pairing_scale analytic), exactly as Analysis 13's a13_core_sampler.build(). K = 2 complete
GAL + AGN catalogues (seed 100, nside 32), field weighting, completeness None at
log10n0 = log10n0_c2 = -24 (allow_out_of_prior), delta = sigma_kde = 0, Om0 0.3075; ONE shared
population (no per-catalog block), every population value fixed at core's fiducial with
gamma = 0 (the mock's input). Free: H0 and fcat_2 = f_AGN. max_likelihood_variance 1e6 (inert,
as every stored run), sel_batch 5000. Every cell records ln L, the PE variance sum, N_eff,
ln mu and the selection term (core's return_diagnostics path; checked equal to the bound call).

Configurations
  (a) events_pureagn.h5 at f_AGN = 1 (= the AGN-catalog-only likelihood): H0 65.5..71 step 0.05
  (b) events.h5 (input f_AGN 0.30): H0 64..74 step 0.25 x f_AGN 0.05..0.55 step 0.025 (41 x 21)
Redraws: per-event posterior samples redrawn in memory by mc_redraw.py (the generator's own
functions; common random numbers across N_s); injection sets j = 1..5 from
make_injection_redraws.py. Each finished scan writes results/scans/<task>.{h5,json} at once;
a resubmission skips every task whose .json exists.

Order: build -> reproduction of the stored A0/A2 grids with the stored samples and injections
(the job STOPS with exit 3 if the posteriors disagree) -> timing at each N_s -> the adapted
plan (results/plan.json) -> tasks in priority order, each guarded against the wall clock.
"""
import argparse
import dataclasses
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A14 = HERE.parent
sys.path.insert(0, str(HERE))
import a14_post as P  # noqa: E402
import mc_redraw as M  # noqa: E402

CORE = "e7c3007"
WORK = Path("/hildafs/projects/phy230014p/magana/gws-agn/working")
D = WORK / "data" / "seed100"
EVENTS = {"a": D / "events" / "events_pureagn.h5", "b": D / "events" / "events.h5"}
SETNAME = {"a": "pureagn", "b": "mixed"}
INJ_RECORD = D / "injections" / "injections_targeted.h5"
REF = {"a": WORK / "analyses/analysis_0_pure_tracer_H0/results/h0_pureagn_targeted_s100.h5",
       "b": WORK / "analyses/analysis_2_complete_catalog_H0_fagn/results/joint_s100.h5"}
TRUTH = {"H0": 67.74, "f": 0.30}
GRID_A = 65.5 + 0.05 * np.arange(111)
# (a) H0 grids per catalog seed: 111 cells of 0.05 centred on the stored A0 posterior, ends on
# multiples of 0.25 so the stored cells are grid cells. Seed 100's (a) is retired (README:
# "Core and the mocks disagree below the minimum mass"); seed 103 since 2026-11-17.
GRID_A_BY_SEED = {100: 65.5 + 0.05 * np.arange(111), 103: 64.5 + 0.05 * np.arange(111)}
SEED = 100          # the catalog realisation of this run (set in main from --seed-a / configs)
LABEL = {"a": "a", "b": "b"}   # task-name prefix per configuration; "a_s103" off seed 100
GRID_B_H0 = 64.0 + 0.25 * np.arange(41)
GRID_B_F = 0.05 + 0.025 * np.arange(21)
REF_CELL = {"a": (67.75, 1.0), "b": (67.75, 0.30)}
NS_LIST = (2000, 8000, 32000)
PEB = {2000: 5, 8000: 5, 32000: 2, 4000: 5}
SEL_BATCH = 5000
MAXVAR = 1e6
# repro gate (posterior level): |median shift| <= 0.2 half-widths, 68%/90% widths within 10%
GATE = {"dmed_hw": 0.2, "wratio_lo": 0.9, "wratio_hi": 1.1}
# base plan (brief) and the floors the adaptation may shrink to
BASE = [("a", 2000, 20), ("a", 8000, 20), ("a", 32000, 10),
        ("b", 2000, 20), ("b", 8000, 12), ("b", 32000, 6)]
FLOOR = {("a", 2000): 12, ("a", 8000): 10, ("a", 32000): 5,
         ("b", 2000): 12, ("b", 8000): 8, ("b", 32000): 3}
EXTRA_32K = {"a": 20, "b": 12}       # "if time remains": redraws at 32000 up to these counts
INJ_SETS = (1, 2, 3, 4, 5)
INJ_NS, INJ_R = 8000, 0


def inj_set_path(j):
    root = Path(os.path.realpath(D)) / "mc_accuracy" / f"inj_set{j}" / f"seed{SEED}" / "injections"
    return root / "injections_targeted.h5", root / "injections_targeted_meta.json"


def set_seed(seed, cfgs):
    """Point every path of this run at catalog realisation ``seed``."""
    global SEED, D, EVENTS, INJ_RECORD, REF, GRID_A
    SEED = int(seed)
    D = WORK / "data" / f"seed{SEED}"
    EVENTS = {"a": D / "events" / "events_pureagn.h5", "b": D / "events" / "events.h5"}
    INJ_RECORD = D / "injections" / "injections_targeted.h5"
    REF = {"a": WORK / f"analyses/analysis_0_pure_tracer_H0/results/h0_pureagn_targeted_s{SEED}.h5",
           "b": WORK / f"analyses/analysis_2_complete_catalog_H0_fagn/results/joint_s{SEED}.h5"}
    if "a" in cfgs:
        if SEED not in GRID_A_BY_SEED:
            sys.exit(f"[fatal] no (a) H0 grid registered for seed {SEED} (GRID_A_BY_SEED)")
        GRID_A = GRID_A_BY_SEED[SEED]
    for c in ("a", "b"):
        LABEL[c] = c if SEED == 100 else f"{c}_s{SEED}"


def log(msg):
    print(f"[a14 {time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)


# ----------------------------------------------------------------------------- model
def build_model(ds, dry=False):
    from darksirens.catalog import settings as cs
    from darksirens.population.utils import configure_normalization_grids
    if f"darksirens-core-{CORE}" not in ds.__file__:
        sys.exit(f"[fatal] darksirens imported from {ds.__file__}, not core {CORE}")
    cs.configure_catalog_evaluation(kernel_window=1e-10, kernel_layout="padded", missing_density="grid")
    ng = configure_normalization_grids(pairing_norm="per_sample", pairing_scale="analytic")
    st = cs.catalog_evaluation_settings()
    settings = {k: str(getattr(st, k)) for k in st.__dataclass_fields__}
    settings.update({"pairing_norm": str(getattr(ng, "pairing_norm", "per_sample")),
                     "pairing_scale": str(getattr(ng, "pairing_scale", "analytic"))})
    cos = ds.Cosmology(H0=(50.0, 100.0), Om0=0.3075)
    probe = ds.model(cosmology=cos, population=ds.Population("powerlaw+peak", fixed=True)).parameters
    fv = dict(zip(probe.population_labels, probe.fixed_population))
    g = [k for k in fv if "gamma" in k.lower()]
    assert len(g) == 1, fv
    fv[g[0]] = 0.0
    tracers = ("agn", "agn") if dry else ("gal", "agn")
    cats = [ds.load_catalog(str(D / "surveys" / f"survey_{t}_complete_ns32.h5")) for t in tracers]
    an = ds.model(cosmology=cos, population=ds.Population("powerlaw+peak", fixed=fv),
                  catalog=cats, catalog_sky_weighting="field", completeness=None,
                  fixed_survey={"log10n0": -24.0, "log10n0_c2": -24.0, "delta": 0.0,
                                "sigma_kde": 0.0, "delta_c2": 0.0, "sigma_kde_c2": 0.0},
                  allow_out_of_prior=True)
    return an, cats, settings, {k: float(v) for k, v in fv.items()}


def make_diag(an, n_events, nsamp, n_draw, peb):
    """The bound call's program with return_diagnostics=True; data operands are arguments."""
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
            pe_event_block=peb, angular_model=an.angular_model, angular_params=angular,
            return_diagnostics=True)
        return (d.log_likelihood, jnp.sum(d.event_mc_variance), d.n_eff, d.log_mu,
                d.selection_log_correction, d.event_log_evidence, d.event_mc_variance)
    return diag


@dataclasses.dataclass
class SelCtx:
    """One injection set bound to the analysis: its operands and the compact-view pixel sets."""
    name: str
    store: object
    gw_sel: object
    model_operands: object
    keep: list
    global_sel: list
    nside: list
    n_draw: float


class Engine:
    def __init__(self, ds, an, cats, dry):
        import jax
        from darksirens.runtime_binding import bind_analysis, required_fit_columns
        self.ds, self.an, self.cats, self.dry = ds, an, cats, dry
        self.jax = jax
        self.bind = bind_analysis
        self.required = required_fit_columns(an)
        self.diags = {}
        self.labels = None
        self.n_rebinds = 0

    def selctx(self, name, inj_store, events_store, peb):
        from darksirens.catalog.geometry import ang2pix_ring
        b = self.bind(self.an, events=events_store, injections=inj_store, max_likelihood_variance=MAXVAR,
                      sel_batch_size=SEL_BATCH, pe_event_block=peb)
        if self.labels is None:
            self.labels = list(b.labels)
            if sorted(self.labels) != ["H0", "fcat_2"]:
                sys.exit(f"[fatal] core labels {self.labels}, expected H0 and fcat_2")
        comps = b.model_operands.components
        keep = [np.asarray(c.compact.unique_pixels, dtype=np.int64) for c in comps]
        nside = [int(c.catalog.nside) for c in self.an.redshift.components]
        gsel = [ang2pix_ring(ns, inj_store.columns["ra"], inj_store.columns["dec"]) for ns in nside]
        return b, SelCtx(name, inj_store, b.gw_selection, b.model_operands, keep, gsel, nside, float(b.n_draw))

    def pe_operand(self, store, ctx):
        """The PE operand on ctx's compact views; None if the redraw needs other views."""
        from darksirens.catalog.compact import _sample_to_compact_rows, unique_inference_pixels
        from darksirens.catalog.geometry import ang2pix_ring
        from darksirens.runtime_binding import _make_runtime_event
        rows = []
        for k, ns in enumerate(ctx.nside):
            gp = ang2pix_ring(ns, store.columns["ra"], store.columns["dec"])
            if not np.array_equal(unique_inference_pixels(gp, ctx.global_sel[k]).astype(np.int64), ctx.keep[k]):
                return None
            rows.append(np.asarray(_sample_to_compact_rows(ctx.keep[k].astype(np.int32), gp), dtype=np.int32))
        pix = rows[0] if len(rows) == 1 else np.stack(rows, axis=1)
        return _make_runtime_event(store, pix, self.required)

    def operands(self, store, ctx, peb):
        """(gw_pe, gw_sel, model_operands) for this PE store and injection context."""
        ev = self.pe_operand(store, ctx)
        if ev is not None:
            return ev, ctx.gw_sel, ctx.model_operands, False
        self.n_rebinds += 1
        log(f"  compact views differ for this redraw: full re-bind ({self.n_rebinds})")
        b = self.bind(self.an, events=store, injections=ctx.store, max_likelihood_variance=MAXVAR,
                      sel_batch_size=SEL_BATCH, pe_event_block=peb)
        return b.gw_pe, b.gw_selection, b.model_operands, True

    def diag_fn(self, n_events, nsamp, n_draw, peb):
        key = (n_events, nsamp, n_draw, peb)
        if key not in self.diags:
            self.diags[key] = make_diag(self.an, n_events, nsamp, n_draw, peb)
        return self.diags[key]

    def theta(self, h0, f):
        import jax.numpy as jnp
        v = {"H0": h0, "fcat_2": f}
        return jnp.asarray([v[l] for l in self.labels], dtype=jnp.float64)

    def scan(self, cells, fn, ops, ref_idx):
        out = {k: np.full(len(cells), np.nan) for k in ("lnL", "pe_var", "n_eff", "log_mu", "sel_corr")}
        ref_ev = None
        times = []
        for i, (h, f) in enumerate(cells):
            t0 = time.time()
            r = fn(self.theta(h, f), *ops)
            vals = [float(x) for x in r[:5]]
            times.append(time.time() - t0)
            for k, v in zip(out, vals):
                out[k][i] = v
            if i == ref_idx:
                ref_ev = (np.asarray(r[5]), np.asarray(r[6]))
        out["seconds"] = np.asarray(times)
        return out, ref_ev


# ----------------------------------------------------------------------------- tasks
def cells_for(cfg, dry):
    if cfg == "a":
        hs = GRID_A[::40][:2] if dry else GRID_A
        return [(h, 1.0) for h in hs], {"H0": hs}
    hs, fs = (GRID_B_H0[14:16], GRID_B_F[9:11]) if dry else (GRID_B_H0, GRID_B_F)
    return [(h, f) for h in hs for f in fs], {"H0": hs, "f": fs}


def ref_index(cfg, cells):
    h, f = REF_CELL[cfg]
    d = [abs(c[0] - h) + abs(c[1] - f) for c in cells]
    return int(np.argmin(d))


def summarize(cfg, axes, lnL):
    if cfg == "a":
        return {"H0": P.summarize_1d(axes["H0"], lnL)}
    return P.summarize_2d(axes["H0"], axes["f"], lnL.reshape(len(axes["H0"]), len(axes["f"])))


def write_scan(outdir, tid, cfg, axes, res, ref_ev, meta):
    import h5py
    sd = outdir / "scans"
    sd.mkdir(parents=True, exist_ok=True)
    shape = (len(axes["H0"]),) if cfg == "a" else (len(axes["H0"]), len(axes["f"]))
    with h5py.File(sd / f"{tid}.h5", "w") as f:
        for k, v in axes.items():
            f.create_dataset(f"{k}_grid", data=np.asarray(v))
        for k in ("lnL", "pe_var", "n_eff", "log_mu", "sel_corr", "seconds"):
            f.create_dataset(k, data=np.asarray(res[k]).reshape(shape) if k != "seconds" else res[k])
        if ref_ev is not None:
            g = f.create_group("ref_cell")
            g.create_dataset("event_log_evidence", data=ref_ev[0])
            g.create_dataset("event_mc_variance", data=ref_ev[1])
        f.attrs["meta_json"] = json.dumps(meta)
    lnL = np.asarray(res["lnL"])
    ri = meta["ref_cell_index"]
    summ = {**meta, "summary": summarize(cfg, axes, lnL),
            "summary_bias_corrected": summarize(cfg, axes, lnL + 0.5 * np.asarray(res["pe_var"])
                                                - 0.5 * meta["n_events"] / np.asarray(res["n_eff"])),
            "lnL_max": float(np.nanmax(lnL)), "n_nonfinite": int((~np.isfinite(lnL)).sum()),
            "at_ref_cell": {"pe_var": float(res["pe_var"][ri]), "n_eff": float(res["n_eff"][ri]),
                            "sigma2_sel_N2_over_Neff": float(meta["n_events"] ** 2 / res["n_eff"][ri])},
            "pe_var_range": [float(np.nanmin(res["pe_var"])), float(np.nanmax(res["pe_var"]))],
            "n_eff_range": [float(np.nanmin(res["n_eff"])), float(np.nanmax(res["n_eff"]))],
            "seconds_per_call_median": float(np.median(res["seconds"][1:])) if len(res["seconds"]) > 1
            else float(res["seconds"][0]),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (sd / f"{tid}.json").write_text(json.dumps(summ, indent=1))
    return summ


def task_id(kind, cfg, ns, idx):
    c = LABEL[cfg]
    return {"pe": f"{c}_pe_n{ns}_r{idx:02d}", "inj": f"{c}_inj{idx}_n{ns}_r{INJ_R:02d}"}[kind]


# ----------------------------------------------------------------------------- reproduction
def ref_summary(cfg, cells_axes):
    """The stored grid restricted to this analysis' domain, on the stored cells."""
    import h5py
    with h5py.File(REF[cfg], "r") as f:
        H = np.asarray(f["H0_grid"])
        L = np.asarray(f["log_likelihood"])
        pv = np.asarray(f["guard/pe_variance_sum"])
        ne = np.asarray(f["guard/Neff"])
        F = np.asarray(f["f_grid"]) if "f_grid" in f else None
        gamma = f.attrs.get("arg_gamma", None)
    hs = cells_axes["H0"]
    mh = (H >= hs.min() - 1e-9) & (H <= hs.max() + 1e-9)
    if cfg == "a":
        return {"H0": H[mh], "lnL": L[mh], "pe_var": pv[mh], "n_eff": ne[mh],
                "summary": {"H0": P.summarize_1d(H[mh], L[mh])}, "gamma_attr": str(gamma)}
    fs = cells_axes["f"]
    mf = (F >= fs.min() - 1e-9) & (F <= fs.max() + 1e-9)
    Ls = L[np.ix_(mh, mf)]
    return {"H0": H[mh], "f": F[mf], "lnL": Ls, "pe_var": pv[np.ix_(mh, mf)], "n_eff": ne[np.ix_(mh, mf)],
            "summary": P.summarize_2d(H[mh], F[mf], Ls), "gamma_attr": str(gamma)}


def compare(cfg, axes, res, outdir):
    ref = ref_summary(cfg, axes)
    lnL = np.asarray(res["lnL"])
    pv = np.asarray(res["pe_var"])
    ne = np.asarray(res["n_eff"])
    if cfg == "a":
        hs = axes["H0"]
        idx = [int(np.argmin(np.abs(hs - h))) for h in ref["H0"]]
        assert np.allclose(hs[idx], ref["H0"], atol=1e-9)
        new_on_ref = lnL[idx]
        mine = {"H0": P.summarize_1d(ref["H0"], new_on_ref)}
        fine = {"H0": P.summarize_1d(hs, lnL)}
        d = new_on_ref - ref["lnL"]
        dpv = pv[idx] / ref["pe_var"]
        dne = ne[idx] / ref["n_eff"]
        w = np.exp(ref["lnL"] - ref["lnL"].max())
    else:
        L2 = lnL.reshape(len(axes["H0"]), len(axes["f"]))
        assert np.allclose(axes["H0"], ref["H0"]) and np.allclose(axes["f"], ref["f"])
        mine = P.summarize_2d(axes["H0"], axes["f"], L2)
        fine = mine
        d = (L2 - ref["lnL"]).ravel()
        dpv = (pv.reshape(L2.shape) / ref["pe_var"]).ravel()
        dne = (ne.reshape(L2.shape) / ref["n_eff"]).ravel()
        w = np.exp(ref["lnL"] - ref["lnL"].max()).ravel()
    sel = w > 1e-3
    params = ["H0"] if cfg == "a" else ["H0", "f"]
    rows, ok = {}, True
    for p in params:
        a, b = mine[p], ref["summary"][p]
        r = {"core_median": a["median"], "stored_median": b["median"],
             "median_shift_over_hw68": (a["median"] - b["median"]) / b["hw68"],
             "w68_ratio": a["w68"] / b["w68"], "w90_ratio": a["w90"] / b["w90"],
             "core_ci68": a["ci68"], "stored_ci68": b["ci68"], "core_ci90": a["ci90"], "stored_ci90": b["ci90"]}
        r["pass"] = bool(abs(r["median_shift_over_hw68"]) <= GATE["dmed_hw"]
                         and GATE["wratio_lo"] <= r["w68_ratio"] <= GATE["wratio_hi"]
                         and GATE["wratio_lo"] <= r["w90_ratio"] <= GATE["wratio_hi"])
        ok &= r["pass"]
        rows[p] = r
    out = {"config": cfg, "seed": SEED, "reference": str(REF[cfg]), "reference_code": "legacy darksirens 2b86a2d",
           "reference_gamma_attr": ref["gamma_attr"], "gate": GATE, "pass": bool(ok), "params": rows,
           "core_fine_grid_summary": fine,
           "dlnL_core_minus_stored": {"mean_in_bulk": float(np.mean(d[sel])), "sd_in_bulk": float(np.std(d[sel])),
                                      "min": float(np.min(d)), "max": float(np.max(d)), "n_bulk_cells": int(sel.sum())},
           "pe_var_ratio_core_over_stored_bulk": [float(np.min(dpv[sel])), float(np.max(dpv[sel]))],
           "n_eff_ratio_core_over_stored_bulk": [float(np.min(dne[sel])), float(np.max(dne[sel]))]}
    (outdir / f"repro_{LABEL[cfg]}.json").write_text(json.dumps(out, indent=1))
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(A14 / "results"))
    ap.add_argument("--plan-hours", type=float, default=12.0)
    ap.add_argument("--skip-repro", action="store_true", help="only after a passed reproduction on record")
    ap.add_argument("--configs", default="a,b",
                    help="comma list of configurations to reproduce and run (a = all-AGN, b = mixed)")
    ap.add_argument("--seed-a", type=int, default=103,
                    help="catalog realisation of configuration (a) (b is always seed 100); a run "
                         "holds one realisation, so (a) off seed 100 runs on its own (--configs a)")
    args = ap.parse_args()
    CFGS = tuple(c for c in ("a", "b") if c in args.configs.split(","))
    if not CFGS:
        sys.exit(f"[fatal] --configs {args.configs!r} names no configuration")
    seed = args.seed_a if "a" in CFGS else 100
    if "b" in CFGS and seed != 100:
        sys.exit(f"[fatal] (b) is seed 100 and (a) is seed {seed}: run them as separate jobs "
                 "(--configs b, --configs a)")
    set_seed(seed, CFGS)
    tag = "" if SEED == 100 else f"_{'_'.join(LABEL[c] for c in CFGS)}"
    dry = args.dry_run
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    t_start = float(os.environ.get("A14_JOB_START", time.time()))
    deadline = float(os.environ.get("A14_DEADLINE", t_start + 14.4 * 3600))
    import jax
    import darksirens as ds
    log(f"host {os.uname().nodename} job {os.environ.get('SLURM_JOB_ID')} devices {jax.devices()} "
        f"darksirens {ds.__file__} dry={dry} deadline in {(deadline - time.time()) / 3600:.2f} h")
    t0 = time.time()
    an, cats, settings, popfixed = build_model(ds, dry)
    eng = Engine(ds, an, cats, dry)
    nsl = (2000, 4000) if dry else NS_LIST
    nev_dry = 20

    # stored event sets (and their recorded observations for the redraws)
    obs, attrs, stored = {}, {}, {}
    for cfg in CFGS:
        o, at, nobs, nsamp = M.read_event_set(EVENTS[cfg])
        if dry:
            o = {k: v[:nev_dry] for k, v in o.items()}
            st = M.read_stored_samples(EVENTS[cfg], keys=M.STORE_COLUMNS)
            cols = {k: v[:nev_dry].ravel() for k, v in st.items()}
            stored[cfg] = M.load_store(ds, cols, at, nev_dry, nsamp)
        else:
            stored[cfg] = ds.load_events(str(EVENTS[cfg]))
        obs[cfg], attrs[cfg] = o, at
    inj_rec = ds.load_injections(str(INJ_RECORD))
    b0, ctx_rec = eng.selctx("record", inj_rec, stored[CFGS[-1]], PEB[2000])
    nobs = int(b0.n_events)
    log(f"built + bound in {time.time() - t0:.0f}s; labels {eng.labels}; n_events {nobs}; "
        f"n_draw {ctx_rec.n_draw:g}; compact rows {[k.size for k in ctx_rec.keep]}; settings {settings}")
    common = {"code": f"darksirens-core {CORE}", "darksirens_file": ds.__file__, "catalog_evaluation": settings,
              "population_fixed": popfixed, "max_likelihood_variance": MAXVAR, "sel_batch_size": SEL_BATCH,
              "Om0": 0.3075, "log10n0": -24.0, "dry_run": dry, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "n_events": nobs, "seed": SEED}
    (outdir / f"run_config{tag}.json").write_text(json.dumps({**common, "labels": eng.labels, "configs": list(CFGS),
                                                        "grid_a_H0": [GRID_A[0], GRID_A[-1], 0.05],
                                                        "grid_b_H0": [GRID_B_H0[0], GRID_B_H0[-1], 0.25],
                                                        "grid_b_f": [GRID_B_F[0], GRID_B_F[-1], 0.025],
                                                        "pe_event_block": PEB, "redraw_root": M.ROOT}, indent=1))

    # the diagnostics program must equal the bound call
    fn2 = eng.diag_fn(nobs, 2000, ctx_rec.n_draw, PEB[2000])
    for (h, f) in ((67.75, 0.30), (69.0, 1.0)):
        v_bound = float(b0(eng.theta(h, f)))
        v_diag = float(fn2(eng.theta(h, f), b0.gw_pe, b0.gw_selection, b0.model_operands)[0])
        log(f"diag vs bound at ({h}, {f}): {v_diag:.10f} vs {v_bound:.10f}")
        if not abs(v_diag - v_bound) <= 1e-8 * max(1.0, abs(v_bound)):
            sys.exit(f"[fatal] the diagnostics program differs from the bound call by {v_diag - v_bound:g}")

    # ---- 1. reproduction of the stored A0 / A2 results (stored samples, stored injections)
    for cfg in [c for c in ("b", "a") if c in CFGS]:
        tid = f"repro_{LABEL[cfg]}"
        if (outdir / "scans" / f"{tid}.json").exists() and (outdir / f"{tid}.json").exists():
            rep = json.loads((outdir / f"{tid}.json").read_text())
            log(f"{tid}: on record, pass={rep['pass']}")
        else:
            cells, axes = cells_for(cfg, dry)
            ops = (b0.gw_pe, b0.gw_selection, b0.model_operands) if cfg == CFGS[-1] else \
                eng.operands(stored[cfg], ctx_rec, PEB[2000])[:3]
            ri = ref_index(cfg, cells)
            t1 = time.time()
            res, ref_ev = eng.scan(cells, fn2, ops, ri)
            meta = {**common, "task": tid, "config": cfg, "label": LABEL[cfg], "kind": "repro", "nsamp": 2000, "redraw": None,
                    "injections": "record", "events_file": str(EVENTS[cfg]), "ref_cell_index": ri,
                    "wall_seconds": time.time() - t1}
            summ = write_scan(outdir, tid, cfg, axes, res, ref_ev, meta)
            log(f"{tid}: {len(cells)} cells in {time.time() - t1:.0f}s; summary {json.dumps(summ['summary'])[:400]}")
            if dry:
                continue
            rep = compare(cfg, axes, res, outdir)
            log(f"{tid}: {json.dumps(rep['params'])}  dlnL {rep['dlnL_core_minus_stored']}")
        if not dry and not rep["pass"] and not args.skip_repro:
            log(f"[FATAL] reproduction check FAILED for configuration ({cfg}, seed {SEED}) against {REF[cfg]}: "
                f"{json.dumps(rep['params'])}. Stopping before any redraw; see {outdir}/{tid}.json")
            sys.exit(3)

    # ---- 2. timing at each N_s (mixed set, record injections, redraw 0)
    timing_path = outdir / f"timing{tag}.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    for ns in nsl:
        if str(ns) in timing and not dry:
            continue
        tt = time.time()
        c0 = CFGS[-1]
        rd = M.redraw(obs[c0], ns, M.SET_CODES[SETNAME[c0]], 0, seed=SEED)
        store = M.load_store(ds, rd, attrs[c0], len(obs[c0]["rho"]), ns)
        del rd
        t_draw = time.time() - tt
        ops = eng.operands(store, ctx_rec, PEB[ns])
        t_ops = time.time() - tt - t_draw
        peb = PEB[ns]
        rec = None
        for attempt in (peb, 1):
            try:
                fn = eng.diag_fn(nobs, ns, ctx_rec.n_draw, attempt)
                tc = time.time()
                fn(eng.theta(67.75, 0.30), *ops[:3])[0].block_until_ready()
                t_compile = time.time() - tc
                ts = []
                for (h, f) in ((66.0, 0.2), (68.0, 0.3), (70.0, 0.4), (67.0, 1.0)):
                    tc = time.time()
                    fn(eng.theta(h, f), *ops[:3])[0].block_until_ready()
                    ts.append(time.time() - tc)
                rec = {"pe_event_block": attempt, "seconds_per_call": float(np.median(ts)), "calls": ts,
                       "first_call_seconds": t_compile, "redraw_and_load_seconds": t_draw,
                       "operand_seconds": t_ops, "rebind": bool(ops[3])}
                PEB[ns] = attempt
                break
            except Exception as e:  # noqa: BLE001
                log(f"timing N_s={ns} pe_event_block={attempt} FAILED: {type(e).__name__}: {str(e)[:300]}")
                eng.diags.pop((nobs, ns, ctx_rec.n_draw, attempt), None)
        if rec is None:
            rec = {"failed": True}
        else:
            try:
                rec["device_memory"] = {k: int(v) for k, v in jax.devices()[0].memory_stats().items()
                                        if k in ("peak_bytes_in_use", "bytes_limit")}
            except Exception:  # noqa: BLE001
                pass
        timing[str(ns)] = rec
        log(f"timing N_s={ns}: {json.dumps(rec)}")
        timing_path.write_text(json.dumps(timing, indent=1))
        del store, ops
    feasible = [ns for ns in nsl if not timing.get(str(ns), {}).get("failed")]
    for ns in nsl:
        if "pe_event_block" in timing.get(str(ns), {}):
            PEB[ns] = timing[str(ns)]["pe_event_block"]

    # ---- 3. the plan
    spc = {int(k): v["seconds_per_call"] for k, v in timing.items() if "seconds_per_call" in v}
    ovh = {int(k): v["redraw_and_load_seconds"] + v["operand_seconds"] + 5.0
           for k, v in timing.items() if "seconds_per_call" in v}
    ncell = {c: len(cells_for(c, dry)[0]) for c in ("a", "b")}

    def cost(cfg, ns, kind="pe"):
        return ncell[cfg] * spc.get(ns, 1e9) + ovh.get(ns, 60.0) + (180.0 if kind == "inj" else 0.0)

    counts = {(c, ns): n for c, ns, n in BASE if ns in feasible and c in CFGS}
    if dry:
        counts = {(c, ns): 1 for c in CFGS for ns in feasible}
    budget = args.plan_hours * 3600 - (time.time() - t_start)
    inj_cost = sum(cost(c, INJ_NS, "inj") for c in CFGS) * len(INJ_SETS)

    def total():
        return sum(n * cost(c, ns) for (c, ns), n in counts.items()) + inj_cost

    while total() > budget and not dry:
        red = [(n * cost(c, ns), (c, ns)) for (c, ns), n in counts.items() if n > FLOOR[(c, ns)]]
        if not red:
            break
        counts[max(red)[1]] -= 1
    tasks = []
    for (c, ns, _n) in BASE:
        if (c, ns) in counts:
            tasks += [("pe", c, ns, r) for r in range(counts[(c, ns)])]
    if dry:
        tasks += [("inj", c, nsl[0], 1) for c in CFGS]
    else:
        tasks += [("inj", c, INJ_NS, j) for c in CFGS for j in INJ_SETS]
        if 32000 in feasible:
            for r in range(max(EXTRA_32K.values())):
                for c in CFGS:
                    if counts.get((c, 32000), 0) <= r < EXTRA_32K[c]:
                        tasks.append(("pe", c, 32000, r))
    plan = {"configs": list(CFGS), "seed": SEED, "seconds_per_call": spc, "overhead_seconds_per_redraw": ovh, "cells": ncell,
            "pe_event_block": {str(k): v for k, v in PEB.items()},
            "counts": {f"{c}@{ns}": n for (c, ns), n in counts.items()},
            "budget_hours": budget / 3600, "estimated_hours_base_plus_inj": total() / 3600,
            "injection_sets": list(INJ_SETS), "injection_tasks_at": {"nsamp": INJ_NS, "pe_redraw": INJ_R},
            "extra_32000_up_to": EXTRA_32K,
            "order": [task_id(k, c, ns, i) for k, c, ns, i in tasks],
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    plan_name = "plan.json" if CFGS == ("a", "b") else f"plan_{'_'.join(LABEL[c] for c in CFGS)}.json"
    (outdir / plan_name).write_text(json.dumps(plan, indent=1))
    log(f"plan: {json.dumps({k: v for k, v in plan.items() if k != 'order'})}")
    log(f"plan order ({len(tasks)} tasks): {plan['order']}")

    # ---- 4. run
    inj_ctx = {}
    skipped_inj = []

    def run_task(kind, cfg, ns, idx):
        tid = task_id(kind, cfg, ns, idx)
        if (outdir / "scans" / f"{tid}.json").exists():
            return "done"
        est = cost(cfg, ns, kind)
        if time.time() + est > deadline:
            log(f"{tid}: skipped, estimated {est / 60:.1f} min exceeds the time left")
            return "time"
        cells, axes = cells_for(cfg, dry)
        if kind == "inj":
            if dry:
                ip, mp = INJ_RECORD, INJ_RECORD
            else:
                ip, mp = inj_set_path(idx)
            if not (ip.exists() and mp.exists()):
                log(f"{tid}: injection set {idx} not there yet ({ip}); left for a resubmission")
                return "missing"
            if idx not in inj_ctx:
                for k in list(inj_ctx):
                    del inj_ctx[k]
                tb = time.time()
                inj = ds.load_injections(str(ip))
                _, inj_ctx[idx] = eng.selctx(f"inj_set{idx}", inj, stored[CFGS[-1]], PEB[ns])
                log(f"  bound injection set {idx}: {inj.n_injections:,} detected of {inj.ndraw:g} "
                    f"in {time.time() - tb:.0f}s")
            ctx = inj_ctx[idx]
            r = INJ_R
        else:
            ctx, r = ctx_rec, idx
        t1 = time.time()
        rd = M.redraw(obs[cfg], ns, M.SET_CODES[SETNAME[cfg]], r, seed=SEED)
        store = M.load_store(ds, rd, attrs[cfg], len(obs[cfg]["rho"]), ns)
        del rd
        gw_pe, gw_sel, mo, rebound = eng.operands(store, ctx, PEB[ns])
        fn = eng.diag_fn(nobs, ns, ctx.n_draw, PEB[ns])
        t_prep = time.time() - t1
        ri = ref_index(cfg, cells)
        res, ref_ev = eng.scan(cells, fn, (gw_pe, gw_sel, mo), ri)
        meta = {**common, "task": tid, "config": cfg, "label": LABEL[cfg], "kind": kind, "nsamp": ns, "redraw": r,
                "redraw_seed": [M.ROOT, M.SET_CODES[SETNAME[cfg]], r, "block"] + ([] if SEED == 100 else [SEED]),
                "injections": "record" if kind == "pe" else f"inj_set{idx}",
                "injection_file": str(INJ_RECORD if kind == "pe" else ip), "n_draw": ctx.n_draw,
                "n_detected_injections": int(ctx.store.n_injections), "events_file": str(EVENTS[cfg]),
                "pe_event_block": PEB[ns], "rebound": bool(rebound), "ref_cell": REF_CELL[cfg],
                "ref_cell_index": ri, "prep_seconds": t_prep, "wall_seconds": time.time() - t1}
        summ = write_scan(outdir, tid, cfg, axes, res, ref_ev, meta)
        s = summ["summary"]
        txt = " ".join(f"{p} {s[p]['median']:.4f} hw68 {s[p]['hw68']:.4f}" for p in ("H0", "f") if p in s)
        log(f"{tid}: {txt}  pe_var@ref {summ['at_ref_cell']['pe_var']:.2f}  "
            f"{time.time() - t1:.0f}s ({summ['seconds_per_call_median']:.3f} s/call)")
        del store, gw_pe
        return "ran"

    for kind, cfg, ns, idx in tasks:
        try:
            st = run_task(kind, cfg, ns, idx)
        except Exception as e:  # noqa: BLE001
            log(f"{task_id(kind, cfg, ns, idx)}: ERROR {type(e).__name__}: {e}")
            traceback.print_exc()
            st = "error"
        if st == "missing":
            skipped_inj.append((kind, cfg, ns, idx))
    for kind, cfg, ns, idx in skipped_inj:      # files may have appeared meanwhile
        try:
            run_task(kind, cfg, ns, idx)
        except Exception as e:  # noqa: BLE001
            log(f"{task_id(kind, cfg, ns, idx)}: ERROR {type(e).__name__}: {e}")
    done = sorted(p.stem for p in (outdir / "scans").glob("*.json"))
    log(f"finished: {len(done)} scans on record; re-binds {eng.n_rebinds}; "
        f"{(deadline - time.time()) / 60:.0f} min before the deadline")


if __name__ == "__main__":
    main()
