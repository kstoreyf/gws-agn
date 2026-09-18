"""Gate C: seed-100 marked recovery, three arms, one marked dataset.

ALL THREE arms read ``seed100/events/events_marked_dmu0p10.h5`` (Gate B's product)
and generate nothing.  The likelihood is Analysis 2's, built through
``a8_likelihood.build`` -- the same code path Gate A validated -- and nothing
about it is reimplemented here.

  ARM J  joint / production.  Both catalogs carry their OWN spatial prior AND
         their own population block.  Free (fcat_2, mu_chi_c2).  2-D grid.

  ARM I  intrinsic only.  DIAGNOSTIC construction, not a physical model: the
         SAME survey file (the GAL complete survey, the denser tracer of the same
         underlying density field) is passed for BOTH catalog slots, so the
         common spatial factor p(z|pix) multiplies straight out of the branch sum
         and every remaining f-dependence comes from the intrinsic populations.
         Free (fcat_2, mu_chi_c2), same grid as Arm J.

  ARM S  spatial only.  mu_chi_c2 == 0 exactly (identical intrinsic populations),
         f inferred from tracer structure alone.  1-D, 101 points, the Analysis-2
         f-scan convention.

``mu_chi_c2`` is the AGN branch's ABSOLUTE spin mean.  It equals the planted
``dmu_chi`` only because the GAL branch's ``mu_chi`` is pinned at the
powerlaw+peak fiducial 0.0; both spellings are carried in every output.

EXACT COMMAND LINES USED FOR THE RECORDED RESULT
------------------------------------------------
    export PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-a8
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
    D=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_8_marked_multitracer_H0_fagn

    $PY $D/scripts/gate_c_three_arms.py --stage timing --arm J
    $PY $D/scripts/gate_c_three_arms.py --stage scan   --arm J
    $PY $D/scripts/gate_c_three_arms.py --stage scan   --arm I
    $PY $D/scripts/gate_c_three_arms.py --stage scan   --arm S
    $PY $D/scripts/gate_c_three_arms.py --stage assemble

Each scan checkpoints after every f row into
``diagnostics/_gate_c_stage_<arm>.json`` and resumes from it, so a crash costs at
most one row.  Do NOT set JAX_PLATFORMS=cpu: this runs on the local H100.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE.parent
DIAG = ANALYSIS_DIR / "diagnostics"
RESULTS = ANALYSIS_DIR / "results"

# --------------------------------------------------------------------------- #
# The registered grid (specification section 8) and the Analysis-2 f convention
# --------------------------------------------------------------------------- #
F_GRID_2D = np.linspace(0.0, 1.0, 41)
MU_GRID_2D = np.linspace(-0.20, 0.25, 61)
F_GRID_1D = np.linspace(0.0, 1.0, 101)

# Two truths, as analyses 0-2 quote them.
TRUTH = {
    "f_agn_planted": 0.30,
    "f_agn_realised": 0.295,
    "dmu_chi_planted": 0.100000,
    "dmu_chi_realised": 0.111924,
    "dmu_chi_realised_err": 0.006815,
    "mu_chi_gal_fixed": 0.0,
    "note": ("mu_chi_c2 is the AGN branch's ABSOLUTE spin mean; dmu_chi = "
             "mu_chi_c2 - mu_chi_gal and the two coincide only because "
             "mu_chi_gal is pinned at the powerlaw+peak fiducial 0.0.  The "
             "realised truth is the detected-set branch-mean difference in true "
             "chi_eff, which already carried +0.011924 before the mark."),
}

ARMS = {
    "J": {"name": "arm_J_joint", "kind": "2d", "surveys": "gal_agn"},
    "I": {"name": "arm_I_intrinsic", "kind": "2d", "surveys": "gal_gal"},
    "S": {"name": "arm_S_spatial", "kind": "1d", "surveys": "gal_agn"},
}


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(repr(o))


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default))
    tmp.replace(path)
    return path


def _stage_path(arm):
    return DIAG / f"_gate_c_stage_{arm}.json"


def _surveys_for(arm):
    sys.path.insert(0, str(HERE))
    import a8_likelihood as a8
    if ARMS[arm]["surveys"] == "gal_gal":
        # THE diagnostic construction: one common spatial prior for both slots.
        return [a8.SURVEY_GAL, a8.SURVEY_GAL]
    return [a8.SURVEY_GAL, a8.SURVEY_AGN]


def _build_arm(arm, verbose=True):
    """Build the marked-data likelihood for one arm (always the NEW shape)."""
    sys.path.insert(0, str(HERE))
    import a8_likelihood as a8
    surveys = _surveys_for(arm)
    cell = a8.build(f"ARM_{arm}", "new", surveys, verbose=verbose,
                    gw_path=a8.GW_PATH_MARKED)
    return a8, cell, surveys


def _fiducial_mu_chi(a8):
    """The pinned GAL-branch spin mean, read from the fiducial vector itself."""
    labels, plain, fid = a8.population_labels_and_fiducial()
    idx = [i for i, p in enumerate(plain) if p == "mu_chi"]
    if len(idx) != 1:
        raise RuntimeError(f"expected exactly one 'mu_chi' slot, got {idx} in {plain}")
    return float(fid[idx[0]]), labels[idx[0]]


def _cell_record(rec):
    """The compact per-cell row kept on disk."""
    g = rec.get("guard", {})
    return {
        "logL": rec["logL"],
        "finite": rec["finite"],
        "Neff": g.get("Neff"),
        "threshold": g.get("threshold"),
        "passes": g.get("passes"),
        "pe_variance_sum": g.get("pe_variance_sum"),
        "sigma2_total": g.get("sigma2_total"),
        "logL_selection": rec.get("logL_selection"),
        "logL_pe": rec.get("logL_pe"),
        "log_mu": rec.get("log_mu"),
        "seconds": rec["seconds"],
        "n_guard_calls": rec.get("n_guard_calls"),
    }


# =========================================================================== #
# Stage "timing": ONE cheap measurement before committing to the full grid
# =========================================================================== #
def stage_timing(args):
    arm = args.arm
    sys.path.insert(0, str(HERE))
    import a8_likelihood as a8
    a8.set_env(guard_record=True)
    import darksirens  # noqa: F401
    import jax
    print("darksirens module file:", darksirens.__file__)
    print("JAX devices:", jax.devices())

    t_build0 = time.time()
    a8, cell, surveys = _build_arm(arm)
    build_s = time.time() - t_build0

    mu_fid, mu_label = _fiducial_mu_chi(a8)
    L = a8.MU_CHI_C2_LABEL
    probe = [(0.295, 0.0), (0.295, 0.10), (0.5, -0.20), (1.0, 0.25), (0.0, 0.25),
             (0.75, 0.20)]
    rows = []
    for f, mu in probe:
        r = cell.evaluate(fcat_2=f, **{L: mu})
        rows.append({"f_agn": f, "mu_chi_c2": mu, **_cell_record(r)})
        print(f"  f={f:<6} mu={mu:+.3f}  logL={r['logL']!r}  "
              f"Neff={rows[-1]['Neff']}  {r['seconds']:.3f}s")
        sys.stdout.flush()

    steady = [r["seconds"] for r in rows[1:]]
    n2d = F_GRID_2D.size * MU_GRID_2D.size
    out = {
        "provenance": a8.provenance(gw_path=a8.GW_PATH_MARKED, survey_paths=surveys),
        "arm": arm,
        "survey_paths": surveys,
        "sampled_labels": cell.labels,
        "per_catalog_pop_params": list(cell.per_catalog_pop_params),
        "n_fixed_parameter_values": len(cell.fixed_parameter_values),
        "mu_chi_fiducial_GAL_branch": mu_fid,
        "mu_chi_fiducial_label": mu_label,
        "nEvents": int(cell.data["nEvents"]),
        "build_seconds": build_s,
        "first_eval_seconds": rows[0]["seconds"],
        "steady_state_median_seconds": float(np.median(steady)),
        "steady_state_min_seconds": float(np.min(steady)),
        "steady_state_max_seconds": float(np.max(steady)),
        "cells": rows,
        "projection": {
            "n_cells_2d": int(n2d),
            "n_cells_1d": int(F_GRID_1D.size),
            "hours_per_2d_arm": float(n2d * np.median(steady) / 3600.0),
            "hours_arm_S": float(F_GRID_1D.size * np.median(steady) / 3600.0),
            "hours_total_all_three_arms": float(
                (2 * n2d + F_GRID_1D.size) * np.median(steady) / 3600.0),
        },
    }
    if mu_fid != 0.0:
        out["WARNING"] = (f"mu_chi fiducial is {mu_fid}, NOT 0.0: mu_chi_c2 is then "
                          f"NOT equal to dmu_chi and every truth comparison must "
                          f"subtract {mu_fid}.")
    _write(DIAG / f"_gate_c_stage_timing_{arm}.json", out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("cells", "provenance")},
                     indent=2, default=_json_default))


# =========================================================================== #
# Stage "scan": the grids
# =========================================================================== #
def stage_scan(args):
    arm = args.arm
    kind = ARMS[arm]["kind"]
    sys.path.insert(0, str(HERE))
    import a8_likelihood as a8
    a8.set_env(guard_record=True)
    import darksirens  # noqa: F401
    import jax
    print("darksirens module file:", darksirens.__file__)
    print("JAX devices:", jax.devices())

    f_grid = F_GRID_2D if kind == "2d" else F_GRID_1D
    mu_grid = MU_GRID_2D if kind == "2d" else np.array([0.0])

    path = _stage_path(arm)
    state = None
    if path.exists() and not args.restart:
        state = json.loads(path.read_text())
        if (state.get("f_grid") != f_grid.tolist()
                or state.get("mu_grid") != mu_grid.tolist()):
            raise RuntimeError(f"{path} holds a DIFFERENT grid; use --restart")
        print(f"resuming from {path}: {len(state['rows'])}/{f_grid.size} rows done")

    t_build0 = time.time()
    a8, cell, surveys = _build_arm(arm)
    build_s = time.time() - t_build0
    mu_fid, _lab = _fiducial_mu_chi(a8)
    L = a8.MU_CHI_C2_LABEL

    if state is None:
        state = {
            "arm": arm,
            "arm_name": ARMS[arm]["name"],
            "kind": kind,
            "provenance": a8.provenance(gw_path=a8.GW_PATH_MARKED,
                                        survey_paths=surveys),
            "survey_paths": surveys,
            "survey_construction": (
                "DIAGNOSTIC: the SAME GAL complete survey is passed for both "
                "catalog slots, so both branches share one spatial prior that is "
                "independent of the sampled f_AGN.  Not a physical model."
                if ARMS[arm]["surveys"] == "gal_gal" else
                "GAL complete survey in slot 1, AGN complete survey in slot 2."),
            "sampled_labels": cell.labels,
            "per_catalog_pop_params": list(cell.per_catalog_pop_params),
            "fixed_parameter_values": cell.fixed_parameter_values,
            "base_coord": {k: float(v) for k, v in zip(cell.labels, cell.base)},
            "mu_chi_fiducial_GAL_branch": mu_fid,
            "nEvents": int(cell.data["nEvents"]),
            "nsamp": int(cell.data["nsamp"]),
            "Ndraw": float(cell.data["Ndraw"]),
            "truth": dict(TRUTH),
            "f_grid": f_grid.tolist(),
            "mu_grid": mu_grid.tolist(),
            "jax_devices": str(jax.devices()),
            "build_seconds": build_s,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "rows": {},
        }

    t0 = time.time()
    n_done_at_start = len(state["rows"])
    for i, f in enumerate(f_grid):
        if str(i) in state["rows"]:
            continue
        row = []
        for mu in mu_grid:
            r = cell.evaluate(fcat_2=float(f), **{L: float(mu)})
            row.append(_cell_record(r))
        state["rows"][str(i)] = row
        state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _write(path, state)
        el = time.time() - t0
        done = len(state["rows"]) - n_done_at_start
        left = f_grid.size - len(state["rows"])
        lls = np.array([c["logL"] for c in row], dtype=float)
        nrej = int(np.sum(~np.array([c["finite"] for c in row])))
        print(f"[{arm}] row {i+1}/{f_grid.size} f={f:.4f} "
              f"max logL={np.nanmax(lls[np.isfinite(lls)]) if np.isfinite(lls).any() else float('nan'):.4f} "
              f"rejected={nrej}/{mu_grid.size} "
              f"elapsed={el/60:.1f}min eta={(el/max(done,1))*left/60:.1f}min")
        sys.stdout.flush()

    print(f"[{arm}] scan complete -> {path}")


# =========================================================================== #
# Stage "assemble"
# =========================================================================== #
def _grids(state):
    f = np.array(state["f_grid"], dtype=float)
    mu = np.array(state["mu_grid"], dtype=float)
    shape = (f.size, mu.size)
    def pull(key, dtype=float, default=np.nan):
        out = np.full(shape, default, dtype=dtype)
        for i in range(f.size):
            row = state["rows"][str(i)]
            for j in range(mu.size):
                v = row[j].get(key)
                out[i, j] = (default if v is None else v)
        return out
    ll = pull("logL")
    fin = np.zeros(shape, dtype=bool)
    for i in range(f.size):
        for j in range(mu.size):
            fin[i, j] = bool(state["rows"][str(i)][j]["finite"])
    return f, mu, ll, fin, pull


def _trapz_weights(x):
    x = np.asarray(x, dtype=float)
    if x.size == 1:
        return np.array([1.0])
    w = np.zeros_like(x)
    d = np.diff(x)
    w[0] = 0.5 * d[0]
    w[-1] = 0.5 * d[-1]
    w[1:-1] = 0.5 * (d[:-1] + d[1:])
    return w


def _marginals(f, mu, ll, fin, marginal_ci):
    """Flat-prior marginals; ``marginal_ci`` is Analysis 2's, imported not copied."""
    llm = np.where(fin, ll, -np.inf)
    mx = np.nanmax(llm[np.isfinite(llm)])
    P = np.where(np.isfinite(llm), np.exp(llm - mx), 0.0)
    out = {"logL_max": float(mx)}
    if mu.size > 1:
        m_f = np.trapz(P, mu, axis=1)
        m_mu = np.trapz(P, f, axis=0)
    else:
        m_f = P[:, 0]
        m_mu = None
    with np.errstate(divide="ignore"):
        logp_f = np.log(m_f)
    out["f"] = marginal_ci(f, logp_f)
    out["f"]["marginal_logp"] = logp_f.tolist()
    if m_mu is not None:
        with np.errstate(divide="ignore"):
            logp_mu = np.log(m_mu)
        out["mu_chi_c2"] = marginal_ci(mu, logp_mu)
        out["mu_chi_c2"]["marginal_logp"] = logp_mu.tolist()
        # posterior correlation on the 2-D grid
        W = np.outer(_trapz_weights(f), _trapz_weights(mu)) * P
        Z = W.sum()
        F2, MU2 = np.meshgrid(f, mu, indexing="ij")
        Ef = (W * F2).sum() / Z
        Em = (W * MU2).sum() / Z
        Vf = (W * (F2 - Ef) ** 2).sum() / Z
        Vm = (W * (MU2 - Em) ** 2).sum() / Z
        Cfm = (W * (F2 - Ef) * (MU2 - Em)).sum() / Z
        out["posterior_moments"] = {
            "mean_f": float(Ef), "mean_mu_chi_c2": float(Em),
            "sd_f": float(np.sqrt(Vf)), "sd_mu_chi_c2": float(np.sqrt(Vm)),
            "cov": float(Cfm),
            "correlation": float(Cfm / np.sqrt(Vf * Vm)),
        }
    i, j = np.unravel_index(np.argmax(llm), llm.shape)
    out["map"] = {"f": float(f[i]), "mu_chi_c2": float(mu[j]),
                  "dmu_chi": float(mu[j]), "logL": float(llm[i, j]),
                  "index": [int(i), int(j)]}
    return out


def _truth_flags(block, truths):
    for tag, val in truths.items():
        block[f"truth_{tag}"] = float(val)
        for lev in ("ci68", "ci90"):
            if lev in block:
                lo, hi = block[lev]
                block[f"truth_{tag}_in_{lev}"] = bool(lo <= val <= hi)
        if "median" in block and np.isfinite(block["median"]):
            block[f"offset_from_{tag}"] = float(block["median"] - val)
    return block


def _guard_report(arm, f, mu, ll, fin, pull):
    """Every rejected cell, and a conservative bound on the mass it could carry."""
    Neff = pull("Neff")
    thr = pull("threshold")
    rejected = ~fin
    rep = {
        "arm": arm,
        "n_cells": int(ll.size),
        "n_rejected": int(rejected.sum()),
        "rejected_fraction": float(rejected.mean()),
        "Neff_min": float(np.nanmin(Neff)),
        "Neff_median": float(np.nanmedian(Neff)),
        "Neff_max": float(np.nanmax(Neff)),
        "threshold_min": float(np.nanmin(thr)),
        "threshold_max": float(np.nanmax(thr)),
        "Neff_over_threshold_min": float(np.nanmin(Neff / thr)),
        "rejected_cells": [],
    }
    idx = np.argwhere(rejected)
    for i, j in idx:
        rep["rejected_cells"].append({
            "i_f": int(i), "j_mu": int(j),
            "f_agn": float(f[i]), "mu_chi_c2": float(mu[j]),
            "dmu_chi": float(mu[j]),
            "Neff": float(Neff[i, j]), "threshold": float(thr[i, j]),
            "Neff_over_threshold": float(Neff[i, j] / thr[i, j]),
        })
    llm = np.where(fin, ll, -np.inf)
    mx = np.nanmax(llm[np.isfinite(llm)])
    P = np.where(np.isfinite(llm), np.exp(llm - mx), 0.0)
    W2 = np.outer(_trapz_weights(f), _trapz_weights(mu))
    Z_acc = float((W2 * P).sum())
    rep["posterior_mass_in_rejected_region_as_scanned"] = 0.0
    rep["posterior_mass_note"] = (
        "A rejected cell returns logL = -inf, so on the grid as scanned it carries "
        "exactly zero mass; that is circular.  The bound below instead FILLS every "
        "rejected cell with the largest posterior density found on the accepted "
        "boundary next to the rejected region, which over-states it because logL "
        "falls monotonically away from the peak there (checked).")
    if rep["n_rejected"] == 0:
        rep["upper_bound_rejected_mass_fraction"] = 0.0
        return rep

    # accepted 8-neighbours of rejected cells
    nb = []
    ni, nj = ll.shape
    for i, j in idx:
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                a, b = i + di, j + dj
                if 0 <= a < ni and 0 <= b < nj and fin[a, b]:
                    nb.append(P[a, b])
    P_bound = float(max(nb)) if nb else 0.0
    P_filled = P.copy()
    P_filled[rejected] = P_bound
    Z_tot = float((W2 * P_filled).sum())
    Z_rej = float((W2 * np.where(rejected, P_filled, 0.0)).sum())
    rep["boundary_max_density_relative_to_peak"] = P_bound
    rep["boundary_max_delta_logL_below_peak"] = (
        float(np.log(P_bound)) if P_bound > 0 else float("-inf"))
    rep["upper_bound_rejected_mass_fraction"] = float(Z_rej / Z_tot)
    rep["accepted_mass_normalisation"] = Z_acc

    # the accepted mass that already sits beyond the first rejected mu
    mu_first = float(mu[idx[:, 1].min()])
    beyond = np.zeros_like(P, dtype=bool)
    beyond[:, mu >= mu_first] = True
    rep["first_rejected_mu_chi_c2"] = mu_first
    rep["accepted_mass_fraction_beyond_first_rejected_mu"] = float(
        (W2 * np.where(beyond & fin, P, 0.0)).sum() / Z_acc)

    # monotonicity of logL in mu over the accepted run-up, per rejected column
    mono = []
    for i in sorted(set(int(x) for x in idx[:, 0])):
        col = llm[i]
        acc = np.where(np.isfinite(col))[0]
        acc = acc[acc >= np.argmax(col)]          # only the falling side
        tail = acc[-4:] if acc.size >= 4 else acc
        d = np.diff(col[tail]) if tail.size > 1 else np.array([np.nan])
        mono.append({"i_f": int(i), "f_agn": float(f[i]),
                     "last_accepted_mu": float(mu[acc[-1]]) if acc.size else None,
                     "last_diffs_logL": d.tolist(),
                     "monotonically_falling": bool(np.all(d < 0))})
    rep["falling_towards_the_guard"] = mono
    rep["all_columns_falling"] = bool(all(m["monotonically_falling"] for m in mono))
    return rep


def stage_assemble(args):
    import h5py
    sys.path.insert(0, str(HERE))
    import a8_likelihood as a8
    scan_h0f = a8.import_scan_h0f()          # marginal_ci, NOT reimplemented
    marginal_ci = scan_h0f.marginal_ci

    guard_all = {"gate": "C", "arms": {}}
    summary_all = {}
    for arm in ("J", "I", "S"):
        path = _stage_path(arm)
        if not path.exists():
            print(f"[skip] {arm}: {path} missing")
            continue
        state = json.loads(path.read_text())
        f, mu, ll, fin, pull = _grids(state)
        n_missing = f.size - len(state["rows"])
        if n_missing:
            raise RuntimeError(f"arm {arm}: {n_missing} rows missing; scan is incomplete")

        marg = _marginals(f, mu, ll, fin, marginal_ci)
        _truth_flags(marg["f"], {"planted": TRUTH["f_agn_planted"],
                                 "realised": TRUTH["f_agn_realised"]})
        if "mu_chi_c2" in marg:
            _truth_flags(marg["mu_chi_c2"],
                         {"planted": TRUTH["dmu_chi_planted"],
                          "realised": TRUTH["dmu_chi_realised"]})
        guard = _guard_report(arm, f, mu, ll, fin, pull)
        guard_all["arms"][arm] = guard

        secs = pull("seconds")
        name = ARMS[arm]["name"]
        summ = {
            "arm": arm,
            "arm_name": name,
            "kind": state["kind"],
            "file": f"results/{name}.h5",
            "provenance": state["provenance"],
            "survey_paths": state["survey_paths"],
            "survey_construction": state["survey_construction"],
            "events_file": state["provenance"]["inputs"]["gw_path_used"],
            "selection_file": state["provenance"]["inputs"]["gwselection_path"],
            "sampled_labels": state["sampled_labels"],
            "per_catalog_pop_params": state["per_catalog_pop_params"],
            "base_coord": state["base_coord"],
            "n_fixed_population_parameters_pinned": len(state["fixed_parameter_values"]),
            "H0_fixed": a8.H0_FID,
            "Om0_fixed": a8.OM0_FID,
            "mu_chi_gal_fixed": state["mu_chi_fiducial_GAL_branch"],
            "truth": dict(TRUTH),
            "grid": {"f": [float(f[0]), float(f[-1]), int(f.size)],
                     "mu_chi_c2": ([float(mu[0]), float(mu[-1]), int(mu.size)]
                                   if mu.size > 1 else [0.0, 0.0, 1])},
            "n_cells": int(ll.size),
            "n_rejected": int((~fin).sum()),
            "logL_max": marg["logL_max"],
            "map": marg["map"],
            "f": marg["f"],
            "timing": {"build_seconds": state.get("build_seconds"),
                       "first_eval_seconds": float(secs.flat[0]),
                       "steady_state_median_seconds": float(np.median(secs.flat[1:])),
                       "total_eval_seconds": float(np.nansum(secs))},
            "guard": guard,
        }
        if "mu_chi_c2" in marg:
            summ["mu_chi_c2"] = marg["mu_chi_c2"]
            summ["dmu_chi"] = dict(marg["mu_chi_c2"])
            summ["dmu_chi"]["note"] = (
                "identical to mu_chi_c2 because mu_chi_gal is pinned at 0.0")
            summ["posterior_moments"] = marg["posterior_moments"]
        summary_all[arm] = summ

        h5_path = RESULTS / f"{name}.h5"
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(h5_path, "w") as h5:
            h5.create_dataset("f_grid", data=f)
            if mu.size > 1:
                h5.create_dataset("mu_chi_c2_grid", data=mu)
            h5.create_dataset("log_likelihood",
                              data=ll if mu.size > 1 else ll[:, 0])
            g = h5.create_group("guard")
            for key in ("Neff", "threshold", "pe_variance_sum", "sigma2_total",
                        "logL_selection", "logL_pe", "log_mu", "seconds"):
                arr = pull(key)
                g.create_dataset(key, data=arr if mu.size > 1 else arr[:, 0])
            g.create_dataset("rejected", data=(~fin) if mu.size > 1 else (~fin)[:, 0])
            g.attrs["n_rejected"] = int((~fin).sum())
            h5.attrs["arm"] = arm
            h5.attrs["arm_name"] = name
            h5.attrs["survey_paths"] = json.dumps(state["survey_paths"])
            h5.attrs["survey_construction"] = state["survey_construction"]
            h5.attrs["events_file"] = state["provenance"]["inputs"]["gw_path_used"]
            h5.attrs["selection_file"] = state["provenance"]["inputs"]["gwselection_path"]
            h5.attrs["labels"] = json.dumps(state["sampled_labels"])
            h5.attrs["base_coord"] = json.dumps(state["base_coord"])
            h5.attrs["truth"] = json.dumps(TRUTH)
            h5.attrs["H0_fixed"] = a8.H0_FID
            h5.attrs["Om0_fixed"] = a8.OM0_FID
            h5.attrs["gws_agn_sha"] = state["provenance"]["gws_agn_sha"]
            h5.attrs["darksirens_sha"] = state["provenance"]["darksirens_sha"]
            h5.attrs["settings"] = json.dumps(state["provenance"]["settings"])
        print(f"wrote {h5_path}")
        _write(RESULTS / f"{name}.json", summ)
        print(f"wrote {RESULTS / (name + '.json')}")

    # cross-arm block
    if {"J", "I", "S"} <= set(summary_all):
        guard_all["cross_arm"] = {
            "f_median": {a: summary_all[a]["f"]["median"] for a in ("J", "I", "S")},
            "mu_median": {a: summary_all[a]["mu_chi_c2"]["median"]
                          for a in ("J", "I")},
        }
    guard_all["provenance"] = summary_all[list(summary_all)[0]]["provenance"]
    guard_all["truth"] = dict(TRUTH)
    guard_all["policy"] = (
        "Specification C6: no result may stand behind a guard-rejected cell.  The "
        "registered grid f in [0,1] x 41, mu_chi_c2 in [-0.20, +0.25] x 61 was run "
        "in full; every rejected cell is listed here with its N_eff, and the mass "
        "it could carry is bounded from above by filling it with the largest "
        "posterior density on the accepted boundary beside it.")
    _write(DIAG / "gate_c_guard.json", guard_all)
    print(f"wrote {DIAG / 'gate_c_guard.json'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["timing", "scan", "assemble"], required=True)
    ap.add_argument("--arm", choices=["J", "I", "S"])
    ap.add_argument("--restart", action="store_true",
                    help="discard the checkpoint and rescan from row 0")
    args = ap.parse_args(argv)
    if args.stage in ("timing", "scan") and not args.arm:
        ap.error("--arm is required for --stage timing/scan")
    return {"timing": stage_timing, "scan": stage_scan,
            "assemble": stage_assemble}[args.stage](args)


if __name__ == "__main__":
    main()
