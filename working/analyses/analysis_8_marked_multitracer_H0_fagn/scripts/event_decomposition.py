"""Specification section 10: the per-event evidence decomposition, joint arm.

WHAT THIS COMPUTES
------------------
At ONE hyperparameter point -- the Arm-J posterior MAP (f = 0.275,
mu_chi_c2 = +0.1075), see ``WHY THE MAP`` below -- each of the 1000 seed-100
events gets

    * its true host label (mock diagnostic ONLY; never an input to anything);
    * log BF_i,spatial      the AGN-over-GAL ratio of the SPATIAL factor;
    * log BF_i,intrinsic    the AGN-over-GAL ratio of the INTRINSIC factor;
    * P_i(AGN)              the combined branch posterior probability.

THE DEFINITION, AND WHY IT IS THE PRODUCTION ONE
------------------------------------------------
The production per-tracer kernel is ``_log_sample_weight_pop_branches`` in
``darksirens/likelihood/core.py``.  Per PE sample s of event i it forms, for
each branch k,

    log f_k  +  log p_k(z_s | pix_{k,s})  +  log p_pop(theta_s | Lambda_k)

sums the branches with ``_mixture_logsumexp``, subtracts the BRANCH-INDEPENDENT
Jacobian and proposal terms, and the event likelihood is the sample average
(``log_evidence_and_mc_variance``).  Write the three pieces as

    A_k(s) = log p_k(z_s | pix_{k,s})                 SPATIAL   (no log f_k)
    B_k(s) = log p_pop(theta_s | Lambda_k)            INTRINSIC
    C(s)   = -log|d(m1src,q,z)/d(m1det,q,dL)| - log prior_wt_s    branch-free

so that the production event likelihood is EXACTLY

    Z_i = (1/n) sum_s exp( logsumexp_k [ log f_k + A_k(s) + B_k(s) ] + C(s) )
        = f_GAL * E_i[GG]  +  f_AGN * E_i[AA]

with the four sample averages, over the SAME samples and the SAME mask,

    E_i[XY] = (1/n) sum_s exp( A_X(s) + B_Y(s) + C(s) ),   X, Y in {G, A}.

E[GG] and E[AA] are the two branches the production kernel actually sums.
E[AG] and E[GA] are the two hybrids -- one factor swapped, the other held --
which is what makes the split exact.  The decomposition is the chain rule
through the AGN-spatial / GAL-intrinsic hybrid:

    log BF_i,spatial   = log E_i[AG] - log E_i[GG]      (spatial swapped first)
    log BF_i,intrinsic = log E_i[AA] - log E_i[AG]      (intrinsic swapped next)
    log BF_i,spatial + log BF_i,intrinsic = log E_i[AA] - log E_i[GG]
                                          = log BF_i,total    EXACTLY

    P_i(AGN) = f_AGN E_i[AA] / (f_GAL E_i[GG] + f_AGN E_i[AA])
             = sigmoid( log(f_AGN/f_GAL) + log BF_i,total ).

The opposite swap order routes through E[GA] instead; the difference between
the two orderings is the single interaction term

    Delta_i = log E[AA] + log E[GG] - log E[AG] - log E[GA],

which is reported per event so the ordering choice is measured, not assumed.
Both orderings give the same sum and the same P_i(AGN).

NOTHING IS REIMPLEMENTED.  The operands (cosmo, surveys, per-catalog EMCatalogs,
per-catalog population vectors, log mixture weights, the GW PE container) are
CAPTURED from the production factory at the seam where it calls
``darksiren_log_likelihood``, and the per-sample arithmetic below calls the same
darksirens functions in the same order as the production kernel.

HOW IT IS VERIFIED (five independent checks, four against results already on
disk, one against a live production call)
-----------------------------------------------------------------------------
Because ``log f_k`` enters the branch additively, the production likelihood
evaluated at a mixture ENDPOINT isolates one of the four E's exactly:

  sum_i log Z_i          == arm_J  guard/logL_pe[11, 41]  (MAP) and a live call
  sum_i log E_i[GG]      == arm_J  guard/logL_pe[ 0,  :]  (f = 0: GAL branch)
  sum_i log E_i[AA]      == arm_J  guard/logL_pe[40, 41]  (f = 1, mu = +0.1075)
  sum_i log E_i[GA]      == arm_I  guard/logL_pe[40, 41]  (Arm I puts the GAL
                                    survey in BOTH slots, so its f = 1 cell is
                                    GAL spatial x AGN intrinsic)
  sum_i log E_i[AG]      == one live production call at (f = 1, mu_chi_c2 = 0)
                                    (that cell is not a grid node)

No scan is recomputed: the four grid numbers are READ from the recorded files.

WHY THE MAP AND NOT THE POSTERIOR MEDIAN
----------------------------------------
The MAP (f = 0.275, mu_chi_c2 = +0.1075) is a node of the registered 41 x 61
grid, so it is a cell the production scan actually evaluated and its logL is on
disk.  The posterior median (0.261653, 0.107386) is an interpolated marginal
summary and lies between nodes, so nothing recorded could check it.  The two
points are 0.3 of a grid step apart in f and 0.06 of a step in mu_chi_c2 --
0.15 sigma and 0.007 sigma of the respective 68% half-widths -- so the choice is
immaterial to the physics and decisive for the verification.

USAGE
-----
    export PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-a8
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
    D=/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/analysis_8_marked_multitracer_H0_fagn
    $PY $D/scripts/event_decomposition.py

Runs on the local H100.  Do NOT set JAX_PLATFORMS=cpu.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE.parent
RESULTS = ANALYSIS_DIR / "results"
sys.path.insert(0, str(HERE))

import a8_likelihood as a8  # noqa: E402

# --------------------------------------------------------------------------- #
# The representative point (Arm J MAP) and the registered grid indices
# --------------------------------------------------------------------------- #
POINT = {
    "kind": "posterior MAP",
    "f_agn": 0.275,
    "mu_chi_c2": 0.10750000000000004,
    "grid_index": [11, 41],
}
TRUTH = {
    "f_agn_planted": 0.30,
    "f_agn_realised": 0.295,
    "dmu_chi_planted": 0.100000,
    "dmu_chi_realised": 0.111924,
    "dmu_chi_realised_err": 0.006815,
    "mu_chi_gal_fixed": 0.0,
}
HOST_LABELS = {0: "GAL", 1: "AGN"}


def _json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"{type(o)} is not JSON serialisable")


# =========================================================================== #
# 1. Capture the production operands
# =========================================================================== #
def capture_operands(cell, coord):
    """Return the exact argument bundle ``darksiren_log_likelihood`` receives.

    The factory's ``_jit_likelihood_body`` is swapped for an EAGER wrapper (same
    table bindings) and ``darksiren_log_likelihood`` for a recorder that returns
    0.0 without doing any work, so one call of the rebuilt closure runs only the
    parameter decode and hands back CONCRETE operands.  Both patches are undone
    before anything else runs; the production likelihood object built outside
    this function is untouched.
    """
    from darksirens.likelihood import factory as F
    from darksirens.utils import cosmology as CO
    from darksirens.redshift.completion import (
        bound_smoothing_operator, smoothing_operator,
    )
    import jax.numpy as jnp

    cap = {}

    def _recorder(*args, **kwargs):
        cap["args"], cap["kwargs"] = args, kwargs
        return jnp.float64(0.0)

    def _eager(body, operands):
        dt, so = CO.distance_table(), smoothing_operator()

        def likelihood(c):
            with CO.bound_distance_table(dt), bound_smoothing_operator(so):
                return body(c, operands)
        likelihood.distance_table = dt
        likelihood.smoothing_operator = so
        return likelihood

    true_dsll, true_jit = F.darksiren_log_likelihood, F._jit_likelihood_body
    F.darksiren_log_likelihood, F._jit_likelihood_body = _recorder, _eager
    try:
        cap_cell = a8.build("CAPTURE", "new", cell.survey_paths, data=cell.data,
                            verbose=False, gw_path=a8.GW_PATH_MARKED)
        cap_cell.likelihood(coord)
    finally:
        F.darksiren_log_likelihood, F._jit_likelihood_body = true_dsll, true_jit
    if "args" not in cap:
        raise RuntimeError("capture seam never fired")
    return cap


# =========================================================================== #
# 2. The per-event evaluator, built from the captured operands
# =========================================================================== #
def build_evaluator(cap):
    """``(start, m) -> dict of per-event log quantities`` for ``m`` events.

    Every line mirrors ``core._log_sample_weight_pop_branches`` and
    ``core._pe_chunk_ldw``; the only addition is that the per-branch spatial and
    intrinsic terms are kept SEPARATE so the four E's can be formed from them.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.scipy.special import logsumexp

    from darksirens.gw.populations import pop_model_parser
    from darksirens.inference.utils import log_jacobian_m1src_q_z_to_m1det_q_dL
    from darksirens.likelihood.selection import log_evidence_and_mc_variance
    from darksirens.redshift.prior import (
        eval_redshift_prior_with_state, prepare_redshift_prior_state,
    )
    from darksirens.utils import cosmology as CO
    from darksirens.utils.cosmology import dL_grid_bounds, z_of_dL

    (cosmo, survey_0, pop_params, gw_pe, em_cat_pe_0, _gw_sel, _em_cat_sel_0,
     nEvents, nsamp, _Ndraw, pop_model, universe_model) = cap["args"]
    kw = cap["kwargs"]

    surveys_all = (survey_0,) + tuple(kw["mixture_surveys"])
    catalogs_pe_all = (em_cat_pe_0,) + tuple(kw["mixture_em_catalogs_pe"])
    pop_params_all = (pop_params,) + tuple(kw["mixture_pop_params"])
    log_w = jnp.asarray(kw["mixture_log_weights"])
    n_cat = int(kw["n_catalogs"])
    csw = kw["catalog_sky_weighting"]
    mark_params_all = tuple(kw["mark_params_all"])
    mark_names_all = tuple(kw["mark_names_all"])
    if universe_model != "dark_sirens" or kw["sky_model"] != "isotropic":
        raise RuntimeError("evaluator pinned to the dark_sirens / isotropic path")
    if n_cat != 2:
        raise RuntimeError(f"section 10 is the K=2 joint arm; got K={n_cat}")

    # core.py: pe_model == universe_model for dark_sirens.
    pe_model = universe_model
    log_p_pop = pop_model_parser(
        pop_model=pop_model, shared_beta=kw["shared_beta"],
        shared_spin=kw["shared_spin"], shared_gamma=kw["shared_gamma"],
    )
    # core.py's `_mark_model_for`: a catalog with no marks runs the plain model.
    states = tuple(
        prepare_redshift_prior_state(
            pe_model, cosmo, surveys_all[k], catalogs_pe_all[k],
            mark_model=(kw["mark_model"] if mark_names_all[k] else "none"),
            mark_params=mark_params_all[k], mark_names=mark_names_all[k],
            materialize_state=kw["materialize_redshift_prior_state"],
            catalog_sky_weighting=csw,
        )
        for k in range(n_cat)
    )

    def _mixture_logsumexp(lps):                      # core.py, verbatim
        stacked = jnp.stack(lps, axis=0)
        finite = jnp.isfinite(stacked)
        safe = jnp.where(finite, stacked, -1e30)
        return jnp.where(jnp.any(finite, axis=0), logsumexp(safe, axis=0), -jnp.inf)

    def _chunk(start, m):
        n = m * nsamp
        sl = lambda arr: lax.dynamic_slice_in_dim(arr, start, n)  # noqa: E731
        m1det, q, dL = sl(gw_pe.m1det), sl(gw_pe.q), sl(gw_pe.dL)
        chieff, pix, prior_wt = sl(gw_pe.chieff), sl(gw_pe.pixels), sl(gw_pe.prior_wt)
        valid = sl(gw_pe.valid) & (prior_wt > 0.0)

        H0_, Om0_, w0_, wa_ = cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa
        dL_lo, dL_hi = dL_grid_bounds(H0_, Om0_, w0_, wa_)
        supported = (dL >= dL_lo) & (dL <= dL_hi)
        dL_c = jnp.clip(dL, dL_lo, dL_hi)
        z = z_of_dL(dL_c, H0_, Om0_, w0_, wa_)
        m1src = m1det / (1.0 + z)

        # A_k: the SPATIAL factor, log p_k(z | pix_k), WITHOUT the log f_k the
        # production kernel adds in `_eval_prior_branches`.
        A = [
            eval_redshift_prior_with_state(
                pe_model, states[k], z, (pix[:, k] if pix.ndim == 2 else pix),
                cosmo, surveys_all[k], catalogs_pe_all[k],
                catalog_sky_weighting=csw,
            )
            for k in range(n_cat)
        ]
        # B_k: the INTRINSIC factor, log p_pop(theta | Lambda_k).
        B = [log_p_pop(m1src, q, z, chieff, pop_params_all[k]) for k in range(n_cat)]
        # C: everything outside the branch sum, subtracted once (core.py).
        C = (-log_jacobian_m1src_q_z_to_m1det_q_dL(z, dL_c, H0_, Om0_, w0_, wa_)
             - jnp.log(prior_wt))

        # The production per-sample weight, reassembled term for term.
        ldw = _mixture_logsumexp([log_w[k] + A[k] + B[k] for k in range(n_cat)]) + C
        ldw = jnp.where(supported & jnp.isfinite(ldw), ldw, -jnp.inf)
        keep = valid & jnp.isfinite(ldw)
        ldw = jnp.where(keep, ldw, -jnp.inf)

        def _rows(x):
            return jnp.where(keep, x, -jnp.inf).reshape(m, nsamp)

        out = {"logZ_prod": jax.vmap(
            lambda r: log_evidence_and_mc_variance(r, nsamp)[0])(_rows(ldw))}
        # The four E's: same samples, same mask, same branch-free factor C.
        for name, (x, y) in {"GG": (0, 0), "AG": (1, 0),
                             "GA": (0, 1), "AA": (1, 1)}.items():
            out["logE_" + name] = jax.vmap(
                lambda r: log_evidence_and_mc_variance(r, nsamp)[0]
            )(_rows(A[x] + B[y] + C))
        out["n_kept"] = keep.reshape(m, nsamp).sum(axis=1)
        return out

    jitted = {}

    def evaluate(start, m):
        if m not in jitted:
            jitted[m] = jax.jit(lambda s: _chunk(s, m))
        return jitted[m](start)

    return evaluate, int(nEvents), int(nsamp), np.asarray(log_w)


# =========================================================================== #
# 3. Driver
# =========================================================================== #
def main():
    import h5py

    a8.set_env(guard_record=True)
    L = a8.MU_CHI_C2_LABEL
    f_map, mu_map = POINT["f_agn"], POINT["mu_chi_c2"]

    t0 = time.time()
    cell = a8.build("DECOMP", "new", [a8.SURVEY_GAL, a8.SURVEY_AGN],
                    verbose=True, gw_path=a8.GW_PATH_MARKED)
    print(f"[decomp] build {time.time() - t0:.1f}s")

    # --- live production evaluations (the only two likelihood calls) -------- #
    rec_map = cell.evaluate(fcat_2=f_map, **{L: mu_map})
    rec_ag = cell.evaluate(fcat_2=1.0, **{L: 0.0})
    print(f"[decomp] production logL(MAP)       = {rec_map['logL']!r}")
    print(f"[decomp] production logL_pe(MAP)    = {rec_map['logL_pe']!r}")
    print(f"[decomp] production logL_pe(f=1,mu=0) = {rec_ag['logL_pe']!r}")

    # --- recorded grid values (read, never recomputed) ---------------------- #
    with h5py.File(RESULTS / "arm_J_joint.h5", "r") as h:
        J_pe = h["guard/logL_pe"][:]
        J_ll = h["log_likelihood"][:]
        f_grid, mu_grid = h["f_grid"][:], h["mu_chi_c2_grid"][:]
    with h5py.File(RESULTS / "arm_I_intrinsic.h5", "r") as h:
        I_pe = h["guard/logL_pe"][:]
    iF, iM = POINT["grid_index"]
    assert abs(f_grid[iF] - f_map) < 1e-12 and abs(mu_grid[iM] - mu_map) < 1e-12

    # --- capture + per-event evaluation ------------------------------------ #
    coord = cell.coord(fcat_2=f_map, **{L: mu_map})
    cap = capture_operands(cell, coord)
    evaluate, nEvents, nsamp, log_w = build_evaluator(cap)
    print(f"[decomp] captured operands; nEvents={nEvents} nsamp={nsamp} "
          f"log_w={log_w}")

    block = int(a8.SETTINGS["pe_event_block"])
    cols = {k: np.empty(nEvents) for k in
            ("logZ_prod", "logE_GG", "logE_AG", "logE_GA", "logE_AA")}
    n_kept = np.empty(nEvents, dtype=np.int64)
    t0 = time.time()
    for s in range(0, nEvents, block):
        m = min(block, nEvents - s)
        out = evaluate(s * nsamp, m)
        for k in cols:
            cols[k][s:s + m] = np.asarray(out[k])
        n_kept[s:s + m] = np.asarray(out["n_kept"])
    print(f"[decomp] per-event pass {time.time() - t0:.1f}s")

    logE = {k[5:]: cols[k] for k in cols if k.startswith("logE_")}
    logZ_prod = cols["logZ_prod"]
    lf_gal, lf_agn = float(log_w[0]), float(log_w[1])
    logZ_recomb = np.logaddexp(lf_gal + logE["GG"], lf_agn + logE["AA"])

    # --- the decomposition -------------------------------------------------- #
    bf_spatial = logE["AG"] - logE["GG"]
    bf_intrinsic = logE["AA"] - logE["AG"]
    bf_total = logE["AA"] - logE["GG"]
    bf_spatial_alt = logE["AA"] - logE["GA"]
    bf_intrinsic_alt = logE["GA"] - logE["GG"]
    interaction = logE["AA"] + logE["GG"] - logE["AG"] - logE["GA"]
    log_odds = (lf_agn - lf_gal) + bf_total
    p_agn = 1.0 / (1.0 + np.exp(-log_odds))

    # --- verification -------------------------------------------------------- #
    def _chk(name, mine, ref, source):
        d = float(mine - ref)
        return {"quantity": name, "sum_over_events": float(mine),
                "production_reference": float(ref), "source": source,
                "abs_difference": abs(d),
                "relative_difference": abs(d) / abs(ref) if ref else None}

    checks = [
        _chk("sum_i log Z_i (joint, MAP)", logZ_prod.sum(), rec_map["logL_pe"],
             "live production call at the MAP, logL - logL_selection"),
        _chk("sum_i log Z_i (joint, MAP)", logZ_prod.sum(), J_pe[iF, iM],
             "results/arm_J_joint.h5 guard/logL_pe[11,41] (recorded)"),
        _chk("sum_i log E_i[GG]", logE["GG"].sum(), J_pe[0, iM],
             "results/arm_J_joint.h5 guard/logL_pe[0,41] (recorded; f=0)"),
        _chk("sum_i log E_i[AA]", logE["AA"].sum(), J_pe[40, iM],
             "results/arm_J_joint.h5 guard/logL_pe[40,41] (recorded; f=1)"),
        _chk("sum_i log E_i[GA]", logE["GA"].sum(), I_pe[40, iM],
             "results/arm_I_intrinsic.h5 guard/logL_pe[40,41] (recorded; "
             "Arm I f=1 is GAL spatial x AGN intrinsic)"),
        _chk("sum_i log E_i[AG]", logE["AG"].sum(), rec_ag["logL_pe"],
             "live production call at (f=1, mu_chi_c2=0), logL - logL_selection"),
    ]
    checks.append({
        "quantity": "logaddexp(log f_G + log E[GG], log f_A + log E[AA]) vs "
                    "the production per-sample branch sum",
        "max_abs_difference_per_event": float(np.max(np.abs(logZ_recomb - logZ_prod))),
        "sum_abs_difference": float(abs(logZ_recomb.sum() - logZ_prod.sum())),
        "source": "internal: the branch sum re-associated event by event",
    })
    checks.append({
        "quantity": "log BF_spatial + log BF_intrinsic - log BF_total",
        "max_abs_difference_per_event":
            float(np.max(np.abs(bf_spatial + bf_intrinsic - bf_total))),
        "source": "internal: the chain-rule split is exact by construction",
    })
    for c in checks[:6]:
        print(f"[check] {c['quantity']:<34s} mine={c['sum_over_events']:.9f} "
              f"ref={c['production_reference']:.9f} "
              f"|d|={c['abs_difference']:.3e}")

    # --- true labels (diagnostic only) --------------------------------------- #
    with h5py.File(a8.GW_PATH_MARKED, "r") as h:
        host_type = h["host_type"][:].astype(int)
        true_z = h["true_z"][:]
        true_chieff = h["true_chieff"][:]
    if host_type.size != nEvents:
        raise RuntimeError("host_type length does not match nEvents")
    is_agn = host_type == 1

    def _sep(x):
        return {
            "median_true_GAL": float(np.median(x[~is_agn])),
            "median_true_AGN": float(np.median(x[is_agn])),
            "median_difference_AGN_minus_GAL":
                float(np.median(x[is_agn]) - np.median(x[~is_agn])),
            "mean_true_GAL": float(np.mean(x[~is_agn])),
            "mean_true_AGN": float(np.mean(x[is_agn])),
        }

    def _auc(score):
        order = np.argsort(score, kind="mergesort")
        ranks = np.empty(score.size, dtype=float)
        ranks[order] = np.arange(1, score.size + 1)
        s = np.sort(score)
        i = 0
        while i < s.size:                                     # average ties
            j = i
            while j + 1 < s.size and s[j + 1] == s[i]:
                j += 1
            if j > i:
                ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
            i = j + 1
        n1, n0 = int(is_agn.sum()), int((~is_agn).sum())
        return float((ranks[is_agn].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    sep = {
        "log_BF_spatial": {**_sep(bf_spatial), "auc": _auc(bf_spatial)},
        "log_BF_intrinsic": {**_sep(bf_intrinsic), "auc": _auc(bf_intrinsic)},
        "log_BF_total": {**_sep(bf_total), "auc": _auc(bf_total)},
        "P_AGN": {**_sep(p_agn), "auc": _auc(p_agn)},
    }
    pred = p_agn > 0.5
    classification = {
        "threshold": 0.5,
        "counts": {
            "true_AGN_called_AGN": int((is_agn & pred).sum()),
            "true_AGN_called_GAL": int((is_agn & ~pred).sum()),
            "true_GAL_called_AGN": int((~is_agn & pred).sum()),
            "true_GAL_called_GAL": int((~is_agn & ~pred).sum()),
        },
        "n_true_AGN": int(is_agn.sum()),
        "n_true_GAL": int((~is_agn).sum()),
        "n_called_AGN": int(pred.sum()),
        "accuracy": float((pred == is_agn).mean()),
        "expected_AGN_count_sum_P": float(p_agn.sum()),
        "realised_AGN_count": int(is_agn.sum()),
        "P_AGN_range": [float(p_agn.min()), float(p_agn.max())],
        "n_with_P_above_0p9": int((p_agn > 0.9).sum()),
        "n_with_P_below_0p1": int((p_agn < 0.1).sum()),
    }
    interaction_summary = {
        "definition": "log E[AA] + log E[GG] - log E[AG] - log E[GA]",
        "median": float(np.median(interaction)),
        "mean": float(np.mean(interaction)),
        "max_abs": float(np.max(np.abs(interaction))),
        "p90_abs": float(np.percentile(np.abs(interaction), 90)),
        "alt_order_log_BF_spatial_median": float(np.median(bf_spatial_alt)),
        "alt_order_log_BF_intrinsic_median": float(np.median(bf_intrinsic_alt)),
        "note": ("the two swap orders differ ONLY by this term; both give the "
                 "same log BF_total and the same P_i(AGN)"),
    }

    # --- write --------------------------------------------------------------- #
    RESULTS.mkdir(exist_ok=True)
    h5_path = RESULTS / "event_decomposition.h5"
    with h5py.File(h5_path, "w") as h:
        h.create_dataset("event_index", data=np.arange(nEvents))
        h.create_dataset("true_host_type", data=host_type)
        h.create_dataset("log_BF_spatial", data=bf_spatial)
        h.create_dataset("log_BF_intrinsic", data=bf_intrinsic)
        h.create_dataset("log_BF_total", data=bf_total)
        h.create_dataset("log_BF_spatial_alt_order", data=bf_spatial_alt)
        h.create_dataset("log_BF_intrinsic_alt_order", data=bf_intrinsic_alt)
        h.create_dataset("interaction", data=interaction)
        h.create_dataset("P_AGN", data=p_agn)
        h.create_dataset("log_Z_event", data=logZ_prod)
        for k, v in logE.items():
            h.create_dataset("logE_" + k, data=v)
        h.create_dataset("n_samples_kept", data=n_kept)
        h.create_dataset("true_z", data=true_z)
        h.create_dataset("true_chieff", data=true_chieff)
        h.attrs["point_kind"] = POINT["kind"]
        h.attrs["f_agn"] = f_map
        h.attrs["mu_chi_c2"] = mu_map
        h.attrs["log_f_gal"] = lf_gal
        h.attrs["log_f_agn"] = lf_agn
        h.attrs["host_type_encoding"] = "0 = GAL, 1 = AGN"
        h.attrs["events_file"] = a8.GW_PATH_MARKED

    summary = {
        "section": "specification section 10 -- per-event evidence decomposition",
        "arm": "J (joint)",
        "point": {
            **POINT,
            "dmu_chi": mu_map,
            "why": ("the MAP is a node of the registered 41 x 61 grid, so the "
                    "production scan evaluated this exact cell and its "
                    "log-likelihood is on disk; the posterior median "
                    "(f = 0.261653, mu_chi_c2 = 0.107386) is an interpolated "
                    "marginal summary lying between nodes, which nothing "
                    "recorded could check.  The two differ by 0.3 grid steps in "
                    "f (0.15 sigma) and 0.06 steps in mu_chi_c2 (0.007 sigma)."),
            "log_f_gal": lf_gal,
            "log_f_agn": lf_agn,
            "production_logL_at_this_point": rec_map["logL"],
            "production_logL_pe_at_this_point": rec_map["logL_pe"],
            "production_logL_selection_at_this_point": rec_map["logL_selection"],
            "recorded_arm_J_logL_at_this_cell": float(J_ll[iF, iM]),
        },
        "truth": TRUTH,
        "definition": {
            "production_kernel":
                "darksirens/likelihood/core.py :: "
                "_log_sample_weight_pop_branches, per PE sample s and branch k: "
                "log f_k + log p_k(z_s|pix_k,s) + log p_pop(theta_s|Lambda_k); "
                "the branches are summed with _mixture_logsumexp and the event "
                "likelihood is the sample average (log_evidence_and_mc_variance)",
            "A_k": "A_k(s) = log p_k(z_s | pix_{k,s})  -- the SPATIAL factor, "
                   "the same term the production kernel builds in "
                   "_eval_prior_branches, minus the log f_k it adds there",
            "B_k": "B_k(s) = log p_pop(theta_s | Lambda_k)  -- the INTRINSIC "
                   "factor, the same log_p_pop call on the same pop_params_all[k]",
            "C": "C(s) = -log|d(m1src,q,z)/d(m1det,q,dL)| - log prior_wt_s  -- "
                 "branch-independent, subtracted once exactly as in production",
            "E": "E_i[XY] = (1/n) sum_s exp(A_X(s) + B_Y(s) + C(s)), X,Y in "
                 "{G,A}, over the SAME samples with the SAME production mask",
            "identity": "Z_i = f_GAL E_i[GG] + f_AGN E_i[AA] is the production "
                        "per-event likelihood",
            "log_BF_spatial": "log E_i[AG] - log E_i[GG]",
            "log_BF_intrinsic": "log E_i[AA] - log E_i[AG]",
            "sum_rule": "log BF_spatial + log BF_intrinsic = log E[AA] - "
                        "log E[GG] = log BF_total, exactly",
            "P_AGN": "f_AGN E[AA] / (f_GAL E[GG] + f_AGN E[AA]) = "
                     "sigmoid(log(f_AGN/f_GAL) + log BF_total)",
            "ordering": "the split is the chain rule through the AGN-spatial / "
                        "GAL-intrinsic hybrid E[AG]; the opposite order routes "
                        "through E[GA] and differs by the interaction term only",
        },
        "verification": checks,
        "separation": sep,
        "classification": classification,
        "ordering_interaction": interaction_summary,
        "provenance": a8.provenance(gw_path=a8.GW_PATH_MARKED,
                                    survey_paths=cell.survey_paths),
        "outputs": {"h5": str(h5_path),
                    "json": str(RESULTS / "event_decomposition.json")},
        "caveats": [
            "The true host label is a mock diagnostic only; it never enters the "
            "likelihood, the decomposition or P_i(AGN).",
            "Seed 100's DETECTED set already separates GAL from AGN in mass "
            "ratio q (KS p = 0.0096; z-stratified Fisher 29.57, the largest of "
            "41 seeds examined, median 7.54).  That is the record's own "
            "property, present before any mark, and the model holds q identical "
            "in both branches, so the branch label is partly identifiable "
            "without the spin mark and log BF_intrinsic here is not a clean "
            "measurement of the spin mark alone.",
            "One realisation.  Seed 100 only.",
            "One hyperparameter point.  These are not marginalised over the "
            "(f_AGN, mu_chi_c2) posterior.",
        ],
    }
    json_path = RESULTS / "event_decomposition.json"
    # results/ is gitignored repo-wide (.gitignore:4) and specification section
    # 11 files the event-level decomposition under Diagnostics, so the same
    # compact record is written there too, byte for byte.
    diag_path = ANALYSIS_DIR / "diagnostics" / "event_decomposition.json"
    summary["outputs"]["diagnostics_json"] = str(diag_path)
    blob = json.dumps(summary, indent=2, default=_json_default)
    json_path.write_text(blob)
    diag_path.parent.mkdir(exist_ok=True)
    diag_path.write_text(blob)
    print(f"[decomp] wrote {json_path}")
    print(f"[decomp] wrote {diag_path}")
    print(f"[decomp] wrote {h5_path}")

    print("\n              median log BF   true-GAL     true-AGN      AUC")
    for k in ("log_BF_spatial", "log_BF_intrinsic", "log_BF_total", "P_AGN"):
        s = sep[k]
        print(f"  {k:<18s} {s['median_true_GAL']:>11.4f} "
              f"{s['median_true_AGN']:>12.4f} {s['auc']:>8.4f}")
    print("  classification at P>0.5:", classification["counts"])
    print("  sum_i P_i(AGN) =", f"{classification['expected_AGN_count_sum_P']:.2f}",
          "vs realised", classification["realised_AGN_count"])


if __name__ == "__main__":
    main()
