#!/usr/bin/env python
"""Analysis 10, pre-generation check: is the registered mass mark admissible?

The Analysis-10 specification plants ONE mass mark on top of the Analysis-8/9
spin mark:

    lambda_peak^AGN = lambda_peak^GAL + Delta_lambda_peak,   Delta = +0.15

where ``lambda_peak`` is the Gaussian-peak mixture fraction of the production
``powerlaw+peak`` model.  The specification requires, BEFORE any generation,

    0 < lambda_peak^fid + 0.15 < 1

and orders a STOP -- reporting the real fiducial -- if it does not hold.  This
script is that check.  It is CPU-only, loads no survey and no likelihood, runs
no sampler, and writes exactly one file:

    ../diagnostics/a10_mass_mark_feasibility.json

What it establishes, in order:

  0. Provenance: the pinned darksirens checkout, its HEAD, a clean worktree, and
     that ``darksirens`` really imports from there.
  1. The twelve ``powerlaw+peak`` parameter slots -- plain name, LaTeX label,
     fiducial, prior bounds -- read off ``model.param_specs`` and
     ``pop_model_prior_parser``, not retyped.
  2. The stick-breaking map v -> w at the fiducial, and its inverse ``_w_to_v``,
     both imported from darksirens.
  3. The composition order, established TWICE and independently: from the
     curated registry entry, and numerically, by integrating the fiducial mass
     density over a peak band and a power-law band at three values of ``v1``.
  4. The admissibility check itself, in both coordinates: the human-readable
     peak fraction and the sampled stick-breaking coordinate ``v1_c2``.
  5. The seed-100 marked mock's own ``true_m1src`` distribution and the
     generator's recorded ``peak_fraction`` (read-only corroboration).
  6. A code-readiness probe (informative only): whether
     ``build_parameter_space`` -- called exactly as the Analysis-8 builder calls
     it -- will emit a per-catalog ``v1_c2`` coordinate at all.

Environment (must hold before this script is run):

    export PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-a8
    export JAX_PLATFORMS=cpu
    export PYTHONDONTWRITEBYTECODE=1
"""
import sys

sys.dont_write_bytecode = True  # never leave a __pycache__ in the Analysis-8 tree

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Pins of record
# --------------------------------------------------------------------------- #
DARKSIRENS_REPO = "/hildafs/projects/phy230014p/magana/src/darksirens-a8"
DARKSIRENS_SHA = "af896cae6f3f3dd1f87dec50046e3a8228f59b39"
DARKSIRENS_BASE = "2b86a2d"
GWS_AGN_REPO = "/hildafs/projects/phy230014p/magana/gws-agn"

A8_SCRIPTS = Path(GWS_AGN_REPO) / (
    "working/analyses/analysis_8_marked_multitracer_H0_fagn/scripts"
)
EVENTS = Path(GWS_AGN_REPO) / "working/data/seed100/events/events_marked_dmu0p10.h5"

POP_MODEL = "powerlaw+peak"
DELTA_LAMBDA_REGISTERED = 0.15

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "diagnostics" / "a10_mass_mark_feasibility.json"

# The two mass bands used to identify which mixture component ``v1`` weights.
PEAK_BAND = (25.0, 45.0)   # brackets the Gaussian peak at mu = 35, sigma = 5
PL_BAND = (5.0, 20.0)      # power-law territory, well below the peak


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", repo, *args], capture_output=True, text=True, check=False
    ).stdout.strip()


def provenance():
    """Assert the pinned checkout, and record what was asserted."""
    import darksirens

    head = _git(DARKSIRENS_REPO, "rev-parse", "HEAD")
    dirty = _git(DARKSIRENS_REPO, "status", "--porcelain")
    ds_file = os.path.realpath(darksirens.__file__)
    ancestor = subprocess.run(
        ["git", "-C", DARKSIRENS_REPO, "merge-base", "--is-ancestor",
         DARKSIRENS_BASE, "HEAD"],
        capture_output=True, check=False,
    ).returncode == 0

    assert head == DARKSIRENS_SHA, f"darksirens HEAD {head} != {DARKSIRENS_SHA}"
    assert dirty == "", f"darksirens worktree dirty:\n{dirty}"
    assert ds_file.startswith(os.path.realpath(DARKSIRENS_REPO) + os.sep), (
        f"darksirens imports from {ds_file}, not from {DARKSIRENS_REPO}"
    )
    assert ancestor, f"pinned base {DARKSIRENS_BASE} is not an ancestor of HEAD"

    rec = {
        "darksirens_repo": DARKSIRENS_REPO,
        "darksirens_head": head,
        "darksirens_head_expected": DARKSIRENS_SHA,
        "darksirens_worktree_clean": True,
        "darksirens_import_path": ds_file,
        "darksirens_base_is_ancestor": {"base": DARKSIRENS_BASE, "ok": ancestor},
        "gws_agn_repo": GWS_AGN_REPO,
        "gws_agn_head": _git(GWS_AGN_REPO, "rev-parse", "HEAD"),
        "gws_agn_dirty": bool(_git(GWS_AGN_REPO, "status", "--porcelain")),
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "jax_platforms": os.environ.get("JAX_PLATFORMS", ""),
        "python": sys.executable,
        "cpu_only": True,
    }
    print("== provenance ==")
    for k in ("darksirens_head", "darksirens_import_path", "gws_agn_head"):
        print(f"  [OK ] {k}: {rec[k]}")
    print(f"  [OK ] darksirens worktree clean; base {DARKSIRENS_BASE} is an ancestor")
    return rec


def param_specs_block():
    """The twelve powerlaw+peak slots: plain name, LaTeX label, fiducial, bounds."""
    from darksirens.gw.populations import get_model, get_fixed_population_params
    from darksirens.gw.populations.registry import pop_model_prior_parser

    kw = dict(shared_beta=True, shared_spin=True, shared_gamma=True)
    model = get_model(POP_MODEL, **kw)
    lows, highs, labels, _kinds, latex_name = pop_model_prior_parser(POP_MODEL, **kw)
    fid = np.asarray(get_fixed_population_params(POP_MODEL, **kw), dtype=float)

    specs = []
    for i, spec in enumerate(model.param_specs):
        specs.append({
            "index": i,
            "name": spec.name,
            "label": labels[i],
            "fiducial": float(fid[i]),
            "low": float(lows[i]),
            "high": float(highs[i]),
        })

    print("\n== powerlaw+peak param_specs (12 slots) ==")
    for s in specs:
        print(f"  {s['index']:2d} {s['name']:<12s} {s['label']:<28s} "
              f"fid={s['fiducial']:<8g} bounds=[{s['low']:g}, {s['high']:g}]")

    return {
        "pop_model": POP_MODEL,
        "latex_name": latex_name,
        "shared_beta": True, "shared_spin": True, "shared_gamma": True,
        "n_slots": len(specs),
        "specs": specs,
        "fiducial_vector": [float(x) for x in fid],
    }, model, fid


def stick_breaking_block(model, fid):
    """v -> w at the fiducial, and the inverse map, both imported from darksirens."""
    import jax.numpy as jnp
    from darksirens.gw.populations.base import _stick_breaking_weights
    from darksirens.gw.populations.grammar import _w_to_v, _stick_breaking_weights_np

    k = model.mixture.k
    n_w = k - 1
    v_fid = [float(x) for x in fid[:n_w]]
    w_fid = [float(x) for x in np.asarray(_stick_breaking_weights(jnp.asarray(v_fid)))]
    v_back = [float(x) for x in _w_to_v(w_fid)]
    roundtrip = float(max(abs(a - b) for a, b in zip(v_fid, v_back)))

    comp_names = [type(c).__name__ for c in model.mixture.mass_components]

    print("\n== stick breaking (base.py::_stick_breaking_weights) ==")
    print(f"  k = {k}, composition order = {comp_names}")
    print(f"  v_fid = {v_fid}  ->  w_fid = {w_fid}")
    print(f"  inverse grammar.py::_w_to_v(w_fid) = {v_back} "
          f"(round-trip max |dv| = {roundtrip:.3e})")

    # what the inverse says about the marks the owner might register
    inverse_table = {}
    for lam in (0.75, 0.90, 0.95, 1.05):
        w = [1.0 - lam, lam]
        inverse_table[f"lambda_peak={lam:g}"] = {
            "w": w,
            "v1": float(_w_to_v(w)[0]),
            "v1_in_bounds": bool(0.0 <= _w_to_v(w)[0] <= 1.0),
        }

    return {
        "source_forward": "darksirens/gw/populations/base.py::_stick_breaking_weights",
        "source_inverse": "darksirens/gw/populations/grammar.py::_w_to_v",
        "k": int(k),
        "n_stick_inputs": int(n_w),
        "composition_order": comp_names,
        "v_fiducial": v_fid,
        "w_fiducial": w_fid,
        "w_fiducial_meaning": dict(zip(comp_names, w_fid)),
        "inverse_of_w_fiducial": v_back,
        "roundtrip_max_abs_dv": roundtrip,
        "numpy_forward_check": [
            float(x) for x in _stick_breaking_weights_np(v_fid)
        ],
        "inverse_at_candidate_peak_fractions": inverse_table,
    }


def composition_order_block(model, fid):
    """Establish that component 0 is the power law and component 1 the Gaussian.

    Two independent lines of evidence:
      (a) the curated registry entry, quoted from source;
      (b) a numerical experiment -- vary v1 and watch which band of m1 gains mass.
    """
    import jax.numpy as jnp
    from darksirens.gw.populations.registry import CURATED

    curated = CURATED[POP_MODEL]
    registry_src = Path(DARKSIRENS_REPO) / "darksirens/gw/populations/registry.py"
    quoted = [
        ln.rstrip("\n")
        for ln in registry_src.read_text().splitlines()
        if "w_PL=0.10" in ln
    ]

    # (b) numerical: p_pop on an m1 grid at fixed (q, z, chieff).  This is a
    # conditional slice, not the m1 marginal, so it is normalised over the grid
    # and only the RELATIVE band weights are used.
    m1 = jnp.linspace(2.0, 100.0, 4001)
    q = 0.9 * jnp.ones_like(m1)
    z = 0.1 * jnp.ones_like(m1)
    chieff = 0.0 * jnp.ones_like(m1)
    grid = np.asarray(m1)

    trials = []
    for v1 in (0.1, 0.25, 0.9):
        theta = np.array(fid, dtype=float)
        theta[0] = v1
        lp = np.asarray(model.log_p_pop(m1, q, z, chieff, jnp.asarray(theta)))
        p = np.nan_to_num(np.exp(lp), nan=0.0, posinf=0.0, neginf=0.0)
        tot = float(np.trapz(p, grid))
        peak = float(np.trapz(np.where(
            (grid >= PEAK_BAND[0]) & (grid <= PEAK_BAND[1]), p, 0.0), grid)) / tot
        pl = float(np.trapz(np.where(
            (grid >= PL_BAND[0]) & (grid <= PL_BAND[1]), p, 0.0), grid)) / tot
        trials.append({"v1": float(v1), "frac_peak_band": peak, "frac_pl_band": pl})
        print(f"  v1={v1:<5g} mass in [{PEAK_BAND[0]:g},{PEAK_BAND[1]:g}] = {peak:.4f}"
              f"   mass in [{PL_BAND[0]:g},{PL_BAND[1]:g}] = {pl:.4f}")

    peak_falls = all(
        trials[i]["frac_peak_band"] > trials[i + 1]["frac_peak_band"]
        for i in range(len(trials) - 1)
    )
    pl_rises = all(
        trials[i]["frac_pl_band"] < trials[i + 1]["frac_pl_band"]
        for i in range(len(trials) - 1)
    )
    assert peak_falls and pl_rises, (
        "v1 does NOT behave as the power-law weight; the composition order "
        "assumed by the Analysis-10 specification is wrong -- STOP and report."
    )

    print("  -> raising v1 moves mass OUT of the peak band and INTO the power-law "
          "band: v1 is the POWER-LAW weight, component 0 = PowerLaw.")

    return {
        "verdict": "component 0 = PowerLaw, component 1 = Gaussian; "
                   "v1 is the POWER-LAW weight and lambda_peak = 1 - v1",
        "evidence_registry": {
            "curated_weights": [float(x) for x in curated.weights],
            "curated_weights_meaning": "leading DESIRED FINAL fractions; the last "
                                       "is implied as 1 - sum",
            "implied_w_full": [float(curated.weights[0]),
                               float(1.0 - sum(curated.weights))],
            "source_line": quoted,
            "source_file": str(registry_src),
        },
        "evidence_numerical": {
            "method": "model.log_p_pop on an m1 grid at fixed q=0.9, z=0.1, "
                      "chieff=0; conditional slice normalised over the grid",
            "m1_grid": {"low": 2.0, "high": 100.0, "n": 4001},
            "peak_band": list(PEAK_BAND),
            "pl_band": list(PL_BAND),
            "trials": trials,
            "peak_band_falls_with_v1": bool(peak_falls),
            "pl_band_rises_with_v1": bool(pl_rises),
        },
    }


def admissibility_block(w_fid):
    """The registered check, in both the human and the sampled coordinate."""
    from darksirens.gw.populations.grammar import _w_to_v

    lam_fid = float(w_fid[1])
    lam_agn = lam_fid + DELTA_LAMBDA_REGISTERED
    ok = bool(0.0 < lam_agn < 1.0)
    v1_c2 = float(_w_to_v([1.0 - lam_agn, lam_agn])[0])

    print("\n== the registered check ==")
    print(f"  lambda_peak^fid            = {lam_fid:.10g}")
    print(f"  Delta_lambda_peak^plant    = {DELTA_LAMBDA_REGISTERED:+.10g}")
    print(f"  lambda_peak^AGN            = {lam_agn:.10g}")
    print(f"  0 < lambda_peak^AGN < 1    = {ok}")
    print(f"  sampled coordinate v1_c2   = 1 - {lam_agn:.10g} = {v1_c2:.10g}"
          f"   (prior bounds [0, 1])")
    if not ok:
        print("  ==> STOP: the registered mark is INADMISSIBLE under the "
              "production parameterisation.")

    return {
        "lambda_peak_definition": "the Gaussian-peak mixture fraction, "
                                  "lambda_peak = w_G = 1 - v1",
        "lambda_peak_fid": lam_fid,
        "delta_lambda_registered": DELTA_LAMBDA_REGISTERED,
        "lambda_peak_agn": lam_agn,
        "check": "0 < lambda_peak_fid + delta < 1",
        "check_passes": ok,
        "offending_value": None if ok else lam_agn,
        "sampled_coordinate": {
            "name": "v1_c2",
            "meaning": "the AGN branch's stick-breaking input = the POWER-LAW "
                       "weight of the AGN branch",
            "required_value": v1_c2,
            "prior_bounds": [0.0, 1.0],
            "in_bounds": bool(0.0 <= v1_c2 <= 1.0),
        },
        "verdict": "ADMISSIBLE" if ok else "INADMISSIBLE -- STOP before generation",
    }


def mock_block():
    """Read-only summary of the seed-100 marked mock's true source masses."""
    import h5py

    with h5py.File(EVENTS, "r") as f:
        m1 = f["true_m1src"][:]
        host = f["host_type"][:]
        attrs = dict(f.attrs)

    meta = json.loads(attrs["metadata_json"])
    pop_rec = meta.get("population", {})

    def summarise(x):
        return {
            "n": int(x.size),
            "median": float(np.median(x)),
            "mean": float(np.mean(x)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "frac_in_peak_band": float(np.mean(
                (x >= PEAK_BAND[0]) & (x <= PEAK_BAND[1]))),
            "frac_below_20": float(np.mean(x < 20.0)),
        }

    out = {
        "path": str(EVENTS),
        "read_only": True,
        "dataset": "true_m1src",
        "peak_band": list(PEAK_BAND),
        "all": summarise(m1),
        "by_host_type": {
            "0_GAL": summarise(m1[host == 0]),
            "1_AGN": summarise(m1[host == 1]),
        },
        "generator_population_record": pop_rec,
        "generator_peak_fraction": float(pop_rec["peak_fraction"]),
        "detection": {
            "detection_rule": str(attrs["detection_rule"]),
            "snr_threshold": float(attrs["snr_threshold"]),
            "note": "rho_opt scales as (Mc_det/30)^(5/6) / dL, so the detection "
                    "rule depends on the CHIRP MASS: a mass mark changes "
                    "detectability branch by branch",
        },
        "dmu_chi_agn": float(attrs["dmu_chi_agn"]),
        "planted_f_agn": float(attrs["planted_f_agn"]),
        "truth_f_agn": float(attrs["truth_f_agn"]),
    }

    print("\n== seed-100 marked mock, true_m1src (read-only) ==")
    a = out["all"]
    print(f"  N = {a['n']}, median = {a['median']:.4f}, "
          f"frac in [25,45] = {a['frac_in_peak_band']:.4f}, "
          f"frac < 20 = {a['frac_below_20']:.4f}")
    for k, v in out["by_host_type"].items():
        print(f"  {k}: n = {v['n']}, frac in [25,45] = {v['frac_in_peak_band']:.4f}, "
              f"median = {v['median']:.4f}")
    print(f"  generator record: peak_fraction = {out['generator_peak_fraction']}")
    return out


def readiness_probe():
    """Informative only: would build_parameter_space emit a per-catalog v1_c2?

    Called exactly as ``a8_likelihood.build`` calls it, with n_catalogs = 2 and
    no data loaded.  This decides nothing about admissibility; it records
    whether the CODE could carry a mass mark if an admissible one is registered.
    """
    sys.path.insert(0, str(A8_SCRIPTS))
    import a8_likelihood as A8
    from darksirens.inference.prior import build_parameter_space

    opts = A8.build_opts([A8.SURVEY_GAL, A8.SURVEY_AGN])
    opts.fix_population = False
    fpv = A8.fixed_parameter_values_for("new")

    results = {}
    for pcp in [("mu_chi_c2",), ("v1_c2", "mu_chi_c2")]:
        key = "+".join(pcp)
        try:
            res = build_parameter_space(
                opts.pop_model, opts.fix_population, opts.fix_cosmology,
                opts.fix_survey, fix_de=opts.fix_de, prior_overrides={},
                fixed_parameter_values=fpv, universe_model=opts.universe_model,
                shared_beta=opts.shared_beta, shared_spin=opts.shared_spin,
                shared_gamma=opts.shared_gamma, sky_model=opts.sky_model,
                mark_model=opts.mark_model, mark_names=opts.mark_names,
                n_catalogs=2, lss_completion_active=[False] * 2,
                use_lss=bool(opts.use_LSS), mark_names_by_catalog=None,
                per_catalog_pop_params=pcp,
            )
            labels = list(res[0])
            results[key] = {"ok": True, "labels": labels, "n_labels": len(labels)}
            print(f"  {key}: OK -> {labels}")
        except Exception as exc:  # informative: record the refusal verbatim
            results[key] = {"ok": False, "error_type": type(exc).__name__,
                            "error": str(exc)}
            print(f"  {key}: REFUSED -> {type(exc).__name__}: {exc}")

    baseline = results.get("mu_chi_c2", {}).get("labels", [])
    extended = results.get("v1_c2+mu_chi_c2", {}).get("labels", [])
    print("\n== code-readiness probe (informative only) ==")
    return {
        "informative_only": True,
        "note": "a PASS here means the resolver would carry a per-catalog mass "
                "weight; it says nothing about whether +0.15 is admissible",
        "n_catalogs": 2,
        "called_as": "a8_likelihood.build (same arguments, no data loaded)",
        "a8_baseline_labels": baseline,
        "a10_extended_labels": extended,
        "new_labels": [x for x in extended if x not in baseline],
        "results": results,
    }


def main():
    print(f"Analysis 10 -- mass-mark feasibility check ({datetime.now(timezone.utc).isoformat()})\n")
    rec = {
        "script": str(Path(__file__).resolve()),
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "pre-registered admissibility check of the Analysis-10 mass "
                   "mark, run BEFORE any generation",
        "registered_mark": {
            "statement": "lambda_peak^AGN = lambda_peak^GAL + Delta_lambda_peak",
            "delta": DELTA_LAMBDA_REGISTERED,
            "spec_condition": "0 < lambda_peak^fid + 0.15 < 1",
            "spec_action_on_failure": "STOP and report the real fiducial value; "
                                      "do not silently alter the registered mark",
        },
    }
    rec["provenance"] = provenance()
    rec["param_specs"], model, fid = param_specs_block()
    rec["stick_breaking"] = stick_breaking_block(model, fid)
    print("\n== composition order, numerically ==")
    rec["composition_order"] = composition_order_block(model, fid)
    rec["admissibility"] = admissibility_block(rec["stick_breaking"]["w_fiducial"])
    rec["mock"] = mock_block()
    rec["readiness_probe"] = readiness_probe()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=False) + "\n")
    print(f"\nwrote {OUT}")

    adm = rec["admissibility"]
    print("\n" + "=" * 72)
    print(f"VERDICT: {adm['verdict']}")
    print(f"  lambda_peak^fid = {adm['lambda_peak_fid']:.10g}; "
          f"{adm['lambda_peak_fid']:.10g} + {DELTA_LAMBDA_REGISTERED:g} = "
          f"{adm['lambda_peak_agn']:.10g}")
    print(f"  sampled coordinate v1_c2 would have to be "
          f"{adm['sampled_coordinate']['required_value']:.10g}, outside [0, 1]")
    print("=" * 72)
    return 0 if adm["check_passes"] else 2


if __name__ == "__main__":
    sys.exit(main())
