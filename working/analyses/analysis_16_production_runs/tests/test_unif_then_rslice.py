#!/usr/bin/env python
"""CPU checks of the eight-parameter runs that sample uniformly first and with rslice after (18 Nov, stage V8,
run names ..._unif_then_rslice): the continuation of a stopped run from its checkpoint (--init-from with
--switch-sample rslice) and the automatic switch (--auto-switch-sample rslice --auto-switch-cpi C).

    <core venv python> tests/test_unif_then_rslice.py [settings] [cost] [continuation] [two_phase] [auto[=CPI]]
        -> tests/results/unif_then_rslice_checks.json (each named section rewritten, the others kept)

  settings      dynesty's rslice after switch_sampler (no options) against a sampler built the way the driver builds
                the production eight-parameter runs (rslice, nothing passed: dynesty's defaults), on a toy problem and on
                a copy of the stopped realization-100 checkpoint: every setting the proposals read.
  cost          the stopped run's seconds per call, calls per iteration by block and by chunk, where the switch at 60
                would have fired on its trajectory, the total iterations extrapolated from its ln L(X) (the run list's numbers).
  continuation  the realization-100 continuation's own run-list arguments (+ --dry-run, the tests' output root, a
                smaller chunk) on a copy of the real checkpoint, with ln L shifted to its scale (tests/driver_lnl_offset.py):
                first process restores the copy, switches, writes its own checkpoint and runs until a hard kill;
                second process resumes from the part's checkpoint: rslice, production settings, iterations continue.
  two_phase     small finished two-phase run (no catalog, 20 events, 20,000 injections, 2,000 samples, nlive 40): phase 1 unif to 150 iterations,
                phase 2 --init-from its state with --switch-sample rslice to convergence; prod_summary on it (old and new
                summary code): the dead points of both phases are in the results and in the posterior.
  auto          --auto-switch-sample on the same small problem: an uninterrupted reference, a run killed before the
                switch and resumed, a run killed after it and resumed (final products equal bit for bit), exactly one
                switch, rslice after resumption; and its summary.
A16_SCRIPTS selects the scripts directory under test (default scripts/), A16_SCRIPTS_OLD the summary code to compare
with (optional); outputs go to tests/unif_then_rslice_runs/.
"""
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

A16 = Path(__file__).resolve().parent.parent
SCRIPTS = Path(os.environ.get("A16_SCRIPTS", str(A16 / "scripts")))
sys.path.insert(0, str(SCRIPTS))
import prod_common as P  # noqa: E402

ROOT = A16 / "tests" / "unif_then_rslice_runs"
OUT = A16 / "tests" / "results" / "unif_then_rslice_checks.json"
LOGS = A16 / "logs" / "dryrun"
STOPPED = ("host_dependent_populations/both_catalogs/complete/host_dependent/realization_100_unif_boot5", "nested_1")
STAGE4_R100 = ["--mode", "nested", "--analysis", "host_dependent_populations", "--catalogs", "both", "--flux", "complete",
               "--events-kind", "host_dependent", "--realization", "100",
               "--injections-lane", "injections/injections_targeted_peak.h5"]
ENV = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH="", XLA_PYTHON_CLIENT_PREALLOCATE="false", PYTHONUNBUFFERED="1")
SETTING_ATTRS = ("method", "sampling", "enlarge", "bootstrap", "walks", "facc", "slices", "scale", "bound_update_interval",
                 "nlive", "first_bound_update_ncall", "first_bound_update_eff", "queue_size", "bounding", "ncdim", "ndim")


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def driver(args, log, harness=False, env=None):
    exe = [P.CORE_PY, str(A16 / "tests" / "driver_lnl_offset.py")] if harness else [P.CORE_PY, str(SCRIPTS / "prod_driver.py")]
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / log, "w") as fh:
        rc = subprocess.run(exe + [str(a) for a in args], stdout=fh, stderr=subprocess.STDOUT,
                            env=dict(ENV, A16_SCRIPTS=str(SCRIPTS), **(env or {}))).returncode
    return rc, (LOGS / log).read_text()


def state(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)["sampler"]


def settings_of(s):
    d = {k: getattr(s, k) for k in SETTING_ATTRS}
    d["evolve_point"] = f"{s.evolve_point.__module__}.{s.evolve_point.__name__}"
    d["propose_point"] = s.propose_point.__func__.__name__
    d["update_proposal"] = s.update_proposal.__func__.__name__
    d["kwargs"] = {k: v for k, v in s.kwargs.items() if k != "cite"}
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in d.items()}


def fresh_production(ndim, nlive=200, seed=1):
    """A sampler built as prod_driver builds a new eight-parameter part (sample rslice, nothing else passed)."""
    import dynesty
    return dynesty.NestedSampler(lambda x: -0.5 * float(np.sum(((x - 0.5) / 0.1) ** 2)), lambda u: u, ndim, nlive=nlive,
                                 bound="multi", sample="rslice", rstate=np.random.default_rng(seed),
                                 first_update={"min_ncall": 2 * nlive, "min_eff": 100.0})


def compare(a, b, skip=()):
    """Settings that differ (skip: adaptive state, e.g. rslice's slice scale once iterations have run)."""
    return {k: [a.get(k), b.get(k)] for k in sorted(set(a) | set(b)) if k not in skip and a.get(k) != b.get(k)}


# ------------------------------------------------------------------------------------------------ settings
def check_settings():
    import dynesty
    import prod_driver as PD
    from darksirens.inference.dynesty_checkpoint import restore_dynesty_sampler
    out = {}
    ref8 = settings_of(fresh_production(8))
    out["production_rslice_8d"] = ref8
    # toy: the driver's unif sampler (dynesty's unif defaults), past its first bound, switched with no options
    toy = dynesty.NestedSampler(lambda x: -0.5 * float(np.sum(((x - 0.5) / 0.1) ** 2)), lambda u: u, 8, nlive=200,
                                bound="multi", sample="unif", rstate=np.random.default_rng(1),
                                first_update={"min_ncall": 400, "min_eff": 100.0})
    toy.run_nested(maxcall=3000, print_progress=False)
    rec = PD.switch_sampler(toy, "rslice")
    st = settings_of(toy)
    out["toy"] = {"switch_record": rec, "differs_from_production": compare(st, ref8)}
    # the stopped realization-100 checkpoint (a copy), restored and switched with no options
    cp = ROOT / "_init" / "state_r100_unif_boot5.save"
    s = restore_dynesty_sampler(str(cp), lambda x: 0.0, lambda u: u)
    before = settings_of(s)
    rec = PD.switch_sampler(s, "rslice")
    st = settings_of(s)
    out["checkpoint"] = {"before": {k: before[k] for k in ("method", "enlarge", "bootstrap", "slices", "bound_update_interval")},
                         "it": int(s.it), "ncall": int(s.ncall), "switch_record": rec,
                         "differs_from_production": compare(st, ref8)}
    out["pass"] = not out["toy"]["differs_from_production"] and not out["checkpoint"]["differs_from_production"]
    return out


# ------------------------------------------------------------------------------------------------ cost
def check_cost(threshold=60.0, chunk=2000, rslice_cpi=60.0):
    """The stopped realization-100 run's numbers behind the run list's estimates: seconds per call (its sessions.json),
    calls per iteration by blocks of 200 iterations and by the driver's chunks (a chunk ends at the first iteration that
    takes its calls above --chunk-calls), where the automatic switch at `threshold` would have fired on its trajectory,
    and the total number of iterations extrapolated from its ln L(X) as a Gaussian (8 dimensions)."""
    from scipy.optimize import curve_fit
    from scipy.special import gammaln
    s = state(ROOT / "_init" / "state_r100_unif_boot5.save")
    ses = json.loads((P.RUNS / STOPPED[0] / STOPPED[1] / "sessions.json").read_text())
    calls = sum(r["calls"] for r in ses)
    nc = np.asarray(s.saved_run["nc"])
    blocks = [{"iterations": [a + 1, min(a + 200, len(nc))], "calls_per_iteration": round(float(nc[a:a + 200].mean()), 2)}
              for a in range(0, len(nc), 200)]
    chunks, i, ncall = [], 0, int(s.ncall - nc.sum())
    while i < len(nc):
        c, i0 = 0, i
        while i < len(nc) and not c > chunk:
            c += int(nc[i])
            i += 1
        ncall += c
        chunks.append({"first": i0 + 1, "last": i, "calls": c, "calls_per_iteration": round(c / (i - i0), 2),
                       "ncall_after": ncall, "complete": c > chunk})
    first = next((x for x in chunks if x["calls_per_iteration"] > threshold and x["complete"]), None)
    before = [x["calls_per_iteration"] for x in chunks if first and x["last"] < first["first"]]
    L, lv = np.asarray(s.saved_run["logl"]), np.asarray(s.saved_run["logvol"])
    fits = []
    for a in (600, 900, 1200):
        f = lambda x, Lm, lnC: Lm - 0.5 * np.exp(lnC + 2 * x / 8.0)       # noqa: E731
        (Lm, lnC), _ = curve_fit(f, lv[a:], L[a:], p0=(L[-1] + 50, 5.0))
        lzl = 4 * np.log(2) + gammaln(5) - 4 * lnC                         # ln Z - L_max of an 8-D Gaussian
        lx_end = lzl + np.log(np.expm1(0.1))                               # where max L * X / Z = e^0.1 - 1
        n_end = len(L) + (lv[-1] - lx_end) * s.nlive                       # iterations at ln X = -1/nlive each
        fits.append({"from_iteration": a + 1, "L_max": round(float(Lm), 1), "lnZ_minus_Lmax": round(float(lzl), 2),
                     "lnX_end": round(float(lx_end), 2), "iterations_total": int(round(n_end))})
    return {"state": {"iterations": int(s.it - 1), "ncall": int(s.ncall), "lnX": float(lv[-1]),
                      "delta_logz": float(np.logaddexp(0, s.live_logl.max() + lv[-1] - s.saved_run["logz"][-1])),
                      "max_live_lnL": float(s.live_logl.max()), "n_floor_points": int((L < -1e299).sum())},
            "seconds_per_call": {"likelihood": sum(r["seconds_in_likelihood"] for r in ses) / calls,
                                 "wall": sum(r["wall_seconds"] for r in ses) / calls, "calls_recorded": calls,
                                 "source": "sessions.json of the stopped part (its last record, before the stop)"},
            "blocks_of_200": blocks, "chunks": chunks,
            "auto_switch_at_threshold": {"threshold": threshold, "first_chunk_above": first,
                                         "largest_chunk_before": max(before) if before else None},
            "extrapolated_iterations": fits, "rslice_calls_per_iteration": rslice_cpi, "pass": first is not None}


# ------------------------------------------------------------------------------------------------ continuation
def check_continuation(argv_extra=()):
    """The run list's continuation arguments with the checkpoint copy in place of the production path."""
    cp = ROOT / "_init" / "state_r100_unif_boot5.save"
    md5_before = md5(cp)
    root = ROOT / "continuation"
    shutil.rmtree(root, ignore_errors=True)
    args = STAGE4_R100 + ["--variant", "unif_then_rslice", "--init-from", str(cp), "--switch-sample", "rslice",
                          "--nested-index", "1"]
    test = ["--dry-run", "--out-root", root, "--chunk-calls", "40"]
    part = root / P.run_name("host_dependent_populations", "both", "complete", "host_dependent", 100,
                             variant="unif_then_rslice") / "nested_1"
    env = {"A16_LNL_OFFSET_STATE": str(cp)}
    rc1, log1 = driver(args + test + ["--kill-after-calls", "120"], "unif_then_rslice_cont_1.log", harness=True, env=env)
    s1 = state(part / "state.save")
    init1 = json.loads((part / "init_from.json").read_text())
    rc2, log2 = driver(args + test + ["--maxiter-more", "2"], "unif_then_rslice_cont_2.log", harness=True, env=env)
    s2 = state(part / "state.save")
    prog = json.loads((part / "progress.json").read_text())
    init2 = json.loads((part / "init_from.json").read_text())
    ref8 = settings_of(fresh_production(8))
    sess = json.loads((part / "sessions.json").read_text())
    res = {"part_dir": str(part.relative_to(A16)), "init_from": init1,
           "process_1": {"exit": rc1, "killed": rc1 == 137, "state_after_kill": {"it": int(s1.it), "ncall": int(s1.ncall),
                                                                                "method": s1.method},
                         "restored_and_switched": "initial state restored" in log1 and "sampler switched" in log1},
           "process_2": {"exit": rc2, "resumed_from_part_checkpoint": f"resumed {part / 'state.save'}" in log2,
                         "init_path_not_repeated": "initial state restored" not in log2,
                         "state_after": {"it": int(s2.it), "ncall": int(s2.ncall), "method": s2.method},
                         "progress_status": prog.get("status"), "progress_sampler": prog.get("sampler"),
                         "progress_init_from": prog.get("init_from")},
           "sessions": [{k: r.get(k) for k in ("calls", "exit")} for r in sess],
           "init_file_unchanged": md5(cp) == md5_before, "init_from_json_unchanged": init1 == init2,
           "state_differs_from_production_rslice": {"after_kill": compare(settings_of(s1), ref8, skip=("scale",)),
                                                    "after_resume": compare(settings_of(s2), ref8, skip=("scale",)),
                                                    "note": "scale (rslice's adapted slice width) excluded: state"},
           "slice_scale": {"after_kill": float(s1.scale), "after_resume": float(s2.scale)},
           "offset_lines": [ln for ln in (log1 + log2).splitlines() if ln.startswith("[offset]")]}
    res["pass"] = bool(res["process_1"]["killed"] and res["process_1"]["restored_and_switched"]
                       and s1.method == "rslice" and s1.it > init1["it"] and rc2 == 0
                       and res["process_2"]["resumed_from_part_checkpoint"] and res["process_2"]["init_path_not_repeated"]
                       and s2.method == "rslice" and s2.it > s1.it and prog.get("sampler", {}).get("method") == "rslice"
                       and not res["state_differs_from_production_rslice"]["after_kill"]
                       and not res["state_differs_from_production_rslice"]["after_resume"]
                       and res["init_file_unchanged"] and res["init_from_json_unchanged"]
                       and part.name == "nested_1")
    return res


# ------------------------------------------------------------------------------------------------ small runs
SMALL = ["--mode", "nested", "--analysis", "single_population", "--catalogs", "none", "--events-kind", "host_dependent",
         "--realization", "100", "--nsamp", "2000", "--limit-events", "20", "--limit-injections", "20000",
         "--nlive", "40", "--dlogz", "0.5", "--checkpoint-every", "1000000"]
# (the dry run's sizes, without --dry-run: a dry run's result is never "finished", and prod_summary reads finished parts)


def summarize_with(scripts_dir, root, name):
    """prod_summary.summarize(name, root) of the given scripts directory, in a subprocess (-> its JSON)."""
    code = (f"import sys, json; sys.path.insert(0, {str(scripts_dir)!r}); import prod_summary as S; "
            f"o = S.summarize({name!r}, {str(root)!r}); print('JSON' + json.dumps(o, default=float))")
    r = subprocess.run([P.CORE_PY, "-c", code], capture_output=True, text=True, env=ENV)
    line = [x for x in r.stdout.splitlines() if x.startswith("JSON")]
    if not line:
        raise RuntimeError(r.stderr[-2000:])
    return json.loads(line[-1][4:])


def check_two_phase():
    root = ROOT / "two_phase"
    shutil.rmtree(root, ignore_errors=True)
    n1 = "phase1"
    rc1, _ = driver(SMALL + ["--out-root", root, "--run-name", n1, "--sample", "unif", "--chunk-calls", "100",
                             "--maxiter-total", "150"], "unif_then_rslice_two_phase_1.log")
    st1 = root / n1 / "nested_1" / "state.save"
    s1 = state(st1)
    n2 = "phase2"
    rc2, log2 = driver(SMALL + ["--out-root", root, "--run-name", n2, "--chunk-calls", "300", "--init-from", st1,
                                "--switch-sample", "rslice"], "unif_then_rslice_two_phase_2.log")
    rj = json.loads((root / n2 / "nested_1" / "result.json").read_text())
    z = np.load(root / n2 / "nested_1" / "samples.npz")
    k = len(s1.saved_run["logl"])
    first_phase_kept = bool(np.array_equal(z["logl"][:k], np.asarray(s1.saved_run["logl"]))
                            and np.array_equal(z["samples"][:k], np.asarray(s1.saved_run["v"])))
    w = z["weights"]
    new = summarize_with(SCRIPTS, root, n2)
    old_dir = os.environ.get("A16_SCRIPTS_OLD")          # the summary code before 18 Nov's change, if given
    old = summarize_with(Path(old_dir), root, n2) if old_dir else new
    res = {"phase1": {"exit": rc1, "iterations": int(s1.it - 1), "ncall": int(s1.ncall), "method": s1.method},
           "phase2": {"exit": rc2, "status": rj.get("status"), "niter_total": rj.get("niter"),
                      "ncall_total_dead_points": rj.get("ncall_total"), "sampler": rj.get("sampler", {}).get("method"),
                      "init_from": rj.get("init_from")},
           "dead_points": {"phase1_iterations": k, "total": int(len(z["logl"])),
                           "phase1_dead_points_kept_bit_for_bit": first_phase_kept,
                           "posterior_weight_in_phase1_points": float(w[:k].sum()),
                           "samples_it_range": [int(z["samples_it"].min()), int(z["samples_it"].max())]},
           "summary_new": {k_: new.get(k_) for k_ in ("posterior", "ln_evidence", "parts", "calls", "gpu_hours",
                                                       "sampler_switches", "init_from", "calls_including_init_state",
                                                       "gpu_hours_including_init_source")},
           "summary_before_change": {k_: old.get(k_) for k_ in ("posterior", "ln_evidence", "parts", "calls", "gpu_hours")}}
    same_post = new["posterior"] == old["posterior"] and new["ln_evidence"] == old["ln_evidence"]
    res["summary_posterior_and_lnZ_identical_old_new"] = same_post
    res["pass"] = bool(rc1 == 0 and rc2 == 0 and rj.get("finished") is True
                       and rj.get("status") == "converged" and first_phase_kept
                       and rj["niter"] + 40 == len(z["logl"]) and same_post       # iterations + the final live points
                       and new.get("calls_including_init_state") == new["calls"] + int(s1.ncall))
    return res


def chunk_trace(log):
    rows = []
    for ln in log.splitlines():
        if "] it " in ln and "ncall" in ln:
            f = ln.split()
            rows.append((int(f[3]), int(f[5])))
    return rows


def check_auto(cpi):
    root = ROOT / "auto"
    shutil.rmtree(root, ignore_errors=True)
    base = SMALL + ["--sample", "unif", "--chunk-calls", "100", "--auto-switch-sample", "rslice",
                    "--auto-switch-cpi", str(cpi)]
    runs = {}
    rc, log = driver(base + ["--out-root", root / "reference"], "unif_then_rslice_auto_ref.log")
    name = P.run_name("single_population", "none", "complete", "host_dependent", 100)
    ref = root / "reference" / name / "nested_1"
    sw = json.loads((ref / "sampler_switch.json").read_text())
    runs["reference"] = {"exit": rc, "switch": sw, "n_switch_lines": log.count("automatic switch at iteration"),
                         "chunks": chunk_trace(log)}
    n_sw = int(sw["ncall"])
    for tag, kill in (("killed_before", max(60, n_sw // 2)), ("killed_after", n_sw + 150)):
        r1, l1 = driver(base + ["--out-root", root / tag, "--kill-after-calls", str(kill)], f"unif_then_rslice_auto_{tag}_1.log")
        part = root / tag / name / "nested_1"
        s_k = state(part / "state.save")
        r2, l2 = driver(base + ["--out-root", root / tag], f"unif_then_rslice_auto_{tag}_2.log")
        za, zb = np.load(ref / "samples.npz"), np.load(part / "samples.npz")
        same = all(np.array_equal(za[k], zb[k]) for k in za.files)
        ra, rb = json.loads((ref / "result.json").read_text()), json.loads((part / "result.json").read_text())
        swb = json.loads((part / "sampler_switch.json").read_text())
        runs[tag] = {"kill_after_calls": kill, "exit_1": r1, "state_at_kill": {"it": int(s_k.it), "ncall": int(s_k.ncall),
                                                                               "method": s_k.method},
                     "exit_2": r2, "resume_line": [x for x in l2.splitlines() if "automatic switch:" in x],
                     "switch_lines_2": l2.count("automatic switch at iteration"),
                     "final_samples_npz_identical_to_reference": same,
                     "logZ_identical": ra["logZ"] == rb["logZ"], "niter": [ra["niter"], rb["niter"]],
                     "switch_identical": {k: swb.get(k) == sw.get(k) for k in ("iterations_done", "ncall", "chunk")},
                     "result_sampler": rb["sampler"]["method"], "result_auto_switch_iterations_done":
                         (rb.get("sampler_auto_switch") or {}).get("iterations_done")}
    summ = summarize_with(SCRIPTS, root / "reference", name)
    runs["summary"] = {k: summ.get(k) for k in ("parts", "sampler_switches", "calls", "posterior")}
    b, a = runs["killed_before"], runs["killed_after"]
    runs["pass"] = bool(rc == 0 and runs["reference"]["n_switch_lines"] == 1 and sw["from"] == "unif" and sw["to"] == "rslice"
                        and sw["chunk"]["calls_per_iteration"] > cpi
                        and b["state_at_kill"]["method"] == "unif" and a["state_at_kill"]["method"] == "rslice"
                        and b["exit_1"] == 137 and a["exit_1"] == 137 and b["exit_2"] == 0 and a["exit_2"] == 0
                        and b["final_samples_npz_identical_to_reference"] and a["final_samples_npz_identical_to_reference"]
                        and b["logZ_identical"] and a["logZ_identical"] and all(b["switch_identical"].values())
                        and all(a["switch_identical"].values()) and a["switch_lines_2"] == 0
                        and "sampler rslice" in " ".join(a["resume_line"]) and a["result_sampler"] == "rslice"
                        and runs["summary"]["sampler_switches"] and len(runs["summary"]["sampler_switches"]) == 1)
    return runs


def main():
    want = sys.argv[1:] or ["settings", "cost", "continuation", "two_phase", "auto"]
    (ROOT / "_init").mkdir(parents=True, exist_ok=True)
    src = P.RUNS / STOPPED[0] / STOPPED[1] / "state.save"
    cp = ROOT / "_init" / "state_r100_unif_boot5.save"
    if not cp.exists():
        shutil.copy2(src, cp)
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res["what"] = __doc__.split("\n\n")[0]
    res["scripts"] = str(SCRIPTS)
    res["checkpoint_copy"] = {"source": str(src), "copy": str(cp), "md5": md5(cp)}
    t0 = time.time()
    for w in want:
        print(f"== {w}", flush=True)
        if w == "settings":
            res[w] = check_settings()
        elif w == "cost":
            res[w] = check_cost()
        elif w == "continuation":
            res[w] = check_continuation()
        elif w == "two_phase":
            res[w] = check_two_phase()
        elif w.startswith("auto"):
            res["auto"] = check_auto(float(w.split("=")[1]) if "=" in w else 40.0)
        print(json.dumps({"section": w, "pass": res.get(w.split("=")[0], {}).get("pass")}), flush=True)
    res["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    res["pass"] = all(res.get(k, {}).get("pass") for k in ("settings", "cost", "continuation", "two_phase", "auto") if k in res)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"-> {OUT} ({time.time() - t0:.0f} s); pass {res['pass']}")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
