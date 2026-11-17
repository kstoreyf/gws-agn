#!/usr/bin/env python
"""Test (b) collection: interruption/resume verdicts and measured seconds per call per model class.

    JAX_PLATFORMS=cpu <core venv python> tests/collect_nested_tests.py
    -> tests/results/nested_tests.json, tests/results/timing.json (input to make_runlist.py --timing)

Pass criteria per class: the killed session exited after 500 calls (exit 137 recorded), the resumed
session restored a checkpoint (it, ncall > 0 at its start) and ran to >= 700 calls past the first
bound update (400 calls); where an uninterrupted reference exists, the resumed run's dead points
equal the reference's bit for bit over every iteration both did (else the first differing
iteration and the size of the difference are reported).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

A16 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(A16 / "scripts"))
sys.path.insert(0, str(A16 / "tests"))
import prod_common as P  # noqa: E402
import compare_resume as CR  # noqa: E402

NR = Path(os.environ.get("A16_NESTED_RUNS", A16 / "tests" / "nested_runs"))
CASES = {
    "no_catalog": ("single_population/no_catalog/complete/equal_populations/realization_100", True, "spc_none_free"),
    "both_single": ("single_population/both_catalogs/complete/equal_populations/realization_100", True, "spc_free"),
    "host_dependent": ("host_dependent_populations/both_catalogs/complete/host_dependent/realization_100", False,
                       "spc_free_hostdep"),
    "free_densities_m18": ("free_densities/both_catalogs/flux_limit_18/equal_populations/realization_100", False,
                           "spc_flux_free_densities"),
}


def main():
    out, timing = {}, {}
    ok_all = True
    for key, (name, has_ref, tkey) in CASES.items():
        d = NR / name
        rd = d / "nested_1_resumed"
        row = {"run": name}
        if not (rd / "sessions.json").exists():
            row["available"] = False
            out[key] = row
            ok_all = False
            continue
        ses = json.loads((rd / "sessions.json").read_text())
        prog = json.loads((rd / "progress.json").read_text()) if (rd / "progress.json").exists() else {}
        killed = [s for s in ses if str(s.get("exit", "")).startswith("killed_by_test")]
        resumed = [s for s in ses if not str(s.get("exit", "")).startswith("killed_by_test")]
        calls = sum(s.get("calls", 0) for s in ses)
        secs = sum(s.get("seconds_in_likelihood", 0.0) for s in ses)
        st = CR.load_state(rd)
        row.update({"available": True, "sessions": ses, "final_it": st["it"] - 1, "final_ncall": st["ncall"],
                    "killed_sessions": len(killed), "resumed_from": prog.get("start_state_this_session"),
                    "status": prog.get("status"), "sampler": prog.get("sampler"),
                    "seconds_per_call": secs / max(calls, 1), "n_neginf_last_session": prog.get("n_neginf_this_session")})
        ok = (len(killed) == 1 and len(resumed) >= 1 and row["final_ncall"] >= 700
              and (row["resumed_from"] or {}).get("ncall", 0) > 0)
        if has_ref:
            refd = d / "nested_1_ref"
            if (refd / "state.save").exists():
                r = subprocess.run([sys.executable, str(A16 / "tests" / "compare_resume.py"), str(refd), str(rd)],
                                   capture_output=True, text=True)
                cmp_ = json.loads(r.stdout[r.stdout.index("{"):]) if "{" in r.stdout else {"error": r.stderr[-500:]}
                row["vs_uninterrupted"] = cmp_
                ok &= bool(cmp_.get("bitwise_identical_all"))
            else:
                row["vs_uninterrupted"] = {"available": False}
                ok = False
        row["pass"] = bool(ok)
        ok_all &= ok
        timing[tkey] = row["seconds_per_call"]
        out[key] = row
        print(f"{key:20s} pass={row['pass']}  it {row['final_it']}  ncall {row['final_ncall']}  "
              f"{row['seconds_per_call']:.3f} s/call  resumed from {row['resumed_from']}  "
              f"bitwise {row.get('vs_uninterrupted', {}).get('bitwise_identical_all')}")
    res = {"pass": bool(ok_all), "cases": out, "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (A16 / "tests" / "results" / "nested_tests.json").write_text(json.dumps(res, indent=1, default=str))
    timing["note"] = "measured by test b (8,000 samples, real. 100); spc_* in seconds per likelihood call"
    tp = A16 / "tests" / "results" / "timing.json"
    old = json.loads(tp.read_text()) if tp.exists() else {}
    old.update(timing)
    tp.write_text(json.dumps(old, indent=1))
    print(f"PASS={ok_all} -> tests/results/nested_tests.json, tests/results/timing.json")


if __name__ == "__main__":
    main()
