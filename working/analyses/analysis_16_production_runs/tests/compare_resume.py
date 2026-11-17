#!/usr/bin/env python
"""Compare an interrupted-and-resumed nested run with an uninterrupted one (same seed, settings).

    python tests/compare_resume.py <reference nested dir> <resumed nested dir> [--out file.json]

Both directories hold a dynesty state.save. dynesty pickles the full sampler state (live points,
bounds, random generator), so a run resumed from a checkpoint must repeat the uninterrupted run
iteration for iteration when the likelihood is deterministic: the dead-point sequence (unit-cube
positions, ln L, calls per iteration) must be identical over the iterations both have done.
Reports the first differing iteration, the number of bitwise-identical iterations, and, if they
differ, the size of the differences (the "statistical" fallback: a nondeterministic GPU reduction
would show as last-bit differences in ln L that then change the path).
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np


def load_state(d):
    with open(Path(d) / "state.save", "rb") as fh:
        D = pickle.load(fh)
    s = D["sampler"] if isinstance(D, dict) and "sampler" in D else D
    sr = s.saved_run
    return {"u": np.asarray(sr["u"]), "logl": np.asarray(sr["logl"]), "nc": np.asarray(sr["nc"]),
            "it": int(s.it), "ncall": int(s.ncall), "blob": np.asarray(sr["blob"]) if "blob" in sr.D else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("resumed")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r, s = load_state(a.ref), load_state(a.resumed)
    n = min(len(r["logl"]), len(s["logl"]))
    same_u = np.all(r["u"][:n] == s["u"][:n], axis=1)
    same_l = r["logl"][:n] == s["logl"][:n]
    same_nc = r["nc"][:n] == s["nc"][:n]
    same = same_u & same_l & same_nc
    first_diff = int(np.argmin(same)) if not same.all() else None
    out = {"reference": a.ref, "resumed": a.resumed, "iterations_compared": int(n),
           "reference_iterations": int(len(r["logl"])), "resumed_iterations": int(len(s["logl"])),
           "reference_ncall": r["ncall"], "resumed_ncall": s["ncall"],
           "bitwise_identical_all": bool(same.all()), "first_differing_iteration": first_diff,
           "n_identical_prefix": int(n if first_diff is None else first_diff),
           "max_abs_dlogl_over_compared": float(np.max(np.abs(r["logl"][:n] - s["logl"][:n]))) if n else None}
    sessions = Path(a.resumed) / "sessions.json"
    if sessions.exists():
        out["resumed_sessions"] = json.loads(sessions.read_text())
    print(json.dumps({k: v for k, v in out.items() if k != "resumed_sessions"}, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    return 0 if same.all() and n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
