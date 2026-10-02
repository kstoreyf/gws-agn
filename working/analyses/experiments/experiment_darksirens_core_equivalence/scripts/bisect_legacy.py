#!/usr/bin/env python
"""Locate the legacy commits between 0c5b3db and c042527 that change this K = 1 likelihood.

    python bisect_legacy.py          # on a rita GPU node (scripts/run_bisect.sbatch)

Cells: H0 = 60, 67, 75 at log10n0 = -3 (cells.bisect_cells). A commit "differs" from
another when any cell's |dlogL| > 1e-3 (material changes only). Every evaluated commit's logL is kept
(results/bisect/<sha>.json), not just good/bad, because the jump may be spread
over several commits.

Pass 1 walks the FIRST-PARENT merge chain 0c5b3db..c042527 and splits every
interval whose endpoints differ until each change sits on one merge, so it finds
all change points, not only the first. Pass 2 repeats this inside each guilty
merge, on the first-parent history of its second parent. Each commit is checked
out as a detached worktree of the legacy repo and evaluated in its own process
(legacy_k1_grid.py --bisect). A commit that fails to build is recorded and its
neighbour is used instead. Worktrees are removed at the end.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path("/hildafs/projects/phy230014p/magana/src/darksirens")
WT = Path("/hildafs/projects/phy230014p/magana/src/ds-bisect")
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "bisect"
BASE_PY = "/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python"
TOL = 1e-3   # material changes only; sub-milli-nat drifts (~1e-6 seen 948db95..c042527) are ignored
MAX_EVALS = 60
GOOD, END = "0c5b3db", "c042527"
evals = {"n": 0}


def git(*a):
    return subprocess.run(["git", "-C", str(REPO), *a], check=True, capture_output=True,
                          text=True).stdout.strip()


def full(sha):
    return git("rev-parse", sha)


def subject(sha):
    return git("log", "-1", "--format=%h %ad %s", "--date=short", sha)


def value(sha):
    """logL at the three cells (tuple), or None if the commit does not build."""
    sha = full(sha)[:10]
    rec = OUT / f"bisect_{sha}.json"
    if not rec.exists():
        if evals["n"] >= MAX_EVALS:
            raise RuntimeError("evaluation budget exhausted")
        evals["n"] += 1
        wt = WT / f"ds-bisect-{sha}"
        if not wt.exists():
            git("worktree", "add", "--detach", str(wt), sha)
        import os
        env = dict(os.environ, PYTHONPATH=str(wt), XLA_PYTHON_CLIENT_PREALLOCATE="false",
                   JAX_PLATFORMS="cuda", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
        r = subprocess.run([BASE_PY, str(HERE / "legacy_k1_grid.py"), "--arm", f"bisect_{sha}",
                            "--expect", f"ds-bisect-{sha}", "--bisect", "--outdir", str(OUT)],
                           env=env, capture_output=True, text=True)
        (OUT / f"bisect_{sha}.log").write_text(r.stdout[-20000:] + "\n--- stderr ---\n" + r.stderr[-20000:])
        if r.returncode != 0 or not rec.exists():
            rec.write_text(json.dumps({"failed": True, "returncode": r.returncode}))
        print(f"[bisect] eval {evals['n']:2d} {subject(sha)} rc={r.returncode}", flush=True)
    d = json.loads(rec.read_text())
    if d.get("failed"):
        return None
    return tuple(row["logL"] for row in d["rows"])


def differs(a, b):
    return max(abs(x - y) for x, y in zip(a, b)) > TOL


def segment(chain, vals, i, j, found):
    """Split chain[i..j] until every change sits between adjacent commits."""
    if vals[i] is None or vals[j] is None or not differs(vals[i], vals[j]):
        return
    if j == i + 1:
        found.append((i, j))
        return
    m = (i + j) // 2
    for cand in (m, m + 1, m - 1, m + 2, m - 2):
        if i < cand < j:
            vals[cand] = value(chain[cand])
            if vals[cand] is not None:
                m = cand
                break
    else:
        found.append((i, j))                         # nothing in between builds
        return
    segment(chain, vals, i, m, found)
    segment(chain, vals, m, j, found)


def walk(chain, label):
    vals = {0: value(chain[0]), len(chain) - 1: value(chain[-1])}
    found = []
    segment(chain, vals, 0, len(chain) - 1, found)
    rows = []
    for i, j in found:
        a, b = vals[i], vals[j]
        rows.append({"from": subject(chain[i]), "to": subject(chain[j]), "to_sha": full(chain[j]),
                     "dlogL": [y - x for x, y in zip(a, b)], "dlogL_shape_60_minus_75":
                     (b[0] - a[0]) - (b[2] - a[2])})
        print(f"[{label}] change at {subject(chain[j])}: dlogL {[round(y - x, 6) for x, y in zip(a, b)]}",
              flush=True)
    return rows, {chain[k][:10]: v for k, v in vals.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    WT.mkdir(parents=True, exist_ok=True)
    fp = [full(GOOD)] + git("rev-list", "--reverse", "--first-parent", f"{GOOD}..{END}").split()
    print(f"[bisect] first-parent chain: {len(fp)} commits", flush=True)
    out = {"tolerance": TOL, "cells": "H0 = 60, 67, 75 at log10n0 = -3", "pass1": None, "pass2": []}
    try:
        rows, vals1 = walk(fp, "pass1")
        out["pass1"] = {"changes": rows, "values": vals1}
        for r in rows:
            m = r["to_sha"]
            parents = git("rev-list", "--parents", "-n", "1", m).split()[1:]
            if len(parents) < 2:
                continue
            p1, p2 = parents[0], parents[1]
            chain = [p1] + git("rev-list", "--reverse", "--first-parent", f"{p1}..{p2}").split()
            rows2, vals2 = walk(chain, "pass2")
            vm, v2 = value(m), value(p2)
            out["pass2"].append({"merge": subject(m), "branch_commits": len(chain) - 1,
                                 "changes": rows2, "values": vals2,
                                 "merge_equals_second_parent": (vm is not None and v2 is not None
                                                                and not differs(vm, v2))})
    except RuntimeError as e:
        out["stopped"] = str(e)
    out["n_evaluations"] = evals["n"]
    (OUT / "summary.json").write_text(json.dumps(out, indent=2))
    for wt in WT.glob("ds-bisect-*"):
        subprocess.run(["git", "-C", str(REPO), "worktree", "remove", "--force", str(wt)],
                       capture_output=True)
    print(f"[bisect] done: {evals['n']} evaluations; wrote {OUT / 'summary.json'}")


if __name__ == "__main__":
    sys.exit(main())
