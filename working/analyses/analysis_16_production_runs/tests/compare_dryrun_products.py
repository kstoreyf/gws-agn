#!/usr/bin/env python
"""Compare the products of the same driver run made by two driver versions (or into two output roots), bit for bit:
every dataset of grid/*.h5 (and its attributes), config.json (run and part) and result.json without the fields that
record time (any key ending in _seconds, the session records), process or the output path, and nested progress/result records the same way.

    python tests/compare_dryrun_products.py ROOT_A ROOT_B RUN [RUN ...] [--out JSON]

Exit 0 when every compared item is identical.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

VOLATILE = {"sessions", "finished_utc", "gpu_hours_total", "seconds_per_call_median", "seconds_per_call_this_session",
            "start_utc", "end_utc", "wall_seconds", "written_utc", "seconds_in_likelihood", "slurm_job_id", "host",
            "cuda_visible_devices", "file"}


def _strip(o, root):
    if isinstance(o, dict):
        return {k: _strip(v, root) for k, v in o.items() if k not in VOLATILE and not k.endswith("_seconds")}
    if isinstance(o, list):
        return [_strip(v, root) for v in o]
    if isinstance(o, str):
        return o.replace(str(root), "<root>")
    return o


def _diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'B' if k not in a else 'A'}")
            else:
                out += _diff(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += _diff(x, y, f"{path}[{i}]")
        return out
    same = (a == b) or (isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b))
    return [] if same else [f"{path}: {a!r} vs {b!r}"]


def compare_run(ra, rb, run):
    import h5py
    A, B = Path(ra) / run, Path(rb) / run
    rows = {}
    files = sorted({p.relative_to(A) for p in A.rglob("*") if p.is_file()})
    for rel in files:
        pa, pb = A / rel, B / rel
        if rel.name in ("sessions.json",) or rel.suffix in (".npz",) and "partial" in rel.name:
            continue
        if not pb.exists():
            rows[str(rel)] = {"identical": False, "why": "missing in B"}
            continue
        if rel.suffix == ".h5":
            bad = []
            with h5py.File(pa, "r") as fa, h5py.File(pb, "r") as fb:
                ka, kb = sorted(fa.keys()), sorted(fb.keys())
                if ka != kb:
                    bad.append(f"datasets {ka} vs {kb}")
                for k in ka:
                    if k in fb and k != "seconds":
                        x, y = fa[k][()], fb[k][()]
                        if x.dtype != y.dtype or x.shape != y.shape or not np.array_equal(x, y, equal_nan=True):
                            bad.append(k)
                if dict(fa.attrs) != dict(fb.attrs) and {k: str(v) for k, v in fa.attrs.items()} != {k: str(v) for k, v in fb.attrs.items()}:
                    bad.append("attrs")
            rows[str(rel)] = {"identical": not bad, "compared": "every dataset except seconds (wall time), attributes",
                              **({"differ": bad} if bad else {})}
        elif rel.suffix == ".json":
            da = _strip(json.loads(pa.read_text()), ra)
            db = _strip(json.loads(pb.read_text()), rb)
            d = _diff(da, db)
            rows[str(rel)] = {"identical": not d, **({"differ": d[:10]} if d else {})}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root_a")
    ap.add_argument("root_b")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = {"root_a": a.root_a, "root_b": a.root_b, "runs": {r: compare_run(a.root_a, a.root_b, r) for r in a.runs}}
    res["identical"] = all(v["identical"] for rr in res["runs"].values() for v in rr.values())
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    sys.exit(0 if res["identical"] else 1)


if __name__ == "__main__":
    main()
