#!/usr/bin/env python
"""Step 3 (b) of the obslike_f0p30_s1 H0-offset diagnosis: the generation attributes of the three event files
(obslike_f0p30_s1, obslike_f0p30_s0, peak_f0p30_s1) and of the two injection files (obslike, peak lanes), per realization:
every HDF5 attribute and every leaf of metadata_json, flattened; reports the keys whose values differ between files of the
same realization (and, per file kind, between realizations). Read-only.
Output: tests/results/obslike_offset_step3_attrs.json
"""
import json
from pathlib import Path

import h5py
import numpy as np

A16 = Path(__file__).resolve().parents[1]
DATA = A16.parents[1] / "data"
REAL = [100, 101, 102, 103, 105]
EV = {"obslike_f0p30_s1": "events/events_marked_dmu0p10_dmuG5_obslike.h5",
      "obslike_f0p30_s0": "events/events_obslike.h5",
      "peak_f0p30_s1": "events/events_marked_dmu0p10_dmuG5_corepop.h5"}
INJ = {"inj_obslike": "injections/injections_targeted_obslike.h5", "inj_peak": "injections/injections_targeted_peak.h5"}
SKIP_LEAF = ("pdet_z_grid",)


def flat(d, pre=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            if k in SKIP_LEAF:
                out[pre + k] = f"<{type(v).__name__} skipped>"
                continue
            out.update(flat(v, pre + k + "."))
    elif isinstance(d, list) and len(d) > 12:
        out[pre[:-1]] = f"<list len {len(d)}: {d[:3]}...>"
    else:
        out[pre[:-1]] = d
    return out


def read(path):
    with h5py.File(path, "r") as f:
        a = {}
        for k, v in f.attrs.items():
            if k == "metadata_json":
                a.update(flat(json.loads(v), "meta."))
            else:
                a["attr." + k] = v.tolist() if isinstance(v, np.ndarray) else (v.item() if hasattr(v, "item") else v)
    return a


def main():
    res = {"events": {}, "injections": {}}
    for kind, files in (("events", EV), ("injections", INJ)):
        per = {}
        for r in REAL:
            A = {lab: read(DATA / f"seed{r}" / fn) for lab, fn in files.items()}
            keys = sorted(set().union(*[set(v) for v in A.values()]))
            diff = {k: {lab: str(A[lab].get(k, "<absent>"))[:200] for lab in A} for k in keys
                    if len({json.dumps(A[lab].get(k, "<absent>"), default=str) for lab in A}) > 1}
            same = {k: str(next(iter(A.values())).get(k))[:200] for k in keys if k not in diff}
            per[str(r)] = {"differ": diff, "identical": same}
        res[kind] = per
        print(f"===== {kind}: keys that differ between files (realization 101)")
        for k, v in per["101"]["differ"].items():
            print(f"  {k}: {v}")
    out = A16 / "tests" / "results" / "obslike_offset_step3_attrs.json"
    out.write_text(json.dumps(res, indent=1, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()
