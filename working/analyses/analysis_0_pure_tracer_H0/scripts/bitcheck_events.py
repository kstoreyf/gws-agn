#!/usr/bin/env python3
"""Full dataset-level bit-identity check between two events files.

Every HDF5 dataset (recursively) must be byte-identical.  Attributes are compared
too but reported separately: `generated_at_utc` inside metadata_json and any
attribute added by a generator extension are allowed to differ, datasets are not.
"""
import argparse, hashlib, json, sys
import h5py, numpy as np


def walk(f):
    out = {}
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            out[name] = obj
    f.visititems(visit)
    return out


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--out", default=None)
args = ap.parse_args()

with h5py.File(args.a, "r") as fa, h5py.File(args.b, "r") as fb:
    da, db = walk(fa), walk(fb)
    keys = sorted(set(da) | set(db))
    rows, bad = {}, []
    for k in keys:
        if k not in da or k not in db:
            rows[k] = {"present_a": k in da, "present_b": k in db, "identical": False}
            bad.append(k); continue
        A, B = da[k][()], db[k][()]
        same = A.shape == B.shape and A.dtype == B.dtype and \
            digest(A) == digest(B)
        rows[k] = {"shape": list(np.shape(A)), "dtype": str(A.dtype),
                   "sha256": digest(A), "identical": bool(same)}
        if not same:
            bad.append(k)
    atk = sorted(set(fa.attrs) | set(fb.attrs))
    adiff = []
    for k in atk:
        if k == "metadata_json":
            continue
        va = fa.attrs.get(k, "<absent>"); vb = fb.attrs.get(k, "<absent>")
        try:
            eq = bool(np.all(va == vb))
        except Exception:
            eq = repr(va) == repr(vb)
        if not eq:
            adiff.append({"attr": k, "record": str(va), "regen": str(vb)})
    ma = json.loads(fa.attrs["metadata_json"]); mb = json.loads(fb.attrs["metadata_json"])
    mdiff = sorted(set(ma) | set(mb))
    mdiff = [k for k in mdiff if json.dumps(ma.get(k), sort_keys=True, default=str)
             != json.dumps(mb.get(k), sort_keys=True, default=str)]

res = {"a_record": args.a, "b_regen": args.b,
       "n_datasets": len(keys), "datasets": rows,
       "datasets_all_identical": len(bad) == 0,
       "dataset_failures": bad,
       "attr_differences_excluding_metadata_json": adiff,
       "metadata_json_keys_that_differ": mdiff,
       "PASS": len(bad) == 0}
txt = json.dumps(res, indent=2)
if args.out:
    open(args.out, "w").write(txt)
print(json.dumps({k: v for k, v in res.items() if k != "datasets"}, indent=2))
print(f"datasets compared: {len(keys)}   all identical: {len(bad) == 0}")
sys.exit(0 if len(bad) == 0 else 1)
