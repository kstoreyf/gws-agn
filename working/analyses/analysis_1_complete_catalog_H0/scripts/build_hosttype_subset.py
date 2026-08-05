#!/usr/bin/env python3
"""Split a gwcat-1.0 event file by `host_type` — the matched-subset control.

WHY.  The four production scans of this analysis are DELIBERATELY mis-specified:
the GAL-catalog analysis is handed 316 events whose true hosts are AGN and are
absent from the GAL catalog, and vice versa.  When such a scan lands far from
truth there are two candidate readings — the mis-specification, or something
wrong with the configuration/data — and they are told apart by exactly one run:
give the GAL catalog only its own 684 events, and the AGN catalog only its own
316.  That subset IS the matched analysis (hosts were drawn from the mixture
(1-f) GAL + f AGN, so conditioning on the branch gives precisely a draw from the
single-tracer catalog prior), and the selection integral is unchanged because
mu(theta) never saw the host type.

HOW.  gwcat-1.0 is flat event-major: per-sample arrays of length nobs*nsamp and
per-event arrays of length nobs, plus a `truth/` group of per-event arrays.  All
three are sliced with the same event index set.  Events in the campaign dataset
are stored `as_drawn`, so the two host-type blocks are interleaved; nothing here
depends on their order.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--host_type", type=int, required=True, choices=[0, 1],
                    help="0 = GAL-hosted events, 1 = AGN-hosted events.")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    with h5py.File(args.in_path, "r") as f:
        attrs = dict(f.attrs)
        nsamp = int(attrs["nsamp"])
        nobs = int(attrs["nobs"])
        host_type = f["host_type"][:]
        if host_type.size != nobs:
            raise SystemExit(f"[fatal] host_type size {host_type.size} != nobs {nobs}")

        sel = np.flatnonzero(host_type == args.host_type)
        n_new = int(sel.size)
        if n_new == 0:
            raise SystemExit(f"[fatal] no events with host_type == {args.host_type}")
        rows = (sel[:, None] * nsamp + np.arange(nsamp)[None, :]).ravel()

        datasets, groups = {}, {}
        for name, obj in f.items():
            if isinstance(obj, h5py.Group):
                groups[name] = {}
                for sub, dset in obj.items():
                    arr = dset[:]
                    if arr.shape[0] == nobs:
                        groups[name][sub] = arr[sel]
                    elif arr.shape[0] == nobs * nsamp:
                        groups[name][sub] = arr[rows]
                    else:
                        raise SystemExit(f"[fatal] {name}/{sub}: size {arr.shape}")
                continue
            arr = obj[:]
            if arr.size == nobs:
                datasets[name] = arr[sel]
            elif arr.size == nobs * nsamp:
                datasets[name] = arr[rows]
            else:
                raise SystemExit(f"[fatal] {name}: unexpected size {arr.size} "
                                 f"(nobs={nobs}, nsamp={nsamp})")

    ht_new = datasets["host_type"]
    n_agn = int(np.sum(ht_new == 1))
    n_gal = int(np.sum(ht_new == 0))
    new_attrs = dict(attrs)
    new_attrs.update({
        "nobs": n_new,
        "n_host_gal": n_gal,
        "n_host_agn": n_agn,
        "truth_f_agn": n_agn / n_new,
        "subset_of": str(args.in_path),
        "subset_parent_nobs": nobs,
        "subset_selector": f"host_type == {args.host_type}",
        "subset_indices": json.dumps([int(i) for i in sel]),
        "subset_built_by": str(Path(__file__).resolve()),
    })

    with h5py.File(args.out_path, "w") as g:
        for k, v in new_attrs.items():
            g.attrs[k] = v
        for name, arr in datasets.items():
            g.create_dataset(name, data=arr)
        for gname, members in groups.items():
            grp = g.create_group(gname)
            for sub, arr in members.items():
                grp.create_dataset(sub, data=arr)

    print(f"{args.in_path} -> {args.out_path}")
    print(f"  events {nobs} -> {n_new}  (gal {n_gal}, agn {n_agn})")


if __name__ == "__main__":
    main()
