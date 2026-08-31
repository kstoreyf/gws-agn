#!/usr/bin/env python3
"""Split a gwcat-1.0 event file into K disjoint contiguous blocks.

WHY.  A single matched-host control returns one number with a likelihood width
attached.  Whether that width is an honest description of how much the answer
moves under resampling is a separate question, and the cheapest way to ask it
without generating new realisations is to cut the event set into disjoint blocks
and scan each one.  The scatter of the block medians, divided by sqrt(K), is a
direct empirical estimate of the full-set sampling error, which can then be
compared against the width the likelihood quotes.

Events in the campaign dataset are stored `as_drawn` (see working/data/README.md),
so any contiguous block is an unbiased sub-realisation and no shuffling is needed
or wanted -- shuffling would break the reproducibility of the block definition.

Layout handling is identical to build_hosttype_subset.py: gwcat-1.0 is flat
event-major, with per-sample arrays of length nobs*nsamp, per-event arrays of
length nobs, and a `truth/` group of per-event arrays.
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
    ap.add_argument("--out_prefix", required=True,
                    help="Blocks are written to <out_prefix>_b{k}.h5 for k = 0..K-1.")
    ap.add_argument("--n_blocks", type=int, default=8)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    with h5py.File(args.in_path, "r") as f:
        attrs = dict(f.attrs)
        nsamp = int(attrs["nsamp"])
        nobs = int(attrs["nobs"])
        edges = np.linspace(0, nobs, args.n_blocks + 1).astype(int)

        payload = {}
        for name, obj in f.items():
            payload[name] = ({sub: dset[:] for sub, dset in obj.items()}
                             if isinstance(obj, h5py.Group) else obj[:])

    written = []
    for k in range(args.n_blocks):
        sel = np.arange(edges[k], edges[k + 1])
        n_new = int(sel.size)
        rows = (sel[:, None] * nsamp + np.arange(nsamp)[None, :]).ravel()

        def cut(arr):
            if arr.shape[0] == nobs:
                return arr[sel]
            if arr.shape[0] == nobs * nsamp:
                return arr[rows]
            raise SystemExit(f"[fatal] unexpected leading size {arr.shape[0]} "
                             f"(nobs={nobs}, nsamp={nsamp})")

        out_path = f"{args.out_prefix}_b{k}.h5"
        new_attrs = {kk: vv for kk, vv in attrs.items() if kk != "subset_indices"}
        ht = cut(payload["host_type"])
        new_attrs.update({
            "nobs": n_new,
            "n_host_gal": int(np.sum(ht == 0)),
            "n_host_agn": int(np.sum(ht == 1)),
            "truth_f_agn": float(np.mean(ht == 1)),
            "block_index": k,
            "block_count": args.n_blocks,
            "block_event_range": json.dumps([int(edges[k]), int(edges[k + 1])]),
            "subset_of": str(args.in_path),
            "subset_parent_nobs": nobs,
            "subset_selector": f"contiguous block {k}/{args.n_blocks}",
            "subset_built_by": str(Path(__file__).resolve()),
        })
        with h5py.File(out_path, "w") as g:
            for kk, vv in new_attrs.items():
                g.attrs[kk] = vv
            for name, obj in payload.items():
                if isinstance(obj, dict):
                    grp = g.create_group(name)
                    for sub, arr in obj.items():
                        grp.create_dataset(sub, data=cut(arr))
                else:
                    g.create_dataset(name, data=cut(obj))
        written.append((out_path, n_new))

    print(f"{args.in_path}  ({nobs} events) -> {args.n_blocks} blocks")
    for p, n in written:
        print(f"  {p}  {n} events")


if __name__ == "__main__":
    main()
