#!/usr/bin/env python3
"""Aggregate the block sweep and test whether a few events carry the H0 pull.

Reads the 2-point H0 grids written by `localize_h0_pull.sh`, forms each block's
per-event pull toward low H0, and asks whether the distribution is consistent with
one common systematic (all blocks alike) or dominated by a minority of blocks.

Also correlates each block's pull against the event truths carried in the parent
gwcat file, so a positive identification points at a physical property rather than
just a block index.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--nblock", type=int, required=True)
    ap.add_argument("--out_json", required=True)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rd = Path(args.results_dir)

    # ---- per-block pull -------------------------------------------------------
    blocks = []
    for b in range(args.nblock):
        p = rd / f"loc_{args.block}_{b}.h5"
        if not p.exists():
            continue
        with h5py.File(p, "r") as f:
            H0 = f["H0_grid"][:]
            ll = f["log_likelihood"][:]
        if ll.size != 2 or not np.all(np.isfinite(ll)):
            blocks.append({"block": b, "dlogL": None,
                           "note": "non-finite (guard-rejected)"})
            continue
        # grid is [H0_LO, H0_TRUE]; positive => prefers the low H0
        blocks.append({"block": b, "H0_lo": float(H0[0]), "H0_true": float(H0[1]),
                       "dlogL": float(ll[0] - ll[1]),
                       "dlogL_per_event": float((ll[0] - ll[1]) / args.block)})

    ok = [b for b in blocks if b.get("dlogL") is not None]
    if not ok:
        raise SystemExit("[fatal] no usable blocks")
    d = np.array([b["dlogL_per_event"] for b in ok])
    order = np.argsort(d)[::-1]

    # ---- event truths per block ---------------------------------------------
    with h5py.File(args.parent, "r") as f:
        truth = {}
        for key in ("true_z", "z", "snr", "true_m1src", "true_chieff"):
            if key in f:
                truth[key] = f[key][:]
        grp = f.get("truth")
        if grp is not None:
            for key in grp.keys():
                truth.setdefault(key, grp[key][:])
    for b in ok:
        lo = b["block"] * args.block
        hi = lo + args.block
        for key, arr in truth.items():
            if arr.shape[0] < hi:
                continue
            seg = np.asarray(arr[lo:hi], dtype=float)
            b[f"mean_{key}"] = float(seg.mean())
            b[f"max_{key}"] = float(seg.max())

    # ---- is it one systematic, or a minority? --------------------------------
    med, mean = float(np.median(d)), float(d.mean())
    mad = float(np.median(np.abs(d - med)))
    # blocks beyond 3 robust sigma (1.4826*MAD) on the high side
    thresh = med + 3.0 * 1.4826 * mad if mad > 0 else med
    outliers = [ok[i]["block"] for i in order if d[i] > thresh]
    # how much of the total pull the top decile carries
    tot = float(d.sum())
    top10 = float(d[order[: max(1, len(d) // 10)]].sum())

    summary = {
        "block_size": args.block, "n_blocks_used": len(ok),
        "dlogL_per_event": {"mean": mean, "median": med, "mad": mad,
                            "min": float(d.min()), "max": float(d.max()),
                            "std": float(d.std(ddof=1)) if d.size > 1 else None},
        "outlier_threshold_3mad": thresh,
        "outlier_blocks": outliers,
        "top_decile_share_of_total_pull": (top10 / tot) if tot != 0 else None,
        "interpretation": (
            "A single common systematic predicts all blocks near the median with "
            "Gaussian scatter. A minority mechanism predicts a heavy high-side tail "
            "and a top-decile share well above 0.1."),
        "blocks_sorted_by_pull": [ok[i] for i in order],
        "all_blocks": blocks,
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2))

    print(f"blocks used: {len(ok)}  block size: {args.block}")
    print(f"dlogL per event: mean={mean:+.4f} median={med:+.4f} "
          f"min={d.min():+.4f} max={d.max():+.4f} MAD={mad:.4f}")
    print(f"top-decile share of total pull: "
          f"{summary['top_decile_share_of_total_pull']:.3f} (0.1 = uniform)")
    print(f"outlier blocks (>3 robust sigma): {outliers}")
    print("\ntop 8 blocks by pull toward low H0:")
    for i in order[:8]:
        b = ok[i]
        extra = " ".join(f"{k}={b[k]:.3f}" for k in b
                         if k.startswith(("mean_", "max_")))
        print(f"  block {b['block']:3d}  dlogL/event={b['dlogL_per_event']:+.4f}  {extra}")
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
