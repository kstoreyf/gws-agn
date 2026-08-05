#!/usr/bin/env python3
"""Build a stratified event subsample of a gwcat-1.0 mock event file.

WHY: master @ 2b86a2d admits a likelihood cell only if
`sigma^2_lnL = sum_i sigma^2_i + N_obs^2/Neff <= max_likelihood_variance` (1.0).
The measured per-event PE reweighting variance on this mock is ~0.016 at the
fagn0.3 truth point (../GUARD_AUDIT.md), so the FIRST term alone is ~16 at
N_obs = 1000 and the mixture configurations are rejected outright.  Since that
term is proportional to N_obs and the selection term to N_obs^2, there is an
event count at which the same mock becomes admissible:

    N * 0.0163 + N^2 / 2.02e5 <= 1   =>   N <~ 57

N = 50 leaves headroom (0.82 + 0.01).  That is also, independently, the event
count of the gw_agn proof-of-concept target (sigma(alpha) = 0.086 at N=50).

HOW: gwcat-1.0 is flat event-major — per-sample arrays of length nobs*nsamp and
per-event arrays of length nobs.  Events in these files are ORDERED
`gal_then_agn` (attr `host_order`), so a head slice would be all
galaxy-hosted.  Selection is therefore STRATIFIED: an even stride within each
host-type block, sized to preserve the planted AGN-hosted fraction.  The exact
subsample truth (n_agn/N, which differs from the parent's by rounding) is
written to the output attrs and must be the truth the scan is scored against.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

PER_EVENT = ("host_type", "true_z", "true_m1src", "true_m2src", "true_chieff")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--n_events", type=int, required=True)
    return ap.parse_args(argv)


def stratified_indices(host_type, n_target):
    """Even-stride pick within each host-type block, preserving proportions."""
    idx_out = []
    types, counts = np.unique(host_type, return_counts=True)
    total = int(host_type.size)
    # Largest-remainder allocation so the per-type counts sum to exactly n_target.
    exact = {int(t): n_target * (int(c) / total) for t, c in zip(types, counts)}
    alloc = {t: int(np.floor(v)) for t, v in exact.items()}
    while sum(alloc.values()) < n_target:
        t = max(exact, key=lambda k: exact[k] - alloc[k])
        alloc[t] += 1
        exact[t] -= 1e-9   # avoid re-picking the same key on ties
    for t in sorted(alloc):
        pool = np.flatnonzero(host_type == t)
        k = alloc[t]
        if k <= 0:
            continue
        if k > pool.size:
            raise SystemExit(f"[fatal] host_type {t}: asked {k} of {pool.size}")
        # Even stride across the block, deterministic, no RNG.
        take = np.linspace(0, pool.size - 1, k).round().astype(int)
        take = np.unique(take)
        while take.size < k:      # de-dup collisions at tiny k
            missing = np.setdiff1d(np.arange(pool.size), take)
            take = np.sort(np.concatenate([take, missing[: k - take.size]]))
        idx_out.append(pool[take])
    return np.sort(np.concatenate(idx_out))


def main(argv=None):
    args = parse_args(argv)
    with h5py.File(args.in_path, "r") as f:
        attrs = dict(f.attrs)
        nsamp = int(attrs["nsamp"])
        nobs = int(attrs["nobs"])
        host_type = f["host_type"][:]
        if host_type.size != nobs:
            raise SystemExit(f"[fatal] host_type size {host_type.size} != nobs {nobs}")

        sel = stratified_indices(host_type, args.n_events)
        n_new = int(sel.size)
        # Per-sample row block for each selected event.
        rows = (sel[:, None] * nsamp + np.arange(nsamp)[None, :]).ravel()

        out = {}
        for name, obj in f.items():
            if not isinstance(obj, h5py.Dataset):
                continue
            arr = obj[:]
            if name in PER_EVENT or arr.size == nobs:
                out[name] = arr[sel]
            elif arr.size == nobs * nsamp:
                out[name] = arr[rows]
            else:
                raise SystemExit(f"[fatal] {name}: unexpected size {arr.size} "
                                 f"(nobs={nobs}, nsamp={nsamp})")

    ht_new = out["host_type"]
    n_agn = int(np.sum(ht_new == 1))
    n_gal = int(np.sum(ht_new == 0))
    truth_alpha = n_agn / n_new

    new_attrs = dict(attrs)
    new_attrs.update({
        "nobs": n_new,
        "n_host_gal": n_gal,
        "n_host_agn": n_agn,
        "subsample_of": str(args.in_path),
        "subsample_parent_nobs": nobs,
        "subsample_n_events": n_new,
        "subsample_mode": "stratified-even-stride-by-host-type",
        "subsample_indices": json.dumps([int(i) for i in sel]),
        "subsample_truth_alpha_agn": truth_alpha,
        "subsample_parent_truth_alpha_agn": attrs["n_host_agn"] / nobs,
        "subsample_built_by": str(Path(__file__).resolve()),
    })

    with h5py.File(args.out_path, "w") as g:
        for k, v in new_attrs.items():
            g.attrs[k] = v
        for name, arr in out.items():
            g.create_dataset(name, data=arr)

    print(f"{args.in_path} -> {args.out_path}")
    print(f"  events {nobs} -> {n_new}  (gal {n_gal}, agn {n_agn})")
    print(f"  truth alpha_AGN: parent {attrs['n_host_agn'] / nobs:.6f} "
          f"-> subsample {truth_alpha:.6f}")


if __name__ == "__main__":
    main()
