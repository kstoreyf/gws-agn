#!/usr/bin/env python3
"""Null test: permute which sky patch each event's distance belongs to.

An f_AGN posterior can be narrow for two very different reasons. It can be
narrow because events land in pixels whose catalog hosts sit at the right
redshift for their measured distance -- the dark-siren information. Or it can be
narrow because, with n0 and delta anchored, the two completed tracer priors
carry different global normalisations, so the mixture weight is pinned by
bookkeeping that never looks at an event.

Permuting the per-event (ra, dec) blocks among events destroys the first and
leaves the second untouched: the same sky patches are occupied and every event
keeps its own distance, masses and localisation area, but no event's distance is
paired with its own host's redshift any more. If the width survives the
permutation, it was never host-association information.
"""
import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--in_path", required=True)
ap.add_argument("--out_path", required=True)
ap.add_argument("--seed", type=int, default=90210)
a = ap.parse_args()

out = Path(a.out_path)
out.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(a.in_path, out)
with h5py.File(out, "r+") as f:
    nobs, nsamp = int(f.attrs["nobs"]), int(f.attrs["nsamp"])
    perm = np.random.default_rng(a.seed).permutation(nobs)
    if np.all(perm == np.arange(nobs)):
        raise SystemExit("identity permutation drawn; pick another seed")
    for key in ("ra", "dec"):
        v = np.asarray(f[key][:]).reshape(nobs, nsamp)
        f[key][...] = v[perm].reshape(-1)
    f.attrs["sky_shuffled"] = True
    f.attrs["sky_shuffle_seed"] = int(a.seed)
    f.attrs["sky_shuffle_note"] = (
        "per-event (ra,dec) blocks permuted among events; host association "
        "destroyed, localisation areas and all other data preserved")
print(f"wrote {out}  ({nobs} events permuted, {int((perm != np.arange(nobs)).sum())} moved)")
