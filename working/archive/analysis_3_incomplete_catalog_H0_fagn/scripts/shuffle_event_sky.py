#!/usr/bin/env python3
"""Null test for the f_AGN measurement: permute which sky patch each event sits in.

An f_AGN posterior can be narrow for two very different reasons.  It can be narrow
because events land in pixels whose catalogue hosts sit at the right redshift for
their measured distance -- that is the dark-siren information the measurement is
supposed to use.  Or it can be narrow because the two tracer priors carry different
global normalisations, so the mixture weight is pinned by bookkeeping that never
looks at an individual event.

Permuting the per-event (ra, dec) sample blocks among events destroys the first and
leaves the second untouched: the same sky patches are occupied, every event keeps
its own distance, masses, spin and localisation area, but no event's distance is
paired with its own host's redshift any more.  If the f posterior survives the
permutation with the same width, it was never host-association information.

Adapted (not imported) from
`working/analyses/experiments/experiment_twotracer_incomplete/scripts/shuffle_event_sky.py`;
this version also permutes `host_index` / `host_type` in step with the sky blocks so
the bookkeeping columns keep describing the patch they travel with, records the
provenance in the file attrs, and refuses to run on a file it has already shuffled.
"""
import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--in_path", required=True)
ap.add_argument("--out_path", required=True)
ap.add_argument("--seed", type=int, default=90210)
a = ap.parse_args()

out = Path(a.out_path)
out.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(a.in_path, out)
with h5py.File(out, "r+") as f:
    if f.attrs.get("sky_shuffled", False):
        raise SystemExit(f"[fatal] {a.in_path} is already sky-shuffled")
    nobs, nsamp = int(f.attrs["nobs"]), int(f.attrs["nsamp"])
    perm = np.random.default_rng(a.seed).permutation(nobs)
    n_moved = int((perm != np.arange(nobs)).sum())
    if n_moved == 0:
        raise SystemExit("identity permutation drawn; pick another seed")
    for key in ("ra", "dec"):
        v = np.asarray(f[key][:]).reshape(nobs, nsamp)
        f[key][...] = v[perm].reshape(-1)
    for key in ("host_index", "host_type"):
        if key in f:
            f[key][...] = np.asarray(f[key][:])[perm]
    f.attrs["sky_shuffled"] = True
    f.attrs["sky_shuffle_seed"] = int(a.seed)
    f.attrs["sky_shuffle_source"] = str(a.in_path)
    f.attrs["sky_shuffle_note"] = (
        "per-event (ra, dec) sample blocks permuted among events, with host_index / "
        "host_type carried along; host association destroyed, localisation areas, "
        "distances, masses, spins and p_pe preserved")
print(f"wrote {out}  ({nobs} events, {n_moved} moved)")
