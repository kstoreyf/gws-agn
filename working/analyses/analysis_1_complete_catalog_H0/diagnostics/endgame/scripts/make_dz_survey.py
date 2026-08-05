#!/usr/bin/env python3
"""ENDGAME -- a survey block identical to the record's except for the DECLARED
photo-z kernel width, written to scratch.

The mock's survey blocks carry ``zgals`` that are BIT-IDENTICAL to the catalog
redshifts ``stage_events`` draws its hosts from, together with a declared
``dzgals = DZ_SCALE * (1 + z)``, ``DZ_SCALE = 3e-3``.  The likelihood's
``p_z(z|pix)`` is therefore a KDE of that bandwidth around EXACT redshifts, while
the mock's hosts sit exactly on them.  Scaling the declared width by ``--scale``
and re-running the (A - B) test is what turns "the declared kernel is the reason
A != B" from an inspection into a measurement.

Only ``dzgals`` changes; ``zgals``, ``wgals`` (all 1.0 on real rows) and ``ngals``
are copied verbatim, and the padding sentinels (z=100, dz=1, w=0) are preserved.
Nothing under working/data is touched.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

DATA = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data")
SCRATCH = Path("/hildafs/projects/phy220048p/magana/gws-agn-data/scratch_truncation_test")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--tracer", choices=["gal", "agn"], default="agn")
    ap.add_argument("--scale", type=float, required=True,
                    help="Multiplier on the declared dz (1.0 = the record).")
    ap.add_argument("--outdir", default=str(SCRATCH / "surveys_dz"))
    args = ap.parse_args(argv)

    src = DATA / f"seed{args.seed}" / "surveys" / f"survey_{args.tracer}_complete_ns32.h5"
    od = Path(args.outdir); od.mkdir(parents=True, exist_ok=True)
    tag = f"x{args.scale:g}".replace(".", "p")
    dst = od / f"survey_{args.tracer}_complete_ns32_dz{tag}.h5"
    if dst.exists():
        print(f"exists: {dst}")
        return 0
    shutil.copy2(src, dst)
    with h5py.File(dst, "r+") as f:
        ng = f["ngals"][:]
        dz = f["dzgals"][:]
        z = f["zgals"][:]
        real = np.arange(dz.shape[1])[None, :] < ng[:, None]
        before = dz[real].copy()
        dz[real] = dz[real] * args.scale
        f["dzgals"][...] = dz
        f.attrs["dz_scale"] = float(f.attrs["dz_scale"]) * args.scale
        f.attrs["dz_convention"] = (f"dz = {float(f.attrs['dz_scale'])} * (1 + z)"
                                    f"  [ENDGAME rescale x{args.scale:g} of the record]")
        f.attrs["endgame_dz_rescale"] = float(args.scale)
        f.attrs["endgame_source_survey"] = str(src)
        print(f"real rows {int(real.sum()):,}  dz {before.min():.6g}..{before.max():.6g}"
              f"  ->  {dz[real].min():.6g}..{dz[real].max():.6g}")
        print(f"zgals unchanged: {np.array_equal(z, f['zgals'][:])}")
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
