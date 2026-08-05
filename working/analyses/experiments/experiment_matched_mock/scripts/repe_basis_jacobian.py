#!/usr/bin/env python3
"""Rewrite p_pe into the likelihood's canonical (m1det, q) basis.

darksirens divides the per-sample weight by ``prior_wt`` in the basis
(m1det, q, dL, chi_eff, sky) -- ``likelihood/core.py`` calls
``log_sample_weight(m1det, q, dL, chieff, pix, prior_wt, ...)`` and
``inference/utils.py`` subtracts ``log(prior_wt)`` there.  gmd draws its PE
samples flat in (m1det, m2det, ...) and stores ``p_pe = 1``, but a density flat
in m2det is NOT flat in q:

    p(m1det, q) = p(m1det, m2det) * |d m2det / d q| = p(m1det, m2det) * m1det

so the correct ``p_pe`` is proportional to ``m1det``.  The selection side is
already in the canonical basis -- ``_selection_pdraw`` documents it and carries
the (1+z) m1src->m1det Jacobian explicitly -- so only the PE side is
mislabelled, and the asymmetry is what makes it a candidate systematic rather
than a cancelling convention.

Only the WITHIN-EVENT variation of m1det matters: a per-event constant in p_pe
shifts that event's log-evidence by an H0-independent constant.  The coupling to
H0 runs through the mass population term, m1src = m1det / (1 + z(dL; H0)).

Copies an events file and overwrites the ``p_pe`` dataset; nothing else changes.
"""
import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--in_path", required=True)
ap.add_argument("--out_path", required=True)
a = ap.parse_args()

out = Path(a.out_path)
out.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(a.in_path, out)
with h5py.File(out, "r+") as f:
    m1det = np.asarray(f["m1det"][:], dtype=np.float64)
    old = np.asarray(f["p_pe"][:], dtype=np.float64)
    if not np.allclose(old, 1.0):
        raise SystemExit("p_pe is not 1; refusing to guess what basis it is in")
    del f["p_pe"]
    f.create_dataset("p_pe", data=m1det, compression="gzip", shuffle=True)
    f.attrs["p_pe_basis"] = "(m1det, q, dL, chieff): p_pe = m1det (Jacobian dm2det/dq)"
    f.attrs["p_pe_rewritten_by"] = str(Path(__file__).resolve())
print(f"wrote {out}  p_pe <- m1det  (mean {m1det.mean():.3f})")
