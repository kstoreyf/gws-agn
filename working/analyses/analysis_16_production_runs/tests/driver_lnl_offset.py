#!/usr/bin/env python
"""Test helper (CPU): run scripts/prod_driver.py unchanged, with the model's ln L shifted by a constant.

    A16_LNL_OFFSET_STATE=<dynesty state> <core venv python> tests/driver_lnl_offset.py <driver arguments ...>

Used to continue a production checkpoint on CPU with a dry-run model (20 events, 20,000 injections): the dry run's
ln L lives on another scale than the production model's, so its values are shifted by one constant, the state's
highest live ln L minus the dry run's corrected ln L at that live point (the same constant in every process that
reads the same state). The sampler, checkpoints, --init-from, --switch-sample and resumption are the driver's own;
the diagnostics-vs-bound check sees the unshifted model. A16_SCRIPTS selects the scripts directory (default
scripts/). Never used by production runs.
"""
import os
import pickle
import signal
import sys
from pathlib import Path

A16 = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.environ.get("A16_SCRIPTS", str(A16 / "scripts")))

import numpy as np  # noqa: E402
import prod_model  # noqa: E402

_make = prod_model.make_model
STATE = os.environ["A16_LNL_OFFSET_STATE"]


def _best_live(path):
    with open(path, "rb") as fh:
        s = pickle.load(fh)["sampler"]
    i = int(np.argmax(s.live_logl))
    return np.asarray(s.live_v[i], float), float(s.live_logl[i])


def make_model(cfg):
    m = _make(cfg)
    ev, chk = m.evaluate, m.check_diag_equals_bound
    box = {}

    def shifted(coords):
        if "offset" not in box:
            x, lmax = _best_live(STATE)
            r0 = ev(dict(zip(cfg["free"], map(float, x))))
            box["offset"] = lmax - r0["lnL_corrected"]
            print(f"[offset] ln L shifted by {box['offset']:.6f} (state max live ln L {lmax:.6f}, dry run "
                  f"{r0['lnL_corrected']:.6f} at that point)", flush=True)
        r = dict(ev(coords))
        r["lnL"] += box["offset"]
        r["lnL_corrected"] += box["offset"]
        return r

    def check(points):
        m.evaluate = ev
        try:
            return chk(points)
        finally:
            m.evaluate = shifted

    m.evaluate = shifted
    m.check_diag_equals_bound = check
    return m


prod_model.make_model = make_model
import prod_driver  # noqa: E402

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(prod_driver.EXIT_DEADLINE))
    sys.exit(prod_driver.main(sys.argv[1:]))
