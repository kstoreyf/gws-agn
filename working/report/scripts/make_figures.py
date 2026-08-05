#!/usr/bin/env python3
"""Regenerate every figure in the paper from the experiments' results files.

Each fig_*.py module is self-contained: it reads the results files it names in
its own docstring and writes one PDF into ../figures/.  Nothing is copied from
an experiment's own figure directory.

Usage
    JAX_PLATFORMS=cpu python scripts/make_figures.py [name ...]

With no arguments, all figures are rebuilt.  Names may be given with or without
the fig_ prefix.
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FIGURES = [
    "fig_f_recovery",
    "fig_joint_h0f",
    "fig_selection_lane",
    "fig_completeness_anchored",
    "fig_completeness_twotracer",
    "fig_n0_significance",
    "fig_n0_degeneracy",
    "fig_bias_budget",
    "fig_closure_waterfall",
    "fig_sample_variance",
    "fig_kernel_threshold",
    "fig_tilt_decomposition",
]


def main(argv):
    wanted = FIGURES
    if argv:
        asked = {a if a.startswith("fig_") else f"fig_{a}" for a in argv}
        unknown = asked - set(FIGURES)
        if unknown:
            raise SystemExit(f"unknown figure(s): {', '.join(sorted(unknown))}")
        wanted = [f for f in FIGURES if f in asked]

    t0 = time.time()
    for name in wanted:
        mod = importlib.import_module(name)
        mod.main()
    print(f"\n{len(wanted)} figure(s) in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main(sys.argv[1:])
