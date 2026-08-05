#!/usr/bin/env python3
"""Build every figure in the paper.

    python scripts/make_figures.py                    # all of them
    python scripts/make_figures.py joint closure      # just these

One module per figure, each exposing `main()`; every one is deterministic,
reads only the result files named in its own docstring, and writes a vector PDF
plus a 300-dpi PNG preview at the size the figure is printed at.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fig_closure
import fig_joint
import fig_null
import fig_pgm
import fig_pure_tracer
import fig_single_tracer

FIGURES = {
    "pgm": fig_pgm.main,                      # the generative model (v3)
    "single_tracer": fig_single_tracer.main,  # GAL-only and AGN-only H0
    "joint": fig_joint.main,                  # (H0, f_AGN) regions + marginals
    "closure": fig_closure.main,              # five realisations, both params
    "null": fig_null.main,                    # the sky-shuffle null on f
    "pure_tracer": fig_pure_tracer.main,      # appendix: one tracer at a time
}


def main(argv: list[str]) -> int:
    wanted = argv[1:] or list(FIGURES)
    unknown = [w for w in wanted if w not in FIGURES]
    if unknown:
        print(f"unknown figure(s): {', '.join(unknown)}")
        print(f"available: {', '.join(FIGURES)}")
        return 1
    for name in wanted:
        FIGURES[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
