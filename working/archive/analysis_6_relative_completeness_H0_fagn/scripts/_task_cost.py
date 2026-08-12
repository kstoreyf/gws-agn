#!/usr/bin/env python
"""Expected wall seconds for one queue task. Used by worker.sh.

    _task_cost.py GLEV ALEV MARGIN OVERHEAD_S

Cost is set by the GALAXY block: at equal depth the AGN catalog holds ~1/100 the
entries (within the horizon, surveys_meta.json: 86,185 vs 8,611,131 complete),
so even `AGN complete` is a smaller block than `GAL m18` and adds a few percent.
Pricing off the AGN level would be meaningless.

The per-level anchors below are analysis 3's measured steady-state s/eval
(results/gates.json).  They are deliberately multiplied by SAFETY = 4: the same
m18 configuration measured 0.185 s/eval in analysis 3 and 0.513 s/eval in
analysis 4, a factor 2.8 discrepancy between directories that no cost model
explains, so the honest response is to overprice and pair this with a generous
walltime.  Worst-case task under this model is ~5 h against a 24 h allocation,
so no task is ever skipped for lack of time.
"""
import sys

MEASURED_GAL = {"complete": 2.97, "m21": 0.997, "m20": 0.516, "m19": 0.275, "m18": 0.185}
SAFETY = 4.0

glev, alev = sys.argv[1], sys.argv[2]
margin, overhead = float(sys.argv[3]), float(sys.argv[4])

per_eval = SAFETY * MEASURED_GAL.get(glev, 3.0)
n = 201 * 41
print(int(per_eval * n * margin + overhead))
