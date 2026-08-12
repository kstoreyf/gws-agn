#!/usr/bin/env python
"""Expected wall seconds for one queue task. Used by arm_worker.sh.

    _task_cost.py GLEV ALEV MARGIN OVERHEAD_S

Per-eval costs are analysis 3's MEASURED steady-state numbers (its
results/gates.json timing pilot on RITA A100-80); an arm changes one scalar, not
the survey blocks, so the cost is the rung's.  A K=2 evaluation's cost is set by
the two blocks' combined width, so the oracle probe (GAL m18 block 189 + AGN
complete block 178) is priced like m19 (pair width 468) — conservative.
"""
import sys

MEASURED = {"complete": 2.97, "m21": 0.997, "m20": 0.516, "m19": 0.275, "m18": 0.185}

glev, alev = sys.argv[1], sys.argv[2]
margin, overhead = float(sys.argv[3]), float(sys.argv[4])

if glev != alev:  # the oracle probe
    per_eval = MEASURED["m19"]
else:
    per_eval = MEASURED.get(glev, 3.0)

n = 201 * 41
print(int(per_eval * n * margin + overhead))
