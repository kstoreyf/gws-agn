#!/usr/bin/env python
"""Expected wall seconds for one queue task. Used by ladder_worker.sh.

    _task_cost.py LEVEL KIND NCHUNK MARGIN OVERHEAD_S

Per-eval costs are the MEASURED ones: results/gates.json for the ladder rungs
(timing pilot), and analysis 3's own complete-pair h0scan for the complete rung.
The fallbacks below are those same measurements, so a missing gates.json degrades
to the right numbers rather than to a guess.
"""
import json
import sys
from pathlib import Path

MEASURED = {"complete": 2.97, "m21": 0.997, "m20": 0.516, "m19": 0.275, "m18": 0.185}

level, kind, nchunk = sys.argv[1], sys.argv[2], sys.argv[3]
margin, overhead = float(sys.argv[4]), float(sys.argv[5])

per_eval = MEASURED.get(level, 3.0)
p = Path("results/gates.json")
if p.exists():
    t = (json.loads(p.read_text()).get("timing") or {}).get(level) or {}
    per_eval = t.get("steady_state_s_per_eval", per_eval)

n = 101 if kind in ("fscan", "fnull") else 201 if kind == "h0scan" else 201 * 41
if nchunk not in ("-", "", "None"):
    n = -(-n // int(nchunk))  # ceil -> the largest chunk

print(int(per_eval * n * margin + overhead))
