#!/usr/bin/env python
"""SLURM `%L` (time left, [D-]HH:MM:SS or MM:SS) -> seconds. Used by ladder_worker.sh."""
import sys

t = sys.argv[1]
d, _, rest = t.partition("-")
days, hms = (int(d), rest) if rest else (0, t)
p = [int(x) for x in hms.split(":")]
while len(p) < 3:
    p.insert(0, 0)
print(days * 86400 + p[0] * 3600 + p[1] * 60 + p[2])
