#!/usr/bin/env python
"""The full (unsuffixed) validation stage is unchanged by the corepop edits: seed 100,
unmodified generator (git HEAD 85f2934) vs modified, both in scratch copies whose inputs
are symlinks to the signed-off seed (numpy AVX-512 off).  Every check's verdict and every
number must agree; only the timestamp may differ.  Writes results/fullval_regression_s100.json.
"""
import json
from pathlib import Path

CP = Path(__file__).resolve().parents[1]
SCR = Path("/hildafs/projects/phy230054p/magana/gws-agn-data-v3-scratch-corepop")


def flat(d, p=""):
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            out.update(flat(v, f"{p}/{k}"))
        return out
    return {p: d}


a = json.loads((SCR / "fullval_pre/seed100/validation/validation.json").read_text())
b = json.loads((SCR / "fullval_post/seed100/validation/validation.json").read_text())
fa, fb = flat(a), flat(b)
diff = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
res = {"pre": {"n_checks": a["n_checks"], "failed": a["failed"]},
       "post": {"n_checks": b["n_checks"], "failed": b["failed"]},
       "verdicts_identical": {k: v["pass"] for k, v in a["checks"].items()}
       == {k: v["pass"] for k, v in b["checks"].items()},
       "n_leaves_compared": len(set(fa) | set(fb)), "differing_leaves": diff,
       "PASS": diff == ["/generated_at_utc"]}
(CP / "results" / "fullval_regression_s100.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
