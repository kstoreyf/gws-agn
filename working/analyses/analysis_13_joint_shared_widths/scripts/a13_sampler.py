#!/usr/bin/env python
"""Analysis 13 -- the Analysis-11 sampler, with this directory's results/queue/diagnostics.

Problem 13 (11D + BOTH shared widths, dmu_chi on [-0.05, 0.30]) is
defined in ../analysis_11_free_common_population/scripts/a11_sampler.py, on the A11
likelihood (``a11_likelihood.build_a11(shared_width=...)``). Nothing is reimplemented
here: the wrapper only points the sampler's output directories at Analysis 13, so
checkpoints, runs and merges land in this directory. Run tags keep the sampler's
``a11_ns_<problem>_...`` form; merged posteriors are ``results/a13_<problem>.{json,npz}``.

    sbatch --job-name=a13_dyn_s1 --time=96:00:00 \\
        --export=ALL,A13_SCRIPT=scripts/a13_sampler.py,A13_ARGS="--stage run --problem 13 --engine dynesty --nlive 200 --seed 1 --first_update_min_eff 100" \\
        scripts/submit_a13_gpu.sbatch
    python scripts/a13_sampler.py --stage merge --out 13 \\
        --runs results/a11_ns_13_dynesty_n200_s1.json results/a11_ns_13_dynesty_n200_s2.json
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
A13 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(A13.parent / "analysis_11_free_common_population" / "scripts"))

import a11_sampler as S  # noqa: E402

S.RESULTS, S.CKPT, S.DIAG = A13 / "results", A13 / "queue", A13 / "diagnostics"
S.PREFIX = "a13"

if __name__ == "__main__":
    S.main()
