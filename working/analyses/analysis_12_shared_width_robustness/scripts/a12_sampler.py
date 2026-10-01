#!/usr/bin/env python
"""Analysis 12 -- the Analysis-11 sampler, with this directory's results/queue/diagnostics.

Problems 12M (11D + one shared sigma_G) and 12chi (11D + one shared sigma_chi) are
defined in ../analysis_11_free_common_population/scripts/a11_sampler.py, on the A11
likelihood (``a11_likelihood.build_a11(shared_width=...)``). Nothing is reimplemented
here: the wrapper only points the sampler's output directories at Analysis 12, so
checkpoints, runs and merges land in this directory. Run tags keep the sampler's
``a11_ns_<problem>_...`` form; merged posteriors are ``results/a12_<problem>.{json,npz}``.

    sbatch --job-name=a12_12M_s1 --time=96:00:00 \\
        --export=ALL,A12_SCRIPT=scripts/a12_sampler.py,A12_ARGS="--stage run --problem 12M --engine dynesty --nlive 200 --seed 1 --first_update_min_eff 100" \\
        scripts/submit_a12_gpu.sbatch
    python scripts/a12_sampler.py --stage merge --out 12M \\
        --runs results/a11_ns_12M_dynesty_n200_s1.json results/a11_ns_12M_dynesty_n200_s2.json
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
A12 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(A12.parent / "analysis_11_free_common_population" / "scripts"))

import a11_sampler as S  # noqa: E402

S.RESULTS, S.CKPT, S.DIAG = A12 / "results", A12 / "queue", A12 / "diagnostics"
S.PREFIX = "a12"

if __name__ == "__main__":
    S.main()
