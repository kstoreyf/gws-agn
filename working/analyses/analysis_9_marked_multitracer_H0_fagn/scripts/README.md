# Analysis 9 scripts

Thin drivers around the Analysis-8 likelihood. Rules, in force:

- **import, never copy.** The likelihood, the opts namespace, the guard spy, the
  KDE window, the pinned population block, the registered \((f, \mu_\chi)\) grid,
  the truths and the posterior convention all come from
  `analysis_8/scripts/{a8_likelihood,gate_c_three_arms}.py` and
  `analysis_2/scripts/scan_h0f.py`. No second likelihood lives here.
- **the Analysis-8 tree is read-only.** `sys.dont_write_bytecode` is set before
  the import so not even a `.pyc` lands there, and `PYTHONDONTWRITEBYTECODE=1` is
  exported by `env_a9.sh`.
- **every write stays under this analysis directory.** The driver's `_write`
  refuses anything else.
- **`PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-a8`** on every
  command. Asserted in-process, and again by `env_a9.sh` before python starts.
- **GPU stages run on RITA via SLURM only.** `a9_scan.py` raises rather than run
  a GPU stage on any other host. The local H100 stays free.
- seed-100 paths stay explicit; exact CLI arguments and upstream SHAs go into
  every output.

## Files

| file | role |
|---|---|
| `a9_scan.py` | the driver: `--stage {provenance,timing,anchor,scan,compare_a8,status,assemble}` |
| `env_a9.sh` | the load-bearing environment; sourced by both harnesses |
| `submit_a9_stage_rita.sbatch` | one GPU, the `timing` and `anchor` stages |
| `submit_a9_rita.sbatch` | one GPU per array task, one interleaved slice of the \(H_0\) axis |

## Order

    # CPU, anywhere
    python a9_scan.py --stage provenance

    # RITA
    sbatch --export=ALL,STAGE="timing anchor" scripts/submit_a9_stage_rita.sbatch

    # CPU — Gate A; the cube does not start until this passes
    python a9_scan.py --stage compare_a8

    # RITA, two GPUs at a time
    sbatch --array=0-7%2 --export=ALL,N_CHUNKS=8 scripts/submit_a9_rita.sbatch

    # CPU, any time
    python a9_scan.py --stage status
    python a9_scan.py --stage assemble

`make_figures.py` and `REPORT.md` come after Gate C, not before.
