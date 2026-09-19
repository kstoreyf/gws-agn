# The 100-seed campaign

Extends analyses 0, 1 and 2 from five catalog realisations to many, so that the
absolute offsets those analyses quote stop being one draw of the universe.  Started
2026-09-17.

| arm | target | where it runs | cost, measured |
|---|---|---|---|
| a0 pure-tracer `H0` | 100 seeds (5 existing + 95 new) | rita, 2x A100-80 | 0.34-0.45 GPU-h/seed |
| a1 matched-host controls | 100 seeds | rita, same workers | 0.22 GPU-h/seed |
| a2 joint `(H0, f_AGN)` | 50 seeds (5 existing + 45 new) | miko, 1x H100 NVL, local | 3.86 GPU-h/seed |

New realisations are seeds **106-200**; a2 takes the first 45 of them (106-150).
Seed 104 is skipped throughout -- it exists but failed its own validation in the
original campaign and was never used.

## Cost

Measured, not extrapolated.  A K=2 evaluation costs **1.686 s on the H100 NVL**
against **3.71 s on the A100-40** the original campaign used (probe:
`analysis_2/results/pilot/pilot_h100_rate.json`, 24 evaluations, seed 100), so the
8241-evaluation joint grid is 3.86 GPU-h rather than 8.5.

- a0 + a1: 95 seeds x ~0.6 GPU-h / 2 GPUs = **~29 h wall** on rita.
- a2: 45 seeds x 3.86 = **174 GPU-h = 7.2 days** on the single local H100.  This
  does not fit one 7-day allocation, which is why the grids are chunked: each seed
  is 8 contiguous `H0` chunks and a worker in a later allocation resumes exactly
  where the last one stopped.
- generation: 12 min/seed, CPU only, 16-way on RM = **~1.2 h** for all 95.

## Disk, and why catalogs are deleted

The binding quota is **phy220048p at 57.06 T of 58.59 T** -- 1.53 TB free.  A seed
as originally generated is 9.4 GB, so 95 of them would be 893 GB, over half of
everything left.

`catalogs/` is 6.7 GB of that 9.4 GB, and `catalog_gal_complete.h5` alone is 5.7 GB.
**No scan driver in analyses 0, 1 or 2 ever opens `catalogs/`** -- they read
`surveys/`, `events/` and `injections/`.  It is an input to the surveys and events
stages only.  `gen_one_seed.sh` therefore deletes it once those stages and the two
pure-tracer event draws have finished, which lands each seed at ~2.7 GB and the
campaign at ~260 GB.  It is reproducible in ~4 minutes with `--stage catalogs` on
the same seed if it is ever wanted again.

Analysis outputs are tiny by comparison -- a0 0.11 MiB, a1 0.05 MiB, a2 1.06 MiB per
seed, ~115 MiB for the campaign -- and land in phy230014p (28.8 GB free).

## Validation, and the V6 caveat

Every seed runs the generator's validation stage and the verdict is recorded in
`manifest/seeds.tsv`, but **a validation failure does not stop the seed from being
scanned**.  This is deliberate.  `V6_injections_and_detection_closure` applies a
Gaussian binomial error to bins holding a *single* detected injection near the
horizon, so its tail is far heavier than the nominal sigma scale; seed 104 failed it
at 7.53 against a `< 6.0` gate while its end-to-end closure -- the physically
meaningful quantity -- was 0.088 sigma, the best of the five original seeds.  See
"A note on seed 104" in `analysis_1/README.md`.

Discarding on V6 at this scale would throw away roughly one seed in six on a check
that is known to be miscalibrated.  Recording the verdict and choosing at
aggregation costs nothing and keeps both options open.  `manifest/seeds.tsv` columns:
seed, status, detail, bytes retained, timestamp.

## The darksirens pin (read this before re-running anything)

Every arm of this campaign runs against **darksirens `2b86a2d`**, held in a
dedicated worktree at `src/darksirens-2b86a2d` and enforced by `pin.sh`'s
`ds_pin_guard`, which refuses to start if the SHA is wrong or the worktree is dirty.

This is not a formality.  Three SHAs are in play in this project:

| SHA | what ran on it |
|---|---|
| `2b86a2d` | analyses 0, 1, 2 of record (SLURM 1059xxx) **and** the seed-100..105 datasets |
| `0c5b3db` | the selection-mode redo of analyses 3-7 and the follow-up campaign |
| `b324bed` | what `src/darksirens` happens to be checked out at, on `feat/likelihood-2d-scan` |

The main checkout is the default for both `scan_h0f.py` (via `DARKSIRENS_SRC`) and
`generate_dataset.py` (via `--darksirens`), so **doing nothing would have run the 95
new realisations on `b324bed`** while the 5 existing ones sat at `2b86a2d`.  The new
and old seeds would then differ by code drift as well as by seed, which defeats the
entire purpose of adding seeds -- the campaign's own executive summary already flags
this failure mode when it notes that archive-vs-new comparisons are "estimator plus
drift".

Generation is affected as well as inference: `generate_dataset.py` calls
`import_gmd(args.darksirens)` in its events, surveys and injections stages, and the
dataset records the SHA it used -- `seed101/META.json` carries
`stages.events.darksirens_sha = 2b86a2d...`.  The first two seeds generated for this
campaign were built on `b324bed` before the guard existed; they were deleted rather
than kept.

The H100 rate probe quoted above (1.686 s/eval) was taken on the then-current
checkout.  It is a hardware measurement, not a result, and nothing in the campaign
depends on it beyond scheduling.

## Two findings from the first 24 hours

### The pin was cosmetic for inference, and it did not matter — measured, not argued

`darksirens` is pip-installed **editable** into the jax env (`darksirens.egg-link`,
`easy-install.pth` -> the MAIN checkout).  `scan_h0f.py` uses `DARKSIRENS_SRC` only
for its *git provenance check* (`merge-base --is-ancestor`), never for the import, so
`import darksirens` resolved to the main checkout at `b324bed` regardless of the pin.
The ancestry guard passed because `b324bed` *is* a descendant of `2b86a2d`.

Whether that mattered was settled by measurement, on one A100 scan of seed 106:

| comparison | max abs diff in logL |
|---|---|
| same GPU (miko), `2b86a2d` vs `b324bed` | **0.000e+00** |
| same code, miko H100 vs rita A100-80 | 3.638e-12 |
| same GPU, same code, repeat run | 0.000e+00 |

The two SHAs are **bit-identical on identical hardware**, so every seed computed
before this was found is valid.  The 3.6e-12 is cross-GPU float drift, the same order
the follow-up campaign already bounded at ~1e-13.  This is consistent with the diff:
`2b86a2d..b324bed` touches exactly one importable file,
`darksirens/gw/populations/registry.py`, and only the `GWTC5FiducialBPL2Peaks`
fiducial inside it -- while this campaign runs `pop_model="powerlaw+peak"`, whose
fiducial vector is byte-identical across the two checkouts.  `likelihood/`,
`catalogs/` and `selection/` are untouched.

`pin.sh` now also exports `PYTHONPATH`, which *does* beat the egg-link, so restarts
are genuinely pinned -- the main checkout is a live feature branch and could move.

### V8 fails on ~38% of seeds, and it is a margin, not a violation

    ok_edge = (z_pe_max < 0.7 * z_max) AND complete_ok AND shell_edge >= z_max

`complete_ok` -- the physical requirement, that the complete survey's `z_max` exceeds
the PE support -- is **True for all 99 seeds**, and **no seed** has `z_pe_max` above
the catalog edge (max margin 0.825 against a catalog edge of 1.0).  The failures are
purely the conservative 0.7 bar.  `z_pe_max` is an extreme-value statistic (max over
1000 events x 2000 PE samples x 51 `H0` values), so it scatters seed to seed: the
range over 99 seeds is 0.535-0.825 with a median of 0.674, and the five record seeds
happened to land at 0.565-0.664, all under the bar.

So, as with V6, the verdict is recorded and not acted on.  Note for aggregation: the
100-seed ensemble samples a wider range of PE-support-to-catalog-edge ratios than the
original five did.  Every one of them is well posed.

## Layout

    pin.sh                 the darksirens pin + ds_pin_guard; sourced by all three arms
    gen_one_seed.sh        generation + pure-tracer draws + validation + prune, one seed
    submit_gen.sbatch      RM array; task i generates seed FIRST_SEED + i
    a01_worker.sh          claims seeds, runs a0's four scans then a1's two controls
    submit_a01_rita.sbatch one worker per rita GPU
    a2_worker.sh           claims joint-grid chunks; own queue dir, cannot disturb
                           the original campaign's 48 claims
    status.sh              one-screen state of all three arms
    queue/a01_seeds.txt    the 95 seeds
    queue100/tasks.txt     360 chunks = 45 seeds x 8
    manifest/seeds.tsv     per-seed generation verdict

Claims are `mkdir`, which is atomic on POSIX: any number of workers share a queue
with no lock server, and a killed worker's queue is re-runnable after
`rm -rf queue*/...claim_*` because a chunk whose output already exists is skipped.

## Resuming

    ./status.sh                                          # where everything is
    sbatch --array=0-94%16 submit_gen.sbatch 106         # generation
    sbatch --array=0-1 submit_a01_rita.sbatch            # a0 + a1
    setsid nohup ./a2_worker.sh > logs/a2_local_worker.log 2>&1 &   # a2, on miko

Aggregation, once the arms finish, uses the existing scripts unchanged:
`analysis_0/scripts/aggregate_pure_tracer.py`, `analysis_1/scripts/aggregate_closure.py`,
and for a2 `analysis_2/scripts/merge_joint.py` then `aggregate_joint.py`.
