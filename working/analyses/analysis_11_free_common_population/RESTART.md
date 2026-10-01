# Analysis 11 — restart checkpoint (2026-09-28)

**Status (2026-10-01): the brief is complete through its §31 mandatory stop (REPORT.md, end).
Analysis 12 is complete in `../analysis_12_shared_width_robustness/`. Nothing is running. Deferred by the
owner: extending the 12χ Δμ_χ box to [−0.05, 0.30] (the conclusion holds with the cut). Not yet
run: the recommended 8-parameter model (both widths free together) on any seed, then the
multi-realisation calibration.**

## What is done (all committed)

| stage | result | where |
|---|---|---|
| spec, likelihood, closure, selection | PASS (C1/C2 at 1 ULP, C3–C5 bitwise; one rejected spin corner) | `GATES.md`, `diagnostics/a11_closure.json`, `diagnostics/a11_selection_file.json` |
| 11A (f, μ_G, Δμ_G) grid | μ_G 35.48 [34.80, 36.30], Δμ_G 3.85 [2.37, 5.23] (90%) | `REPORT.md` §2, `figs/fig_11A` |
| 11B (f, μ_χ, Δμ_χ) grid | μ_χ −0.0054 [−0.020, 0.009], Δμ_χ 0.126 [0.094, 0.158] | `REPORT.md` §3, `figs/fig_11B` |
| gate after 11A/11B | PASS | `REPORT.md` |
| S0 sampler preflight | PASS (barrier off bitwise, 2.99 s/call, 28.5 GB peak) | `diagnostics/a11_s0_sampler_preflight.json` |
| S1 sampler validation | dynesty PASS against spline-interpolated A10-J grid quantiles (≤ 0.06 half-widths) | `STATE.md`, `diagnostics/a11_s1_compare_*.json` |
| 11C 5-D, fixed H0 | Δμ_G 3.93 [2.49, 5.29], Δμ_χ 0.128 [0.096, 0.160]; D vs A10-J 1.43 / 1.28 | `REPORT.md` §4–6, `figs/fig_11C`, `diagnostics/a11_11C_comparisons.json` |
| event assignment | RMS ΔP 0.028, Spearman 0.994 | `diagnostics/a11_event_assignment.json` |
| fixed-H0 owner gate | PASS on all five conditions | `REPORT.md` |

Owner decisions on record: tinyns is excluded (dynesty only), and 11D is on hold.

## Local-only products (git-ignored; on HildaFS, needed to resume)

- `results/a11_11A.{h5,json}`, `results/a11_11B.{h5,json}`: assembled grids.
- `diagnostics/_a11_11{A,B}_c*of4.jsonl`: grid checkpoints (re-assemble with
  `a11_scan.py --stage assemble --arm 11A|11B`).
- `results/a11_ns_a10J_dynesty_n300_s1.{json,npz}`: S1 run.
- `results/a11_ns_11C_dynesty_n200_s{1,2}.{json,npz}` and `results/a11_11C.{json,npz}`:
  11C runs and their merge.
- `results/a11_ns_11D_dynesty_n200_s{1,2}.{json,npz}` and `results/a11_11D.{json,npz}`:
  11D runs and their merge.
- `queue/*.ckpt`, `queue/*.results.pkl`: dynesty checkpoints and pickled Results (the
  merge input).

## Environment

- GPU: rita only, `--partition=RITA-GPU --qos=rita`, account `phy220048p`. There is no
  separate "priority" QOS; RITA-GPU accepts only `rita` and `rita-s`.
- `source ../analysis_9_spin_marked_H0_fagn/scripts/env_a9.sh` (PYTHONPATH →
  darksirens-a8 at `af896ca`, clean). Inputs: events `events_marked_dmu0p10_dmuG5.h5`
  (md5 `427990378e29…`), injections `injections_targeted.h5` (md5 `e8a611a27f1f…`).
- The provenance blocks list `events_marked_dmu0p10.h5`: a stale default label that
  Analysis 10 also printed. The likelihood uses `a11_likelihood.GW_PATH_A11`; the bitwise C1
  closure against A10-J confirms this.

## Known gotchas

- rita often fails a job at start with `slurmstepd: error: TaskProlog failed` (0 rows written,
  about 5 s). Resubmit that index once, unchanged; every driver resumes from its checkpoint.
- dynesty 2 builds its first ellipsoid only once efficiency drops below 10% by default.
  Production runs pass `--first_update_min_eff 100` (first bound after 2·nlive calls).
- Grid quantiles from a linear CDF between coarse nodes widen 90% ends by up to 0.22
  half-widths; compare samplers against spline-interpolated grid quantiles.
- `np.round` can emit `-0.0` on axes; `a11_scan._key` folds it.

## To resume 11D (only when the owner releases it)

The problem `11D` already exists in `scripts/a11_sampler.py`: (H0 [60, 76], f, μ_G, Δμ_G,
μ_χ, Δμ_χ), with H0 on Analysis 10's contained C10-J window. Same configuration as 11C, one
seed per rita GPU:

    cd working/analyses/analysis_11_free_common_population
    for SEED in 1 2; do
      sbatch --job-name=a11_11D_s$SEED --time=96:00:00 \
        --export=ALL,A11_SCRIPT=scripts/a11_sampler.py,A11_ARGS="--stage run --problem 11D --engine dynesty --nlive 200 --seed $SEED --first_update_min_eff 100" \
        scripts/submit_a11_gpu.sbatch
    done
    # afterwards (CPU):
    python scripts/a11_sampler.py --stage merge \
        --runs results/a11_ns_11D_dynesty_n200_s1.json results/a11_ns_11D_dynesty_n200_s2.json --out 11D

Cost estimate: 11C used ≈ 10k calls per run. One more dimension and the H0 compression
(≈ 2 nats) should give ≈ 12–20k calls per run, ≈ 20–35 GPU-h in total, about a day of wall
time. A run that is killed resumes from `queue/<tag>.ckpt` when resubmitted with identical
arguments.

After 11D, following brief §§18–21: compare with C10-J (H0 67.66 [65.39, 69.43]); report
W68/W90 ratios and the median shift, the Δμ_G and Δμ_χ degradation, and ρ(H0, ·) for all
five population coordinates, with ρ(H0, μ_G) against ρ(H0, Δμ_G) as the key comparison.
Then run the one §21 mechanism diagnostic (H0 profile with μ_G = 35, μ_χ = 0 pinned and the
offsets at their representative values) and write figures for 11D (p(H0) A10 vs A11; the
(H0, μ_G), (H0, Δμ_G), (H0, μ_χ), (H0, Δμ_χ) planes). Analysis 12 (a shared σ_G, then a
shared σ_χ, one at a time) comes only after 11D closes. Finish at the mandatory owner stop
(brief §31) with the 15-line summary.
