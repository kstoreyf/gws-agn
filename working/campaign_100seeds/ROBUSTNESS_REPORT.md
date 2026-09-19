# Seed-ensemble robustness and calibration test

*Run 2026-09-17/18, stopped by the owner once the ensemble was large enough to be
informative. Treated throughout as a calibration exercise, not a new measurement:
nothing here changes a central value, and the point is the spread, not the mean.*

The campaign of record rests on **five** catalog realisations, and its own executive
summary flags that as the binding limitation. This test asks what the five-seed
numbers would have looked like with ~90.

| arm | realisations of record | realisations here |
|---|---|---|
| a0 pure-tracer `H0` | 5 | **93** |
| a1 matched-host controls | 5 | **91** |
| a2 joint `(H0, f_AGN)` | 5 | **9** |

Seed 121 is excluded: it is the only `V3_pe_calibration` failure in 99 seeds
(per-event KS on chirp mass, p = 5.5e-5) although its bijections are exact and its
measurement pulls are healthy (mean 0.008, sd 1.03). One seed in a hundred is free
to drop. The 36 `V8` failures are **not** excluded -- see the README: `complete_ok`
holds for every seed and no seed has PE support outside the catalog, so those are
margin warnings, not defects.

## 1. The central values hold, now at much higher precision

| arm, tracer | mean offset in `H0` | t (dof) | p |
|---|---|---|---|
| a0 GAL | **-0.007 +- 0.220** | -0.03 (92) | 0.975 |
| a0 AGN | **+0.068 +- 0.062** | +1.10 (92) | 0.276 |
| a1 GAL | +0.058 +- 0.257 | +0.23 (90) | 0.821 |
| a1 AGN | +0.161 +- 0.119 | +1.35 (90) | 0.180 |

Units \kmsmpc. Every arm is consistent with zero offset. The five-seed campaign could
only bound the GAL mean offset to about +-1 \kmsmpc; it is now +-0.22, and still zero.

For the joint measurement, 9 realisations against 5:

| | 5 seeds | 9 seeds |
|---|---|---|
| `H0` offset | +0.41 +- 0.55 | **+0.50 +- 0.33** |
| `f_AGN` offset (vs planted) | -0.013 +- 0.019 | **-0.018 +- 0.021** |
| correlation `rho` | 0.105 +- 0.035 | 0.084 +- 0.027 |

## 2. The finding: the sparse tracer's intervals are too narrow

Ratio of realisation-to-realisation scatter to the interval the method quotes for
itself. One means the quoted uncertainty is the right size.

| arm | GAL (dense) | AGN (sparse) |
|---|---|---|
| a0, 93 realisations | **1.004** | **1.261** |
| a1, 91 realisations | 0.961 | 1.375 |

and the same thing read as coverage:

| | 68% interval | 90% interval |
|---|---|---|
| GAL | 71.0% | 89.2% |
| AGN | **50.5%** | **76.3%** |
| nominal | 68% | 90% |

The dense-tracer interval is textbook. The sparse-tracer interval covers truth about
half the time when it claims 68%, and three quarters of the time when it claims 90%;
its quoted width is low by roughly 26-37%. Two independent arms agree -- a0 draws a
fresh 1000-event set per realisation, a1 splits one mixture draw on host type -- so
this is a property of the sparse-tracer likelihood, not of one event selection.

**This is invisible at five realisations.** With five draws, 68% coverage is resolved
only to +-1 count, and the AGN arm's five-seed coverage was inside expectation by
that measure.

## 3. Constraining power, now measured rather than estimated

At equal `N = 1000` on the same catalog, 68% half-widths:

    GAL  2.115 \kmsmpc      AGN  0.474 \kmsmpc
    ratio AGN/GAL = 0.224 +- 0.006   (mean of per-seed ratios 0.234 +- 0.006)

The sparse tracer is ~4.5x the more constraining at equal event count, measured to
2.7% over 93 realisations.

**Carry the caveat from section 2**: if the AGN interval is inflated by its measured
1.26 calibration factor, the honest ratio is ~0.28, i.e. ~3.5x rather than ~4.5x.

## 4. What this does and does not license

- It **does** license dropping "every number is one seed" for a0 and a1, and stating
  the closure as an ensemble result.
- It **does** oblige any quoted sparse-tracer `H0` interval to carry the coverage
  caveat, or to be widened by the measured factor.
- It **does not** change any central value, and it is not a new measurement.
- a2 at 9 realisations is better than 5 but still small; its coverage counts
  (`H0` 5/9 at 68%, 8/9 at 90%) are consistent with nominal and not much more.

## Provenance

darksirens `2b86a2d`, the SHA of record for analyses 0-2 and for the seed datasets.
Verified bit-identical to the working checkout `b324bed` on identical hardware
(max abs diff in logL = 0.000e+00); see the README section on the pin. Data under
`phy220048p/magana/gws-agn-data-v3`, generated with the record's 1.5e8/4.0e8
injection draws and the v3 PE family.

Sources: `analysis_0/results/h0_pure_tracer_ens93.json`,
`analysis_1/results/closure_seeds_ens91.json`,
`analysis_2/results/joint_summary_ens9.json`.
