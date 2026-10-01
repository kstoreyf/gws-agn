# Analysis 9 — mechanism follow-up: why the spin mark sharpens `H0`

**Date.** 2026-09-21. **Sources.** `diagnostics/a9_mechanism_fixed_f.json`
(`scripts/a9_mechanism_fixed_f.py`, CPU, slices of the recorded S9 and J9 cubes),
`diagnostics/a9_event_routing_h0_67p74_from_a8.{h5,json}` (`scripts/a9_event_routing_from_a8.py`,
CPU, Analysis 8's recorded per-event decomposition), `diagnostics/a9_event_routing.{h5,json}`
(`scripts/a9_event_routing.py`, rita job 1335485, 7 min 54 s, one A100-80), figures
`figs/fig_mechanism_fixed_f.{pdf,png}` and `figs/fig_mechanism_event_routing.{pdf,png}`.
darksirens `af896ca` on the pinned base `2b86a2d`, worktree clean; gws-agn `408990a` at run
time. Seed 100 only; the marked mock `events_marked_dmu0p10.h5` and the existing injections;
no new realisation, no new production scan. The likelihood was evaluated only in the routing
job, and every one of its sums was checked against the recorded cubes before being used.

## The question

Analysis 9 measured that freeing the AGN-branch spin offset `Δμ_χ` narrows the dark-siren
`H0` interval by 11.2% (68%) and 10.0% (90%) on the matched 28-node lattice, while
`ρ(H0, Δμ_χ) = +0.025` and freeing `H0` costs `Δμ_χ` 0.1% of its width. The first report
read this as **mechanism A**: the mark tightens the global host fraction `f_AGN` by 4%, and
`f_AGN` sets the weighting between the two tracers whose redshift structure carries `H0`.
The competing reading is **mechanism B**: the mark changes each event's GAL/AGN branch
weight, so each event listens to a different mixture of the two tracers' redshift priors,
and `H0` sharpens event by event with the global fraction playing no part. The two are
separated by holding `f_AGN` fixed in both arms: mechanism A then has nothing left to act
through, mechanism B is untouched.

## 1. The fixed-`f` diagnostic (primary result)

`f_AGN = 0.275` is the MAP node of both arms. The spatial-only arm S9 is sliced at that node
(A9F-S); the marked arm J9 is sliced at the same node and marginalised over `Δμ_χ` (A9F-J);
both are integrated on the 28 `H0` nodes J9 was scanned on, which are bitwise S9 nodes.

| `p(H0 | f_AGN = 0.275)` | median | MAP | 68% | width68 | 90% | width90 |
|---|---|---|---|---|---|---|
| A9F-S spatial-only | 69.0892 | 69.00 | [68.0744, 70.1124] | 2.0381 | [67.4141, 70.8031] | 3.3890 |
| A9F-J marked, `Δμ_χ` marginalised | 69.0881 | 69.00 | [68.1795, 69.9891] | **1.8096** | [67.6137, 70.6587] | **3.0449** |
| A9F-JM marked, `Δμ_χ = +0.1075` fixed | 69.0908 | 69.00 | [68.1847, 69.9897] | 1.8051 | [67.6218, 70.6591] | 3.0372 |

| ratio | 68% | 90% | median shift | MAP shift |
|---|---|---|---|---|
| **`R^F` = A9F-J / A9F-S** | **0.8879** | **0.8985** | −0.0011 | 0.00 |
| A9F-JM / A9F-S (mark fixed) | 0.8857 | 0.8962 | +0.0016 | 0.00 |
| A9F-JM / A9F-J (cost of the mark's own uncertainty) | 0.9975 | 0.9975 | +0.0027 | 0.00 |
| Analysis 9, `f` marginalised (reference) | 0.8879 | 0.8996 | +0.0004 | 0.00 |

**With the global mixture fraction fixed and identical in both arms, the whole `H0` gain
remains.** The fixed-`f` ratio equals the marginalised one to four digits at 68% and is
0.001 smaller at 90%.

**Two-factor decomposition.** The marginalised ratio is exactly the product of the fixed-`f`
ratio and a global-mixture factor,
`[w(J9 marg)/w(S9 marg)] = [w(J9|f)/w(S9|f)] × [(w(J9 marg)/w(J9|f)) / (w(S9 marg)/w(S9|f))]`.
The identity holds to 1e-16 on the recorded widths.

| level | total ratio | event-level factor | global-mixture factor | share of the log-gain, event-level / mixture |
|---|---|---|---|---|
| 68% | 0.87887 | 0.87890 | **0.99997** | 99.97% / 0.03% |
| 90% | 0.89964 | 0.89848 | **1.00130** | 101.2% / −1.2% |

The tighter `f_AGN` posterior contributes nothing to the `H0` gain at 68% and works
slightly *against* it at 90%. Marginalising over the mark itself costs 0.25% of the marked
width (A9F-JM against A9F-J), so knowing the mark exactly would add almost nothing: the
cosmological information is carried by the branch structure the mark induces, not by the
value of `Δμ_χ`.

**Robustness in `f`.** Repeating the fixed-`f` ratio at every `f` node inside J9's 90%
interval:

| `f` held fixed | 0.200 | 0.225 | 0.250 | **0.275** | 0.300 | 0.325 |
|---|---|---|---|---|---|---|
| `R68^F` | 0.8906 | 0.8915 | 0.8899 | 0.8879 | 0.8855 | 0.8831 |
| `R90^F` | 0.8941 | 0.8939 | 0.8951 | 0.8985 | 0.9047 | 0.9020 |

The spread is 0.008 at 68% and 0.011 at 90%; no node comes near 1. Neither arm rejects any
cell at these `f`. As a consistency check, the J9 rows bracketing `Δμ_χ = 0` (nodes −0.0050
and +0.0025; zero is not a node) reproduce the S9 row to within 0.19 nats in the PE term and
bracket S9's width (ratios 1.0096 and 0.9951 at 68%).

## 2. Where in the likelihood the sharpening lives

At `f = 0.275`, `Δμ_χ = +0.1075`, the second difference of the log-likelihood at
`H0 = 69.0` (step 0.5, in nats per (km s⁻¹ Mpc⁻¹)²), split into the event (PE) and the
selection term, read from the cubes:

| | PE term | selection term | total |
|---|---|---|---|
| J9 (marked) | −1.8913 | +0.4229 | −1.4684 |
| S9 (spatial) | −1.5224 | +0.5103 | −1.0122 |
| J9 − S9 | **−0.3689** | **−0.0873** | −0.4562 |

81% of the added curvature is in the event term, 19% in the selection term. The selection
term's expectation is spin-independent — detection depends on chirp mass and distance only —
so its 19% is the injection estimator's response to the reweighted AGN branch (the same
injections carry different weights when the AGN population changes), not a mechanism; it is
part of the uncertainty on the size of the gain, not part of its cause. The slopes at 69.0
are +0.145 (PE) and −0.139 (selection): the two tilts cancel, which is why the mark shows up
as a sharpening with the median unmoved rather than as a shift.

## 3. Event-level routing

The Analysis-8 decomposition is exact: at fixed hyperparameters each event's likelihood is
`Z_i = f_G E_i[GG] + f_A E_i[AA]`, with `E_i[XY]` the PE-sample average of the tracer-X
spatial factor times the population-Y intrinsic factor. In the spatial-only arm both branches
carry the GAL population, so `Z_i^S = f_G E_i[GG] + f_A E_i[AG]`. Therefore, with the true
label used nowhere,

    P_i(AGN | spatial)      = f_A E_i[AG] / (f_G E_i[GG] + f_A E_i[AG])
    P_i(AGN | spatial+mark) = f_A E_i[AA] / (f_G E_i[GG] + f_A E_i[AA])
    ΔP_i = the difference.

Computed at the Analysis-9 point `(H0, f, Δμ_χ) = (69.0, 0.275, +0.1075)` for all 1000
events, and at nine more `H0` nodes from 67.0 to 71.0 for the per-event profiles.

**Verification.** At every one of the ten nodes, `Σ_i ln Z_i^J` reproduces J9's recorded
`guard/logL_pe[H0, 0.275, 0.1075]` and `Σ_i ln Z_i^S` reproduces S9's recorded
`guard/logL_pe[H0, 0.275]`: 34 of 34 comparisons pass, 26 bitwise, the rest at exactly one
ULP of the compared value (1.8e-12 on a PE term of −1.6e4). A live production evaluation at
the representative point agrees with the cube at exactly 0.0 in all four of its terms. The
per-event curvatures sum to the cube's PE curvature to 2e-12 (marked) and 1.3e-11 (spatial).

**How much the mark moves the branch assignment** (H0 = 69.0; the anchor 67.74 from
Analysis 8's recorded file in brackets):

| | value |
|---|---|
| `ΔP_i` percentiles 5 / 16 / 50 / 84 / 95 | −0.166 / −0.104 / −0.008 / +0.112 / +0.208 |
| median \|ΔP_i\| | 0.0679 [0.0674] |
| RMS \|ΔP_i\| | 0.1115 [0.1113] |
| events with \|ΔP_i\| > 0.05 / > 0.1 / > 0.2 | 602 / **348** [339] / 75 [74] |
| events crossing `P = 0.5` | **71** [70]: 55 upward, 16 downward [55, 15] |
| events called AGN at 0.5 | 52 → 91 |
| `Σ_i P_i(AGN)` | 276.3 → 275.3 (realised count 295) |

A third of the events change their branch probability by more than 0.1 and 7% cross the
midpoint, while the expected AGN count moves by −1.0: the mark **re-sorts** the events between
the two tracers, it does not change how many belong to each. That is the event-level
statement of what section 1 found globally.

**Are the re-routed events the ones carrying the new `H0` information?** Per event,
`I_i = −∂²ln Z_i/∂H0²` at 69.0 from the 68.5/69.0/69.5 nodes, and `ΔI_i = I_i^J − I_i^S`.
`Σ_i ΔI_i = +0.3689`, exactly the PE-term curvature change of section 2.

| event set | N | share of `Σ ΔI_i` | share of `Σ |ΔI_i|` |
|---|---|---|---|
| \|ΔP_i\| > 0.05 | 602 | **91%** | 74% |
| \|ΔP_i\| > 0.1 | 348 | 58% | 49% |
| \|ΔP_i\| > 0.2 | 75 | **60%** | 16% |
| crossers of 0.5 | 71 | **61%** | 11% |
| the other 925 events (\|ΔP_i\| ≤ 0.2) | 925 | 40%, mean \|ΔI\| ≤ 9e-4 per bin | — |

Rank correlation of `|ΔP_i|` with `|ΔI_i|` is +0.34 (p = 2e-28) and with the `H0` range of
`ΔlnL_i` is +0.48 (p = 5e-60). Binned in `ΔP_i`, only the two extreme bins carry a mean `ΔI_i`
distinguishable from zero (+0.0057 for `ΔP > +0.2`, −0.0036 for `ΔP < −0.2`); the middle six
bins are consistent with noise. So the added curvature is concentrated in the tail of
strongly re-routed events — 7.5% of the events carry 60% of it — with the bulk of the
sample contributing a small, noisy remainder. Events whose `H0` profile peak moves between
the two arms have median `|ΔP_i|` 0.149 against 0.063 for the rest (the peak-node count
itself, 69, is weak: 956 events peak at a window edge in both arms).

**Interpretation only** (true labels read after the fact): the 55 upward crossers are 32
true AGN and 23 true GAL; the 16 downward are 3 true AGN and 13 true GAL; accuracy at 0.5
rises from 0.715 to 0.734; median `ΔP_i` is +0.035 for true AGN hosts and −0.020 for true
GAL hosts. The mark moves events in the right direction on average and is far from a
classifier — as Analysis 8 already found — which is consistent with the gain being a
likelihood-weighting effect rather than a labelling one.

## 4. Conclusion

**Mechanism B.** Fixing the global mixture fraction removes none of the `H0` improvement:
the marked interval is 0.8879 (68%) and 0.8985 (90%) of the spatial-only one at
`f_AGN = 0.275`, against 0.8879 and 0.8996 with `f` marginalised, and the global-mixture
factor is 0.99997 and 1.0013. The tighter `f_AGN` is a by-product, not the channel. The
central mechanism on this realisation is

    intrinsic population information → event-level GAL/AGN branch weight
    → which tracer's redshift structure each event listens to → H0,

with the added `H0` curvature concentrated in the events whose branch probability the mark
moves most (7.5% of events carrying 60% of it), 81% of the fixed-`f` curvature change in the
event term, and the mark's own uncertainty costing 0.25%. The ladder
`ρ(H0, f_AGN)` = +0.068 → +0.059 → +0.015 that the first report offered as evidence for
mechanism A is real but is a consequence: once each event's branch is better resolved, `H0`
no longer has to co-vary with the global fraction.

Analysis 9's `REPORT.md` and `STATE.md` have been corrected accordingly (marked
"Corrected 2026-09-21"). No number in the Analysis-9 measurement changes; only the
attribution does.

**What this is not.** One realisation, one hyperparameter point for the routing statistics
(the fixed-`f` result holds across the `f` nodes inside the 90% interval). No coverage, no
calibration, no claim about real BBHs in AGN.

## 5. The mass-ratio wording, corrected

Analyses 8 and 9 described the finite-realisation GAL/AGN difference in mass ratio as making
the branch label "partly identifiable from `q`". That is too strong under the model: both
inferred branches use the same `q` distribution, so `q` supplies no GAL-versus-AGN likelihood
ratio when the branch populations differ only in `χ_eff`. The wording of record is now:

> Seed 100 contains an accidental finite-realization difference in the true mass-ratio
> distributions of the two host populations. Since the inferred GAL and AGN branches share
> the same `q` distribution, this does not directly create branch evidence. Correlations
> among `q`, mass, distance and `χ_eff` in the event posterior can nevertheless indirectly
> affect recovery of the environmental spin parameters.

Applied to Analysis 8 `REPORT.md`, `STATE.md`, `GATES.md` and Analysis 9 `REPORT.md`,
`STATE.md`, `GATES.md`. The recorded JSON caveats (`event_decomposition.json`) are run
records and were left as written. No `q`-mark experiment was run.
