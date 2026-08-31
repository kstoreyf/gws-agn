# The v3 measurement family: literature prescriptions, the adopted model, and why it closes

Owner-approved redesign, 2026-08-01.  This document records (i) the published
mock-PE prescriptions that were checked, with citations and the numbers taken
from each, (ii) the measurement family actually adopted here (`PE_MODEL = "v3"`
in `generate_dataset.py`), (iii) every derivation the implementation depends on —
in particular the exact `p_pe` Jacobian in darksirens' canonical basis — and
(iv) the argument, term by term, for why the detected-set score identity closes
by construction under this family and did not under the previous one.

`darksirens` was **READ-ONLY at `2b86a2d`** throughout.  Nothing in this document
changes the estimator; it changes only what the mock hands it.

---

## 0. Why the family had to change

`CLOSURE.md` ends (§15) with a measured statement: with the mass PE drawn from
the **exact** flat-prior posterior of the realised measurement, with every catalog
galaxy at its true `(ra, dec, z)`, with the mock's own host prior and the exact
selection function, the per-event score residual is still

```
r = <d ln Z_i/dH0> - d ln mu/dH0 = -6.20e-4     (matched GAL, per event)
```

and the split localises it entirely in the **posterior-averaging step of the mass
channel**: `(C - A)_pop = -1.274e-3 +- 0.113e-3` (11.3 sigma) against
`(A - B)_pop = -4.4e-5 +- 1.2e-5`.

The previous family drew

```
obs_m1 ~ N(m1det, f1 * m1det),      f1 = 0.08     <- width set by the LATENT mass
obs_m2 ~ N(m2det, f2 * m2det),      f2 = 0.10     <- likewise
ln obs_dL ~ N(ln dL, 0.10)                        <- constant log-width
```

Three properties of that family are, jointly, the reason no repair inside it
closed the identity:

1. **The mass widths are functions of the latent parameter.**  The likelihood
   `N(obs; m, f m)` carries a `theta`-dependent normalisation `1/(f m)`, so the
   flat-prior posterior is skewed by construction (mean `+2f^2` above `obs`) and
   the ensemble mean of a non-linear functional of it need not equal the
   functional at the truth.  Convention (c2) made the *stored samples* exact for
   that likelihood; it did not remove the latent-dependence itself.
2. **`m1det` and `m2det` are measured independently.**  Real interferometric
   measurements constrain the chirp mass `Mc` to a fraction of a percent and the
   mass ratio hardly at all; independent 8 % / 10 % component masses are an
   unphysically strong and *uncorrelated* mass measurement.  18.4 % of the
   previous dataset's PE samples had `q = m2det/m1det > 1` and were discarded by
   the population prior — a symptom of the same thing.
3. **The mass–distance channels are uncorrelated.**  The real degeneracy runs
   through the SNR: `rho ~ Mc_det^(5/6)/dL`, so a chirp-mass error *is* a
   distance error.  Breaking that link makes the spectral-siren lever
   (`sigma/m = 8-10 %` against a `35 +- 5 Msun` peak) far stronger and far more
   curved than any real catalog's.

The redesign fixes all three at once by moving to the literature's own mock-PE
basis, in which **every measurement width is a function of the observed SNR**,
i.e. of data.

---

## 1. The literature check

Three sources were fetched and read.  What each contributes is recorded below;
the code constants carry the same citations in `generate_dataset.py`.

### 1.1 Fishbach, Holz & Farr (2018), arXiv:1805.10270 §4.2 — the original prescription

The synthetic-detection model that the whole later lineage is built on:

```
(29)   rho_obs  =  rho + N(0, 1)                      detection: rho_obs >= 8
(30)   log M_obs = log( M (1+z) ) + N(0, 8 sigma_M / rho_obs),   sigma_M = 0.04
(31)   eta_obs   = eta            + N(0, 8 sigma_eta / rho_obs), sigma_eta = 0.03
```

Two structural points, both adopted here:

* **the SNR noise is additive with unit variance and the threshold is on the
  *observed* SNR**, so detection is a deterministic function of recorded data;
* **every other width scales as `8/rho_obs`**, i.e. as one over the *observed*
  SNR — never one over a latent SNR.

### 1.2 Fishbach & Holz (2020), arXiv:1905.12669 Appendix B — the "general framework"

The appendix `GWMockCat`'s README points to.  It quotes

```
sigma_Mc/Mc = (8/rho) * sqrt( 0.01 + (0.2 z/(1+z))^2 )
sigma_eta   = 0.03 * (8/rho)
detection: single-detector SNR >= 8
```

and states that the framework produces *"typical 90 % measurement uncertainties
on the source-frame component masses of ~50 %"* — the calibration target used in
§2.3 below.

**Not adopted: the redshift-dependent chirp-mass width.**  The `0.2 z/(1+z)`
term is a function of the source's **latent redshift**, which is exactly what
campaign convention (b) forbids: a width that depends on a latent makes the
recorded PE width carry information the fixed-width posterior cannot represent,
and it breaks the detected-set score identity (measured cost `-0.49 +- 0.08
km/s/Mpc` even under the exact likelihood, darksirens PR #335).  The
`GWMockCat` release itself drops it in favour of a constant coefficient, and
that is what is used here.

### 1.3 Farah, Edelman, Zevin, Fishbach, Maria Ezquiaga, Farr & Holz (2023) — `GWMockCat`

*"Things That Might Go Bump in the Night: Assessing Structure in the Binary Black
Hole Mass Spectrum"*, ApJ **955**, 107 (2023); arXiv:2301.00834; Appendix A.
Code: <https://git.ligo.org/amanda.farah/GWMockCat> (CC0).

The package is the operative reference — the appendix describes it and the code
fixes the constants.  `GWMockCat/posterior_utils.py`:

```python
uncert_default = {
    "threshold_snr": 8,
    "snr": 1.0,
    "mc": 0.08,
    "Theta": 0.15 * 1.4,
    "eta": 0.022,
}
...
rho_obs = rho_true + uncert["snr"] * rng.normal(size=len(m1))
det_sel = rho_obs > uncert["threshold_snr"]
...
smc  = uncert["threshold_snr"] / rho_obs * uncert["mc"]
mcobs = rng.lognormal(mean=np.log(mc), sigma=smc)
seta = uncert["threshold_snr"] / rho_obs * uncert["eta"]
etaobs = eta + seta * truncnorm.rvs((0.0 - eta)/seta, (0.25 - eta)/seta, ...)
st   = uncert["threshold_snr"] / rho_obs * uncert["Theta"]
...
mc_samps  = rng.lognormal(mean=np.log(mcobs[i]), sigma=smc[i], size=PEsamps)
eta_samps = truncnorm(..., loc=etaobs[i], scale=seta[i]).rvs(...)
rho_samps = truncnorm((0.0 - rho_obs[i])/uncert["snr"], np.inf,
                      loc=rho_obs[i], scale=uncert["snr"]).rvs(...)
```

and `GWMockCat/parser.py` adds `--Xeff_uncert  default=0.2`.

So the released defaults are

| quantity | symbol | value | meaning |
|---|---|---|---|
| SNR threshold | `rho_th` | 8 | on the **observed** SNR |
| SNR noise | `sigma_rho` | 1.0 | additive, `rho_obs = rho_true + N(0, 1)` |
| detector-frame chirp mass | `alpha_Mc` | 0.08 | `sigma_lnMc = alpha_Mc * (8/rho_obs)` |
| symmetric mass ratio | `alpha_eta` | 0.022 | `sigma_eta = alpha_eta * (8/rho_obs)` |
| projection factor `Theta` | `alpha_Theta` | 0.21 | `sigma_Theta = alpha_Theta * (8/rho_obs)` |
| effective spin | `alpha_chi` | 0.2 | `sigma_chieff = alpha_chi * (8/rho_obs)` |

Three structural facts read off the code, all of which the design below inherits:

* **`dL` is never measured directly.**  `GWMockCat` samples `(Mc, eta, rho,
  Theta)` and *derives* the luminosity distance,
  `dL = Theta * rho_opt(m1, m2)/rho` (`transforms.py::redshift`).  The distance
  observable **is** the SNR.  This is the single most important structural point
  of the check — see §3.1.
* **the PE draws are centred on the OBSERVED value with the SAME width**, i.e.
  they are flat-prior posteriors of a likelihood whose width is data-derived —
  precisely the campaign's convention (b)+(c);
* **the parameters are correlated only through the SNR**: the README says
  *"These parameters are correlated through their dependence on SNR, but beyond
  that no parameter correlation is assumed. Then, all samples are transformed to
  detector-frame component masses and luminosity distance, resulting in somewhat
  realistic degeneracies between component masses."*

### 1.4 `gwbench` (Borhanian 2021, arXiv:2010.15202) — the Fisher-matrix family

Consulted for the *form* rather than for numbers.  A Fisher-matrix forecast makes
every 1-sigma width scale as `1/rho` at fixed intrinsic parameters, which is the
`a/rho` family used throughout below, and it makes the sky area scale as
`1/rho^2`, i.e. `sigma_ang ~ 1/rho` — the campaign's existing sky convention.
The Fisher route was **not** adopted as the generator's width source, for one
structural reason: a Fisher matrix is evaluated at the **true** parameters, so
`sigma_Fisher(theta_true)` is a latent-dependent width and is forbidden by
convention (b) exactly as the Fishbach & Holz `z`-dependent term is.  Evaluating
the Fisher matrix at the *observed* point would restore data-dependence but adds
a waveform/PSD dependency the campaign does not need.  The `GWMockCat` family —
Fisher-like `1/rho` scaling with calibrated constants, evaluated at `rho_obs` —
is the same physics with none of that machinery.

### 1.5 Sky localisation

The campaign's existing convention

```
sigma_ang = clip( 35 deg / rho_sigma , 1 deg, 12 deg ),
rho_sigma = network amplitude on darksirens' own rho_ref = 11.5 scale
```

is already a member of the `a/rho` family, and `1/rho` is the standard
localisation scaling (`Delta Omega ~ rho^-2`; Fairhurst 2009, arXiv:0908.2356;
Berry et al. 2015, arXiv:1411.6934).  It is **kept verbatim**, with the single
change that `rho_sigma` is now formed from the recorded `rho_obs` rather than
recomputed from the observed masses and distance:

```
rho_sigma = (SNR_REF_SIGMA / SNR_REF_DETECT) * rho_obs = 1.83165 * rho_obs
```

so `sigma_ang = clip(19.1069 deg / rho_obs, 1, 12)`.  The realised width
distribution is essentially unchanged from the v2 dataset (`sigma_ang` was in
`[1.0, 2.39] deg`), which is the continuity the owner asked for.

---

## 2. The adopted model

`PE_MODEL = "v3"`.  Constants live in `generate_dataset.py` under
`# --- v3 measurement family ---` and are recorded in every seed's `META.json`.

### 2.1 Parameters, data, and the bijection

The campaign forbids a projection latent (convention (a)), so the mock's
amplitude is projection-free and

```
rho_opt(theta) = SNR_REF_DETECT * (Mc_det/30)^(5/6) * (1000 Mpc / dL)
```

is an **exact function of `(Mc_det, dL)`**.  That makes

```
(Mc_det, q, rho, chieff, ra, dec)   <-->   (m1det, m2det, dL, chieff, ra, dec)
```

a bijection, with

```
Mc_det = m1det q^(3/5) / (1+q)^(1/5),     q = m2det/m1det
m1det  = Mc_det (1+q)^(1/5) q^(-3/5),     m2det = q m1det
dL     = 1000 * SNR_REF_DETECT * (Mc_det/30)^(5/6) / rho
```

so the SNR **is** the distance coordinate.  This is `GWMockCat`'s own
construction with `Theta` removed.

### 2.2 The generative order (all-observable, sequential)

```
1.  theta_true  = (m1det, m2det, dL, chieff, ra, dec)   from population + host
2.  rho_true    = rho_opt(theta_true)
3.  rho_obs     = rho_true + sigma_rho * N(0,1)          sigma_rho = 1.0
4.  DETECTION   = [ rho_obs >= 8 ]                       deterministic in DATA
5.  widths from rho_obs, and ONLY from rho_obs:
       sigma_lnMc   = A_MC   * (8/rho_obs)               A_MC   = 0.08
       sigma_lnq    = A_Q    * (8/rho_obs)               A_Q    = 0.60
       sigma_chieff = A_CHI  * (8/rho_obs)               A_CHI  = 0.20
       sigma_ang    = clip( 35 deg / (1.83165 rho_obs), 1 deg, 12 deg )
6.  observations, each an UNBOUNDED Gaussian in its own variable:
       ln Mc_obs = ln Mc_det + sigma_lnMc   * N(0,1)
       ln q_obs  = ln q      + sigma_lnq    * N(0,1)
       chieff_obs= chieff    + sigma_chieff * N(0,1)
       dec_obs   = dec       + sigma_ang    * N(0,1)          (b2: dec FIRST)
       sigma_ra  = sigma_ang / max(cos dec_obs, 0.1)
       ra_obs    = (ra + sigma_ra * N(0,1)) mod 2 pi
```

**No observation is clipped or truncated.**  That is a deliberate change from v2
and it is not cosmetic: clipping the *data* (as v2 did for `obs_m1`, `obs_m2`,
`obs_chieff` and `obs_dec`) makes the measurement model *censored*, i.e. gives
the likelihood a `theta`-dependent normalisation
`P(obs = boundary | theta) = 1 - Phi(...)`, and the exact flat-prior posterior is
then no longer a simple truncated normal.  Truncating the **prior** is free;
truncating the **data** is not.  v3 therefore truncates only priors (§2.4).

The same reasoning is why the mass ratio is measured as `ln q` with unbounded
noise rather than as `eta` on `[0, 0.25]`: `GWMockCat`'s truncated-normal `eta`
observation has normalisation `Z(eta) = Phi((0.25-eta)/s) - Phi(-eta/s)`, which
depends on `eta`.  With this campaign's `beta = 1` pairing and a `35 Msun` peak
the detected `eta` median is **0.2473** — i.e. the median event sits `0.17
sigma` from the boundary — so the censoring would be active for most of the
catalog, not a corner case.  `ln q` with a prior truncation at `q <= 1` carries
the same physics with no `theta`-dependent normalisation anywhere.

### 2.3 Calibration of `A_Q`

`GWMockCat` gives `sigma_eta = 0.022 (8/rho_obs)`.  With
`eta = q/(1+q)^2`, `d eta/d ln q = q(1-q)/(1+q)^3`, so

```
A_Q(q) = 0.022 * (1+q)^3 / ( q (1-q) )
```

which is `0.63` at `q = 0.75` and `0.86` at the v2 detected median `q = 0.812`.
The divergence at `q -> 1` is real physics (`eta` is stationary at equal mass) but
makes a single reference value necessary.  Two independent anchors were used:

* **the literature's own component-mass statement.**  Fishbach & Holz (2020)
  Appendix B: 90 % component-mass uncertainties of `~50 %`.  With
  `d ln m1det/d ln q = q/(5(1+q)) - 3/5 = -0.51` at `q ~ 0.8`, and `sigma_lnMc`
  small, `sigma_ln m1det ~ 0.51 * sigma_lnq`.  A 90 % interval of +-50 % is
  `sigma_ln m1 ~ 0.30`, i.e. `sigma_lnq ~ 0.59` at the detected median SNR
  (`rho ~ 10`), i.e. `A_Q ~ 0.74`; reading `50 %` as the *full* 90 % width gives
  `A_Q ~ 0.37`.
* **a real event.**  GW150914 (Abbott et al. 2016, arXiv:1602.03840) measured
  `q = 0.86 (+0.14 / -0.21)` at network `rho ~ 24`, i.e. `sigma_lnq ~ 0.20` and
  `A_Q = 0.20 * 24/8 = 0.60`.

**Adopted: `A_Q = 0.60`**, i.e. `sigma_lnq = 0.60 * (8/rho_obs)`, which is the
GW150914 anchor and sits inside the range the two `eta`-conversion reference
points (`0.63` at `q = 0.75`) and the component-mass statement bracket.  The
sensitivity of the closure to this number is *not* a modelling risk: `A_Q` sets
how strong the mass lever is, and the score identity closes for **any** `A_Q`
because the width is a function of data (that is the whole point of the family).
`A_Q` is a single config constant and is recorded in `META.json`.

### 2.4 The PE: exact flat-prior posteriors, channel by channel

The PE prior is declared **flat in `(ln Mc_det, ln q, rho, chieff, ra, dec)`**
on the physical support

```
S = { q <= 1 } x { rho > 0 } x { |chieff| <= 1 } x { |dec| <= pi/2 } x { ra in [0,2pi) }
```

Because every channel's likelihood is an unbounded Gaussian in exactly the
variable the prior is flat in, each posterior is a (possibly truncated) normal
about the **observed** value with the **stored** width — no shift, no skew, no
quantile table:

| channel | posterior | truncation |
|---|---|---|
| `ln Mc_det` | `N(ln Mc_obs, sigma_lnMc)` | none |
| `ln q` | `N(ln q_obs, sigma_lnq)` | `ln q <= 0` (prior) |
| `rho` | `N(rho_obs, sigma_rho)` | `rho > 0` (prior; `rho_obs >= 8`, so inert at `8 sigma`) |
| `chieff` | `N(chieff_obs, sigma_chieff)` | `[-1, 1]` (prior) |
| `dec` | `N(dec_obs, sigma_ang)` | `[-pi/2, pi/2]` (prior) |
| `ra` | wrapped `N(ra_obs, sigma_ra)` | `mod 2 pi` |

Every truncated draw is an exact inverse-CDF truncated normal
(`scipy`-free: `Phi^-1( Phi(a) + u (Phi(b) - Phi(a)) )` in `ndtr/ndtri`).
Samples are then mapped through the bijection of §2.1 to
`(m1det, m2det, dL, chieff, ra, dec)`.

Note that `q <= 1` now holds for **every** PE sample by construction, against
18.4 % `q > 1` in v2 — the previous dataset was spending a fifth of its samples
on a region the population prior sets to zero.

### 2.5 `p_pe`: the exact Jacobian in darksirens' canonical basis

darksirens divides each sample by `p_pe`, the **PE prior density expressed in the
canonical basis** `x = (m1det, q, dL, chieff)`.  (The v2 file stored
`p_pe ~ m1det`, which is exactly this rule applied to a prior flat in
`y = (m1det, m2det, dL, chieff)`: `|dy/dx| = |d m2det/d q| = m1det`.)

Here `y = (ln Mc_det, ln q, rho, chieff)` and `p_pe = const * |dy/dx|`.  Write
`A = 3/(5q) - 1/(5(1+q))`.  From
`ln Mc = ln m1det + (3/5) ln q - (1/5) ln(1+q)` and
`ln rho = const + (5/6) ln Mc - ln dL`,

```
d(ln Mc, ln q, rho) / d(m1det, q, dL)  =
   [  1/m1det          A            0     ]
   [  0                1/q          0     ]
   [ (5/6) rho/m1det  (5/6) rho A  -rho/dL ]
```

Expanding along the third column (only the `(3,3)` entry is non-zero),

```
|dy/dx| = (rho / dL) * (1 / (m1det q))
```

so, with `rho = 1000 * SNR_REF_DETECT * (Mc_det/30)^(5/6) / dL` evaluated at the
sample,

```
                 rho                       Mc_det^(5/6)
   p_pe    ~  -----------      ~      -------------------------
              dL m1det q                 dL^2  m1det  q
```

`chieff` maps identically (no factor) and the sky prior is flat in `(ra, dec)`,
the same convention as v2, so `p_pe` carries no sky factor.  The stored column is
normalised to mean 1 per event (darksirens renormalises per event, so only the
shape matters).  `p_pe_unity` is retained as before, and is likewise a statement
about the prior only.

**Validation of this derivation is structural, not asserted**: `--stage
validation` check **V3c** recomputes `p_pe` from the stored samples by two
independent routes — the closed form above, and a numerical Jacobian of the
bijection by central differences — and requires agreement to `1e-10` relative.

### 2.6 What is stored

`events.h5` gains, in `truth/`, the recorded measurement in the measurement
basis and every width, so **every width is recomputable from stored data**:

```
obs_rho          rho_obs                     (THE detection statistic)
obs_lnmc         ln Mc_det_obs
obs_lnq          ln q_obs
obs_chieff       chieff_obs
obs_dec, obs_ra
obs_sigma_rho    sigma_rho          ( = SIGMA_RHO, constant )
obs_sig_lnmc     A_MC  * 8/rho_obs
obs_sig_lnq      A_Q   * 8/rho_obs
obs_sig_chieff   A_CHI * 8/rho_obs
obs_sigma_ang    clip(35 deg/(1.83165 rho_obs), 1, 12)
obs_sig_ra       sigma_ang / max(cos dec_obs, 0.1)
```

plus the derived point estimates `obs_m1det, obs_m2det, obs_dL` (the bijection
evaluated at the observation) for diagnostics; **the PE never reads them**.
`snr_obs` is kept as an alias of `obs_rho` so downstream tooling that reads
`truth/snr_obs` keeps working, and `snr_true` is `rho_opt(theta_true)`.

---

## 3. Why the identity closes

Write `d` for the recorded data of one event, `theta` for its source parameters.
Under a correctly specified detected-set likelihood, for **every** function `h`,

```
C = mean_i E_post_i[h]      A = mean_i h(theta_i^true)      B = E_model-det[h]
E[C] = E[A] = B
```

by the tower property, **provided** (i) the posterior darksirens forms is the true
`p(theta | d)`, and (ii) detection is a deterministic function of the data the
estimator conditions on.  Both now hold by construction.

### 3.1 The estimator's likelihood is the true likelihood

The true data likelihood factorises sequentially,

```
p(d|theta) = N(rho_obs; rho_opt(theta), sigma_rho)
           * N(ln Mc_obs;  ln Mc_det(theta), sigma_lnMc(rho_obs))
           * N(ln q_obs;   ln q(theta),      sigma_lnq(rho_obs))
           * N(chieff_obs; chieff,           sigma_chi(rho_obs))
           * N(dec_obs;    dec,              sigma_ang(rho_obs))
           * N(ra_obs;     ra,               sigma_ra(rho_obs, dec_obs))
```

— every conditioning variable on the right of a `;` is either `theta` or data
already drawn, so this is a proper factorisation, and every width is data.

darksirens does not evaluate this expression; it reconstructs it from the stored
samples and `p_pe`, computing
`Zhat_i = (1/N) sum_s p_target(theta_s)/p_pe(theta_s)` with
`theta_s ~ pi_PE * L / Z`.  That equals `INT p_target L` — the exact evidence —
**iff** the samples really are draws from `pi_PE L` and `p_pe` really is `pi_PE`
in the same basis.  §2.4 and §2.5 are exactly those two statements, and V3/V3c
test them.

**This is the point at which the naive reading of the brief fails and had to be
adapted.**  If `rho_obs` were drawn as `N(rho_opt(theta), 1)` *and* `dL` were
additionally measured through its own lognormal channel, then `rho_obs` would be
recorded data whose distribution depends on `theta`, and the factor
`N(rho_obs; rho_opt(theta), sigma_rho)` would be a `theta`-dependent piece of the
true likelihood that darksirens — which only ever sees `(m1det, q, dL, chieff,
sky)` samples and `p_pe` — cannot represent.  The posterior would be wrong and
`(C - A)` would not close.  The literature does not do this: `GWMockCat` derives
`dL` from `rho` rather than measuring it separately (§1.3).  Adopting that
structure makes the SNR channel *be* the distance channel, so nothing is
double-counted and nothing is dropped.  The realised distance precision is
unchanged in scale:

```
ln dL = const + (5/6) ln Mc_det - ln rho
sigma_lndL = sqrt( (5/6 * 0.64/rho)^2 + (1/rho)^2 ) = 1.133/rho
```

i.e. `14.2 %` at `rho = 8` and `11.3 %` at the detected median `rho ~ 10`,
against v2's flat `10 %`.

### 3.2 Detection is deterministic in the estimator's data

`1[rho_obs >= 8]` reads one stored number.  Because the widths are `a/rho_obs`,
`rho_obs` is also recoverable from any stored width, so conditioning on the PE
widths already conditions on the detection statistic.  There is no latent left
inside the detection decision, which is convention (a) in its strongest form.

The closed-form selection function becomes a single Gaussian CDF,

```
P_det(theta) = Phi( ( rho_opt(theta) - 8 ) / sigma_rho )
```

replacing v2's two-dimensional Gauss-Hermite average over the mass-noise latents.
`mu(H0)` therefore reduces, exactly as before, to a one-dimensional kernel:

```
rho_opt = a(m1src, q) * b(z, H0),
  a = SNR_REF_DETECT * (Mc_src/30)^(5/6),
  b = 1000 (1+z)^(5/6) / dL(z;H0)     and   dL(z;H0) = (H0_fid/H0) dL(z;H0_fid)
=> b(z,H0) = b(z,fid) * H0/H0_fid  EXACTLY
=> F(z;H0) = G(b),   G(b) = E_a[ Phi((a b - 8)/sigma_rho) ],
   d ln mu/dH0 = < G'(b) b > / < G(b) > / H0
```

with `G'` from the same construction with the normal PDF in place of the CDF.
`attr_selmu_oracle.py` carries this as its `--pe_model v3` kernel; every anchor
of `CLOSURE.md` §11.3 is re-run against it.

### 3.3 The prior matches the truth — the D3 remedy

The remaining requirement is `p_target = p_true`.  The mass/spin channels already
match to `1.3e-8` (`ATTRIBUTION.md` A1) and the host-acceptance convention matches
(`CLOSURE.md` §13).  The one measured mismatch was the redshift channel:
`CLOSURE.md` §15.4 showed the survey block declares `dz = 3e-3 (1+z)` on
redshifts copied **bit-for-bit** from the catalog the hosts are drawn from, so
the model smooths a comb that carries no error — a `7.6 sigma` effect in
`(A - B)_pz`.

v3 **realises the declared error**.  The catalogs now carry two redshift columns:

```
z        the TRUE redshift  (GLASS; the host draw and the event's truth)
z_obs    z + N(0, DZ_SCALE (1+z))     DZ_SCALE = 3e-3, sub-seed "photoz"
```

and `stage_surveys` pixelates on **`z_obs`** with the declared width
`dz = DZ_SCALE (1+z_obs)`.  The generative statement is then consistent:

* hosts are drawn from the true catalog and the event happens at `z_true`;
* the survey — and therefore the likelihood — sees only `z_obs`;
* darksirens' per-galaxy kernel is `g(z) N(z; z_obs_g, sigma_g) / Z(z_obs_g)`
  with `Z(z_obs_g) = INT g(z) N(z; z_obs_g, sigma_g) dz`
  (`CLOSURE.md` §13), which is **exactly** the Bayesian posterior for that
  galaxy's true redshift given its catalog entry, under the prior `g(z)` —
  and `g(z) ~ dV_c/dz` is the mock's own galaxy redshift distribution
  (constant comoving density).

So `p_z(z|pix)` is the correct prior for the host's true redshift given the
catalog, to `O(sigma_z^2)`: the only residual is that the truth used
`dz = DZ_SCALE(1+z_true)` while the block declares `dz = DZ_SCALE(1+z_obs)`, a
relative width mismatch of `O(DZ_SCALE) = 3e-3`, and that the within-pixel
clustering makes `g(z)` only approximately the local prior.  Both are second
order against the `+5.8e-4` the mismatch was worth.  Check **V9** measures the
realised scatter `z_obs - z_true` against the declaration.

`z_obs` may in principle be negative for a galaxy at `z ~ 0`; the realised
minimum catalog redshift is `1.26e-3` (GAL) with only **3** of 151,179,870 rows
below `z = 0.005`, so the expected number of negative entries is `<= 1`.  They are
**not clipped** (clipping would re-introduce censoring); the realised count is
recorded in `META.json` and in V9.

### 3.4 What is *not* claimed

Two approximations of the catalog prior survive v3 and are unchanged, both
already measured and both consistent with zero on matched GAL:

* the **nside-32 pixelisation** (`+9.06e-5 +- 1.95e-4`, `CLOSURE.md` §7), i.e.
  sky and redshift independent inside a `1.83 deg` pixel;
* the finite **PE Monte Carlo** at `nsamp = 2000` (2-3 % of `r` in v2).

And the selection integral's estimator carries a common-mode Monte-Carlo error on
`d ln mu/dH0` (`CLOSURE.md` §14.2) which is **carried explicitly** in every number
quoted for v3 rather than left out.

---

## 4. Implementation map

| where | what |
|---|---|
| `generate_dataset.py` `PE_MODEL` | `"v3"`; `--pe_model {v2,v3}` selects, default `v3` |
| `SIGMA_RHO, A_MC, A_Q, A_CHI, SKY_A_DEG, SNR_REF_SIGMA` | the constants of §2 |
| `rho_opt`, `mc_q_to_m1m2`, `m1m2_to_mc_q`, `dl_from_mc_rho`, `rho_from_mc_dl` | the bijection of §2.1 |
| `observe_v3` | steps 2-6 of §2.2 |
| `detect_v3` | `rho_obs >= SNR_THRESHOLD` |
| `posterior_samples_v3` | §2.4 + the map to storage basis |
| `p_pe_v3` | §2.5 closed form |
| `stage_catalogs` | the `z_obs` column, sub-seed `photoz` |
| `stage_surveys` | pixelates on `z_obs` with `dz = DZ_SCALE (1+z_obs)` |
| `stage_injections` | same detection rule; `observe_v3(need_sky=False)` is just the `rho_obs` draw |
| `stage_validation` | V1/V2/V2b/V3 rewritten for v3; V3c (`p_pe` Jacobian) and V9 (photo-z) are new |
| `analyses/.../scripts/attr_selmu_pdet.py` | `--pe_model v3`: `P_det = Phi((rho_opt-8)/sigma_rho)` |
| `analyses/.../scripts/attr_selmu_oracle.py` | `--pe_model v3`: the `G(b)` kernel of §3.2 |
| `analyses/.../scripts/build_catalog_skyindex.py` | lexsorts on `z_obs`, the survey block's own key |

---

## 5. Citations

* Fishbach, M., Holz, D. E. & Farr, W. M. 2018, ApJL 863, L41, arXiv:1805.10270 —
  the `rho_obs = rho + N(0,1)`, `rho_obs >= 8`, `8/rho_obs` width family.
* Fishbach, M. & Holz, D. E. 2020, ApJL 891, L31, arXiv:1905.12669, Appendix B —
  the "general framework" `GWMockCat` is built on; the `~50 %` 90 % component-mass
  statement used to calibrate `A_Q`.
* Farah, A. M., Edelman, B., Zevin, M., Fishbach, M., Maria Ezquiaga, J.,
  Farr, B. & Holz, D. E. 2023, ApJ 955, 107, arXiv:2301.00834, Appendix A —
  `GWMockCat`; the adopted constants `sigma_rho = 1`, `alpha_Mc = 0.08`,
  `alpha_eta = 0.022`, `alpha_chi = 0.2`, `rho_th = 8`, and the structure in which
  `dL` is derived from `rho`.
  Code: <https://git.ligo.org/amanda.farah/GWMockCat> (CC0), `GWMockCat/posterior_utils.py`,
  `GWMockCat/parser.py`, `GWMockCat/transforms.py`.
* Borhanian, S. 2021, CQG 38, 175014, arXiv:2010.15202 — `gwbench`; the
  Fisher-matrix `1/rho` family, consulted for the form and rejected as a
  *generator* width source because a Fisher matrix is evaluated at the latent.
* Abbott, B. P. et al. (LVC) 2016, PRL 116, 061102, arXiv:1602.03840 — GW150914
  `q = 0.86 (+0.14/-0.21)` at `rho ~ 24`, the `A_Q` anchor.
* Fairhurst, S. 2009, NJP 11, 123006, arXiv:0908.2356; Berry, C. P. L. et al.
  2015, ApJ 804, 114, arXiv:1411.6934 — sky-area `~ rho^-2` scaling behind the
  retained `35 deg/rho` convention.
* Ng, K. K. Y. et al. 2018, arXiv:1805.03046 — the `psi` reparametrisation
  `GWMockCat` uses for `chi_eff`; **not** adopted here (see §2.2: v3 measures
  `chi_eff` directly with an unbounded Gaussian and truncates only the prior).

---

## 6. Realised on seed 100 (the first v3 dataset)

`/hildafs/projects/phy220048p/magana/gws-agn-data-v3/seed100`, generated
2026-08-01, **12/12 validation checks pass**.

| quantity | v2 (the previous record) | v3 |
|---|---|---|
| `rho_obs` min / median / max | 8.003 / 10.11 / 199.3 | 8.003 / **10.03** / 240.3 |
| `sigma_lnMc` median | (not a channel) | **0.0638** |
| `sigma_lnq` median | (not a channel) | **0.478** |
| `sigma_ln dL` median | 0.100 (flat) | **0.1137** (`= 1.13/rho`) |
| `sigma_ln m1det` median | 0.080 | **0.177** |
| `sigma_ang` | `[1.00, 2.39] deg` | **`[1.00, 2.39] deg`** (unchanged) |
| PE samples with `q > 1` | **18.4 %** | **0 %** |
| horizon `z_max(detected)` | 0.3565 | 0.3105 |
| detected fraction | 7.605e-3 | 7.850e-3 |
| events GAL / AGN | 720 / 280 | 705 / 295 |
| max PE redshift over `H0 in [50,100]` (bar 0.700) | 0.655 | **0.652** |
| targeted injections detected | 2,095,518 / 1.5e8 | 2,205,380 / 1.5e8 |
| popuni injections detected | 1,175,596 / 4.0e8 | 1,230,471 / 4.0e8 |

The measurement model's own certificates, from `--stage validation`:

| check | result |
|---|---|
| **V1** the detection rule | `rho_obs` is one stored number; every detected event clears 8, every stored rejection fails it; `rho_true` recomputed from the truth to `< 1e-12`; and the *truncated* pull `(rho_obs − rho_opt)/sigma_rho` — a normal truncated at `(8 − rho_opt)` on the detected set — has a **uniform PIT, KS `p = 0.61`** |
| **V2** widths from data | `sigma_lnMc`, `sigma_lnq`, `sigma_chieff`, `sigma_ang` all recomputed from the stored `rho_obs` alone, **bitwise** |
| **V2b** the RA width | `sigma_ang/max(cos dec_obs, 0.1)` bitwise; measurement pulls `N(0,1)` |
| **V3** the PE | the stored measurement-basis columns are the exact bijection of the storage basis (`<= 4.4e-16`); pooled PIT/KS `p` = 0.92 (`ln Mc`), 0.50 (`ln q`), 0.75 (`rho`), 0.32 (`chieff`), 0.49 (`dec`), 0.65 (RA pull); measurement-side pull sd = 1.004 / 1.010 / 0.957 |
| **V3c** the `p_pe` Jacobian | the stored column equals the closed form `rho/(dL m1det q)` **exactly** (`0.0`), and that closed form equals an independent **numerical** Jacobian to `6.2e-10` |
| **V3b** the generative replay | `3e6` fresh proposals through the same `observe_v3`/`detect_v3`: SNR-noise KS `p = 0.40`, detected-set two-sample KS `p = 0.55` (`rho` pull) and `0.48` (`z`); and **`P_det = Phi((rho_opt − 8)/sigma_rho)` against brute force: `0.0082793` vs `0.008281`, a `+0.03 sigma` binomial pull** |
| **V9** the realised photo-z | the survey block's `zgals` are the catalog's `z_obs` **bitwise**, are **not** the true redshifts (max `|Δ| = 0.0276` GAL, `0.0126` AGN), the declared `dzgals` equal `DZ_SCALE (1+z_obs)` **bitwise**, and the realised pull has sd `0.99996` / `0.99958` with KS `p = 0.65` / `0.87`.  One catalog row of 151,179,870 has `z_obs < 0` (`−3.9e-4`); it is **not** clipped |
| **V8** the catalog edge | max PE redshift over the scanned `H0` range **0.652** against the `0.7 z_max = 0.700` bar |

and the exact selection function, whose closed form is now a single Gaussian CDF:

* `attr_selmu_pdet.py --pe_model v3` against the generator's own
  `observe_v3`/`detect_v3` on 30 points spanning `P_det ∈ [0.003, 0.999]` at
  `2e7` draws each: **max `|P_MC − P_exact| = 1.09e-4`** (the Monte-Carlo error
  itself), max pull 2.54, **mean pull `−0.16 ± 0.20`**;
* `attr_selmu_oracle.py --pe_model v3`: the `G(b)` kernel reproduces a direct
  unbinned sum over the `(m1src, q)` grid to `1e-7` where the host measure lives;
  the catalog lattice halves to `4.3e-10` (GAL) / `1.3e-9` (AGN); the analytic
  derivative and the `dh`-halving finite difference agree to `1.3e-8`; and every
  anchor of `CLOSURE.md` §11.3 holds (`|N_obs − ngals| ≤ 1.3e-11`,
  `|Z_global − Ntot| ≤ 6.0e-8`, `log_kw` table vs darksirens `≤ 3.0e-7`).
