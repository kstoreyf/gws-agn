#!/usr/bin/env python3
"""THE campaign dataset: one generative model, one master seed, every lesson baked in.

This is the single entrypoint that produces the dataset every subsequent gws-agn
analysis reads.  It is seed-parameterised (``--seed``); one master seed fixes every
sub-stream, and all outputs for that seed live under ``seed<SEED>/``.

THE GENERATIVE MODEL (executed in this order)
=============================================
1.  ``catalogs``   GLASS lognormal two-tracer catalogs on ONE shared density field
                   (GAL b=1.2, AGN b=2.0) at LITERATURE-ANCHORED comoving number
                   densities (1e-3 / 1e-5 Mpc^-3, see SCHECHTER / AGN_LF) out to
                   z_max = 1.0, plus per-object absolute/apparent magnitudes drawn
                   from the same Schechter function so a flux limit can act later.
2.  ``events``     GW sources injected on the COMPLETE catalogs, hosts drawn from
                   the planted mixture ``(1-f) GAL + f AGN``; masses/spins from the
                   powerlaw+peak fiducials; ONE observed measurement per source;
                   detection and PE both built from THAT measurement.
3.  ``surveys``    Isotropic apparent-magnitude limits -> incomplete catalog pairs,
                   then HEALPix pixelation of every catalog into darksirens survey
                   files.
4.  ``injections`` Two selection campaigns under the SAME detection rule: a
                   catalog-targeted three-branch mixture and a plain
                   population+uniform cross-check lane.
5.  ``validation`` Every required check, written to ``validation/`` -- FAILS LOUDLY.
                   V8 additionally requires the events' PE redshift support, mapped
                   through z(dL; H0) over the WHOLE scanned H0 range, to stay inside
                   0.7 * z_max of the catalog.

THE v3 MEASUREMENT FAMILY (2026-08-01)  --  see working/data/DESIGN_PE.md
========================================================================
``--pe_model v3`` (the default) replaces the previous family, in which the mass
widths were ``f * m_TRUE`` -- functions of the LATENT parameter.  No measurement
model in that family closes the detected-set score identity: with the exact
per-event posterior, exact host positions, the mock's own host prior and the exact
selection function, ``(C - A)`` in the mass channel was ``-1.274e-3 +- 0.113e-3``,
an 11.3 sigma violation (CLOSURE.md 15).  v3 is the literature-standard
all-observable family:

    rho_obs = rho_opt(theta) + N(0, 1)              DETECTION: rho_obs >= 8
    sigma_lnMc  = 0.08 * (8/rho_obs)                ln Mc_obs ~ N(ln Mc_det, .)
    sigma_lnq   = 0.60 * (8/rho_obs)                ln q_obs  ~ N(ln q,      .)
    sigma_chi   = 0.20 * (8/rho_obs)                chi_obs   ~ N(chi,       .)
    sigma_ang   = clip(35 deg / (1.83165 rho_obs), 1, 12)   dec_obs, then ra_obs
    PE = the EXACT flat-prior posterior in (ln Mc_det, ln q, rho, chieff, ra, dec),
         truncated ONLY in the PRIOR (q <= 1, rho > 0, |chi| <= 1, |dec| <= pi/2);
         p_pe = rho/(dL m1det q) in the canonical (m1det, q, dL, chieff) basis.

EVERY width is a function of OBSERVED data, so the generative likelihood is exactly
invertible and the identity closes by construction.  dL is NOT measured separately:
with no projection latent (convention (a)) ``rho_opt`` is an exact function of
(Mc_det, dL), so the SNR IS the distance observable -- GWMockCat's own construction
(Farah et al. 2023, ApJ 955, 107; arXiv:2301.00834 App. A).  Recording rho_obs AND
measuring dL would leave a theta-dependent factor N(rho_obs; rho_opt(theta), 1) that
darksirens cannot represent.  Constants: GWMockCat / Fishbach, Holz & Farr (2018,
arXiv:1805.10270) eqs. 29-31.  NO recorded value is ever clipped -- clipping the
DATA censors the likelihood; the physical ranges live on the PE PRIOR instead.

D3 -- THE DECLARED PHOTO-Z IS NOW REALISED
==========================================
The catalogs carry ``z_obs = z + N(0, DZ_SCALE (1+z))`` and the survey blocks
pixelate ``z_obs`` with the declared width ``dz = DZ_SCALE (1+z_obs)``.  ``z`` (the
true redshift) still drives the host draw and the event's truth.  Before this the
block declared ``dz = 3e-3 (1+z)`` on redshifts copied bit-for-bit from the catalog
the hosts were drawn from -- a kernel on redshifts carrying no error at all, and a
7.6 sigma ``(A - B)`` violation (CLOSURE.md 15.4).  Gated by validation V9.

THE THREE NON-NEGOTIABLE CONVENTIONS
====================================
These are the campaign's three hard-won lessons.  Each was a measured, multi-sigma
bias in an earlier mock; each is now structural.  (b)/(c) below are stated in their
v2 form; v3 strengthens both -- in v3 EVERY width, not only the sky width, is a
function of the recorded data.

(a) DETECTION IS A DETERMINISTIC FUNCTION OF THE OBSERVED DATA.
    ``rho_obs = snr_ref * (Mc_det_obs/30)^(5/6) * (1000 Mpc / d_obs) >= 8``, computed
    from the SAME recorded measurement the posterior conditions on.  No true-redshift
    cut, no projection latent, no separate noise draw for the PE.
    WHY: a population likelihood evaluates ``prod_i [int p(d_i|th) p(th|L) dth] / mu^N``,
    which is the detected-set likelihood only when ``1[det(d_i)] = 1`` on the observed
    set.  A latent-dependent detection rule leaves an extra ``P(det|th)`` INSIDE each
    event integral that no population code evaluates.  Measured cost of getting this
    wrong: H0 biased -1.57 +- 0.18 km/s/Mpc (8.5 sigma) instead of -0.80 +- 0.16.

(b) THE MEASUREMENT WIDTHS COME FROM THE OBSERVED DATA, SEQUENTIALLY.
    Measure dL and the masses FIRST, then set ``sigma_ang = clip(35/rho_opt(observed
    values), 1, 12) deg``, and only then draw the sky offsets.
    WHY: ``sigma_ang ~ dL / Mc_det^(5/6)`` is itself an H0-sensitive observable.
    Freezing it at its LATENT true value makes the recorded sky width carry distance
    information the fixed-width sky posterior cannot represent, breaking the
    detected-set score identity.  Measured cost: -0.49 +- 0.08 km/s/Mpc even under the
    EXACT likelihood (darksirens PR #335).

    (b2) THE SAME RULE INSIDE THE SKY PAIR: ``dec`` IS MEASURED BEFORE ``ra``.
    The RA offset is drawn with width ``sigma_ang / max(cos dec, 0.1)``, and the
    ``dec`` that enters it must be the one already RECORDED, not the latent one.
    So ``dec_obs`` is drawn first and ``sig_ra = sigma_ang / max(cos dec_obs, 0.1)``
    is a deterministic function of stored data, recomputable from the file and
    stored as ``obs_sig_ra``; the PE then uses that same number.  Under the earlier
    (pre-2026-08-01) convention ``observe()`` used ``cos dec_TRUE`` while
    ``posterior_samples()`` used ``cos dec_obs``, so the recorded RA posterior width
    was wrong by ``|cos dec_obs / cos dec_true - 1|`` = 2.3 % mean / 4.6 % rms /
    54 % max on seed 100 -- the sky twin of the mass defect in (c2), and the one
    place convention (b) was not actually honoured.  See ATTRIBUTION.md A4.5/A5.2.

(c) PE SAMPLES ARE THE EXACT FLAT-PRIOR POSTERIOR OF THAT MEASUREMENT.
    For multiplicative distance noise ``ln d_obs ~ N(ln dL, s)`` the flat-in-dL
    posterior is EXACTLY ``ln dL ~ N(ln d_obs + s^2, s)`` -- lognormal about the
    OBSERVATION, shifted +s^2 by the volume factor.
    WHY: clouds centred on TRUTH stored with ``p_pe = 1`` are mislabelled as
    flat-prior posteriors and inject an O(sigma^2) distance-scale (hence H0) bias.
    Measured cost: -1.14 km/s/Mpc at sigma_dL = 0.10, scaling as sigma^2
    (darksirens PR #332).

    (c2) THE MASS CHANNELS ARE *NOT* GAUSSIAN ABOUT THE OBSERVATION.
    The realised mass measurement is ``obs ~ N(m, f m)`` with ``f`` CONSTANT, so the
    width is proportional to the LATENT mass and the flat-prior posterior is
        p(m | obs)  ~  (1 / (f m)) exp[ -(obs - m)^2 / (2 f^2 m^2) ],
    which is skewed: its mean sits ``+2 f^2`` (+1.32 % at f = 0.08) ABOVE ``obs``.
    Storing instead a fixed-width Gaussian about ``obs`` -- and, worse, one whose
    width ``f * m_true`` is computed from the LATENT mass -- mislabels the PE and
    biases the spectral-siren mass channel.  Measured cost on the matched-GAL
    control: 39.5 % of the per-event score residual ``r`` and +2.15 km/s/Mpc
    (ATTRIBUTION.md A2/A3).  Note the *minimal* repair (Gaussian of width ``f*obs``)
    is measurably WORSE: it is the O(f^2) SHAPE, not the width, that matters.
    The exact posterior is drawn here by inverse CDF -- the additive-noise branch of
    the same machinery ``code/generate_gwsamples.py`` provides, and the direct
    analogue of what PR #335 did for ``sigma_ang``.

REUSE / ATTRIBUTION
===================
Adapted from validated campaign components (each read before adapting):
  * ``code/make_mocks.py``                       GLASS two-tracer catalog construction
  * ``code/generate_gwsamples.py``               exact flat-prior PE, pe_centering=obs
  * ``code/pixelize_catalogs.py``                survey-file pixelation
  * ``experiments/experiment_matched_mock/scripts/build_obsdet_mock.py``
        the observed-data detection rule (a) and the sequential sky width (b)
  * ``experiments/experiment_twotracer_seeds/scripts/build_targeted_injections_k2.py``
        the catalog-targeted injection mixture with exact pdraw
  * ``experiments/experiment_completeness_anchored/scripts/pixelate_complete_catalog.py``
    and ``experiment_twotracer_incomplete/scripts/materialise_tracer_catalogs.py``
        the isotropic magnitude-limit machinery
The population samplers, cosmology grids, ``_selection_pdraw`` and ``_pixelate_catalog``
are IMPORTED from darksirens' own ``generate_mock_data.py`` so the mock is the
inference's model by construction.  The pinned checkout and its SHA are recorded in
``META.json``.

ENVIRONMENTS
============
``glass`` needs numpy >= 2 (it calls ``np.trapezoid`` and the array-API
``__array_namespace__``), while the project ``jax`` env is pinned at numpy 1.26 /
scipy 1.12 -- upgrading it would break scipy and the whole inference stack.  So stage
``catalogs`` re-executes THIS FILE under a dedicated venv (``--glass_python``); every
other stage runs in the project env.  Both interpreters and every package version are
recorded in ``META.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

# --------------------------------------------------------------------------------
# Fiducials.  These are the campaign's numbers; changing one changes the dataset.
# --------------------------------------------------------------------------------
# --- GLASS density field / tracers (code/make_mocks.py, configs/data_glass_prod_*) --
# v2: the tracer amplitudes are COMOVING NUMBER DENSITIES with a literature anchor
# (see SCHECHTER / AGN_LF below), converted to GLASS's per-arcmin^2 `ngal` convention
# inside _glass_build so the realised catalogs have constant comoving density.
GLASS = dict(
    n_comoving_gal=1.0e-3,   # Mpc^-3 comoving  (bright, L >~ L*, see SCHECHTER)
    n_comoving_agn=1.0e-5,   # Mpc^-3 comoving  (luminous AGN, see AGN_LF)
    bias_gal=1.2,
    bias_agn=2.0,
    z_min=0.0,
    z_max=1.0,           # edge well past the GW data; enforced by validation V8
    nside=128,           # density-field resolution (v1: 64)
    lmax=256,            # v1: 64 -- too low to resolve the planted bias contrast
    dx_mpc=200.0,        # comoving shell thickness
    h=0.6774,
    Om0=0.3075,
    Ob0=0.0486,
)

# --- galaxy luminosity function: the GLADE-lineage B-band Schechter function -----
# GLADE (Dalya et al. 2018, arXiv:1804.05709) / GLADE+ (Dalya et al. 2022,
# arXiv:2110.06184) use phi* = 1.6e-2 h^3 Mpc^-3, alpha = -1.07, M_B* = -20.47.
#
# ARITHMETIC (computed in _schechter_number_density / _schechter_cut, recorded in
# META): with h = 0.7, phi* = 1.6e-2 * 0.7^3 = 5.488e-3 Mpc^-3 and
#     n(> x L*) = phi* * Gamma(alpha + 1, x),      Gamma = UPPER incomplete gamma.
# The brief's nominal cut x = 0.25 gives Gamma(-0.07, 0.25) = 1.08383 and
#     n(> 0.25 L*) = 5.948e-3 Mpc^-3,
# which is 5.9x the intended 1e-3 Mpc^-3 and would put 9.8e8 galaxies inside
# z <= 1 -- ~20 GB of catalog plus ~15 GB of survey blocks, over the storage budget
# (34 GB free on /hildafs) by several times.  The campaign's binding number is the
# DENSITY (1e-3 Mpc^-3, ~1.6e8 rows to z = 1), so the cut is solved FROM it on the
# same luminosity function:
#     n(> x L*) = 1e-3  =>  Gamma(-0.07, x) = 0.182216  =>  x = 1.0908,
# i.e. the sample is "L > 1.09 L*", M_B < -20.564 -- the classic L* bright-galaxy
# sample, which is what a 1e-3 Mpc^-3 host catalog physically is.
SCHECHTER = dict(phi_star_h3=1.6e-2, h_for_phi_star=0.7, alpha=-1.07,
                 M_B_star=-20.47,
                 reference="GLADE arXiv:1804.05709 / GLADE+ arXiv:2110.06184")

# --- AGN space density anchor ----------------------------------------------------
AGN_LF = dict(
    n_comoving=1.0e-5,
    definition="luminous AGN, log10 L_X(2-10 keV) >~ 43.7 erg/s",
    reference="Swift-BAT/BASS X-ray luminosity function lineage (Ananna et al.; "
              "BASS DR2). Integrated space density of the luminous class; adopted "
              "as a pinned constant here rather than re-integrated.",
    note="ratio n_gal/n_agn = 100, the same contrast v1 planted arbitrarily.")

# --- analysis / GW cosmology (pinned; the inference's truth) ---------------------
H0_FID = 67.74
OM0_FID = 0.3075
W0_FID = -1.0
WA_FID = 0.0
ZMAX_GRID = 2.0          # cosmology-grid / injection-proposal depth

# --- GW population + measurement model ------------------------------------------
GAMMA = 0.0              # host acceptance weight (1+z)^(gamma-1)
F_AGN = 0.30             # planted AGN-hosted fraction
N_EVENTS = 1000
N_SAMP = 2000
SNR_THRESHOLD = 8.0
# Amplitude scale of the DETECTION statistic in observed-data mode.  Calibrated in
# experiment_matched_mock so the observed-data rule reproduces the control rule's
# detection fraction after the projection latent is dropped (horizon z ~ 0.27).
SNR_REF_DETECT = 6.278363879917771
# Amplitude scale of the SKY-WIDTH model.  This is gmd's own rho_ref and is the scale
# sigma_ang has always used; it is NOT the detection scale.
SNR_REF_SIGMA = 11.5
SIGMA_DL = 0.10          # fractional, multiplicative (lognormal)   [v2 only]
SIG_M1_FRAC = 0.08                                                 # [v2 only]
SIG_M2_FRAC = 0.10                                                 # [v2 only]
SIGMA_CHIEFF = 0.08                                                # [v2 only]
CHIEFF_AMAX = 0.99

# --- v3 measurement family (2026-08-01) ------------------------------------------
# THE literature-standard mock-PE family; see working/data/DESIGN_PE.md for the
# full derivation, the citations and the calibration of every constant.
#
#   rho_obs = rho_opt(theta) + N(0, SIGMA_RHO)        detection: rho_obs >= 8
#   sigma_x = A_x * (SNR_THRESHOLD / rho_obs)         for every other channel
#
# and the measurement basis is (ln Mc_det, ln q, rho, chieff, ra, dec), which is a
# BIJECTION of (m1det, m2det, dL, chieff, ra, dec) because the campaign forbids a
# projection latent, so rho_opt is an exact function of (Mc_det, dL).  dL is
# therefore NOT measured on its own: the SNR IS the distance observable, exactly as
# in GWMockCat (Farah et al. 2023, ApJ 955, 107, arXiv:2301.00834, App. A;
# git.ligo.org/amanda.farah/GWMockCat, transforms.py::redshift).  Measuring dL
# separately AND recording rho_obs would leave a theta-dependent likelihood factor
# N(rho_obs; rho_opt(theta), sigma_rho) that darksirens cannot represent, and the
# detected-set score identity would not close -- see DESIGN_PE.md 3.1.
PE_MODEL_DEFAULT = "v3"
SIGMA_RHO = 1.0          # GWMockCat uncert_default["snr"]  (Fishbach+2018 eq. 29)
A_MC = 0.08              # GWMockCat uncert_default["mc"]:  sigma_lnMc = 0.08 * 8/rho
A_Q = 0.60               # sigma_lnq = 0.60 * 8/rho; calibrated in DESIGN_PE.md 2.3
A_CHI = 0.20             # GWMockCat --Xeff_uncert:         sigma_chi  = 0.20 * 8/rho
SKY_A_DEG = 35.0         # the campaign's own sky constant, retained verbatim:
#                          sigma_ang = clip(SKY_A_DEG / rho_sigma, 1, 12) deg with
#                          rho_sigma = (SNR_REF_SIGMA/SNR_REF_DETECT) * rho_obs
SKY_CLIP_DEG = (1.0, 12.0)
COS_DEC_FLOOR = 0.1      # the RA-width floor of convention (b2), unchanged
CHIEFF_RANGE = (-1.0, 1.0)     # PE PRIOR support (never a clip on the DATA)
# The catalog's declared photo-z error is now REALISED (D3 / DESIGN_PE.md 3.3):
# catalogs carry z_obs = z + N(0, DZ_SCALE (1+z)) and the survey blocks pixelate on
# z_obs with dz = DZ_SCALE (1+z_obs).  PHOTOZ_SURVEY_DEFAULT = "obs" is the fix;
# "true" reproduces the pre-2026-08-01 (dishonest) convention for regression only.
PHOTOZ_SURVEY_DEFAULT = "obs"

# --- exact flat-prior mass posterior, convention (c2) ----------------------------
# In y = obs/m the posterior p(m|obs) ~ (1/(f m)) exp[-(obs-m)^2/(2 f^2 m^2)] becomes
#     p(y) ~ (1/y) N(y; 1, f)          -- INDEPENDENT of obs,
# so ONE quantile table per f serves every event and m = obs / Q(u).  The 1/y factor
# is a log-divergence at y -> 0 (the 1/m tail of the posterior), whose coefficient is
# exp(-1/(2 f^2)) = 1.1e-34 (f = 0.08) / 1.9e-22 (f = 0.10) per e-fold, so the tail is
# proper for every reachable cap; PEX_Y_CAP truncates it at m = 40 * obs and PEX_N_SIG
# at m = obs/(1 + 12 f).  Convergence of the table is CHECKED in stage validation
# (grid doubled, cap x10, n_sig 12 -> 16): median |dQ/Q| = 4e-12, max 2e-9 over the
# u range 2e6 samples per seed can actually reach.
PEX_N_GRID = (1 << 21) + 1
PEX_Y_CAP = 1.0 / 40.0
PEX_N_SIG = 12.0

# --- EM survey model -------------------------------------------------------------
# v2: absolute magnitudes come from the SAME Schechter function the density anchor
# uses, truncated at the cut, so a flux limit acts on a physical luminosity
# distribution instead of the v1 placeholder N(-21, 1).
MAG_LIMITS = (21.0, 20.0, 19.0, 18.0)

# --- storage ---------------------------------------------------------------------
# CAT_DTYPE is THE storage dtype of the catalog and survey columns.  It is a single
# config constant: every catalog column, every pixelated survey block, and every
# in-memory catalog load (`load_catalog`) is written and read at this precision.
#
# It is **float64** since 2026-07-31, and the reason is not precision but
# evaluability.  darksirens' general incomplete-catalog likelihood `dark_sirens`
# builds its observed-density KDE in `darksirens.redshift.completion._kde_dndz_obs`,
# where the truncated-kernel mass
#     mass = ndtr((5 - z)/0.05) - ndtr(-z/0.05);   mass = max(mass, 1e-300)
# is evaluated in the CATALOG'S storage dtype while the kernel itself is promoted to
# the package zgrid's float64.  The survey blocks pad short pixel rows at z = 100, so
# for every padded slot `mass` underflows to exactly 0 -- and 1e-300 is not
# representable in float32, so the clamp cannot rescue it.  The float64 kernel is 0
# there too, 0/0 = NaN, and the `* real_gal` mask cannot remove a NaN.  Every catalog
# row carrying any padding comes back all-NaN, the NaN reaches the survey-global field
# normalizer, and `dark_sirens` returns -inf in every cell of every grid.  In float64
# the clamp holds and the model runs.  `dark_sirens_complete` never touches that KDE,
# which is why the float32 dataset was fine for the complete-catalog-only campaign and
# stopped being fine the moment one nested likelihood had to carry every run.
# See working/experiments/experiment_model_equivalence/README.md.
#
# The precision itself was never the constraint: float32's 6e-8 relative precision is
# five orders below the KDE width dz = 3e-3 (1+z), and the two dtypes give the same
# science (`dark_sirens_complete` moves by <= 2e-7 km/s/Mpc in the posterior median).
# The cost of float64 is disk: ~2.5x on the catalogs, ~2.5x on the surveys.
CAT_DTYPE = "float64"

# --- survey pixelation -----------------------------------------------------------
NSIDE_SURVEY = 32
DZ_SCALE = 3.0e-3        # dz = DZ_SCALE * (1 + z); the validated kernel width
Z_PAD, DZ_PAD, W_PAD = 100.0, 1.0, 0.0   # gmd _pixelate_catalog padding sentinels

# --- selection campaigns ---------------------------------------------------------
NDRAW_INJ = 120_000_000
MIX_POPULATION = 0.65
MIX_UNIFORM = 0.10
MIX_TARGETED_AGN = 0.25
PDRAW_FLOOR = 1.0e-300

# --- the targeted branch's H0-range-covering distance kernel ---------------------
# The scan runs H0 over [H0_SCAN_MIN, H0_SCAN_MAX].  For flat wCDM with w0 = -1 and
# Om0 pinned, dL(z; H0) = (H0_FID/H0) dL(z; H0_FID) EXACTLY, so an injection stored
# at true redshift z is re-read by the likelihood at trial H0 as the redshift z'
# with dL_fid(z') = dL_fid(z) * H0/H0_FID.  A branch that plants draws AT the
# catalog redshifts therefore only lines up with the catalog prior at H0 = H0_FID --
# which is exactly what made v1's AGN N_eff collapse away from the fiducial.
#
# v2 widens the branch in redshift instead: for host j the branch draws z UNIFORMLY
# over
#     [ R_LO * (z_j - NSIG_PAD sigma_j) ,  min(Z_TGT_CAP, R_HI * (z_j + NSIG_PAD sigma_j)) ]
# with R_LO = H0_FID/H0_SCAN_MAX and R_HI = H0_FID/H0_SCAN_MIN, so the image of the
# branch under EVERY trial H0 in the scanned range still covers the host's kernel.
# The density is a flat box, hence exact in closed form and cheap to evaluate; the
# other two branches (population, uniform) have full support, so the cap and the
# host cut cannot open a hole in pdraw.
H0_SCAN_MIN, H0_SCAN_MAX = 50.0, 100.0
TGT_R_LO = H0_FID / H0_SCAN_MAX
TGT_R_HI = H0_FID / H0_SCAN_MIN
TGT_NSIG_PAD = 4.0        # kernel padding, in units of the host's sigma_z
TGT_HOST_ZMAX = 0.50      # hosts deeper than this can never land inside the horizon
TGT_Z_CAP = 0.50          # upper cap on the box (draws above it are never detected)

DARKSIRENS_REPO = Path("/hildafs/projects/phy230014p/magana/src/darksirens")

STAGES = ("catalogs", "events", "surveys", "injections", "validation")
TRACERS = ("gal", "agn")


# ================================================================================
# seeds: one master seed, explicit auditable children
# ================================================================================
def sub_seeds(seed: int) -> dict[str, int]:
    """Every random stream in the dataset, derived from the master seed.

    Offsets 1-7 below are the RECORD's streams and are the only ones a default
    (no-flag) run ever touches.  Offsets 8+ are reserved for explicitly requested
    EXTRA event stages via ``--seed_events`` (analysis 0 uses 8 = pure-GAL,
    9 = pure-AGN); nothing in the generator derives them, so an extra draw can
    never collide with the record.
    """
    return {
        "glass_field": seed * 1000 + 1,     # GLASS matter field + tracer positions
        "magnitudes": seed * 1000 + 2,      # absolute magnitudes of both tracers
        "photoz": seed * 1000 + 7,          # catalog photo-z scatter (D3, v3)
        "events": seed * 1000 + 3,          # host draw, masses/spins, measurement, PE
        "injections_targeted": seed * 1000 + 4,
        "injections_popuni": seed * 1000 + 5,
        "validation": seed * 1000 + 6,      # subsampling inside the checks only
    }


def seed_dir(seed: int, root: Path | None = None) -> Path:
    return (root or HERE) / f"seed{seed}"


# ================================================================================
# small utilities
# ================================================================================
def _git(repo, *args):
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception as exc:                              # pragma: no cover
        return f"<unavailable: {exc}>"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _pkg_versions(mods):
    out = {}
    for name in mods:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "<no __version__>")
        except Exception as exc:
            out[name] = f"<not importable: {exc}>"
    return out


def _write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    _log(f"wrote {path}")


def _read_json(path: Path):
    return json.loads(Path(path).read_text())


def _merge_meta(seed: int, key: str, payload, root: Path | None = None):
    """Accumulate stage results into META.json without losing earlier stages."""
    meta_path = seed_dir(seed, root) / "META.json"
    meta = _read_json(meta_path) if meta_path.exists() else {}
    meta.setdefault("dataset", "gws-agn campaign dataset")
    meta.setdefault("master_seed", seed)
    meta.setdefault("sub_seeds", sub_seeds(seed))
    meta.setdefault("generator", str(Path(__file__).resolve()))
    meta.setdefault("stages", {})
    meta["stages"][key] = payload
    meta["last_updated_utc"] = _now()
    _write_json(meta_path, meta)
    return meta


def import_gmd(worktree: Path = DARKSIRENS_REPO):
    """darksirens' own mock generator: population samplers, grids, pdraw, pixelation."""
    gmd_dir = Path(worktree) / "scripts/mock_dark_sirens"
    if not (gmd_dir / "generate_mock_data.py").exists():
        raise SystemExit(f"generate_mock_data.py not found under {gmd_dir}")
    sys.path.insert(0, str(gmd_dir))
    import generate_mock_data as gmd            # noqa: E402
    return gmd


# ================================================================================
# the galaxy luminosity function  --  density anchor and magnitudes
# ================================================================================
def _upper_incomplete_gamma(a, x0):
    """Gamma(a, x0) = int_x0^inf t^(a-1) e^-t dt, valid for a <= 0 (scipy's
    gammaincc is not).  Quadrature, split at 1 where the integrand turns over."""
    from scipy.integrate import quad
    f = lambda t: t ** (a - 1.0) * np.exp(-t)          # noqa: E731
    v1, _ = quad(f, x0, 1.0, limit=400)
    v2, _ = quad(f, 1.0, 300.0, limit=400)
    return v1 + v2


def schechter_phi_star():
    """phi* in Mpc^-3 (the tabulated value is in h^3 Mpc^-3)."""
    return SCHECHTER["phi_star_h3"] * SCHECHTER["h_for_phi_star"] ** 3


def schechter_number_density(x_cut):
    """n(> x_cut L*) = phi* Gamma(alpha+1, x_cut)  [Mpc^-3]."""
    return schechter_phi_star() * _upper_incomplete_gamma(
        SCHECHTER["alpha"] + 1.0, float(x_cut))


def schechter_cut_for_density(n_target):
    """The luminosity cut x = L/L* whose Schechter integral is ``n_target``."""
    from scipy.optimize import brentq
    return float(brentq(lambda x: schechter_number_density(x) - n_target,
                        1.0e-3, 200.0, xtol=1.0e-12, rtol=1.0e-14))


def sample_schechter_abs_mag(rng, n, x_cut, n_grid=400_001, x_max=200.0):
    """Absolute magnitudes of a Schechter sample truncated at L > x_cut L*.

    Inverse-CDF on a log-spaced luminosity grid (the density x^alpha e^-x is smooth
    and monotone above the cut, so trapezoid + interp is exact to ~1e-10 here).
    M = M* - 2.5 log10(L/L*)."""
    a = SCHECHTER["alpha"]
    x = np.geomspace(float(x_cut), x_max, n_grid)
    p = x ** a * np.exp(-x)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
    cdf /= cdf[-1]
    u = rng.uniform(size=n)
    xs = np.interp(u, cdf, x)
    return (SCHECHTER["M_B_star"] - 2.5 * np.log10(xs)).astype(np.float64)


def dl_interpolator(acosmo, z_max, n=200_001):
    """np.interp table for luminosity distance [Mpc].  astropy's per-element
    hypergeometric evaluation is far too slow at 1e8 rows; the table is accurate to
    <1e-9 relative (checked in validation V4)."""
    import astropy.units as u
    zg = np.linspace(0.0, float(z_max) * 1.02, n)
    dg = acosmo.luminosity_distance(zg).to_value(u.Mpc)
    return zg, dg


# ================================================================================
# vectorised survey pixelation (gmd's _pixelate_catalog, without the python loop)
# ================================================================================
def pixelate_catalog_vec(ra, dec, z, dz, w, nside, dtype=None):
    """Same output contract as gmd ``_pixelate_catalog`` -- (npix, max_gals) padded
    ``zgals``/``dzgals``/``wgals`` with the 100.0 / 1.0 / 0.0 sentinels and a
    per-row ``ngals`` count -- but built with a lexsort instead of a per-object
    python loop (gmd's loop is O(1e8) interpreter iterations for a v2 catalog).

    Rows come out sorted in z within their real prefix.  That is exactly the
    invariant darksirens' windowed catalog-KDE evaluator needs
    (``darksirens.redshift.catalog._rows_sorted_for_windowing``); gmd's loop
    emitted catalog order, so this also removes the load-time re-sort."""
    import healpy as hp
    dtype = np.dtype(CAT_DTYPE) if dtype is None else np.dtype(dtype)
    npix = hp.nside2npix(nside)
    pix = hp.ang2pix(nside, np.pi / 2.0 - np.asarray(dec), np.asarray(ra))
    counts = np.bincount(pix, minlength=npix).astype(np.int32)
    max_gals = max(1, int(counts.max()))
    order = np.lexsort((np.asarray(z), pix))            # by pixel, then by z
    pix_s = pix[order]
    starts = np.concatenate([[0], np.cumsum(counts)])[:-1]
    col = np.arange(pix_s.size, dtype=np.int64) - starts[pix_s]
    flat = pix_s.astype(np.int64) * max_gals + col
    zgals = np.full((npix, max_gals), 100.0, dtype=dtype)
    dzgals = np.full((npix, max_gals), 1.0, dtype=dtype)
    wgals = np.zeros((npix, max_gals), dtype=dtype)
    zgals.reshape(-1)[flat] = np.asarray(z)[order]
    dzgals.reshape(-1)[flat] = np.asarray(dz)[order]
    wgals.reshape(-1)[flat] = np.asarray(w)[order]
    return {"zgals": zgals, "dzgals": dzgals, "wgals": wgals, "ngals": counts}


# ================================================================================
# the measurement model  --  conventions (a) and (b) live here
# ================================================================================
def snr_amplitude(m1det, m2det, dl, snr_ref):
    """Projection-free, noise-free network SNR from DETECTOR-frame masses.

    Identical in form to gmd ``_network_snr`` with ``projection = 1``; taking
    detector-frame masses is what lets it be evaluated on OBSERVED quantities, where
    no redshift exists."""
    mchirp_det = (m1det * m2det) ** 0.6 / (m1det + m2det) ** 0.2
    return snr_ref * (mchirp_det / 30.0) ** (5.0 / 6.0) * (1000.0 / dl)


def sigma_ang_from_amplitude(rho):
    """Sky width in radians from an amplitude.  clip(35/rho, 1, 12) deg."""
    return np.deg2rad(np.clip(35.0 / rho, 1.0, 12.0))


_PEX_CACHE: dict = {}


def exact_mass_posterior_table(f, n_grid=None, y_cap=None, n_sig=None):
    """Quantile table of ``y = obs/m`` under the exact flat-prior posterior of
    ``obs ~ N(m, f m)``  --  convention (c2).

    Change of variables: with ``y = obs/m``, ``dm = -obs/y^2 dy`` and
        p(m|obs) dm  ~  (1/(f m)) exp[-(obs-m)^2/(2 f^2 m^2)] dm
                     =  (1/(f y)) exp[-(y-1)^2/(2 f^2)] dy,
    i.e. a Gaussian in ``y`` about 1 with sd ``f``, tilted by ``1/y``.  The shape is
    obs-INDEPENDENT, so one table serves every event.  Integrating on a uniform
    ``ln y`` grid absorbs the ``1/y`` exactly (``dy/y = d ln y``), leaving a plain
    Gaussian integrand -- which is why a trapezoid rule converges to 1e-12 here.

    Returns ``(cdf, y)`` for ``np.interp(u, cdf, y)``.
    """
    n_grid = PEX_N_GRID if n_grid is None else int(n_grid)
    y_cap = PEX_Y_CAP if y_cap is None else float(y_cap)
    n_sig = PEX_N_SIG if n_sig is None else float(n_sig)
    key = (float(f), n_grid, y_cap, n_sig)
    if key in _PEX_CACHE:
        return _PEX_CACHE[key]
    lny = np.linspace(np.log(y_cap), np.log(1.0 + n_sig * f), n_grid)
    y = np.exp(lny)
    d = np.exp(-0.5 * ((y - 1.0) / f) ** 2)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(lny))])
    cdf /= cdf[-1]
    _PEX_CACHE[key] = (cdf, y)
    return cdf, y


def exact_mass_posterior_cdf(m, obs, f, **kw):
    """CDF of the exact flat-prior posterior at ``m`` -- the PIT used in validation.

    ``m`` decreasing in ``y`` means ``P(m' <= m) = P(y' >= obs/m) = 1 - G(obs/m)``.
    """
    cdf, y = exact_mass_posterior_table(f, **kw)
    return 1.0 - np.interp(np.asarray(obs) / np.asarray(m), y, cdf)


def sample_exact_mass_posterior(rng, obs, f, nsamp, **kw):
    """``nsamp`` inverse-CDF draws from ``p(m|obs)`` for a scalar ``obs``."""
    cdf, y = exact_mass_posterior_table(f, **kw)
    return float(obs) / np.interp(rng.random(nsamp), cdf, y)


def observe(rng, m1det, m2det, chi, dl, ra, dec, *, need_sky=True):
    """ONE measurement per source.  Convention (b) is the ORDER of the draws.

    Distance noise is multiplicative (lognormal); masses, spin and sky are additive.
    Clips are applied HERE, before the detection statistic, so the object the SNR is
    computed from is bit-identically the object the posterior conditions on.

    SEQUENTIAL SKY WIDTH: dL and the masses are measured first, ``sigma_ang`` is
    derived from those OBSERVED values on the ``SNR_REF_SIGMA`` scale, and only then
    are the sky offsets drawn.  ``sigma_ang`` is therefore a deterministic function of
    the RECORDED data and the fixed-width sky posterior built from it is exact.

    SEQUENTIAL WITHIN THE SKY PAIR (convention (b2)): ``dec`` is drawn FIRST, then
    ``sig_ra = sigma_ang / max(cos dec_OBS, 0.1)`` is formed from that recorded value
    and only then is the RA offset drawn.  ``sig_ra`` is stored, so like
    ``sigma_ang`` it is recomputable from the file and the PE's fixed-width RA
    posterior built from it is exact.  Using ``cos dec_TRUE`` here (the convention in
    force before 2026-08-01) left the recorded RA posterior width wrong by 2.3 %
    (mean) / 54 % (max) -- see the module docstring and ATTRIBUTION.md A4.5.

    ``need_sky=False`` skips the sky and spin draws.  Detection depends only on
    (dL, m1det, m2det), so the detected set is distributionally identical; this is
    used for the selection campaigns, where the sky/spin OBSERVATIONS are never
    stored (injections store TRUE parameters -- mu(theta) is an integral over true
    parameters and only the DETECTION decision sees the noise)."""
    n = np.asarray(dl, dtype=float).shape[0]
    sig_m1 = SIG_M1_FRAC * m1det
    sig_m2 = SIG_M2_FRAC * m2det
    obs_dl = dl * np.exp(SIGMA_DL * rng.normal(size=n))
    obs_m1 = np.clip(rng.normal(m1det, sig_m1), 2.0, None)
    obs_m2 = np.clip(rng.normal(m2det, sig_m2), 1.0, None)
    # (b) the width is a function of the DATA just measured, not of the truth.
    sigma_ang = sigma_ang_from_amplitude(
        snr_amplitude(obs_m1, obs_m2, obs_dl, SNR_REF_SIGMA))
    out = {
        "dL": obs_dl, "m1det": obs_m1, "m2det": obs_m2,
        "sigma_dl": np.full(n, SIGMA_DL),
        "sig_m1": sig_m1, "sig_m2": sig_m2,
        "sigma_ang": sigma_ang,
    }
    if need_sky:
        out["chieff"] = np.clip(rng.normal(chi, SIGMA_CHIEFF), -1.0, 1.0)
        # (b2) dec FIRST, then the RA width from the dec just RECORDED.
        obs_dec = np.clip(dec + rng.normal(0.0, sigma_ang), -0.5 * np.pi, 0.5 * np.pi)
        sig_ra = sigma_ang / np.maximum(np.cos(obs_dec), 0.1)
        out["dec"] = obs_dec
        out["sig_ra"] = sig_ra
        out["ra"] = (ra + rng.normal(0.0, sig_ra)) % (2.0 * np.pi)
    return out


def detect_from_observation(obs):
    """(a) detection is a deterministic function of the recorded measurement."""
    rho_obs = snr_amplitude(obs["m1det"], obs["m2det"], obs["dL"], SNR_REF_DETECT)
    return rho_obs >= SNR_THRESHOLD, rho_obs


def posterior_samples(rng, obs, nsamp):
    """(c) EXACT flat-prior posterior draws GIVEN the stored observation.

    Distance: with ``ln d_obs ~ N(ln dL, s)`` the flat-in-dL posterior is
    ``ln dL ~ N(ln d_obs + s^2, s)``.  Derivation: p(dL|d_obs) ~ exp(-(ln d_obs -
    ln dL)^2 / 2s^2); substituting u = ln dL brings a Jacobian e^u, and completing
    the square gives u ~ N(ln d_obs + s^2, s).  This closed form is EXACT.  The
    validation stage re-derives this posterior numerically and KS-tests the stored
    samples against it.

    Masses (convention (c2)): the noise is ADDITIVE but its width is proportional to
    the LATENT mass, ``obs ~ N(m, f m)``, so the flat-prior posterior is NOT a
    Gaussian about ``obs``:
        p(m | obs)  ~  (1 / (f m)) exp[ -(obs - m)^2 / (2 f^2 m^2) ],
    and it is drawn here by INVERSE CDF -- the additive-noise branch of the
    campaign's ``code/generate_gwsamples.py`` machinery, implemented in
    ``exact_mass_posterior_table`` and shared across events (the shape in
    ``y = obs/m`` does not depend on ``obs``).  The clips at 2 / 1 Msun are retained
    from the measurement model but are inert: the table's own support starts at
    ``obs/(1 + 12 f)`` >= 6.3 (m1) / 2.6 (m2) Msun on this dataset, and validation
    records the realised clip fraction (0).

    Spin and sky have constant, data-derived widths, so a Gaussian centred on the
    OBSERVED value IS the exact flat-prior posterior.  The RA width is taken from
    the STORED ``sig_ra`` (convention (b2)), which is the identical number
    ``observe()`` measured with -- not a re-derivation that could drift from it.
    The order of the draws below is immaterial for the PE (each channel's posterior
    is conditionally independent given the stored observation); it follows the
    measurement order only for readability."""
    nobs = obs["dL"].shape[0]
    out = {k: [] for k in ("ra", "dec", "dL", "m1det", "m2det", "chieff")}
    for i in range(nobs):
        s = float(obs["sigma_dl"][i])
        sa = float(obs["sigma_ang"][i])
        out["dL"].append(rng.lognormal(np.log(obs["dL"][i]) + s * s, s, nsamp))
        out["dec"].append(np.clip(obs["dec"][i] + rng.normal(0.0, sa, nsamp),
                                  -0.5 * np.pi, 0.5 * np.pi))
        out["ra"].append((obs["ra"][i]
                          + rng.normal(0.0, float(obs["sig_ra"][i]), nsamp))
                         % (2.0 * np.pi))
        out["m1det"].append(np.clip(
            sample_exact_mass_posterior(rng, obs["m1det"][i], SIG_M1_FRAC, nsamp),
            2.0, None))
        out["m2det"].append(np.clip(
            sample_exact_mass_posterior(rng, obs["m2det"][i], SIG_M2_FRAC, nsamp),
            1.0, None))
        out["chieff"].append(np.clip(rng.normal(obs["chieff"][i], SIGMA_CHIEFF, nsamp),
                                     -1.0, 1.0))
    return {k: np.concatenate(v) for k, v in out.items()}


# ================================================================================
# THE v3 MEASUREMENT FAMILY  --  every width is a function of rho_obs, i.e. DATA
# ================================================================================
# See working/data/DESIGN_PE.md for the derivations, the literature citations and
# the calibration of A_MC / A_Q / A_CHI.  The short version:
#
#   parameters   (Mc_det, q, rho, chieff, ra, dec)      <-->   (m1det, m2det, dL, ...)
#   data         (rho_obs, ln Mc_obs, ln q_obs, chieff_obs, dec_obs, ra_obs)
#   widths       sigma_x = A_x * (SNR_THRESHOLD / rho_obs)          [DATA ONLY]
#   detection    rho_obs >= SNR_THRESHOLD                           [DATA ONLY]
#   PE           the exact flat-prior posterior in (ln Mc, ln q, rho, chieff, sky),
#                truncated ONLY in the PRIOR (q <= 1, rho > 0, |chi| <= 1,
#                |dec| <= pi/2); no observation is ever clipped, because clipping
#                the DATA censors the likelihood and gives it a theta-dependent
#                normalisation -- the defect (c2) had to repair in v2.
#
# The rho channel IS the distance channel (GWMockCat's own construction with the
# projection factor removed, which convention (a) forbids anyway), so nothing is
# double counted and the estimator's likelihood is the true likelihood.

def mc_of_m1q(m1det, q):
    """Detector-frame chirp mass from (m1det, q).  Mc = m1 q^(3/5) / (1+q)^(1/5)."""
    return m1det * q ** 0.6 / (1.0 + q) ** 0.2


def m1_of_mc_q(mc, q):
    """Inverse of :func:`mc_of_m1q`."""
    return mc * (1.0 + q) ** 0.2 / q ** 0.6


def rho_opt_of_mc_dl(mc_det, dl):
    """The projection-free optimal SNR on the DETECTION amplitude scale.

    Identical to ``snr_amplitude`` -- written in terms of the chirp mass so the
    bijection (Mc_det, q, rho) <-> (m1det, m2det, dL) is explicit."""
    return SNR_REF_DETECT * (mc_det / 30.0) ** (5.0 / 6.0) * (1000.0 / dl)


def dl_of_mc_rho(mc_det, rho):
    """The inverse: the distance a source of chirp mass ``mc_det`` needs to give
    optimal SNR ``rho``.  This is what makes rho the distance coordinate."""
    return 1000.0 * SNR_REF_DETECT * (mc_det / 30.0) ** (5.0 / 6.0) / rho


def sigma_ang_v3(rho_obs):
    """Sky width from the RECORDED SNR.  The campaign's own 35 deg / rho_sigma
    convention, with ``rho_sigma`` now formed from ``rho_obs`` instead of being
    recomputed from the observed masses and distance:

        rho_sigma = (SNR_REF_SIGMA / SNR_REF_DETECT) * rho_obs = 1.83165 rho_obs

    so ``sigma_ang = clip(19.1069 deg / rho_obs, 1, 12)`` -- the same realised
    width distribution as v2, now a function of one stored number."""
    rho_sigma = (SNR_REF_SIGMA / SNR_REF_DETECT) * np.asarray(rho_obs, float)
    return np.deg2rad(np.clip(SKY_A_DEG / rho_sigma, *SKY_CLIP_DEG))


def v3_widths(rho_obs):
    """Every measurement width of the v3 family, from rho_obs alone."""
    r = np.asarray(rho_obs, float)
    k = SNR_THRESHOLD / r
    return {"sig_lnmc": A_MC * k, "sig_lnq": A_Q * k, "sig_chieff": A_CHI * k,
            "sigma_ang": sigma_ang_v3(r)}


def _trunc_norm(rng, loc, scale, lo, hi, size=None):
    """Exact truncated-normal draws by inverse CDF (scipy-free, vectorised).

    ``lo``/``hi`` may be +-inf.  This is the ONLY sampler v3 uses for a truncated
    channel; the truncation is always a PRIOR truncation, never a censored
    likelihood."""
    from scipy.special import ndtr, ndtri
    loc = np.asarray(loc, float)
    scale = np.asarray(scale, float)
    a = ndtr((np.asarray(lo, float) - loc) / scale)
    b = ndtr((np.asarray(hi, float) - loc) / scale)
    u = rng.random(size if size is not None else np.broadcast(loc, scale).shape)
    return loc + scale * ndtri(np.clip(a + u * (b - a), 1e-300, 1.0 - 1e-16))


def observe_v3(rng, m1det, m2det, chi, dl, ra, dec, *, need_sky=True):
    """ONE measurement per source in the v3 family.

    Draw order (this IS the model; do not reorder):
      1. rho_obs = rho_opt(theta) + N(0, SIGMA_RHO)
      2. widths from rho_obs and NOTHING else
      3. ln Mc_obs, ln q_obs, chieff_obs, then dec_obs, then ra_obs
         (convention (b2): dec is measured before ra, and the RA width is formed
         from the dec ALREADY RECORDED).

    ``need_sky=False`` draws only rho_obs -- which is the whole detection rule --
    and is what the selection campaigns use.  No clip is applied to any recorded
    value anywhere in this function."""
    n = np.asarray(dl, dtype=float).shape[0]
    mc_det = mc_of_m1q(np.asarray(m1det, float), np.asarray(m2det, float)
                       / np.asarray(m1det, float))
    rho_true = rho_opt_of_mc_dl(mc_det, dl)
    rho_obs = rho_true + SIGMA_RHO * rng.normal(size=n)
    out = {"rho": rho_obs, "rho_true": rho_true,
           "sigma_rho": np.full(n, SIGMA_RHO)}
    if not need_sky:
        return out
    # widths: functions of the recorded rho_obs alone.  rho_obs can be <= 0 for a
    # (never detected) proposal, so the widths are formed on a floored copy and the
    # rows that matter -- the detected ones, rho_obs >= 8 -- are untouched by it.
    r_safe = np.maximum(rho_obs, 1e-3)
    w = v3_widths(r_safe)
    q_true = np.asarray(m2det, float) / np.asarray(m1det, float)
    out["sig_lnmc"] = w["sig_lnmc"]
    out["sig_lnq"] = w["sig_lnq"]
    out["sig_chieff"] = w["sig_chieff"]
    out["sigma_ang"] = w["sigma_ang"]
    out["lnmc"] = np.log(mc_det) + w["sig_lnmc"] * rng.normal(size=n)
    out["lnq"] = np.log(q_true) + w["sig_lnq"] * rng.normal(size=n)
    out["chieff"] = np.asarray(chi, float) + w["sig_chieff"] * rng.normal(size=n)
    # (b2) dec FIRST, then the RA width from the dec just RECORDED.  dec_obs is NOT
    # clipped: |dec| <= pi/2 is a statement about the PARAMETER, and it is imposed
    # on the PE prior instead (see posterior_samples_v3).
    obs_dec = np.asarray(dec, float) + w["sigma_ang"] * rng.normal(size=n)
    sig_ra = w["sigma_ang"] / np.maximum(np.cos(obs_dec), COS_DEC_FLOOR)
    out["dec"] = obs_dec
    out["sig_ra"] = sig_ra
    out["ra"] = (np.asarray(ra, float) + sig_ra * rng.normal(size=n)) % (2.0 * np.pi)
    # derived point estimates -- DIAGNOSTIC ONLY; the PE never reads them.  For a
    # proposal with rho_obs near zero the widths blow up and these overflow; those
    # rows are never detected and nothing downstream reads them, so the overflow is
    # ignored rather than clipped (clipping would misrepresent the recorded value).
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        q_o = np.exp(out["lnq"])
        mc_o = np.exp(out["lnmc"])
        m1_o = m1_of_mc_q(mc_o, q_o)
        out["m1det"] = m1_o
        out["m2det"] = q_o * m1_o
        out["dL"] = dl_of_mc_rho(mc_o, np.maximum(rho_obs, 1e-3))
    return out


def detect_v3(obs):
    """(a) detection is a deterministic function of ONE recorded number."""
    rho_obs = obs["rho"]
    return rho_obs >= SNR_THRESHOLD, rho_obs


def posterior_samples_v3(rng, obs, nsamp):
    """(c) EXACT flat-prior posterior draws GIVEN the stored observation.

    The PE prior is flat in ``(ln Mc_det, ln q, rho, chieff, ra, dec)`` on the
    physical support ``q <= 1, rho > 0, |chieff| <= 1, |dec| <= pi/2``.  Every
    channel's likelihood is an UNBOUNDED Gaussian in exactly that variable, so
    every posterior is a (possibly prior-truncated) normal about the OBSERVED value
    with the STORED width -- no shift, no skew, no quantile table:

        ln Mc  ~ N(ln Mc_obs, sig_lnmc)
        ln q   ~ N(ln q_obs,  sig_lnq )   truncated to ln q <= 0
        rho    ~ N(rho_obs,   sigma_rho)  truncated to rho > 0   (inert at 8 sigma)
        chieff ~ N(chi_obs,   sig_chieff) truncated to [-1, 1]
        dec    ~ N(dec_obs,   sigma_ang)  truncated to [-pi/2, pi/2]
        ra     ~ wrapped N(ra_obs, sig_ra)

    and the samples are then mapped through the bijection

        m1det = Mc (1+q)^(1/5) q^(-3/5),  m2det = q m1det,
        dL    = 1000 SNR_REF_DETECT (Mc/30)^(5/6) / rho

    so ``q <= 1`` holds for EVERY sample (v2 spent 18.4 % of its samples on
    ``q > 1``, which the population prior discards)."""
    nobs = obs["rho"].shape[0]
    keys = ("ra", "dec", "dL", "m1det", "m2det", "chieff", "rho", "mc_det", "q")
    out = {k: [] for k in keys}
    half_pi = 0.5 * np.pi
    for i in range(nobs):
        s_mc = float(obs["sig_lnmc"][i])
        s_q = float(obs["sig_lnq"][i])
        s_ch = float(obs["sig_chieff"][i])
        s_an = float(obs["sigma_ang"][i])
        s_ra = float(obs["sig_ra"][i])
        s_rho = float(obs["sigma_rho"][i])
        lnmc = float(obs["lnmc"][i]) + s_mc * rng.normal(size=nsamp)
        lnq = _trunc_norm(rng, float(obs["lnq"][i]), s_q, -np.inf, 0.0, nsamp)
        rho = _trunc_norm(rng, float(obs["rho"][i]), s_rho, 0.0, np.inf, nsamp)
        chi = _trunc_norm(rng, float(obs["chieff"][i]), s_ch,
                          CHIEFF_RANGE[0], CHIEFF_RANGE[1], nsamp)
        dec = _trunc_norm(rng, float(obs["dec"][i]), s_an, -half_pi, half_pi, nsamp)
        ra = (float(obs["ra"][i]) + s_ra * rng.normal(size=nsamp)) % (2.0 * np.pi)
        mc = np.exp(lnmc)
        q = np.exp(lnq)
        m1 = m1_of_mc_q(mc, q)
        out["mc_det"].append(mc)
        out["q"].append(q)
        out["rho"].append(rho)
        out["m1det"].append(m1)
        out["m2det"].append(q * m1)
        out["dL"].append(dl_of_mc_rho(mc, rho))
        out["chieff"].append(chi)
        out["dec"].append(dec)
        out["ra"].append(ra)
    return {k: np.concatenate(v) for k, v in out.items()}


def p_pe_v3(m1det, q, dL, mc_det=None):
    """The PE PRIOR density in darksirens' canonical ``(m1det, q, dL, chieff)``
    basis -- the number darksirens divides each sample by.

    With the prior flat in ``y = (ln Mc_det, ln q, rho, chieff)`` and
    ``x = (m1det, q, dL, chieff)``, writing ``A = 3/(5q) - 1/(5(1+q))``,

        d(ln Mc, ln q, rho)/d(m1det, q, dL) =
            [ 1/m1det          A            0       ]
            [ 0                1/q          0       ]
            [ (5/6) rho/m1det  (5/6) rho A  -rho/dL ]

    whose determinant is ``-(rho/dL) * 1/(m1det q)``, so

        p_pe  ~  rho / (dL m1det q)  ~  Mc_det^(5/6) / (dL^2 m1det q).

    (The v2 convention ``p_pe ~ m1det`` is the same rule applied to a prior flat in
    ``(m1det, m2det, dL, chieff)``.)  chieff maps identically and the sky prior is
    flat in (ra, dec), the same convention as v2, so neither contributes a factor.
    Validated in stage validation, check V3c, against a numerical Jacobian."""
    mc = mc_of_m1q(m1det, q) if mc_det is None else mc_det
    rho = rho_opt_of_mc_dl(mc, dL)
    return rho / (dL * m1det * q)


# ================================================================================
# STAGE 1 -- GLASS catalogs  (runs under the dedicated glass venv)
# ================================================================================
def stage_catalogs(args):
    """Re-execute this file under the glass venv unless we are already there."""
    if not args._glass_worker:
        py = Path(args.glass_python)
        if not py.exists():
            raise SystemExit(
                f"glass interpreter not found: {py}\n"
                "Create it with:\n"
                f"  python -m venv {py.parents[1]}\n"
                f"  {py} -m pip install 'glass==2025.1' 'glass.ext.camb==2023.6' camb h5py")
        cmd = [str(py), str(Path(__file__).resolve()), "--stage", "catalogs",
               "--seed", str(args.seed), "--outroot", str(args.outroot),
               "--_glass_worker"]
        if args.overwrite:
            cmd.append("--overwrite")
        _log(f"catalogs: delegating to the glass venv -> {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return
    _glass_build(args)


def _glass_build(args):
    """GLASS lognormal two-tracer catalogs on ONE shared density field.

    Adapted from ``code/make_mocks.py::create_mock_catalog_glass`` -- same CAMB
    parameters, same 200 Mpc linear shells, same volume-weighted dN/dz, same
    ``positions_from_delta`` bias assignment.  Both tracers are drawn from the SAME
    realisation of the matter field, which is what gives them a shared large-scale
    structure with different bias (1.2 vs 2.0).

    Magnitudes are attached here so the catalog files are self-contained for the
    flux-limit stage.  AGN INHERIT THEIR HOST GALAXY'S APPARENT MAGNITUDE: an AGN
    lives in a galaxy, so its absolute magnitude is drawn from the same galaxy
    luminosity model -- v2: the Schechter function truncated at the cut that
    defines the density anchor -- and converted at the AGN's own redshift.  One flux
    limit therefore thins both tracers with the same C(z), which keeps the
    completeness ladder a clean single axis.

    v2 amplitudes are COMOVING NUMBER DENSITIES.  GLASS's `ngal` is a surface
    density per arcmin^2 spread over the shells by ``glass.partition`` in
    proportion to ``dndz``, so

        N_total = A_sky[arcmin^2] * nbar * INT(dndz dz)   and   n_comoving = N/V_c,

    which is inverted here for nbar.  ``dndz`` is the volume weight, so the
    realised catalog has CONSTANT comoving number density by construction.
    """
    import h5py
    import camb
    from cosmology import Cosmology
    import glass
    import glass.ext.camb
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u

    out = seed_dir(args.seed, args.outroot) / "catalogs"
    out.mkdir(parents=True, exist_ok=True)
    done = [out / f"catalog_{t}_complete.h5" for t in TRACERS]
    if all(p.exists() for p in done) and not args.overwrite:
        _log(f"catalogs: {[str(p) for p in done]} exist; use --overwrite to regenerate")
        return

    seeds = sub_seeds(args.seed)
    t0 = time.perf_counter()
    g = GLASS
    Oc = g["Om0"] - g["Ob0"]
    _log(f"glass: CAMB params h={g['h']} Om0={g['Om0']} Ob0={g['Ob0']} (Oc={Oc:.4f})")
    pars = camb.set_params(H0=100 * g["h"], omch2=Oc * g["h"] ** 2,
                           ombh2=g["Ob0"] * g["h"] ** 2,
                           NonLinear=camb.model.NonLinear_both)
    cosmo = Cosmology.from_camb(pars)

    zb = glass.distance_grid(cosmo, g["z_min"], g["z_max"], dx=g["dx_mpc"])
    shells = glass.linear_windows(zb)
    _log(f"glass: {len(shells)} shells, shell edges {zb[0]:.4f} .. {zb[-1]:.4f}")

    cls = glass.ext.camb.matter_cls(pars, g["lmax"], shells)
    _log(f"glass: CAMB angular spectra done ({len(cls)} spectra) "
         f"[{time.perf_counter() - t0:.1f} s]")
    fields = glass.lognormal_fields(shells)
    cls = glass.discretized_cls(cls, nside=g["nside"], lmax=g["lmax"], ncorr=3)
    gls = glass.solve_gaussian_spectra(fields, cls)
    _log(f"glass: gaussian spectra solved [{time.perf_counter() - t0:.1f} s]")

    # ONE rng, ONE matter realisation, both tracers painted on it.
    rng = np.random.default_rng(seeds["glass_field"])
    matter = list(glass.generate(fields, gls, g["nside"], ncorr=3, rng=rng))
    _log(f"glass: matter field realised ({len(matter)} shells @ nside {g['nside']}) "
         f"[{time.perf_counter() - t0:.1f} s]")

    # --- comoving number density -> GLASS surface density -----------------------
    # v1 used ``glass.partition(z_bins, nbar * dndz, shells)`` + ``glass.redshifts``.
    # BOTH steps are wrong for a constant-comoving-density catalog:
    #   * partition's NNLS fit minimises the ABSOLUTE L2 residual against
    #     dN/dz ~ z^2, so near z = 0, where the target is ~0, the fit's relative
    #     error is enormous -- the first run of v2 came out 7.3x too dense below
    #     z = 0.05 and 21% too dense inside the GW horizon (v1 had the opposite
    #     sign: NO galaxy at all below z = 0.047).
    #   * ``glass.redshifts`` then spreads a shell's objects ~ W_i(z), i.e. nearly
    #     uniformly IN REDSHIFT across a 200 Mpc shell, while a constant comoving
    #     density needs ~ W_i(z) dV_c/dz -- a large distortion in the first shells.
    # ``glass.linear_windows`` is a PARTITION OF UNITY on [zb[1], zb[-2]], so
    # setting  N_i = n * INT W_i(z) (dV_c/dz) dz  and drawing z within shell i from
    # the density ~ W_i(z) (dV_c/dz)(z) gives EXACTLY
    #     dN/dz = n (dV_c/dz) sum_i W_i(z) = n dV_c/dz     for zb[1] <= z <= zb[-2],
    # with linear ramps over the first and last shell half-widths (z < 0.0457 and
    # z > 0.9230 here), which are recorded in META and excluded from the density
    # gate in validation V4.
    acosmo = FlatLambdaCDM(H0=H0_FID, Om0=OM0_FID)
    a_sky_arcmin2 = float((4.0 * np.pi * u.sr).to(u.arcmin ** 2).value)
    v_c = float(4.0 / 3.0 * np.pi
                * acosmo.comoving_distance(g["z_max"]).to_value(u.Mpc) ** 3)

    def dVdz_full_sky(zz):
        """dV_c/dz over the full sky [Mpc^3 per unit z]."""
        return (4.0 * np.pi
                * acosmo.differential_comoving_volume(zz).to_value(u.Mpc ** 3 / u.sr))

    shell_int, shell_cdf = [], []
    for w in shells:
        za = np.asarray(w.za, dtype=float)
        dens = np.asarray(w.wa, dtype=float) * dVdz_full_sky(za)
        cdf = np.concatenate([[0.0],
                              np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(za))])
        shell_int.append(float(cdf[-1]))
        shell_cdf.append((za, cdf / max(cdf[-1], 1e-300)))
    shell_int = np.asarray(shell_int)
    flat_lo, flat_hi = float(zb[1]), float(zb[-2])

    nbar = {t: g[f"n_comoving_{t}"] * shell_int / a_sky_arcmin2 for t in TRACERS}
    x_cut = schechter_cut_for_density(g["n_comoving_gal"])
    _log(f"glass: V_c(z<{g['z_max']}) = {v_c:.4e} Mpc^3; constant-density plateau "
         f"z in [{flat_lo:.4f}, {flat_hi:.4f}]; sum_i INT W_i dV/dz = "
         f"{shell_int.sum():.4e} Mpc^3 ({shell_int.sum() / v_c:.4f} of V_c)")
    _log(f"glass: N_target GAL {g['n_comoving_gal'] * shell_int.sum():,.0f}  "
         f"AGN {g['n_comoving_agn'] * shell_int.sum():,.0f}; Schechter cut "
         f"L > {x_cut:.4f} L* "
         f"(M_B < {SCHECHTER['M_B_star'] - 2.5 * np.log10(x_cut):.4f})")

    tracers = {"gal": {"bias": g["bias_gal"], "N": nbar["gal"]},
               "agn": {"bias": g["bias_agn"], "N": nbar["agn"]}}

    cdt = np.dtype(CAT_DTYPE)
    cats = {}
    for name, tr in tracers.items():
        lons, lats, zs = [], [], []
        for i, delta_i in enumerate(matter):
            za_i, cdf_i = shell_cdf[i]
            for lon, lat, count in glass.positions_from_delta(
                    tr["N"][i], delta_i, bias=tr["bias"], rng=rng):
                zz = np.interp(rng.uniform(size=int(count)), cdf_i, za_i)
                lons.append(np.asarray(lon, dtype=cdt))
                lats.append(np.asarray(lat, dtype=cdt))
                zs.append(np.asarray(zz, dtype=cdt))
        lon = np.concatenate(lons) % cdt.type(360.0)
        del lons
        lat = np.concatenate(lats); del lats
        z = np.concatenate(zs); del zs
        cats[name] = {"ra": np.deg2rad(lon).astype(cdt),
                      "dec": np.deg2rad(lat).astype(cdt), "z": z}
        del lon, lat
        _log(f"glass: tracer {name}: {z.size:,} objects, "
             f"z in [{z.min():.5f}, {z.max():.5f}] [{time.perf_counter() - t0:.1f} s]")

    # --- magnitudes (own stream, independent of GLASS rng consumption) -----------
    mrng = np.random.default_rng(seeds["magnitudes"])
    zg_tab, dl_tab = dl_interpolator(acosmo, float(max(c["z"].max()
                                                       for c in cats.values())))
    for name in TRACERS:
        c = cats[name]
        abs_mag = sample_schechter_abs_mag(mrng, c["z"].size, x_cut)
        dl_pc = np.interp(c["z"].astype(np.float64), zg_tab, dl_tab) * 1.0e6
        c["abs_mag"] = abs_mag.astype(cdt)
        c["app_mag"] = (abs_mag
                        + 5.0 * np.log10(np.maximum(dl_pc, 10.0) / 10.0)).astype(cdt)
        del abs_mag, dl_pc

    # --- D3: REALISE the declared photo-z error (own stream) ---------------------
    # `z` stays the TRUE redshift: the hosts are drawn from it and the events happen
    # at it.  `z_obs` is what the SURVEY records, and stage_surveys pixelates on it
    # with the declared width dz = DZ_SCALE (1 + z_obs).  darksirens' per-galaxy
    # kernel g(z) N(z; z_obs, sigma)/Z(z_obs) is then exactly the posterior for that
    # galaxy's true redshift given its catalog entry, so the model's p_z(z|pix) IS
    # the distribution of the host's true redshift given the catalog.  Before this,
    # the block declared dz = 3e-3(1+z) on redshifts copied bit-for-bit from the
    # catalog the hosts were drawn from -- a 7.6 sigma (A - B) effect, CLOSURE.md
    # 15.4.  NOT clipped at z >= 0: clipping would censor.  See DESIGN_PE.md 3.3.
    zrng = np.random.default_rng(seeds["photoz"])
    photoz_stats = {}
    for name in TRACERS:
        c = cats[name]
        zt = c["z"].astype(np.float64)
        zo = zt + DZ_SCALE * (1.0 + zt) * zrng.normal(size=zt.size)
        c["z_obs"] = zo.astype(cdt)
        photoz_stats[name] = {
            "n_negative_z_obs": int((zo < 0.0).sum()),
            "z_obs_min": float(zo.min()), "z_obs_max": float(zo.max()),
            "pull_mean": float(np.mean((zo - zt) / (DZ_SCALE * (1.0 + zt)))),
            "pull_sd": float(np.std((zo - zt) / (DZ_SCALE * (1.0 + zt))))}
        del zt, zo

    rec = {"generated_at_utc": _now(), "elapsed_s": time.perf_counter() - t0,
           "glass_config": g, "seed_glass_field": seeds["glass_field"],
           "seed_magnitudes": seeds["magnitudes"],
           "seed_photoz": seeds["photoz"],
           "photoz_model": {
               "column": "z_obs",
               "form": "z_obs = z + N(0, DZ_SCALE (1+z)),  DZ_SCALE = "
                       f"{DZ_SCALE:g}",
               "why": ("D3 / DESIGN_PE.md 3.3 -- the survey block declares this "
                       "kernel, so the catalog must actually carry the error.  "
                       "z (true) drives the host draw and the event's truth; "
                       "z_obs is what the survey and hence the likelihood see."),
               "not_clipped": True,
               "realised": photoz_stats},
           "shell_edges": zb.tolist(), "n_shells": len(shells),
           "storage_dtype": CAT_DTYPE,
           "number_density": {
               "target_comoving_Mpc^-3": {t: g[f"n_comoving_{t}"] for t in TRACERS},
               "V_comoving_to_zmax_Mpc^3": v_c,
               "sky_area_arcmin2": a_sky_arcmin2,
               "shell_volume_integrals_Mpc^3": shell_int.tolist(),
               "sum_shell_volume_over_Vc": float(shell_int.sum() / v_c),
               "constant_density_plateau": [flat_lo, flat_hi],
               "plateau_note": (
                   "linear_windows are a partition of unity only on "
                   "[zb[1], zb[-2]]; dN/dz = n dV_c/dz exactly there and ramps "
                   "linearly to 0 over the first/last shell half-width"),
               "nbar_per_arcmin2_per_shell": {t: nbar[t].tolist() for t in TRACERS},
               "N_target": {t: g[f"n_comoving_{t}"] * float(shell_int.sum())
                            for t in TRACERS},
               "agn_lf": AGN_LF},
           "magnitude_model": {
               "form": "Schechter luminosity function truncated at L > x_cut L*",
               "schechter": SCHECHTER,
               "phi_star_Mpc^-3": schechter_phi_star(),
               "x_cut_L_over_Lstar": x_cut,
               "M_B_faint_limit": SCHECHTER["M_B_star"] - 2.5 * np.log10(x_cut),
               "n_at_0.25Lstar_Mpc^-3": schechter_number_density(0.25),
               "n_at_x_cut_Mpc^-3": schechter_number_density(x_cut),
               "cut_arithmetic": (
                   "n(>x L*) = phi* Gamma(alpha+1, x); phi* = 1.6e-2 h^3 with h=0.7; "
                   "the brief's nominal x=0.25 gives 5.948e-3 Mpc^-3 (5.9x the "
                   "intended 1e-3 and ~9.8e8 rows to z=1, over the storage budget), "
                   "so x was solved from n = 1e-3 instead -> x = 1.0908."),
               "app_mag": "abs_mag + 5 log10(dL[pc]/10), dL at H0=67.74 Om0=0.3075 "
                          "via a 200k-point interpolation table",
               "agn_note": "AGN inherit their host galaxy's apparent magnitude: their "
                           "absolute magnitudes come from the same galaxy luminosity "
                           "model, evaluated at the AGN's own redshift."},
           "interpreter": sys.executable,
           "packages": _pkg_versions(["numpy", "scipy", "glass", "camb", "cosmology",
                                      "healpy", "healpix", "h5py", "astropy"]),
           "tracers": {}}

    for name in TRACERS:
        c = cats[name]
        p = out / f"catalog_{name}_complete.h5"
        with h5py.File(p, "w") as f:
            f.attrs["mock_data"] = True
            f.attrs["tracer"] = name
            f.attrs["complete"] = True
            f.attrs["description"] = (
                f"Complete GLASS lognormal {name.upper()} tracer catalog "
                f"(bias {tracers[name]['bias']}), shared density field, before any "
                "flux limit. ra/dec in radians.")
            f.attrs["bias"] = float(tracers[name]["bias"])
            f.attrs["nbar_per_arcmin2_total"] = float(nbar[name].sum())
            f.attrs["n_comoving_target"] = float(g[f"n_comoving_{name}"])
            f.attrs["n_comoving_plateau"] = [flat_lo, flat_hi]
            f.attrs["schechter_x_cut"] = float(x_cut)
            f.attrs["n_hosts"] = int(c["z"].size)
            f.attrs["z_max_catalog"] = float(c["z"].max())
            f.attrs["seed_glass_field"] = seeds["glass_field"]
            f.attrs["seed_magnitudes"] = seeds["magnitudes"]
            f.attrs["seed_photoz"] = seeds["photoz"]
            f.attrs["photoz_convention"] = (
                f"z_obs = z + N(0, {DZ_SCALE:g} (1+z)); z is TRUE, z_obs is what "
                "the survey records (D3, DESIGN_PE.md 3.3)")
            f.attrs["built_by"] = str(Path(__file__).resolve())
            f.attrs["built_at_utc"] = _now()
            for k in ("ra", "dec", "z", "z_obs", "abs_mag", "app_mag"):
                f.create_dataset(k, data=c[k], compression="gzip", shuffle=True)
        rec["tracers"][name] = {
            "path": str(p), "n": int(c["z"].size), "bias": tracers[name]["bias"],
            "n_comoving_realised_over_shell_volume":
                float(c["z"].size / shell_int.sum()),
            "n_comoving_target": float(g[f"n_comoving_{name}"]),
            "z_min": float(c["z"].min()), "z_max": float(c["z"].max()),
            "app_mag_min": float(c["app_mag"].min()),
            "app_mag_max": float(c["app_mag"].max()),
            "size_bytes": p.stat().st_size}
        _log(f"wrote {p} ({c['z'].size:,} objects, {p.stat().st_size / 1e6:.1f} MB)")

    rec["density_ratio_gal_over_agn"] = (rec["tracers"]["gal"]["n"]
                                         / rec["tracers"]["agn"]["n"])
    _write_json(out / "glass_field_meta.json", rec)
    _merge_meta(args.seed, "catalogs", rec, args.outroot)
    _log(f"catalogs done in {time.perf_counter() - t0:.1f} s")


# ================================================================================
# STAGE 2 -- events
# ================================================================================
def load_catalog(path, keys=("ra", "dec", "z", "z_obs", "abs_mag", "app_mag"),
                 dtype=None):
    """Load a catalog at the dataset's storage precision (``CAT_DTYPE``, float64).

    Catalogs run to 1.6e8 rows, so a full five-column load is ~6 GB; pass a narrower
    ``keys`` when only positions are needed.  Reading at the storage dtype is what
    makes the event host draw (``stage_events``) see the same redshifts the survey
    blocks and the likelihood do."""
    import h5py
    dtype = np.dtype(CAT_DTYPE) if dtype is None else np.dtype(dtype)
    with h5py.File(path, "r") as f:
        return {k: np.asarray(f[k][:], dtype=dtype) for k in keys if k in f}


def stage_events(args):
    import h5py
    gmd = import_gmd(args.darksirens)
    sd = seed_dir(args.seed, args.outroot)
    out = sd / "events"
    out.mkdir(parents=True, exist_ok=True)
    _NEV = int(getattr(args, "n_events", None) or N_EVENTS)
    _NSA = int(getattr(args, "nsamp", None) or N_SAMP)
    _sfx = getattr(args, "events_suffix", "") or ""
    ev_path = out / f"events{_sfx}.h5"
    if ev_path.exists() and not args.overwrite:
        _log(f"events: {ev_path} exists; use --overwrite to regenerate")
        return

    seeds = sub_seeds(args.seed)
    # --- the two knobs an EXTRA event draw may turn (defaults = the record) ------
    # ``--f_agn`` and ``--seed_events`` both default to None, in which case _FAGN is
    # the module constant F_AGN and _SEED_EV is the record's derived sub-seed, so a
    # no-flag run is bit-identical to the record.  They exist so a single catalog
    # realisation can carry additional, INDEPENDENT event sets (analysis 0's
    # pure-tracer draws) without touching the signed-off events.h5.
    _FAGN = F_AGN if getattr(args, "f_agn", None) is None else float(args.f_agn)
    _SEED_EV = (seeds["events"] if getattr(args, "seed_events", None) is None
                else int(args.seed_events))
    if not 0.0 <= _FAGN <= 1.0:
        raise SystemExit(f"--f_agn must lie in [0, 1]; got {_FAGN}")
    rng = np.random.default_rng(_SEED_EV)
    cosmo = gmd._build_cosmology(H0_FID, OM0_FID, W0_FID, WA_FID)
    grids = gmd._cosmology_grids(cosmo, ZMAX_GRID)
    pop = gmd.PopulationConfig(gamma=GAMMA)

    pe_model = getattr(args, "pe_model", PE_MODEL_DEFAULT)
    _observe = observe_v3 if pe_model == "v3" else observe
    _detect = detect_v3 if pe_model == "v3" else detect_from_observation
    _post = posterior_samples_v3 if pe_model == "v3" else posterior_samples
    _log(f"events: measurement family {pe_model}")

    # only (ra, dec, z) are needed here -- the host draw uses the TRUE redshift.
    cats = {t: load_catalog(sd / "catalogs" / f"catalog_{t}_complete.h5",
                            keys=("ra", "dec", "z"))
            for t in TRACERS}
    n_gal, n_agn = cats["gal"]["z"].size, cats["agn"]["z"].size
    _log(f"events: hosts GAL {n_gal:,}  AGN {n_agn:,}  planted f_AGN={_FAGN}"
         f"  seed_events={_SEED_EV}")

    # gmd's rate weighting, verbatim: accept ∝ (1+z)^(gamma-1), normalised by its max
    # over the grid so the acceptance probability is <= 1.
    rate_gmax = max(1.0, (1.0 + float(grids["z"][-1])) ** (pop.gamma - 1.0))

    keep, n_tried, n_pass_snr = [], 0, 0
    rej_keep, n_rej_target = [], 20_000
    n_have = 0
    ntry = max(4 * _NEV, 100_000)
    t0 = time.perf_counter()
    while n_have < _NEV:
        # host draw: Bernoulli(f_AGN) tracer choice, uniform within the tracer.
        # Both index arrays are drawn every batch so rng consumption is deterministic.
        u_type = rng.uniform(size=ntry)
        i_gal = rng.integers(0, n_gal, ntry)
        i_agn = rng.integers(0, n_agn, ntry)
        is_agn = u_type < _FAGN
        host_idx = np.where(is_agn, i_agn, i_gal)
        z = np.where(is_agn, cats["agn"]["z"][i_agn], cats["gal"]["z"][i_gal])
        ra = np.where(is_agn, cats["agn"]["ra"][i_agn], cats["gal"]["ra"][i_gal])
        dec = np.where(is_agn, cats["agn"]["dec"][i_agn], cats["gal"]["dec"][i_gal])
        dl = gmd._interp_dl(z, grids)

        m1, use_peak = gmd._sample_powerlaw_peak_m1(rng, ntry, pop, return_component=True)
        q = gmd._sample_q(rng, m1, pop, use_peak=use_peak)
        m2 = q * m1
        chi = gmd._sample_chieff(rng, ntry, pop)

        m1det, m2det = m1 * (1.0 + z), m2 * (1.0 + z)
        obs = _observe(rng, m1det, m2det, chi, dl, ra, dec, need_sky=True)
        det_snr, rho_obs = _detect(obs)
        rho_true = snr_amplitude(m1det, m2det, dl, SNR_REF_DETECT)
        acc = rng.uniform(size=ntry) < (1.0 + z) ** (pop.gamma - 1.0) / rate_gmax
        det = det_snr & acc

        n_tried += ntry
        n_pass_snr += int(det_snr.sum())

        if n_rej_target > 0:
            rmask = ~det_snr
            take = min(int(rmask.sum()), n_rej_target)
            if take:
                idx = np.where(rmask)[0][:take]
                rej_keep.append({"obs_dL": obs["dL"][idx], "obs_m1det": obs["m1det"][idx],
                                 "obs_m2det": obs["m2det"][idx], "rho_obs": rho_obs[idx],
                                 "true_dl": dl[idx], "true_m1det": m1det[idx],
                                 "true_m2det": m2det[idx], "true_z": z[idx],
                                 "rho_true": rho_true[idx]})
                n_rej_target -= take

        if not np.any(det):
            continue
        rec = {"z": z[det], "ra": ra[det], "dec": dec[det], "dl": dl[det],
               "m1src": m1[det], "m2src": m2[det], "q": q[det], "chieff": chi[det],
               "m1det": m1det[det], "m2det": m2det[det],
               "host_type": is_agn[det].astype(np.int64),
               "host_index": host_idx[det].astype(np.int64),
               "snr_obs": rho_obs[det], "snr_true": rho_true[det]}
        for k, v in obs.items():
            rec[f"obs_{k}"] = v[det]
        keep.append(rec)
        n_have += int(det.sum())
        _log(f"events: {n_have}/{_NEV} detected from {n_tried:,} proposed")

    n_detected_total = int(n_have)
    truth = {k: np.concatenate([x[k] for x in keep])[:_NEV] for k in keep[0]}
    obs = {k[4:]: v for k, v in truth.items() if k.startswith("obs_")}
    # The loop stops on a batch boundary, so the last batch usually overshoots.
    # ``det_frac`` uses EVERY detection found in ``n_tried`` proposals -- that is the
    # unbiased per-proposal detection probability.  _NEV/n_tried would discard the
    # overshoot and bias the number low.
    det_frac = n_detected_total / n_tried

    # --- PE: exact flat-prior posteriors of the STORED measurement ---------------
    post = _post(rng, obs, _NSA)
    z_pe = np.interp(post["dL"], grids["dl"], grids["z"])
    post["m1src"] = post["m1det"] / (1.0 + z_pe)
    post["m2src"] = post["m2det"] / (1.0 + z_pe)
    # p_pe: the PE PRIOR at each sample, which is what darksirens divides by --  NOT
    # the posterior the samples were drawn from.  darksirens re-normalises p_pe per
    # event, so only the SHAPE matters; it is stored mean-1 per event for
    # readability.  `p_pe_unity` is the alternative stored convention (prior taken
    # flat in the canonical basis); it too is a statement about the PRIOR only.
    #
    # v2: the prior is flat in (m1det, m2det, dL, chieff, ra, dec); the canonical
    #     basis is (m1det, q = m2det/m1det, dL, chieff) and dm2det = m1det dq, so
    #     p_pe ~ m1det.
    # v3: the prior is flat in (ln Mc_det, ln q, rho, chieff, ra, dec) and the same
    #     rule gives p_pe ~ rho/(dL m1det q) -- see p_pe_v3 and DESIGN_PE.md 2.5.
    if pe_model == "v3":
        q_pe = post["q"]
        raw = p_pe_v3(post["m1det"], q_pe, post["dL"], mc_det=post["mc_det"])
    else:
        raw = post["m1det"]
    raw_e = raw.reshape(_NEV, _NSA)
    post["p_pe"] = (raw_e / raw_e.mean(axis=1, keepdims=True)).ravel()
    p_pe_unity = np.ones(_NEV * _NSA)
    # `mc_det`, `q` and `rho` are the measurement-basis coordinates of the same
    # samples; they are stored so the PE can be PIT-tested channel by channel
    # without re-deriving the bijection.  They are exact functions of the stored
    # (m1det, m2det, dL) and are re-derived and compared bitwise in validation.
    if pe_model != "v3":
        post.pop("mc_det", None)
        post.pop("q", None)
        post.pop("rho", None)

    n_agn_ev = int(truth["host_type"].sum())
    uniq_agn = np.unique(truth["host_index"][truth["host_type"] == 1])
    uniq_gal = np.unique(truth["host_index"][truth["host_type"] == 0])
    cnt_agn = np.bincount(truth["host_index"][truth["host_type"] == 1].astype(int))
    cnt_gal = np.bincount(truth["host_index"][truth["host_type"] == 0].astype(int))

    meas_v3 = {
        "family": "v3 (GWMockCat lineage; see working/data/DESIGN_PE.md)",
        "basis": "(ln Mc_det, ln q, rho, chieff, ra, dec)",
        "snr": f"rho_obs = rho_opt(theta) + N(0, {SIGMA_RHO}); rho_opt = "
               f"{SNR_REF_DETECT} (Mc_det/30)^(5/6) (1000 Mpc/dL); NO projection "
               "latent, so rho is an exact function of (Mc_det, dL) and the SNR IS "
               "the distance observable (dL is never measured separately)",
        "widths": {
            "sigma_lnMc": f"A_MC * ({SNR_THRESHOLD}/rho_obs), A_MC = {A_MC}",
            "sigma_lnq": f"A_Q * ({SNR_THRESHOLD}/rho_obs), A_Q = {A_Q}",
            "sigma_chieff": f"A_CHI * ({SNR_THRESHOLD}/rho_obs), A_CHI = {A_CHI}",
            "sigma_ang": f"clip({SKY_A_DEG} deg / "
                         f"(({SNR_REF_SIGMA}/{SNR_REF_DETECT}) rho_obs), "
                         f"{SKY_CLIP_DEG[0]}, {SKY_CLIP_DEG[1]}) deg",
            "sigma_ra": f"sigma_ang / max(cos dec_obs, {COS_DEC_FLOOR})  (b2)"},
        "constants_source": {
            "sigma_rho, A_MC, A_CHI, threshold": "GWMockCat (Farah et al. 2023, "
                "ApJ 955, 107; arXiv:2301.00834 App. A) posterior_utils.uncert_default "
                "+ parser.py defaults: snr=1.0, mc=0.08, Xeff_uncert=0.2, "
                "threshold_snr=8; original prescription Fishbach, Holz & Farr 2018 "
                "(arXiv:1805.10270) eqs. 29-31",
            "A_Q": "converted from GWMockCat eta_uncert=0.022 and anchored on "
                   "GW150914 q = 0.86 (+0.14/-0.21) at rho ~ 24 "
                   "(arXiv:1602.03840); see DESIGN_PE.md 2.3",
            "sky": "the campaign's own 35 deg/rho_sigma convention, retained; "
                   "1/rho is the Fisher scaling (Fairhurst 2009 arXiv:0908.2356)"},
        "no_clipping_of_data": (
            "NO recorded value is clipped or truncated.  Clipping the DATA censors "
            "the likelihood and gives it a theta-dependent normalisation; the "
            "physical ranges q <= 1, rho > 0, |chieff| <= 1, |dec| <= pi/2 are "
            "imposed on the PE PRIOR instead, which is exact."),
    }
    pe_v3 = {
        "ln Mc_det": "N(ln Mc_obs, sig_lnmc)                       (exact)",
        "ln q": "N(ln q_obs, sig_lnq) truncated to ln q <= 0       (exact)",
        "rho": "N(rho_obs, sigma_rho) truncated to rho > 0         (exact)",
        "chieff": "N(chieff_obs, sig_chieff) truncated to [-1, 1]  (exact)",
        "dec": "N(dec_obs, sigma_ang) truncated to [-pi/2, pi/2]   (exact)",
        "ra": "wrapped N(ra_obs, sig_ra)                           (exact)",
        "map_to_storage": "m1det = Mc (1+q)^(1/5) q^(-3/5); m2det = q m1det; "
                          f"dL = 1000 * {SNR_REF_DETECT} (Mc/30)^(5/6) / rho",
        "p_pe": "PE PRIOR in the canonical (m1det, q, dL, chieff) basis = "
                "|d(ln Mc, ln q, rho)/d(m1det, q, dL)| = rho/(dL m1det q); "
                "see DESIGN_PE.md 2.5 and validation check V3c",
        "q_le_1_by_construction": True,
    }
    meta = {
        "generated_at_utc": _now(), "seed_events": _SEED_EV,
        "seed_events_is_record_default": bool(_SEED_EV == seeds["events"]),
        "n_events": _NEV, "nsamp": _NSA,
        "events_suffix": _sfx,
        "pe_model_version": pe_model,
        "planted_f_agn": _FAGN,
        "planted_f_agn_is_record_default": bool(_FAGN == F_AGN),
        "gamma": GAMMA,
        "cosmology": {"H0": H0_FID, "Om0": OM0_FID, "w0": W0_FID, "wa": WA_FID,
                      "zmax_grid": ZMAX_GRID},
        "measurement_model": meas_v3 if pe_model == "v3" else {
            "sigma_dL_fractional_multiplicative": SIGMA_DL,
            "m1det_fractional": SIG_M1_FRAC, "m2det_fractional": SIG_M2_FRAC,
            "chieff_absolute": SIGMA_CHIEFF,
            "sky_width": "clip(35/rho_opt(OBSERVED m1det,m2det,dL), 1, 12) deg "
                         f"on the snr_ref={SNR_REF_SIGMA} amplitude scale, drawn "
                         "sequentially AFTER the distance/mass measurement",
            "ra_width": "sigma_ang / max(cos dec_OBS, 0.1), stored as obs_sig_ra; "
                        "dec is measured BEFORE ra (convention b2, 2026-08-01)",
            "mass_noise": "obs ~ N(m, f*m) with f constant -- the width is set by the "
                          "LATENT mass, so the flat-prior posterior is skewed"},
        "pe_model": pe_v3 if pe_model == "v3" else {
            "dL": "ln dL ~ N(ln d_obs + s^2, s)  (exact, closed form)",
            "m1det/m2det": "p(m|obs) ~ (1/(f m)) exp[-(obs-m)^2/(2 f^2 m^2)], drawn "
                           "by inverse CDF on a uniform ln(obs/m) grid  (exact; "
                           "convention c2, 2026-08-01)",
            "pex_grid": {"n_grid": PEX_N_GRID, "y_cap": PEX_Y_CAP,
                         "n_sig": PEX_N_SIG,
                         "support": "obs/(1+n_sig f) <= m <= obs/y_cap"},
            "ra": "N(ra_obs, obs_sig_ra)   dec: N(dec_obs, sigma_ang)   (exact)",
            "chieff": "N(chieff_obs, sigma_chieff)  (exact)",
            "p_pe": "PE PRIOR in the canonical (m1det, q, dL, chieff) basis, "
                    "proportional to m1det; unchanged by the (b2)/(c2) fixes",
            "changed_2026_08_01": [
                "c2: mass PE is the exact flat-prior posterior of obs ~ N(m, f m), "
                "not a Gaussian of latent width f*m_true about obs",
                "b2: dec measured before ra; RA width from cos(dec_obs), stored as "
                "obs_sig_ra and reused verbatim by the PE"],
            "detected_set_unaffected": (
                "detection reads only (obs m1det, m2det, dL), which neither fix "
                "touches: (c2) acts after the event loop, (b2) reorders two draws "
                "of equal count inside the sky block that detection never reads.  "
                "The injections stage calls observe(need_sky=False) and never reads "
                "events.h5, so both selection campaigns are bit-identical too.")},
        "detection": {"rule": ("rho_obs >= snr_threshold, rho_obs = rho_opt(theta) "
                               "+ N(0, sigma_rho)" if pe_model == "v3"
                               else "rho_obs(observed data) >= snr_threshold"),
                      "snr_threshold": SNR_THRESHOLD,
                      "snr_ref_detect": SNR_REF_DETECT,
                      "snr_ref_sigma_ang": SNR_REF_SIGMA,
                      "sigma_rho": SIGMA_RHO if pe_model == "v3" else None,
                      "P_det_closed_form": ("Phi((rho_opt(theta) - 8)/sigma_rho)"
                                            if pe_model == "v3" else None),
                      "true_redshift_cut": None, "projection_latent": False,
                      "pe_shares_noise_draw": True},
        "realised": {
            "n_proposed": int(n_tried),
            "n_detected_total": n_detected_total,
            "detected_fraction": det_frac,
            "detected_fraction_first_nobs_only": _NEV / n_tried,
            "detected_fraction_snr_only": n_pass_snr / n_tried,
            "horizon_z_max_detected": float(truth["z"].max()),
            "z_median_detected": float(np.median(truth["z"])),
            "z_min_detected": float(truth["z"].min()),
            "dL_max_detected_Mpc": float(truth["dl"].max()),
            "n_host_gal": int(_NEV - n_agn_ev), "n_host_agn": n_agn_ev,
            "realised_f_agn": n_agn_ev / _NEV,
            "unique_agn_hosts": int(uniq_agn.size),
            "unique_gal_hosts": int(uniq_gal.size),
            "max_events_per_agn_host": int(cnt_agn.max()) if cnt_agn.size else 0,
            "max_events_per_gal_host": int(cnt_gal.max()) if cnt_gal.size else 0,
            "agn_host_multiplicity_hist": {
                str(k): int(v) for k, v in
                zip(*np.unique(cnt_agn[cnt_agn > 0], return_counts=True))},
            "gal_host_multiplicity_hist": {
                str(k): int(v) for k, v in
                zip(*np.unique(cnt_gal[cnt_gal > 0], return_counts=True))},
            "snr_obs_min": float(truth["snr_obs"].min()),
            "snr_obs_max": float(truth["snr_obs"].max()),
            "frac_detected_with_true_snr_below_threshold":
                float((truth["snr_true"] < SNR_THRESHOLD).mean()),
        },
        "darksirens_worktree": str(args.darksirens),
        "darksirens_sha": _git(args.darksirens, "rev-parse", "HEAD"),
        "population": {k: getattr(pop, k) for k in pop.__dataclass_fields__},
    }

    with h5py.File(ev_path, "w") as f:
        f.attrs["format_version"] = "gwcat-1.0"
        f.attrs["mock_data"] = True
        f.attrs["nobs"] = int(_NEV)
        f.attrs["nsamp"] = int(_NSA)
        f.attrs["pe_cosmology_H0"] = float(H0_FID)
        f.attrs["pe_cosmology_Om0"] = float(OM0_FID)
        f.attrs["chi_eff_in_p_pe"] = True
        f.attrs["chi_eff_amax"] = float(CHIEFF_AMAX)
        f.attrs["pe_centering"] = "observed"
        f.attrs["pe_model"] = pe_model
        f.attrs["pe_mass_model"] = (
            "exact_flatprior_posterior_in_(lnMc,lnq,rho); widths a_x*8/rho_obs"
            if pe_model == "v3" else "exact_flatprior_posterior_of_N(m,f*m)")
        f.attrs["ra_width_convention"] = "sigma_ang/max(cos dec_obs, 0.1)"
        if pe_model == "v3":
            f.attrs["sigma_rho"] = float(SIGMA_RHO)
            f.attrs["a_mc"] = float(A_MC)
            f.attrs["a_q"] = float(A_Q)
            f.attrs["a_chi"] = float(A_CHI)
            f.attrs["sky_a_deg"] = float(SKY_A_DEG)
        f.attrs["pop_model"] = "powerlaw+peak"
        f.attrs["shared_beta"] = True
        f.attrs["shared_spin"] = True
        f.attrs["shared_gamma"] = True
        f.attrs["detection_rule"] = "observed-data"
        f.attrs["detection_shares_noise_with_pe"] = True
        f.attrs["sky_width"] = "observed"
        f.attrs["snr_threshold"] = float(SNR_THRESHOLD)
        f.attrs["snr_ref"] = float(SNR_REF_DETECT)
        f.attrs["snr_ref_sigma_ang"] = float(SNR_REF_SIGMA)
        f.attrs["p_pe_basis"] = (
            "rho/(dL m1det q): the PE prior is flat in (ln Mc_det, ln q, rho, "
            "chieff, ra, dec) and the canonical basis is (m1det, q, dL, chieff); "
            "stored mean-1 per event"
            if pe_model == "v3" else
            "proportional to m1det: the PE prior is flat in "
            "(m1det, m2det, dL, chieff) and the canonical basis "
            "is (m1det, q=m2det/m1det, dL, chieff); stored "
            "mean-1 per event")
        f.attrs["host_order"] = "as_drawn"
        f.attrs["n_host_gal"] = int(_NEV - n_agn_ev)
        f.attrs["n_host_agn"] = int(n_agn_ev)
        f.attrs["truth_f_agn"] = float(n_agn_ev / _NEV)
        f.attrs["planted_f_agn"] = float(_FAGN)
        f.attrs["seed_events"] = int(_SEED_EV)
        f.attrs["gamma"] = float(GAMMA)
        f.attrs["dL_fractional_uncertainty"] = (
            float(np.sqrt((5.0 / 6.0 * A_MC * SNR_THRESHOLD) ** 2 + SIGMA_RHO ** 2))
            if pe_model == "v3" else float(SIGMA_DL))
        if pe_model == "v3":
            f.attrs["dL_fractional_uncertainty_note"] = (
                "the number above is a_dL with sigma_ln dL = a_dL/rho_obs; dL is "
                "derived from (Mc_det, rho), so its width is not a free constant")
        f.attrs["n_proposed_for_events"] = int(n_tried)
        f.attrs["metadata_json"] = json.dumps(meta, default=str)
        for k, v in post.items():
            f.create_dataset(k, data=v, compression="gzip", shuffle=True)
        f.create_dataset("p_pe_unity", data=p_pe_unity, compression="gzip", shuffle=True)
        # per-event convenience columns (campaign tooling reads these at top level)
        for k in ("host_type", "host_index"):
            f.create_dataset(k, data=truth[k])
        for k, src in (("true_z", "z"), ("true_m1src", "m1src"),
                       ("true_m2src", "m2src"), ("true_chieff", "chieff"),
                       ("true_dL", "dl")):
            f.create_dataset(k, data=truth[src])
        g = f.create_group("truth")
        for k, v in truth.items():
            g.create_dataset(k, data=v)
    _log(f"wrote {ev_path} ({ev_path.stat().st_size / 1e6:.1f} MB)")

    if rej_keep:
        rp = out / f"events_rejected_sample{_sfx}.h5"
        rej = {k: np.concatenate([c[k] for c in rej_keep]) for k in rej_keep[0]}
        with h5py.File(rp, "w") as f:
            f.attrs["description"] = (
                "Sample of proposals REJECTED by the observed-data SNR cut, with their "
                "recorded measurement. Lets the validation prove detection is a "
                "two-sided deterministic function of the data.")
            f.attrs["snr_threshold"] = float(SNR_THRESHOLD)
            f.attrs["snr_ref_detect"] = float(SNR_REF_DETECT)
            for k, v in rej.items():
                f.create_dataset(k, data=v, compression="gzip", shuffle=True)
        _log(f"wrote {rp} ({rej['obs_dL'].size:,} rejected proposals)")

    _write_json(out / f"events_meta{_sfx}.json", meta)
    if not _sfx:
        _merge_meta(args.seed, "events", meta, args.outroot)
    _log(f"events done in {time.perf_counter() - t0:.1f} s: "
         f"horizon z_max={meta['realised']['horizon_z_max_detected']:.4f}, "
         f"detected fraction {det_frac:.4e}, "
         f"f_AGN realised {meta['realised']['realised_f_agn']:.3f}")


# ================================================================================
# STAGE 3 -- survey selection + pixelation
# ================================================================================
def stage_surveys(args):
    """Isotropic apparent-magnitude limits, then pixelation into survey files.

    The flux limit is ISOTROPIC and has no hard redshift cut, so completeness C(z) is
    a CONSEQUENCE of survey depth rather than an input, and the incompleteness the
    inference must model is isotropic by construction (an anisotropic completeness
    modelled as isotropic would imprint a sky-density contrast -- exactly the channel
    that identifies f_AGN).
    """
    import h5py
    gmd = import_gmd(args.darksirens)
    sd = seed_dir(args.seed, args.outroot)
    cat_dir, srv_dir = sd / "catalogs", sd / "surveys"
    srv_dir.mkdir(parents=True, exist_ok=True)

    ev_meta = _read_json(sd / "events" / "events_meta.json")
    horizon = float(ev_meta["realised"]["horizon_z_max_detected"])
    _log(f"surveys: measured GW horizon z = {horizon:.4f} (completeness reference)")

    photoz_survey = getattr(args, "photoz_survey", PHOTOZ_SURVEY_DEFAULT)
    cats = {t: load_catalog(cat_dir / f"catalog_{t}_complete.h5") for t in TRACERS}
    if photoz_survey == "obs" and "z_obs" not in cats["gal"]:
        raise SystemExit(
            "surveys: --photoz_survey obs needs the catalogs' z_obs column (D3); "
            "regenerate --stage catalogs, or pass --photoz_survey true.")
    _log(f"surveys: pixelating on the catalog's "
         f"{'OBSERVED (photo-z) redshift z_obs' if photoz_survey == 'obs' else 'TRUE redshift z'}")
    levels = [("complete", None)] + [(f"m{int(m)}", m) for m in MAG_LIMITS]
    rec = {"generated_at_utc": _now(), "nside": NSIDE_SURVEY,
           "photoz_survey": photoz_survey,
           "z_column": ("z_obs" if photoz_survey == "obs" else "z"),
           "dz_convention": f"dz = {DZ_SCALE} * (1 + z_"
                            f"{'obs' if photoz_survey == 'obs' else 'true'})",
           "dz_scale": DZ_SCALE, "z_pad": Z_PAD, "dz_pad": DZ_PAD, "w_pad": W_PAD,
           "horizon_z": horizon, "magnitude_limits": list(MAG_LIMITS),
           "completeness": {}, "catalogs": {}, "surveys": {}}

    edges = np.linspace(0.0, horizon, 7)
    for tag, mlim in levels:
        rec["completeness"][tag] = {}
        for t in TRACERS:
            c = cats[t]
            keep = (np.ones(c["z"].size, dtype=bool) if mlim is None
                    else c["app_mag"] < mlim)
            in_h = c["z"] <= horizon
            cz = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                b = (c["z"] > lo) & (c["z"] <= hi)
                cz.append(float(keep[b].mean()) if b.any() else None)
            comp = {"mag_limit": mlim,
                    "n_kept": int(keep.sum()), "n_total": int(keep.size),
                    "C_all_z": float(keep.mean()),
                    "C_within_horizon": float(keep[in_h].mean()) if in_h.any() else None,
                    "n_within_horizon_kept": int((keep & in_h).sum()),
                    "n_within_horizon_total": int(in_h.sum()),
                    "C_of_z_bins": {"edges": edges.tolist(), "C": cz}}
            rec["completeness"][tag][t] = comp

            # no copy at the complete level: `c` is up to 1.6e8 rows in v2
            sub = c if mlim is None else {k: v[keep] for k, v in c.items()}
            # incomplete catalog pairs, alongside the complete pair
            if mlim is not None:
                cp = cat_dir / f"catalog_{t}_{tag}.h5"
                if cp.exists() and not args.overwrite:
                    _log(f"surveys: {cp} exists; reusing")
                else:
                    with h5py.File(cp, "w") as f:
                        f.attrs["mock_data"] = True
                        f.attrs["tracer"] = t
                        f.attrs["complete"] = False
                        f.attrs["mag_limit"] = float(mlim)
                        f.attrs["source_complete_catalog"] = str(
                            cat_dir / f"catalog_{t}_complete.h5")
                        f.attrs["description"] = (
                            f"{t.upper()} tracer after the ISOTROPIC flux limit "
                            f"app_mag < {mlim}. AGN carry their host galaxy's "
                            "apparent magnitude, so one limit thins both tracers "
                            "with the same C(z).")
                        f.attrs["completeness_json"] = json.dumps(comp, default=str)
                        for k, v in sub.items():
                            f.create_dataset(k, data=v, compression="gzip", shuffle=True)
                    _log(f"wrote {cp} ({sub['z'].size:,} objects)")
                rec["catalogs"][f"{t}_{tag}"] = {
                    "path": str(cp), "n": int(sub["z"].size),
                    "size_bytes": cp.stat().st_size}
            else:
                cp = cat_dir / f"catalog_{t}_complete.h5"
                rec["catalogs"][f"{t}_{tag}"] = {
                    "path": str(cp), "n": int(sub["z"].size),
                    "size_bytes": cp.stat().st_size}

            # --- pixelate into the darksirens survey format --------------------
            sp = srv_dir / f"survey_{t}_{tag}_ns{NSIDE_SURVEY}.h5"
            if sp.exists() and not args.overwrite:
                _log(f"surveys: {sp} exists; reusing")
            else:
                # D3 (DESIGN_PE.md 3.3): the block carries the survey's OBSERVED
                # redshift, which now really does carry the declared error.  The
                # declared width is formed from the RECORDED value, because that is
                # all the survey knows; the difference from DZ_SCALE (1+z_true) is
                # O(DZ_SCALE^2) and is measured by validation check V9.
                zsrv = sub["z_obs"] if photoz_survey == "obs" else sub["z"]
                dz = (DZ_SCALE * (1.0 + zsrv)).astype(zsrv.dtype)
                w = np.ones_like(zsrv)
                pix = pixelate_catalog_vec(sub["ra"], sub["dec"], zsrv, dz, w,
                                           NSIDE_SURVEY,
                                           dtype=np.dtype(CAT_DTYPE))
                # gmd pads z with 100.0 / dz with 1.0 / w with 0.0 -- the same
                # sentinels convert_catalogs.py used, so downstream kernel sums that
                # mask on `ngals` are unaffected by the padding.
                ngals = np.asarray(pix["ngals"])
                occupied = int((ngals > 0).sum())
                npix = int(ngals.size)
                with h5py.File(sp, "w") as f:
                    for k, v in pix.items():
                        f.create_dataset(k, data=np.asarray(v),
                                         compression="gzip", shuffle=True)
                    f.attrs["nside"] = int(NSIDE_SURVEY)
                    f.attrs["tracer"] = t
                    f.attrs["level"] = tag
                    f.attrs["mag_limit"] = (float(mlim) if mlim is not None else -1.0)
                    f.attrs["n_hosts"] = int(sub["z"].size)
                    f.attrs["z_min"] = float(zsrv.min())
                    f.attrs["z_max"] = float(zsrv.max())
                    f.attrs["z_min_true"] = float(sub["z"].min())
                    f.attrs["z_max_true"] = float(sub["z"].max())
                    f.attrs["dz_scale"] = float(DZ_SCALE)
                    f.attrs["z_column"] = ("z_obs" if photoz_survey == "obs" else "z")
                    f.attrs["photoz_realised"] = bool(photoz_survey == "obs")
                    f.attrs["dz_convention"] = (
                        f"dz = {DZ_SCALE} * (1 + z_obs); zgals ARE the catalog's "
                        "photo-z redshifts and really do carry this error (D3)"
                        if photoz_survey == "obs"
                        else f"dz = {DZ_SCALE} * (1 + z)")
                    f.attrs["occupied_pixels"] = occupied
                    f.attrs["empty_pixel_fraction"] = float(1.0 - occupied / npix)
                    f.attrs["source_catalog"] = str(cp)
                    f.attrs["completeness_json"] = json.dumps(comp, default=str)
                    f.attrs["built_by"] = str(Path(__file__).resolve())
                    f.attrs["built_at_utc"] = _now()
                _log(f"wrote {sp} ({sub['z'].size:,} hosts, "
                     f"{100 * (1 - occupied / npix):.2f}% empty pixels)")
            with h5py.File(sp, "r") as f:
                rec["surveys"][f"{t}_{tag}"] = {
                    "path": str(sp), "n_hosts": int(f.attrs["n_hosts"]),
                    "empty_pixel_fraction": float(f.attrs["empty_pixel_fraction"]),
                    "max_hosts_per_pixel": int(f["zgals"].shape[1]),
                    "size_bytes": sp.stat().st_size}

    _write_json(srv_dir / "surveys_meta.json", rec)
    _merge_meta(args.seed, "surveys", rec, args.outroot)
    _log("surveys done")


# ================================================================================
# STAGE 4 -- selection campaigns
# ================================================================================
class SurveyPixelMap:
    """The AGN tracer's KDE support, read from the file the LIKELIHOOD conditions on.

    Adapted from ``build_targeted_injections_k2.py``.  Reading the PIXELATED SURVEY
    (not the raw catalog) is the point: that file's ``zgals``/``dzgals`` ARE the
    kernel centres and widths the target density uses, so the proposal cannot drift
    away from the target through a pixelisation or kernel-width mismatch.

    v2 -- H0-RANGE-COVERING BOXES.  v1's branch drew ``z ~ TN(z_j, sigma_j)``, i.e.
    it planted injections exactly ON the catalog kernels of the FIDUCIAL cosmology.
    Because the likelihood re-reads a stored injection at trial ``H0`` as the
    redshift ``z'`` with ``dL_fid(z') = dL_fid(z) H0/H0_FID``, that branch only
    overlaps the catalog prior near ``H0_FID`` and its ``N_eff`` collapsed by three
    orders of magnitude across the scan.

    Each retained host now carries a UNIFORM BOX

        [ L_j, U_j ] = [ max(0, R_LO (z_j - n sigma_j)),
                         min(z_cap, R_HI (z_j + n sigma_j)) ],
        R_LO = H0_FID/H0_max,  R_HI = H0_FID/H0_min,

    whose image under every trial ``H0`` in the scanned range still contains the
    host's kernel.  The density is the flat mixture
    ``(1/N_kept) sum_{j in pix} 1[L_j <= z <= U_j] / (U_j - L_j)`` -- closed form,
    so ``pdraw`` stays EXACT (validation V6 recomputes it from the flat host list).
    Hosts deeper than ``host_zmax`` are dropped and the box is capped at ``z_cap``:
    neither can ever be detected, and the population/uniform branches keep full
    support, so no hole is opened in ``pdraw``."""

    def __init__(self, path, zmax, host_zmax=TGT_HOST_ZMAX, z_cap=TGT_Z_CAP,
                 r_lo=TGT_R_LO, r_hi=TGT_R_HI, nsig=TGT_NSIG_PAD):
        import h5py
        import healpy as hp
        with h5py.File(path, "r") as f:
            zgals = np.asarray(f["zgals"][:], dtype=np.float64)
            dzgals = np.asarray(f["dzgals"][:], dtype=np.float64)
            wgals = np.asarray(f["wgals"][:], dtype=np.float64)
            ngals = np.asarray(f["ngals"][:], dtype=np.int64)
            nside = int(f.attrs["nside"])
            tracer = f.attrs.get("tracer", Path(path).stem)
        npix, maxcol = zgals.shape
        if npix != hp.nside2npix(nside):
            raise SystemExit(f"{path}: npix {npix} inconsistent with nside {nside}")
        valid = np.arange(maxcol)[None, :] < ngals[:, None]
        w = wgals[valid]
        if w.size and not np.allclose(w, w.flat[0], rtol=0, atol=0):
            raise SystemExit(f"{path}: catalog weights are not uniform; the targeted "
                             "branch draws hosts uniformly and its density assumes "
                             "equal weights")
        self.path, self.name, self.nside, self.npix = str(path), str(tracer), nside, npix
        self.apix = float(hp.nside2pixarea(nside))
        self.resol = float(hp.nside2resol(nside))
        self.zmax = float(zmax)
        self.N_catalog = int(ngals.sum())
        self.empty_pixel_fraction_catalog = float(1.0 - (ngals > 0).mean())

        # --- flat host list, then the box and the host cut ----------------------
        z_all = zgals[valid]
        s_all = dzgals[valid]
        pix_all = np.repeat(np.arange(npix), ngals)
        cap = float(min(z_cap, zmax))
        lo_all = np.maximum(0.0, r_lo * (z_all - nsig * s_all))
        hi_all = np.minimum(cap, r_hi * (z_all + nsig * s_all))
        keep = (z_all <= float(host_zmax)) & (hi_all > lo_all)
        self.host_zmax, self.z_cap = float(host_zmax), cap
        self.r_lo, self.r_hi, self.nsig = float(r_lo), float(r_hi), float(nsig)
        self.z_hosts = z_all[keep]
        self.sig_hosts = s_all[keep]
        self.lo_hosts = lo_all[keep]
        self.hi_hosts = hi_all[keep]
        self.pix_hosts = pix_all[keep]
        self.N_hosts = int(self.z_hosts.size)
        if self.N_hosts == 0:
            raise SystemExit(f"{path}: no host survives z <= {host_zmax}")

        # --- repack the retained hosts into padded rows -------------------------
        counts = np.bincount(self.pix_hosts, minlength=npix).astype(np.int64)
        mc = int(max(1, counts.max()))
        order = np.argsort(self.pix_hosts, kind="stable")
        starts = np.concatenate([[0], np.cumsum(counts)])[:-1]
        col = np.arange(self.N_hosts) - starts[self.pix_hosts[order]]
        flat = self.pix_hosts[order] * mc + col
        self.maxcol = mc
        self.counts = counts
        self.n_occupied = int((counts > 0).sum())
        self.empty_pixel_fraction = float(1.0 - (counts > 0).mean())
        self.finite_pad = np.zeros((npix, mc), dtype=bool)
        self.lo_pad = np.zeros((npix, mc))
        self.hi_pad = np.ones((npix, mc))
        self.finite_pad.reshape(-1)[flat] = True
        self.lo_pad.reshape(-1)[flat] = self.lo_hosts[order]
        self.hi_pad.reshape(-1)[flat] = self.hi_hosts[order]
        self.width_pad = np.where(self.finite_pad, self.hi_pad - self.lo_pad, 1.0)

    def pix_of(self, ra, dec):
        import healpy as hp
        return hp.ang2pix(self.nside, 0.5 * np.pi - np.asarray(dec), np.asarray(ra))

    def box_sum(self, z_rows, pix_rows):
        """sum_{j in pix} 1[L_j <= z <= U_j] / (U_j - L_j)  (host-count NOT divided)."""
        z_rows = np.asarray(z_rows, dtype=np.float64)
        if z_rows.size == 0:
            return np.zeros(0)
        rows = np.asarray(pix_rows)
        lo, hi = self.lo_pad[rows], self.hi_pad[rows]
        inside = self.finite_pad[rows] & (z_rows[:, None] >= lo) & (z_rows[:, None] <= hi)
        return np.where(inside, 1.0 / self.width_pad[rows], 0.0).sum(axis=1)

    def sample_z_at_hosts(self, rng, j):
        """z ~ U[L_j, U_j] -- exactly the density ``box_sum`` assumes."""
        return self.lo_hosts[j] + rng.uniform(size=j.shape[0]) * (
            self.hi_hosts[j] - self.lo_hosts[j])


def sample_uniform_in_pixels(rng, pix_tgt, nside, resol, maxiter=2000):
    """Uniform-on-sphere WITHIN each target RING pixel (bounding-box rejection).

    Rejection of a uniform proposal restricted to a superset of the pixel is exactly
    uniform on the pixel, which is what makes the targeted sky density the constant
    1/Omega_pix inside it."""
    import healpy as hp
    n = pix_tgt.shape[0]
    theta_c, phi_c = hp.pix2ang(nside, pix_tgt)
    dt = 2.0 * resol
    cos_lo = np.cos(np.minimum(theta_c + dt, np.pi))
    cos_hi = np.cos(np.maximum(theta_c - dt, 0.0))
    sin_c = np.sin(np.clip(theta_c, 1.0e-6, np.pi - 1.0e-6))
    dphi = np.minimum(2.5 * resol / np.maximum(sin_c, 1.0e-2), np.pi)
    out_theta, out_phi = np.empty(n), np.empty(n)
    todo = np.ones(n, dtype=bool)
    for _ in range(maxiter):
        idx = np.where(todo)[0]
        if idx.size == 0:
            break
        u = rng.uniform(cos_lo[idx], cos_hi[idx])
        th = np.arccos(np.clip(u, -1.0, 1.0))
        ph = (phi_c[idx] + rng.uniform(-dphi[idx], dphi[idx])) % (2.0 * np.pi)
        acc = hp.ang2pix(nside, th, ph) == pix_tgt[idx]
        hit = idx[acc]
        out_theta[hit], out_phi[hit] = th[acc], ph[acc]
        todo[hit] = False
    n_fallback = int(todo.sum())
    if n_fallback:
        out_theta[todo], out_phi[todo] = theta_c[todo], phi_c[todo]
    return out_phi, 0.5 * np.pi - out_theta, n_fallback


def p_targeted_density(gmd, z, ra, dec, m1src, q, chi, grids, ddldz_grid, pop, pmap):
    """One targeted branch's proposal density in the canonical basis, per steradian.

        q_t = p_ms(m1src,q,chi) * (1/Omega_pix)
                * sum_{j in pix} (1/N_t) U(z; L_j, U_j)
        p_targeted = q_t / [ (1+z) dL'(z) ]

    Structurally identical to gmd's ``_p_population`` -- same ``_mass_spin_pdf``, same
    ``np.gradient(dl, z)``, same ``(1+z)`` -- with ``p_z(z)/(4 pi)`` replaced by the
    catalog z-sky factor, so all branch densities live in one measure and mix."""
    ksum = pmap.box_sum(z, pmap.pix_of(ra, dec))
    p_zsky = ksum / pmap.N_hosts / pmap.apix
    ddldz = np.interp(z, grids["z"], ddldz_grid)
    jac = ddldz * (1.0 + z)
    msp = gmd._mass_spin_pdf(m1src, q, chi, pop)
    return msp * p_zsky / np.maximum(jac, 1.0e-300)


def _chunked_pdraw(fn, n, chunk=200_000):
    """gmd's mass pdf allocates (n, 500) grids; chunk so peak RAM stays bounded."""
    if n == 0:
        return np.zeros(0)
    return np.concatenate([fn(slice(i, min(i + chunk, n)))
                           for i in range(0, n, chunk)])


# --- multiprocessing worker state ------------------------------------------------
_W = {}


def _inj_init(lane, darksirens, agn_survey, zmax, nz_edges, mix, pe_model="v3"):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    gmd = import_gmd(Path(darksirens))
    cosmo = gmd._build_cosmology(H0_FID, OM0_FID, W0_FID, WA_FID)
    grids = gmd._cosmology_grids(cosmo, zmax)
    _W.update(lane=lane, gmd=gmd, grids=grids, mix=tuple(mix), pe_model=pe_model,
              ddldz=np.gradient(grids["dl"], grids["z"]),
              pop=gmd.PopulationConfig(gamma=GAMMA), zmax=zmax,
              nz_edges=np.asarray(nz_edges),
              pmap=(SurveyPixelMap(agn_survey, zmax) if lane == "targeted" else None))


def _inj_batch(task):
    """One reproducible batch.  Seeds come from SeedSequence children, so the result
    is independent of how many workers run or in what order they finish."""
    bidx, ss, nb = task
    gmd, grids, pop = _W["gmd"], _W["grids"], _W["pop"]
    rng = np.random.default_rng(ss)
    m1lo, m1hi = gmd._M1DET_RANGE
    lane = _W["lane"]

    if lane == "targeted":
        pmap = _W["pmap"]
        mw = _W["mix"]
        edges = np.cumsum([mw[0], mw[1], mw[2]])
        branch = np.searchsorted(edges, rng.uniform(size=nb), side="right").astype(np.int8)
        np.clip(branch, 0, 2, out=branch)
    else:
        # gmd's own population+uniform: Bernoulli(0.9) population else uniform.
        branch = np.where(rng.uniform(size=nb) < 0.9, 0, 1).astype(np.int8)

    z = np.empty(nb); ra = np.empty(nb); dec = np.empty(nb)
    m1src = np.empty(nb); q = np.empty(nb); chi = np.empty(nb)

    is_pop = branch == 0
    npop = int(is_pop.sum())
    if npop:
        zc = gmd._sample_uniform_comoving_z(rng, grids, npop)
        rac, decc = gmd._sample_sky(rng, npop)
        m1c, use_peak = gmd._sample_powerlaw_peak_m1(rng, npop, pop, return_component=True)
        z[is_pop], ra[is_pop], dec[is_pop] = zc, rac, decc
        m1src[is_pop] = m1c
        q[is_pop] = gmd._sample_q(rng, m1c, pop, use_peak=use_peak)
        chi[is_pop] = gmd._sample_chieff(rng, npop, pop)

    is_unif = branch == 1
    nunif = int(is_unif.sum())
    if nunif:
        zc = gmd._sample_uniform_comoving_z(rng, grids, nunif)
        rac, decc = gmd._sample_sky(rng, nunif)
        z[is_unif], ra[is_unif], dec[is_unif] = zc, rac, decc
        m1src[is_unif] = rng.uniform(m1lo, m1hi, nunif) / (1.0 + zc)
        q[is_unif] = rng.uniform(0.0, 1.0, nunif)
        chi[is_unif] = rng.uniform(-1.0, 1.0, nunif)

    n_fallback = 0
    if lane == "targeted":
        pmap = _W["pmap"]
        is_t = branch == 2
        nt = int(is_t.sum())
        if nt:
            j = rng.integers(0, pmap.N_hosts, nt)
            zc = pmap.sample_z_at_hosts(rng, j)
            rac, decc, n_fallback = sample_uniform_in_pixels(
                rng, pmap.pix_hosts[j], pmap.nside, pmap.resol)
            m1c, use_peak = gmd._sample_powerlaw_peak_m1(rng, nt, pop,
                                                         return_component=True)
            z[is_t], ra[is_t], dec[is_t] = zc, rac, decc
            m1src[is_t] = m1c
            q[is_t] = gmd._sample_q(rng, m1c, pop, use_peak=use_peak)
            chi[is_t] = gmd._sample_chieff(rng, nt, pop)

    m2src = q * m1src
    dl = gmd._interp_dl(z, grids)
    # --- DETECTION: the events' own rule, convention (a) ------------------------
    _obs_fn = observe_v3 if _W.get("pe_model", "v3") == "v3" else observe
    _det_fn = detect_v3 if _W.get("pe_model", "v3") == "v3" else detect_from_observation
    obs = _obs_fn(rng, m1src * (1.0 + z), m2src * (1.0 + z), chi, dl, ra, dec,
                  need_sky=False)
    det, _rho = _det_fn(obs)

    zd, rad, decd = z[det], ra[det], dec[det]
    m1d, qd, chid = m1src[det], q[det], chi[det]
    nd = int(det.sum())

    p_pop = _chunked_pdraw(lambda s: gmd._selection_pdraw(
        "population", m1d[s], qd[s], chid[s], zd[s], grids, pop), nd)
    p_unif = _chunked_pdraw(lambda s: gmd._selection_pdraw(
        "uniform", m1d[s], qd[s], chid[s], zd[s], grids, pop), nd)
    if lane == "targeted":
        mw = _W["mix"]
        p_tgt = _chunked_pdraw(lambda s: p_targeted_density(
            gmd, zd[s], rad[s], decd[s], m1d[s], qd[s], chid[s],
            grids, _W["ddldz"], pop, _W["pmap"]), nd)
        pdraw = np.maximum(mw[0] * p_pop + mw[1] * p_unif
                           + mw[2] * p_tgt, PDRAW_FLOOR)
    else:
        p_tgt = np.zeros(nd)
        pdraw = np.maximum(0.9 * p_pop + 0.1 * p_unif, PDRAW_FLOOR)

    # P_det(z) bookkeeping on the POPULATION branch (shared by both lanes and by the
    # event draw), so the lanes can be cross-checked against each other and against
    # the realised event detection fraction.
    e = _W["nz_edges"]
    h_prop = np.histogram(z[is_pop], bins=e)[0]
    h_det = np.histogram(z[is_pop & det], bins=e)[0]

    nb_branch = 3 if lane == "targeted" else 2
    out = {
        "m1det": m1d * (1.0 + zd), "m2det": qd * m1d * (1.0 + zd),
        "m1src": m1d, "m2src": m2src[det], "dL": dl[det], "chieff": chid,
        "ra": rad, "dec": decd, "pdraw": pdraw, "z": zd,
        "branch": branch[det].astype(np.float64),
        "pdraw_population": p_pop, "pdraw_uniform": p_unif,
    }
    if lane == "targeted":
        out["pdraw_targeted_agn"] = p_tgt
    stats = {
        "n_proposed": nb, "n_detected": nd,
        "n_proposed_branch": [int((branch == b).sum()) for b in range(nb_branch)],
        "n_detected_branch": [int(((branch == b) & det).sum()) for b in range(nb_branch)],
        "n_pixel_fallback": n_fallback,
        "hist_proposed_pop": h_prop, "hist_detected_pop": h_det,
    }
    return bidx, out, stats


def stage_injections(args):
    import h5py
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    gmd = import_gmd(args.darksirens)
    sd = seed_dir(args.seed, args.outroot)
    out_dir = sd / "injections"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = sub_seeds(args.seed)
    agn_survey = sd / "surveys" / f"survey_agn_complete_ns{NSIDE_SURVEY}.h5"

    # Fine at low z (where the events are), coarse above -- this grid is only used
    # for the P_det(z) cross-check.
    nz_edges = np.concatenate([np.linspace(0.0, 0.5, 1001),
                               np.linspace(0.5, ZMAX_GRID, 31)[1:]])

    mix = (args.mix_population, args.mix_uniform, args.mix_targeted)
    if abs(sum(mix) - 1.0) > 1e-12:
        raise SystemExit(f"targeted mixture weights must sum to 1, got {mix}")

    lanes = []
    if args.lane in ("both", "targeted"):
        lanes.append(("targeted", seeds["injections_targeted"],
                      out_dir / "injections_targeted.h5",
                      args.ndraw_targeted or args.ndraw))
    if args.lane in ("both", "popuni"):
        lanes.append(("popuni", seeds["injections_popuni"],
                      out_dir / "injections_popuni.h5",
                      args.ndraw_popuni or args.ndraw))

    jobs = args.jobs or max(1, min(48, (os.cpu_count() or 8) - 8))
    all_rec = {}
    for lane, lseed, path, ndraw in lanes:
        if path.exists() and not args.overwrite:
            _log(f"injections: {path} exists; use --overwrite to regenerate")
            all_rec[lane] = _read_json(out_dir / f"injections_{lane}_meta.json")
            continue
        t0 = time.perf_counter()
        nb = args.batch_size
        n_batches = int(np.ceil(ndraw / nb))
        sizes = [min(nb, ndraw - i * nb) for i in range(n_batches)]
        children = np.random.SeedSequence(lseed).spawn(n_batches)
        _log(f"injections[{lane}]: {ndraw:,} proposals in {n_batches} batches "
             f"of {nb:,} on {jobs} workers")

        chunks, stats = [None] * n_batches, [None] * n_batches
        # spawn, not fork: gmd imports jax at module level and forking a
        # thread-carrying interpreter is a known hazard.  Each worker re-imports
        # cleanly; the per-batch SeedSequence children make the result independent
        # of worker count and completion order either way.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx,
                                 initializer=_inj_init,
                                 initargs=(lane, str(args.darksirens),
                                           str(agn_survey), ZMAX_GRID,
                                           nz_edges, mix,
                                           getattr(args, "pe_model",
                                                   PE_MODEL_DEFAULT))) as ex:
            ndone = 0
            for bidx, arrs, st in ex.map(
                    _inj_batch, [(i, children[i], sizes[i]) for i in range(n_batches)],
                    chunksize=1):
                chunks[bidx], stats[bidx] = arrs, st
                ndone += 1
                if ndone % 10 == 0 or ndone == n_batches:
                    _log(f"injections[{lane}]: batch {ndone}/{n_batches} "
                         f"[{time.perf_counter() - t0:.0f} s]")

        keys = list(chunks[0].keys())
        arrays = {k: np.concatenate([c[k] for c in chunks]) for k in keys}
        n_proposed = int(sum(s["n_proposed"] for s in stats))
        n_detected = int(sum(s["n_detected"] for s in stats))
        nb_branch = len(stats[0]["n_proposed_branch"])
        n_prop_b = [int(sum(s["n_proposed_branch"][b] for s in stats))
                    for b in range(nb_branch)]
        n_det_b = [int(sum(s["n_detected_branch"][b] for s in stats))
                   for b in range(nb_branch)]
        n_fb = int(sum(s["n_pixel_fallback"] for s in stats))
        h_prop = np.sum([s["hist_proposed_pop"] for s in stats], axis=0)
        h_det = np.sum([s["hist_detected_pop"] for s in stats], axis=0)

        inv = 1.0 / arrays["pdraw"]
        neff = float(inv.sum() ** 2 / np.square(inv).sum())
        branch_names = (["population", "uniform", "targeted_agn"] if lane == "targeted"
                        else ["population", "uniform"])
        extra = ["z", "branch", "pdraw_population", "pdraw_uniform"] + (
            ["pdraw_targeted_agn"] if lane == "targeted" else [])
        proposal = (f"mixture(population={mix[0]},uniform={mix[1]},"
                    f"targeted_agn={mix[2]})" if lane == "targeted"
                    else "population+uniform")
        tgt_cfg = None
        if lane == "targeted":
            pm = SurveyPixelMap(agn_survey, ZMAX_GRID)
            tgt_cfg = {
                "branch": "per-host UNIFORM BOX in redshift, H0-range covering",
                "H0_scan_range": [H0_SCAN_MIN, H0_SCAN_MAX],
                "r_lo": TGT_R_LO, "r_hi": TGT_R_HI,
                "nsigma_kernel_pad": TGT_NSIG_PAD,
                "host_zmax": TGT_HOST_ZMAX, "z_cap": TGT_Z_CAP,
                "n_hosts_in_catalog": pm.N_catalog,
                "n_hosts_retained": pm.N_hosts,
                "retained_fraction": pm.N_hosts / pm.N_catalog,
                "occupied_pixels_retained": pm.n_occupied,
                "empty_pixel_fraction_retained": pm.empty_pixel_fraction,
                "box_width_min": float((pm.hi_hosts - pm.lo_hosts).min()),
                "box_width_median": float(np.median(pm.hi_hosts - pm.lo_hosts)),
                "box_width_max": float((pm.hi_hosts - pm.lo_hosts).max())}
            del pm

        rec = {
            "generated_at_utc": _now(), "lane": lane, "seed": lseed,
            "ndraw": n_proposed, "n_detected": n_detected,
            "detected_fraction": n_detected / n_proposed,
            "Neff_population_only": neff,
            "selection_proposal": proposal,
            "branch_names": branch_names,
            "n_proposed_branch": dict(zip(branch_names, n_prop_b)),
            "n_detected_branch": dict(zip(branch_names, n_det_b)),
            "n_pixel_placement_fallback": n_fb,
            "targeted_survey": (str(agn_survey) if lane == "targeted" else None),
            "targeted_branch": tgt_cfg,
            "mixture_weights": list(mix) if lane == "targeted" else [0.9, 0.1],
            "pe_model_version": getattr(args, "pe_model", PE_MODEL_DEFAULT),
            "detection": {"rule": (
                              "rho_obs = rho_opt(theta) + N(0, sigma_rho) >= "
                              "snr_threshold"
                              if getattr(args, "pe_model", PE_MODEL_DEFAULT) == "v3"
                              else "rho_obs(observed data) >= snr_threshold"),
                          "snr_threshold": SNR_THRESHOLD,
                          "snr_ref_detect": SNR_REF_DETECT,
                          "sigma_rho": (SIGMA_RHO if getattr(
                              args, "pe_model", PE_MODEL_DEFAULT) == "v3" else None),
                          "P_det_closed_form": (
                              "Phi((rho_opt(theta) - 8)/sigma_rho)"
                              if getattr(args, "pe_model", PE_MODEL_DEFAULT) == "v3"
                              else None),
                          "note": "identical to the events' rule; injections store "
                                  "TRUE parameters because mu(theta) is an integral "
                                  "over true parameters and only the DETECTION "
                                  "decision sees the measurement noise"},
            "batch_size": nb, "n_batches": n_batches, "jobs": jobs,
            "elapsed_s": time.perf_counter() - t0,
            "pdet_z_grid": {"edges": nz_edges.tolist(),
                            "n_proposed_population": h_prop.tolist(),
                            "n_detected_population": h_det.tolist()},
        }

        with h5py.File(path, "w") as f:
            f.attrs["format_version"] = "gwcat-selection-1.0"
            f.attrs["mock_data"] = True
            f.attrs["ndraw"] = int(n_proposed)          # TOTAL PROPOSED
            f.attrs["Ndraw"] = int(n_proposed)
            f.attrs["Neff"] = float(neff)
            f.attrs["n_detected"] = int(n_detected)
            f.attrs["selection_proposal"] = proposal
            f.attrs["chi_eff_swap_applied"] = True
            f.attrs["chi_eff_amax"] = float(CHIEFF_AMAX)
            f.attrs["cosmology_H0"] = float(H0_FID)
            f.attrs["cosmology_Om0"] = float(OM0_FID)
            f.attrs["pop_model"] = "powerlaw+peak"
            f.attrs["shared_beta"] = True
            f.attrs["shared_spin"] = True
            f.attrs["shared_gamma"] = True
            f.attrs["detection_rule"] = ("observed-data: SNR of ONE recorded "
                                         "measurement >= snr_threshold")
            f.attrs["detection_data"] = "observed"
            f.attrs["pe_model"] = getattr(args, "pe_model", PE_MODEL_DEFAULT)
            if getattr(args, "pe_model", PE_MODEL_DEFAULT) == "v3":
                f.attrs["sigma_rho"] = float(SIGMA_RHO)
            f.attrs["snr_threshold"] = float(SNR_THRESHOLD)
            f.attrs["snr_ref"] = float(SNR_REF_DETECT)
            f.attrs["zmax_proposal"] = float(ZMAX_GRID)
            f.attrs["m1det_range_uniform"] = np.asarray(gmd._M1DET_RANGE, dtype=float)
            f.attrs["seed"] = int(lseed)
            if lane == "targeted":
                f.attrs["proposal_mix_population"] = float(mix[0])
                f.attrs["proposal_mix_uniform"] = float(mix[1])
                f.attrs["proposal_mix_targeted_agn"] = float(mix[2])
                f.attrs["targeted_agn_survey"] = str(agn_survey)
                f.attrs["targeted_nside"] = int(NSIDE_SURVEY)
                f.attrs["targeted_branch_json"] = json.dumps(tgt_cfg, default=str)
            for b, name in enumerate(branch_names):
                f.attrs[f"n_proposed_{name}_branch"] = n_prop_b[b]
                f.attrs[f"n_detected_{name}_branch"] = n_det_b[b]
            f.attrs["metadata_json"] = json.dumps(rec, default=str)
            for key in gmd.SELECTION_KEYS + extra:
                f.create_dataset(key, data=np.asarray(arrays[key], dtype=np.float64),
                                 compression="gzip", shuffle=True)
        rec["path"] = str(path)
        rec["size_bytes"] = path.stat().st_size
        _log(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB): "
             f"detected {n_detected:,}/{n_proposed:,} "
             f"({n_detected / n_proposed:.4e}), Neff(pop-only)={neff:.1f}, "
             f"{rec['elapsed_s']:.0f} s")
        _write_json(out_dir / f"injections_{lane}_meta.json", rec)
        all_rec[lane] = rec

    _merge_meta(args.seed, "injections", all_rec, args.outroot)


# ================================================================================
# STAGE 5 -- validation
# ================================================================================
def stage_validation(args):
    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.special import ndtr
    from scipy.stats import kstest

    gmd = import_gmd(args.darksirens)
    sd = seed_dir(args.seed, args.outroot)
    vdir = sd / "validation"
    vdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(sub_seeds(args.seed)["validation"])
    results, failures = {}, []

    def check(name, ok, detail):
        results[name] = {"pass": bool(ok), **detail}
        _log(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failures.append(name)

    ev_path = sd / "events" / "events.h5"
    with h5py.File(ev_path, "r") as f:
        nobs, nsamp = int(f.attrs["nobs"]), int(f.attrs["nsamp"])
        pe_model = str(f.attrs.get("pe_model", "v2"))
        tr = {k: np.asarray(f["truth"][k][:]) for k in f["truth"].keys()}
        _pe_keys = ["dL", "m1det", "m2det", "chieff", "ra", "dec", "p_pe",
                    "m1src", "m2src"]
        if pe_model == "v3":
            _pe_keys += ["mc_det", "q", "rho"]
        pe = {k: np.asarray(f[k][:]) for k in _pe_keys}
    _log(f"validation: measurement family {pe_model}")

    if pe_model == "v2":
        # ---------------------------------------------------------------- V1 detection
        _log("V1: detection is a deterministic function of the observed data")
        rho_re = snr_amplitude(tr["obs_m1det"], tr["obs_m2det"], tr["obs_dL"],
                               SNR_REF_DETECT)
        exact = bool(np.array_equal(rho_re, tr["snr_obs"]))
        all_pass = bool(np.all(rho_re >= SNR_THRESHOLD))
        rp = sd / "events" / "events_rejected_sample.h5"
        rej_ok, n_rej, rej_exact = True, 0, True
        if rp.exists():
            with h5py.File(rp, "r") as f:
                rj = {k: np.asarray(f[k][:]) for k in f.keys()}
            rho_rej = snr_amplitude(rj["obs_m1det"], rj["obs_m2det"], rj["obs_dL"],
                                    SNR_REF_DETECT)
            rej_exact = bool(np.array_equal(rho_rej, rj["rho_obs"]))
            rej_ok = bool(np.all(rho_rej < SNR_THRESHOLD))
            n_rej = int(rho_rej.size)
        frac_true_below = float((tr["snr_true"] < SNR_THRESHOLD).mean())
        rej_true_above = (float((rj["rho_true"] >= SNR_THRESHOLD).mean())
                          if rp.exists() else None)
        check("V1_detection_deterministic_in_data",
              exact and all_pass and rej_ok and rej_exact,
              {"recomputed_equals_stored_bitwise": exact,
               "all_detected_rho_obs_ge_threshold": all_pass,
               "n_rejected_checked": n_rej,
               "all_rejected_rho_obs_lt_threshold": rej_ok,
               "rejected_recompute_bitwise": rej_exact,
               "frac_detected_with_TRUE_snr_below_threshold": frac_true_below,
               "frac_rejected_with_TRUE_snr_above_threshold": rej_true_above,
               "note": "the last two are the Malmquist/Eddington scatter across the "
                       "threshold that only a data-space cut can produce; a true-parameter "
                       "cut would give exactly 0 for both"})

        # ------------------------------------------------------------- V2 sky width
        _log("V2: sigma_ang is a function of the observed data, not of the truth")
        sa_obs = sigma_ang_from_amplitude(
            snr_amplitude(tr["obs_m1det"], tr["obs_m2det"], tr["obs_dL"], SNR_REF_SIGMA))
        sa_true = sigma_ang_from_amplitude(
            snr_amplitude(tr["m1det"], tr["m2det"], tr["dl"], SNR_REF_SIGMA))
        sa_exact = bool(np.array_equal(sa_obs, tr["obs_sigma_ang"]))
        rel = np.abs(sa_true - tr["obs_sigma_ang"]) / tr["obs_sigma_ang"]
        differs = bool(np.median(rel) > 1e-3)
        check("V2_sigma_ang_from_observed_amplitude", sa_exact and differs,
              {"recomputed_from_observed_equals_stored_bitwise": sa_exact,
               "truth_derived_differs": differs,
               "median_rel_diff_truth_vs_stored": float(np.median(rel)),
               "max_rel_diff_truth_vs_stored": float(rel.max()),
               "frac_events_differing_gt_1pct": float((rel > 0.01).mean()),
               "sigma_ang_deg_min": float(np.rad2deg(tr["obs_sigma_ang"].min())),
               "sigma_ang_deg_max": float(np.rad2deg(tr["obs_sigma_ang"].max()))})

        # ------------------------------------------------------------ V3 PE calibration
        _log("V3: PE samples are the exact flat-prior posterior of the stored measurement")
        s = SIGMA_DL
        dL_e = pe["dL"].reshape(nobs, nsamp)
        m1_e = pe["m1det"].reshape(nobs, nsamp)
        m2_e = pe["m2det"].reshape(nobs, nsamp)
        ch_e = pe["chieff"].reshape(nobs, nsamp)
        mu_ln = np.log(tr["obs_dL"]) + s * s

        def _pit(x, mu, sig, lo=None, hi=None):
            """PIT under the exact flat-prior posterior.

            The additively measured channels are CLIPPED at hard physical bounds, which
            puts a point mass on the bound rather than truncating.  Testing the clipped
            samples against an untruncated Gaussian would be wrong, so the check is run
            on the interior samples against the CONDITIONAL (truncated) law -- which is
            the same test whenever nothing is clipped."""
            u = ndtr((x - mu) / sig)
            ulo = ndtr((lo - mu) / sig) if lo is not None else 0.0 * u
            uhi = ndtr((hi - mu) / sig) if hi is not None else 0.0 * u + 1.0
            keep = np.ones_like(x, dtype=bool)
            if lo is not None:
                keep &= x > lo
            if hi is not None:
                keep &= x < hi
            return ((u - ulo) / (uhi - ulo))[keep], float(1.0 - keep.mean())

        def _ks(u, n=200_000):
            if u.size > n:
                u = u[rng.choice(u.size, n, replace=False)]
            r = kstest(u, "uniform")
            return float(r.pvalue), float(r.statistic)

        # -------------------------------------------- V2b RA width, convention (b2)
        # The measurement width of RA and the width the PE was built with must be the
        # SAME number, and that number must be recomputable from the STORED observables.
        # Before 2026-08-01 observe() used cos(dec_TRUE) while posterior_samples() used
        # cos(dec_obs); this check is what would have caught it.
        _log("V2b: the RA width is a function of the OBSERVED dec, and the PE uses it")

        def _wrap(x):
            return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi

        sig_ra_re = tr["obs_sigma_ang"] / np.maximum(np.cos(tr["obs_dec"]), 0.1)
        sig_ra_true = tr["obs_sigma_ang"] / np.maximum(np.cos(tr["dec"]), 0.1)
        ra_exact = bool(np.array_equal(sig_ra_re, tr["obs_sig_ra"]))
        rel_ra = np.abs(sig_ra_true / tr["obs_sig_ra"] - 1.0)
        # measurement-side pull: (ra_obs - ra_true)/sig_ra ~ N(0,1) iff observe() drew
        # with the width the file records.
        ks_xi_ra = kstest(_wrap(tr["obs_ra"] - tr["ra"]) / tr["obs_sig_ra"], "norm")
        ks_xi_dec = kstest((tr["obs_dec"] - tr["dec"]) / tr["obs_sigma_ang"], "norm")
        ks_xi_ra_latent = kstest(_wrap(tr["obs_ra"] - tr["ra"]) / sig_ra_true, "norm")
        # PE side: the stored RA samples against N(ra_obs, obs_sig_ra).
        ra_e = pe["ra"].reshape(nobs, nsamp)
        dec_e = pe["dec"].reshape(nobs, nsamp)
        p_ra_pe, _ = _ks(ndtr(_wrap(ra_e - tr["obs_ra"][:, None])
                              / tr["obs_sig_ra"][:, None]).ravel())
        p_dec_pe, _ = _ks(ndtr((dec_e - tr["obs_dec"][:, None])
                               / tr["obs_sigma_ang"][:, None]).ravel())
        ratio_ra = _wrap(ra_e - tr["obs_ra"][:, None]).std(axis=1, ddof=1) / tr["obs_sig_ra"]
        ratio_dec = ((dec_e - tr["obs_dec"][:, None]).std(axis=1, ddof=1)
                     / tr["obs_sigma_ang"])
        # Pooled chi-like pull on the realised scatter.  sd of one event's ratio is
        # 1/sqrt(2 nsamp); over nobs events the mean has sd 1/sqrt(2 nsamp nobs) = 5e-4,
        # so the pre-fix 2.2 % mean RA width error would show here at ~44 sigma.
        # Events within 8 sigma_ang of a POLE are excluded: their dec posterior is
        # genuinely truncated by the |dec| <= pi/2 clip, so the realised sd is smaller
        # than sigma_ang for a physical reason, not a bookkeeping one.
        pole = (0.5 * np.pi - np.abs(tr["obs_dec"])) / tr["obs_sigma_ang"] < 8.0
        n_pole = int(pole.sum())
        n_ok = max(int((~pole).sum()), 1)
        ra_pull = float((ratio_ra[~pole].mean() - 1.0) * np.sqrt(2.0 * nsamp * n_ok))
        dec_pull = float((ratio_dec[~pole].mean() - 1.0) * np.sqrt(2.0 * nsamp * n_ok))
        dec_clip_frac = float((np.abs(dec_e) >= 0.5 * np.pi - 1e-15).mean())
        ok_ra = (ra_exact and p_ra_pe > 1e-4 and p_dec_pe > 1e-4
                 and ks_xi_ra.pvalue > 1e-4 and ks_xi_dec.pvalue > 1e-4
                 and abs(ra_pull) < 6.0 and abs(dec_pull) < 6.0)
        check("V2b_ra_width_from_observed_dec", ok_ra,
              {"pe_width_recomputed_from_stored_equals_stored_bitwise": ra_exact,
               "median_rel_width_error_if_latent_dec_were_used": float(np.median(rel_ra)),
               "mean_rel_width_error_if_latent_dec_were_used": float(rel_ra.mean()),
               "rms_rel_width_error_if_latent_dec_were_used":
                   float(np.sqrt((rel_ra ** 2).mean())),
               "max_rel_width_error_if_latent_dec_were_used": float(rel_ra.max()),
               "ks_measurement_pull_ra_pvalue": float(ks_xi_ra.pvalue),
               "ks_measurement_pull_dec_pvalue": float(ks_xi_dec.pvalue),
               "ks_measurement_pull_ra_with_LATENT_dec_pvalue":
                   float(ks_xi_ra_latent.pvalue),
               "ks_pe_ra_pooled_pvalue": p_ra_pe, "ks_pe_dec_pooled_pvalue": p_dec_pe,
               "pe_ra_width_ratio_mean": float(ratio_ra[~pole].mean()),
               "pe_ra_width_ratio_pull_sigma": ra_pull,
               "pe_dec_width_ratio_mean": float(ratio_dec[~pole].mean()),
               "pe_dec_width_ratio_pull_sigma": dec_pull,
               "n_events_within_8_sigma_of_a_pole": n_pole,
               "pe_dec_clip_fraction_at_pi_over_2": dec_clip_frac,
               "note": "the RA width is a deterministic function of the RECORDED dec "
                       "(convention b2); the 'if_latent_dec_were_used' entries are the "
                       "size of the defect this convention removes"})

        u_dL = ndtr((np.log(dL_e) - mu_ln[:, None]) / s).ravel()
        # Mass channels, convention (c2): the PIT is taken under the EXACT flat-prior
        # posterior of obs ~ N(m, f m).  The table used for the CHECK is deliberately
        # finer than the one that drew the samples (grid x2, cap x10, n_sig 12 -> 16), so
        # this is not a self-consistency test of a single grid.
        CHK = dict(n_grid=(1 << 22) + 1, y_cap=PEX_Y_CAP / 10.0, n_sig=16.0)

        def _pit_mass(x, obs, f, lo):
            u = exact_mass_posterior_cdf(x, obs, f, **CHK)
            ulo = exact_mass_posterior_cdf(np.full_like(np.asarray(obs, float), lo),
                                           obs, f, **CHK)
            keep = x > lo
            return ((u - ulo) / (1.0 - ulo))[keep], float(1.0 - keep.mean())

        u_m1, clip_m1 = _pit_mass(m1_e, tr["obs_m1det"][:, None], SIG_M1_FRAC, 2.0)
        u_m2, clip_m2 = _pit_mass(m2_e, tr["obs_m2det"][:, None], SIG_M2_FRAC, 1.0)
        u_ch, clip_ch = _pit(ch_e, tr["obs_chieff"][:, None], SIGMA_CHIEFF, lo=-1.0, hi=1.0)
        p_dL, d_dL = _ks(u_dL)
        p_m1, _ = _ks(u_m1.ravel())
        p_m2, _ = _ks(u_m2.ravel())
        p_ch, _ = _ks(u_ch.ravel())

        # PER-EVENT KS on >= 200 events: each event's own 2000 dL samples against its own
        # analytic posterior, then a meta-test that the p-values are uniform.
        n_ev_ks = min(nobs, 1000)
        ev_ids = rng.choice(nobs, n_ev_ks, replace=False)
        pv = np.array([kstest(ndtr((np.log(dL_e[i]) - mu_ln[i]) / s), "uniform").pvalue
                       for i in ev_ids])
        ks_meta = kstest(pv, "uniform")
        # the same per-event test on both mass channels, under the exact posterior
        pv_m1 = np.array([kstest(exact_mass_posterior_cdf(
            m1_e[i], tr["obs_m1det"][i], SIG_M1_FRAC, **CHK), "uniform").pvalue
            for i in ev_ids])
        pv_m2 = np.array([kstest(exact_mass_posterior_cdf(
            m2_e[i], tr["obs_m2det"][i], SIG_M2_FRAC, **CHK), "uniform").pvalue
            for i in ev_ids])
        ks_meta_m1 = kstest(pv_m1, "uniform")
        ks_meta_m2 = kstest(pv_m2, "uniform")

        # Independent numerical re-derivation of the exact MASS posterior, the direct
        # analogue of the distance check below: build p(m|obs) ~ (1/(f m)) exp[-(obs-m)^2
        # / (2 f^2 m^2)] on a dense grid IN m (no change of variables), integrate, and
        # compare its CDF with the y-space table the sampler inverts.  This tests the
        # change of variables and the quadrature together.
        mass_cdf_err = 0.0
        for f_frac, o_col, lo_m in ((SIG_M1_FRAC, tr["obs_m1det"], 2.0),
                                    (SIG_M2_FRAC, tr["obs_m2det"], 1.0)):
            for i in rng.choice(nobs, 12, replace=False):
                ob = float(o_col[i])
                g = np.geomspace(ob / (1.0 + PEX_N_SIG * f_frac), ob / PEX_Y_CAP, 400_001)
                lp = -np.log(f_frac * g) - 0.5 * ((ob - g) / (f_frac * g)) ** 2
                p = np.exp(lp - lp.max())
                c = np.concatenate([[0.0],
                                    np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(g))])
                c /= c[-1]
                mass_cdf_err = max(mass_cdf_err, float(np.abs(
                    c - exact_mass_posterior_cdf(g, ob, f_frac)).max()))

        # Convergence of the production quantile table itself: refine the grid, widen the
        # cap and the sigma range, and compare the quantile functions over the u range
        # nobs*nsamp draws can reach.
        u_conv = np.linspace(1.0 / (nobs * nsamp * 10.0), 1.0 - 1.0 / (nobs * nsamp * 10.0),
                             200_001)
        pex_conv = {}
        for f_frac, nm in ((SIG_M1_FRAC, "m1"), (SIG_M2_FRAC, "m2")):
            c0, y0 = exact_mass_posterior_table(f_frac)
            q0 = np.interp(u_conv, c0, y0)
            for lab, kw in (("grid_x2", dict(n_grid=2 * PEX_N_GRID - 1)),
                            ("cap_x10", dict(y_cap=PEX_Y_CAP / 10.0)),
                            ("n_sig_16", dict(n_sig=16.0))):
                c1, y1 = exact_mass_posterior_table(f_frac, **kw)
                pex_conv[f"{nm}_{lab}_max_rel_dQ"] = float(
                    np.abs(np.interp(u_conv, c1, y1) / q0 - 1.0).max())
        # the direct signature of (c2): E_post[m]/obs = 1 + 2 f^2 + O(f^4)
        pex_mean_shift = {
            "m1_measured": float((m1_e / tr["obs_m1det"][:, None]).mean()),
            "m1_predicted_1p2f2": 1.0 + 2.0 * SIG_M1_FRAC ** 2,
            "m2_measured": float((m2_e / tr["obs_m2det"][:, None]).mean()),
            "m2_predicted_1p2f2": 1.0 + 2.0 * SIG_M2_FRAC ** 2}

        # Independent numerical re-derivation of the flat-prior distance posterior.
        # This tests the CLOSED FORM itself, not just the sampler: build
        # p(dL | d_obs) ~ exp(-(ln d_obs - ln dL)^2 / 2 s^2) on a grid, integrate, and
        # compare the numeric CDF with the lognormal closed form the sampler uses.  This
        # is the role ``code/generate_gwsamples.py``'s inverse-CDF machinery plays for
        # the additive-noise variant; here it is a CHECK, because the multiplicative
        # posterior has an exact closed form.
        max_cdf_err = 0.0
        for i in rng.choice(nobs, 25, replace=False):
            d_obs = float(tr["obs_dL"][i])
            grid = np.geomspace(d_obs * np.exp(-12 * s), d_obs * np.exp(12 * s), 400_001)
            logp = -0.5 * ((np.log(d_obs) - np.log(grid)) / s) ** 2
            p = np.exp(logp - logp.max())
            cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(grid))])
            cdf /= cdf[-1]
            closed = ndtr((np.log(grid) - (np.log(d_obs) + s * s)) / s)
            max_cdf_err = max(max_cdf_err, float(np.abs(cdf - closed).max()))

        ok_pe = (p_dL > 1e-4 and p_m1 > 1e-4 and p_m2 > 1e-4 and p_ch > 1e-4
                 and ks_meta.pvalue > 1e-4 and ks_meta_m1.pvalue > 1e-4
                 and ks_meta_m2.pvalue > 1e-4 and max_cdf_err < 1e-4
                 and mass_cdf_err < 1e-4
                 and max(pex_conv.values()) < 1e-6)
        check("V3_pe_calibration", ok_pe,
              {"n_events": nobs, "nsamp": nsamp, "n_events_per_event_ks": int(n_ev_ks),
               "ks_dL_pooled_pvalue": p_dL, "ks_dL_pooled_stat": d_dL,
               "ks_m1det_pooled_pvalue": p_m1, "ks_m2det_pooled_pvalue": p_m2,
               "ks_chieff_pooled_pvalue": p_ch,
               "per_event_ks_pvalue_uniformity_pvalue": float(ks_meta.pvalue),
               "per_event_ks_pvalue_min": float(pv.min()),
               "per_event_ks_frac_below_0.01": float((pv < 0.01).mean()),
               "per_event_ks_m1det_uniformity_pvalue": float(ks_meta_m1.pvalue),
               "per_event_ks_m2det_uniformity_pvalue": float(ks_meta_m2.pvalue),
               "per_event_ks_m1det_frac_below_0.01": float((pv_m1 < 0.01).mean()),
               "per_event_ks_m2det_frac_below_0.01": float((pv_m2 < 0.01).mean()),
               "closed_form_vs_numeric_cdf_max_abs_err": max_cdf_err,
               "exact_mass_posterior_numeric_cdf_max_abs_err": mass_cdf_err,
               "exact_mass_posterior_table_convergence": pex_conv,
               "exact_mass_posterior_mean_shift": pex_mean_shift,
               "clip_fraction_m1det": clip_m1, "clip_fraction_m2det": clip_m2,
               "clip_fraction_chieff": clip_ch,
               "note": "u_dL is the PIT under ln dL ~ N(ln d_obs + s^2, s); the MASS "
                       "channels use the exact flat-prior posterior of obs ~ N(m, f m) "
                       "(convention c2), evaluated on a table finer than the sampler's; "
                       "chieff uses the clip-conditional Gaussian PIT"})

        # ------------------------------------- V3b the stored events are a fair draw
        _log("V3b: independent replication of the generative draw")
        cosmo_v = gmd._build_cosmology(H0_FID, OM0_FID, W0_FID, WA_FID)
        grids_v = gmd._cosmology_grids(cosmo_v, ZMAX_GRID)
        pop_v = gmd.PopulationConfig(gamma=GAMMA)
        catv = {t: load_catalog(sd / "catalogs" / f"catalog_{t}_complete.h5",
                                keys=("ra", "dec", "z")) for t in TRACERS}
        rng_v = np.random.default_rng(sub_seeds(args.seed)["validation"] + 7)
        n_rep = 3_000_000
        ng, na = catv["gal"]["z"].size, catv["agn"]["z"].size
        u_t = rng_v.uniform(size=n_rep)
        ig, ia = rng_v.integers(0, ng, n_rep), rng_v.integers(0, na, n_rep)
        isa = u_t < F_AGN
        zv = np.where(isa, catv["agn"]["z"][ia], catv["gal"]["z"][ig])
        rav = np.where(isa, catv["agn"]["ra"][ia], catv["gal"]["ra"][ig])
        decv = np.where(isa, catv["agn"]["dec"][ia], catv["gal"]["dec"][ig])
        dlv = gmd._interp_dl(zv, grids_v)
        m1v, upv = gmd._sample_powerlaw_peak_m1(rng_v, n_rep, pop_v, return_component=True)
        qv = gmd._sample_q(rng_v, m1v, pop_v, use_peak=upv)
        chiv = gmd._sample_chieff(rng_v, n_rep, pop_v)
        obsv = observe(rng_v, m1v * (1 + zv), qv * m1v * (1 + zv), chiv, dlv, rav, decv)
        detv, _ = detect_from_observation(obsv)
        accv = rng_v.uniform(size=n_rep) < (1.0 + zv) ** (GAMMA - 1.0)
        kept = detv & accv
        eps_all = np.log(obsv["dL"] / dlv) / SIGMA_DL
        ks_eps_all = kstest(eps_all[rng_v.choice(n_rep, 200_000, replace=False)], "norm")
        eps_stored = np.log(tr["obs_dL"] / tr["dl"]) / SIGMA_DL
        ks_eps_2s = kstest(eps_stored, eps_all[kept])
        ks_z_2s = kstest(tr["z"], zv[kept])
        f_rep = float(isa[kept].mean())
        ok_rep = (ks_eps_all.pvalue > 1e-4 and ks_eps_2s.pvalue > 1e-3
                  and ks_z_2s.pvalue > 1e-3)
        check("V3b_generative_replication", ok_rep,
              {"n_replicated_proposals": n_rep, "n_replicated_detected": int(kept.sum()),
               "ks_eps_all_proposals_vs_N01_pvalue": float(ks_eps_all.pvalue),
               "ks_eps_detected_stored_vs_replica_pvalue": float(ks_eps_2s.pvalue),
               "ks_z_detected_stored_vs_replica_pvalue": float(ks_z_2s.pvalue),
               "mean_eps_all_proposals": float(eps_all.mean()),
               "mean_eps_detected_stored": float(eps_stored.mean()),
               "mean_eps_detected_replica": float(eps_all[kept].mean()),
               "replica_f_agn": f_rep,
               "replica_detected_fraction": float(kept.mean()),
               "note": "mean_eps over DETECTED events is significantly negative while it "
                       "is zero over all proposals -- the Malmquist/Eddington tilt that "
                       "only a data-space detection cut produces"})

        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        ax[0].hist(u_dL, bins=50, range=(0, 1), density=True, color="0.4")
        ax[0].axhline(1.0, color="crimson", lw=1.2)
        ax[0].set_title(r"$d_L$ PIT (pooled)"); ax[0].set_xlabel("PIT")
        ax[0].set_ylabel("density")
        ax[1].hist(pv, bins=25, range=(0, 1), density=True, color="0.4")
        ax[1].axhline(1.0, color="crimson", lw=1.2)
        ax[1].set_title("per-event KS $p$-values"); ax[1].set_xlabel("$p$")
        ax[2].hist(eps_all[rng_v.choice(n_rep, 200_000, replace=False)], bins=60,
                   density=True, histtype="step", label="all proposals", lw=1.4)
        ax[2].hist(eps_stored, bins=40, density=True, histtype="step",
                   label="stored detected", lw=1.4)
        ax[2].hist(eps_all[kept], bins=60, density=True, histtype="step",
                   label="replica detected", lw=1.2, ls="--")
        ax[2].set_xlabel(r"$\varepsilon = \ln(d_{\rm obs}/d_L)/s$")
        ax[2].legend(fontsize=7); ax[2].set_title("measurement noise + Malmquist tilt")
        fig.suptitle("PE calibration and generative replication")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(vdir / f"fig_pe_calibration.{ext}", dpi=140)
        plt.close(fig)

    # ==========================================================================
    #  v3 MEASUREMENT-FAMILY CHECKS  (DESIGN_PE.md)
    # ==========================================================================
    if pe_model == "v3":
        from scipy.special import ndtri

        def _phi(x):
            return ndtr(np.asarray(x, float))

        def _tn_pit(x, mu, sig, lo=-np.inf, hi=np.inf):
            """PIT under N(mu, sig) truncated to [lo, hi] -- the EXACT flat-prior
            posterior of every v3 channel (the truncation is a PRIOR truncation,
            so the likelihood carries no theta-dependent normalisation)."""
            a = _phi((lo - mu) / sig)
            b = _phi((hi - mu) / sig)
            return np.clip((_phi((x - mu) / sig) - a) / np.maximum(b - a, 1e-300),
                           0.0, 1.0)

        def _wrap(d):
            return (np.asarray(d) + np.pi) % (2.0 * np.pi) - np.pi

        rho_obs = tr["obs_rho"]
        w_re = v3_widths(rho_obs)
        mc_true = mc_of_m1q(tr["m1det"], tr["q"])
        rho_true_re = rho_opt_of_mc_dl(mc_true, tr["dl"])

        # ------------------------------------------------------------ V1 detection
        _log("V1: detection is a deterministic function of ONE recorded number")
        alias_ok = bool(np.array_equal(rho_obs, tr["snr_obs"]))
        all_pass = bool(np.all(rho_obs >= SNR_THRESHOLD))
        rho_true_ok = float(np.max(np.abs(rho_true_re / tr["snr_true"] - 1.0)))
        rp = sd / "events" / "events_rejected_sample.h5"
        rej_ok, n_rej = True, 0
        if rp.exists():
            with h5py.File(rp, "r") as f:
                rj = {k: np.asarray(f[k][:]) for k in f.keys()}
            rej_ok = bool(np.all(rj["rho_obs"] < SNR_THRESHOLD))
            n_rej = int(rj["rho_obs"].size)
        # the detection rule's own generative test: on the DETECTED set the pull
        # (rho_obs - rho_true)/sigma_rho is a normal TRUNCATED at (8 - rho_true),
        # so its PIT must be uniform.  This is the sharpest possible statement that
        # rho_obs ~ N(rho_opt, sigma_rho) and that the cut is on rho_obs.
        u_rho_det = _tn_pit(rho_obs, rho_true_re, SIGMA_RHO,
                            lo=SNR_THRESHOLD, hi=np.inf)
        ks_rho_det = kstest(u_rho_det, "uniform")
        frac_true_below = float((tr["snr_true"] < SNR_THRESHOLD).mean())
        rej_true_above = (float((rj["rho_true"] >= SNR_THRESHOLD).mean())
                          if rp.exists() else None)
        check("V1_detection_deterministic_in_data",
              alias_ok and all_pass and rej_ok and rho_true_ok < 1e-12
              and ks_rho_det.pvalue > 1e-4,
              {"obs_rho_equals_snr_obs_bitwise": alias_ok,
               "all_detected_rho_obs_ge_threshold": all_pass,
               "n_rejected_checked": n_rej,
               "all_rejected_rho_obs_lt_threshold": rej_ok,
               "rho_true_recompute_max_rel_err": rho_true_ok,
               "ks_truncated_rho_pull_pvalue": float(ks_rho_det.pvalue),
               "rho_obs_min": float(rho_obs.min()),
               "rho_obs_median": float(np.median(rho_obs)),
               "rho_obs_max": float(rho_obs.max()),
               "frac_detected_with_TRUE_snr_below_threshold": frac_true_below,
               "frac_rejected_with_TRUE_snr_above_threshold": rej_true_above,
               "note": "the last two are the Malmquist/Eddington scatter across the "
                       "threshold that only a data-space cut can produce"})

        # -------------------------------------------------------- V2 widths = f(data)
        _log("V2: EVERY measurement width is recomputable from the stored rho_obs")
        wid = {"obs_sig_lnmc": w_re["sig_lnmc"], "obs_sig_lnq": w_re["sig_lnq"],
               "obs_sig_chieff": w_re["sig_chieff"],
               "obs_sigma_ang": w_re["sigma_ang"]}
        w_exact = {k: bool(np.array_equal(v, tr[k])) for k, v in wid.items()}
        sig_rho_ok = bool(np.array_equal(tr["obs_sigma_rho"],
                                         np.full(nobs, SIGMA_RHO)))
        check("V2_widths_from_observed_snr", all(w_exact.values()) and sig_rho_ok,
              {"recomputed_equals_stored_bitwise": w_exact,
               "sigma_rho_constant": sig_rho_ok,
               "sigma_ang_deg_min": float(np.rad2deg(tr["obs_sigma_ang"].min())),
               "sigma_ang_deg_max": float(np.rad2deg(tr["obs_sigma_ang"].max())),
               "sig_lnmc_min": float(tr["obs_sig_lnmc"].min()),
               "sig_lnmc_max": float(tr["obs_sig_lnmc"].max()),
               "sig_lnq_min": float(tr["obs_sig_lnq"].min()),
               "sig_lnq_max": float(tr["obs_sig_lnq"].max()),
               "note": "in v3 EVERY channel's width -- not only the sky -- is a "
                       "function of one recorded number, which is what makes the "
                       "generative likelihood exactly invertible"})

        # --------------------------------------- V2b RA width, convention (b2)
        _log("V2b: the RA width is a function of the OBSERVED dec, and the PE uses it")
        sig_ra_re = tr["obs_sigma_ang"] / np.maximum(np.cos(tr["obs_dec"]),
                                                     COS_DEC_FLOOR)
        ra_exact = bool(np.array_equal(sig_ra_re, tr["obs_sig_ra"]))
        sig_ra_true = tr["obs_sigma_ang"] / np.maximum(np.cos(tr["dec"]),
                                                       COS_DEC_FLOOR)
        rel_ra = np.abs(sig_ra_true / tr["obs_sig_ra"] - 1.0)
        xi_ra = _wrap(tr["obs_ra"] - tr["ra"]) / tr["obs_sig_ra"]
        xi_dec = (tr["obs_dec"] - tr["dec"]) / tr["obs_sigma_ang"]
        ks_xi_ra = kstest(xi_ra, "norm")
        ks_xi_dec = kstest(xi_dec, "norm")
        check("V2b_ra_width_from_observed_dec",
              ra_exact and ks_xi_ra.pvalue > 1e-4 and ks_xi_dec.pvalue > 1e-4,
              {"recomputed_equals_stored_bitwise": ra_exact,
               "ks_ra_measurement_pull_pvalue": float(ks_xi_ra.pvalue),
               "ks_dec_measurement_pull_pvalue": float(ks_xi_dec.pvalue),
               "median_rel_diff_truthdec_vs_stored": float(np.median(rel_ra)),
               "max_rel_diff_truthdec_vs_stored": float(rel_ra.max())})

        # ------------------------------------------------ V3 PE calibration, per channel
        _log("V3: PE samples are the exact flat-prior posterior, channel by channel")
        m1_e = pe["m1det"].reshape(nobs, nsamp)
        m2_e = pe["m2det"].reshape(nobs, nsamp)
        dL_e = pe["dL"].reshape(nobs, nsamp)
        ch_e = pe["chieff"].reshape(nobs, nsamp)
        ra_e = pe["ra"].reshape(nobs, nsamp)
        dec_e = pe["dec"].reshape(nobs, nsamp)
        q_e = pe["q"].reshape(nobs, nsamp)
        mc_e = pe["mc_det"].reshape(nobs, nsamp)
        rho_e = pe["rho"].reshape(nobs, nsamp)
        # the stored measurement-basis columns must be EXACTLY the bijection of the
        # stored storage-basis columns (this is what makes the PIT a test of the PE
        # and not of the bookkeeping)
        bij = {
            "q_vs_m2/m1": float(np.max(np.abs(q_e / (m2_e / m1_e) - 1.0))),
            "mc_vs_m1q": float(np.max(np.abs(mc_e / mc_of_m1q(m1_e, q_e) - 1.0))),
            "rho_vs_mc_dL": float(np.max(np.abs(
                rho_e / rho_opt_of_mc_dl(mc_e, dL_e) - 1.0))),
            "m1_vs_mc_q": float(np.max(np.abs(m1_e / m1_of_mc_q(mc_e, q_e) - 1.0))),
            "dL_vs_mc_rho": float(np.max(np.abs(
                dL_e / dl_of_mc_rho(mc_e, rho_e) - 1.0)))}
        bij_ok = max(bij.values()) < 1e-11
        q_le_1 = float((q_e > 1.0).mean())

        B = lambda a: np.broadcast_to(np.asarray(a)[:, None], (nobs, nsamp))
        u_mc = _tn_pit(np.log(mc_e), B(tr["obs_lnmc"]), B(tr["obs_sig_lnmc"]))
        u_q = _tn_pit(np.log(q_e), B(tr["obs_lnq"]), B(tr["obs_sig_lnq"]),
                      hi=0.0)
        u_rho = _tn_pit(rho_e, B(tr["obs_rho"]), B(tr["obs_sigma_rho"]), lo=0.0)
        u_ch = _tn_pit(ch_e, B(tr["obs_chieff"]), B(tr["obs_sig_chieff"]),
                       lo=CHIEFF_RANGE[0], hi=CHIEFF_RANGE[1])
        u_dec = _tn_pit(dec_e, B(tr["obs_dec"]), B(tr["obs_sigma_ang"]),
                        lo=-0.5 * np.pi, hi=0.5 * np.pi)
        xi_ra_pe = _wrap(ra_e - B(tr["obs_ra"])) / B(tr["obs_sig_ra"])
        n_sub = min(200_000, nobs * nsamp)
        sub = rng.choice(nobs * nsamp, n_sub, replace=False)
        pooled = {}
        for nm, u in (("mc", u_mc), ("q", u_q), ("rho", u_rho), ("chieff", u_ch),
                      ("dec", u_dec)):
            pooled[nm] = float(kstest(u.ravel()[sub], "uniform").pvalue)
        pooled["ra_pull"] = float(kstest(xi_ra_pe.ravel()[sub], "norm").pvalue)
        # per-event KS, then a meta-KS that the p-values are uniform
        n_ev_ks = min(nobs, 1000)
        ev_idx = rng.choice(nobs, n_ev_ks, replace=False)
        meta = {}
        for nm, u in (("mc", u_mc), ("q", u_q), ("rho", u_rho), ("dec", u_dec)):
            pv = np.array([kstest(u[i], "uniform").pvalue for i in ev_idx])
            meta[nm] = float(kstest(pv, "uniform").pvalue)
            if nm == "mc":
                pv_mc = pv
        # measurement-side pulls: every channel's observation about the TRUTH
        pulls = {
            "lnmc": (tr["obs_lnmc"] - np.log(mc_true)) / tr["obs_sig_lnmc"],
            "lnq": (tr["obs_lnq"] - np.log(tr["q"])) / tr["obs_sig_lnq"],
            "chieff": (tr["obs_chieff"] - tr["chieff"]) / tr["obs_sig_chieff"]}
        ks_pulls = {k: float(kstest(v, "norm").pvalue) for k, v in pulls.items()}
        ok_v3 = (bij_ok and q_le_1 == 0.0
                 and min(pooled.values()) > 1e-4 and min(meta.values()) > 1e-4
                 and min(ks_pulls.values()) > 1e-4)
        check("V3_pe_calibration", ok_v3,
              {"bijection_max_rel_err": bij, "bijection_ok": bij_ok,
               "frac_pe_samples_with_q_gt_1": q_le_1,
               "pooled_ks_pvalue": pooled,
               "per_event_ks_uniformity_pvalue": meta,
               "measurement_pull_ks_pvalue": ks_pulls,
               "measurement_pull_mean": {k: float(v.mean())
                                         for k, v in pulls.items()},
               "measurement_pull_sd": {k: float(v.std(ddof=1))
                                       for k, v in pulls.items()},
               "sigma_lndL_realised_median": float(np.median(
                   np.std(np.log(dL_e), axis=1))),
               "note": "every channel is a (prior-)truncated normal about the "
                       "OBSERVED value with the STORED width -- the exact "
                       "flat-prior posterior of an unbounded-Gaussian likelihood"})

        # ------------------------------------------- V3c the p_pe Jacobian is exact
        _log("V3c: p_pe is the PE prior in the canonical basis (closed form == "
             "numerical Jacobian)")
        raw = p_pe_v3(m1_e, q_e, dL_e, mc_det=mc_e)
        p_closed = raw / raw.mean(axis=1, keepdims=True)
        p_stored = pe["p_pe"].reshape(nobs, nsamp)
        # NOTE: darksirens renormalises p_pe per event, so only the shape matters;
        # the file stores it mean-1 per event, which is the same normalisation.
        rel_ppe = float(np.max(np.abs(p_closed / p_stored - 1.0)))
        # an INDEPENDENT numerical Jacobian of y = (ln Mc, ln q, rho) wrt
        # x = (m1det, q, dL) by central differences, on a random subset
        ns_j = 20_000
        js = rng.choice(nobs * nsamp, ns_j, replace=False)
        x0 = np.stack([pe["m1det"][js], pe["q"][js], pe["dL"][js]], axis=1)

        def _y(x):
            m1, q, dl = x[:, 0], x[:, 1], x[:, 2]
            mc = mc_of_m1q(m1, q)
            return np.stack([np.log(mc), np.log(q),
                             rho_opt_of_mc_dl(mc, dl)], axis=1)

        J = np.empty((ns_j, 3, 3))
        for c in range(3):
            h = 1e-6 * np.abs(x0[:, c]) + 1e-12
            xp = x0.copy(); xp[:, c] += h
            xm = x0.copy(); xm[:, c] -= h
            J[:, :, c] = (_y(xp) - _y(xm)) / (2.0 * h)[:, None]
        det_num = np.abs(np.linalg.det(J))
        det_cl = p_pe_v3(x0[:, 0], x0[:, 1], x0[:, 2])
        rel_jac = float(np.max(np.abs(det_num / det_cl - 1.0)))
        check("V3c_p_pe_jacobian", rel_ppe < 1e-10 and rel_jac < 1e-5,
              {"stored_vs_closed_form_max_rel": rel_ppe,
               "closed_form_vs_numerical_jacobian_max_rel": rel_jac,
               "n_numerical_jacobian_samples": ns_j,
               "formula": "p_pe ~ |d(ln Mc, ln q, rho)/d(m1det, q, dL)| "
                          "= rho/(dL m1det q)",
               "note": "DESIGN_PE.md 2.5; the v2 convention p_pe ~ m1det is the "
                       "same rule for a prior flat in (m1det, m2det, dL, chieff)"})

        # ------------------------------------- V3b the stored events are a fair draw
        _log("V3b: independent replication of the generative draw")
        cosmo_v = gmd._build_cosmology(H0_FID, OM0_FID, W0_FID, WA_FID)
        grids_v = gmd._cosmology_grids(cosmo_v, ZMAX_GRID)
        pop_v = gmd.PopulationConfig(gamma=GAMMA)
        catv = {t: load_catalog(sd / "catalogs" / f"catalog_{t}_complete.h5",
                                keys=("ra", "dec", "z")) for t in TRACERS}
        rng_v = np.random.default_rng(sub_seeds(args.seed)["validation"] + 7)
        n_rep = 3_000_000
        ng, na = catv["gal"]["z"].size, catv["agn"]["z"].size
        u_t = rng_v.uniform(size=n_rep)
        ig, ia = rng_v.integers(0, ng, n_rep), rng_v.integers(0, na, n_rep)
        isa = u_t < F_AGN
        zv = np.where(isa, catv["agn"]["z"][ia], catv["gal"]["z"][ig])
        rav = np.where(isa, catv["agn"]["ra"][ia], catv["gal"]["ra"][ig])
        decv = np.where(isa, catv["agn"]["dec"][ia], catv["gal"]["dec"][ig])
        dlv = gmd._interp_dl(zv, grids_v)
        m1v, upv = gmd._sample_powerlaw_peak_m1(rng_v, n_rep, pop_v,
                                                return_component=True)
        qv = gmd._sample_q(rng_v, m1v, pop_v, use_peak=upv)
        chiv = gmd._sample_chieff(rng_v, n_rep, pop_v)
        obsv = observe_v3(rng_v, m1v * (1 + zv), qv * m1v * (1 + zv), chiv, dlv,
                          rav, decv)
        detv, _ = detect_v3(obsv)
        accv = rng_v.uniform(size=n_rep) < (1.0 + zv) ** (GAMMA - 1.0)
        kept = detv & accv
        eps_all = (obsv["rho"] - obsv["rho_true"]) / SIGMA_RHO
        ks_eps_all = kstest(eps_all[rng_v.choice(n_rep, 200_000, replace=False)],
                            "norm")
        eps_stored = (tr["obs_rho"] - tr["snr_true"]) / SIGMA_RHO
        ks_eps_2s = kstest(eps_stored, eps_all[kept])
        ks_z_2s = kstest(tr["z"], zv[kept])
        # the closed-form P_det the exact selection oracle uses, against this replay
        pdet_cf = float(np.mean(_phi((obsv["rho_true"] - SNR_THRESHOLD) / SIGMA_RHO)))
        pdet_mc = float(detv.mean())
        pdet_pull = ((pdet_mc - pdet_cf)
                     / np.sqrt(max(pdet_cf * (1 - pdet_cf) / n_rep, 1e-300)))
        ok_rep = (ks_eps_all.pvalue > 1e-4 and ks_eps_2s.pvalue > 1e-3
                  and ks_z_2s.pvalue > 1e-3 and abs(pdet_pull) < 5.0)
        check("V3b_generative_replication", ok_rep,
              {"n_replicated_proposals": n_rep,
               "n_replicated_detected": int(kept.sum()),
               "ks_rho_noise_all_proposals_vs_N01_pvalue": float(ks_eps_all.pvalue),
               "ks_rho_pull_detected_stored_vs_replica_pvalue": float(ks_eps_2s.pvalue),
               "ks_z_detected_stored_vs_replica_pvalue": float(ks_z_2s.pvalue),
               "mean_rho_pull_all_proposals": float(eps_all.mean()),
               "mean_rho_pull_detected_stored": float(eps_stored.mean()),
               "mean_rho_pull_detected_replica": float(eps_all[kept].mean()),
               "P_det_closed_form_Phi": pdet_cf,
               "P_det_brute_force": pdet_mc,
               "P_det_binomial_pull": float(pdet_pull),
               "replica_f_agn": float(isa[kept].mean()),
               "replica_detected_fraction": float(kept.mean()),
               "note": "P_det(theta) = Phi((rho_opt - 8)/sigma_rho) is the closed "
                       "form the exact selection oracle uses; it is validated here "
                       "against the generator's own observe_v3/detect_v3"})

        # ------------------------------------------- V9 the photo-z is REALISED (D3)
        _log("V9: the survey's declared photo-z error is realised in the catalog")
        d3 = {}
        d3_ok = True
        for t in TRACERS:
            with h5py.File(sd / "catalogs" / f"catalog_{t}_complete.h5", "r") as f:
                has_zobs = "z_obs" in f
                zt = np.asarray(f["z"][:], float) if has_zobs else None
                zo = np.asarray(f["z_obs"][:], float) if has_zobs else None
            if not has_zobs:
                d3[t] = {"z_obs_column_present": False}
                d3_ok = False
                continue
            pull = (zo - zt) / (DZ_SCALE * (1.0 + zt))
            spath = sd / "surveys" / f"survey_{t}_complete_ns{NSIDE_SURVEY}.h5"
            with h5py.File(spath, "r") as f:
                zg = np.asarray(f["zgals"][:])
                dzg = np.asarray(f["dzgals"][:])
                ngl = np.asarray(f["ngals"][:])
                zcol = str(f.attrs.get("z_column", "z"))
            msk = np.arange(zg.shape[1])[None, :] < ngl[:, None]
            zsv = zg[msk]
            dzsv = dzg[msk]
            blk_vs_zobs = float(np.max(np.abs(np.sort(zsv) - np.sort(zo))))
            blk_vs_ztrue = float(np.max(np.abs(np.sort(zsv) - np.sort(zt))))
            dz_ok = float(np.max(np.abs(dzsv - DZ_SCALE * (1.0 + zsv))))
            sub_p = pull[rng.choice(pull.size, min(200_000, pull.size),
                                    replace=False)]
            ks_p = float(kstest(sub_p, "norm").pvalue)
            ok_t = (zcol == "z_obs" and blk_vs_zobs == 0.0 and blk_vs_ztrue > 0.0
                    and dz_ok == 0.0 and ks_p > 1e-4)
            d3_ok = d3_ok and ok_t
            d3[t] = {"z_obs_column_present": True, "survey_z_column": zcol,
                     "block_vs_catalog_z_obs_maxabs": blk_vs_zobs,
                     "block_vs_catalog_z_TRUE_maxabs": blk_vs_ztrue,
                     "declared_dz_vs_DZ_SCALE_1pz_maxabs": dz_ok,
                     "photoz_pull_mean": float(pull.mean()),
                     "photoz_pull_sd": float(pull.std(ddof=1)),
                     "photoz_pull_ks_pvalue": ks_p,
                     "n_negative_z_obs": int((zo < 0.0).sum()),
                     "z_obs_min": float(zo.min()), "z_true_min": float(zt.min()),
                     "pass": bool(ok_t)}
            del zt, zo, pull, zsv, dzsv, zg, dzg
        check("V9_photoz_realised", d3_ok,
              {"per_tracer": d3, "dz_scale": DZ_SCALE,
               "note": "the survey block must pixelate z_obs (a REAL error of the "
                       "declared width) and must NOT be bit-identical to the true "
                       "catalog redshift -- CLOSURE.md 15.4, DESIGN_PE.md 3.3"})

        fig, ax = plt.subplots(1, 4, figsize=(16, 3.4))
        for nm, u in (("ln Mc", u_mc), ("ln q", u_q), (r"$\rho$", u_rho),
                      ("dec", u_dec)):
            ax[0].hist(u.ravel()[sub], bins=50, range=(0, 1), density=True,
                       histtype="step", lw=1.3, label=nm)
        ax[0].axhline(1.0, color="crimson", lw=1.0)
        ax[0].set_xlabel("PIT"); ax[0].set_ylabel("density")
        ax[0].set_title("PE PIT, per channel"); ax[0].legend(fontsize=7)
        ax[1].hist(pv_mc, bins=25, range=(0, 1), density=True, color="0.4")
        ax[1].axhline(1.0, color="crimson", lw=1.0)
        ax[1].set_xlabel("$p$"); ax[1].set_title(r"per-event KS $p$, $\ln M_c$")
        ax[2].hist(eps_all[rng_v.choice(n_rep, 200_000, replace=False)], bins=60,
                   density=True, histtype="step", lw=1.4, label="all proposals")
        ax[2].hist(eps_stored, bins=40, density=True, histtype="step", lw=1.4,
                   label="stored detected")
        ax[2].hist(eps_all[kept], bins=60, density=True, histtype="step", lw=1.2,
                   ls="--", label="replica detected")
        ax[2].set_xlabel(r"$(\rho_{\rm obs}-\rho_{\rm opt})/\sigma_\rho$")
        ax[2].legend(fontsize=7); ax[2].set_title("SNR noise + Malmquist tilt")
        ax[3].hist(np.std(np.log(dL_e), axis=1), bins=40, color="0.4")
        ax[3].set_xlabel(r"realised $\sigma_{\ln d_L}$ per event")
        ax[3].set_title(r"$\sigma_{\ln d_L}\simeq 1.13/\rho$")
        fig.suptitle("v3 PE calibration and generative replication")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(vdir / f"fig_pe_calibration.{ext}", dpi=140)
        plt.close(fig)


    # ------------------------------------------------------- V4 catalog cross-checks
    # v2: the archived reference is a DIFFERENT model (v1 densities, z_max 1.565,
    # lmax 64), so comparing against it would be meaningless.  The check is now
    # internal and stronger: does the catalog realise the TARGET comoving number
    # density and the planted bias contrast?
    _log("V4: catalog realises the target comoving densities and the planted bias")
    cats = {t: load_catalog(sd / "catalogs" / f"catalog_{t}_complete.h5",
                            keys=("ra", "dec", "z")) for t in TRACERS}

    import healpy as hp
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u

    acos = FlatLambdaCDM(H0=H0_FID, Om0=OM0_FID)
    cat_meta = _read_json(sd / "catalogs" / "glass_field_meta.json")
    z_edge = float(GLASS["z_max"])
    plat_lo, plat_hi = cat_meta["number_density"]["constant_density_plateau"]

    # --- realised comoving number density on the constant-density plateau -------
    # Outside [plat_lo, plat_hi] the linear_windows basis ramps to zero (documented
    # in the catalogs stage), so the density gate is applied on the plateau only
    # and the ramps are reported separately.
    dens = {}
    z_sh = np.linspace(plat_lo, plat_hi, 10)
    v_plat = float(4.0 / 3.0 * np.pi
                   * (acos.comoving_distance(plat_hi).to_value(u.Mpc) ** 3
                      - acos.comoving_distance(plat_lo).to_value(u.Mpc) ** 3))
    for t in TRACERS:
        z_t = cats[t]["z"].astype(np.float64)
        in_plat = (z_t >= plat_lo) & (z_t <= plat_hi)
        n_real = float(in_plat.sum() / v_plat)
        n_tgt = float(GLASS[f"n_comoving_{t}"])
        shell_n = []
        for lo, hi in zip(z_sh[:-1], z_sh[1:]):
            dv = float(4.0 / 3.0 * np.pi
                       * (acos.comoving_distance(hi).to_value(u.Mpc) ** 3
                          - acos.comoving_distance(lo).to_value(u.Mpc) ** 3))
            shell_n.append(float(((z_t > lo) & (z_t <= hi)).sum() / dv))
        dens[t] = {"n_realised_plateau_Mpc^-3": n_real, "n_target_Mpc^-3": n_tgt,
                   "rel_err": n_real / n_tgt - 1.0,
                   "plateau": [plat_lo, plat_hi],
                   "n_objects": int(z_t.size), "n_objects_in_plateau": int(in_plat.sum()),
                   "z_min": float(z_t.min()), "z_max": float(z_t.max()),
                   "shell_edges": z_sh.tolist(),
                   "n_in_shells_Mpc^-3": shell_n,
                   "shell_rms_frac_dev": float(np.std(np.asarray(shell_n) / n_tgt - 1.0))}

    bins = np.linspace(0, z_edge * 1.02, 80)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, t in zip(ax, TRACERS):
        hn, _ = np.histogram(cats[t]["z"], bins=bins)
        ctrs = 0.5 * (bins[1:] + bins[:-1])
        dv = np.diff(4.0 / 3.0 * np.pi
                     * acos.comoving_distance(bins).to_value(u.Mpc) ** 3)
        a.step(ctrs, hn / dv, where="mid",
               label=f"seed{args.seed} ({cats[t]['z'].size:,})", lw=1.6)
        a.axhline(GLASS[f"n_comoving_{t}"], color="crimson", ls="--", lw=1.2,
                  label=f"target {GLASS[f'n_comoving_{t}']:.1e}")
        a.set_yscale("log")
        a.set_xlabel("z"); a.set_title(f"{t.upper()} comoving number density")
        a.legend(fontsize=8)
    ax[0].set_ylabel(r"$n(z)$ [Mpc$^{-3}$]")

    ratio_new = cats["gal"]["z"].size / cats["agn"]["z"].size
    ratio_target = GLASS["n_comoving_gal"] / GLASS["n_comoving_agn"]

    # --- clustering, measured properly ------------------------------------------
    # The GAL auto-amplitude per z-shell, delta2 = <delta^2> - 1/nbar, is
    # well-measured and is compared new-vs-archive: that is the statement that the
    # shared density field is the same model.
    #
    # The AGN/GAL BIAS RATIO must NOT use the AGN auto-variance: with ~0.05-0.3 AGN
    # per pixel the shot-noise subtraction is a difference of two nearly equal noisy
    # numbers.  The CROSS-correlation is shot-noise free, so the estimator used is
    #     b_agn/b_gal = sum_s w_s <delta_a delta_g>_s / sum_s w_s <delta_g^2 - 1/n_g>_s
    # with w_s = nbar_agn,s (inverse AGN shot-noise weight), errors by delete-one
    # jackknife over 48 sky patches.
    nside_b, njk = 32, 2
    npix_b = hp.nside2npix(nside_b)
    jk_of_pix = hp.ang2pix(njk, *hp.pix2ang(nside_b, np.arange(npix_b)))
    shells = [(0.1, 0.25), (0.25, 0.40), (0.40, 0.55), (0.55, 0.70),
              (0.70, 0.85), (0.85, 1.00)]

    def pixz_new(t):
        return (hp.ang2pix(nside_b, 0.5 * np.pi - cats[t]["dec"].astype(np.float64),
                           cats[t]["ra"].astype(np.float64)),
                cats[t]["z"])

    def clustering(get):
        pg, zg = get("gal")
        pa, za = get("agn")
        num, den, auto = np.zeros(npix_b), np.zeros(npix_b), {}
        for lo, hi in shells:
            cg = np.bincount(pg[(zg >= lo) & (zg < hi)],
                             minlength=npix_b).astype(float)
            ca = np.bincount(pa[(za >= lo) & (za < hi)],
                             minlength=npix_b).astype(float)
            mug, mua = cg.mean(), ca.mean()
            dg, da = cg / mug - 1.0, ca / mua - 1.0
            num += mua * dg * da
            den += mua * (dg * dg - 1.0 / mug)
            auto[f"{lo}-{hi}"] = {
                "delta2_gal": float(dg.var() - 1.0 / mug),
                "delta2_gal_err": float(np.sqrt(2.0 / npix_b) * dg.var()),
                "mean_gal_per_pixel": float(mug), "mean_agn_per_pixel": float(mua)}
        r = float(num.sum() / den.sum())
        vals = np.array([num[jk_of_pix != k].sum() / den[jk_of_pix != k].sum()
                         for k in range(hp.nside2npix(njk))])
        n = vals.size
        err = float(np.sqrt((n - 1) / n * np.sum((vals - vals.mean()) ** 2)))
        return r, err, auto

    r_new, e_new, auto_new = clustering(pixz_new)
    b_expect = GLASS["bias_agn"] / GLASS["bias_gal"]
    bias_pull = (abs(r_new - b_expect) / e_new) if e_new else np.inf
    # "detected" = the measurement separates the planted contrast from no contrast
    bias_detection_sigma = (abs(r_new - 1.0) / e_new) if e_new else 0.0

    a = ax[2]
    xs = np.arange(len(shells))
    a.errorbar(xs, [auto_new[f"{lo}-{hi}"]["delta2_gal"] for lo, hi in shells],
               yerr=[auto_new[f"{lo}-{hi}"]["delta2_gal_err"] for lo, hi in shells],
               fmt="o", label=f"seed{args.seed}")
    a.axhline(0, color="0.7", lw=0.8)
    a.set_xticks(xs)
    a.set_xticklabels([f"{lo:g}-{hi:g}" for lo, hi in shells], rotation=45, fontsize=7)
    a.set_ylabel(r"$\langle\delta_g^2\rangle - 1/\bar n$  (nside 32)")
    a.set_title(f"GAL clustering per shell; "
                r"$b_{\rm AGN}/b_{\rm GAL}$ = "
                f"{r_new:.3f}±{e_new:.3f} (planted {b_expect:.3f})")
    a.legend(fontsize=8)
    fig.suptitle("GLASS v2 catalogs: realised comoving densities and clustering")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(vdir / f"fig_dndz_overlay.{ext}", dpi=140)
    plt.close(fig)

    ok_cat = (all(abs(v["rel_err"]) < 0.03 for v in dens.values())
              and abs(ratio_new / ratio_target - 1.0) < 0.05
              and bias_pull < 3.0
              and all(v["shell_rms_frac_dev"] < 0.10 for v in dens.values()))
    check("V4_catalog_densities_and_clustering", ok_cat,
          {"comoving_number_density": dens,
           "density_ratio_gal_over_agn": ratio_new,
           "density_ratio_target": ratio_target,
           "schechter_x_cut": cat_meta["magnitude_model"]["x_cut_L_over_Lstar"],
           "schechter_n_at_0.25Lstar": cat_meta["magnitude_model"][
               "n_at_0.25Lstar_Mpc^-3"],
           "gal_clustering_amplitude": auto_new,
           "bias_ratio_agn_over_gal": {
               "estimator": "shot-noise-free AGN x GAL cross-correlation, "
                            "nbar_agn-weighted stack over 6 z shells at nside 32, "
                            "delete-one jackknife over 48 sky patches",
               "measured": r_new, "err": e_new, "planted": b_expect,
               "pull_vs_planted": float(bias_pull),
               "sigma_from_no_contrast": float(bias_detection_sigma),
               "resolved": bool(bias_detection_sigma > 3.0)},
           "note": "v2 raises the field to nside 128 / lmax 256 and the AGN count to "
                   "~1.6e6, so the planted b_AGN/b_GAL = 1.667 is now measurable; "
                   "the gate is consistency with the planted value at 3 sigma and "
                   "'resolved' records whether it also separates from 1."})

    # ------------------------------------------------------- V5 planted-f bookkeeping
    _log("V5: planted-f bookkeeping")
    n_agn = int(tr["host_type"].sum())
    n_gal = nobs - n_agn
    p = F_AGN
    sig = np.sqrt(nobs * p * (1 - p))
    pull = (n_agn - nobs * p) / sig
    cnt_agn = np.bincount(tr["host_index"][tr["host_type"] == 1].astype(int))
    cnt_gal = np.bincount(tr["host_index"][tr["host_type"] == 0].astype(int))
    check("V5_planted_f_bookkeeping", abs(pull) < 4.0,
          {"planted_f_agn": F_AGN, "n_host_gal": n_gal, "n_host_agn": n_agn,
           "realised_f_agn": n_agn / nobs,
           "binomial_pull": float(pull),
           "unique_agn_hosts": int((cnt_agn > 0).sum()),
           "unique_gal_hosts": int((cnt_gal > 0).sum()),
           "max_events_per_agn_host": int(cnt_agn.max()) if cnt_agn.size else 0,
           "max_events_per_gal_host": int(cnt_gal.max()) if cnt_gal.size else 0,
           "agn_multiplicity_hist": {str(int(k)): int(v) for k, v in
                                     zip(*np.unique(cnt_agn[cnt_agn > 0],
                                                    return_counts=True))},
           "gal_multiplicity_hist": {str(int(k)): int(v) for k, v in
                                     zip(*np.unique(cnt_gal[cnt_gal > 0],
                                                    return_counts=True))}})

    # ------------------------------------------------------- V6 injections + closure
    _log("V6: injection pdraw recomputation and detected-fraction consistency")
    cosmo = gmd._build_cosmology(H0_FID, OM0_FID, W0_FID, WA_FID)
    grids = gmd._cosmology_grids(cosmo, ZMAX_GRID)
    ddldz = np.gradient(grids["dl"], grids["z"])
    pop = gmd.PopulationConfig(gamma=GAMMA)
    pmap = SurveyPixelMap(sd / "surveys" / f"survey_agn_complete_ns{NSIDE_SURVEY}.h5",
                          ZMAX_GRID)
    inj_rec, pdet_curves = {}, {}
    for lane in ("targeted", "popuni"):
        ip = sd / "injections" / f"injections_{lane}.h5"
        if not ip.exists():
            continue
        with h5py.File(ip, "r") as f:
            d = {k: np.asarray(f[k][:]) for k in
                 ("m1det", "m1src", "m2src", "chieff", "ra", "dec", "pdraw", "z",
                  "branch", "pdraw_population", "pdraw_uniform")}
            if lane == "targeted":
                d["pdraw_targeted_agn"] = np.asarray(f["pdraw_targeted_agn"][:])
            ndraw = int(f.attrs["ndraw"])
        n = d["pdraw"].size
        _lm = _read_json(sd / "injections" / f"injections_{lane}_meta.json")
        w = list(_lm["mixture_weights"])
        tot = w[0] * d["pdraw_population"] + w[1] * d["pdraw_uniform"]
        if lane == "targeted":
            tot = tot + w[2] * d["pdraw_targeted_agn"]
        bitwise = bool(np.array_equal(d["pdraw"], np.maximum(tot, PDRAW_FLOOR)))

        idx = rng.choice(n, 200, replace=False)
        # z from the canonical coordinates, NOT the stored z column: exact because
        # m1det = (1+z) m1src.  Shares no code with the write path.
        zc = d["m1det"][idx] / d["m1src"][idx] - 1.0
        qs = d["m2src"][idx] / d["m1src"][idx]
        m1s, chis = d["m1src"][idx], d["chieff"][idx]
        pp = gmd._selection_pdraw("population", m1s, qs, chis, zc, grids, pop)
        pu = gmd._selection_pdraw("uniform", m1s, qs, chis, zc, grids, pop)
        mix = w[0] * pp + w[1] * pu
        if lane == "targeted":
            # independent targeted density: flat host-array scan, not the padded table
            pix_s = pmap.pix_of(d["ra"][idx], d["dec"][idx])
            p_re = np.empty(idx.size)
            for k in range(idx.size):
                inp = pmap.pix_hosts == pix_s[k]
                lj, uj = pmap.lo_hosts[inp], pmap.hi_hosts[inp]
                if lj.size:
                    on = (zc[k] >= lj) & (zc[k] <= uj)
                    ks = float(np.sum(1.0 / (uj[on] - lj[on]))) if on.any() else 0.0
                else:
                    ks = 0.0
                dd = float(np.interp(zc[k], grids["z"], ddldz))
                msp = float(gmd._mass_spin_pdf(np.array([m1s[k]]), np.array([qs[k]]),
                                               np.array([chis[k]]), pop)[0])
                p_re[k] = msp * (ks / pmap.N_hosts / pmap.apix) / max(dd * (1 + zc[k]),
                                                                     1e-300)
            mix = mix + w[2] * p_re
        mix = np.maximum(mix, PDRAW_FLOOR)
        relerr = np.abs(mix - d["pdraw"][idx]) / d["pdraw"][idx]
        m = _read_json(sd / "injections" / f"injections_{lane}_meta.json")
        e = np.asarray(m["pdet_z_grid"]["edges"])
        npr = np.asarray(m["pdet_z_grid"]["n_proposed_population"], dtype=float)
        nde = np.asarray(m["pdet_z_grid"]["n_detected_population"], dtype=float)
        pdet_curves[lane] = {"edges": e, "pdet": np.where(npr > 0, nde / np.maximum(npr, 1), 0.0),
                             "n": npr}
        inj_rec[lane] = {
            "n_detected": n, "ndraw": ndraw, "detected_fraction": n / ndraw,
            "mixture_exact_from_stored_components_bitwise": bitwise,
            "pdraw_recompute_nsamp": int(idx.size),
            "pdraw_recompute_max_rel_err": float(relerr.max()),
            "pdraw_recompute_median_rel_err": float(np.median(relerr)),
            "pdraw_all_positive_finite": bool(np.all(d["pdraw"] > 0)
                                              and np.all(np.isfinite(d["pdraw"]))),
            "n_rows_pdraw_at_floor": int((d["pdraw"] <= PDRAW_FLOOR).sum()),
        }

    # P_det(z) must agree between lanes (same rule, same population branch).
    #
    # This is the standard TWO-PROPORTION z test, with the variance formed from the
    # POOLED estimate and a minimum expected count so the normal approximation is
    # legitimate.  The pre-2026-08-01 version estimated the variance from ONE arm's
    # own p_det, which makes the z-score diverge whenever that arm happens to draw a
    # near-zero count in a deep bin: on seed 101 a bin at z = 0.295 held 1 detection
    # in 5,444 targeted proposals against 37 in 20,102 popuni ones, and the
    # one-arm variance turned a ~2.5-sigma Poisson fluctuation into 8.0 sigma.  With
    # the pooled variance the same seed's 163 well-populated bins give a z
    # distribution of mean -0.08, sd 1.02, max |z| 3.24 -- i.e. the lanes agree.
    # (The same artefact is the most likely reason seed 104 was condemned by this
    # check in the v2 campaign.)  An AGGREGATE comparison over all compared bins is
    # added because it is the low-variance statement a real lane mismatch would fail.
    lane_ok, pdet_cmp = True, {}
    if len(pdet_curves) == 2:
        a, b = pdet_curves["targeted"], pdet_curves["popuni"]
        na, nb = a["n"], b["n"]
        da, db = a["pdet"] * na, b["pdet"] * nb
        m = (na > 2000) & (nb > 2000) & (da >= 25) & (db >= 25)
        pp = (da[m] + db[m]) / (na[m] + nb[m])
        sig = np.sqrt(pp * (1.0 - pp) * (1.0 / na[m] + 1.0 / nb[m]))
        z_dev = (a["pdet"][m] - b["pdet"][m]) / np.maximum(sig, 1e-300)
        # the aggregate over EVERY bin with enough proposals (no count cut)
        mg = (na > 2000) & (nb > 2000)
        Da, Db = float(da[mg].sum()), float(db[mg].sum())
        Na, Nb = float(na[mg].sum()), float(nb[mg].sum())
        Pp = (Da + Db) / (Na + Nb)
        sig_g = np.sqrt(Pp * (1.0 - Pp) * (1.0 / Na + 1.0 / Nb))
        z_agg = (Da / Na - Db / Nb) / max(sig_g, 1e-300)
        lane_ok = bool(z_dev.size > 0 and np.max(np.abs(z_dev)) < 6.0
                       and abs(z_agg) < 6.0)
        pdet_cmp = {"n_bins_compared": int(m.sum()),
                    "test": "two-proportion z with the POOLED variance; bins need "
                            ">2000 proposals and >=25 detections in BOTH lanes",
                    "max_abs_pooled_z": float(np.max(np.abs(z_dev)))
                                        if z_dev.size else 0.0,
                    "mean_pooled_z": float(z_dev.mean()) if z_dev.size else 0.0,
                    "sd_pooled_z": float(z_dev.std(ddof=1)) if z_dev.size > 1 else 0.0,
                    "aggregate_pdet_targeted": Da / Na,
                    "aggregate_pdet_popuni": Db / Nb,
                    "aggregate_ratio": (Da / Na) / (Db / Nb),
                    "aggregate_pooled_z": float(z_agg)}

    # end-to-end closure: predict the EVENT detection fraction from P_det(z)
    ev_meta = _read_json(sd / "events" / "events_meta.json")
    pd_curve = pdet_curves.get("popuni") or next(iter(pdet_curves.values()))
    ctr = 0.5 * (pd_curve["edges"][1:] + pd_curve["edges"][:-1])
    pdet_of_z = lambda zz: np.interp(zz, ctr, pd_curve["pdet"])
    pred = 0.0
    for t, wt in (("gal", 1.0 - F_AGN), ("agn", F_AGN)):
        zz = cats[t]["z"]
        pred += wt * np.mean(pdet_of_z(zz) * (1.0 + zz) ** (GAMMA - 1.0))
    meas = float(ev_meta["realised"]["detected_fraction"])
    n_tried = int(ev_meta["realised"]["n_proposed"])
    n_det_tot = int(ev_meta["realised"]["n_detected_total"])
    # Poisson error on the measured fraction
    sig_meas = np.sqrt(n_det_tot) / n_tried
    closure_z = abs(pred - meas) / sig_meas
    ok_inj = (all(v["mixture_exact_from_stored_components_bitwise"] for v in inj_rec.values())
              and all(v["pdraw_recompute_max_rel_err"] < 1e-8 for v in inj_rec.values())
              and all(v["pdraw_all_positive_finite"] for v in inj_rec.values())
              and lane_ok and closure_z < 5.0)
    check("V6_injections_and_detection_closure", ok_inj,
          {"lanes": inj_rec, "pdet_lane_comparison": pdet_cmp,
           "event_detected_fraction_measured": meas,
           "event_detected_fraction_predicted_from_injections": float(pred),
           "closure_sigma": float(closure_z),
           "note": "the prediction is E_host[(1+z)^(gamma-1) P_det(z)] over the "
                   "planted host mixture, with P_det(z) measured on the injection "
                   "lanes' population branch"})

    # ------------------------------------------- V8 catalog edge vs the PE support
    # The campaign's measured -4.09 km/s/Mpc catalog-edge lever comes from events
    # whose redshift support runs into a catalog that stops.  The requirement is
    # therefore not "z_max is big" but "the PE support, mapped through EVERY H0 the
    # scan visits, stays well inside the catalog".  dL(z; H0) = (H0_FID/H0)
    # dL(z; H0_FID) exactly here, so the worst case is the largest PE dL at the
    # LARGEST H0 -- but the whole grid is scanned anyway.
    _log("V8: catalog/survey redshift edge clears the events' PE support")
    z_edge_cat = float(GLASS["z_max"])
    bar = 0.7 * z_edge_cat
    h0_probe = np.linspace(H0_SCAN_MIN, H0_SCAN_MAX, 51)
    dl_pe = pe["dL"]
    z_of_h0 = []
    for h0 in h0_probe:
        cz = gmd._build_cosmology(float(h0), OM0_FID, W0_FID, WA_FID)
        gz = gmd._cosmology_grids(cz, ZMAX_GRID)
        z_of_h0.append(float(np.interp(dl_pe.max(), gz["dl"], gz["z"])))
    z_pe_max = float(np.max(z_of_h0))
    i_worst = int(np.argmax(z_of_h0))
    # per-survey z grids: nothing may be truncated inside the support
    srv_edges = {}
    for t in TRACERS:
        for tag in ["complete"] + [f"m{int(m)}" for m in MAG_LIMITS]:
            sp = sd / "surveys" / f"survey_{t}_{tag}_ns{NSIDE_SURVEY}.h5"
            with h5py.File(sp, "r") as f:
                srv_edges[f"{t}_{tag}"] = {"z_max": float(f.attrs["z_max"]),
                                           "n_hosts": int(f.attrs["n_hosts"])}
    shell_edge = float(_read_json(sd / "catalogs"
                                  / "glass_field_meta.json")["shell_edges"][-1])
    complete_ok = all(srv_edges[f"{t}_complete"]["z_max"] > z_pe_max
                      for t in TRACERS)
    ok_edge = bool(z_pe_max < bar and complete_ok and shell_edge >= z_edge_cat)
    check("V8_catalog_edge_clears_pe_support", ok_edge,
          {"catalog_z_max_nominal": z_edge_cat,
           "glass_last_shell_edge": shell_edge,
           "bar_0.7_zmax": bar,
           "max_pe_dL_Mpc": float(dl_pe.max()),
           "max_pe_redshift_over_H0_grid": z_pe_max,
           "worst_case_H0": float(h0_probe[i_worst]),
           "margin_fraction_of_zmax": float(z_pe_max / z_edge_cat),
           "complete_survey_z_max_exceeds_support": complete_ok,
           "survey_edges": srv_edges,
           "note": "z' solves dL(z'; H0) = dL_PE; the maximum is taken over every "
                   "PE sample of every event and over 51 H0 values spanning the "
                   "scanned range"})

    # --------------------------------------------- V7 darksirens format contract
    _log("V7: every file loads through darksirens' own loaders")
    fmt = {}
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
        from darksirens.gw.utils import load_gw_samples, load_selection_samples
        from darksirens.catalogs.io import load_survey
        o = load_gw_samples(str(ev_path))
        ppe = np.asarray(o[6]).reshape(o[7], o[8])
        fmt["events"] = {"nobs": int(o[7]), "nsamp": int(o[8]),
                         "p_pe_per_event_sum": float(ppe.sum(axis=1).max()),
                         "dL_min": float(np.min(o[2])), "dL_max": float(np.max(o[2]))}
        for lane in ("targeted", "popuni"):
            ip = sd / "injections" / f"injections_{lane}.h5"
            if ip.exists():
                so = load_selection_samples(str(ip))
                fmt[f"selection_{lane}"] = {"n": int(np.asarray(so[0]).size),
                                            "ndraw": int(so[-1])}
        for t in TRACERS:
            for tag in ["complete"] + [f"m{int(m)}" for m in MAG_LIMITS]:
                load_survey(str(sd / "surveys"
                                / f"survey_{t}_{tag}_ns{NSIDE_SURVEY}.h5"))
        fmt["surveys_loaded"] = 2 * (1 + len(MAG_LIMITS))
        fmt["darksirens_importable"] = True
        ok_fmt = (abs(fmt["events"]["p_pe_per_event_sum"] - 1.0) < 1e-9
                  and fmt["events"]["nobs"] == N_EVENTS)
    except Exception as exc:
        fmt = {"darksirens_importable": False, "error": repr(exc)}
        ok_fmt = False
    check("V7_darksirens_format_contract", ok_fmt, fmt)

    # ------------------------------------------------------------------- figures
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].hist(tr["z"], bins=50, color="0.4")
    ax[0].axvline(tr["z"].max(), color="crimson", lw=1.2,
                  label=f"horizon {tr['z'].max():.3f}")
    ax[0].set_xlabel("true z"); ax[0].set_ylabel("events"); ax[0].legend(fontsize=8)
    ax[0].set_title("detected events")
    ax[1].plot(ctr, pd_curve["pdet"], lw=1.6)
    ax[1].set_xlim(0, 0.5); ax[1].set_yscale("log")
    ax[1].set_xlabel("z"); ax[1].set_ylabel(r"$P_{\rm det}(z)$")
    ax[1].set_title("detection efficiency (population branch)")
    ax[2].scatter(tr["snr_true"], tr["snr_obs"], s=3, alpha=0.4)
    lim = [min(tr["snr_true"].min(), 6), max(tr["snr_obs"].max(), 10)]
    ax[2].plot(lim, lim, color="0.6", lw=0.8)
    ax[2].axhline(SNR_THRESHOLD, color="crimson", lw=1.0)
    ax[2].axvline(SNR_THRESHOLD, color="crimson", lw=1.0, ls="--")
    ax[2].set_xlabel(r"$\rho$(true params)"); ax[2].set_ylabel(r"$\rho_{\rm obs}$")
    ax[2].set_title("detection acts on the DATA")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(vdir / f"fig_detection.{ext}", dpi=140)
    plt.close(fig)

    srv = _read_json(sd / "surveys" / "surveys_meta.json")
    fig, ax = plt.subplots(figsize=(6, 4))
    for t, ls in (("gal", "-"), ("agn", "--")):
        xs, ys = [], []
        for tag in ["complete"] + [f"m{int(m)}" for m in MAG_LIMITS]:
            c = srv["completeness"][tag][t]
            xs.append(c["mag_limit"] if c["mag_limit"] else 24.0)
            ys.append(c["C_within_horizon"])
        ax.plot(xs, ys, ls, marker="o", label=t.upper())
    ax.set_xlabel("apparent-magnitude limit"); ax.invert_xaxis()
    ax.set_ylabel(f"C(z <= {srv['horizon_z']:.3f})")
    ax.set_title("completeness within the GW horizon")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(vdir / f"fig_completeness.{ext}", dpi=140)
    plt.close(fig)

    summary = {"generated_at_utc": _now(), "seed": args.seed,
               "n_checks": len(results), "n_failed": len(failures),
               "failed": failures, "checks": results}
    _write_json(vdir / "validation.json", summary)
    _merge_meta(args.seed, "validation", summary, args.outroot)
    if failures:
        raise SystemExit(f"VALIDATION FAILED: {failures}  (see {vdir / 'validation.json'})")
    _log(f"validation: all {len(results)} checks PASSED")


# ================================================================================
# inventory + META finalisation
# ================================================================================
def finalise_meta(args):
    sd = seed_dir(args.seed, args.outroot)
    inv = []
    for p in sorted(sd.rglob("*")):
        if p.is_file():
            inv.append({"path": str(p.relative_to(sd)), "bytes": p.stat().st_size})
    payload = {
        "generated_at_utc": _now(),
        "master_seed": args.seed,
        "sub_seeds": sub_seeds(args.seed),
        "generator": str(Path(__file__).resolve()),
        "gws_agn_repo_head": _git(REPO_ROOT, "rev-parse", "HEAD"),
        "gws_agn_repo_branch": _git(REPO_ROOT, "rev-parse", "--abbrev-ref", "HEAD"),
        "darksirens_worktree": str(args.darksirens),
        "darksirens_sha": _git(args.darksirens, "rev-parse", "HEAD"),
        "darksirens_branch": _git(args.darksirens, "rev-parse", "--abbrev-ref", "HEAD"),
        "python_main": sys.executable,
        "python_glass": str(args.glass_python),
        "packages_main": _pkg_versions(["numpy", "scipy", "h5py", "healpy", "astropy",
                                        "matplotlib", "jax"]),
        "config": {
            "glass": GLASS,
            "cosmology": {"H0": H0_FID, "Om0": OM0_FID, "w0": W0_FID, "wa": WA_FID,
                          "zmax_grid": ZMAX_GRID},
            "events": {"n": N_EVENTS, "nsamp": N_SAMP, "f_agn": F_AGN, "gamma": GAMMA,
                       "snr_threshold": SNR_THRESHOLD,
                       "snr_ref_detect": SNR_REF_DETECT,
                       "snr_ref_sigma_ang": SNR_REF_SIGMA,
                       "sigma_dL": SIGMA_DL, "sig_m1_frac": SIG_M1_FRAC,
                       "sig_m2_frac": SIG_M2_FRAC, "sigma_chieff": SIGMA_CHIEFF,
                       "pe_model": getattr(args, "pe_model", PE_MODEL_DEFAULT),
                       "v3_measurement_family": {
                           "sigma_rho": SIGMA_RHO, "A_MC": A_MC, "A_Q": A_Q,
                           "A_CHI": A_CHI, "sky_a_deg": SKY_A_DEG,
                           "sky_clip_deg": list(SKY_CLIP_DEG),
                           "width_law": "sigma_x = A_x * (8/rho_obs)",
                           "reference": "working/data/DESIGN_PE.md"}},
            "survey": {"nside": NSIDE_SURVEY, "dz": f"{DZ_SCALE} * (1+z)",
                       "photoz_survey": getattr(args, "photoz_survey",
                                                PHOTOZ_SURVEY_DEFAULT),
                       "mag_limits": list(MAG_LIMITS),
                       "storage_dtype": CAT_DTYPE},
            "luminosity_function": {**SCHECHTER,
                                    "phi_star_Mpc^-3": schechter_phi_star(),
                                    "n_at_0.25Lstar": schechter_number_density(0.25),
                                    "x_cut_for_1e-3": schechter_cut_for_density(
                                        GLASS["n_comoving_gal"])},
            "agn_lf": AGN_LF,
            "injections": {"ndraw_targeted": args.ndraw_targeted or args.ndraw,
                           "ndraw_popuni": args.ndraw_popuni or args.ndraw,
                           "targeted_mixture": {"population": args.mix_population,
                                                "uniform": args.mix_uniform,
                                                "targeted_agn": args.mix_targeted},
                           "targeted_branch": {
                               "kind": "per-host uniform box in z, H0-range covering",
                               "H0_scan_range": [H0_SCAN_MIN, H0_SCAN_MAX],
                               "r_lo": TGT_R_LO, "r_hi": TGT_R_HI,
                               "nsigma_pad": TGT_NSIG_PAD,
                               "host_zmax": TGT_HOST_ZMAX, "z_cap": TGT_Z_CAP},
                           "crosscheck_proposal": "population+uniform (0.9/0.1)"},
        },
        "conventions": {
            "a_detection": "rho_obs computed from the recorded measurement >= 8; no "
                           "true-redshift cut, no projection latent, no separate PE "
                           "noise draw",
            "b_sky_width": "sigma_ang = clip(35/rho_opt(OBSERVED dL, masses), 1, 12) "
                           "deg, drawn sequentially before the sky offsets",
            "c_pe": "exact flat-prior posterior of that measurement; "
                    "ln dL ~ N(ln d_obs + s^2, s)",
            "b2_ra_width": {
                "value": "sig_ra = sigma_ang / max(cos dec_OBS, 0.1), stored as "
                         "obs_sig_ra; dec is drawn BEFORE ra and the PE reuses the "
                         "stored number verbatim",
                "changed_at_utc": "2026-08-01",
                "was": "observe() used cos(dec_TRUE) while posterior_samples() used "
                       "cos(dec_obs)",
                "why": "the sky twin of (c2): the recorded RA posterior width was "
                       "wrong by |cos dec_obs/cos dec_true - 1| = 2.1-2.4 % mean, "
                       "3.7-5.0 % rms, 26-57 % max across the five seeds, i.e. the "
                       "one place convention (b) was not actually honoured "
                       "(ATTRIBUTION.md A4.5, A5.2).  Gated by validation V2b.",
                "cost_of_the_fix": "events stage only; the detected set is bit-"
                                   "identical (verify_events_regen.py)"},
            "c2_mass_pe": {
                "value": "p(m|obs) ~ (1/(f m)) exp[-(obs-m)^2/(2 f^2 m^2)], drawn by "
                         "inverse CDF; E_post[m]/obs = 1 + 2 f^2 + O(f^4)",
                "changed_at_utc": "2026-08-01",
                "was": "N(m; obs, f * m_TRUE) -- a fixed-width Gaussian about the "
                       "observation whose width came from the LATENT mass",
                "why": "the realised measurement is obs ~ N(m, f m) with f constant, "
                       "whose flat-prior posterior is skewed, not Gaussian.  Measured "
                       "cost on the matched-GAL control: 39.5 % of the per-event "
                       "score residual r and +2.15 km/s/Mpc (ATTRIBUTION.md A2/A3).  "
                       "The naive repair (Gaussian of width f*obs) is WORSE.  Gated "
                       "by validation V3.",
                "cost_of_the_fix": "events stage only"},
            "v3_measurement_family": {
                "value": ("rho_obs = rho_opt(theta) + N(0, 1), detection on rho_obs, "
                          "EVERY other width = A_x * (8/rho_obs); PE = the exact "
                          "flat-prior posterior in (ln Mc_det, ln q, rho, chieff, "
                          "ra, dec), truncated only in the PRIOR; dL is DERIVED from "
                          "(Mc_det, rho), never measured on its own"),
                "changed_at_utc": "2026-08-01",
                "was": ("independent m1det/m2det Gaussians of LATENT width f*m_true "
                        "and an independent lognormal dL of constant log-width"),
                "why": ("no measurement model in the latent-width family closes the "
                        "detected-set score identity: with the exact per-event "
                        "posterior, exact host positions and the exact selection "
                        "function, (C - A) in the mass channel was -1.274e-3 +- "
                        "0.113e-3, an 11.3 sigma violation (CLOSURE.md 15).  In v3 "
                        "every width is a function of OBSERVED data, so the "
                        "generative likelihood is exactly invertible and the "
                        "identity closes by construction.  Constants are the "
                        "GWMockCat / Fishbach-Holz-Farr published values; see "
                        "working/data/DESIGN_PE.md for the citations, the "
                        "calibration of A_Q and the p_pe Jacobian derivation.  "
                        "Gated by validation V1, V2, V2b, V3, V3b, V3c."),
                "P_det_closed_form": "Phi((rho_opt(theta) - 8)/sigma_rho)",
                "p_pe": "rho/(dL m1det q) in the canonical (m1det, q, dL, chieff) "
                        "basis",
                "cost_of_the_fix": "full regeneration: catalogs (D3), events, "
                                   "surveys, injections"},
            "d3_photoz_realised": {
                "value": (f"catalogs carry z_obs = z + N(0, {DZ_SCALE:g} (1+z)) and "
                          "the survey blocks pixelate z_obs with the declared width "
                          f"dz = {DZ_SCALE:g} (1+z_obs); z (true) still drives the "
                          "host draw and the event's truth"),
                "changed_at_utc": "2026-08-01",
                "was": ("the survey block declared dz = 3e-3 (1+z) on redshifts "
                        "copied BIT-FOR-BIT from the catalog the hosts are drawn "
                        "from -- a kernel on redshifts that carry no error at all"),
                "why": ("CLOSURE.md 15.4 measured that as a 7.6 sigma (A - B) "
                        "violation in the catalog p_z channel.  With the error "
                        "realised, darksirens' per-galaxy kernel "
                        "g(z) N(z; z_obs, sigma)/Z(z_obs) IS the posterior for the "
                        "galaxy's true redshift given its catalog entry, so "
                        "p_z(z|pix) is the correct prior for the host's true "
                        "redshift.  Gated by validation V9."),
                "cost_of_the_fix": "catalogs -> surveys -> injections -> everything"},
            "why_catalogs_surveys_injections_are_untouched": (
                "SUPERSEDED 2026-08-01 -- v3 + D3 regenerate the whole pipeline.  "
                "The pre-v3 statement follows.  "
                "The catalogs precede the events stage.  The surveys are pixelations "
                "of the catalogs; the only thing they read from the events is the "
                "realised horizon z, which is a function of the DETECTED SET and is "
                "bit-identical.  The injections never open events.h5 at all: they "
                "call observe(need_sky=False), which neither (b2) (inside the "
                "need_sky block) nor (c2) (in posterior_samples) touches, and they "
                "store TRUE parameters, so mu(theta) is unchanged.  Verified: the "
                "detected set, snr_obs, sigma_ang, obs_chieff and every realised "
                "bookkeeping number are bit-identical across the regeneration."),
            "d_storage_dtype": {
                "value": CAT_DTYPE,
                "changed_at_utc": "2026-07-31",
                "was": "float32",
                "why": (
                    "darksirens' general incomplete-catalog likelihood `dark_sirens` "
                    "cannot be evaluated at all on float32 survey blocks: its "
                    "observed-density KDE (redshift/completion.py::_kde_dndz_obs) "
                    "computes the truncated-kernel mass in the catalog's storage "
                    "dtype and clamps it at 1e-300, which is not representable in "
                    "float32, so every z = 100 padded slot yields 0/0 = NaN, every "
                    "padded catalog row comes back all-NaN, and the likelihood is "
                    "-inf in every cell.  float64 storage makes the clamp hold.  The "
                    "campaign therefore uses one nested likelihood (`dark_sirens` at "
                    "log10n0 = -24, bitwise equal to `dark_sirens_complete`) for "
                    "every run.  See "
                    "working/experiments/experiment_model_equivalence/README.md."),
                "consequence_for_the_data": (
                    "The stored NUMBERS change, not the model.  Catalog columns are "
                    "no longer rounded to float32, so `stage_events` draws its hosts "
                    "from float64 redshifts/positions: the events of a given seed "
                    "differ from the float32-era events of the same seed at the ~1e-7 "
                    "relative level (float32 eps = 1.2e-7), which is five orders "
                    "below the catalog KDE width dz = 3e-3 (1+z).  Every downstream "
                    "product (surveys, injections, validation) is regenerated from "
                    "these, so the seed is internally consistent; it is NOT "
                    "bit-comparable with a float32-era run of the same seed.  The "
                    "reference model's posterior moved by <= 2e-7 km/s/Mpc under the "
                    "dtype change, i.e. the science is unchanged."),
            },
        },
        "file_inventory": inv,
        "total_bytes": sum(x["bytes"] for x in inv),
    }
    meta_path = sd / "META.json"
    meta = _read_json(meta_path) if meta_path.exists() else {}
    meta.update(payload)
    _write_json(meta_path, meta)
    _log(f"META.json finalised: {len(inv)} files, "
         f"{payload['total_bytes'] / 1e9:.2f} GB")


# ================================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=100, help="Master seed (default 100).")
    p.add_argument("--stage", default="all",
                   choices=("all", *STAGES, "meta"),
                   help="Run one stage, or 'all'.")
    p.add_argument("--outroot", default=str(HERE),
                   help="Directory that will contain seed<SEED>/.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--darksirens", default=str(DARKSIRENS_REPO),
                   help="Pinned darksirens checkout supplying generate_mock_data.py.")
    p.add_argument("--glass_python", default=str(HERE / ".venv_glass/bin/python"),
                   help="Interpreter with glass/camb installed (numpy >= 2).")
    p.add_argument("--reference_dir",
                   default=str(REPO_ROOT / "working/archive/gw_agn_darksirens/data"),
                   help="UNUSED since v2: the archived reference is a different "
                        "generative model (v1 densities, z_max 1.565, lmax 64), so "
                        "V4 now checks the realised comoving density and the planted "
                        "bias contrast internally instead.")
    p.add_argument("--ndraw", type=int, default=NDRAW_INJ,
                   help="Proposals per selection lane (default for both lanes).")
    p.add_argument("--ndraw_targeted", type=int, default=None,
                   help="Override --ndraw for the catalog-targeted lane.")
    p.add_argument("--ndraw_popuni", type=int, default=None,
                   help="Override --ndraw for the population+uniform lane.")
    p.add_argument("--mix_population", type=float, default=MIX_POPULATION)
    p.add_argument("--mix_uniform", type=float, default=MIX_UNIFORM)
    p.add_argument("--mix_targeted", type=float, default=MIX_TARGETED_AGN)
    p.add_argument("--batch_size", type=int, default=2_000_000)
    p.add_argument("--jobs", type=int, default=None)
    p.add_argument("--lane", default="both", choices=("both", "targeted", "popuni"))
    p.add_argument("--pe_model", default=PE_MODEL_DEFAULT, choices=("v2", "v3"),
                   help="Measurement family.  v3 (default, 2026-08-01) is the "
                        "literature-standard all-observable family of "
                        "working/data/DESIGN_PE.md: rho_obs = rho_opt + N(0,1), "
                        "detection on rho_obs, every width a_x*8/rho_obs, PE in "
                        "(ln Mc, ln q, rho, chieff, sky).  v2 is the pre-redesign "
                        "family (latent-width component masses) and is kept only "
                        "for regression.")
    p.add_argument("--photoz_survey", default=PHOTOZ_SURVEY_DEFAULT,
                   choices=("obs", "true"),
                   help="Which catalog redshift the SURVEY blocks carry.  'obs' "
                        "(default) is the D3 fix: the block pixelates the catalog's "
                        "photo-z redshift z_obs, so its declared "
                        "dz = 3e-3 (1+z) describes a REAL error.  'true' reproduces "
                        "the pre-2026-08-01 convention for regression.")
    p.add_argument("--n_events", type=int, default=None,
                   help="PILOT ONLY: override N_EVENTS for a small closure pilot.")
    p.add_argument("--nsamp", type=int, default=None,
                   help="PILOT ONLY: override N_SAMP.")
    p.add_argument("--events_suffix", default="",
                   help="PILOT ONLY: write events<SUFFIX>.h5 instead of events.h5.")
    p.add_argument("--f_agn", type=float, default=None,
                   help="EXTRA DRAWS ONLY: planted AGN-hosted fraction for the "
                        f"events stage.  Unset = the module constant F_AGN "
                        f"({F_AGN}), i.e. the dataset of record.  Set together "
                        "with --seed_events and --events_suffix to add an "
                        "independent single-tracer event set (f_agn=0 pure GAL, "
                        "f_agn=1 pure AGN) beside the record's mixture without "
                        "touching it.")
    p.add_argument("--seed_events", type=int, default=None,
                   help="EXTRA DRAWS ONLY: explicit sub-seed for the events stage. "
                        "Unset = the record's derivation SEED*1000+3.  Offsets 1-7 "
                        "are taken by sub_seeds(); use SEED*1000+8 or higher so an "
                        "extra draw is independent of every recorded stream.")
    p.add_argument("--_glass_worker", action="store_true",
                   help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.outroot = Path(args.outroot)
    args.darksirens = Path(args.darksirens)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    stages = STAGES if args.stage == "all" else (
        () if args.stage == "meta" else (args.stage,))
    fns = {"catalogs": stage_catalogs, "events": stage_events,
           "surveys": stage_surveys, "injections": stage_injections,
           "validation": stage_validation}
    for s in stages:
        _log(f"===== STAGE {s} (seed {args.seed}) =====")
        fns[s](args)
    if args.stage in ("all", "meta") or args.stage == "validation":
        finalise_meta(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
