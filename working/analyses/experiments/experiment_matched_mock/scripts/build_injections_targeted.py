#!/usr/bin/env python3
"""Catalog-TARGETED selection injections for the experiment_matched_mock deep mock.

WHY
---
darksirens estimates the selection integral mu(theta) by reweighting a frozen set
of DETECTED injections.  Under ``--catalog_sky_weighting field`` with a host
catalog and ``--universe_model dark_sirens_complete`` the integrand is
CATALOG-CONDITIONED: the target density carries

    p_z(z, pix) = p_cat(z | pix) * N_obs[pix] / N_obs_total,
    p_cat(z | pix) = g(z) * sum_{i in pix} (w_i / W_pix) * N(z; z_i, sig_i) / Z_i,
    Z_i = int N(z; z_i, sig_i) g(z) dz,   g(z) = dV_c/dz (1+z)^delta,

i.e. an injection only carries weight if its redshift lands within a few
``sig_i = sqrt(dzgals^2 + sigma_kde^2)`` of an actual catalog host IN ITS OWN
HEALPix pixel.  For this mock ``dzgals = 3e-3`` and the detected events sit at
z <~ 0.32, where the 1e6-host catalog has only ~0.6 hosts per nside-64 pixel
below z = 0.4 -- so a population/uniform proposal puts essentially all of its
detected injections at redshifts where their own pixel holds no host, and
p_target = 0 there.  Measured on this mock:

    gmd population+uniform, 120,000,000 proposed / 343,702 detected
      generator-printed (population) N_eff = 5133
      darksirens catalog-conditioned N_eff =  682     (needs > 5*N_obs = 5000)

This generator adds a third proposal branch that draws injections AT catalog
hosts, so a controlled fraction of the detected rows is guaranteed to sit on the
catalog's redshift-KDE support.

PROPOSAL
--------
Three-branch mixture; every row's stored ``pdraw`` is the EXACT mixture density
evaluated at that row's coordinates, regardless of which branch produced it:

    pdraw = f_pop * p_population + f_unif * p_uniform + f_tgt * p_targeted

* ``p_population`` / ``p_uniform`` are ``gmd._selection_pdraw("population", ...)``
  and ``gmd._selection_pdraw("uniform", ...)`` -- byte-identical to the densities
  the library's own generator stores, so the population and uniform branches are
  the same objects gmd would have drawn (z uniform-in-comoving-volume on
  [0, zmax], sky isotropic, masses/spins from the fiducial powerlaw+peak
  samplers resp. uniform m1det/q/chi).
* the TARGETED branch draws
    - host j uniformly from the 1e6-host catalog,
    - z ~ Normal(z_j, sigma_j) truncated to (0, zmax) (renormalised, exact),
    - sky uniform WITHIN host j's nside-64 RING pixel (bounding-box rejection),
    - m1src / q / chieff from the same fiducial population samplers as the
      population branch (with gmd's per-component pairing mask).
  ``sigma_j = sqrt(sigma_t^2 + (sigma_t_rel * z_j)^2)``; ``sigma_t_rel = 0``
  (the default) gives the constant width matched to the survey KDE.

TARGETED-BRANCH DENSITY (derived from how the draw is actually made)
-------------------------------------------------------------------
In the sampling variables (m1src, q, chi, z, Omega) the targeted branch is a
finite mixture over the N hosts,

    q_tgt(m1src,q,chi, z,Omega)
        = p_ms(m1src,q,chi) * sum_j (1/N) * TN(z; z_j, sigma_j)
                                      * 1{Omega in P_j} / Omega_pix
        = p_ms(m1src,q,chi) * (1/Omega_pix)
              * sum_{j : P_j = P(Omega)} (1/N) TN(z; z_j, sigma_j),

because the sky draw is a DISCRETE pixel choice (host j fixes the pixel) followed
by a uniform draw inside that equal-area pixel -- so for a query point only the
hosts sharing its pixel contribute.  ``TN`` is the (0, zmax)-truncated normal

    TN(z; z_j, s) = phi((z - z_j)/s) / s / [Phi((zmax - z_j)/s) - Phi(-z_j/s)],

and ``p_ms = gmd._mass_spin_pdf``.  Normalisation is exact by construction:
integrating over Omega gives Omega_pix/Omega_pix = 1 per host and TN integrates
to 1 on the sampled support, so sum_j (1/N) * 1 * 1 = 1.

darksirens consumes densities in the canonical basis (m1det, q, chi, dL) and gmd
writes them per STERADIAN of sky (``_selection_pdraw`` divides by 4*pi).  The map
(m1src, z) -> (m1det, dL) at fixed (q, chi, Omega) is m1det = (1+z) m1src,
dL = dL(z), whose Jacobian is triangular with determinant (1+z) * dL'(z).  Hence

    p_targeted(m1det, q, chi, dL, Omega)
        = q_tgt / [ (1+z) * dL'(z) ]

which is structurally IDENTICAL to gmd's ``_p_population`` -- the same
``_mass_spin_pdf``, the same ``np.gradient(dl, z)`` derivative, the same
``(1+z)`` factor -- with ``p_z(z)/(4 pi)`` replaced by the catalog z-sky factor
``[sum_j (1/N) TN] / Omega_pix``.  All three branch densities are therefore in
the same measure and may be mixed directly.

(darksirens' own target uses a DISCRETE pixel probability rather than a
per-steradian density, i.e. it differs from this convention by one global factor
Omega_pix.  That factor is row-independent because HEALPix pixels are equal-area,
so it cancels out of N_eff and out of every parameter-dependent part of logL --
exactly as it does for gmd's own injection files, whose convention this matches.)

OUTPUT
------
``gwcat-selection-1.0``, DETECTED ROWS ONLY, datasets = ``gmd.SELECTION_KEYS``
= [m1det, m2det, m1src, m2src, dL, chieff, ra, dec, pdraw], plus auditability
extras (``z``, ``branch``, ``pdraw_population``, ``pdraw_uniform``,
``pdraw_targeted``; the loader ignores unknown datasets), and ``ndraw`` /
``Ndraw`` attrs equal to the TOTAL number PROPOSED (not detected).

DETECTION
---------
``gmd._network_snr(m1src, m2src, z, dL, rng) >= snr_threshold`` -- the SAME noisy
network SNR, with the same Beta(2,5)**0.5 projection draw, that generated the
events (``gmd._draw_events_until_detected``).  This is the one substantive
difference from ``working/gw_agn_darksirens/scripts/build_injections.py``, whose
mock used a hard true-redshift cut.  If the detection rule here were not the
events' own rule the selection integral would be inconsistent with the data and
the whole exercise would be void.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import healpy as hp
from scipy.special import ndtr  # standard-normal CDF Phi(x), vectorised

EXP_ROOT = Path(__file__).resolve().parents[1]

# --- pinned darksirens worktree that generated the mock ----------------------
DEFAULT_WORKTREE = ("/tmp/claude-88592/-hildafs-projects-phy230014p-magana-gws-agn/"
                    "6b9abc89-f874-41de-9ed3-c0ca4def231c/scratchpad/wt-2b86a2d")

# --- fiducials of the deep mock (deep_mock_z2 / deep_mock_z2_big) -------------
H0_FID = 67.74
OM0_FID = 0.3075
W0_FID = -1.0
WA_FID = 0.0
ZMAX = 2.0                 # cosmology-grid / proposal upper bound
SNR_THRESHOLD = 8.0        # gmd --snr-threshold used for events + injections
NSIDE = 64                 # survey / catalog HEALPix nside (RING)
CHIEFF_AMAX = 0.99
PDRAW_FLOOR = 1.0e-300     # gmd._selection_pdraw's np.maximum floor, retained

# --- proposal defaults -------------------------------------------------------
MIX_POPULATION = 0.65
MIX_UNIFORM = 0.10
MIX_TARGETED = 0.25
SIGMA_T = 3.0e-3           # matched to the survey's dzgals KDE width
SIGMA_T_REL = 0.0          # optional relative broadening: s_j^2 += (rel*z_j)^2

DEFAULT_CATALOG = EXP_ROOT / "data_derived/deep_mock_z2/mock_galaxy_catalog_complete.h5"

SELECTION_EXTRA_KEYS = ["z", "branch", "pdraw_population", "pdraw_uniform",
                        "pdraw_targeted"]
BRANCH_NAMES = ("population", "uniform", "targeted")


# =============================================================================
# provenance helpers
# =============================================================================
def _git(repo, *args):
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception as exc:  # pragma: no cover - provenance best-effort
        return f"<unavailable: {exc}>"


# =============================================================================
# catalog pixel map
# =============================================================================
class CatalogPixelMap:
    """Immutable view of the host catalog for the catalog-targeted branch.

    Holds per-host (z, RING nside pixel, kernel width, truncation norm) plus a
    padded pixel -> hosts table so the kernel sum

        S(z, pix) = sum_{j in pix} TN(z; z_j, sigma_j)

    can be evaluated vectorised for arbitrary query rows.  Only the hosts in the
    query's OWN pixel contribute (the targeted sky draw is a discrete pixel
    choice), so the padded table with <= max_hosts_per_pixel columns is exact.

    Catalog ``ra``/``dec`` are RADIANS (gmd ``_generate_complete_catalog`` ->
    ``_sample_sky``), matching gmd ``_pixelate_catalog``'s
    ``hp.ang2pix(nside, pi/2 - dec, ra)`` RING convention and darksirens'
    ``inference/loaders.py`` pixelisation of the injection sky positions.
    """

    def __init__(self, path, nside, sigma_t, sigma_t_rel, zmax):
        with h5py.File(path, "r") as f:
            z = np.asarray(f["z"][:], dtype=np.float64)
            ra = np.asarray(f["ra"][:], dtype=np.float64)
            dec = np.asarray(f["dec"][:], dtype=np.float64)
        if float(ra.max()) > 2.0 * np.pi + 1e-6 or float(np.abs(dec).max()) > 0.5 * np.pi + 1e-6:
            raise SystemExit("catalog ra/dec are not radians -- refusing to guess units")
        self.path = str(path)
        self.nside = int(nside)
        self.npix = int(hp.nside2npix(nside))
        self.N_hosts = int(z.size)
        self.apix = float(hp.nside2pixarea(nside))
        self.resol = float(hp.nside2resol(nside))
        self.sigma_t = float(sigma_t)
        self.sigma_t_rel = float(sigma_t_rel)
        self.zmax = float(zmax)
        self.z_hosts = z
        self.ra_hosts = ra
        self.dec_hosts = dec
        self.pix_hosts = hp.ang2pix(nside, 0.5 * np.pi - dec, ra)
        self.sig_hosts = np.sqrt(sigma_t**2 + (sigma_t_rel * z) ** 2)
        # (0, zmax) truncation normalisation of the per-host proposal kernel.
        self.tnorm_hosts = (ndtr((self.zmax - z) / self.sig_hosts)
                            - ndtr((0.0 - z) / self.sig_hosts))

        # Padded pixel -> hosts table (NaN pad); lookup[pix] = compact row or -1.
        order = np.argsort(self.pix_hosts, kind="stable")
        pix_sorted = self.pix_hosts[order]
        uniq, counts = np.unique(pix_sorted, return_counts=True)
        n_occ = int(uniq.size)
        self.maxcol = int(counts.max())
        self.n_occupied = n_occ
        col = np.concatenate([np.arange(c) for c in counts])
        rowidx = np.repeat(np.arange(n_occ), counts)
        self.z_pad = np.full((n_occ, self.maxcol), np.nan)
        self.sig_pad = np.full((n_occ, self.maxcol), 1.0)
        self.tnorm_pad = np.full((n_occ, self.maxcol), 1.0)
        self.z_pad[rowidx, col] = z[order]
        self.sig_pad[rowidx, col] = self.sig_hosts[order]
        self.tnorm_pad[rowidx, col] = self.tnorm_hosts[order]
        self.finite_pad = np.isfinite(self.z_pad)
        self.lookup = np.full(self.npix, -1, dtype=np.int64)
        self.lookup[uniq] = np.arange(n_occ)
        self.counts = np.zeros(self.npix, dtype=np.int64)
        self.counts[uniq] = counts

    def pix_of(self, ra, dec):
        return hp.ang2pix(self.nside, 0.5 * np.pi - np.asarray(dec), np.asarray(ra))

    def kernel_sum(self, z_rows, pix_rows):
        """sum_{j in pix} TN(z; z_j, sigma_j) per row (0 if the pixel is empty)."""
        z_rows = np.asarray(z_rows, dtype=np.float64)
        out = np.zeros(z_rows.shape[0], dtype=np.float64)
        ci = self.lookup[np.asarray(pix_rows)]
        valid = ci >= 0
        if not valid.any():
            return out
        zc = z_rows[valid]
        rows = ci[valid]
        zmat = self.z_pad[rows]
        sig = self.sig_pad[rows]
        tn = self.tnorm_pad[rows]
        fin = self.finite_pad[rows]
        x = (zc[:, None] - np.where(fin, zmat, 0.0)) / sig
        pdf = np.exp(-0.5 * x * x) / (sig * np.sqrt(2.0 * np.pi))
        out[valid] = np.where(fin, pdf / tn, 0.0).sum(axis=1)
        return out


def sample_truncated_normal_at_hosts(rng, z_j, sig_j, zmax):
    """z ~ Normal(z_j, sig_j) truncated to (0, zmax), by rejection.

    Rejection resamples the SAME host, so the accepted draw is exactly the
    renormalised truncated normal whose density ``CatalogPixelMap.kernel_sum``
    evaluates (normalisation Phi((zmax - z_j)/s) - Phi(-z_j/s))."""
    z = z_j + rng.normal(0.0, 1.0, size=z_j.shape[0]) * sig_j
    bad = (z <= 0.0) | (z >= zmax)
    it = 0
    while bad.any():
        n = int(bad.sum())
        z[bad] = z_j[bad] + rng.normal(0.0, 1.0, size=n) * sig_j[bad]
        bad = (z <= 0.0) | (z >= zmax)
        it += 1
        if it > 10_000:
            raise RuntimeError("truncated-normal rejection failed to converge")
    return z


def sample_uniform_in_pixels(rng, pix_tgt, nside, resol, maxiter=2000):
    """Uniform-on-sphere WITHIN each target RING pixel.

    Bounding-box proposal (uniform in cos(theta) x uniform in phi around the
    pixel centre) rejected on ``ang2pix == pix_tgt``.  Rejection of a uniform
    proposal restricted to a superset of the pixel is exactly uniform on the
    pixel, which is what makes the targeted branch's sky density the constant
    1/Omega_pix inside the pixel.  Returns (ra, dec, n_fallback); a nonzero
    fallback count means some row was placed at its pixel centre (still the
    correct pixel, but not uniform) and is reported in the validation record."""
    n = pix_tgt.shape[0]
    theta_c, phi_c = hp.pix2ang(nside, pix_tgt)
    dt = 2.0 * resol
    cos_lo = np.cos(np.minimum(theta_c + dt, np.pi))
    cos_hi = np.cos(np.maximum(theta_c - dt, 0.0))
    sin_c = np.sin(np.clip(theta_c, 1.0e-6, np.pi - 1.0e-6))
    dphi = np.minimum(2.5 * resol / np.maximum(sin_c, 1.0e-2), np.pi)
    out_theta = np.empty(n)
    out_phi = np.empty(n)
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
        out_theta[hit] = th[acc]
        out_phi[hit] = ph[acc]
        todo[hit] = False
    n_fallback = int(todo.sum())
    if n_fallback:
        out_theta[todo] = theta_c[todo]
        out_phi[todo] = phi_c[todo]
    return out_phi, 0.5 * np.pi - out_theta, n_fallback


# =============================================================================
# densities
# =============================================================================
def p_targeted_density(gmd, z, ra, dec, m1src, q, chi, grids, ddldz_grid, pop, pixmap):
    """Targeted-branch proposal density in the canonical basis, per steradian.

        p_tgt = p_ms(m1src,q,chi) * [ (1/N) sum_{j in pix} TN(z; z_j, s_j) ]
                / Omega_pix / ( (1+z) * dL'(z) )

    Same mass/spin factor, same ``np.gradient(dl, z)`` derivative and same
    ``(1+z)`` detector-frame Jacobian as ``gmd._selection_pdraw("population")``;
    only the z-sky factor differs (catalog kernel sum instead of
    ``p_z(z)/(4 pi)``)."""
    ksum = pixmap.kernel_sum(z, pixmap.pix_of(ra, dec))
    p_zsky = ksum / pixmap.N_hosts / pixmap.apix
    ddldz = np.interp(z, grids["z"], ddldz_grid)
    jac = ddldz * (1.0 + z)
    msp = gmd._mass_spin_pdf(m1src, q, chi, pop)
    return msp * p_zsky / np.maximum(jac, 1.0e-300)


def mixture_pdraw(gmd, mix, z, ra, dec, m1src, q, chi, grids, ddldz_grid, pop, pixmap):
    """Exact 3-branch mixture density and its components (floored like gmd)."""
    f_pop, f_unif, f_tgt = mix
    p_pop = gmd._selection_pdraw("population", m1src, q, chi, z, grids, pop)
    p_unif = gmd._selection_pdraw("uniform", m1src, q, chi, z, grids, pop)
    if f_tgt > 0.0:
        p_tgt = p_targeted_density(gmd, z, ra, dec, m1src, q, chi,
                                   grids, ddldz_grid, pop, pixmap)
    else:
        p_tgt = np.zeros_like(p_pop)
    total = np.maximum(f_pop * p_pop + f_unif * p_unif + f_tgt * p_tgt, PDRAW_FLOOR)
    return total, p_pop, p_unif, p_tgt


# =============================================================================
# draw
# =============================================================================
def draw_injections(gmd, ndraw, seed, grids, ddldz_grid, pop, pixmap, mix,
                    snr_threshold, batch_size, zmax, verbose=True):
    """Draw the 3-branch mixture over batches; keep detected rows only.

    Per batch: one uniform for the categorical branch split, then the
    population-branch draws, the uniform-branch draws, the targeted-branch
    draws, then ONE ``gmd._network_snr`` call over the whole batch (so every
    proposal gets one independent Beta(2,5)**0.5 projection draw exactly as in
    ``gmd._draw_events_until_detected`` / ``gmd._draw_selection_batch``)."""
    rng = np.random.default_rng(seed)
    m1lo, m1hi = gmd._M1DET_RANGE
    f_pop, f_unif, f_tgt = mix
    thr_unif = f_pop + f_unif
    chunks = []
    n_proposed = n_detected = 0
    n_prop = [0, 0, 0]
    n_det = [0, 0, 0]
    n_pix_fallback = 0
    ci = 0
    while n_proposed < ndraw:
        nb = int(min(batch_size, ndraw - n_proposed))
        bu = rng.uniform(size=nb)
        is_pop = bu < f_pop
        is_unif = (bu >= f_pop) & (bu < thr_unif)
        is_tgt = bu >= thr_unif
        z = np.empty(nb); ra = np.empty(nb); dec = np.empty(nb)
        m1src = np.empty(nb); q = np.empty(nb); chi = np.empty(nb)
        branch = np.empty(nb, dtype=np.int8)

        npop = int(is_pop.sum())
        if npop:
            zc = gmd._sample_uniform_comoving_z(rng, grids, npop)
            rac, decc = gmd._sample_sky(rng, npop)
            m1c, use_peak = gmd._sample_powerlaw_peak_m1(rng, npop, pop,
                                                         return_component=True)
            qc = gmd._sample_q(rng, m1c, pop, use_peak=use_peak)
            chic = gmd._sample_chieff(rng, npop, pop)
            z[is_pop] = zc; ra[is_pop] = rac; dec[is_pop] = decc
            m1src[is_pop] = m1c; q[is_pop] = qc; chi[is_pop] = chic
            branch[is_pop] = 0

        nunif = int(is_unif.sum())
        if nunif:
            zc = gmd._sample_uniform_comoving_z(rng, grids, nunif)
            rac, decc = gmd._sample_sky(rng, nunif)
            m1det_u = rng.uniform(m1lo, m1hi, nunif)
            qc = rng.uniform(0.0, 1.0, nunif)
            chic = rng.uniform(-1.0, 1.0, nunif)
            z[is_unif] = zc; ra[is_unif] = rac; dec[is_unif] = decc
            m1src[is_unif] = m1det_u / (1.0 + zc); q[is_unif] = qc
            chi[is_unif] = chic; branch[is_unif] = 1

        ntgt = int(is_tgt.sum())
        if ntgt:
            j = rng.integers(0, pixmap.N_hosts, ntgt)
            zc = sample_truncated_normal_at_hosts(
                rng, pixmap.z_hosts[j], pixmap.sig_hosts[j], zmax)
            rac, decc, nfb = sample_uniform_in_pixels(
                rng, pixmap.pix_hosts[j], pixmap.nside, pixmap.resol)
            n_pix_fallback += nfb
            m1c, use_peak = gmd._sample_powerlaw_peak_m1(rng, ntgt, pop,
                                                         return_component=True)
            qc = gmd._sample_q(rng, m1c, pop, use_peak=use_peak)
            chic = gmd._sample_chieff(rng, ntgt, pop)
            z[is_tgt] = zc; ra[is_tgt] = rac; dec[is_tgt] = decc
            m1src[is_tgt] = m1c; q[is_tgt] = qc; chi[is_tgt] = chic
            branch[is_tgt] = 2

        m2src = q * m1src
        dl = gmd._interp_dl(z, grids)
        # ---- detection: THE MOCK'S OWN RULE (noisy network SNR) --------------
        snr = gmd._network_snr(m1src, m2src, z, dl, rng)
        det = snr >= snr_threshold

        zd = z[det]; rad = ra[det]; decd = dec[det]
        m1d = m1src[det]; qd = q[det]; chid = chi[det]
        p_draw, p_pop, p_unif, p_tgt = mixture_pdraw(
            gmd, mix, zd, rad, decd, m1d, qd, chid, grids, ddldz_grid, pop, pixmap)
        chunks.append({
            "m1det": m1d * (1.0 + zd),
            "m2det": qd * m1d * (1.0 + zd),
            "m1src": m1d,
            "m2src": m2src[det],
            "dL": dl[det],
            "chieff": chid,
            "ra": rad,
            "dec": decd,
            "pdraw": p_draw,
            "z": zd,
            "branch": branch[det].astype(np.float64),
            "pdraw_population": p_pop,
            "pdraw_uniform": p_unif,
            "pdraw_targeted": p_tgt,
        })
        for b in (0, 1, 2):
            inb = branch == b
            n_prop[b] += int(inb.sum())
            n_det[b] += int((inb & det).sum())
        n_proposed += nb
        n_detected += int(det.sum())
        ci += 1
        if verbose:
            print(f"    batch {ci:4d}: proposed={n_proposed:,}/{ndraw:,}  "
                  f"detected={n_detected:,}  "
                  f"(pop={n_det[0]:,} unif={n_det[1]:,} tgt={n_det[2]:,})",
                  flush=True)

    arrays = {k: np.concatenate([c[k] for c in chunks])
              for k in gmd.SELECTION_KEYS + SELECTION_EXTRA_KEYS}
    return {**arrays, "Ndraw": n_proposed, "n_detected": n_detected,
            "n_proposed_branch": n_prop, "n_detected_branch": n_det,
            "n_pixel_fallback": n_pix_fallback}


def neff_population(pdraw):
    """Generator-style (population-only) N_eff: the effective sample size of the
    weights 1/pdraw that estimate the POPULATION selection integral.  This is the
    number gmd prints; it is NOT the number darksirens' catalog-conditioned
    integral sees (see the module docstring)."""
    inv = 1.0 / np.asarray(pdraw, dtype=np.float64)
    if inv.size == 0:
        return 0.0
    return float(inv.sum() ** 2 / np.square(inv).sum())


# =============================================================================
# write
# =============================================================================
def write_file(out_path, sel, args, mix, pixmap, gmd, meta):
    neff_pop = neff_population(sel["pdraw"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        # --- gwcat-selection-1.0 contract ---
        f.attrs["format_version"] = "gwcat-selection-1.0"
        f.attrs["mock_data"] = True
        f.attrs["ndraw"] = int(sel["Ndraw"])       # TOTAL PROPOSED
        f.attrs["Ndraw"] = int(sel["Ndraw"])
        f.attrs["chi_eff_swap_applied"] = True
        f.attrs["chi_eff_amax"] = float(CHIEFF_AMAX)
        f.attrs["cosmology_H0"] = float(args.H0)
        f.attrs["cosmology_Om0"] = float(args.Om0)
        f.attrs["Neff"] = float(neff_pop)          # population-only, informational
        f.attrs["pop_model"] = "powerlaw+peak"
        f.attrs["shared_beta"] = True
        f.attrs["shared_spin"] = True
        f.attrs["shared_gamma"] = True
        # --- proposal / provenance ---
        f.attrs["proposal_mode"] = args.proposal_mode
        f.attrs["selection_proposal"] = (
            f"mixture(population={mix[0]},uniform={mix[1]},"
            f"catalog_targeted={mix[2]})")
        f.attrs["proposal_mix_population"] = float(mix[0])
        f.attrs["proposal_mix_uniform"] = float(mix[1])
        f.attrs["proposal_mix_targeted"] = float(mix[2])
        f.attrs["targeted_sigma_t"] = float(args.sigma_t)
        f.attrs["targeted_sigma_t_rel"] = float(args.sigma_t_rel)
        f.attrs["targeted_catalog"] = pixmap.path
        f.attrs["targeted_catalog_n_hosts"] = int(pixmap.N_hosts)
        f.attrs["targeted_nside"] = int(pixmap.nside)
        f.attrs["targeted_pixarea_sr"] = float(pixmap.apix)
        f.attrs["targeted_max_hosts_per_pixel"] = int(pixmap.maxcol)
        f.attrs["targeted_occupied_pixels"] = int(pixmap.n_occupied)
        f.attrs["detection_rule"] = "gmd._network_snr >= snr_threshold (noisy network SNR)"
        f.attrs["snr_threshold"] = float(args.snr_threshold)
        f.attrs["zmax_proposal"] = float(args.zmax)
        f.attrs["m1det_range_uniform"] = np.asarray(gmd._M1DET_RANGE, dtype=np.float64)
        f.attrs["seed"] = int(args.seed)
        f.attrs["batch_size"] = int(args.batch_size)
        f.attrs["n_detected"] = int(sel["n_detected"])
        for b, name in enumerate(BRANCH_NAMES):
            f.attrs[f"n_proposed_{name}_branch"] = int(sel["n_proposed_branch"][b])
            f.attrs[f"n_detected_{name}_branch"] = int(sel["n_detected_branch"][b])
        f.attrs["n_pixel_placement_fallback"] = int(sel["n_pixel_fallback"])
        f.attrs["generated_at_utc"] = meta["generated_at_utc"]
        f.attrs["generator_script"] = meta["generator_script"]
        f.attrs["darksirens_worktree"] = meta["darksirens_worktree"]
        f.attrs["darksirens_worktree_sha"] = meta["darksirens_worktree_sha"]
        f.attrs["gmd_file"] = meta["gmd_file"]
        f.attrs["gws_agn_repo_head"] = meta["gws_agn_repo_head"]
        f.attrs["metadata_json"] = json.dumps(meta, default=str)
        # --- datasets: gmd.SELECTION_KEYS order, detected rows only ---
        for key in gmd.SELECTION_KEYS:
            f.create_dataset(key, data=np.asarray(sel[key], dtype=np.float64),
                             compression="gzip", shuffle=True)
        for key in SELECTION_EXTRA_KEYS:
            f.create_dataset(key, data=np.asarray(sel[key], dtype=np.float64),
                             compression="gzip", shuffle=True)
    return neff_pop


# =============================================================================
# validation
# =============================================================================
def validate(out_path, gmd, grids, ddldz_grid, pop, pixmap, mix, args, rng):
    """Independent recomputation of the stored mixture density plus bookkeeping.

    The p_targeted arm of the check shares NO code with the write path: it scans
    the FULL catalog for the hosts in each row's pixel (``pix_hosts == pix``)
    instead of using the padded pixel table, and takes z from the stored
    canonical coordinates (z = m1det/m1src - 1, exact since m1det = (1+z) m1src)
    instead of the stored z column."""
    with h5py.File(out_path, "r") as f:
        m1det = np.asarray(f["m1det"]); m1src = np.asarray(f["m1src"])
        m2src = np.asarray(f["m2src"]); dL = np.asarray(f["dL"])
        chieff = np.asarray(f["chieff"]); ra = np.asarray(f["ra"])
        dec = np.asarray(f["dec"]); pdraw = np.asarray(f["pdraw"], dtype=np.float64)
        z = np.asarray(f["z"]); branch = np.asarray(f["branch"]).astype(int)
        p_pop_s = np.asarray(f["pdraw_population"])
        p_unif_s = np.asarray(f["pdraw_uniform"])
        p_tgt_s = np.asarray(f["pdraw_targeted"])
        ndraw = int(f.attrs["ndraw"])
        n_det_attr = int(f.attrs["n_detected"])
        neff_attr = float(f.attrs["Neff"])
    n_det = int(pdraw.size)
    f_pop, f_unif, f_tgt = mix

    # (a) stored total == exact mixture of stored components (bit-level).
    mix_from_components = np.maximum(
        f_pop * p_pop_s + f_unif * p_unif_s + f_tgt * p_tgt_s, PDRAW_FLOOR)
    mixture_exact_bitwise = bool(np.array_equal(pdraw, mix_from_components))

    # (b) full independent recomputation on a random subsample.
    nsub = int(min(args.validate_nsamp, n_det))
    sidx = rng.choice(n_det, size=nsub, replace=False)
    z_coords = m1det[sidx] / m1src[sidx] - 1.0
    q_s = m2src[sidx] / m1src[sidx]
    m1_s = m1src[sidx]; chi_s = chieff[sidx]
    ra_s = ra[sidx]; dec_s = dec[sidx]
    pix_s = pixmap.pix_of(ra_s, dec_s)

    p_pop_re = gmd._selection_pdraw("population", m1_s, q_s, chi_s, z_coords, grids, pop)
    p_unif_re = gmd._selection_pdraw("uniform", m1_s, q_s, chi_s, z_coords, grids, pop)
    p_tgt_re = np.empty(nsub)
    sqrt2pi = np.sqrt(2.0 * np.pi)
    for k in range(nsub):
        zk = z_coords[k]
        in_pix = pixmap.pix_hosts == pix_s[k]           # direct full-catalog scan
        zj = pixmap.z_hosts[in_pix]
        if zj.size:
            sj = np.sqrt(args.sigma_t**2 + (args.sigma_t_rel * zj) ** 2)
            nj = ndtr((args.zmax - zj) / sj) - ndtr((0.0 - zj) / sj)
            ksum = float(np.sum(np.exp(-0.5 * ((zk - zj) / sj) ** 2) / (sj * sqrt2pi) / nj))
        else:
            ksum = 0.0
        p_zsky = ksum / pixmap.N_hosts / pixmap.apix
        ddldz = float(np.interp(zk, grids["z"], ddldz_grid))
        msp = float(gmd._mass_spin_pdf(np.array([m1_s[k]]), np.array([q_s[k]]),
                                       np.array([chi_s[k]]), pop)[0])
        p_tgt_re[k] = msp * p_zsky / max(ddldz * (1.0 + zk), 1.0e-300)
    mix_re = np.maximum(f_pop * p_pop_re + f_unif * p_unif_re + f_tgt * p_tgt_re,
                        PDRAW_FLOOR)
    rel = np.abs(mix_re - pdraw[sidx]) / pdraw[sidx]
    rel_tgt = np.abs(p_tgt_re - p_tgt_s[sidx]) / np.maximum(p_tgt_s[sidx], 1e-300)

    # (c) pixel-placement exactness of the targeted branch.
    tgt_rows = branch == 2
    pix_ok = True
    n_tgt_rows = int(tgt_rows.sum())
    if n_tgt_rows:
        # every targeted row must sit in a pixel that holds >= 1 host
        pix_ok = bool(np.all(pixmap.counts[pixmap.pix_of(ra[tgt_rows], dec[tgt_rows])] > 0))

    # (d) branch bookkeeping / support.
    counts_det = [int((branch == b).sum()) for b in range(3)]
    on_support = p_tgt_s > 0.0

    rec = {
        "file": str(out_path),
        "ndraw_total_proposed": ndraw,
        "n_detected": n_det,
        "n_detected_attr": n_det_attr,
        "frac_detected": n_det / ndraw,
        "detected_by_branch": dict(zip(BRANCH_NAMES, counts_det)),
        "detected_frac_of_rows_by_branch": {
            k: (c / n_det if n_det else 0.0) for k, c in zip(BRANCH_NAMES, counts_det)},
        "pdraw_all_positive": bool(np.all(pdraw > 0.0)),
        "pdraw_all_finite": bool(np.all(np.isfinite(pdraw))),
        "pdraw_min": float(pdraw.min()), "pdraw_max": float(pdraw.max()),
        "n_rows_pdraw_at_floor": int((pdraw <= PDRAW_FLOOR).sum()),
        "mixture_exact_from_stored_components_bitwise": mixture_exact_bitwise,
        "pdraw_recompute_nsamp": nsub,
        "pdraw_recompute_max_rel_err": float(rel.max()),
        "pdraw_recompute_median_rel_err": float(np.median(rel)),
        "p_targeted_recompute_max_rel_err": float(rel_tgt.max()),
        "n_rows_p_targeted_positive": int(on_support.sum()),
        "frac_rows_on_catalog_support": float(on_support.mean()),
        "frac_rows_on_catalog_support_by_branch": {
            k: (float(on_support[branch == b].mean()) if counts_det[b] else 0.0)
            for b, k in enumerate(BRANCH_NAMES)},
        "targeted_rows_all_in_occupied_pixel": pix_ok,
        "z_min": float(z.min()), "z_max": float(z.max()),
        "dL_min": float(dL.min()), "dL_max": float(dL.max()),
        "Neff_population_only": neff_population(pdraw),
        "Neff_population_only_attr": neff_attr,
        "size_mb": out_path.stat().st_size / 1e6,
    }
    return rec


# =============================================================================
# main
# =============================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_path", required=True)
    p.add_argument("--proposal_mode", default="catalog_targeted",
                   choices=["catalog_targeted", "popuni"],
                   help="catalog_targeted (default): 3-branch mixture including the "
                        "catalog-targeted branch. popuni: the same generator with the "
                        "targeted weight folded into the population branch (an A/B "
                        "control that isolates the targeting).")
    p.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    p.add_argument("--worktree", default=DEFAULT_WORKTREE,
                   help="Pinned darksirens checkout supplying generate_mock_data.py.")
    p.add_argument("--ndraw", type=int, required=True, help="TOTAL proposals.")
    p.add_argument("--seed", type=int, default=71001)
    p.add_argument("--batch_size", type=int, default=1_000_000)
    p.add_argument("--mix_population", type=float, default=MIX_POPULATION)
    p.add_argument("--mix_uniform", type=float, default=MIX_UNIFORM)
    p.add_argument("--mix_targeted", type=float, default=MIX_TARGETED)
    p.add_argument("--sigma_t", type=float, default=SIGMA_T)
    p.add_argument("--sigma_t_rel", type=float, default=SIGMA_T_REL)
    p.add_argument("--snr_threshold", type=float, default=SNR_THRESHOLD)
    p.add_argument("--zmax", type=float, default=ZMAX)
    p.add_argument("--nside", type=int, default=NSIDE)
    p.add_argument("--H0", type=float, default=H0_FID)
    p.add_argument("--Om0", type=float, default=OM0_FID)
    p.add_argument("--validate_nsamp", type=int, default=300)
    p.add_argument("--validation_json", default=None)
    p.add_argument("--no_validate", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    gmd_dir = Path(args.worktree) / "scripts/mock_dark_sirens"
    if not (gmd_dir / "generate_mock_data.py").exists():
        raise SystemExit(f"generate_mock_data.py not found under {gmd_dir}")
    sys.path.insert(0, str(gmd_dir))
    import generate_mock_data as gmd  # noqa: E402

    if args.proposal_mode == "popuni":
        mix = (args.mix_population + args.mix_targeted, args.mix_uniform, 0.0)
    else:
        mix = (args.mix_population, args.mix_uniform, args.mix_targeted)
    if abs(sum(mix) - 1.0) > 1e-12:
        raise SystemExit(f"mixture weights must sum to 1; got {mix} -> {sum(mix)}")

    cosmo = gmd._build_cosmology(args.H0, args.Om0, W0_FID, WA_FID)
    grids = gmd._cosmology_grids(cosmo, zmax=args.zmax)
    ddldz_grid = np.gradient(grids["dl"], grids["z"])   # same array gmd's pdraw uses
    pop = gmd.PopulationConfig()

    pixmap = CatalogPixelMap(args.catalog, args.nside, args.sigma_t,
                             args.sigma_t_rel, args.zmax)

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator_script": str(Path(__file__).resolve()),
        "darksirens_worktree": str(Path(args.worktree)),
        "darksirens_worktree_sha": _git(args.worktree, "rev-parse", "HEAD"),
        "gmd_file": gmd.__file__,
        "gws_agn_repo_head": _git(EXP_ROOT, "rev-parse", "HEAD"),
        "proposal_mode": args.proposal_mode,
        "mixture": {"population": mix[0], "uniform": mix[1], "targeted": mix[2]},
        "sigma_t": args.sigma_t, "sigma_t_rel": args.sigma_t_rel,
        "snr_threshold": args.snr_threshold, "zmax": args.zmax,
        "nside": args.nside, "seed": args.seed, "ndraw_requested": args.ndraw,
        "catalog": pixmap.path, "catalog_n_hosts": pixmap.N_hosts,
        "cosmology": {"H0": args.H0, "Om0": args.Om0, "w0": W0_FID, "wa": WA_FID},
        "population": {k: getattr(pop, k) for k in pop.__dataclass_fields__},
    }

    print("=" * 92)
    print("CATALOG-TARGETED SELECTION INJECTIONS  (experiment_matched_mock deep mock)")
    print(f"  darksirens worktree : {meta['darksirens_worktree']} @ "
          f"{meta['darksirens_worktree_sha'][:10]}")
    print(f"  gmd                 : {gmd.__file__}")
    print(f"  catalog             : {pixmap.path}")
    print(f"    N_hosts={pixmap.N_hosts:,}  nside={pixmap.nside}  "
          f"occupied={pixmap.n_occupied:,}/{pixmap.npix:,}  "
          f"max hosts/pix={pixmap.maxcol}  Omega_pix={pixmap.apix:.6e} sr")
    print(f"  mixture             : pop={mix[0]}  unif={mix[1]}  targeted={mix[2]}")
    print(f"  sigma_t             : {args.sigma_t}  (rel={args.sigma_t_rel})")
    print(f"  detection           : gmd._network_snr >= {args.snr_threshold}")
    print(f"  zmax / cosmology    : {args.zmax}  H0={args.H0} Om0={args.Om0}")
    print(f"  ndraw (proposals)   : {args.ndraw:,}  seed={args.seed}  "
          f"batch={args.batch_size:,}")
    print("=" * 92, flush=True)

    sel = draw_injections(gmd, args.ndraw, args.seed, grids, ddldz_grid, pop,
                          pixmap, mix, args.snr_threshold, args.batch_size,
                          args.zmax)
    out_path = Path(args.out_path)
    neff_pop = write_file(out_path, sel, args, mix, pixmap, gmd, meta)
    print(f"\nwrote {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")
    print(f"  proposed={sel['Ndraw']:,}  detected={sel['n_detected']:,}  "
          f"frac={sel['n_detected'] / sel['Ndraw']:.6e}")
    print(f"  proposed by branch={sel['n_proposed_branch']}  "
          f"detected by branch={sel['n_detected_branch']}")
    print(f"  pixel-placement fallbacks={sel['n_pixel_fallback']}")
    print(f"  N_eff (population-only, generator style) = {neff_pop:.1f}")

    if args.no_validate:
        return 0
    rng = np.random.default_rng(args.seed + 900_001)
    rec = validate(out_path, gmd, grids, ddldz_grid, pop, pixmap, mix, args, rng)
    rec["meta"] = meta
    print("\n--- validation ---")
    for k, v in rec.items():
        if k == "meta":
            continue
        print(f"  {k}: {v}")
    if args.validation_json:
        Path(args.validation_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.validation_json).write_text(json.dumps(rec, indent=2, default=str))
        print(f"\nwrote {args.validation_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
