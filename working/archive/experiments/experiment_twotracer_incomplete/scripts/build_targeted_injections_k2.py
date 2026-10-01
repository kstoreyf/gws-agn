#!/usr/bin/env python3
"""Catalog-TARGETED selection injections for the K=2 deep two-tracer mock.

WHY
---
Under ``--catalog_sky_weighting field`` the selection integral mu(theta) is
CATALOG-CONDITIONED: an injection carries weight only if its redshift lands
within a few ``sig_i = sqrt(dzgals^2 + sigma_kde^2)`` of an actual catalog host
IN ITS OWN HEALPix pixel.  For a K=2 mixture the target is a fcat-weighted
combination of the per-tracer conditioned densities, so as f_AGN rises the
integral leans on the SPARSE tracer and N_eff collapses.  Measured on this mock
with gmd's ``population+uniform`` proposal (120,000,000 proposed):

    N_obs = 200  ->  catalog-conditioned N_eff = 497   (needs > 5*N_obs = 1000)

which rejects every high-f cell and rails the posterior against the guard
boundary (see ``../RESULTS.md``, population+uniform run).

Coarsening the pixelisation is the remedy that works for a DENSE catalog, and it
is NOT available here: it washes out the number-density contrast that identifies
f_AGN in the first place.  The proposal has to be fixed instead.

WHAT THIS ADDS OVER THE K=1 GENERATOR
-------------------------------------
This is the two-tracer generalisation of
``../../experiment_matched_mock/scripts/build_injections_targeted.py``.  Two
changes:

1. **Any number of targeted branches**, one per tracer, with independent
   mixture weights (``--target survey.h5:weight``).  The default plants the
   whole targeted weight on the sparse AGN tracer, which is the component that
   starves; a GAL arm can be added without touching the code.
2. **The targeted branch reads the PIXELATED SURVEY file, not the raw catalog.**
   That file *is* the object the likelihood conditions on -- its ``zgals`` /
   ``dzgals`` are literally the KDE centres and widths the target density uses
   -- so the proposal is guaranteed to cover exactly the support that carries
   weight, with no possibility of a pixelisation or kernel-width mismatch
   between generator and inference.  (The K=1 generator read the raw catalog and
   re-derived both, which is correct only as long as the two stay in step.)

PROPOSAL
--------
(2 + T)-branch mixture over population, uniform and one branch per targeted
tracer.  Every row's stored ``pdraw`` is the EXACT mixture density at that row's
coordinates, whichever branch produced it:

    pdraw = f_pop * p_population + f_unif * p_uniform + sum_t f_t * p_targeted_t

* ``p_population`` / ``p_uniform`` are ``gmd._selection_pdraw(...)`` -- the same
  densities gmd's own generator stores, so those two branches are byte-for-byte
  the objects gmd would have drawn.
* targeted branch t draws
    - host j uniformly from tracer t's hosts,
    - z ~ Normal(z_j, dzgals_j) truncated to (0, zmax) (renormalised, exact),
    - sky uniform WITHIN host j's pixel (bounding-box rejection),
    - m1src / q / chieff from the fiducial population samplers.

TARGETED-BRANCH DENSITY
-----------------------
The sky draw is a DISCRETE pixel choice (host j fixes the pixel) followed by a
uniform draw inside that equal-area pixel, so only hosts sharing the query
point's pixel contribute:

    q_t(m1src,q,chi, z,Omega)
        = p_ms(m1src,q,chi) * (1/Omega_pix)
              * sum_{j : pix_j = pix(Omega)} (1/N_t) TN(z; z_j, sig_j),
    TN(z; z_j, s) = phi((z - z_j)/s) / s / [Phi((zmax - z_j)/s) - Phi(-z_j/s)].

Normalisation is exact: the Omega integral gives Omega_pix/Omega_pix = 1 per
host and TN integrates to 1 on the sampled support, so sum_j (1/N_t) = 1.

Mapping to the canonical basis (m1det, q, chi, dL) at fixed (q, chi, Omega) has
Jacobian determinant (1+z) dL'(z), hence

    p_targeted_t = q_t / [ (1+z) * dL'(z) ]

which is structurally IDENTICAL to gmd's ``_p_population`` -- same
``_mass_spin_pdf``, same ``np.gradient(dl, z)``, same ``(1+z)`` -- with
``p_z(z)/(4 pi)`` replaced by the catalog z-sky factor.  All branch densities
are therefore in one measure and mix directly.

(darksirens' own target uses a DISCRETE pixel probability rather than a
per-steradian density, i.e. differs from this convention by one global factor
Omega_pix.  HEALPix pixels are equal-area so that factor is row-independent and
cancels out of N_eff and every parameter-dependent part of logL -- exactly as it
does for gmd's own injection files, whose convention this matches.  All targeted
tracers must therefore share one nside, which is checked.)

DETECTION
---------
``gmd._network_snr(...) >= snr_threshold`` -- the SAME noisy network SNR, with
the same Beta(2,5)**0.5 projection draw, that generated the events via
``gmd._draw_events_until_detected``.  If the detection rule here were not the
events' own rule the selection integral would be inconsistent with the data.

OUTPUT
------
``gwcat-selection-1.0``, DETECTED ROWS ONLY, datasets = ``gmd.SELECTION_KEYS``
plus auditability extras (``z``, ``branch``, ``pdraw_population``,
``pdraw_uniform``, ``pdraw_targeted_<tracer>``; the loader ignores unknown
datasets).  ``ndraw`` / ``Ndraw`` attrs are the TOTAL PROPOSED, not detected.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import healpy as hp
import numpy as np
from scipy.special import ndtr  # standard-normal CDF Phi(x), vectorised

EXP_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_WORKTREE = "/hildafs/projects/phy230014p/magana/src/darksirens-pefix"

# --- fiducials of the deep two-tracer mock (data_derived/twotracer_meta.json) --
H0_FID = 67.74
OM0_FID = 0.3075
W0_FID = -1.0
WA_FID = 0.0
ZMAX = 2.0
SNR_THRESHOLD = 8.0
CHIEFF_AMAX = 0.99
PDRAW_FLOOR = 1.0e-300     # gmd._selection_pdraw's np.maximum floor, retained

# --- proposal defaults: the whole targeted weight on the sparse tracer --------
MIX_POPULATION = 0.65
MIX_UNIFORM = 0.10
# remaining 0.25 is split across --target entries by their declared weights


# =============================================================================
# provenance
# =============================================================================
def _git(repo, *args):
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception as exc:  # pragma: no cover - provenance best-effort
        return f"<unavailable: {exc}>"


# =============================================================================
# tracer pixel map, built from the PIXELATED SURVEY FILE
# =============================================================================
class SurveyPixelMap:
    """The tracer's KDE support, read from the file the likelihood conditions on.

    ``zgals``/``dzgals`` are (npix, maxcol) padded tables; row ``p`` holds the
    ``ngals[p]`` hosts of pixel ``p`` in its leading columns (gmd
    ``_pixelate_catalog`` pads z with 100.0, dz with 1.0, w with 0.0).  That
    layout is already exactly the padded lookup the kernel sum

        S(z, pix) = sum_{j in pix} TN(z; z_j, sig_j)

    needs, so no re-derivation from a raw catalog is involved anywhere.
    """

    def __init__(self, path, zmax, name=None):
        with h5py.File(path, "r") as f:
            zgals = np.asarray(f["zgals"][:], dtype=np.float64)
            dzgals = np.asarray(f["dzgals"][:], dtype=np.float64)
            wgals = np.asarray(f["wgals"][:], dtype=np.float64)
            ngals = np.asarray(f["ngals"][:], dtype=np.int64)
            nside = int(f.attrs["nside"])
            tracer = f.attrs.get("tracer", name or Path(path).stem)
        npix, maxcol = zgals.shape
        if npix != hp.nside2npix(nside):
            raise SystemExit(f"{path}: npix {npix} inconsistent with nside {nside}")
        valid = np.arange(maxcol)[None, :] < ngals[:, None]

        # Uniform host choice is the correct weighted choice only if the catalog
        # weights are equal; refuse rather than silently mis-specify the density.
        w = wgals[valid]
        if w.size and not np.allclose(w, w.flat[0], rtol=0, atol=0):
            raise SystemExit(
                f"{path}: catalog weights are not uniform; the targeted branch "
                "draws hosts uniformly and its density assumes equal weights")

        self.path = str(path)
        self.name = str(tracer)
        self.nside = nside
        self.npix = npix
        self.maxcol = int(maxcol)
        self.apix = float(hp.nside2pixarea(nside))
        self.resol = float(hp.nside2resol(nside))
        self.zmax = float(zmax)
        self.counts = ngals
        self.N_hosts = int(ngals.sum())
        self.n_occupied = int((ngals > 0).sum())
        self.empty_pixel_fraction = float(1.0 - (ngals > 0).mean())

        self.finite_pad = valid
        self.z_pad = np.where(valid, zgals, 0.0)
        self.sig_pad = np.where(valid, dzgals, 1.0)
        self.tnorm_pad = np.where(
            valid,
            ndtr((self.zmax - self.z_pad) / self.sig_pad)
            - ndtr((0.0 - self.z_pad) / self.sig_pad),
            1.0)

        # Flat per-host arrays for the draw.  Boolean-mask extraction is
        # row-major, i.e. pixel order, matching the repeat() below.
        self.z_hosts = zgals[valid]
        self.sig_hosts = dzgals[valid]
        self.pix_hosts = np.repeat(np.arange(npix), ngals)
        if self.z_hosts.size != self.N_hosts:
            raise SystemExit(f"{path}: host bookkeeping mismatch")

    def pix_of(self, ra, dec):
        return hp.ang2pix(self.nside, 0.5 * np.pi - np.asarray(dec), np.asarray(ra))

    def kernel_sum(self, z_rows, pix_rows):
        """sum_{j in pix} TN(z; z_j, sig_j) per row (0 where the pixel is empty)."""
        z_rows = np.asarray(z_rows, dtype=np.float64)
        if z_rows.size == 0:
            return np.zeros(0)
        rows = np.asarray(pix_rows)
        zmat = self.z_pad[rows]
        sig = self.sig_pad[rows]
        tn = self.tnorm_pad[rows]
        fin = self.finite_pad[rows]
        x = (z_rows[:, None] - zmat) / sig
        pdf = np.exp(-0.5 * x * x) / (sig * np.sqrt(2.0 * np.pi))
        return np.where(fin, pdf / tn, 0.0).sum(axis=1)


def sample_truncated_normal_at_hosts(rng, z_j, sig_j, zmax):
    """z ~ Normal(z_j, sig_j) truncated to (0, zmax), by rejection.

    Rejection resamples the SAME host, so the accepted draw is exactly the
    renormalised truncated normal whose density ``kernel_sum`` evaluates."""
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

    Bounding-box proposal (uniform in cos(theta) x uniform in phi about the
    pixel centre) rejected on ``ang2pix == pix_tgt``.  Rejection of a uniform
    proposal restricted to a superset of the pixel is exactly uniform on the
    pixel, which is what makes the targeted sky density the constant
    1/Omega_pix inside it.  A nonzero fallback count means some row was placed
    at its pixel centre (right pixel, not uniform within it) and is reported."""
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
def p_targeted_density(gmd, z, ra, dec, m1src, q, chi, grids, ddldz_grid, pop, pmap):
    """One targeted branch's proposal density in the canonical basis, per sr."""
    ksum = pmap.kernel_sum(z, pmap.pix_of(ra, dec))
    p_zsky = ksum / pmap.N_hosts / pmap.apix
    ddldz = np.interp(z, grids["z"], ddldz_grid)
    jac = ddldz * (1.0 + z)
    msp = gmd._mass_spin_pdf(m1src, q, chi, pop)
    return msp * p_zsky / np.maximum(jac, 1.0e-300)


def mixture_pdraw(gmd, f_pop, f_unif, tgt_weights, pmaps,
                  z, ra, dec, m1src, q, chi, grids, ddldz_grid, pop):
    """Exact mixture density and its per-branch components (floored like gmd)."""
    p_pop = gmd._selection_pdraw("population", m1src, q, chi, z, grids, pop)
    p_unif = gmd._selection_pdraw("uniform", m1src, q, chi, z, grids, pop)
    total = f_pop * p_pop + f_unif * p_unif
    p_tgts = []
    for w, pmap in zip(tgt_weights, pmaps):
        p_t = (p_targeted_density(gmd, z, ra, dec, m1src, q, chi,
                                  grids, ddldz_grid, pop, pmap)
               if w > 0.0 else np.zeros_like(p_pop))
        p_tgts.append(p_t)
        total = total + w * p_t
    return np.maximum(total, PDRAW_FLOOR), p_pop, p_unif, p_tgts


# =============================================================================
# draw
# =============================================================================
def draw_injections(gmd, ndraw, seed, grids, ddldz_grid, pop, pmaps,
                    f_pop, f_unif, tgt_weights, snr_threshold, batch_size,
                    zmax, verbose=True):
    """Draw the mixture over batches; keep detected rows only.

    Per batch: one uniform picks each row's branch, then the population,
    uniform and targeted draws are made, then ONE ``gmd._network_snr`` call
    over the whole batch -- so every proposal gets one independent
    Beta(2,5)**0.5 projection draw exactly as in
    ``gmd._draw_events_until_detected`` / ``gmd._draw_selection_batch``."""
    rng = np.random.default_rng(seed)
    m1lo, m1hi = gmd._M1DET_RANGE
    n_tgt = len(pmaps)
    # branch codes: 0 population, 1 uniform, 2 + t targeted tracer t
    edges = np.cumsum([f_pop, f_unif, *tgt_weights])
    chunks = []
    n_proposed = n_detected = 0
    n_branches = 2 + n_tgt
    n_prop = [0] * n_branches
    n_det = [0] * n_branches
    n_pix_fallback = [0] * n_tgt
    ci = 0
    while n_proposed < ndraw:
        nb = int(min(batch_size, ndraw - n_proposed))
        bu = rng.uniform(size=nb)
        # np.searchsorted on the cumulative weights assigns each row its branch.
        branch = np.searchsorted(edges, bu, side="right").astype(np.int8)
        np.clip(branch, 0, n_branches - 1, out=branch)

        z = np.empty(nb)
        ra = np.empty(nb)
        dec = np.empty(nb)
        m1src = np.empty(nb)
        q = np.empty(nb)
        chi = np.empty(nb)

        is_pop = branch == 0
        npop = int(is_pop.sum())
        if npop:
            zc = gmd._sample_uniform_comoving_z(rng, grids, npop)
            rac, decc = gmd._sample_sky(rng, npop)
            m1c, use_peak = gmd._sample_powerlaw_peak_m1(rng, npop, pop,
                                                         return_component=True)
            qc = gmd._sample_q(rng, m1c, pop, use_peak=use_peak)
            chic = gmd._sample_chieff(rng, npop, pop)
            z[is_pop] = zc
            ra[is_pop] = rac
            dec[is_pop] = decc
            m1src[is_pop] = m1c
            q[is_pop] = qc
            chi[is_pop] = chic

        is_unif = branch == 1
        nunif = int(is_unif.sum())
        if nunif:
            zc = gmd._sample_uniform_comoving_z(rng, grids, nunif)
            rac, decc = gmd._sample_sky(rng, nunif)
            m1det_u = rng.uniform(m1lo, m1hi, nunif)
            z[is_unif] = zc
            ra[is_unif] = rac
            dec[is_unif] = decc
            m1src[is_unif] = m1det_u / (1.0 + zc)
            q[is_unif] = rng.uniform(0.0, 1.0, nunif)
            chi[is_unif] = rng.uniform(-1.0, 1.0, nunif)

        for t, pmap in enumerate(pmaps):
            is_t = branch == (2 + t)
            nt = int(is_t.sum())
            if not nt:
                continue
            j = rng.integers(0, pmap.N_hosts, nt)
            zc = sample_truncated_normal_at_hosts(
                rng, pmap.z_hosts[j], pmap.sig_hosts[j], zmax)
            rac, decc, nfb = sample_uniform_in_pixels(
                rng, pmap.pix_hosts[j], pmap.nside, pmap.resol)
            n_pix_fallback[t] += nfb
            m1c, use_peak = gmd._sample_powerlaw_peak_m1(rng, nt, pop,
                                                         return_component=True)
            z[is_t] = zc
            ra[is_t] = rac
            dec[is_t] = decc
            m1src[is_t] = m1c
            q[is_t] = gmd._sample_q(rng, m1c, pop, use_peak=use_peak)
            chi[is_t] = gmd._sample_chieff(rng, nt, pop)

        m2src = q * m1src
        dl = gmd._interp_dl(z, grids)
        # ---- detection: THE MOCK'S OWN RULE (noisy network SNR) --------------
        snr = gmd._network_snr(m1src, m2src, z, dl, rng)
        det = snr >= snr_threshold

        zd, rad, decd = z[det], ra[det], dec[det]
        m1d, qd, chid = m1src[det], q[det], chi[det]
        p_draw, p_pop, p_unif, p_tgts = mixture_pdraw(
            gmd, f_pop, f_unif, tgt_weights, pmaps,
            zd, rad, decd, m1d, qd, chid, grids, ddldz_grid, pop)
        chunk = {
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
        }
        for t, pmap in enumerate(pmaps):
            chunk[f"pdraw_targeted_{pmap.name}"] = p_tgts[t]
        chunks.append(chunk)

        for b in range(n_branches):
            inb = branch == b
            n_prop[b] += int(inb.sum())
            n_det[b] += int((inb & det).sum())
        n_proposed += nb
        n_detected += int(det.sum())
        ci += 1
        if verbose:
            print(f"    batch {ci:4d}: proposed={n_proposed:,}/{ndraw:,}  "
                  f"detected={n_detected:,}  by-branch={n_det}", flush=True)

    keys = list(chunks[0].keys())
    arrays = {k: np.concatenate([c[k] for c in chunks]) for k in keys}
    return {**arrays, "Ndraw": n_proposed, "n_detected": n_detected,
            "n_proposed_branch": n_prop, "n_detected_branch": n_det,
            "n_pixel_fallback": n_pix_fallback}


def neff_population(pdraw):
    """Generator-style (population-only) N_eff of the weights 1/pdraw.

    This is the number gmd prints.  It is NOT the catalog-conditioned N_eff the
    inference's selection integral sees -- for this mock the two differ by
    almost two orders of magnitude, which is exactly why this file exists."""
    inv = 1.0 / np.asarray(pdraw, dtype=np.float64)
    if inv.size == 0:
        return 0.0
    return float(inv.sum() ** 2 / np.square(inv).sum())


# =============================================================================
# write
# =============================================================================
def write_file(out_path, sel, args, f_pop, f_unif, tgt_weights, pmaps, gmd, meta):
    neff_pop = neff_population(sel["pdraw"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    branch_names = ["population", "uniform"] + [f"targeted_{p.name}" for p in pmaps]
    extra_keys = (["z", "branch", "pdraw_population", "pdraw_uniform"]
                  + [f"pdraw_targeted_{p.name}" for p in pmaps])
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
        mix_desc = ",".join([f"population={f_pop}", f"uniform={f_unif}"]
                            + [f"targeted_{p.name}={w}"
                               for p, w in zip(pmaps, tgt_weights)])
        f.attrs["selection_proposal"] = f"mixture({mix_desc})"
        f.attrs["proposal_mix_population"] = float(f_pop)
        f.attrs["proposal_mix_uniform"] = float(f_unif)
        for p, w in zip(pmaps, tgt_weights):
            f.attrs[f"proposal_mix_targeted_{p.name}"] = float(w)
            f.attrs[f"targeted_{p.name}_survey"] = p.path
            f.attrs[f"targeted_{p.name}_n_hosts"] = int(p.N_hosts)
            f.attrs[f"targeted_{p.name}_empty_pixel_fraction"] = p.empty_pixel_fraction
        f.attrs["targeted_nside"] = int(pmaps[0].nside) if pmaps else 0
        f.attrs["targeted_pixarea_sr"] = float(pmaps[0].apix) if pmaps else 0.0
        f.attrs["detection_rule"] = ("gmd._network_snr >= snr_threshold "
                                     "(noisy network SNR)")
        f.attrs["snr_threshold"] = float(args.snr_threshold)
        f.attrs["zmax_proposal"] = float(args.zmax)
        f.attrs["m1det_range_uniform"] = np.asarray(gmd._M1DET_RANGE, dtype=np.float64)
        f.attrs["seed"] = int(args.seed)
        f.attrs["batch_size"] = int(args.batch_size)
        f.attrs["n_detected"] = int(sel["n_detected"])
        for b, name in enumerate(branch_names):
            f.attrs[f"n_proposed_{name}_branch"] = int(sel["n_proposed_branch"][b])
            f.attrs[f"n_detected_{name}_branch"] = int(sel["n_detected_branch"][b])
        for p, nfb in zip(pmaps, sel["n_pixel_fallback"]):
            f.attrs[f"n_pixel_placement_fallback_{p.name}"] = int(nfb)
        for k, v in meta.items():
            if isinstance(v, str):
                f.attrs[k] = v
        f.attrs["metadata_json"] = json.dumps(meta, default=str)
        # --- datasets: gmd.SELECTION_KEYS order, detected rows only ---
        for key in gmd.SELECTION_KEYS + extra_keys:
            f.create_dataset(key, data=np.asarray(sel[key], dtype=np.float64),
                             compression="gzip", shuffle=True)
    return neff_pop, branch_names


# =============================================================================
# validation
# =============================================================================
def validate(out_path, gmd, grids, ddldz_grid, pop, pmaps, f_pop, f_unif,
             tgt_weights, args, rng, branch_names):
    """Independent recomputation of the stored mixture density plus bookkeeping.

    The targeted arm of the check shares NO code with the write path: it scans
    the FLAT host arrays for the hosts in each row's pixel instead of using the
    padded pixel table, and takes z from the stored canonical coordinates
    (z = m1det/m1src - 1, exact since m1det = (1+z) m1src) instead of the
    stored z column."""
    with h5py.File(out_path, "r") as f:
        m1det = np.asarray(f["m1det"])
        m1src = np.asarray(f["m1src"])
        m2src = np.asarray(f["m2src"])
        dL = np.asarray(f["dL"])
        chieff = np.asarray(f["chieff"])
        ra = np.asarray(f["ra"])
        dec = np.asarray(f["dec"])
        pdraw = np.asarray(f["pdraw"], dtype=np.float64)
        z = np.asarray(f["z"])
        branch = np.asarray(f["branch"]).astype(int)
        p_pop_s = np.asarray(f["pdraw_population"])
        p_unif_s = np.asarray(f["pdraw_uniform"])
        p_tgt_s = [np.asarray(f[f"pdraw_targeted_{p.name}"]) for p in pmaps]
        ndraw = int(f.attrs["ndraw"])
        n_det_attr = int(f.attrs["n_detected"])
        neff_attr = float(f.attrs["Neff"])
    n_det = int(pdraw.size)

    # (a) stored total == exact mixture of stored components (bit-level).
    total = f_pop * p_pop_s + f_unif * p_unif_s
    for w, pt in zip(tgt_weights, p_tgt_s):
        total = total + w * pt
    mixture_exact_bitwise = bool(np.array_equal(pdraw, np.maximum(total, PDRAW_FLOOR)))

    # (b) full independent recomputation on a random subsample.
    nsub = int(min(args.validate_nsamp, n_det))
    sidx = rng.choice(n_det, size=nsub, replace=False)
    z_coords = m1det[sidx] / m1src[sidx] - 1.0
    q_s = m2src[sidx] / m1src[sidx]
    m1_s = m1src[sidx]
    chi_s = chieff[sidx]

    p_pop_re = gmd._selection_pdraw("population", m1_s, q_s, chi_s, z_coords, grids, pop)
    p_unif_re = gmd._selection_pdraw("uniform", m1_s, q_s, chi_s, z_coords, grids, pop)
    mix_re = f_pop * p_pop_re + f_unif * p_unif_re
    sqrt2pi = np.sqrt(2.0 * np.pi)
    rel_tgt_max = {}
    for w, pmap, pt in zip(tgt_weights, pmaps, p_tgt_s):
        pix_s = pmap.pix_of(ra[sidx], dec[sidx])
        p_re = np.empty(nsub)
        for k in range(nsub):
            zk = z_coords[k]
            in_pix = pmap.pix_hosts == pix_s[k]      # direct flat-array scan
            zj = pmap.z_hosts[in_pix]
            if zj.size:
                sj = pmap.sig_hosts[in_pix]
                nj = ndtr((args.zmax - zj) / sj) - ndtr((0.0 - zj) / sj)
                ksum = float(np.sum(np.exp(-0.5 * ((zk - zj) / sj) ** 2)
                                    / (sj * sqrt2pi) / nj))
            else:
                ksum = 0.0
            ddldz = float(np.interp(zk, grids["z"], ddldz_grid))
            msp = float(gmd._mass_spin_pdf(np.array([m1_s[k]]), np.array([q_s[k]]),
                                           np.array([chi_s[k]]), pop)[0])
            p_re[k] = msp * (ksum / pmap.N_hosts / pmap.apix) / max(
                ddldz * (1.0 + zk), 1.0e-300)
        mix_re = mix_re + w * p_re
        rel_tgt_max[pmap.name] = float(np.abs(p_re - pt[sidx]).max()
                                       / max(pt[sidx].max(), 1e-300))
    mix_re = np.maximum(mix_re, PDRAW_FLOOR)
    rel = np.abs(mix_re - pdraw[sidx]) / pdraw[sidx]

    # (c) targeted rows must land in a pixel that holds >= 1 host of their tracer.
    pix_ok = {}
    for t, pmap in enumerate(pmaps):
        rows = branch == (2 + t)
        pix_ok[pmap.name] = (bool(np.all(pmap.counts[pmap.pix_of(ra[rows], dec[rows])] > 0))
                             if rows.any() else True)

    counts_det = [int((branch == b).sum()) for b in range(len(branch_names))]
    support = {p.name: (pt > 0.0) for p, pt in zip(pmaps, p_tgt_s)}

    return {
        "file": str(out_path),
        "ndraw_total_proposed": ndraw,
        "n_detected": n_det,
        "n_detected_attr": n_det_attr,
        "frac_detected": n_det / ndraw,
        "detected_by_branch": dict(zip(branch_names, counts_det)),
        "pdraw_all_positive": bool(np.all(pdraw > 0.0)),
        "pdraw_all_finite": bool(np.all(np.isfinite(pdraw))),
        "pdraw_min": float(pdraw.min()), "pdraw_max": float(pdraw.max()),
        "n_rows_pdraw_at_floor": int((pdraw <= PDRAW_FLOOR).sum()),
        "mixture_exact_from_stored_components_bitwise": mixture_exact_bitwise,
        "pdraw_recompute_nsamp": nsub,
        "pdraw_recompute_max_rel_err": float(rel.max()),
        "pdraw_recompute_median_rel_err": float(np.median(rel)),
        "p_targeted_recompute_max_abs_err_over_scale": rel_tgt_max,
        "frac_rows_on_catalog_support": {k: float(v.mean()) for k, v in support.items()},
        "frac_rows_on_catalog_support_by_branch": {
            k: {bn: (float(v[branch == b].mean()) if counts_det[b] else 0.0)
                for b, bn in enumerate(branch_names)}
            for k, v in support.items()},
        "targeted_rows_all_in_occupied_pixel": pix_ok,
        "z_min": float(z.min()), "z_max": float(z.max()),
        "dL_min": float(dL.min()), "dL_max": float(dL.max()),
        "Neff_population_only": neff_population(pdraw),
        "Neff_population_only_attr": neff_attr,
        "size_mb": out_path.stat().st_size / 1e6,
    }


# =============================================================================
# main
# =============================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_path", required=True)
    p.add_argument("--target", action="append", default=None, metavar="SURVEY:WEIGHT",
                   help="Pixelated survey file to target and its mixture weight, "
                        "e.g. data_derived/survey_agn_ns32.h5:0.25. Repeatable. "
                        "Default: the AGN survey at 0.25.")
    p.add_argument("--mix_population", type=float, default=MIX_POPULATION)
    p.add_argument("--mix_uniform", type=float, default=MIX_UNIFORM)
    p.add_argument("--worktree", default=DEFAULT_WORKTREE,
                   help="Pinned darksirens checkout supplying generate_mock_data.py.")
    p.add_argument("--ndraw", type=int, required=True, help="TOTAL proposals.")
    p.add_argument("--seed", type=int, default=73101)
    p.add_argument("--batch_size", type=int, default=2_000_000)
    p.add_argument("--snr_threshold", type=float, default=SNR_THRESHOLD)
    p.add_argument("--zmax", type=float, default=ZMAX)
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

    targets = args.target or [str(EXP_ROOT / "data_derived/survey_agn_ns32.h5") + ":0.25"]
    pmaps, tgt_weights = [], []
    for spec in targets:
        path, _, w = spec.rpartition(":")
        if not path:
            raise SystemExit(f"--target must be SURVEY:WEIGHT, got {spec!r}")
        pmaps.append(SurveyPixelMap(path, args.zmax))
        tgt_weights.append(float(w))
    if len({p.nside for p in pmaps}) > 1:
        raise SystemExit("all targeted tracers must share one nside (the Omega_pix "
                         "convention factor must be row-independent)")
    if len({p.name for p in pmaps}) != len(pmaps):
        raise SystemExit("targeted tracer names must be distinct (they name datasets)")

    f_pop, f_unif = args.mix_population, args.mix_uniform
    total_w = f_pop + f_unif + sum(tgt_weights)
    if abs(total_w - 1.0) > 1e-12:
        raise SystemExit(f"mixture weights must sum to 1; got {total_w}")

    cosmo = gmd._build_cosmology(args.H0, args.Om0, W0_FID, WA_FID)
    grids = gmd._cosmology_grids(cosmo, zmax=args.zmax)
    ddldz_grid = np.gradient(grids["dl"], grids["z"])   # same array gmd's pdraw uses
    pop = gmd.PopulationConfig()

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator_script": str(Path(__file__).resolve()),
        "darksirens_worktree": str(Path(args.worktree)),
        "darksirens_worktree_sha": _git(args.worktree, "rev-parse", "HEAD"),
        "gmd_file": gmd.__file__,
        "gws_agn_repo_head": _git(EXP_ROOT, "rev-parse", "HEAD"),
        "mixture": {"population": f_pop, "uniform": f_unif,
                    **{f"targeted_{p.name}": w for p, w in zip(pmaps, tgt_weights)}},
        "targets": [{"survey": p.path, "tracer": p.name, "n_hosts": p.N_hosts,
                     "nside": p.nside, "empty_pixel_fraction": p.empty_pixel_fraction,
                     "max_hosts_per_pixel": p.maxcol, "weight": w}
                    for p, w in zip(pmaps, tgt_weights)],
        "snr_threshold": args.snr_threshold, "zmax": args.zmax,
        "seed": args.seed, "ndraw_requested": args.ndraw,
        "cosmology": {"H0": args.H0, "Om0": args.Om0, "w0": W0_FID, "wa": WA_FID},
        "population": {k: getattr(pop, k) for k in pop.__dataclass_fields__},
    }

    print("=" * 92)
    print("CATALOG-TARGETED SELECTION INJECTIONS  (K=2 deep two-tracer mock)")
    print(f"  darksirens worktree : {meta['darksirens_worktree']} @ "
          f"{meta['darksirens_worktree_sha'][:10]}")
    print(f"  gmd                 : {gmd.__file__}")
    for p, w in zip(pmaps, tgt_weights):
        print(f"  target '{p.name}'     : {p.path}")
        print(f"    weight={w}  N_hosts={p.N_hosts:,}  nside={p.nside}  "
              f"occupied={p.n_occupied:,}/{p.npix:,}  max hosts/pix={p.maxcol}  "
              f"Omega_pix={p.apix:.6e} sr")
    print(f"  mixture             : pop={f_pop}  unif={f_unif}  "
          f"targeted={tgt_weights}")
    print(f"  detection           : gmd._network_snr >= {args.snr_threshold}")
    print(f"  zmax / cosmology    : {args.zmax}  H0={args.H0} Om0={args.Om0}")
    print(f"  ndraw (proposals)   : {args.ndraw:,}  seed={args.seed}  "
          f"batch={args.batch_size:,}")
    print("=" * 92, flush=True)

    sel = draw_injections(gmd, args.ndraw, args.seed, grids, ddldz_grid, pop,
                          pmaps, f_pop, f_unif, tgt_weights, args.snr_threshold,
                          args.batch_size, args.zmax)
    out_path = Path(args.out_path)
    neff_pop, branch_names = write_file(out_path, sel, args, f_pop, f_unif,
                                        tgt_weights, pmaps, gmd, meta)
    print(f"\nwrote {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")
    print(f"  proposed={sel['Ndraw']:,}  detected={sel['n_detected']:,}  "
          f"frac={sel['n_detected'] / sel['Ndraw']:.6e}")
    print(f"  branches={branch_names}")
    print(f"  proposed by branch={sel['n_proposed_branch']}  "
          f"detected by branch={sel['n_detected_branch']}")
    print(f"  pixel-placement fallbacks={sel['n_pixel_fallback']}")
    print(f"  N_eff (population-only, generator style) = {neff_pop:.1f}")

    if args.no_validate:
        return 0
    rng = np.random.default_rng(args.seed + 900_001)
    rec = validate(out_path, gmd, grids, ddldz_grid, pop, pmaps, f_pop, f_unif,
                   tgt_weights, args, rng, branch_names)
    rec["meta"] = meta
    print("\n--- validation ---")
    for k, v in rec.items():
        if k != "meta":
            print(f"  {k}: {v}")
    if args.validation_json:
        Path(args.validation_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.validation_json).write_text(json.dumps(rec, indent=2, default=str))
        print(f"\nwrote {args.validation_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
