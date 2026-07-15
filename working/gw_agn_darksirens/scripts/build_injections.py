#!/usr/bin/env python
"""Build darksirens gwcat-selection-1.0 injection files for the gw_agn mock.

The gw_agn mock's "detection" is a HARD TRUE-redshift cut ``z <= z_detect_max``
(z_detect_max = 1.0) -- there is NO SNR threshold.  This script mirrors
``gmd._draw_selection_batch`` exactly EXCEPT that the network-SNR detection is
replaced by the true-z cut, and the mass/spin proposal is a 90/10 DEFENSIVE
MIXTURE of gmd's "population" and "uniform" branches:

  * proposal z ~ uniform-in-comoving-volume on [0, zmax_proposal=1.2]
    (``gmd._sample_uniform_comoving_z`` on the grids built for zmax=1.2);
  * sky uniform (``gmd._sample_sky``);
  * per draw, with prob 0.9 the "population" branch: m1src / q / chieff from
    the fiducial powerlaw+peak samplers (``gmd._sample_powerlaw_peak_m1`` /
    ``gmd._sample_q`` / ``gmd._sample_chieff``); with prob 0.1 the "uniform"
    branch exactly as in ``gmd._draw_selection_batch``: m1det ~ U over
    ``gmd._M1DET_RANGE`` = (2, 200), q ~ U(0, 1), chi ~ U(-1, 1),
    m1src = m1det / (1 + z);
  * detection ``det = z <= z_detect_max`` (NO ``gmd._network_snr`` call);
  * pdraw for EVERY detected row = the EXACT mixture density
        0.9 * gmd._selection_pdraw("population", m1src, q, chi, z, grids, pop)
      + 0.1 * gmd._selection_pdraw("uniform",    m1src, q, chi, z, grids, pop)
    evaluated at that row's (m1src, q, chi, z) regardless of which branch
    generated it, with gmd's np.maximum(., 1e-300) floor retained.

WHY THE MIXTURE (estimator-variance pathology found on GB smoke): darksirens'
PL+G population is a PER-COMPONENT mixture in which the Gaussian peak
component's pairing carries NO m_min taper (the mixture code sets
mmin, dmmin = M_LO, 0.01 for components without m_min_spec), so its p_pop at
m2 slightly above m_min=5 is floored at w_G*phi_G(m1)*q^beta ~ e^-19, while
gmd's ``_mass_spin_pdf`` tapers the WHOLE mixture with S_low(m2; 5, 3) ~ e^-50
there.  With a pure-population proposal one injection (m1src=5.32, q=0.95)
got importance weight p_pop/pdraw ~ e^30, Neff -> 1.0, and logL = -inf
everywhere via the hard selection-Neff guard.  The 10% uniform component
floors the proposal density: population-branch draws always lie inside the
uniform branch's support (m1det < ~176 < 200, q in (0, 1], chi in [-1, 1]),
so pdraw_total >= 0.1 * pdraw_uniform > 0 for every detected row and no
single importance weight can blow up.  Physics/detection are UNCHANGED.

Detected rows only are stored, datasets = ``gmd.SELECTION_KEYS`` =
[m1det, m2det, m1src, m2src, dL, chieff, ra, dec, pdraw] (plus provenance
extras ``pdraw_population``/``pdraw_uniform``, ignored by the loader), in a
gwcat-selection-1.0 file consumed by ``darksirens.gw.utils.load_selection_samples``
(see RECON.md "Injection file").

Analytic cross-check: because the proposal is uniform-in-comoving-volume on
[0, 1.2] and detection is z <= 1.0, the detected FRACTION must equal
Vc(z=1.0)/Vc(z=1.2) to MC error -- i.e. the value of the (normalized) comoving-
volume CDF at z=1.0, ``np.interp(1.0, grids["z"], grids["vc_cdf"])``.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import astropy.units as u
import healpy as hp
from scipy.special import ndtr  # standard-normal CDF Phi(x), vectorised

# --- fixed conventions (RECON.md "Truth / fiducials", "Injection file") ------
H0_FID = 67.74
OM0_FID = 0.3075
W0_FID = -1.0
WA_FID = 0.0
ZMAX_PROPOSAL = 1.2        # uniform-in-comoving-volume proposal upper bound
Z_DETECT_MAX = 1.0         # hard TRUE-z detection cut (replaces the SNR cut)
CHIEFF_AMAX = 0.99
MIX_POPULATION = 0.9       # defensive-mixture weight: gmd "population" branch
MIX_UNIFORM = 0.1          # defensive-mixture weight: gmd "uniform" branch
PDRAW_FLOOR = 1.0e-300     # gmd._selection_pdraw's np.maximum floor, retained

DIAGNOSIS_NOTE = (
    "Defensive 90/10 population/uniform mixture proposal. Reason: darksirens' "
    "PL+G population is a PER-COMPONENT mixture in which the Gaussian peak "
    "component's pairing carries NO m_min taper (mixture code sets mmin,dmmin="
    "M_LO,0.01 for components without m_min_spec), so its p_pop at m2 slightly "
    "above m_min=5 is floored at w_G*phi_G(m1)*q^beta ~ e^-19, while gmd's "
    "_mass_spin_pdf tapers the WHOLE mixture with S_low(m2;5,3) ~ e^-50 there. "
    "With the pure-population proposal one injection (m1src=5.32, q=0.95) got "
    "importance weight p_pop/pdraw ~ e^30, Neff -> 1.0, and logL=-inf "
    "everywhere via the hard selection-Neff guard. The 10% uniform component "
    "floors pdraw_total >= 0.1*pdraw_uniform > 0 for every detected row "
    "(population draws always lie inside the uniform branch's support: "
    "m1det < ~176 < 200, q in (0,1], chi in [-1,1]), bounding all importance "
    "weights. pdraw for EVERY detected row is the exact mixture density "
    "0.9*gmd._selection_pdraw('population',...) + "
    "0.1*gmd._selection_pdraw('uniform',...) at that row's coordinates, "
    "regardless of generating branch. Physics/detection unchanged (z<=1.0 cut)."
)

DARKSIRENS_REPO = "/hildafs/projects/phy230014p/magana/src/darksirens"
DARKSIRENS_MERGE_BASE = "d387b4f"
GMD_DIR = "/hildafs/projects/phy230014p/magana/src/darksirens/scripts/mock_dark_sirens"

# --- AGN-targeted proposal lane (--proposal_mode agncat) ----------------------
# New 3-lane mixture 0.65 population + 0.10 uniform + 0.25 agn-targeted.  The
# population and uniform lanes are byte-compatible with the popuni generator
# above (same gmd samplers / densities); the third lane draws injections AT AGN
# catalog objects so the field-convention selection integral mu_AGN (which
# weights injections by the count-weighted, narrow-kernel AGN field) is well
# conditioned.  See the module task spec for the exact p_agn density.
MIX_AGN_POP = 0.65         # lane weight: gmd "population" branch
MIX_AGN_UNIF = 0.10        # lane weight: gmd "uniform" branch
MIX_AGN_CAT = 0.25         # lane weight: AGN-catalog-targeted branch
PROPOSAL_MIX_LABEL = "0.65pop+0.10unif+0.25agncat"
AGN_LANE_SIGMA_Z = 0.01    # proposal redshift kernel: z = z_j + N(0, s), s here
FIELD_KERNEL_SLOPE = 3.0e-3  # field-convention AGN kernel sigma_z = 3e-3*(1+z)
NSIDE = 64                 # HEALPix nside of the AGN catalog (RECON.md)
AGN_CATALOG = Path(__file__).resolve().parents[3] / "working/gw_agn/data/glass_prod/mock_catalog.h5"

INJECTION_SETS_AGNCAT = [
    ("injections_cat.h5", 2_000_000, 52001),
    ("injections_cat_B.h5", 2_000_000, 52002),
]
EXTRA_KEYS_AGNCAT = ["pdraw_population", "pdraw_uniform", "pdraw_agncat", "z", "lane"]
WT_MASTER = ("/tmp/claude-88592/-hildafs-projects-phy230014p-magana-gws-agn/"
             "89590650-74f5-413e-8311-7f0160636741/scratchpad/wt-master")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = REPO_ROOT / "working/gw_agn_darksirens/data"

# gmd is a standalone importable module (no darksirens import); RECON.md line 17.
sys.path.insert(0, GMD_DIR)
import generate_mock_data as gmd  # noqa: E402

# (name, ndraw, seed) per RECON task spec.
INJECTION_SETS = [
    ("injections.h5", 2_000_000, 42001),
    ("injections_B.h5", 2_000_000, 42002),
    ("injections_small.h5", 400_000, 42003),
]
BATCH_SIZE = 250_000


def git_head(repo):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc:  # pragma: no cover - provenance best-effort
        return f"<unavailable: {exc}>"


def git_branch(repo):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    except Exception as exc:  # pragma: no cover
        return f"<unavailable: {exc}>"


EXTRA_KEYS = ["pdraw_population", "pdraw_uniform"]   # provenance/validation extras


def draw_injection_set(ndraw, seed, grids, pop, batch_size, verbose=True):
    """Mirror gmd._draw_selection_batch over batches with (a) detection =
    z <= Z_DETECT_MAX instead of the SNR threshold (no gmd._network_snr call)
    and (b) a 90/10 defensive mixture of gmd's population / uniform mass-spin
    branches.  Returns detected-only arrays + counts.

    RNG consumption per batch: z (uniform-Vc), sky (ra, sin_dec), branch mask,
    then population draws (m1src, q, chi) for the population subset, then
    uniform draws (m1det, q, chi) for the uniform subset.

    pdraw for every detected row is the exact mixture density
    MIX_POPULATION * gmd._selection_pdraw("population", ...) +
    MIX_UNIFORM * gmd._selection_pdraw("uniform", ...) at that row's
    coordinates, regardless of which branch generated it.
    """
    rng = np.random.default_rng(seed)
    m1lo, m1hi = gmd._M1DET_RANGE
    chunks = []
    n_proposed = 0
    n_detected = 0
    n_det_pop = 0
    n_prop_pop = 0
    ci = 0
    while n_proposed < ndraw:
        n_batch = int(min(batch_size, ndraw - n_proposed))
        # --- z / sky proposals unchanged (as _draw_selection_batch) -----------
        z = gmd._sample_uniform_comoving_z(rng, grids, n_batch)
        ra, dec = gmd._sample_sky(rng, n_batch)
        dl = gmd._interp_dl(z, grids)
        # --- defensive-mixture branch selection --------------------------------
        use_pop = rng.uniform(size=n_batch) < MIX_POPULATION
        n_pop = int(use_pop.sum())
        m1src = np.empty(n_batch)
        q = np.empty(n_batch)
        chi = np.empty(n_batch)
        if n_pop:
            # gmd population branch, unchanged.
            m1src[use_pop] = gmd._sample_powerlaw_peak_m1(rng, n_pop, pop)
            q[use_pop] = gmd._sample_q(rng, m1src[use_pop], pop)
            chi[use_pop] = gmd._sample_chieff(rng, n_pop, pop)
        n_uni = n_batch - n_pop
        if n_uni:
            # gmd uniform branch, exactly as in _draw_selection_batch.
            uni = ~use_pop
            m1det_u = rng.uniform(m1lo, m1hi, n_uni)
            q[uni] = rng.uniform(0.0, 1.0, n_uni)
            chi[uni] = rng.uniform(-1.0, 1.0, n_uni)
            m1src[uni] = m1det_u / (1.0 + z[uni])
        m2src = q * m1src
        # --- detection: hard true-z cut (replaces snr >= snr_threshold) -------
        det = z <= Z_DETECT_MAX
        # --- exact mixture density on the detected subset ----------------------
        p_pop = gmd._selection_pdraw("population", m1src[det], q[det], chi[det],
                                     z[det], grids, pop)
        p_uni = gmd._selection_pdraw("uniform", m1src[det], q[det], chi[det],
                                     z[det], grids, pop)
        p_draw = np.maximum(MIX_POPULATION * p_pop + MIX_UNIFORM * p_uni, PDRAW_FLOOR)
        # Sanity: every detected row must lie inside the uniform branch's support
        # so pdraw_uniform genuinely floors the mixture density.
        m1det_det = (m1src * (1.0 + z))[det]
        assert m1det_det.min() > m1lo and m1det_det.max() < m1hi, (
            f"m1det outside uniform support: [{m1det_det.min()}, {m1det_det.max()}]")
        chunks.append({
            "m1det": m1det_det,
            "m2det": (q * m1src * (1.0 + z))[det],
            "m1src": m1src[det],
            "m2src": m2src[det],
            "dL": dl[det],
            "chieff": chi[det],
            "ra": ra[det],
            "dec": dec[det],
            "pdraw": p_draw,
            "pdraw_population": p_pop,
            "pdraw_uniform": p_uni,
        })
        n_proposed += n_batch
        n_detected += int(det.sum())
        n_det_pop += int((det & use_pop).sum())
        n_prop_pop += n_pop
        ci += 1
        if verbose:
            print(f"    batch {ci:2d}: proposed={n_proposed:,}/{ndraw:,}  "
                  f"detected={n_detected:,}", flush=True)

    arrays = {key: np.concatenate([c[key] for c in chunks])
              for key in gmd.SELECTION_KEYS + EXTRA_KEYS}
    return {**arrays, "Ndraw": n_proposed, "n_detected": n_detected,
            "n_proposed_population_branch": n_prop_pop,
            "n_detected_population_branch": n_det_pop,
            "n_detected_uniform_branch": n_detected - n_det_pop}


def neff_from_pdraw(pdraw):
    inv = 1.0 / np.asarray(pdraw, dtype=np.float64)
    if inv.size == 0:
        return 0.0
    return float(inv.sum() ** 2 / np.square(inv).sum())


def write_injection_file(out_path, sel, seed, meta_common):
    neff = neff_from_pdraw(sel["pdraw"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        # --- gwcat-selection-1.0 contract attrs ---
        f.attrs["format_version"] = "gwcat-selection-1.0"
        f.attrs["mock_data"] = True
        f.attrs["ndraw"] = int(sel["Ndraw"])          # TOTAL proposed
        f.attrs["chi_eff_swap_applied"] = True
        f.attrs["chi_eff_amax"] = float(CHIEFF_AMAX)
        f.attrs["cosmology_H0"] = float(H0_FID)
        f.attrs["cosmology_Om0"] = float(OM0_FID)
        f.attrs["Neff"] = float(neff)
        # --- provenance attrs ---
        f.attrs["seed"] = int(seed)
        f.attrs["zmax_proposal"] = float(ZMAX_PROPOSAL)
        f.attrs["z_detect_max"] = float(Z_DETECT_MAX)
        f.attrs["selection_proposal"] = "mixture(population=0.9,uniform=0.1)"
        f.attrs["proposal_mix_population"] = float(MIX_POPULATION)
        f.attrs["proposal_mix_uniform"] = float(MIX_UNIFORM)
        f.attrs["m1det_range_uniform"] = np.asarray(gmd._M1DET_RANGE, dtype=np.float64)
        f.attrs["detection_rule"] = "true_z_cut"
        f.attrs["n_detected"] = int(sel["n_detected"])
        f.attrs["n_detected_population_branch"] = int(sel["n_detected_population_branch"])
        f.attrs["n_detected_uniform_branch"] = int(sel["n_detected_uniform_branch"])
        f.attrs["generated_at_utc"] = meta_common["generated_at_utc"]
        f.attrs["generator_script"] = meta_common["script"]
        f.attrs["gws_agn_repo_head"] = meta_common["gws_agn_repo_head"]
        f.attrs["darksirens_repo_head"] = meta_common["darksirens_repo_head"]
        f.attrs["darksirens_repo_branch"] = meta_common["darksirens_repo_branch"]
        f.attrs["darksirens_file"] = meta_common["darksirens_file"]
        f.attrs["darksirens_merge_base_required"] = DARKSIRENS_MERGE_BASE
        # --- datasets: exactly gmd.SELECTION_KEYS order, detected rows only ---
        for key in gmd.SELECTION_KEYS:
            f.create_dataset(key, data=np.asarray(sel[key], dtype=np.float64),
                             compression="gzip", shuffle=True)
        # --- provenance extras (loader ignores unknown datasets) ---
        for key in EXTRA_KEYS:
            f.create_dataset(key, data=np.asarray(sel[key], dtype=np.float64),
                             compression="gzip", shuffle=True)
    return neff


def validate_file(out_path, grids, pop, analytic_frac, before=None):
    """Load with the real loader and cross-check n_det, z-range, pdraw, Neff,
    plus the defensive-mixture tail statistics."""
    from darksirens.gw.utils import load_selection_samples

    (m1det, m2det, dL, chieff, ra, dec, pdraw, ndraw) = load_selection_samples(str(out_path))
    m1det = np.asarray(m1det); m2det = np.asarray(m2det); dL = np.asarray(dL)
    chieff = np.asarray(chieff); ra = np.asarray(ra); dec = np.asarray(dec)
    pdraw = np.asarray(pdraw, dtype=np.float64)

    n_det = int(pdraw.size)
    # Re-invert dL through the SAME grids to recover z; must be <= Z_DETECT_MAX.
    z_recon = np.interp(dL, grids["dl"], grids["z"])

    with h5py.File(out_path, "r") as f:
        neff_attr = float(f.attrs["Neff"])
        n_det_attr = int(f.attrs["n_detected"])
        m1src = np.asarray(f["m1src"]); m2src = np.asarray(f["m2src"])
        p_pop = np.asarray(f["pdraw_population"])
        p_uni = np.asarray(f["pdraw_uniform"])

    # Stored pdraw must be exactly the mixture of the stored components.
    mix_exact = bool(np.array_equal(
        pdraw, np.maximum(MIX_POPULATION * p_pop + MIX_UNIFORM * p_uni, PDRAW_FLOOR)))
    # End-to-end recompute of both components at (m1src, q, chi, z_recon).
    q_recon = m2src / m1src
    p_pop_re = gmd._selection_pdraw("population", m1src, q_recon, chieff, z_recon, grids, pop)
    p_uni_re = gmd._selection_pdraw("uniform", m1src, q_recon, chieff, z_recon, grids, pop)
    mix_re = MIX_POPULATION * p_pop_re + MIX_UNIFORM * p_uni_re
    mix_recompute_max_rel_err = float(np.max(np.abs(mix_re - pdraw) / pdraw))

    # Defensive-mixture tail statistics.
    inv_pdraw_max = float((1.0 / pdraw).max())
    pop_share = MIX_POPULATION * p_pop / pdraw          # 0.9*p_pop / p_total
    floor_ok = bool(pdraw.min() >= MIX_UNIFORM * p_uni.min())

    frac_detected = n_det / ndraw
    all_finite = bool(
        np.all(np.isfinite(m1det)) and np.all(np.isfinite(m2det))
        and np.all(np.isfinite(dL)) and np.all(np.isfinite(chieff))
        and np.all(np.isfinite(ra)) and np.all(np.isfinite(dec))
        and np.all(np.isfinite(pdraw)) and np.all(np.isfinite(m1src))
        and np.all(np.isfinite(m2src))
    )
    rec = {
        "file": out_path.name,
        "ndraw": int(ndraw),
        "n_detected": n_det,
        "n_detected_attr": n_det_attr,
        "frac_detected": float(frac_detected),
        "analytic_frac": float(analytic_frac),
        "frac_rel_err_pct": float(100.0 * (frac_detected - analytic_frac) / analytic_frac),
        "z_recon_min": float(z_recon.min()),
        "z_recon_max": float(z_recon.max()),
        "z_within_cut": bool(z_recon.max() <= Z_DETECT_MAX + 1e-9),
        "dL_min": float(dL.min()),
        "dL_max": float(dL.max()),
        "pdraw_min": float(pdraw.min()),
        "pdraw_max": float(pdraw.max()),
        "pdraw_all_finite_pos": bool(np.all(np.isfinite(pdraw)) and np.all(pdraw > 0.0)),
        "all_finite": all_finite,
        "mixture_exact_from_components": mix_exact,
        "mixture_recompute_max_rel_err": mix_recompute_max_rel_err,
        "inv_pdraw_max": inv_pdraw_max,
        "pop_share_max": float(pop_share.max()),
        "pdraw_uniform_min": float(p_uni.min()),
        "floor_ok_min_pdraw_ge_0p1_puni_min": floor_ok,
        "Neff": float(neff_from_pdraw(pdraw)),
        "Neff_attr": neff_attr,
        "size_mb": out_path.stat().st_size / 1e6,
    }
    if before is not None:
        rec["inv_pdraw_max_before"] = float(before["inv_pdraw_max_old"])
        rec["Neff_before"] = float(before["Neff_old"])
        rec["inv_pdraw_max_reduction_factor"] = float(
            before["inv_pdraw_max_old"] / inv_pdraw_max)
    return rec


# =============================================================================
# AGN-targeted proposal lane (--proposal_mode agncat)
# =============================================================================
class AgnPixelMap:
    """Immutable view of the AGN catalog for the agn-targeted proposal lane.

    Holds the per-object (z, RING nside-64 pixel) plus a padded pixel->objects
    table so the kernel sum  sum_{j in pix} truncnorm(z; z_j, sigma_j)  can be
    evaluated vectorised for arbitrary rows.  ra/dec in the catalog are DEGREES
    (RECON.md) and are converted to radians here.
    """

    def __init__(self, path, nside):
        with h5py.File(path, "r") as f:
            z = np.asarray(f["z_agn"][:], dtype=np.float64)
            ra = np.deg2rad(np.asarray(f["ra_agn"][:], dtype=np.float64))
            dec = np.deg2rad(np.asarray(f["dec_agn"][:], dtype=np.float64))
        self.path = str(path)
        self.nside = int(nside)
        self.N_agn = int(z.size)
        self.apix = float(hp.nside2pixarea(nside))
        self.resol = float(hp.nside2resol(nside))
        self.z_agn = z
        self.ra_agn = ra
        self.dec_agn = dec
        # RING pixel index, theta = pi/2 - dec, phi = ra (RECON.md convention).
        self.pix_agn = hp.ang2pix(nside, 0.5 * np.pi - dec, ra)
        # Padded pixel->object table (NaN pad); lookup[pix] = compact row or -1.
        order = np.argsort(self.pix_agn, kind="stable")
        pix_sorted = self.pix_agn[order]
        z_sorted = z[order]
        uniq, counts = np.unique(pix_sorted, return_counts=True)
        n_occ = uniq.size
        self.maxcol = int(counts.max())
        self.n_occupied = int(n_occ)
        z_pad = np.full((n_occ, self.maxcol), np.nan, dtype=np.float64)
        col = np.concatenate([np.arange(c) for c in counts]) if n_occ else np.array([], int)
        rowidx = np.repeat(np.arange(n_occ), counts)
        z_pad[rowidx, col] = z_sorted
        self.z_pad = z_pad
        self.lookup = np.full(hp.nside2npix(nside), -1, dtype=np.int64)
        self.lookup[uniq] = np.arange(n_occ)

    def pix_of(self, ra, dec):
        return hp.ang2pix(self.nside, 0.5 * np.pi - dec, ra)

    def kernel_sum(self, z_rows, pix_rows, sigma_const=None):
        """Per-row  sum_{j in pix_rows} truncnorm(z_rows; z_j, sigma_j)  with the
        z>0 truncation normalization Phi(z_j / sigma_j).

        ``sigma_const`` set -> constant kernel width (the proposal, s=0.01);
        ``None`` -> per-object field kernel sigma_j = FIELD_KERNEL_SLOPE*(1+z_j).
        Rows whose pixel holds no AGN object return 0 (p_agn support = AGN pixels).
        """
        out = np.zeros(np.shape(z_rows)[0], dtype=np.float64)
        ci = self.lookup[pix_rows]
        valid = ci >= 0
        if not valid.any():
            return out
        zc = np.asarray(z_rows)[valid]
        zmat = self.z_pad[ci[valid]]                       # (nv, maxcol), NaN pad
        finite = np.isfinite(zmat)
        sig = float(sigma_const) if sigma_const is not None \
            else FIELD_KERNEL_SLOPE * (1.0 + np.where(finite, zmat, 0.0))
        x = (zc[:, None] - zmat) / sig
        pdf = np.exp(-0.5 * x * x) / (sig * np.sqrt(2.0 * np.pi))
        norm = ndtr(np.where(finite, zmat, 0.0) / sig)     # Phi(z_j/sig), z>0 trunc
        tn = np.where(finite, pdf / norm, 0.0)
        out[valid] = tn.sum(axis=1)
        return out


def _sample_agn_z(rng, z_j, s):
    """z_j + N(0, s) truncated to z>0 by rejection (exact renormalized Gaussian)."""
    z = z_j + rng.normal(0.0, s, size=z_j.shape[0])
    bad = z <= 0.0
    while bad.any():
        z[bad] = z_j[bad] + rng.normal(0.0, s, size=int(bad.sum()))
        bad = z <= 0.0
    return z


def _sample_uniform_in_pixels(rng, pix_tgt, nside, resol):
    """Uniform-on-sphere WITHIN each target nside pixel (RING), by a bounding-box
    (uniform-in-cos(theta), uniform-in-phi) proposal rejected on ang2pix == pix.
    Exact: rejection preserves uniformity, box is chosen to cover the pixel."""
    n = pix_tgt.shape[0]
    theta_c, phi_c = hp.pix2ang(nside, pix_tgt)            # RING centers
    dt = 2.0 * resol
    theta_lo = np.maximum(theta_c - dt, 0.0)
    theta_hi = np.minimum(theta_c + dt, np.pi)
    cos_lo = np.cos(theta_hi)                              # smaller cos
    cos_hi = np.cos(theta_lo)                              # larger cos
    sin_c = np.sin(np.clip(theta_c, 1.0e-6, np.pi - 1.0e-6))
    dphi = np.minimum(2.5 * resol / np.maximum(sin_c, 1.0e-2), np.pi)
    out_theta = np.empty(n)
    out_phi = np.empty(n)
    todo = np.ones(n, dtype=bool)
    for _ in range(1000):
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
    if todo.any():  # measure-zero stragglers -> pixel center (exact membership)
        out_theta[todo] = theta_c[todo]
        out_phi[todo] = phi_c[todo]
    ra = out_phi
    dec = 0.5 * np.pi - out_theta
    return ra, dec


def _p_agn_density(z, ra, dec, m1src, q, chi, grids, pop, pixmap):
    """Proposal density of the AGN-targeted lane in canonical coords
    (m1det, q, chi, dL, Omega):

        p_agn = [ (1/N_agn) sum_{j in pix(Omega)} truncnorm(z; z_j, s) ] / apix
                * p_massspin_pop(m1src, q, chi)
                / ( (1+z) * dL'(z) )

    The z-sky factor (1/N_agn) sum(...) / apix replaces the population lane's
    p_z(z)/(4 pi); the mass/spin factor and the (1+z) dL'(z) detector-frame
    Jacobian are IDENTICAL to gmd._selection_pdraw('population', ...)."""
    pix = pixmap.pix_of(ra, dec)
    ksum = pixmap.kernel_sum(z, pix, sigma_const=AGN_LANE_SIGMA_Z)
    p_zsky = ksum / pixmap.N_agn / pixmap.apix
    ddldz = np.interp(z, grids["z"], np.gradient(grids["dl"], grids["z"]))
    jac = ddldz * (1.0 + z)
    msp = gmd._mass_spin_pdf(m1src, q, chi, pop)
    return msp * p_zsky / np.maximum(jac, 1.0e-300)


def draw_injection_set_agncat(ndraw, seed, grids, pop, pixmap, batch_size, verbose=True):
    """Draw the 3-lane mixture (0.65 pop + 0.10 unif + 0.25 agncat) over batches.

    Per batch RNG order: one uniform for the categorical lane split (thresholds
    0.65 / 0.75), then population-lane draws, then uniform-lane draws, then
    agn-lane draws (object index, truncated-normal z, uniform-in-pixel sky,
    population masses/spin).  Detection = z <= Z_DETECT_MAX.  pdraw for every
    detected row is the exact mixture density
    0.65 p_pop + 0.10 p_unif + 0.25 p_agn at that row's coordinates, regardless
    of the generating lane."""
    rng = np.random.default_rng(seed)
    m1lo, m1hi = gmd._M1DET_RANGE
    thr_unif = MIX_AGN_POP + MIX_AGN_UNIF
    chunks = []
    n_proposed = n_detected = 0
    n_prop = [0, 0, 0]
    n_det = [0, 0, 0]
    ci = 0
    while n_proposed < ndraw:
        nb = int(min(batch_size, ndraw - n_proposed))
        lane_u = rng.uniform(size=nb)
        is_pop = lane_u < MIX_AGN_POP
        is_unif = (lane_u >= MIX_AGN_POP) & (lane_u < thr_unif)
        is_agn = lane_u >= thr_unif
        z = np.empty(nb); ra = np.empty(nb); dec = np.empty(nb)
        m1src = np.empty(nb); q = np.empty(nb); chi = np.empty(nb)
        lane = np.empty(nb, dtype=np.int8)
        # --- population lane (byte-compatible with the popuni population lane) --
        npop = int(is_pop.sum())
        if npop:
            zc = gmd._sample_uniform_comoving_z(rng, grids, npop)
            rac, decc = gmd._sample_sky(rng, npop)
            m1c, use_peak = gmd._sample_powerlaw_peak_m1(rng, npop, pop, return_component=True)
            qc = gmd._sample_q(rng, m1c, pop, use_peak=use_peak)
            chic = gmd._sample_chieff(rng, npop, pop)
            z[is_pop] = zc; ra[is_pop] = rac; dec[is_pop] = decc
            m1src[is_pop] = m1c; q[is_pop] = qc; chi[is_pop] = chic; lane[is_pop] = 0
        # --- uniform lane (byte-compatible with the popuni uniform lane) --------
        nunif = int(is_unif.sum())
        if nunif:
            zc = gmd._sample_uniform_comoving_z(rng, grids, nunif)
            rac, decc = gmd._sample_sky(rng, nunif)
            m1det_u = rng.uniform(m1lo, m1hi, nunif)
            qc = rng.uniform(0.0, 1.0, nunif)
            chic = rng.uniform(-1.0, 1.0, nunif)
            m1c = m1det_u / (1.0 + zc)
            z[is_unif] = zc; ra[is_unif] = rac; dec[is_unif] = decc
            m1src[is_unif] = m1c; q[is_unif] = qc; chi[is_unif] = chic; lane[is_unif] = 1
        # --- agn-targeted lane --------------------------------------------------
        nagn = int(is_agn.sum())
        if nagn:
            j = rng.integers(0, pixmap.N_agn, nagn)
            zc = _sample_agn_z(rng, pixmap.z_agn[j], AGN_LANE_SIGMA_Z)
            rac, decc = _sample_uniform_in_pixels(rng, pixmap.pix_agn[j], NSIDE, pixmap.resol)
            m1c, use_peak = gmd._sample_powerlaw_peak_m1(rng, nagn, pop, return_component=True)
            qc = gmd._sample_q(rng, m1c, pop, use_peak=use_peak)
            chic = gmd._sample_chieff(rng, nagn, pop)
            z[is_agn] = zc; ra[is_agn] = rac; dec[is_agn] = decc
            m1src[is_agn] = m1c; q[is_agn] = qc; chi[is_agn] = chic; lane[is_agn] = 2
        m2src = q * m1src
        det = z <= Z_DETECT_MAX
        # --- exact mixture density on the detected subset -----------------------
        zd = z[det]; rad = ra[det]; decd = dec[det]
        m1d = m1src[det]; qd = q[det]; chid = chi[det]
        p_pop = gmd._selection_pdraw("population", m1d, qd, chid, zd, grids, pop)
        p_uni = gmd._selection_pdraw("uniform", m1d, qd, chid, zd, grids, pop)
        p_agn = _p_agn_density(zd, rad, decd, m1d, qd, chid, grids, pop, pixmap)
        p_draw = np.maximum(
            MIX_AGN_POP * p_pop + MIX_AGN_UNIF * p_uni + MIX_AGN_CAT * p_agn, PDRAW_FLOOR)
        chunks.append({
            "m1det": (m1src * (1.0 + z))[det],
            "m2det": (q * m1src * (1.0 + z))[det],
            "m1src": m1d,
            "m2src": m2src[det],
            "dL": gmd._interp_dl(zd, grids),
            "chieff": chid,
            "ra": rad,
            "dec": decd,
            "pdraw": p_draw,
            "pdraw_population": p_pop,
            "pdraw_uniform": p_uni,
            "pdraw_agncat": p_agn,
            "z": zd,
            "lane": lane[det].astype(np.float64),
        })
        for L in (0, 1, 2):
            in_l = lane == L
            n_prop[L] += int(in_l.sum())
            n_det[L] += int((in_l & det).sum())
        n_proposed += nb
        n_detected += int(det.sum())
        ci += 1
        if verbose:
            print(f"    batch {ci:2d}: proposed={n_proposed:,}/{ndraw:,}  "
                  f"detected={n_detected:,}", flush=True)

    arrays = {key: np.concatenate([c[key] for c in chunks])
              for key in gmd.SELECTION_KEYS + EXTRA_KEYS_AGNCAT}
    return {**arrays, "Ndraw": n_proposed, "n_detected": n_detected,
            "n_proposed_lane": n_prop, "n_detected_lane": n_det}


def write_injection_file_agncat(out_path, sel, seed, pixmap, meta_common):
    neff = neff_from_pdraw(sel["pdraw"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        # --- gwcat-selection-1.0 contract attrs (same schema as popuni) ---
        f.attrs["format_version"] = "gwcat-selection-1.0"
        f.attrs["mock_data"] = True
        f.attrs["ndraw"] = int(sel["Ndraw"])
        f.attrs["chi_eff_swap_applied"] = True
        f.attrs["chi_eff_amax"] = float(CHIEFF_AMAX)
        f.attrs["cosmology_H0"] = float(H0_FID)
        f.attrs["cosmology_Om0"] = float(OM0_FID)
        f.attrs["Neff"] = float(neff)
        # --- provenance / new-lane attrs ---
        f.attrs["seed"] = int(seed)
        f.attrs["zmax_proposal"] = float(ZMAX_PROPOSAL)
        f.attrs["z_detect_max"] = float(Z_DETECT_MAX)
        f.attrs["selection_proposal"] = (
            f"mixture(population={MIX_AGN_POP},uniform={MIX_AGN_UNIF},"
            f"agncat={MIX_AGN_CAT})")
        f.attrs["proposal_mix"] = PROPOSAL_MIX_LABEL
        f.attrs["agn_lane_sigma_z"] = float(AGN_LANE_SIGMA_Z)
        f.attrs["proposal_mix_population"] = float(MIX_AGN_POP)
        f.attrs["proposal_mix_uniform"] = float(MIX_AGN_UNIF)
        f.attrs["proposal_mix_agncat"] = float(MIX_AGN_CAT)
        f.attrs["agn_catalog"] = pixmap.path
        f.attrs["agn_nside"] = int(pixmap.nside)
        f.attrs["agn_n_objects"] = int(pixmap.N_agn)
        f.attrs["agn_pixarea"] = float(pixmap.apix)
        f.attrs["m1det_range_uniform"] = np.asarray(gmd._M1DET_RANGE, dtype=np.float64)
        f.attrs["detection_rule"] = "true_z_cut"
        f.attrs["n_detected"] = int(sel["n_detected"])
        f.attrs["n_proposed_population_lane"] = int(sel["n_proposed_lane"][0])
        f.attrs["n_proposed_uniform_lane"] = int(sel["n_proposed_lane"][1])
        f.attrs["n_proposed_agncat_lane"] = int(sel["n_proposed_lane"][2])
        f.attrs["n_detected_population_lane"] = int(sel["n_detected_lane"][0])
        f.attrs["n_detected_uniform_lane"] = int(sel["n_detected_lane"][1])
        f.attrs["n_detected_agncat_lane"] = int(sel["n_detected_lane"][2])
        f.attrs["generated_at_utc"] = meta_common["generated_at_utc"]
        f.attrs["generator_script"] = meta_common["script"]
        f.attrs["gws_agn_repo_head"] = meta_common["gws_agn_repo_head"]
        f.attrs["darksirens_repo_head"] = meta_common["darksirens_repo_head"]
        f.attrs["darksirens_repo_branch"] = meta_common["darksirens_repo_branch"]
        f.attrs["darksirens_file"] = meta_common["darksirens_file"]
        f.attrs["darksirens_merge_base_required"] = DARKSIRENS_MERGE_BASE
        # --- datasets: gmd.SELECTION_KEYS order, detected rows only ---
        for key in gmd.SELECTION_KEYS:
            f.create_dataset(key, data=np.asarray(sel[key], dtype=np.float64),
                             compression="gzip", shuffle=True)
        # --- auditability extras (loader ignores unknown datasets) ---
        for key in EXTRA_KEYS_AGNCAT:
            f.create_dataset(key, data=np.asarray(sel[key], dtype=np.float64),
                             compression="gzip", shuffle=True)
    return neff


# --- validation: loader round-trip is run in a subprocess against wt-master ---
_LOADER_CHECK_SRC = r'''
import json, sys
import numpy as np
from darksirens.gw.utils import load_selection_samples
import darksirens
path = sys.argv[1]
m1det, m2det, dL, chieff, ra, dec, pdraw, ndraw = load_selection_samples(path)
def arr(x): return np.asarray(x, dtype=np.float64)
m1det, m2det, dL = arr(m1det), arr(m2det), arr(dL)
chieff, ra, dec, pdraw = arr(chieff), arr(ra), arr(dec), arr(pdraw)
out = {
    "darksirens_file": darksirens.__file__,
    "n_loaded": int(pdraw.size),
    "ndraw": int(ndraw),
    "all_finite": bool(np.all(np.isfinite(m1det)) and np.all(np.isfinite(m2det))
        and np.all(np.isfinite(dL)) and np.all(np.isfinite(chieff))
        and np.all(np.isfinite(ra)) and np.all(np.isfinite(dec))
        and np.all(np.isfinite(pdraw))),
    "pdraw_all_pos": bool(np.all(pdraw > 0.0)),
    "pdraw_min": float(pdraw.min()), "pdraw_max": float(pdraw.max()),
    "ra_min": float(ra.min()), "ra_max": float(ra.max()),
    "dec_min": float(dec.min()), "dec_max": float(dec.max()),
    "dL_min": float(dL.min()), "dL_max": float(dL.max()),
}
print("LOADERJSON:" + json.dumps(out))
'''


def _loader_roundtrip(out_path):
    """Run darksirens.gw.utils.load_selection_samples in a subprocess with
    PYTHONPATH=wt-master (the darksirens master checkout) and return its stats."""
    src_path = Path(WT_MASTER).parent / "_loader_check.py"
    src_path.write_text(_LOADER_CHECK_SRC)
    env = dict(os.environ)
    env["PYTHONPATH"] = WT_MASTER + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    res = subprocess.run([sys.executable, str(src_path), str(out_path)],
                         capture_output=True, text=True, env=env)
    line = next((ln for ln in res.stdout.splitlines() if ln.startswith("LOADERJSON:")), None)
    if line is None:
        raise RuntimeError(f"loader subprocess failed:\nSTDOUT:\n{res.stdout}\n"
                           f"STDERR:\n{res.stderr}")
    return json.loads(line[len("LOADERJSON:"):])


def _field_neff(dL, ra, dec, pdraw, m1src, m2src, chieff, grids, pop, pixmap):
    """Field-convention AGN-selection Neff proxy: the effective sample size of the
    importance weights that estimate the field-mode AGN selection integral
    mu_AGN = (1/Ndraw) sum_det p_target(d) / pdraw(d) over the DETECTED rows.

    The field-mode target density for the count-weighted AGN catalog is

        p_target(d) = p_massspin_pop(m1,q,chi)
                      * [ (1/N_agn) sum_{i in pix} truncnorm(z; z_i, sigma_i) ]/apix
                      / ( (1+z) dL'(z) ),     sigma_i = 3e-3 (1+z_i),

    i.e. the population mass/spin density times the count-weighted AGN FIELD shape
    (evaluated at z(dL; H0=67.74) and the row's pixel) with the same detector-frame
    Jacobian as pdraw's population/agn lanes.  Because the population mass/spin
    factor and the (1+z) dL'(z) Jacobian appear identically in pdraw's population
    and agn-targeted lanes, they cancel in w=p_target/pdraw -- so w isolates the
    SPATIAL (z-sky) conditioning that the agn-targeted lane fixes.  (Evaluating
    the spec's literal field/pdraw instead leaves an uncancelled 1/p_massspin that
    is dominated by rare-mass tails and hides the spatial improvement; reported as
    ``Neff_field_over_pdraw_literal`` for reference.)  Returns
    Neff=(sum w)^2/sum w^2 and diagnostics."""
    z_field = np.interp(dL, grids["dl"], grids["z"])
    pix = pixmap.pix_of(ra, dec)
    field_zsky = pixmap.kernel_sum(z_field, pix, sigma_const=None) / pixmap.N_agn / pixmap.apix
    ddldz = np.interp(z_field, grids["z"], np.gradient(grids["dl"], grids["z"]))
    jac = ddldz * (1.0 + z_field)
    q = m2src / m1src
    msp = gmd._mass_spin_pdf(m1src, q, chieff, pop)
    p_target = msp * field_zsky / np.maximum(jac, 1.0e-300)
    w = p_target / pdraw
    sw = float(w.sum()); sw2 = float(np.square(w).sum())
    neff = (sw * sw / sw2) if sw2 > 0.0 else 0.0
    w_lit = field_zsky / pdraw
    sw_l = float(w_lit.sum()); sw2_l = float(np.square(w_lit).sum())
    neff_lit = (sw_l * sw_l / sw2_l) if sw2_l > 0.0 else 0.0
    return {"Neff": float(neff), "max_w": float(w.max()),
            "in_field_frac": float(np.mean(field_zsky > 0.0)),
            "Neff_field_over_pdraw_literal": float(neff_lit),
            "n_rows": int(pdraw.size)}


def validate_file_agncat(out_path, grids, pop, pixmap, analytic, old_path, seed):
    """Full agncat validation: loader round-trip (subprocess/wt-master), detected
    fraction vs 3-lane analytic, p_agn exactness (200 rows, first-principles), and
    the field-mode Neff proxy (new vs old injections.h5)."""
    rng = np.random.default_rng(seed + 900_000)
    with h5py.File(out_path, "r") as f:
        m1det = np.asarray(f["m1det"]); m1src = np.asarray(f["m1src"])
        m2src = np.asarray(f["m2src"]); dL = np.asarray(f["dL"])
        chieff = np.asarray(f["chieff"]); ra = np.asarray(f["ra"]); dec = np.asarray(f["dec"])
        pdraw = np.asarray(f["pdraw"], dtype=np.float64)
        p_pop = np.asarray(f["pdraw_population"]); p_uni = np.asarray(f["pdraw_uniform"])
        p_agn = np.asarray(f["pdraw_agncat"]); z = np.asarray(f["z"]); lane = np.asarray(f["lane"])
        ndraw = int(f.attrs["ndraw"]); neff_attr = float(f.attrs["Neff"])
        n_det_attr = int(f.attrs["n_detected"])
        n_prop_lane = [int(f.attrs[f"n_proposed_{n}_lane"]) for n in ("population", "uniform", "agncat")]
        n_det_lane = [int(f.attrs[f"n_detected_{n}_lane"]) for n in ("population", "uniform", "agncat")]
    n_det = int(pdraw.size)

    # (1) loader round-trip against darksirens master (wt-master).
    loader = _loader_roundtrip(out_path)

    # (2) mixture exactness: stored pdraw == mixture of stored components.
    mix_from_components = np.maximum(
        MIX_AGN_POP * p_pop + MIX_AGN_UNIF * p_uni + MIX_AGN_CAT * p_agn, PDRAW_FLOOR)
    mixture_exact = bool(np.array_equal(pdraw, mix_from_components))

    # (3) detected fraction vs 3-lane analytic.
    frac_detected = n_det / ndraw
    frac_rel_err_pct = 100.0 * (frac_detected - analytic["frac_total"]) / analytic["frac_total"]

    # (4) p_agn EXACTNESS: first-principles direct sum over each row's pixel, for
    #     up to 200 random detected rows.  z is taken from the canonical stored
    #     coordinates (z = m1det/m1src - 1) so this path shares no code with the
    #     vectorised padded-table sum used at write time.
    nsamp = int(min(200, n_det))
    sidx = rng.choice(n_det, size=nsamp, replace=False)
    z_from_coords = m1det[sidx] / m1src[sidx] - 1.0     # exact: m1det=(1+z)m1src
    q_s = m2src[sidx] / m1src[sidx]
    m1src_s = m1src[sidx]; chi_s = chieff[sidx]
    pix_s = pixmap.pix_of(ra[sidx], dec[sidx])
    apix = pixmap.apix; s = AGN_LANE_SIGMA_Z
    ddldz_grid = np.gradient(grids["dl"], grids["z"])
    p_agn_bruteforce = np.empty(nsamp)
    for k in range(nsamp):
        zk = z_from_coords[k]
        in_pix = pixmap.pix_agn == pix_s[k]              # direct catalog scan
        zj = pixmap.z_agn[in_pix]
        if zj.size:
            tn = (np.exp(-0.5 * ((zk - zj) / s) ** 2) / (s * np.sqrt(2.0 * np.pi))) / ndtr(zj / s)
            ksum = float(tn.sum())
        else:
            ksum = 0.0
        p_zsky = ksum / pixmap.N_agn / apix
        ddldz = float(np.interp(zk, grids["z"], ddldz_grid))
        jac = ddldz * (1.0 + zk)
        msp = float(gmd._mass_spin_pdf(np.array([m1src_s[k]]), np.array([q_s[k]]),
                                       np.array([chi_s[k]]), pop)[0])
        p_agn_bruteforce[k] = msp * p_zsky / max(jac, 1.0e-300)
    stored = p_agn[sidx]
    # Floored relative error: for genuine nonzero rows denom==stored (true rel);
    # for rows where p_agn underflowed to 0 (deep-Gaussian-tail pixels whose AGN
    # objects sit far in z, density ~1e-300) the floor makes the metric compare
    # to ~1e-300 so the same underflow in the first-principles path scores ~0.
    denom = np.maximum(stored, 1.0e-300)
    rel = np.abs(p_agn_bruteforce - stored) / denom
    p_agn_max_rel_err = float(rel.max())
    n_agn_pix_rows = int((stored > 0.0).sum())
    p_agn_zero_rows_bruteforce_max = float(
        p_agn_bruteforce[stored == 0.0].max()) if int((stored == 0.0).sum()) else 0.0

    # (5) field-mode Neff proxy: new file vs old injections.h5.
    new_field = _field_neff(dL, ra, dec, pdraw, m1src, m2src, chieff, grids, pop, pixmap)
    old_field = None
    if old_path is not None and Path(old_path).exists():
        with h5py.File(old_path, "r") as f:
            dL_o = np.asarray(f["dL"]); ra_o = np.asarray(f["ra"])
            dec_o = np.asarray(f["dec"]); pd_o = np.asarray(f["pdraw"], dtype=np.float64)
            m1s_o = np.asarray(f["m1src"]); m2s_o = np.asarray(f["m2src"])
            chi_o = np.asarray(f["chieff"])
        old_field = _field_neff(dL_o, ra_o, dec_o, pd_o, m1s_o, m2s_o, chi_o, grids, pop, pixmap)

    rec = {
        "file": out_path.name,
        "ndraw": ndraw,
        "n_detected": n_det,
        "n_detected_attr": n_det_attr,
        "loader_n_loaded": loader["n_loaded"],
        "loader_ndraw": loader["ndraw"],
        "loader_darksirens_file": loader["darksirens_file"],
        "loader_all_finite": loader["all_finite"],
        "loader_pdraw_all_pos": loader["pdraw_all_pos"],
        "n_proposed_lane": n_prop_lane,
        "n_detected_lane": n_det_lane,
        "frac_detected": float(frac_detected),
        "analytic_frac_total": float(analytic["frac_total"]),
        "analytic_frac_lanes": analytic["frac_lanes"],
        "measured_frac_lanes": [n_det_lane[L] / n_prop_lane[L] if n_prop_lane[L] else 0.0
                                for L in range(3)],
        "frac_rel_err_pct": float(frac_rel_err_pct),
        "mixture_exact_from_components": mixture_exact,
        "p_agn_exactness_nsamp": nsamp,
        "p_agn_nonzero_rows_in_sample": n_agn_pix_rows,
        "p_agn_max_rel_err": p_agn_max_rel_err,
        "p_agn_zero_rows_bruteforce_max": p_agn_zero_rows_bruteforce_max,
        "z_min": float(z.min()), "z_max": float(z.max()),
        "z_within_cut": bool(z.max() <= Z_DETECT_MAX + 1e-9),
        "pdraw_min": float(pdraw.min()), "pdraw_max": float(pdraw.max()),
        "Neff": float(neff_from_pdraw(pdraw)), "Neff_attr": neff_attr,
        "field_neff_new": new_field,
        "field_neff_old": old_field,
        "field_neff_improvement_x": (new_field["Neff"] / old_field["Neff"]
                                     if old_field and old_field["Neff"] > 0 else None),
        "size_mb": out_path.stat().st_size / 1e6,
    }
    return rec


def _analytic_frac_lanes(grids, pixmap):
    """Analytic detected fraction per lane and for the mixture.

    pop & unif lanes: z ~ uniform-in-comoving-volume on [0, zmax], so
    P(z<=1) = vc_cdf(1.0).  agn lane: z = z_j + N(0,s) truncated to z>0, j uniform
    over the catalog, so P(z<=1) = mean_j [Phi((1-z_j)/s) - Phi(-z_j/s)]/Phi(z_j/s)."""
    f_vc = float(np.interp(Z_DETECT_MAX, grids["z"], grids["vc_cdf"]))
    s = AGN_LANE_SIGMA_Z
    zj = pixmap.z_agn
    p_det_agn = (ndtr((Z_DETECT_MAX - zj) / s) - ndtr((0.0 - zj) / s)) / ndtr(zj / s)
    f_agn = float(np.mean(p_det_agn))
    frac_total = MIX_AGN_POP * f_vc + MIX_AGN_UNIF * f_vc + MIX_AGN_CAT * f_agn
    return {"frac_lanes": [f_vc, f_vc, f_agn], "frac_total": float(frac_total)}


def build_agncat(args, grids, pop, meta_common):
    """Build + validate the agn-targeted injection files (proposal_mode=agncat)."""
    out_dir = args.out_dir
    # Guarantee we import the darksirens MASTER generator: the byte content of the
    # imported gmd file must equal origin/master's (verified independent of branch).
    diff = subprocess.run(
        ["git", "-C", DARKSIRENS_REPO, "diff", "--quiet", "origin/master", "HEAD",
         "--", "scripts/mock_dark_sirens/generate_mock_data.py"])
    assert diff.returncode == 0, (
        "imported generate_mock_data.py differs from origin/master; import the "
        "master copy (e.g. from the wt-master worktree) before running agncat.")

    pixmap = AgnPixelMap(AGN_CATALOG, NSIDE)
    analytic = _analytic_frac_lanes(grids, pixmap)
    old_path = out_dir / "injections.h5"

    print("=" * 90)
    print(f"AGN-targeted proposal  ({PROPOSAL_MIX_LABEL})   sigma_z={AGN_LANE_SIGMA_Z}")
    print(f"  AGN catalog: {pixmap.path}")
    print(f"  N_agn={pixmap.N_agn:,}  occupied pixels={pixmap.n_occupied:,}/"
          f"{hp.nside2npix(NSIDE):,} ({100.0*pixmap.n_occupied/hp.nside2npix(NSIDE):.1f}%)  "
          f"max objs/pix={pixmap.maxcol}  apix={pixmap.apix:.6e} sr")
    print(f"  analytic detected fraction: pop/unif={analytic['frac_lanes'][0]:.6f}  "
          f"agn={analytic['frac_lanes'][2]:.6f}  mixture={analytic['frac_total']:.6f}")
    print("=" * 90)

    meta_common = {**meta_common, "proposal_mode": "agncat",
                   "proposal_mix": PROPOSAL_MIX_LABEL,
                   "agn_lane_sigma_z": AGN_LANE_SIGMA_Z,
                   "agn_catalog": pixmap.path, "agn_n_objects": pixmap.N_agn,
                   "analytic_frac_lanes": analytic["frac_lanes"],
                   "analytic_frac_total": analytic["frac_total"]}

    file_records = []
    for name, ndraw, seed in INJECTION_SETS_AGNCAT:
        out_path = out_dir / name
        print(f"\nBuilding {name}: Ndraw={ndraw:,} seed={seed}  "
              f"(expected n_det ~ {int(round(ndraw * analytic['frac_total'])):,})")
        sel = draw_injection_set_agncat(ndraw, seed, grids, pop, pixmap, args.batch_size)
        neff = write_injection_file_agncat(out_path, sel, seed, pixmap, meta_common)
        rec = {"file": name, "ndraw": int(sel["Ndraw"]),
               "n_detected": int(sel["n_detected"]),
               "n_proposed_lane": sel["n_proposed_lane"],
               "n_detected_lane": sel["n_detected_lane"],
               "frac_detected": sel["n_detected"] / sel["Ndraw"],
               "Neff": float(neff), "seed": int(seed),
               "size_mb": out_path.stat().st_size / 1e6}
        file_records.append(rec)
        print(f"  wrote {out_path}  ({rec['size_mb']:.2f} MB)  n_det={rec['n_detected']:,}  "
              f"lanes prop={sel['n_proposed_lane']} det={sel['n_detected_lane']}  "
              f"frac={rec['frac_detected']:.6f}  Neff={rec['Neff']:.1f}")

    meta = {**meta_common, "batch_size": args.batch_size, "files": file_records}
    meta_path = out_dir / "injections_cat_meta.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=False)
    print(f"\nwrote {meta_path}")

    if args.no_validate:
        return 0

    print("\n" + "=" * 90)
    print("VALIDATION (loader round-trip via wt-master; detected-frac; p_agn "
          "exactness; field-mode Neff proxy)")
    print("=" * 90)
    val_records = []
    all_ok = True
    for rec in file_records:
        v = validate_file_agncat(out_dir / rec["file"], grids, pop, pixmap, analytic,
                                 old_path, rec["seed"])
        val_records.append(v)
        ok = (v["loader_n_loaded"] == v["n_detected"] == v["n_detected_attr"]
              and v["loader_all_finite"] and v["loader_pdraw_all_pos"]
              and v["loader_ndraw"] == v["ndraw"]
              and v["mixture_exact_from_components"]
              and abs(v["frac_rel_err_pct"]) < 0.5
              and v["z_within_cut"]
              and v["p_agn_max_rel_err"] < 1.0e-12
              and (v["field_neff_improvement_x"] is None
                   or v["field_neff_improvement_x"] >= 20.0))
        all_ok = all_ok and ok
        nf = v["field_neff_new"]; of = v["field_neff_old"]
        print(f"\n  {v['file']}  ({v['size_mb']:.2f} MB)")
        print(f"    loader(wt-master): n_loaded={v['loader_n_loaded']:,}  "
              f"ndraw={v['loader_ndraw']:,}  finite={v['loader_all_finite']}  "
              f"pdraw>0={v['loader_pdraw_all_pos']}")
        print(f"      darksirens.__file__={v['loader_darksirens_file']}")
        print(f"    n_detected={v['n_detected']:,} (attr={v['n_detected_attr']:,})  "
              f"lanes prop={v['n_proposed_lane']} det={v['n_detected_lane']}")
        print(f"    frac_detected={v['frac_detected']:.6f}  analytic={v['analytic_frac_total']:.6f}"
              f"  rel_err={v['frac_rel_err_pct']:+.3f}%")
        print(f"      per-lane analytic={['%.5f'%x for x in v['analytic_frac_lanes']]}  "
              f"measured={['%.5f'%x for x in v['measured_frac_lanes']]}")
        print(f"    mixture exact from stored components: {v['mixture_exact_from_components']}")
        print(f"    p_agn exactness: nsamp={v['p_agn_exactness_nsamp']} "
              f"(in-AGN-pixel rows={v['p_agn_nonzero_rows_in_sample']})  "
              f"max rel err={v['p_agn_max_rel_err']:.2e}  "
              f"(zero-row bruteforce max={v['p_agn_zero_rows_bruteforce_max']:.1e})")
        print(f"    z range=[{v['z_min']:.4f}, {v['z_max']:.6f}] within_cut={v['z_within_cut']}  "
              f"pdraw=[{v['pdraw_min']:.3e}, {v['pdraw_max']:.3e}]  Neff={v['Neff']:.1f}")
        print(f"    FIELD-MODE AGN-selection Neff proxy (w = p_target/pdraw, mass-spin cancels):")
        print(f"      NEW {v['file']}: Neff={nf['Neff']:.1f}  max_w={nf['max_w']:.3e}  "
              f"in_field_frac={nf['in_field_frac']:.4f}  (n_rows={nf['n_rows']:,})")
        if of is not None:
            print(f"      OLD injections.h5 : Neff={of['Neff']:.1f}  max_w={of['max_w']:.3e}  "
                  f"in_field_frac={of['in_field_frac']:.4f}  (n_rows={of['n_rows']:,})")
            print(f"      improvement = {v['field_neff_improvement_x']:.1f}x  (target >= 20x)")
            print(f"      [ref] spec-literal field/pdraw Neff: new={nf['Neff_field_over_pdraw_literal']:.1f}"
                  f"  old={of['Neff_field_over_pdraw_literal']:.1f}  (1/mass-spin-tail dominated)")
        print(f"    -> OK={ok}")

    val_path = out_dir / "injections_cat_validation.json"
    with open(val_path, "w") as fh:
        json.dump(val_records, fh, indent=2, sort_keys=False)
    print(f"\nwrote {val_path}")
    print(f"\nALL AGNCAT INJECTION FILES VALID: {all_ok}")
    return 0 if all_ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--no-validate", action="store_true")
    p.add_argument("--proposal_mode", choices=["popuni", "agncat"], default="popuni",
                   help="popuni (default): existing 90/10 population+uniform mixture "
                        "(injections{,_B,_small}.h5, byte-compatible). agncat: 3-lane "
                        "0.65pop+0.10unif+0.25agncat mixture drawing at AGN catalog "
                        "objects (injections_cat{,_B}.h5).")
    args = p.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # merge-base guard (RECON.md line 14).
    mb = subprocess.run(
        ["git", "-C", DARKSIRENS_REPO, "merge-base", "--is-ancestor",
         DARKSIRENS_MERGE_BASE, "HEAD"])
    assert mb.returncode == 0, f"{DARKSIRENS_MERGE_BASE} is not an ancestor of darksirens HEAD"

    # Grids + population EXACTLY per RECON task spec.
    cosmo = gmd._build_cosmology(H0_FID, OM0_FID, W0_FID, WA_FID)
    grids = gmd._cosmology_grids(cosmo, zmax=ZMAX_PROPOSAL)
    pop = gmd.PopulationConfig()

    # Analytic detected fraction under the uniform-Vc proposal:
    #   proposal draws z via inverse-CDF of grids["vc_cdf"] (Vc-normalized to 1
    #   at z=1.2), so P(z <= 1.0) = vc_cdf(1.0) = Vc(1.0)/Vc(1.2).
    analytic_frac_grid = float(np.interp(Z_DETECT_MAX, grids["z"], grids["vc_cdf"]))
    vc_1p0 = cosmo.comoving_volume(Z_DETECT_MAX).to_value(u.Mpc**3)
    vc_1p2 = cosmo.comoving_volume(ZMAX_PROPOSAL).to_value(u.Mpc**3)
    analytic_frac_astropy = float(vc_1p0 / vc_1p2)

    meta_common = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "gws_agn_repo_head": git_head(REPO_ROOT),
        "gws_agn_repo_branch": git_branch(REPO_ROOT),
        "darksirens_repo_head": git_head(DARKSIRENS_REPO),
        "darksirens_repo_branch": git_branch(DARKSIRENS_REPO),
        "darksirens_file": gmd.__file__,
        "cosmology": {"H0": H0_FID, "Om0": OM0_FID, "w0": W0_FID, "wa": WA_FID},
        "zmax_proposal": ZMAX_PROPOSAL,
        "z_detect_max": Z_DETECT_MAX,
        "proposal_mix_population": MIX_POPULATION,
        "proposal_mix_uniform": MIX_UNIFORM,
        "m1det_range_uniform": list(gmd._M1DET_RANGE),
        "proposal_diagnosis_note": DIAGNOSIS_NOTE,
        "analytic_frac_vc_cdf": analytic_frac_grid,
        "analytic_frac_astropy": analytic_frac_astropy,
    }

    # --- agn-targeted proposal lane branch (leaves the popuni path untouched) --
    if args.proposal_mode == "agncat":
        return build_agncat(args, grids, pop, meta_common)

    # Capture pre-overwrite pdraw tail stats (pure-population files, if present)
    # so the fix's variance reduction is documented in meta/validation.
    before_stats = {}
    for name, _, _ in INJECTION_SETS:
        prev = out_dir / name
        if prev.exists():
            try:
                with h5py.File(prev, "r") as f:
                    pd_old = np.asarray(f["pdraw"], dtype=np.float64)
                    before_stats[name] = {
                        "pdraw_min_old": float(pd_old.min()),
                        "inv_pdraw_max_old": float((1.0 / pd_old).max()),
                        "Neff_old": float(f.attrs["Neff"]),
                        "n_detected_old": int(pd_old.size),
                    }
            except Exception as exc:
                before_stats[name] = {"error": str(exc)}
    if before_stats:
        meta_common["before_fix"] = before_stats

    print("=" * 90)
    print("Analytic detected fraction  Vc(1.0)/Vc(1.2):")
    print(f"    grids vc_cdf(1.0)     = {analytic_frac_grid:.6f}")
    print(f"    astropy Vc(1.0)/Vc(1.2)= {analytic_frac_astropy:.6f}")
    print("=" * 90)

    file_records = []
    for name, ndraw, seed in INJECTION_SETS:
        out_path = out_dir / name
        print(f"\nBuilding {name}: Ndraw={ndraw:,} seed={seed} "
              f"mix=({MIX_POPULATION}/{MIX_UNIFORM}) "
              f"(expected n_det ~ {int(round(ndraw * analytic_frac_grid)):,})")
        sel = draw_injection_set(ndraw, seed, grids, pop, args.batch_size)
        neff = write_injection_file(out_path, sel, seed, meta_common)
        rec = {
            "file": name,
            "ndraw": int(sel["Ndraw"]),
            "n_detected": int(sel["n_detected"]),
            "n_detected_population_branch": int(sel["n_detected_population_branch"]),
            "n_detected_uniform_branch": int(sel["n_detected_uniform_branch"]),
            "frac_detected": sel["n_detected"] / sel["Ndraw"],
            "Neff": float(neff),
            "seed": int(seed),
            "size_mb": out_path.stat().st_size / 1e6,
        }
        file_records.append(rec)
        print(f"  wrote {out_path}  ({rec['size_mb']:.2f} MB)  "
              f"n_det={rec['n_detected']:,} "
              f"(pop={rec['n_detected_population_branch']:,} "
              f"uni={rec['n_detected_uniform_branch']:,})  "
              f"frac={rec['frac_detected']:.6f}  Neff={rec['Neff']:.1f}")

    meta = {**meta_common, "batch_size": args.batch_size, "files": file_records}
    meta_path = out_dir / "injections_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=False)
    print(f"\nwrote {meta_path}")

    if args.no_validate:
        return 0

    # ----------------------------- validation --------------------------------
    print("\n" + "=" * 90)
    print("VALIDATION via darksirens.gw.utils.load_selection_samples")
    print("=" * 90)
    val_records = []
    all_ok = True
    for rec in file_records:
        v = validate_file(out_dir / rec["file"], grids, pop, analytic_frac_grid,
                          before=before_stats.get(rec["file"]))
        val_records.append(v)
        ok = (v["z_within_cut"] and v["pdraw_all_finite_pos"] and v["all_finite"]
              and abs(v["frac_rel_err_pct"]) < 0.5
              and v["n_detected"] == v["n_detected_attr"]
              and v["mixture_exact_from_components"]
              and v["mixture_recompute_max_rel_err"] < 1.0e-6
              and v["floor_ok_min_pdraw_ge_0p1_puni_min"])
        all_ok = all_ok and ok
        print(f"\n  {v['file']}  ({v['size_mb']:.2f} MB)")
        print(f"    Ndraw={v['ndraw']:,}  n_detected={v['n_detected']:,}  "
              f"(attr={v['n_detected_attr']:,})")
        print(f"    frac_detected={v['frac_detected']:.6f}  analytic={v['analytic_frac']:.6f}  "
              f"rel_err={v['frac_rel_err_pct']:+.3f}%")
        print(f"    z_recon range=[{v['z_recon_min']:.4f}, {v['z_recon_max']:.6f}]  "
              f"within_cut(<=1.0)={v['z_within_cut']}")
        print(f"    dL range=[{v['dL_min']:.2f}, {v['dL_max']:.2f}] Mpc")
        print(f"    pdraw range=[{v['pdraw_min']:.3e}, {v['pdraw_max']:.3e}]  "
              f"finite&pos={v['pdraw_all_finite_pos']}  all_finite={v['all_finite']}")
        print(f"    mixture exact from stored components: {v['mixture_exact_from_components']}  "
              f"recompute max rel err: {v['mixture_recompute_max_rel_err']:.2e}")
        print(f"    tail: max 1/pdraw = {v['inv_pdraw_max']:.3e}"
              + (f"  (before: {v['inv_pdraw_max_before']:.3e}, "
                 f"reduction x{v['inv_pdraw_max_reduction_factor']:.3e})"
                 if "inv_pdraw_max_before" in v else ""))
        print(f"    max[0.9*pdraw_pop/pdraw_total] = {v['pop_share_max']:.6f}  "
              f"min(pdraw) >= 0.1*min(pdraw_uniform): "
              f"{v['floor_ok_min_pdraw_ge_0p1_puni_min']} "
              f"(0.1*min(p_uni)={0.1 * v['pdraw_uniform_min']:.3e})")
        print(f"    Neff={v['Neff']:.1f} (attr={v['Neff_attr']:.1f})"
              + (f"  before: {v['Neff_before']:.1f}" if "Neff_before" in v else "")
              + f"  ->  OK={ok}")

    val_path = out_dir / "injections_validation.json"
    with open(val_path, "w") as f:
        json.dump(val_records, f, indent=2, sort_keys=False)
    print(f"\nwrote {val_path}")
    print(f"\nALL INJECTION FILES VALID: {all_ok}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
