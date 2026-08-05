#!/usr/bin/env python3
"""Deep mock whose SELECTION ACTS ON THE DATA THE POSTERIOR CONDITIONS ON.

THE DEFECT UNDER TEST
---------------------
``generate_mock_data`` decides detection and builds the posterior from two
INDEPENDENT random draws:

* ``_draw_events_until_detected`` keeps a source if
  ``_network_snr(...) >= snr_threshold``, where ``_network_snr`` multiplies the
  true amplitude by a fresh ``projection = Beta(2,5)**0.5`` -- a latent
  orientation factor drawn per source;
* ``_posterior_samples`` then draws a SEPARATE noise realisation to make the
  observation the PE conditions on.

So detection is decided by a latent ``w`` that never appears in the data.  Write
the detected-set likelihood out:

    p(d, det | theta) = int dw p(w) 1[det(theta, w)] p(d | theta)
                      = P(det | theta) p(d | theta)
    p({d_i} | Lambda, det)
        = prod_i [ int dtheta p(d_i|theta) p(theta|Lambda) P(det|theta) ]
          / mu(Lambda)

i.e. the correct per-event integrand carries an extra ``P(det|theta)`` INSIDE
the integral.  darksirens -- like every population code -- instead computes

    prod_i [ int dtheta p(d_i|theta) p(theta|Lambda) ] / mu(Lambda),

which is the right answer only when detection is a DETERMINISTIC FUNCTION OF THE
DATA, so that ``1[det(d_i)] = 1`` for every observed event and drops out.  The
mock violates that premise; the inference is not wrong, the mock is.

THE FIX MEASURED HERE
---------------------
``--detection observed-data`` draws the observation ONCE and uses it for both:

    d_obs      = dL_true * exp(sigma_dL * eps)          (one measurement)
    m1det_obs  = m1det_true + sigma_m1 * eps_1          (likewise m2, chi_eff, sky)
    rho_obs    = snr_ref * (Mchirp_det_obs / 30)^(5/6) * (1000 / d_obs)
    detected   <=>  rho_obs >= snr_threshold

and the PE then conditions on exactly that observation.  ``rho_obs`` is a
function of the observed data alone, so ``1[det(d)] = 1`` on the detected set
and the inference's likelihood is the correct one.  The stochasticity that the
projection factor used to supply is now supplied by the measurement noise --
which is what actually makes a real event detectable or not, and which is the
same randomness the posterior sees.  This also restores the Malmquist/Eddington
scatter across the threshold that the old rule had no way to produce.

The projection latent CANNOT be retained in this mode.  Keeping it would leave
detection depending on a variable absent from the data, i.e. an extra
``P(det | d, theta)`` inside the per-event integral -- a different mis-
specification, not a fix.  Dropping it raises the horizon at fixed ``snr_ref``,
so ``--calibrate`` solves for the ``snr_ref`` that reproduces the control arm's
detection fraction, keeping the two arms' event populations comparable.

``--detection true-params`` is the control: gmd's current rule, independent
projection and independent PE noise.  Both arms share this script, these seeds,
these catalogs and these ancillary uncertainty models, so an A/B difference is
attributable to the detection rule alone.

ANCILLARY UNCERTAINTIES, IDENTICAL IN BOTH ARMS
-----------------------------------------------
* ``sigma_dL`` is required to be an explicit constant.  gmd's default
  ``clip(1.8/rho, 0.08, 0.35)`` would make the PE width depend on the SNR that
  the noise itself determines -- circular in observed-data mode.  The campaign's
  runs all pin it (0.10), so requiring it costs nothing.
* ``sigma_ang`` uses ``clip(35/rho_opt, 1, 12)`` deg with ``rho_opt`` the
  OPTIMAL (projection-free, noise-free) SNR, a deterministic function of the
  source parameters.  gmd uses the projected SNR, which does not exist in
  observed-data mode; using ``rho_opt`` in BOTH arms keeps the arms differing
  only in the detection rule.  The control arm is therefore not bit-identical to
  gmd's own output -- it is validated instead against the published five-seed
  baseline offset.

MODES
-----
``catalog``     : draw a fresh complete host catalog with gmd's own
                  ``_generate_complete_catalog``, in gmd's on-disk format.  Used
                  to extend the closure test beyond the five catalog
                  realisations the baseline happened to have, because the
                  catalog realisation -- not the events -- is the dominant
                  variance term (seed-to-seed sd 1.09 against a per-seed 68%
                  half-width of 0.49).
``events``      : draw detected events at the hosts of an existing complete
                  catalog and write a ``gwcat-1.0`` file.  The catalog and its
                  pixelated survey are REUSED from the baseline runs, so the
                  catalog realisation is shared with the published numbers.
``injections``  : draw a ``population+uniform`` selection campaign under the
                  same detection rule and write ``gwcat-selection-1.0``.
                  Injections store TRUE parameters (mu(theta) is an integral
                  over true parameters); only the detection decision is noisy.
``calibrate``   : report the detection fraction of both rules over a common
                  proposal and solve for the observed-data ``snr_ref`` that
                  matches the control.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

EXP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKTREE = "/hildafs/projects/phy230014p/magana/src/darksirens-pefix"

H0_FID = 67.74
OM0_FID = 0.3075
W0_FID = -1.0
WA_FID = 0.0
ZMAX = 2.0
SNR_THRESHOLD = 8.0
SNR_REF = 11.5              # gmd's rho_ref in _network_snr
CHIEFF_AMAX = 0.99
DETECTIONS = ("true-params", "observed-data")


def _git(repo, *args):
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception as exc:  # pragma: no cover - provenance best-effort
        return f"<unavailable: {exc}>"


# =============================================================================
# detection statistic
# =============================================================================
def snr_optimal(m1det, m2det, dl, snr_ref):
    """Projection-free, noise-free network SNR from DETECTOR-frame masses.

    Same functional form as ``gmd._network_snr`` with ``projection = 1``; that
    function takes source-frame masses and multiplies the chirp mass by (1+z),
    which is identically the detector-frame chirp mass."""
    mchirp_det = (m1det * m2det) ** 0.6 / (m1det + m2det) ** 0.2
    return snr_ref * (mchirp_det / 30.0) ** (5.0 / 6.0) * (1000.0 / dl)


def sigma_ang_from(rho_opt, sky_uncertainty_deg=None):
    """Sky uncertainty in radians; deterministic in the source parameters."""
    deg = (sky_uncertainty_deg if sky_uncertainty_deg is not None
           else np.clip(35.0 / rho_opt, 1.0, 12.0))
    return np.deg2rad(deg)


def observe(rng, m1det, m2det, chi, dl, ra, dec, sigma_dl, sig_m1_frac,
            sig_m2_frac, sig_chi, sigma_ang, snr_ref_sigma=None):
    """ONE measurement per source, in the same parameterisation PR #332 uses.

    Distance carries multiplicative noise; masses, spin and sky are additive.
    The clips are applied here, before the detection statistic, so the object
    the SNR is computed from is bit-identically the object the PE conditions
    on.

    SEQUENTIAL SKY WIDTH (darksirens PR #335): when ``sigma_ang is None`` the
    masses and distance are measured FIRST and the sky width is derived from
    those OBSERVED values -- ``clip(35/rho, 1, 12) deg`` with ``rho`` the
    projection-free amplitude of the observed detector-frame masses and
    distance on the ``snr_ref_sigma`` scale -- before the sky offsets are
    drawn.  ``sigma_ang`` is then a deterministic function of the RECORDED
    data, so a fixed-width sky posterior built from it is exact.  A
    truth-derived width is not: ``sigma_ang ∝ dL/Mc_det^(5/6)`` is itself an
    H0-sensitive observable, and freezing it at its latent true value breaks
    the detected-set score identity (measured -0.49 ± 0.08 on H0 even under
    the exact likelihood; see ORACLE_FINDINGS.md §8).

    The rng draw ORDER (dL, m1det, m2det, chieff, ra, dec) is unchanged in
    both modes and the sigma_ang computation consumes no rng, so everything
    except the two sky columns stays bit-identical to the latent-width
    output."""
    n = m1det.shape[0]
    sig_m1 = sig_m1_frac * m1det
    sig_m2 = sig_m2_frac * m2det
    obs_dl = dl * np.exp(sigma_dl * rng.normal(size=n))
    obs_m1 = np.clip(rng.normal(m1det, sig_m1), 2.0, None)
    obs_m2 = np.clip(rng.normal(m2det, sig_m2), 1.0, None)
    if sigma_ang is None:
        rho_obs_opt = snr_optimal(obs_m1, obs_m2, obs_dl, snr_ref_sigma)
        sigma_ang = np.deg2rad(np.clip(35.0 / rho_obs_opt, 1.0, 12.0))
    return {
        "dL": obs_dl,
        "m1det": obs_m1,
        "m2det": obs_m2,
        "chieff": np.clip(rng.normal(chi, sig_chi), -1.0, 1.0),
        "ra": (ra + rng.normal(0.0, sigma_ang / np.maximum(np.cos(dec), 0.1))) % (2.0 * np.pi),
        "dec": np.clip(dec + rng.normal(0.0, sigma_ang), -0.5 * np.pi, 0.5 * np.pi),
        "sigma_dl": np.full(n, sigma_dl),
        "sig_m1": sig_m1,
        "sig_m2": sig_m2,
        "sigma_ang": sigma_ang,
    }


def detect(gmd, rng, detection, m1src, m2src, z, dl, ra, dec, chi, args):
    """Apply the arm's detection rule.  Returns (mask, observation-or-None).

    In observed-data mode the returned observation is the one the caller MUST
    hand to the PE -- that sharing is the entire point of the fix."""
    m1det = m1src * (1.0 + z)
    m2det = m2src * (1.0 + z)
    rho_opt = snr_optimal(m1det, m2det, dl, args.snr_ref_control)
    sigma_ang = sigma_ang_from(rho_opt, args.sky_uncertainty_deg)
    if detection == "true-params":
        # gmd's rule verbatim: fresh projection latent on the TRUE amplitude.
        snr = gmd._network_snr(m1src, m2src, z, dl, rng)
        return snr >= args.snr_threshold, None, snr, sigma_ang
    # PR #335 convention: with --sky_width observed (and no explicit constant),
    # observe() derives the sky width sequentially from the observed amplitude
    # on the SAME snr scale sigma_ang has always used (snr_ref_control).
    sig_pass = (None if (args.sky_width == "observed"
                         and args.sky_uncertainty_deg is None) else sigma_ang)
    obs = observe(rng, m1det, m2det, chi, dl, ra, dec, args.dL_fractional_uncertainty,
                  args.m1det_fractional_uncertainty, args.m2det_fractional_uncertainty,
                  args.chieff_uncertainty, sig_pass,
                  snr_ref_sigma=args.snr_ref_control)
    rho_obs = snr_optimal(obs["m1det"], obs["m2det"], obs["dL"], args.snr_ref)
    return rho_obs >= args.snr_threshold, obs, rho_obs, sigma_ang


# =============================================================================
# posterior samples
# =============================================================================
def posterior_samples(rng, obs, nsamp):
    """Flat-prior posterior draws GIVEN the stored observation (PR #332 forms).

    Distance: with ``ln d_obs ~ N(ln dL, s)`` the flat-in-dL posterior is
    ``ln dL ~ N(ln d_obs + s^2, s)`` -- lognormal about the OBSERVATION, shifted
    ``+s^2`` by the volume factor, not about the truth.  The additively measured
    quantities have Gaussian flat-prior posteriors centred on their observed
    values with the same widths."""
    nobs = obs["dL"].shape[0]
    out = {k: [] for k in ("ra", "dec", "dL", "m1det", "m2det", "chieff", "p_pe")}
    for i in range(nobs):
        s = float(obs["sigma_dl"][i])
        sa = float(np.atleast_1d(obs["sigma_ang"])[i]
                   if np.ndim(obs["sigma_ang"]) else obs["sigma_ang"])
        out["dL"].append(rng.lognormal(np.log(obs["dL"][i]) + s * s, s, nsamp))
        out["ra"].append((obs["ra"][i]
                          + rng.normal(0.0, sa / max(np.cos(obs["dec"][i]), 0.1), nsamp))
                         % (2.0 * np.pi))
        out["dec"].append(np.clip(obs["dec"][i] + rng.normal(0.0, sa, nsamp),
                                  -0.5 * np.pi, 0.5 * np.pi))
        out["m1det"].append(np.clip(rng.normal(obs["m1det"][i], obs["sig_m1"][i], nsamp),
                                    2.0, None))
        out["m2det"].append(np.clip(rng.normal(obs["m2det"][i], obs["sig_m2"][i], nsamp),
                                    1.0, None))
        out["chieff"].append(np.clip(rng.normal(obs["chieff"][i], 0.08, nsamp), -1.0, 1.0))
        out["p_pe"].append(np.ones(nsamp))
    return {k: np.concatenate(v) for k, v in out.items()}


# =============================================================================
# events
# =============================================================================
def draw_events(gmd, rng, args, catalog, grids, pop):
    """gmd's rejection loop, with the arm's detection rule substituted.

    The rate weighting ``(1+z)**(gamma-1)`` acceptance is gmd's, kept verbatim
    so the detected redshift distribution matches the inference's population
    model exactly as in the baseline."""
    zmax_grid = float(grids["z"][-1])
    rate_gmax = max(1.0, (1.0 + zmax_grid) ** (pop.gamma - 1.0))
    kept = []
    n_have = 0
    n_tried = 0
    while n_have < args.nobs:
        ntry = max(4 * args.nobs, 256)
        host_idx = rng.integers(0, len(catalog["z"]), ntry)
        z = catalog["z"][host_idx]
        ra = catalog["ra"][host_idx]
        dec = catalog["dec"][host_idx]
        dl = gmd._interp_dl(z, grids)
        m1, use_peak = gmd._sample_powerlaw_peak_m1(rng, ntry, pop, return_component=True)
        q = gmd._sample_q(rng, m1, pop, use_peak=use_peak)
        m2 = q * m1
        chi = gmd._sample_chieff(rng, ntry, pop)
        det, obs, snr, sigma_ang = detect(gmd, rng, args.detection,
                                          m1, m2, z, dl, ra, dec, chi, args)
        det = det & (rng.uniform(size=ntry) < (1.0 + z) ** (pop.gamma - 1.0) / rate_gmax)
        n_tried += ntry
        if not np.any(det):
            continue
        rec = {"z": z[det], "ra": ra[det], "dec": dec[det], "dl": dl[det],
               "m1": m1[det], "m2": m2[det], "q": q[det], "chi": chi[det],
               "snr": snr[det], "sigma_ang": np.broadcast_to(sigma_ang, z.shape)[det]}
        if obs is not None:
            for k, v in obs.items():
                rec[f"obs_{k}"] = (np.broadcast_to(v, z.shape)[det]
                                   if np.ndim(v) else np.full(int(det.sum()), v))
        kept.append(rec)
        n_have += int(det.sum())
    truth = {k: np.concatenate([x[k] for x in kept])[:args.nobs] for k in kept[0]}
    return truth, n_tried


def build_events(gmd, args, meta):
    rng = np.random.default_rng(args.seed)
    cosmo = gmd._build_cosmology(args.H0, args.Om0, W0_FID, WA_FID)
    grids = gmd._cosmology_grids(cosmo, args.zmax)
    pop = gmd.PopulationConfig(gamma=args.gamma)

    with h5py.File(args.catalog, "r") as f:
        cat = {k: np.asarray(f[k][:], dtype=float) for k in ("ra", "dec", "z")}
    print(f"catalog: {args.catalog}  {cat['z'].size:,} hosts")

    truth, n_tried = draw_events(gmd, rng, args, cat, grids, pop)
    print(f"events: {args.nobs} detected from {n_tried:,} proposed "
          f"(frac={args.nobs / n_tried:.4e})")
    print(f"  z in [{truth['z'].min():.4f}, {truth['z'].max():.4f}]  "
          f"median {np.median(truth['z']):.4f}")
    print(f"  detection statistic in [{truth['snr'].min():.2f}, {truth['snr'].max():.2f}]")

    if args.detection == "observed-data":
        # THE SHARED DRAW: the PE conditions on the very observation that was
        # tested against the threshold.  No new noise is generated here.
        obs = {k[4:]: v for k, v in truth.items() if k.startswith("obs_")}
    else:
        # gmd's behaviour: a fresh, independent measurement for the PE.
        rho_opt = snr_optimal(truth["m1"] * (1.0 + truth["z"]),
                              truth["m2"] * (1.0 + truth["z"]),
                              truth["dl"], args.snr_ref_control)
        obs = observe(rng, truth["m1"] * (1.0 + truth["z"]),
                      truth["m2"] * (1.0 + truth["z"]), truth["chi"], truth["dl"],
                      truth["ra"], truth["dec"], args.dL_fractional_uncertainty,
                      args.m1det_fractional_uncertainty,
                      args.m2det_fractional_uncertainty, args.chieff_uncertainty,
                      sigma_ang_from(rho_opt, args.sky_uncertainty_deg))
    post = posterior_samples(rng, obs, args.nsamp)
    z_pe = np.interp(post["dL"], grids["dl"], grids["z"])
    post["m1src"] = post["m1det"] / (1.0 + z_pe)
    post["m2src"] = post["m2det"] / (1.0 + z_pe)

    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.attrs["format_version"] = "gwcat-1.0"
        f.attrs["mock_data"] = True
        f.attrs["nobs"] = int(args.nobs)
        f.attrs["nsamp"] = int(args.nsamp)
        f.attrs["pe_cosmology_H0"] = float(args.H0)
        f.attrs["pe_cosmology_Om0"] = float(args.Om0)
        f.attrs["chi_eff_in_p_pe"] = True
        f.attrs["chi_eff_amax"] = float(CHIEFF_AMAX)
        f.attrs["pe_centering"] = "observed"
        f.attrs["pop_model"] = "powerlaw+peak"
        f.attrs["shared_beta"] = True
        f.attrs["shared_spin"] = True
        f.attrs["shared_gamma"] = True
        f.attrs["detection_rule"] = args.detection
        f.attrs["detection_shares_noise_with_pe"] = args.detection == "observed-data"
        f.attrs["sky_width"] = args.sky_width
        f.attrs["snr_threshold"] = float(args.snr_threshold)
        f.attrs["snr_ref"] = float(args.snr_ref)
        f.attrs["n_proposed_for_events"] = int(n_tried)
        f.attrs["metadata_json"] = json.dumps(meta, default=str)
        for k, v in post.items():
            f.create_dataset(k, data=v, compression="gzip", shuffle=True)
        g = f.create_group("truth")
        for k, v in truth.items():
            g.create_dataset(k, data=v)
        for k, v in obs.items():
            if k not in truth and np.ndim(v):
                g.create_dataset(f"pe_obs_{k}", data=np.broadcast_to(v, (args.nobs,)))
    print(f"wrote {out}")
    return {"n_proposed": n_tried, "detected_fraction": args.nobs / n_tried,
            "z_median": float(np.median(truth["z"])),
            "z_max": float(truth["z"].max())}


# =============================================================================
# injections
# =============================================================================
def build_injections(gmd, args, meta):
    """population+uniform selection campaign under the arm's detection rule.

    Stored coordinates are the TRUE parameters and ``pdraw`` is gmd's own
    ``_selection_pdraw("population+uniform", ...)`` -- unchanged, because
    mu(theta) is an integral over true parameters and only the DETECTION
    decision is affected by measurement noise.  This mirrors a real injection
    campaign: true injected parameters, recovery decided on noisy data."""
    rng = np.random.default_rng(args.seed)
    cosmo = gmd._build_cosmology(args.H0, args.Om0, W0_FID, WA_FID)
    grids = gmd._cosmology_grids(cosmo, args.zmax)
    pop = gmd.PopulationConfig(gamma=args.gamma)
    m1lo, m1hi = gmd._M1DET_RANGE

    chunks = []
    n_proposed = n_detected = 0
    ci = 0
    while n_proposed < args.ndraw:
        nb = int(min(args.batch_size, args.ndraw - n_proposed))
        z = gmd._sample_uniform_comoving_z(rng, grids, nb)
        ra, dec = gmd._sample_sky(rng, nb)
        dl = gmd._interp_dl(z, grids)
        # gmd's population+uniform: Bernoulli(0.9) population else uniform.
        use_pop = rng.uniform(size=nb) < 0.9
        m1_pop, use_peak = gmd._sample_powerlaw_peak_m1(rng, nb, pop,
                                                        return_component=True)
        q_pop = gmd._sample_q(rng, m1_pop, pop, use_peak=use_peak)
        chi_pop = gmd._sample_chieff(rng, nb, pop)
        m1det_u = rng.uniform(m1lo, m1hi, nb)
        q_u = rng.uniform(0.0, 1.0, nb)
        chi_u = rng.uniform(-1.0, 1.0, nb)
        m1src = np.where(use_pop, m1_pop, m1det_u / (1.0 + z))
        q = np.where(use_pop, q_pop, q_u)
        chi = np.where(use_pop, chi_pop, chi_u)
        m2src = q * m1src

        det, _obs, _snr, _sa = detect(gmd, rng, args.detection,
                                      m1src, m2src, z, dl, ra, dec, chi, args)
        p_draw = gmd._selection_pdraw("population+uniform", m1src[det], q[det],
                                      chi[det], z[det], grids, pop)
        zd = z[det]
        chunks.append({
            "m1det": m1src[det] * (1.0 + zd), "m2det": m2src[det] * (1.0 + zd),
            "m1src": m1src[det], "m2src": m2src[det], "dL": dl[det],
            "chieff": chi[det], "ra": ra[det], "dec": dec[det], "pdraw": p_draw,
            "z": zd,
        })
        n_proposed += nb
        n_detected += int(det.sum())
        ci += 1
        print(f"    batch {ci:4d}: proposed={n_proposed:,}/{args.ndraw:,}  "
              f"detected={n_detected:,}", flush=True)

    arrays = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    inv = 1.0 / arrays["pdraw"]
    neff = float(inv.sum() ** 2 / np.square(inv).sum())

    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.attrs["format_version"] = "gwcat-selection-1.0"
        f.attrs["mock_data"] = True
        f.attrs["ndraw"] = int(n_proposed)
        f.attrs["Ndraw"] = int(n_proposed)
        f.attrs["Neff"] = neff
        f.attrs["selection_proposal"] = "population+uniform"
        f.attrs["chi_eff_swap_applied"] = True
        f.attrs["chi_eff_amax"] = float(CHIEFF_AMAX)
        f.attrs["cosmology_H0"] = float(args.H0)
        f.attrs["cosmology_Om0"] = float(args.Om0)
        f.attrs["pop_model"] = "powerlaw+peak"
        f.attrs["shared_beta"] = True
        f.attrs["shared_spin"] = True
        f.attrs["shared_gamma"] = True
        f.attrs["detection_rule"] = args.detection
        f.attrs["snr_threshold"] = float(args.snr_threshold)
        f.attrs["snr_ref"] = float(args.snr_ref)
        f.attrs["n_detected"] = int(n_detected)
        f.attrs["metadata_json"] = json.dumps(meta, default=str)
        for k in gmd.SELECTION_KEYS + ["z"]:
            f.create_dataset(k, data=np.asarray(arrays[k], dtype=np.float64),
                             compression="gzip", shuffle=True)
    print(f"wrote {out}  detected={n_detected:,}/{n_proposed:,} "
          f"(frac={n_detected / n_proposed:.6e})  Neff(population-only)={neff:.1f}")
    return {"n_detected": n_detected, "ndraw": n_proposed,
            "detected_fraction": n_detected / n_proposed, "neff_population": neff}


# =============================================================================
# complete host catalog
# =============================================================================
def build_catalog(gmd, args, meta):
    """A fresh complete catalog, drawn by gmd's own routine, in gmd's format."""
    rng = np.random.default_rng(args.seed)
    cosmo = gmd._build_cosmology(args.H0, args.Om0, W0_FID, WA_FID)
    grids = gmd._cosmology_grids(cosmo, args.zmax)
    cat = gmd._generate_complete_catalog(rng, args.n_galaxies, grids,
                                         gmd.SurveyConfig())
    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.attrs["mock_data"] = True
        f.attrs["description"] = ("Complete isotropic, uniform-in-comoving-volume "
                                  "mock galaxy catalog before EM incompleteness.")
        f.attrs["metadata_json"] = json.dumps(meta, default=str)
        for k, v in cat.items():
            f.create_dataset(k, data=v, compression="gzip", shuffle=True)
    print(f"wrote {out}  {args.n_galaxies:,} hosts, z in "
          f"[{cat['z'].min():.4f}, {cat['z'].max():.4f}]")
    return {"n_galaxies": int(args.n_galaxies), "z_min": float(cat["z"].min()),
            "z_max": float(cat["z"].max())}


# =============================================================================
# calibration
# =============================================================================
def calibrate(gmd, args, meta):
    """Solve for the observed-data snr_ref matching the control's detection rate.

    Both rules are evaluated on ONE common set of proposed sources drawn at the
    catalog's hosts, so the comparison is paired and the only difference is the
    rule."""
    rng = np.random.default_rng(args.seed)
    cosmo = gmd._build_cosmology(args.H0, args.Om0, W0_FID, WA_FID)
    grids = gmd._cosmology_grids(cosmo, args.zmax)
    pop = gmd.PopulationConfig(gamma=args.gamma)
    with h5py.File(args.catalog, "r") as f:
        cat = {k: np.asarray(f[k][:], dtype=float) for k in ("ra", "dec", "z")}

    n = args.calibrate_n
    host_idx = rng.integers(0, cat["z"].size, n)
    z = cat["z"][host_idx]
    ra, dec = cat["ra"][host_idx], cat["dec"][host_idx]
    dl = gmd._interp_dl(z, grids)
    m1, use_peak = gmd._sample_powerlaw_peak_m1(rng, n, pop, return_component=True)
    q = gmd._sample_q(rng, m1, pop, use_peak=use_peak)
    m2 = q * m1
    chi = gmd._sample_chieff(rng, n, pop)

    frac_ctrl = float((gmd._network_snr(m1, m2, z, dl, rng)
                       >= args.snr_threshold).mean())

    m1det, m2det = m1 * (1.0 + z), m2 * (1.0 + z)
    sigma_ang = sigma_ang_from(snr_optimal(m1det, m2det, dl, args.snr_ref_control),
                               args.sky_uncertainty_deg)
    obs = observe(rng, m1det, m2det, chi, dl, ra, dec, args.dL_fractional_uncertainty,
                  args.m1det_fractional_uncertainty, args.m2det_fractional_uncertainty,
                  args.chieff_uncertainty, sigma_ang)
    # rho_obs is linear in snr_ref, so the whole curve comes from one draw.
    base = snr_optimal(obs["m1det"], obs["m2det"], obs["dL"], 1.0)

    def frac_at(ref):
        return float((ref * base >= args.snr_threshold).mean())

    lo, hi = 0.05 * args.snr_ref_control, args.snr_ref_control
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if frac_at(mid) < frac_ctrl:
            lo = mid
        else:
            hi = mid
    ref = 0.5 * (lo + hi)
    rec = {"n_pilot": n, "detected_fraction_control": frac_ctrl,
           "snr_ref_control": args.snr_ref_control,
           "snr_ref_observed_data": ref,
           "detected_fraction_observed_data_at_solution": frac_at(ref),
           "detected_fraction_observed_data_at_control_ref": frac_at(args.snr_ref_control)}
    print(json.dumps(rec, indent=2))
    return rec


# =============================================================================
# main
# =============================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True,
                   choices=["catalog", "events", "injections", "calibrate"])
    p.add_argument("--detection", required=True, choices=DETECTIONS)
    p.add_argument("--n_galaxies", type=int, default=1_000_000)
    p.add_argument("--out_path", default=None)
    p.add_argument("--catalog", default=None, help="Complete host catalog (events/calibrate).")
    p.add_argument("--worktree", default=DEFAULT_WORKTREE)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--nobs", type=int, default=1000)
    p.add_argument("--nsamp", type=int, default=2000)
    p.add_argument("--ndraw", type=int, default=120_000_000)
    p.add_argument("--batch_size", type=int, default=4_000_000)
    p.add_argument("--calibrate_n", type=int, default=4_000_000)
    p.add_argument("--snr_threshold", type=float, default=SNR_THRESHOLD)
    p.add_argument("--snr_ref", type=float, default=SNR_REF,
                   help="Amplitude scale of the arm's detection statistic. For "
                        "observed-data use the --mode calibrate solution.")
    p.add_argument("--snr_ref_control", type=float, default=SNR_REF,
                   help="Amplitude scale used for rho_opt (sky-uncertainty model) "
                        "and for the control rule; identical in both arms.")
    p.add_argument("--dL_fractional_uncertainty", type=float, required=True)
    p.add_argument("--m1det_fractional_uncertainty", type=float, default=0.08)
    p.add_argument("--m2det_fractional_uncertainty", type=float, default=0.10)
    p.add_argument("--chieff_uncertainty", type=float, default=0.08)
    p.add_argument("--sky_uncertainty_deg", type=float, default=None)
    p.add_argument("--sky_width", choices=("latent", "observed"), default="latent",
                   help="Sky-noise width model in observed-data mode.  'latent' "
                        "(default, reproduces all pre-PR#335 outputs): "
                        "clip(35/rho_opt(true params), 1, 12) deg.  'observed' "
                        "(darksirens PR #335): draw dL/m1det/m2det first, derive "
                        "the width from the OBSERVED amplitude, then draw the "
                        "sky offsets.  Only the two sky columns change; the rng "
                        "stream and every other draw stay bit-identical.  "
                        "Ignored when --sky_uncertainty_deg is given, and in "
                        "the control arm.")
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--H0", type=float, default=H0_FID)
    p.add_argument("--Om0", type=float, default=OM0_FID)
    p.add_argument("--zmax", type=float, default=ZMAX)
    p.add_argument("--summary_json", default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    gmd_dir = Path(args.worktree) / "scripts/mock_dark_sirens"
    if not (gmd_dir / "generate_mock_data.py").exists():
        raise SystemExit(f"generate_mock_data.py not found under {gmd_dir}")
    sys.path.insert(0, str(gmd_dir))
    import generate_mock_data as gmd  # noqa: E402

    if args.mode in ("events", "calibrate") and not args.catalog:
        raise SystemExit("--catalog is required for this mode")
    if args.mode != "calibrate" and not args.out_path:
        raise SystemExit("--out_path is required for this mode")

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator_script": str(Path(__file__).resolve()),
        "darksirens_worktree": str(Path(args.worktree)),
        "darksirens_worktree_sha": _git(args.worktree, "rev-parse", "HEAD"),
        "gmd_file": gmd.__file__,
        "gws_agn_repo_head": _git(EXP_ROOT, "rev-parse", "HEAD"),
        "args": vars(args),
    }

    print("=" * 92)
    print(f"OBSERVED-DATA DETECTION MOCK  mode={args.mode}  arm={args.detection}")
    print(f"  darksirens          : {meta['darksirens_worktree']} @ "
          f"{meta['darksirens_worktree_sha'][:10]}")
    print(f"  seed={args.seed}  sigma_dL={args.dL_fractional_uncertainty}  "
          f"snr_ref={args.snr_ref}  threshold={args.snr_threshold}")
    print("=" * 92, flush=True)

    if args.mode == "calibrate":
        rec = calibrate(gmd, args, meta)
    elif args.mode == "catalog":
        rec = build_catalog(gmd, args, meta)
    elif args.mode == "events":
        rec = build_events(gmd, args, meta)
    else:
        rec = build_injections(gmd, args, meta)

    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps({**rec, "meta": meta},
                                                      indent=2, default=str))
        print(f"wrote {args.summary_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
