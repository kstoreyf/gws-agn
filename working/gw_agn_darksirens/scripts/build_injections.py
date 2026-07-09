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
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import astropy.units as u

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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--no-validate", action="store_true")
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
