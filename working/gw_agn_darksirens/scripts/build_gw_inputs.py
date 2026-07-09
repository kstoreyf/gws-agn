#!/usr/bin/env python
"""Build darksirens gwcat-1.0 GW PE files from gw_agn's event-truth + PE sets.

For each event set (recovery fagn{0.0,0.3,0.7,1.0} and coverage
gal/agn r00..r09) this reads the gw_agn glass_prod files

  * ``gws_<set>.h5``           -> datasets ``i_gw_gal``/``i_gw_agn`` = integer
                                  indices into the full ``mock_catalog.h5``
                                  ``z_gal``/``z_agn`` arrays (host truth).
  * ``gwsamples_<set>_dLunc0.1_obs.h5`` -> per-host-type PE clouds
                                  ``dL_{gal,agn}``/``ra_*``/``dec_*`` of shape
                                  ``(N_type, 2000)`` (masses there are ignored).

and writes the darksirens input contract consumed by
``darksirens.gw.utils.load_gw_samples`` (see RECON.md "GW PE files"):

  attrs  format_version="gwcat-1.0", mock_data=True, nobs, nsamp=2000,
         pe_cosmology_H0=67.74, pe_cosmology_Om0=0.3075,
         chi_eff_in_p_pe=True, chi_eff_amax=0.99
  data   dL, ra, dec  (copied VERBATIM from the PE file, event-major:
                       gal block then agn block, each event's 2000 samples
                       contiguous)
         m1det, m2det, m1src, m2src, chieff, p_pe  (rebuilt from the fiducial
                       powerlaw+peak population -- gw_agn's stored N(35,5)
                       masses are NOT used)

Mass/spin recipe (per RECON.md "GW PE files"):
  * dL/ra/dec are copied; only masses+chieff are regenerated.
  * rng = np.random.default_rng(seed_gw + MASS_SEED_OFFSET), seed_gw read from
    the truth-file attrs (recovery 5000..5003, coverage-gal 6000+, coverage-agn
    7000+); MASS_SEED_OFFSET = 900000. One rng stream per event set.
  * Events are processed in host order (gal block then agn block). For EACH
    event, in a single interleaved pass over that one rng stream:
        truth : m1src = gmd._sample_powerlaw_peak_m1(rng, 1, pop)
                q     = gmd._sample_q(rng, [m1src], pop);  m2src = q * m1src
                chi   = gmd._sample_chieff(rng, 1, pop)
                m*det_true = m*src * (1 + z_true)         (z_true = host z)
        cloud : m1det ~ N(m1det_true, 0.08*m1det_true) clip >=2
                m2det ~ N(m2det_true, 0.10*m2det_true) clip >=1
                chieff ~ N(chi_true, 0.08) clip [-1,1]        (2000 samples)
                p_pe   = 1
    which reproduces the gmd._posterior_samples mass/chieff conventions
    exactly (no per-sample m1>=m2 re-sort, matching gmd).
  * m1src/m2src datasets = m1det/(1+z_pe), m2det/(1+z_pe) where z_pe = z(dL) at
    H0=67.74/Om0=0.3075 via interp-inversion of a dense luminosity_distance
    grid (dL clipped to grid range). These are only the chi-eff-swap proxies,
    which mock_data=True skips in the loader, so a simple inversion suffices
    (follows generate_multitracer_mock.py:253-255 / generate_mock_data.py:954).

Provenance: source file paths, seeds, git SHAs and darksirens.__file__ are
stored as attrs; per-event truths are stored as EXTRA datasets true_z,
true_m1src, true_m2src, true_chieff, host_type (0=gal, 1=agn) -- the loader
ignores unknown datasets, so these are carried only for provenance/validation.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

# --- fixed conventions (RECON.md "Truth / fiducials", "GW PE files") ---------
H0_FID = 67.74
OM0_FID = 0.3075
W0_FID = -1.0
WA_FID = 0.0
NSAMP = 2000
MASS_SEED_OFFSET = 900000
CHIEFF_AMAX = 0.99
Z_GRID_MAX = 3.0          # dense z(dL) inversion grid upper bound (covers PE dL tails)
Z_GRID_N = 20_000

DARKSIRENS_REPO = "/hildafs/projects/phy230014p/magana/src/darksirens"
DARKSIRENS_MERGE_BASE = "d387b4f"
GMD_DIR = "/hildafs/projects/phy230014p/magana/src/darksirens/scripts/mock_dark_sirens"

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = REPO_ROOT / "working/gw_agn/data/glass_prod"
DEFAULT_OUT_DIR = REPO_ROOT / "working/gw_agn_darksirens/data"

# gmd is a standalone importable module (no darksirens import); RECON.md line 17.
sys.path.insert(0, GMD_DIR)
import generate_mock_data as gmd  # noqa: E402


# --- event-set registry ------------------------------------------------------
def build_sets():
    """Return list of (tag, truth_name, pe_name, out_name) for the 24 sets."""
    sets = []
    fagn_keys = ["0.0", "0.3", "0.7", "1.0"]
    fagn_seeds = [5000, 5001, 5002, 5003]
    for key, seed in zip(fagn_keys, fagn_seeds):
        sets.append((
            f"fagn{key}",
            f"gws_fagn{key}_lam0.5_seedgw{seed}.h5",
            f"gwsamples_fagn{key}_lam0.5_seedgw{seed}_dLunc0.1_obs.h5",
            f"gw_fagn{key}.h5",
        ))
    for host in ("gal", "agn"):
        for r in range(10):
            sets.append((
                f"cov_{host}_r{r:02d}",
                f"gws_cov_{host}_r{r:02d}.h5",
                f"gwsamples_cov_{host}_r{r:02d}_dLunc0.1_obs.h5",
                f"gw_cov_{host}_r{r:02d}.h5",
            ))
    return sets


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


def cosmology_grid():
    """Dense (z, dL) grid for the H0=67.74/Om0=0.3075 z(dL) interp-inversion."""
    cosmo = gmd._build_cosmology(H0_FID, OM0_FID, W0_FID, WA_FID)
    grids = gmd._cosmology_grids(cosmo, zmax=Z_GRID_MAX, ngrid=Z_GRID_N)
    return grids["z"], grids["dl"]


def z_of_dl(dL, z_grid, dl_grid):
    """z(dL) by interp-inversion; dL clipped to the grid's [dl_min, dl_max]."""
    dL_clipped = np.clip(dL, dl_grid[0], dl_grid[-1])
    return np.interp(dL_clipped, dl_grid, z_grid)


def read_truth(truth_path):
    with h5py.File(truth_path, "r") as f:
        i_gal = np.asarray(f["i_gw_gal"], dtype=np.int64)
        i_agn = np.asarray(f["i_gw_agn"], dtype=np.int64)
        attrs = {k: f.attrs[k] for k in f.attrs}
    return i_gal, i_agn, attrs


def read_pe_block(pe_path, host):
    """Read (dL, ra, dec) PE clouds for one host type; shape (N_type, 2000)."""
    with h5py.File(pe_path, "r") as f:
        n_type = int(f.attrs[f"N_gw_{host}"])
        if n_type == 0:
            return np.empty((0, NSAMP)), np.empty((0, NSAMP)), np.empty((0, NSAMP))
        dL = np.asarray(f[f"dL_{host}"], dtype=np.float64)
        ra = np.asarray(f[f"ra_{host}"], dtype=np.float64)
        dec = np.asarray(f[f"dec_{host}"], dtype=np.float64)
    return dL, ra, dec


def build_one(tag, truth_path, pe_path, out_path, mock_catalog, z_grid, dl_grid, meta_common):
    """Build one gwcat-1.0 PE file. Returns a per-file provenance dict."""
    i_gal, i_agn, truth_attrs = read_truth(truth_path)
    seed_gw = int(truth_attrs["seed_gw"])
    mass_seed = seed_gw + MASS_SEED_OFFSET

    z_gal_all = mock_catalog["z_gal"]
    z_agn_all = mock_catalog["z_agn"]

    # Host z truth in event order = gal block then agn block.
    z_true_gal = z_gal_all[i_gal]
    z_true_agn = z_agn_all[i_agn]
    z_true = np.concatenate([z_true_gal, z_true_agn])
    host_type = np.concatenate([
        np.zeros(len(i_gal), dtype=np.int8),
        np.ones(len(i_agn), dtype=np.int8),
    ])
    nobs = len(z_true)

    # Copy PE dL/ra/dec (gal block then agn block) and assert block sizes agree.
    dL_gal, ra_gal, dec_gal = read_pe_block(pe_path, "gal")
    dL_agn, ra_agn, dec_agn = read_pe_block(pe_path, "agn")
    assert dL_gal.shape[0] == len(i_gal), (
        f"{tag}: PE N_gw_gal={dL_gal.shape[0]} != truth i_gw_gal={len(i_gal)}")
    assert dL_agn.shape[0] == len(i_agn), (
        f"{tag}: PE N_gw_agn={dL_agn.shape[0]} != truth i_gw_agn={len(i_agn)}")

    def stack(gal_arr, agn_arr):
        blocks = [b for b in (gal_arr, agn_arr) if b.shape[0] > 0]
        return np.concatenate(blocks, axis=0) if blocks else np.empty((0, NSAMP))

    dL_pe = stack(dL_gal, dL_agn)      # (nobs, 2000)
    ra_pe = stack(ra_gal, ra_agn)
    dec_pe = stack(dec_gal, dec_agn)
    assert dL_pe.shape == (nobs, NSAMP), f"{tag}: dL_pe shape {dL_pe.shape}"

    # --- fiducial-population masses/chieff: one rng stream for this set -------
    pop = gmd.PopulationConfig()
    rng = np.random.default_rng(mass_seed)

    true_m1src = np.empty(nobs)
    true_m2src = np.empty(nobs)
    true_chieff = np.empty(nobs)

    m1det = np.empty((nobs, NSAMP))
    m2det = np.empty((nobs, NSAMP))
    chieff = np.empty((nobs, NSAMP))

    for i in range(nobs):
        # truth: one draw per event from the fiducial powerlaw+peak population.
        m1s = float(gmd._sample_powerlaw_peak_m1(rng, 1, pop)[0])
        q = float(gmd._sample_q(rng, np.array([m1s]), pop)[0])
        m2s = q * m1s
        chi = float(gmd._sample_chieff(rng, 1, pop)[0])
        true_m1src[i] = m1s
        true_m2src[i] = m2s
        true_chieff[i] = chi

        zt = z_true[i]
        m1det_true = m1s * (1.0 + zt)
        m2det_true = m2s * (1.0 + zt)
        # PE clouds, gmd._posterior_samples conventions (no m1>=m2 re-sort).
        m1det[i] = np.clip(rng.normal(m1det_true, 0.08 * m1det_true, NSAMP), 2.0, None)
        m2det[i] = np.clip(rng.normal(m2det_true, 0.10 * m2det_true, NSAMP), 1.0, None)
        chieff[i] = np.clip(rng.normal(chi, 0.08, NSAMP), -1.0, 1.0)

    # m1src/m2src proxies via z(dL) (only for the chi-eff swap; mock_data skips).
    z_pe = z_of_dl(dL_pe, z_grid, dl_grid)          # (nobs, 2000)
    m1src = m1det / (1.0 + z_pe)
    m2src = m2det / (1.0 + z_pe)
    p_pe = np.ones((nobs, NSAMP))

    # --- write flat, event-major (nobs*nsamp) ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["format_version"] = "gwcat-1.0"
        f.attrs["mock_data"] = True
        f.attrs["nobs"] = int(nobs)
        f.attrs["nsamp"] = int(NSAMP)
        f.attrs["pe_cosmology_H0"] = float(H0_FID)
        f.attrs["pe_cosmology_Om0"] = float(OM0_FID)
        f.attrs["chi_eff_in_p_pe"] = True
        f.attrs["chi_eff_amax"] = float(CHIEFF_AMAX)
        # provenance attrs
        f.attrs["event_set_tag"] = tag
        f.attrs["source_truth_path"] = str(truth_path)
        f.attrs["source_pe_path"] = str(pe_path)
        f.attrs["source_mock_catalog"] = str(meta_common["mock_catalog_path"])
        f.attrs["seed_gw"] = int(seed_gw)
        f.attrs["mass_seed"] = int(mass_seed)
        f.attrs["mass_seed_offset"] = int(MASS_SEED_OFFSET)
        f.attrs["host_order"] = "gal_then_agn"
        f.attrs["n_host_gal"] = int(len(i_gal))
        f.attrs["n_host_agn"] = int(len(i_agn))
        for k in ("f_agn", "lambda_agn", "z_max_gw"):
            if k in truth_attrs:
                f.attrs[f"truth_{k}"] = truth_attrs[k]
        f.attrs["generated_at_utc"] = meta_common["generated_at_utc"]
        f.attrs["gws_agn_repo_head"] = meta_common["gws_agn_repo_head"]
        f.attrs["darksirens_repo_head"] = meta_common["darksirens_repo_head"]
        f.attrs["darksirens_file"] = meta_common["darksirens_file"]
        f.attrs["darksirens_merge_base_required"] = DARKSIRENS_MERGE_BASE
        f.attrs["generator_script"] = meta_common["script"]

        for key, val in (
            ("dL", dL_pe), ("ra", ra_pe), ("dec", dec_pe),
            ("m1det", m1det), ("m2det", m2det),
            ("m1src", m1src), ("m2src", m2src),
            ("chieff", chieff), ("p_pe", p_pe),
        ):
            f.create_dataset(key, data=val.reshape(-1), compression="gzip", shuffle=True)

        # extra provenance datasets (ignored by the loader)
        f.create_dataset("true_z", data=z_true)
        f.create_dataset("true_m1src", data=true_m1src)
        f.create_dataset("true_m2src", data=true_m2src)
        f.create_dataset("true_chieff", data=true_chieff)
        f.create_dataset("host_type", data=host_type)

    return {
        "tag": tag,
        "out": str(out_path),
        "nobs": int(nobs),
        "n_host_gal": int(len(i_gal)),
        "n_host_agn": int(len(i_agn)),
        "seed_gw": int(seed_gw),
        "mass_seed": int(mass_seed),
        "truth": str(truth_path),
        "pe": str(pe_path),
        "z_true_min": float(z_true.min()),
        "z_true_max": float(z_true.max()),
        "size_mb": out_path.stat().st_size / 1e6,
    }


def validate_file(out_path, z_grid, dl_grid, n_probe=3):
    """Load with the real loader; check shapes/finiteness and z(dL) recovery."""
    from darksirens.gw.utils import load_gw_samples

    (m1det, m2det, dL, chieff, ra, dec, p_pe, nEvents, nsamp) = load_gw_samples(str(out_path))
    arrs = {
        "m1det": np.asarray(m1det), "m2det": np.asarray(m2det), "dL": np.asarray(dL),
        "chieff": np.asarray(chieff), "ra": np.asarray(ra), "dec": np.asarray(dec),
        "p_pe": np.asarray(p_pe),
    }
    all_finite = all(np.all(np.isfinite(a)) for a in arrs.values())

    with h5py.File(out_path, "r") as f:
        nobs_attr = int(f.attrs["nobs"])
        true_z = np.asarray(f["true_z"])

    dL_ev = arrs["dL"].reshape(nEvents, nsamp)
    z_pe = z_of_dl(dL_ev, z_grid, dl_grid)          # (nEvents, 2000)
    med_z = np.median(z_pe, axis=1)
    sigma_z = np.maximum(0.07 * true_z, 1e-3)       # ~10% dL error -> sigma_z ~ 0.07 z
    dz_in_sigma = (med_z - true_z) / sigma_z

    # 3 probe events spread across the file
    if nEvents >= n_probe:
        idx = np.linspace(0, nEvents - 1, n_probe).round().astype(int)
    else:
        idx = np.arange(nEvents)
    probes = [
        (int(j), float(true_z[j]), float(med_z[j]), float(dz_in_sigma[j])) for j in idx
    ]

    return {
        "tag": out_path.stem,
        "nEvents": int(nEvents),
        "nsamp": int(nsamp),
        "nobs_attr": nobs_attr,
        "ok_shapes": (nEvents == nobs_attr) and (nsamp == NSAMP),
        "all_finite": bool(all_finite),
        "mean_abs_dz_sigma": float(np.mean(np.abs(dz_in_sigma))),
        "max_abs_dz_sigma": float(np.max(np.abs(dz_in_sigma))),
        "frac_within_1p5sigma": float(np.mean(np.abs(dz_in_sigma) <= 1.5)),
        "probes": probes,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--mock-catalog", type=Path, default=None)
    p.add_argument("--no-validate", action="store_true")
    args = p.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    mock_catalog_path = args.mock_catalog or (data_dir / "mock_catalog.h5")

    # merge-base guard (RECON.md line 14)
    mb = subprocess.run(
        ["git", "-C", DARKSIRENS_REPO, "merge-base", "--is-ancestor",
         DARKSIRENS_MERGE_BASE, "HEAD"])
    assert mb.returncode == 0, f"{DARKSIRENS_MERGE_BASE} is not an ancestor of darksirens HEAD"

    with h5py.File(mock_catalog_path, "r") as f:
        mock_catalog = {
            "z_gal": np.asarray(f["z_gal"], dtype=np.float64),
            "z_agn": np.asarray(f["z_agn"], dtype=np.float64),
        }

    z_grid, dl_grid = cosmology_grid()

    meta_common = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "mock_catalog_path": str(mock_catalog_path),
        "gws_agn_repo_head": git_head(REPO_ROOT),
        "gws_agn_repo_branch": git_branch(REPO_ROOT),
        "darksirens_repo_head": git_head(DARKSIRENS_REPO),
        "darksirens_repo_branch": git_branch(DARKSIRENS_REPO),
        "darksirens_file": gmd.__file__,
        "cosmology": {"H0": H0_FID, "Om0": OM0_FID, "w0": W0_FID, "wa": WA_FID},
        "nsamp": NSAMP,
        "mass_seed_offset": MASS_SEED_OFFSET,
        "chi_eff_amax": CHIEFF_AMAX,
    }

    sets = build_sets()
    file_records = []
    print(f"Building {len(sets)} gwcat-1.0 PE files -> {out_dir}")
    for tag, truth_name, pe_name, out_name in sets:
        truth_path = data_dir / truth_name
        pe_path = data_dir / pe_name
        out_path = out_dir / out_name
        rec = build_one(tag, truth_path, pe_path, out_path,
                        mock_catalog, z_grid, dl_grid, meta_common)
        file_records.append(rec)
        print(f"  [{tag:16s}] nobs={rec['nobs']:4d} "
              f"(gal={rec['n_host_gal']:4d} agn={rec['n_host_agn']:4d}) "
              f"mass_seed={rec['mass_seed']} z_true=[{rec['z_true_min']:.3f},"
              f"{rec['z_true_max']:.3f}] -> {out_name} ({rec['size_mb']:.2f} MB)")

    meta = {**meta_common, "files": file_records}
    meta_path = out_dir / "gw_inputs_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=False)
    print(f"wrote {meta_path}")

    if args.no_validate:
        return 0

    # ----------------------------- validation --------------------------------
    print("\n" + "=" * 100)
    print("VALIDATION via darksirens.gw.utils.load_gw_samples "
          "(median z(dL) at H0=67.74 vs true_z; sigma_z ~ 0.07*z)")
    print("=" * 100)
    header = (f"{'file':22s} {'nEv':>5s} {'nsamp':>6s} {'shapes':>7s} "
              f"{'finite':>7s} {'mean|dz|/s':>11s} {'max|dz|/s':>10s} {'frac<=1.5s':>11s}")
    print(header)
    print("-" * len(header))
    val_records = []
    all_ok = True
    for rec in file_records:
        v = validate_file(Path(rec["out"]), z_grid, dl_grid)
        val_records.append(v)
        ok = v["ok_shapes"] and v["all_finite"]
        all_ok = all_ok and ok
        print(f"{v['tag']:22s} {v['nEvents']:5d} {v['nsamp']:6d} "
              f"{str(v['ok_shapes']):>7s} {str(v['all_finite']):>7s} "
              f"{v['mean_abs_dz_sigma']:11.3f} {v['max_abs_dz_sigma']:10.3f} "
              f"{v['frac_within_1p5sigma']:11.3f}")

    # detailed 3-event probes for two representative files
    print("\nPer-event z(dL) probes (event_idx: true_z -> med_z_pe  [dz/sigma_z]):")
    for v in val_records:
        if v["tag"] in ("gw_fagn0.3", "gw_cov_gal_r00", "gw_cov_agn_r00", "gw_fagn0.0",
                        "gw_fagn0.7", "gw_fagn1.0"):
            parts = "  ".join(
                f"ev{j}: {tz:.3f}->{mz:.3f} [{dz:+.2f}]" for (j, tz, mz, dz) in v["probes"])
            print(f"  {v['tag']:16s} {parts}")

    val_path = out_dir / "gw_inputs_validation.json"
    with open(val_path, "w") as f:
        json.dump(val_records, f, indent=2, sort_keys=False)
    print(f"\nwrote {val_path}")
    print(f"\nALL FILES VALID: {all_ok}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
