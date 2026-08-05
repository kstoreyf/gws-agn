#!/usr/bin/env python3
"""PROBE 4 -- analytic continuum GAL surveys (GPU).

Question: does the matched-GAL H0 offset survive when the catalog's redshift
structure is replaced by a PERFECTLY SMOOTH, analytically known continuum?

Two synthetic survey files are built with the seed-100 GAL block's on-disk
conventions EXACTLY -- nside 32, dz = 3e-3 (1+z), float32, z-sorted real prefix,
100.0 / 1.0 / 0.0 padding sentinels, ``ngals`` = the number of real slots:

  4a  per-pixel z-continuum.  Every pixel keeps its REAL galaxy count
      ``ngals[pix]`` (so the sky structure -- the field weight
      ``N_obs[pix]/N_obs_total`` -- is untouched), but its ``ngals[pix]``
      redshifts are replaced by the QUANTILES of the analytic
      ``dN/dz ∝ dV_c/dz`` on [0, z_max]:  z_i = F^-1((i+1/2)/n).  Deterministic
      quantile placement carries neither clustering nor shot noise in z, and with
      unit weights the resulting kernel mixture is the analytic continuum to
      O(sigma^2 (ln g)'').

  4b  fully uniform sky.  Every pixel is IDENTICAL: the same continuum rows and
      the same count (the survey mean, so the total host number is preserved to
      within rounding).  Sky structure and z structure are both gone; the prior
      is a known smooth function of z alone.

Why quantiles and not "a fine z grid with weights ∝ dV_c/dz": darksirens takes
the per-pixel amplitude from ``ngals`` (``prior._row_counts``) and its real-slot
mask is ``arange < ngals`` (``catalog._row_real_mask``), so the row length IS the
pixel count -- a short weighted grid would silently rewrite every pixel's field
weight.  Unit weights on the count-many quantiles keep both the amplitude and the
shape analytic.

The seed-100 matched-GAL events and the targeted injection lane are unchanged
(the targeted branch is built from the AGN survey's kernels, so it does not know
about the GAL catalog at all), as are the H0 grid, the window and the guard.

Reading.  If the offset PERSISTS on 4b -- a perfectly known smooth prior -- the
defect is in likelihood/convention territory and is reproducible analytically.
If it VANISHES, the clustering/realisation interaction is implicated.

  build     : write the two synthetic survey files
  scan      : run the two scans
  analyse   : compare against the real-catalog control and write the JSON + figure
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data")
BULK = Path("/hildafs/projects/phy220048p/magana/gws-agn-data/derived/"
            "analysis_1_complete_catalog_H0/probe4")

DZ_SCALE = 3.0e-3
Z_PAD, DZ_PAD, W_PAD = np.float32(100.0), np.float32(1.0), np.float32(0.0)
OM0_FID = 0.3075
H0_TRUE = 67.74


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["build", "scan", "analyse", "all"])
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--kde_window", type=int, default=4096)
    ap.add_argument("--sel_batch_size", type=int, default=200000)
    ap.add_argument("--pe_event_block", type=int, default=100)
    ap.add_argument("--h0_grid", nargs=3, type=float, default=[50.0, 100.0, 201])
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args(argv)


# --------------------------------------------------------------------------- #
# the analytic continuum
# --------------------------------------------------------------------------- #
def comoving_volume_cdf(z_max, om0=OM0_FID, n=400_001):
    """CDF of dN/dz ∝ dV_c/dz on [0, z_max] for flat wCDM with w0 = -1.

    dV_c/dz ∝ D_c(z)^2 / E(z) with D_c ∝ ∫ dz/E; the H0 scale cancels in the
    normalised CDF, so the SHAPE depends only on Om0 -- there is no H0
    convention to get wrong here.
    """
    zg = np.linspace(0.0, float(z_max), n)
    E = np.sqrt(om0 * (1.0 + zg) ** 3 + (1.0 - om0))
    inv = 1.0 / E
    Dc = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(zg))])
    dVdz = Dc ** 2 * inv
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dVdz[1:] + dVdz[:-1]) * np.diff(zg))])
    cdf /= cdf[-1]
    return zg, cdf, dVdz


def continuum_quantiles(n, zg, cdf):
    """The n mid-point quantiles of the continuum -- ascending, float32."""
    u = (np.arange(n, dtype=np.float64) + 0.5) / n
    return np.interp(u, cdf, zg).astype(np.float32)


def write_survey(path, zgals, ngals, nside, provenance, z_ref):
    import h5py
    dzgals = np.where(zgals == Z_PAD, DZ_PAD,
                      (np.float32(DZ_SCALE) * (np.float32(1.0) + zgals)))
    wgals = np.where(zgals == Z_PAD, W_PAD, np.float32(1.0)).astype(np.float32)
    dzgals = dzgals.astype(np.float32)
    real = np.arange(zgals.shape[1])[None, :] < ngals[:, None]
    assert np.all(zgals[~real] == Z_PAD), "padding slots not at the sentinel"
    # darksirens' own invariant (catalogs/io.sort_survey_rows_by_z): the real
    # prefix is non-decreasing; the real->padding step is exempt.
    cols = np.arange(1, zgals.shape[1])[None, :]
    assert bool(np.all((np.diff(zgals, axis=1) >= 0)
                       | (cols >= ngals[:, None]))), "row z-sort invariant violated"
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for k, v in (("zgals", zgals), ("dzgals", dzgals),
                     ("wgals", wgals), ("ngals", ngals.astype(np.int32))):
            f.create_dataset(k, data=v, compression="gzip", shuffle=True)
        f.attrs["nside"] = int(nside)
        f.attrs["tracer"] = "gal"
        f.attrs["level"] = "complete"
        f.attrs["mag_limit"] = -1.0
        f.attrs["n_hosts"] = int(ngals.sum())
        f.attrs["z_min"] = float(zgals[real].min())
        f.attrs["z_max"] = float(zgals[real].max())
        f.attrs["dz_scale"] = float(DZ_SCALE)
        f.attrs["dz_convention"] = f"dz = {DZ_SCALE} * (1 + z)"
        f.attrs["occupied_pixels"] = int((ngals > 0).sum())
        f.attrs["empty_pixel_fraction"] = float(1.0 - (ngals > 0).mean())
        f.attrs["synthetic"] = True
        f.attrs["z_reference_catalog"] = z_ref
        for k, v in provenance.items():
            f.attrs[k] = v
    return path


def stage_build(args):
    import h5py
    src = DATA / f"seed{args.seed}" / "surveys" / "survey_gal_complete_ns32.h5"
    with h5py.File(src, "r") as f:
        ngals_real = np.asarray(f["ngals"]).astype(np.int64)
        nside = int(f.attrs["nside"])
        z_max = float(f.attrs["z_max"])
        z_min_real = float(f.attrs["z_min"])
        n_hosts = int(f.attrs["n_hosts"])
        zg_real_row0 = None
    print(f"[build] source {src}\n        npix={ngals_real.size} "
          f"ngals in [{ngals_real.min()}, {ngals_real.max()}] sum={ngals_real.sum():,} "
          f"z_max={z_max:.6f}")

    zg, cdf, dVdz = comoving_volume_cdf(z_max)

    # --- how faithful is the analytic continuum to the REAL global dN/dz? ----
    #     (the low-z ramp / catalog-edge check the README flags).  Also build
    #     the EMPIRICAL global CDF, used by the 4b-emp control arm.
    with h5py.File(src, "r") as f:
        Z = np.asarray(f["zgals"])
    real = np.arange(Z.shape[1])[None, :] < ngals_real[:, None]
    zr = Z[real].astype(np.float64)
    fine = np.linspace(0.0, z_max, 2001)          # 5e-4 bins; kernel is 3e-3
    hfine, _ = np.histogram(zr, bins=fine)
    cdf_emp = np.concatenate([[0.0], np.cumsum(hfine)]).astype(np.float64)
    cdf_emp /= cdf_emp[-1]

    edges = np.linspace(0.0, z_max, 81)
    hist, _ = np.histogram(zr, bins=edges)
    frac_real = hist / hist.sum()
    frac_ana = np.diff(np.interp(edges, zg, cdf))
    ratio = np.where(frac_ana > 0, frac_real / np.maximum(frac_ana, 1e-30), np.nan)
    plateau = (edges[:-1] >= 0.0457) & (edges[1:] <= 0.9230)
    ev_band = (edges[:-1] >= 0.0457) & (edges[1:] <= 0.40)
    fidelity = {
        "bin_edges": edges.tolist(),
        "frac_real": frac_real.tolist(),
        "frac_analytic": frac_ana.tolist(),
        "ratio_real_over_analytic": np.where(np.isfinite(ratio), ratio, None).tolist(),
        "note": ("On the constant-comoving-density plateau the ratio is a CONSTANT "
                 "(the analytic CDF is normalised over the whole [0, z_max] while "
                 "the real catalog is deficient at both ends), so the SHAPE agrees "
                 "there to the quoted spread.  The ends do differ: a low-z deficit "
                 "below the first 200 Mpc GLASS shell edge (z ~ 0.0457) and the "
                 "partial last shell above z ~ 0.923.  Arm 4b-emp replaces the "
                 "analytic CDF with the catalog's own (still perfectly smooth, "
                 "still clustering-free) global dN/dz to control for exactly this."),
        "plateau_ratio_mean": float(np.nanmean(ratio[plateau])),
        "plateau_ratio_spread_max_abs_rel": float(
            np.nanmax(np.abs(ratio[plateau] / np.nanmean(ratio[plateau]) - 1.0))),
        "event_band_ratio_spread_max_abs_rel": float(
            np.nanmax(np.abs(ratio[ev_band] / np.nanmean(ratio[plateau]) - 1.0))),
        "catalog_fraction_below_z0p0457_real": float(frac_real[edges[1:] <= 0.0501].sum()),
        "catalog_fraction_below_z0p0457_analytic": float(frac_ana[edges[1:] <= 0.0501].sum()),
    }
    del Z, zr, real
    print(f"[build] analytic-vs-real dN/dz: plateau ratio "
          f"{fidelity['plateau_ratio_mean']:.4f} +- "
          f"{fidelity['plateau_ratio_spread_max_abs_rel']:.4f} (max rel spread); "
          f"event band {fidelity['event_band_ratio_spread_max_abs_rel']:.4f}")

    prov = {"built_by": str(Path(__file__).resolve()),
            "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_survey": str(src),
            "Om0_for_continuum": OM0_FID,
            "continuum": "dN/dz proportional to dV_c/dz, mid-point quantiles, "
                         "unit weights"}

    out = {"fidelity_vs_real_global_dNdz": fidelity, "files": {}}

    # ---- 4a: real per-pixel counts, continuum redshifts ---------------------
    p4a = BULK / f"survey_gal_probe4a_continuum_s{args.seed}_ns32.h5"
    if p4a.exists() and not args.overwrite:
        print(f"[build] {p4a} exists; reusing")
    else:
        nmax = int(ngals_real.max())
        Zs = np.full((ngals_real.size, nmax), Z_PAD, dtype=np.float32)
        t0 = time.time()
        cache = {}
        for r, n in enumerate(ngals_real):
            n = int(n)
            if n == 0:
                continue
            q = cache.get(n)
            if q is None:
                q = continuum_quantiles(n, zg, cdf)
                if len(cache) < 4000:
                    cache[n] = q
            Zs[r, :n] = q
        print(f"[build] 4a rows built in {time.time()-t0:.1f}s "
              f"({len(cache)} distinct counts cached)")
        write_survey(p4a, Zs, ngals_real, nside,
                     {**prov, "variant": "4a per-pixel z-continuum, real counts"},
                     str(src))
        print(f"[build] wrote {p4a} ({p4a.stat().st_size/1e6:.0f} MB)")
        del Zs
    out["files"]["4a"] = str(p4a)

    # ---- 4b: fully uniform sky ---------------------------------------------
    p4b = BULK / f"survey_gal_probe4b_uniform_s{args.seed}_ns32.h5"
    if p4b.exists() and not args.overwrite:
        print(f"[build] {p4b} exists; reusing")
    else:
        npix = ngals_real.size
        n_uni = int(round(n_hosts / npix))
        q = continuum_quantiles(n_uni, zg, cdf)
        Zs = np.broadcast_to(q, (npix, n_uni)).copy()
        ng = np.full(npix, n_uni, dtype=np.int64)
        write_survey(p4b, Zs, ng, nside,
                     {**prov, "variant": "4b uniform sky, identical continuum "
                                         "row in every pixel"},
                     str(src))
        print(f"[build] wrote {p4b} ({p4b.stat().st_size/1e6:.0f} MB); "
              f"n per pixel = {n_uni}, total {n_uni*npix:,} "
              f"(real {n_hosts:,})")
        del Zs
    out["files"]["4b"] = str(p4b)

    # ---- 4b-emp: uniform sky, the catalog's OWN global dN/dz ----------------
    # Control for 4b's only known mis-specification: the analytic dV_c/dz form
    # differs from the real catalog at the two shell ends.  Same construction,
    # same uniform sky, quantiles taken from the empirical global CDF instead --
    # still deterministic, still clustering- and shot-noise-free in z.
    p4be = BULK / f"survey_gal_probe4bemp_uniform_s{args.seed}_ns32.h5"
    if p4be.exists() and not args.overwrite:
        print(f"[build] {p4be} exists; reusing")
    else:
        npix = ngals_real.size
        n_uni = int(round(n_hosts / npix))
        u = (np.arange(n_uni, dtype=np.float64) + 0.5) / n_uni
        q = np.interp(u, cdf_emp, fine).astype(np.float32)
        Zs = np.broadcast_to(q, (npix, n_uni)).copy()
        ng = np.full(npix, n_uni, dtype=np.int64)
        write_survey(p4be, Zs, ng, nside,
                     {**prov, "variant": "4b-emp uniform sky, EMPIRICAL global "
                                         "dN/dz of the real seed-100 GAL catalog "
                                         "(2000-bin CDF), identical row in every pixel",
                      "continuum": "dN/dz = the real catalog's own global dN/dz"},
                     str(src))
        print(f"[build] wrote {p4be} ({p4be.stat().st_size/1e6:.0f} MB)")
        del Zs
    out["files"]["4b_emp"] = str(p4be)

    out["counts"] = {
        "npix": int(ngals_real.size),
        "ngals_real_min": int(ngals_real.min()),
        "ngals_real_max": int(ngals_real.max()),
        "ngals_real_sum": int(ngals_real.sum()),
        "uniform_n_per_pixel": int(round(n_hosts / ngals_real.size)),
        "z_max": z_max, "z_min_real_catalog": z_min_real,
    }
    p = ROOT / "results" / "probe4_build.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"[build] wrote {p}")
    return out


# --------------------------------------------------------------------------- #
def stage_scan(args):
    build = json.loads((ROOT / "results" / "probe4_build.json").read_text())
    sd = DATA / f"seed{args.seed}"
    jobs = [("probe4a_gal_continuum", build["files"]["4a"]),
            ("probe4b_gal_uniform", build["files"]["4b"]),
            ("probe4bemp_gal_uniform", build["files"]["4b_emp"])]
    for tag, survey in jobs:
        h5 = ROOT / "results" / f"{tag}.h5"
        if h5.exists():
            print(f"[scan] {tag} exists; skipping")
            continue
        cmd = [sys.executable, str(HERE / "scan_h0f.py"),
               "--universe_model", "dark_sirens_complete",
               "--catalog_sky_weighting", "field",
               "--scan", "h0",
               "--h0_grid", *[str(x) for x in args.h0_grid],
               "--h0_true", str(H0_TRUE),
               "--survey_path", survey,
               "--gw_path", str(ROOT / "data_derived" / "events_gal_hosted.h5"),
               "--gwselection_path", str(sd / "injections" / "injections_targeted.h5"),
               "--selection_neff_guard", "hard",
               "--max_likelihood_variance", "1e6",
               "--sel_batch_size", str(args.sel_batch_size),
               "--pe_event_block", str(args.pe_event_block),
               "--kde_window", str(args.kde_window),
               "--kde_window_nsigma", "8.0",
               "--outdir", str(ROOT / "results"),
               "--out_tag", tag]
        print(f"[{time.strftime('%H:%M:%S')}] scanning {tag}")
        t0 = time.time()
        with open(ROOT / "logs" / f"{tag}.log", "w") as fh:
            subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.STDOUT,
                           cwd=str(ROOT))
        print(f"[{time.strftime('%H:%M:%S')}] {tag} done in {time.time()-t0:.0f}s")


# --------------------------------------------------------------------------- #
def summarise(H0, logL):
    ok = np.isfinite(logL)
    ll = np.where(ok, logL, -np.inf)
    p = np.exp(ll - np.max(ll[ok]))
    p = np.where(ok, p, 0.0)
    p = p / np.trapz(p, H0)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(H0))])
    cdf /= cdf[-1]
    q = lambda t: float(np.interp(t, cdf, H0))            # noqa: E731
    imap = int(np.argmax(ll))
    return {"map": float(H0[imap]), "map_at_edge": imap in (0, len(H0) - 1),
            "median": q(0.5), "ci68": [q(0.16), q(0.84)], "ci90": [q(0.05), q(0.95)],
            "offset": q(0.5) - H0_TRUE,
            "half68": 0.5 * (q(0.84) - q(0.16)),
            "truth_in_ci68": bool(q(0.16) <= H0_TRUE <= q(0.84)),
            "truth_in_ci90": bool(q(0.05) <= H0_TRUE <= q(0.95)),
            "mass_in_lowest_cell": float(p[0] * (H0[1] - H0[0]) )}


def lowz_audit(args):
    """Per-pixel galaxy counts below a redshift ladder, real vs each synthetic
    survey.  A dense catalog is only 'dense' where the volume is: inside the GW
    horizon a nside-32 pixel holds ~1e3 galaxies and below z ~ 0.05 it holds a
    handful, so this records how discrete the catalog prior actually is where the
    events live -- and that the continuum arms match the real counts there rather
    than inventing coverage."""
    import h5py
    bp = ROOT / "results" / "probe4_build.json"
    b = json.loads(bp.read_text())
    if "lowz_audit" in b and not args.overwrite:
        return b["lowz_audit"]
    ladder = [0.02, 0.05, 0.10, 0.1320, 0.20, 0.3565]
    files = {"real": str(DATA / f"seed{args.seed}" / "surveys"
                         / "survey_gal_complete_ns32.h5")}
    files.update(b["files"])
    audit = {"ladder": ladder,
             "note": "per-pixel counts of catalog galaxies below each z "
                     "(0.1320 = the events' median z; 0.3565 = the GW horizon)"}
    for name, p in files.items():
        with h5py.File(p, "r") as f:
            Z = np.asarray(f["zgals"]); NG = np.asarray(f["ngals"]).astype(np.int64)
        real = np.arange(Z.shape[1])[None, :] < NG[:, None]
        rec = {}
        for zq in ladder:
            c = ((Z < zq) & real).sum(axis=1)
            rec[f"z<{zq:g}"] = {"mean": float(c.mean()), "min": int(c.min()),
                                "max": int(c.max()),
                                "sd": float(c.std()),
                                "n_pixels_zero": int((c == 0).sum())}
        audit[name] = rec
        del Z, real
    b["lowz_audit"] = audit
    bp.write_text(json.dumps(b, indent=2))
    print("[lowz]", json.dumps({k: v for k, v in audit.items()
                                if k not in ("ladder", "note")}, indent=1))
    return audit


def stage_analyse(args):
    import h5py
    try:
        lowz_audit(args)
    except Exception as exc:                       # diagnostic only
        print(f"[lowz] skipped: {exc}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = {"real_catalog (ctrl_gal_matched)": "ctrl_gal_matched",
            "4a per-pixel continuum": "probe4a_gal_continuum",
            "4b uniform sky continuum": "probe4b_gal_uniform",
            "4b-emp uniform sky, empirical dN/dz": "probe4bemp_gal_uniform"}
    res = {"probe": 4, "name": "analytic_continuum_survey", "seed": args.seed,
           "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "events": "data_derived/events_gal_hosted.h5 (720 GAL-hosted, seed 100)",
           "injections": "injections_targeted.h5",
           "arms": {}}
    bp = ROOT / "results" / "probe4_build.json"
    if bp.exists():
        b = json.loads(bp.read_text())
        res["lowz_audit"] = b.get("lowz_audit")
        res["build"] = {"counts": b["counts"],
                        "fidelity_vs_real_global_dNdz": {
                            k: v for k, v in b["fidelity_vs_real_global_dNdz"].items()
                            if not k.startswith(("frac_", "bin_"))},
                        "files": b["files"]}
    curves = {}
    for name, tag in arms.items():
        p = ROOT / "results" / f"{tag}.h5"
        if not p.exists():
            print(f"[miss] {p}")
            continue
        with h5py.File(p, "r") as f:
            H0 = np.asarray(f["H0_grid"]); ll = np.asarray(f["log_likelihood"])
            neff = np.asarray(f["guard/Neff"]); rej = np.asarray(f["guard/rejected"])
        curves[name] = (H0, ll)
        res["arms"][name] = {"tag": tag, "n_rejected": int(rej.sum()),
                             "min_Neff": float(neff.min()), **summarise(H0, ll)}

    if "4b uniform sky continuum" in res["arms"]:
        off_real = res["arms"]["real_catalog (ctrl_gal_matched)"]["offset"]
        off_4a = res["arms"].get("4a per-pixel continuum", {}).get("offset")
        off_4b = res["arms"]["4b uniform sky continuum"]["offset"]
        off_4be = res["arms"].get("4b-emp uniform sky, empirical dN/dz",
                                  {}).get("offset")
        res["verdict"] = {
            "offset_real_catalog": off_real,
            "offset_4a": off_4a,
            "offset_4b": off_4b,
            "offset_4b_emp": off_4be,
            "fraction_of_real_offset_surviving_4a": (off_4a / off_real
                                                     if off_4a is not None else None),
            "fraction_of_real_offset_surviving_4b": off_4b / off_real,
            "fraction_of_real_offset_surviving_4b_emp": (off_4be / off_real
                                                         if off_4be is not None else None),
        }
        persists_4b = abs(off_4b) > 0.5 * abs(off_real)
        closes_4be = (off_4be is not None and abs(off_4be) < 0.25 * abs(off_real))
        if persists_4b and closes_4be:
            summary = (
                "SPLIT READING. The offset SURVIVES a perfectly smooth analytic "
                "dV_c/dz continuum (so it is not a clustering, pixelation or KDE "
                "artefact and is reproducible with no catalog realisation at all), "
                "but it VANISHES when the same perfectly smooth continuum is built "
                "from the catalog's OWN measured dN/dz. The estimator is therefore "
                "hypersensitive to the effective per-pixel redshift prior: the two "
                "smooth priors differ by <1% in shape across the event band and by "
                "the shell-edge ends, and that moves the recovered H0 by ~7.")
        elif persists_4b:
            summary = ("BIAS PERSISTS ON A PERFECTLY SMOOTH KNOWN PRIOR -- "
                       "likelihood/convention territory, reproducible analytically")
        elif abs(off_4b) < 0.25 * abs(off_real):
            summary = ("BIAS LARGELY VANISHES ON THE CONTINUUM -- the "
                       "clustering / catalog-realisation interaction is implicated")
        else:
            summary = ("PARTIAL: the continuum removes some but not most of the "
                       "offset")
        res["verdict"]["summary"] = summary

    # ---- probe 3's decomposition re-run on these same arms ------------------
    # (scripts/run_probe4_decomp.sh; --survey_override on probe3_decomposition.py)
    # This is what turns "the answer moved" into "which term moved".
    decomp = {"": "real_catalog (ctrl_gal_matched)",
              "_p4a": "4a per-pixel continuum",
              "_p4b": "4b uniform sky continuum",
              "_p4bemp": "4b-emp uniform sky, empirical dN/dz"}
    res["decomposition_by_arm"] = {}
    for suf, name in decomp.items():
        dp = ROOT / "results" / f"probe3_decomp_gal_s{args.seed}{suf}.json"
        if not dp.exists():
            continue
        d = json.loads(dp.read_text())
        n = d["nobs"]
        s = d["at_truth"]["dnumerator_dH0"] / n
        m = d["dlnmu_dH0_at_truth"]
        H0g = np.asarray(d["H0_grid"]); lm = np.asarray(d["log_mu"])
        res["decomposition_by_arm"][name] = {
            "source": str(dp), "nobs": n,
            "peak_total": d["peak_total"], "offset_total": d["offset_total"],
            "per_event_numerator_slope_at_truth": s,
            "dlnmu_dH0_at_truth": m,
            "score_residual_per_event": s - m,
            "score_residual_relative": s / m - 1.0,
            "d2total_dH02_per_event": d["at_truth"]["d2total_dH02_per_event"],
            "log_mu_at_truth": float(lm[int(np.argmin(np.abs(H0g - 67.74)))]),
        }

    p = ROOT / "results" / "probe4_continuum.json"
    p.write_text(json.dumps(res, indent=2))
    print(json.dumps(res.get("decomposition_by_arm", {}), indent=2))
    print(json.dumps(res.get("arms", {}), indent=2))
    print(json.dumps(res.get("verdict", {}), indent=2))
    print(f"Wrote {p}")

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for name, (H0, ll) in curves.items():
        p_ = np.exp(ll - np.nanmax(ll))
        p_ = p_ / np.trapz(p_, H0)
        ax.plot(H0, p_, lw=1.7, label=f"{name}  (median {res['arms'][name]['median']:.2f})")
    ax.axvline(H0_TRUE, color="k", ls=":", lw=1.2, label="truth 67.74")
    ax.set_xlabel(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_ylabel(r"$p(H_0)$")
    ax.set_title("Probe 4 — seed-100 matched-GAL events against analytic continuum catalogs")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figs" / f"probe4_continuum.{ext}", dpi=150)
    print("Wrote figs/probe4_continuum.{png,pdf}")


def main(argv=None):
    args = parse_args(argv)
    if args.stage in ("build", "all"):
        stage_build(args)
    if args.stage in ("scan", "all"):
        stage_scan(args)
    if args.stage in ("analyse", "all"):
        stage_analyse(args)


if __name__ == "__main__":
    main()
