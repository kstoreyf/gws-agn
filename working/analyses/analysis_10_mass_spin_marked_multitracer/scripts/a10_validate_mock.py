#!/usr/bin/env python
"""Analysis 10 -- validation record for the branch-conditioned mass-mark mock.

Twelve registered checks on ``seed100/events/events_marked_dmu0p10_dmuG5.h5``, the
seed-100 mock whose AGN-hosted branch draws its primary mass from a Gaussian peak at
mu_G = 40 while the GAL-hosted branch keeps mu_G = 35, both branches carrying the
Analysis-8 spin mark dmu_chi_agn = +0.10.

Deterministic and CPU-only: it reads the five HDF5 products, re-derives the two
population configurations from the pinned darksirens checkout, re-runs the primary-mass
sampler on a fixed seed, and compares md5 digests against a baseline recorded before the
generator was touched.  It evaluates no likelihood and writes nothing outside
``analysis_10_mass_spin_marked_multitracer/diagnostics/``.

    export PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-a8
    export JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
    python scripts/a10_validate_mock.py
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import h5py
import numpy as np
from scipy import stats

REPO = Path("/hildafs/projects/phy230014p/magana/gws-agn")
DARKSIRENS = Path("/hildafs/projects/phy230014p/magana/src/darksirens-a8")
DATA = REPO / "working/data/seed100"
EV = DATA / "events"
A10 = REPO / "working/analyses/analysis_10_mass_spin_marked_multitracer"
GEN = REPO / "working/data/generate_dataset.py"

NEW = EV / "events_marked_dmu0p10_dmuG5.h5"          # the production mock
CTRL = EV / "events_a10ctrl_dmu0p10_dmuG0.h5"        # patched generator, dmu_G_agn = 0
HEADCTRL = EV / "events_a10headctrl_dmu0p10.h5"      # pristine HEAD generator, same node
A8 = EV / "events_marked_dmu0p10.h5"                 # the Analysis-8 marked mock

PLANTED_F_AGN = 0.30
PLANTED_DMU_CHI = 0.10
PLANTED_DMU_G = 5.0
MU_G_GAL = 35.0
SEED_EVENTS_RECORD = 100 * 1000 + 3
NEW_ATTRS = {"dmu_G_agn", "mu_G_gal", "mu_G_agn"}
SAMPLER_N = 2_000_000
SAMPLER_SEED = 20261010
SAMPLER_NBINS = 100

# The three generator runs behind this record.  Each was a single-core CPU job on the
# SLURM RM partition, all three pinned to the same node so the comparison in check 12 is
# free of the 1-ULP pow-kernel drift between nodes that Analysis 8 GATES.md Gate B
# documents.
RUNS = [
    {"role": "bitwise control (patched generator at the default)",
     "argv": "--seed 100 --stage events --darksirens .../darksirens-a8 "
             "--dmu_chi_agn 0.10 --dmu_G_agn 0.0 "
             "--events_suffix _a10ctrl_dmu0p10_dmuG0",
     "generator": "working/data/generate_dataset.py (patched)",
     "slurm_job": 1335487, "host": "r008.opa.vera.psc.edu",
     "wall_seconds": 44, "events_stage_seconds": 6.9},
    {"role": "pristine reference (HEAD generator, same node)",
     "argv": "--seed 100 --stage events --outroot working/data "
             "--darksirens .../darksirens-a8 --dmu_chi_agn 0.10 "
             "--events_suffix _a10headctrl_dmu0p10",
     "generator": "git show HEAD:working/data/generate_dataset.py (unpatched)",
     "slurm_job": 1335488, "host": "r008.opa.vera.psc.edu",
     "wall_seconds": 23, "events_stage_seconds": 6.0},
    {"role": "production mock",
     "argv": "--seed 100 --stage events --darksirens .../darksirens-a8 "
             "--dmu_chi_agn 0.10 --dmu_G_agn 5.0 "
             "--events_suffix _marked_dmu0p10_dmuG5",
     "generator": "working/data/generate_dataset.py (patched)",
     "slurm_job": 1335489, "host": "r008.opa.vera.psc.edu",
     "wall_seconds": 24, "events_stage_seconds": 7.1},
]

# md5 digests of every file Analyses 0-9 stand on, recorded on 2026-09-21 BEFORE
# generate_dataset.py was edited.  events/events_marked_dmu0p10.h5 reproduces the digest
# Analysis 8 recorded for it in GATES.md (7dcb8bcc...), so the baseline is anchored to a
# value that predates this session.
BASELINE_MD5 = {
    "events/events.h5": "adeabf81c627778092fdc76015ec6c36",
    "events/events_marked_dmu0p10.h5": "7dcb8bccba7f7a360a6da2741d4cf9d1",
    "injections/injections_popuni.h5": "cb1991fbf431fb04bf3975be98e86cc8",
    "injections/injections_targeted.h5": "e8a611a27f1f0699adc1768b2a3e395a",
    "surveys/survey_agn_complete_ns32.h5": "6dcc38d17bd5cf6b006fde1427524e3e",
    "surveys/survey_agn_m18_ns32.h5": "7bd5e50423ae1f649a426b3ee987a866",
    "surveys/survey_agn_m19_ns32.h5": "ad16ada39876c4e700e477e3b5d5b5d3",
    "surveys/survey_agn_m20_ns32.h5": "73f09666bdd4aad6bdc769b525a235de",
    "surveys/survey_agn_m21_ns32.h5": "d8f84f1f7d522b51f5c81961a12a7b3d",
    "surveys/survey_gal_complete_ns32.h5": "3568cf69344b13d13de13a7e7b1d4fc6",
    "surveys/survey_gal_m18_ns32.h5": "c09d37ef065749f90c497040269be2e7",
    "surveys/survey_gal_m19_ns32.h5": "b9c19d7d9a79368f48ddfcadd4ed041d",
    "surveys/survey_gal_m20_ns32.h5": "a2d24f3797d1f96e72f1a8d06739a85e",
    "surveys/survey_gal_m21_ns32.h5": "d261eecfeb70b566cb2ef385a1945cf8",
    "catalogs/catalog_agn_complete.h5": "ca10cb1cf1f2d0abedc702d0f98d1037",
    "catalogs/catalog_agn_m18.h5": "662db188cd8db9225a06b118975d4572",
    "catalogs/catalog_agn_m19.h5": "9fa4d5c75e26ef932dad6134d01432f0",
    "catalogs/catalog_agn_m20.h5": "0f2d43ca0c0881324aece3fb7ac38d02",
    "catalogs/catalog_agn_m21.h5": "6d656c6ffcf2990deef3c5d00d9b6a90",
    "catalogs/catalog_gal_complete.h5": "ecb3f4ad05098052792923f53b214e31",
    "catalogs/catalog_gal_m18.h5": "8a6193eaf443bfed6f56911384b9cc23",
    "catalogs/catalog_gal_m19.h5": "7fe074428361390a2cec7b168a2a5db2",
    "catalogs/catalog_gal_m20.h5": "e7f4d3f97450d713be735727d1c4b213",
    "catalogs/catalog_gal_m21.h5": "4f75ed97687ca53be9fb6af381d942a9",
}
CATALOG_KEYS = [k for k in BASELINE_MD5 if k.startswith("catalogs/")]
SURVEY_KEYS = [k for k in BASELINE_MD5 if k.startswith("surveys/")]

# Functions the mass mark must not have touched.  Their source text is compared
# character for character between HEAD and the working tree.
UNTOUCHED_FUNCS = (
    "observe", "observe_v3", "detect_from_observation", "detect_v3", "snr_amplitude",
    "posterior_samples", "posterior_samples_v3", "p_pe_v3", "sub_seeds", "seed_dir",
    "stage_catalogs", "stage_surveys", "stage_injections", "stage_validation",
)


# --------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------
def md5(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def import_gmd():
    sys.path.insert(0, str(DARKSIRENS / "scripts/mock_dark_sirens"))
    import generate_mock_data as gmd            # noqa: E402
    return gmd


def dataset_paths(f: h5py.File) -> list[str]:
    out: list[str] = []
    f.visititems(lambda n, o: out.append(n) if isinstance(o, h5py.Dataset) else None)
    return sorted(out)


def compare_h5(a: Path, b: Path) -> dict:
    """Bitwise dataset comparison plus an attribute/metadata diff of two events files."""
    with h5py.File(a) as fa, h5py.File(b) as fb:
        pa, pb = dataset_paths(fa), dataset_paths(fb)
        differ, detail = [], {}
        for k in (pa if pa == pb else []):
            x, y = fa[k][()], fb[k][()]
            same = (x.dtype == y.dtype and x.shape == y.shape
                    and x.tobytes() == y.tobytes())
            if not same:
                differ.append(k)
                if np.issubdtype(x.dtype, np.floating) and x.shape == y.shape:
                    d = np.abs(x - y)
                    detail[k] = {"n_differing": int((d != 0).sum()), "n": int(x.size),
                                 "max_abs_diff": float(d.max())}
        ka, kb = set(fa.attrs), set(fb.attrs)
        attr_differ = []
        for k in sorted(ka & kb):
            if k == "metadata_json":
                continue
            va, vb = fa.attrs[k], fb.attrs[k]
            same = (np.array_equal(va, vb) if isinstance(va, np.ndarray) else va == vb)
            if not same:
                attr_differ.append(k)
        ma = json.loads(fa.attrs["metadata_json"])
        mb = json.loads(fb.attrs["metadata_json"])
        for m in (ma, mb):
            m.pop("generated_at_utc", None)
            m.pop("branch_mass", None)
        meta_differ = sorted(k for k in set(ma) | set(mb)
                             if ma.get(k, "<absent>") != mb.get(k, "<absent>"))
    return {
        "a": a.name, "b": b.name,
        "n_datasets": len(pa), "dataset_sets_equal": pa == pb,
        "datasets_bitwise_identical": (pa == pb) and not differ,
        "datasets_differing": differ, "dataset_diff_detail": detail,
        "attrs_only_in_a": sorted(ka - kb), "attrs_only_in_b": sorted(kb - ka),
        "shared_attrs_differing": attr_differ,
        "metadata_keys_differing": meta_differ,
    }


def read_events(path: Path) -> dict:
    with h5py.File(path) as f:
        meta = json.loads(f.attrs["metadata_json"])
        return {
            "attrs": {k: (v.item() if isinstance(v, np.generic) else v)
                      for k, v in f.attrs.items() if k != "metadata_json"},
            "meta": meta,
            "host_type": f["host_type"][()],
            "m1src": f["true_m1src"][()],
            "m2src": f["true_m2src"][()],
            "chieff": f["true_chieff"][()],
            "z": f["true_z"][()],
            "q": f["truth/q"][()],
            "snr_true": f["truth/snr_true"][()],
        }


def branch_stats(ev: dict, host: int) -> dict:
    m = ev["host_type"] == host
    m1, chi, q = ev["m1src"][m], ev["chieff"][m], ev["q"][m]
    lo, med, hi = np.percentile(m1, [16.0, 50.0, 84.0])
    return {"n": int(m.sum()),
            "m1src_median": float(med), "m1src_p16": float(lo), "m1src_p84": float(hi),
            "m1src_mean": float(m1.mean()),
            "m1src_sem": float(m1.std(ddof=1) / np.sqrt(m.sum())),
            "chieff_mean": float(chi.mean()),
            "chieff_sem": float(chi.std(ddof=1) / np.sqrt(m.sum())),
            "q_median": float(np.median(q))}


def peak_responsibility(m1: np.ndarray, gmd, pop) -> float:
    """Mean posterior probability that a DETECTED primary came from the Gaussian peak,
    under that branch's own population.  A detected-set descriptive: the generator does
    not store the component flag, and detection reweights the mixture."""
    p_pk = gmd._peak_pdf(m1, pop.peak_mu, pop.peak_sigma)
    p_pl = gmd._powerlaw_pdf(m1, pop.alpha, pop.mmin, pop.mmax, pop.dm_min, pop.dm_max)
    w = pop.peak_fraction * p_pk
    return float(np.mean(w / (w + (1.0 - pop.peak_fraction) * p_pl)))


def mixture_density(m: np.ndarray, gmd, pop) -> np.ndarray:
    return (pop.peak_fraction * gmd._peak_pdf(m, pop.peak_mu, pop.peak_sigma)
            + (1.0 - pop.peak_fraction)
            * gmd._powerlaw_pdf(m, pop.alpha, pop.mmin, pop.mmax, pop.dm_min, pop.dm_max))


def sampler_check(gmd, pop, label: str) -> dict:
    """Draw SAMPLER_N primaries from ``_sample_powerlaw_peak_m1`` under ``pop`` and test

    (a) the PEAK-lane masses against the truncated Gaussian the code draws, by KS;
    (b) the FULL m1 histogram against the analytic PL + G density, by an
        equal-probability binned chi-squared.
    """
    rng = np.random.default_rng(SAMPLER_SEED)
    m1, use_peak = gmd._sample_powerlaw_peak_m1(
        rng, SAMPLER_N, pop, return_component=True)
    lo, hi = float(gmd._MASS_NORM_GRID[0]), float(gmd._MASS_NORM_GRID[-1])

    # (a) peak lanes vs the truncated Gaussian, exact CDF
    tn = stats.truncnorm((lo - pop.peak_mu) / pop.peak_sigma,
                         (hi - pop.peak_mu) / pop.peak_sigma,
                         loc=pop.peak_mu, scale=pop.peak_sigma)
    ks = stats.kstest(m1[use_peak], tn.cdf)

    # (b) full mixture, equal-probability bins under the analytic density
    grid = np.linspace(lo, hi, 200_001)
    dens = mixture_density(grid, gmd, pop)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(grid))])
    cdf /= cdf[-1]
    edges = np.interp(np.linspace(0.0, 1.0, SAMPLER_NBINS + 1), cdf, grid)
    edges[0], edges[-1] = -np.inf, np.inf
    obs = np.histogram(m1, bins=edges)[0]
    exp = SAMPLER_N / SAMPLER_NBINS
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    dof = SAMPLER_NBINS - 1
    return {
        "label": label, "peak_mu": float(pop.peak_mu), "peak_sigma": float(pop.peak_sigma),
        "n_draws": int(SAMPLER_N), "rng_seed": int(SAMPLER_SEED),
        "n_peak_lane": int(use_peak.sum()),
        "peak_lane_fraction": float(use_peak.mean()),
        "peak_lane_mean": float(m1[use_peak].mean()),
        "peak_lane_std": float(m1[use_peak].std(ddof=1)),
        "peak_lane_ks_D": float(ks.statistic), "peak_lane_ks_p": float(ks.pvalue),
        "full_chi2": chi2, "full_chi2_dof": dof, "full_chi2_per_dof": chi2 / dof,
        "full_chi2_p": float(stats.chi2.sf(chi2, dof)),
        "n_bins": SAMPLER_NBINS,
    }


def func_source(text: str, name: str) -> str:
    """Source of a top-level ``def name`` (to its next top-level statement)."""
    lines = text.splitlines(keepends=True)
    start = next(i for i, ln in enumerate(lines) if ln.startswith(f"def {name}("))
    end = start + 1
    while end < len(lines) and (lines[end].startswith((" ", "\t", ")", "#"))
                                or not lines[end].strip()):
        end += 1
    return "".join(lines[start:end]).rstrip() + "\n"


def verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


# --------------------------------------------------------------------------------
def main() -> int:
    gmd = import_gmd()
    pop = gmd.PopulationConfig(gamma=0.0)
    pop_agn = dataclasses.replace(pop, peak_mu=pop.peak_mu + PLANTED_DMU_G)

    for p in (NEW, CTRL, HEADCTRL, A8):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")

    new, a8 = read_events(NEW), read_events(A8)
    gal, agn = branch_stats(new, 0), branch_stats(new, 1)
    gal8, agn8 = branch_stats(a8, 0), branch_stats(a8, 1)
    bm = new["meta"]["branch_mass"]
    bs = new["meta"]["branch_spin"]

    digests = {k: md5(DATA / k) for k in BASELINE_MD5}
    changed = sorted(k for k, v in digests.items() if v != BASELINE_MD5[k])

    ctrl_vs_head = compare_h5(HEADCTRL, CTRL)
    ctrl_vs_a8 = compare_h5(A8, CTRL)
    new_vs_ctrl = compare_h5(CTRL, NEW)

    samp_gal = sampler_check(gmd, pop, "GAL branch, pop (mu_G = 35)")
    samp_agn = sampler_check(gmd, pop_agn, "AGN branch, pop_agn (mu_G = 40)")

    head_src = git(REPO, "show", f"HEAD:working/data/generate_dataset.py")
    cur_src = GEN.read_text()
    func_changed = [f for f in UNTOUCHED_FUNCS
                    if func_source(head_src, f) != func_source(cur_src, f)]
    numstat = git(REPO, "diff", "--numstat", "--", "working/data/generate_dataset.py")
    added, removed = (int(x) for x in numstat.split()[:2]) if numstat else (0, 0)
    gen_diff = git(REPO, "diff", "--", "working/data/generate_dataset.py")
    diff_sha = hashlib.sha256(gen_diff.encode()).hexdigest()
    hunks = [ln for ln in gen_diff.splitlines() if ln.startswith("@@")]

    with h5py.File(DATA / "catalogs/catalog_gal_complete.h5") as f:
        n_gal_cat = int(f["z"].shape[0])
    with h5py.File(DATA / "catalogs/catalog_agn_complete.h5") as f:
        n_agn_cat = int(f["z"].shape[0])

    ks_q = stats.ks_2samp(new["q"][new["host_type"] == 0], new["q"][new["host_type"] == 1])
    dchi = agn["chieff_mean"] - gal["chieff_mean"]
    dchi_se = float(np.hypot(agn["chieff_sem"], gal["chieff_sem"]))
    dchi8 = agn8["chieff_mean"] - gal8["chieff_mean"]
    dchi8_se = float(np.hypot(agn8["chieff_sem"], gal8["chieff_sem"]))
    dm1 = agn["m1src_median"] - gal["m1src_median"]

    pop_fields = {f: getattr(pop, f) for f in pop.__dataclass_fields__}
    shared_fields = [f for f in pop.__dataclass_fields__ if f != "peak_mu"]
    fields_equal = {f: getattr(pop, f) == getattr(pop_agn, f) for f in shared_fields}

    C: list[dict] = []

    C.append({
        "id": 1, "name": "one realisation, seed 100",
        "statement": "the mock is a single draw on the record's event sub-seed",
        "numbers": {"attrs.seed_events": new["attrs"]["seed_events"],
                    "sub_seeds(100)['events']": SEED_EVENTS_RECORD,
                    "meta.seed_events_is_record_default":
                        new["meta"]["seed_events_is_record_default"],
                    "attrs.nobs": int(new["attrs"]["nobs"]),
                    "attrs.nsamp": int(new["attrs"]["nsamp"]),
                    "meta.planted_f_agn_is_record_default":
                        new["meta"]["planted_f_agn_is_record_default"]},
        "verdict": verdict(new["attrs"]["seed_events"] == SEED_EVENTS_RECORD
                           and new["meta"]["seed_events_is_record_default"] is True
                           and int(new["attrs"]["nobs"]) == 1000
                           and int(new["attrs"]["nsamp"]) == 2000
                           and new["meta"]["planted_f_agn_is_record_default"] is True)})

    cat_srv_ok = not [k for k in CATALOG_KEYS + SURVEY_KEYS if k in changed]
    C.append({
        "id": 2, "name": "same LSS and survey catalogs",
        "statement": "the events stage read the seed-100 complete catalogs and left "
                     "every catalog and survey file byte-identical",
        "numbers": {"n_catalog_files": len(CATALOG_KEYS),
                    "n_survey_files": len(SURVEY_KEYS),
                    "catalog_or_survey_md5_changed": [k for k in changed
                                                      if k in CATALOG_KEYS + SURVEY_KEYS],
                    "catalog_gal_complete_rows": n_gal_cat,
                    "catalog_agn_complete_rows": n_agn_cat,
                    "generator_log_hosts": "GAL 151,179,870  AGN 1,514,567",
                    "rows_match_generator_log":
                        bool(n_gal_cat == 151_179_870 and n_agn_cat == 1_514_567)},
        "verdict": verdict(cat_srv_ok and n_gal_cat == 151_179_870
                           and n_agn_cat == 1_514_567)})

    C.append({
        "id": 3, "name": "GAL branch generated from mu_G = 35",
        "statement": "the GAL-hosted branch keeps the record's Gaussian-peak location",
        "numbers": {"meta.branch_mass.mu_G_gal": bm["mu_G_gal"],
                    "attrs.mu_G_gal": float(new["attrs"]["mu_G_gal"]),
                    "DETECTED-SET true_m1src median [Msun]": gal["m1src_median"],
                    "DETECTED-SET true_m1src p16 [Msun]": gal["m1src_p16"],
                    "DETECTED-SET true_m1src p84 [Msun]": gal["m1src_p84"],
                    "DETECTED-SET mean peak-component responsibility":
                        peak_responsibility(new["m1src"][new["host_type"] == 0],
                                            gmd, pop),
                    "DETECTED-SET n": gal["n"],
                    "sampler check (mu_G = 35)": samp_gal},
        "note": "the median/percentiles/responsibility are DETECTED-SET descriptive "
                "statistics, not the hyperparameter: detection reweights the mixture "
                "and the peak lanes carry a hard truncation at [1, 200] Msun",
        "verdict": verdict(bm["mu_G_gal"] == MU_G_GAL
                           and float(new["attrs"]["mu_G_gal"]) == MU_G_GAL
                           and samp_gal["peak_lane_ks_p"] > 1e-3
                           and samp_gal["full_chi2_p"] > 1e-3)})

    C.append({
        "id": 4, "name": "AGN branch generated from mu_G = 40",
        "statement": "the AGN-hosted branch draws its primary mass from a peak at 40",
        "numbers": {"meta.branch_mass.mu_G_agn": bm["mu_G_agn"],
                    "meta.branch_mass.dmu_G_agn": bm["dmu_G_agn"],
                    "attrs.mu_G_agn": float(new["attrs"]["mu_G_agn"]),
                    "attrs.dmu_G_agn": float(new["attrs"]["dmu_G_agn"]),
                    "DETECTED-SET true_m1src median [Msun]": agn["m1src_median"],
                    "DETECTED-SET true_m1src p16 [Msun]": agn["m1src_p16"],
                    "DETECTED-SET true_m1src p84 [Msun]": agn["m1src_p84"],
                    "DETECTED-SET mean peak-component responsibility":
                        peak_responsibility(new["m1src"][new["host_type"] == 1],
                                            gmd, pop_agn),
                    "DETECTED-SET n": agn["n"],
                    "DETECTED-SET median shift AGN - GAL [Msun]": dm1,
                    "sampler check (mu_G = 40)": samp_agn},
        "note": "same caveat as check 3; the planted mark is +5.00 exactly and is "
                "carried separately from these detected-set descriptives",
        "verdict": verdict(bm["mu_G_agn"] == MU_G_GAL + PLANTED_DMU_G
                           and bm["dmu_G_agn"] == PLANTED_DMU_G
                           and float(new["attrs"]["mu_G_agn"]) == MU_G_GAL + PLANTED_DMU_G
                           and float(new["attrs"]["dmu_G_agn"]) == PLANTED_DMU_G
                           and samp_agn["peak_lane_ks_p"] > 1e-3
                           and samp_agn["full_chi2_p"] > 1e-3)})

    C.append({
        "id": 5, "name": "both branches share peak fraction 0.90",
        "statement": "the mixture weight is untouched by the mark",
        "numbers": {"pop.peak_fraction": pop.peak_fraction,
                    "pop_agn.peak_fraction": pop_agn.peak_fraction,
                    "meta.branch_mass.peak_fraction": bm["peak_fraction"],
                    "meta.population.peak_fraction":
                        new["meta"]["population"]["peak_fraction"]},
        "verdict": verdict(pop.peak_fraction == 0.90 and pop_agn.peak_fraction == 0.90
                           and bm["peak_fraction"] == 0.90
                           and new["meta"]["population"]["peak_fraction"] == 0.90)})

    C.append({
        "id": 6, "name": "shared PL slope, mass limits and tapers",
        "statement": "pop and pop_agn are equal in every dataclass field but peak_mu",
        "numbers": {"fields_compared": shared_fields,
                    "all_shared_fields_equal": all(fields_equal.values()),
                    "fields_unequal": [f for f, v in fields_equal.items() if not v],
                    "pop.peak_mu": pop.peak_mu, "pop_agn.peak_mu": pop_agn.peak_mu,
                    "pop fields": pop_fields,
                    "meta.branch_mass.peak_sigma": bm["peak_sigma"]},
        "verdict": verdict(all(fields_equal.values())
                           and pop_agn.peak_mu == pop.peak_mu + PLANTED_DMU_G)})

    C.append({
        "id": 7, "name": "shared q distribution",
        "statement": "both branches pair through the same beta and the same "
                     "per-component taper: the generator calls _sample_q(rng, m1, pop) "
                     "once, with pop, for every lane",
        "numbers": {"pop.beta": pop.beta, "pop_agn.beta": pop_agn.beta,
                    "generator call": "q = gmd._sample_q(rng, m1, pop, use_peak=use_peak)",
                    "DETECTED-SET KS D (GAL vs AGN q)": float(ks_q.statistic),
                    "DETECTED-SET KS p": float(ks_q.pvalue),
                    "DETECTED-SET q median GAL": gal["q_median"],
                    "DETECTED-SET q median AGN": agn["q_median"]},
        "note": "the KS numbers are DESCRIPTIVE ONLY.  The pairing model is shared by "
                "construction; a detected-set q difference is an expected consequence "
                "of the mass mark (the AGN peak sits 5 Msun higher, so its per-lane "
                "q support and its detectability both move) and of seed 100's own "
                "unplanted GAL/AGN q difference recorded in Analysis 8 Gate B",
        "verdict": verdict(pop.beta == pop_agn.beta)})

    C.append({
        "id": 8, "name": "spin difference +0.10",
        "statement": "the Analysis-8 spin mark is carried unchanged",
        "numbers": {"meta.branch_spin.dmu_chi_agn": bs["dmu_chi_agn"],
                    "attrs.dmu_chi_agn": float(new["attrs"]["dmu_chi_agn"]),
                    "attrs.mu_chi_gal": float(new["attrs"]["mu_chi_gal"]),
                    "attrs.mu_chi_agn": float(new["attrs"]["mu_chi_agn"]),
                    "planted": PLANTED_DMU_CHI,
                    "DETECTED-SET realised mean chi_eff AGN - GAL": dchi,
                    "DETECTED-SET standard error": dchi_se,
                    "DETECTED-SET realised, A8 marked mock": dchi8,
                    "DETECTED-SET standard error, A8 marked mock": dchi8_se},
        "verdict": verdict(bs["dmu_chi_agn"] == PLANTED_DMU_CHI
                           and float(new["attrs"]["dmu_chi_agn"]) == PLANTED_DMU_CHI)})

    pe_keys = ("pe_model", "a_mc", "a_q", "a_chi", "sigma_rho", "sky_a_deg")
    pe_diff = [k for k in pe_keys if new["attrs"][k] != a8["attrs"][k]]
    C.append({
        "id": 9, "name": "v3 PE unchanged",
        "statement": "every measurement-family constant equals the Analysis-8 mock's",
        "numbers": {**{f"attrs.{k}": new["attrs"][k] for k in pe_keys},
                    "keys_differing_from_A8": pe_diff,
                    "PE function sources identical to HEAD":
                        [f for f in ("observe_v3", "posterior_samples_v3", "p_pe_v3")
                         if f not in func_changed]},
        "verdict": verdict(not pe_diff
                           and not {"observe_v3", "posterior_samples_v3",
                                    "p_pe_v3"} & set(func_changed))})

    det_keys = ("detection_rule", "snr_threshold", "snr_ref")
    det_diff = [k for k in det_keys if new["attrs"][k] != a8["attrs"][k]]
    C.append({
        "id": 10, "name": "observed-SNR detection unchanged",
        "statement": "the detection rule and its constants are the Analysis-8 mock's, "
                     "and the generator diff is confined to the mass-mark hunk, the "
                     "argparse entry and the metadata/attribute records",
        "numbers": {**{f"attrs.{k}": new["attrs"][k] for k in det_keys},
                    "keys_differing_from_A8": det_diff,
                    "git diff lines added": added, "git diff lines removed": removed,
                    "git diff hunks": hunks,
                    "git diff sha256": diff_sha,
                    "untouched functions compared": list(UNTOUCHED_FUNCS),
                    "untouched functions whose source changed": func_changed},
        "note": "removed == 0 means the change is purely additive: no existing line of "
                "generate_dataset.py was altered or deleted",
        "verdict": verdict(not det_diff and removed == 0 and not func_changed
                           and len(hunks) == 6)})

    C.append({
        "id": 11, "name": "analyses 0-9 files unchanged",
        "statement": "every file Analyses 0-9 stand on carries its pre-edit md5",
        "numbers": {"n_files_checked": len(BASELINE_MD5),
                    "files_changed": changed,
                    "events/events_marked_dmu0p10.h5":
                        digests["events/events_marked_dmu0p10.h5"],
                    "events/events.h5": digests["events/events.h5"]},
        "verdict": verdict(not changed)})

    C.append({
        "id": 12, "name": "control behaves",
        "statement": "the patched generator at --dmu_G_agn 0.0 reproduces the pristine "
                     "HEAD generator bit for bit on the same node",
        "numbers": {
            "patched vs pristine HEAD (same node r008)": {
                "n_datasets": ctrl_vs_head["n_datasets"],
                "datasets_bitwise_identical": ctrl_vs_head["datasets_bitwise_identical"],
                "datasets_differing": ctrl_vs_head["datasets_differing"],
                "attrs_only_in_patched": ctrl_vs_head["attrs_only_in_b"],
                "shared_attrs_differing": ctrl_vs_head["shared_attrs_differing"],
                "metadata_keys_differing": ctrl_vs_head["metadata_keys_differing"]},
            "patched control vs the Analysis-8 marked mock": {
                "datasets_differing_n": len(ctrl_vs_a8["datasets_differing"]),
                "datasets_differing": ctrl_vs_a8["datasets_differing"],
                "max_abs_diff_true_m1src":
                    ctrl_vs_a8["dataset_diff_detail"].get("truth/m1src", {}),
                "shared_attrs_differing": ctrl_vs_a8["shared_attrs_differing"],
                "diagnosis": "1-ULP environment drift in the tapered power-law inverse "
                             "CDF, NOT the generator change: host labels, host indices, "
                             "redshifts, sky and source chi_eff are bit-identical, the "
                             "detected set and its order are unchanged, and the same "
                             "drift (11 of 1000 primary masses, max 1.42e-14) is "
                             "recorded in Analysis 8 GATES.md Gate B as a different "
                             "pow kernel on the generating node"},
            "production vs control (the mark itself)": {
                "datasets_differing_n": len(new_vs_ctrl["datasets_differing"]),
                "datasets_identical_n": (new_vs_ctrl["n_datasets"]
                                         - len(new_vs_ctrl["datasets_differing"])),
                "datasets_identical": [k for k in dataset_paths(h5py.File(NEW))
                                       if k not in new_vs_ctrl["datasets_differing"]]}},
        "verdict": verdict(ctrl_vs_head["datasets_bitwise_identical"]
                           and set(ctrl_vs_head["attrs_only_in_b"]) == NEW_ATTRS
                           and not ctrl_vs_head["attrs_only_in_a"]
                           and not ctrl_vs_head["shared_attrs_differing"]
                           and ctrl_vs_head["metadata_keys_differing"] == ["events_suffix"])})

    record = {
        "title": "Analysis 10 -- mock validation record",
        "date_utc": new["meta"]["generated_at_utc"],
        "product": str(NEW),
        "mark": {
            "planted_f_agn": PLANTED_F_AGN,
            "realised_detected_f_agn": float(new["attrs"]["truth_f_agn"]),
            "realised_n_host_agn": int(new["attrs"]["n_host_agn"]),
            "realised_n_host_gal": int(new["attrs"]["n_host_gal"]),
            "planted_dmu_chi_agn": PLANTED_DMU_CHI,
            "realised_detected_dmu_chi": dchi, "realised_detected_dmu_chi_se": dchi_se,
            "planted_dmu_G_agn": PLANTED_DMU_G,
            "mu_G_gal": MU_G_GAL, "mu_G_agn": MU_G_GAL + PLANTED_DMU_G,
            "detected_set_m1src_median_gal": gal["m1src_median"],
            "detected_set_m1src_median_agn": agn["m1src_median"],
            "detected_set_m1src_median_shift": dm1,
            "note": "the planted Delta mu_G = +5.00 is a hyperparameter of the DRAW; "
                    "the detected-set median shift is a different quantity, damped by "
                    "the shared power-law component, the peak truncation and the "
                    "SNR-weighted selection, and is reported beside it, never as it"},
        "branches": {"gal": gal, "agn": agn,
                     "gal_a8_marked_mock": gal8, "agn_a8_marked_mock": agn8},
        "realised": {
            "new": {"n_proposed_for_events": int(new["attrs"]["n_proposed_for_events"]),
                    "n_detected_total": new["meta"]["realised"]["n_detected_total"],
                    "detected_fraction": new["meta"]["realised"]["detected_fraction"],
                    "detected_fraction_snr_only":
                        new["meta"]["realised"]["detected_fraction_snr_only"],
                    "horizon_z_max_detected":
                        new["meta"]["realised"]["horizon_z_max_detected"],
                    "z_median_detected": new["meta"]["realised"]["z_median_detected"],
                    "realised_f_agn": new["meta"]["realised"]["realised_f_agn"],
                    "unique_agn_hosts": new["meta"]["realised"]["unique_agn_hosts"],
                    "unique_gal_hosts": new["meta"]["realised"]["unique_gal_hosts"],
                    "snr_obs_min": new["meta"]["realised"]["snr_obs_min"],
                    "snr_obs_max": new["meta"]["realised"]["snr_obs_max"]},
            "a8_marked_mock": {
                "n_proposed_for_events": int(a8["attrs"]["n_proposed_for_events"]),
                "n_detected_total": a8["meta"]["realised"]["n_detected_total"],
                "detected_fraction": a8["meta"]["realised"]["detected_fraction"],
                "detected_fraction_snr_only":
                    a8["meta"]["realised"]["detected_fraction_snr_only"],
                "horizon_z_max_detected":
                    a8["meta"]["realised"]["horizon_z_max_detected"],
                "z_median_detected": a8["meta"]["realised"]["z_median_detected"],
                "realised_f_agn": a8["meta"]["realised"]["realised_f_agn"],
                "unique_agn_hosts": a8["meta"]["realised"]["unique_agn_hosts"],
                "unique_gal_hosts": a8["meta"]["realised"]["unique_gal_hosts"],
                "snr_obs_min": a8["meta"]["realised"]["snr_obs_min"],
                "snr_obs_max": a8["meta"]["realised"]["snr_obs_max"]}},
        "sampler_checks": {"gal_mu35": samp_gal, "agn_mu40": samp_agn},
        "rng_child_derivation": bm["rng_child_derivation"],
        "branch_mass_rule": bm["rule"],
        "runs": RUNS,
        "files_written_md5": {
            q.name: md5(q) for q in sorted(
                list(EV.glob("*a10*")) + list(EV.glob("*_marked_dmu0p10_dmuG5*")))},
        "protected_md5_after": digests,
        "provenance": {
            "gws_agn_head": git(REPO, "rev-parse", "HEAD"),
            "gws_agn_generator_diff_sha256": diff_sha,
            "gws_agn_generator_lines_added": added,
            "gws_agn_generator_lines_removed": removed,
            "darksirens_worktree": str(DARKSIRENS),
            "darksirens_head": git(DARKSIRENS, "rev-parse", "HEAD"),
            "darksirens_clean": git(DARKSIRENS, "status", "--porcelain") == "",
            "darksirens_sha_in_product": new["meta"]["darksirens_sha"],
            "python": sys.executable,
            "numpy": np.__version__, "h5py": h5py.__version__,
            "generation_host": "r008.opa.vera.psc.edu (SLURM RM, CPU only)"},
        "checks": C,
        "n_pass": sum(c["verdict"] == "PASS" for c in C), "n_checks": len(C),
    }

    outdir = A10 / "diagnostics"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "a10_mock_validation.json").write_text(
        json.dumps(record, indent=2, default=str) + "\n")

    rows = ["| # | check | the number that decides it | verdict |",
            "|---|---|---|---|"]
    decider = {
        1: lambda: f"`seed_events` = {new['attrs']['seed_events']} = SEED*1000+3; "
                   f"`nobs` = {int(new['attrs']['nobs'])}, `nsamp` = "
                   f"{int(new['attrs']['nsamp'])}; record defaults True",
        2: lambda: f"{len(CATALOG_KEYS)} catalog + {len(SURVEY_KEYS)} survey files, "
                   f"0 md5 changes; complete catalogs {n_gal_cat:,} GAL / "
                   f"{n_agn_cat:,} AGN rows, matching the events log",
        3: lambda: f"`mu_G_gal` = {bm['mu_G_gal']}; detected GAL median m1src = "
                   f"{gal['m1src_median']:.3f} (16/84 {gal['m1src_p16']:.3f}/"
                   f"{gal['m1src_p84']:.3f}) Msun, n = {gal['n']}; sampler KS p = "
                   f"{samp_gal['peak_lane_ks_p']:.3f}, chi2/dof = "
                   f"{samp_gal['full_chi2_per_dof']:.3f}",
        4: lambda: f"`mu_G_agn` = {bm['mu_G_agn']}, `dmu_G_agn` = {bm['dmu_G_agn']}; "
                   f"detected AGN median m1src = {agn['m1src_median']:.3f} (16/84 "
                   f"{agn['m1src_p16']:.3f}/{agn['m1src_p84']:.3f}) Msun, n = "
                   f"{agn['n']}; sampler KS p = {samp_agn['peak_lane_ks_p']:.3f}, "
                   f"chi2/dof = {samp_agn['full_chi2_per_dof']:.3f}",
        5: lambda: f"`peak_fraction` = {pop.peak_fraction} in pop, pop_agn, "
                   f"`branch_mass` and `population`",
        6: lambda: f"{len(shared_fields)}/{len(shared_fields)} shared dataclass fields "
                   f"equal; peak_mu {pop.peak_mu} -> {pop_agn.peak_mu}",
        7: lambda: f"beta = {pop.beta} in both; detected-set q KS D = "
                   f"{ks_q.statistic:.4f}, p = {ks_q.pvalue:.4f} (descriptive)",
        8: lambda: f"`dmu_chi_agn` = {bs['dmu_chi_agn']}; detected-set realised "
                   f"{dchi:+.6f} +/- {dchi_se:.6f}",
        9: lambda: "pe_model = " + str(new["attrs"]["pe_model"]) + ", a_mc = "
                   + f"{new['attrs']['a_mc']}, a_q = {new['attrs']['a_q']}, a_chi = "
                   + f"{new['attrs']['a_chi']}, sigma_rho = "
                   + f"{new['attrs']['sigma_rho']}, sky_a_deg = "
                   + f"{new['attrs']['sky_a_deg']}: 0 keys differ from the A8 mock",
        10: lambda: f"detection_rule = {new['attrs']['detection_rule']}, "
                    f"snr_threshold = {new['attrs']['snr_threshold']}, snr_ref = "
                    f"{new['attrs']['snr_ref']}: 0 keys differ; git diff +{added}/-"
                    f"{removed} over {len(hunks)} hunks; "
                    f"{len(UNTOUCHED_FUNCS)} untouched functions byte-identical to HEAD",
        11: lambda: f"{len(BASELINE_MD5)}/{len(BASELINE_MD5)} files carry their "
                    f"pre-edit md5; 0 changed",
        12: lambda: f"patched vs pristine HEAD on r008: "
                    f"{ctrl_vs_head['n_datasets']}/{ctrl_vs_head['n_datasets']} "
                    f"datasets bitwise identical, only the 3 new attrs added; "
                    f"production vs control: "
                    f"{len(new_vs_ctrl['datasets_differing'])}/"
                    f"{new_vs_ctrl['n_datasets']} datasets move",
    }
    for c in C:
        rows.append(f"| {c['id']} | {c['name']} | {decider[c['id']]()} | "
                    f"**{c['verdict']}** |")
    md = [
        "# Analysis 10 — mock validation record", "",
        f"`{NEW}`", "",
        f"Twelve registered checks, {record['n_pass']}/{record['n_checks']} PASS. "
        "Numbers are read from `diagnostics/a10_mock_validation.json`, which "
        "`scripts/a10_validate_mock.py` writes; nothing here is retyped.", "",
        *rows, "",
        "## Planted against realised", "",
        "| quantity | planted | realised (detected set) |",
        "|---|---|---|",
        f"| `f_AGN` | {PLANTED_F_AGN:.2f} | "
        f"{float(new['attrs']['truth_f_agn']):.3f} "
        f"({int(new['attrs']['n_host_agn'])}/1000) |",
        f"| Δμ_χ | {PLANTED_DMU_CHI:+.2f} | {dchi:+.6f} ± {dchi_se:.6f} |",
        f"| Δμ_G [M⊙] | {PLANTED_DMU_G:+.2f} | median m1src shift {dm1:+.3f} "
        "(a different quantity — see below) |", "",
        "The mass mark is a hyperparameter of the **draw**: the AGN branch's Gaussian "
        "peak sits at 40 M⊙ against the GAL branch's 35. The detected-set median shift "
        "is not that number and must not be quoted as it — the shared 10% power-law "
        "component, the peak truncation and the SNR-weighted selection all damp it.", "",
        "## Detected-set branch descriptives", "",
        "| branch | n | median m1src | 16/84 | mean χ_eff | median q |",
        "|---|---|---|---|---|---|",
        f"| GAL (μ_G = 35) | {gal['n']} | {gal['m1src_median']:.3f} | "
        f"{gal['m1src_p16']:.3f} / {gal['m1src_p84']:.3f} | "
        f"{gal['chieff_mean']:+.6f} ± {gal['chieff_sem']:.6f} | {gal['q_median']:.4f} |",
        f"| AGN (μ_G = 40) | {agn['n']} | {agn['m1src_median']:.3f} | "
        f"{agn['m1src_p16']:.3f} / {agn['m1src_p84']:.3f} | "
        f"{agn['chieff_mean']:+.6f} ± {agn['chieff_sem']:.6f} | {agn['q_median']:.4f} |",
        "", "## Yield, against the Analysis-8 marked mock", "",
        "| quantity | this mock | A8 marked mock |", "|---|---|---|",
        f"| proposals for 1000 events | "
        f"{int(new['attrs']['n_proposed_for_events']):,} | "
        f"{int(a8['attrs']['n_proposed_for_events']):,} |",
        f"| detected fraction | "
        f"{new['meta']['realised']['detected_fraction']:.6e} | "
        f"{a8['meta']['realised']['detected_fraction']:.6e} |",
        f"| detected fraction, SNR only | "
        f"{new['meta']['realised']['detected_fraction_snr_only']:.6e} | "
        f"{a8['meta']['realised']['detected_fraction_snr_only']:.6e} |",
        f"| horizon z_max | "
        f"{new['meta']['realised']['horizon_z_max_detected']:.6f} | "
        f"{a8['meta']['realised']['horizon_z_max_detected']:.6f} |",
        f"| realised f_AGN | {new['meta']['realised']['realised_f_agn']:.3f} | "
        f"{a8['meta']['realised']['realised_f_agn']:.3f} |", "",
        "## Provenance", "",
        f"- gws-agn HEAD `{record['provenance']['gws_agn_head']}`; "
        f"`generate_dataset.py` diff +{added}/-{removed}, sha256 `{diff_sha[:16]}…`",
        f"- darksirens `{record['provenance']['darksirens_head']}` in "
        f"`{DARKSIRENS}`, worktree clean = "
        f"{record['provenance']['darksirens_clean']}; the product records "
        f"`darksirens_sha` = `{new['meta']['darksirens_sha']}`",
        f"- generated on {record['provenance']['generation_host']}; "
        f"numpy {np.__version__}, h5py {h5py.__version__}",
        f"- AGN-branch child RNG stream: `{bm['rng_child_derivation']}`", "",
        "## The three runs behind this record", "",
        "| role | SLURM job | host | wall | events stage |", "|---|---|---|---|---|",
        *[f"| {r['role']} | {r['slurm_job']} | {r['host'].split('.')[0]} | "
          f"{r['wall_seconds']} s | {r['events_stage_seconds']} s |" for r in RUNS],
        "",
        "The sampler check draws the same 2,000,000 primaries under both populations "
        "from one seed, so the two lanes carry the identical peak-lane count "
        f"({samp_gal['n_peak_lane']:,}) and the identical dispersion "
        f"({samp_gal['peak_lane_std']:.6f} M\u2299); the mark is a pure location "
        f"shift, {samp_gal['peak_lane_mean']:.4f} -> "
        f"{samp_agn['peak_lane_mean']:.4f} M\u2299.", "",
    ]
    (outdir / "a10_mock_validation.md").write_text("\n".join(md))

    for c in C:
        print(f"[{c['verdict']}] {c['id']:2d}  {c['name']}")
    print(f"\n{record['n_pass']}/{record['n_checks']} PASS")
    print("wrote", outdir / "a10_mock_validation.json")
    print("wrote", outdir / "a10_mock_validation.md")
    return 0 if record["n_pass"] == record["n_checks"] else 2


if __name__ == "__main__":
    sys.exit(main())
