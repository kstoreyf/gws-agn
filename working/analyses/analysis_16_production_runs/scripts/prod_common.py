"""Analysis 16 (production runs): shared definitions. Imports neither jax nor darksirens.

Everything a production run can be configured with lives here: the priors (one range per
parameter in every run), the population inputs of each mock family, the event kinds, the
catalog and flux-limit files, the analysis classes (which parameters each frees), the run
naming vocabulary, the number density from catalog counts, and the grid posterior summaries.

The driver (prod_driver.py, core venv), the run-list generator, the launcher and the summary
script (any python with numpy/h5py/yaml) all import this module.
"""
from __future__ import annotations

import json
import math
import sys
import zlib
from pathlib import Path

import numpy as np

WORK = Path("/hildafs/projects/phy230014p/magana/gws-agn/working")
A16 = WORK / "analyses" / "analysis_16_production_runs"
RUNS = A16 / "runs"
DATA = WORK / "data"
A14_SCRIPTS = WORK / "analyses" / "analysis_14_monte_carlo_accuracy" / "scripts"
A15_SCRIPTS = WORK / "analyses" / "analysis_15_host_normalisation" / "scripts"
# Schechter selection fits of the flux-limited catalogs (true-z fits on seed 100's catalogs; the
# luminosity function and the limits are generator constants, so every realization uses them)
FITS = WORK / "analyses" / "selection_redo" / "scripts" / "selection_fits_truez"

CORE = "e7c3007"
CORE_SRC = Path(f"/hildafs/projects/phy230014p/magana/src/darksirens-core-{CORE}")
CORE_PY = f"/hildafs/projects/phy230014p/magana/envs/darksirens-core-{CORE}/bin/python"
JAX_PY = "/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python"

OM0 = 0.3075               # the mocks' Omega_m (fixed in every run)
H0_N0_REF = 100.0          # h-scaled densities: n0 [Mpc^-3] = 10**log10n * (H0/100)**3
ZMAX_PRODUCTION = 1.10     # production convention: darksirens' redshift grid ends at z = 1.10
C_KMS = 299792.458

# ----------------------------------------------------------------------------- priors
# One uniform range per parameter, the same in every run (owner, 17 Nov).
PRIORS = {
    "H0": (50.0, 100.0),           # km/s/Mpc
    "f_agn": (0.0, 1.0),
    "mu_G": (20.0, 50.0),          # mass-peak location [Msun]
    "sigma_G": (1.0, 10.0),        # mass-peak width [Msun]
    "mu_chi": (-0.2, 0.3),         # spin mean
    "sigma_chi": (0.01, 1.0),      # spin width
    "dmu_G": (-10.0, 10.0),        # AGN offset of the mass-peak location [Msun]
    "dmu_chi": (-0.05, 0.30),      # AGN offset of the spin mean
    "log10n_gal": (-4.0, -1.0),    # log10 galaxy number density [h^3 Mpc^-3]
    "log10n_agn": (-6.0, -4.0),    # log10 AGN number density [h^3 Mpc^-3]
}
PARAM_ORDER = ["H0", "f_agn", "mu_G", "sigma_G", "mu_chi", "sigma_chi", "dmu_G", "dmu_chi",
               "log10n_gal", "log10n_agn"]

# ----------------------------------------------------------------------------- population
# powerlaw+peak (darksirens-core), shared beta/spin/gamma. ASCII name -> core label.
POP_LABELS = {
    "v1": "$v_1$", "alpha": r"$\alpha_{\rm PL}$", "m_min": r"$m_{\min,\rm PL}$",
    "m_max": r"$m_{\max,\rm PL}$", "dm_min": r"$\delta m_{\min,\rm PL}$",
    "dm_max": r"$\delta m_{\max,\rm PL}$", "mu_G": r"$\mu_{\rm G}$", "sigma_G": r"$\sigma_{\rm G}$",
    "beta": r"$\beta$", "mu_chi": r"$\mu_\chi$", "sigma_chi": r"$\sigma_\chi$", "gamma": r"$\gamma$",
}
POP_NAMES = list(POP_LABELS)
# The inputs each mock family was drawn with (data/corepop/README.md, "Core's density"); the
# host-dependent set's AGN class has mu_G + dmu_G and mu_chi + dmu_chi. Stage 5's family adds an
# entry here (and its events suffix) when it exists.
POPULATION_INPUTS = {
    "corepop": {"v1": 0.1, "alpha": 2.3, "m_min": 5.0, "m_max": 80.0, "dm_min": 3.0, "dm_max": 10.0,
                "mu_G": 35.0, "sigma_G": 5.0, "beta": 1.0, "mu_chi": 0.0, "sigma_chi": 0.1, "gamma": 0.0},
    # the observed-like spectrum (GWTC-3 power law + peak mapped onto core; data/obslike/README.md,
    # results/obslike_paper_values.json "core_theta"); spin and redshift as corepop
    "obslike": {"v1": 0.908, "alpha": 3.5, "m_min": 5.0, "m_max": 87.0, "dm_min": 4.9, "dm_max": 0.01,
                "mu_G": 34.0, "sigma_G": 3.6, "beta": 1.1, "mu_chi": 0.0, "sigma_chi": 0.1, "gamma": 0.0},
}
# AGN offsets per unit of the experiment's population difference s (the host-dependent sets are s = 1)
AGN_OFFSET_INPUTS = {"corepop": {"dmu_G": 5.0, "dmu_chi": 0.10}, "obslike": {"dmu_G": 5.0, "dmu_chi": 0.10}}
# each family's injection set (data/seedNNN/injections/injections_targeted<suffix>.h5)
INJECTION_SUFFIX = {"corepop": "", "obslike": "_obslike"}
POP_FREE = ("mu_G", "sigma_G", "mu_chi", "sigma_chi")
OFFSETS = ("dmu_G", "dmu_chi")
# the events-file attributes that state the planted classes (generate_dataset.py writes them)
EVENT_TRUTH_ATTRS = ("planted_f_agn", "truth_f_agn", "n_host_agn", "n_host_gal", "mu_G_gal", "mu_G_agn",
                     "mu_chi_gal", "mu_chi_agn", "dmu_G_agn", "dmu_chi_agn", "mass_spectrum", "population_code",
                     "seed_events")


def family_of_suffix(suffix: str) -> str:
    """The population family of an events suffix (`_obslike` sets are the observed-like family)."""
    return "obslike" if suffix.endswith("_obslike") else "corepop"

# ----------------------------------------------------------------------------- events
# kind -> (file stem before the suffix, redraw set code, planted f_AGN)
# "set" is any other event set (the short paper's experiment): file events<suffix>.h5, its planted
# f_AGN and class populations read from the file's attributes.
EVENT_KINDS = {
    "equal_populations": ("events", 2, 0.30),
    "agn_hosted": ("events_pureagn", 1, 1.0),
    "galaxy_hosted": ("events_puregal", 3, 0.0),
    "host_dependent": ("events_marked_dmu0p10_dmuG5", 4, 0.30),
    "set": ("events", 5, None),
}
DEFAULT_SUFFIX = "_corepop"
REDRAW_ROOT = 16_100_2026      # Analysis 16's stream root (A14's is 14_100_2026)
REDRAW_BLOCK = 2000


def events_path(realization: int, kind: str, suffix: str = DEFAULT_SUFFIX) -> Path:
    stem = EVENT_KINDS[kind][0]
    return DATA / f"seed{int(realization)}" / "events" / f"{stem}{suffix}.h5"


def default_injection_lane(family: str = "corepop") -> str:
    """The family's default injection lane, relative to the realization's directory."""
    return f"injections/injections_targeted{INJECTION_SUFFIX[family]}.h5"


def present_lane(lane: str) -> bool:
    """The record's (present) targeted lane, which serves equal populations of the peak family."""
    return lane.strip("/") == default_injection_lane("corepop")


def injections_path(realization: int, family: str = "corepop", lane: str | None = None) -> Path:
    """Each realization's own injection set (every run uses its realization's; never shared): the lane
    (a file relative to data/seedNNN/) given, else the family's default lane."""
    lane = lane or default_injection_lane(family)
    lane = lane.replace("seed<R>/", "").replace("seed{R}/", "").lstrip("/")
    return DATA / f"seed{int(realization)}" / lane


# Analysis 14's further injection sets of a realization (same proposal and draw count as the
# record's targeted lane, independent streams S*1000+900+j): pooled with the realization's own set for
# the analyses whose posterior is narrow (every merger AGN-hosted, matched to the AGN catalog)
EXTRA_LANE_GLOB = "mc_accuracy/inj_set*/seed{R}/injections/injections_targeted.h5"
NSAMP_AGN_ONLY = 32000       # Analysis 14: 32000 samples per merger for the all-AGN-hosted analyses


def extra_injection_lanes(realization: int, family: str = "corepop") -> list:
    """The realization's further injection sets that exist and are complete (lanes relative to the
    realization's directory), sorted by set number; the observed-like family has none."""
    if family != "corepop":
        return []
    root = DATA / f"seed{int(realization)}"
    pat = EXTRA_LANE_GLOB.replace("{R}", str(int(realization)))
    out = []
    for p in sorted(root.glob(pat), key=lambda q: int(q.parts[-4].replace("inj_set", "") or 0)):
        if injections_ready(p):
            out.append(str(p.relative_to(root)))
    return out


def variant_tag(nsamp: int, n_sets: int, prefix: str = "") -> str:
    """Run-name suffix of a run at non-default Monte Carlo settings, e.g. s32000_inj6."""
    return f"{prefix}s{int(nsamp)}_inj{int(n_sets)}"


def injections_ready(path) -> bool:
    """An injection file is usable once it and its _meta.json exist (the generator writes the meta
    after the file; a set still being written has no meta yet)."""
    p = Path(path)
    return p.exists() and p.with_name(p.stem + "_meta.json").exists()


def kind_and_suffix_of_file(name: str):
    """events file name (events<...>.h5) -> (kind, suffix) of the production vocabulary when the file
    is one of the four production kinds of a family (so it shares their redraw), else ("set", suffix)."""
    stem = name[:-3] if name.endswith(".h5") else name
    for fam_sfx in ("_corepop", "_obslike"):
        for kind, (kstem, _, _) in EVENT_KINDS.items():
            if kind != "set" and stem == kstem + fam_sfx:
                return kind, fam_sfx
    if not stem.startswith("events"):
        raise ValueError(f"not an events file name: {name}")
    return "set", stem[len("events"):]


def read_event_truth(path) -> dict:
    """The planted values an events file records (attrs; None where absent)."""
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        for k in EVENT_TRUTH_ATTRS:
            v = f.attrs.get(k)
            if isinstance(v, bytes):
                v = v.decode()
            elif isinstance(v, (np.floating, np.integer)):
                v = v.item()
            out[k] = v
    return out


def family_code(suffix: str) -> int:
    """A stable integer per events suffix (enters the redraw stream, so families never share one)."""
    return int(zlib.crc32(suffix.encode()) % 1_000_003)


def redraw_key(realization: int, kind: str, suffix: str, redraw_index: int = 0) -> list:
    """SeedSequence entropy prefix of a production redraw; block b appends [b]."""
    return [REDRAW_ROOT, int(realization), EVENT_KINDS[kind][1], family_code(suffix), int(redraw_index)]


# ----------------------------------------------------------------------------- catalogs
CATALOG_SETS = {"both": ("gal", "agn"), "galaxy": ("gal",), "agn": ("agn",), "none": ()}
CATALOG_DIR_NAMES = {"both": "both_catalogs", "galaxy": "galaxy_catalog", "agn": "agn_catalog",
                     "none": "no_catalog"}
FLUX_LIMITS = ("complete", "21", "20", "19", "18")


def rung(flux: str) -> str:
    return "complete" if flux == "complete" else f"m{flux}"


def survey_path(realization: int, tracer: str, flux: str) -> Path:
    return DATA / f"seed{int(realization)}" / "surveys" / f"survey_{tracer}_{rung(flux)}_ns32.h5"


def fit_path(tracer: str, flux: str) -> Path:
    return FITS / f"fit_{tracer}_{rung(flux)}_schechter.json"


# ----------------------------------------------------------------------------- analyses
ANALYSES = ("known_population", "single_population", "host_dependent_populations", "free_densities",
            # the short paper's experiment (paper_short_independent/EXPERIMENT.md); populations at inputs
            "mass_spectrum_alone", "galaxy_catalog_alone", "two_tracers_one_population",
            "two_tracers_two_populations", "hosts_known")
EXPERIMENT_ANALYSES = ANALYSES[4:]
# the catalogs each experiment analysis uses (hosts_known: the AGN catalog for AGN-hosted mergers and the
# galaxy catalog for galaxy-hosted ones)
EXPERIMENT_CATALOGS = {"mass_spectrum_alone": "none", "galaxy_catalog_alone": "galaxy",
                       "two_tracers_one_population": "both", "two_tracers_two_populations": "both",
                       "hosts_known": "both"}


def free_parameters(analysis: str, catalogs: str, flux: str, populations_differ: bool = False) -> list:
    """The sampled parameters of an analysis class, in PARAM_ORDER. populations_differ: the events'
    AGN class has its own population (only mass_spectrum_alone depends on it: f_AGN is then free)."""
    tr = CATALOG_SETS[catalogs]
    if analysis in EXPERIMENT_ANALYSES:
        if catalogs != EXPERIMENT_CATALOGS[analysis]:
            raise ValueError(f"{analysis} uses catalogs {EXPERIMENT_CATALOGS[analysis]!r}, not {catalogs!r}")
        if flux != "complete":
            raise ValueError(f"{analysis} is defined on complete catalogs")
        if analysis in ("two_tracers_one_population", "two_tracers_two_populations"):
            return ["H0", "f_agn"]
        if analysis == "mass_spectrum_alone" and populations_differ:
            return ["H0", "f_agn"]
        return ["H0"]
    out = ["H0"]
    if catalogs == "both":
        out.append("f_agn")
    if analysis == "known_population":
        pass
    elif analysis == "single_population":
        out += list(POP_FREE)
    elif analysis == "host_dependent_populations":
        if catalogs != "both":
            raise ValueError("host_dependent_populations needs both catalogs (the AGN class is catalog 2)")
        out += list(POP_FREE) + list(OFFSETS)
    elif analysis == "free_densities":
        if flux == "complete" or not tr:
            raise ValueError("free_densities needs flux-limited catalogs (complete catalogs have no density)")
        out += [p for p, t in (("log10n_gal", "gal"), ("log10n_agn", "agn")) if t in tr]
    else:
        raise ValueError(f"unknown analysis {analysis!r}")
    return [p for p in PARAM_ORDER if p in out]


def set_name(family: str, f: float, s: float) -> str:
    """The experiment's name of an event set: <family>_f<f>_s<s>, e.g. mock_f0p30_s1."""
    ss = f"{s:g}".replace(".", "p")
    return f"{family}_f{f:.2f}".replace(".", "p") + f"_s{ss}"


def run_name(analysis, catalogs, flux, kind, realization, density="counts", agn_density_factor=1.0,
             family="", extra="", set_label="", variant=""):
    """runs/<analysis>/<catalogs>/<flux>[_<density>]/<events>[_<family>]/realization_NNN[_<variant>][/<extra>];
    set_label (experiment runs) replaces <events>[_<family>]; variant marks non-default Monte Carlo
    settings (e.g. s32000_inj6)."""
    flux_part = "complete" if flux == "complete" else f"flux_limit_{flux}"
    if flux != "complete" and analysis != "free_densities":
        dpart = {"counts": "densities_from_counts", "given": "densities_given"}[density]
        if agn_density_factor != 1.0:
            dpart += {0.5: "_agn_density_half", 2.0: "_agn_density_double"}.get(
                agn_density_factor, f"_agn_density_times_{agn_density_factor:g}".replace(".", "p"))
        flux_part += "_" + dpart
    ev = set_label or (kind + (f"_{family}" if family else ""))
    parts = [analysis, CATALOG_DIR_NAMES[catalogs], flux_part, ev,
             f"realization_{int(realization):03d}" + (f"_{variant}" if variant else "")]
    if extra:
        parts.append(extra)
    return "/".join(parts)


# ----------------------------------------------------------------------------- densities
def schechter_fit(path) -> dict:
    s = json.loads(Path(path).read_text())["strata"][0]
    if s["family"] != "schechter":
        raise SystemExit(f"[fatal] {path}: family {s['family']}")
    return {k: float(s[k]) for k in ("m_lim", "Mstar_hat", "alpha", "M_faint_offset")}


def comoving_distance(z, H0, Om0=OM0, n=200001):
    """chi(z) [Mpc], flat LCDM, fine trapezoid in ln(1+z)."""
    z = np.asarray(z, float)
    zz = np.expm1(np.linspace(0.0, np.log1p(max(float(np.max(z)), 1e-6)), n))
    E = np.sqrt(Om0 * (1 + zz) ** 3 + 1 - Om0)
    chi = np.concatenate([[0.0], np.cumsum(0.5 * (1 / E[1:] + 1 / E[:-1]) * np.diff(zz))]) * C_KMS / H0
    return np.interp(z, zz, chi)


def z_complete(fit: dict, Om0=OM0) -> float:
    """The redshift to which a Schechter-selected catalog is complete: C(z) = 1 exactly below it.

    darksirens' selection completeness is C = clip[Gamma(a+1, x_lim)/Gamma(a+1, x_faint), 0, 1]
    with x = L/L*; it is 1 while x_lim <= x_faint, i.e. while the limiting absolute magnitude
    m_lim - DM(z) - M* is fainter than the luminosity function's faint cut M* + M_faint_offset.
    C is h-scaled (M* = Mstar_hat + 5 log10 h), so z_c solves DM(z_c; H0=100) =
    m_lim - Mstar_hat - M_faint_offset and does not depend on H0.
    """
    dm_c = fit["m_lim"] - fit["Mstar_hat"] - fit["M_faint_offset"]
    lo, hi = 1e-6, 20.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        dl = (1 + mid) * comoving_distance(mid, 100.0, Om0)
        if 5 * np.log10(dl) + 25.0 < dm_c:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def catalog_count_below(path, z_max: float) -> int:
    """Catalog galaxies (the survey file's own redshifts, zgals; padded entries excluded) below z_max."""
    import h5py
    with h5py.File(path, "r") as f:
        ng = np.asarray(f["ngals"], dtype=np.int64)
        z = f["zgals"]
        n = 0
        step = 1024
        for a in range(0, ng.size, step):
            blk = np.asarray(z[a:a + step])
            cols = np.arange(blk.shape[1])[None, :]
            real = cols < ng[a:a + step, None]
            n += int(np.sum(real & (blk < z_max)))
    return n


def density_from_counts(survey, fit_json, Om0=OM0) -> dict:
    """A class's number density from its catalog counts within the redshift to which it is complete.

        n = N(z < z_c) / V_c(z < z_c),   V_c = (4 pi / 3) chi(z_c)^3 (full sky),

    with z_c from z_complete(fit). Quoted in h^3 Mpc^-3 (V_c at H0 = 100), so that the density the
    likelihood uses at any H0, n * (H0/100)^3 (core's n0_units = 'h_scaled'), equals N(z<z_c) /
    V_c(z<z_c; H0): the counts fix the density in redshift space, at every H0.
    """
    fit = schechter_fit(fit_json)
    zc = z_complete(fit, Om0)
    n = catalog_count_below(survey, zc)
    chi100 = float(comoving_distance(zc, 100.0, Om0))
    v100 = 4.0 * math.pi / 3.0 * chi100 ** 3
    nh = n / v100
    return {"z_complete": zc, "n_below_z_complete": n, "V_c_h-3Mpc3": v100, "n_h3Mpc-3": nh,
            "log10n_h_scaled": math.log10(nh) if n > 0 else float("-inf"),
            "n_Mpc-3_at_H0_67.74": nh * (67.74 / 100.0) ** 3,
            "survey": str(survey), "fit": str(fit_json), "fit_values": fit}


# ----------------------------------------------------------------------------- summaries
_trapz = getattr(np, "trapezoid", None) or np.trapz
QS = (0.05, 0.16, 0.5, 0.84, 0.95)


def _cdf(x, p):
    c = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
    return c / c[-1]


def summarize_1d(x, logp):
    """Median, 68%/90% equal-tailed intervals of a gridded 1-D log density (Analysis 14's a14_post)."""
    x = np.asarray(x, float)
    logp = np.asarray(logp, float)
    ok = np.isfinite(logp)
    if x.size < 2 or ok.sum() < 2:
        nan = float("nan")
        return {"median": nan, "ci68": [nan, nan], "ci90": [nan, nan], "hw68": nan, "w68": nan,
                "w90": nan, "mean": nan, "sd": nan, "map": nan, "mass_at_edges": [nan, nan]}
    p = np.where(ok, np.exp(logp - np.max(logp[ok])), 0.0)
    c = _cdf(x, p)
    q = np.interp(QS, c, x)
    norm = _trapz(p, x)
    mean = float(_trapz(p * x, x) / norm)
    sd = float(np.sqrt(max(_trapz(p * (x - mean) ** 2, x) / norm, 0.0)))
    return {"median": float(q[2]), "ci68": [float(q[1]), float(q[3])], "ci90": [float(q[0]), float(q[4])],
            "hw68": float(0.5 * (q[3] - q[1])), "w68": float(q[3] - q[1]), "w90": float(q[4] - q[0]),
            "mean": mean, "sd": sd, "map": float(x[int(np.nanargmax(np.where(ok, logp, -np.inf)))]),
            "mass_at_edges": [float(p[0] / p.max()), float(p[-1] / p.max())]}


def marginals_2d(h, f, lnL):
    lnL = np.asarray(lnL, float)
    m = np.nanmax(np.where(np.isfinite(lnL), lnL, -np.inf))
    p = np.where(np.isfinite(lnL), np.exp(lnL - m), 0.0)
    with np.errstate(divide="ignore"):
        return np.log(_trapz(p, f, axis=1)), np.log(_trapz(p, h, axis=0))


def summarize_2d(h, f, lnL, names=("H0", "f")):
    lh, lf = marginals_2d(h, f, lnL)
    sh, sf = summarize_1d(h, lh), summarize_1d(f, lf)
    lnL = np.asarray(lnL, float)
    p = np.where(np.isfinite(lnL), np.exp(lnL - np.nanmax(np.where(np.isfinite(lnL), lnL, -np.inf))), 0.0)
    p = p / p.sum()
    H, F = np.meshgrid(h, f, indexing="ij")
    eh, ef = (p * H).sum(), (p * F).sum()
    cov = (p * (H - eh) * (F - ef)).sum()
    rho = cov / np.sqrt((p * (H - eh) ** 2).sum() * (p * (F - ef) ** 2).sum())
    return {names[0]: sh, names[1]: sf, "rho": float(rho)}


def summarize_samples(s, names, weights=None):
    """Medians, 68%/90% intervals of (weighted) samples."""
    s = np.asarray(s, float)
    w = None if weights is None else np.asarray(weights, float) / np.sum(weights)
    out = {}
    for i, n in enumerate(names):
        x = s[:, i]
        if w is None:
            q = np.quantile(x, QS)
            mean, sd = float(x.mean()), float(x.std())
        else:
            o = np.argsort(x)
            c = np.cumsum(w[o]) - 0.5 * w[o]
            q = np.interp(QS, c, x[o])
            mean = float(np.sum(w * x))
            sd = float(np.sqrt(np.sum(w * (x - mean) ** 2)))
        out[n] = {"median": float(q[2]), "ci68": [float(q[1]), float(q[3])],
                  "ci90": [float(q[0]), float(q[4])], "mean": mean, "sd": sd}
    return out


def half_variance_corrected(lnL, pe_var, n_eff, n_events):
    """Analysis 14's correction: ln L + sum_i sigma_i^2 / 2 - N / (2 N_eff) (summary_bias_corrected)."""
    return np.asarray(lnL) + 0.5 * np.asarray(pe_var) - 0.5 * n_events / np.asarray(n_eff)


def log(tag, msg):
    import time
    print(f"[{tag} {time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)


def ensure_paths():
    for p in (str(A14_SCRIPTS),):
        if p not in sys.path:
            sys.path.insert(0, p)
