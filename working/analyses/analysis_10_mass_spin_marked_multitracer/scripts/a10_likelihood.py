"""Analysis-10 likelihood builder: TWO per-catalog population coordinates.

Analysis 8 gave the AGN branch its own spin mean (``mu_chi_c2``).  Analysis 10
adds the AGN branch's own GAUSSIAN-PEAK LOCATION, ``G.mu_c2`` -- slot 6 of
``powerlaw+peak``, LaTeX label ``$\\mu_{\\rm G}$_c2``, fiducial 35 Msun, prior
bounds [20, 50].  The reporting coordinate is

    dmu_G = mu_{G,c2} - 35        (planted truth +5, i.e. mu_{G,AGN} = 40)

alongside Analysis 8's

    dmu_chi = mu_{chi,c2} - 0     (planted truth +0.10; the base fiducial IS 0,
                                   so the label value and the offset coincide)

NOTHING about the recipe is reimplemented here.  ``build_a10`` calls
``a8_likelihood.build(..., mode='new')`` VERBATIM -- same opts namespace, same
nuisance defaults, same guard spy, same catalog-KDE window, same
``fixed_parameter_values_for('new')`` with all twelve base population slots
pinned, same ``fix_population=False``.  The only difference is the tuple handed
to ``build_parameter_space``/``make_likelihood``, and that is injected by a
context manager that patches the two darksirens entry points a8.build imports
INSIDE its body, then restores them.  a8's body is the code of record; this
module only steers three arguments.

Three shapes are exposed because the closure stage needs all of them from ONE
``data`` object (``load_all_data`` is the expensive step and is survey/events
dependent only):

    build_spatial_reference  a8.build(mode='old')  -- one shared population
                             outside the branch sum (the analysis-2 shape)
    build_spin_only          a8.build(mode='new')  -- ('mu_chi_c2',), Analysis 8
    build_a10                a8.build(mode='new')  -- ('G.mu_c2', 'mu_chi_c2')

Environment (must hold before this module is imported):

    source .../analysis_9_marked_multitracer_H0_fagn/scripts/env_a9.sh
      -> PYTHONPATH=/hildafs/projects/phy230014p/magana/src/darksirens-a8
      -> XLA_PYTHON_CLIENT_PREALLOCATE=false

Trap 1 (Analysis 9's): ``DARKSIRENS_SRC`` does NOT steer the import -- the
editable install does.  Without PYTHONPATH pointing at darksirens-a8 the run
silently becomes Analysis 2 with fewer coordinates.  ``assert_a10_labels``
below is the cheap end-of-chain check.

Trap 2 (Analysis 8's): a coordinate can be EMITTED by the parameter space and
still be dead inside the likelihood.  Emitting ``$\\mu_{\\rm G}$_c2`` proves
nothing; the liveness probe is a measured change in logL (and, for the
selection term, in N_eff) when the coordinate moves.  Gate B's f = 1 axis is
exactly that probe for the selection factor.

The A8 and A9 trees are READ-ONLY: ``sys.dont_write_bytecode`` is set before
they are imported so no ``.pyc`` is written into them.
"""
import sys

sys.dont_write_bytecode = True                      # A8/A9 trees are read-only

import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np

A8_DIR = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/"
               "analysis_8_marked_multitracer_H0_fagn")
A8_SCRIPTS = A8_DIR / "scripts"
A9_DIR = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/"
               "analysis_9_marked_multitracer_H0_fagn")
A10_DIR = Path(__file__).resolve().parent.parent

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
if str(A8_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(A8_SCRIPTS))

import a8_likelihood as a8                          # noqa: E402  (the recipe)

# --------------------------------------------------------------------------- #
# The new coordinate
# --------------------------------------------------------------------------- #
MU_G_C2_LABEL = "$\\mu_{\\rm G}$_c2"   # what build_parameter_space emits
MU_G_PLAIN = "G.mu_c2"                 # what per_catalog_pop_params takes
MU_G_FID = 35.0                        # powerlaw+peak slot 6 fiducial (Msun)
MU_G_BOUNDS = (20.0, 50.0)             # the registry prior bounds
DMU_G_PLANT = 5.0                      # the planted AGN-branch offset

MU_CHI_C2_LABEL = a8.MU_CHI_C2_LABEL   # "$\\mu_\\chi$_c2"
MU_CHI_FID = 0.0
DMU_CHI_PLANT = 0.10

PER_CATALOG_A10 = ("G.mu_c2", "mu_chi_c2")
PER_CATALOG_SPIN_ONLY = ("mu_chi_c2",)

# Pass-throughs, so callers need import only this module.
set_env = a8.set_env
provenance = a8.provenance
configure_kde = a8.configure_kde
install_guard_spy = a8.install_guard_spy
population_labels_and_fiducial = a8.population_labels_and_fiducial
SURVEY_GAL = a8.SURVEY_GAL
SURVEY_AGN = a8.SURVEY_AGN
GW_PATH = a8.GW_PATH
GW_PATH_MARKED = a8.GW_PATH_MARKED
GWSEL_PATH = a8.GWSEL_PATH
H0_FID = a8.H0_FID
OM0_FID = a8.OM0_FID


def dmuG_to_label(dmu):
    """Reporting coordinate -> sampled coordinate: ``mu_G,c2 = 35 + dmu_G``."""
    return MU_G_FID + float(dmu)


def label_to_dmuG(value):
    return float(value) - MU_G_FID


def dmuchi_to_label(dmu):
    """The spin fiducial is 0, so this is the identity -- asserted, not assumed."""
    return MU_CHI_FID + float(dmu)


# --------------------------------------------------------------------------- #
# Fiducial checks (cheap, no likelihood trace; run them before trusting a scan)
# --------------------------------------------------------------------------- #
def fiducial_slot(plain_name):
    """``(index, latex label, fiducial value)`` of a base population slot."""
    labels, plain, fid = a8.population_labels_and_fiducial()
    idx = [i for i, p in enumerate(plain) if p == plain_name]
    if len(idx) != 1:
        raise RuntimeError(
            f"expected exactly one '{plain_name}' slot; got {idx} in {plain}")
    i = idx[0]
    return i, labels[i], float(fid[i])


def assert_fiducials():
    """The two reporting coordinates are OFFSETS; pin what they are offsets from."""
    out = {}
    i_g, lab_g, fid_g = fiducial_slot("G.mu")
    if abs(fid_g - MU_G_FID) > 0:
        raise RuntimeError(
            f"[fatal] G.mu fiducial is {fid_g}, not {MU_G_FID}: dmu_G = "
            f"mu_G_c2 - {MU_G_FID} is the WRONG reporting coordinate.")
    if lab_g + "_c2" != MU_G_C2_LABEL:
        raise RuntimeError(
            f"[fatal] G.mu LaTeX label is {lab_g!r}; the per-catalog label would "
            f"be {lab_g + '_c2'!r}, not {MU_G_C2_LABEL!r}.")
    out["G.mu"] = {"slot": i_g, "label": lab_g, "fiducial": fid_g}

    i_c, lab_c, fid_c = fiducial_slot("mu_chi")
    if fid_c != MU_CHI_FID:
        raise RuntimeError(
            f"[fatal] mu_chi fiducial is {fid_c}, not 0, so mu_chi_c2 is NOT "
            f"dmu_chi and every truth comparison must subtract it.")
    if lab_c + "_c2" != MU_CHI_C2_LABEL:
        raise RuntimeError(
            f"[fatal] mu_chi LaTeX label is {lab_c!r}, not {a8.MU_CHI_C2_LABEL!r} "
            f"without the suffix.")
    out["mu_chi"] = {"slot": i_c, "label": lab_c, "fiducial": fid_c}
    return out


EXPECTED_LABELS_A10 = [
    "H0", "log10n0", "delta", "sigma_kde",
    "log10n0_c2", "delta_c2", "sigma_kde_c2",
    MU_G_C2_LABEL, MU_CHI_C2_LABEL, "fcat_2",
]


def assert_a10_labels(cell):
    """The end-of-chain check for Trap 1: the ten coordinates, in order."""
    if list(cell.labels) != EXPECTED_LABELS_A10:
        raise RuntimeError(
            f"[fatal] A10 parameter space is {list(cell.labels)}, expected "
            f"{EXPECTED_LABELS_A10}.  Most likely darksirens did not resolve "
            f"under darksirens-a8 (PYTHONPATH), so the per-catalog population "
            f"blocks are absent.")
    return list(cell.labels)


# --------------------------------------------------------------------------- #
# The cell: a8's, plus the one base-point entry a8 cannot know about
# --------------------------------------------------------------------------- #
class A10LikelihoodCell(a8.LikelihoodCell):
    """a8's ``LikelihoodCell`` with ``$\\mu_{\\rm G}$_c2`` in the base point.

    a8's ``__init__`` raises ``KeyError`` for any sampled label it has no value
    for, which is the behaviour that caught Analysis 8's dead coordinate and
    must NOT be relaxed in A8.  The extra default is therefore injected here,
    around a8's own ``__init__``, and removed again: a8's tree is read-only and
    a8's module state is left exactly as it was found.
    """

    EXTRA_BASE_POINT = {MU_G_C2_LABEL: MU_G_FID}

    def __init__(self, **kw):
        saved = dict(a8.NUISANCE_DEFAULTS)
        a8.NUISANCE_DEFAULTS.update(self.EXTRA_BASE_POINT)
        try:
            super().__init__(**kw)
        finally:
            a8.NUISANCE_DEFAULTS.clear()
            a8.NUISANCE_DEFAULTS.update(saved)
        # The base point is dmu_G = 0 / dmu_chi = 0 by construction.
        for lbl, val in self.EXTRA_BASE_POINT.items():
            if self.base_point.get(lbl) != val:
                raise RuntimeError(
                    f"[fatal] base point for {lbl!r} is "
                    f"{self.base_point.get(lbl)!r}, expected {val!r}")

    # -- reporting-coordinate front door ----------------------------------- #
    def evaluate_at(self, H0=None, fcat_2=None, dmu_chi=None, dmu_G=None):
        """Evaluate at the REPORTING coordinates; unset ones keep the base point.

        ``dmu_chi``/``dmu_G`` are offsets from the base fiducials (0 and 35);
        the conversion is asserted by :func:`assert_fiducials`.
        """
        ov = {}
        if H0 is not None:
            ov["H0"] = float(H0)
        if fcat_2 is not None:
            ov["fcat_2"] = float(fcat_2)
        if dmu_chi is not None:
            ov[MU_CHI_C2_LABEL] = dmuchi_to_label(dmu_chi)
        if dmu_G is not None:
            ov[MU_G_C2_LABEL] = dmuG_to_label(dmu_G)
        rec = self.evaluate(**ov)
        rec["dmu_chi"] = (None if dmu_chi is None else float(dmu_chi))
        rec["dmu_G"] = (None if dmu_G is None else float(dmu_G))
        return rec


# --------------------------------------------------------------------------- #
# The three shapes
# --------------------------------------------------------------------------- #
@contextmanager
def _steer(per_catalog_pop_params, cell_class):
    """Steer a8.build's three arguments without touching a8's body or A8's tree.

    a8.build imports ``build_parameter_space`` and ``make_likelihood`` INSIDE
    its body, so patching them on their defining modules is seen by that call
    and by nothing that ran earlier.  ``a8.LikelihoodCell`` is a module global
    a8.build reads at return time.  All three are restored in ``finally``.
    """
    from darksirens.inference import prior as ds_prior
    from darksirens.likelihood import factory as ds_factory

    pcp = tuple(per_catalog_pop_params)
    true_bps = ds_prior.build_parameter_space
    true_ml = ds_factory.make_likelihood
    true_cell = a8.LikelihoodCell

    def bps(*args, **kw):
        kw["per_catalog_pop_params"] = pcp
        return true_bps(*args, **kw)

    def ml(*args, **kw):
        opts = kw.get("opts", args[0] if args else None)
        if opts is None:
            raise RuntimeError("make_likelihood called without opts")
        opts.per_catalog_pop_params = pcp
        return true_ml(*args, **kw)

    ds_prior.build_parameter_space = bps
    ds_factory.make_likelihood = ml
    a8.LikelihoodCell = cell_class
    try:
        yield
    finally:
        ds_prior.build_parameter_space = true_bps
        ds_factory.make_likelihood = true_ml
        a8.LikelihoodCell = true_cell


def build_a10(name, survey_paths, data=None, verbose=True, gw_path=None,
              per_catalog_pop_params=PER_CATALOG_A10):
    """The Analysis-10 shape: a8's 'new' recipe with TWO per-catalog coordinates.

    ``data`` may be reused across all three shapes provided it was loaded with
    the same survey list AND the same events file.
    """
    assert_fiducials()
    with _steer(per_catalog_pop_params, A10LikelihoodCell):
        cell = a8.build(name, "new", survey_paths, data=data, verbose=verbose,
                        gw_path=gw_path)
    if tuple(per_catalog_pop_params) == PER_CATALOG_A10:
        assert_a10_labels(cell)
    return cell


def build_spin_only(name, survey_paths, data=None, verbose=True, gw_path=None):
    """Analysis 8's shape verbatim: one per-catalog coordinate, ``mu_chi_c2``."""
    return a8.build(name, "new", survey_paths, data=data, verbose=verbose,
                    gw_path=gw_path)


def build_spatial_reference(name, survey_paths, data=None, verbose=True,
                            gw_path=None):
    """The shared-population (analysis-2) shape: no per-catalog population."""
    return a8.build(name, "old", survey_paths, data=data, verbose=verbose,
                    gw_path=gw_path)


# --------------------------------------------------------------------------- #
# CPU self-test: the parameter space resolves and the labels are as registered.
# No likelihood trace -- that is GPU work and must run on rita via SLURM.
# --------------------------------------------------------------------------- #
def _selftest():
    import json
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from darksirens.inference.prior import build_parameter_space
    import darksirens

    out = {"darksirens_module_file": darksirens.__file__,
           "pythonpath": os.environ.get("PYTHONPATH", "")}
    if not darksirens.__file__.startswith(a8.DARKSIRENS_REPO):
        raise SystemExit(f"[fatal] darksirens resolved at {darksirens.__file__}")

    out["fiducials"] = assert_fiducials()

    opts = a8.build_opts([a8.SURVEY_GAL, a8.SURVEY_AGN])
    opts.fix_population = False
    fpv = a8.fixed_parameter_values_for("new")

    def space(pcp):
        res = build_parameter_space(
            opts.pop_model, opts.fix_population, opts.fix_cosmology,
            opts.fix_survey, fix_de=opts.fix_de, prior_overrides={},
            fixed_parameter_values=fpv, universe_model=opts.universe_model,
            shared_beta=opts.shared_beta, shared_spin=opts.shared_spin,
            shared_gamma=opts.shared_gamma, sky_model=opts.sky_model,
            mark_model=opts.mark_model, mark_names=opts.mark_names,
            n_catalogs=opts.n_catalogs,
            lss_completion_active=[False] * opts.n_catalogs,
            use_lss=bool(opts.use_LSS), mark_names_by_catalog=None,
            per_catalog_pop_params=tuple(pcp),
        )
        labels, lo, hi = list(res[0]), np.asarray(res[1]), np.asarray(res[2])
        return {"labels": labels,
                "bounds": {l: [float(a), float(b)]
                           for l, a, b in zip(labels, lo, hi)}}

    out["A10"] = space(PER_CATALOG_A10)
    out["A8_spin_only"] = space(PER_CATALOG_SPIN_ONLY)
    if out["A10"]["labels"] != EXPECTED_LABELS_A10:
        raise SystemExit(f"[fatal] labels {out['A10']['labels']}")
    b = out["A10"]["bounds"]
    if b[MU_G_C2_LABEL] != list(MU_G_BOUNDS):
        raise SystemExit(f"[fatal] {MU_G_C2_LABEL} bounds {b[MU_G_C2_LABEL]}")
    if b[MU_CHI_C2_LABEL] != [-1.0, 1.0]:
        raise SystemExit(f"[fatal] {MU_CHI_C2_LABEL} bounds {b[MU_CHI_C2_LABEL]}")
    out["checks"] = {
        "labels_match_expected": True,
        "mu_G_c2_bounds_20_50": True,
        "mu_chi_c2_bounds_m1_1": True,
        "dmuG_to_label(0)": dmuG_to_label(0.0),
        "dmuG_to_label(+5)": dmuG_to_label(DMU_G_PLANT),
        "dmuG_to_label(-10)": dmuG_to_label(-10.0),
        "dmuG_to_label(+10)": dmuG_to_label(10.0),
    }
    print(json.dumps(out, indent=2))
    print("\n[a10_likelihood] SELFTEST PASSED (CPU, no likelihood trace)")
    return out


if __name__ == "__main__":
    _selftest()
