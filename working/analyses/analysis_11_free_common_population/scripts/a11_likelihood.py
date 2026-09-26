"""Analysis-11 likelihood: the Analysis-10 two-mark likelihood with the
REFERENCE population's Gaussian-peak location and spin mean released.

Analysis 10 pinned all twelve base (catalog-1, GAL) population slots at the
powerlaw+peak fiducial and sampled only the AGN branch's absolute copies
``$\\mu_{\\rm G}$_c2`` and ``$\\mu_\\chi$_c2``.  Analysis 11 unpins exactly two
base slots, ``$\\mu_{\\rm G}$`` (G.mu, slot 6) and ``$\\mu_\\chi$`` (mu_chi,
slot 9), and reports the population as a reference plus an environmental
departure:

    mu_G      = mu_{G,c1}                 (reference, truth 35 Msun)
    dmu_G     = mu_{G,c2} - mu_{G,c1}     (environmental, truth +5 Msun)
    mu_chi    = mu_{chi,c1}               (reference, truth 0)
    dmu_chi   = mu_{chi,c2} - mu_{chi,c1} (environmental, truth +0.10)

NOTHING about the likelihood is reimplemented.  ``build_a11`` is
``a10_likelihood.build_a10``'s recipe verbatim -- ``a8_likelihood.build(...,
mode='new')`` under ``a10_likelihood._steer(('G.mu_c2', 'mu_chi_c2'), ...)`` --
with ONE extra steer: ``a8.fixed_parameter_values_for`` returns the Analysis-10
pinned set minus the two base labels, so ``build_parameter_space`` emits them
as sampled coordinates.  ``ParameterDecoder.decode_mixture`` reads catalog 1's
population from the base labels and catalog 2's from the ``_c2`` labels
(absolute, falling back elementwise to catalog 1), both through the same
``resolve_parameter_values`` whether a value is sampled or fixed, so at
(mu_G, mu_chi) = (35, 0) the Analysis-11 likelihood is the Analysis-10 one --
checked bitwise by ``a11_closure.py``, not assumed.

The reporting map (``A11LikelihoodCell.evaluate_at``) is applied BEFORE the
production call; the physical population implementation is untouched.

Environment: as Analysis 10 (``source .../analysis_9.../scripts/env_a9.sh``,
PYTHONPATH -> darksirens-a8 at af896ca).  The A8/A9/A10 trees are read-only.
"""
import sys

sys.dont_write_bytecode = True

import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np

A10_SCRIPTS = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/analyses/"
                   "analysis_10_mass_spin_marked_multitracer/scripts")
A11_DIR = Path(__file__).resolve().parent.parent
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
if str(A10_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(A10_SCRIPTS))

import a10_likelihood as A10                        # noqa: E402

a8 = A10.a8

# --------------------------------------------------------------------------- #
# The mock and inputs of record (Analysis 10's production mock, reused)
# --------------------------------------------------------------------------- #
DATA = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data/seed100")
GW_PATH_A11 = str(DATA / "events" / "events_marked_dmu0p10_dmuG5.h5")
GW_MD5_A11 = "427990378e299850a9c0708d389bc0bf"
INJ_MD5 = "e8a611a27f1f0699adc1768b2a3e395a"
SURVEY_GAL, SURVEY_AGN = A10.SURVEY_GAL, A10.SURVEY_AGN
H0_FID = A10.H0_FID                                  # 67.74

TRUTH = {"mu_G": 35.0, "dmu_G": 5.0, "mu_chi": 0.0, "dmu_chi": 0.10,
         "f_agn_planted": 0.30, "H0": 67.74}

# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
_iG, MU_G_LABEL, _fidG = A10.fiducial_slot("G.mu")          # '$\mu_{\rm G}$'
_iC, MU_CHI_LABEL, _fidC = A10.fiducial_slot("mu_chi")      # '$\mu_\chi$'
MU_G_C2_LABEL = A10.MU_G_C2_LABEL
MU_CHI_C2_LABEL = A10.MU_CHI_C2_LABEL
FREED_BASE = (MU_G_LABEL, MU_CHI_LABEL)
if (_fidG, _fidC) != (35.0, 0.0):
    raise RuntimeError(f"[fatal] base fiducials are ({_fidG}, {_fidC}), not (35, 0)")

# Science-exploration domains (owner brief, section 4).
DOMAIN = {"mu_G": (31.0, 39.0), "dmu_G": (-4.0, 10.0),
          "mu_chi": (-0.10, 0.10), "dmu_chi": (-0.05, 0.20), "f_agn": (0.0, 1.0)}
MU_G_ABS_BOUNDS = A10.MU_G_BOUNDS                    # (20, 50): registry prior
MU_CHI_ABS_BOUNDS = (-1.0, 1.0)                      # production spin support


def check_point(mu_G, dmu_G, mu_chi, dmu_chi):
    """The brief's validity rules on the ABSOLUTE branch means."""
    agn_G = mu_G + dmu_G
    agn_chi = mu_chi + dmu_chi
    lo, hi = MU_G_ABS_BOUNDS
    if not (lo < mu_G < hi and lo < agn_G < hi):
        raise ValueError(f"mass means ({mu_G}, {agn_G}) outside ({lo}, {hi})")
    lo, hi = MU_CHI_ABS_BOUNDS
    if not (lo < mu_chi < hi and lo < agn_chi < hi):
        raise ValueError(f"spin means ({mu_chi}, {agn_chi}) outside ({lo}, {hi})")


# --------------------------------------------------------------------------- #
# The one extra steer
# --------------------------------------------------------------------------- #
@contextmanager
def _free_base(labels=FREED_BASE):
    """``a8.fixed_parameter_values_for`` minus the named base labels."""
    true_fpv = a8.fixed_parameter_values_for

    def fpv(mode):
        out = true_fpv(mode)
        if mode == "new":
            for lbl in labels:
                if lbl not in out:
                    raise RuntimeError(f"[fatal] {lbl!r} is not a pinned base label")
                del out[lbl]
        return out

    a8.fixed_parameter_values_for = fpv
    try:
        yield
    finally:
        a8.fixed_parameter_values_for = true_fpv


class A11LikelihoodCell(A10.A10LikelihoodCell):
    """A10's cell with base-point values for the two released base labels."""

    EXTRA_BASE_POINT = {MU_G_C2_LABEL: 35.0, MU_G_LABEL: 35.0, MU_CHI_LABEL: 0.0}

    def evaluate_at(self, H0=None, fcat_2=None, mu_G=35.0, dmu_G=0.0,
                    mu_chi=0.0, dmu_chi=0.0):
        """Evaluate at the REPORTING coordinates (reference + environmental)."""
        mu_G, dmu_G, mu_chi, dmu_chi = (float(mu_G), float(dmu_G),
                                        float(mu_chi), float(dmu_chi))
        check_point(mu_G, dmu_G, mu_chi, dmu_chi)
        ov = {MU_G_LABEL: mu_G, MU_G_C2_LABEL: mu_G + dmu_G,
              MU_CHI_LABEL: mu_chi, MU_CHI_C2_LABEL: mu_chi + dmu_chi}
        if H0 is not None:
            ov["H0"] = float(H0)
        if fcat_2 is not None:
            ov["fcat_2"] = float(fcat_2)
        rec = self.evaluate(**ov)
        rec.update({"mu_G": mu_G, "dmu_G": dmu_G, "mu_chi": mu_chi, "dmu_chi": dmu_chi})
        return rec


def build_a11(name, survey_paths=(SURVEY_GAL, SURVEY_AGN), data=None, verbose=True,
              gw_path=GW_PATH_A11):
    A10.assert_fiducials()
    with A10._steer(A10.PER_CATALOG_A10, A11LikelihoodCell), _free_base():
        cell = a8.build(name, "new", list(survey_paths), data=data, verbose=verbose,
                        gw_path=gw_path)
    assert_a11_labels(cell)
    return cell


def build_a10_reference(name, survey_paths=(SURVEY_GAL, SURVEY_AGN), data=None,
                        verbose=True, gw_path=GW_PATH_A11):
    """The frozen Analysis-10 shape on the same data object (closure reference)."""
    return A10.build_a10(name, list(survey_paths), data=data, verbose=verbose,
                         gw_path=gw_path)


def assert_a11_labels(cell):
    """The A10 labels, in their A10 order, plus exactly the two base labels."""
    labels = list(cell.labels)
    extra = [lbl for lbl in labels if lbl not in A10.EXPECTED_LABELS_A10]
    kept = [lbl for lbl in labels if lbl in A10.EXPECTED_LABELS_A10]
    if sorted(extra) != sorted(FREED_BASE) or kept != A10.EXPECTED_LABELS_A10:
        raise RuntimeError(f"[fatal] A11 parameter space {labels}: extra {extra}")
    return labels


def selftest():
    """CPU: fiducials, labels and the fixed-value map, no data, no likelihood."""
    fpv_new = a8.fixed_parameter_values_for("new")
    with _free_base():
        fpv_a11 = a8.fixed_parameter_values_for("new")
    dropped = sorted(set(fpv_new) - set(fpv_a11))
    assert a8.fixed_parameter_values_for("new") == fpv_new, "steer not restored"
    print(f"base labels released: {dropped}")
    print(f"pinned base labels kept: {len(fpv_a11) - 1} of {len(fpv_new) - 1}")
    assert dropped == sorted(FREED_BASE)
    for p in [(35, 5, 0, 0.1), (31, -4, -0.1, -0.05), (39, 10, 0.1, 0.2)]:
        check_point(*p)
    print("selftest OK")


if __name__ == "__main__":
    selftest()
