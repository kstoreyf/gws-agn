#!/usr/bin/env python3
"""Analysis 9 mechanism follow-up, part A -- fixed-f and fixed-mark slices.

    python scripts/a9_mechanism_fixed_f.py     # writes diagnostics/a9_mechanism_fixed_f.json

Deterministic, CPU-only, and it recomputes NOTHING through the likelihood: every
number below is a reduction of the two finished cubes, results/s9_spatial.h5 and
results/j9_marked.h5.  Seed 100 only.  All writes are confined to this analysis
directory by _write().

The question this file answers is where the Analysis-9 H0 sharpening comes from.
Section 11 compares the two arms with f_AGN (and, in J9, dmu_chi) marginalised
away, so the marked arm gets two things at once: a different H0 likelihood shape
at any given host fraction, and a different weighting of the host fractions that
are summed over.  Holding f at the MAP node 0.275 separates them:

  A9F-S   p(H0 | f = 0.275, spatial-only)         S9 column 11
  A9F-J   p(H0 | f = 0.275, marked)               J9 column 11, dmu_chi marginalised
  A9F-JM  p(H0 | f = 0.275, dmu_chi = +0.1075)    J9 cell (., 11, 41), mark fixed too

and the ratio of the first two against the fully marginalised ratio is an exact
two-factor identity,

    width[J9 marg]     width[J9 | f]       width[J9 marg] / width[J9 | f]
    --------------  =  -------------  x  ---------------------------------
    width[S9 marg]     width[S9 | f]       width[S9 marg] / width[S9 | f]
                       \___________/      \_______________________________/
                        routing factor            global-mixture factor

verified here to ~1e-12.  The same ratio is then swept over every f node inside
J9's own 90 % credible interval, and the fixed-f log-likelihood difference is
split into its PE and selection parts along H0.

Conventions, identical to the production summariser and to a9_post.py:

  * flat priors on every axis, so the posterior is the exponentiated
    log-likelihood and a guard-rejected cell (logL = -inf) carries zero density;
  * marginals by trapezoid integration over the other axes (np.trapz);
  * medians and equal-tailed 68/90 % intervals from
    analysis_2/scripts/scan_h0f.py::marginal_ci, imported, not copied;
  * MAP is the argmax node of the density being summarised, not an interpolant;
  * the PRIMARY H0 lattice is the 28 J9 nodes.  Every J9 node is bitwise an S9
    node, so S9 can be re-integrated on exactly that lattice and the two arms
    compared at identical quadrature and identical range, exactly the matching
    a9_post.py::h0_matched_lattice registered.  The full 202-node S9 axis is
    reported alongside for A9F-S only, as a quadrature reference.

Both widths are reported at 68 % AND at 90 % everywhere.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.dont_write_bytecode = True

A9 = Path(__file__).resolve().parent.parent
RESULTS = A9 / "results"
DIAG = A9 / "diagnostics"
ANALYSES = A9.parent
A2_SCRIPTS = ANALYSES / "analysis_2_complete_catalog_H0_fagn" / "scripts"

sys.path.insert(0, str(A2_SCRIPTS))
import scan_h0f  # noqa: E402

marginal_ci = scan_h0f.marginal_ci

F_NODE = 0.275          # the (H0, f, dmu) MAP node in f, and Analysis 8's anchor
F_INDEX = 11
MU_NODE = 0.1075        # the MAP node in dmu_chi; mu_chi_c2 IS dmu_chi (mu_GAL = 0)
MU_INDEX = 41
MU_BRACKET = (26, 27)   # the two nodes bracketing dmu_chi = 0: -0.0050 and +0.0025
H0_REF = 69.0           # re-centring node for the PE/selection profiles
H0_STEP = 0.5           # lattice step used for the second differences at H0_REF
F_WINDOW = (0.189, 0.343)   # J9's own 90 % credible interval in f_AGN


# --------------------------------------------------------------------------- #
# writing, reductions
# --------------------------------------------------------------------------- #
def _write(path: Path, obj) -> Path:
    path = Path(path).resolve()
    if A9 not in path.parents:
        raise RuntimeError(f"[fatal] refusing to write outside {A9}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    print(f"wrote {path}")
    return path


def _density(ll: np.ndarray) -> np.ndarray:
    """exp(logL - max) with every non-finite cell (guard reject) set to zero."""
    ll = np.asarray(ll, float)
    fin = np.isfinite(ll)
    if not fin.any():
        raise RuntimeError("[fatal] no finite log-likelihood in this slice")
    out = np.zeros_like(ll)
    out[fin] = np.exp(ll[fin] - ll[fin].max())
    return out


def _norm(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p / np.trapz(p, x)


def _summary(x: np.ndarray, p: np.ndarray) -> dict:
    """median / MAP / equal-tailed 68 and 90 % of a 1-D flat-prior density."""
    x = np.asarray(x, float)
    p = np.asarray(p, float)
    with np.errstate(divide="ignore"):
        ci = marginal_ci(x, np.log(p))
    return {
        "n_nodes": int(x.size),
        "median": ci["median"],
        "MAP": float(x[int(np.argmax(p))]),
        "ci68": ci["ci68"],
        "ci90": ci["ci90"],
        "width68": float(ci["ci68"][1] - ci["ci68"][0]),
        "width90": float(ci["ci90"][1] - ci["ci90"][0]),
    }


def _second_difference(y: np.ndarray, i: int, h: float) -> float:
    return float((y[i + 1] - 2.0 * y[i] + y[i - 1]) / (h * h))


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load() -> dict:
    with h5py.File(RESULTS / "s9_spatial.h5", "r") as h:
        S = {"H0": np.asarray(h["H0_grid"][:], float),
             "f": np.asarray(h["f_grid"][:], float),
             "ll": np.asarray(h["log_likelihood"][:], float),
             "pe": np.asarray(h["guard/logL_pe"][:], float),
             "sel": np.asarray(h["guard/logL_selection"][:], float),
             "rejected": np.asarray(h["guard/rejected"][:], bool)}
    with h5py.File(RESULTS / "j9_marked.h5", "r") as h:
        J = {"H0": np.asarray(h["H0_grid"][:], float),
             "f": np.asarray(h["f_grid"][:], float),
             "mu": np.asarray(h["mu_chi_c2_grid"][:], float),
             "ll": np.asarray(h["log_likelihood"][:], float),
             "pe": np.asarray(h["guard/logL_pe"][:], float),
             "sel": np.asarray(h["guard/logL_selection"][:], float),
             "rejected": np.asarray(h["guard/rejected"][:], bool)}

    # the registered node identifications, asserted rather than assumed
    if abs(float(S["f"][F_INDEX]) - F_NODE) >= 1e-12:
        raise RuntimeError(f"[fatal] S9 f[{F_INDEX}] is not {F_NODE}")
    if abs(float(J["f"][F_INDEX]) - F_NODE) >= 1e-12:
        raise RuntimeError(f"[fatal] J9 f[{F_INDEX}] is not {F_NODE}")
    if not np.array_equal(S["f"], J["f"]):
        raise RuntimeError("[fatal] the two arms do not share the f axis")
    if abs(float(J["mu"][MU_INDEX]) - MU_NODE) >= 1e-12:
        raise RuntimeError(f"[fatal] J9 mu[{MU_INDEX}] is not {MU_NODE}")
    if np.any(J["mu"] == 0.0):
        raise RuntimeError("[fatal] an exact mu = 0 node exists; the bracket is wrong")

    idx = []
    for v in J["H0"]:
        hit = np.where(S["H0"] == v)[0]
        if hit.size != 1:
            raise RuntimeError(f"[fatal] J9 node {v} is not a unique S9 node")
        idx.append(int(hit[0]))
    idx = np.asarray(idx)
    if not np.all(S["H0"][idx] == J["H0"]):
        raise RuntimeError("[fatal] the J9 lattice is not bitwise a subset of S9's")
    J["s9_index"] = idx

    ref = np.where(J["H0"] == H0_REF)[0]
    if ref.size != 1:
        raise RuntimeError(f"[fatal] H0 = {H0_REF} is not a unique J9 node")
    J["ref"] = int(ref[0])
    if not (abs(J["H0"][J["ref"] + 1] - H0_REF - H0_STEP) < 1e-12
            and abs(H0_REF - J["H0"][J["ref"] - 1] - H0_STEP) < 1e-12):
        raise RuntimeError(f"[fatal] H0 = {H0_REF} is not centred in a {H0_STEP} step")
    return {"S": S, "J": J}


# --------------------------------------------------------------------------- #
# the three fixed-f posteriors, and the fully marginalised pair they sit inside
# --------------------------------------------------------------------------- #
def s9_fixed_f(S, idx, j):
    """p(H0 | f = f[j], spatial) on the matched lattice and on the full axis."""
    full = _norm(S["H0"], _density(S["ll"][:, j]))
    sub = _norm(S["H0"][idx], _density(S["ll"][idx, j]))
    return sub, full


def j9_fixed_f(J, j):
    """p(H0 | f = f[j], marked), dmu_chi marginalised by trapezoid."""
    P = _density(J["ll"][:, j, :])
    return _norm(J["H0"], np.trapz(P, J["mu"], axis=1))


def j9_fixed_f_fixed_mark(J, j, k=MU_INDEX):
    return _norm(J["H0"], _density(J["ll"][:, j, k]))


def s9_marginal(S, idx):
    P = _density(S["ll"][idx, :])
    return _norm(S["H0"][idx], np.trapz(P, S["f"], axis=1))


def j9_marginal(J):
    P = _density(J["ll"])
    return _norm(J["H0"], np.trapz(np.trapz(P, J["mu"], axis=2), J["f"], axis=1))


# --------------------------------------------------------------------------- #
def main() -> dict:
    d = load()
    S, J = d["S"], d["J"]
    idx = J["s9_index"]
    h0 = J["H0"]

    # ---- 1a/1b/1c: the three fixed-f posteriors ---------------------------- #
    p_S_sub, p_S_full = s9_fixed_f(S, idx, F_INDEX)
    p_J = j9_fixed_f(J, F_INDEX)
    p_JM = j9_fixed_f_fixed_mark(J, F_INDEX)

    A9F_S = _summary(h0, p_S_sub)
    A9F_S_full = _summary(S["H0"], p_S_full)
    A9F_J = _summary(h0, p_J)
    A9F_JM = _summary(h0, p_JM)

    # ---- the fully marginalised pair (the section-11 comparison) ----------- #
    p_S_marg = s9_marginal(S, idx)
    p_J_marg = j9_marginal(J)
    S_marg = _summary(h0, p_S_marg)
    J_marg = _summary(h0, p_J_marg)

    def ratios(a, b):
        return {"R68": a["width68"] / b["width68"],
                "R90": a["width90"] / b["width90"],
                "median_shift": a["median"] - b["median"],
                "MAP_shift": a["MAP"] - b["MAP"]}

    # ---- 1e: the two-factor identity --------------------------------------- #
    decomposition = {}
    for lev in ("68", "90"):
        w = lambda blk: blk[f"width{lev}"]            # noqa: E731
        total = w(J_marg) / w(S_marg)
        routing = w(A9F_J) / w(A9F_S)
        mixture = (w(J_marg) / w(A9F_J)) / (w(S_marg) / w(A9F_S))
        decomposition[f"level_{lev}"] = {
            "width_S9_marginalised": w(S_marg),
            "width_J9_marginalised": w(J_marg),
            "width_S9_at_f_fixed": w(A9F_S),
            "width_J9_at_f_fixed": w(A9F_J),
            "total_ratio_marginalised": total,
            "routing_factor": routing,
            "global_mixture_factor": mixture,
            "product_of_the_two_factors": routing * mixture,
            "identity_residual": routing * mixture - total,
            "identity_holds_to_1e-12": bool(abs(routing * mixture - total) < 1e-12),
            "log_total": float(np.log(total)),
            "fraction_of_log_gain_routing": float(np.log(routing) / np.log(total)),
            "fraction_of_log_gain_mixture": float(np.log(mixture) / np.log(total)),
        }

    # ---- 1f: the ratio swept over f inside J9's own 90 % interval ----------- #
    lo, hi = F_WINDOW
    f_nodes = [int(k) for k in range(S["f"].size) if lo <= S["f"][k] <= hi]
    sweep = []
    for k in f_nodes:
        sub_k, _ = s9_fixed_f(S, idx, k)
        p_k = j9_fixed_f(J, k)
        sS, sJ = _summary(h0, sub_k), _summary(h0, p_k)
        sweep.append({
            "f_index": k, "f_AGN": float(S["f"][k]),
            "is_the_planted_f": bool(abs(S["f"][k] - 0.30) < 1e-12),
            "is_the_MAP_f": bool(k == F_INDEX),
            "S9_width68": sS["width68"], "S9_width90": sS["width90"],
            "J9_width68": sJ["width68"], "J9_width90": sJ["width90"],
            "R68_F": sJ["width68"] / sS["width68"],
            "R90_F": sJ["width90"] / sS["width90"],
            "S9_median": sS["median"], "J9_median": sJ["median"],
            "median_shift": sJ["median"] - sS["median"],
        })
    r68 = np.array([r["R68_F"] for r in sweep])
    r90 = np.array([r["R90_F"] for r in sweep])

    def at_f(vals, target):
        """The swept ratio at one f node, matched on the node not on a literal."""
        hit = [i for i, s in enumerate(sweep) if abs(s["f_AGN"] - target) < 1e-12]
        if len(hit) != 1:
            raise RuntimeError(f"[fatal] f = {target} is not a unique swept node")
        return float(vals[hit[0]])

    # ---- 1g: the PE / selection split along H0 at f = 0.275 ---------------- #
    r = J["ref"]
    prof = {
        "J9_pe": J["pe"][:, F_INDEX, MU_INDEX],
        "J9_selection": J["sel"][:, F_INDEX, MU_INDEX],
        "S9_pe": S["pe"][idx, F_INDEX],
        "S9_selection": S["sel"][idx, F_INDEX],
    }
    prof["J9_total"] = prof["J9_pe"] + prof["J9_selection"]
    prof["S9_total"] = prof["S9_pe"] + prof["S9_selection"]
    centred = {k: v - v[r] for k, v in prof.items()}
    delta_pe = centred["J9_pe"] - centred["S9_pe"]
    delta_sel = centred["J9_selection"] - centred["S9_selection"]

    curvature = {k: _second_difference(v, r, H0_STEP) for k, v in prof.items()}
    curvature["delta_pe_J9_minus_S9"] = curvature["J9_pe"] - curvature["S9_pe"]
    curvature["delta_selection_J9_minus_S9"] = (curvature["J9_selection"]
                                                - curvature["S9_selection"])
    curvature["delta_total_J9_minus_S9"] = curvature["J9_total"] - curvature["S9_total"]

    split = {
        "definition": ("logL = logL_pe + logL_selection cell by cell; J9 is read at "
                       f"(f = {F_NODE}, dmu_chi = {MU_NODE}) and S9 at f = {F_NODE}, "
                       f"both on the 28 matched nodes, each profile re-centred at "
                       f"its own value at H0 = {H0_REF}"),
        "H0_nodes": h0.tolist(),
        "H0_reference_index": r,
        "raw_at_the_reference_node": {k: float(v[r]) for k, v in prof.items()},
        "profiles_recentred": {k: v.tolist() for k, v in centred.items()},
        "delta_pe_J9_minus_S9_recentred": delta_pe.tolist(),
        "delta_selection_J9_minus_S9_recentred": delta_sel.tolist(),
        "delta_total_J9_minus_S9_recentred": (delta_pe + delta_sel).tolist(),
        "extrema": {
            "max_abs_delta_pe": float(np.max(np.abs(delta_pe))),
            "max_abs_delta_selection": float(np.max(np.abs(delta_sel))),
            "H0_at_max_abs_delta_pe": float(h0[int(np.argmax(np.abs(delta_pe)))]),
            "H0_at_max_abs_delta_selection":
                float(h0[int(np.argmax(np.abs(delta_sel)))]),
        },
        "second_difference_at_H0_69": {
            "step": H0_STEP,
            "nodes_used": [float(h0[r - 1]), float(h0[r]), float(h0[r + 1])],
            "units": "nats per (km/s/Mpc)^2",
            **{k: float(v) for k, v in curvature.items()},
        },
    }

    # ---- 1h: the J9 rows bracketing dmu_chi = 0 against S9 ------------------ #
    bracket = []
    s9_pe_c = centred["S9_pe"]
    s9_sel_c = centred["S9_selection"]
    for k in MU_BRACKET:
        pe_c = J["pe"][:, F_INDEX, k] - J["pe"][r, F_INDEX, k]
        sel_c = J["sel"][:, F_INDEX, k] - J["sel"][r, F_INDEX, k]
        p_k = j9_fixed_f_fixed_mark(J, F_INDEX, k)
        s_k = _summary(h0, p_k)
        bracket.append({
            "mu_index": int(k), "mu_chi_c2": float(J["mu"][k]),
            "max_abs_delta_logL_pe_vs_S9": float(np.max(np.abs(pe_c - s9_pe_c))),
            "max_abs_delta_logL_selection_vs_S9":
                float(np.max(np.abs(sel_c - s9_sel_c))),
            "median": s_k["median"], "MAP": s_k["MAP"],
            "ci68": s_k["ci68"], "ci90": s_k["ci90"],
            "width68": s_k["width68"], "width90": s_k["width90"],
            "width68_over_S9": s_k["width68"] / A9F_S["width68"],
            "width90_over_S9": s_k["width90"] / A9F_S["width90"],
        })
    brackets_S9 = (min(b["width68"] for b in bracket) <= A9F_S["width68"]
                   <= max(b["width68"] for b in bracket))

    out = {
        "analysis": "analysis_9_marked_multitracer_H0_fagn",
        "diagnostic": "mechanism part A -- fixed-f and fixed-mark H0 posteriors",
        "seed": 100,
        "recomputed_through_the_likelihood": False,
        "inputs": [str(RESULTS / "s9_spatial.h5"), str(RESULTS / "j9_marked.h5")],
        "conventions": {
            "prior": "flat on every axis; guard-rejected cells carry zero density",
            "marginals": "np.trapz over the other axes",
            "intervals": ("equal-tailed, from analysis_2/scripts/scan_h0f.py::"
                          "marginal_ci (imported)"),
            "MAP": "argmax node of the density being summarised",
            "primary_lattice": ("the 28 J9 H0 nodes; every one is bitwise an S9 node, "
                                "so both arms are integrated at identical quadrature"),
            "f_fixed_at": F_NODE, "f_index": F_INDEX,
            "mu_fixed_at": MU_NODE, "mu_index": MU_INDEX,
            "note_mu": ("mu_chi_c2 IS dmu_chi because mu_chi_GAL is pinned at 0; "
                        "there is no exact mu = 0 node, the neighbours are "
                        f"{float(J['mu'][MU_BRACKET[0]]):+.4f} and "
                        f"{float(J['mu'][MU_BRACKET[1]]):+.4f}"),
        },
        "A9F_S_spatial_f_fixed": {
            "matched_28_node_lattice": A9F_S,
            "full_202_node_axis": A9F_S_full,
            "quadrature_ratio_matched_over_full_68":
                A9F_S["width68"] / A9F_S_full["width68"],
            "quadrature_ratio_matched_over_full_90":
                A9F_S["width90"] / A9F_S_full["width90"],
        },
        "A9F_J_marked_f_fixed_mu_marginalised": A9F_J,
        "A9F_JM_marked_f_fixed_mu_fixed": A9F_JM,
        "fully_marginalised_reference": {
            "S9_matched_lattice": S_marg, "J9": J_marg,
            "ratio68": J_marg["width68"] / S_marg["width68"],
            "ratio90": J_marg["width90"] / S_marg["width90"],
            "registered_in_REPORT": {"width68_ratio": 0.8879, "width90_ratio": 0.8996},
        },
        "ratios_at_fixed_f": {
            "A9F_J_over_A9F_S": ratios(A9F_J, A9F_S),
            "A9F_JM_over_A9F_S": ratios(A9F_JM, A9F_S),
            "A9F_JM_over_A9F_J": ratios(A9F_JM, A9F_J),
        },
        "two_factor_decomposition": decomposition,
        "robustness_in_f": {
            "window": {"J9_90pc_interval": list(F_WINDOW),
                       "nodes": [float(S["f"][k]) for k in f_nodes]},
            "table": sweep,
            "R68_F": {"min": float(r68.min()), "max": float(r68.max()),
                      "median": float(np.median(r68)),
                      "at_f_0p275": at_f(r68, 0.275),
                      "at_f_0p300": at_f(r68, 0.300),
                      "at_f_0p250": at_f(r68, 0.250)},
            "R90_F": {"min": float(r90.min()), "max": float(r90.max()),
                      "median": float(np.median(r90)),
                      "at_f_0p275": at_f(r90, 0.275),
                      "at_f_0p300": at_f(r90, 0.300),
                      "at_f_0p250": at_f(r90, 0.250)},
        },
        "pe_vs_selection_split": split,
        "mu_zero_bracket_against_S9": {
            "check": ("J9 at dmu_chi ~ 0 should reproduce S9, which holds dmu_chi "
                      "at exactly 0; these are the two J9 rows bracketing it"),
            "rows": bracket,
            "S9_reference": {"width68": A9F_S["width68"], "width90": A9F_S["width90"],
                             "median": A9F_S["median"], "MAP": A9F_S["MAP"]},
            "S9_width68_lies_between_the_two_bracket_rows": bool(brackets_S9),
        },
        "curves_for_the_figure": {
            "H0_nodes": h0.tolist(),
            "A9F_S": p_S_sub.tolist(),
            "A9F_J": p_J.tolist(),
            "A9F_JM": p_JM.tolist(),
            "normalisation": "np.trapz(p, H0_nodes) = 1 on the matched lattice",
        },
        "guard": {
            "S9_rejected_cells_at_f_0p275": int(S["rejected"][idx, F_INDEX].sum()),
            "J9_rejected_cells_at_f_0p275": int(J["rejected"][:, F_INDEX, :].sum()),
        },
    }
    _write(DIAG / "a9_mechanism_fixed_f.json", out)

    # ---- console ----------------------------------------------------------- #
    print(f"\n  p(H0 | f = {F_NODE}) on the 28 matched nodes")
    print(f"    {'':<10s}{'median':>10s}{'MAP':>8s}"
          f"{'ci68':>22s}{'w68':>9s}{'ci90':>22s}{'w90':>9s}")
    for name, blk in (("A9F-S", A9F_S), ("A9F-J", A9F_J), ("A9F-JM", A9F_JM)):
        print(f"    {name:<10s}{blk['median']:10.4f}{blk['MAP']:8.2f}"
              f"   [{blk['ci68'][0]:8.4f},{blk['ci68'][1]:8.4f}]{blk['width68']:9.4f}"
              f"   [{blk['ci90'][0]:8.4f},{blk['ci90'][1]:8.4f}]{blk['width90']:9.4f}")
    rr = out["ratios_at_fixed_f"]["A9F_J_over_A9F_S"]
    print(f"    R68^F = {rr['R68']:.4f}   R90^F = {rr['R90']:.4f}")
    for lev in ("68", "90"):
        b = decomposition[f"level_{lev}"]
        print(f"    {lev}%: total {b['total_ratio_marginalised']:.6f} = routing "
              f"{b['routing_factor']:.6f} x mixture {b['global_mixture_factor']:.6f} "
              f"(residual {b['identity_residual']:.2e}; log split "
              f"{b['fraction_of_log_gain_routing']:.3f} / "
              f"{b['fraction_of_log_gain_mixture']:.3f})")
    print(f"    R68^F(f) over {len(sweep)} nodes: min {r68.min():.4f} "
          f"median {np.median(r68):.4f} max {r68.max():.4f}")
    print(f"    R90^F(f) over {len(sweep)} nodes: min {r90.min():.4f} "
          f"median {np.median(r90):.4f} max {r90.max():.4f}")
    c = split["second_difference_at_H0_69"]
    print(f"    d2/dH0^2 at {H0_REF}: PE  J9 {c['J9_pe']:+.5f}  S9 {c['S9_pe']:+.5f}"
          f"   SEL J9 {c['J9_selection']:+.5f}  S9 {c['S9_selection']:+.5f}")
    return out


if __name__ == "__main__":
    main()
