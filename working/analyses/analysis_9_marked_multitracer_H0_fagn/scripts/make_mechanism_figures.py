#!/usr/bin/env python3
"""Analysis 9 mechanism figures, part A.

    python scripts/make_mechanism_figures.py      # writes fig_mechanism_fixed_f.{pdf,png}

Deterministic and derived: it opens ONE file, diagnostics/a9_mechanism_fixed_f.json,
and draws what is in it.  No cube is re-reduced here and no likelihood is touched,
so the figure cannot drift from the diagnostic.  Every number that reaches the
canvas is printed to stdout beside the value recorded in that JSON, the same
convention scripts/make_figures.py uses, so the figure can be diffed without
opening either file.

  fig_mechanism_fixed_f
    (a) p(H0 | f_AGN = 0.275) on the 28-node matched lattice, for the three
        fixed-f posteriors: spatial-only (A9F-S), marked with the spin offset
        marginalised (A9F-J), and marked with the spin offset held at its MAP
        node +0.1075 (A9F-JM).  Shaded 90 % equal-tailed intervals; the planted
        H0 = 67.74 as a thin reference.
    (b) the fixed-f width ratio R^F(f) = width[J9 | f] / width[S9 | f] at every
        f node inside J9's own 90 % credible interval, at 68 % and at 90 %,
        against the fully marginalised ratios 0.8879 and 0.8996 as dashed
        references.

House conventions, from ../../../paper/scripts/figstyle.py, which is the visual
system: nothing here invents a colour.  EVERY plotted interval and band is 90 %;
the 68 % numbers go to stdout and belong in text and tables, never on an axis --
panel (b) is the one place 68 % appears on a canvas, and it is a width RATIO, not
an interval.  Panel (a) keeps the Analysis-8 arm identities that
scripts/make_figures.py registered: marked/joint = blue, spatial = aqua; the
fixed-mark variant takes the next slot of figstyle's fixed categorical order,
orange, which is the one pair the palette validator cleared against blue for all
pairs.  Panel (b) draws no arm, so it uses the ordered single-hue ramp -- light
for 68 %, dark for 90 % -- which is the documented slot for an ordered magnitude.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
A9 = HERE.parent
DIAG = A9 / "diagnostics"
FIGS = A9 / "figs"
PAPER_SCRIPTS = A9.parent.parent / "paper" / "scripts"

sys.path.insert(0, str(PAPER_SCRIPTS))
import figstyle as fs  # noqa: E402

C_S9, C_J9, C_JM = fs.C["aqua"], fs.C["blue"], fs.C["orange"]
C_68, C_90 = fs.RAMP[1], fs.RAMP[3]
H0_TRUTH = fs.H0_TRUTH


def _write_guard(path: Path) -> Path:
    path = Path(path).resolve()
    if A9 not in path.parents:
        raise RuntimeError(f"[fatal] refusing to write outside {A9}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def check(label: str, drawn: float, want: float, tol: float = 1e-9) -> None:
    d = abs(drawn - want)
    print(f"      {label:<44s} drawn {drawn: .6f}   json {want: .6f}   "
          f"|d| {d:.2e} {'OK' if d <= tol else 'MISMATCH'}")


def curve(ax, x, p, s, color, label, lw):
    """One density with its 90 % equal-tailed interval shaded.

    A9F-J and A9F-JM differ by 0.25 % in width, so they overplot; the line
    weights are staggered (widest curve underneath) purely so both remain
    visible, which is itself the reading.
    """
    xs, ys = fs.ci_band(x, p, s["ci90"][0], s["ci90"][1])
    ax.fill_between(xs, 0.0, ys, color=color, alpha=0.14, lw=0, zorder=2)
    ax.plot(x, p, color=color, lw=lw, zorder=3, label=label)


def main() -> None:
    fs.use()
    src = DIAG / "a9_mechanism_fixed_f.json"
    D = json.loads(src.read_text())
    print(f"  reading {src}")

    cur = D["curves_for_the_figure"]
    h0 = np.asarray(cur["H0_nodes"], float)
    arms = (("A9F-S  spatial-only", "A9F_S", C_S9,
             D["A9F_S_spatial_f_fixed"]["matched_28_node_lattice"], 1.4),
            ("A9F-JM  marked, mark fixed", "A9F_JM", C_JM,
             D["A9F_JM_marked_f_fixed_mu_fixed"], 2.6),
            ("A9F-J  marked", "A9F_J", C_J9,
             D["A9F_J_marked_f_fixed_mu_marginalised"], 1.2))

    sweep = D["robustness_in_f"]["table"]
    fnodes = np.array([r["f_AGN"] for r in sweep], float)
    r68 = np.array([r["R68_F"] for r in sweep], float)
    r90 = np.array([r["R90_F"] for r in sweep], float)
    ref68 = D["fully_marginalised_reference"]["registered_in_REPORT"]["width68_ratio"]
    ref90 = D["fully_marginalised_reference"]["registered_in_REPORT"]["width90_ratio"]

    # ---- every drawn number against its recorded value --------------------- #
    print("\n  [a] p(H0 | f = 0.275), 28-node matched lattice, 90 % shaded")
    for name, key, _, s, _lw in arms:
        p = np.asarray(cur[key], float)
        print(f"    {name}")
        check("normalisation trapz(p, H0)", float(np.trapz(p, h0)), 1.0)
        check("MAP node (argmax of the drawn curve)",
              float(h0[int(np.argmax(p))]), s["MAP"])
        check("90 % low  (shaded edge)", s["ci90"][0], s["ci90"][0])
        check("90 % high (shaded edge)", s["ci90"][1], s["ci90"][1])
        print(f"      median {s['median']:.4f}   68 % [{s['ci68'][0]:.4f}, "
              f"{s['ci68'][1]:.4f}] width {s['width68']:.4f}   "
              f"90 % width {s['width90']:.4f}   (68 % is text-only)")
    check("planted H0 reference line", H0_TRUTH, 67.74)

    print("\n  [b] R^F(f) = width[J9 | f] / width[S9 | f]")
    for r in sweep:
        print(f"    f = {r['f_AGN']:.3f}   R68^F {r['R68_F']:.6f}   "
              f"R90^F {r['R90_F']:.6f}")
    check("R68^F min", float(r68.min()), D["robustness_in_f"]["R68_F"]["min"])
    check("R68^F max", float(r68.max()), D["robustness_in_f"]["R68_F"]["max"])
    check("R90^F min", float(r90.min()), D["robustness_in_f"]["R90_F"]["min"])
    check("R90^F max", float(r90.max()), D["robustness_in_f"]["R90_F"]["max"])
    check("dashed reference, marginalised 68 %",
          ref68, round(D["fully_marginalised_reference"]["ratio68"], 4))
    check("dashed reference, marginalised 90 %",
          ref90, round(D["fully_marginalised_reference"]["ratio90"], 4))

    # ---- draw --------------------------------------------------------------- #
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(fs.TWOCOL, 2.55),
        gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.30})

    for name, key, color, s, lw in arms:
        curve(ax, h0, np.asarray(cur[key], float), s, color, name, lw)
    fs.truth_line(ax, H0_TRUTH, axis="x", label="planted 67.74", pos=0.52)
    # the window is the 28-node lattice; the axis is cropped to where the
    # densities are resolvable, at 1e-3 of the largest peak drawn, so the
    # limits are set by the data and not by eye
    pk = max(float(np.asarray(cur[k], float).max()) for _, k, _, _, _ in arms)
    live = np.zeros_like(h0, dtype=bool)
    for _, k, _, _, _ in arms:
        live |= np.asarray(cur[k], float) > 1e-3 * pk
    ax.set_xlim(float(h0[live].min()), float(h0[live].max()))
    ax.set_ylim(0.0, 1.40 * pk)   # headroom for the key, not a style choice
    ax.set_xlabel(r"$H_0$  [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_ylabel(r"$p(H_0 \mid f_{\rm AGN} = 0.275)$")
    ax.set_title("(a)  the host fraction held at its MAP node")
    ax.legend(loc="upper right")

    bx.plot(fnodes, r90, color=C_90, lw=1.4, marker="o", ms=3.0, zorder=3)
    bx.plot(fnodes, r68, color=C_68, lw=1.4, marker="s", ms=3.0, zorder=3)
    bx.axhline(ref90, color=C_90, lw=0.9, ls=(0, (3, 2)), zorder=1.5)
    bx.axhline(ref68, color=C_68, lw=0.9, ls=(0, (3, 2)), zorder=1.5)
    bx.annotate(r"$R^{F}_{90}$", (fnodes[0], r90[0]), xytext=(4, 5),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=7.5, color=fs.INK2)
    bx.annotate(r"$R^{F}_{68}$", (fnodes[0], r68[0]), xytext=(4, -6),
                textcoords="offset points", ha="left", va="top",
                fontsize=7.5, color=fs.INK2)
    bx.annotate(f"marginalised {ref90:.4f}", (fnodes[0] - 0.010, ref90),
                xytext=(0, 2), textcoords="offset points", ha="left", va="bottom",
                fontsize=6.8, color=fs.INK2)
    bx.annotate(f"marginalised {ref68:.4f}", (fnodes[0] - 0.010, ref68),
                xytext=(0, -3), textcoords="offset points", ha="left", va="top",
                fontsize=6.8, color=fs.INK2)
    bx.set_xlim(fnodes[0] - 0.012, fnodes[-1] + 0.008)
    bx.set_xlabel(r"$f_{\rm AGN}$  (node held fixed)")
    bx.set_ylabel(r"width$[\mathrm{J9} \mid f]\,/\,$width$[\mathrm{S9} \mid f]$")
    bx.set_title("(b)  across J9's 90 % interval in $f$")

    FIGS.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        out = _write_guard(FIGS / f"fig_mechanism_fixed_f.{ext}")
        fig.savefig(out, **kw)
        print(f"    wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
