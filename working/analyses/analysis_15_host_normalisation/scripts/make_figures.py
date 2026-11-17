#!/usr/bin/env python
"""Analysis 15 figure from results/normalisation_summary.json (deterministic; CPU, seconds).

    python make_figures.py      -> figs/normalisation_fagn.{pdf,png}

One panel: for each configuration (one row), the measured f_AGN median with its 90% interval
(filled circle and bar) and the normalisation identity's prediction from the configuration's
reference grid (open diamond and thin bar, 90%), with the stored legacy result on the same cells
where one exists (cross). Rows are grouped: complete catalogs (grid to 5 or to 1.10, AGN
density x1 or x2, and the zero-density reference), then m<21 and m<18. The input f_AGN = 0.30
is the black vertical line. Direct labels; no legend.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A15 = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(A15.parent / "common"))
import figstyle as fs  # noqa: E402

fs.use()
import matplotlib.pyplot as plt  # noqa: E402

ROWS = [("ref_complete_n0m24", "complete, zero density"),
        ("p1_complete", r"complete, $z<5$"),
        ("c_complete_x2", r"complete, $z<5$, AGN $n_0\times2$"),
        ("b_complete_x1", r"complete, $z<1.10$"),
        ("b_complete_x2", r"complete, $z<1.10$, AGN $n_0\times2$"),
        (None, None),
        ("p1_m21", r"$m<21$, $z<5$"),
        ("p1_m18", r"$m<18$, $z<5$"),
        ("d_m18_x05", r"$m<18$, $z<1.10$, AGN $n_0\times0.5$"),
        ("d_m18_x1", r"$m<18$, $z<1.10$"),
        ("d_m18_x2", r"$m<18$, $z<1.10$, AGN $n_0\times2$")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(A15 / "results"))
    ap.add_argument("--out", default=str(A15 / "figs" / "normalisation_fagn"))
    args = ap.parse_args()
    S = json.loads((Path(args.results) / "normalisation_summary.json").read_text())
    sc = S["scans"]
    fig, ax = plt.subplots(1, 1, figsize=(fs.ONECOL * 1.55, 3.6), constrained_layout=True)
    y = 0.0
    yt, yl = [], []
    first = {"m": True, "p": True, "l": True}
    cm, cp, cl = fs.C["blue"], fs.C["orange"], "0.45"
    for name, lab in ROWS:
        if name is None:
            y -= 0.6
            continue
        r = sc.get(name, {})
        yt.append(y)
        yl.append(lab)
        if r.get("status") == "done":
            m = r["measured"]["f"]
            ax.plot(m["ci90"], [y, y], color=cm, lw=1.6, solid_capstyle="butt", zorder=3)
            ax.plot([m["median"]], [y], "o", color=cm, ms=4.2, zorder=4)
            if first["m"]:
                ax.annotate("measured", (m["ci90"][1], y), xytext=(3, 0), textcoords="offset points",
                            ha="left", va="center", color=cm, fontsize=7)
                first["m"] = False
            pr = r.get("prediction") or {}
            if "predicted" in pr:
                p = pr["predicted"]["f"]
                yy = y - 0.28
                ax.plot(p["ci90"], [yy, yy], color=cp, lw=0.8, zorder=2)
                ax.plot([p["median"]], [yy], "D", mfc="white", mec=cp, mew=0.9, ms=3.6, zorder=3)
                if first["p"]:
                    ax.annotate("identity prediction", (p["ci90"][1], yy), xytext=(3, 0), textcoords="offset points",
                                ha="left", va="center", color=cp, fontsize=7)
                    first["p"] = False
            lg = r.get("legacy_same_cells")
            if lg is not None:
                ax.plot([lg["f"]["median"]], [y + 0.24], "x", color=cl, ms=3.6, mew=0.9, zorder=3)
                if first["l"]:
                    ax.annotate("legacy 0c5b3db", (lg["f"]["median"], y + 0.24), xytext=(-4, 0),
                                textcoords="offset points", ha="right", va="center", color=cl, fontsize=7)
                    first["l"] = False
        else:
            ax.text(0.31, y, "not run", color="0.5", fontsize=7, va="center")
        y -= 1.0
    fin = S.get("f_input", 0.30)
    ax.axvline(fin, color="black", lw=0.8, zorder=1)
    ax.annotate("input", (fin, y + 0.4), xytext=(3, 2), textcoords="offset points", ha="left", va="bottom",
                color="black", fontsize=7)
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=7.5)
    ax.set_ylim(y + 0.4, 0.75)
    ax.set_xlabel(r"$f_{\rm AGN}$ (median and 90% interval)")
    ax.set_xlim(0.0, 0.62)
    ax.tick_params(axis="y", length=0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200, metadata={"CreationDate": None} if ext == "pdf" else None)
    print(f"wrote {out}.pdf and {out}.png")


if __name__ == "__main__":
    main()
