#!/usr/bin/env python
"""Specification section 10 figure: the per-event evidence plane. Deterministic.

    python fig_event_evidence_plane.py   # writes ../figs/fig_event_evidence_plane.{pdf,png}

Reads ONLY results/event_decomposition.{h5,json}; computes nothing.

Left  -- log BF_spatial against log BF_intrinsic, points by TRUE host label.
Right -- the same plane, points by the combined P_i(AGN) the inference assigns
         without ever seeing that label.

The x axis is symlog (linear inside |x| < 1): the spatial Bayes factor spans
-295 to +6.7, because an event whose sky-and-distance volume holds no AGN host
candidate is excluded from the AGN branch outright, while the spin mark can only
ever shift the intrinsic factor by a few nats.  A linear axis would hide that.

The rule is the P = 0.5 boundary, log BF_spatial + log BF_intrinsic =
log(f_GAL/f_AGN) = +0.969 at this hyperparameter point -- a property of the
mixture prior, not a classifier threshold chosen here.

Colours: Okabe-Ito, the campaign's fixed categorical order (analyses 3-7).
"""
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
A8 = HERE.parent
RESULTS = A8 / "results"
FIGS = A8 / "figs"

GAL_C, AGN_C = "#0072B2", "#D55E00"       # Okabe-Ito blue / vermillion
INK, MUTED = "#1a1a1a", "#6b6b6b"


def main():
    with h5py.File(RESULTS / "event_decomposition.h5", "r") as h:
        x = h["log_BF_spatial"][:]
        y = h["log_BF_intrinsic"][:]
        p = h["P_AGN"][:]
        t = h["true_host_type"][:]
        f_agn = float(h.attrs["f_agn"])
        mu = float(h.attrs["mu_chi_c2"])
        lf_g, lf_a = float(h.attrs["log_f_gal"]), float(h.attrs["log_f_agn"])
    summary = json.loads((RESULTS / "event_decomposition.json").read_text())
    auc = summary["separation"]["P_AGN"]["auc"]
    is_agn = t == 1
    boundary = lf_g - lf_a                      # log BF_total at P = 0.5

    FIGS.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), sharex=True, sharey=True)

    xs = np.concatenate([-np.logspace(np.log10(400), 0, 200), np.linspace(-1, 8, 200)])
    for ax in axes:
        ax.plot(xs, boundary - xs, color=INK, lw=1.0, ls="--", zorder=1)
        ax.axhline(0.0, color=MUTED, lw=0.6, zorder=0)
        ax.axvline(0.0, color=MUTED, lw=0.6, zorder=0)
        ax.set_xscale("symlog", linthresh=1.0, linscale=1.4)
        ax.set_xlim(-400, 9)
        ax.set_ylim(-3.2, 2.9)
        ax.set_xticks([-100, -10, -1, 0, 1, 5])
        ax.set_xticklabels(["-100", "-10", "-1", "0", "1", "5"])
        ax.tick_params(labelsize=9)
        ax.set_xlabel(r"$\log\,\mathrm{BF}_{i,\rm spatial}$")

    axes[0].scatter(x[~is_agn], y[~is_agn], s=13, c=GAL_C, alpha=0.65,
                    lw=0, label=f"true GAL ({(~is_agn).sum()})", zorder=3)
    axes[0].scatter(x[is_agn], y[is_agn], s=13, c=AGN_C, alpha=0.8,
                    lw=0, label=f"true AGN ({is_agn.sum()})", zorder=4)
    axes[0].legend(loc="lower left", frameon=False, fontsize=9)
    axes[0].set_ylabel(r"$\log\,\mathrm{BF}_{i,\rm intrinsic}$")
    axes[0].set_title("true host label (mock diagnostic)", fontsize=10, color=INK)

    sc = axes[1].scatter(x, y, s=13, c=p, cmap="cividis", vmin=0.0, vmax=1.0,
                         lw=0, zorder=3)
    cb = fig.colorbar(sc, ax=axes[1], pad=0.02)
    cb.set_label(r"$P_i(\mathrm{AGN})$", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    axes[1].set_title(r"inferred $P_i(\mathrm{AGN})$, AUC = %.3f" % auc,
                      fontsize=10, color=INK)

    fig.suptitle(r"seed 100, joint arm at the MAP  "
                 r"$f_{\rm AGN}=%.3f$, $\Delta\mu_\chi=%+.4f$   "
                 r"(dashed: $P_i=0.5$)" % (f_agn, mu),
                 fontsize=10, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("pdf", "png"):
        out = FIGS / f"fig_event_evidence_plane.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
