#!/usr/bin/env python3
"""Render analysis_4's README results banner and arm tables from arms_summary.json.

Rewrites, in place, the two marked regions of ../README.md:

  <!-- RESULTS_BANNER -->  ... <!-- /RESULTS_BANNER -->
  <!-- ARM_TABLES -->      ... <!-- /ARM_TABLES -->

(the second block is appended under "## Results" if it is not there yet).  Every
number is read from results/arms_summary.json, so the README cannot drift from
the grids; running this against a partial campaign is safe and says so.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
README = ROOT / "README.md"
LEVELS = ["m21", "m20", "m19", "m18"]
PRETTY = {"m21": "m < 21", "m20": "m < 20", "m19": "m < 19", "m18": "m < 18"}
ARM_ORDER = ["a05", "a07", "a09", "exact", "a11", "a13", "a20"]


def replace_block(text, tag, body):
    o, c = f"<!-- {tag} -->", f"<!-- /{tag} -->"
    if o in text and c in text:
        head, rest = text.split(o, 1)
        _, tail = rest.split(c, 1)
        return f"{head}{o}\n{body}\n{c}{tail}"
    return f"{text.rstrip()}\n\n{o}\n{body}\n{c}\n"


def banner(S):
    p = S["progress"]
    partial = p["n_grids_present"] < p["n_grids_expected"]
    lines = []
    if partial:
        lines.append(f"> **Campaign in progress — {p['n_grids_present']} of "
                     f"{p['n_grids_expected']} grids on disk.** Numbers below "
                     "cover the arms that have landed.")
        lines.append(">")
    # the headline, taken from the deepest rung with a full ±2 sweep
    best = None
    for lev in LEVELS:
        arms = S["rungs"][lev]["arms"]
        if all(arms.get(a, {}).get("present") for a in ("a05", "a20", "exact")):
            best = lev
            break
    if best:
        arms = S["rungs"][best]["arms"]
        lo, ex, hi = arms["a05"], arms["exact"], arms["a20"]
        lines.append(
            f"> Mis-anchoring the completion's AGN density propagates almost "
            f"entirely into $f_{{\\rm AGN}}$ itself: at {PRETTY[best]}, halving "
            f"the assumed density moves the recovered fraction to "
            f"{lo['f_vs_realised']['median']:.3f} and doubling it to "
            f"{hi['f_vs_realised']['median']:.3f}, against "
            f"{ex['f_vs_realised']['median']:.3f} at the true anchor and a "
            f"realised {S['truth']['f_realised']:.3f} — a "
            f"{lo['vs_exact']['f']['delta_median_over_ref_halfwidth68']:+.1f}σ "
            f"and {hi['vs_exact']['f']['delta_median_over_ref_halfwidth68']:+.1f}σ "
            f"shift in units of the exact arm's own 68 % half-width.")
        lines.append(">")
        lines.append(
            f"> The error moves with the median, so the *detection* of an AGN "
            f"component survives: the significance runs "
            f"{lo['significance_f']:.1f}σ → {ex['significance_f']:.1f}σ → "
            f"{hi['significance_f']:.1f}σ across the same factor-4 range in "
            f"assumed density. $H_0$ is the resilient parameter, shifting "
            f"{lo['vs_exact']['H0']['delta_median_over_ref_halfwidth68']:+.2f} to "
            f"{hi['vs_exact']['H0']['delta_median_over_ref_halfwidth68']:+.2f} of "
            f"its own half-width over the same range.")
    O = S.get("oracle") or {}
    if O.get("present"):
        lines.append(">")
        f = O["f_vs_realised"]
        frac = O.get("bias_removed_fraction")
        lines.append(
            f"> The oracle probe settles the faintest rung: handing the model a "
            f"complete AGN survey while the galaxies stay at m < 18 gives "
            f"$f_{{\\rm AGN}} = {f['median']:.3f} \\pm {f['halfwidth68']:.3f}$, "
            f"offset {f['offset']:+.3f}"
            + (f" — {100 * frac:.0f} % of the m < 18 bias removed."
               if frac is not None else "."))
    return "\n".join(lines) if lines else "> *(no arms on disk yet)*"


def arm_table(S, lev):
    arms = S["rungs"][lev]["arms"]
    got = [a for a in ARM_ORDER if arms.get(a, {}).get("present")]
    if not got:
        return None
    out = [f"**{PRETTY[lev]}**", "",
           "| assumed / true $n_{0,\\rm AGN}$ | $\\log_{10} n_{0,c2}$ | "
           "$H_0$ | offset | $f_{\\rm AGN}$ | offset | $f$ significance | "
           "$\\Delta f$ vs exact |",
           "|---:|---:|:---|---:|:---|---:|---:|---:|"]
    for a in got:
        r = arms[a]
        H, F = r["H0"], r["f_vs_realised"]
        d = (f"{r['vs_exact']['f']['delta_median_over_ref_halfwidth68']:+.2f}σ"
             if r.get("vs_exact") else "—")
        name = ("**1.0** (exact)" if a == "exact" else f"{r['factor']:g}")
        out.append(
            f"| {name} | {r['log10n0_c2']:.3f} | "
            f"${H['median']:.2f}^{{+{H['plus68']:.2f}}}_{{-{H['minus68']:.2f}}}$ | "
            f"{H['offset']:+.2f} | "
            f"${F['median']:.3f}^{{+{F['plus68']:.3f}}}_{{-{F['minus68']:.3f}}}$ | "
            f"{F['offset']:+.3f} | {r['significance_f']:.1f}σ | {d} |")
    return "\n".join(out)


def main():
    S = json.loads((RES / "arms_summary.json").read_text())
    t = S["truth"]
    tables = [f"All numbers: seed {S['seed']}, targeted-injection lane, "
              f"truth $H_0 = {t['H0']}$, realised host fraction "
              f"{t['f_realised']:.3f} ({t['n_host_agn']}/{t['n_events']}), "
              f"planted {t['f_planted']}.  The exact arm is analysis_3's own "
              f"grid, referenced not rerun.  Significance is median / 68 % "
              f"half-width; the last column is the shift from the exact arm in "
              f"units of that arm's half-width.", ""]
    for lev in LEVELS:
        tb = arm_table(S, lev)
        if tb:
            tables += [tb, ""]
    O = S.get("oracle") or {}
    if O.get("present"):
        H, F = O["H0"], O["f_vs_realised"]
        tables += [
            "**Oracle probe** — galaxies at m < 18, AGN survey complete, both "
            "densities at truth", "",
            "| | $H_0$ | offset | $f_{\\rm AGN}$ | offset |",
            "|:--|:--|---:|:--|---:|",
            f"| oracle | ${H['median']:.2f}^{{+{H['plus68']:.2f}}}_{{-{H['minus68']:.2f}}}$ | "
            f"{H['offset']:+.2f} | "
            f"${F['median']:.3f}^{{+{F['plus68']:.3f}}}_{{-{F['minus68']:.3f}}}$ | "
            f"{F['offset']:+.3f} |", ""]
    body = "\n".join(tables).rstrip()

    text = README.read_text()
    text = replace_block(text, "RESULTS_BANNER", banner(S))
    if "<!-- ARM_TABLES -->" not in text:
        text = text.rstrip() + "\n\n## Results\n\n"
    text = replace_block(text, "ARM_TABLES", body)
    README.write_text(text)
    print(f"Wrote {README}")


if __name__ == "__main__":
    main()
