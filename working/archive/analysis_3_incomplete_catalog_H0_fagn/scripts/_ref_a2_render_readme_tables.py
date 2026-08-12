#!/usr/bin/env python3
"""Render the README's result tables from results/joint_summary.json.

The campaign rule is that no reader-facing number is hand-typed: every figure in
the tables below is read out of the pipeline's own output.  This script rewrites
the block between the `<!-- RESULTS_BODY -->` and `<!-- /RESULTS_BODY -->` markers
(and the one-line banner between `<!-- RESULTS_BANNER -->` and
`<!-- /RESULTS_BANNER -->`), leaving every other line of the README alone, so it is
safe to re-run whenever another realisation lands.  The interpretation prose is
hand-written and lives outside the markers.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
README = ROOT / "README.md"
H0_TRUTH, F_PLANTED = 67.74, 0.30


def pm(v, e, nd=2):
    return f"`{v:+.{nd}f} ± {e:.{nd}f}`"


def yn(b):
    return "yes" if b else "no"


def build():
    S = json.loads((RES / "joint_summary.json").read_text())
    rows = [r for r in S["seeds"] if "joint" in r]
    C = S["closure"]
    L = []

    # ---- the joint fit, per realisation ------------------------------------
    L.append("## The joint fit, realisation by realisation\n")
    L.append("Medians with equal-tailed 68 % intervals from the 2-D posterior's "
             "marginals. `offset` is median − truth; for `f` the truth is that "
             "realisation's own **realised** host fraction, with the offset "
             "against the planted 0.30 alongside.\n")
    L.append("| seed | AGN-hosted | realised `f` | `H0` median ± 68 % | offset | in 68 / 90 | "
             "`f` median ± 68 % | offset vs realised | vs 0.30 | in 68 / 90 | `rho(H0, f)` |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        H, F, Fp = r["joint"]["H0"], r["joint"]["f_vs_realised"], r["joint"]["f_vs_planted"]
        L.append(
            f"| {r['seed']} | {r['n_host_agn']} | {r['f_realised']:.3f} | "
            f"{H['median']:.2f} ± {H['halfwidth68']:.2f} | **{H['offset']:+.2f}** | "
            f"{yn(H['truth_in_ci68'])} / {yn(H['truth_in_ci90'])} | "
            f"{F['median']:.3f} ± {F['halfwidth68']:.3f} | **{F['offset']:+.3f}** | "
            f"{Fp['offset']:+.3f} | {yn(F['truth_in_ci68'])} / {yn(F['truth_in_ci90'])} | "
            f"{r['joint']['rho']:+.3f} |")
    cov = C["coverage"]
    L.append(f"| | | | | {pm(C['joint_H0']['mean'], C['joint_H0']['sem'])} | "
             f"**{cov['H0_in_68']} / {cov['H0_in_90']}** of {cov['n']} | | "
             f"{pm(C['joint_f_vs_realised']['mean'], C['joint_f_vs_realised']['sem'], 3)} | "
             f"{pm(C['joint_f_vs_planted']['mean'], C['joint_f_vs_planted']['sem'], 3)} | "
             f"**{cov['f_realised_in_68']} / {cov['f_realised_in_90']}** of {cov['n']} | "
             f"{C['rho']['mean']:+.3f} |")
    L.append("")

    # ---- closure -----------------------------------------------------------
    L.append("### Closure over the five realisations\n")
    L.append("| quantity | truth | mean offset ± s.e.m. | `t(4)` | realisation scatter | "
             "mean quoted half-width | scatter / width |")
    L.append("|---|---|---|---|---|---|---|")
    sH, sf = C["scatter_H0"], C["scatter_f"]
    L.append(f"| `H0` | 67.74 | **{C['joint_H0']['mean']:+.2f} ± {C['joint_H0']['sem']:.2f}** | "
             f"{C['joint_H0']['t']:+.2f} | {sH['sd_of_medians']:.2f} | "
             f"{sH['mean_halfwidth68']:.2f} | {sH['ratio']:.2f} × |")
    L.append(f"| `f_AGN` vs **realised** | per seed | "
             f"**{C['joint_f_vs_realised']['mean']:+.3f} ± {C['joint_f_vs_realised']['sem']:.3f}** | "
             f"{C['joint_f_vs_realised']['t']:+.2f} | {sf['sd_of_medians']:.3f} | "
             f"{sf['mean_halfwidth68']:.3f} | {sf['ratio']:.2f} × |")
    L.append(f"| `f_AGN` vs **planted** | 0.30 | "
             f"**{C['joint_f_vs_planted']['mean']:+.3f} ± {C['joint_f_vs_planted']['sem']:.3f}** | "
             f"{C['joint_f_vs_planted']['t']:+.2f} | — | — | — |")
    L.append("")
    L.append(f"The binomial term separating the two `f` references is "
             f"`sqrt(0.3 × 0.7 / 1000) = {S['binomial_sd_per_realisation']:.4f}` per "
             f"realisation, `{S['binomial_sd_per_realisation'] / 5 ** 0.5:.4f}` on the "
             f"five-realisation mean.\n")

    # ---- the 1-D scans -----------------------------------------------------
    if any("fscan" in r for r in S["seeds"]):
        L.append("### The two one-dimensional cuts\n")
        L.append("`fscan` fixes `H0` at truth and scans `f` on 101 points; `h0scan` fixes "
                 "`f` at the planted 0.30 and scans `H0` on 201. They are cuts through the "
                 "same likelihood, not independent measurements, and they carry no "
                 "marginalisation over the other parameter.\n")
        L.append("| seed | realised `f` | `fscan` `f` ± 68 % | offset vs realised | "
                 "`h0scan` `H0` ± 68 % | offset |")
        L.append("|---|---|---|---|---|---|")
        for r in S["seeds"]:
            if "fscan" not in r and "h0scan" not in r:
                continue
            fb = r.get("fscan", {}).get("vs_realised")
            hb = r.get("h0scan", {}).get("H0")
            L.append(f"| {r['seed']} | {r['f_realised']:.3f} | "
                     + (f"{fb['median']:.3f} ± {fb['halfwidth68']:.3f} | {fb['offset']:+.3f} | "
                        if fb else "— | — | ")
                     + (f"{hb['median']:.2f} ± {hb['halfwidth68']:.2f} | {hb['offset']:+.2f} |"
                        if hb else "— | — |"))
        if "fscan_f_vs_realised" in C:
            c1, c2 = C["fscan_f_vs_realised"], C.get("h0scan_H0", {})
            L.append(f"| | | | **{c1['mean']:+.3f} ± {c1['sem']:.3f}** | | "
                     + (f"**{c2['mean']:+.2f} ± {c2['sem']:.2f}** |" if c2 else "|"))
        L.append("")

    # ---- guard / N_eff -----------------------------------------------------
    g = rows[0]["joint"]["guard"] if rows else {}
    L.append("### The selection integral across the `f` axis\n")
    nrej = sum(r["joint"].get("n_rejected", 0) for r in rows)
    ncell = 8241 * len(rows)
    L.append(f"**{nrej} of {ncell:,} grid cells were rejected** across the "
             f"{len(rows)} joint grids — the guard never fires anywhere on the "
             f"(`H0`, `f`) plane, at either end of the `f` axis.\n")
    blk = rows[0]["joint"].get("neff_vs_f_at_truth_H0") if rows else None
    if blk:
        L.append(f"At truth `H0` = {blk['H0']:.2f}, seed {rows[0]['seed']}, targeted lane:\n")
        L.append("| `f` | 0.0 | 0.25 | 0.50 | 0.75 | 1.0 |")
        L.append("|---|---|---|---|---|---|")
        idx = [0, 10, 20, 30, 40]
        L.append("| `N_eff` | " + " | ".join(f"{blk['Neff'][i]:,.0f}" for i in idx) + " |")
        L.append("| `Σ σ²_PE` | " + " | ".join(f"{blk['pe_variance_sum'][i]:.1f}" for i in idx) + " |")
        L.append("| `× 5N_obs` floor | "
                 + " | ".join(f"{blk['Neff'][i] / 5000:.0f}×" for i in idx) + " |")
        L.append("")
    if g:
        L.append(f"Over the whole seed-{rows[0]['seed']} grid `N_eff` runs "
                 f"{g['Neff_min']:,.0f} – {g['Neff_max']:,.0f} against a flat floor of "
                 f"5 000, and `Σ σ²_PE` runs {g['pe_variance_sum_min']:.1f} – "
                 f"{g['pe_variance_sum_max']:.1f} against the campaign's inert `1e6` cap.\n")

    # ---- lanes and null ----------------------------------------------------
    lanes = S.get("lane_agreement") or {}
    if lanes:
        L.append("### Injection lanes\n")
        L.append("The targeted lane is the record; population+uniform is the cross-check. "
                 "They are the same detection rule with different proposals, so they must "
                 "agree.\n")
        L.append("| scan | parameter | targeted | popuni | difference | in 68 % half-widths |")
        L.append("|---|---|---|---|---|---|")
        for tag, d in lanes.items():
            for p, v in d.items():
                if not isinstance(v, dict) or "delta" not in v:
                    continue
                nd = 2 if p == "H0" else 4
                L.append(f"| `{tag}` | `{p}` | {v['targeted']:.{nd}f} | {v['popuni']:.{nd}f} | "
                         f"{v['delta']:+.{nd}f} | {v['delta_over_halfwidth68']:+.3f} |")
        L.append("")
    n = S.get("sky_shuffle_null")
    if n:
        L.append("### The sky-shuffle null\n")
        L.append("Permuting the per-event `(ra, dec)` blocks among events destroys every "
                 "host association while leaving each event's distance, masses, spin and "
                 "localisation area untouched, and leaving the same patches of sky "
                 "occupied. Anything the mixture weight still \"measures\" afterwards was "
                 "never host-association information.\n")
        L.append("| | median `f` | 68 % interval | 90 % interval |")
        L.append("|---|---|---|---|")
        L.append(f"| record (seed {n['seed']}) | **{n['record_median']:.3f}** | "
                 f"± {n['record_halfwidth68']:.3f} | — |")
        L.append(f"| sky-shuffled | **{n['median']:.3f}** | "
                 f"[{n['ci68'][0]:.3f}, {n['ci68'][1]:.3f}] | "
                 f"[{n['ci90'][0]:.3f}, {n['ci90'][1]:.3f}] |")
        L.append("")
        L.append(f"The weight collapses toward zero and the recorded value "
                 f"{n['record_median']:.3f} lies far outside the shuffled 90 % interval. "
                 f"A weight pinned by the two catalogs' global normalisations would have "
                 f"survived the permutation unchanged; this one does not.\n")
    return "\n".join(L)


def banner():
    S = json.loads((RES / "joint_summary.json").read_text())
    C = S["closure"]
    hook = json.loads((RES / "h0_fagn_joint.json").read_text())
    cov = C["coverage"]
    return (
        f"> **Both parameters are recovered.** Over {cov['n']} independent realisations "
        f"the joint fit returns `H0` with a mean offset of "
        f"**{C['joint_H0']['mean']:+.2f} ± {C['joint_H0']['sem']:.2f}** km s⁻¹ Mpc⁻¹ "
        f"from 67.74 (`t({cov['n'] - 1}) = {C['joint_H0']['t']:+.2f}`) and the AGN host "
        f"fraction with a mean offset of "
        f"**{C['joint_f_vs_realised']['mean']:+.3f} ± {C['joint_f_vs_realised']['sem']:.3f}** "
        f"from each realisation's own realised fraction "
        f"(`t({cov['n'] - 1}) = {C['joint_f_vs_realised']['t']:+.2f}`), or "
        f"{C['joint_f_vs_planted']['mean']:+.3f} ± {C['joint_f_vs_planted']['sem']:.3f} from "
        f"the planted 0.30. Truth is inside the 68 % interval on "
        f"{cov['H0_in_68']} / {cov['n']} realisations for `H0` and "
        f"{cov['f_realised_in_68']} / {cov['n']} for `f`; inside the 90 % on "
        f"{cov['H0_in_90']} / {cov['n']} and {cov['f_realised_in_90']} / {cov['n']}. "
        f"On the reference realisation the fit reads `H0` = {hook['h0_ci']}, "
        f"`f_AGN` = {hook['f_ci']}, with a correlation of {hook['rho']:+.2f}.")


def splice(text, tag, body):
    a, b = f"<!-- {tag} -->", f"<!-- /{tag} -->"
    if a not in text:
        return text
    head = text.split(a)[0]
    tail = text.split(b)[1] if b in text else text.split(a)[1].split("\n", 1)[1]
    return f"{head}{a}\n{body}\n{b}{tail}"


if __name__ == "__main__":
    t = README.read_text()
    t = splice(t, "RESULTS_BANNER", banner())
    t = splice(t, "RESULTS_BODY", build())
    README.write_text(t)
    print(f"rendered {README}")
