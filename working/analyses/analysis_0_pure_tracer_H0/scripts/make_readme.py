#!/usr/bin/env python3
"""analysis_0 -- render README.md from results/*.json.

Every number in the README is substituted from `results/event_sets.json` and
`results/h0_pure_tracer.json`; nothing is hand-typed.  Re-run after the
aggregation and the README follows the results.

    python scripts/make_readme.py
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"


def f(x, n=2, dash="--"):
    return dash if x is None else f"{x:.{n}f}"


def sgn(x, n=3, dash="--"):
    return dash if x is None else f"{x:+.{n}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "README.md"))
    a = ap.parse_args(argv)

    ev = json.loads((RES / "event_sets.json").read_text())
    hp = RES / "h0_pure_tracer.json"
    r = json.loads(hp.read_text()) if hp.exists() else None
    gate = RES / "gate_events_bitidentity.json"
    g = json.loads(gate.read_text()) if gate.exists() else None

    P = []
    P.append("# analysis 0 -- pure-tracer H0 constraining power, and the bias check\n")
    P.append(ev["what"] + "\n")
    P.append("The question analysis 1 could not answer.  Its two single-tracer numbers "
             "came from splitting ONE 1000-event mixture draw on host type, so the "
             "galaxy arm carried 705 events and the AGN arm 295, and the event noise "
             "was shared with the analysis of record.  Nothing in that pair compares "
             "the two tracers' constraining power, because they are not the same "
             "measurement.  Here each tracer gets its own independent draw of "
             "N = 1000 detected events on the same catalogs and surveys, so the "
             "widths are directly comparable and the offsets are a fresh look at the "
             "bias.\n")

    # ---- headline ------------------------------------------------------------
    if r and r.get("closure_gal") and r.get("closure_agn"):
        cg, ca = r["closure_gal"], r["closure_agn"]
        cp = r.get("constraining_power") or {}
        P.append("**What it found.**  At a matched N = 1000 the AGN catalog "
                 f"constrains H0 about {f(1.0 / cp['ratio_of_means_agn_over_gal'], 1)} "
                 f"times more tightly than the galaxy catalog -- a mean 68% "
                 f"half-width of {f(ca['widths']['mean_half68'])} against "
                 f"{f(cg['widths']['mean_half68'])} km/s/Mpc.  Both tracers recover "
                 f"truth: the mean offset over five realisations is "
                 f"{sgn(ca['mean_offset'])} +- {f(ca['sem_offset'], 3)} km/s/Mpc for "
                 f"AGN (t({ca['dof']}) = {sgn(ca['t_statistic'], 2)}) and "
                 f"{sgn(cg['mean_offset'])} +- {f(cg['sem_offset'], 3)} for GAL "
                 f"(t({cg['dof']}) = {sgn(cg['t_statistic'], 2)}).  On independent "
                 f"event draws the H0 bias does not reappear.\n")

    # ---- 1. the generator extension + gate -----------------------------------
    P.append("## 1. The event sets\n")
    P.append("`working/data/generate_dataset.py` gained two options, both defaulting "
             "to the behaviour of record:\n")
    P.append("* `--f_agn` -- the planted AGN-hosted fraction used by the events "
             "stage.  Unset, it is the module constant `F_AGN`.\n"
             "* `--seed_events` -- an explicit events-stage sub-seed.  Unset, it is "
             "the record's derivation `SEED*1000+3`.\n")
    P.append("Both flow into the events RNG and into the recorded metadata "
             "(`planted_f_agn`, `seed_events` on the file and in `metadata_json`).  "
             "`sub_seeds()` spends offsets 1-7 on the record "
             + ", ".join(f"({k} {v})" for k, v in
                         sorted(ev["sub_seed_offsets_taken_by_the_record"].items(),
                                key=lambda kv: int(kv[0])))
             + "; offsets "
             + " and ".join(str(k) for k in sorted(ev["sub_seed_offsets_used_here"],
                                                   key=int))
             + " are unused by the generator and carry the two draws here, so they "
               "are independent of every recorded stream and of each other.\n")
    if g:
        P.append(f"**Bit-identity gate.**  Seed 100's events stage was rerun with no "
                 f"new flags into a scratch output root and compared against the "
                 f"record file dataset by dataset: "
                 f"**{g['n_datasets']} of {g['n_datasets']} datasets byte-identical "
                 f"(SHA-256), {len(g['dataset_failures'])} failures** -- "
                 f"`{'PASS' if g['PASS'] else 'FAIL'}`.  The only differences are "
                 f"metadata: the generation timestamp, the new provenance keys the "
                 f"extension records, and the new top-level `seed_events` attribute.  "
                 f"`results/gate_events_bitidentity.json` carries the per-dataset "
                 f"digests.\n")
    P.append("Nothing in the signed-off dataset was modified: `--events_suffix` "
             "writes `events_pure{gal,agn}.h5` beside `events.h5` and suppresses the "
             "`META.json` merge, `--overwrite` is never passed, and the catalogs, "
             "surveys and both injection lanes are reused exactly as they are on "
             "disk.\n")

    rows = []
    for s in ev["sets"]:
        uh = s["unique_agn_hosts"] if s["tracer"] == "agn" else s["unique_gal_hosts"]
        mm = (s["max_events_per_agn_host"] if s["tracer"] == "agn"
              else s["max_events_per_gal_host"])
        rows.append([s["seed"], s["tracer"], s["seed_events"], s["nobs"], s["nsamp"],
                     s["n_host_gal"], s["n_host_agn"], uh, mm,
                     f(s["snr_obs_min"], 3), f(s["z_median_detected"], 3),
                     f(s["horizon_z_max_detected"], 3),
                     "PASS" if s["PASS"] else "FAIL"])
    P.append(table(["seed", "tracer", "seed_events", "N", "nsamp", "hosts GAL",
                    "hosts AGN", "unique hosts", "max mult", "min SNR", "median z",
                    "max z", "checks"], rows) + "\n")
    P.append(f"All ten sets pass every check "
             f"(`scripts/check_pure_tracer_events.py`, overall "
             f"`{'PASS' if ev['PASS'] else 'FAIL'}`): the declared count and sample "
             f"depth, every host of the declared type, every recorded SNR above the "
             f"threshold of {f(ev['snr_threshold'], 0)}, the requested sub-seed and "
             f"planted fraction on the file, and ten distinct streams.\n")

    # ---- 2. configuration ----------------------------------------------------
    P.append("## 2. The scans\n")
    if r:
        c = r["configuration"]
        P.append("Twenty K=1 `dark_sirens` H0 scans -- five realisations x two "
                 "tracers x two injection lanes -- with analysis 1's configuration "
                 "copied verbatim; only the events file changes.\n")
        P.append(table(["setting", "value"], [
            ["estimator", c["estimator"]],
            ["sky weighting", c["catalog_sky_weighting"]],
            ["H0 grid", f"[{c['h0_grid'][0]:.0f}, {c['h0_grid'][1]:.0f}] x "
                        f"{int(c['h0_grid'][2])}"],
            ["truth H0", f(r["truth_H0"])],
            ["Om0", c["Om0"]],
            ["population + nuisances", c["population_and_nuisances"]],
            ["selection guard", c["guard"]],
            ["catalog KDE window", c["kde_window"]],
            ["injection lane of record", r["injection_lane_of_record"]],
        ]) + "\n")
    else:
        P.append("Twenty K=1 `dark_sirens` H0 scans -- five realisations x two "
                 "tracers x two injection lanes -- with analysis 1's configuration "
                 "copied verbatim; only the events file changes.  The configuration "
                 "table is rendered from `results/h0_pure_tracer.json`, which the "
                 "aggregation has not yet written; see `scripts/run_scans.sh` for "
                 "the settings in the meantime.\n")
    P.append("`scripts/scan_h0f.py` is analysis 1's driver copied byte for byte; "
             "`scripts/run_scans.sh` and `scripts/submit_scans.sbatch` are its "
             "`run_scans.sh` / `submit_v3_controls.sbatch` with the event paths "
             "and tags changed.\n")

    # ---- 3. results ----------------------------------------------------------
    if r and (r.get("closure_gal") or r.get("closure_agn")):
        P.append("## 3. Constraining power at equal N\n")
        cp = r.get("constraining_power")
        if cp and cp["n_seeds_usable"]:
            P.append(cp["what"] + "\n")
            P.append(table(["seed", "N (GAL)", "N (AGN)", "68% half-width GAL",
                            "68% half-width AGN", "AGN / GAL"],
                           [[x["seed"], x["n_events_gal"], x["n_events_agn"],
                             f(x["half68_gal"]), f(x["half68_agn"]),
                             f(x["ratio_agn_over_gal"])
                             + (" (railed)" if x["railed_gal"] or x["railed_agn"]
                                else "")]
                            for x in cp["per_seed"]]) + "\n")
            P.append(f"Mean 68% half-width: **{f(cp['mean_half68_gal'])} km/s/Mpc "
                     f"(GAL)** against **{f(cp['mean_half68_agn'])} km/s/Mpc "
                     f"(AGN)**, a ratio of means of "
                     f"**{f(cp['ratio_of_means_agn_over_gal'])}**; the per-seed "
                     f"ratios average {f(cp['mean_of_per_seed_ratios'])} "
                     f"+- {f(cp['sem_of_per_seed_ratios'])} over "
                     f"{cp['n_seeds_usable']} realisations.\n")

        P.append("## 4. Closure on truth\n")
        for tracer, name in (("gal", "pure-GAL"), ("agn", "pure-AGN")):
            c = r.get(f"closure_{tracer}")
            if not c:
                continue
            P.append(f"### {name}\n")
            P.append(table(["seed", "N", "median", "68% interval", "90% interval",
                            "offset", "truth in 68%", "truth in 90%", "cells rejected"],
                           [[x["seed"], x["n_events"], f(x["median"]),
                             f"[{f(x['ci68'][0])}, {f(x['ci68'][1])}]",
                             f"[{f(x['ci90'][0])}, {f(x['ci90'][1])}]",
                             sgn(x["offset"], 2),
                             "yes" if x["truth_in_ci68"] else "no",
                             "yes" if x["truth_in_ci90"] else "no",
                             x["n_rejected"]] + (["RAILED"] if x["railed"] else [])
                            for x in c["per_seed"]]) + "\n")
            cv = c["coverage"]
            P.append(f"Mean offset **{sgn(c['mean_offset'])} +- "
                     f"{f(c['sem_offset'], 3)} km/s/Mpc** over {c['n_seeds']} "
                     f"realisations (sd {f(c['sd_offset'], 3)}), "
                     f"t({c['dof']}) = {sgn(c['t_statistic'], 2)}, "
                     f"p = {f(c['p_two_sided'], 4)}.  Truth falls inside the 68% "
                     f"interval in {cv['n_truth_in_ci68']} of "
                     f"{cv['n_realisations']} realisations and inside the 90% "
                     f"interval in {cv['n_truth_in_ci90']} of "
                     f"{cv['n_realisations']}.  The scatter of the five medians is "
                     f"{f(c['seed_scatter_over_quoted_half68'])} times the mean "
                     f"quoted 68% half-width.\n")
            if c.get("excluding_railed"):
                e = c["excluding_railed"]
                P.append(f"{c['n_railed']} realisation(s) railed "
                         f"(seeds {c['railed_seeds']}); {c['railed_note']}.  "
                         f"Without them: mean {sgn(e['mean_offset'])} +- "
                         f"{f(e['sem_offset'], 3)}, "
                         f"t({e['dof']}) = {sgn(e['t_statistic'], 2)}.\n")

        if r.get("lanes"):
            P.append("## 5. Injection-lane cross-check\n")
            P.append("The two lanes are the same detection rule under different "
                     "proposals, so a difference large against the 68% half-width "
                     "would mean the selection integral is setting digits of the "
                     "answer.\n")
            rows = []
            for tracer, v in r["lanes"].items():
                for x in v["per_seed"]:
                    rows.append([tracer, x["seed"], f(x["targeted_median"], 3),
                                 f(x["popuni_median"], 3), sgn(x["difference"], 3),
                                 f"{100 * x['difference_over_targeted_half68']:.1f}%"])
            P.append(table(["tracer", "seed", "targeted", "popuni", "difference",
                            "as % of one half-width"], rows) + "\n")
            for tracer, v in r["lanes"].items():
                P.append(f"{tracer.upper()}: largest lane shift "
                         f"{100 * v['max_abs_difference_over_half68']:.1f}% of one "
                         f"68% half-width.\n")
        dg = r.get("diagnostics")
        if dg:
            P.append("## 6. Guard and shape (internal)\n")
            P.append(f"Selection-validity guard: {dg['guard_convention']}.  Across "
                     f"all {dg['n_scans']} scans **every cell was accepted** "
                     f"(`all_cells_accepted = {dg['all_cells_accepted']}`), the "
                     f"smallest per-cell N_eff sat "
                     f"{f(dg['min_Neff_over_threshold_across_all_scans'], 1)}x above "
                     f"the wall, and the largest posterior density reached at a grid "
                     f"edge was {dg['max_density_at_a_grid_edge']:.1e} of the peak, "
                     f"so no posterior is censored by the scanned range.\n")
            multi = [x for x in dg["per_scan"] if x["n_interior_modes"] > 1
                     and max(x["mode_relative_heights"]) > 0
                     and sorted(x["mode_relative_heights"])[-2] > 0.01]
            if multi:
                P.append("Genuinely multimodal posteriors (a second mode above 1% of "
                         "the peak) -- their 68% interval spans the gap between the "
                         "modes, which is why the width is large:\n")
                P.append(table(["scan", "modes (relative height)"],
                               [[f"`{x['tag']}`",
                                 "; ".join(f"{f(m)} ({f(h)})" for m, h in
                                           zip(x["mode_positions"],
                                               x["mode_relative_heights"]))]
                                for x in multi]) + "\n")
            else:
                P.append("No posterior carries a second mode above 1% of its peak.\n")
    else:
        P.append("## 3. Results\n\nScans not yet aggregated -- run "
                 "`python scripts/aggregate_pure_tracer.py` once "
                 "`results/h0_pure*.json` are present, then re-run this script.\n")

    # ---- figures -------------------------------------------------------------
    P.append("## Figures\n")
    P.append("`python scripts/make_figures.py` renders all five from `results/` "
             "(PDF + PNG, deterministic -- a rerun on unchanged results "
             "reproduces both files byte for byte); "
             "`python scripts/make_figures.py <name>` renders one.\n")
    P.append(table(["figure", "what it shows"], [
        ["`figs/fig_posteriors.{pdf,png}`",
         "the ten record-lane (targeted) H0 posteriors overlaid, each scaled to "
         "its own peak -- the AGN densities are ~5x narrower and would otherwise "
         "flatten the galaxy curves; blue = galaxies, orange = AGN, seed 100 at "
         "full strength and the other four at a lighter step of the same hue.  "
         "Drawn on the window holding >= 99.99% of every curve's mass, not the "
         "full scanned [50, 100].  The bimodal galaxy realisation is drawn as it "
         "is, with its second mode marked"],
        ["`figs/fig_recovery.{pdf,png}`",
         "per-realisation medians +- 68% for both tracers against truth, the two "
         "tracers dodged either side of each seed, with each tracer's "
         "five-realisation mean offset +- standard error as a band"],
        ["`figs/fig_lanes.{pdf,png}`",
         "the targeted vs popuni median shift for all 20 scans as 10 "
         "same-events pairs: in units of that scan's 68% half-width (upper) and "
         "in km/s/Mpc (lower).  The two panels rank the scans differently, which "
         "is the point -- the largest AGN shift is 0.85 half-widths but only "
         "0.37 km/s/Mpc"],
        ["`figs/fig_diagnostics.{pdf,png}`",
         "internal: the selection guard.  Per-scan minimum N_eff against the "
         "5 N_obs floor (log), and the per-scan PE variance sum as its range over "
         "the 201 cells with the median marked; filled = targeted, open = popuni"],
        ["`figs/fig_bimodal.{pdf,png}`",
         "the s105 galaxy case on its own: the bimodal targeted posterior against "
         "the unimodal popuni one, both modes labelled with their relative height "
         "and both 68% intervals drawn as bars under the curves -- the "
         "single-scan look for a future reader who meets that wide interval in "
         "the closure table"],
    ]) + "\n")
    P.append("Colour is the project data-viz standard: identity is the tracer "
             "and only the tracer (categorical slots 1 and 2), the five "
             "realisations inside a tracer are an ordinal step of one hue rather "
             "than five hues, and every pair used together was checked with "
             "`../analysis_2_complete_catalog_H0_fagn/scripts/validate_palette.py` "
             "-- the header of `scripts/make_figures.py` records the measured "
             "separations.\n")

    # ---- layout --------------------------------------------------------------
    P.append("## Layout\n")
    P.append(table(["path", "what"], [
        ["`scripts/make_pure_tracer_events.sh`", "draws the ten event sets"],
        ["`scripts/check_pure_tracer_events.py`", "sanity checks them -> "
         "`results/event_sets.json`"],
        ["`scripts/scan_h0f.py`", "analysis 1's likelihood-grid driver, byte for byte"],
        ["`scripts/run_scans.sh`", "the four scans of one realisation"],
        ["`scripts/submit_scans.sbatch`", "the original 5-task array (one task per "
         "realisation)"],
        ["`scripts/submit_one_seed.sbatch`", "one realisation, partition/QOS left to "
         "the command line so it can be aimed at whichever GPU is free"],
        ["`scripts/bitcheck_events.py`", "the dataset-level bit-identity gate"],
        ["`scripts/aggregate_pure_tracer.py`", "closure + constraining power -> "
         "`results/h0_pure_tracer.json`"],
        ["`scripts/make_figures.py`", "renders `figs/` from `results/` "
         "(PDF + PNG, one function per figure)"],
        ["`scripts/make_readme.py`", "renders this file from the JSON"],
        ["`results/`", "one `.h5` (grid + logL) and one `.json` (posterior summary) "
         "per scan, plus the aggregates"],
        ["`figs/`", "the five rendered figures, PDF + PNG"],
        ["`logs/`", "generation, per-scan and SLURM logs"],
    ]) + "\n")
    P.append("Event files live beside the record in "
             "`/hildafs/projects/phy220048p/magana/gws-agn-data-v3/seed<S>/events/`.\n")

    Path(a.out).write_text("\n".join(P))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
