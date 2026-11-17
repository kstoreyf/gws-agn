#!/usr/bin/env python
"""Write values_independent_results.tex (a black \\newcommand for every macro whose real result exists, nothing
for the others, so the red \\providecommand of values_independent.tex stays) and notes/STATUS.md (every macro
and plotted element: real or placeholder, and the runs it waits for).

    PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
    $PY scripts/build_values.py

main.tex inputs the results file BEFORE values_independent.tex:
    \\InputIfFileExists{values_independent_results.tex}{}{}
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import results as R  # noqa: E402

OUT_TEX = R.PAPER / "values_independent_results.tex"
OUT_STATUS = R.PAPER / "notes" / "STATUS.md"


def run_index():
    """run name -> (stage, hold reason or '', data exists) from both run lists."""
    import yaml
    idx = {}
    for f in ("RUNLIST.yaml", "RUNLIST_experiment.yaml"):
        p = R.runs_root() / f
        if not p.exists():
            continue
        d = yaml.safe_load(p.read_text())
        for st in d["stages"]:
            for r in st["runs"]:
                idx[r["name"]] = (str(st["stage"]), r.get("hold") or "", bool(r.get("data_exists", True)))
    return idx


def waits_for(entry, idx):
    """One line: the run pattern(s), how many summaries are missing, the run list's stage and any hold."""
    if entry.get("combine") == "derived":
        return "derived: " + entry["expr"]
    pats = entry.get("summary") or ""
    pats = pats if isinstance(pats, list) else [pats]
    if not any(pats):
        return "no summary named in results_map.json"
    reals = R.realizations(entry)
    if entry["combine"] in ("single", "log10_bayes", "membership", "joint"):
        reals = reals[:1]
    parts = []
    for pat in pats:
        names = [pat.format(R=f"{r:03d}").rsplit("/summary.json", 1)[0] for r in reals]
        miss = [r for r, n in zip(reals, names) if not (R.runs_root() / n / "summary.json").exists()]
        info = [idx.get(n) for n in names]
        stages = sorted({i[0] for i in info if i})
        holds = sorted({i[1] for i in info if i and i[1]})
        nodata = [r for r, i in zip(reals, info) if i and not i[2]]
        notin = [r for r, i in zip(reals, info) if i is None]
        s = f"`{pat.rsplit('/realization_', 1)[0]}` ({len(reals) - len(miss)}/{len(reals)} summaries"
        if miss:
            s += f"; missing {','.join(map(str, miss))}"
        s += ")"
        if stages:
            s += f", stage {'/'.join(stages)}"
        if notin:
            s += f"; realizations {','.join(map(str, notin))} not in the run lists"
        if nodata:
            s += f"; no events file yet for {','.join(map(str, nodata))}"
        if holds:
            s += "; HELD: " + "; ".join(holds)
        parts.append(s)
    return " / ".join(parts)


def main():
    m = R.load_map()
    exp = R.expected()
    idx = run_index()
    stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    lines = ["%% values_independent_results.tex -- WRITTEN by scripts/build_values.py; do not edit.",
             f"%% {stamp}. Real results of the short paper (black), from the run summaries named in",
             "%% scripts/results_map.json. A macro whose result does not exist yet is absent here, so the red",
             "%% \\providecommand of values_independent.tex (input after this file) supplies its expected value.", ""]
    rows = []
    n_real = 0
    for name, entry in m["macros"].items():
        r = R.macro(name)
        if name not in exp:
            print(f"  WARNING: {name} is in results_map.json but not in values_independent.tex")
        if r.real:
            n_real += 1
            lines.append(f"\\newcommand{{\\{name}}}{{{r.tex}}}  % {r.how}")
        rows.append((name, r, entry))
    unmapped = [k for k, (b, red, _) in exp.items() if red and k not in m["macros"]]
    OUT_TEX.write_text("\n".join(lines) + "\n")

    st = [f"# Status of the short paper's numbers and figures ({stamp})", "",
          "Written by `scripts/build_values.py` from `scripts/results_map.json`; do not edit. **real** = black "
          "`\\newcommand` in `values_independent_results.tex` (from the run summaries); **placeholder** = the red "
          "expected value of `values_independent.tex`. Run paths are under "
          "`analyses/analysis_16_production_runs/runs/`; a run's stage is that of `RUNLIST.yaml` (1-4) or "
          "`RUNLIST_experiment.yaml` (E1-E3).", "",
          f"**{n_real} of {len(rows)} result macros real.**", "",
          "## Macros", "", "| macro | status | printed | meaning | waits for |", "|---|---|---|---|---|"]
    for name, r, entry in rows:
        printed = r.tex if r.real else exp[name][2]
        st.append(f"| `\\{name}` | {'**real**' if r.real else 'placeholder'} | `{printed}` | {entry.get('meaning', '')} | "
                  f"{'' if r.real else waits_for(entry, idx)} |")
    if unmapped:
        st += ["", "Red macros of `values_independent.tex` that `results_map.json` does not map (they stay red): "
               + ", ".join(f"`\\{k}`" for k in unmapped)]
    st += ["", "## Figures", "",
           "Fig. 2 `figures/fig_anchor`, Fig. 3 `figures/fig1_budget`, Fig. 4 `figures/fig_populations` (drawn by "
           "`scripts/make_figures.py`). **real** = every run it needs exists (final palette); **partial k/n** = drawn "
           "from the k realizations that exist, still in red; **placeholder** = expected value, red.", "",
           "| figure element | status | value drawn | waits for |", "|---|---|---|---|"]

    def tag(real, k, n):
        return "**real**" if real else (f"partial {k}/{n}" if k else "placeholder")

    for w in ("one_realization", "combined"):
        J = R.anchor_joint(w)
        e = m["series"]["fig_anchor"][w]
        st.append(f"| Fig. 2 {e['label']} | {tag(J['real'], J['n_have'], J['n_want'])} | 90% region of (H0, f_AGN); {J['how']} | "
                  f"{'' if J['real'] else waits_for(e, idx)} |")
    for panel in ("a", "b"):
        for key, s in R.fig1_points(panel).items():
            entries = m["series"][f"fig1{panel}"][key]["points"]
            for (xs, e), p in zip(entries.items(), s["points"]):
                where = "line" if p["x"] is None else f"f = {p['x']:g}"
                src = f"= `\\{e['macro']}`: " + waits_for(m["macros"][e["macro"]], idx) if "macro" in e else waits_for(e, idx)
                st.append(f"| Fig. 3({panel}) {s['label']}, {where} | {tag(p['real'], p['n_have'], p['n_want'])} | "
                          f"{p['y']:.2f} [{p['lo']:.2f}, {p['hi']:.2f}] | {'' if p['real'] else src} |")
    for w in ("one_population", "each_tracer"):
        d = R.fig3_density(w)
        e = m["series"]["fig3"][w]
        s = d["summary"]
        st.append(f"| Fig. 4(a) {e['label']} | {tag(d['real'], d['n_have'], d['n_real'])} | median {s['median']:.2f}, "
                  f"90% [{s['ci90'][0]:.2f}, {s['ci90'][1]:.2f}] | {'' if d['real'] else waits_for(e, idx) + '; ' + d['how']} |")
    d = R.fig3_joint()
    e = m["series"]["fig3"]["joint"]
    st.append(f"| Fig. 4(b) {e['label']} | {'**real**' if d['real'] else 'placeholder'} | 90% region | "
              f"{'' if d['real'] else waits_for(e, idx) + '; ' + d['how']} |")
    for w in ("positions", "masses_spins"):
        d = R.fig2_membership(w)
        e = m["series"]["fig2"][w]
        st.append(f"| Fig. 4(c) {e['label']} | {'**real**' if d['real'] else 'placeholder'} | means {d['mean_agn']:.3f} (AGN), "
                  f"{d['mean_gal']:.3f} (galaxies) | {'' if d['real'] else waits_for(e, idx) + '; ' + d['how']} |")
    OUT_STATUS.write_text("\n".join(st) + "\n")
    print(f"wrote {OUT_TEX.name}: {n_real} of {len(rows)} macros real; {OUT_STATUS.relative_to(R.PAPER)}")
    print("main.tex needs, before \\input{values_independent.tex}:  \\InputIfFileExists{values_independent_results.tex}{}{}")


if __name__ == "__main__":
    main()
