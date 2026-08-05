#!/usr/bin/env python3
"""Aggregate the per-configuration guard probes into one table.

Reads ../results/guard_audit/*.json (written by diag_variance_guard.py) and
reports, per configuration, the measured variance budget of darksirens master's
total-variance guard:

    sigma^2_lnL = pe_variance_sum + N_obs^2 / Neff   must be <=
    max_likelihood_variance (default 1.0, the GWTC-4.0/5.0 threshold)

For K=2 mixtures the guard is evaluated once per mixture member, so a config can
carry several records; the binding one is the WORST (largest sigma^2_total), and
a config is admitted only if every member passes.
"""
import argparse
import json
from pathlib import Path


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", required=True)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = []
    for path in sorted(Path(args.audit_dir).glob("*.json")):
        d = json.loads(path.read_text())
        recs = d.get("guard_records") or []
        if not recs:
            rows.append({"tag": path.stem, "status": "no-guard-record",
                         "logL": d.get("logL")})
            continue
        worst = max(recs, key=lambda r: r["sigma2_total"])
        rows.append({
            "tag": path.stem,
            "universe_model": d["universe_model"],
            "sky_weighting": d["catalog_sky_weighting"],
            "n_catalogs": len(d["survey_paths"]),
            "nEvents": d["nEvents"],
            "nsamp": d["nsamp"],
            "Ndraw": d["Ndraw"],
            "n_members": len(recs),
            "Neff": worst["Neff"],
            "pe_variance_sum": worst["pe_variance_sum"],
            "selection_variance": worst["selection_variance_N2_over_Neff"],
            "sigma2_total": worst["sigma2_total"],
            "over_budget_factor": worst["sigma2_total"] / 1.0,
            "passes_default_guard": all(r["passes"] for r in recs),
            "passes_legacy_floor": all(r["passes_legacy_floor"] for r in recs),
            "min_max_likelihood_variance": max(r["sigma2_total"] for r in recs),
            "logL_at_default_budget": d.get("logL"),
        })

    passing = [r for r in rows if r.get("passes_default_guard")]
    out = {
        "default_max_likelihood_variance": 1.0,
        "n_configs": len(rows),
        "n_pass_default_guard": len(passing),
        "passing_tags": [r["tag"] for r in passing],
        "rows": rows,
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2))

    hdr = ("| config | model | K | N_obs | Neff | Σσ²_PE | N²/Neff | σ²_total | "
           "≤1.0? | legacy 5N? |")
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    lines = [
        "# Guard audit — darksirens master @ 2b86a2d",
        "",
        "Total-variance guard (`darksirens/likelihood/selection.py`): a cell is",
        "admitted only if `Neff > max(5·N_obs, N_obs²/(V − Σσ²_PE))` with",
        "`V = max_likelihood_variance` (default **1.0**, the GWTC-4.0/5.0 bound on",
        "σ²_lnL). Equivalently `σ²_total = Σσ²_PE + N_obs²/Neff ≤ V`. This criterion",
        "did not exist in the #212-era code the previous run used, which faced only",
        "the legacy `Neff > 5·N_obs` floor (last column).",
        "",
        hdr, sep,
    ]
    for r in sorted(rows, key=lambda r: r["tag"]):
        if r.get("status") == "no-guard-record":
            lines.append(f"| {r['tag']} | — | — | — | — | — | — | — | ERROR | — |")
            continue
        lines.append(
            "| {tag} | {um} | {k} | {n} | {neff:.3g} | {pv:.3g} | {sv:.3g} | "
            "{st:.3g} | {ok} | {lg} |".format(
                tag=r["tag"],
                um=("dscf" if r["universe_model"] == "dark_sirens_complete" else "dsf"),
                k=r["n_catalogs"], n=r["nEvents"], neff=r["Neff"],
                pv=r["pe_variance_sum"], sv=r["selection_variance"],
                st=r["sigma2_total"],
                ok=("**PASS**" if r["passes_default_guard"] else "fail"),
                lg=("pass" if r["passes_legacy_floor"] else "fail"),
            )
        )
    lines += ["", f"**{len(passing)} of {len(rows)} configurations pass the default "
                  f"guard.**", ""]
    if passing:
        lines += ["Passing: " + ", ".join(f"`{r['tag']}`" for r in passing), ""]
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.out_json} and {args.out_md}")
    print(f"{len(passing)}/{len(rows)} configs pass the default guard")


if __name__ == "__main__":
    main()
