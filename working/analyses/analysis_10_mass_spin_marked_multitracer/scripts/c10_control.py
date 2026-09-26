#!/usr/bin/env python
"""Analysis 10, gate C10-3 -- the single-tracer spectral-siren control.

THE CONSTRUCTION (owner decision 2026-09-25)
--------------------------------------------
Two seed-100 mocks in which every host is a GAL galaxy and each event's
population branch is drawn independently of its host with probability 0.30
(``generate_dataset.py --f_agn 0 --branch_frac_independent 0.30``):

    B5  events_single_b0p30_dmu0p10_dmuG5.h5   marked branch mu_G = 40, dmu_chi +0.10
    B0  events_single_b0p30_dmu0p10_dmuG0.h5   marked branch mu_G = 35, dmu_chi +0.10

On these mocks the [GAL, GAL] two-mark model is CORRECTLY specified: both
branches share the GAL spatial prior, so routing cannot carry any redshift
information, and the second branch's peak location is the only
branch-dependent H0 information (a spectral siren).  The mixture weight f is
then the branch fraction.

THE ARMS (each a 2-D (H0, f) grid, H0 on [58, 78] at 0.5 + 67.74, f on F_GRID)
------------------------------------------------------------------------------
    B5M  B5, [GAL, GAL], marks pinned at the planted (+0.10, +5)   correct model
    B0M  B0, [GAL, GAL], marks pinned at the planted (+0.10,  0)   correct model
    B5U  B5, [GAL, GAL], marks 0                                    mass misspecified

W(B5M) / W(B0M) is the H0 width the second peak buys through the spectral-siren
channel alone (the twins differ in their detected sets because the heavier
branch is louder; that caveat is reported).  B5U - B5M is the spectral-siren
H0 offset from ignoring the heavier peak, to set against the production
C10-S - P1 = +3.02.

USAGE
-----
    python scripts/c10_control.py --stage run --arm B5M     # GPU, one arm per job
    python scripts/c10_control.py --stage assemble          # CPU
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c10_scan as CS                              # noqa: E402

A10, A8, A9, C = CS.A10, CS.A8, CS.A9, CS.C
DIAG, RESULTS = CS.DIAG, CS.RESULTS
EVENTS = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data/seed100/events")

MOCKS = {"B5": EVENTS / "events_single_b0p30_dmu0p10_dmuG5.h5",
         "B0": EVENTS / "events_single_b0p30_dmu0p10_dmuG0.h5"}
ARMS = {
    "B5M": {"mock": "B5", "dmu_chi": 0.10, "dmu_G": 5.0, "model": "correct (both marks)"},
    "B0M": {"mock": "B0", "dmu_chi": 0.10, "dmu_G": 0.0, "model": "correct (spin mark only)"},
    "B5U": {"mock": "B5", "dmu_chi": 0.0, "dmu_G": 0.0, "model": "unmarked (mass misspecified)"},
}
H0_WINDOW = (58.0, 78.0)          # the first pass; --h0_hi extends an arm with additive rows


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _ckpt(arm):
    return DIAG / f"_c10_ctl_{arm}.jsonl"


def stage_run(args):
    spec = ARMS[args.arm]
    gw = MOCKS[spec["mock"]]
    env = A9._gpu_setup(f"ctl_{args.arm}")
    md5 = _md5(gw)
    h0_axis = CS.j_h0_axis(min(H0_WINDOW[0], args.h0_lo or 1e9),
                           max(H0_WINDOW[1], args.h0_hi or 0.0))
    path = _ckpt(args.arm)
    hdr, rows, dropped = CS._read_jsonl(path)
    if hdr is not None and hdr.get("events_md5") != md5:
        raise SystemExit(f"[fatal] {path.name} was written for events md5 {hdr.get('events_md5')}")
    done = {CS._key1(r["H0"]) for r in rows}
    todo = [h0 for h0 in h0_axis if CS._key1(h0) not in done]
    print(f"[ctl-{args.arm}] {gw.name} md5 {md5}; marks ({spec['dmu_chi']}, {spec['dmu_G']}); "
          f"{len(done)}/{h0_axis.size} H0 rows done, {len(todo)} to do; {dropped} dropped")
    if not todo:
        return
    t0 = time.time()
    cell = A10.build_a10(f"C10_CTL_{args.arm}", [A8.SURVEY_GAL, A8.SURVEY_GAL],
                         verbose=True, gw_path=gw)
    # a10_closure.assert_cell, with the events-file check pointed at THIS mock
    if list(cell.labels) != list(A10.EXPECTED_LABELS_A10):
        raise SystemExit(f"[fatal] labels {list(cell.labels)}")
    if (int(cell.data["nEvents"]), int(cell.data["nsamp"])) != (C.N_EVENTS_EXPECTED,
                                                                C.NSAMP_EXPECTED):
        raise SystemExit("[fatal] nEvents/nsamp differ from the record")
    if str(cell.opts.gw_path) != str(gw):
        raise SystemExit(f"[fatal] events file {cell.opts.gw_path} != {gw}")
    print(f"  [OK ] {args.arm}: {len(cell.labels)} labels, events={gw.name}")
    print(f"build: {time.time() - t0:.1f}s")
    if hdr is None:
        CS._append_jsonl(path, {
            "record": "header", "arm": args.arm, "spec": spec,
            "events_file": str(gw), "events_md5": md5,
            "survey_paths": [str(A8.SURVEY_GAL), str(A8.SURVEY_GAL)],
            "selection_file": str(cell.opts.gwselection_path),
            "darksirens_sha": A9.DARKSIRENS_A8_SHA,
            "H0_grid": h0_axis.tolist(), "f_grid": CS.F_GRID.tolist(),
            "host": os.uname().nodename, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "environment": env, "written_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    t0 = time.time()
    for n, h0 in enumerate(todo, start=1):
        cells = [CS._eval(cell, h0, f, spec["dmu_chi"], spec["dmu_G"]) for f in CS.F_GRID]
        CS._append_jsonl(path, {"record": "row", "H0": float(h0), "key": CS._key1(h0),
                                "cells": cells,
                                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        lls = np.array([c["logL"] for c in cells], dtype=float)
        fin = np.isfinite(lls)
        el = time.time() - t0
        print(f"[ctl-{args.arm}] row {n}/{len(todo)} H0={h0:.2f} "
              f"maxlogL={(lls[fin].max() if fin.any() else float('nan')):.4f} "
              f"rejected={int((~fin).sum())}/{lls.size} eta={el / n * (len(todo) - n) / 60:.1f}min")
        sys.stdout.flush()
    print(f"[ctl-{args.arm}] done -> {path}")


def stage_assemble(args):
    import h5py
    mci = A8.import_scan_h0f().marginal_ci
    f = CS.F_GRID
    out = {"construction": __doc__.split("THE ARMS")[0].strip(), "arms": {},
           "H0_window": list(H0_WINDOW)}
    with h5py.File(RESULTS / "c10_control.h5", "w") as h5:
        h5.create_dataset("f_grid", data=f)
        for arm, spec in ARMS.items():
            hdr, rows, _ = CS._read_jsonl(_ckpt(arm))
            got = {CS._key1(r["H0"]): r for r in rows}
            # each arm's own window: [lowest, highest row present] on the 0.5 lattice
            h0_axis = CS.j_h0_axis(min(float(r["H0"]) for r in rows),
                                   max(float(r["H0"]) for r in rows))
            miss = [float(h) for h in h0_axis if CS._key1(h) not in got]
            if miss:
                print(f"[assemble] {arm}: {len(miss)} H0 rows missing; skipped")
                out["arms"][arm] = {"complete": False, "missing_H0": miss}
                continue
            ll = np.array([[c["logL"] for c in got[CS._key1(h)]["cells"]] for h in h0_axis])
            fin = np.isfinite(ll)
            L = np.where(fin, ll, -np.inf)
            mx = L[fin].max()
            P = np.exp(L - mx)
            lpH = np.log(np.trapz(P, f, axis=1))
            lpF = np.log(np.trapz(P, h0_axis, axis=0))
            H, F = mci(h0_axis, lpH), mci(f, lpF)
            iH, iF = np.unravel_index(np.argmax(L), L.shape)
            pH = np.exp(lpH - lpH.max())
            pF = np.exp(lpF - lpF.max())
            out["arms"][arm] = {
                "complete": True, "spec": spec, "events_md5": hdr["events_md5"],
                "n_cells": int(ll.size), "n_rejected": int((~fin).sum()),
                "H0": H, "f": F,
                "H0_width68": H["ci68"][1] - H["ci68"][0],
                "H0_width90": H["ci90"][1] - H["ci90"][0],
                "map": {"H0": float(h0_axis[iH]), "f": float(f[iF])},
                "edges_over_peak": {"H0": [float(pH[0]), float(pH[-1])],
                                    "f": [float(pF[0]), float(pF[-1])]},
                "H0_contained_1e-6": bool(max(pH[0], pH[-1]) <= 1e-6),
            }
            out["arms"][arm]["H0_window"] = [float(h0_axis[0]), float(h0_axis[-1])]
            g = h5.create_group(arm)
            g.create_dataset("H0_grid", data=h0_axis)
            g.create_dataset("log_likelihood", data=ll)
            g.create_dataset("rejected", data=~fin)
            g.attrs["spec"] = json.dumps(spec)
            g.attrs["events_md5"] = hdr["events_md5"]
    a = out["arms"]
    if all(a.get(k, {}).get("complete") for k in ("B5M", "B0M")):
        out["spectral_siren_width_ratio_B5M_over_B0M"] = {
            "68": a["B5M"]["H0_width68"] / a["B0M"]["H0_width68"],
            "90": a["B5M"]["H0_width90"] / a["B0M"]["H0_width90"]}
    if all(a.get(k, {}).get("complete") for k in ("B5M", "B5U")):
        out["unmodelled_peak_offset_B5U_minus_B5M"] = (
            a["B5U"]["H0"]["median"] - a["B5M"]["H0"]["median"])
        out["production_reference_C10S_minus_P1"] = 3.02
    CS._write(RESULTS / "c10_control.json", out)
    for arm, b in a.items():
        if b.get("complete"):
            print(f"  {arm}: H0 {b['H0']['median']:.3f} 90% {np.round(b['H0']['ci90'], 3).tolist()} "
                  f"w68 {b['H0_width68']:.3f} w90 {b['H0_width90']:.3f}; f {b['f']['median']:.3f}; "
                  f"rejected {b['n_rejected']}; H0 edges {b['edges_over_peak']['H0']}")
    for k in ("spectral_siren_width_ratio_B5M_over_B0M", "unmodelled_peak_offset_B5U_minus_B5M"):
        if k in out:
            print(f"  {k}: {out[k]}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("run", "assemble"))
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--h0_lo", type=float, default=None,
                    help="run: extend this arm's H0 axis down to this edge (additive rows)")
    ap.add_argument("--h0_hi", type=float, default=None,
                    help="run: extend this arm's H0 axis to this upper edge (additive rows)")
    args = ap.parse_args(argv)
    if args.stage == "run":
        if args.arm is None:
            raise SystemExit("[fatal] --stage run needs --arm")
        return stage_run(args)
    return stage_assemble(args)


if __name__ == "__main__":
    main()
