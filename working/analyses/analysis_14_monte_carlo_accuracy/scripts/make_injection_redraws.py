#!/usr/bin/env python
"""Independent redraws of the seed-100 targeted injection set (Analysis 14).

    python make_injection_redraws.py --sets 1 2 3 4 5 [--jobs 48] [--seed 100]

Each set j is the generator's own ``injections`` stage (working/data/generate_dataset.py,
README section 4), lane ``targeted``, 1.5e8 proposals (the record's draw count), the
record's three-branch proposal (0.65 population + 0.10 uniform + 0.25 AGN-targeted) and the
record's darksirens checkout (2b86a2d, as META.json records). The ONLY change is the
stream: the targeted lane's sub-seed is SEED*1000 + 900 + j instead of the record's
SEED*1000 + 4 (offsets 900+ are not used anywhere else in the generator). SEED is the
catalog realisation (--seed; 100 for configuration (b), 103 for (a) since 2026-11-17).

Output (never the signed-off files):
    <seedSEED target>/mc_accuracy/inj_set<j>/seedSEED/injections/injections_targeted.h5
    (+ _meta.json and that root's META.json). The stage reads the pixelated complete AGN
    survey from <root>/seed100/surveys/, which is a symlink to the record's file; the
    deleted-catalog question does not arise, the stage never reads the catalogs.
"""
import argparse
import os
import sys
from pathlib import Path

DATA = Path("/hildafs/projects/phy230014p/magana/gws-agn/working/data")
DARKSIRENS = "/hildafs/projects/phy230014p/magana/src/darksirens-2b86a2d"
NDRAW_T = 150_000_000
SEED_OFFSET = 900


def rec_dir(seed: int) -> Path:
    return Path(os.path.realpath(DATA / f"seed{seed}"))


def set_root(j: int, seed: int = 100) -> Path:
    return rec_dir(seed) / "mc_accuracy" / f"inj_set{j}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=100)
    args = ap.parse_args()
    SEED = args.seed
    REC = rec_dir(SEED)
    import json
    sha = json.loads((REC / "META.json").read_text()).get("darksirens_sha", "")
    if not sha.startswith("2b86a2d"):
        sys.exit(f"[inj] seed {SEED} was generated with darksirens {sha}, not 2b86a2d; set DARKSIRENS")
    sys.path.insert(0, str(DATA))
    import generate_dataset as g

    orig = g.sub_seeds
    for j in args.sets:
        root = set_root(j, SEED)
        sd = root / f"seed{SEED}"
        dst = sd / "injections" / "injections_targeted.h5"
        if dst.exists():
            print(f"[inj] set {j}: {dst} exists, skipping", flush=True)
            continue
        (sd / "surveys").mkdir(parents=True, exist_ok=True)
        link = sd / "surveys" / "survey_agn_complete_ns32.h5"
        if not link.exists():
            link.symlink_to(REC / "surveys" / "survey_agn_complete_ns32.h5")
        new_seed = SEED * 1000 + SEED_OFFSET + j

        def patched(seed, _orig=orig, _s=new_seed):
            d = dict(_orig(seed))
            d["injections_targeted"] = _s
            return d

        g.sub_seeds = patched
        argv = ["--seed", str(SEED), "--stage", "injections", "--outroot", str(root),
                "--lane", "targeted", "--ndraw_targeted", str(NDRAW_T),
                "--darksirens", DARKSIRENS]
        if args.jobs:
            argv += ["--jobs", str(args.jobs)]
        print(f"[inj] set {j}: sub-seed {new_seed} -> {dst}", flush=True)
        g.main(argv)
        g.sub_seeds = orig
        if not dst.exists():
            sys.exit(f"[inj] set {j}: {dst} was not written")
    print("[inj] done", flush=True)


if __name__ == "__main__":
    main()
