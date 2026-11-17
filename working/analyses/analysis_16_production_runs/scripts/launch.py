#!/usr/bin/env python
"""Analysis 16 launcher: put a stage (or one run) on the GPU lane, or show where things stand.

    python scripts/launch.py status [--stage S] [-v]       runs, tasks, states, estimates, lane jobs
    python scripts/launch.py stage S [S ...] [--dry-run]    queue the stages' unfinished tasks whose data
                                                            exist and start a lane job where none is queued
    python scripts/launch.py run NAME [--part P] [--dry-run] [--hours H]
                                                            one-off job for one run (chained like the lane)

Stages: 1-4 (runs/RUNLIST.yaml, production) and E1, E2, E3 (runs/RUNLIST_experiment.yaml, the short
paper's experiment); both lists are written by make_runlist.py.

ONE LANE (default since 17 Nov: this project may run one job at a time on RITA-GPU). The lane is
runs/_lanes/lane1.txt, run by scripts/submit_prod.sbatch as job agn-prod-l1. Every job this launcher
submits, and every successor a lane job submits, depends (afterany) on all of the user's pending or
running RITA-GPU jobs whose names start with "agn-" (the Analysis 14/16 test jobs included, never other
projects' jobs, and not a job on hold: its owner releases and chains it), so exactly one agn job
runs at a time. Runs marked "hold" in a run list (host-dependent sets on the present injection lane) are
not queued; make_runlist.py lifts the hold when data/experiment_sets.json names their lane. The lane file is kept sorted by (stage order
[1, then E1/E2, E3, then 2, 3, 4], the run's priority [experiment: f = 0.3 sets first, then f = 0.1 and
0.5, then the rest], estimated GPU hours), so a stage added later still enters ahead of costlier ones.
A lane job re-reads its file before every task and submits its own successor while work remains.
Tasks whose events or injection file does not exist (yet) are left out: rerun `stage S` once they do.

TWO LANES (the earlier mode, for when both A100s are ours again): pass --lanes 2 (or export
A16_LANES=2 before launching): tasks are balanced over lane1.txt/lane2.txt by estimated GPU hours, and
each lane chains only behind its own jobs (agn-prod-l1, agn-prod-l2), so two run at once.
Any python with PyYAML (the core venv or the jax env).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prod_common as P  # noqa: E402
import lane_worker as W  # noqa: E402

STAGE_ORDER = {1: 1.0, "E1": 1.5, "E2": 1.5, "E3": 1.7, 2: 2.0, 3: 3.0, 4: 4.0}


def norm_stage(s):
    s = str(s).strip()
    return int(s) if s.isdigit() else s.upper()


def all_runs():
    for doc in W.runlist_docs():
        for st in doc["stages"]:
            for r in st["runs"]:
                yield norm_stage(st["stage"]), r


def order_key(stage, r, part):
    return (STAGE_ORDER.get(stage, 9.0), float(r.get("priority", 0.0)), r["est_gpu_hours_per_part"], r["name"], part)


def lane_load(k, est):
    return sum(est.get(t, 0.0) for t in W.read_lane(k) if W.task_state(*t) == "todo")


def write_lane(k, tasks, order):
    W.LANES.mkdir(parents=True, exist_ok=True)
    tasks = sorted(set(tasks), key=lambda t: order.get(t, (99, 0, 1e9, t[0], t[1])))
    (W.LANES / f"lane{k}.txt").write_text(
        "# <run name>|<part>; sorted by (stage order, priority, estimated GPU hours); written by launch.py\n"
        + "\n".join(f"{n}|{p}" for n, p in tasks) + "\n")


def cmd_status(a):
    jobs = W.agn_jobs()
    tot = {}
    want = norm_stage(a.stage) if a.stage else None
    for stage, r in all_runs():
        if want is not None and stage != want:
            continue
        states = [W.task_state(r["name"], p, r) for p in r["parts"]]
        key = "/".join(states)
        tot.setdefault(stage, {}).setdefault(key, [0, 0.0])
        tot[stage][key][0] += 1
        tot[stage][key][1] += r["est_gpu_hours"]
        if a.verbose:
            print(f"  [{stage}] {r['name']:110s} {','.join(states):24s} est {r['est_gpu_hours']:.2f} h"
                  f"{'' if W.data_ready(r) else '  (no data yet)'}")
    for stage, d in sorted(tot.items(), key=lambda kv: STAGE_ORDER.get(kv[0], 9)):
        print(f"stage {stage}: " + "; ".join(f"{k}: {n} runs ({h:.1f} est GPU-h)" for k, (n, h) in sorted(d.items())))
    for k in (1, 2):
        t = W.read_lane(k)
        if not t and k == 2:
            continue
        st = [W.task_state(*x) for x in t]
        print(f"lane {k}: {len(t)} tasks, {st.count('todo')} to do, {st.count('finished')} finished, "
              f"{st.count('failed')} failed")
    for j in jobs:
        print(f"  agn job {j['id']} {j['name']} {j['state']} {j['used']}/{j['limit']} {j['dep']} {j['reason']}")


def submit(lane=None, task=None, hours=None, dry=False, n_lanes=1):
    cmd = ["sbatch", "--parsable"]
    if lane:
        cmd += [f"--job-name=agn-prod-l{lane}", f"--export=ALL,LANE={lane},A16_LANES={n_lanes}"]
    else:
        cmd += ["--job-name=agn-prod-run", f"--export=ALL,TASK={task},A16_LANES={n_lanes}"]
    dep = W.chain_dependency(lane if n_lanes > 1 else None)
    if dep:
        cmd.append(dep)
    if hours:
        h = int(hours)
        m = int(round((hours - h) * 60))
        cmd += [f"--time={h:02d}:{m:02d}:00"]
        i = next(i for i, c in enumerate(cmd) if c.startswith("--export"))
        cmd[i] += f",A16_WALL_MINUTES={60 * h + m}"
    cmd.append(str(HERE / "submit_prod.sbatch"))
    print(("[dry-run] " if dry else "") + " ".join(cmd))
    if dry:
        return None
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout.strip(), r.stderr.strip())
    return r.stdout.strip()


def cmd_stage(a):
    n_lanes = a.lanes
    stages = [norm_stage(s) for s in a.stage]
    est, order, new, missing, held = {}, {}, [], [], []
    for stage, r in all_runs():
        for p in r["parts"]:
            t = (r["name"], p)
            est[t] = r["est_gpu_hours_per_part"]
            order[t] = order_key(stage, r, p)
        if stage in stages:
            if r.get("hold"):
                held.append(r["name"])
                continue
            if not W.data_ready(r):
                missing.append(r["name"])
                continue
            new += [(r["name"], p) for p in r["parts"] if W.task_state(r["name"], p, r) == "todo"]
    lanes = {k: W.read_lane(k) for k in range(1, n_lanes + 1)}
    if n_lanes == 1 and W.read_lane(2):
        # back to one lane: lane 2's unfinished tasks move to lane 1
        lanes[1] += [t for t in W.read_lane(2) if W.task_state(*t) == "todo"]
        if not a.dry_run:
            (W.LANES / "lane2.txt").rename(W.LANES / "lane2.txt.retired")
    present = {t for v in lanes.values() for t in v}
    new = sorted([t for t in set(new) if t not in present], key=lambda t: order[t])
    load = {k: lane_load(k, est) for k in lanes}
    for t in new:
        k = min(load, key=load.get)
        lanes[k].append(t)
        load[k] += est[t]
    print(f"stages {stages}: {len(new)} new tasks ({sum(est[t] for t in new):.1f} est GPU-h); "
          f"{len(missing)} runs wait for data; {len(held)} runs held (see 'hold' in the run list); {n_lanes} lane(s)")
    for k in lanes:
        print(f"  lane {k}: {sum(1 for t in lanes[k] if W.task_state(*t) == 'todo')} tasks to do, "
              f"{load[k]:.1f} est GPU-h")
    if a.dry_run:
        for t in sorted(set(lanes[1]), key=lambda t: order.get(t, (99,)))[:12]:
            print(f"  e.g. {t[0]}|{t[1]}  est {est.get(t, 0):.2f} h")
        print(f"  dependency for a new lane job: {W.chain_dependency(None) or '(none)'}")
        return
    for k in lanes:
        write_lane(k, lanes[k], order)
    jobs = W.agn_jobs()
    for k in lanes:
        if not any(W.task_state(*t) == "todo" for t in lanes[k]):
            continue
        mine = [j for j in jobs if j["name"] == f"agn-prod-l{k}"]
        if mine:
            print(f"  lane {k}: job(s) {[j['id'] for j in mine]} already queued; it will pick the new tasks up")
            continue
        submit(lane=k, n_lanes=n_lanes)


def cmd_run(a):
    rr = [r for _, r in all_runs() if r["name"] == a.name]
    if not rr:
        sys.exit(f"[fatal] {a.name} is in neither run list")
    r = rr[0]
    parts = [a.part] if a.part else r["parts"]
    for p in parts:
        st = W.task_state(r["name"], p, r)
        if st != "todo":
            print(f"{r['name']}|{p}: {st}; not submitted")
            continue
        submit(task=f"{r['name']}|{p}", hours=a.hours, dry=a.dry_run, n_lanes=a.lanes)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lanes", type=int, choices=(1, 2), default=int(os.environ.get("A16_LANES", "1")),
                    help="GPU lanes (default 1, or $A16_LANES)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status")
    s.add_argument("--stage")
    s.add_argument("-v", "--verbose", action="store_true")
    s = sub.add_parser("stage")
    s.add_argument("stage", nargs="+")
    s.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("run")
    s.add_argument("name")
    s.add_argument("--part")
    s.add_argument("--hours", type=float, default=None)
    s.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    {"status": cmd_status, "stage": cmd_stage, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
