#!/usr/bin/env python
"""Analysis 16 launcher: put a stage (or one run) on the two GPU lanes, or show where things stand.

    python scripts/launch.py status [--stage N]            runs, tasks, states, estimates, lane jobs
    python scripts/launch.py stage N [--dry-run]           queue stage N's unfinished tasks on the lanes
                                                           and start a lane job where none is queued
    python scripts/launch.py run NAME [--part P] [--dry-run] [--hours H]
                                                           one-off job(s) for one run (outside the lanes)

Lanes: runs/_lanes/lane1.txt and lane2.txt, one GPU each (scripts/submit_prod.sbatch,
job names agn-prod-l1 / agn-prod-l2). Each new task goes to the lane with the smaller remaining
estimated GPU hours; each lane file is kept sorted by (stage, estimated hours per task), so
cheap stages run first and a stage added later still enters ahead of costlier ones. A lane job
re-reads its file before every task and submits its own successor while work remains, so the GPUs
stay busy and an interrupted task resumes from its checkpoint. Tasks whose events file does not
exist yet are left out (rerun `stage N` once the data exist).
Any python with PyYAML (the core venv or the jax env).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prod_common as P  # noqa: E402
import lane_worker as W  # noqa: E402

N_LANES = 2


def runlist():
    import yaml
    return yaml.safe_load((P.RUNS / "RUNLIST.yaml").read_text())


def all_runs(doc):
    for st in doc["stages"]:
        for r in st["runs"]:
            yield st["stage"], r


def lane_jobs():
    r = subprocess.run(["squeue", "-h", "--me", "-o", "%i %j %T %M %l"], capture_output=True, text=True)
    out = {}
    for line in r.stdout.splitlines():
        jid, name, state, used, lim = line.split()
        if name.startswith("agn-prod"):
            out.setdefault(name, []).append((jid, state, used, lim))
    return out


def lane_load(k, est):
    return sum(est.get(t, 0.0) for t in W.read_lane(k) if W.task_state(*t) == "todo")


def write_lane(k, tasks, order):
    W.LANES.mkdir(parents=True, exist_ok=True)
    tasks = sorted(set(tasks), key=lambda t: order.get(t, (99, 1e9, t)))
    (W.LANES / f"lane{k}.txt").write_text(
        "# <run name>|<part>; sorted by (stage, estimated GPU hours); written by launch.py\n"
        + "\n".join(f"{n}|{p}" for n, p in tasks) + "\n")


def cmd_status(a):
    doc = runlist()
    jobs = lane_jobs()
    tot = {}
    for stage, r in all_runs(doc):
        if a.stage and stage != a.stage:
            continue
        states = [W.task_state(r["name"], p) for p in r["parts"]]
        key = (stage, "/".join(sorted(set(states))))
        tot.setdefault(stage, {}).setdefault("/".join(states), [0, 0.0])
        tot[stage]["/".join(states)][0] += 1
        tot[stage]["/".join(states)][1] += r["est_gpu_hours"]
        if a.verbose:
            print(f"  [{stage}] {r['name']:110s} {','.join(states):24s} est {r['est_gpu_hours']:.2f} h"
                  f"{'' if r['data_exists'] else '  (no data yet)'}")
    for stage, d in sorted(tot.items()):
        print(f"stage {stage}: " + "; ".join(f"{k}: {n} runs ({h:.1f} est GPU-h)" for k, (n, h) in sorted(d.items())))
    for k in range(1, N_LANES + 1):
        t = W.read_lane(k)
        st = [W.task_state(*x) for x in t]
        print(f"lane {k}: {len(t)} tasks, {st.count('todo')} to do, {st.count('finished')} finished, "
              f"{st.count('failed')} failed; jobs {jobs.get(f'agn-prod-l{k}', [])}")
    if "agn-prod-run" in jobs:
        print(f"one-off jobs: {jobs['agn-prod-run']}")


def submit(lane=None, task=None, hours=None, dry=False):
    cmd = ["sbatch", "--parsable"]
    if lane:
        cmd += [f"--job-name=agn-prod-l{lane}", f"--export=ALL,LANE={lane}"]
    else:
        cmd += ["--job-name=agn-prod-run", f"--export=ALL,TASK={task}"]
    if hours:
        h = int(hours)
        m = int(round((hours - h) * 60))
        cmd += [f"--time={h:02d}:{m:02d}:00"]
        cmd[cmd.index(next(c for c in cmd if c.startswith("--export")))] += f",A16_WALL_MINUTES={60 * h + m}"
    cmd.append(str(HERE / "submit_prod.sbatch"))
    print(("[dry-run] " if dry else "") + " ".join(cmd))
    if dry:
        return None
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout.strip(), r.stderr.strip())
    return r.stdout.strip()


def cmd_stage(a):
    doc = runlist()
    est, order, new = {}, {}, []
    for stage, r in all_runs(doc):
        for p in r["parts"]:
            t = (r["name"], p)
            est[t] = r["est_gpu_hours_per_part"]
            order[t] = (stage, r["est_gpu_hours_per_part"], r["name"], p)
            if stage == a.stage and r["data_exists"] and W.task_state(*t) == "todo":
                new.append(t)
    missing = [r["name"] for stage, r in all_runs(doc) if stage == a.stage and not r["data_exists"]]
    lanes = {k: W.read_lane(k) for k in range(1, N_LANES + 1)}
    present = {t for v in lanes.values() for t in v}
    new = sorted([t for t in new if t not in present], key=lambda t: order[t])
    load = {k: lane_load(k, est) for k in lanes}
    for t in new:
        k = min(load, key=load.get)
        lanes[k].append(t)
        load[k] += est[t]
    print(f"stage {a.stage}: {len(new)} new tasks ({sum(est[t] for t in new):.1f} est GPU-h); "
          f"{len(missing)} runs wait for data")
    for k in lanes:
        print(f"  lane {k}: {sum(1 for t in lanes[k] if W.task_state(*t) == 'todo')} tasks to do, "
              f"{load[k]:.1f} est GPU-h")
    if a.dry_run:
        for t in new[:10]:
            print(f"  e.g. {t[0]}|{t[1]}  est {est[t]:.2f} h")
        return
    for k in lanes:
        write_lane(k, lanes[k], order)
    jobs = lane_jobs()
    for k in lanes:
        if not any(W.task_state(*t) == "todo" for t in lanes[k]):
            continue
        if jobs.get(f"agn-prod-l{k}"):
            print(f"  lane {k}: job(s) {jobs[f'agn-prod-l{k}']} already queued; it will pick the new tasks up")
            continue
        submit(lane=k)


def cmd_run(a):
    doc = runlist()
    rr = [r for _, r in all_runs(doc) if r["name"] == a.name]
    if not rr:
        sys.exit(f"[fatal] {a.name} is not in RUNLIST.yaml")
    r = rr[0]
    parts = [a.part] if a.part else r["parts"]
    for p in parts:
        st = W.task_state(r["name"], p)
        if st != "todo":
            print(f"{r['name']}|{p}: {st}; not submitted")
            continue
        submit(task=f"{r['name']}|{p}", hours=a.hours, dry=a.dry_run)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status")
    s.add_argument("--stage", type=int)
    s.add_argument("-v", "--verbose", action="store_true")
    s = sub.add_parser("stage")
    s.add_argument("stage", type=int)
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
