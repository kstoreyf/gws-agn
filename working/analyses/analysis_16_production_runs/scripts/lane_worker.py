#!/usr/bin/env python
"""Analysis 16 lane worker: run the tasks of one lane (one GPU) until its job's wall time is used.

    (inside submit_prod.sbatch)  python lane_worker.py --lane 1
    (one-off)                    python lane_worker.py --task <run name>|<part>

A lane is runs/_lanes/lane<K>.txt, one task per line ("<run name>|<part>", part = grid, nested_1,
nested_2), in the order the launcher wrote them (cheap stages first). On start the worker submits
its own successor whenever unfinished tasks remain, so a lane keeps one job running and one waiting,
and a run cut by the wall clock or by a crash is picked up again from its checkpoint. One lane (the
default, A16_LANES=1): the successor depends (afterany) on every pending or running RITA-GPU job of
the user named agn-* (this one included), so one agn job runs at a time; two lanes: on this lane's
own jobs only. It then takes the first unfinished task, runs the driver with the job's deadline, writes
the run's summary (prod_summary.py -> runs/<run>/summary.json) when all its parts are finished, and
moves on until the deadline is near. Runs are looked up in RUNLIST.yaml and RUNLIST_experiment.yaml
(re-read when a lane line names a run the worker does not know yet).

Per task (runs/<run>/<part>/attempts.json): a task whose driver exited with an error (not 0, not 75)
three times is marked failed and skipped; a missing events file (exit 4) is skipped without counting.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prod_common as P  # noqa: E402

LANES = P.RUNS / "_lanes"
MAX_ERRORS = 3
MIN_SECONDS_TO_START = 1200


def log(msg):
    P.log("lane", msg)


RUNLISTS = ("RUNLIST.yaml", "RUNLIST_experiment.yaml")


def runlist_docs():
    import yaml
    return [yaml.safe_load((P.RUNS / f).read_text()) for f in RUNLISTS if (P.RUNS / f).exists()]


def load_runlist():
    by = {}
    for doc in runlist_docs():
        for st in doc["stages"]:
            for r in st["runs"]:
                by[r["name"]] = r
    return by


def data_ready(entry):
    """The run's events file and injection set exist (a planned set may not exist yet)."""
    if not entry.get("args"):
        return False
    ev = entry.get("events_file")
    inj = entry.get("injections_file")
    return bool(ev and Path(ev).exists() and (inj is None or P.injections_ready(inj)))


def agn_jobs():
    """The user's pending/running RITA-GPU jobs named agn-* (never another project's jobs)."""
    r = subprocess.run(["squeue", "-h", "--me", "-p", "RITA-GPU", "-o", "%i|%j|%T|%M|%l|%E|%r"],
                       capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines():
        f = line.strip().split("|")
        if len(f) < 7 or not f[1].startswith("agn-"):
            continue
        if f[2] in ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED", "RESIZING"):
            out.append({"id": f[0], "name": f[1], "state": f[2], "used": f[3], "limit": f[4], "dep": f[5],
                        "reason": f[6], "held": f[6] in ("JobHeldUser", "JobHeldAdmin")})
    return out


def chain_dependency(lane=None):
    """--dependency for a new agn job: one lane (lane None) -> afterany on every agn-* job of the user
    on RITA-GPU; two lanes -> afterany on this lane's own jobs."""
    jobs = [j for j in agn_jobs() if not j["held"]]   # a job on hold is released (and chained) by its owner
    if lane is not None:
        jobs = [j for j in jobs if j["name"] == f"agn-prod-l{lane}"]
    ids = sorted({j["id"] for j in jobs}, key=lambda x: int(x.split("_")[0]) if x.split("_")[0].isdigit() else 0)
    return f"--dependency=afterany:{':'.join(ids)}" if ids else None


def part_dir(name, part):
    return P.RUNS / name / part


def task_state(name, part, entry=None):
    d = part_dir(name, part)
    rj = d / "result.json"
    if rj.exists():
        try:
            if json.loads(rj.read_text()).get("finished"):
                return "finished"
        except json.JSONDecodeError:
            pass
    att = d / "attempts.json"
    if att.exists():
        rows = json.loads(att.read_text())
        if sum(1 for r in rows if r.get("outcome") == "error") >= MAX_ERRORS:
            return "failed"
        if rows and rows[-1].get("outcome") == "missing_data":
            ev = rows[-1].get("events_file")
            inj = rows[-1].get("injections_file")
            if (ev and not Path(ev).exists()) or (inj and not P.injections_ready(inj)):
                return "missing_data"
    if entry is not None and entry.get("hold"):
        return "held"
    if entry is not None and not data_ready(entry):
        return "missing_data"
    return "todo"


def read_lane(k):
    p = LANES / f"lane{k}.txt"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, part = line.split("|")
            out.append((name, part))
    return out


def driver_args(entry, part):
    args = list(entry["args"])
    if part.startswith("nested_"):
        args += ["--nested-index", part.split("_")[1]]
    return args


def submit_successor(lane):
    sb = HERE / "submit_prod.sbatch"
    jid = os.environ.get("SLURM_JOB_ID")
    n_lanes = int(os.environ.get("A16_LANES", "1"))
    cmd = ["sbatch", "--parsable", f"--job-name=agn-prod-l{lane}", f"--export=ALL,LANE={lane},A16_LANES={n_lanes}"]
    dep = chain_dependency(lane if n_lanes > 1 else None)
    if dep is None and jid:
        dep = f"--dependency=afterany:{jid}"
    elif dep and jid and jid not in dep:
        dep += f":{jid}"
    if dep:
        cmd.append(dep)
    cmd.append(str(sb))
    r = subprocess.run(cmd, capture_output=True, text=True)
    log(f"successor: {' '.join(cmd)} -> {r.stdout.strip()} {r.stderr.strip()}")
    return r.stdout.strip()


def lock_holder(name, part):
    """The Slurm job holding this task's lock, if that job is still pending or running."""
    lk = part_dir(name, part) / "LOCK"
    if not lk.exists():
        return None
    try:
        jid = json.loads(lk.read_text()).get("slurm_job_id")
    except json.JSONDecodeError:
        return None
    if not jid or jid == os.environ.get("SLURM_JOB_ID"):
        return None
    r = subprocess.run(["squeue", "-h", "-j", str(jid), "-o", "%T"], capture_output=True, text=True)
    return jid if r.stdout.strip() in ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING") else None


def run_task(entry, name, part, deadline):
    d = part_dir(name, part)
    d.mkdir(parents=True, exist_ok=True)
    (d / "LOCK").write_text(json.dumps({"slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                                        "host": os.uname().nodename, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ")}))
    att = d / "attempts.json"
    rows = json.loads(att.read_text()) if att.exists() else []
    row = {"slurm_job_id": os.environ.get("SLURM_JOB_ID"), "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "events_file": entry.get("events_file"), "injections_file": entry.get("injections_file"),
           "outcome": "running"}
    rows.append(row)
    att.write_text(json.dumps(rows, indent=1))
    args = driver_args(entry, part)
    env = dict(os.environ, A16_DEADLINE=str(deadline))
    log(f"task {name}|{part}: {' '.join(args)}")
    t0 = time.time()
    rc = subprocess.run([P.CORE_PY, str(HERE / "prod_driver.py"), *args], env=env).returncode
    row["exit_code"] = rc
    row["wall_seconds"] = time.time() - t0
    row["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    st = task_state(name, part)
    row["outcome"] = ("finished" if st == "finished" else "deadline" if rc == 75 else
                      "missing_data" if rc == 4 else "error" if rc != 0 else "stopped")
    att.write_text(json.dumps(rows, indent=1))
    (d / "LOCK").unlink(missing_ok=True)
    log(f"task {name}|{part}: exit {rc}, {row['outcome']}, {row['wall_seconds'] / 60:.1f} min")
    if row["outcome"] == "finished" and all(task_state(name, p) == "finished" for p in entry["parts"]):
        rs = subprocess.run([P.CORE_PY, str(HERE / "prod_summary.py"), name], capture_output=True, text=True,
                            env=dict(os.environ, JAX_PLATFORMS="cpu"))
        log(f"summary {name}: {(rs.stdout.strip() or rs.stderr.strip()[-500:])}")
    return row["outcome"]


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--lane", type=int)
    g.add_argument("--task", help="'<run name>|<part>' (one-off job)")
    ap.add_argument("--no-successor", action="store_true")
    a = ap.parse_args()
    start = float(os.environ.get("A16_JOB_START", time.time()))
    deadline = float(os.environ.get("A16_DEADLINE", start + 47 * 3600))
    runlist = load_runlist()
    tasks = read_lane(a.lane) if a.lane else [tuple(a.task.split("|"))]
    todo = [(n, p) for n, p in tasks if task_state(n, p) == "todo"]
    log(f"job {os.environ.get('SLURM_JOB_ID')} lane {a.lane} task {a.task}: {len(todo)} of {len(tasks)} tasks "
        f"to do; deadline in {(deadline - time.time()) / 3600:.2f} h")
    if not todo:
        log("nothing to do; no successor")
        return 0
    if a.lane and not a.no_successor:
        submit_successor(a.lane)
    done_here = set()
    while True:
        if time.time() + MIN_SECONDS_TO_START > deadline:
            log("too little time left to start another task; the successor continues")
            break
        # re-read the lane each time: the launcher may have added (cheaper) tasks meanwhile
        tasks = read_lane(a.lane) if a.lane else tasks
        pick = None
        for name, part in tasks:
            if (name, part) in done_here or task_state(name, part, runlist.get(name)) != "todo":
                continue
            holder = lock_holder(name, part)
            if holder:
                log(f"[skip] {name}|{part} is held by job {holder}")
                continue
            pick = (name, part)
            break
        if pick is None:
            break
        name, part = pick
        done_here.add(pick)
        if name not in runlist:
            runlist = load_runlist()
        if name not in runlist:
            log(f"[skip] {name} is in neither run list")
            continue
        out = run_task(runlist[name], name, part, deadline)
        if out == "deadline":
            break
    left = [(n, p) for n, p in tasks if task_state(n, p) == "todo"]
    log(f"exiting: {len(left)} tasks left in this lane")
    return 0


if __name__ == "__main__":
    sys.exit(main())
