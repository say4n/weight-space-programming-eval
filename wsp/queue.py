"""A job queue so several models can be evaluated without babysitting.

Each job is one (model, experiment, protocol) triple. Status lives in
`results/queue.json`, so the queue survives interruptions and can be inspected with
`wsp queue status`. `wsp queue run` executes whatever is still pending, a few jobs
at a time, and records each one's exit code.

Work is done by shelling out to `wsp <exp> ...` rather than importing the modules:
one crashed experiment must not take the queue down, and each run gets a clean
process with its own memory.
"""

import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .experiments import EFFORT, SKIP

REPO = Path(__file__).resolve().parent.parent
QUEUE_PATH = REPO / "results" / "queue.json"
_LOCK = threading.Lock()

# exp8 is not reproducible on this hardware (no CUDA/vLLM); see README.
DEFAULT_EXPS = ["exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7"]


def slug(model):
    return model.replace("/", "_").replace(":", "-")


def load():
    try:
        rows = json.loads(QUEUE_PATH.read_text())
        return rows if isinstance(rows, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save(rows):
    """Write atomically: several workers update the queue at once."""
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        tmp = QUEUE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rows, indent=1))
        os.replace(tmp, QUEUE_PATH)


def _key(job):
    return (job["model"], job["exp"], job.get("protocol"))


def jobs_for(model, provider, exps):
    out = []
    for exp in exps:
        if exp in SKIP:
            continue
        protocols = ("weights_only", "attention") if exp == "exp7" else (None,)
        for proto in protocols:
            out.append(
                {
                    "model": model,
                    "provider": provider,
                    "exp": exp,
                    "protocol": proto,
                    "status": "pending",
                    "added": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "started": None,
                    "finished": None,
                    "returncode": None,
                    "note": None,
                }
            )
    return out


def add(model, provider, exps, rows=None):
    """Enqueue jobs, skipping any (model, experiment, protocol) already present."""
    rows = rows if rows is not None else load()
    seen = {_key(r) for r in rows}
    added = 0
    for job in jobs_for(model, provider, exps):
        if _key(job) in seen:
            continue
        rows.append(job)
        seen.add(_key(job))
        added += 1
    return rows, added


def command(job):
    """The `wsp` invocation for one job, at the effort the post used."""
    cmd = [
        "uv",
        "run",
        "wsp",
        job["exp"],
        "--model",
        job["model"],
        "--provider",
        job["provider"],
        "--thinking",
        EFFORT[job["exp"]],
        "--tag",
        slug(job["model"]),
    ]
    if job.get("protocol"):
        cmd += ["--protocol", job["protocol"]]
    if job["exp"] == "exp6":
        cmd += ["--generate"]
    return cmd


def _execute(job):
    cmd = command(job)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=8 * 3600)
        rc, tail = p.returncode, (p.stdout or "")[-1500:] + (p.stderr or "")[-500:]
    except subprocess.TimeoutExpired:
        rc, tail = -1, "timed out after 8h"
    return rc, tail, round(time.time() - t0, 1)


def run(pending, workers=2):
    """Execute the given jobs, updating and persisting status as they finish."""

    def one(job):
        job["status"] = "running"
        job["started"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        save(all_rows)
        rc, tail, secs = _execute(job)
        job["returncode"] = rc
        job["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        job["status"] = "done" if rc == 0 else "failed"
        job["note"] = tail.strip()[-600:] if rc != 0 else f"{secs}s"
        save(all_rows)
        label = f"{job['exp']}:{job['protocol']}" if job.get("protocol") else job["exp"]
        print(f"  [{job['status']}] {label} {job['model']} ({secs}s)", flush=True)
        return job

    global all_rows
    all_rows = load()
    by_key = {_key(r): r for r in all_rows}
    targets = [by_key[_key(j)] for j in pending if _key(j) in by_key]

    print(f"queue: {len(targets)} job(s), {workers} at a time", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        done = list(ex.map(one, targets))
    ok = sum(1 for j in done if j["status"] == "done")
    print(f"\nqueue: {ok}/{len(done)} finished cleanly", flush=True)
    return done


def status_rows():
    rows = load()
    if not rows:
        print("queue is empty; add work with `wsp queue add --model X`")
        return rows
    wm = max(len(r["model"]) for r in rows)
    print(f"{'model':{wm}s}  {'exp':5s} {'protocol':12s} {'status':8s} note")
    for r in rows:
        note = (r.get("note") or "")[:60].replace("\n", " ")
        print(
            f"{r['model']:{wm}s}  {r['exp']:5s} {str(r.get('protocol') or '-'):12s} "
            f"{r['status']:8s} {note}"
        )
    return rows
