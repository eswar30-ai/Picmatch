"""The queue: one audit at a time, each job's state saved to a file.

One worker thread, so audits never compete for the GPU with each other (they
still share it with Picmatch and with whatever else uses Ollama). Every state
change is written to `var/jobs/<audit_id>.json`, so `GET /api/audit/{id}`
answers after a restart, and a restart gives every unfinished job an answer:
one that never started is queued again; one that was running is marked FAILED
and not retried - it may be what stopped the process, and retrying it on every
start would turn one bad file into a crash loop.
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from idf import settings

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="idf-audit")
UNFINISHED = ("QUEUED", "PROCESSING")


def new_job_id() -> str:
    return f"{int(time.time())}_{uuid.uuid4().hex[:8]}"


def safe_name(raw: str | None) -> str:
    """A filename reduced to a bare name Windows can store."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", Path(raw or "").name).strip(" .")
    return name or "upload"


def _state_file(job_id: str) -> Path:
    return settings.JOBS / f"{job_id}.json"


def persist(job: dict[str, Any]) -> None:
    target = _state_file(job["job_id"])
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    temporary.replace(target)


def load(job_id: str) -> dict[str, Any] | None:
    """The job from memory, else from its state file; None if there is no such job."""
    if not re.fullmatch(r"\d+_[0-9a-f]{8}", job_id or ""):
        return None
    with JOBS_LOCK:
        if job_id in JOBS:
            return dict(JOBS[job_id])
    path = _state_file(job_id)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def waiting() -> int:
    with JOBS_LOCK:
        return sum(1 for j in JOBS.values() if j["state"] == "QUEUED")


def counts() -> dict[str, int]:
    with JOBS_LOCK:
        states = [j["state"] for j in JOBS.values()]
    return {s: states.count(s) for s in UNFINISHED}


def audit_files(folder: Path) -> dict[int, Path]:
    """The evidence files saved under `<id>_<name>`, by id."""
    out: dict[int, Path] = {}
    for path in folder.iterdir():
        head, _, rest = path.name.partition("_")
        if path.is_file() and rest and head.isdigit():
            out[int(head)] = path
    return out


def enqueue(job_id: str, folder: Path, files: int, caller: str | None = None) -> None:
    with JOBS_LOCK:
        ahead = sum(1 for j in JOBS.values() if j["state"] == "QUEUED")
        JOBS[job_id] = {"job_id": job_id, "state": "QUEUED", "queue_position": ahead + 1,
                        "files": files, "message": "Queued", "queued_at": time.time(), "caller": caller}
        snapshot = dict(JOBS[job_id])
    persist(snapshot)
    _worker.submit(run_job, job_id, folder)


def run_job(job_id: str, folder: Path) -> None:
    from idf import callback, ocr
    from idf.audit import callback_body, run
    from idf.package import parse

    def mark(**fields):
        with JOBS_LOCK:
            JOBS[job_id].update(fields)
            snapshot = dict(JOBS[job_id])
        persist(snapshot)

    mark(state="PROCESSING", started_at=time.time())
    try:
        package = parse(json.loads((folder / "package.json").read_text(encoding="utf-8")))
        health = ocr.vlm_health()
        if not (health.get("up") and health.get("model_present")):
            raise ocr.ModelDidNotAnswer(
                f"{health.get('reason') or 'The vision model is unavailable'} - this audit was not "
                f"run. Start Ollama with {health.get('model')}, then submit it again.")
        mark(message=f"Judging {len(package.evidence)} evidence file(s)")
        result = run(package, audit_files(folder))
        result["vlm"] = {"model": health.get("model")}
        target = settings.AUDITS / f"{job_id}.json"
        target.write_text(json.dumps(result, ensure_ascii=False, default=str), encoding="utf-8")
        result["callback"] = callback.post(package.callback_url, callback_body(result))
        target.write_text(json.dumps(result, ensure_ascii=False, default=str), encoding="utf-8")
        state = "ACCEPTED" if result["status"] == "approved" else "NEEDS_REVIEW"
        mark(state=state, finished_at=time.time(),
             message=f"{result['status'].replace('_', ' ')}, score {result['score']}: {result['comments']}")
        with JOBS_LOCK:
            took = JOBS[job_id]["finished_at"] - JOBS[job_id]["started_at"]
        print(f"audit {job_id} {package.id_key}={package.target_id}: {result['status']} {result['score']} "
              f"in {took:.1f}s ({len(package.evidence)} file(s))", flush=True)
    except Exception as exc:                                    # noqa: BLE001
        traceback.print_exc()
        mark(state="FAILED", finished_at=time.time(), message=f"{type(exc).__name__}: {exc}")
        print(f"audit {job_id} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)


def recover() -> dict[str, int]:
    """Give every job a previous process left unfinished an answer (see the module's docstring)."""
    requeued = failed = 0
    for path in sorted(settings.JOBS.glob("*.json")):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if job.get("state") not in UNFINISHED:
            continue
        folder = settings.UPLOADS / job["job_id"]
        if job["state"] == "QUEUED" and (folder / "package.json").is_file():
            job["message"] = "Queued again after the service restarted"
            with JOBS_LOCK:
                JOBS[job["job_id"]] = job
            persist(job)
            _worker.submit(run_job, job["job_id"], folder)
            requeued += 1
        else:
            job.update(state="FAILED", finished_at=time.time(), message=(
                "Interrupted by a service restart while it was running. Not retried; submit it again."))
            persist(job)
            failed += 1
    return {"requeued": requeued, "failed": failed}
