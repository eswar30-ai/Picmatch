"""The API IDF's backend calls.

    POST /api/audit            multipart: `package` (the ai-package JSON as a string), and
                               `file_<id>` per evidence item - or its path in the package,
                               and the file is fetched from IDF's server
    GET  /api/audit/{id}       409 while it runs, 422 if it failed, 200 with the verdict
    GET  /api/health           whether Ollama and the model are up, and the queue

The same contract as the OCR platform's `/api/audit` (D:\\ocr-platform, docs/API.md).
"""

from __future__ import annotations

import json
import shutil
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from idf import access, fetch, jobs, ocr, settings
from idf.audit import answer_body
from idf.package import BadPackage, Evidence, parse

AUDIT_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".pdf"}
#: Keys of the package never written to disk: nothing here needs them.
_CONTACT = ("contact_person", "contact_phone", "contact_email")
_SUFFIX_OF_MIME = {"image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png", "image/webp": ".webp",
                   "application/pdf": ".pdf"}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    recovered = jobs.recover()
    if any(recovered.values()):
        print(f"after the restart: {recovered['requeued']} audit(s) queued again, "
              f"{recovered['failed']} marked FAILED", flush=True)
    if access.is_open():
        print("WARNING OPEN_AUDIT_API is on - anyone who can reach this host can submit audits "
              "and read their verdicts without a key", file=sys.stderr, flush=True)
    # The OCR engine takes a second or two to build; build it now, not on the first audit.
    threading.Thread(target=ocr.engine, daemon=True).start()
    yield


app = FastAPI(title="IDF audit", lifespan=lifespan)


@app.exception_handler(StarletteHTTPException)
async def refused(request: Request, exc: StarletteHTTPException) -> Any:
    """Every refusal (400, 401, 404, 413, 415, 429, 503) with its reason, in the
    log as well as in the answer: the access log line shows only the status.
    "Still running" (409) and a failed audit (422) are answers, not refusals."""
    print(f"{exc.status_code} for {access.client_key(request)} {request.method} {request.url.path}: {exc.detail}",
          file=sys.stderr, flush=True)
    return await http_exception_handler(request, exc)


def fetched_suffix(item: Evidence) -> str:
    """The file type of an evidence file to be fetched: from its path, its name, or its mime type."""
    for candidate in (urlsplit(item.url).path, item.name):
        suffix = Path(candidate or "").suffix.lower()
        if suffix:
            return suffix
    return _SUFFIX_OF_MIME.get(item.mime_type, "")


def scrubbed(data: dict[str, Any]) -> dict[str, Any]:
    """The package as received, without the contact details."""
    inner = data["data"] if isinstance(data.get("data"), dict) and "target_type" not in data else data
    institution = inner.get("institution")
    if isinstance(institution, dict):
        inner["institution"] = {k: v for k, v in institution.items() if k not in _CONTACT}
    return data


def save_upload(upload: UploadFile, target: Path, max_bytes: int) -> None:
    """Write an uploaded file to disk, refusing it with 413 past `max_bytes`."""
    size = 0
    with target.open("wb") as out:
        while chunk := upload.file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(413, f"The file is larger than {settings.MAX_UPLOAD_SIZE_MB} MB.")
            out.write(chunk)
    if size == 0:
        raise HTTPException(400, f"{upload.filename or 'A file'} is empty.")


@app.post("/api/audit")
async def start_audit(request: Request) -> JSONResponse:
    access.require(request)
    caller = access.client_key(request)
    wait = access.submissions.blocked_for(caller)
    if wait:
        raise HTTPException(429, f"Too many submissions - the limit is {access.rate_limit()} a minute. "
                                 f"Try again in {wait}s.", headers={"Retry-After": str(wait)})
    if jobs.waiting() >= access.max_queued():
        raise HTTPException(503, f"{jobs.waiting()} audits are already waiting. Try again in a minute.",
                            headers={"Retry-After": "60"})
    form = await request.form()
    raw = form.get("package")
    if not isinstance(raw, str) or not raw.strip():
        raise HTTPException(400, "Send the ai-package JSON as the form field `package`.")
    try:
        data = json.loads(raw)
        package = parse(data)
    except ValueError as exc:                     # json.JSONDecodeError and BadPackage both
        reason = str(exc) if isinstance(exc, BadPackage) else "the package is not valid JSON"
        raise HTTPException(400, f"The package cannot be audited: {reason}.") from exc

    # Each file is sent as file_<id>, or fetched from IDF's server by the path the
    # package gives (idf/fetch.py) when it is not.
    uploads: list[tuple[int, UploadFile]] = []
    fetches: list[tuple[int, str, str]] = []
    for item in package.evidence:
        upload = form.get(f"file_{item.id}")
        if isinstance(upload, UploadFile):
            suffix = Path(upload.filename or item.name or "").suffix.lower()
        elif item.url:
            suffix = fetched_suffix(item)
        else:
            raise HTTPException(400, f"No file was sent for evidence {item.id} ({item.name or item.title}), and the "
                                     f"package gives no path to fetch it from: send it as the form field "
                                     f"file_{item.id}, or put its path on the server in its `url`.")
        if suffix not in AUDIT_SUFFIXES:
            raise HTTPException(415, f"Evidence {item.id}: '{suffix or 'no extension'}' is not audited. "
                                     f"Send a JPG, PNG, WEBP or PDF.")
        if isinstance(upload, UploadFile):
            uploads.append((item.id, upload))
        else:
            fetches.append((item.id, item.url, jobs.safe_name(Path(item.name).stem + suffix if item.name else
                                                              f"evidence{suffix}")))
    access.submissions.record(caller)

    job_id = jobs.new_job_id()
    folder = settings.UPLOADS / job_id
    folder.mkdir(parents=True, exist_ok=True)
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    try:
        for evidence_id, upload in uploads:
            target = folder / f"{evidence_id}_{jobs.safe_name(upload.filename)}"
            await run_in_threadpool(save_upload, upload, target, max_bytes)
            try:
                await run_in_threadpool(fetch.check_file, target)
            except fetch.FetchFailed as exc:
                raise HTTPException(400, f"Evidence {evidence_id}: the file sent as file_{evidence_id} "
                                         f"cannot be audited: {exc}.") from None
        for evidence_id, url, name in fetches:
            try:
                await run_in_threadpool(fetch.download, url, folder / f"{evidence_id}_{name}", max_bytes=max_bytes)
            except fetch.FetchFailed as exc:
                raise HTTPException(400, f"Evidence {evidence_id}: the file at {url} cannot be audited: {exc}.") from None
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    (folder / "package.json").write_text(json.dumps(scrubbed(data), ensure_ascii=False), encoding="utf-8")
    jobs.enqueue(job_id, folder, len(uploads) + len(fetches), access.caller_of(request))
    return JSONResponse({"audit_id": job_id, "state": "QUEUED"}, status_code=202)


@app.get("/api/audit/{audit_id}")
def audit_answer(audit_id: str, request: Request) -> Any:
    access.require(request)
    job = jobs.load(audit_id)
    if job is None:
        raise HTTPException(404, "No such audit")
    state = job.get("state")
    if state in jobs.UNFINISHED:
        return JSONResponse({"audit_id": audit_id, "state": state, "message": job.get("message")},
                            status_code=409, headers={"Retry-After": "10"})
    if state == "FAILED":
        return JSONResponse({"audit_id": audit_id, "state": state, "message": job.get("message")}, status_code=422)
    path = settings.AUDITS / f"{audit_id}.json"
    if not path.is_file():
        raise HTTPException(404, "The verdict is no longer on disk")
    result = json.loads(path.read_text(encoding="utf-8"))
    return {"audit_id": audit_id, "state": state,
            "elapsed_seconds": round((job.get("finished_at") or time.time()) - (job.get("started_at") or 0), 1)
            if job.get("started_at") else None,
            **answer_body(result)}


@app.api_route("/", methods=["GET", "HEAD"])
def root() -> dict[str, str]:
    """What this is and where to go: for a browser opening the address, a link
    preview, or an uptime check - all of which got 404 here before."""
    return {"service": "IDF audit", "health": "/api/health", "submit": "POST /api/audit",
            "result": "GET /api/audit/{audit_id}"}


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Whether the vision model is up, and how many audits wait. No key, no audit data."""
    vlm = ocr.vlm_health()
    return {"ollama": "up" if vlm["up"] else "down", "model": vlm["model"],
            "model_present": vlm["model_present"], "queue": jobs.counts()}
