"""POST /api/audit, GET /api/audit/{id}, GET /api/health: the contract IDF's
backend codes against. The judgement itself is faked; the queue is real."""

import io
import json
import time

import pytest
from PIL import Image

from test_idf_package import CERTIFICATION


@pytest.fixture
def client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from idf import access, settings
    from idf.app import app
    for name in ("UPLOADS", "AUDITS", "JOBS"):
        monkeypatch.setattr(settings, name, tmp_path / name.lower())
        getattr(settings, name).mkdir()
    monkeypatch.setattr(access, "submissions", access.Window(access.rate_limit, 60))
    return TestClient(app, headers={"X-API-Key": "test-key"})


@pytest.fixture
def judged(monkeypatch):
    """The model up and every audit judged by a fake; what it was given is recorded."""
    from idf import audit as idf_audit, ocr
    monkeypatch.setattr(ocr, "vlm_health", lambda: {"up": True, "model_present": True, "model": "m"})
    seen = {}

    def fake_run(package, paths):
        seen["names"] = {i: p.name for i, p in paths.items()}
        return {"target_type": "certification", "application_id": package.target_id, "number": package.number,
                "status": "rejected", "score": 61, "comments": "c",
                "checks": [{"check_type": "overall", "status": "rejected", "score": 61, "details": "d"}],
                "evidence": [], "claims": [], "model_answers": {"39": "{...}"}}
    monkeypatch.setattr(idf_audit, "run", fake_run)
    monkeypatch.setattr("idf.callback.post", lambda url, body: {"posted": False, "status": None, "detail": "off"})
    return seen


def jpg():
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(buffer, "JPEG")
    return buffer.getvalue()


def files():
    return {"file_39": ("gym.jpg", jpg(), "image/jpeg"), "file_40": ("posters.jpg", jpg(), "image/jpeg"),
            "file_42": ("policy.pdf", b"%PDF-1.4 x", "application/pdf")}


def no_paths():
    package = json.loads(json.dumps(CERTIFICATION))
    for item in package["data"]["evidence_files"]:
        item.pop("url")
    return package


def wait(client, audit_id, seconds=10):
    for _ in range(seconds * 10):
        r = client.get(f"/api/audit/{audit_id}")
        if r.status_code != 409:
            return r
        time.sleep(0.1)
    pytest.fail("the audit did not finish")


def test_an_audit_is_queued_judged_and_fetched(client, judged):
    from idf import settings
    r = client.post("/api/audit", data={"package": json.dumps(CERTIFICATION)}, files=files())
    assert r.status_code == 202, r.text
    audit_id = r.json()["audit_id"]
    r = wait(client, audit_id)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["state"] == "REJECTED" and body["status"] == "rejected" and body["score"] == 61
    assert body["application_id"] == 26 and "model_answers" not in body and body["callback"]["posted"] is False
    assert body["elapsed_seconds"] is not None
    assert judged["names"] == {39: "39_gym.jpg", 40: "40_posters.jpg", 42: "42_policy.pdf"}
    stored = json.loads((settings.UPLOADS / audit_id / "package.json").read_text(encoding="utf-8"))
    assert "contact_email" not in stored["data"]["institution"] and stored["data"]["institution"]["name"]
    assert json.loads((settings.JOBS / f"{audit_id}.json").read_text(encoding="utf-8"))["state"] == "REJECTED"


def test_what_cannot_be_audited_is_refused_with_the_reason(client):
    r = client.post("/api/audit", data={"package": json.dumps(no_paths())},
                    files={"file_39": ("gym.jpg", jpg(), "image/jpeg")})
    assert r.status_code == 400 and "evidence 40" in r.json()["detail"]
    bad = {**files(), "file_42": ("policy.docx", b"PK", "application/octet-stream")}
    assert client.post("/api/audit", data={"package": json.dumps(CERTIFICATION)}, files=bad).status_code == 415
    r = client.post("/api/audit", data={"package": "{not json"}, files=files())
    assert r.status_code == 400 and "not valid JSON" in r.json()["detail"]
    r = client.post("/api/audit", data={"package": json.dumps({"target_type": "survey"})}, files=files())
    assert r.status_code == 400 and "target_type" in r.json()["detail"]
    assert client.post("/api/audit", data={}).status_code == 400


def test_a_placeholder_is_refused_before_it_is_queued(client):
    from idf import settings
    placeholder = {**files(), "file_39": ("gym.jpg", bytes.fromhex("ffd8ffe000104a4649460001"), "image/jpeg")}
    r = client.post("/api/audit", data={"package": json.dumps(CERTIFICATION)}, files=placeholder)
    assert r.status_code == 400 and "Evidence 39" in r.json()["detail"] and "12 bytes" in r.json()["detail"]
    not_pdf = {**files(), "file_42": ("policy.pdf", b"hello", "application/pdf")}
    r = client.post("/api/audit", data={"package": json.dumps(CERTIFICATION)}, files=not_pdf)
    assert r.status_code == 400 and "not a PDF" in r.json()["detail"]
    assert list(settings.UPLOADS.iterdir()) == [] and list(settings.JOBS.iterdir()) == []


def test_a_file_over_the_limit_is_refused(client, monkeypatch):
    from idf import settings
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_MB", 0)
    r = client.post("/api/audit", data={"package": json.dumps(CERTIFICATION)}, files=files())
    assert r.status_code == 413 and list(settings.UPLOADS.iterdir()) == []


def test_files_not_sent_are_fetched_from_idf_by_their_path(client, judged, monkeypatch):
    from idf import fetch
    fetched = []

    def fake_download(url, target, *, base=None, max_bytes):
        fetched.append((url, target.name))
        target.write_bytes(b"%PDF-1.4 x" if target.suffix == ".pdf" else jpg())
        return target
    monkeypatch.setattr(fetch, "download", fake_download)
    r = client.post("/api/audit", data={"package": json.dumps(CERTIFICATION)},
                    files={"file_39": files()["file_39"]})
    assert r.status_code == 202, r.text
    assert wait(client, r.json()["audit_id"]).status_code == 200
    assert fetched == [("/uploads/y.jpg", "40_posters.jpg"), ("/uploads/z.pdf", "42_policy.pdf")]
    assert judged["names"] == {39: "39_gym.jpg", 40: "40_posters.jpg", 42: "42_policy.pdf"}


def test_a_path_that_cannot_be_fetched_is_refused_with_the_reason(client, monkeypatch):
    from idf import fetch, settings

    def placeholder(url, target, *, base=None, max_bytes):
        raise fetch.FetchFailed("not a readable image (12 bytes: OSError)")
    monkeypatch.setattr(fetch, "download", placeholder)
    r = client.post("/api/audit", data={"package": json.dumps(CERTIFICATION)})
    assert r.status_code == 400 and "Evidence 39" in r.json()["detail"] and "12 bytes" in r.json()["detail"]
    assert list(settings.UPLOADS.iterdir()) == []


def test_a_failed_audit_answers_422_and_an_unknown_one_404(client, monkeypatch):
    from idf import ocr
    monkeypatch.setattr(ocr, "vlm_health", lambda: {"up": False, "model_present": False, "model": "m",
                                                    "reason": "Ollama is not reachable"})
    r = client.post("/api/audit", data={"package": json.dumps(CERTIFICATION)}, files=files())
    r = wait(client, r.json()["audit_id"])
    assert r.status_code == 422 and "Ollama" in r.json()["message"] and r.json()["state"] == "FAILED"
    assert client.get("/api/audit/1700000000_deadbeef").status_code == 404
    assert client.get("/api/audit/..%2F..%2Fsecrets").status_code == 404


def test_a_key_is_needed_unless_the_switch_is_on(client, monkeypatch):
    from fastapi.testclient import TestClient
    from idf.app import app
    anonymous = TestClient(app)
    assert anonymous.post("/api/audit", data={"package": "{}"}, files=files()).status_code == 401
    assert anonymous.get("/api/audit/1700000000_deadbeef").status_code == 401
    wrong = TestClient(app, headers={"X-API-Key": "nope"})
    assert wrong.get("/api/audit/1700000000_deadbeef").status_code == 401
    monkeypatch.setenv("OPEN_AUDIT_API", "true")
    assert anonymous.post("/api/audit", data={"package": "{not json"}).status_code == 400    # past the gate
    assert anonymous.get("/api/audit/1700000000_deadbeef").status_code == 404


def test_too_many_submissions_from_one_address_wait(client, monkeypatch):
    monkeypatch.setenv("UPLOAD_RATE_LIMIT_PER_MINUTE", "2")
    bad = {"package": json.dumps({"target_type": "survey"})}
    from idf import access
    access.submissions.record("testclient")
    access.submissions.record("testclient")
    r = client.post("/api/audit", data=bad)
    assert r.status_code == 429 and int(r.headers["Retry-After"]) > 0


def test_health_says_whether_the_model_is_up_without_a_key(client, monkeypatch):
    from fastapi.testclient import TestClient
    from idf import ocr
    from idf.app import app
    monkeypatch.setattr(ocr, "vlm_health", lambda: {"up": True, "model_present": True, "model": "qwen2.5vl:7b"})
    body = TestClient(app).get("/api/health").json()
    assert body == {"ollama": "up", "model": "qwen2.5vl:7b", "model_present": True,
                    "queue": {"QUEUED": 0, "PROCESSING": 0}}


def test_a_restart_queues_waiting_audits_again_and_fails_the_running_one(client, judged):
    from idf import jobs, settings
    waiting, running = "1700000000_aaaaaaaa", "1700000001_bbbbbbbb"
    folder = settings.UPLOADS / waiting
    folder.mkdir()
    (folder / "package.json").write_text(json.dumps(CERTIFICATION), encoding="utf-8")
    for name, (filename, data, _) in files().items():
        (folder / f"{name[5:]}_{filename}").write_bytes(data)
    jobs.persist({"job_id": waiting, "state": "QUEUED", "message": "Queued", "queued_at": time.time()})
    jobs.persist({"job_id": running, "state": "PROCESSING", "message": "Judging", "queued_at": time.time()})
    assert jobs.recover() == {"requeued": 1, "failed": 1}
    assert wait(client, waiting).status_code == 200
    r = client.get(f"/api/audit/{running}")
    assert r.status_code == 422 and "restart" in r.json()["message"]


def test_the_root_says_what_this_is_for_a_browser_or_an_uptime_check(client):
    from fastapi.testclient import TestClient
    from idf.app import app
    plain = TestClient(app)                     # no key: it carries no audit data
    assert plain.get("/").status_code == 200 and plain.get("/").json()["service"] == "IDF audit"
    assert plain.head("/").status_code == 200


def test_a_finished_audit_reads_accepted_or_rejected_even_if_saved_as_needs_review(client, judged):
    """Audits finished before 2026-11-19 were saved as NEEDS_REVIEW, which their app
    showed as "in review"; fetched now, the state follows the verdict."""
    from idf import settings
    r = client.post("/api/audit", data={"package": json.dumps(CERTIFICATION)}, files=files())
    audit_id = r.json()["audit_id"]
    assert wait(client, audit_id).status_code == 200
    job_file = settings.JOBS / f"{audit_id}.json"
    job = json.loads(job_file.read_text(encoding="utf-8"))
    job_file.write_text(json.dumps({**job, "state": "NEEDS_REVIEW"}), encoding="utf-8")
    from idf import jobs
    with jobs.JOBS_LOCK:
        jobs.JOBS.pop(audit_id, None)
    assert client.get(f"/api/audit/{audit_id}").json()["state"] == "REJECTED"
