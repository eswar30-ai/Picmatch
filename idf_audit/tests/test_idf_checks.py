"""idf/checks.py and idf/audit.py: the verdict from readings, with
fake readers. Values invented."""

from pathlib import Path

import pytest

from idf import audit, checks, package
from idf.ocr import ModelDidNotAnswer
from test_idf_package import CERTIFICATION, PROGRAM


def read(shows="yes", confidence=0.9, text=(), reason="r", parsed=True):
    return {"kind": "photo", "shows": shows, "confidence": confidence, "reason": reason,
            "description": "d", "text": list(text), "parsed": parsed, "model_answer": "{}"}


@pytest.mark.parametrize("r,status", [
    (read("yes", 0.9), "approved"), (read("yes", 0.5), "changes_required"),
    (read("no", 0.9), "rejected"), (read("no", 0.5), "changes_required"),
    (read("unsure", 0.9), "changes_required"), (read("yes", 0.9, parsed=False), "changes_required"),
])
def test_a_files_status_follows_the_verdict_and_how_sure_it_was(r, status):
    assert checks.evidence_status(r)[0] == status


def test_worst_status_wins():
    assert checks.worst(["approved", "changes_required", "rejected"]) == "rejected"
    assert checks.worst(["approved", "changes_required"]) == "changes_required"
    assert checks.worst(["approved"]) == "approved" and checks.worst([]) == "approved"


def test_evidence_review_names_what_was_not_shown():
    p = package.parse(CERTIFICATION)
    reads = {39: read("yes", 0.95), 40: read("no", 0.8, reason="a blank wall"), 42: read("unsure", 0.3, reason="not legible")}
    c = checks.evidence_review("evidence_review", p, reads, "their requirement")
    assert c["status"] == "rejected" and "1 of 3 files" in c["details"]
    assert "PA Promotion Signage does not: a blank wall" in c["details"] and "PA Policy Document cannot be told" in c["details"]


def test_a_question_is_related_to_a_requirement_by_a_shared_content_word():
    p = package.parse(CERTIFICATION)
    assert checks.related(p.answers[0], p.evidence[0])          # exercise ~ Safe Exercise Area
    assert not checks.related(p.answers[0], p.evidence[1])      # signage


def test_detail_match_backs_a_yes_with_the_file_about_it_and_finds_the_name():
    p = package.parse(CERTIFICATION)
    reads = {39: read("yes", 0.9, text=["EXAMPLE DIGITAL CAMPUS", "GYM"]), 40: read("yes", 0.9), 42: read("yes", 0.8)}
    c = checks.detail_evidence_match(p, reads)
    assert c["status"] == "approved" and "printed on file(s) 39" in c["details"] and "1 of 1" in c["details"]


def test_a_yes_contradicted_by_its_file_rejects():
    p = package.parse(CERTIFICATION)
    reads = {39: read("no", 0.9), 40: read("yes", 0.9), 42: read("yes", 0.8)}
    c = checks.detail_evidence_match(p, reads)
    assert c["status"] == "rejected" and "Q1" in c["details"]


def test_no_checkable_claim_is_review_not_approval():
    data = {**CERTIFICATION["data"], "questionnaire_answers": []}
    c = checks.detail_evidence_match(package.parse(data), {39: read()})
    assert c["status"] == "changes_required" and "no \"Yes\" answer" in c["details"]


def test_location_check_rolls_up_the_findings():
    p = package.parse(CERTIFICATION)
    findings = {39: {"status": "approved", "score": 95, "details": "agree"},
                40: {"status": "changes_required", "score": 40, "details": "no GPS"}, 42: None}
    c = checks.location_validation(p, findings)
    assert c["status"] == "changes_required" and c["score"] == 68 and "PA Promotion Signage: no GPS" in c["details"]
    assert c["counted"] is True


def test_gps_counts_only_when_a_photo_requires_it():
    # The owner, 2026-11-14: GPS aside until IDF makes it mandatory (requires_gps).
    optional = package.parse({**CERTIFICATION["data"], "evidence_files": [
        {**e, "requires_gps": False} for e in CERTIFICATION["data"]["evidence_files"]]})
    c = checks.location_validation(optional, {39: None, 40: None, 42: None})
    assert c["counted"] is False and "not checked" in c["details"] and "requires_gps" in c["details"]
    assert c["status"] == "approved"
    good = [{"check_type": "evidence_review", "status": "approved", "score": 90, "details": ""},
            {"check_type": "detail_evidence_match", "status": "approved", "score": 80, "details": ""}]
    final = checks.overall(good + [c])
    assert final["status"] == "approved" and final["score"] == 85                # the location check left out
    counted = {**c, "counted": True, "status": "changes_required", "score": 40}
    assert checks.overall(good + [counted])["status"] == "changes_required"


def test_overall_is_the_worst_check_and_a_rejection_caps_the_score():
    good = [{"check_type": "a", "status": "approved", "score": 90, "details": ""}] * 3
    assert checks.overall(good)["status"] == "approved" and checks.overall(good)["score"] == 90
    bad = good + [{"check_type": "d", "status": "rejected", "score": 0, "details": ""}]
    o = checks.overall(bad)
    assert o["status"] == "rejected" and o["score"] <= 30 and "d" in o["details"]


def test_program_content_review_by_rule():
    p = package.parse(PROGRAM)
    assert checks.content_review(p)["status"] == "approved"
    broken = package.parse({**PROGRAM, "dates": {"start_date": "2026-11-22", "end_date": "2026-11-20"}, "schedule": []})
    c = checks.content_review(broken)
    assert c["status"] == "changes_required" and "before the start" in c["details"] and "no agenda" in c["details"]


def test_program_details_match_looks_for_title_venue_and_dates_on_the_files():
    p = package.parse(PROGRAM)
    reads = {1: read(text=["ANNUAL FITNESS DRIVE 2026", "EXAMPLE STADIUM", "20-22 NOV 2026"]), 2: read(text=[])}
    c = checks.program_details_match(p, reads)
    assert c["status"] == "approved" and "title is printed on file(s) 1" in c["details"]
    assert "venue is named on file(s) 1" in c["details"] and "dates appear on file(s) 1" in c["details"]
    assert "organiser is not named" in c["details"]
    assert checks._dates_on("Nov 2026", ("2026-11-20", "2026-11-22")) and checks._dates_on("22/11/2026", ("2026-11-20", "2026-11-22"))
    assert not checks._dates_on("2026", ("2026-11-20", "2026-11-22")) and not checks._dates_on("20 NOV", ("2026-11-20", "2026-11-22"))
    c = checks.program_details_match(p, {1: read(text=["EXAMPLE HEALTH ALLIANCE", "EXAMPLE STADIUM"]), 2: read(text=[])})
    assert c["status"] == "approved" and "organiser is named on file(s) 1" in c["details"]
    assert checks.program_details_match(p, {1: read(text=[]), 2: read(text=[])})["status"] == "changes_required"


# --- the audit as a whole -------------------------------------------------------

def fake_reader(answers):
    def reader(path: Path, claim: str):
        return answers[path.name]
    return reader


def test_run_gives_their_callback_body_and_our_detail(tmp_path):
    p = package.parse(CERTIFICATION)
    files = {39: tmp_path / "gym.jpg", 40: tmp_path / "posters.jpg", 42: tmp_path / "policy.pdf"}
    for f in files.values():
        f.write_bytes(b"x")
    answers = {"gym.jpg": read("yes", 0.95, text=["FITNESS CENTER"]), "posters.jpg": read("yes", 0.9),
               "policy.pdf": {**read("yes", 0.8), "kind": "document"}}
    result = audit.run(p, files, read_photo=fake_reader(answers), read_document=fake_reader(answers))
    assert result["target_type"] == "certification" and result["application_id"] == 26
    assert [c["check_type"] for c in result["checks"]] == [
        "evidence_review", "image_location_validation", "detail_evidence_match", "overall"]
    assert result["status"] == "changes_required"                 # the files carry no GPS
    assert [e["id"] for e in result["evidence"]] == [39, 40, 42] and result["evidence"][2]["location"] is None
    body = audit.callback_body(result)
    assert set(body) == {"status", "score", "comments", "checks", "evidence"} and 0 <= body["score"] <= 100
    assert body["evidence"] == [{"id": e["id"], "ai_status": e["ai_status"], "ai_note": e["ai_note"]}
                                for e in result["evidence"]]
    assert "model_answers" not in audit.answer_body(result) and "evidence" in audit.answer_body(result)


def test_each_file_gets_one_verdict_and_one_note():
    near = {"status": "approved", "score": 95, "details": "the file's GPS is 2 m from the declared position: they agree"}
    far = {"status": "rejected", "score": 10, "details": "the file's GPS is 5004 m from the declared position: not the declared place"}
    none = {"status": "changes_required", "score": 40, "details": "the file carries no GPS data, so the declared position cannot be checked"}
    shows = {**read("yes", 0.95), "reason": "A FITNESS CENTER sign over gym machines."}
    assert audit.file_verdict(shows, near) == ("approved", "A FITNESS CENTER sign over gym machines.")
    assert audit.file_verdict(shows, None) == ("approved", "A FITNESS CENTER sign over gym machines.")
    status, note = audit.file_verdict(shows, far)                 # the right picture, in the wrong place
    assert status == "rejected" and note.startswith("A FITNESS") and "Location: the file's GPS is 5004 m" in note
    status, note = audit.file_verdict(shows, none)
    assert status == "changes_required" and "no GPS data" in note
    status, note = audit.file_verdict({**read("no", 0.9), "reason": "A car park."}, near)
    assert status == "rejected" and note == "Does not show it: A car park."
    status, note = audit.file_verdict({**read("unsure", 0.5), "reason": ""}, None)
    assert status == "changes_required" and note == "Cannot be told from the file: no clear answer"
    status, note = audit.file_verdict({**read("yes", 0.9), "reason": "x" * 900}, far)
    assert len(note) <= audit.NOTE_MAX


def test_the_claim_tells_the_model_what_the_file_is_for():
    p = package.parse(CERTIFICATION)
    claim = audit.claim_for(p, p.evidence[0])
    assert '"Safe Exercise Area"' in claim and "Example Digital Campus" in claim and "exercise" in claim.lower()
    q = package.parse(PROGRAM)
    assert "Annual Fitness Drive 2026" in audit.claim_for(q, q.evidence[0]) and "2026-11-20" in audit.claim_for(q, q.evidence[0])


def test_a_missing_file_or_a_silent_model_fails_the_audit(tmp_path):
    p = package.parse(CERTIFICATION)
    with pytest.raises(ValueError, match="evidence 40, 42"):
        audit.run(p, {39: tmp_path / "a.jpg"}, read_photo=lambda *a: read(), read_document=lambda *a: read())
    files = {39: tmp_path / "a.jpg", 40: tmp_path / "b.jpg", 42: tmp_path / "c.pdf"}
    for f in files.values():
        f.write_bytes(b"x")

    def down(path, claim):
        raise ModelDidNotAnswer("down")
    with pytest.raises(ModelDidNotAnswer):
        audit.run(p, files, read_photo=down, read_document=down)


def test_a_program_audit_runs_its_own_checks(tmp_path):
    p = package.parse(PROGRAM)
    files = {1: tmp_path / "track.jpg", 2: tmp_path / "agenda.pdf"}
    for f in files.values():
        f.write_bytes(b"x")
    answers = {"track.jpg": read("yes", 0.9, text=["EXAMPLE STADIUM"]), "agenda.pdf": {**read("yes", 0.9), "kind": "document"}}
    result = audit.run(p, files, read_photo=fake_reader(answers), read_document=fake_reader(answers))
    assert result["program_id"] == 29 and [c["check_type"] for c in result["checks"]] == [
        "supporting_materials", "content_review", "program_details_match", "overall"]


def test_the_same_file_uploaded_for_two_requirements_is_flagged(tmp_path):
    p = package.parse(CERTIFICATION)
    files = {39: tmp_path / "gym.jpg", 40: tmp_path / "posters.jpg", 42: tmp_path / "policy.pdf"}
    files[39].write_bytes(b"same picture")
    files[40].write_bytes(b"same picture")
    files[42].write_bytes(b"a policy")
    answers = {"gym.jpg": read("yes", 0.95), "posters.jpg": read("yes", 0.95),
               "policy.pdf": {**read("yes", 0.9), "kind": "document"}}
    result = audit.run(p, files, read_photo=fake_reader(answers), read_document=fake_reader(answers))
    review = result["checks"][0]
    assert review["status"] == "changes_required" and "the same file" in review["details"]
    by_id = {e["id"]: e for e in result["evidence"]}
    assert by_id[39]["ai_status"] == "changes_required" and "PA Promotion Signage" in by_id[39]["ai_note"]
    assert by_id[40]["same_file_as"] == [39] and by_id[42]["same_file_as"] == []
    assert result["status"] != "approved"


def test_files_are_read_by_their_full_path(tmp_path, monkeypatch):
    # The caller may run from any folder, so a relative path would point
    # somewhere else by the time it is read.
    p = package.parse(CERTIFICATION)
    monkeypatch.chdir(tmp_path)
    for name in ("gym.jpg", "posters.jpg", "policy.pdf"):
        (tmp_path / name).write_bytes(b"x")
    seen = []
    def reader(path, claim):
        seen.append(path)
        return read("yes", 0.9)
    audit.run(p, {39: Path("gym.jpg"), 40: Path("posters.jpg"), 42: Path("policy.pdf")},
              read_photo=reader, read_document=reader)
    assert all(path.is_absolute() for path in seen) and seen[0] == tmp_path / "gym.jpg"


def test_the_real_readers_read_each_files_text_ahead_and_judge_in_order(tmp_path, monkeypatch):
    """Text is read by the worker, the model is asked in the main thread, one
    file at a time and in the package's order - and the result is the one the
    readers give on their own."""
    import threading

    from idf import evidence
    p = package.parse(CERTIFICATION)
    files = {39: tmp_path / "gym.jpg", 40: tmp_path / "posters.jpg", 42: tmp_path / "policy.pdf"}
    for f in files.values():
        f.write_bytes(b"x")
    texts = {"gym.jpg": "FITNESS CENTER", "posters.jpg": "TAKE THE STAIRS", "policy.pdf": " ".join(["policy"] * 30)}
    main, text_threads, asked = threading.current_thread(), set(), []

    def photo_text(path):
        text_threads.add(threading.current_thread())
        return [texts[path.name]]

    def document_text(path):
        text_threads.add(threading.current_thread())
        return texts[path.name]

    def ask(prompt, image=None):
        assert threading.current_thread() is main
        asked.append(next(name for name, text in texts.items() if text in prompt))
        return '{"shows": "yes", "confidence": 0.9, "reason": "r", "description": "d", "text_seen": []}'

    monkeypatch.setattr(evidence, "photo_text", photo_text)
    monkeypatch.setattr(evidence, "document_text", document_text)
    monkeypatch.setattr(evidence, "ask", ask)
    result = audit.run(p, files)
    assert asked == ["gym.jpg", "posters.jpg", "policy.pdf"]
    assert main not in text_threads and len(text_threads) == 1
    assert [e["text"] for e in result["evidence"]] == [["FITNESS CENTER"], ["TAKE THE STAIRS"], [texts["policy.pdf"]]]


def test_a_model_that_does_not_answer_still_fails_the_audit_with_text_read_ahead(tmp_path, monkeypatch):
    from idf import evidence
    p = package.parse(CERTIFICATION)
    files = {39: tmp_path / "gym.jpg", 40: tmp_path / "posters.jpg", 42: tmp_path / "policy.pdf"}
    monkeypatch.setattr(evidence, "photo_text", lambda path: ["WORDS"])
    monkeypatch.setattr(evidence, "document_text", lambda path: "words " * 30)

    def down(prompt, image=None):
        raise ModelDidNotAnswer("down")
    monkeypatch.setattr(evidence, "ask", down)
    with pytest.raises(ModelDidNotAnswer):
        audit.run(p, files)
