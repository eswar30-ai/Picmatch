"""scripts/idf_sample.py: the sample packages have every field IDF's ai-package
endpoints were seen to return, parse, and come with a file per evidence item."""

import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import idf_sample  # noqa: E402

from idf import location, package  # noqa: E402

CERTIFICATION_KEYS = {"target_type", "application_id", "application_no", "application_type", "institution",
                      "questionnaire_answers", "evidence_files", "submitted_at", "ai_validation_context"}
INSTITUTION_KEYS = {"name", "category_id", "category", "address", "country", "region", "city", "pincode", "full_address",
                    "contact_person", "contact_email", "contact_phone", "details"}
EVIDENCE_KEYS = {"id", "requirement_id", "requirement_title", "kind", "name", "mime_type", "url", "source",
                 "latitude", "longitude", "captured_at", "requires_gps"}
PROGRAM_KEYS = {"target_type", "program_id", "program_no", "program_name", "title", "program_type", "program_type_id",
                "description", "organization", "venue", "dates", "schedule", "banner_image_url", "evidence_materials",
                "submitted_at", "ai_validation_context"}
CONTEXT_KEYS = {"prompt_context", "required_checks", "callback_url", "callback_method", "callback_headers"}


def test_the_sample_packages_carry_their_fields():
    c = idf_sample.certification_package(False)["data"]
    assert set(c) == CERTIFICATION_KEYS and set(c["institution"]) == INSTITUTION_KEYS
    assert all(set(e) == EVIDENCE_KEYS for e in c["evidence_files"])
    assert all(set(a) == {"question_id", "section", "question", "answer_type", "answer", "raw"} for a in c["questionnaire_answers"])
    assert set(c["ai_validation_context"]) == CONTEXT_KEYS
    assert [k["check_type"] for k in c["ai_validation_context"]["required_checks"]] == [
        "evidence_review", "image_location_validation", "detail_evidence_match", "overall"]
    p = idf_sample.program_package()["data"]
    assert set(p) == PROGRAM_KEYS and set(p["ai_validation_context"]) == CONTEXT_KEYS
    assert all(set(e) == {"id", "name", "mime_type", "url"} for e in p["evidence_materials"])


def test_the_samples_parse_and_the_files_match(tmp_path):
    cert, prog = idf_sample.write(tmp_path)
    for folder in (cert, prog):
        pk = package.parse(json.loads((folder / "package.json").read_text(encoding="utf-8")))
        for e in pk.evidence:
            files = [f for f in folder.iterdir() if f.name.startswith(f"{e.id}_")]
            assert len(files) == 1, e
    pk = package.parse(json.loads((cert / "package.json").read_text(encoding="utf-8")))
    assert [e.kind for e in pk.evidence] == ["photo", "photo", "photo", "document"]
    exif = location.exif_of(cert / "43_gym_floor.jpg")
    assert exif["latitude"] is not None and location.check(pk.evidence[0], exif)["status"] == "approved"
    assert Image.open(cert / "45_badminton_court.jpg").size == (1200, 800)
    assert (cert / "46_active_workplace_policy_2026.pdf").read_bytes()[:5] == b"%PDF-"
    assert package.parse(json.loads((prog / "package.json").read_text(encoding="utf-8"))).organizer == "Example Health Alliance"


def test_contradict_moves_the_signage_photo_away(tmp_path):
    cert, _ = idf_sample.write(tmp_path, contradict=True)
    pk = package.parse(json.loads((cert / "package.json").read_text(encoding="utf-8")))
    found = location.check(pk.evidence[1], location.exif_of(cert / "44_stairs_poster.jpg"))
    assert found["status"] == "rejected" and found["distance_m"] > 4000
