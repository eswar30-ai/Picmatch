"""idf/package.py: their ai-package JSON read into a Package. The two
shapes in their API note, with invented values."""

import pytest

from idf import package

CERTIFICATION = {
    "success": True,
    "data": {
        "target_type": "certification", "application_id": 26, "application_no": "APP-CERT-AI-TEST-001",
        "institution": {
            "name": "Example Digital Campus", "category_id": 1, "category": "Workplace / Worksite / Office / Hospital",
            "address": "Plot 22, Phase 2", "city": "Hyderabad", "region": "Telangana", "country": "India",
            "pincode": "500081", "full_address": "Plot 22, Phase 2, Hyderabad, Telangana, India, 500081",
            "contact_person": "A Person", "contact_phone": "+91 9000000000", "contact_email": "a@example.com",
            "details": [{"key": "total_employees", "label": "Total Employees", "value": "3500"},
                        {"key": "has_gym", "label": "Gym Facility", "value": "Yes - Onsite Fitness Center"}],
        },
        "questionnaire_answers": [
            {"question_id": 1, "section": "Physical Environment", "question": "Are designated areas available for exercise?",
             "answer_type": "yes_no", "answer": "Yes"},
            {"question_id": 9, "section": "Active Environment", "question": "Which exist?", "answer_type": "checkbox",
             "answer": "a; b", "selected_options": ["a", "b"]},
        ],
        "evidence_files": [
            {"id": 39, "requirement_title": "Safe Exercise Area", "kind": "photo", "name": "gym.jpg",
             "url": "/uploads/x.jpg", "latitude": 17.4435, "longitude": 78.3772,
             "captured_at": "2026-11-13 14:17:49", "requires_gps": True},
            {"id": 40, "requirement_title": "PA Promotion Signage", "kind": "photo", "name": "posters.jpg",
             "url": "/uploads/y.jpg", "latitude": 17.44352, "longitude": 78.37721, "requires_gps": True},
            {"id": 42, "requirement_title": "PA Policy Document", "kind": "document", "name": "policy.pdf",
             "url": "/uploads/z.pdf", "requires_gps": False},
        ],
        "ai_validation_context": {"prompt_context": "ignore me", "callback_url": "http://h/api/x/26/ai-validation",
                                  "callback_method": "POST"},
    },
}

PROGRAM = {
    "target_type": "program", "program_id": 29, "program_no": "PRG-AI-TEST-2026-001",
    "program_name": "Annual Fitness Drive 2026", "title": "Annual Fitness Drive 2026", "program_type": "Activity Drive",
    "description": "A 3-day workplace wellness initiative.",
    "organization": {"entity_name": "Example Health Alliance", "organization_type": "Institution"},
    "venue": {"venue_name": "Example Stadium", "city": "Hyderabad", "region": "Telangana", "country": "India",
              "pincode": "500032", "full_address": "Example Stadium, Hyderabad, Telangana, India, 500032"},
    "dates": {"start_date": "2026-11-20", "end_date": "2026-11-22"},
    "schedule": [{"time": "06:30 AM", "title": "Morning Aerobics"}, {"time": "08:00 AM", "title": "5K Walkathon"}],
    "evidence_materials": [{"id": 1, "name": "track.jpg", "mime_type": "image/jpeg", "url": "/u/a.jpg"},
                           {"id": 2, "name": "agenda.pdf", "mime_type": "application/pdf", "url": "/u/b.pdf"}],
    "ai_validation_context": {"callback_url": "http://h/api/programs/29/ai-validation"},
}


def test_a_certification_package_is_read_with_or_without_the_reply_wrapper():
    for data in (CERTIFICATION, CERTIFICATION["data"]):
        p = package.parse(data)
        assert (p.target_type, p.target_id, p.number, p.id_key) == ("certification", 26, "APP-CERT-AI-TEST-001", "application_id")
        assert p.name == "Example Digital Campus" and p.address.startswith("Plot 22")
        assert p.details == (("Total Employees", "3500"), ("Gym Facility", "Yes - Onsite Fitness Center"))
        assert [e.id for e in p.photos] == [39, 40] and [e.id for e in p.documents] == [42]
        assert p.evidence[0].latitude == 17.4435 and p.evidence[0].requires_gps is True
        assert p.evidence[1].captured_at is None
        assert p.callback_url == "http://h/api/x/26/ai-validation"


def test_contact_details_are_not_kept():
    p = package.parse(CERTIFICATION)
    text = repr(p)
    assert "A Person" not in text and "9000000000" not in text and "example.com" not in text


def test_answers_keep_question_text_and_know_a_yes():
    p = package.parse(CERTIFICATION)
    assert p.answers[0].is_yes and p.answers[0].question.startswith("Are designated")
    assert not p.answers[1].is_yes and p.answers[1].selected_options == ("a", "b")


def test_a_program_package_is_read_with_kind_from_the_mime_type():
    p = package.parse(PROGRAM)
    assert (p.target_type, p.target_id, p.id_key) == ("program", 29, "program_id")
    assert p.name == "Annual Fitness Drive 2026" and p.category == "Activity Drive"
    assert p.organizer == "Example Health Alliance"
    assert p.dates == ("2026-11-20", "2026-11-22") and p.schedule[1] == ("08:00 AM", "5K Walkathon")
    assert [(e.id, e.kind) for e in p.evidence] == [(1, "photo"), (2, "document")]
    assert p.evidence[0].title == "" and p.evidence[0].requires_gps is False


@pytest.mark.parametrize("broken,reason", [
    ({"target_type": "survey"}, "target_type"),
    ({**CERTIFICATION["data"], "evidence_files": []}, "no evidence"),
    ({**CERTIFICATION["data"], "evidence_files": [{"name": "x.jpg"}]}, "needs an id"),
    ({**CERTIFICATION["data"], "evidence_files": [{"id": 1}, {"id": 1}]}, "share an id"),
    ({**CERTIFICATION["data"], "institution": {}}, "institution.name"),
    ({**CERTIFICATION["data"], "application_id": "twenty"}, "must be a number"),
    ("not an object", "JSON object"),
])
def test_a_package_that_cannot_be_audited_says_why(broken, reason):
    with pytest.raises(package.BadPackage, match=reason):
        package.parse(broken)


def test_long_text_is_cut_and_whitespace_folded():
    data = {**CERTIFICATION["data"], "institution": {**CERTIFICATION["data"]["institution"], "name": "A  \n B" + "x" * 1000}}
    p = package.parse(data)
    assert p.name.startswith("A B") and len(p.name) == package.MAX_TEXT


def test_fields_from_their_submission_shape():
    data = {**CERTIFICATION["data"], "submitted_at": "2026-11-13 08:52:44",
            "questionnaire_answers": [
                {"question_id": 1, "question": "q", "answer_type": "yes_no", "answer": "",
                 "raw": {"bool": 1, "number": None, "text": None, "option_id": None}},
                {"question_id": 2, "question": "q", "answer_type": "yes_no", "answer": "Yes",
                 "raw": {"bool": 0, "number": None, "text": None, "option_id": None}},
                {"question_id": 3, "question": "q", "answer_type": "yes_no", "answer": "Yes"}],
            "evidence_files": [{**CERTIFICATION["data"]["evidence_files"][0], "source": "Camera"}]}
    p = package.parse(data)
    assert p.submitted_at == "2026-11-13 08:52:44" and p.evidence[0].source == "camera"
    assert [a.is_yes for a in p.answers] == [True, False, True]      # raw.bool decides; the text when it is absent
    q = package.parse({**PROGRAM, "schedule": [{"start_time": "08:30", "title": "Registration"}]})
    assert q.schedule == (("08:30", "Registration"),)


def test_each_evidence_keeps_the_path_of_its_file_on_their_server():
    p = package.parse(CERTIFICATION)
    assert [e.url for e in p.evidence] == ["/uploads/x.jpg", "/uploads/y.jpg", "/uploads/z.pdf"]
    data = {**CERTIFICATION["data"], "evidence_files": [
        {"id": 1, "name": "a.pdf", "file_url": "http://localhost:5000/uploads/a.pdf"},
        {"id": 2, "name": "b.jpg", "file": {"url": "/uploads/b.jpg", "mime_type": "image/jpeg"}},
        {"id": 3, "name": "c.jpg", "path": "uploads/c.jpg"},
        {"id": 4, "name": "d.jpg"}]}
    p = package.parse(data)
    assert [e.url for e in p.evidence] == ["http://localhost:5000/uploads/a.pdf", "/uploads/b.jpg", "uploads/c.jpg", ""]
    assert p.evidence[1].mime_type == "image/jpeg"


# What IDF actually sent on 2026-11-14: flat fields, the names their submission API uses.
FLAT_PROGRAM = {
    "target_type": "program", "program_id": 10, "title": "Example Conference 2026",
    "about": "National annual conference on diabetic foot prevention and clinical workshops.",
    "venue": "Convention Centre", "location": "Hyderabad",
    "start_date": "2026-11-20", "end_date": "2026-11-22",
    "organizer_name": "Example Society", "organization_type_text": "Society",
    "agenda": [{"start_time": "09:00", "title": "Inauguration and Keynote Address"},
               {"start_time": "11:00", "title": "Wound Care Workshop"}],
    "evidence_files": [{"id": 1, "title": "Conference Backdrop", "url": "/uploads/a.jpg"}],
}


def test_a_program_in_the_flat_shape_of_their_submission_is_read_whole():
    p = package.parse(FLAT_PROGRAM)
    assert p.description.startswith("National annual conference")
    assert p.dates == ("2026-11-20", "2026-11-22")
    assert p.schedule == (("09:00", "Inauguration and Keynote Address"), ("11:00", "Wound Care Workshop"))
    assert p.address == "Convention Centre, Hyderabad" and p.organizer == "Example Society"
    assert p.evidence[0].title == "Conference Backdrop" and p.evidence[0].kind == "photo"
    # description wins over about and program_description; the ai-package names still win over the flat ones.
    p = package.parse({**FLAT_PROGRAM, "description": "D", "program_description": "P"})
    assert p.description == "D"
    p = package.parse({**FLAT_PROGRAM, "program_description": "P", "about": None})
    assert p.description == "P"
    p = package.parse({**FLAT_PROGRAM, "dates": {"start_date": "2026-12-01", "end_date": "2026-12-02"},
                       "schedule": [{"time": "08:00 AM", "title": "Walk"}]})
    assert p.dates == ("2026-12-01", "2026-12-02") and p.schedule == (("08:00 AM", "Walk"),)
    # The venue as text with the submission's city fields.
    p = package.parse({**FLAT_PROGRAM, "location": None, "city_text": "Mumbai", "region_text": "Maharashtra",
                       "country_text": "India", "pincode": "400001"})
    assert p.address == "Convention Centre, Mumbai, Maharashtra, India, 400001"


def test_a_certification_with_only_a_name_takes_its_category_from_facility_type():
    p = package.parse({"target_type": "certification", "application_id": 202,
                       "institution": {"name": "Example Institute"}, "facility_type": "Centre of Excellence",
                       "evidence_files": [{"id": 50, "title": "Training Hall", "file_name": "x.jpg"}]})
    assert p.category == "Centre of Excellence" and p.name == "Example Institute"
    assert p.evidence[0].name == "x.jpg" and p.evidence[0].title == "Training Hall"
