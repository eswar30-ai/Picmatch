"""Photo authenticity (AUTHENTICITY_CHECKS): the photo taken at the institution's
own coordinates, not a fake-GPS one, and not a photo another application
already submitted."""

from dataclasses import replace

import pytest
from PIL import Image, ImageDraw

from idf import audit, fingerprints, location, package
from test_idf_checks import read
from test_idf_package import CERTIFICATION

SITE = (17.44350, 78.37720)


def with_site(**institution):
    data = {**CERTIFICATION["data"],
            "institution": {**CERTIFICATION["data"]["institution"], "latitude": SITE[0], "longitude": SITE[1],
                            **institution}}
    return package.parse(data)


def photo(eid=39, **fields):
    base = package.parse(CERTIFICATION).evidence[0]
    return replace(base, id=eid, **fields)


def north(metres):
    """A point `metres` north of SITE."""
    return SITE[0] + metres / 111_195, SITE[1]


# --- the package --------------------------------------------------------------

def test_the_institutions_coordinates_and_the_photos_gps_details_are_read():
    p = with_site(location_source="geocoded")
    assert p.site == SITE and p.site_source == "geocoded" and p.owner == "certification:26"
    assert with_site().site_source == "pin"
    data = {**CERTIFICATION["data"], "evidence_files": [
        {**CERTIFICATION["data"]["evidence_files"][0], "gps_accuracy_m": "12", "mock_location": "true"}]}
    e = package.parse(data).evidence[0]
    assert e.gps_accuracy_m == 12 and e.mock_location is True


@pytest.mark.parametrize("lat,lon", [(None, None), (0, 0), (95, 10), ("x", 1)])
def test_a_missing_or_impossible_position_is_no_site(lat, lon):
    assert with_site(latitude=lat, longitude=lon).site is None


# --- taken at the institution -------------------------------------------------

@pytest.mark.parametrize("metres,status", [(50, "approved"), (800, "changes_required"), (5000, "rejected")])
def test_the_files_gps_against_the_marked_location(metres, status):
    lat, lon = north(metres)
    out = location.against_site(photo(), {"latitude": lat, "longitude": lon}, SITE, "pin")
    assert out["status"] == status and abs(out["distance_m"] - metres) <= 2


def test_a_geocoded_location_is_rough_so_far_is_a_question_not_a_rejection():
    lat, lon = north(5000)
    out = location.against_site(photo(), {"latitude": lat, "longitude": lon}, SITE, "geocoded")
    assert out["status"] == "changes_required" and "approximate" in out["details"]
    lat, lon = north(2000)
    assert location.against_site(photo(), {"latitude": lat, "longitude": lon}, SITE, "geocoded")["status"] == "approved"


def test_a_fake_gps_photo_is_rejected_wherever_it_claims_to_be():
    out = location.against_site(photo(mock_location=True), {"latitude": SITE[0], "longitude": SITE[1]}, SITE)
    assert out["status"] == "rejected" and "fake-GPS" in out["details"]


def test_a_camera_photo_is_placed_by_the_apps_record_and_a_gallery_photo_only_by_its_file():
    lat, lon = north(40)
    assert location.against_site(photo(source="camera", latitude=lat, longitude=lon), {}, SITE)["status"] == "approved"
    out = location.against_site(photo(source="upload", latitude=lat, longitude=lon), {}, SITE)
    assert out["status"] == "rejected" and "gallery" in out["details"]


def test_a_poorly_placed_phone_does_not_approve():
    out = location.against_site(photo(gps_accuracy_m=900), {"latitude": SITE[0], "longitude": SITE[1]}, SITE)
    assert out["status"] == "changes_required" and "900 m" in out["details"]


def test_the_site_leads_and_photos_that_disagree_with_each_other_only_make_it_worse():
    lat, lon = north(20)
    e = photo(latitude=lat, longitude=lon)
    far = north(6000)
    at_site = location.check(e, {"latitude": lat, "longitude": lon}, site=SITE, site_source="pin")
    assert at_site["status"] == "approved" and "marked location" in at_site["details"]
    assert at_site["site"]["distance_m"] <= 25
    # The file's GPS 6 km from where the app says, and from the institution: rejected.
    moved = location.check(e, {"latitude": far[0], "longitude": far[1]}, site=SITE, site_source="pin")
    assert moved["status"] == "rejected"
    assert location.check(e, {"latitude": lat, "longitude": lon})["status"] == "approved"   # no site: as before


# --- the same photo in another application ------------------------------------

def picture(path, seed, size=(640, 480)):
    img = Image.new("RGB", size, (30 + seed * 40 % 200, 120, 200 - seed * 30 % 150))
    draw = ImageDraw.Draw(img)
    for i in range(6):
        draw.rectangle([40 + i * 90, 60 + (i * seed * 37) % 200, 100 + i * 90, 400], fill=(250 - i * 30, 40 * i % 255, 90))
    img.save(path, "JPEG", quality=90)
    return path


@pytest.fixture
def index(tmp_path, monkeypatch):
    monkeypatch.setattr(fingerprints, "index_file", lambda: tmp_path / "fingerprints.jsonl")
    return tmp_path


def test_a_resized_copy_has_the_same_fingerprint_and_another_picture_does_not(tmp_path):
    original = picture(tmp_path / "a.jpg", 1)
    Image.open(original).resize((320, 240)).save(tmp_path / "small.jpg", "JPEG", quality=60)
    other = picture(tmp_path / "b.jpg", 4)
    a, small, b = (fingerprints.dhash(p) for p in (original, tmp_path / "small.jpg", other))
    assert fingerprints.distance(a, small) <= fingerprints.MATCH_BITS < fingerprints.distance(a, b)
    (tmp_path / "x.pdf").write_bytes(b"%PDF-1.4")
    assert fingerprints.dhash(tmp_path / "x.pdf") is None


def test_a_photo_is_reused_only_when_another_owner_submitted_it(index):
    gym = picture(index / "gym.jpg", 1)
    assert fingerprints.reused("certification:5", {43: gym}) == {}
    assert fingerprints.remember("certification:5", "1795_aaaa0000", {43: gym}) == 1
    assert fingerprints.reused("certification:5", {43: gym}) == {}         # its own application, resubmitted
    found = fingerprints.reused("certification:9", {7: gym})
    assert found[7]["owner"] == "certification:5" and found[7]["evidence"] == 43 and found[7]["bits"] == 0
    assert fingerprints.describe(found[7]).startswith("certification 5 (evidence 43, audit 1795_aaaa0000")


def test_the_first_to_submit_a_photo_owns_it(index):
    gym = picture(index / "gym.jpg", 1)
    fingerprints.remember("certification:5", "1795_aaaa0000", {43: gym})      # 5 submits it first
    assert fingerprints.reused("certification:9", {7: gym})                    # 9 copies it: flagged
    fingerprints.remember("certification:9", "1795_bbbb0000", {7: gym})
    assert fingerprints.reused("certification:5", {43: gym}) == {}             # 5 again: still its own
    assert fingerprints.reused("certification:9", {7: gym})[7]["owner"] == "certification:5"   # 9 again: still flagged
    found = fingerprints.reused("certification:12", {1: gym})                  # a third: points at the original
    assert found[1]["owner"] == "certification:5"


def files_for(tmp_path):
    files = {39: picture(tmp_path / "gym.jpg", 1), 40: picture(tmp_path / "posters.jpg", 2),
             42: tmp_path / "policy.pdf"}
    files[42].write_bytes(b"%PDF-1.4")
    return files


def all_shown(path, claim):
    return {**read("yes", 0.95), "kind": "document"} if path.suffix == ".pdf" else read("yes", 0.95)


@pytest.mark.skip(reason="the authenticity checks are paused in audit.py until IDF approves them")
def test_with_the_checks_on_a_reused_photo_rejects_the_application(index, monkeypatch):
    monkeypatch.setenv("AUTHENTICITY_CHECKS", "true")
    files = files_for(index)
    fingerprints.remember("certification:5", "1795_aaaa0000", {43: files[39]})
    result = audit.run(with_site(), files, read_photo=all_shown, read_document=all_shown)
    gym = next(e for e in result["evidence"] if e["id"] == 39)
    assert gym["ai_status"] == "rejected" and gym["ai_note"].startswith("The same photo was already submitted by certification 5")
    assert gym["reused_from"]["owner"] == "certification:5"
    assert result["status"] == "rejected" and "already submitted by certification 5" in result["checks"][0]["details"]


@pytest.mark.skip(reason="the authenticity checks are paused in audit.py until IDF approves them")
def test_with_the_checks_on_a_photo_far_from_the_institution_rejects(index, monkeypatch):
    monkeypatch.setenv("AUTHENTICITY_CHECKS", "true")
    lat, lon = north(8000)
    p = with_site(latitude=lat, longitude=lon)          # the institution is 8 km from where the photos were taken
    p = replace(p, evidence=tuple(replace(e, source="camera") for e in p.evidence))
    result = audit.run(p, files_for(index), read_photo=all_shown, read_document=all_shown)
    gps = next(c for c in result["checks"] if c["check_type"] == "image_location_validation")
    assert gps["status"] == "rejected" and "not taken at the institution" in gps["details"]
    assert result["status"] == "rejected"


def test_with_the_checks_off_the_site_and_the_index_are_not_used(index, monkeypatch):
    monkeypatch.delenv("AUTHENTICITY_CHECKS", raising=False)
    files = files_for(index)
    fingerprints.remember("certification:5", "1795_aaaa0000", {43: files[39]})
    lat, lon = north(8000)
    result = audit.run(with_site(latitude=lat, longitude=lon), files, read_photo=all_shown, read_document=all_shown)
    assert all(e["reused_from"] is None for e in result["evidence"])
    assert "institution" not in next(c for c in result["checks"] if c["check_type"] == "image_location_validation")["details"]
