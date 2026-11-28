"""Demo of the photo authenticity checks (AUTHENTICITY_CHECKS): five
certification applications from five kinds of institution, each with photos
the AI accepts on content, differing only in where and whose the photos are.

    1  Example Fitness Club   gym        photos taken at the gym                    -> ACCEPTED
    2  Example Public School  school     a picture picked from the gallery, no GPS  -> REJECTED
    3  Example City Hospital  hospital   photos taken 7 km from the hospital        -> REJECTED
    4  Example Tech Office    workplace  the phone reported fake GPS                -> REJECTED
    5  Example Degree College college    a photo application 1 already submitted    -> REJECTED

    .venv/Scripts/python.exe scripts/idf_authenticity_demo.py --url http://127.0.0.1:5008 [--key KEY] [--out FOLDER]

The service must run with AUTHENTICITY_CHECKS=true. The applications are sent
in order and the queue judges one at a time, so application 1's photos are
fingerprinted before application 5 is judged. Every name, place and picture is
invented. Writes each package, its files and the answer to --out.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from idf_sample import font, gps_exif  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "var" / "authenticity_demo"
EARTH_M_PER_DEG = 111_195


def offset(lat: float, lon: float, north_m: float) -> tuple[float, float]:
    return round(lat + north_m / EARTH_M_PER_DEG, 6), lon


# --- pictures: drawn differently for each application, so only #5 is a reused photo -----

# The fingerprint ignores colour, so each facility and each poster has its own layout.
POSTERS = [((100, 150, 800, 1050), (255, 140, 0), ["WALK 10,000", "STEPS A DAY", "TAKE THE STAIRS"]),
           ((60, 520, 840, 1150), (20, 120, 200), ["MOVE MORE", "SIT LESS", "STRETCH HOURLY"]),
           ((60, 60, 840, 640), (200, 30, 60), ["30 MINUTES", "OF ACTIVITY", "EVERY DAY"]),
           ((40, 60, 520, 1140), (40, 150, 90), ["CYCLE", "TO WORK", "STAY", "ACTIVE"]),
           ((380, 60, 860, 1140), (120, 40, 160), ["DANCE", "YOGA", "JOG", "PLAY"])]


def court(variant: int) -> Image.Image:
    """A sports facility, drawn: 0 badminton court, 1 basketball court, 2 running
    track, 3 tennis court seen in perspective."""
    if variant == 2:
        img = Image.new("RGB", (1400, 900), (60, 140, 70))
        d = ImageDraw.Draw(img)
        d.ellipse((100, 150, 1300, 850), fill=(180, 60, 50))
        d.ellipse((300, 280, 1100, 720), fill=(70, 160, 80))
        for r in range(1, 4):
            d.ellipse((100 + r * 50, 150 + r * 32, 1300 - r * 50, 850 - r * 32), outline="white", width=3)
        d.text((480, 40), "RUNNING TRACK", fill="white", font=font(56, True))
        return img
    if variant == 3:
        img = Image.new("RGB", (1200, 800), (150, 190, 230))
        d = ImageDraw.Draw(img)
        d.rectangle((0, 260, 1200, 800), fill=(60, 140, 80))
        d.polygon([(380, 300), (820, 300), (1100, 760), (100, 760)], fill=(40, 90, 170), outline="white", width=6)
        d.line((240, 530, 960, 530), fill="white", width=10)               # the net
        for x in range(240, 961, 40):
            d.line((x, 500, x, 530), fill=(230, 230, 230), width=2)
        d.line((600, 300, 600, 760), fill="white", width=4)
        d.text((390, 110), "TENNIS COURT", fill="white", font=font(60, True))
        return img
    if variant == 1:
        img = Image.new("RGB", (1200, 800), (170, 100, 50))
        d = ImageDraw.Draw(img)
        d.rectangle((350, 90, 850, 760), outline="white", width=8)
        d.line((350, 425, 850, 425), fill="white", width=6)
        d.ellipse((540, 365, 660, 485), outline="white", width=5)
        d.rectangle((560, 90, 640, 160), outline="white", width=5)
        d.text((360, 20), "BASKETBALL COURT", fill="white", font=font(44, True))
        return img
    img = Image.new("RGB", (1200, 800), (30, 110, 60))
    d = ImageDraw.Draw(img)
    d.rectangle((150, 100, 1050, 700), outline="white", width=8)
    d.line((600, 100, 600, 700), fill="white", width=6)
    d.line((150, 400, 1050, 400), fill="white", width=4)
    d.rectangle((560, 60, 640, 110), fill=(60, 60, 60))
    d.text((360, 20), "BADMINTON COURT 1", fill="white", font=font(44, True))
    return img


def poster(variant: int, institution: str) -> Image.Image:
    box, colour, lines = POSTERS[variant]
    img = Image.new("RGB", (900, 1200), (235, 225, 210))
    d = ImageDraw.Draw(img)
    d.rectangle(box, fill=colour, outline="black", width=8)
    for i, line in enumerate(lines):
        d.text((box[0] + 40, box[1] + 60 + i * 95), line, fill="white", font=font(60, True))
    d.text((box[0] + 40, box[3] - 70), institution.upper()[:24], fill="white", font=font(30, True))
    return img


def save(img: Image.Image, path: Path, gps: tuple[float, float] | None, taken: str) -> Path:
    if gps:
        img.save(path, "JPEG", quality=92, exif=gps_exif(gps[0], gps[1], taken))
    else:
        img.save(path, "JPEG", quality=92)
    return path


# --- the applications -------------------------------------------------------------

SCENARIOS = [
    dict(app=7001, name="Example Fitness Club", category="Gym / Fitness Center", site=(17.44350, 78.37720),
         story="Genuine: both photos taken in the app at the gym", expect="ACCEPTED"),
    dict(app=7002, name="Example Public School", category="School", site=(17.38500, 78.48670),
         story="A picture picked from the phone's gallery (e.g. downloaded from Google): no GPS in it",
         expect="REJECTED"),
    dict(app=7003, name="Example City Hospital", category="Hospital", site=(17.41000, 78.45000),
         story="Photos taken about 7 km away from the hospital", expect="REJECTED"),
    dict(app=7004, name="Example Tech Office", category="Workplace / Office", site=(17.45000, 78.38000),
         story="The phone reported a fake-GPS (mock) location", expect="REJECTED"),
    dict(app=7005, name="Example Degree College", category="College / University", site=(17.49000, 78.39000),
         story="Re-uses the court photo application 7001 submitted", expect="REJECTED"),
]


def answers() -> list[dict]:
    return [
        {"question_id": 1, "section": "Infrastructure", "question": "Are sports or fitness facilities available on the premises?",
         "answer_type": "yes_no", "answer": "Yes", "raw": {"bool": 1}},
        {"question_id": 2, "section": "Promotion", "question": "Is physical activity promotion signage displayed?",
         "answer_type": "yes_no", "answer": "Yes", "raw": {"bool": 1}},
    ]


def build(index: int, scenario: dict, folder: Path, first_court: Path | None) -> tuple[dict, dict[int, Path]]:
    """One application's package and files."""
    folder.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    taken = (now - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
    submitted = now.strftime("%Y-%m-%d %H:%M:%S")
    lat, lon = scenario["site"]
    at_site = offset(lat, lon, 25)                     # 25 m from the marked location
    where, source, mock = at_site, "camera", False
    file_gps: tuple[float, float] | None = at_site
    if index == 1:                                     # from the gallery: no GPS in the file
        source, file_gps = "upload", None
    elif index == 2:                                   # taken 7 km away
        where = file_gps = offset(lat, lon, 7000)
    elif index == 3:                                   # a fake-GPS app put the phone "at" the office
        mock = True
    eid = 100 + index * 10
    files = {eid + 1: folder / f"{eid + 1}_facility.jpg", eid + 2: folder / f"{eid + 2}_signage.jpg"}
    if index == 4 and first_court:                     # application 7001's court photo, resized and re-saved
        img = Image.open(first_court).resize((1000, 667))
        save(img, files[eid + 1], file_gps, taken)
    else:
        save(court(index), files[eid + 1], file_gps, taken)
    save(poster(index, scenario["name"]), files[eid + 2], file_gps, taken)

    def item(fid: int, rid: int, title: str, name: str) -> dict:
        return {"id": fid, "requirement_id": rid, "requirement_title": title, "kind": "photo", "name": name,
                "mime_type": "image/jpeg", "source": source, "latitude": where[0], "longitude": where[1],
                "gps_accuracy_m": 8, "mock_location": mock, "captured_at": submitted, "requires_gps": True}

    data = {
        "target_type": "certification", "application_id": scenario["app"],
        "application_no": f"APP-DEMO-{scenario['app']}", "application_type": "new",
        "institution": {"name": scenario["name"], "category": scenario["category"],
                        "full_address": f"{scenario['name']}, Hyderabad, Telangana, India",
                        "latitude": lat, "longitude": lon, "location_source": "pin"},
        "questionnaire_answers": answers(),
        "evidence_files": [item(eid + 1, 1, "Sports / Fitness Facilities", files[eid + 1].name.split("_", 1)[1]),
                           item(eid + 2, 2, "Physical Activity Promotion Signage", files[eid + 2].name.split("_", 1)[1])],
        "submitted_at": submitted,
    }
    (folder / "package.json").write_text(json.dumps({"success": True, "data": data}, indent=1), encoding="utf-8")
    return data, files


def submit(url: str, key: str | None, data: dict, files: dict[int, Path]) -> str:
    headers = {"X-API-Key": key} if key else {}
    handles = {f"file_{eid}": (path.name, path.open("rb"), "image/jpeg") for eid, path in files.items()}
    try:
        reply = requests.post(f"{url}/api/audit", headers=headers, data={"package": json.dumps(data)},
                              files=handles, timeout=60)
    finally:
        for _, handle, _ in handles.values():
            handle.close()
    reply.raise_for_status()
    return reply.json()["audit_id"]


def wait(url: str, key: str | None, audit_id: str, limit_s: int = 1800) -> dict:
    headers = {"X-API-Key": key} if key else {}
    end = time.time() + limit_s
    while time.time() < end:
        reply = requests.get(f"{url}/api/audit/{audit_id}", headers=headers, timeout=30)
        if reply.status_code != 409:
            return {"http": reply.status_code, **reply.json()}
        time.sleep(3)
    return {"http": 409, "state": "STILL RUNNING"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:5008")
    ap.add_argument("--key", default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    url = args.url.rstrip("/")

    built, first_court = [], None
    for i, scenario in enumerate(SCENARIOS):
        data, files = build(i, scenario, args.out / str(scenario["app"]), first_court)
        if i == 0:
            first_court = next(iter(files.values()))
        built.append((scenario, data, files))
    ids = [(scenario, submit(url, args.key, data, files)) for scenario, data, files in built]
    for scenario, audit_id in ids:
        print(f"submitted {scenario['app']} {scenario['name']}: {audit_id}", flush=True)

    summary = []
    for scenario, audit_id in ids:
        answer = wait(url, args.key, audit_id)
        (args.out / str(scenario["app"]) / "answer.json").write_text(json.dumps(answer, indent=1, ensure_ascii=False),
                                                                     encoding="utf-8")
        gps = next((c for c in answer.get("checks", []) if c["check_type"] == "image_location_validation"), {})
        files = [(e["id"], e["ai_status"], e["ai_note"]) for e in answer.get("evidence", [])]
        summary.append({"application": scenario["app"], "institution": f"{scenario['name']} ({scenario['category']})",
                        "scenario": scenario["story"], "expected": scenario["expect"], "audit_id": audit_id,
                        "state": answer.get("state"), "score": answer.get("score"),
                        "location_check": f"{gps.get('status')} {gps.get('score')}: {gps.get('details', '')}",
                        "files": files})
        print(f"\n{scenario['app']} {scenario['name']} ({scenario['category']}): {scenario['story']}")
        print(f"  -> {answer.get('state')} (score {answer.get('score')}), expected {scenario['expect']}")
        print(f"  location: {gps.get('status')} {gps.get('score')}: {gps.get('details', '')[:300]}")
        for eid, status, note in files:
            print(f"  file {eid}: {status}: {note[:220]}")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
