"""Make a sample IDF audit - package and files - in the exact shape IDF's
`ai-package` endpoints return (read from their test instance on 2026-11-13),
with invented values and drawn pictures, so the audit can be run and shown
without a real applicant.

    .venv/Scripts/python.exe scripts/idf_sample.py [FOLDER] [--contradict]

Writes FOLDER/certification and FOLDER/program (default
var/sample in this folder), each holding `package.json` and one file per
evidence item named `<id>_<name>`, ready for:

    .venv/Scripts/python.exe scripts/audit.py FOLDER/certification/package.json --files FOLDER/certification

or for `POST /api/audit` with `file_<id>` fields. `--contradict` makes the
certification package one a reviewer should refuse: the exercise-area photo is
a blank wall and the signage photo was taken 5 km from the declared position.

Nothing here is a real place, person or organisation. The files go outside the
repository by default and are blocked from it by .gitignore in any case.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

from PIL import ExifTags, Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT = Path(__file__).resolve().parents[1] / "var" / "sample"
WHEN = "2026-11-13 08:52:44"
LAT, LON = 17.44350, 78.37720                       # an invented point in Hyderabad
BACKEND = "http://localhost:4000"                   # as their packages say: their server speaking of itself


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in (("arialbd.ttf" if bold else "arial.ttf"), "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"):
        for folder in ("C:/Windows/Fonts", "/usr/share/fonts/truetype/dejavu"):
            try:
                return ImageFont.truetype(f"{folder}/{name}", size)
            except OSError:
                continue
    return ImageFont.load_default(size)


def gps_exif(lat: float, lon: float, taken: str) -> Image.Exif:
    exif = Image.Exif()
    gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
    for ref_tag, tag, value, refs in ((1, 2, lat, "NS"), (3, 4, lon, "EW")):
        gps[ref_tag] = refs[0] if value >= 0 else refs[1]
        v = abs(value)
        d = int(v)
        m = int((v - d) * 60)
        s = (v - d - m / 60) * 3600
        gps[tag] = (Fraction(d), Fraction(m), Fraction(round(s * 1000), 1000))
    exif.get_ifd(ExifTags.IFD.Exif)[36867] = taken.replace("-", ":")
    return exif


def save(img: Image.Image, path: Path, lat: float | None = None, lon: float | None = None) -> None:
    if lat is None:
        img.save(path, "JPEG", quality=92)
    else:
        img.save(path, "JPEG", quality=92, exif=gps_exif(lat, lon, WHEN))


# --- the pictures ----------------------------------------------------------------

def gym() -> Image.Image:
    img = Image.new("RGB", (1200, 800), (200, 200, 205))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, 1200, 300), fill=(240, 240, 245))
    for x in range(80, 1100, 220):
        d.rectangle((x, 420, x + 150, 720), fill=(40, 40, 40))
        d.rectangle((x + 20, 380, x + 130, 420), fill=(90, 90, 90))
    d.rectangle((300, 60, 900, 200), fill="white", outline="black", width=6)
    d.text((330, 90), "FITNESS CENTER", fill="black", font=font(64, True))
    return img


def poster() -> Image.Image:
    img = Image.new("RGB", (900, 1200), (235, 225, 210))
    d = ImageDraw.Draw(img)
    d.rectangle((100, 150, 800, 1050), fill=(255, 140, 0), outline="black", width=8)
    f = font(64, True)
    d.text((140, 300), "WALK 10,000", fill="white", font=f)
    d.text((140, 400), "STEPS A DAY", fill="white", font=f)
    d.text((140, 600), "TAKE THE STAIRS", fill="black", font=f)
    return img


def court() -> Image.Image:
    img = Image.new("RGB", (1200, 800), (30, 110, 60))
    d = ImageDraw.Draw(img)
    d.rectangle((150, 100, 1050, 700), outline="white", width=8)
    d.line((600, 100, 600, 700), fill="white", width=6)
    d.line((150, 400, 1050, 400), fill="white", width=4)
    d.rectangle((560, 60, 640, 110), fill=(60, 60, 60))
    d.text((360, 20), "BADMINTON COURT 1", fill="white", font=font(44, True))
    return img


def wall() -> Image.Image:
    return Image.new("RGB", (1200, 800), (185, 180, 175))


def track(title: str, organizer: str, dates: str) -> Image.Image:
    img = Image.new("RGB", (1400, 900), (60, 140, 70))
    d = ImageDraw.Draw(img)
    d.ellipse((100, 150, 1300, 850), fill=(180, 60, 50))
    d.ellipse((300, 280, 1100, 720), fill=(70, 160, 80))
    for r in range(1, 4):
        d.ellipse((100 + r * 50, 150 + r * 32, 1300 - r * 50, 850 - r * 32), outline="white", width=3)
    d.rectangle((150, 20, 1250, 140), fill="white", outline="black", width=5)
    d.text((180, 35), title.upper(), fill="black", font=font(48, True))
    d.text((180, 95), f"{organizer}  |  {dates}", fill=(40, 40, 40), font=font(30))
    return img


def standee(lines: list[str]) -> Image.Image:
    img = Image.new("RGB", (800, 1400), (245, 245, 240))
    d = ImageDraw.Draw(img)
    d.rectangle((60, 60, 740, 1340), fill=(20, 90, 160), outline="black", width=8)
    y = 140
    for line in lines:
        d.text((100, y), line, fill="white", font=font(40, True))
        y += 110
    return img


def policy_pdf(path: Path, institution: str) -> None:
    page = Image.new("RGB", (1240, 1754), "white")
    d = ImageDraw.Draw(page)
    f = font(36)
    lines = ["ACTIVE WORKPLACE POLICY 2026", institution, "",
             "1. Every employee is encouraged to take a five minute activity break each hour.",
             "2. Stairs are the preferred route between floors; signage promotes their use.",
             "3. The onsite fitness centre is open to all staff from 6 am to 9 pm.",
             "4. Weekly team sports sessions are held on the campus grounds.",
             "5. Standing desks are available on request from the facilities team.",
             "6. This policy is reviewed every year by the wellness committee."]
    y = 120
    for line in lines:
        d.text((100, y), line, fill="black", font=f)
        y += 70
    page.save(path, "PDF")


# --- the packages, field for field as their API gives them -----------------------

def _answer(qid: int, section: str, question: str, kind: str = "yes_no", answer: str = "Yes",
            raw_text: str | None = None) -> dict:
    return {"question_id": qid, "section": section, "question": question, "answer_type": kind, "answer": answer,
            "raw": {"bool": 1 if kind == "yes_no" else None, "number": None,
                    "text": raw_text if raw_text is not None else (answer if kind != "yes_no" else None),
                    "option_id": None}}


def _evidence(eid: int, rid: int, title: str, kind: str, name: str, mime: str, lat: float | None,
              lon: float | None, requires_gps: bool) -> dict:
    return {"id": eid, "requirement_id": rid, "requirement_title": title, "kind": kind, "name": name,
            "mime_type": mime, "url": f"/uploads/{name}", "source": "camera" if kind == "photo" else "upload",
            "latitude": lat, "longitude": lon, "captured_at": WHEN, "requires_gps": requires_gps}


def certification_package(contradict: bool) -> dict:
    name = "Example Digital Campus"
    files = [
        _evidence(43, 1, "Safe Exercise Area", "photo", "gym_floor.jpg", "image/jpg", LAT, LON, True),
        _evidence(44, 2, "PA Promotion Signage", "photo", "stairs_poster.jpg", "image/jpg", LAT, LON, True),
        _evidence(45, 3, "Sports / Fitness Facilities", "photo", "badminton_court.jpg", "image/jpg", LAT, LON, True),
        _evidence(46, 4, "PA Policy Document", "document", "active_workplace_policy_2026.pdf", "application/pdf",
                  None, None, False),
    ]
    sections = ("Infrastructure / Amenities", "Organizational / Policy", "Individual Level", "Scalability / Advocacy")
    answers = [
        _answer(1, sections[0], "Are designated areas available for exercise, such as converted spaces or an on-site gym within the premises?"),
        _answer(2, sections[0], "Are active design elements installed, such as standing desks, desk risers, or under-desk walkers?"),
        _answer(3, sections[0], "Is active infrastructure in place: walking trails or paths, well-lit accessible stairwells, bicycle parking?"),
        _answer(4, sections[0], "Are shower facilities and changing rooms available for active commuting and midday exercise?"),
        _answer(5, sections[1], "Are annual sports competitions or PA-related events organized for employees (and families)?"),
        _answer(6, sections[1], "Is an annual health check-up provided?"),
        _answer(7, sections[1], "Are periodic physical activity information resources, sessions or messages provided (posters, signage, emails)?"),
        _answer(8, sections[1], "Is there access to online or offline PA / fitness classes (concessions for classes or memberships)?"),
        _answer(9, sections[1], "Do any of the following activities exist at the workplace as an ACTIVE office policy? (Select all that apply)",
                "checkbox", "Ergonomic breaks, breaking sedentary time, or five-minute activity breaks; Walking meetings", "[15,17]"),
        _answer(11, sections[2], "What percentage of employees participate in a regular (at least three days a week) physical activity program?",
                "single_choice", "Fully implemented as active company policy"),
        _answer(12, sections[3], "Is staff assigned to organize ongoing or periodic advocacy to improve environmental conditions for physical activity?"),
    ]
    details = [
        {"key": "number_of_rooms", "label": "Number of Rooms", "value": "120"},
        {"key": "total_employees", "label": "Total Employees", "value": "3500"},
        {"key": "distance_from_main_road_km", "label": "Distance from Main Road (km)", "value": "0.5"},
        {"key": "company_type", "label": "Company / Industry Type", "value": "Information Technology"},
        {"key": "parking_availability", "label": "Parking Availability", "value": "Yes"},
        {"key": "floors_in_building", "label": "Number of Floors", "value": "6"},
        {"key": "has_cafeteria", "label": "Cafeteria / Canteen Available", "value": "Yes"},
        {"key": "has_gym", "label": "On-site Gym / Fitness Room", "value": "Yes - Onsite Fitness Center"},
        {"key": "has_standing_desks", "label": "Standing Desks Available", "value": "Yes"},
        {"key": "annual_health_budget_inr", "label": "Annual Health & Wellness Budget", "value": "2500000"},
        {"key": "wellness_coordinator_name", "label": "Wellness Coordinator Name", "value": "Wellness Team"},
    ]
    if contradict:
        files[1]["latitude"], files[1]["longitude"] = LAT + 0.045, LON          # ~5 km north of where the photo says
    return {"success": True, "message": "OK", "data": {
        "target_type": "certification", "application_id": 9001, "application_no": "APP-CERT-SAMPLE-0001",
        "application_type": "new",
        "institution": {
            "name": name, "category_id": 1, "category": "Workplace / Worksite / Office / Hospital",
            "address": "Plot 22, Phase 2, Example Tech Park", "country": "India", "region": "Telangana",
            "city": "Hyderabad", "pincode": "500081",
            "full_address": "Plot 22, Phase 2, Example Tech Park, Hyderabad, Telangana, India, 500081",
            "contact_person": "Sample Person", "contact_email": "sample@example.com", "contact_phone": "+91 0000000000",
            "details": details,
        },
        "questionnaire_answers": answers,
        "evidence_files": files,
        "submitted_at": WHEN,
        "ai_validation_context": {
            "prompt_context": f'Certification Application for "{name}" (Category: "Workplace / Worksite / Office / Hospital"). '
                              "Total Questions: 11. Evidence Files Attached: 4.",
            "required_checks": [
                {"check_type": "evidence_review", "description": "Check that evidence files meet requirements (e.g. safe exercise area, fitness facility)."},
                {"check_type": "image_location_validation", "description": "Validate that image metadata and location correspond with registered institution."},
                {"check_type": "detail_evidence_match", "description": "Verify that evidence matches the institution details and questionnaire answers."},
                {"check_type": "overall", "description": "Final recommendation: approved, rejected, or changes_required."},
            ],
            "callback_url": f"{BACKEND}/api/certification/applications/9001/ai-validation",
            "callback_method": "POST",
            "callback_headers": {"Content-Type": "application/json", "X-API-Key": "<their key>"},
        },
    }}


def program_package() -> dict:
    title = "Annual Corporate Fitness Drive 2026"
    return {"success": True, "message": "OK", "data": {
        "target_type": "program", "program_id": 9002, "program_no": "PRG-SAMPLE-2026-0001",
        "program_name": title, "title": title, "program_type": "Activity Drive", "program_type_id": 1,
        "description": "A 3-day workplace wellness initiative with track fitness drills, active posture clinics, "
                       "and sports challenges to encourage daily physical activity.",
        "organization": {"entity_name": "Example Health Alliance", "organization_type": "Institution"},
        "venue": {"venue_name": "Example Stadium Athletics Complex", "city": "Hyderabad", "region": "Telangana",
                  "country": "India", "pincode": "500032",
                  "full_address": "Example Stadium Athletics Complex, Old Highway Road, Hyderabad, Telangana, India, 500032"},
        "dates": {"start_date": "2026-11-20", "end_date": "2026-11-22"},
        "schedule": [
            {"time": "06:30 AM", "start_time": "06:30", "title": "Morning Aerobics & Group Stretching on Grounds"},
            {"time": "08:00 AM", "start_time": "08:00", "title": "5K Walkathon & Cycling Sprint"},
            {"time": "11:00 AM", "start_time": "11:00", "title": "Desk Ergonomics & Active Posture Masterclass"},
            {"time": "04:00 PM", "start_time": "16:00", "title": "Inter-Corporate Badminton & Table Tennis Tournament"},
        ],
        "banner_image_url": "/uploads/evidence/stadium_track.jpg",
        "evidence_materials": [
            {"id": 1, "name": "stadium_track.jpg", "mime_type": "image/jpeg", "url": "/uploads/evidence/stadium_track.jpg"},
            {"id": 2, "name": "fitness_standee.jpg", "mime_type": "image/jpeg", "url": "/uploads/evidence/fitness_standee.jpg"},
        ],
        "submitted_at": WHEN,
        "ai_validation_context": {
            "prompt_context": f'Program: "{title}" (Activity Drive). Dates: 2026-11-20 to 2026-11-22.',
            "required_checks": [
                {"check_type": "supporting_materials", "description": "Verify if uploaded images/materials match the context in the form (e.g. playground, sports court, classroom)."},
                {"check_type": "content_review", "description": "Verify that the text title and description represent a valid, active program."},
                {"check_type": "program_details_match", "description": "Verify that the venue, dates, and agenda are realistic and logically consistent."},
                {"check_type": "overall", "description": "Final recommendation: status must be approved, rejected, or changes_required."},
            ],
            "callback_url": f"{BACKEND}/api/programs/9002/ai-validation",
            "callback_method": "POST",
            "callback_headers": {"Content-Type": "application/json", "X-API-Key": "<their key>"},
        },
    }}


def write(folder: Path, contradict: bool = False) -> tuple[Path, Path]:
    cert = folder / "certification"
    prog = folder / "program"
    cert.mkdir(parents=True, exist_ok=True)
    prog.mkdir(parents=True, exist_ok=True)

    c = certification_package(contradict)
    (cert / "package.json").write_text(json.dumps(c, indent=1), encoding="utf-8")
    save(wall() if contradict else gym(), cert / "43_gym_floor.jpg", LAT, LON)
    save(poster(), cert / "44_stairs_poster.jpg", LAT, LON)
    save(court(), cert / "45_badminton_court.jpg", LAT, LON)
    policy_pdf(cert / "46_active_workplace_policy_2026.pdf", c["data"]["institution"]["name"])

    p = program_package()
    (prog / "package.json").write_text(json.dumps(p, indent=1), encoding="utf-8")
    d = p["data"]
    save(track(d["title"], d["organization"]["entity_name"], "20-22 NOV 2026"), prog / "1_stadium_track.jpg")
    save(standee(["FITNESS DRIVE 2026", "06:30 AEROBICS", "08:00 5K WALKATHON", "11:00 POSTURE CLASS",
                  "16:00 BADMINTON", "EXAMPLE STADIUM"]), prog / "2_fitness_standee.jpg")
    return cert, prog


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", type=Path, default=DEFAULT)
    ap.add_argument("--contradict", action="store_true", help="a certification package a reviewer should refuse")
    args = ap.parse_args()
    cert, prog = write(args.folder, args.contradict)
    for folder in (cert, prog):
        print(folder)
        for f in sorted(folder.iterdir()):
            print(f"  {f.name}  {f.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
