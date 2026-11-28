"""Their `ai-package` JSON, read into one Package.

Two shapes arrive (`target_type`): a certification application - an institution,
its questionnaire and evidence tied to requirements - and a program - an event
with a venue, dates, a schedule and free photos. Both become the same Package so
the checks read one thing.

What is deliberately NOT read: the contact person, phone and email (nothing
here needs them, so they are never stored or shown to the model), and
`prompt_context`, which is their summary for a model and is not used as one -
the prompts are this service's own, and every value from the package goes
into them as quoted data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

TARGET_TYPES = ("certification", "program")
#: Longest value taken from the package, in characters: a name, a question, an
#: address. Longer is cut, so a package cannot carry a page of text into a prompt.
MAX_TEXT = 400
MAX_EVIDENCE = 20


class BadPackage(ValueError):
    """The package cannot be audited; the message says what is missing."""


@dataclass(frozen=True)
class Evidence:
    id: int
    title: str                  # the requirement it was uploaded for; a program's photo has none
    kind: str                   # "photo" or "document"
    name: str
    latitude: float | None = None
    longitude: float | None = None
    captured_at: str | None = None
    requires_gps: bool = False
    source: str = ""            # "camera" (taken in their app) or "upload" (picked from a file)
    url: str = ""               # the file's path on IDF's server, fetched when the file is not sent
    mime_type: str = ""
    gps_accuracy_m: float | None = None     # how sure the phone was of its position, in metres
    mock_location: bool = False             # the phone reported a fake-GPS (mock) position


@dataclass(frozen=True)
class Answer:
    question_id: int
    question: str
    answer: str
    answer_type: str = ""
    section: str = ""
    selected_options: tuple[str, ...] = ()
    raw_bool: bool | None = None    # their stored answer (`raw.bool`), when the question is yes/no

    @property
    def is_yes(self) -> bool:
        if self.raw_bool is not None:
            return self.raw_bool
        return self.answer.strip().lower() in {"yes", "true", "1"}


@dataclass(frozen=True)
class Package:
    target_type: str            # certification | program
    target_id: int
    number: str
    name: str                   # the institution's name, or the program's title
    category: str               # the institution's category, or the program's type
    address: str
    description: str = ""
    organizer: str = ""         # a program's organising body (`organization.entity_name`)
    details: tuple[tuple[str, str], ...] = ()
    answers: tuple[Answer, ...] = ()
    schedule: tuple[tuple[str, str], ...] = ()
    dates: tuple[str, str] | None = None
    evidence: tuple[Evidence, ...] = ()
    callback_url: str | None = None
    submitted_at: str | None = None
    site: tuple[float, float] | None = None     # the institution's (or venue's) own coordinates
    site_source: str = ""                       # "pin" (marked on site or on a map) or "geocoded" (from the address)

    @property
    def owner(self) -> str:
        """Who the evidence belongs to: one application or program, resubmitted or not."""
        return f"{self.target_type}:{self.target_id}"

    @property
    def id_key(self) -> str:
        return "application_id" if self.target_type == "certification" else "program_id"

    @property
    def photos(self) -> list[Evidence]:
        return [e for e in self.evidence if e.kind == "photo"]

    @property
    def documents(self) -> list[Evidence]:
        return [e for e in self.evidence if e.kind == "document"]


def _text(value: Any, limit: int = MAX_TEXT) -> str:
    s = re.sub(r"\s+", " ", str(value if value is not None else "")).strip()
    return s[:limit]


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _flag(value: Any) -> bool:
    """A yes/no their JSON may send as true, 1 or "true"."""
    return value is True or value == 1 or str(value).strip().lower() in {"true", "1", "yes"}


def _site(block: Any) -> tuple[tuple[float, float] | None, str]:
    """The coordinates of an institution or venue, and how they were found
    (`location_source`: "pin" unless it says "geocoded"). None for a missing,
    zero or impossible position."""
    if not isinstance(block, dict):
        return None, ""
    lat, lon = _number(block.get("latitude")), _number(block.get("longitude"))
    if lat is None or lon is None or (lat, lon) == (0.0, 0.0) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None, ""
    source = _text(block.get("location_source"), 20).lower()
    return (lat, lon), ("geocoded" if source == "geocoded" else "pin")


def _int(value: Any, what: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise BadPackage(f"{what} must be a number, not {value!r}") from None


def _kind(item: dict[str, Any]) -> str:
    kind = _text(item.get("kind") or item.get("evidence_kind")).lower()
    if kind in ("photo", "document"):
        return kind
    mime = _text(item.get("mime_type")).lower()
    if mime.startswith("image/"):
        return "photo"
    if mime:
        return "document"
    suffix = _text(item.get("name")).lower().rsplit(".", 1)[-1]
    return "document" if suffix == "pdf" else "photo"


def _path(item: dict[str, Any]) -> str:
    """Where IDF keeps the file: `url` in their ai-package, `file_url` in their
    submission, `file.url` in their submission's reply; `path` / `file_path` too."""
    nested = item.get("file") if isinstance(item.get("file"), dict) else {}
    for value in (item.get("url"), item.get("file_url"), item.get("path"), item.get("file_path"), nested.get("url")):
        if isinstance(value, str) and value.strip():
            return value.strip()[:1000]
    return ""


def _evidence(items: Any) -> tuple[Evidence, ...]:
    if not isinstance(items, list) or not items:
        raise BadPackage("the package lists no evidence files, and there is nothing to audit without them")
    if len(items) > MAX_EVIDENCE:
        raise BadPackage(f"{len(items)} evidence files; at most {MAX_EVIDENCE} are audited in one package")
    out = []
    for item in items:
        if not isinstance(item, dict) or "id" not in item:
            raise BadPackage("every evidence file needs an id")
        out.append(Evidence(
            id=_int(item["id"], "an evidence id"),
            title=_text(item.get("requirement_title") or item.get("title")),
            kind=_kind(item),
            name=_text(item.get("name") or item.get("file_name"), 255),
            latitude=_number(item.get("latitude")),
            longitude=_number(item.get("longitude")),
            captured_at=_text(item.get("captured_at"), 40) or None,
            requires_gps=bool(item.get("requires_gps", False)),
            source=_text(item.get("source"), 20).lower(),
            url=_path(item),
            mime_type=_text(item.get("mime_type")
                            or (item["file"].get("mime_type") if isinstance(item.get("file"), dict) else ""), 60).lower(),
            gps_accuracy_m=_number(item.get("gps_accuracy_m")),
            mock_location=_flag(item.get("mock_location")),
        ))
    if len({e.id for e in out}) != len(out):
        raise BadPackage("two evidence files share an id")
    return tuple(out)


def _answers(items: Any) -> tuple[Answer, ...]:
    out = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or "question_id" not in item:
            continue
        options = item.get("selected_options") or []
        raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
        flag = raw.get("bool")
        answer = item.get("answer")
        if answer is None and options:
            answer = "; ".join(str(o) for o in options)
        out.append(Answer(
            question_id=_int(item["question_id"], "a question id"),
            question=_text(item.get("question")),
            answer=_text(answer),
            answer_type=_text(item.get("answer_type"), 40).lower(),
            section=_text(item.get("section"), 120),
            selected_options=tuple(_text(o) for o in options if isinstance(options, list)),
            raw_bool=bool(flag) if flag in (0, 1, True, False) else None,
        ))
    return tuple(out)


def _details(items: Any) -> tuple[tuple[str, str], ...]:
    if isinstance(items, dict):
        return tuple((_text(k, 120), _text(v)) for k, v in items.items() if _text(v))
    out = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and _text(item.get("value")):
            out.append((_text(item.get("label") or item.get("key"), 120), _text(item.get("value"))))
    return tuple(out)


def _place(block: dict[str, Any]) -> str:
    full = _text(block.get("full_address"))
    if full:
        return full
    parts = [_text(block.get(k)) for k in ("venue_name", "address", "city", "region", "country", "pincode")]
    return ", ".join(p for p in parts if p)


def _flat_place(data: dict[str, Any]) -> str:
    """A program's venue in the flat submission shape: `venue` as text, then the
    place it is in (`location`, or `city_text`, `region_text`, `country_text`, `pincode`)."""
    parts = [_text(data.get(k)) for k in ("venue", "location", "city_text", "region_text", "country_text", "pincode")]
    return ", ".join(dict.fromkeys(p for p in parts if p))


def parse(data: Any) -> Package:
    """A Package from their JSON - the `data` object, or the whole
    `{"success": true, "data": {...}}` reply around it."""
    if isinstance(data, dict) and isinstance(data.get("data"), dict) and "target_type" not in data:
        data = data["data"]
    if not isinstance(data, dict):
        raise BadPackage("the package must be a JSON object")
    target = _text(data.get("target_type"), 40).lower()
    if target not in TARGET_TYPES:
        raise BadPackage(f"target_type must be one of {', '.join(TARGET_TYPES)}, not {target!r}")
    context = data.get("ai_validation_context") or {}
    callback = _text(context.get("callback_url"), 500) if isinstance(context, dict) else ""

    if target == "certification":
        institution = data.get("institution") or {}
        if not isinstance(institution, dict) or not _text(institution.get("name")):
            raise BadPackage("a certification package needs institution.name")
        site, site_source = _site(institution)
        return Package(
            site=site, site_source=site_source,
            target_type=target,
            target_id=_int(data.get("application_id"), "application_id"),
            number=_text(data.get("application_no"), 60),
            name=_text(institution.get("name")),
            category=_text(institution.get("category") or data.get("category") or data.get("facility_type")),
            address=_place(institution),
            details=_details(institution.get("details")),
            answers=_answers(data.get("questionnaire_answers")),
            evidence=_evidence(data.get("evidence_files")),
            callback_url=callback or None,
            submitted_at=_text(data.get("submitted_at"), 40) or None,
        )

    # Two shapes of the same program: their ai-package (`dates{}`, `schedule`, a `venue{}`
    # object, `organization{}`) and the flat fields of their submission API, which is what
    # they sent on 2026-11-14 (`start_date`, `agenda`, `about`, `venue` as text,
    # `organizer_name`). The ai-package names win where both are present.
    venue = data.get("venue")
    dates = data.get("dates") if isinstance(data.get("dates"), dict) else {}
    organization = data.get("organization") if isinstance(data.get("organization"), dict) else {}
    if not _text(data.get("title") or data.get("program_name")):
        raise BadPackage("a program package needs a title")
    start = _text(dates.get("start_date") or data.get("start_date"), 20)
    end = _text(dates.get("end_date") or data.get("end_date"), 20)
    agenda = data.get("schedule") if isinstance(data.get("schedule"), list) else data.get("agenda")
    site, site_source = _site(venue)
    return Package(
        site=site, site_source=site_source,
        target_type=target,
        target_id=_int(data.get("program_id"), "program_id"),
        number=_text(data.get("program_no"), 60),
        name=_text(data.get("title") or data.get("program_name")),
        category=_text(data.get("program_type") or data.get("program_type_text")),
        address=_place(venue) if isinstance(venue, dict) else _flat_place(data),
        description=_text(data.get("description") or data.get("about") or data.get("program_description"), 1500),
        organizer=_text(organization.get("entity_name") or data.get("organizer_name")),
        schedule=tuple((_text(s.get("time") or s.get("start_time"), 20), _text(s.get("title")))
                       for s in (agenda if isinstance(agenda, list) else []) if isinstance(s, dict) and _text(s.get("title"))),
        dates=(start, end) if start or end else None,
        evidence=_evidence(data.get("evidence_materials") or data.get("evidence_files")),
        callback_url=callback or None,
        submitted_at=_text(data.get("submitted_at"), 40) or None,
    )
