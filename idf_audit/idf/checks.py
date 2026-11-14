"""The checks IDF's backend expects, from the readings - code only, so the
same readings always give the same verdict.

    certification   evidence_review, image_location_validation, detail_evidence_match, overall
    program         supporting_materials, content_review, program_details_match, overall

Each check is `{"check_type", "status", "score", "details"}`: score 0 to 100;
details one or two sentences a reviewer can act on. The overall status is the
worst of the three, never better: one contradicted claim is not averaged away
by two good photos.

The checks themselves work in three statuses - `approved`, `changes_required`
(cannot be told either way), `rejected` - but IDF wants only two (decision of
2026-11-14). `decided` turns each `changes_required` into `approved` at
APPROVE_SCORE or above and `rejected` below it; audit.run applies it to every
status in the result, last, so what the checks found is unchanged and the
details still say what was not confirmed.
"""

from __future__ import annotations

import re
from typing import Any

from idf import settings
from idf.helpers import _on_page
from idf.evidence import text_of
from idf.package import Answer, Evidence, Package

STATUSES = ("approved", "changes_required", "rejected")
#: A verdict the model gave at less than this confidence is a question, not an answer.
SURE = 0.70
#: Question words too common to tie a question to a requirement ("area", "available").
_STOP = {"that", "this", "with", "have", "does", "your", "from", "which", "there", "they", "them",
         "than", "been", "into", "such", "also", "what", "when", "where", "will", "would", "area",
         "areas", "available", "following", "exist", "select", "options", "option", "below", "above",
         "workplace", "office", "campus", "institution", "school", "hospital", "please", "provide",
         "about", "other", "more", "least", "between", "within", "some", "every", "each", "both"}


def worst(statuses: list[str]) -> str:
    present = set(statuses)
    for status in ("rejected", "changes_required"):
        if status in present:
            return status
    return "approved"


#: What cannot be told either way is approved at this score or above, rejected below it.
APPROVE_SCORE = settings.number("APPROVE_SCORE", 80)


def decide(status: str, score: float) -> str:
    """`approved` or `rejected`: a `changes_required` is decided by its score."""
    if status != "changes_required":
        return status
    return "approved" if score >= APPROVE_SCORE else "rejected"


def decided(check: dict[str, Any]) -> dict[str, Any]:
    """A check with its `changes_required` decided by score, saying so in its details."""
    if check["status"] != "changes_required":
        return check
    status = decide(check["status"], check["score"])
    side = "at or above" if status == "approved" else "below"
    return _check(check["check_type"], status, check["score"],
                  f"{check['details'].rstrip('.')}. Not confirmed either way; the score {check['score']} "
                  f"is {side} {APPROVE_SCORE}, so {status}")


def _check(check_type: str, status: str, score: float, details: str) -> dict[str, Any]:
    return {"check_type": check_type, "status": status, "score": int(round(max(0.0, min(100.0, score)))),
            "details": details.strip()[:600]}


def evidence_status(read: dict[str, Any]) -> tuple[str, int]:
    """One piece of evidence's own status and score from the model's verdict.
    `yes` at SURE or better is approved; `no` at SURE or better rejected;
    everything else - unsure, an unparsed answer, a verdict given without
    confidence - is for a person."""
    shows, confidence = read.get("shows"), float(read.get("confidence") or 0.0)
    if not read.get("parsed", True):
        return "changes_required", 20
    if shows == "yes":
        return ("approved", round(confidence * 100)) if confidence >= SURE else ("changes_required", round(confidence * 100))
    if shows == "no":
        return ("rejected", round((1 - confidence) * 100)) if confidence >= SURE else ("changes_required", 35)
    return "changes_required", 40


def label(e: Evidence) -> str:
    return e.title or e.name or f"file {e.id}"


def evidence_review(check_type: str, package: Package, reads: dict[int, dict[str, Any]],
                    what: str, same: dict[int, list[int]] | None = None) -> dict[str, Any]:
    """Does each file show what it was uploaded for? `what` names the claim in
    the details ("its requirement", "the program"). `same` maps an evidence id to
    the others whose file is identical: one picture is not evidence for two things."""
    verdicts = [(e, *evidence_status(reads[e.id])) for e in package.evidence if e.id in reads]
    if not verdicts:
        return _check(check_type, "changes_required", 0, "no evidence file was read")
    shown = [e for e, s, _ in verdicts if s == "approved"]
    contradicted = [(e, reads[e.id]) for e, s, _ in verdicts if s == "rejected"]
    unsure = [(e, reads[e.id]) for e, s, _ in verdicts if s == "changes_required"]
    parts = [f"{len(shown)} of {len(verdicts)} files show {what}"
             + (f" ({', '.join(label(e) for e in shown)})" if shown else "")]
    for e, read in contradicted:
        parts.append(f"{label(e)} does not: {read.get('reason') or 'the model saw something else'}")
    for e, read in unsure:
        parts.append(f"{label(e)} cannot be told: {read.get('reason') or 'no clear answer'}")
    score = sum(sc for _, _, sc in verdicts) / len(verdicts)
    status = worst([s for _, s, _ in verdicts])
    groups = {tuple(sorted([eid, *others])) for eid, others in (same or {}).items()}
    by_id = {e.id: e for e in package.evidence}
    for group in sorted(groups):
        parts.append("the same file was uploaded for " + " and ".join(label(by_id[i]) for i in group if i in by_id))
    if groups:
        status, score = worst([status, "changes_required"]), min(score, 50)
    return _check(check_type, status, score, "; ".join(parts))


def location_validation(package: Package, findings: dict[int, dict[str, Any] | None]) -> dict[str, Any]:
    """GPS counts only when IDF marks a photo `requires_gps`, its way of making it
    mandatory (the owner, 2026-11-14). Otherwise the check is still returned -
    IDF expects all four - but says it was not checked and carries
    `counted: false`, and `overall` leaves it out of the status and the score."""
    kind = "image_location_validation"
    if not any(e.requires_gps for e in package.photos):
        return {**_check(kind, "approved", 100,
                         "not checked: no photo in this package is marked requires_gps, so GPS is not judged "
                         "and does not count towards the overall result"), "counted": False}
    found = [(e, findings[e.id]) for e in package.photos if findings.get(e.id)]
    if not found:
        return {**_check(kind, "changes_required", 50, "a photo requires GPS but none could be checked"),
                "counted": True}
    parts = [f"{label(e)}: {f['details']}" for e, f in found]
    score = sum(f["score"] for _, f in found) / len(found)
    return {**_check(kind, worst([f["status"] for _, f in found]), score, "; ".join(parts)), "counted": True}


def _content_words(text: str) -> set[str]:
    words = {w.lower().rstrip("s") for w in re.findall(r"[A-Za-z]{4,}", text)}
    return {w for w in words if w and w not in _STOP and len(w) >= 3}


def related(answer: Answer, evidence: Evidence) -> bool:
    """A question and a requirement are about the same thing when they share
    a content word: "designated areas for exercise" and "Safe Exercise Area"."""
    return bool(_content_words(answer.question) & _content_words(evidence.title))


def claims(package: Package, reads: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Each "Yes" in the questionnaire, against the evidence about the same
    thing: supported, contradicted, unsure, or unchecked (no related file)."""
    out = []
    for answer in package.answers:
        if answer.answer_type not in ("", "yes_no", "boolean", "bool") or not answer.is_yes:
            continue
        about = [e for e in package.evidence if e.id in reads and related(answer, e)]
        statuses = [evidence_status(reads[e.id])[0] for e in about]
        if not about:
            outcome = "unchecked"
        elif "approved" in statuses:
            outcome = "supported"
        elif all(s == "rejected" for s in statuses):
            outcome = "contradicted"
        else:
            outcome = "unsure"
        out.append({"question_id": answer.question_id, "question": answer.question[:160],
                    "evidence": [e.id for e in about], "outcome": outcome})
    return out


def name_on_evidence(name: str, reads: dict[int, dict[str, Any]]) -> list[int]:
    """The evidence files on which the institution's or program's name is printed."""
    return [eid for eid, read in reads.items() if name and _on_page(name, text_of(read))
            and any(w in text_of(read).upper() for w in re.findall(r"[A-Za-z]{3,}", name.upper()))]


def detail_evidence_match(package: Package, reads: dict[int, dict[str, Any]]) -> dict[str, Any]:
    kind = "detail_evidence_match"
    checked = claims(package, reads)
    by = {o: [c for c in checked if c["outcome"] == o] for o in ("supported", "contradicted", "unsure", "unchecked")}
    named = name_on_evidence(package.name, reads)
    parts = []
    if named:
        parts.append(f"the name \"{package.name}\" is printed on file(s) {', '.join(str(i) for i in named)}")
    else:
        parts.append("the name is not printed on any file (not a fault: a gym photo need not show it)")
    judged = len(by["supported"]) + len(by["contradicted"]) + len(by["unsure"])
    if checked:
        parts.append(f"{len(by['supported'])} of {len(checked)} \"Yes\" answers are backed by a file that shows it")
        for c in by["contradicted"]:
            parts.append(f"Q{c['question_id']} \"{c['question'][:80]}\" is contradicted by file(s) "
                         + ", ".join(str(i) for i in c["evidence"]))
        if by["unsure"]:
            parts.append("unclear from the evidence: " + ", ".join(f"Q{c['question_id']}" for c in by["unsure"]))
        if by["unchecked"]:
            parts.append("no file is about: " + ", ".join(f"Q{c['question_id']}" for c in by["unchecked"]))
    else:
        parts.append("no \"Yes\" answer in the questionnaire to check against the evidence")
    if by["contradicted"]:
        status = "rejected"
    elif judged and not by["unsure"]:
        status = "approved"
    else:
        status = "changes_required"
    score = (100 * len(by["supported"]) / judged) if judged else 50
    if named and judged == 0:
        score = 70
    return _check(kind, status, score, "; ".join(parts))


def content_review(package: Package) -> dict[str, Any]:
    """A program's own description, dates and agenda, by rule: are the dates
    in order and the agenda there? Not judged for substance - that is the
    reviewer's, and saying so is better than a model's guess."""
    problems = []
    start = end = None
    if package.dates:
        from idf.location import parse_time

        start, end = (parse_time(d + " 00:00") for d in package.dates)
        if not (start and end):
            problems.append("a date could not be read")
        elif end < start:
            problems.append("the end date is before the start date")
    else:
        problems.append("no dates")
    if len(package.description) < 20:
        problems.append("no description to speak of")
    if not package.schedule:
        problems.append("no agenda")
    days = (end - start).days + 1 if start and end and end >= start else None
    summary = (f"{len(package.schedule)} agenda item(s), "
               + (f"{days} day(s) from {package.dates[0]} to {package.dates[1]}, " if days else "")
               + f"description of {len(package.description.split())} words; checked by rule, not for substance")
    if problems:
        return _check("content_review", "changes_required", 40, "; ".join(problems) + ". " + summary)
    return _check("content_review", "approved", 85, summary)


def _dates_on(text: str, dates: tuple[str, str]) -> bool:
    """A program's dates are on a file when its year is printed with the day
    or the month of either date: "20-22 NOV 2026", "20/11/2026", "Nov 2026"."""
    import calendar

    low = text.lower()
    for d in dates:
        m = re.match(r"(\d{4})-(\d{2})-(\d{2})", d or "")
        if not m or m.group(1) not in text:
            continue
        month = calendar.month_abbr[int(m.group(2))].lower()
        if month in low or re.search(rf"(?<!\d)0?{int(m.group(3))}(?!\d)", text):
            return True
    return False


def program_details_match(package: Package, reads: dict[int, dict[str, Any]]) -> dict[str, Any]:
    kind = "program_details_match"
    named = name_on_evidence(package.name, reads)
    venue_words = package.address.split(",")[0] if package.address else ""
    at_venue = [eid for eid, read in reads.items() if venue_words and _on_page(venue_words, text_of(read))
                and _content_words(venue_words) & _content_words(text_of(read))]
    dated = [eid for eid, read in reads.items() if package.dates and _dates_on(text_of(read), package.dates)]
    by_organizer = name_on_evidence(package.organizer, reads)
    parts = []
    parts.append(f"the title is printed on file(s) {', '.join(map(str, named))}" if named
                 else "the title is not printed on any file")
    parts.append(f"the venue is named on file(s) {', '.join(map(str, at_venue))}" if at_venue
                 else "the venue is not named on any file")
    if package.dates:
        parts.append(f"the dates appear on file(s) {', '.join(map(str, dated))}" if dated
                     else "the dates do not appear on any file")
    if package.organizer:
        parts.append(f"the organiser is named on file(s) {', '.join(map(str, by_organizer))}" if by_organizer
                     else "the organiser is not named on any file")
    hits = sum(1 for x in (named, at_venue, dated, by_organizer) if x)
    if hits >= 2:
        return _check(kind, "approved", 70 + 10 * hits, "; ".join(parts))
    return _check(kind, "changes_required", 40 + 15 * hits,
                  "; ".join(parts) + " - a photo need not carry text, so a person should look")


def overall(checks: list[dict[str, Any]]) -> dict[str, Any]:
    checks = [c for c in checks if c.get("counted", True)]
    status = worst([c["status"] for c in checks])
    score = sum(c["score"] for c in checks) / len(checks) if checks else 0
    if status == "approved":
        details = "Every check passed; recommended for approval at manual review"
    elif status == "rejected":
        failed = [c["check_type"] for c in checks if c["status"] == "rejected"]
        details = "The evidence contradicts the application (" + ", ".join(failed) + "); recommended for rejection, a person to confirm"
        score = min(score, 30)
    else:
        pending = [c["check_type"] for c in checks if c["status"] == "changes_required"]
        details = "Not confirmed: " + ", ".join(pending)
    return _check("overall", status, score, details)


def certification_checks(package: Package, reads: dict[int, dict[str, Any]],
                         findings: dict[int, dict[str, Any] | None],
                         same: dict[int, list[int]] | None = None) -> list[dict[str, Any]]:
    checks = [evidence_review("evidence_review", package, reads, "their requirement", same),
              location_validation(package, findings),
              detail_evidence_match(package, reads)]
    return checks + [overall(checks)]


def program_checks(package: Package, reads: dict[int, dict[str, Any]],
                   same: dict[int, list[int]] | None = None) -> list[dict[str, Any]]:
    checks = [evidence_review("supporting_materials", package, reads, "the program", same),
              content_review(package),
              program_details_match(package, reads)]
    return checks + [overall(checks)]
