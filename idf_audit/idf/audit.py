"""One audit: a Package and its files in, the verdict out.

`run` is the whole judgement and is pure apart from the two readers it is
given: the API's job calls it with the real ones, tests with fakes. A reader that raises `ModelDidNotAnswer` fails the audit: no verdict is
built from a photo the model never judged. A file that cannot be read - a
broken image, a damaged PDF - fails only itself: it is judged "could not be
read", and the rest of the audit goes on.
"""

from __future__ import annotations

import hashlib
import sys
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from idf import checks, evidence, location
from idf.ocr import ModelDidNotAnswer
from idf.package import Evidence, Package

Reader = Callable[[Path, str], dict[str, Any]]

#: Keys of the stored result that go back to IDF: their callback body, no more.
CALLBACK_KEYS = ("status", "score", "comments", "checks")
#: Longest `ai_note`, in characters: one line for the reviewer beside the file.
NOTE_MAX = 300


def claim_for(package: Package, item: Evidence) -> str:
    """What this file is meant to show, in words the model is asked to verify."""
    if package.target_type == "certification":
        what = f'the requirement "{item.title}"' if item.title else "the application"
        return (f"This {item.kind} was uploaded as evidence for {what} of a certification application "
                f'by the institution "{package.name}" (category: {package.category or "not given"}), '
                f"located at {package.address or 'an address not given'}."
                + (" It should show a safe place where people can exercise." if "exercise" in item.title.lower() else "")
                + (" It should show signage, posters or boards promoting physical activity." if "signage" in item.title.lower() else "")
                + (" It should be a written policy promoting physical activity at this institution." if item.kind == "document" and "policy" in item.title.lower() else ""))
    when = f" from {package.dates[0]} to {package.dates[1]}" if package.dates else ""
    who = f' organised by "{package.organizer}"' if package.organizer else ""
    return (f'This {item.kind} was uploaded as supporting material for the program "{package.name}" '
            f'({package.category or "program"}){who}{when} at {package.address or "a venue not given"}. '
            f"The program describes itself as: {package.description or 'no description given'}. "
            f"It should show the venue, the activities or the promotion of this program.")


def comments_for(package: Package, result_checks: list[dict[str, Any]], reads: dict[int, dict[str, Any]]) -> str:
    """One line for the reviewer's list, from the checks before `checks.decided`:
    what the evidence contradicts, and what it could not confirm either way."""
    final = checks.decided(result_checks[-1])
    shown = sum(1 for r in reads.values() if checks.evidence_status(r)[0] == "approved")
    head = f"{shown} of {len(reads)} evidence file(s) show what they were uploaded for"
    rejected = [c["check_type"] for c in result_checks[:-1] if c["status"] == "rejected"]
    pending = [c["check_type"] for c in result_checks[:-1] if c["status"] == "changes_required"]
    if not rejected and not pending:
        return f"{head}; every check passed."
    parts = ([f"contradicted on {', '.join(rejected)}"] if rejected else []) + \
            ([f"not confirmed on {', '.join(pending)}"] if pending else [])
    side = f"{checks.APPROVE_SCORE} or above" if final["status"] == "approved" else f"below {checks.APPROVE_SCORE}"
    by_score = f" (score {final['score']}, {side})" if result_checks[-1]["status"] == "changes_required" else ""
    return f"{head}; {'; '.join(parts)}: {final['status']}{by_score}."


def same_files(files: dict[int, Path]) -> dict[int, list[int]]:
    """Evidence ids whose files are byte for byte the same, each against the
    others: `{43: [45], 45: [43]}`. One photo uploaded for two requirements
    cannot be evidence for both."""
    by_hash: dict[str, list[int]] = {}
    for eid, path in files.items():
        by_hash.setdefault(hashlib.sha256(path.read_bytes()).hexdigest(), []).append(eid)
    return {eid: [o for o in group if o != eid] for group in by_hash.values() if len(group) > 1 for eid in group}


def file_verdict(read: dict[str, Any], finding: dict[str, Any] | None,
                 same_as: list[str] | None = None) -> tuple[str, str, int]:
    """One file's `ai_status`, `ai_note` and the score behind the status, for
    IDF's evidence record: what the file shows and, for a photo whose position
    is checked, where it was taken, together. The worse of the two decides: the
    right picture taken 5 km from the declared place is not approved. A file
    also uploaded for another requirement (`same_as`, their titles) is at best
    `changes_required`, at a score of at most 50. Three-way, like the checks:
    audit.run decides a `changes_required` by the score."""
    status, score = checks.evidence_status(read)
    reason = (read.get("reason") or "").strip()
    if status == "approved":
        note = reason or "Shows what it was uploaded for."
    elif status == "rejected":
        note = f"Does not show it: {reason or 'the model saw something else'}"
    else:
        note = f"Cannot be told from the file: {reason or 'no clear answer'}"
    if finding:
        status = checks.worst([status, finding["status"]])
        if finding["status"] != "approved":
            note = f"{note.rstrip('.')}. Location: {finding['details']}"
            score = min(score, finding["score"])
    if same_as:
        status = checks.worst([status, "changes_required"])
        note = f"The same file was also uploaded for {', '.join(same_as)}. {note}"
        score = min(score, 50)
    if len(note) > NOTE_MAX:
        note = note[:NOTE_MAX - 3].rstrip() + "..."
    return status, note, score


def kinds_from_files(package: Package, files: dict[int, Path]) -> Package:
    """The package with every PDF a document, whatever it was labelled: IDF's
    backend has sent PDFs as photos (`image/...`), and a PDF read as a photo
    cannot be opened (2026-11-24, audits 1795516313_3b3c0790 and
    1795516895_03f1e4ce). Only that way round: an image labelled a document -
    a photographed policy - is read as a document on purpose, its text by OCR."""
    fixed = tuple(replace(e, kind="document") if e.kind != "document" and files[e.id].suffix.lower() == ".pdf"
                  else e for e in package.evidence)
    return package if fixed == package.evidence else replace(package, evidence=fixed)


def unreadable(item: Evidence, exc: BaseException) -> dict[str, Any]:
    """The reading of a file that could not be opened: not judged, so `unsure`
    with nothing behind it (`parsed: False`, a score of 20) - decided by score
    like any file that cannot be told, and saying why."""
    return {"kind": item.kind, "shows": "unsure", "confidence": 0.0,
            "reason": f"the file could not be read ({type(exc).__name__}), so it was not judged",
            "description": "", "text": [], "parsed": False, "model_answer": None}


def run(package: Package, files: dict[int, Path], *, read_photo: Reader = evidence.read_photo,
        read_document: Reader = evidence.read_document) -> dict[str, Any]:
    files = {eid: Path(path).resolve() for eid, path in files.items()}   # full paths, whatever folder the caller ran from
    missing = [e.id for e in package.evidence if e.id not in files]
    if missing:
        raise ValueError(f"no file was received for evidence {', '.join(map(str, missing))}")
    package = kinds_from_files(package, files)
    reads: dict[int, dict[str, Any]] = {}
    findings: dict[int, dict[str, Any] | None] = {}
    # The text of every file is read in order by one worker (CPU), so the next
    # file's text is ready while the model (GPU) judges this one: the audit
    # takes the model's time plus one file's OCR, not both added up per file.
    # Only for the real readers - one passed in (a test's) reads its own way.
    ahead = ThreadPoolExecutor(max_workers=1, thread_name_prefix="idf-text")
    try:
        texts: dict[int, Future] = {}
        for item in package.evidence:
            if item.kind == "photo" and read_photo is evidence.read_photo:
                texts[item.id] = ahead.submit(evidence.photo_text, files[item.id])
            elif item.kind != "photo" and read_document is evidence.read_document:
                texts[item.id] = ahead.submit(evidence.document_text, files[item.id])
        for item in package.evidence:
            path, claim = files[item.id], claim_for(package, item)
            try:
                if item.id not in texts:
                    reads[item.id] = (read_photo if item.kind == "photo" else read_document)(path, claim)
                elif item.kind == "photo":
                    reads[item.id] = evidence.read_photo(path, claim, ocr_lines=texts[item.id].result())
                else:
                    reads[item.id] = evidence.read_document(path, claim, text=texts[item.id].result())
            except ModelDidNotAnswer:
                raise                                   # the model, not the file: the audit fails
            except Exception as exc:                    # noqa: BLE001 - this file only
                print(f"evidence {item.id} ({path.name}) could not be read: {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)
                traceback.print_exc()
                reads[item.id] = unreadable(item, exc)
            findings[item.id] = (location.check(item, location.exif_of(path), peers=package.photos)
                                 if item.kind == "photo" else None)
    finally:
        ahead.shutdown(wait=False, cancel_futures=True)     # a failed audit does not wait for text it will not use
    same = same_files(files)

    if package.target_type == "certification":
        result_checks = checks.certification_checks(package, reads, findings, same)
    else:
        result_checks = checks.program_checks(package, reads, same)
    # IDF gets only approved or rejected: every `changes_required` - checks, files,
    # their locations - is decided by its own score here, after the checks ran.
    comments = comments_for(package, result_checks, reads)
    result_checks = [checks.decided(c) for c in result_checks]
    final = result_checks[-1]
    items = []
    for item in package.evidence:
        read = reads[item.id]
        status, score = checks.evidence_status(read)
        twins = [e for e in package.evidence if e.id in same.get(item.id, [])]
        ai_status, ai_note, file_score = file_verdict(read, findings.get(item.id), [checks.label(e) for e in twins])
        finding = findings.get(item.id)
        if finding:
            finding = {**finding, "status": checks.decide(finding["status"], finding["score"])}
        items.append({"id": item.id, "title": item.title, "kind": item.kind, "name": item.name,
                      "ai_status": checks.decide(ai_status, file_score), "ai_note": ai_note,
                      "same_file_as": [e.id for e in twins],
                      "status": checks.decide(status, score), "score": score, "shows": read["shows"],
                      "confidence": read["confidence"], "reason": read["reason"],
                      "description": read["description"], "text": read["text"],
                      "location": finding})
    return {
        "target_type": package.target_type, package.id_key: package.target_id, "number": package.number,
        "status": final["status"], "score": final["score"],
        "comments": comments,
        "checks": result_checks,
        "evidence": items,
        "claims": checks.claims(package, reads) if package.target_type == "certification" else [],
        "model_answers": {str(e.id): reads[e.id].get("model_answer") for e in package.evidence},
    }


def callback_body(result: dict[str, Any]) -> dict[str, Any]:
    """Their callback body: the verdict, the four checks, and one line per file
    for the `ai_status` / `ai_note` of their evidence records."""
    body = {key: result[key] for key in CALLBACK_KEYS}
    body["evidence"] = [{"id": e["id"], "ai_status": e["ai_status"], "ai_note": e["ai_note"]}
                        for e in result["evidence"]]
    return body


def answer_body(result: dict[str, Any]) -> dict[str, Any]:
    """What our own GET returns: the callback body plus the per-file detail;
    never the model's raw answers, which stay server-side."""
    return {key: value for key, value in result.items() if key != "model_answers"}
