"""The IDF audit service: judge a certification application or a program
against its evidence, and answer in the shape the IDF backend expects.

Copied from the OCR platform (D:\\ocr-platform, core/idf, decision D29) with
only what the audit needs: RapidOCR for the text, Qwen2.5-VL through Ollama for
the judgement, EXIF for where a photo was taken.

    package.py    their `ai-package` JSON, read into a Package; contact details dropped
    evidence.py   one photo or document read against the claim it was uploaded for
    location.py   where and when a photo was taken, from its own EXIF, against the claim
    checks.py     the four checks and the overall verdict, from the readings
    audit.py      runs the whole thing for one package and its files
    fetch.py      an evidence file fetched from IDF's server by its path
    callback.py   posts the verdict to their callback URL, when that is switched on
    ocr.py        RapidOCR, a PDF's own text, and whether Ollama is up
    helpers.py    small shared helpers
    settings.py   the .env and where data lives
    access.py     the key (or the open switch) and the per-address limit
    jobs.py       the queue: one audit at a time, surviving a restart
    app.py        the API: POST /api/audit, GET /api/audit/{id}, GET /api/health

The judgement is advice to a human reviewer (their pipeline goes AI audit ->
manual review -> scoring). So the rule throughout the checks is: `approved`
only when the evidence shows it, `rejected` only when the evidence contradicts
it, and `changes_required` whenever it cannot tell - never a confident answer
from nothing. IDF wants only two outcomes (2026-11-14), so each
`changes_required` is then decided by its score (checks.decided, APPROVE_SCORE)
and the details say it was not confirmed.
"""
