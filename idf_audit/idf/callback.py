"""Posting a verdict to IDF's callback - when that is switched on.

IDF polls our `GET /api/audit/{id}` for the verdict; the callback is the
other way they offered, and off unless `.env` says otherwise:

    IDF_POST_CALLBACK=true          post the verdict when an audit finishes
    IDF_CALLBACK_BASE=http://...    their backend. The package's `callback_url`
                                    gives the path (`/api/certification/applications/26/
                                    ai-validation`); the host is always this one. Their
                                    test packages name `http://localhost:4000/...`, their
                                    server speaking of itself, so the host in the package
                                    is not used - and a package sent by a caller cannot
                                    make "post the verdict" into "post to anywhere"
    IDF_API_KEY=...                 sent as X-API-Key, their machine-to-machine key

A failed post never fails the audit: the verdict is stored and theirs to
fetch; the post's outcome is recorded beside it.
"""

from __future__ import annotations

import sys
from typing import Any
from urllib.parse import urlsplit

import requests

from idf import settings


def target(url: str | None, base: str | None) -> str | None:
    """Where the verdict is posted: `base` plus the path of the package's
    `callback_url`. None when either is missing, or the URL has no path."""
    if not url or not base:
        return None
    parts = urlsplit(url.strip())
    path = parts.path if parts.scheme or url.startswith("/") else ""
    if not path or path == "/":
        return None
    return base.rstrip("/") + path + (f"?{parts.query}" if parts.query else "")


def post(url: str | None, body: dict[str, Any]) -> dict[str, Any]:
    """`{"posted": bool, "status": http status or None, "detail": why, "url": where}`."""
    if not settings.enabled("IDF_POST_CALLBACK"):
        return {"posted": False, "status": None, "url": None,
                "detail": "IDF_POST_CALLBACK is off; the verdict is fetched by GET"}
    base = settings.value("IDF_CALLBACK_BASE")
    if not url:
        return {"posted": False, "status": None, "url": None, "detail": "the package names no callback_url"}
    where = target(url, base)
    if not where:
        return {"posted": False, "status": None, "url": None,
                "detail": f"no callback target: IDF_CALLBACK_BASE is {base or 'unset'} and callback_url has no path"}
    headers = {"Content-Type": "application/json"}
    key = settings.value("IDF_API_KEY")
    if key:
        headers["X-API-Key"] = key
    try:
        response = requests.post(where, json=body, headers=headers, timeout=settings.number("IDF_CALLBACK_TIMEOUT", 20))
    except requests.RequestException as exc:
        print(f"IDF callback to {where} failed: {type(exc).__name__}", file=sys.stderr, flush=True)
        return {"posted": False, "status": None, "url": where, "detail": f"{type(exc).__name__}: the callback did not answer"}
    ok = 200 <= response.status_code < 300
    return {"posted": ok, "status": response.status_code, "url": where,
            "detail": "accepted" if ok else f"the callback answered {response.status_code}: {response.text[:200]}"}
