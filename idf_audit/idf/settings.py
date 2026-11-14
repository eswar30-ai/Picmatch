"""Where this service's settings and data live.

One `.env`, in the service's folder (D:\\Picmatch\\idf_audit\\.env), loaded into
the environment here. A variable already set in the environment wins - tests
rely on that.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def value(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def enabled(key: str) -> bool:
    return value(key).strip().lower() in {"1", "true", "yes"}


def number(key: str, default: int) -> int:
    try:
        return int(value(key).strip() or default)
    except ValueError:
        return default


VAR = Path(value("IDF_DATA_DIR") or (ROOT / "var"))
UPLOADS = VAR / "uploads"        # one folder per audit: package.json and the evidence files
AUDITS = VAR / "audits"          # one verdict per audit, <audit_id>.json
JOBS = VAR / "jobs"              # one state file per audit, so the queue survives a restart
for _folder in (UPLOADS, AUDITS, JOBS):
    _folder.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_SIZE_MB = number("MAX_UPLOAD_SIZE_MB", 50)
