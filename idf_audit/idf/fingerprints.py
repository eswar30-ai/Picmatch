"""The same photo in another application: a fingerprint of every photo audited.

A fingerprint is a 64-bit difference hash (dHash): the photo shrunk to 9 x 8
grey pixels, one bit per pair of neighbours, brighter or not. Resizing,
recompressing or a light edit keeps it within a few bits; a different
picture is some 32 bits away. Two photos within MATCH_BITS bits are taken to
be the same picture.

Kept in `var/fingerprints.jsonl`, one line per photo of a finished audit:
which application or program it belonged to (`owner`), its evidence id and the
audit. A photo is "reused" when it matches one that belonged to a *different*
owner: an application submitted again, with the same photos, is not flagged
against itself.

Used only when AUTHENTICITY_CHECKS is on. Nothing leaves this server.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from idf import settings

MATCH_BITS = 6
_LOCK = threading.Lock()


def index_file() -> Path:
    return settings.VAR / "fingerprints.jsonl"


def dhash(path: Path) -> str | None:
    """The photo's fingerprint as 16 hex digits; None when it cannot be opened as an image."""
    try:
        with Image.open(path) as image:
            grey = ImageOps.exif_transpose(image).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    except Exception:                                               # noqa: BLE001
        return None
    pixels = grey.tobytes()                      # one byte per grey pixel, row by row
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return f"{bits:016x}"


def distance(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _records() -> list[dict[str, Any]]:
    path = index_file()
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def reused(owner: str, photos: dict[int, Path]) -> dict[int, dict[str, Any]]:
    """For each photo (evidence id -> file) another owner submitted first, the
    first such record, with its `bits` apart. "First" by the index's order: a
    photo this owner submitted before anyone else is theirs, and a copy someone
    made of it later does not flag this owner when it submits again."""
    records = [r for r in _records() if r.get("hash")]
    out: dict[int, dict[str, Any]] = {}
    for eid, path in photos.items():
        mine = dhash(path)
        if mine is None:
            continue
        near = [(i, distance(mine, r["hash"]), r) for i, r in enumerate(records)]
        near = [(i, bits, r) for i, bits, r in near if bits <= MATCH_BITS]
        own_first = min((i for i, _, r in near if r.get("owner") == owner), default=None)
        before = [(i, bits, r) for i, bits, r in near
                  if r.get("owner") != owner and (own_first is None or i < own_first)]
        if before:
            _, bits, record = min(before, key=lambda match: match[0])
            out[eid] = {**record, "bits": bits}
    return out


def remember(owner: str, audit_id: str, photos: dict[int, Path]) -> int:
    """Add the photos of a finished audit to the index; how many were added."""
    lines = []
    for eid, path in photos.items():
        mine = dhash(path)
        if mine is not None:
            lines.append(json.dumps({"hash": mine, "owner": owner, "evidence": eid, "audit": audit_id,
                                     "at": time.strftime("%Y-%m-%d %H:%M:%S")}))
    if lines:
        with _LOCK, index_file().open("a", encoding="utf-8") as out:
            out.write("\n".join(lines) + "\n")
    return len(lines)


def describe(record: dict[str, Any]) -> str:
    """A match in words: "certification 9001 (evidence 43, audit ..., 2026-11-25 10:12)"."""
    kind, _, number = str(record.get("owner", "")).partition(":")
    return (f"{kind} {number} (evidence {record.get('evidence')}, audit {record.get('audit')}, "
            f"{str(record.get('at', ''))[:16]})")
