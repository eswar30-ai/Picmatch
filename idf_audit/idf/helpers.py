"""Small helpers the audit shares, copied unchanged from the OCR platform
(D:\\ocr-platform: core/scene.py and core/checks.py) so the verdicts match:
reading the model's JSON, preparing a photo for the model, and matching text."""

from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path
from typing import Any

#: Photos larger than this on their longest side are scaled down before the model sees them.
MAX_SIDE = 4000


def _unfenced(raw: str) -> str:
    text = raw.strip()
    fence = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```$", text, re.S)
    return fence.group(1).strip() if fence else text


def _as_json(raw: str) -> dict[str, Any] | None:
    """The first JSON object in the model's answer, fenced or not; None if there is none."""
    text = _unfenced(raw)
    for candidate in (text, text[text.find("{"): text.rfind("}") + 1] if "{" in text else ""):
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _confidence(value: Any) -> float:
    """A confidence from 0 to 1; "92" or "92.0" read as 0.92; anything else 0."""
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0
    if 1.0 < number <= 100.0:
        number /= 100.0
    return round(number, 4) if 0.0 <= number <= 1.0 else 0.0


def _letters(line: str) -> str:
    return re.sub(r"[^a-z0-9]", "", line.lower())


def reconcile(ocr_lines: list[str], seen: list[str]) -> list[str]:
    """RapidOCR's lines, in its order; a line the model also saw takes the
    model's spelling (it tends to get case and punctuation right). A line only
    the model saw is not added: it may not be in the photo at all."""
    by_letters = {_letters(s): s for s in seen if _letters(s)}
    return [by_letters.get(_letters(line), line) for line in ocr_lines]


def _photo_for_model(path: Path) -> str:
    """The photo upright, as base64 JPEG, at most MAX_SIDE pixels on a side."""
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        upright = ImageOps.exif_transpose(image) or image
        upright = upright.convert("RGB")
        if max(upright.size) > MAX_SIDE:
            upright.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        buffer = io.BytesIO()
        upright.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _digits(s: Any) -> str:
    return re.sub(r"\D", "", str(s or ""))


def _on_page(value: Any, page_text: str) -> bool:
    """Is this value printed on the page? Digits compared as digits, words
    as words, so OCR spacing and punctuation do not decide it."""
    v = str(value or "").strip()
    if not v:
        return True
    if sum(c.isdigit() for c in v) >= 4:
        return _digits(v) in _digits(page_text) or _digits(v)[-4:] in _digits(page_text)
    words = [w for w in re.findall(r"[A-Za-z]{3,}", v.upper())]
    text = page_text.upper()
    # Two thirds of its words, rounded up: one shared word ("KUMAR") is not
    # the same name, and OCR may still lose one word of a long one.
    return not words or sum(w in text for w in words) >= -(-2 * len(words) // 3)
