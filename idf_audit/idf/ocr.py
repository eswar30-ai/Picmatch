"""The text printed on a photo or a PDF page, and the vision model's health.

The OCR is RapidOCR with the settings the OCR platform's classifier uses
(detection box threshold 0.35, unclip ratio 2.0, text score 0.35), called
directly: the platform reached it through its classifier pipeline, which this
service does not carry. Deterministic and independent of the vision model,
which is what makes it useful for checking the model's reading.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import requests

from idf import settings

DET_BOX_THRESH = 0.35
DET_UNCLIP_RATIO = 2.0
TEXT_SCORE = 0.35


class ModelDidNotAnswer(RuntimeError):
    """The vision model could not be asked or did not answer. The audit fails:
    no verdict is built from a photo the model never judged."""


_engine = None
_engine_lock = threading.Lock()


def engine():
    """One RapidOCR engine for the process, built on first use (a second or two)."""
    global _engine
    with _engine_lock:
        if _engine is None:
            from rapidocr_onnxruntime import RapidOCR

            _engine = RapidOCR(det_box_thresh=DET_BOX_THRESH, det_unclip_ratio=DET_UNCLIP_RATIO,
                               text_score=TEXT_SCORE)
        return _engine


def page_text(images: list[Path]) -> str:
    """The text printed on these images, one line per detected line."""
    reader = engine()
    lines: list[str] = []
    for image in images:
        result, _ = reader(str(Path(image).resolve()))
        lines += [line[1] for line in (result or [])]
    return "\n".join(lines)


def pdf_text(path: Path, pages: int = 2) -> str:
    """The text a PDF carries itself, for its first pages - exact where it
    exists. Empty for a scanned PDF."""
    import pypdfium2 as pdfium

    try:
        pdf = pdfium.PdfDocument(str(path))
    except Exception:                                            # noqa: BLE001
        return ""
    try:
        out = []
        for index in range(min(pages, len(pdf))):
            page = pdf[index]
            out.append(page.get_textpage().get_text_range())
        return "\n".join(out)
    except Exception:                                            # noqa: BLE001
        return ""
    finally:
        pdf.close()


def vlm_health() -> dict[str, Any]:
    """Is Ollama up, and does it have the model? Asked right before each audit."""
    base = settings.value("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = settings.value("OLLAMA_MODEL", "qwen2.5vl:7b")
    try:
        reply = requests.get(f"{base}/api/tags", timeout=5)
        reply.raise_for_status()
        names = {m.get("name") for m in reply.json().get("models", [])}
    except Exception as exc:                                     # noqa: BLE001
        return {"up": False, "model_present": False, "model": model, "base": base,
                "reason": f"Ollama is not reachable at {base} ({type(exc).__name__})"}
    present = model in names or f"{model}:latest" in names
    return {"up": True, "model_present": present, "model": model, "base": base,
            "reason": None if present else f"Ollama at {base} does not have {model} (ollama pull {model})"}
