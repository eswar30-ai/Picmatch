"""An evidence file fetched from IDF's server, by the path their package gives.

IDF uploads the applicant's files to its own server and puts each one's path in
the package (`evidence_files[].url`: `/uploads/efa55f1a....jpg`). When a file is
not sent with the audit, it is downloaded from there before anything is judged.

Only from their server: the host is always `IDF_BASE_URL` (or `IDF_CALLBACK_BASE`)
and only the path is taken from the package. Their packages name
`http://localhost:5000/...` - their server speaking of itself - and a package
comes from a caller, so following the URL as given would let anyone who sends a
package make this server fetch any address it can reach. Redirects are not
followed for the same reason.

Only a real image or PDF: their test server holds 12-byte JPEG headers and an
18-byte "PDF" where photos should be, and judging a placeholder would be a
verdict on nothing.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

import requests

from idf import settings

TIMEOUT = 30


class FetchFailed(ValueError):
    """The file could not be fetched, or is not one that can be audited; the message says which."""


def server() -> str:
    return settings.value("IDF_BASE_URL") or settings.value("IDF_CALLBACK_BASE")


def url_for(url: str | None, base_url: str | None) -> str | None:
    """`base_url` plus the path of `url`, whatever host `url` names."""
    if not url or not base_url:
        return None
    parts = urlsplit(url.strip())
    path = parts.path if (parts.scheme or parts.netloc) else url.strip().split("?", 1)[0]
    if not path.startswith("/"):
        path = "/" + path
    if path == "/":
        return None
    return base_url.rstrip("/") + path + (f"?{parts.query}" if parts.query else "")


def check_file(path: Path) -> None:
    """Raise FetchFailed unless `path` opens as an image or starts like a PDF:
    applied to sent files too, so a placeholder is refused, never judged."""
    head = path.read_bytes()[:5]
    if path.suffix.lower() == ".pdf":
        if head != b"%PDF-":
            raise FetchFailed(f"not a PDF ({path.stat().st_size} bytes)")
        return
    from PIL import Image

    try:
        with Image.open(path) as image:
            image.load()
    except Exception as exc:                                        # noqa: BLE001
        raise FetchFailed(f"not a readable image ({path.stat().st_size} bytes: {type(exc).__name__})") from None


def download(url: str, target: Path, *, base: str | None = None, max_bytes: int) -> Path:
    """Fetch `url`'s path from IDF's server into `target`; raise FetchFailed
    with the reason when it cannot be had or is not a real image or PDF."""
    base_url = base if base is not None else server()
    where = url_for(url, base_url)
    if not where:
        raise FetchFailed("IDF_BASE_URL is not set, so there is no server to fetch it from"
                          if not base_url else f"the path {url!r} names no file")
    headers = {}
    key = settings.value("IDF_API_KEY")
    if key:
        headers["X-API-Key"] = key
    try:
        reply = requests.get(where, headers=headers, timeout=TIMEOUT, stream=True, allow_redirects=False)
    except requests.RequestException as exc:
        raise FetchFailed(f"{where} did not answer ({type(exc).__name__})") from None
    try:
        if reply.status_code != 200:
            raise FetchFailed(f"{where} answered {reply.status_code}")
        size = 0
        with target.open("wb") as out:
            for chunk in reply.iter_content(64 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise FetchFailed(f"{where} is larger than {max_bytes // (1024 * 1024) or max_bytes} "
                                      f"{'MB' if max_bytes >= 1024 * 1024 else 'bytes'}")
                out.write(chunk)
        check_file(target)
    except FetchFailed:
        target.unlink(missing_ok=True)
        raise
    finally:
        reply.close()
    return target
