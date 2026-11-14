"""idf/fetch.py: an evidence file fetched from IDF's own server by the
path the package gives - only from that server, and only a real image or PDF."""

import io

import pytest
import requests
from PIL import Image

from idf import fetch


def jpg():
    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), "white").save(buffer, "JPEG")
    return buffer.getvalue()


class Reply:
    def __init__(self, status=200, body=b"", ctype="image/jpeg"):
        self.status_code, self._body, self.headers = status, body, {"Content-Type": ctype}

    def iter_content(self, size):
        for i in range(0, len(self._body), size):
            yield self._body[i:i + size]

    def close(self):
        pass


def serve(monkeypatch, replies):
    asked = []

    def fake_get(url, headers=None, timeout=None, stream=None, allow_redirects=None):
        asked.append((url, headers, allow_redirects))
        return replies[url]
    monkeypatch.setattr(requests, "get", fake_get)
    return asked


BASE = "http://192.168.214.23:4000"


def test_the_host_is_always_theirs():
    assert fetch.url_for("/uploads/a.jpg", BASE) == f"{BASE}/uploads/a.jpg"
    assert fetch.url_for("http://localhost:5000/uploads/a.jpg", BASE) == f"{BASE}/uploads/a.jpg"
    assert fetch.url_for("http://evil.example/x.jpg", BASE) == f"{BASE}/x.jpg"
    assert fetch.url_for("uploads/a.jpg", BASE) == f"{BASE}/uploads/a.jpg"
    assert fetch.url_for("/uploads/a.jpg", "") is None and fetch.url_for("", BASE) is None


def test_a_photo_is_fetched_and_kept(monkeypatch, tmp_path):
    asked = serve(monkeypatch, {f"{BASE}/uploads/a.jpg": Reply(body=jpg())})
    path = fetch.download("/uploads/a.jpg", tmp_path / "43_a.jpg", base=BASE, max_bytes=10_000_000)
    assert path.read_bytes()[:2] == b"\xff\xd8" and asked[0][2] is False       # redirects not followed


@pytest.mark.parametrize("reply,words", [
    (Reply(404, b'{"error":"not found"}', "application/json"), "answered 404"),
    (Reply(body=b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01"), "not a readable image"),   # their 12-byte placeholders
    (Reply(body=b"%PDF-"[:3] + b" not a pdf", ctype="application/pdf"), "not a PDF"),
    (Reply(body=b"x" * 2000), "larger than"),
])
def test_what_cannot_be_audited_says_why(monkeypatch, tmp_path, reply, words):
    name = "43_a.pdf" if reply.headers["Content-Type"] == "application/pdf" else "43_a.jpg"
    serve(monkeypatch, {f"{BASE}/uploads/{name}": reply})
    with pytest.raises(fetch.FetchFailed, match=words):
        fetch.download(f"/uploads/{name}", tmp_path / name, base=BASE, max_bytes=1000)
    assert not (tmp_path / name).exists()


def test_a_server_that_does_not_answer_is_a_fetch_failure(monkeypatch, tmp_path):
    def down(*a, **k):
        raise requests.ConnectionError("down")
    monkeypatch.setattr(requests, "get", down)
    with pytest.raises(fetch.FetchFailed, match="did not answer"):
        fetch.download("/uploads/a.jpg", tmp_path / "a.jpg", base=BASE, max_bytes=1000)
    with pytest.raises(fetch.FetchFailed, match="IDF_BASE_URL"):
        fetch.download("/uploads/a.jpg", tmp_path / "a.jpg", base="", max_bytes=1000)
