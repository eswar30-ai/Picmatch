"""idf/callback.py: off by default, and only ever to the configured host."""

import requests

from idf import settings
from idf import callback


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("IDF_POST_CALLBACK", raising=False)
    out = callback.post("http://h/x", {"status": "approved"})
    assert out["posted"] is False and "off" in out["detail"]


def test_the_host_is_ours_and_the_path_is_the_packages():
    base = "http://192.168.214.23:4000"
    # Their test packages say localhost: their server speaking of itself.
    assert callback.target("http://localhost:4000/api/certification/applications/26/ai-validation", base) \
        == "http://192.168.214.23:4000/api/certification/applications/26/ai-validation"
    assert callback.target("http://192.168.214.23:4000/api/programs/29/ai-validation?x=1", base + "/") \
        == "http://192.168.214.23:4000/api/programs/29/ai-validation?x=1"
    assert callback.target("http://evil.example/api/x", base) == "http://192.168.214.23:4000/api/x"
    assert callback.target("/api/x", base) == "http://192.168.214.23:4000/api/x"
    assert callback.target("http://192.168.214.23:4000/api/x", "") is None
    assert callback.target("http://h/", base) is None
    assert callback.target("", base) is None


def test_posts_with_their_key_and_records_the_answer(monkeypatch):
    monkeypatch.setenv("IDF_POST_CALLBACK", "true")
    monkeypatch.setenv("IDF_CALLBACK_BASE", "http://h:4000")
    monkeypatch.setenv("IDF_API_KEY", "k")
    sent = {}
    class R:
        status_code = 200
        text = "ok"
    def fake_post(url, json=None, headers=None, timeout=None):
        sent.update(url=url, body=json, headers=headers)
        return R()
    monkeypatch.setattr(requests, "post", fake_post)
    out = callback.post("http://localhost:4000/api/a/1/ai-validation", {"status": "approved"})
    assert out["posted"] is True and sent["headers"]["X-API-Key"] == "k" and sent["body"] == {"status": "approved"}
    assert sent["url"] == "http://h:4000/api/a/1/ai-validation" and out["url"] == sent["url"]
    monkeypatch.delenv("IDF_CALLBACK_BASE")
    out = callback.post("http://other/api", {"status": "approved"})
    assert out["posted"] is False and "IDF_CALLBACK_BASE" in out["detail"]


def test_a_callback_that_does_not_answer_is_recorded_not_raised(monkeypatch):
    monkeypatch.setenv("IDF_POST_CALLBACK", "true")
    monkeypatch.setenv("IDF_CALLBACK_BASE", "http://h:4000")
    def down(*a, **k):
        raise requests.ConnectionError("down")
    monkeypatch.setattr(requests, "post", down)
    out = callback.post("http://h:4000/api/a/1/ai-validation", {})
    assert out["posted"] is False and "ConnectionError" in out["detail"]
    assert settings.enabled("IDF_POST_CALLBACK")
