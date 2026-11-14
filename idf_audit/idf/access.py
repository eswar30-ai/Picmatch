"""Who may call, and how often.

    OPEN_AUDIT_API=true      no key needed (the owner's call for IDF's backend, 2026-11-14)
    IDF_AUDIT_API_KEYS=a,b   otherwise one of these as X-API-Key
    UPLOAD_RATE_LIMIT_PER_MINUTE=10   submissions per address per minute
    MAX_QUEUED=100           audits that may wait before new ones get 503
"""

from __future__ import annotations

import hmac
import sys
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from idf import settings


def is_open() -> bool:
    return settings.enabled("OPEN_AUDIT_API")


def keys() -> list[str]:
    return [k.strip() for k in settings.value("IDF_AUDIT_API_KEYS").split(",") if k.strip()]


def client_key(request: Request) -> str:
    """The caller's address. X-Forwarded-For is not trusted: any client can set it."""
    return request.client.host if request.client else "unknown"


def _supplied(request: Request) -> str:
    return request.headers.get("X-API-Key") or request.query_params.get("api_key") or ""


def has_key(request: Request) -> bool:
    supplied = _supplied(request)
    return bool(supplied) and any(hmac.compare_digest(supplied, k) for k in keys())


def require(request: Request) -> None:
    if is_open() or has_key(request):
        return
    print(f"401 for {client_key(request)} {request.method} {request.url.path}: no valid X-API-Key",
          file=sys.stderr, flush=True)
    raise HTTPException(401, "Send a valid X-API-Key header.")


def caller_of(request: Request) -> str:
    return f"{'API key' if has_key(request) else 'No key'} · {client_key(request)}"


class Window:
    """At most `limit` events per `seconds` per key, in memory."""

    def __init__(self, limit_of, seconds: float) -> None:
        self.limit_of, self.seconds = limit_of, seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _trim(self, key: str, now: float) -> deque[float]:
        bucket = self._hits[key]
        while bucket and bucket[0] < now - self.seconds:
            bucket.popleft()
        return bucket

    def blocked_for(self, key: str) -> int:
        """Seconds until `key` may try again; 0 when it may now."""
        now = time.time()
        with self._lock:
            bucket = self._trim(key, now)
            if len(bucket) < self.limit_of():
                return 0
            return max(1, int(bucket[0] + self.seconds - now) + 1)

    def record(self, key: str) -> None:
        with self._lock:
            self._hits[key].append(time.time())


def rate_limit() -> int:
    return settings.number("UPLOAD_RATE_LIMIT_PER_MINUTE", 10)


def max_queued() -> int:
    return settings.number("MAX_QUEUED", 100)


submissions = Window(rate_limit, 60)
