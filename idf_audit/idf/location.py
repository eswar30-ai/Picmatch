"""Where and when a photo was taken, from the file itself, against the claim.

The package declares a latitude, longitude and `captured_at` for a photo: the
position their app recorded, and - read from their submission API - the time the
evidence was submitted, not the time the picture was taken (every `captured_at`
equals `submitted_at` to the second). The file's own EXIF is the independent
witness: a phone writes the position and time into the JPEG as it takes the
picture. Its time is held to "before submission, and not long before"; its
position is compared with the declared one.

A photo taken in their app (`source: camera`) is usually a resized copy without
EXIF. Its position is then the one the app recorded as it was taken, and what can
be checked is that the application's photos were taken in one place.

No geocoding: turning the institution's address into coordinates needs an
outside service, which this service does not call. Until the
package carries the institution's own coordinates, "within the premises" is
not a judgement this check can make, and it says so instead of guessing.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from idf.package import Evidence

#: Within this of the declared position the file agrees; beyond FAR it contradicts.
NEAR_M = 200
FAR_M = 2000
#: A photo taken more than this long before the evidence was submitted is old
#: enough for a person to ask whether it still shows the place as it is.
MAX_AGE_DAYS = 90
#: EXIF writes the phone's local time, the package may be UTC: a file "taken
#: after submission" by less than this is a time zone, not a fault.
ZONE_HOURS = 14
#: Camera photos of one application further apart than this were not taken at one site.
SAME_SITE_M = 1000

_GPS_IFD = 0x8825
_EXIF_IFD = 0x8769
_DATE_ORIGINAL = 36867
_DATE_DIGITIZED = 36868
_DATE_TIME = 306


def _degrees(value: Any, ref: Any) -> float | None:
    try:
        d, m, s = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if any(math.isnan(v) for v in (d, m, s)):
        return None
    degrees = d + m / 60 + s / 3600
    return -degrees if str(ref or "").strip().upper() in ("S", "W") else degrees


def exif_of(path: Path) -> dict[str, Any]:
    """`latitude`, `longitude` and `taken_at` ("YYYY-MM-DD HH:MM:SS") from the
    file's EXIF; each None when the file does not carry it."""
    out: dict[str, Any] = {"latitude": None, "longitude": None, "taken_at": None}
    try:
        from PIL import Image

        with Image.open(path) as image:
            exif = image.getexif()
            gps = exif.get_ifd(_GPS_IFD)
            inner = exif.get_ifd(_EXIF_IFD)
            when = inner.get(_DATE_ORIGINAL) or inner.get(_DATE_DIGITIZED) or exif.get(_DATE_TIME)
    except Exception:                                               # noqa: BLE001
        return out
    if gps:
        lat = _degrees(gps.get(2), gps.get(1))
        lon = _degrees(gps.get(4), gps.get(3))
        if lat is not None and lon is not None and (lat, lon) != (0.0, 0.0):
            out["latitude"], out["longitude"] = round(lat, 6), round(lon, 6)
    stamp = parse_time(str(when)) if when else None
    out["taken_at"] = stamp.strftime("%Y-%m-%d %H:%M:%S") if stamp else None
    return out


def parse_time(value: str | None) -> datetime | None:
    """EXIF writes `2026:11:13 14:17:49`; the package `2026-11-13 14:17:49` or
    ISO with a `T`. All read; anything else is None."""
    text = str(value or "").strip()
    match = re.match(r"^(\d{4})[:\-](\d{2})[:\-](\d{2})[ T](\d{2}):(\d{2})(?::(\d{2}))?", text)
    if not match:
        return None
    try:
        return datetime(*(int(g or 0) for g in match.groups()))
    except ValueError:
        return None


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _camera(evidence: Evidence, peers: Sequence[Evidence]) -> dict[str, Any]:
    """A photo taken in their app, without a position in the file: placed by the
    app's own record, and checked against the application's other photos."""
    what = "the position recorded by the app as the photo was taken"
    unchecked = "not checked against the institution's own location, which the package does not carry"
    others = [(o, distance_m(evidence.latitude, evidence.longitude, o.latitude, o.longitude))
              for o in peers if o.id != evidence.id and o.latitude is not None and o.longitude is not None]
    if not others:
        return {"status": "approved", "score": 70,
                "details": f"camera photo without GPS in the file: {what}, the only photo with a position; {unchecked}"}
    far, metres = max(others, key=lambda pair: pair[1])
    if metres <= SAME_SITE_M:
        return {"status": "approved", "score": 75,
                "details": f"camera photo without GPS in the file: {what} is within {round(metres)} m of the "
                           f"application's other {len(others)} photo(s), one site; {unchecked}"}
    return {"status": "changes_required", "score": 40,
            "details": f"camera photo without GPS in the file: {what} is {round(metres)} m from another photo of "
                       f"this application (evidence {far.id}), not one site"}


def check(evidence: Evidence, exif: dict[str, Any], peers: Sequence[Evidence] = ()) -> dict[str, Any] | None:
    """The finding for one photo, or None when no location is required of it.
    `peers` are the application's other photos, for a camera photo without EXIF.

    status    approved          the file's GPS is within NEAR_M of the declared position; or a
                                camera photo without EXIF taken where the application's other
                                photos were
              changes_required  nothing to compare, the two between NEAR_M and FAR_M apart, a
                                camera photo away from the others, or the file's time after
                                submission or more than MAX_AGE_DAYS before it
              rejected          the file's GPS is more than FAR_M from the declared position
    """
    declared = evidence.latitude is not None and evidence.longitude is not None
    in_file = exif.get("latitude") is not None and exif.get("longitude") is not None
    # GPS is judged only when IDF marks the photo `requires_gps` - its way of making
    # it mandatory. Until then it is set aside (the owner, 2026-11-14).
    if not evidence.requires_gps:
        return None
    finding: dict[str, Any] = {"declared": [evidence.latitude, evidence.longitude] if declared else None,
                               "in_file": [exif.get("latitude"), exif.get("longitude")] if in_file else None,
                               "distance_m": None, "taken_at": exif.get("taken_at"),
                               "captured_at": evidence.captured_at, "source": evidence.source or None}
    if declared and in_file:
        d = distance_m(evidence.latitude, evidence.longitude, exif["latitude"], exif["longitude"])
        finding["distance_m"] = round(d)
        where = f"the file's GPS is {round(d)} m from the declared position"
        if d <= NEAR_M:
            finding.update(status="approved", score=95, details=f"{where}: they agree")
        elif d <= FAR_M:
            finding.update(status="changes_required", score=60, details=f"{where}: close, not the same place")
        else:
            finding.update(status="rejected", score=10, details=f"{where}: not the declared place")
    elif declared and evidence.source == "camera":
        finding.update(_camera(evidence, peers))
    elif declared:
        finding.update(status="changes_required", score=40,
                       details="the file carries no GPS data, so the declared position cannot be checked")
    elif in_file:
        finding.update(status="changes_required", score=50,
                       details="the file carries GPS data but the package declares no position to compare")
    else:
        finding.update(status="changes_required", score=30,
                       details="no position in the file and none declared")

    taken, submitted = parse_time(exif.get("taken_at")), parse_time(evidence.captured_at)
    if taken and submitted:
        hours = (submitted - taken).total_seconds() / 3600
        finding["days_before_submission"] = round(hours / 24, 1)
        if hours < -ZONE_HOURS:
            finding["details"] += (f"; the file says it was taken {taken:%Y-%m-%d %H:%M}, "
                                   f"after the evidence was submitted ({submitted:%Y-%m-%d %H:%M})")
            if finding["status"] == "approved":
                finding.update(status="changes_required", score=min(finding["score"], 50))
        elif hours > MAX_AGE_DAYS * 24:
            finding["details"] += (f"; taken {taken:%Y-%m-%d}, {round(hours / 24)} days before submission: "
                                   f"older than {MAX_AGE_DAYS} days")
            if finding["status"] == "approved":
                finding.update(status="changes_required", score=min(finding["score"], 60))
        else:
            finding["details"] += f"; taken {max(0, round(hours / 24))} day(s) before submission"
    else:
        finding["details"] += "; time not compared (" + (
            "the file carries no time" if not taken else "no submission time") + ")"
    return finding
