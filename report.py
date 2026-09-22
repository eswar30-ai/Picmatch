"""Self-contained HTML contact sheet of the match results.

The point of this file is that you can *see* every decision: each hit shows the
actual face crop the score came from, so a wrong person in someone's results
takes one glance to spot instead of opening 400 photos.
"""

import base64
import html
import os

import cv2

import gpu_engine as ge

_CSS = """
:root{--bg:#12100e;--card:#1c1916;--ink:#f2ede4;--soft:#a49b8c;--line:#2e2a25;
      --ok:#5fbf7a;--maybe:#e0a33a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:14px/1.5 "Segoe UI",Inter,system-ui,sans-serif}
.wrap{max-width:1200px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:26px;margin:0 0 4px}
.sub{color:var(--soft);margin:0 0 28px;font-size:13px}
.person{background:var(--card);border:1px solid var(--line);border-radius:8px;
        padding:16px 18px;margin-bottom:18px}
.person h2{font-size:17px;margin:0 0 2px}
.meta{color:var(--soft);font-size:12px;margin:0 0 12px}
.band{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--soft);
      margin:14px 0 8px;border-top:1px solid var(--line);padding-top:10px}
.grid{display:flex;flex-wrap:wrap;gap:10px}
figure{margin:0;width:104px}
figure img{width:104px;height:104px;object-fit:cover;border-radius:5px;display:block;
           border:2px solid transparent}
figure.match img{border-color:var(--ok)}
figure.review img{border-color:var(--maybe)}
figcaption{font-size:10px;color:var(--soft);margin-top:4px;word-break:break-all;
           line-height:1.35}
figcaption b{color:var(--ink);font-weight:600}
.empty{color:var(--soft);font-style:italic;font-size:13px}
table{border-collapse:collapse;width:100%;margin-bottom:28px;font-size:13px}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line)}
th{color:var(--soft);font-weight:500;font-size:11px;letter-spacing:.06em;
   text-transform:uppercase}
"""


def _thumb(img, bbox, size=112, pad=0.35):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 = int(max(0, x1 - bw * pad))
    y1 = int(max(0, y1 - bh * pad))
    x2 = int(min(w, x2 + bw * pad))
    y2 = int(min(h, y2 + bh * pad))
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def write(path, result, summary, folder, settings, per_person_limit=80):
    """Render the report. `result` is matcher.match()'s return value."""
    # decode each photo at most once, no matter how many people it feeds
    wanted = {}
    for name, hits in result["by_person"].items():
        for h in hits[:per_person_limit]:
            wanted.setdefault(h["photo"], []).append(h)

    thumbs = {}
    for photo, hits in wanted.items():
        img = ge.load_image(os.path.join(folder, photo))
        if img is None:
            continue
        for h in hits:
            thumbs[(photo, h["face_idx"])] = _thumb(img, h["bbox"])

    parts = ["<!doctype html><meta charset='utf-8'>",
             "<title>PicMatch GPU - match report</title>",
             f"<style>{_CSS}</style><div class='wrap'>",
             "<h1>Match report</h1>",
             f"<p class='sub'>{html.escape(settings)}</p>",
             "<table><tr><th>Person</th><th>Confident</th><th>Needs a look</th>"
             "<th>Best score</th></tr>"]
    for row in summary:
        parts.append(f"<tr><td>{html.escape(row['person'])}</td><td>{row['matches']}</td>"
                     f"<td>{row['review']}</td><td>{row['best_score']}</td></tr>")
    parts.append("</table>")

    for row in summary:
        name = row["person"]
        hits = result["by_person"].get(name, [])[:per_person_limit]
        parts.append("<div class='person'>")
        parts.append(f"<h2>{html.escape(name)}</h2>")
        parts.append(f"<p class='meta'>{row['matches']} confident &middot; "
                     f"{row['review']} to review</p>")
        for status, label in (("match", "confident"), ("review", "borderline - check these")):
            group = [h for h in hits if h["status"] == status]
            if not group:
                continue
            parts.append(f"<div class='band'>{label} ({len(group)})</div><div class='grid'>")
            for h in group:
                src = thumbs.get((h["photo"], h["face_idx"]))
                if not src:
                    continue
                q = h["quality"]
                cap = (f"<b>{h['score']:.3f}</b><br>{html.escape(h['photo'])}<br>"
                       f"{int(q['size'])}px")
                parts.append(f"<figure class='{status}'><img src='{src}' "
                             f"title='{html.escape(h['photo'])} score {h['score']:.3f} "
                             f"thr {h['threshold_used']} blur {q['blur']:.0f} "
                             f"pose {q['pose']:.2f}'>"
                             f"<figcaption>{cap}</figcaption></figure>")
            parts.append("</div>")
        if not hits:
            parts.append("<p class='empty'>no photos matched</p>")
        parts.append("</div>")

    parts.append("</div>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(parts))
    return path
