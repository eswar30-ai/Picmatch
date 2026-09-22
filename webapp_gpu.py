"""Browser UI for the GPU pipeline.

Same idea as the original app: drop in a photo per person, get back the group
photos they appear in. Differences that matter:

  * the group library is indexed once on disk, so a search is a matrix multiply
    and returns instantly no matter how large the library gets
  * results are split into confident and borderline, with the score and the
    actual matched face shown - nothing is silently dropped, and a wrong match
    is visible rather than mysterious
  * the accept threshold is a slider on the page, so you can see the trade-off
    instead of guessing
"""

import base64
import io
import os
import threading

import cv2
import numpy as np
from flask import Flask, request, render_template_string, send_from_directory, abort, jsonify

import gpu_engine as ge
import face_index
import matcher

PHOTOS = os.environ.get("PMG_PHOTOS", "group_photos")
DATA = os.environ.get("PMG_DATA", "data")
PORT = int(os.environ.get("PMG_PORT", 5002))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

_engine = None
_index = None
_lock = threading.Lock()
_status = ""


def engine():
    global _engine
    with _lock:
        if _engine is None:
            _engine = ge.FaceEngine()
        return _engine


def index():
    global _index, _status
    with _lock:
        if _index is None:
            idx = face_index.FaceIndex(DATA)
            if not idx.load() or not idx.photos:
                _status = ("No index yet - run:  python pmg.py index --photos "
                           + PHOTOS)
            _index = idx
        return _index


def rebuild_index():
    global _index
    idx = face_index.FaceIndex(DATA)
    idx.load()
    idx.build(PHOTOS, engine(), progress=False)
    idx.save()
    with _lock:
        _index = idx
    return idx


def _thumb_data(photo, bbox, size=104):
    img = ge.load_image(os.path.join(PHOTOS, photo))
    if img is None:
        return None
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1, y1 = int(max(0, x1 - bw * .35)), int(max(0, y1 - bh * .35))
    x2, y2 = int(min(w, x2 + bw * .35)), int(min(h, y2 + bh * .35))
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return ("data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
            if ok else None)


PAGE = """
<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PicMatch GPU</title><style>
:root{--bg:#12100e;--card:#1c1916;--ink:#f2ede4;--soft:#a49b8c;--line:#2e2a25;
      --ok:#5fbf7a;--maybe:#e0a33a;--accent:#c8552d}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:15px/1.55 "Segoe UI",Inter,system-ui,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:44px 20px 80px}
h1{font-size:30px;margin:0 0 6px}
.sub{color:var(--soft);font-size:13px;margin:0 0 30px}
form{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:22px}
.drop{border:2px dashed var(--line);border-radius:8px;padding:34px;text-align:center;
      position:relative;color:var(--soft)}
.drop input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
.row{display:flex;gap:22px;align-items:center;margin-top:18px;flex-wrap:wrap}
label.k{font-size:12px;color:var(--soft);display:block;margin-bottom:4px}
input[type=range]{width:190px;accent-color:var(--accent)}
button{background:var(--accent);color:#fff;border:0;border-radius:6px;
       padding:12px 26px;font-size:15px;font-weight:600;cursor:pointer}
button:disabled{opacity:.45;cursor:not-allowed}
.note{color:var(--soft);font-size:12px;margin-top:12px}
.person{background:var(--card);border:1px solid var(--line);border-radius:10px;
        padding:16px 18px;margin-top:18px}
.person h2{font-size:18px;margin:0 0 2px}
.meta{color:var(--soft);font-size:12px;margin:0 0 12px}
.band{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--soft);
      margin:12px 0 8px;border-top:1px solid var(--line);padding-top:10px}
.grid{display:flex;flex-wrap:wrap;gap:10px}
figure{margin:0;width:104px}
figure img{width:104px;height:104px;object-fit:cover;border-radius:5px;
           border:2px solid transparent;display:block}
figure.match img{border-color:var(--ok)}figure.review img{border-color:var(--maybe)}
figcaption{font-size:10px;color:var(--soft);margin-top:4px;word-break:break-all}
figcaption b{color:var(--ink)}
a{color:inherit;text-decoration:none}
.banner{border-radius:8px;padding:12px 16px;font-size:13px;margin-bottom:20px;
        border:1px solid var(--line);background:var(--card)}
.banner.bad{border-color:var(--accent);color:#ffb59b}
</style></head><body><div class="wrap">
<h1>PicMatch <span style="color:var(--accent)">GPU</span></h1>
<p class="sub">{{ engine_line }} &middot; {{ n_photos }} photos indexed, {{ n_faces }} faces</p>
{% if status %}<div class="banner bad">{{ status }}</div>{% endif %}

<form method="post" action="/search" enctype="multipart/form-data">
  <div class="drop"><b>Drop reference photos here</b><br>
    one file per person, the file name becomes the name<br>
    <span style="font-size:12px">several photos of the same person? name them
    <code>Ravi_1.jpg</code>, <code>Ravi_2.jpg</code> - anything after the last
    underscore is ignored</span>
    <input type="file" name="photos" accept="image/*" multiple required
           onchange="document.getElementById('go').disabled=!this.files.length;
                     document.getElementById('n').textContent=this.files.length+' file(s)'">
  </div>
  <p class="note" id="n"></p>
  <div class="row">
    <div><label class="k">confident at <output id="ao">{{ accept }}</output></label>
      <input type="range" name="accept" min="0.25" max="0.60" step="0.01" value="{{ accept }}"
             oninput="ao.value=this.value"></div>
    <div><label class="k">show borderline from <output id="ro">{{ review }}</output></label>
      <input type="range" name="review" min="0.20" max="0.50" step="0.01" value="{{ review }}"
             oninput="ro.value=this.value"></div>
    <button id="go" type="submit" disabled>Search</button>
  </div>
  <p class="note">Green = confident. Amber = borderline, check it yourself.
     The number under each face is the cosine similarity that produced it.</p>
</form>

{% for block in results %}
<div class="person">
  <h2>{{ block.name }}</h2>
  <p class="meta">{{ block.n_match }} confident &middot; {{ block.n_review }} borderline
     &middot; best {{ block.best }}</p>
  {% for status, label, hits in block.groups %}
    {% if hits %}
    <div class="band">{{ label }} ({{ hits|length }})</div>
    <div class="grid">
    {% for h in hits %}
      <figure class="{{ status }}">
        <a href="/photo/{{ h.photo }}" target="_blank">
          <img src="{{ h.thumb }}" title="{{ h.photo }} - score {{ '%.3f'|format(h.score) }}">
        </a>
        <figcaption><b>{{ '%.3f'|format(h.score) }}</b><br>{{ h.photo }}</figcaption>
      </figure>
    {% endfor %}
    </div>
    {% endif %}
  {% endfor %}
  {% if not block.n_match and not block.n_review %}
    <p class="meta">nothing matched - try lowering the borderline slider</p>
  {% endif %}
</div>
{% endfor %}
</div></body></html>
"""


@app.route("/")
def home():
    idx = index()
    return render_template_string(
        PAGE, engine_line=engine().describe(), n_photos=len(idx.photos),
        n_faces=idx.total_faces(), status=_status, results=[],
        accept=matcher.ACCEPT, review=matcher.REVIEW)


@app.route("/search", methods=["POST"])
@app.route("/api/match", methods=["POST"])
def search():
    idx = index()
    eng = engine()
    api_response = request.path == "/api/match"
    accept = float(request.form.get("accept", matcher.ACCEPT))
    review = float(request.form.get("review", matcher.REVIEW))

    uploads = [f for f in request.files.getlist("photos") if f and f.filename]
    refs, names, missing = {}, [], []
    for f in uploads:
        stem = os.path.splitext(os.path.basename(f.filename))[0]
        name = stem.rsplit("_", 1)[0] if stem.rsplit("_", 1)[-1].isdigit() else stem
        face = eng.primary_face(f.read())
        if face is None:
            missing.append(f.filename)
            continue
        refs.setdefault(name, []).append(face["embedding"])

    if not refs:
        if api_response:
            return jsonify(error="No face found in any of the uploaded photos."), 400
        return render_template_string(
            PAGE, engine_line=eng.describe(), n_photos=len(idx.photos),
            n_faces=idx.total_faces(), results=[], accept=accept, review=review,
            status="No face found in: " + ", ".join(missing))

    names = sorted(refs)
    mats, owner = [], []
    for i, n in enumerate(names):
        mats.append(np.stack(refs[n]))
        owner.extend([i] * len(refs[n]))
    ref_mat = np.vstack(mats).astype(np.float32)
    result = matcher.match(idx, names, ref_mat, np.asarray(owner, np.int32),
                           accept=accept, review=review)

    blocks = []
    for name in names:
        hits = result["by_person"][name]
        seen, trimmed = set(), []
        for h in hits:                       # one entry per photo, best face wins
            if h["photo"] in seen:
                continue
            seen.add(h["photo"])
            trimmed.append(h)
        for h in trimmed[:120]:
            h["thumb"] = _thumb_data(h["photo"], h["bbox"])
        m = [h for h in trimmed[:120] if h["status"] == "match"]
        r = [h for h in trimmed[:120] if h["status"] == "review"]
        blocks.append({
            "name": name, "n_match": len(m), "n_review": len(r),
            "best": f"{trimmed[0]['score']:.3f}" if trimmed else "-",
            "groups": [("match", "confident", m),
                       ("review", "borderline - check these", r)],
        })

    if api_response:
      require_all = request.form.get("require_all") in ("1", "true", "on")
      by_photo = {}
      for name in names:
        for hit in result["by_person"][name]:
          if hit["status"] != "match":
            continue
          by_photo.setdefault(hit["photo"], set()).add(name)

      api_results = []
      for photo, matched_names in by_photo.items():
        if require_all and len(matched_names) != len(names):
          continue
        api_results.append({
          "filename": photo,
          "matched_names": sorted(matched_names),
          "original_url": None,
          "url": "/photo/" + photo,
        })
      return jsonify({
        "indexed_group_photos": len(idx.photos),
        "people": names,
        "require_all": require_all,
        "results": api_results,
        "skipped": missing,
      })

    note = ("No face detected in: " + ", ".join(missing)) if missing else ""
    return render_template_string(
        PAGE, engine_line=eng.describe(), n_photos=len(idx.photos),
        n_faces=idx.total_faces(), results=blocks, accept=accept,
        review=review, status=note)


@app.route("/photo/<path:name>")
def photo(name):
    folder = os.path.abspath(PHOTOS)
    target = os.path.abspath(os.path.join(folder, name))
    if not target.startswith(folder + os.sep) or not os.path.isfile(target):
        abort(404)
    return send_from_directory(folder, name)


@app.route("/rescan", methods=["POST"])
def rescan():
    rebuild_index()
    return home()


if __name__ == "__main__":
    print("PicMatch GPU on http://127.0.0.1:%d  (photos: %s)" % (PORT, PHOTOS))
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
