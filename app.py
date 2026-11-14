"""PicMatch - find every group photo a person appears in.

Same app as always: same routes, same JSON, same folders. What changed is the
engine underneath. YuNet + SFace on CPU are gone; detection is SCRFD-10GF and
embeddings are ArcFace w600k_r50, both running through onnxruntime's CUDA
provider (see gpu_engine.py). The group library lives in an on-disk face index
under data/ instead of embeddings_cache.json, so a search is one matrix
multiply however large the library grows, and matching goes through matcher.py
- each detected face is claimed by at most one person, and small, blurry or
side-on faces have to clear a stricter line before they are believed.
"""

import argparse
import csv
import io
import ipaddress
import json
import os
import threading
import time

import numpy as np
import requests
from flask import Flask, request, render_template_string, send_from_directory, jsonify, send_file, abort
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps

import gpu_engine as ge
import face_index
import matcher
from fix_orientation import EXIF_ORIENTATION_TAG, pil_to_bgr, rotate_pil_clockwise


GROUP_PHOTO_FOLDER = os.environ.get("PICMATCH_PHOTOS", "group_photos")
DATA_DIR = os.environ.get("PICMATCH_DATA", "data")
GALLERY_INDEX_FILE = os.environ.get("PICMATCH_GALLERY_INDEX", "gallery_index.json")
API_URL = os.environ.get("PICMATCH_GALLERY_API", "https://services.heterohcl.com/dfs-icon/api/gallery?summit_id=1")
FETCH_INTERVAL_SECONDS = int(os.environ.get("PICMATCH_FETCH_INTERVAL", 15 * 60))
AUTO_SYNC_GALLERY = os.environ.get("PICMATCH_AUTO_SYNC", "1") != "0"
SINGLE_PERSON_FOLDER = os.environ.get("PICMATCH_SINGLE_PERSON", "single_person")
IMAGE_EXTS = ge.IMAGE_EXTS
# ArcFace cosine scale - nothing to do with the old 0.51 SFace line.
ACCEPT = float(os.environ.get("PICMATCH_THRESHOLD", matcher.ACCEPT))
REVIEW = float(os.environ.get("PICMATCH_REVIEW", matcher.REVIEW))
INDEX_DELAY_SECONDS = float(os.environ.get("PICMATCH_INDEX_DELAY", "0.8"))
# ========================================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 48 * 1024 * 1024

model_error = None
gallery_index = {}
gallery_lock = threading.RLock()
_engine = None
_index = None
_engine_lock = threading.RLock()
_index_lock = threading.RLock()
_gallery_thread = None
_gallery_thread_lock = threading.RLock()
_gallery_sync_lock = threading.RLock()
_gallery_sync_in_progress = False

_user_activity_lock = threading.Lock()
_active_user_requests = 0
_user_activity_event = threading.Event()
_user_activity_event.set()


class UserGPUContext:
    """Signals that an interactive user request is using the GPU.
    Background indexing checks this and yields immediately to avoid GPU crashes and 502s."""
    def __enter__(self):
        global _active_user_requests
        with _user_activity_lock:
            _active_user_requests += 1
            _user_activity_event.clear()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _active_user_requests
        with _user_activity_lock:
            _active_user_requests = max(0, _active_user_requests - 1)
            if _active_user_requests == 0:
                _user_activity_event.set()


def engine():
    """The one shared GPU engine. Loads both ONNX models on first use."""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = ge.FaceEngine()
        return _engine


def ensure_models():
    """Load the two ONNX models once, lazily. Never crashes the app if
    they're missing - just reports the problem on the page instead."""
    global model_error
    try:
        engine()
        model_error = None
        return True
    except ge.ModelError as exc:
        model_error = str(exc)
    except Exception as exc:
        model_error = f"Couldn't start the GPU engine: {exc}"
    return False


def face_db():
    """The saved face index for GROUP_PHOTO_FOLDER, loaded on first use."""
    global _index
    with _index_lock:
        if _index is None:
            idx = face_index.FaceIndex(DATA_DIR)
            idx.load()
            _index = idx
        return _index


def save_index():
    with _index_lock:
        try:
            face_db().save()
        except Exception:
            app.logger.exception("could not write the face index")


def total_indexed_photos():
    """Number of photos in the index that have at least one detected face."""
    with _index_lock:
        idx = face_db()
        return len([p for p in idx.photos if idx.faces.get(p)])


def total_indexed_faces():
    """Total number of detected faces across all photos in the index."""
    with _index_lock:
        idx = face_db()
        return idx.total_faces()


def index_photo(fname):
    """Compute face embeddings for one photo already on disk in
    GROUP_PHOTO_FOLDER and fold them into the index. Returns True if at least
    one face was found."""
    path = os.path.join(GROUP_PHOTO_FOLDER, fname)
    if not os.path.isfile(path):
        return False
    eng = engine()
    try:
        faces = eng.analyze(path)
    except Exception:
        app.logger.exception("could not index %s", fname)
        return False

    with _index_lock:
        idx = face_db()
        idx.folder = idx.folder or GROUP_PHOTO_FOLDER
        idx.params = idx.params or face_index.params_for(eng)
        idx.faces[fname] = [{"embedding": f["embedding"], "bbox": f["bbox"],
                             "quality": f["quality"]} for f in faces]
        try:
            idx.stamps[fname] = face_index._stamp(path, idx.params)
        except OSError:
            idx.stamps.pop(fname, None)
        if fname not in idx.photos:
            idx.photos = sorted(set(idx.photos) | {fname})

    return bool(faces)


def rebuild_index():
    """(Re)scan GROUP_PHOTO_FOLDER and index the faces in every photo.

    Photos whose file hasn't changed since last time are carried over from the
    saved index, so restarts are near-instant and adding 20 photos to a
    5000-photo library costs 20 photos of work.
    """
    global model_error
    if not os.path.isdir(GROUP_PHOTO_FOLDER):
        model_error = f"Folder '{GROUP_PHOTO_FOLDER}' doesn't exist. Edit GROUP_PHOTO_FOLDER at the top of app.py."
        return
    if not ensure_models():
        return
    with _index_lock:
        idx = face_db()
        try:
            idx.build(GROUP_PHOTO_FOLDER, engine(), progress=False)
            idx.save()
        except SystemExit as exc:      # folder holds no images
            model_error = str(exc)
            return

build_group_cache = rebuild_index  # alias for backwards compatibility


def load_gallery_index():
    global gallery_index
    try:
        if os.path.isfile(GALLERY_INDEX_FILE):
            with open(GALLERY_INDEX_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                with gallery_lock:
                    gallery_index.clear()
                    gallery_index.update(data)
        else:
            with gallery_lock:
                gallery_index.clear()
    except Exception:
        with gallery_lock:
            gallery_index.clear()


def fetch_gallery_json(url=None):
    target = url or API_URL
    resp = requests.get(target, timeout=20)
    resp.raise_for_status()
    return resp.json()


def download_gallery_image(url, dest_folder=GROUP_PHOTO_FOLDER):
    os.makedirs(dest_folder, exist_ok=True)
    filename = os.path.basename(url.split("?")[0])
    dest_path = os.path.join(dest_folder, filename)
    if os.path.isfile(dest_path):
        return dest_path, False
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(1024 * 8):
            if chunk:
                f.write(chunk)
    return dest_path, True


download_image = download_gallery_image  # alias for backwards compatibility


def process_pending_indexing(photo_names, delay=INDEX_DELAY_SECONDS):
    """Paced background indexing of photos with cooperative yielding to user requests.
    Prevents GPU lock contention, thermal throttling, WDDM resets, and 502 Bad Gateway."""
    if not photo_names:
        return 0

    indexed_count = 0
    total = len(photo_names)
    app.logger.info("Starting throttled background indexing for %d photo(s) (cooldown=%.2fs)...", total, delay)

    for i, fname in enumerate(photo_names, 1):
        # 1. Cooperative yield: If any user request is active or arrives, pause indexing immediately!
        while _active_user_requests > 0:
            app.logger.info("Background indexing paused: user request active. Yielding GPU...")
            _user_activity_event.wait(timeout=2.0)

        # 2. Safely index single photo on GPU with error isolation
        try:
            if index_photo(fname):
                indexed_count += 1
            if indexed_count % 5 == 0 or i == total:
                save_index()
        except Exception as e:
            app.logger.warning("Error indexing photo %s: %s (skipping and continuing)", fname, e)

        # 3. Cooldown pause between images to keep GPU utilization and VRAM stable
        time.sleep(delay)

        # 4. Periodically free unreferenced memory
        if i % 10 == 0:
            import gc
            gc.collect()

    save_index()
    app.logger.info("Throttled background indexing completed: %d/%d photos indexed with faces.",
                    indexed_count, total)
    return indexed_count


def update_gallery_index(auto_index=True):
    """Fetch remote gallery images, download new ones into group_photos,
    update gallery_index.json, and optionally index new faces into face_db."""
    global gallery_index
    try:
        data = fetch_gallery_json(API_URL)
    except Exception as e:
        app.logger.warning("Failed to fetch gallery JSON: %s", e)
        return {"status": False, "error": str(e), "new_downloaded": 0, "new_indexed": 0}

    if not data or not data.get("status"):
        app.logger.warning("Gallery API did not return success: %s", data)
        return {"status": False, "error": "API returned non-success", "new_downloaded": 0, "new_indexed": 0}

    image_entries = []
    days = data.get("data", {}).get("days", [])
    for day in days:
        for img in day.get("images", []):
            url = img.get("image_url") or img.get("image_path")
            if url:
                image_entries.append({"original_url": url})

    if not image_entries:
        return {"status": True, "new_downloaded": 0, "new_indexed": 0, "total_images": len(gallery_index)}

    new_downloaded = []
    new_entries = []
    for entry in image_entries:
        url = entry["original_url"]
        try:
            dest_path, is_new = download_gallery_image(url, GROUP_PHOTO_FOLDER)
            fname = os.path.basename(dest_path)
            new_entries.append((fname, url, dest_path))
            if is_new:
                new_downloaded.append(fname)
                app.logger.info("  downloaded new gallery photo: %s", fname)
        except Exception as e:
            app.logger.error("Failed to download %s: %s", url, e)

    if new_entries:
        with gallery_lock:
            for fname, url, dest_path in new_entries:
                gallery_index[fname] = {"original_url": url, "local_path": dest_path}
            try:
                with open(GALLERY_INDEX_FILE, "w", encoding="utf-8") as f:
                    json.dump(gallery_index, f, indent=2)
            except Exception as e:
                app.logger.error("Failed to save gallery index file: %s", e)

    new_indexed = 0
    if new_downloaded and auto_index and ensure_models():
        new_indexed = process_pending_indexing(new_downloaded, delay=INDEX_DELAY_SECONDS)
        app.logger.info("Gallery sync: %d new photos downloaded, %d indexed with faces",
                        len(new_downloaded), new_indexed)

    return {
        "status": True,
        "total_images": len(gallery_index),
        "new_downloaded": len(new_downloaded),
        "new_indexed": new_indexed
    }


def run_gallery_sync(auto_index=True):
    """Run update_gallery_index with single-flight concurrency lock."""
    global _gallery_sync_in_progress
    with _gallery_sync_lock:
        if _gallery_sync_in_progress:
            app.logger.info("Gallery sync already in progress, skipping redundant trigger.")
            return {
                "status": True,
                "message": "Gallery sync already in progress",
                "in_progress": True,
                "total_images": len(gallery_index)
            }
        _gallery_sync_in_progress = True

    try:
        return update_gallery_index(auto_index=auto_index)
    finally:
        with _gallery_sync_lock:
            _gallery_sync_in_progress = False


def _periodic_gallery_fetch_loop():
    try:
        run_gallery_sync(auto_index=True)
    except Exception:
        app.logger.exception("Initial gallery sync encountered an error")

    while True:
        time.sleep(FETCH_INTERVAL_SECONDS)
        try:
            run_gallery_sync(auto_index=True)
        except Exception:
            app.logger.exception("Periodic gallery sync encountered an error")


def start_gallery_background_sync():
    global _gallery_thread
    with _gallery_thread_lock:
        if _gallery_thread is None or not _gallery_thread.is_alive():
            _gallery_thread = threading.Thread(target=_periodic_gallery_fetch_loop, daemon=True)
            _gallery_thread.start()
            app.logger.info("Background gallery sync started (interval: %ds)", FETCH_INTERVAL_SECONDS)


def batch_match_single_person(folder=SINGLE_PERSON_FOLDER, accept=ACCEPT, review=REVIEW):
    """Offline batch match for single_person reference photos against indexed group photos.
    Uses GPU/FaceIndex directly and saves results_by_person.csv and results_by_photo.csv.
    Does NOT use deprecated embeddings_cache.json."""
    if not os.path.isdir(folder):
        print(f"Folder '{folder}' does not exist.")
        return False
    if not ensure_models():
        print("Models could not be loaded:", model_error)
        return False

    eng = engine()
    people = {}
    skipped = []
    print(f"Loading reference people from {folder}...")
    for fname in ge.list_images(folder):
        name = os.path.splitext(fname)[0]
        face = eng.primary_face(os.path.join(folder, fname))
        if face is None:
            skipped.append(fname)
            continue
        people.setdefault(name, []).append(face["embedding"])

    if not people:
        print("No usable reference faces found.")
        return False

    names = sorted(people.keys())
    mats, owner = [], []
    for i, name in enumerate(names):
        vectors = np.stack(people[name]).astype(np.float32)
        mats.append(vectors)
        owner.extend([i] * len(vectors))

    idx = face_db()
    result = matcher.match(idx, names, np.vstack(mats), np.asarray(owner, np.int32),
                           accept=accept, review=review)

    with open("results_by_person.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["person", "num_group_photos", "group_photo_filenames"])
        for name in names:
            matched_photos = sorted({h["photo"] for h in result["by_person"][name] if h["status"] == "match"})
            writer.writerow([name, len(matched_photos), ", ".join(matched_photos)])

    with open("results_by_photo.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group_photo", "people_present"])
        for photo in idx.photos:
            matched_names = sorted({h["person"] for h in result["by_photo"].get(photo, []) if h["status"] == "match"})
            writer.writerow([photo, ", ".join(matched_names)])

    print(f"Saved results_by_person.csv and results_by_photo.csv for {len(names)} people.")
    return True


def reference_embeddings(files):
    """Turn uploaded reference photos into one identity per person.

    Keyed by filename without extension, so Ravi.jpg enrols "Ravi". Several
    shots of one person can be uploaded as Ravi_1.jpg, Ravi_2.jpg and are
    merged into a single identity - more angles per person is the cheapest
    accuracy gain there is. Uses the largest face in each photo, so a
    bystander in the background can't take over someone's identity.
    """
    people, skipped = {}, []
    eng = engine()
    for f in files:
        stem = os.path.splitext(os.path.basename(f.filename))[0]
        head, _, tail = stem.rpartition("_")
        name = head if head and tail.isdigit() else stem
        face = eng.primary_face(f.read())
        if face is None:
            skipped.append(f.filename)
            continue
        people.setdefault(name, []).append(face["embedding"])
    return people, skipped


def find_matches(people, require_all, accept=None, review=None):
    """[(filename, [names]), ...] for every indexed photo containing someone.

    Scoring goes through matcher.match, so each indexed face is claimed by at
    most one person - the best-scoring one, and only when it beats the
    runner-up by a margin. That is what stops one ambiguous face being handed
    to three different people at once.
    """
    names = sorted(people)
    if not names:
        return []
    mats, owner = [], []
    for i, name in enumerate(names):
        vectors = np.stack(people[name]).astype(np.float32)
        mats.append(vectors)
        owner.extend([i] * len(vectors))

    result = matcher.match(face_db(), names, np.vstack(mats),
                           np.asarray(owner, np.int32),
                           accept=ACCEPT if accept is None else accept,
                           review=REVIEW if review is None else review)

    by_photo = {}
    for name in names:
        for hit in result["by_person"][name]:
            if hit["status"] == "match":
                by_photo.setdefault(hit["photo"], set()).add(name)

    results = [(photo, sorted(found)) for photo, found in by_photo.items()
               if not (require_all and len(found) < len(names))]
    # photos with more of the uploaded people together float to the top
    results.sort(key=lambda r: (-len(r[1]), r[0]))
    return results


def settings_summary():
    try:
        return (f"{engine().describe()} accept={ACCEPT} review={REVIEW} "
                f"margin={matcher.MARGIN}")
    except Exception as exc:
        return f"engine not loaded: {exc}"


def unique_path(folder, filename):
    """Return a save path that won't overwrite an existing file, appending
    _1, _2, ... to the name if needed."""
    base, ext = os.path.splitext(filename)
    candidate = filename
    i = 1
    while os.path.isfile(os.path.join(folder, candidate)):
        candidate = f"{base}_{i}{ext}"
        i += 1
    return os.path.join(folder, candidate)

def get_uploaded_files(field_name="photos"):

    app.logger.debug("request.files keys: %s", list(request.files.keys()))
    files = [f for f in request.files.getlist(field_name) if f and f.filename]
    if not files:
        # fallback to any uploaded files (useful for single-file uploads
        # where clients use a different field name)
        files = [f for f in request.files.values() if f and f.filename]
    return files


def count_group_photos():
    if not os.path.isdir(GROUP_PHOTO_FOLDER):
        return 0
    return len([f for f in os.listdir(GROUP_PHOTO_FOLDER) if f.lower().endswith(IMAGE_EXTS)])


# ---------------------------------------------------------------- templates
INDEX_HTML = """
<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Light Table &mdash; Photo Match</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap" rel="stylesheet">
<style>
  :root{--bg:#EAE7E0;--surface:#FFFFFF;--ink:#211D19;--ink-soft:#706A5E;--accent:#C81D25;--accent-soft:rgba(200,29,37,.08);--line:#D8D3C7}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:'Inter',sans-serif;min-height:100vh}
  .wrap{max-width:640px;margin:0 auto;padding:64px 24px 80px}
  .eyebrow{font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:.12em;color:var(--ink-soft);text-transform:uppercase;margin:0 0 14px}
  .eyebrow::before{content:"\\25CF ";color:var(--accent)}
  h1{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:clamp(32px,5vw,44px);line-height:1.05;margin:0 0 12px;letter-spacing:-.01em}
  .sub{color:var(--ink-soft);font-size:16px;line-height:1.5;margin:0 0 40px;max-width:46ch}
  .dropzone{position:relative;border:2px dashed var(--line);border-radius:4px;background:var(--surface);padding:48px 24px;text-align:center;transition:border-color .15s ease,background .15s ease}
  .dropzone.drag{border-color:var(--accent);background:var(--accent-soft)}
  .dropzone p{margin:0 0 4px;font-size:15px}
  .dropzone .hint{color:var(--ink-soft);font-size:13px}
  .dropzone input[type=file]{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:pointer}
  #filename{font-family:'IBM Plex Mono',monospace;font-size:13px;color:var(--accent);margin-top:14px;min-height:18px}
  button{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:15px;background:var(--ink);color:var(--surface);border:none;padding:14px 28px;border-radius:4px;cursor:pointer;margin-top:24px;width:100%}
  button:hover{background:var(--accent)}
  button:focus-visible{outline:3px solid var(--accent);outline-offset:2px}
  button:disabled{opacity:.5;cursor:not-allowed}
  .status{display:flex;align-items:center;gap:8px;font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--ink-soft);margin-top:28px;letter-spacing:.02em}
  .status .count{color:var(--ink);font-weight:600}
  .rescan-row{margin-top:6px}
  .rescan-row button{all:unset;cursor:pointer;font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--ink-soft);text-decoration:underline;text-underline-offset:3px}
  .rescan-row button:hover{color:var(--accent)}
  .banner{font-family:'IBM Plex Mono',monospace;font-size:13px;padding:12px 16px;border-radius:4px;margin-bottom:24px}
  .banner.error{background:var(--accent-soft);color:var(--accent);border:1px solid var(--accent)}
  .banner.ok{background:#eef4ec;color:#2f5e3f;border:1px solid #b9d3bd}
  .check-row{display:flex;align-items:center;gap:9px;margin-top:18px;font-size:14px;color:var(--ink-soft);cursor:pointer;user-select:none}
  .check-row input{width:16px;height:16px;accent-color:var(--accent);cursor:pointer}
</style></head><body>
<div class="wrap">
  <p class="eyebrow">Light Table / Photo Match</p>
  <h1>Find the frames<br>they're in.</h1>
  <p class="sub">Upload one photo per person &mdash; select several at once to check for a whole group in the same shot.</p>

  {% if error %}<div class="banner error">{{ error }}</div>{% endif %}
  {% if message %}<div class="banner ok">{{ message }}</div>{% endif %}

  <form action="/match" method="post" enctype="multipart/form-data" id="uploadForm">
    <div class="dropzone" id="dropzone">
      <p>Drop photos here, or click to browse</p>
      <p class="hint">JPG, PNG, or WEBP &middot; select multiple to check several people</p>
      <input type="file" name="photos" accept="image/*" id="fileInput" required multiple>
    </div>
    <p id="filename"></p>
    <label class="check-row"><input type="checkbox" name="require_all"> <span>Only show photos with everyone together</span></label>
    <button type="submit" id="submitBtn" disabled>Find matches</button>
  </form>

  <div class="status"><span>&#9670; auto-indexed:</span> <span class="count">{{ num_cached }}</span> <span>group photo(s)</span></div>
  <div class="rescan-row"><a href="/admin" style="color:var(--ink-soft);text-decoration:underline;text-underline-offset:3px">+ add group photos</a> &middot; <a href="/docs" style="color:var(--ink-soft);text-decoration:underline;text-underline-offset:3px">API docs</a></div>
</div>
<script>
  const dz = document.getElementById('dropzone');
  const input = document.getElementById('fileInput');
  const nameEl = document.getElementById('filename');
  const btn = document.getElementById('submitBtn');
  function showNames(files){
    if(!files || !files.length) return;
    if(files.length === 1){ nameEl.textContent = '\\u2192 ' + files[0].name; }
    else { nameEl.textContent = '\\u2192 ' + files.length + ' photos: ' + Array.from(files).map(f => f.name).join(', '); }
    btn.disabled = false;
  }
  input.addEventListener('change', () => showNames(input.files));
  ['dragenter','dragover'].forEach(evt => dz.addEventListener(evt, e => { e.preventDefault(); dz.classList.add('drag'); }));
  ['dragleave','drop'].forEach(evt => dz.addEventListener(evt, e => { e.preventDefault(); dz.classList.remove('drag'); }));
  dz.addEventListener('drop', e => { if(e.dataTransfer.files.length){ input.files = e.dataTransfer.files; showNames(input.files); } });
</script>
</body></html>
"""

ADMIN_HTML = """
<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Add photos &mdash; Light Table</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap" rel="stylesheet">
<style>
  :root{--bg:#EAE7E0;--surface:#FFFFFF;--ink:#211D19;--ink-soft:#706A5E;--accent:#C81D25;--accent-soft:rgba(200,29,37,.08);--line:#D8D3C7}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:'Inter',sans-serif;min-height:100vh}
  .wrap{max-width:640px;margin:0 auto;padding:64px 24px 80px}
  .eyebrow{font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:.12em;color:var(--ink-soft);text-transform:uppercase;margin:0 0 14px}
  .eyebrow::before{content:"\\25CF ";color:var(--accent)}
  h1{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:clamp(32px,5vw,44px);line-height:1.05;margin:0 0 12px;letter-spacing:-.01em}
  .sub{color:var(--ink-soft);font-size:16px;line-height:1.5;margin:0 0 40px;max-width:46ch}
  .dropzone{position:relative;border:2px dashed var(--line);border-radius:4px;background:var(--surface);padding:48px 24px;text-align:center;transition:border-color .15s ease,background .15s ease}
  .dropzone.drag{border-color:var(--accent);background:var(--accent-soft)}
  .dropzone p{margin:0 0 4px;font-size:15px}
  .dropzone .hint{color:var(--ink-soft);font-size:13px}
  .dropzone input[type=file]{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:pointer}
  #filename{font-family:'IBM Plex Mono',monospace;font-size:13px;color:var(--accent);margin-top:14px;min-height:18px}
  button{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:15px;background:var(--ink);color:var(--surface);border:none;padding:14px 28px;border-radius:4px;cursor:pointer;margin-top:24px;width:100%}
  button:hover{background:var(--accent)}
  button:focus-visible{outline:3px solid var(--accent);outline-offset:2px}
  button:disabled{opacity:.5;cursor:not-allowed}
  .status{display:flex;align-items:center;gap:8px;font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--ink-soft);margin-top:28px;letter-spacing:.02em}
  .status .count{color:var(--ink);font-weight:600}
  .banner{font-family:'IBM Plex Mono',monospace;font-size:13px;padding:12px 16px;border-radius:4px;margin-bottom:24px}
  .banner.error{background:var(--accent-soft);color:var(--accent);border:1px solid var(--accent)}
  .banner.ok{background:#eef4ec;color:#2f5e3f;border:1px solid #b9d3bd}
  a.back{display:inline-block;margin-top:28px;color:var(--ink-soft);font-family:'IBM Plex Mono',monospace;font-size:13px;text-decoration:none;border-bottom:1px solid var(--line);padding-bottom:2px}
  a.back:hover{color:var(--accent);border-color:var(--accent)}
</style></head><body>
<div class="wrap">
  <p class="eyebrow">Light Table / Admin</p>
  <h1>Add group photos</h1>
  <p class="sub">Upload new pictures straight into your group_photos library. They're indexed right away, so people can be matched against them immediately &mdash; no separate rescan needed.</p>

  {% if error %}<div class="banner error">{{ error }}</div>{% endif %}
  {% if message %}<div class="banner ok">{{ message }}</div>{% endif %}

  <form action="/admin/upload" method="post" enctype="multipart/form-data" id="uploadForm">
    <div class="dropzone" id="dropzone">
      <p>Drop group photos here, or click to browse</p>
      <p class="hint">JPG, PNG, or WEBP &middot; select as many as you like</p>
      <input type="file" name="photos" accept="image/*" id="fileInput" required multiple>
    </div>
    <p id="filename"></p>
    <button type="submit" id="submitBtn" disabled>Upload to library</button>
  </form>

  <div class="status"><span>&#9670; on disk:</span> <span class="count">{{ total_files }}</span> <span>photo(s), {{ num_cached }} indexed</span></div>
  <a class="back" href="/">&larr; back to search</a>
</div>
<script>
  const dz = document.getElementById('dropzone');
  const input = document.getElementById('fileInput');
  const nameEl = document.getElementById('filename');
  const btn = document.getElementById('submitBtn');
  function showNames(files){
    if(!files || !files.length) return;
    if(files.length === 1){ nameEl.textContent = '\\u2192 ' + files[0].name; }
    else { nameEl.textContent = '\\u2192 ' + files.length + ' photos: ' + Array.from(files).map(f => f.name).join(', '); }
    btn.disabled = false;
  }
  input.addEventListener('change', () => showNames(input.files));
  ['dragenter','dragover'].forEach(evt => dz.addEventListener(evt, e => { e.preventDefault(); dz.classList.add('drag'); }));
  ['dragleave','drop'].forEach(evt => dz.addEventListener(evt, e => { e.preventDefault(); dz.classList.remove('drag'); }));
  dz.addEventListener('drop', e => { if(e.dataTransfer.files.length){ input.files = e.dataTransfer.files; showNames(input.files); } });
</script>
</body></html>
"""

RESULTS_HTML = """
<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Matches &mdash; Light Table</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap" rel="stylesheet">
<style>
  :root{--bg:#EAE7E0;--surface:#FFFFFF;--ink:#211D19;--ink-soft:#706A5E;--accent:#C81D25;--accent-soft:rgba(200,29,37,.08);--line:#D8D3C7}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:'Inter',sans-serif}
  .wrap{max-width:1040px;margin:0 auto;padding:56px 24px 80px}
  .eyebrow{font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:.12em;color:var(--ink-soft);text-transform:uppercase;margin:0 0 14px}
  .eyebrow::before{content:"\\25CF ";color:var(--accent)}
  h1{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:clamp(26px,4vw,36px);margin:0 0 8px;letter-spacing:-.01em}
  .meta{color:var(--ink-soft);font-size:14px;margin:0 0 6px;font-family:'IBM Plex Mono',monospace}
  .meta b{color:var(--ink)}
  .meta.warn{color:var(--accent)}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:26px;margin-top:34px}

  .frame{background:var(--surface);border-radius:4px;border:1px solid var(--line);position:relative;opacity:0;animation:rise .4s ease forwards;cursor:pointer}
  .frame:hover{border-color:var(--ink-soft)}
  .frame:focus-visible{outline:3px solid var(--accent);outline-offset:3px}
  .frame:nth-child(1){animation-delay:.02s}.frame:nth-child(2){animation-delay:.06s}.frame:nth-child(3){animation-delay:.10s}
  .frame:nth-child(4){animation-delay:.14s}.frame:nth-child(5){animation-delay:.18s}.frame:nth-child(6){animation-delay:.22s}
  .frame:nth-child(n+7){animation-delay:.24s}
  @keyframes rise{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}

  .thumb{overflow:hidden;border-radius:4px 4px 0 0;position:relative}
  .thumb img{width:100%;height:200px;object-fit:cover;display:block;transition:transform .2s ease}
  .frame:hover .thumb img{transform:scale(1.04)}
  .expand-hint{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:rgba(33,29,25,0);transition:background .15s ease}
  .frame:hover .expand-hint{background:rgba(33,29,25,.15)}
  .expand-hint svg{opacity:0;transition:opacity .15s ease}
  .frame:hover .expand-hint svg{opacity:1}

  .stamp{padding:10px 12px 12px;font-family:'IBM Plex Mono',monospace;border-top:1px solid var(--line);border-radius:0 0 4px 4px}
  .stamp .fname{display:block;font-size:12px;color:var(--ink);font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin-bottom:8px}
  .tags{display:flex;flex-wrap:wrap;gap:5px}
  .tag{background:var(--accent-soft);color:var(--accent);padding:2px 9px;border-radius:20px;font-size:11px}
  .tag.tag-full{background:var(--accent);color:#fff}

  .empty{border:2px dashed var(--line);border-radius:4px;padding:64px 24px;text-align:center;color:var(--ink-soft);font-family:'IBM Plex Mono',monospace;font-size:14px;margin-top:34px}
  a.back{display:inline-block;margin-top:44px;color:var(--ink-soft);font-family:'IBM Plex Mono',monospace;font-size:13px;text-decoration:none;border-bottom:1px solid var(--line);padding-bottom:2px}
  a.back:hover{color:var(--accent);border-color:var(--accent)}
  a.back:focus-visible{outline:3px solid var(--accent);outline-offset:4px}
  @media (prefers-reduced-motion:reduce){.frame{opacity:1;animation:none}.thumb img{transition:none}}

  /* ---- lightbox ---- */
  .lightbox{display:none;position:fixed;inset:0;background:rgba(20,17,15,.94);z-index:1000;flex-direction:column}
  .lightbox.open{display:flex}
  .lb-toolbar{display:flex;align-items:center;gap:12px;padding:14px 20px;font-family:'IBM Plex Mono',monospace;color:#EAE7E0;font-size:13px}
  .lb-toolbar button,.lb-toolbar a{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.25);color:#EAE7E0;width:36px;height:36px;border-radius:4px;display:flex;align-items:center;justify-content:center;cursor:pointer;text-decoration:none;font-size:19px;line-height:1}
  .lb-toolbar button:hover,.lb-toolbar a:hover{background:var(--accent);border-color:var(--accent)}
  .lb-toolbar button:focus-visible,.lb-toolbar a:focus-visible{outline:2px solid #fff;outline-offset:2px}
  #lbZoomLevel{min-width:44px;text-align:center}
  #lbClose{margin-left:auto;font-size:22px}
  .lb-stage{flex:1;overflow:hidden;display:flex;align-items:center;justify-content:center;cursor:grab}
  .lb-stage.dragging{cursor:grabbing}
  .lb-stage img{max-width:90vw;max-height:80vh;user-select:none;-webkit-user-drag:none}
  .lb-caption{text-align:center;color:var(--ink-soft);font-family:'IBM Plex Mono',monospace;font-size:12px;padding:6px 0 18px}
</style></head><body>
<div class="wrap">
  <p class="eyebrow">Light Table / Photo Match</p>
  <h1>Matches</h1>
  <p class="meta">Checking for: <b>{{ people|join(', ') }}</b></p>
  {% if skipped %}<p class="meta warn">Couldn't find a face in: {{ skipped|join(', ') }} &mdash; skipped</p>{% endif %}
  {% if require_all %}<p class="meta">Showing only photos with everyone together</p>{% endif %}

  {% if results %}
    <p class="meta">{{ results|length }} group photo(s), sorted by how many of them are in each</p>
    <div class="grid">
      {% for fname, names in results %}
      <div class="frame" data-full="/group_photo/{{ fname }}" data-name="{{ fname }}" tabindex="0" role="button" aria-label="View {{ fname }} full size">
        <div class="thumb">
          <img src="/group_photo/{{ fname }}" alt="{{ fname }}" loading="lazy">
          <div class="expand-hint"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2"><path d="M9 3H5a2 2 0 0 0-2 2v4M15 3h4a2 2 0 0 1 2 2v4M9 21H5a2 2 0 0 1-2-2v-4M15 21h4a2 2 0 0 0 2-2v-4"/></svg></div>
        </div>
        <div class="stamp">
          <span class="fname">{{ fname }}</span>
          <span class="tags">
            {% for n in names %}<span class="tag{% if names|length == people|length %} tag-full{% endif %}">{{ n }}</span>{% endfor %}
          </span>
        </div>
      </div>
      {% endfor %}
    </div>
  {% else %}
    <div class="empty">No group photo matches {{ 'all of them together' if require_all else 'any of them' }}.<br>Try clearer, front-facing reference photos{% if require_all %}, or untick "everyone together"{% endif %}.</div>
  {% endif %}
  <a class="back" href="/">&larr; try again</a>
</div>

<div class="lightbox" id="lightbox">
  <div class="lb-toolbar">
    <button id="lbZoomOut" aria-label="Zoom out">&minus;</button>
    <span id="lbZoomLevel">100%</span>
    <button id="lbZoomIn" aria-label="Zoom in">+</button>
    <a id="lbDownload" aria-label="Download this photo">&#8595;</a>
    <button id="lbClose" aria-label="Close">&times;</button>
  </div>
  <div class="lb-stage" id="lbStage"><img id="lbImg" src="" alt=""></div>
  <p class="lb-caption" id="lbCaption"></p>
</div>

<script>
(function(){
  const lightbox = document.getElementById('lightbox');
  const lbImg = document.getElementById('lbImg');
  const lbStage = document.getElementById('lbStage');
  const zoomLevelEl = document.getElementById('lbZoomLevel');
  const lbDownload = document.getElementById('lbDownload');
  const lbCaption = document.getElementById('lbCaption');
  let scale = 1, tx = 0, ty = 0, dragging = false, startX = 0, startY = 0;

  function applyTransform(){
    lbImg.style.transform = 'translate(' + tx + 'px, ' + ty + 'px) scale(' + scale + ')';
    zoomLevelEl.textContent = Math.round(scale * 100) + '%';
  }
  function resetView(){ scale = 1; tx = 0; ty = 0; applyTransform(); }
  function zoomBy(delta){
    scale = Math.min(Math.max(scale + delta, 1), 4);
    if (scale === 1) { tx = 0; ty = 0; }
    applyTransform();
  }
  function openLightbox(frame){
    lbImg.src = frame.dataset.full;
    lbCaption.textContent = frame.dataset.name;
    lbDownload.href = frame.dataset.full;
    lbDownload.setAttribute('download', frame.dataset.name);
    resetView();
    lightbox.classList.add('open');
  }
  function closeLightbox(){ lightbox.classList.remove('open'); }

  document.querySelectorAll('.frame').forEach(frame => {
    frame.addEventListener('click', () => openLightbox(frame));
    frame.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openLightbox(frame); }
    });
  });

  document.getElementById('lbZoomIn').addEventListener('click', () => zoomBy(0.25));
  document.getElementById('lbZoomOut').addEventListener('click', () => zoomBy(-0.25));
  document.getElementById('lbClose').addEventListener('click', closeLightbox);
  lightbox.addEventListener('click', e => { if (e.target === lightbox) closeLightbox(); });

  lbStage.addEventListener('wheel', e => {
    e.preventDefault();
    zoomBy(e.deltaY < 0 ? 0.15 : -0.15);
  }, { passive: false });

  lbStage.addEventListener('mousedown', e => {
    if (scale <= 1) return;
    dragging = true; lbStage.classList.add('dragging');
    startX = e.clientX - tx; startY = e.clientY - ty;
  });
  window.addEventListener('mousemove', e => {
    if (!dragging) return;
    tx = e.clientX - startX; ty = e.clientY - startY;
    applyTransform();
  });
  window.addEventListener('mouseup', () => { dragging = false; lbStage.classList.remove('dragging'); });

  window.addEventListener('keydown', e => {
    if (!lightbox.classList.contains('open')) return;
    if (e.key === 'Escape') closeLightbox();
    if (e.key === '+' || e.key === '=') zoomBy(0.25);
    if (e.key === '-') zoomBy(-0.25);
  });
})();
</script>
</body></html>
"""


# --------------------------------------------------------------------- routes
@app.route("/")
def index():
    err = model_error if not ensure_models() else None
    return render_template_string(INDEX_HTML, num_cached=total_indexed_photos(), error=err, message=None)


@app.route("/rescan", methods=["POST", "GET"])
def rescan():
    """Deprecated: indexing is now fully automatic in the background."""
    msg = f"Automatic background indexing is active — {total_indexed_photos()} photos currently indexed."
    return render_template_string(INDEX_HTML, num_cached=total_indexed_photos(), error=None, message=msg)


@app.route("/admin")
def admin():
    return render_template_string(ADMIN_HTML, num_cached=total_indexed_photos(),
                                   total_files=count_group_photos(), error=None, message=None)


@app.route("/admin/upload", methods=["POST"])
def admin_upload():
    with UserGPUContext():
        os.makedirs(GROUP_PHOTO_FOLDER, exist_ok=True)

        files = [f for f in request.files.getlist("photos") if f and f.filename]
        if not files:
            return render_template_string(ADMIN_HTML, num_cached=total_indexed_photos(),
                                           total_files=count_group_photos(),
                                           error="Choose at least one photo to upload.", message=None)

        saved, rejected, straightened = [], [], []
        for f in files:
          safe_name = secure_filename(f.filename)
          if not safe_name or not safe_name.lower().endswith(IMAGE_EXTS):
            rejected.append(f.filename)
            continue
          dest = unique_path(GROUP_PHOTO_FOLDER, safe_name)
       
          try:
            img_pil = ImageOps.exif_transpose(Image.open(f.stream))
            turn = 0
            if ensure_models():
              turn, _ = engine().detect_rotation(pil_to_bgr(img_pil))
            if turn:
              img_pil = rotate_pil_clockwise(img_pil, turn)
              straightened.append(f"{os.path.basename(dest)} ({turn}°)")
            exif = img_pil.getexif()
            exif.pop(EXIF_ORIENTATION_TAG, None)  
            save_kwargs = {"exif": exif.tobytes()} if len(exif) else {}
            img_pil.convert("RGB").save(dest, **save_kwargs)
          except Exception:
       
            app.logger.exception("could not re-save %s upright; storing it as uploaded", safe_name)
            straightened = [s for s in straightened if not s.startswith(os.path.basename(dest))]
            f.stream.seek(0)
            with open(dest, "wb") as out:
              out.write(f.stream.read())
          saved.append(os.path.basename(dest))

        indexed = 0
        if saved and ensure_models():
            for fname in saved:
                if index_photo(fname):
                    indexed += 1
            save_index()

        parts = []
        if saved:
            parts.append(f"Added {len(saved)} photo(s): {', '.join(saved)} ({indexed} indexed, "
                          f"{len(saved) - indexed} had no detectable face but are still saved)")
        if straightened:
            parts.append(f"Straightened sideways photo(s): {', '.join(straightened)}")
        if rejected:
            parts.append(f"Skipped (not an image file): {', '.join(rejected)}")

        return render_template_string(
            ADMIN_HTML, num_cached=total_indexed_photos(), total_files=count_group_photos(),
            message=" · ".join(parts) if saved else None,
            error=None if saved else " · ".join(parts) if parts else "Nothing was uploaded.",
        )


@app.route("/match", methods=["POST"])
def match():
    with UserGPUContext():
        if not ensure_models():
            return render_template_string(INDEX_HTML, num_cached=total_indexed_photos(), error=model_error, message=None)

        if total_indexed_photos() == 0:
            rebuild_index()

        files = get_uploaded_files("photos")
        if not files:
          return render_template_string(INDEX_HTML, num_cached=total_indexed_photos(),
                           error="Choose at least one photo first.", message=None)

        require_all = request.form.get("require_all") == "on"

        people, skipped = reference_embeddings(files)
        if not people:
            return render_template_string(INDEX_HTML, num_cached=total_indexed_photos(),
                                           error="No face found in any of those photos - try clearer, front-facing shots.",
                                           message=None)

        results = find_matches(people, require_all)

        return render_template_string(RESULTS_HTML, results=results, people=sorted(people.keys()),
                                       skipped=skipped, require_all=require_all)


@app.route("/api/match", methods=["POST"])
def api_match():
    with UserGPUContext():
        if not ensure_models():
            return jsonify(error=model_error), 500

        if total_indexed_photos() == 0:
            rebuild_index()

        files = get_uploaded_files("photos")
        if not files:
            return jsonify(error="Choose at least one photo first.", received_keys=list(request.files.keys())), 400

        require_all = request.form.get("require_all") == "on"

        people, skipped = reference_embeddings(files)
        if not people:
            return jsonify(error="No face found in any of those photos - try clearer, front-facing shots."), 400

        with gallery_lock:
            gallery_copy = dict(gallery_index)

        results = []
        for fname, matched_names in find_matches(people, require_all):
            orig_entry = gallery_copy.get(fname)
            orig = orig_entry.get("original_url") if isinstance(orig_entry, dict) else None
            results.append({
                "filename": fname,
                "url": f"/group_photo/{fname}",
                "original_url": orig,
                "matched_names": matched_names,
            })

        return jsonify({
            "people": sorted(people.keys()),
            "skipped": skipped,
            "require_all": require_all,
            "results": results,
            "indexed_group_photos": total_indexed_photos(),
        })


@app.route("/gallery", methods=["GET"])
def gallery_endpoint():
    """Gallery endpoint synced from gallery_search.py on single port."""
    with gallery_lock:
        return jsonify({"status": True, "images": dict(gallery_index)})


@app.route("/api/gallery/sync", methods=["POST", "GET"])
def gallery_sync():
    """Trigger manual gallery synchronization and indexing (non-blocking)."""
    with _gallery_sync_lock:
        is_running = _gallery_sync_in_progress

    if is_running:
        return jsonify({
            "status": True,
            "message": "Gallery sync already running in background",
            "sync_running": True,
            "total_gallery_images": len(gallery_index),
            "indexed_photos": total_indexed_photos()
        })

    threading.Thread(target=run_gallery_sync, kwargs={"auto_index": True}, daemon=True).start()
    return jsonify({
        "status": True,
        "message": "Gallery sync started in background",
        "sync_running": True,
        "total_gallery_images": len(gallery_index),
        "indexed_photos": total_indexed_photos()
    })


@app.route("/api/gallery/status", methods=["GET"])
def gallery_status():
    """Get current synchronization and indexing status."""
    with _gallery_sync_lock:
        is_running = _gallery_sync_in_progress
    with gallery_lock:
        total_gal = len(gallery_index)
    return jsonify({
        "status": True,
        "sync_running": is_running,
        "total_gallery_images": total_gal,
        "indexed_photos": total_indexed_photos(),
        "total_faces": total_indexed_faces()
    })


@app.route("/group_photo/<path:filename>")
def group_photo(filename):
  path = os.path.join(GROUP_PHOTO_FOLDER, filename)
  if not os.path.isfile(path):
    abort(404)
  try:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    buf = io.BytesIO()
    fmt = img.format or "JPEG"
    img.save(buf, format=fmt)
    buf.seek(0)
    return send_file(buf, mimetype=f"image/{fmt.lower()}")
  except Exception:
    return send_from_directory(GROUP_PHOTO_FOLDER, filename)


DOCS_HTML = """
<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>API Docs &mdash; Light Table</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap" rel="stylesheet">
<style>
  :root{--bg:#EAE7E0;--surface:#FFFFFF;--ink:#211D19;--ink-soft:#706A5E;--accent:#C81D25;--line:#D8D3C7}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:'Inter',sans-serif;min-height:100vh}
  .wrap{max-width:760px;margin:0 auto;padding:64px 24px 80px}
  h1{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:clamp(32px,5vw,44px);margin:0 0 16px}
  .section{margin-top:32px}
  .section h2{font-size:20px;margin:0 0 12px}
  .section p, .section pre, .section ul{margin:0 0 16px;line-height:1.6}
  .card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:22px;}
  pre{background:rgba(33,29,25,.04);padding:16px;border-radius:8px;overflow-x:auto}
  code{font-family:'IBM Plex Mono',monospace;background:rgba(33,29,25,.06);padding:.15em .35em;border-radius:.35em}
  a{color:var(--accent);text-decoration:none}
  a:hover{text-decoration:underline}
</style>
</head><body>
<div class="wrap">
  <h1>Light Table API Docs</h1>
  <p>Quick reference for the built-in routes exposed by this photo matching app.</p>

  <div class="section card">
    <h2>GET /</h2>
    <p>Loads the main search page where you can upload reference photos and find matching group photos.</p>
  </div>

  <div class="section card">
    <h2>POST /match</h2>
    <p>Upload one or more reference photos to search the indexed group photos.</p>
    <p>Form fields:</p>
    <ul>
      <li><code>photos</code> — one or more image files. The file name is the person's name; several shots of one person can be sent as <code>Ravi_1.jpg</code>, <code>Ravi_2.jpg</code> and are merged into one identity</li>
      <li><code>require_all</code> — optional checkbox; when present, only returns photos containing every uploaded person</li>
    </ul>
  </div>

  <div class="section card">
    <h2>GET /api/gallery/status</h2>
    <p>Returns current background sync and GPU indexing status along with total photo and face counts. Group photos are automatically embedded in the background.</p>
  </div>

  <div class="section card">
    <h2>GET /gallery</h2>
    <p>Returns JSON dictionary of all synchronized remote gallery images mapped to their local paths and original URLs.</p>
  </div>

  <div class="section card">
    <h2>POST /api/gallery/sync</h2>
    <p>Triggers asynchronous synchronization with the remote summit gallery API and throttled background GPU indexing.</p>
  </div>


  <div class="section card">
    <h2>GET /admin</h2>
    <p>Loads the admin page for uploading new group photos into the app library.</p>
  </div>

  <div class="section card">
    <h2>POST /admin/upload</h2>
    <p>Upload group photos to be saved into <code>group_photos/</code> and indexed automatically.</p>
    <p>Form fields:</p>
    <ul>
      <li><code>photos</code> — one or more image files</li>
    </ul>
  </div>

  <div class="section card">
    <h2>GET /group_photo/&lt;filename&gt;</h2>
    <p>Serves a specific stored group photo by filename.</p>
  </div>

  <div class="section card">
    <h2>POST /api/match</h2>
    <p>Returns a JSON-only result set with group photo matches, original remote URLs, and local image URLs.</p>
    <p>Form fields:</p>
    <ul>
      <li><code>photos</code> — one or more image files. The file name is the person's name; several shots of one person can be sent as <code>Ravi_1.jpg</code>, <code>Ravi_2.jpg</code> and are merged into one identity</li>
      <li><code>require_all</code> — optional checkbox field; include if you want only photos containing every uploaded person</li>
    </ul>
  </div>

  <div class="section card">
    <h2>GET /docs</h2>
    <p>This page.</p>
  </div>

  <div class="section">
    <h2>Example cURL</h2>
    <pre>curl -F "photos=@person1.jpg" -F "photos=@person2.jpg" http://127.0.0.1:5001/api/match</pre>
  </div>
</div>
</body></html>
"""


@app.route("/docs")
def docs():
    return render_template_string(DOCS_HTML)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Photo match web app & gallery service")
    parser.add_argument("--host", default=os.environ.get("PICMATCH_HOST", "0.0.0.0"),
                        help="Host/IP address to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PICMATCH_PORT", os.environ.get("PORT", 5001))),
                        help="Port to listen on (default: 5001)")
    parser.add_argument("--no-gallery-sync", action="store_true",
                        help="Disable automatic background gallery sync")
    parser.add_argument("--idf-port", type=int, default=int(os.environ.get("IDF_PORT", 5008)),
                        help="Port for the IDF audit service started alongside (default: 5008)")
    parser.add_argument("--no-idf", action="store_true",
                        help="Do not start the IDF audit service (idf_audit/) alongside Picmatch")
    args = parser.parse_args(argv)

    try:
        ipaddress.ip_address(args.host)
    except ValueError:
        parser.error(f"Invalid host address: {args.host}")

    if not (1 <= args.port <= 65535 and 1 <= args.idf_port <= 65535):
        parser.error("Port must be between 1 and 65535")
    if not args.no_idf and args.idf_port == args.port:
        parser.error("Picmatch and IDF need different ports")

    return args


def start_idf(host, port):
    """Start the IDF audit service (idf_audit/) beside Picmatch, in its own
    environment - the two can never share one (idf_audit/requirements.txt).
    Returns the running service, or None when it is not started here."""
    if os.environ.get("PICMATCH_START_IDF") == "0":       # run.py starts it itself
        return None
    import run as launcher

    if launcher.port_taken(port):
        print(f"IDF audit: port {port} is already in use - using what is running there, not starting another.")
        return None
    python = launcher.ROOT / "idf_audit" / ".venv" / launcher.PYTHON
    if not python.is_file():
        print(f"IDF audit: not started - no environment at {python.parent.parent} (see idf_audit/README.md).")
        return None
    service = launcher.Service("idf", python, ["run.py", "--host", host, "--port", str(port)],
                               launcher.ROOT / "idf_audit")
    service.start()
    print(f"IDF audit: its log is {launcher.LOGS}\\idf-<date>.log")
    return service


def main(argv=None):
    args = parse_args(argv)
    idf = None if args.no_idf else start_idf(args.host, args.idf_port)
    try:
        serve(args, idf)
    finally:
        if idf is not None:
            idf.stop(wait=15)        # Ctrl+C reaches it too; this waits for it to finish


def serve(args, idf):

    print("Loading models and initializing index ...")
    print(settings_summary())
    if ensure_models():
        idx = face_db()
        if not idx.photos:
            rebuild_index()
        load_gallery_index()
        faces = total_indexed_faces()
        photos = total_indexed_photos()
        print(f"Loaded index: {photos} group photo(s), {faces} face(s).")
    else:
        print(f"Warning: {model_error}")
        print("The app will still start - fix the issue above, then click 'rescan' on the page.")

    if not args.no_gallery_sync and AUTO_SYNC_GALLERY:
        start_gallery_background_sync()

    print(f"\nOpen this address in your browser: http://{args.host}:{args.port}\n")
    print(f"Endpoints: Web UI: http://{args.host}:{args.port}/ | Gallery: http://{args.host}:{args.port}/gallery | Docs: http://{args.host}:{args.port}/docs\n")
    if idf is not None:
        print(f"IDF audit: http://{args.host}:{args.idf_port}/api/health | POST http://{args.host}:{args.idf_port}/api/audit\n")
    app.run(debug=False, port=args.port, host=args.host)


if __name__ == "__main__":
    main()
