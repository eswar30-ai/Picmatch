"""Face engine - GPU pipeline behind the original CPU-era interface.

The functions here are the same ones the rest of the project has always
called (`ensure_models`, `embed_primary_face`, `best_similarity`,
`EmbeddingCache`, ...) so `match_faces.py`, `gallery_search.py` and
`fix_orientation.py` keep working unchanged. What sits underneath them has
changed completely: YuNet + SFace on CPU are gone, replaced by SCRFD-10GF
detection and ArcFace w600k_r50 embeddings running on CUDA via onnxruntime.

Two consequences worth knowing about:

  * embeddings are 512-d ArcFace vectors, not 128-d SFace ones, so the cosine
    scale moved - the accept line lives around 0.42 now (see matcher.py), not
    0.51. `is_match` already reads the new default.
  * old embeddings_cache.json entries are stamped with the engine version and
    are therefore ignored, not misread. The first run re-embeds; on GPU that
    costs seconds.

The CPU implementation is kept next door in face_engine_cpu_backup.py.
"""

import hashlib
import json
import os
import threading

import numpy as np

import gpu_engine as ge
import matcher

# ============================= CONFIG =================================
IMAGE_EXTS = ge.IMAGE_EXTS
DET_SIZE = ge.DET_SIZE
DET_THRESHOLD = ge.DET_THRESH
MIN_FACE_PX = ge.MIN_FACE_PX
# ArcFace cosine scale: same person 0.45-0.85, different people below 0.25.
COSINE_THRESHOLD = float(os.environ.get("PICMATCH_THRESHOLD", matcher.ACCEPT))
AUTO_ORIENT = os.environ.get("PICMATCH_AUTO_ORIENT", "1") != "0"
FORCE_CPU = os.environ.get("PICMATCH_FORCE_CPU", "0") == "1"
ENGINE_VERSION = ge.ENGINE_VERSION
# ========================================================================

ModelError = ge.ModelError

_engine = None
_engine_lock = threading.Lock()


# ------------------------------------------------------------------ models
def engine():
    """The one shared FaceEngine. Loads both ONNX models on first use."""
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is None:
            _engine = ge.FaceEngine(force_cpu=FORCE_CPU)
        return _engine


def ensure_models():
    """Load the models once. Raises ModelError if they're missing."""
    eng = engine()
    return eng.detector, eng.recognizer


def settings_summary():
    try:
        return (f"{engine().describe()} threshold={COSINE_THRESHOLD} "
                f"auto_orient={'on' if AUTO_ORIENT else 'off'}")
    except ModelError as exc:
        return f"engine not loaded: {exc}"


# ------------------------------------------------------------------ images
load_image = ge.load_image
list_images = ge.list_images


def detect_faces(img, auto_orient=None):
    """(working_image, faces) - crop from the image handed back, not the input.

    `faces` are the raw SCRFD rows (x1, y1, x2, y2, score) in the coordinate
    space of the returned image.
    """
    if img is None:
        return None, np.zeros((0, 5), np.float32)
    orient = AUTO_ORIENT if auto_orient is None else auto_orient
    work, dets, _kps = engine().detect(img, auto_orient=orient)
    return work, dets


def detect_upright_rotation(img):
    """(degrees_clockwise, per-rotation stats) for a photo stored sideways."""
    if img is None:
        return 0, {}
    return engine().detect_rotation(img)


def rotate_clockwise(img, degrees):
    """Rotate a BGR array by 0/90/180/270 degrees clockwise."""
    code = dict(ge.FaceEngine._ROTATIONS).get(degrees % 360)
    return img if code is None else ge.cv2.rotate(img, code)


# -------------------------------------------------------------- embeddings
def embed_all_faces(source):
    """Every face in the photo -> list of 512-d unit vectors, largest first."""
    return [f["embedding"] for f in engine().analyze(source, auto_orient=AUTO_ORIENT)]


def embed_primary_face(source):
    """The largest face only - for a reference photo of one person."""
    face = engine().primary_face(source, auto_orient=AUTO_ORIENT)
    return None if face is None else face["embedding"]


def similarity(a, b):
    return float(np.dot(np.asarray(a, np.float32).ravel(),
                        np.asarray(b, np.float32).ravel()))


def best_similarity(ref, candidates):
    if candidates is None or not len(candidates):
        return 0.0
    return float(np.max(np.asarray(candidates, np.float32)
                        @ np.asarray(ref, np.float32).ravel()))


def is_match(score, threshold=None):
    return score >= (COSINE_THRESHOLD if threshold is None else threshold)


# ------------------------------------------------------------ embedding cache
class EmbeddingCache:
    """Per-photo embeddings on disk, keyed by file name.

    An entry is reused only while the file's size/mtime and the engine
    settings that produced it are unchanged, so switching engines or detector
    settings can never serve a stale vector.
    """

    def __init__(self, path):
        self.path = path
        self._data = {}
        self._lock = threading.Lock()
        self.load()

    @staticmethod
    def _stamp(file_path):
        st = os.stat(file_path)
        key = (f"{ENGINE_VERSION}|{DET_SIZE}|{DET_THRESHOLD}|{MIN_FACE_PX}|"
               f"{AUTO_ORIENT}|{st.st_size}|{int(st.st_mtime)}")
        return hashlib.sha1(key.encode()).hexdigest()

    def load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self._data = {
                k: {"stamp": v["stamp"],
                    "embeddings": [np.asarray(e, np.float32) for e in v["embeddings"]]}
                for k, v in raw.items()
            }
        except Exception:
            self._data = {}

    def save(self):
        try:
            with self._lock:
                raw = {k: {"stamp": v["stamp"],
                           "embeddings": [e.tolist() for e in v["embeddings"]]}
                       for k, v in self._data.items()}
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(raw, fh)
            os.replace(tmp, self.path)
        except Exception:
            pass

    def get(self, name, file_path):
        try:
            stamp = self._stamp(file_path)
        except OSError:
            return []
        with self._lock:
            hit = self._data.get(name)
            if hit is not None and hit["stamp"] == stamp:
                return hit["embeddings"]
        embeds = embed_all_faces(file_path)
        with self._lock:
            self._data[name] = {"stamp": stamp, "embeddings": embeds}
        return embeds

    def drop(self, name):
        with self._lock:
            self._data.pop(name, None)

    def prune(self, keep_names):
        keep = set(keep_names)
        with self._lock:
            for gone in [k for k in self._data if k not in keep]:
                del self._data[gone]
