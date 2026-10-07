"""GPU face engine: SCRFD-10GF detection + ArcFace w600k_r50 embeddings.

Replaces the YuNet + SFace CPU pair used by the original app. Both models come
from InsightFace's `buffalo_l` pack and run through onnxruntime's CUDA provider,
so we can afford a much larger detection resolution - which is where most of the
"missing images" came from - and a far more discriminative embedding, which is
where the "wrong person in the group" came from.

Nothing here imports the `insightface` package: it needs a C compiler on Windows
and pins old numpy. We run the ONNX files directly instead.
"""

import io
import os
import threading

import cv2
import numpy as np
from PIL import Image, ImageOps

import cuda_dlls  # noqa: F401  - must precede onnxruntime, wires up the CUDA DLLs
import onnxruntime as ort

ort.set_default_logger_severity(3)

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_HERE, "models")
DET_MODEL = os.path.join(MODEL_DIR, "det_10g.onnx")
REC_MODEL = os.path.join(MODEL_DIR, "w600k_r50.onnx")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

# ------------------------------------------------------------------ defaults
# Detection long side. 640 is the InsightFace default; we run much higher
# because group photos put faces at 30-60px and 640 throws them away.
DET_SIZE = int(os.environ.get("PMG_DET_SIZE", 1280))
DET_THRESH = float(os.environ.get("PMG_DET_THRESH", 0.30))
NMS_THRESH = 0.4
MIN_FACE_PX = int(os.environ.get("PMG_MIN_FACE_PX", 20))

# how much better a rotated frame must score before we believe the photo is
# stored sideways rather than merely awkward
_ORIENT_MARGIN = 0.08

ENGINE_VERSION = "gpu-1"


class ModelError(RuntimeError):
    """Model files missing or the ONNX runtime refused to load them."""


# ---------------------------------------------------------------- providers
def available_providers(force_cpu=False):
    """Preferred execution providers, GPU first when one is usable."""
    if force_cpu:
        return ["CPUExecutionProvider"]
    have = set(ort.get_available_providers())
    order = []
    if "CUDAExecutionProvider" in have:
        cuda_opts = {
            "device_id": 0,
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "HEURISTIC",
            "do_copy_in_default_stream": True,
        }
        mem_limit = os.environ.get("PMG_GPU_MEM_LIMIT") or os.environ.get("PICMATCH_GPU_MEM_LIMIT")
        if mem_limit:
            try:
                cuda_opts["gpu_mem_limit"] = int(mem_limit)
            except ValueError:
                pass
        order.append(("CUDAExecutionProvider", cuda_opts))
    if "TensorrtExecutionProvider" in have and os.environ.get("PMG_TRT") == "1":
        order.insert(0, "TensorrtExecutionProvider")
    order.append("CPUExecutionProvider")
    return order


def _make_session(path, providers):
    if not os.path.isfile(path):
        raise ModelError(
            f"Missing model: {path}\nRun:  python fetch_models.py"
        )
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    try:
        return ort.InferenceSession(path, sess_options=opts, providers=providers)
    except Exception as exc:
        if providers != ["CPUExecutionProvider"]:
            return ort.InferenceSession(
                path, sess_options=opts, providers=["CPUExecutionProvider"])
        raise ModelError(f"Couldn't load {os.path.basename(path)}: {exc}") from exc


# ------------------------------------------------------------------- images
def load_image(source, max_dim=2560):
    """Path or bytes -> BGR uint8 array, EXIF orientation applied.

    Goes through PIL because cv2.imread silently returns None for non-ASCII
    Windows paths and ignores the EXIF orientation flag. Caps maximum dimension
    to avoid massive memory spikes from raw camera photos.
    """
    try:
        if isinstance(source, (bytes, bytearray)):
            pil = Image.open(io.BytesIO(bytes(source)))
        else:
            with open(source, "rb") as fh:
                pil = Image.open(io.BytesIO(fh.read()))
        pil = ImageOps.exif_transpose(pil).convert("RGB")
        w, h = pil.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            new_w, new_h = max(32, int(round(w * scale))), max(32, int(round(h * scale)))
            pil = pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
        return np.asarray(pil)[:, :, ::-1].copy()
    except Exception:
        try:
            raw = (np.frombuffer(bytes(source), np.uint8)
                   if isinstance(source, (bytes, bytearray))
                   else np.fromfile(source, np.uint8))
            img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if img is not None and max(img.shape[:2]) > max_dim:
                scale = max_dim / max(img.shape[:2])
                new_w, new_h = max(32, int(round(img.shape[1] * scale))), max(32, int(round(img.shape[0] * scale)))
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return img
        except Exception:
            return None


def list_images(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS))


# ------------------------------------------------------------------- SCRFD
def _distance2bbox(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points, distance):
    preds = []
    for i in range(0, distance.shape[1], 2):
        preds.append(points[:, 0] + distance[:, i])
        preds.append(points[:, 1] + distance[:, i + 1])
    return np.stack(preds, axis=-1)


def _nms(dets, thresh):
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= thresh)[0] + 1]
    return keep


class SCRFD:
    """SCRFD-10GF (det_10g.onnx) with 5 keypoints, 3 FPN levels, 2 anchors."""

    _STRIDES = (8, 16, 32)
    _NUM_ANCHORS = 2
    _FMC = 3

    def __init__(self, path=DET_MODEL, providers=None):
        self.session = _make_session(path, providers or available_providers())
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self._center_cache = {}

    def provider(self):
        return self.session.get_providers()[0]

    def _centers(self, height, width, stride):
        key = (height, width, stride)
        centers = self._center_cache.get(key)
        if centers is None:
            grid = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            centers = (grid * stride).reshape(-1, 2)
            if self._NUM_ANCHORS > 1:
                centers = np.stack([centers] * self._NUM_ANCHORS, axis=1).reshape(-1, 2)
            self._center_cache[key] = centers
        return centers

    def _forward(self, blob, in_h, in_w, thresh):
        outs = self.session.run(self.output_names, {self.input_name: blob})
        boxes_all, kps_all, scores_all = [], [], []
        for idx, stride in enumerate(self._STRIDES):
            scores = outs[idx].reshape(-1)
            bbox_preds = outs[idx + self._FMC].reshape(-1, 4) * stride
            kps_preds = outs[idx + self._FMC * 2].reshape(-1, 10) * stride
            centers = self._centers(in_h // stride, in_w // stride, stride)
            pos = np.where(scores >= thresh)[0]
            if not len(pos):
                continue
            boxes_all.append(_distance2bbox(centers[pos], bbox_preds[pos]))
            kps_all.append(_distance2kps(centers[pos], kps_preds[pos]).reshape(-1, 5, 2))
            scores_all.append(scores[pos])
        if not scores_all:
            return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
        return (np.vstack(boxes_all).astype(np.float32),
                np.vstack(kps_all).astype(np.float32),
                np.concatenate(scores_all).astype(np.float32))

    def detect(self, img, det_size=DET_SIZE, thresh=DET_THRESH, max_num=0):
        """Detect faces in a BGR image.

        Returns (boxes[N,5] as x1,y1,x2,y2,score  ,  kps[N,5,2]) in the
        coordinate space of the image passed in.
        """
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)

        # Aspect-matched letterbox rounded to a multiple of 32. Keeping the
        # aspect ratio (rather than a square canvas) spends every pixel of the
        # budget on real image instead of padding.
        scale = det_size / max(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        in_h = max(32, int(np.ceil(new_h / 32) * 32))
        in_w = max(32, int(np.ceil(new_w / 32) * 32))
        interp = cv2.INTER_LINEAR if scale <= 1 else cv2.INTER_CUBIC
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
        canvas = np.zeros((in_h, in_w, 3), np.uint8)
        canvas[:new_h, :new_w] = resized

        blob = cv2.dnn.blobFromImage(
            canvas, 1.0 / 128.0, (in_w, in_h), (127.5, 127.5, 127.5), swapRB=True)
        out = self._forward(blob, in_h, in_w, thresh)
        if len(out) == 2:
            return out
        boxes, kps, scores = out

        boxes /= scale
        kps /= scale
        dets = np.hstack([boxes, scores[:, None]]).astype(np.float32)
        keep = _nms(dets, NMS_THRESH)
        dets, kps = dets[keep], kps[keep]

        if max_num > 0 and len(dets) > max_num:
            area = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
            order = np.argsort(-area)[:max_num]
            dets, kps = dets[order], kps[order]
        return dets, kps


# ----------------------------------------------------------------- ArcFace
_ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def _umeyama(src, dst):
    """Least-squares similarity transform (scikit-image's algorithm, inlined)."""
    num, dim = src.shape
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_demean, dst_demean = src - src_mean, dst - dst_mean
    A = dst_demean.T @ src_demean / num
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1
    T = np.eye(dim + 1, dtype=np.float64)
    U, S, V = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.eye(dim + 1) * np.nan
    if rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V
    scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale
    return T


def norm_crop(img, kps, size=112):
    """Align a face to the canonical ArcFace 112x112 layout using its 5 points."""
    dst = _ARCFACE_DST * (size / 112.0)
    M = _umeyama(np.asarray(kps, np.float64), np.asarray(dst, np.float64))[:2]
    return cv2.warpAffine(img, M.astype(np.float32), (size, size),
                          borderValue=0.0, flags=cv2.INTER_LINEAR)


class ArcFace:
    """w600k_r50.onnx -> 512-d unit embedding."""

    def __init__(self, path=REC_MODEL, providers=None):
        self.session = _make_session(path, providers or available_providers())
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.size = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 112

    def provider(self):
        return self.session.get_providers()[0]

    def embed_batch(self, crops, flip=True, max_chunk=32):
        """crops: list of aligned BGR 112x112 images -> (N,512) unit vectors.

        Each face is embedded twice - as-is and mirrored - and the two are
        summed. Processed in sub-batches to cap peak VRAM allocation.
        """
        if not len(crops):
            return np.zeros((0, 512), np.float32)
        batch = list(crops)
        if flip:
            batch += [cv2.flip(c, 1) for c in crops]

        feats_chunks = []
        for i in range(0, len(batch), max_chunk):
            sub = batch[i:i + max_chunk]
            blob = cv2.dnn.blobFromImages(
                sub, 1.0 / 127.5, (self.size, self.size),
                (127.5, 127.5, 127.5), swapRB=True)
            f = self.session.run(self.output_names, {self.input_name: blob})[0]
            feats_chunks.append(np.asarray(f, np.float32))

        feats = np.vstack(feats_chunks) if feats_chunks else np.zeros((0, 512), np.float32)
        if flip:
            n = len(crops)
            feats = _l2(feats[:n]) + _l2(feats[n:])
        return _l2(feats)


def _l2(x):
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        return x / (np.linalg.norm(x) + 1e-9)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


# ------------------------------------------------------------------ quality
def face_quality(crop, box, det_score):
    """Cheap per-face quality signals used to suppress junk detections.

    A 20px motion-blurred face produces an embedding that is close to noise,
    and noise is exactly what lands on the wrong person at threshold.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    size = float(min(box[2] - box[0], box[3] - box[1]))
    return {"det_score": float(det_score), "size": size, "blur": blur}


def pose_offness(kps):
    """0 = frontal, grows with yaw/roll. Derived from the 5 keypoints."""
    le, re, nose = kps[0], kps[1], kps[2]
    eye_mid = (le + re) / 2.0
    eye_dist = float(np.linalg.norm(re - le)) + 1e-6
    yaw = abs(float(nose[0] - eye_mid[0])) / eye_dist
    roll = abs(float(np.arctan2(re[1] - le[1], re[0] - le[0])))
    return float(yaw + roll * 0.5)


# ------------------------------------------------------------------- engine
class Face(dict):
    """bbox, kps, det_score, embedding, quality - a plain dict with dot access."""

    __getattr__ = dict.get


class FaceEngine:
    def __init__(self, det_size=DET_SIZE, det_thresh=DET_THRESH,
                 min_face_px=MIN_FACE_PX, force_cpu=False, tiles=False,
                 det_scales=None):
        providers = available_providers(force_cpu)
        self.detector = SCRFD(providers=providers)
        self.recognizer = ArcFace(providers=providers)
        self.det_size = det_size
        self.det_thresh = det_thresh
        self.min_face_px = min_face_px
        self.tiles = tiles
        # SCRFD regresses boxes from fixed-size anchors, so its usable face-size
        # window is bounded at BOTH ends: a face filling a 1280px canvas is off
        # the top of the range and gets missed entirely, while a 40px face in a
        # crowd is off the bottom at 640. Running two scales and merging covers
        # close-up portraits and distant faces in the same pass.
        self.det_scales = tuple(det_scales) if det_scales else             tuple(sorted({det_size, 640}, reverse=True))
        self._lock = threading.RLock()

    def describe(self):
        return (f"{self.detector.provider()} | det_scales={list(self.det_scales)} "
                f"det_thresh={self.det_thresh} min_face={self.min_face_px}px "
                f"tiles={'on' if self.tiles else 'off'}")

    def _detect_multiscale(self, img):
        dets_all, kps_all = [], []
        for scale in self.det_scales:
            d, k = self.detector.detect(img, scale, self.det_thresh)
            if len(d):
                dets_all.append(d)
                kps_all.append(k)
        if not dets_all:
            return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
        dets, kps = np.vstack(dets_all), np.vstack(kps_all)
        keep = _nms(dets, NMS_THRESH)
        return dets[keep], kps[keep]

    # ---- detection with recall helpers ----
    def _detect_tiled(self, img):
        """Full-frame pass plus 2x2 overlapping tiles, merged with NMS.

        Each tile is detected at the full det_size budget, so faces get
        roughly twice the effective resolution. Only worth it on crowded or
        very large photos; costs ~5x the detection time.
        """
        dets, kps = self._detect_multiscale(img)
        h, w = img.shape[:2]
        oy, ox = int(h * 0.15), int(w * 0.15)
        boxes_all = [dets]
        kps_all = [kps]
        for ty in (0, 1):
            for tx in (0, 1):
                y0 = max(0, ty * h // 2 - oy)
                y1 = min(h, (ty + 1) * h // 2 + oy)
                x0 = max(0, tx * w // 2 - ox)
                x1 = min(w, (tx + 1) * w // 2 + ox)
                tile = img[y0:y1, x0:x1]
                if tile.size == 0:
                    continue
                d, k = self.detector.detect(tile, self.det_size, self.det_thresh)
                if not len(d):
                    continue
                d[:, [0, 2]] += x0
                d[:, [1, 3]] += y0
                k[:, :, 0] += x0
                k[:, :, 1] += y0
                boxes_all.append(d)
                kps_all.append(k)
        dets = np.vstack([b for b in boxes_all if len(b)]) if any(len(b) for b in boxes_all) \
            else np.zeros((0, 5), np.float32)
        kps = np.vstack([k for k in kps_all if len(k)]) if any(len(k) for k in kps_all) \
            else np.zeros((0, 5, 2), np.float32)
        if len(dets):
            keep = _nms(dets, 0.35)
            dets, kps = dets[keep], kps[keep]
        return dets, kps

    _ROTATIONS = ((0, None), (90, cv2.ROTATE_90_CLOCKWISE),
                  (180, cv2.ROTATE_180), (270, cv2.ROTATE_90_COUNTERCLOCKWISE))

    def detect_rotation(self, img):
        """How far clockwise this frame has to turn to sit upright.

        Returns (degrees, {degrees: (faces, mean_score)}). Cameras write plenty
        of sideways frames with no EXIF flag, and a sideways face is a face the
        detector never sees.
        """
        with self._lock:
            stats = {}
            for degrees, code in self._ROTATIONS:
                turned = img if code is None else cv2.rotate(img, code)
                dets, _ = self._detect_multiscale(turned)
                stats[degrees] = (len(dets),
                                  float(dets[:, 4].mean()) if len(dets) else 0.0)

            best_deg, best_n, best_conf = 0, *stats[0]
            for degrees in (90, 180, 270):
                n, conf = stats[degrees]
                if n > best_n or (n == best_n and n and conf > best_conf + _ORIENT_MARGIN):
                    best_deg, best_n, best_conf = degrees, n, conf
            return best_deg, stats

    def detect(self, img, auto_orient=True):
        """Detect faces, rotating the frame if the photo was stored sideways.

        Returns (image_used, dets, kps). Always crop from the returned image.
        """
        with self._lock:
            dets, kps = (self._detect_tiled(img) if self.tiles
                         else self._detect_multiscale(img))
            if len(dets) or not auto_orient:
                return img, dets, kps
            # nothing upright - the frame may be rotated with no EXIF flag
            for code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE,
                         cv2.ROTATE_180):
                turned = cv2.rotate(img, code)
                d, k = self._detect_multiscale(turned)
                if len(d):
                    return turned, d, k
            return img, dets, kps

    def analyze(self, source, auto_orient=True):
        """Path/bytes/array -> list of Face with embeddings. GPU-batched."""
        img = source if isinstance(source, np.ndarray) else load_image(source)
        if img is None:
            return []
        with self._lock:
            img, dets, kps = self.detect(img, auto_orient)
            if not len(dets):
                return []

            crops, kept = [], []
            for i, box in enumerate(dets):
                if min(box[2] - box[0], box[3] - box[1]) < self.min_face_px:
                    continue
                crop = norm_crop(img, kps[i], self.recognizer.size)
                if crop is None or crop.size == 0:
                    continue
                crops.append(crop)
                kept.append(i)
            if not crops:
                return []

            embeds = self.recognizer.embed_batch(crops)

        faces = []
        for slot, i in enumerate(kept):
            box = dets[i]
            q = face_quality(crops[slot], box, box[4])
            q["pose"] = pose_offness(kps[i])
            faces.append(Face(
                bbox=box[:4].astype(np.float32),
                det_score=float(box[4]),
                kps=kps[i].astype(np.float32),
                embedding=embeds[slot],
                quality=q,
            ))
        faces.sort(key=lambda f: -(f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
        return faces

    def primary_face(self, source, auto_orient=True):
        """Largest face - used for reference photos of a single person."""
        faces = self.analyze(source, auto_orient)
        return faces[0] if faces else None


# ----------------------------------------------------------------- scoring
def cosine(a, b):
    return float(np.dot(_l2(a), _l2(b)))


def usable_face(face, min_size=28.0, min_blur=8.0, min_det=0.45):
    """Is this face worth trusting for identity decisions?

    Tiny, near-flat, or barely-detected faces are the ones that produce
    coincidental high scores against strangers.
    """
    q = face["quality"]
    return q["size"] >= min_size and q["blur"] >= min_blur and q["det_score"] >= min_det
