"""Build and hold the face index for a folder of group photos.

One row per detected face: a 512-d ArcFace embedding plus the quality signals
we use later to decide whether that face is trustworthy enough to name.

The index is incremental - photos whose size/mtime haven't changed are carried
over from the previous run, so adding 20 photos to a 5000-photo library costs
20 photos of work.
"""

import concurrent.futures as cf
import json
import os
import time

import numpy as np

import gpu_engine as ge

INDEX_NPZ = "index.npz"
INDEX_JSON = "index.json"


def params_for(engine):
    """The settings fingerprint stored beside the index.

    Anything in here changing the embeddings must appear in this string, or a
    stale vector produced under different settings would be served as fresh.
    """
    return (f"det{list(engine.det_scales)}|thr{engine.det_thresh}|"
            f"min{engine.min_face_px}|tiles{int(engine.tiles)}")


def _stamp(path, params):
    st = os.stat(path)
    return f"{ge.ENGINE_VERSION}|{params}|{st.st_size}|{int(st.st_mtime)}"


class FaceIndex:
    """Embeddings for every face in a photo folder, keyed by file name."""

    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.folder = None
        self.photos = []            # file names, index-aligned with per-photo lists
        self.stamps = {}            # name -> stamp
        self.faces = {}             # name -> list of dicts (embedding + quality)
        self.params = ""

    # ------------------------------------------------------------- persistence
    @property
    def npz_path(self):
        return os.path.join(self.data_dir, INDEX_NPZ)

    @property
    def json_path(self):
        return os.path.join(self.data_dir, INDEX_JSON)

    def load(self):
        try:
            with open(self.json_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            blob = np.load(self.npz_path)
        except Exception:
            return False
        embeds = blob["embeddings"].astype(np.float32)
        bboxes = blob["bboxes"].astype(np.float32)
        quals = blob["quality"].astype(np.float32)   # det_score, size, blur, pose
        counts = meta["counts"]
        self.folder = meta.get("folder")
        self.params = meta.get("params", "")
        self.stamps = meta.get("stamps", {})
        self.photos = meta["photos"]
        self.faces = {}
        at = 0
        for name, n in zip(self.photos, counts):
            rows = []
            for j in range(at, at + n):
                rows.append({
                    "embedding": embeds[j],
                    "bbox": bboxes[j],
                    "quality": {"det_score": float(quals[j, 0]), "size": float(quals[j, 1]),
                                "blur": float(quals[j, 2]), "pose": float(quals[j, 3])},
                })
            self.faces[name] = rows
            at += n
        return True

    def save(self):
        os.makedirs(self.data_dir, exist_ok=True)
        names = self.photos
        counts = [len(self.faces.get(n, [])) for n in names]
        rows = [f for n in names for f in self.faces.get(n, [])]
        embeds = (np.stack([r["embedding"] for r in rows]) if rows
                  else np.zeros((0, 512), np.float32))
        bboxes = (np.stack([r["bbox"] for r in rows]) if rows
                  else np.zeros((0, 4), np.float32))
        quals = np.array([[r["quality"]["det_score"], r["quality"]["size"],
                           r["quality"]["blur"], r["quality"]["pose"]] for r in rows],
                         np.float32) if rows else np.zeros((0, 4), np.float32)
        tmp_npz = os.path.join(self.data_dir, "index_tmp.npz")
        tmp_json = os.path.join(self.data_dir, "index_tmp.json")
        np.savez_compressed(tmp_npz, embeddings=embeds,
                            bboxes=bboxes, quality=quals)
        with open(tmp_json, "w", encoding="utf-8") as fh:
            json.dump({"folder": self.folder, "params": self.params,
                       "photos": names, "counts": counts, "stamps": self.stamps}, fh)
        os.replace(tmp_npz, self.npz_path)
        os.replace(tmp_json, self.json_path)

    # ------------------------------------------------------------------ build
    def build(self, folder, engine, workers=4, progress=True, force=False):
        """(Re)index `folder`. Returns (indexed, reused, failed, total_faces)."""
        names = ge.list_images(folder)
        if not names:
            raise SystemExit(f"No images found in {folder}")

        params = params_for(engine)
        if force or params != self.params or os.path.abspath(folder) != (
                os.path.abspath(self.folder) if self.folder else None):
            self.faces, self.stamps = {}, {}
        self.folder = folder
        self.params = params

        todo = []
        for name in names:
            path = os.path.join(folder, name)
            try:
                stamp = _stamp(path, params)
            except OSError:
                continue
            if self.stamps.get(name) == stamp and name in self.faces:
                continue
            todo.append((name, path, stamp))

        reused = len(names) - len(todo)
        failed, total_new = [], 0
        started = time.time()

        # Decode on worker threads (JPEG decode is the CPU bottleneck), run the
        # nets on the main thread so the GPU sees one clean stream of work.
        def _decode(item):
            name, path, stamp = item
            return name, stamp, ge.load_image(path)

        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            for i, (name, stamp, img) in enumerate(pool.map(_decode, todo), 1):
                if img is None:
                    failed.append(name)
                    continue
                try:
                    faces = engine.analyze(img)
                except Exception as exc:
                    failed.append(f"{name} ({exc})")
                    continue
                self.faces[name] = [
                    {"embedding": f["embedding"], "bbox": f["bbox"], "quality": f["quality"]}
                    for f in faces
                ]
                self.stamps[name] = stamp
                total_new += len(faces)
                if progress and (i % 10 == 0 or i == len(todo)):
                    rate = i / max(1e-6, time.time() - started)
                    eta = (len(todo) - i) / max(1e-6, rate)
                    print(f"\r  indexed {i}/{len(todo)}  {rate:.1f} img/s  "
                          f"eta {eta / 60:.1f}m  faces so far {total_new}   ",
                          end="", flush=True)
        if progress and todo:
            print()

        keep = set(names)
        self.faces = {k: v for k, v in self.faces.items() if k in keep}
        self.stamps = {k: v for k, v in self.stamps.items() if k in keep}
        self.photos = names
        return len(todo) - len(failed), reused, failed, self.total_faces()

    # ------------------------------------------------------------------ query
    def total_faces(self):
        return sum(len(v) for v in self.faces.values())

    def matrix(self):
        """(embeddings[N,512], owners[N] photo names, rows[N] face dicts)."""
        owners, rows = [], []
        for name in self.photos:
            for f in self.faces.get(name, []):
                owners.append(name)
                rows.append(f)
        if not rows:
            return np.zeros((0, 512), np.float32), [], []
        return np.stack([r["embedding"] for r in rows]), owners, rows
