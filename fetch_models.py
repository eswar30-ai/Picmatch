"""Download the InsightFace buffalo_l models into ./models.

We only need two of the five files in the pack:
  det_10g.onnx    SCRFD-10GF face detector with 5 keypoints
  w600k_r50.onnx  ArcFace ResNet50 trained on WebFace600K -> 512-d embeddings
"""

import os
import sys
import zipfile
import shutil
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
NEEDED = ("det_10g.onnx", "w600k_r50.onnx")

URLS = [
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    "https://huggingface.co/immich-app/antelopev2/resolve/main/buffalo_l.zip",
    "https://huggingface.co/maze/faceX/resolve/main/buffalo_l.zip",
]


def have_all():
    return all(os.path.isfile(os.path.join(MODEL_DIR, n)) for n in NEEDED)


def _progress(count, block, total):
    if total <= 0:
        return
    pct = min(100, count * block * 100 // total)
    sys.stdout.write(f"\r  {pct:3d}%  ({count * block // (1024 * 1024)} MB)")
    sys.stdout.flush()


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if have_all():
        print("Models already present in", MODEL_DIR)
        return 0

    tmpdir = tempfile.mkdtemp()
    zip_path = os.path.join(tmpdir, "buffalo_l.zip")
    last_error = None
    for url in URLS:
        try:
            print("Downloading", url)
            urllib.request.urlretrieve(url, zip_path, _progress)
            print()
            break
        except Exception as exc:
            last_error = exc
            print("\n  failed:", exc)
    else:
        print("\nCouldn't download buffalo_l.zip:", last_error)
        print("Download it manually and unzip det_10g.onnx and w600k_r50.onnx into")
        print(" ", MODEL_DIR)
        return 1

    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            base = os.path.basename(member)
            if base in NEEDED:
                with zf.open(member) as src, \
                        open(os.path.join(MODEL_DIR, base), "wb") as dst:
                    shutil.copyfileobj(src, dst)
                print("  extracted", base)
    shutil.rmtree(tmpdir, ignore_errors=True)

    missing = [n for n in NEEDED if not os.path.isfile(os.path.join(MODEL_DIR, n))]
    if missing:
        print("Archive did not contain:", ", ".join(missing))
        return 1
    print("Done ->", MODEL_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
