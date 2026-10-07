"""Gallery search & synchronization service for PicMatch GPU.

Synced with app.py to run on a single port.
Eliminates duplicate processes, duplicate ONNX GPU allocations, and deprecated
embeddings_cache.json storage.
"""

import argparse
import os
import sys

import app as picmatch_app
from app import (
    app,
    update_gallery_index,
    fetch_gallery_json,
    download_gallery_image,
    download_image,
    load_gallery_index,
    gallery_index,
    gallery_lock,
    face_db,
    engine,
    GROUP_PHOTO_FOLDER,
    GALLERY_INDEX_FILE,
    API_URL,
    FETCH_INTERVAL_SECONDS,
    batch_match_single_person,
    total_indexed_photos,
    total_indexed_faces,
)

# Backwards-compatible aliases
INDEX_FILE = GALLERY_INDEX_FILE
index_lock = gallery_lock


def build_and_match_all():
    """Runs batch matching for single_person photos using GPU FaceIndex.
    Replaces deprecated fe.EmbeddingCache('embeddings_cache.json') with fast matcher.
    """
    return batch_match_single_person()


def main():
    parser = argparse.ArgumentParser(description="PicMatch Gallery & Web Service")
    parser.add_argument("--host", default=os.environ.get("PICMATCH_HOST", "0.0.0.0"),
                        help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PICMATCH_PORT", os.environ.get("PORT", 5001))),
                        help="Port to listen on (default: 5001)")
    parser.add_argument("--sync-only", action="store_true",
                        help="Fetch and index gallery photos once, then exit without starting web server")
    parser.add_argument("--match-offline", action="store_true",
                        help="Run offline matching for single_person folder and update CSV files")
    parser.add_argument("--no-gallery-sync", action="store_true",
                        help="Disable automatic background gallery sync")
    args = parser.parse_args()

    if args.sync_only:
        print("Running one-off gallery synchronization...")
        res = update_gallery_index(auto_index=True)
        print("Sync result:", res)
        return

    if args.match_offline:
        print("Running offline single_person batch matching...")
        build_and_match_all()
        return

    # Run the unified app on the single port
    argv = ["--host", args.host, "--port", str(args.port)]
    if args.no_gallery_sync:
        argv.append("--no-gallery-sync")
    picmatch_app.main(argv)


if __name__ == "__main__":
    main()
