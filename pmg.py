"""PicMatch GPU - command line.

  python pmg.py index  --photos group_photos
  python pmg.py match  --photos group_photos --refs single_person
  python pmg.py match  --refs single_person --accept 0.42 --copy-to out
  python pmg.py scores --refs single_person       # threshold diagnostics
  python pmg.py check  --refs single_person       # reference photo sanity
"""

import argparse
import csv
import os
import shutil
import sys
import time

import numpy as np

import gpu_engine as ge
import face_index
import matcher
import people as people_mod
import report as report_mod


def _engine(args):
    scales = ([int(v) for v in args.det_scales.split(",")]
              if args.det_scales else None)
    eng = ge.FaceEngine(det_size=args.det_size, det_thresh=args.det_thresh,
                        min_face_px=args.min_face, force_cpu=args.cpu,
                        tiles=args.tiles, det_scales=scales)
    print("engine:", eng.describe())
    if eng.detector.provider() == "CPUExecutionProvider" and not args.cpu:
        print("  ! running on CPU - the CUDA provider failed to start. "
              "Check that onnxruntime-gpu and the nvidia-*-cu12 wheels are installed.")
    return eng


def _load_index(args, engine=None, build_if_stale=True):
    idx = face_index.FaceIndex(args.data)
    idx.load()
    if not build_if_stale:
        return idx
    engine = engine or _engine(args)
    t = time.time()
    done, reused, failed, faces = idx.build(args.photos, engine, workers=args.workers,
                                            force=args.rebuild)
    idx.save()
    print(f"index: {len(idx.photos)} photos ({done} new, {reused} unchanged), "
          f"{faces} faces, {time.time() - t:.1f}s")
    for f in failed[:10]:
        print("  ! could not read", f)
    if len(failed) > 10:
        print(f"  ! ...and {len(failed) - 10} more")
    empty = [p for p in idx.photos if not idx.faces.get(p)]
    if empty:
        print(f"  note: {len(empty)} photo(s) have no detectable face "
              f"(try --det-scales 1920,1280,640 or --tiles)")
    return idx


# --------------------------------------------------------------------- index
def cmd_index(args):
    _load_index(args)
    return 0


# --------------------------------------------------------------------- match
def cmd_match(args):
    engine = _engine(args)
    idx = _load_index(args, engine)
    persons, _problems = people_mod.enroll(args.refs, engine)
    if not persons:
        print("No people enrolled - nothing to match.")
        return 1

    names, refs, owner = people_mod.stack(persons)
    result = matcher.match(idx, names, refs, owner, accept=args.accept,
                           review=args.review, margin=args.margin,
                           use_quality=not args.no_quality)
    summary = matcher.summarise(result, names)

    print(f"\n{'person':<20}{'confident':>10}{'review':>9}{'best':>8}")
    print("-" * 47)
    for row in summary:
        flag = "  not in this library" if row.get("absent") else ""
        print(f"{row['person'][:19]:<20}{row['matches']:>10}{row['review']:>9}"
              f"{row['best_score']:>8.3f}{flag}")
    total_photos = len({h['photo'] for hits in result['by_person'].values()
                        for h in hits if h['status'] == 'match'})
    print("-" * 47)
    print(f"{'photos with someone':<20}{total_photos:>10} of {len(idx.photos)}")
    absent = [r["person"] for r in summary if r.get("absent")]
    if absent:
        print(f"\n{len(absent)} enrolled people never scored above {args.review} against "
              f"any of the {idx.total_faces()} indexed faces - they are almost certainly "
              f"not in this photo set: " + ", ".join(absent))

    ambiguous = [f for f in result["faces"] if f["status"] == "ambiguous"]
    if ambiguous:
        print(f"\n{len(ambiguous)} face(s) fit two enrolled people equally well and were "
              f"left unassigned (--margin {args.margin}). Add a second reference photo "
              f"for the people involved.")

    os.makedirs(args.data, exist_ok=True)
    p_csv = os.path.join(args.data, "results_by_person.csv")
    with open(p_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["person", "photo", "score", "status", "threshold_used",
                    "face_px", "blur", "pose", "runner_up", "runner_up_score"])
        for name in names:
            for h in result["by_person"][name]:
                q = h["quality"]
                w.writerow([name, h["photo"], f"{h['score']:.4f}", h["status"],
                            h["threshold_used"], int(q["size"]), f"{q['blur']:.0f}",
                            f"{q['pose']:.2f}", h["runner_up"],
                            f"{h['runner_up_score']:.4f}"])

    ph_csv = os.path.join(args.data, "results_by_photo.csv")
    with open(ph_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["photo", "confident_people", "review_people", "faces_detected"])
        for photo in idx.photos:
            hits = result["by_photo"].get(photo, [])
            conf = sorted({h["person"] for h in hits if h["status"] == "match"})
            rev = sorted({h["person"] for h in hits if h["status"] == "review"})
            w.writerow([photo, "; ".join(conf), "; ".join(rev), len(idx.faces.get(photo, []))])
    print(f"\nwrote {p_csv}\n      {ph_csv}")

    if not args.no_report:
        html_path = os.path.join(args.data, "report.html")
        settings = (f"{engine.describe()} | accept={args.accept} review={args.review} "
                    f"margin={args.margin} | {len(idx.photos)} photos, "
                    f"{idx.total_faces()} faces, {len(names)} people")
        report_mod.write(html_path, result, summary, args.photos, settings)
        print("      " + html_path + "   <- open this to eyeball every match")

    if args.copy_to:
        copied = 0
        for name in names:
            photos = sorted({h["photo"] for h in result["by_person"][name]
                             if h["status"] == "match"})
            review = sorted({h["photo"] for h in result["by_person"][name]
                             if h["status"] == "review"}) if args.include_review else []
            for sub, group in (("", photos), ("_review", review)):
                if not group:
                    continue
                dest = os.path.join(args.copy_to, _safe(name) + sub)
                os.makedirs(dest, exist_ok=True)
                for photo in group:
                    src = os.path.join(args.photos, photo)
                    try:
                        shutil.copy2(src, os.path.join(dest, photo))
                        copied += 1
                    except Exception as exc:
                        print("  ! copy failed", photo, exc)
        print(f"      copied {copied} file(s) into {args.copy_to}/")
    return 0


def _safe(name):
    return "".join(c if c.isalnum() or c in " ._-" else "_" for c in name).strip() or "person"


# -------------------------------------------------------------------- scores
def cmd_scores(args):
    """Show where the accept line falls relative to the actual score spread."""
    engine = _engine(args)
    idx = _load_index(args, engine)
    persons, _ = people_mod.enroll(args.refs, engine, verbose=False)
    names, refs, owner = people_mod.stack(persons)
    embeds, owners, rows = idx.matrix()
    if not len(embeds):
        print("empty index")
        return 1

    sim = embeds @ refs.T
    per_person = np.full((len(embeds), len(names)), -1.0, np.float32)
    for p in range(len(names)):
        cols = np.where(owner == p)[0]
        per_person[:, p] = sim[:, cols].max(axis=1)
    best = per_person.max(axis=1)

    print(f"\n{len(embeds)} indexed faces vs {len(names)} people")
    print("best-match score distribution:")
    for lo in np.arange(0.15, 0.75, 0.05):
        n = int(((best >= lo) & (best < lo + 0.05)).sum())
        bar = "#" * min(60, n * 60 // max(1, len(embeds) // 4))
        print(f"  {lo:.2f}-{lo + 0.05:.2f}  {n:6d}  {bar}")
    print("\nfaces above each candidate threshold:")
    for t in (0.30, 0.34, 0.38, 0.40, 0.42, 0.45, 0.50):
        n = int((best >= t).sum())
        photos = len({o for o, b in zip(owners, best) if b >= t})
        print(f"  accept={t:.2f}  {n:5d} faces  in {photos:4d} photos")
    print("\nA genuine face pair usually scores 0.45+ with this model; unrelated "
          "people land under 0.25. If your confident count is low, look at the "
          "0.30-0.40 band in report.html before lowering the line.")
    return 0


# --------------------------------------------------------------------- check
def cmd_check(args):
    """Sanity-check the reference photos - the usual source of bad results."""
    engine = _engine(args)
    persons, problems = people_mod.enroll(args.refs, engine, verbose=True)
    names, refs, owner = people_mod.stack(persons)
    if len(names) < 2:
        return 0
    per = np.full((len(names), len(names)), 0.0, np.float32)
    sim = refs @ refs.T
    for a in range(len(names)):
        ra = np.where(owner == a)[0]
        for b in range(len(names)):
            rb = np.where(owner == b)[0]
            per[a, b] = sim[np.ix_(ra, rb)].max()
    np.fill_diagonal(per, -1)
    print("\nEnrolled people who look alike to the model "
          "(these are the pairs that get mixed up):")
    pairs = [(per[a, b], names[a], names[b])
             for a in range(len(names)) for b in range(a + 1, len(names))]
    pairs.sort(reverse=True)
    for score, a, b in pairs[:10]:
        flag = "  <-- add more reference photos for both" if score > 0.35 else ""
        print(f"  {score:.3f}  {a} / {b}{flag}")
    singles = [n for n in names if len(persons[n]['embeddings']) == 1]
    if singles:
        print(f"\n{len(singles)} of {len(names)} people have only one reference photo. "
              f"Two or three shots per person (different angle/light) is the biggest "
              f"single accuracy gain available:")
        print("  put them in " + os.path.join(args.refs, "<Person Name>", "*.jpg"))
    return 0


# ----------------------------------------------------------------------- cli
def main(argv=None):
    ap = argparse.ArgumentParser(prog="pmg", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["index", "match", "scores", "check"])
    ap.add_argument("--photos", default="group_photos", help="folder of group photos")
    ap.add_argument("--refs", default="single_person",
                    help="reference photos: Name.jpg or Name/*.jpg")
    ap.add_argument("--data", default="data", help="where the index and results live")

    ap.add_argument("--det-size", type=int, default=ge.DET_SIZE,
                    help="detector long side; raise to find smaller faces")
    ap.add_argument("--det-scales", default=None,
                    help="comma-separated detector sizes to merge, e.g. 1920,1280,640 "
                         "(default: <det-size>,640)")
    ap.add_argument("--det-thresh", type=float, default=ge.DET_THRESH)
    ap.add_argument("--min-face", type=int, default=ge.MIN_FACE_PX)
    ap.add_argument("--tiles", action="store_true",
                    help="also detect on 2x2 overlapping tiles (slow, best recall)")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    ap.add_argument("--workers", type=int, default=4, help="image decode threads")
    ap.add_argument("--rebuild", action="store_true", help="ignore the cached index")

    ap.add_argument("--accept", type=float, default=matcher.ACCEPT)
    ap.add_argument("--review", type=float, default=matcher.REVIEW)
    ap.add_argument("--margin", type=float, default=matcher.MARGIN)
    ap.add_argument("--no-quality", action="store_true",
                    help="disable quality-scaled thresholds")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--copy-to", help="copy matched photos into <dir>/<person>/")
    ap.add_argument("--include-review", action="store_true",
                    help="with --copy-to, also write <person>_review/")

    args = ap.parse_args(argv)
    return {"index": cmd_index, "match": cmd_match,
            "scores": cmd_scores, "check": cmd_check}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
