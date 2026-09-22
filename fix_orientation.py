import argparse
import os
import shutil

import cv2
import numpy as np
from PIL import Image, ImageOps

import face_engine as fe

DEFAULT_FOLDERS = ("group_photos", "single_person")
BACKUP_ROOT = "orientation_backup"
EXIF_ORIENTATION_TAG = 274
CONFIDENT_LEAD = 0.12


def read_exif_orientation(path):
    try:
        with Image.open(path) as im:
            return im.getexif().get(EXIF_ORIENTATION_TAG)
    except Exception:
        return None


def load_exif_corrected(path):
    """Pixels with any EXIF orientation already applied, as a PIL RGB image."""
    with Image.open(path) as im:
        im.load()
        fmt = im.format
        corrected = ImageOps.exif_transpose(im)
    corrected.format = fmt
    return corrected


def pil_to_bgr(pil_img):
    return np.array(pil_img.convert("RGB"))[:, :, ::-1].copy()


_PIL_ROTATION = {
    90: Image.ROTATE_270,    
    180: Image.ROTATE_180,
    270: Image.ROTATE_90,
}


def rotate_pil_clockwise(pil_img, degrees):
    transpose = _PIL_ROTATION.get(degrees % 360)
    return pil_img if transpose is None else pil_img.transpose(transpose)


def plan_for(path):
    exif = read_exif_orientation(path)
    exif_rotates = exif not in (None, 0, 1)

    pil = load_exif_corrected(path)
    bgr = pil_to_bgr(pil)

    rotation, stats = fe.detect_upright_rotation(bgr)

    ranked = sorted(stats.items(), key=lambda kv: -kv[1][1])
    lead = 0.0
    if len(ranked) > 1:
        lead = ranked[0][1][1] - ranked[1][1][1]
    faces_found = stats.get(rotation, (0, 0.0))[0]

    if faces_found == 0:
        verdict = "NO FACES"
    elif rotation == 0 and not exif_rotates:
        verdict = "already straight"
    elif rotation != 0 and lead < CONFIDENT_LEAD:
        verdict = "LOW CONFIDENCE"
    elif rotation == 0:
        verdict = "bake in EXIF"
    else:
        verdict = f"rotate {rotation} deg"

    return {
        "path": path,
        "exif": exif,
        "exif_rotates": exif_rotates,
        "rotation": rotation,
        "stats": stats,
        "lead": lead,
        "faces": faces_found,
        "verdict": verdict,
        "needs_write": rotation != 0 or exif_rotates,
        "pil": pil,
    }


def apply_plan(plan, folder, backup_root=BACKUP_ROOT):
    """Rewrite one file upright, backing the original up first."""
    path = plan["path"]
    backup_dir = os.path.join(backup_root, folder)
    os.makedirs(backup_dir, exist_ok=True)
    backup_path = os.path.join(backup_dir, os.path.basename(path))
    if not os.path.exists(backup_path):
        shutil.copy2(path, backup_path)

    upright = rotate_pil_clockwise(plan["pil"], plan["rotation"])

    exif = upright.getexif()
    if EXIF_ORIENTATION_TAG in exif:
        del exif[EXIF_ORIENTATION_TAG]

    fmt = (plan["pil"].format or "JPEG").upper()
    save_kwargs = {"exif": exif.tobytes()} if len(exif) else {}
    if fmt in ("JPEG", "JPG"):
        save_kwargs.update(quality="keep", subsampling="keep", optimize=True)
        upright = upright.convert("RGB")
    try:
        upright.save(path, format=fmt, **save_kwargs)
    except Exception:
        save_kwargs.pop("quality", None)
        save_kwargs.pop("subsampling", None)
        upright.convert("RGB").save(path, format=fmt, quality=95, **save_kwargs)
    return backup_path


def contact_sheet(rows, out_path, tile=190):
    tiles = []
    for folder, plan, changed in rows:
        before = plan["pil"]
        after = rotate_pil_clockwise(before, plan["rotation"])

        def thumb(img):
            im = img.convert("RGB").copy()
            im.thumbnail((tile, tile))
            canvas = Image.new("RGB", (tile, tile), (245, 243, 238))
            canvas.paste(im, ((tile - im.width) // 2, (tile - im.height) // 2))
            return pil_to_bgr(canvas)

        pair = np.hstack([thumb(before), np.full((tile, 6, 3), 255, np.uint8), thumb(after)])
        label = np.full((26, pair.shape[1], 3), 255, np.uint8)
        name = os.path.basename(plan["path"])
        text = f"{name[:24]} {plan['verdict']}"
        cv2.putText(label, text, (3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        tiles.append(np.vstack([label, pair]))

    if not tiles:
        return None
    per_row = 4
    grid = [np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)]
    width = max(r.shape[1] for r in grid)
    grid = [np.hstack([r, np.full((r.shape[0], width - r.shape[1], 3), 255, np.uint8)])
            for r in grid]
    cv2.imwrite(out_path, np.vstack(grid))
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Straighten sideways photos in group_photos/ and single_person/")
    parser.add_argument("--apply", action="store_true",
                        help="actually rewrite files (default is a dry run)")
    parser.add_argument("--folder", action="append", dest="folders",
                        help="limit to one folder; repeatable")
    parser.add_argument("--include-uncertain", action="store_true",
                        help="also rotate files whose orientation was a close call")
    parser.add_argument("--contact-sheet", metavar="PNG",
                        help="write a before/after thumbnail sheet for review")
    args = parser.parse_args()

    try:
        fe.ensure_models()
    except fe.ModelError as exc:
        print("Error:", exc)
        return

    folders = args.folders or list(DEFAULT_FOLDERS)
    print(fe.settings_summary())
    print("mode:", "APPLY (files will be rewritten)" if args.apply else "dry run (no changes)")

    changed_rows, all_rows = [], []
    for folder in folders:
        if not os.path.isdir(folder):
            print(f"\n! folder not found: {folder}")
            continue
        names = fe.list_images(folder)
        print(f"\n=== {folder} ({len(names)} image(s)) ===")
        print(f"{'file':<24}{'exif':>6}{'faces':>7}{'turn':>7}{'lead':>7}  verdict")
        for name in names:
            plan = plan_for(os.path.join(folder, name))
            all_rows.append((folder, plan))
            print(f"{name:<24}{str(plan['exif']):>6}{plan['faces']:>7}"
                  f"{plan['rotation']:>6}d{plan['lead']:>7.2f}  {plan['verdict']}")
            uncertain = plan["verdict"] == "LOW CONFIDENCE"
            if plan["needs_write"] and (not uncertain or args.include_uncertain):
                changed_rows.append((folder, plan))

    print(f"\n{len(changed_rows)} of {len(all_rows)} file(s) need straightening.")
    uncertain = [p for _, p in all_rows if p["verdict"] == "LOW CONFIDENCE"]
    if uncertain and not args.include_uncertain:
        print(f"{len(uncertain)} skipped as a close call "
              f"({', '.join(os.path.basename(p['path']) for p in uncertain)}). "
              f"Re-run with --include-uncertain to rotate them too.")
    no_faces = [p for _, p in all_rows if p["verdict"] == "NO FACES"]
    if no_faces:
        print(f"{len(no_faces)} file(s) had no detectable face, so orientation could not be "
              f"judged: {', '.join(os.path.basename(p['path']) for p in no_faces)}")

    if args.contact_sheet:
        sheet_rows = [(f, p, True) for f, p in (changed_rows or all_rows)]
        written = contact_sheet(sheet_rows, args.contact_sheet)
        if written:
            print("wrote contact sheet:", written)

    if not args.apply:
        if changed_rows:
            print("\nNothing was written. Re-run with --apply to rewrite these files "
                  f"(originals are copied to {BACKUP_ROOT}/ first).")
        return

    for folder, plan in changed_rows:
        backup = apply_plan(plan, folder)
        print(f"  straightened {plan['path']}  (original -> {backup})")

    if changed_rows:
        print(f"\nDone. {len(changed_rows)} file(s) rewritten upright.")
        print("The embedding cache keys on file mtime, so the next run of "
              "match_faces.py / app.py re-indexes the changed photos automatically.")
    else:
        print("\nNothing to do - every file was already straight.")


if __name__ == "__main__":
    main()
