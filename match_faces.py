import csv
import os

import face_engine as fe

# ============================= CONFIG =================================
SINGLE_PERSON_FOLDER = "single_person"  
GROUP_PHOTO_FOLDER = "group_photos"      
CACHE_FILE = "embeddings_cache.json"     
# ========================================================================


def load_reference_people(folder):
    people = {}
    skipped = []
    for fname in fe.list_images(folder):
        name = os.path.splitext(fname)[0]
        embedding = fe.embed_primary_face(os.path.join(folder, fname))
        if embedding is None:
            skipped.append(fname)
            print(f"  ! no face found in {fname} - skipping this person")
            continue
        people[name] = embedding
        print(f"  loaded: {name}")
    return people, skipped


def main():
    try:
        fe.ensure_models()
    except fe.ModelError as exc:
        print("Error:", exc)
        return

    print(fe.settings_summary())

    for folder in (SINGLE_PERSON_FOLDER, GROUP_PHOTO_FOLDER):
        if not os.path.isdir(folder):
            print(f"Folder '{folder}' doesn't exist. Edit the CONFIG section at the "
                  f"top of this script to point at the right path.")
            return

    print("\nReading reference photos from", SINGLE_PERSON_FOLDER)
    people, _ = load_reference_people(SINGLE_PERSON_FOLDER)
    if not people:
        print("No usable reference faces found. Stopping.")
        return

    print("\nScanning group photos from", GROUP_PHOTO_FOLDER)
    cache = fe.EmbeddingCache(CACHE_FILE)
    group_photos = fe.list_images(GROUP_PHOTO_FOLDER)
    cache.prune(group_photos)

    results_by_person = {name: [] for name in people}
    results_by_photo = {}

    for fname in group_photos:
        group_embeds = cache.get(fname, os.path.join(GROUP_PHOTO_FOLDER, fname))
        if not group_embeds:
            print(f"  ! no faces detected in {fname}")
            results_by_photo[fname] = []
            continue

        matched = []
        for name, ref in people.items():
            if fe.is_match(fe.best_similarity(ref, group_embeds)):
                results_by_person[name].append(fname)
                matched.append(name)
        results_by_photo[fname] = sorted(matched)
        print(f"  {fname}: {len(group_embeds)} face(s) detected, "
              f"{len(matched)} identified")

    cache.save()

    print("\n=== RESULTS: person -> group photos ===")
    with open("results_by_person.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["person", "num_group_photos", "group_photo_filenames"])
        for name, files in results_by_person.items():
            if files:
                print(f"{name} - present in {len(files)} group picture(s): {', '.join(files)}")
            else:
                print(f"{name} - not found in any group picture")
            writer.writerow([name, len(files), ", ".join(files)])

    print("\n=== RESULTS: group photo -> people present ===")
    with open("results_by_photo.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group_photo", "people_present"])
        for fname, names in results_by_photo.items():
            print(f"{fname}: {', '.join(names) if names else '(nobody matched)'}")
            writer.writerow([fname, ", ".join(names)])

    print("\nSaved results_by_person.csv and results_by_photo.csv")


if __name__ == "__main__":
    main()
