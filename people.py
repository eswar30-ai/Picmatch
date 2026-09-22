"""Turn a folder of reference photos into one identity per person.

Two layouts are supported:

  single_person/Bharath.jpg           -> person "Bharath", one reference
  single_person/Bharath/*.jpg         -> person "Bharath", several references

Several references per person is the single cheapest accuracy win available:
one frontal studio shot does not describe how someone looks at 30 degrees off
axis under a different light, and that mismatch is what makes real matches fall
under the threshold and get dropped.
"""

import os

import numpy as np

import gpu_engine as ge


class Person(dict):
    __getattr__ = dict.get


def _ref_files(folder):
    """-> {person_name: [file paths]}"""
    people = {}
    for entry in sorted(os.listdir(folder)):
        path = os.path.join(folder, entry)
        if os.path.isdir(path):
            shots = [os.path.join(path, f) for f in sorted(os.listdir(path))
                     if f.lower().endswith(ge.IMAGE_EXTS)]
            if shots:
                people.setdefault(entry, []).extend(shots)
        elif entry.lower().endswith(ge.IMAGE_EXTS):
            name = os.path.splitext(entry)[0]
            people.setdefault(name, []).append(path)
    return people


def enroll(folder, engine, prune_outliers=True, verbose=True):
    """Build reference embeddings for every person in `folder`.

    Returns (people, problems) where people maps name -> Person with an
    (K, 512) `embeddings` array.
    """
    if not os.path.isdir(folder):
        raise SystemExit(f"Reference folder not found: {folder}")

    people, problems = {}, []
    for name, paths in _ref_files(folder).items():
        vectors, used, notes = [], [], []
        for path in paths:
            faces = engine.analyze(path)
            if not faces:
                problems.append(f"{name}: no face found in {os.path.basename(path)}")
                continue
            # analyze() returns largest-first; flag references where a
            # bystander is nearly as prominent as the subject
            if len(faces) > 1:
                a0 = _area(faces[0]["bbox"])
                a1 = _area(faces[1]["bbox"])
                if a1 > 0.6 * a0:
                    notes.append(f"{os.path.basename(path)} has {len(faces)} similar-sized "
                                 f"faces - largest was used, crop it if that's wrong")
            vectors.append(faces[0]["embedding"])
            used.append(path)

        if not vectors:
            problems.append(f"{name}: SKIPPED - no usable reference photo")
            continue

        emb = np.stack(vectors).astype(np.float32)
        if prune_outliers and len(emb) >= 3:
            emb, used, dropped = _drop_outliers(emb, used)
            for d in dropped:
                notes.append(f"{os.path.basename(d)} looks like a different person "
                             f"- dropped from this identity")

        people[name] = Person(name=name, embeddings=emb, sources=used, notes=notes)
        problems.extend(f"{name}: {n}" for n in notes)

    if verbose:
        print(f"Enrolled {len(people)} people from {folder} "
              f"({sum(len(p['embeddings']) for p in people.values())} reference faces)")
        for p in problems:
            print("  !", p)
    return people, problems


def _area(box):
    return float((box[2] - box[0]) * (box[3] - box[1]))


def _drop_outliers(emb, paths, min_agreement=0.35):
    """Remove reference shots that disagree with the rest of the person's set.

    Guards against the classic mistake of enrolling someone from a photo where
    the app picked the wrong face.
    """
    sim = emb @ emb.T
    np.fill_diagonal(sim, np.nan)
    agreement = np.nanmean(sim, axis=1)
    keep = agreement >= min_agreement
    if keep.sum() < 1:
        return emb, paths, []
    dropped = [p for p, k in zip(paths, keep) if not k]
    return emb[keep], [p for p, k in zip(paths, keep) if k], dropped


def stack(people):
    """All reference vectors in one matrix plus the owning person per row."""
    names = sorted(people)
    mats, owner = [], []
    for i, name in enumerate(names):
        emb = people[name]["embeddings"]
        mats.append(emb)
        owner.extend([i] * len(emb))
    if not mats:
        return names, np.zeros((0, 512), np.float32), np.zeros((0,), np.int32)
    return names, np.vstack(mats).astype(np.float32), np.asarray(owner, np.int32)
