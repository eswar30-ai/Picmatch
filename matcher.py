"""Decide which people appear in which photos.

The two failure modes we're fixing:

  Missing photos   - one reference embedding, a low-resolution detector and a
                     threshold tuned for a weak model meant genuine faces
                     scored under the line. Fixed with better embeddings, more
                     references per person, and a review band so borderline
                     hits surface instead of vanishing.

  Wrong photos     - every enrolled person was scored against every face
                     independently, so one ambiguous face could be handed to
                     three different people at once, and junk detections
                     (tiny, blurry, back-of-head) scored high by accident.
                     Fixed with a winner-takes-the-face rule, a margin over the
                     runner-up, and quality-scaled thresholds.
"""

import numpy as np

# ArcFace w600k_r50 cosine scale: same person typically 0.45-0.85,
# different people typically below 0.25.
ACCEPT = 0.42      # confident match
# Review floor. Measured on this library: impostor scores tail off around 0.35
# and genuine matches start around 0.50, so anything under ~0.36 is noise. Set
# it lower and people who are absent from the library start collecting dozens
# of "possible" hits, which is the irrelevant-results complaint all over again.
REVIEW = 0.36      # shown separately as "possible", never silently dropped
MARGIN = 0.03      # best person must beat the runner-up by this much

# quality gates - a face below these gets a stricter threshold, not a free pass
MIN_SIZE = 32.0
MIN_BLUR = 12.0
MIN_DET = 0.45
STRONG_PROFILE = 1.5


def quality_penalty(q):
    """Extra cosine a doubtful face must earn before we believe it.

    Small, blurry or side-on faces carry little identity information, so their
    scores drift upward toward everybody. Charging them a few points is what
    keeps strangers out of a person's results.
    """
    penalty = 0.0
    if q["size"] < MIN_SIZE:
        penalty += 0.06
    elif q["size"] < 48:
        penalty += 0.02
    if q["blur"] < MIN_BLUR:
        penalty += 0.05
    elif q["blur"] < 40:
        penalty += 0.02
    if q["det_score"] < MIN_DET:
        penalty += 0.04
    if q["pose"] > STRONG_PROFILE:
        penalty += 0.03
    return penalty


def match(index, people_names, refs, ref_owner, accept=ACCEPT, review=REVIEW,
          margin=MARGIN, use_quality=True):
    """Score every indexed face against every enrolled person.

    Returns a dict:
      by_person[name] = [ {photo, score, status, face_idx, bbox, quality}, ... ]
      by_photo[photo] = [ {person, score, status, ...}, ... ]
      faces           = per-face detail, for reports and threshold tuning
    """
    embeds, owners, rows = index.matrix()
    by_person = {n: [] for n in people_names}
    by_photo = {}
    face_details = []
    top_score = {n: 0.0 for n in people_names}
    if not len(embeds) or not len(refs):
        return {"by_person": by_person, "by_photo": by_photo, "faces": face_details,
                "top_score": top_score}

    # (faces x references) cosine, then collapse references down to their person
    sim = embeds @ refs.T
    n_people = len(people_names)
    per_person = np.full((len(embeds), n_people), -1.0, np.float32)
    for p in range(n_people):
        cols = np.where(ref_owner == p)[0]
        if len(cols):
            per_person[:, p] = sim[:, cols].max(axis=1)
            # true best across the whole library, even when it never clears the
            # review line - "0.29" tells you someone is absent, "0.000" doesn't
            top_score[people_names[p]] = float(per_person[:, p].max())

    order = np.argsort(-per_person, axis=1)
    best_idx = order[:, 0]
    best = per_person[np.arange(len(embeds)), best_idx]
    second = (per_person[np.arange(len(embeds)), order[:, 1]] if n_people > 1
              else np.full(len(embeds), -1.0, np.float32))

    for i, photo in enumerate(owners):
        q = rows[i]["quality"]
        penalty = quality_penalty(q) if use_quality else 0.0
        score = float(best[i])
        runner = float(second[i])
        person = people_names[int(best_idx[i])]
        gap = score - runner

        if score >= accept + penalty and gap >= margin:
            status = "match"
        elif score >= accept + penalty and gap < margin:
            status = "ambiguous"          # two enrolled people fit equally well
        elif score >= review + penalty:
            status = "review"
        else:
            status = "none"

        detail = {
            "photo": photo, "face_idx": i, "person": person, "score": score,
            "runner_up": people_names[int(order[i, 1])] if n_people > 1 else None,
            "runner_up_score": runner, "status": status,
            "bbox": rows[i]["bbox"].tolist(), "quality": q,
            "threshold_used": round(accept + penalty, 4),
        }
        face_details.append(detail)

        if status in ("match", "review"):
            by_person[person].append(detail)
            by_photo.setdefault(photo, []).append(detail)

    for name in by_person:
        by_person[name].sort(key=lambda d: -d["score"])
    for photo in by_photo:
        by_photo[photo].sort(key=lambda d: -d["score"])
    return {"by_person": by_person, "by_photo": by_photo, "faces": face_details,
            "top_score": top_score}


def summarise(result, people_names):
    """Per-person counts of confident vs borderline photos."""
    out = []
    for name in people_names:
        hits = result["by_person"].get(name, [])
        photos_match = {h["photo"] for h in hits if h["status"] == "match"}
        photos_review = {h["photo"] for h in hits if h["status"] == "review"} - photos_match
        top = result.get("top_score", {}).get(name, 0.0)
        out.append({"person": name, "matches": len(photos_match),
                    "review": len(photos_review), "best_score": round(top, 3),
                    "absent": top < REVIEW})
    out.sort(key=lambda r: -r["matches"])
    return out
