# PicMatch

Upload a photo of a person, get back every group photo they appear in.

Routes, JSON and folders are exactly what they always were — `group_photos/`,
`single_person/`, `POST /api/match`, `GET /group_photo/<file>`. What changed is
the engine: the CPU YuNet + SFace pair has been replaced by SCRFD-10GF
detection and ArcFace w600k_r50 embeddings running on CUDA through
onnxruntime. There is no CPU pipeline left to switch back to (the old files are
kept as `app_cpu_backup.py` / `face_engine_cpu_backup.py` for reference only).

## Run it

```bat
venv\Scripts\python app.py                 :: http://127.0.0.1:5001
venv\Scripts\python app.py --host 0.0.0.0 --port 5001
```

First start loads both ONNX models and reads `data/index.npz`; photos whose
file hasn't changed are reused, so it comes up in seconds.

`app.py` also starts the IDF audit service from `idf_audit/` on port 5008
(`--idf-port` for another, `--no-idf` for Picmatch alone), in IDF's own
environment, and stops it when Picmatch stops. If something already listens on
that port, it is left alone and used as is.

## Picmatch and IDF on one domain

`run.py` starts Picmatch (port 5001) and the IDF audit service in `idf_audit/`
(port 5008) together, each in its own environment — `venv\` and
`idf_audit\.venv\`, which can never be merged (`idf_audit/requirements.txt`).
Both print to the same window, Ctrl+C stops both. Both listen on the server's
localhost only, so nginx on the same server is the only way in from outside.

```bat
python run.py                              :: 127.0.0.1:5001 Picmatch + 127.0.0.1:5008 IDF
python run.py --host 0.0.0.0               :: also reachable directly from other machines
```

Their paths don't overlap, so nginx sends IDF's three to 5008 and everything
else to 5001, inside the existing `server { }` block for the domain:

```nginx
# IDF audit: POST /api/audit, GET /api/audit/{id}, GET /api/health
location ^~ /api/audit {
    proxy_pass http://127.0.0.1:5008;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $remote_addr;
    proxy_set_header X-Forwarded-Proto $scheme;
    client_max_body_size 300m;      # a package's evidence files, up to 50 MB each
    proxy_read_timeout 300s;
}
location = /api/health {
    proxy_pass http://127.0.0.1:5008;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $remote_addr;
}

# Picmatch: everything else (already there)
location / {
    proxy_pass http://127.0.0.1:5001;
    ...
}
```

IDF's per-caller rate limit uses the address in `X-Forwarded-For` only when the
request comes from 127.0.0.1 (uvicorn's default). If nginx runs on another
machine, start with `FORWARDED_ALLOW_IPS=<nginx's IP>` set, or every caller
shares one limit. On a public domain, keep `OPEN_AUDIT_API=false` and give IDF's
backend a key from `IDF_AUDIT_API_KEYS` (`idf_audit/.env`).

## What changed and why it helps

| | before (CPU) | now (GPU) |
|---|---|---|
| detector | YuNet, images squashed to 1600px, faces under 32px dropped | SCRFD-10GF at **two scales merged** (1280 + 640), faces down to 20px |
| embedding | SFace 128-d | ArcFace **w600k_r50** 512-d (InsightFace `buffalo_l`) |
| runtime | CPU only | CUDA via onnxruntime-gpu, ~0.4 s per 24MP photo |
| group library | `embeddings_cache.json`, re-scored per request | on-disk face index in `data/`, one matrix multiply per search |
| enrolment | one photo, one face per person | several photos per person (`Ravi_1.jpg`, `Ravi_2.jpg` merge into one identity) |
| decision | every person scored independently at a flat 0.51 | winner-takes-the-face, margin over the runner-up, stricter line for small/blurry/profile faces |

The embedding model is the big one. SFace separates people by roughly 0.1
cosine in a crowd photo; ArcFace separates them by 0.3-0.4. That gap is what
makes a threshold reliable — and why the accept line is now **0.42**, not 0.51.

## Setup

```bat
venv\Scripts\python -m pip install -r requirements.txt
venv\Scripts\python fetch_models.py     :: models\det_10g.onnx + w600k_r50.onnx
```

`cuda_dlls.py` puts the pip CUDA DLLs on the Windows loader path and must be
imported before `onnxruntime` — `gpu_engine.py` already does that. If CUDA
can't start, everything still runs on CPU and the page says so.

## Command line

```bat
:: index the library (incremental - unchanged photos are reused)
venv\Scripts\python pmg.py index

:: sanity-check the reference photos first; this catches most bad results
venv\Scripts\python pmg.py check

:: match, write CSVs and an HTML contact sheet of every decision
venv\Scripts\python pmg.py match

:: file the results into out\<person>\
venv\Scripts\python pmg.py match --copy-to out --include-review

:: see where the accept line sits against the real score spread
venv\Scripts\python pmg.py scores
```

Those default to `--photos group_photos --refs single_person --data data`, and
write `data\results_by_person.csv`, `data\results_by_photo.csv` and
**`data\report.html`** — open that one first. It shows the actual face crop
behind every score, so a wrong match is obvious at a glance.

`webapp_gpu.py` (port 5002) is the diagnostic UI: same matching, but with
threshold sliders and the confident/borderline split shown side by side. Handy
when tuning; `app.py` is the app.

## Tuning

Only three numbers matter, all in `matcher.py` and overridable per run:

- **accept** (0.42) — confident match. Lower to catch more, raise to cut false
  positives. Below 0.34 you start seeing strangers. `PICMATCH_THRESHOLD` for
  the web app, `--accept` for the CLI.
- **review** (0.36) — between review and accept is reported separately rather
  than thrown away. This is where "missing" photos usually are.
- **margin** (0.03) — how far the best person must beat the second-best before
  a face is assigned to them, so one face can't be handed to three people.

If photos are still missing after that, it's detection, not recognition: try
`--det-scales 1920,1280,640`, or `--tiles` for a full-resolution 2x2 pass
(~5x slower, best recall on large crowd shots).

**The highest-value change is on your side, not in the code**: give each person
two or three reference photos instead of one — different angle, different
lighting. Name them `Ravi_1.jpg`, `Ravi_2.jpg` for the web upload, or put them
in `single_person\<Person Name>\*.jpg` for the CLI. A single frontal shot
cannot describe someone turned 30 degrees away, which is why those photos fall
under the line.

## Files

- `app.py` — the web app: `/`, `/match`, `/api/match`, `/admin`,
  `/admin/upload`, `/rescan`, `/group_photo/<file>`, `/docs`
- `face_engine.py` — the long-standing helper interface, now GPU-backed, used
  by `match_faces.py`, `gallery_search.py` and `fix_orientation.py`
- `gpu_engine.py` — SCRFD detector, ArcFace recogniser, alignment, quality signals
- `face_index.py` — incremental per-photo face index (`data/index.npz`)
- `people.py` — reference photo enrolment and outlier pruning
- `matcher.py` — thresholds, margin rule, quality penalties
- `report.py` — the HTML contact sheet
- `pmg.py` — command line
- `webapp_gpu.py` — diagnostic UI with threshold sliders
- `cuda_dlls.py` — Windows CUDA DLL path fix, import before onnxruntime
- `fetch_models.py` — downloads `det_10g.onnx` and `w600k_r50.onnx` 
- `SFA.py` — unrelated: duplicate-report spreadsheet tooling
l