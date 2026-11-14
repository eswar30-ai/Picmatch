# IDF audit service

Judges an IDF certification application or program against its evidence photos
and documents, and answers in the shape IDF's backend expects: `approved`,
`changes_required` or `rejected`, a score, and four checks. The verdict is
advice to IDF's manual reviewer, never a decision.

It is the IDF audit from the OCR platform (`D:\ocr-platform`, decision D29),
taken out to run on its own: RapidOCR reads the text in each photo, Qwen2.5-VL
through Ollama judges whether the file shows what it was uploaded for, and
EXIF gives where a photo was taken. Nothing else of the platform comes with it.

It lives in Picmatch's folder but shares nothing with Picmatch: its own
environment (`.venv`), its own settings (`.env`), its own data (`var/`), its own
port. Picmatch's `venv` has onnxruntime-gpu and opencv-contrib-python; RapidOCR
would install the CPU builds over them, so the two environments are never mixed.

## Run it

You need Python 3.10 and Ollama serving `qwen2.5vl:7b`.

```bat
py -3.10 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env              :: then set OPEN_AUDIT_API or IDF_AUDIT_API_KEYS
python run.py                       :: prints http://<this machine's IP>:8090
```

`python run.py --port 8091` for another port, `--host 127.0.0.1` for this
machine only. Windows Firewall must let the port in for IDF's server to reach it
(once, as administrator):

```powershell
New-NetFirewallRule -DisplayName "IDF audit 8090" -Direction Inbound -Protocol TCP -LocalPort 8090 -Action Allow
```

## The API

| Call | What |
| --- | --- |
| `POST /api/audit` | multipart: `package` (the `ai-package` JSON as a string) and `file_<id>` per evidence item, or the file's path in the package's `url` and it is fetched from `IDF_BASE_URL`. `202 {"audit_id", "state"}` |
| `GET /api/audit/{audit_id}` | `409` while it runs (ask again after `Retry-After`), `422` if it failed, `200` with the verdict |
| `GET /api/health` | Whether Ollama and the model are up, and how many audits wait. No key, no audit data |

The verdict: `status`, `score`, `comments`, `checks` (four, by `check_type`),
and per file `ai_status` and `ai_note`, with the detail behind them. GPS is
judged only for photos marked `requires_gps`; otherwise the location check says
"not checked" and does not count.

Errors: `400` bad package, missing file, or a file that is not a readable image
or PDF; `401` no valid key; `413` over `MAX_UPLOAD_SIZE_MB`; `415` file type
not audited; `429` and `503` wait `Retry-After` seconds.

## Settings (`.env`)

| Setting | Default | What |
| --- | --- | --- |
| `OPEN_AUDIT_API` | `false` | `true`: no key needed. Anyone who reaches the host can then submit and read audits |
| `IDF_AUDIT_API_KEYS` | | Keys accepted as `X-API-Key`, comma-separated |
| `IDF_BASE_URL` | `http://192.168.214.23:4000` | IDF's server, for files given by path |
| `IDF_POST_CALLBACK` | `false` | Post the verdict to the package's callback; off, IDF fetches it by GET |
| `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | `http://127.0.0.1:11434`, `qwen2.5vl:7b` | The vision model |
| `MAX_UPLOAD_SIZE_MB`, `UPLOAD_RATE_LIMIT_PER_MINUTE`, `MAX_QUEUED` | 50, 10, 100 | Limits |
| `IDF_DATA_DIR` | `var/` here | Where packages, files, verdicts and job states are kept |

Audits run one at a time. Each job's state is saved, so a verdict can still be
fetched after a restart; a waiting audit is queued again, a running one is
marked FAILED. Nothing is deleted automatically.

## From the command line, and tests

```bat
.venv\Scripts\python scripts\idf_sample.py                          :: sample packages in var\sample
.venv\Scripts\python scripts\audit.py var\sample\certification\package.json --files var\sample\certification
.venv\Scripts\python -m pytest tests -q
```

## Layout

```
run.py              starts the service on this folder's .venv
idf/app.py          the API
idf/jobs.py         the queue, saved to var/jobs
idf/access.py       key or open switch, per-address limit
idf/package.py      their ai-package (or flat submission) JSON, read
idf/evidence.py     one file judged by OCR + the model
idf/location.py     a photo's EXIF position and time against the declared ones
idf/checks.py       the four checks and the overall verdict
idf/audit.py        one audit, start to finish
idf/fetch.py        a file fetched from IDF's server by its path
idf/callback.py     the verdict posted to IDF, when switched on
idf/ocr.py          RapidOCR, a PDF's own text, Ollama's health
idf/helpers.py      small shared helpers
scripts/            audit.py (one audit, no service), idf_sample.py (sample packages)
tests/              87 tests, run against a throwaway data folder
```

## Checked against the platform (2026-11-14)

The same 8 images through both OCRs: identical text, character for character.
The same 14 packages (IDF's 11 from their first calls, and 3 samples) through
both: every status the same, and the requests sent to the model byte for byte
the same. The model's own wording, and once its confidence (0.95 against 0.90),
varied - and varied the same way between two runs of the platform alone.
