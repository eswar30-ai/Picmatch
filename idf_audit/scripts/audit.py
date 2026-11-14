"""Run one IDF audit from the command line, without the service.

    .venv/Scripts/python.exe scripts/audit.py package.json --files FOLDER [--out verdict.json]

`package.json` is their ai-package reply (or its `data`); FOLDER holds one file
per evidence item, named `<id>_<anything>` or exactly the `name` the package
gives it. Prints the verdict IDF would receive, plus the per-file detail.
Ollama must be running: the judgement is the model's.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from idf.audit import answer_body, run                                          # noqa: E402
from idf.package import parse                                                   # noqa: E402


def find_files(package, folder: Path) -> dict[int, Path]:
    out = {}
    for item in package.evidence:
        by_id = [p for p in folder.iterdir() if p.is_file() and p.name.startswith(f"{item.id}_")]
        by_name = folder / item.name if item.name else None
        path = by_id[0] if by_id else (by_name if by_name and by_name.is_file() else None)
        if path is None:
            sys.exit(f"no file for evidence {item.id} ({item.name}) in {folder}")
        out[item.id] = path
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("package", type=Path)
    ap.add_argument("--files", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    package = parse(json.loads(args.package.read_text(encoding="utf-8")))
    result = run(package, find_files(package, args.files))
    text = json.dumps(answer_body(result), indent=2, ensure_ascii=False)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"written to {args.out}")
    print(text)


if __name__ == "__main__":
    main()
