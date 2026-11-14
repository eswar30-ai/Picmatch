"""Start the IDF audit service.

    python run.py                      # http://<this machine's IP>:8090, reachable from the network
    python run.py --port 8091          # another port
    python run.py --host 127.0.0.1     # this machine only

Any `python` will do: if it is not this service's own `.venv` interpreter, this
re-runs itself with that one. The environment is this folder's `.venv`, never
Picmatch's `venv` (requirements.txt says why). First time on a machine:

    py -3.10 -m venv .venv
    .venv\\Scripts\\python -m pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _in_venv() -> bool:
    return Path(sys.prefix).resolve() == (ROOT / ".venv").resolve()


def lan_addresses() -> list[str]:
    """This machine's IPv4 addresses other machines can use: not loopback, not
    the 169.254 addresses Windows gives an adapter with no network."""
    found: list[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = info[4][0]
            if not address.startswith(("127.", "169.254.")) and address not in found:
                found.append(address)
    except OSError:
        pass
    return found


def address_towards(url: str) -> str | None:
    """The address this machine uses to reach `url`'s host - what that host sees
    us as. A UDP "connect" picks the route and sends nothing."""
    from urllib.parse import urlsplit

    host = urlsplit(url).hostname
    if not host:
        return None
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect((host, 9))
            return probe.getsockname()[0]
    except OSError:
        return None


def main() -> int:
    if not _in_venv():
        if not VENV_PYTHON.is_file():
            print(f"No environment at {VENV_PYTHON.parent.parent}. Build it first:\n"
                  "  py -3.10 -m venv .venv\n"
                  "  .venv\\Scripts\\python -m pip install -r requirements.txt", file=sys.stderr)
            return 1
        # A child process rather than os.exec*: on Windows exec does not replace
        # the process, it starts a second one and returns, which detaches Ctrl+C.
        return subprocess.call([str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])

    parser = argparse.ArgumentParser(description="Start the IDF audit service.")
    parser.add_argument("--host", default="0.0.0.0",
                        help="0.0.0.0 (the default) listens on every network address; 127.0.0.1 = this machine only")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from idf import ocr, settings

    vlm = ocr.vlm_health()
    print(f"Ollama at {vlm['base']}: " + ("running" if vlm["up"] else "NOT reachable")
          + ("" if not vlm["up"] else f", {vlm['model']} " + ("present" if vlm["model_present"] else "MISSING")))
    if not (vlm["up"] and vlm["model_present"]):
        print("  Audits will fail until it is: start Ollama and `ollama pull " + vlm["model"] + "`.")
    if args.host == "0.0.0.0":
        idf = settings.value("IDF_BASE_URL")
        towards = address_towards(idf) if idf else None
        addresses = lan_addresses()
        if towards and towards not in addresses:
            addresses.insert(0, towards)
        for address in addresses or ["127.0.0.1"]:
            note = "   <- the address IDF's server reaches us at" if address == towards else ""
            print(f"Open http://{address}:{args.port}{note}")
        print(f"     http://127.0.0.1:{args.port} on this machine\n")
    else:
        print(f"Open http://{args.host}:{args.port}\n")

    import uvicorn

    # One process, one worker: the queue lives in it, and audits run one at a time.
    uvicorn.run("idf.app:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
