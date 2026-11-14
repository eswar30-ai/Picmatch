"""Start Picmatch and the IDF audit service together.

    python run.py                      # Picmatch on 127.0.0.1:5001, IDF audit on 127.0.0.1:5008
    python run.py --host 0.0.0.0       # also reachable from other machines, not only via nginx
    python run.py --picmatch-port 5001 --idf-port 5008

Both listen on this server's localhost only: nginx, on the same server, is the
only way in from outside.

Any `python` will do. Each service runs in its own process with its own
environment - Picmatch in venv\\, IDF in idf_audit\\.venv\\ - because the two
cannot share one: IDF's CPU onnxruntime and opencv-python would replace
Picmatch's onnxruntime-gpu and opencv-contrib-python (idf_audit/requirements.txt).

Both print here, each line marked [picmatch] or [idf], and every line is also
kept in logs\\idf-<date>.log and logs\\picmatch-<date>.log for 30 days. Ctrl+C stops both.
A service that stops on its own after running a while is started again; one that
stops within a minute of starting is a setup error, so it is reported and left
stopped while the other keeps running.

nginx sends /api/audit and /api/health to the IDF port and everything else to
Picmatch's (README.md, "Picmatch and IDF on one domain").
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = "Scripts/python.exe" if os.name == "nt" else "bin/python"
#: A service that dies sooner than this after starting is not restarted.
STARTUP_GRACE = 60
#: Everything a service prints is also kept here, one file per service per day:
#: logs/idf-2026-11-14.log, logs/picmatch-2026-11-14.log. Older ones are deleted.
LOGS = ROOT / "logs"
KEEP_LOG_DAYS = 30

_print_lock = threading.Lock()


def say(name: str, line: str) -> None:
    with _print_lock:
        print(f"[{name}] {line}", flush=True)


class Service:
    def __init__(self, name: str, python: Path, args: list[str], cwd: Path,
                 env: dict[str, str] | None = None) -> None:
        self.name, self.python, self.args, self.cwd = name, python, args, cwd
        self.env = env or {}
        self.proc: subprocess.Popen[str] | None = None
        self.started = 0.0
        self.given_up = False
        self._log_lock = threading.Lock()
        self._log_day = ""
        self._log_file = None

    def log(self, line: str) -> None:
        """Append a line to today's log file for this service."""
        with self._log_lock:
            try:
                day = time.strftime("%Y-%m-%d")
                if day != self._log_day:
                    if self._log_file:
                        self._log_file.close()
                    LOGS.mkdir(exist_ok=True)
                    self._log_file = open(LOGS / f"{self.name}-{day}.log", "a", encoding="utf-8")
                    self._log_day = day
                    prune_logs()
                self._log_file.write(f"{time.strftime('%H:%M:%S')} {line}\n")
                self._log_file.flush()
            except OSError as exc:              # a full or locked disk must not stop the service
                say("run", f"cannot write {self.name}'s log: {exc}")

    def start(self) -> None:
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", **self.env}
        self.proc = subprocess.Popen(
            [str(self.python), *self.args], cwd=self.cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
        self.started = time.monotonic()
        self.log(f"===== started (pid {self.proc.pid}): {' '.join(self.args)}")
        threading.Thread(target=self._relay, args=(self.proc,), daemon=True).start()

    def _relay(self, proc: subprocess.Popen[str]) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            say(self.name, line)
            self.log(line)
        self.log(f"===== stopped (pid {proc.pid}, exit code {proc.wait()})")

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self, wait: float) -> None:
        """Wait for it to finish (Ctrl+C reaches it too), then end it if it has not."""
        if not self.running():
            return
        assert self.proc is not None
        try:
            self.proc.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def prune_logs() -> None:
    cutoff = time.time() - KEEP_LOG_DAYS * 86400
    for old in LOGS.glob("*.log"):
        try:
            if old.stat().st_mtime < cutoff:
                old.unlink()
        except OSError:
            pass


def port_taken(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Start Picmatch and the IDF audit service together.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="127.0.0.1 (the default) = this server only, reached through nginx; "
                             "0.0.0.0 = every network address")
    parser.add_argument("--picmatch-port", type=int, default=5001)
    parser.add_argument("--idf-port", type=int, default=5008)
    args = parser.parse_args()

    if args.picmatch_port == args.idf_port:
        parser.error("Picmatch and IDF need different ports")

    services = [
        Service("picmatch", ROOT / "venv" / PYTHON,
                ["app.py", "--host", args.host, "--port", str(args.picmatch_port)], ROOT,
                env={"PICMATCH_START_IDF": "0"}),     # this launcher starts IDF itself
        Service("idf", ROOT / "idf_audit" / ".venv" / PYTHON,
                ["run.py", "--host", args.host, "--port", str(args.idf_port)], ROOT / "idf_audit"),
    ]

    problems = []
    for service in services:
        if not service.python.is_file():
            problems.append(f"{service.name}: no environment at {service.python.parent.parent} "
                            f"(see {'README.md' if service.name == 'picmatch' else 'idf_audit/README.md'})")
    for name, port in (("picmatch", args.picmatch_port), ("idf", args.idf_port)):
        if port_taken(port):
            problems.append(f"{name}: port {port} is already in use - is it already running?")
    if problems:
        print("Not started:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1

    for service in services:
        service.start()
    say("run", f"Picmatch on port {args.picmatch_port}, IDF audit on port {args.idf_port}. Ctrl+C stops both.")
    say("run", f"Logs: {LOGS}\\<picmatch|idf>-<date>.log")

    try:
        while any(not s.given_up for s in services):
            time.sleep(1)
            for service in services:
                if service.given_up or service.running():
                    continue
                assert service.proc is not None
                code, ran = service.proc.returncode, time.monotonic() - service.started
                if ran >= STARTUP_GRACE:
                    say("run", f"{service.name} stopped (exit code {code}) after {ran:.0f}s - starting it again")
                    service.start()
                else:
                    others = [s.name for s in services if s is not service and not s.given_up]
                    say("run", f"{service.name} stopped (exit code {code}) {ran:.0f}s after starting - "
                               f"not restarting it; see its error above."
                               + (f" {others[0]} keeps running." if others else ""))
                    service.given_up = True
        return 1
    except KeyboardInterrupt:
        say("run", "stopping both ...")
        try:
            for service in services:
                service.stop(wait=15)
        except KeyboardInterrupt:
            for service in services:
                if service.running():
                    service.proc.kill()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
