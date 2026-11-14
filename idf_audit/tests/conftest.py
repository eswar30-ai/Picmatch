"""Every test runs against a throwaway data folder and known settings: never
this machine's var/ or .env values (a variable already set in the environment
wins over .env, so these are set before anything imports idf.settings)."""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))      # tests import each other's fixtures

os.environ["IDF_DATA_DIR"] = tempfile.mkdtemp(prefix="idf_audit_tests_")
os.environ["OPEN_AUDIT_API"] = "false"
os.environ["IDF_AUDIT_API_KEYS"] = "test-key"
os.environ["IDF_POST_CALLBACK"] = "false"
os.environ["IDF_BASE_URL"] = "http://192.168.214.23:4000"
