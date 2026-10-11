"""Isolate tests from the developer's real data directory.

Settings are read at import time, so the data dir must be redirected before
any ``app`` module is imported. Without this, API tests wrote synthetic demo
frames into the same ``data/evidence`` folder the UI serves, overwriting real
evidence images.
"""

from __future__ import annotations

import os
import tempfile

_TEST_DATA_DIR = tempfile.mkdtemp(prefix="vista-test-data-")
os.environ["VISTA_DATA_DIR"] = _TEST_DATA_DIR
