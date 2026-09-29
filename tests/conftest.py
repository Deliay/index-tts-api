"""Test-time configuration.

Point the engine's scratch directory at a temporary location so importing
``index_tts_api.config`` (and creating the app) never writes into the repository.
Must run before any ``index_tts_api`` import.
"""

import os
import tempfile

os.environ.setdefault("INDEX_TTS_WORK_DIR", tempfile.mkdtemp(prefix="index-tts-api-tests-"))
