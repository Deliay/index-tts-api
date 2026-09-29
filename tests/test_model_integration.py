"""End-to-end test against the real IndexTTS-2.5 checkpoints.

Needs a GPU and downloaded checkpoints, so it is excluded from the default run:

    uv run pytest -m model
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from index_tts_api.api.deps import get_engine
from index_tts_api.config import settings
from index_tts_api.main import app
from index_tts_api.services.tts import IndexTTS25Engine

pytestmark = pytest.mark.model


def _reference_audio() -> Path | None:
    """Find a reference clip supplied by the upstream examples downloader."""
    for candidate in (
        Path(settings.model_dir) / "examples" / "voice_01.wav",
        Path("examples") / "voice_01.wav",
        Path("checkpoints") / "examples" / "voice_01.wav",
    ):
        if candidate.is_file():
            return candidate
    return None


def test_synthesize_with_real_model() -> None:
    prompt = _reference_audio()
    if prompt is None:
        pytest.skip("no reference audio found; run the upstream examples downloader first")
    if not Path(settings.cfg_path).is_file():
        pytest.skip(f"IndexTTS-2.5 checkpoints not found at {settings.model_dir}")

    engine = IndexTTS25Engine(settings)
    app.dependency_overrides[get_engine] = lambda: engine
    try:
        with TestClient(app) as client:
            response = client.post(
                "/synthesize",
                data={"text": "大家好，这是 IndexTTS 二点五的接口测试。", "lang": "ZH"},
                files={"speaker_prompt": (prompt.name, prompt.read_bytes(), "audio/wav")},
            )
    finally:
        app.dependency_overrides.clear()
        engine.unload()

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "audio/wav"
    assert response.content.startswith(b"RIFF")
    assert len(response.content) > 44  # more than a bare WAV header
