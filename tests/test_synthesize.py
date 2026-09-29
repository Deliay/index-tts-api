"""Endpoint tests for ``POST /synthesize``.

These use a fake engine so they run without a GPU or checkpoints; the real
IndexTTS-2.5 wiring is exercised by the model-dependent test marked ``model``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from index_tts_api.api.deps import get_engine
from index_tts_api.main import app
from index_tts_api.services.tts import (
    ModelLoadError,
    ModelState,
    SynthesisError,
    SynthesisRequest,
    SynthesisResult,
    pcm16_wav,
)

SAMPLE_RATE = 22050


def wav_bytes(sample_rate: int = SAMPLE_RATE) -> bytes:
    samples = (np.sin(np.linspace(0.0, 40.0, sample_rate // 4)) * 8000).astype(np.int16)
    return pcm16_wav(sample_rate, samples)


class FakeEngine:
    """Stands in for :class:`IndexTTS25Engine` without touching torch."""

    def __init__(self, work_dir: Path, *, error: Exception | None = None) -> None:
        self.work_dir = work_dir
        self.error = error
        self.state = ModelState.READY
        self.model_version = "2.5"
        self.requests: list[SynthesisRequest] = []

    def materialize_prompt(self, data: bytes, filename: str | None) -> Path:
        path = self.work_dir / (filename or "prompt.wav")
        path.write_bytes(data)
        return path

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return SynthesisResult(audio=wav_bytes(), sample_rate=SAMPLE_RATE, duration_seconds=0.25)


@pytest.fixture
def engine(tmp_path: Path) -> FakeEngine:
    return FakeEngine(tmp_path)


@pytest.fixture
def client(engine: FakeEngine):
    app.dependency_overrides[get_engine] = lambda: engine
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def post_synthesize(client: TestClient, **overrides):
    data = {"text": "Hello world", "lang": "EN"}
    data.update({k: v for k, v in overrides.items() if not isinstance(v, (bytes, tuple))})
    files = {"speaker_prompt": ("voice.wav", wav_bytes(), "audio/wav")}
    files.update({k: v for k, v in overrides.items() if isinstance(v, (bytes, tuple))})
    return client.post("/synthesize", data=data, files=files)


def test_returns_wav_audio(client: TestClient, engine: FakeEngine) -> None:
    response = post_synthesize(client)

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["x-sample-rate"] == str(SAMPLE_RATE)
    assert response.content.startswith(b"RIFF")

    (request,) = engine.requests
    assert request.text == "Hello world"
    assert request.lang == "EN"
    assert request.speaker_prompt.exists()


def test_language_is_normalized_to_uppercase(client: TestClient, engine: FakeEngine) -> None:
    response = post_synthesize(client, lang="zh")

    assert response.status_code == 200
    assert engine.requests[0].lang == "ZH"


def test_rejects_unsupported_language(client: TestClient, engine: FakeEngine) -> None:
    response = post_synthesize(client, lang="FR")

    assert response.status_code == 422
    assert "lang" in response.json()["detail"]
    assert engine.requests == []


def test_rejects_emo_text_without_qwen_model(client: TestClient, engine: FakeEngine) -> None:
    response = post_synthesize(client, use_emo_text="true")

    assert response.status_code == 422
    assert "INDEX_TTS_USE_QWEN_EMO" in response.json()["detail"]


def test_rejects_malformed_emo_vector(client: TestClient) -> None:
    response = post_synthesize(client, emo_vector="1,2")

    assert response.status_code == 422


def test_accepts_emotion_vector(client: TestClient, engine: FakeEngine) -> None:
    response = post_synthesize(client, emo_vector="0,0,0.8,0,0,0,0,0")

    assert response.status_code == 200
    assert engine.requests[0].emo_vector == [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_requires_text(client: TestClient) -> None:
    response = client.post(
        "/synthesize",
        data={"lang": "EN"},
        files={"speaker_prompt": ("voice.wav", wav_bytes(), "audio/wav")},
    )

    assert response.status_code == 422


def test_reports_model_load_failure(client: TestClient, engine: FakeEngine) -> None:
    engine.error = ModelLoadError("CUDA is not available")

    response = post_synthesize(client)

    assert response.status_code == 503
    assert "CUDA is not available" in response.json()["detail"]


def test_reports_inference_failure(client: TestClient, engine: FakeEngine) -> None:
    engine.error = SynthesisError("empty audio")

    response = post_synthesize(client)

    assert response.status_code == 500
    assert "empty audio" in response.json()["detail"]


@pytest.mark.parametrize(
    "overrides",
    [{"duration_factor": "3.0"}, {"emo_alpha": "1.5"}, {"lang": "EN", "speaker_prompt": ("v.wav", b"", "audio/wav")}],
)
def test_rejects_out_of_range_and_empty_inputs(client: TestClient, overrides: dict) -> None:
    response = post_synthesize(client, **overrides)

    assert response.status_code == 422 or response.status_code == 400
