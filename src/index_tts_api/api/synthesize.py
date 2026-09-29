"""Speech synthesis endpoint backed by IndexTTS-2.5."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from index_tts_api.api.deps import get_engine
from index_tts_api.config import settings
from index_tts_api.services.tts import (
    EMOTION_ORDER,
    MODEL_NAME,
    SUPPORTED_LANGUAGES,
    IndexTTS25Engine,
    ModelLoadError,
    SynthesisError,
    SynthesisRequest,
    TTSError,
)

router = APIRouter(tags=["synthesis"])

_EMO_VECTOR_DOC = "Comma-separated intensities, in the order: " + ", ".join(EMOTION_ORDER)

WAV_MEDIA_TYPE = "audio/wav"


def _parse_emo_vector(raw: str | None) -> list[float] | None:
    """Turn ``"0,0,0.8,0,0,0,0,0"`` into ``[0.0, 0.0, 0.8, ...]``."""
    if raw is None or not raw.strip():
        return None

    parts = [part.strip() for part in raw.split(",")]
    expected = len(EMOTION_ORDER)
    if len(parts) != expected:
        raise HTTPException(
            status_code=422,
            detail=f"`emo_vector` needs {expected} values in the order: {', '.join(EMOTION_ORDER)}",
        )
    try:
        vector = [float(part) for part in parts]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="`emo_vector` values must be numbers") from exc
    if any(value < 0.0 or value > 1.0 for value in vector):
        raise HTTPException(status_code=422, detail="`emo_vector` values must be between 0 and 1")
    return vector


@router.post(
    "/synthesize",
    summary="Synthesize speech with IndexTTS-2.5",
    response_class=Response,
    responses={
        200: {
            "content": {WAV_MEDIA_TYPE: {}},
            "description": "Mono 16-bit PCM WAV at 22.05 kHz.",
        },
        422: {"description": "Invalid request parameters."},
        503: {"description": "Model could not be loaded."},
    },
)
async def synthesize(
    text: str = Form(..., min_length=1, description="Text to speak."),
    lang: str = Form(
        ...,
        description=f"Language of the text; one of {', '.join(SUPPORTED_LANGUAGES)}.",
    ),
    speaker_prompt: UploadFile = File(..., description="Reference voice clip to clone."),
    emo_prompt: UploadFile | None = File(None, description="Optional emotional reference clip."),
    emo_alpha: float = Form(1.0, ge=0.0, le=1.0, description="Emotion strength when `emo_prompt` is set."),
    emo_vector: str | None = Form(None, description=_EMO_VECTOR_DOC),
    emo_text: str | None = Form(None, description="Emotion description used with `use_emo_text`."),
    use_emo_text: bool = Form(False, description="Infer emotion from the text itself."),
    use_random: bool = Form(False, description="Sample emotion vectors randomly."),
    duration_factor: float = Form(1.0, ge=0.5, le=2.0, description="Above 1 is slower, below 1 is faster."),
    text_normalization: bool = Form(True, description="Normalize numbers/dates before synthesis."),
    max_text_tokens_per_segment: int = Form(120, ge=1),
    interval_silence: int = Form(200, ge=0, description="Silence between text segments, in milliseconds."),
    engine: IndexTTS25Engine = Depends(get_engine),
) -> Response:
    """Clone ``speaker_prompt`` and speak ``text`` with it."""
    normalized_lang = lang.strip().upper()
    if normalized_lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=422,
            detail=f"unsupported `lang` {lang!r}; expected one of {', '.join(SUPPORTED_LANGUAGES)}",
        )

    if use_emo_text and not settings.use_qwen_emo:
        raise HTTPException(
            status_code=422,
            detail="`use_emo_text` requires the emotion model; start with INDEX_TTS_USE_QWEN_EMO=true",
        )

    emo_vector_values = _parse_emo_vector(emo_vector)

    speaker_bytes = await speaker_prompt.read()
    if not speaker_bytes:
        raise HTTPException(status_code=400, detail="`speaker_prompt` is empty")

    emo_bytes: bytes | None = None
    if emo_prompt is not None:
        emo_bytes = await emo_prompt.read()
        if not emo_bytes:
            raise HTTPException(status_code=400, detail="`emo_prompt` is empty")

    try:
        request = SynthesisRequest(
            text=text,
            lang=normalized_lang,
            speaker_prompt=engine.materialize_prompt(speaker_bytes, speaker_prompt.filename),
            emo_prompt=(
                None if emo_bytes is None else engine.materialize_prompt(emo_bytes, emo_prompt.filename if emo_prompt else None)
            ),
            emo_alpha=emo_alpha,
            emo_vector=emo_vector_values,
            emo_text=emo_text,
            use_emo_text=use_emo_text,
            use_random=use_random,
            duration_factor=duration_factor,
            text_normalization=text_normalization,
            max_text_tokens_per_segment=max_text_tokens_per_segment,
            interval_silence=interval_silence,
        )
    except TTSError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = await run_in_threadpool(engine.synthesize, request)
    except ModelLoadError as exc:
        raise HTTPException(status_code=503, detail=f"{MODEL_NAME} is unavailable: {exc}") from exc
    except SynthesisError as exc:
        raise HTTPException(status_code=500, detail=f"synthesis failed: {exc}") from exc
    except TTSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Response(
        content=result.audio,
        media_type=WAV_MEDIA_TYPE,
        headers={
            "Content-Disposition": 'inline; filename="speech.wav"',
            "X-Sample-Rate": str(result.sample_rate),
            "X-Audio-Duration": f"{result.duration_seconds:.3f}",
        },
    )
