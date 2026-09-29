"""IndexTTS-2.5 inference engine.

This module wraps ``indextts.infer_v2_5.IndexTTS2`` — the IndexTTS-2.5 pipeline
from https://github.com/index-tts/index-tts. Only that pipeline is supported:
the module is imported lazily inside :meth:`IndexTTS25Engine._build_model` so the
API can start (and serve ``/health``) without a GPU or downloaded checkpoints.
"""

from __future__ import annotations

import hashlib
import io
import logging
import threading
import wave
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from index_tts_api.config import Settings

logger = logging.getLogger(__name__)

MODEL_NAME = "IndexTTS-2.5"

#: Languages IndexTTS-2.5 is trained on.
SUPPORTED_LANGUAGES: tuple[str, ...] = ("ZH", "EN", "JA", "ES", "AR")

#: Order of the 8 floats in the ``emo_vector`` control input.
EMOTION_ORDER: tuple[str, ...] = (
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "melancholic",
    "surprised",
    "calm",
)

#: Audio extensions we are willing to hand to librosa.
_AUDIO_SUFFIXES = frozenset(
    {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".oga", ".opus", ".aac", ".wma", ".webm"}
)


class ModelState(str, Enum):
    """Lifecycle state of the inference engine."""

    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


class TTSError(RuntimeError):
    """Base class for synthesis failures."""


class ModelLoadError(TTSError):
    """The IndexTTS-2.5 checkpoints could not be loaded."""


class SynthesisError(TTSError):
    """Inference ran but did not produce usable audio."""


@dataclass(frozen=True)
class SynthesisRequest:
    """A single, already-validated synthesis request."""

    text: str
    lang: str
    speaker_prompt: Path
    emo_prompt: Path | None = None
    emo_alpha: float = 1.0
    emo_vector: list[float] | None = None
    emo_text: str | None = None
    use_emo_text: bool = False
    use_random: bool = False
    duration_factor: float = 1.0
    text_normalization: bool = True
    max_text_tokens_per_segment: int = 120
    interval_silence: int = 200


@dataclass(frozen=True)
class SynthesisResult:
    """Generated audio plus a little metadata."""

    audio: bytes
    sample_rate: int
    duration_seconds: float


def pcm16_wav(sample_rate: int, samples: np.ndarray) -> bytes:
    """Encode mono PCM-16 ``samples`` as an in-memory WAV file."""
    array = np.asarray(samples)
    if array.ndim == 2:
        array = array.squeeze()
    if array.ndim != 1:
        raise SynthesisError(f"expected a mono waveform, got shape {np.asarray(samples).shape}")
    if array.dtype != np.int16:
        array = np.clip(array, -32768, 32767).astype(np.int16)

    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(int(sample_rate))
            handle.writeframes(array.tobytes())
        return buffer.getvalue()


class IndexTTS25Engine:
    """Thread-safe, lazily-loaded IndexTTS-2.5 model.

    The underlying pipeline is a single GPU-resident model that keeps
    per-reference-audio caches, so calls are serialized behind a lock. Run
    :meth:`synthesize` from a worker thread (see ``starlette.concurrency``).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: object | None = None
        self._model_version: str | None = None
        self._state = ModelState.NOT_LOADED
        self._error: str | None = None
        self._lock = threading.RLock()
        self._prompt_dir = Path(settings.work_dir) / "prompts"
        self._prompt_dir.mkdir(parents=True, exist_ok=True)

    # -- state -------------------------------------------------------------------

    @property
    def state(self) -> ModelState:
        return self._state

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def is_ready(self) -> bool:
        return self._state is ModelState.READY

    @property
    def model_version(self) -> str | None:
        return self._model_version

    # -- lifecycle ---------------------------------------------------------------

    def load(self) -> None:
        """Load the checkpoints if they are not already resident."""
        with self._lock:
            self._load_locked()

    def unload(self) -> None:
        """Drop the model and release GPU memory."""
        with self._lock:
            self._model = None
            self._model_version = None
            self._state = ModelState.NOT_LOADED
            self._error = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # pragma: no cover - best-effort cleanup
                logger.debug("torch cleanup skipped", exc_info=True)

    def _load_locked(self) -> None:
        if self._state is ModelState.READY:
            return

        self._state = ModelState.LOADING
        self._error = None
        try:
            self._model = self._build_model()
        except Exception as exc:
            self._state = ModelState.ERROR
            self._error = str(exc) if isinstance(exc, TTSError) else f"{type(exc).__name__}: {exc}"
            logger.exception("Failed to load %s", MODEL_NAME)
            if isinstance(exc, TTSError):
                raise
            raise ModelLoadError(self._error) from exc
        self._state = ModelState.READY

    def _build_model(self) -> object:
        # Imported here: pulls in torch and friends, which is slow and must not
        # be required just to start the API.
        from indextts.infer_v2_5 import IndexTTS2

        settings = self._settings
        logger.info("Loading %s from %s", MODEL_NAME, settings.model_dir)
        model = IndexTTS2(
            cfg_path=settings.cfg_path,
            model_dir=settings.model_dir,
            use_bf16=settings.use_bf16,
            device=settings.device,
            use_cuda_kernel=settings.use_cuda_kernel,
            use_qwen_emo=settings.use_qwen_emo,
            use_torch_compile=settings.use_torch_compile,
        )

        version = getattr(model, "model_version", None)
        self._model_version = None if version is None else str(version)
        # `config.yaml` declares `version: 2.5` for the checkpoints we support;
        # reject anything else rather than silently serving another model.
        if self._model_version is not None and not self._model_version.startswith("2.5"):
            raise ModelLoadError(
                f"checkpoints declare version {self._model_version!r}, but only IndexTTS-2.5 is supported"
            )
        return model

    # -- inference ---------------------------------------------------------------

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        """Run inference, blocking until the whole clip has been generated."""
        with self._lock:
            if self._state is not ModelState.READY:
                self._load_locked()
            return self._synthesize_locked(request)

    def _synthesize_locked(self, request: SynthesisRequest) -> SynthesisResult:
        model = self._model
        if model is None:  # pragma: no cover - guarded by _load_locked
            raise ModelLoadError("model is not loaded")

        infer = getattr(model, "infer")
        try:
            output = infer(
                spk_audio_prompt=str(request.speaker_prompt),
                text=request.text,
                lang=request.lang,
                output_path=None,
                emo_audio_prompt=None if request.emo_prompt is None else str(request.emo_prompt),
                emo_alpha=request.emo_alpha,
                emo_vector=request.emo_vector,
                emo_text=request.emo_text,
                use_emo_text=request.use_emo_text,
                use_random=request.use_random,
                duration_factor=request.duration_factor,
                text_normalization=request.text_normalization,
                max_text_tokens_per_segment=request.max_text_tokens_per_segment,
                interval_silence=request.interval_silence,
                verbose=self._settings.verbose,
            )
        except TTSError:
            raise
        except Exception as exc:
            raise SynthesisError(f"{type(exc).__name__}: {exc}") from exc

        if not (isinstance(output, tuple) and len(output) == 2):
            raise SynthesisError("IndexTTS-2.5 returned no audio")

        sample_rate, samples = output
        sample_rate = int(sample_rate)
        audio = pcm16_wav(sample_rate, samples)
        return SynthesisResult(
            audio=audio,
            sample_rate=sample_rate,
            duration_seconds=float(np.asarray(samples).size) / sample_rate,
        )

    # -- request scratch files ---------------------------------------------------

    def materialize_prompt(self, data: bytes, filename: str | None) -> Path:
        """Persist a reference clip, reusing the same path for identical bytes.

        Stable paths across identical uploads matter: the engine caches speaker
        embeddings per ``spk_audio_prompt`` and skips recomputation on a hit.
        """
        if not data:
            raise TTSError("reference audio is empty")

        suffix = Path(filename or "").suffix.lower()
        if suffix not in _AUDIO_SUFFIXES:
            suffix = ".wav"

        path = self._prompt_dir / f"{hashlib.sha256(data).hexdigest()}{suffix}"
        if not path.exists():
            temporary = path.with_name(f"{path.name}.{threading.get_ident()}.part")
            temporary.write_bytes(data)
            temporary.replace(path)
        return path
