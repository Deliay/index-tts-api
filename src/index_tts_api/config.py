"""Runtime configuration sourced from environment variables."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

ENV_PREFIX = "INDEX_TTS_"

DEFAULT_MODEL_DIR = "checkpoints"


def _env(name: str, default: str) -> str:
    return os.getenv(f"{ENV_PREFIX}{name}", default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(f"{ENV_PREFIX}{name}")
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional(name: str) -> str | None:
    raw = os.getenv(f"{ENV_PREFIX}{name}")
    if raw is None or not raw.strip():
        return None
    return raw.strip()


@dataclass(frozen=True)
class Settings:
    """Application settings, overridable via ``INDEX_TTS_*`` variables."""

    app_name: str
    host: str
    port: int
    reload: bool

    # --- IndexTTS-2.5 inference -------------------------------------------------
    model_dir: str
    """Directory holding the IndexTTS-2.5 checkpoints (``config.yaml`` + weights)."""

    cfg_path: str
    """Path to the model ``config.yaml`` (defaults to ``<model_dir>/config.yaml``)."""

    device: str | None
    """Torch device, e.g. ``cuda:0`` or ``cpu``. ``None`` selects automatically."""

    use_bf16: bool
    use_cuda_kernel: bool
    use_qwen_emo: bool
    use_torch_compile: bool
    verbose: bool
    eager_load: bool
    """Load the model during startup instead of on the first request."""

    work_dir: str
    """Directory for request scratch files (reference-audio cache)."""


def load_settings() -> Settings:
    """Build settings from the current environment."""
    model_dir = str(Path(_env("MODEL_DIR", DEFAULT_MODEL_DIR)).expanduser().absolute())
    cfg_path = _env("CFG_PATH", os.path.join(model_dir, "config.yaml"))
    work_dir = str(Path(_env("WORK_DIR", os.path.join(tempfile.gettempdir(), "index-tts-api"))).expanduser())

    return Settings(
        app_name=_env("APP_NAME", "IndexTTS API"),
        host=_env("HOST", "0.0.0.0"),
        port=int(_env("PORT", "8000")),
        reload=_env_bool("RELOAD", False),
        model_dir=model_dir,
        cfg_path=cfg_path,
        device=_env_optional("DEVICE"),
        use_bf16=_env_bool("USE_BF16", True),
        use_cuda_kernel=_env_bool("USE_CUDA_KERNEL", False),
        use_qwen_emo=_env_bool("USE_QWEN_EMO", False),
        use_torch_compile=_env_bool("USE_TORCH_COMPILE", False),
        verbose=_env_bool("VERBOSE", False),
        eager_load=_env_bool("EAGER_LOAD", False),
        work_dir=work_dir,
    )


settings = load_settings()
