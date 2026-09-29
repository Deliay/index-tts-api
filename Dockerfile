# syntax=docker/dockerfile:1

# =============================================================================
# Builder — resolve the locked environment into a self-contained virtualenv.
#
# No compiler is required: every locked dependency ships a wheel except seven
# pure-Python sdists (antlr4-python3-runtime, argbind, distance, jieba,
# openai-whisper, randomname, unidic-lite).
# =============================================================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /uvx /usr/local/bin/

ENV DEBIAN_FRONTEND=noninteractive \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_DOWNLOADS=automatic \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer — metadata only, so editing the source keeps this cache warm.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-dev --no-editable --no-install-project

# Application layer.
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

# Pre-build the WeTextProcessing tagger caches (2.4 MB of .fst files) so the
# runtime reuses them and never writes into site-packages on first request.
RUN /opt/venv/bin/python -c "from indextts.utils.front import TextNormalizer; TextNormalizer(enable_glossary=True).load()"

# =============================================================================
# Runtime — the CUDA runtime plus the virtualenv. No toolchain, no checkpoints.
# =============================================================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime

# libgomp1 is the only extra system library torch and llvmlite need: soundfile
# bundles its own libsndfile and nothing on the inference path imports OpenCV.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1000 --user-group app

# The virtualenv is not relocatable: its interpreter lives under /opt/python.
COPY --from=builder /opt/python /opt/python
COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    INDEX_TTS_HOST=0.0.0.0 \
    INDEX_TTS_PORT=8000 \
    INDEX_TTS_MODEL_DIR=/app/checkpoints \
    INDEX_TTS_WORK_DIR=/tmp/index-tts-api

WORKDIR /app
RUN mkdir -p /app/checkpoints /tmp/index-tts-api \
 && chown app:app /app/checkpoints /tmp/index-tts-api

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# Checkpoints are mounted at /app/checkpoints (see README), never baked in.
CMD ["index-tts-api"]
