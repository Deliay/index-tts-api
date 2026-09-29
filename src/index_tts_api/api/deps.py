"""Shared FastAPI dependencies."""

from __future__ import annotations

from fastapi import Request

from index_tts_api.services.tts import IndexTTS25Engine


def get_engine(request: Request) -> IndexTTS25Engine:
    """Return the process-wide IndexTTS-2.5 engine attached to the app."""
    engine: IndexTTS25Engine = request.app.state.engine
    return engine
