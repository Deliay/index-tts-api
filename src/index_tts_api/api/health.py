"""Health check endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from index_tts_api import __version__
from index_tts_api.api.deps import get_engine
from index_tts_api.config import settings
from index_tts_api.services.tts import MODEL_NAME, IndexTTS25Engine, ModelState

router = APIRouter(tags=["health"])


class ModelStatus(BaseModel):
    """Inference backend state, useful as a readiness signal."""

    name: str
    state: ModelState
    version: str | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """Payload returned by the health check endpoint."""

    status: str
    service: str
    version: str
    model: ModelStatus


@router.get("/health", response_model=HealthResponse, summary="Service health check")
async def health(engine: IndexTTS25Engine = Depends(get_engine)) -> HealthResponse:
    """Report whether the service is up and whether the model is loaded."""
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=__version__,
        model=ModelStatus(
            name=MODEL_NAME,
            state=engine.state,
            version=engine.model_version,
            error=engine.error,
        ),
    )
