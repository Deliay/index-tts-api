"""Application entry point for the IndexTTS API."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from index_tts_api import __version__
from index_tts_api.api.health import router as health_router
from index_tts_api.api.synthesize import router as synthesize_router
from index_tts_api.config import settings
from index_tts_api.services.tts import IndexTTS25Engine, ModelLoadError

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Optionally warm the model up, then release it on shutdown."""
    if settings.eager_load:
        try:
            await run_in_threadpool(app.state.engine.load)
        except ModelLoadError:
            # Keep serving so /health can report why the backend is unavailable.
            logger.exception("Eager model load failed; /health will report the error")
    try:
        yield
    finally:
        app.state.engine.unload()


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        lifespan=lifespan,
    )
    app.state.engine = IndexTTS25Engine(settings)
    app.include_router(health_router)
    app.include_router(synthesize_router)
    return app


app = create_app()


def main() -> None:
    """Serve the API with uvicorn (``uv run index-tts-api``)."""
    uvicorn.run(
        "index_tts_api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
    )


if __name__ == "__main__":
    main()
