from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api.router import api_router
from app.core.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, get_logger, request_id_var

logger = get_logger("app.main")

DESCRIPTION = """
Satellite flood extent analysis for FloodSight AI.

Analysis is asynchronous. `POST /api/analyses` persists an analysis and schedules a
worker job, then returns `202 Accepted`; that response confirms the job was recorded,
not that flood detection has succeeded. Poll `/api/analyses/{id}/status` until the
status is terminal, then read `/facts`, `/map` and `/report`.

Flood extents are model predictions derived from Sentinel-1 radar imagery. They are
not confirmed ground truth. Every measurement is reported with the imagery, dates,
model checkpoint and processing configuration used to produce it.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info(
        "application_starting",
        extra={
            "event": "application_starting",
            "environment": settings.environment,
            "version": settings.app_version,
        },
    )
    yield
    from app.db.session import dispose_engine

    dispose_engine()
    logger.info("application_stopped", extra={"event": "application_stopped"})


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or get_settings()
    configure_logging(cfg)

    app = FastAPI(
        title="FloodSight AI Analysis API",
        description=DESCRIPTION,
        version=cfg.app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.cors_allow_origins),
        allow_credentials=cfg.cors_allow_credentials,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Accept"],
        expose_headers=["Location", "Content-Disposition", "Retry-After"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        identifier = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        token = request_id_var.set(identifier)
        started = time.perf_counter()
        try:
            response: Response = await call_next(request)
        except Exception:
            logger.error(
                "request_failed",
                extra={
                    "event": "request_failed",
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                },
                exc_info=True,
            )
            raise
        response.headers["X-Request-ID"] = identifier
        response.headers["X-Response-Time-ms"] = str(
            int((time.perf_counter() - started) * 1000)
        )
        logger.info(
            "request_completed",
            extra={
                "event": "request_completed",
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": int((time.perf_counter() - started) * 1000),
            },
        )
        request_id_var.reset(token)
        return response

    register_exception_handlers(app)
    app.include_router(api_router)

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {
            "service": cfg.app_name,
            "version": cfg.app_version,
            "api": f"{cfg.api_prefix}/analyses",
            "docs": "/docs",
            "health": "/health/live",
        }

    return app


app = create_app()