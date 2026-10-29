from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.deps import database_session, settings_dependency
from app.core.config import Settings
from app.core.logging import get_logger
from app.schemas.common import HealthResponse
from app.services.prediction_providers import (
    get_prediction_provider,
    resolve_provider_id,
)

logger = get_logger("app.health")
router = APIRouter(prefix="/health", tags=["health"])


def _check_database(session: Session) -> dict[str, Any]:
    try:
        session.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("health_database_failed", extra={"event": "health_database_failed",
                                                         "exception": type(exc).__name__})
        return {"status": "unavailable", "detail": type(exc).__name__}
    return {"status": "ok"}


def _check_model(settings: Settings) -> dict[str, Any]:
    """Report the configured prediction provider, never substituting one for another."""
    try:
        provider_id = resolve_provider_id(settings)
    except Exception as exc:  # noqa: BLE001
        message = getattr(exc, "public_message", None) or str(exc)
        return {"status": "unavailable", "reason": message}
    payload = get_prediction_provider(settings, provider_id).describe().as_dict()
    if payload.get("synthetic"):
        payload["note"] = (
            "The synthetic test provider is selected. Results are synthetic_test_data "
            "and are not flood predictions."
        )
    return payload


@router.get("/live", response_model=HealthResponse, summary="Liveness probe")
def live(settings: Settings = Depends(settings_dependency)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        checks={"process": {"status": "ok"}},
    )


@router.get("/ready", response_model=HealthResponse, summary="Readiness probe")
def ready(
    response: Response,
    session: Session = Depends(database_session),
    settings: Settings = Depends(settings_dependency),
) -> HealthResponse:
    database = _check_database(session)
    if database["status"] != "ok":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="not_ready",
            service=settings.app_name,
            version=settings.app_version,
            environment=settings.environment,
            checks={"database": database},
        )
    return HealthResponse(
        status="ready",
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        checks={"database": database},
    )


@router.get("/model", response_model=HealthResponse, summary="Prediction provider readiness")
def model_readiness(
    settings: Settings = Depends(settings_dependency),
) -> HealthResponse:
    """Report whether flood mapping can run, and with which provider.

    Provider readiness is exposed separately so basic API health never depends on a
    GPU or on trained weights being present.
    """
    check = _check_model(settings)
    return HealthResponse(
        status="ready" if check["status"] == "ok" else "model_unavailable",
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        checks={
            "model": check,
            "checked_at": datetime.now(tz=timezone.utc).isoformat(),
        },
    )


@router.get("/ready/capabilities", response_model=HealthResponse, summary="Accepting work")
def capabilities(
    session: Session = Depends(database_session),
    settings: Settings = Depends(settings_dependency),
) -> HealthResponse:
    """Whether the API can currently accept new analysis work."""
    database = _check_database(session)
    model = _check_model(settings)
    accepting = database["status"] == "ok" and model["status"] == "ok"
    return HealthResponse(
        status="accepting" if accepting else "not_accepting",
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        checks={
            "database": database,
            "model": model,
            "accepts_new_analyses": accepting,
            "produces_real_predictions": bool(accepting and not model.get("synthetic")),
        },
    )