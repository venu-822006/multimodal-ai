from __future__ import annotations

import re
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.enums import FailureCode
from app.core.logging import get_logger, request_id_var

logger = get_logger("app.errors")

_ABSOLUTE_PATH = re.compile(r"(?:^|[\s\"'(])(/[^\s\"')]*){2,}")
_SECRET_PATTERN = re.compile(
    r"(?i)(sig|token|password|secret|apikey|api_key)=[^\s&\"']+"
)


class ErrorBody(BaseModel):
    code: str = Field(description="Stable machine readable error code")
    message: str = Field(description="Human readable, sanitized description")
    details: dict[str, Any] | None = None
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class FloodSightError(Exception):
    code: str = "internal_error"
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        failure_code: FailureCode | str | None = None,
        details: dict[str, Any] | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.public_message = message
        self.failure_code = str(failure_code or self.code)
        self.details = details
        if retryable is not None:
            self.retryable = retryable


HTTP_422 = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422)


class DomainValidationError(FloodSightError):
    code = "validation_error"
    status_code = HTTP_422


class NotFoundError(FloodSightError):
    code = "not_found"
    status_code = status.HTTP_404_NOT_FOUND


class ConflictError(FloodSightError):
    code = "conflict"
    status_code = status.HTTP_409_CONFLICT


class ResultNotReadyError(FloodSightError):
    code = "result_not_ready"
    status_code = status.HTTP_202_ACCEPTED


class CapacityError(FloodSightError):
    code = "capacity_exceeded"
    status_code = status.HTTP_429_TOO_MANY_REQUESTS


class AuthenticationError(FloodSightError):
    code = "unauthenticated"
    status_code = status.HTTP_401_UNAUTHORIZED


class ForbiddenError(FloodSightError):
    code = "forbidden"
    status_code = status.HTTP_403_FORBIDDEN


class RateLimitError(FloodSightError):
    code = "too_many_requests"
    status_code = status.HTTP_429_TOO_MANY_REQUESTS


class DependencyError(FloodSightError):
    code = "dependency_unavailable"
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    retryable = True


class UpstreamError(FloodSightError):
    code = "imagery_unavailable"
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    retryable = True


class PermanentProcessingError(FloodSightError):
    code = "processing_error"
    retryable = False


class RetryableProcessingError(FloodSightError):
    code = "processing_error"
    retryable = True


class AnalysisCancelledError(FloodSightError):
    code = "cancelled"
    retryable = False

    def __init__(self, message: str = "Analysis was cancelled before completion.") -> None:
        super().__init__(message, failure_code=FailureCode.CANCELLED)


def sanitise_public_message(message: str, *, limit: int = 480) -> str:
    cleaned = _SECRET_PATTERN.sub(r"\1=[redacted]", str(message))
    cleaned = _ABSOLUTE_PATH.sub("[path]", cleaned)
    cleaned = " ".join(cleaned.split())
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 1].rstrip() + "…"
    return cleaned or "Unspecified error."


def error_payload(
    code: str, message: str, details: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": sanitise_public_message(message),
            "details": details,
            "request_id": request_id_var.get(),
        }
    }


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(FloodSightError)
    async def _handle_domain_error(request: Request, exc: FloodSightError) -> JSONResponse:
        logger.warning(
            "domain_error",
            extra={
                "event": "domain_error",
                "path": request.url.path,
                "error_code": exc.failure_code,
                "retryable": exc.retryable,
                "reason": sanitise_public_message(exc.public_message),
            },
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=error_payload(exc.failure_code, exc.public_message, exc.details),
            headers={"Retry-After": "30"} if exc.status_code == 503 else None,
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        fields = []
        for error in exc.errors():
            location = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
            fields.append(
                {
                    "field": location or "body",
                    "type": error.get("type", "invalid"),
                    "message": str(error.get("msg", "Invalid value")),
                }
            )
        logger.info(
            "request_validation_failed",
            extra={"event": "request_validation_failed", "path": request.url.path,
                   "fields": [item["field"] for item in fields]},
        )
        return JSONResponse(
            status_code=HTTP_422,
            content=error_payload(
                "validation_error", "Request payload failed validation.", {"fields": fields}
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        message = exc.detail if isinstance(exc.detail, str) else "Request could not be processed."
        code = "not_found" if exc.status_code == 404 else "http_error"
        return JSONResponse(
            status_code=exc.status_code,
            content=error_payload(code, message),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(SQLAlchemyError)
    async def _handle_database(request: Request, exc: SQLAlchemyError) -> JSONResponse:
        logger.error(
            "database_error",
            extra={
                "event": "database_error",
                "path": request.url.path,
                "exception": type(exc).__name__,
            },
            exc_info=exc,
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=error_payload(
                "database_unavailable",
                "The analysis database is currently unavailable. Please retry shortly.",
            ),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "unhandled_exception",
            extra={"event": "unhandled_exception", "path": request.url.path,
                   "exception": type(exc).__name__},
            exc_info=exc,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_payload(
                "internal_error",
                "An unexpected server error occurred while processing the request.",
            ),
        )