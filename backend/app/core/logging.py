from __future__ import annotations

import json
import logging
import sys
import traceback
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from app.core.config import Settings

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
analysis_id_var: ContextVar[str] = ContextVar("analysis_id", default="-")
job_id_var: ContextVar[str] = ContextVar("job_id", default="-")

_RESERVED = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)

_SENSITIVE_MARKERS = (
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "sig=",
    "signature",
    "private",
    "cookie",
    "session",
    "access_key",
    "client_secret",
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
            "analysis_id": analysis_id_var.get(),
            "job_id": job_id_var.get(),
        }
        if not payload["analysis_id"] or payload["analysis_id"] == "-":
            payload.pop("analysis_id")
        if not payload["job_id"] or payload["job_id"] == "-":
            payload.pop("job_id")
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_") or key in payload:
                continue
            payload[key] = _sanitise_value(value)
        if record.exc_info:
            exc_type, exc_value, exc_tb = record.exc_info
            payload["error"] = {
                "type": exc_type.__name__ if exc_type else "UnknownError",
                "message": _sanitise_text(str(exc_value)),
                "stack": [
                    f"{frame.filename}:{frame.lineno}"
                    for frame in traceback.extract_tb(exc_tb)[-8:]
                ],
            }
        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Human readable formatter that also surfaces structured ``extra`` fields.

    Records logged with ``extra={"event": ...}`` carry no ``%`` placeholders, so the
    base formatter is called without arguments and the context is appended instead.
    """

    def __init__(self, fmt: str | None = None, datefmt: str | None = None) -> None:
        super().__init__(fmt or "%(asctime)s %(levelname)-8s %(name)s %(message)s", datefmt)

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        context = {
            key: _sanitise_value(value)
            for key, value in record.__dict__.items()
            if key not in _RESERVED and not key.startswith("_") and not isinstance(value, bool)
        }
        if context:
            pairs = " ".join(f"{key}={context[key]}" for key in sorted(context))
            return f"{message} | {pairs}"
        return message


def _sanitise_text(value: str) -> str:
    lowered = value.lower()
    if any(marker in lowered for marker in _SENSITIVE_MARKERS):
        return "[redacted]"
    return value


def _sanitise_value(value: Any) -> Any:
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        return _sanitise_text(value)
    if isinstance(value, dict):
        return {k: _sanitise_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitise_value(item) for item in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _sanitise_text(str(value))


def new_request_id() -> str:
    return uuid.uuid4().hex


def configure_logging(settings: Settings) -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    handler = logging.StreamHandler(stream=sys.stdout)
    if settings.log_format.lower() == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(TextFormatter())
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)
    logging.getLogger("uvicorn.access").handlers = []
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("rasterio").setLevel(logging.WARNING)
    logging.getLogger("pystac_client").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)