from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import Field, field_validator

from app.schemas.common import ApiModel

_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_EMAIL_MAX_LENGTH = 320
_PASSWORD_PAYLOAD_MAX_LENGTH = 256


class RegisterRequest(ApiModel):
    email: Annotated[str, Field(min_length=3, max_length=_EMAIL_MAX_LENGTH)]
    password: Annotated[str, Field(min_length=1, max_length=_PASSWORD_PAYLOAD_MAX_LENGTH)]
    full_name: Annotated[str | None, Field(default=None, max_length=200)] = None

    @field_validator("email")
    @classmethod
    def _validate_email(cls, value: str) -> str:
        candidate = value.strip()
        if len(candidate) > _EMAIL_MAX_LENGTH or not _EMAIL_PATTERN.match(candidate):
            raise ValueError("Enter a valid email address.")
        return candidate

    @field_validator("full_name")
    @classmethod
    def _normalise_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = " ".join(value.split())
        return cleaned or None


class LoginRequest(ApiModel):
    email: Annotated[str, Field(min_length=1, max_length=_EMAIL_MAX_LENGTH)]
    password: Annotated[str, Field(min_length=1, max_length=_PASSWORD_PAYLOAD_MAX_LENGTH)]
    remember: bool = True


class UserOut(ApiModel):
    id: UUID
    email: str
    full_name: str | None = None
    is_active: bool
    created_at: datetime


class AuthResponse(ApiModel):
    user: UserOut


__all__ = ["AuthResponse", "LoginRequest", "RegisterRequest", "UserOut"]
