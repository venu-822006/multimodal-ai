from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request, Response, status

from app.api.deps import current_user, get_auth_service, settings_dependency
from app.core.config import Settings
from app.core.exceptions import ForbiddenError
from app.core.logging import get_logger
from app.models.user import User
from app.schemas.auth import AuthResponse, LoginRequest, RegisterRequest, UserOut
from app.services.auth_service import AuthService

logger = get_logger("app.api.auth")

router = APIRouter(prefix="/auth", tags=["auth"])


def _client_ip(request: Request) -> str | None:
    return request.client.host if request.client is not None else None


def _assert_trusted_origin(request: Request, settings: Settings) -> None:
    """Defence in depth for cookie-authenticated state changes.

    SameSite already blocks cross-site cookie transmission in modern browsers; this
    additionally rejects a state-changing request whose declared Origin is not a
    configured client origin. Non-browser clients omit Origin and are unaffected.
    """
    origin = request.headers.get("origin")
    if not origin:
        return
    allowed = {value.rstrip("/") for value in settings.cors_allow_origins}
    if origin.rstrip("/") not in allowed:
        raise ForbiddenError("This request did not originate from a trusted client.")


def _session_cookie(settings: Settings) -> dict[str, Any]:
    return {
        "key": settings.auth_session_cookie_name,
        "httponly": True,
        "secure": settings.auth_session_cookie_secure,
        "samesite": settings.auth_session_cookie_samesite,
        "path": "/",
        "domain": settings.auth_session_cookie_domain,
    }


def _attach_session_cookie(
    response: Response, token: str, settings: Settings, *, persistent: bool
) -> None:
    max_age = settings.auth_session_ttl_seconds if persistent else None
    response.set_cookie(value=token, max_age=max_age, **_session_cookie(settings))


@router.post(
    "/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register an account",
)
def register(
    payload: RegisterRequest,
    request: Request,
    response: Response,
    service: AuthService = Depends(get_auth_service),
    settings: Settings = Depends(settings_dependency),
) -> AuthResponse:
    _assert_trusted_origin(request, settings)
    user, token = service.register(
        email=payload.email,
        password=payload.password,
        full_name=payload.full_name,
        user_agent=request.headers.get("user-agent"),
        ip_address=_client_ip(request),
    )
    _attach_session_cookie(response, token, settings, persistent=True)
    return AuthResponse(user=UserOut.model_validate(user))


@router.post("/login", response_model=AuthResponse, summary="Sign in")
def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    service: AuthService = Depends(get_auth_service),
    settings: Settings = Depends(settings_dependency),
) -> AuthResponse:
    _assert_trusted_origin(request, settings)
    user, token = service.authenticate(
        email=payload.email,
        password=payload.password,
        user_agent=request.headers.get("user-agent"),
        ip_address=_client_ip(request),
    )
    _attach_session_cookie(response, token, settings, persistent=payload.remember)
    return AuthResponse(user=UserOut.model_validate(user))


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT, summary="Sign out")
def logout(
    request: Request,
    service: AuthService = Depends(get_auth_service),
    settings: Settings = Depends(settings_dependency),
) -> Response:
    _assert_trusted_origin(request, settings)
    service.logout(request.cookies.get(settings.auth_session_cookie_name))
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    response.delete_cookie(**_session_cookie(settings))
    return response


@router.get("/me", response_model=UserOut, summary="Current user")
def me(user: User = Depends(current_user)) -> UserOut:
    return UserOut.model_validate(user)


__all__ = ["router"]
