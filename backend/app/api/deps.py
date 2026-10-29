from __future__ import annotations

from collections.abc import Iterator

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.exceptions import AuthenticationError
from app.db.session import SessionFactoryProvider
from app.models.user import AuthSession, User
from app.services.analysis_service import AnalysisService
from app.services.auth_service import AuthService
from app.services.report_service import ReportService


def database_session() -> Iterator[Session]:
    provider = SessionFactoryProvider()
    yield from provider()


def settings_dependency() -> Settings:
    return get_settings()


def get_analysis_service(
    session: Session = Depends(database_session),
    settings: Settings = Depends(settings_dependency),
) -> AnalysisService:
    return AnalysisService(session, settings)


def get_auth_service(
    session: Session = Depends(database_session),
    settings: Settings = Depends(settings_dependency),
) -> AuthService:
    return AuthService(session, settings)


def current_session(
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
    settings: Settings = Depends(settings_dependency),
) -> AuthSession:
    token = request.cookies.get(settings.auth_session_cookie_name)
    auth_session = auth_service.resolve_session(token)
    if auth_session is None:
        raise AuthenticationError("Authentication is required for this endpoint.")
    return auth_session


def current_user(auth_session: AuthSession = Depends(current_session)) -> User:
    return auth_session.user


def get_report_service(settings: Settings = Depends(settings_dependency)) -> ReportService:
    return ReportService(page_size=settings.report_page_size)


def public_base_url(request: Request, settings: Settings = Depends(settings_dependency)) -> str:
    if settings.report_public_base_url:
        return settings.report_public_base_url.rstrip("/")
    return str(request.base_url).rstrip("/")