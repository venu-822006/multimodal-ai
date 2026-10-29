from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import AuthenticationError, ConflictError, DomainValidationError, RateLimitError
from app.core.logging import get_logger
from app.core.security import (
    generate_session_token,
    hash_client_key,
    hash_password,
    hash_session_token,
    normalise_email,
    password_needs_rehash,
    spend_password_time,
    verify_password,
)
from app.models.analysis import utcnow
from app.models.user import AuthSession, LoginAttempt, User
from app.repositories.auth import (
    AuthSessionRepository,
    LoginAttemptRepository,
    UserRepository,
)

logger = get_logger("app.auth")

_INVALID_CREDENTIALS_MESSAGE = "Invalid email or password."


def _truncate(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned[:limit] if cleaned else None


class AuthService:
    def __init__(self, session: Session, settings: Settings) -> None:
        self._session = session
        self._settings = settings
        self._users = UserRepository(session)
        self._sessions = AuthSessionRepository(session)
        self._attempts = LoginAttemptRepository(session)

    def register(
        self,
        *,
        email: str,
        password: str,
        full_name: str | None,
        user_agent: str | None,
        ip_address: str | None,
    ) -> tuple[User, str]:
        self._validate_password(password)
        email_normalized = normalise_email(email)
        if self._users.get_by_normalized_email(email_normalized) is not None:
            raise ConflictError("An account with this email already exists.")
        user = User(
            email=email.strip(),
            email_normalized=email_normalized,
            password_hash=hash_password(password),
            full_name=full_name,
            is_active=True,
        )
        try:
            self._users.add(user)
        except IntegrityError:
            self._session.rollback()
            raise ConflictError("An account with this email already exists.") from None
        token = self._create_session(
            user, user_agent=user_agent, ip_address=ip_address, now=utcnow()
        )
        self._session.commit()
        logger.info("user_registered", extra={"event": "user_registered", "user_id": str(user.id)})
        return user, token

    def authenticate(
        self,
        *,
        email: str,
        password: str,
        user_agent: str | None,
        ip_address: str | None,
    ) -> tuple[User, str]:
        now = utcnow()
        email_normalized = normalise_email(email)
        client_hash = hash_client_key(email_normalized, ip_address)
        self._raise_if_locked(client_hash, now=now)
        user = self._users.get_by_normalized_email(email_normalized)
        if user is None or not user.is_active:
            spend_password_time()
            self._record_failure(client_hash, now=now)
            self._session.commit()
            raise AuthenticationError(
                _INVALID_CREDENTIALS_MESSAGE, failure_code="invalid_credentials"
            )
        if not verify_password(user.password_hash, password):
            self._record_failure(client_hash, now=now)
            self._session.commit()
            raise AuthenticationError(
                _INVALID_CREDENTIALS_MESSAGE, failure_code="invalid_credentials"
            )
        self._attempts.reset(client_hash)
        if password_needs_rehash(user.password_hash):
            user.password_hash = hash_password(password)
        token = self._create_session(
            user, user_agent=user_agent, ip_address=ip_address, now=now
        )
        self._session.commit()
        logger.info("user_signed_in", extra={"event": "user_signed_in", "user_id": str(user.id)})
        return user, token

    def resolve_session(self, token: str | None, *, now: datetime | None = None) -> AuthSession | None:
        if not token:
            return None
        moment = now or utcnow()
        auth_session = self._sessions.get_active_by_token_hash(
            hash_session_token(token), now=moment
        )
        if auth_session is None:
            return None
        user = auth_session.user
        if user is None or not user.is_active:
            return None
        elapsed = (moment - auth_session.last_used_at).total_seconds()
        if elapsed >= self._settings.auth_login_touch_interval_seconds:
            self._sessions.touch(auth_session, now=moment)
            self._session.commit()
        return auth_session

    def logout(self, token: str | None, *, now: datetime | None = None) -> bool:
        if not token:
            return False
        moment = now or utcnow()
        auth_session = self._sessions.get_active_by_token_hash(
            hash_session_token(token), now=moment
        )
        if auth_session is None:
            return False
        self._sessions.revoke(auth_session, now=moment)
        self._session.commit()
        logger.info(
            "user_signed_out",
            extra={"event": "user_signed_out", "user_id": str(auth_session.user_id)},
        )
        return True

    def _validate_password(self, password: str) -> None:
        minimum = self._settings.auth_password_min_length
        maximum = self._settings.auth_password_max_length
        if not password.strip():
            raise DomainValidationError("Password must not be blank.")
        if len(password) < minimum:
            raise DomainValidationError(f"Password must be at least {minimum} characters.")
        if len(password) > maximum:
            raise DomainValidationError(f"Password must be at most {maximum} characters.")

    def _create_session(
        self,
        user: User,
        *,
        user_agent: str | None,
        ip_address: str | None,
        now: datetime,
    ) -> str:
        token = generate_session_token()
        auth_session = AuthSession(
            user_id=user.id,
            token_hash=hash_session_token(token),
            user_agent=_truncate(user_agent, 300),
            ip_address=_truncate(ip_address, 64),
            created_at=now,
            last_used_at=now,
            expires_at=now + timedelta(seconds=self._settings.auth_session_ttl_seconds),
        )
        self._sessions.add(auth_session)
        return token

    def _raise_if_locked(self, client_hash: str, *, now: datetime) -> None:
        attempt = self._attempts.get(client_hash)
        if attempt is None or attempt.locked_until is None:
            return
        if attempt.locked_until > now:
            retry_after = max(1, int((attempt.locked_until - now).total_seconds()))
            raise RateLimitError(
                "Too many failed sign-in attempts. Please try again later.",
                details={"retry_after_seconds": retry_after},
            )

    def _record_failure(self, client_hash: str, *, now: datetime) -> None:
        attempt = self._attempts.get(client_hash)
        if attempt is None:
            attempt = LoginAttempt(client_hash=client_hash, failed_attempts=0)
            self._attempts.add(attempt)
        attempt.failed_attempts += 1
        attempt.last_failed_at = now
        if attempt.failed_attempts >= self._settings.auth_login_max_attempts:
            attempt.locked_until = now + timedelta(
                seconds=self._settings.auth_login_lockout_seconds
            )


__all__ = ["AuthService"]
