from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session, selectinload

from app.models.user import AuthSession, LoginAttempt, User


class UserRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, user: User) -> User:
        self._session.add(user)
        self._session.flush()
        return user

    def get(self, user_id: UUID) -> User | None:
        return self._session.get(User, user_id)

    def get_by_normalized_email(self, email_normalized: str) -> User | None:
        return self._session.scalar(
            select(User).where(User.email_normalized == email_normalized)
        )


class AuthSessionRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, auth_session: AuthSession) -> AuthSession:
        self._session.add(auth_session)
        self._session.flush()
        return auth_session

    def get_active_by_token_hash(
        self, token_hash: str, *, now: datetime
    ) -> AuthSession | None:
        return self._session.scalar(
            select(AuthSession)
            .where(
                AuthSession.token_hash == token_hash,
                AuthSession.revoked_at.is_(None),
                AuthSession.expires_at > now,
            )
            .options(selectinload(AuthSession.user))
        )

    def touch(self, auth_session: AuthSession, *, now: datetime) -> None:
        auth_session.last_used_at = now

    def revoke(self, auth_session: AuthSession, *, now: datetime) -> None:
        auth_session.revoked_at = now

    def revoke_all_for_user(self, user_id: UUID, *, now: datetime) -> int:
        result = self._session.execute(
            update(AuthSession)
            .where(AuthSession.user_id == user_id, AuthSession.revoked_at.is_(None))
            .values(revoked_at=now)
        )
        return result.rowcount or 0


class LoginAttemptRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, attempt: LoginAttempt) -> LoginAttempt:
        self._session.add(attempt)
        self._session.flush()
        return attempt

    def get(self, client_hash: str) -> LoginAttempt | None:
        return self._session.scalar(
            select(LoginAttempt).where(LoginAttempt.client_hash == client_hash)
        )

    def reset(self, client_hash: str) -> None:
        self._session.execute(
            delete(LoginAttempt).where(LoginAttempt.client_hash == client_hash)
        )


__all__ = ["AuthSessionRepository", "LoginAttemptRepository", "UserRepository"]
