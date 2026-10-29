from __future__ import annotations

import hashlib
import secrets
import unicodedata

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

_PASSWORD_HASHER = PasswordHasher()
_DUMMY_PASSWORD = "floodsight-dummy-password-for-timing-equalisation"
_DUMMY_HASH = _PASSWORD_HASHER.hash(_DUMMY_PASSWORD)

SESSION_TOKEN_BYTES = 32


def normalise_email(value: str) -> str:
    """Case-fold and strip an email address for uniqueness comparison."""
    return unicodedata.normalize("NFKC", value).strip().casefold()


def hash_password(password: str) -> str:
    return _PASSWORD_HASHER.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _PASSWORD_HASHER.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def password_needs_rehash(password_hash: str) -> bool:
    try:
        return _PASSWORD_HASHER.check_needs_rehash(password_hash)
    except InvalidHashError:
        return True


def spend_password_time() -> None:
    """Burn one verifier computation so unknown accounts are not distinguishable
    from wrong passwords by response timing."""
    verify_password(_DUMMY_HASH, _DUMMY_PASSWORD)


def generate_session_token() -> str:
    return secrets.token_urlsafe(SESSION_TOKEN_BYTES)


def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def hash_client_key(email_normalized: str, ip_address: str | None) -> str:
    material = f"{email_normalized}|{ip_address or ''}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def constant_time_equals(left: str, right: str) -> bool:
    return secrets.compare_digest(left, right)


__all__ = [
    "SESSION_TOKEN_BYTES",
    "constant_time_equals",
    "generate_session_token",
    "hash_client_key",
    "hash_password",
    "hash_session_token",
    "normalise_email",
    "password_needs_rehash",
    "spend_password_time",
    "verify_password",
]
