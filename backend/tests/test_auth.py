from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select, update

from app.models.user import AuthSession, User

COOKIE_NAME = "floodsight_session"
VALID_PASSWORD = "correct-horse-42"
TRUSTED_ORIGIN = "http://localhost:3000"


def _register_payload(**overrides):
    payload = {
        "email": "analyst@example.org",
        "password": VALID_PASSWORD,
        "full_name": "Flood Analyst",
    }
    payload.update(overrides)
    return payload


def _register(client, **overrides):
    return client.post("/api/auth/register", json=_register_payload(**overrides))


class TestRegistration:
    def test_valid_registration_creates_account_and_session(self, client):
        response = _register(client)
        assert response.status_code == 201, response.text
        body = response.json()
        assert set(body.keys()) == {"user"}
        assert body["user"]["email"] == "analyst@example.org"
        assert body["user"]["full_name"] == "Flood Analyst"
        assert body["user"]["is_active"] is True
        uuid.UUID(body["user"]["id"])
        assert response.cookies.get(COOKIE_NAME)

    def test_password_is_hashed_and_never_returned(self, client, session):
        raw_password = VALID_PASSWORD
        response = _register(client, password=raw_password)
        assert response.status_code == 201, response.text
        assert raw_password not in response.text
        assert "$argon2" not in response.text
        assert "password_hash" not in response.text

        user = session.scalar(select(User).where(User.email == "analyst@example.org"))
        assert user is not None
        assert user.password_hash != raw_password
        assert user.password_hash.startswith("$argon2id$")

    def test_duplicate_email_is_rejected(self, client):
        assert _register(client).status_code == 201
        duplicate = _register(client)
        assert duplicate.status_code == 409
        assert duplicate.json()["error"]["code"] == "conflict"

    def test_duplicate_email_is_case_insensitive(self, client):
        assert _register(client, email="Analyst@Example.org").status_code == 201
        duplicate = _register(client, email="ANALYST@EXAMPLE.ORG")
        assert duplicate.status_code == 409

    @pytest.mark.parametrize("bad_email", ["not-an-email", "no-at-sign.org", "a@b", "@example.org", "x y@example.org"])
    def test_invalid_email_is_rejected(self, client, bad_email):
        response = _register(client, email=bad_email)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    def test_short_password_is_rejected(self, client):
        response = _register(client, password="short")
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    def test_blank_password_is_rejected(self, client):
        response = _register(client, password="   ")
        assert response.status_code == 422

    def test_missing_fields_are_rejected(self, client):
        response = client.post("/api/auth/register", json={"email": "a@b.org"})
        assert response.status_code == 422

    def test_unknown_fields_are_rejected(self, client):
        response = _register(client, is_admin=True)
        assert response.status_code == 422

    def test_registration_does_not_leak_internal_details(self, client):
        response = _register(client, password="short")
        body = response.json()
        assert "traceback" not in response.text.lower()
        assert body["error"]["message"]
        assert "Traceback" not in body["error"]["message"]

    def test_trusted_origin_is_accepted(self, client):
        response = client.post(
            "/api/auth/register",
            json=_register_payload(),
            headers={"Origin": TRUSTED_ORIGIN},
        )
        assert response.status_code == 201, response.text

    def test_untrusted_origin_is_rejected(self, client):
        response = client.post(
            "/api/auth/register",
            json=_register_payload(),
            headers={"Origin": "http://evil.example"},
        )
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "forbidden"


class TestLogin:
    def test_correct_credentials_authenticate(self, client):
        assert _register(client).status_code == 201
        client.cookies.clear()
        response = client.post(
            "/api/auth/login",
            json={"email": "analyst@example.org", "password": VALID_PASSWORD},
        )
        assert response.status_code == 200, response.text
        assert response.json()["user"]["email"] == "analyst@example.org"
        assert response.cookies.get(COOKIE_NAME)

    def test_wrong_password_is_rejected_generically(self, client):
        assert _register(client).status_code == 201
        client.cookies.clear()
        response = client.post(
            "/api/auth/login",
            json={"email": "analyst@example.org", "password": "wrong-password"},
        )
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "invalid_credentials"
        assert response.json()["error"]["message"] == "Invalid email or password."

    def test_unknown_email_matches_wrong_password_response(self, client):
        assert _register(client).status_code == 201
        client.cookies.clear()
        unknown = client.post(
            "/api/auth/login",
            json={"email": "nobody@example.org", "password": VALID_PASSWORD},
        )
        assert unknown.status_code == 401
        assert unknown.json()["error"]["code"] == "invalid_credentials"
        assert unknown.json()["error"]["message"] == "Invalid email or password."

    def test_malformed_login_request_is_rejected(self, client):
        response = client.post("/api/auth/login", json={"email": "analyst@example.org"})
        assert response.status_code == 422

    def test_login_does_not_accept_password_alone(self, client):
        response = client.post("/api/auth/login", json={"password": VALID_PASSWORD})
        assert response.status_code == 422

    def test_failed_attempts_lock_the_account_temporarily(self, client):
        assert _register(client).status_code == 201
        client.cookies.clear()
        for _ in range(5):
            attempt = client.post(
                "/api/auth/login",
                json={"email": "analyst@example.org", "password": "wrong-password"},
            )
            assert attempt.status_code == 401
        locked = client.post(
            "/api/auth/login",
            json={"email": "analyst@example.org", "password": VALID_PASSWORD},
        )
        assert locked.status_code == 429
        assert locked.json()["error"]["code"] == "too_many_requests"

    def test_successful_login_resets_failed_attempt_counter(self, client):
        assert _register(client).status_code == 201
        client.cookies.clear()
        for _ in range(3):
            client.post(
                "/api/auth/login",
                json={"email": "analyst@example.org", "password": "wrong-password"},
            )
        good = client.post(
            "/api/auth/login",
            json={"email": "analyst@example.org", "password": VALID_PASSWORD},
        )
        assert good.status_code == 200
        client.cookies.clear()
        after = client.post(
            "/api/auth/login",
            json={"email": "analyst@example.org", "password": "wrong-password"},
        )
        assert after.status_code == 401

    def test_remember_me_controls_cookie_persistence(self, client):
        assert _register(client).status_code == 201
        client.cookies.clear()
        persistent = client.post(
            "/api/auth/login",
            json={"email": "analyst@example.org", "password": VALID_PASSWORD, "remember": True},
        )
        assert "Max-Age" in persistent.headers.get("set-cookie", "")
        client.cookies.clear()
        session_only = client.post(
            "/api/auth/login",
            json={"email": "analyst@example.org", "password": VALID_PASSWORD, "remember": False},
        )
        cookie_header = session_only.headers.get("set-cookie", "")
        assert COOKIE_NAME in cookie_header
        assert "Max-Age" not in cookie_header


class TestCurrentUser:
    def test_me_requires_authentication(self, client):
        response = client.get("/api/auth/me")
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "unauthenticated"

    def test_me_returns_safe_profile(self, client):
        assert _register(client).status_code == 201
        response = client.get("/api/auth/me")
        assert response.status_code == 200, response.text
        body = response.json()
        assert set(body.keys()) == {"id", "email", "full_name", "is_active", "created_at"}
        assert body["email"] == "analyst@example.org"
        assert "password" not in response.text

    def test_invalid_cookie_is_rejected(self, client):
        client.cookies.set(COOKIE_NAME, "not-a-real-token")
        response = client.get("/api/auth/me")
        assert response.status_code == 401

    def test_expired_session_is_rejected(self, client, session):
        assert _register(client).status_code == 201
        assert client.get("/api/auth/me").status_code == 200
        session.execute(
            update(AuthSession).values(
                expires_at=datetime.now(tz=timezone.utc) - timedelta(seconds=5)
            )
        )
        session.commit()
        assert client.get("/api/auth/me").status_code == 401

    def test_revoked_session_is_rejected(self, client, session):
        assert _register(client).status_code == 201
        session.execute(
            update(AuthSession).values(revoked_at=datetime.now(tz=timezone.utc))
        )
        session.commit()
        assert client.get("/api/auth/me").status_code == 401

    def test_session_token_is_stored_hashed(self, client, session):
        assert _register(client).status_code == 201
        raw_token = client.cookies.get(COOKIE_NAME)
        stored = session.scalar(select(AuthSession))
        assert stored is not None
        assert stored.token_hash != raw_token
        assert len(stored.token_hash) == 64


class TestLogout:
    def test_logout_invalidates_the_session(self, client):
        assert _register(client).status_code == 201
        assert client.get("/api/auth/me").status_code == 200
        logout = client.post("/api/auth/logout")
        assert logout.status_code == 204
        assert client.get("/api/auth/me").status_code == 401

    def test_reusing_logged_out_credential_fails(self, client):
        assert _register(client).status_code == 201
        stale_cookie = client.cookies.get(COOKIE_NAME)
        assert client.post("/api/auth/logout").status_code == 204
        client.cookies.clear()
        client.cookies.set(COOKIE_NAME, stale_cookie)
        assert client.get("/api/auth/me").status_code == 401

    def test_logout_without_session_is_idempotent(self, client):
        response = client.post("/api/auth/logout")
        assert response.status_code == 204

    def test_logout_clears_the_cookie(self, client):
        assert _register(client).status_code == 201
        response = client.post("/api/auth/logout")
        set_cookie = response.headers.get("set-cookie", "")
        assert COOKIE_NAME in set_cookie


class TestSessionRenewal:
    def test_page_reload_restores_authentication(self, client):
        assert _register(client).status_code == 201
        first = client.get("/api/auth/me")
        reload_view = client.get("/api/auth/me")
        assert first.status_code == 200
        assert reload_view.status_code == 200
        assert reload_view.json()["email"] == first.json()["email"]
