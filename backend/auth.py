from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUTH_COOKIE_NAME = "cppt_session"
OAUTH_STATE_COOKIE_NAME = "cppt_oauth_state"
OAUTH_VERIFIER_COOKIE_NAME = "cppt_oauth_verifier"


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _now() -> int:
    return int(datetime.now(timezone.utc).timestamp())


class OAuthLoginError(RuntimeError):
    """An OAuth failure safe to expose without leaking credentials."""


@dataclass(frozen=True)
class OAuthClientConfig:
    client_id: str
    app_name: str
    redirect_uri: str


class OAuthClientConfigStore:
    """Resolve the registered Control Center application at runtime."""

    def __init__(self, client_key: str, apps_dsn: str = "", fallback_redirect_uri: str = "") -> None:
        self.client_key = client_key.strip()
        self.apps_dsn = apps_dsn.strip()
        self.fallback_redirect_uri = fallback_redirect_uri.strip()

    def load(self) -> OAuthClientConfig:
        if not self.client_key:
            raise OAuthLoginError("OAuth 客户端应用标识未配置。")
        if not self.apps_dsn:
            if not self.fallback_redirect_uri:
                raise OAuthLoginError("未配置 Control Center apps 查询连接或回调地址。")
            return OAuthClientConfig(self.client_key, self.client_key, self.fallback_redirect_uri)
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise OAuthLoginError("读取 Control Center apps 需要 psycopg。") from exc
        try:
            with psycopg.connect(self.apps_dsn, row_factory=dict_row) as connection:
                rows = connection.execute(
                    """
                    SELECT key AS client_id, name AS app_name,
                           gateway_url AS redirect_uri, is_enabled
                    FROM apps
                    WHERE key = %s AND gateway_url IS NOT NULL
                    """,
                    (self.client_key,),
                ).fetchall()
        except Exception as exc:  # noqa: BLE001 - hide database details from browser
            raise OAuthLoginError("读取 Control Center 应用配置失败。") from exc
        if len(rows) != 1:
            raise OAuthLoginError("Control Center 应用标识不存在或配置重复。")
        row = rows[0]
        enabled = row.get("is_enabled")
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() in {"1", "true", "yes", "on"}
        if not bool(enabled):
            raise OAuthLoginError("Control Center 应用已停用。")
        client_id = str(row.get("client_id") or "").strip()
        redirect_uri = str(row.get("redirect_uri") or "").strip()
        if client_id != self.client_key or not redirect_uri:
            raise OAuthLoginError("Control Center 应用配置无效。")
        return OAuthClientConfig(client_id, str(row.get("app_name") or client_id), redirect_uri)


class AuthStore:
    def __init__(self, backend: str, sqlite_path: Path, postgres_dsn: str = "") -> None:
        self.backend = backend
        self.sqlite_path = sqlite_path
        self.postgres_dsn = postgres_dsn

    def initialize(self) -> None:
        if self.backend == "postgres":
            self._initialize_postgres()
            return
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.sqlite_path)
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cppt_users (
                    id TEXT PRIMARY KEY,
                    sub TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    email TEXT,
                    profile TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def _initialize_postgres(self) -> None:
        with self._pg_connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cppt_users (
                    id TEXT PRIMARY KEY,
                    sub TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    email TEXT,
                    profile JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )

    def _pg_connect(self) -> Any:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise RuntimeError("PostgreSQL 存储需要 psycopg。") from exc
        return psycopg.connect(self.postgres_dsn, row_factory=dict_row)

    @staticmethod
    def _user_from_profile(profile: dict[str, Any]) -> tuple[str, str | None]:
        display_name = str(
            profile.get("name")
            or profile.get("preferred_username")
            or profile.get("username")
            or profile.get("email")
            or "中台用户"
        ).strip()[:200]
        email = profile.get("email")
        return display_name or "中台用户", str(email).strip() if email else None

    def upsert_user(self, profile: dict[str, Any]) -> dict[str, Any]:
        sub = str(profile.get("sub") or "").strip()
        if not sub:
            raise OAuthLoginError("中台用户信息缺少 sub。")
        user_id = f"cppt-{hashlib.sha256(sub.encode('utf-8')).hexdigest()[:32]}"
        display_name, email = self._user_from_profile(profile)
        timestamp = datetime.now(timezone.utc).isoformat()
        serialized = json.dumps(profile, ensure_ascii=False)
        if self.backend == "postgres":
            with self._pg_connect() as connection:
                row = connection.execute(
                    """
                    INSERT INTO cppt_users(id, sub, display_name, email, profile, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s)
                    ON CONFLICT (sub) DO UPDATE SET
                        display_name = EXCLUDED.display_name,
                        email = EXCLUDED.email,
                        profile = EXCLUDED.profile,
                        updated_at = EXCLUDED.updated_at
                    RETURNING id, sub, display_name, email
                    """,
                    (user_id, sub, display_name, email, serialized, timestamp, timestamp),
                ).fetchone()
            return dict(row)
        connection = sqlite3.connect(self.sqlite_path)
        try:
            row = connection.execute(
                """
                INSERT INTO cppt_users(id, sub, display_name, email, profile, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sub) DO UPDATE SET
                    display_name = excluded.display_name,
                    email = excluded.email,
                    profile = excluded.profile,
                    updated_at = excluded.updated_at
                RETURNING id, sub, display_name, email
                """,
                (user_id, sub, display_name, email, serialized, timestamp, timestamp),
            ).fetchone()
            connection.commit()
        finally:
            connection.close()
        return {"id": row[0], "sub": row[1], "display_name": row[2], "email": row[3]}


class SessionManager:
    def __init__(self, secret: str, ttl_seconds: int = 3600) -> None:
        if len(secret.encode("utf-8")) < 32:
            raise RuntimeError("OAUTH_SESSION_SECRET 至少需要 32 字节。")
        self.secret = secret.encode("utf-8")
        self.ttl_seconds = max(60, ttl_seconds)

    def create(self, user: dict[str, Any]) -> str:
        issued = _now()
        payload = {
            "uid": user["id"], "sub": user["sub"],
            "name": user.get("display_name") or "中台用户", "email": user.get("email"),
            "iat": issued, "exp": issued + self.ttl_seconds,
            "nonce": secrets.token_urlsafe(8),
        }
        encoded = _b64encode(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        signature = hmac.new(self.secret, encoded.encode("ascii"), hashlib.sha256).digest()
        return f"{encoded}.{_b64encode(signature)}"

    def read(self, token: str | None) -> dict[str, Any] | None:
        if not token or "." not in token:
            return None
        encoded, signature = token.split(".", 1)
        try:
            expected = hmac.new(self.secret, encoded.encode("ascii"), hashlib.sha256).digest()
            if not hmac.compare_digest(expected, _b64decode(signature)):
                return None
            payload = json.loads(_b64decode(encoded))
        except (ValueError, TypeError, json.JSONDecodeError, UnicodeError):
            return None
        if not isinstance(payload, dict) or int(payload.get("exp", 0)) <= _now():
            return None
        return payload


def create_pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return verifier, _b64encode(digest)


def exchange_code(*, token_url: str, client_id: str, redirect_uri: str, code: str, code_verifier: str, timeout: float) -> dict[str, Any]:
    form = urllib.parse.urlencode({
        "grant_type": "authorization_code", "client_id": client_id,
        "redirect_uri": redirect_uri, "code": code, "code_verifier": code_verifier,
    }).encode("utf-8")
    request = urllib.request.Request(token_url, data=form, headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        raise OAuthLoginError("中台 token 请求失败。") from exc
    if not isinstance(result, dict) or not result.get("access_token"):
        raise OAuthLoginError("中台 token 响应无效。")
    return result


def fetch_userinfo(*, userinfo_url: str, access_token: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(userinfo_url, headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        raise OAuthLoginError("中台 userinfo 请求失败。") from exc
    if not isinstance(result, dict):
        raise OAuthLoginError("中台 userinfo 响应无效。")
    return result
