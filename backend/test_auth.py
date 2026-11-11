from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.auth import AuthStore, OAuthClientConfigStore, SessionManager, create_pkce


class AuthStoreTests(unittest.TestCase):
    def test_sub_maps_to_stable_local_user(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = AuthStore("sqlite", Path(directory) / "runtime.db")
            store.initialize()
            first = store.upsert_user({"sub": "central-42", "name": "Alice", "email": "a@example.com"})
            second = store.upsert_user({"sub": "central-42", "name": "Alice Updated"})
            self.assertEqual(first["id"], second["id"])
            self.assertEqual(second["display_name"], "Alice Updated")


class SessionManagerTests(unittest.TestCase):
    def test_signed_session_round_trip_and_tamper_detection(self) -> None:
        manager = SessionManager("t" * 32, ttl_seconds=3600)
        token = manager.create({"id": "cppt-user", "sub": "central-42", "display_name": "Alice"})
        payload = manager.read(token)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["uid"], "cppt-user")
        self.assertIsNone(manager.read(token + "x"))

    def test_expired_session_is_rejected(self) -> None:
        manager = SessionManager("t" * 32, ttl_seconds=60)
        with patch("backend.auth._now", side_effect=[1000, 1061]):
            token = manager.create({"id": "cppt-user", "sub": "central-42"})
            self.assertIsNone(manager.read(token))


class OAuthConfigTests(unittest.TestCase):
    def test_local_fallback_uses_registered_client_key(self) -> None:
        config = OAuthClientConfigStore("cppt-platform", fallback_redirect_uri="http://localhost:3100/").load()
        self.assertEqual(config.client_id, "cppt-platform")
        self.assertEqual(config.redirect_uri, "http://localhost:3100/")

    def test_pkce_challenge_is_url_safe_and_verifier_is_random(self) -> None:
        verifier, challenge = create_pkce()
        self.assertGreaterEqual(len(verifier), 43)
        self.assertNotIn("=", challenge)
        self.assertNotEqual(verifier, challenge)


if __name__ == "__main__":
    unittest.main()
