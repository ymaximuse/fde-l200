"""Offline tests for SQLite persistence, seeding, and Vault secrets."""

import asyncio
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

import httpx

from agent.config import AppConfig, DatabaseConfig, SecretsConfig
from agent.infra.backends import get_profile_store, set_profile_store
from agent.infra.db import SqliteNooglerProfileStore, configure_storage
from agent.infra.secrets import (
    EnvSecretManager,
    VaultSecretManager,
    create_secret_manager,
    hydrate_runtime_secrets,
)
from agent.mocks import MockNooglerProfileStore
from agent.tools.enablement_tools import update_enablement_progress
from scripts.setup_db import seed_local_db
from tests.helpers import make_context


class TestSqlitePersistenceAndSeeding(unittest.TestCase):
    def tearDown(self) -> None:
        set_profile_store(MockNooglerProfileStore())

    def test_setup_script_seeds_profiles_sessions_and_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "local_dev.db"
            summary = asyncio.run(
                seed_local_db(db_path=db_path, app_name="agent", reset=True)
            )
            self.assertEqual(
                summary["all_profiles"], ["alex_noogler", "jordan_fde", "priya_fde"]
            )
            self.assertEqual(len(summary["seeded_sessions"]), 3)

            # Verify both ADK session tables and noogler_profiles table exist.
            with sqlite3.connect(db_path) as conn:
                tables = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
            self.assertTrue(
                {
                    "sessions",
                    "events",
                    "app_states",
                    "user_states",
                    "noogler_profiles",
                }.issubset(tables)
            )

            # Idempotent re-run does not duplicate rows or sessions.
            rerun = asyncio.run(
                seed_local_db(db_path=db_path, app_name="agent", reset=False)
            )
            self.assertEqual(rerun["seeded_profiles"], [])
            self.assertEqual(rerun["seeded_sessions"], [])

    def test_profile_updates_persist_to_sqlite_profile_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "profiles.db"
            cfg = AppConfig(
                database=DatabaseConfig(backend="sqlite", url=f"sqlite:///{db_path}")
            )
            session_service = configure_storage(cfg)
            try:
                self.assertIsInstance(get_profile_store(), SqliteNooglerProfileStore)
                ctx = make_context("priya_fde")
                res = update_enablement_progress(
                    "L300", "in_progress", labs_completed=6, tool_context=ctx
                )
                self.assertEqual(res["status"], "ok")

                # Verify fresh store instance reads the updated 6/7 L300 progress.
                fresh_store = SqliteNooglerProfileStore(db_path)
                reloaded_profile = fresh_store.get_profile("priya_fde")
                self.assertEqual(
                    reloaded_profile["enablement"]["L300"]["labs_completed"], 6
                )
                self.assertEqual(
                    reloaded_profile["enablement"]["L400"]["status"], "not_started"
                )
            finally:
                asyncio.run(session_service.close())


class TestVaultSecrets(unittest.TestCase):
    def _make_fake_vault_transport(self) -> tuple[httpx.MockTransport, list[dict]]:
        versions: list[dict[str, str]] = []
        seen_headers: list[dict] = []

        def _handler(request: httpx.Request) -> httpx.Response:
            seen_headers.append(dict(request.headers))
            if request.headers.get("x-vault-token") != "test-vault-token":
                return httpx.Response(403, json={"errors": ["permission denied"]})

            if request.url.path != "/v1/secret/data/fde-duckie":
                return httpx.Response(404, json={"errors": []})

            if request.method == "GET":
                if not versions:
                    return httpx.Response(404, json={"errors": []})
                ver_param = request.url.params.get("version")
                if ver_param is not None:
                    idx = int(ver_param) - 1
                    if idx < 0 or idx >= len(versions):
                        return httpx.Response(404, json={"errors": []})
                    payload = versions[idx]
                    ver = idx + 1
                else:
                    payload = versions[-1]
                    ver = len(versions)
                return httpx.Response(
                    200,
                    json={"data": {"data": payload, "metadata": {"version": ver}}},
                )

            if request.method == "POST":
                body = json.loads(request.content.decode("utf-8"))
                versions.append(dict(body.get("data", {})))
                return httpx.Response(200, json={"data": {"version": len(versions)}})

            return httpx.Response(405)

        return httpx.MockTransport(_handler), seen_headers

    def test_vault_kv_v2_crud_and_versioning(self) -> None:
        transport, seen_headers = self._make_fake_vault_transport()
        vault = VaultSecretManager(
            addr="http://127.0.0.1:8200",
            mount="secret",
            path="fde-duckie",
            token="test-vault-token",
            namespace="fde-ns",
            transport=transport,
        )

        self.assertEqual(vault.list_secrets(), [])
        self.assertIsNone(vault.get_secret("GOOGLE_API_KEY"))

        v1 = vault.set_secret("GOOGLE_API_KEY", "key-v1")
        v2 = vault.set_secret("GOOGLE_API_KEY", "key-v2")
        self.assertEqual(v1, 1)
        self.assertEqual(v2, 2)
        self.assertEqual(vault.get_secret("GOOGLE_API_KEY"), "key-v2")
        self.assertEqual(vault.get_secret("GOOGLE_API_KEY", version="1"), "key-v1")
        self.assertEqual(vault.list_secrets(), ["GOOGLE_API_KEY"])

        self.assertTrue(vault.delete_secret("GOOGLE_API_KEY"))
        self.assertFalse(vault.delete_secret("GOOGLE_API_KEY"))
        self.assertIsNone(vault.get_secret("GOOGLE_API_KEY"))
        self.assertTrue(
            all(h.get("x-vault-namespace") == "fde-ns" for h in seen_headers)
        )

    def test_hydrate_runtime_secrets_populates_env_and_respects_existing(self) -> None:
        transport, _ = self._make_fake_vault_transport()
        vault = VaultSecretManager(token="test-vault-token", transport=transport)
        vault.set_secret("TEST_HYDRATE_KEY", "secret-from-vault")

        os.environ.pop("TEST_HYDRATE_KEY", None)
        try:
            loaded = hydrate_runtime_secrets(
                names=("TEST_HYDRATE_KEY",), manager=vault, strict=True
            )
            self.assertEqual(loaded, ["TEST_HYDRATE_KEY"])
            self.assertEqual(os.environ["TEST_HYDRATE_KEY"], "secret-from-vault")

            # Does not overwrite already-set env var when overwrite=False.
            vault.set_secret("TEST_HYDRATE_KEY", "new-secret")
            loaded_again = hydrate_runtime_secrets(
                names=("TEST_HYDRATE_KEY",), manager=vault, overwrite=False, strict=True
            )
            self.assertEqual(loaded_again, [])
            self.assertEqual(os.environ["TEST_HYDRATE_KEY"], "secret-from-vault")
        finally:
            os.environ.pop("TEST_HYDRATE_KEY", None)

    def test_create_secret_manager_backends_and_graceful_fallback(self) -> None:
        vault_cfg = AppConfig(secrets=SecretsConfig(backend="vault"))
        self.assertIsInstance(create_secret_manager(vault_cfg), VaultSecretManager)

        env_cfg = AppConfig(secrets=SecretsConfig(backend="env"))
        self.assertIsInstance(create_secret_manager(env_cfg), EnvSecretManager)

        # Unauthenticated Vault fails gracefully when strict=False.
        broken_vault = VaultSecretManager(token=None)
        old_token = os.environ.pop("VAULT_TOKEN", None)
        try:
            hydrated = hydrate_runtime_secrets(
                names=("NONEXISTENT_KEY_XYZ",), manager=broken_vault, strict=False
            )
            self.assertEqual(hydrated, [])
        finally:
            if old_token is not None:
                os.environ["VAULT_TOKEN"] = old_token


if __name__ == "__main__":
    unittest.main()
