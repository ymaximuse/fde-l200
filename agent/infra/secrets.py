"""HashiCorp Vault secrets manager integration and swappable backend seam (vault / env / gcp).

Local dev uses HashiCorp Vault (`VaultSecretManager`) via its standard KV v2 HTTP
API (`/v1/<mount>/data/<path>`), reading authentication tokens from `VAULT_TOKEN`
or `~/.vault-token` (standard Vault CLI convention). Production can switch to
`secrets.backend = "gcp"` in `config.json`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Protocol, Sequence

import httpx

from agent.config import AppConfig, get_config

logger = logging.getLogger("fde_duckie.secrets")

DEFAULT_RUNTIME_SECRET_NAMES: tuple[str, ...] = (
    "GOOGLE_API_KEY",
    "OTEL_EXPORTER_OTLP_HEADERS",
)


class SecretManager(Protocol):
    """Backend-agnostic secrets manager interface."""

    def get_secret(self, name: str, *, version: str = "latest") -> str | None:
        ...

    def set_secret(self, name: str, value: str) -> int:
        ...

    def list_secrets(self) -> list[str]:
        ...

    def delete_secret(self, name: str) -> bool:
        ...


def _resolve_vault_token(explicit_token: str | None = None) -> str | None:
    """Resolve Vault token from explicit arg, VAULT_TOKEN env, or ~/.vault-token."""
    if explicit_token:
        return explicit_token.strip()
    env_token = os.getenv("VAULT_TOKEN")
    if env_token:
        return env_token.strip()
    token_file = Path.home() / ".vault-token"
    if token_file.is_file():
        return token_file.read_text(encoding="utf-8").strip() or None
    return None


class VaultSecretManager:
    """HashiCorp Vault KV v2 client over Vault's native HTTP API."""

    def __init__(
        self,
        *,
        addr: str = "http://127.0.0.1:8200",
        mount: str = "secret",
        path: str = "fde-duckie",
        token: str | None = None,
        namespace: str | None = None,
        timeout: float = 5.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._addr = addr.rstrip("/")
        self._mount = mount.strip("/")
        self._path = path.strip("/")
        self._explicit_token = token
        self._namespace = namespace
        self._timeout = timeout
        self._transport = transport

    @property
    def data_url(self) -> str:
        return f"{self._addr}/v1/{self._mount}/data/{self._path}"

    def _headers(self) -> dict[str, str]:
        token = _resolve_vault_token(self._explicit_token)
        if not token:
            raise RuntimeError(
                "Vault token not found. Set VAULT_TOKEN or authenticate via `vault login`."
            )
        headers = {"X-Vault-Token": token}
        if self._namespace:
            headers["X-Vault-Namespace"] = self._namespace
        return headers

    def _client(self) -> httpx.Client:
        return httpx.Client(
            timeout=self._timeout,
            transport=self._transport,
        )

    def _read_kv(self, *, version: str = "latest") -> dict[str, Any]:
        params = {} if version == "latest" else {"version": str(version)}
        with self._client() as client:
            resp = client.get(self.data_url, headers=self._headers(), params=params)
        if resp.status_code == 404:
            return {}
        resp.raise_for_status()
        body = resp.json()
        inner = body.get("data", {})
        data = inner.get("data")
        return dict(data) if isinstance(data, dict) else {}

    def _write_kv(self, data: dict[str, Any]) -> int:
        with self._client() as client:
            resp = client.post(
                self.data_url,
                headers=self._headers(),
                json={"data": data},
            )
        resp.raise_for_status()
        body = resp.json()
        return int(body.get("data", {}).get("version", 1))

    def get_secret(self, name: str, *, version: str = "latest") -> str | None:
        data = self._read_kv(version=version)
        val = data.get(name.strip())
        return str(val) if val is not None else None

    def set_secret(self, name: str, value: str) -> int:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Secret name must be non-empty.")
        current = self._read_kv()
        current[clean_name] = value
        return self._write_kv(current)

    def list_secrets(self) -> list[str]:
        return sorted(self._read_kv().keys())

    def delete_secret(self, name: str) -> bool:
        clean_name = name.strip()
        current = self._read_kv()
        if clean_name not in current:
            return False
        del current[clean_name]
        self._write_kv(current)
        return True


class EnvSecretManager:
    """Environment-variable backed SecretManager (for CI / containers)."""

    def get_secret(self, name: str, *, version: str = "latest") -> str | None:
        return os.getenv(name)

    def set_secret(self, name: str, value: str) -> int:
        os.environ[name] = value
        return 1

    def list_secrets(self) -> list[str]:
        return sorted(
            name for name in DEFAULT_RUNTIME_SECRET_NAMES if os.getenv(name) is not None
        )

    def delete_secret(self, name: str) -> bool:
        return os.environ.pop(name, None) is not None


class GcpSecretManager:
    """Google Cloud Secret Manager adapter (lazy-imports google-cloud-secret-manager)."""

    def __init__(self, project_id: str) -> None:
        if not project_id:
            raise ValueError("secrets.gcp_project_id is required when backend='gcp'.")
        self._project_id = project_id

    def _client(self):
        from google.cloud import secretmanager

        return secretmanager.SecretManagerServiceClient()

    def get_secret(self, name: str, *, version: str = "latest") -> str | None:
        client = self._client()
        resource = f"projects/{self._project_id}/secrets/{name}/versions/{version}"
        try:
            response = client.access_secret_version(request={"name": resource})
            return response.payload.data.decode("utf-8")
        except Exception:
            return None

    def set_secret(self, name: str, value: str) -> int:
        client = self._client()
        parent = f"projects/{self._project_id}/secrets/{name}"
        response = client.add_secret_version(
            request={"parent": parent, "payload": {"data": value.encode("utf-8")}}
        )
        return int(response.name.rsplit("/", 1)[-1])

    def list_secrets(self) -> list[str]:
        client = self._client()
        parent = f"projects/{self._project_id}"
        return sorted(
            s.name.rsplit("/", 1)[-1] for s in client.list_secrets(request={"parent": parent})
        )

    def delete_secret(self, name: str) -> bool:
        client = self._client()
        resource = f"projects/{self._project_id}/secrets/{name}"
        client.delete_secret(request={"name": resource})
        return True


def create_secret_manager(config: AppConfig | None = None) -> SecretManager:
    """Instantiate the configured SecretManager backend."""
    cfg = config or get_config()
    backend = cfg.secrets.backend

    if backend == "vault":
        return VaultSecretManager(
            addr=os.getenv("VAULT_ADDR") or cfg.secrets.vault_addr,
            mount=cfg.secrets.vault_mount,
            path=cfg.secrets.vault_path,
            namespace=os.getenv("VAULT_NAMESPACE") or cfg.secrets.vault_namespace,
        )
    if backend == "env":
        return EnvSecretManager()
    if backend == "gcp":
        project_id = cfg.secrets.gcp_project_id or os.getenv("GOOGLE_CLOUD_PROJECT", "")
        return GcpSecretManager(project_id=project_id)
    raise ValueError(f"Unsupported secrets backend: '{backend}'")


def hydrate_runtime_secrets(
    config: AppConfig | None = None,
    *,
    names: Sequence[str] = DEFAULT_RUNTIME_SECRET_NAMES,
    overwrite: bool = False,
    manager: SecretManager | None = None,
    strict: bool = False,
) -> list[str]:
    """Load runtime secrets from SecretManager into os.environ (without leaking values).

    When ``strict=False`` (default), gracefully skips if Vault is unreachable or
    unauthenticated so local `.env` fallback still works seamlessly.
    """
    active_manager = manager or create_secret_manager(config)
    hydrated: list[str] = []
    for name in names:
        if not overwrite and os.getenv(name):
            continue
        try:
            value = active_manager.get_secret(name)
        except Exception as exc:
            if strict:
                raise
            logger.debug("Skipping secret '%s' hydration from backend: %s", name, exc)
            continue
        if value:
            os.environ[name] = value
            hydrated.append(name)
    return hydrated
