"""Infrastructure seams: backend protocols, SQLite storage, and secrets management."""

from agent.infra.backends import (
    DuckieClient,
    FdeKnowledgeBase,
    NooglerProfileStore,
    get_duckie_client,
    get_knowledge_base,
    get_profile_store,
    set_duckie_client,
    set_knowledge_base,
    set_profile_store,
)
from agent.infra.db import (
    SqliteNooglerProfileStore,
    configure_storage,
    create_session_service,
    resolve_sqlite_path,
)
from agent.infra.secrets import (
    DEFAULT_RUNTIME_SECRET_NAMES,
    EnvSecretManager,
    GcpSecretManager,
    SecretManager,
    VaultSecretManager,
    create_secret_manager,
    hydrate_runtime_secrets,
)

__all__ = [
    "DEFAULT_RUNTIME_SECRET_NAMES",
    "DuckieClient",
    "EnvSecretManager",
    "FdeKnowledgeBase",
    "GcpSecretManager",
    "NooglerProfileStore",
    "SecretManager",
    "SqliteNooglerProfileStore",
    "VaultSecretManager",
    "configure_storage",
    "create_secret_manager",
    "create_session_service",
    "get_duckie_client",
    "get_knowledge_base",
    "get_profile_store",
    "hydrate_runtime_secrets",
    "resolve_sqlite_path",
    "set_duckie_client",
    "set_knowledge_base",
    "set_profile_store",
]
