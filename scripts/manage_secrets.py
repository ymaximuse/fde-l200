"""CLI helper for reading/writing FDE Duckie secrets in HashiCorp Vault (KV v2).

Usage:
    # Start local dev Vault first:
    #   vault server -dev -dev-root-token-id=dev-root-token
    #   export VAULT_ADDR=http://127.0.0.1:8200 VAULT_TOKEN=dev-root-token

    uv run python scripts/manage_secrets.py set GOOGLE_API_KEY
    uv run python scripts/manage_secrets.py import-env
    uv run python scripts/manage_secrets.py list
    uv run python scripts/manage_secrets.py get GOOGLE_API_KEY
    uv run python scripts/manage_secrets.py delete GOOGLE_API_KEY
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.config import load_config  # noqa: E402
from agent.infra.secrets import DEFAULT_RUNTIME_SECRET_NAMES, create_secret_manager  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage FDE Duckie secrets in HashiCorp Vault (or configured backend)."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional JSON config file path or inline JSON string.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_set = sub.add_parser("set", help="Store a new KV v2 version of a secret.")
    p_set.add_argument("name", help="Secret key name (e.g. GOOGLE_API_KEY).")
    p_set.add_argument(
        "--value",
        default=None,
        help="Secret value (omit to prompt interactively without echoing).",
    )
    p_set.add_argument(
        "--from-env",
        action="store_true",
        help="Read the secret value from the matching environment variable.",
    )

    p_get = sub.add_parser("get", help="Fetch and print a secret value.")
    p_get.add_argument("name", help="Secret key name.")
    p_get.add_argument("--version", default="latest", help="KV v2 version number or 'latest'.")

    sub.add_parser("list", help="List secret key names stored under the configured path.")

    p_del = sub.add_parser("delete", help="Remove a secret key from the configured path.")
    p_del.add_argument("name", help="Secret key name.")

    p_imp = sub.add_parser(
        "import-env",
        help="Import standard runtime secrets from .env into HashiCorp Vault.",
    )
    p_imp.add_argument(
        "--env-file",
        default=str(PROJECT_ROOT / ".env"),
        help="Path to .env file to import from.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    manager = create_secret_manager(cfg)

    if args.command == "set":
        if args.from_env:
            value = os.getenv(args.name)
            if not value:
                raise SystemExit(f"Environment variable '{args.name}' is not set.")
        elif args.value is not None:
            value = args.value
        else:
            value = getpass.getpass(f"Enter value for {args.name}: ")
        version = manager.set_secret(args.name, value)
        print(f"Stored secret '{args.name}' in {cfg.secrets.backend} (version={version}).")
        return

    if args.command == "get":
        value = manager.get_secret(args.name, version=args.version)
        if value is None:
            raise SystemExit(f"Secret '{args.name}' (version={args.version}) not found.")
        print(value)
        return

    if args.command == "list":
        names = manager.list_secrets()
        if not names:
            print("(no secrets stored)")
            return
        for name in names:
            print(name)
        return

    if args.command == "delete":
        removed = manager.delete_secret(args.name)
        print(
            f"Deleted secret '{args.name}'."
            if removed
            else f"Secret '{args.name}' not found."
        )
        return

    if args.command == "import-env":
        env_path = Path(args.env_file)
        if not env_path.is_file():
            raise SystemExit(f"Env file not found: {env_path}")
        values = dotenv_values(env_path)
        imported = []
        for name in DEFAULT_RUNTIME_SECRET_NAMES:
            val = values.get(name)
            if val and not val.startswith("your-"):
                ver = manager.set_secret(name, val)
                imported.append(f"{name} (v{ver})")
        print(
            f"Imported into {cfg.secrets.backend}: {', '.join(imported)}"
            if imported
            else "No non-placeholder runtime secrets found to import."
        )


if __name__ == "__main__":
    main()
