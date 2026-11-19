output "vault_secret_path" {
  description = "HashiCorp Vault KV v2 path holding FDE Duckie secrets."
  value       = "${local.vault_mount}/data/${local.vault_path}"
}

output "database_url" {
  description = "Configured local SQLite database URI from config.json."
  value       = local.config.database.url
}

output "run_cli_command" {
  description = "Command to run FDE Duckie interactively."
  value       = "uv run python main.py --config config.json"
}

output "run_adk_web_command" {
  description = "Command to run ADK Web UI backed by the local SQLite database."
  value       = "uv run adk web --session_service_uri ${local.config.database.url}"
}
