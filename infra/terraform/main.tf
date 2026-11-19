terraform {
  required_version = ">= 1.6.0"

  required_providers {
    vault = {
      source  = "hashicorp/vault"
      version = "~> 4.5"
    }
  }
}

locals {
  project_root = abspath("${path.module}/../..")
  config       = jsondecode(file("${local.project_root}/config.json"))
  vault_mount  = local.config.secrets.vault_mount
  vault_path   = local.config.secrets.vault_path

  vault_secret_data = merge(
    var.google_api_key != "" ? { GOOGLE_API_KEY = var.google_api_key } : {},
    var.otel_exporter_otlp_headers != "" ? { OTEL_EXPORTER_OTLP_HEADERS = var.otel_exporter_otlp_headers } : {},
  )
}

# Authenticates via VAULT_TOKEN (or ~/.vault-token); reads address/namespace from config.json.
provider "vault" {
  address   = local.config.secrets.vault_addr
  namespace = local.config.secrets.vault_namespace
}

# 1. Seed runtime secrets into HashiCorp Vault KV v2.
resource "vault_kv_secret_v2" "fde_duckie" {
  count               = length(local.vault_secret_data) > 0 ? 1 : 0
  mount               = local.vault_mount
  name                = local.vault_path
  delete_all_versions = true
  data_json           = jsonencode(local.vault_secret_data)
}

# 2. Initialize ADK SQLite tables and seed FDE Noogler profiles & starter sessions.
resource "terraform_data" "sqlite_seed" {
  triggers_replace = {
    db_url       = local.config.database.url
    reset_db     = tostring(var.reset_db)
    profiles_sha = filesha256("${local.project_root}/agent/mocks/data/noogler_profiles.json")
  }

  provisioner "local-exec" {
    working_dir = local.project_root
    command     = "${var.python_bin} scripts/setup_db.py --config config.json${var.reset_db ? " --reset" : ""}"
  }
}
