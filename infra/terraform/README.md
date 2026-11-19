# FDE Duckie Local Terraform Config

Single-provider (`hashicorp/vault`) Terraform config that reads `config.json` directly, seeds runtime secrets into local HashiCorp Vault KV v2, and initializes/seeds the local SQLite database via Terraform's built-in `terraform_data`.

## Usage

1. Start local HashiCorp Vault in dev mode:
   ```bash
   vault server -dev -dev-root-token-id=dev-root-token
   ```
   *(Or via Docker: `docker run --rm -p 8200:8200 -e VAULT_DEV_ROOT_TOKEN_ID=dev-root-token hashicorp/vault:latest`)*

2. Apply Terraform to seed Vault secrets and initialize SQLite:
   ```bash
   cd infra/terraform
   export VAULT_TOKEN="dev-root-token"
   export TF_VAR_google_api_key="your-api-key"
   terraform init
   terraform apply
   ```
