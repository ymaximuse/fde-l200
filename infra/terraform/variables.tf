variable "python_bin" {
  type        = string
  description = "Python executable used to run scripts/setup_db.py."
  default     = ".venv/bin/python"
}

variable "google_api_key" {
  type        = string
  description = "Google AI Studio API key to store in HashiCorp Vault KV v2."
  default     = ""
  sensitive   = true
}

variable "otel_exporter_otlp_headers" {
  type        = string
  description = "Optional OTLP exporter auth headers to store in HashiCorp Vault KV v2."
  default     = ""
  sensitive   = true
}

variable "reset_db" {
  type        = bool
  description = "When true, recreates the local SQLite database from scratch on apply."
  default     = false
}
