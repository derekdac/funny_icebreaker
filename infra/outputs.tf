output "secret_name" {
  value       = google_secret_manager_secret.google_api_key.secret_id
  description = "Secret Manager ID used by ColdOpen at runtime."
}

output "runner_service_account" {
  value       = google_service_account.coldopen_runner.email
  description = "Dedicated least-privilege service account for ColdOpen."
}
