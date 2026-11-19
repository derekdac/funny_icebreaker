terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.30"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

locals {
  required_apis = toset([
    "aiplatform.googleapis.com",
    "secretmanager.googleapis.com",
    "cloudtrace.googleapis.com",
    "logging.googleapis.com",
    "dlp.googleapis.com",
  ])
}

resource "google_project_service" "coldopen_apis" {
  for_each           = local.required_apis
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

resource "google_secret_manager_secret" "google_api_key" {
  secret_id = "coldopen-google-api-key"
  replication {
    auto {}
  }
  depends_on = [google_project_service.coldopen_apis]
}

resource "google_service_account" "coldopen_runner" {
  account_id   = "coldopen-agent-runner"
  display_name = "ColdOpen ADK Agent Runner"
}

resource "google_secret_manager_secret_iam_member" "runner_secret_accessor" {
  secret_id = google_secret_manager_secret.google_api_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.coldopen_runner.email}"
}
