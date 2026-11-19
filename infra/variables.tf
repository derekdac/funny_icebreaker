variable "project_id" {
  type        = string
  description = "GCP project ID for ColdOpen Vertex AI, Secret Manager, and observability resources."
}

variable "region" {
  type        = string
  default     = "us-central1"
  description = "Default GCP region for Vertex AI and supporting services."
}
