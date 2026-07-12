# Terraform variables — configurable per environment.

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "azure-ai-rag"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "location" {
  description = "Azure region for all resources"
  type        = string
  default     = "uksouth"
}

variable "tags" {
  description = "Tags applied to all resources"
  type        = map(string)
  default = {
    project     = "azure-ai-rag-pipeline"
    managed_by  = "terraform"
    purpose     = "portfolio"
  }
}

# OpenAI
variable "openai_sku" {
  description = "Azure OpenAI SKU (S0 for pay-as-you-go)"
  type        = string
  default     = "S0"
}

variable "openai_chat_deployment" {
  description = "Name of the chat model deployment"
  type        = string
  default     = "gpt-4o"
}

variable "openai_chat_model" {
  description = "OpenAI chat model name"
  type        = string
  default     = "gpt-4o"
}

variable "openai_chat_model_version" {
  description = "OpenAI chat model version"
  type        = string
  default     = "2024-08-06"
}

variable "openai_embedding_deployment" {
  description = "Name of the embedding model deployment"
  type        = string
  default     = "text-embedding-3-large"
}

variable "openai_embedding_model" {
  description = "OpenAI embedding model name"
  type        = string
  default     = "text-embedding-3-large"
}

variable "openai_embedding_model_version" {
  description = "OpenAI embedding model version"
  type        = string
  default     = "1"
}

variable "openai_scale_type" {
  description = "OpenAI scaling type (Standard or Provisioned)"
  type        = string
  default     = "Standard"
}

variable "openai_capacity" {
  description = "OpenAI capacity (in thousands of TPM for Standard)"
  type        = number
  default     = 10
}

# AI Search
variable "search_sku" {
  description = "Azure AI Search SKU (free, basic, standard)"
  type        = string
  default     = "basic"
}

variable "semantic_search_sku" {
  description = "Semantic search SKU (free or standard)"
  type        = string
  default     = "free"
}

# Document Intelligence
variable "doc_intelligence_sku" {
  description = "Azure Document Intelligence SKU"
  type        = string
  default     = "S0"
}
