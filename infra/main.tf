# Azure AI RAG Pipeline — Terraform infrastructure
#
# Architecture decision: Terraform over Bicep or ARM templates.
#
# Why: Terraform is cloud-agnostic (works with AWS, GCP, Azure), has a
# mature state management system (remote backends with locking), and is
# the most widely adopted IaC tool across the industry. Bicep is Azure-only.
# ARM templates are JSON-based and hard to maintain at scale.
#
# Architecture decision: Separate tfvars files per environment over
# inline variable defaults.
#
# Why: The .tfvars file for development uses free-tier Azure resources.
# The production .tfvars file uses reserved capacity and geo-redundant
# storage. Inline defaults would require conditionals that make the
# configuration harder to read.

terraform {
  required_version = ">= 1.6, < 2.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.100"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  # Architecture decision: Local backend for portfolio project.
  # Why: Remote backends (Azure Storage, Terraform Cloud) are production
  # requirements for team collaboration and state locking. For a portfolio
  # project that demonstrates the Terraform configuration, a local backend
  # is sufficient. The backend block can be updated to use Azure Storage
  # without changing any resource definitions.
}

provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy = false
    }
  }
}

# ------------------------------------------------------------------
# Resource group — all Azure resources in a single group for easy
# lifecycle management and cost tracking.
# ------------------------------------------------------------------

resource "azurerm_resource_group" "main" {
  name     = "rg-${var.project_name}-${var.environment}"
  location = var.location
  tags     = var.tags
}

# ------------------------------------------------------------------
# Azure OpenAI Service — chat and embedding models
# ------------------------------------------------------------------
# Architecture decision: Separate OpenAI account per environment.
# Why: Development can use a pay-as-you-go account without quota limits.
# Production uses a committed-throughput account for predictable cost and
# performance. Separating environments prevents accidental rate limiting
# from development load affecting production users.

resource "azurerm_cognitive_account" "openai" {
  name                = "cog-${var.project_name}-openai-${var.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  kind                = "OpenAI"
  sku_name            = var.openai_sku

  custom_subdomain_name = "${var.project_name}-openai-${var.environment}"

  tags = var.tags
}

resource "azurerm_cognitive_deployment" "chat" {
  name                 = var.openai_chat_deployment
  cognitive_account_id = azurerm_cognitive_account.openai.id

  model {
    format  = "OpenAI"
    name    = var.openai_chat_model
    version = var.openai_chat_model_version
  }

  scale {
    type     = var.openai_scale_type
    capacity = var.openai_capacity
  }
}

resource "azurerm_cognitive_deployment" "embedding" {
  name                 = var.openai_embedding_deployment
  cognitive_account_id = azurerm_cognitive_account.openai.id

  model {
    format  = "OpenAI"
    name    = var.openai_embedding_model
    version = var.openai_embedding_model_version
  }

  scale {
    type     = var.openai_scale_type
    capacity = var.openai_capacity
  }
}

# ------------------------------------------------------------------
# Azure AI Search — vector + keyword hybrid search
# ------------------------------------------------------------------

resource "azurerm_search_service" "search" {
  name                = "srch-${var.project_name}-${var.environment}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = var.search_sku

  # Architecture decision: Semantic search enabled.
  # Why: Semantic search reranks results using Microsoft's cross-encoder
  # model, which significantly improves RAG accuracy for financial queries.
  semantic_search_sku = var.semantic_search_sku

  tags = var.tags
}

# ------------------------------------------------------------------
# Azure Document Intelligence — document extraction
# ------------------------------------------------------------------

resource "azurerm_cognitive_account" "doc_intelligence" {
  name                = "cog-${var.project_name}-docintel-${var.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  kind                = "FormRecognizer"
  sku_name            = var.doc_intelligence_sku

  custom_subdomain_name = "${var.project_name}-docintel-${var.environment}"

  tags = var.tags
}

# ------------------------------------------------------------------
# Key Vault — stores API keys for application consumption
# ------------------------------------------------------------------
# Architecture decision: Key Vault over .env files in production.
# Why: .env files on a server can be read by anyone with SSH access.
# Key Vault provides access logging, rotation policies, and RBAC.
# The application reads secrets via managed identity on AKS.

resource "azurerm_key_vault" "main" {
  name                = "kv-${var.project_name}-${var.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"

  tags = var.tags
}

resource "azurerm_key_vault_secret" "openai_key" {
  name         = "azure-openai-api-key"
  value        = azurerm_cognitive_account.openai.primary_access_key
  key_vault_id = azurerm_key_vault.main.id
}

resource "azurerm_key_vault_secret" "search_key" {
  name         = "azure-search-api-key"
  value        = azurerm_search_service.search.primary_key
  key_vault_id = azurerm_key_vault.main.id
}

resource "azurerm_key_vault_secret" "doc_intelligence_key" {
  name         = "azure-doc-intelligence-api-key"
  value        = azurerm_cognitive_account.doc_intelligence.primary_access_key
  key_vault_id = azurerm_key_vault.main.id
}

data "azurerm_client_config" "current" {}

# ------------------------------------------------------------------
# Outputs — connection details for the application .env file
# ------------------------------------------------------------------

output "openai_endpoint" {
  value     = azurerm_cognitive_account.openai.endpoint
  sensitive = false
}

output "search_endpoint" {
  value     = "https://${azurerm_search_service.search.name}.search.windows.net"
  sensitive = false
}

output "doc_intelligence_endpoint" {
  value     = azurerm_cognitive_account.doc_intelligence.endpoint
  sensitive = false
}

output "key_vault_uri" {
  value     = azurerm_key_vault.main.vault_uri
  sensitive = false
}

output "resource_group" {
  value     = azurerm_resource_group.main.name
  sensitive = false
}
