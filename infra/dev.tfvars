# Terraform dev environment configuration.
# Copy to terraform.tfvars and customise.
# For production, create a prod.tfvars with higher SKUs and reserved capacity.

project_name = "azure-ai-rag"
environment  = "dev"
location     = "uksouth"

# Use free/basic tiers for development
search_sku           = "free"
semantic_search_sku  = "free"
doc_intelligence_sku = "S0"
openai_capacity      = 10

tags = {
  project    = "azure-ai-rag-pipeline"
  managed_by = "terraform"
  purpose    = "portfolio"
  cost_center = "personal-development"
}
