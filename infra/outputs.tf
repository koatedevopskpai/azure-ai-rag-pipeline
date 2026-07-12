# Terraform outputs.
# Run `terraform output -json` to get connection details for .env file.

output "openai_endpoint" {
  description = "Azure OpenAI service endpoint"
  value       = azurerm_cognitive_account.openai.endpoint
}

output "openai_chat_deployment" {
  description = "Chat model deployment name"
  value       = var.openai_chat_deployment
}

output "openai_embedding_deployment" {
  description = "Embedding model deployment name"
  value       = var.openai_embedding_deployment
}

output "search_endpoint" {
  description = "Azure AI Search endpoint"
  value       = "https://${azurerm_search_service.search.name}.search.windows.net"
}

output "search_index_name" {
  description = "Default search index name"
  value       = "rag-documents"
}

output "doc_intelligence_endpoint" {
  description = "Azure Document Intelligence endpoint"
  value       = azurerm_cognitive_account.doc_intelligence.endpoint
}

output "resource_group_name" {
  description = "Azure resource group name"
  value       = azurerm_resource_group.main.name
}
