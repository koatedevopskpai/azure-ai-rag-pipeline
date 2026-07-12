"""
Azure AI RAG Pipeline — production-grade financial research assistant.

Architecture decision: Package-level imports over relative imports.

Why: PEP 8 recommends absolute imports for clarity. In a project of
this size, `from app.config import settings` is unambiguous regardless
of where the caller sits in the package tree. Relative imports like
`from ..config import settings` break when the package structure changes.
"""
