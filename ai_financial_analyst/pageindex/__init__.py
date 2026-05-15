"""PageIndex — page-level document indexing and retrieval.

Public API:
  index_document(file_bytes, filename, user_id, subllm, scope='user') → document_id
  search_documents(query, user_id, top_k=8) → list[PageResult]
  get_page(page_id, user_id) → PageResult | None
  get_document_page_by_number(document_id, page_number, user_id) → PageResult | None
  list_user_documents(user_id) → list[dict]
"""
from .pipeline import index_document
from .retriever import (
    search_documents,
    get_page,
    get_document_page_by_number,
    list_user_documents,
    PageResult,
)

__all__ = [
    "index_document",
    "search_documents",
    "get_page",
    "get_document_page_by_number",
    "list_user_documents",
    "PageResult",
]
