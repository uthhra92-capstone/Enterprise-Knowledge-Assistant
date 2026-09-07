"""Embedding model factory (OpenAI)."""

from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from .config import settings


def get_embeddings() -> OpenAIEmbeddings:
    """Return a configured ``OpenAIEmbeddings`` client."""
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key or None,
        base_url=settings.openai_base_url,
        max_retries=settings.max_retries,
        timeout=settings.request_timeout,
    )
