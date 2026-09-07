"""Centralised configuration.

All tunables are read once from environment variables (a local ``.env`` file is
loaded automatically). Import ``settings`` anywhere in the package.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of runtime configuration."""

    # --- credentials / endpoints ---
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL") or None
    cohere_api_key: str = os.getenv("COHERE_API_KEY", "")

    # --- models ---
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # --- paths ---
    data_dir: Path = Path(os.getenv("DATA_DIR", "data/documents"))
    storage_dir: Path = Path(os.getenv("STORAGE_DIR", "storage"))

    # --- vector store ---
    vector_backend: str = os.getenv("VECTOR_BACKEND", "faiss").strip().lower()

    # --- chunking ---
    chunk_size: int = _int("CHUNK_SIZE", 1000)
    chunk_overlap: int = _int("CHUNK_OVERLAP", 150)

    # --- retrieval ---
    top_k_vector: int = _int("TOP_K_VECTOR", 8)
    top_k_bm25: int = _int("TOP_K_BM25", 8)
    top_k_fused: int = _int("TOP_K_FUSED", 10)
    top_k_final: int = _int("TOP_K_FINAL", 4)
    rrf_k: int = _int("RRF_K", 60)

    # --- reranking ---
    reranker: str = os.getenv("RERANKER", "llm").strip().lower()

    # --- answering ---
    memory_window: int = _int("MEMORY_WINDOW", 6)
    score_threshold: float = _float("SCORE_THRESHOLD", 0.20)
    temperature: float = _float("TEMPERATURE", 0.0)
    max_retries: int = _int("OPENAI_MAX_RETRIES", 3)
    request_timeout: float = _float("REQUEST_TIMEOUT", 60.0)

    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key != "sk-your-key-here")


settings = Settings()
