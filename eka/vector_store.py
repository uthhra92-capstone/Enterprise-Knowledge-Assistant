"""Local vector database with two interchangeable backends.

* ``FaissVectorStore``  - FAISS ``IndexFlatIP`` over L2-normalised vectors
                          (exact cosine search). Default.
* ``NumpyVectorStore``  - pure-NumPy cosine search, zero native deps. Automatic
                          fallback if FAISS is not installed; also selectable
                          with ``VECTOR_BACKEND=numpy``.

Both persist to the same ``storage/`` folder and expose the same surface
(``from_documents`` / ``save`` / ``load`` / ``search`` / ``similarity``), so the
rest of the pipeline does not care which one is active. ChromaDB could be dropped
in the same way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from langchain_core.documents import Document

from .config import settings

_EPS = 1e-8

EMB_FILE = "embeddings.npy"
CHUNKS_FILE = "chunks.jsonl"
FAISS_FILE = "index.faiss"


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / (norms + _EPS)).astype(np.float32)


def _embed_documents(embeddings, texts: list[str], batch_size: int = 128) -> np.ndarray:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        vectors.extend(embeddings.embed_documents(texts[start : start + batch_size]))
        print(f"[vector_store] embedded {min(start + batch_size, len(texts))}/{len(texts)} chunks")
    return _l2_normalize(np.asarray(vectors, dtype=np.float32))


def _write_chunks(directory: Path, documents: Sequence[Document]) -> None:
    with open(directory / CHUNKS_FILE, "w", encoding="utf-8") as handle:
        for doc in documents:
            handle.write(
                json.dumps(
                    {"page_content": doc.page_content, "metadata": doc.metadata},
                    ensure_ascii=False,
                )
                + "\n"
            )


def _read_chunks(directory: Path) -> list[Document]:
    documents: list[Document] = []
    with open(directory / CHUNKS_FILE, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            documents.append(
                Document(page_content=record["page_content"], metadata=record["metadata"])
            )
    return documents


class _BaseVectorStore:
    """Shared query helpers; subclasses implement the ANN index itself."""

    def __init__(self, embeddings, matrix: np.ndarray, documents: Sequence[Document]) -> None:
        self._embeddings = embeddings
        self._matrix = np.ascontiguousarray(matrix, dtype=np.float32)  # L2-normalised
        self._documents: list[Document] = list(documents)
        self._id_to_row = {
            doc.metadata["chunk_id"]: row for row, doc in enumerate(self._documents)
        }

    def embed_query(self, text: str) -> np.ndarray:
        vector = np.asarray(self._embeddings.embed_query(text), dtype=np.float32)
        return vector / (np.linalg.norm(vector) + _EPS)

    def similarity(self, query_vector: np.ndarray, chunk_id: str) -> float:
        """Cosine similarity between an embedded query and a stored chunk."""
        row = self._id_to_row.get(chunk_id)
        if row is None:
            return 0.0
        return float(self._matrix[row] @ query_vector)

    @property
    def documents(self) -> list[Document]:
        return list(self._documents)

    def __len__(self) -> int:
        return len(self._documents)

    # subclasses override
    def search(self, query: str, k: int) -> list[tuple[Document, float]]:  # pragma: no cover
        raise NotImplementedError


class NumpyVectorStore(_BaseVectorStore):
    backend = "numpy"

    @classmethod
    def from_documents(cls, documents, embeddings, batch_size: int = 128) -> "NumpyVectorStore":
        matrix = _embed_documents(embeddings, [d.page_content for d in documents], batch_size)
        return cls(embeddings, matrix, documents)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / EMB_FILE, self._matrix)
        _write_chunks(directory, self._documents)

    @classmethod
    def load(cls, directory: str | Path, embeddings) -> "NumpyVectorStore":
        directory = Path(directory)
        matrix = np.load(directory / EMB_FILE)
        return cls(embeddings, matrix, _read_chunks(directory))

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        if not self._documents:
            return []
        query_vector = self.embed_query(query)
        sims = self._matrix @ query_vector
        k = max(1, min(k, len(self._documents)))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(self._documents[i], float(sims[i])) for i in top]


class FaissVectorStore(_BaseVectorStore):
    backend = "faiss"

    def __init__(self, embeddings, matrix, documents, index=None) -> None:
        super().__init__(embeddings, matrix, documents)
        import faiss  # noqa: PLC0415

        if index is None:
            index = faiss.IndexFlatIP(self._matrix.shape[1])
            if len(self._matrix):
                index.add(self._matrix)
        self._index = index

    @classmethod
    def from_documents(cls, documents, embeddings, batch_size: int = 128) -> "FaissVectorStore":
        matrix = _embed_documents(embeddings, [d.page_content for d in documents], batch_size)
        return cls(embeddings, matrix, documents)

    def save(self, directory: str | Path) -> None:
        import faiss  # noqa: PLC0415

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / FAISS_FILE))
        np.save(directory / EMB_FILE, self._matrix)  # kept for the grounding gate / fallback
        _write_chunks(directory, self._documents)

    @classmethod
    def load(cls, directory: str | Path, embeddings) -> "FaissVectorStore":
        import faiss  # noqa: PLC0415

        directory = Path(directory)
        documents = _read_chunks(directory)
        matrix = np.load(directory / EMB_FILE)
        faiss_path = directory / FAISS_FILE
        index = faiss.read_index(str(faiss_path)) if faiss_path.exists() else None
        return cls(embeddings, matrix, documents, index=index)

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        if not self._documents:
            return []
        query_vector = np.ascontiguousarray(self.embed_query(query)[None, :], dtype=np.float32)
        k = max(1, min(k, len(self._documents)))
        scores, indices = self._index.search(query_vector, k)
        return [
            (self._documents[i], float(s))
            for i, s in zip(indices[0], scores[0])
            if i != -1
        ]


def resolve_vector_store(backend: str | None = None) -> type[_BaseVectorStore]:
    """Pick the vector-store class from config, degrading gracefully."""
    backend = (backend or settings.vector_backend).strip().lower()
    if backend == "numpy":
        return NumpyVectorStore
    try:
        import faiss  # noqa: F401, PLC0415

        return FaissVectorStore
    except ImportError:
        print("[vector_store] faiss-cpu not installed - using the NumPy store instead")
        return NumpyVectorStore


# Backwards-compatible alias
LocalVectorStore = NumpyVectorStore
