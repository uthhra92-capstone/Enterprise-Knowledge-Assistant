"""Split raw documents into retrieval-sized, overlapping chunks."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings


def split_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Return a list of chunk ``Document`` objects.

    Every chunk gets a stable ``chunk_id`` in its metadata (``<source>#<n>`` plus a
    page suffix for PDFs). The ``chunk_id`` is the join key used by hybrid fusion,
    reranking and the grounding check.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)

    per_source_counter: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        index = per_source_counter.get(source, 0)
        per_source_counter[source] = index + 1

        page = chunk.metadata.get("page")
        chunk_id = f"{source}#{index}"
        if page is not None:
            chunk_id += f"@p{page}"
        chunk.metadata["chunk_id"] = chunk_id

    return chunks
