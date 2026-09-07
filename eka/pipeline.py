"""End-to-end orchestration: ingest the corpus and answer questions.

    query
      -> condense with conversation memory
      -> hybrid retrieval (semantic + BM25)
      -> reciprocal-rank fusion
      -> reranking
      -> grounding check (semantic-similarity gate)
      -> LLM answer with numbered, cited context
      -> parse citations -> source list
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .chunking import split_documents
from .config import settings
from .embeddings import get_embeddings
from .loaders import load_documents
from .memory import ConversationMemory, condense_question
from .prompts import ANSWER_TEMPLATE, NOT_FOUND_MESSAGE, SYSTEM_PROMPT
from .reranker import get_reranker
from .retrievers import HybridRetriever, build_bm25
from .vector_store import EMB_FILE, resolve_vector_store

_CITATION_RE = re.compile(r"\[(\d{1,2})\]")
MANIFEST_FILE = "manifest.json"


def _looks_like_not_found(text: str) -> bool:
    normalised = re.sub(r"[^a-z ]", "", text.lower())
    return "could not find this information in the available documents" in normalised


@dataclass
class RetrievedChunk:
    source: str
    chunk_id: str
    snippet: str
    vector_score: float
    fused_score: float
    rerank_score: float | None
    vector_rank: int | None
    bm25_rank: int | None


@dataclass
class Answer:
    text: str
    cited_sources: list[str]
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    standalone_question: str = ""
    grounded: bool = True
    best_similarity: float = 0.0
    elapsed_s: float = 0.0


class KnowledgeAssistant:
    """Holds the loaded index and answers questions against it."""

    def __init__(self, vector_store, bm25, llm: ChatOpenAI | None = None) -> None:
        self.vector_store = vector_store
        self.retriever = HybridRetriever(vector_store, bm25)
        self.reranker = get_reranker()
        self.llm = llm or ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.temperature,
            api_key=settings.openai_api_key or None,
            base_url=settings.openai_base_url,
            max_retries=settings.max_retries,
            timeout=settings.request_timeout,
        )

    # --------------------------------------------------------------- lifecycle
    @classmethod
    def ingest(
        cls, data_dir: str | Path | None = None, storage_dir: str | Path | None = None
    ) -> "KnowledgeAssistant":
        """Build the index from documents on disk and persist it, then load it."""
        data_dir = Path(data_dir or settings.data_dir)
        storage_dir = Path(storage_dir or settings.storage_dir)

        print(f"[ingest] loading documents from {data_dir} ...")
        documents = load_documents(data_dir)
        if not documents:
            raise RuntimeError(
                f"No supported documents found in '{data_dir}'. "
                "Add .pdf / .docx / .txt / .md files and try again."
            )

        chunks = split_documents(documents)
        print(f"[ingest] {len(documents)} document sections -> {len(chunks)} chunks")

        embeddings = get_embeddings()
        store_cls = resolve_vector_store()
        store = store_cls.from_documents(chunks, embeddings)
        store.save(storage_dir)
        print(f"[ingest] vector backend: {store_cls.backend}")

        sources = sorted({doc.metadata.get("source", "unknown") for doc in documents})
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "embedding_model": settings.embedding_model,
            "llm_model": settings.llm_model,
            "vector_backend": store_cls.backend,
            "num_sources": len(sources),
            "num_chunks": len(chunks),
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "sources": sources,
        }
        (storage_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[ingest] index written to {storage_dir.resolve()}")

        return cls.load(storage_dir)

    @classmethod
    def load(cls, storage_dir: str | Path | None = None) -> "KnowledgeAssistant":
        storage_dir = Path(storage_dir or settings.storage_dir)
        if not cls.index_exists(storage_dir):
            raise FileNotFoundError(
                f"No index found in '{storage_dir}'. Run:  python ingest.py"
            )
        embeddings = get_embeddings()
        store = resolve_vector_store().load(storage_dir, embeddings)
        bm25 = build_bm25(store.documents, settings.top_k_bm25)
        return cls(store, bm25)

    @staticmethod
    def index_exists(storage_dir: str | Path | None = None) -> bool:
        storage_dir = Path(storage_dir or settings.storage_dir)
        return (storage_dir / EMB_FILE).exists() and (storage_dir / "chunks.jsonl").exists()

    @staticmethod
    def manifest(storage_dir: str | Path | None = None) -> dict:
        storage_dir = Path(storage_dir or settings.storage_dir)
        path = storage_dir / MANIFEST_FILE
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    # -------------------------------------------------------------------- query
    def answer(self, question: str, memory: ConversationMemory | None = None) -> Answer:
        started = perf_counter()
        memory = memory or ConversationMemory()

        standalone = condense_question(memory, question, self.llm)
        candidates = self.retriever.retrieve(standalone)
        reranked = self.reranker.rerank(standalone, candidates, settings.top_k_final)

        # --- grounding gate: is anything we retrieved actually about the query?
        query_vector = self.vector_store.embed_query(standalone)
        best_similarity = max(
            (self.vector_store.similarity(query_vector, c.chunk_id) for c in reranked),
            default=0.0,
        )

        retrieved = [
            RetrievedChunk(
                source=c.source,
                chunk_id=c.chunk_id,
                snippet=" ".join(c.document.page_content[:300].split()),
                vector_score=round(self.vector_store.similarity(query_vector, c.chunk_id), 4),
                fused_score=round(c.fused_score, 5),
                rerank_score=(round(c.rerank_score, 3) if c.rerank_score is not None else None),
                vector_rank=c.vector_rank,
                bm25_rank=c.bm25_rank,
            )
            for c in reranked
        ]

        if not reranked or best_similarity < settings.score_threshold:
            return Answer(
                text=NOT_FOUND_MESSAGE,
                cited_sources=[],
                retrieved=retrieved,
                standalone_question=standalone,
                grounded=False,
                best_similarity=best_similarity,
                elapsed_s=perf_counter() - started,
            )

        context = "\n\n".join(self._format_context_block(i, c) for i, c in enumerate(reranked, 1))
        messages = [
            SystemMessage(SYSTEM_PROMPT),
            *memory.recent_messages(),
            HumanMessage(ANSWER_TEMPLATE.format(context=context, question=question)),
        ]

        try:
            text = self.llm.invoke(messages).content.strip()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"The language model request failed: {exc}") from exc

        cited_sources = self._resolve_citations(text, reranked)
        return Answer(
            text=text,
            cited_sources=cited_sources,
            retrieved=retrieved,
            standalone_question=standalone,
            grounded=not _looks_like_not_found(text),
            best_similarity=best_similarity,
            elapsed_s=perf_counter() - started,
        )

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _format_context_block(index: int, candidate) -> str:
        meta = candidate.document.metadata
        header = f"[{index}] Source: {meta.get('source', 'unknown')}"
        if meta.get("page") is not None:
            header += f" (page {meta['page']})"
        return f"{header}\n{candidate.document.page_content.strip()}"

    @staticmethod
    def _resolve_citations(text: str, reranked) -> list[str]:
        if _looks_like_not_found(text):
            return []
        indices = sorted(
            {int(m) for m in _CITATION_RE.findall(text) if 1 <= int(m) <= len(reranked)}
        )
        sources: list[str] = []
        for i in indices:
            source = reranked[i - 1].source
            if source not in sources:
                sources.append(source)
        if sources:
            return sources
        # Model answered from context but did not emit [n] tags - fall back to the
        # sources of the chunks that were actually placed in the context.
        for candidate in reranked:
            if candidate.source not in sources:
                sources.append(candidate.source)
        return sources
