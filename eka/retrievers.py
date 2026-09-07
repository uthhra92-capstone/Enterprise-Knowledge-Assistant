"""Hybrid retrieval: semantic (vector) + keyword (BM25), fused with RRF."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from .config import settings
from .vector_store import LocalVectorStore


@dataclass
class Candidate:
    """A retrieved chunk plus the scores it accumulated along the pipeline."""

    document: Document
    vector_score: float = 0.0          # raw cosine similarity (0 if keyword-only)
    vector_rank: int | None = None     # rank in the semantic result list
    bm25_rank: int | None = None       # rank in the keyword result list
    fused_score: float = 0.0           # reciprocal-rank-fusion score
    rerank_score: float | None = None  # score assigned by the reranker

    @property
    def chunk_id(self) -> str:
        return self.document.metadata["chunk_id"]

    @property
    def source(self) -> str:
        return self.document.metadata.get("source", "unknown")


def build_bm25(documents: list[Document], k: int) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = k
    return retriever


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    k: int,
    weights: list[float] | None = None,
) -> list[tuple[Document, float]]:
    """Combine several ranked lists into one.

    score(d) = sum_i  weight_i * 1 / (k + rank_i(d))
    """
    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for ranked, weight in zip(ranked_lists, weights):
        for rank, doc in enumerate(ranked):
            cid = doc.metadata["chunk_id"]
            docs[cid] = doc
            scores[cid] = scores.get(cid, 0.0) + weight * (1.0 / (k + rank + 1))
    ordered = sorted(scores, key=scores.__getitem__, reverse=True)
    return [(docs[cid], scores[cid]) for cid in ordered]


class HybridRetriever:
    """Runs both retrievers and returns fused ``Candidate`` objects."""

    def __init__(self, vector_store: LocalVectorStore, bm25: BM25Retriever) -> None:
        self.vector_store = vector_store
        self.bm25 = bm25

    def retrieve(self, query: str) -> list[Candidate]:
        vector_hits = self.vector_store.search(query, settings.top_k_vector)

        try:
            keyword_hits = self.bm25.invoke(query)[: settings.top_k_bm25]
        except Exception as exc:  # noqa: BLE001
            print(f"[retrievers] BM25 search failed, continuing semantic-only: {exc}")
            keyword_hits = []

        fused = reciprocal_rank_fusion(
            [[doc for doc, _ in vector_hits], keyword_hits],
            k=settings.rrf_k,
            weights=[1.0, 1.0],
        )[: settings.top_k_fused]

        vector_rank = {doc.metadata["chunk_id"]: i for i, (doc, _) in enumerate(vector_hits)}
        vector_sim = {doc.metadata["chunk_id"]: s for doc, s in vector_hits}
        bm25_rank = {doc.metadata["chunk_id"]: i for i, doc in enumerate(keyword_hits)}

        candidates: list[Candidate] = []
        for doc, fused_score in fused:
            cid = doc.metadata["chunk_id"]
            candidates.append(
                Candidate(
                    document=doc,
                    vector_score=vector_sim.get(cid, 0.0),
                    vector_rank=vector_rank.get(cid),
                    bm25_rank=bm25_rank.get(cid),
                    fused_score=fused_score,
                )
            )
        return candidates
