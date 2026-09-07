"""Reranking - reorder the fused candidates so the best context reaches the LLM.

Three backends, selected by the ``RERANKER`` env var:

* ``llm``    - an LLM scores every passage 0-10 (default, no extra service/key)
* ``cohere`` - Cohere Rerank API (needs ``cohere`` + ``COHERE_API_KEY``)
* ``none``   - keep the fusion order (useful for comparison / offline demos)
"""

from __future__ import annotations

from typing import Protocol, Sequence

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .config import settings
from .prompts import RERANK_SYSTEM
from .retrievers import Candidate


class Reranker(Protocol):
    def rerank(self, query: str, candidates: Sequence[Candidate], top_n: int) -> list[Candidate]: ...


# --------------------------------------------------------------------------- LLM
class PassageScore(BaseModel):
    index: int = Field(description="1-based index of the passage being scored")
    relevance: float = Field(description="relevance to the question, 0 (none) to 10 (perfect)")


class RerankResult(BaseModel):
    scores: list[PassageScore]


class LLMReranker:
    def __init__(self, llm: ChatOpenAI | None = None) -> None:
        self._llm = llm or ChatOpenAI(
            model=settings.llm_model,
            temperature=0.0,
            api_key=settings.openai_api_key or None,
            base_url=settings.openai_base_url,
            max_retries=settings.max_retries,
            timeout=settings.request_timeout,
        )
        self._structured = self._llm.with_structured_output(RerankResult)

    def rerank(self, query: str, candidates: Sequence[Candidate], top_n: int) -> list[Candidate]:
        candidates = list(candidates)
        if not candidates:
            return []

        passages = "\n\n".join(
            f"[{i}] {c.document.page_content[:800].strip()}" for i, c in enumerate(candidates, 1)
        )
        user_msg = (
            f"Question: {query}\n\nPassages:\n{passages}\n\n"
            "Return a relevance score for every passage index above."
        )
        try:
            result = self._structured.invoke(
                [("system", RERANK_SYSTEM), ("human", user_msg)]
            )
            by_index = {s.index: float(s.relevance) for s in result.scores}
        except Exception as exc:  # noqa: BLE001 - degrade to fusion order
            print(f"[reranker] LLM rerank failed, keeping fusion order: {exc}")
            by_index = {}

        for i, candidate in enumerate(candidates, 1):
            candidate.rerank_score = by_index.get(i)

        candidates.sort(
            key=lambda c: c.rerank_score if c.rerank_score is not None else c.fused_score,
            reverse=True,
        )
        return candidates[:top_n]


# ------------------------------------------------------------------------ Cohere
class CohereReranker:
    def __init__(self) -> None:
        import cohere  # noqa: PLC0415

        if not settings.cohere_api_key:
            raise RuntimeError("RERANKER=cohere but COHERE_API_KEY is not set.")
        self._client = cohere.Client(settings.cohere_api_key)
        self._model = "rerank-english-v3.0"

    def rerank(self, query: str, candidates: Sequence[Candidate], top_n: int) -> list[Candidate]:
        candidates = list(candidates)
        if not candidates:
            return []
        response = self._client.rerank(
            model=self._model,
            query=query,
            documents=[c.document.page_content for c in candidates],
            top_n=min(top_n, len(candidates)),
        )
        ordered: list[Candidate] = []
        for item in response.results:
            candidate = candidates[item.index]
            candidate.rerank_score = float(item.relevance_score)
            ordered.append(candidate)
        return ordered


# -------------------------------------------------------------------------- Noop
class NoopReranker:
    def rerank(self, query: str, candidates: Sequence[Candidate], top_n: int) -> list[Candidate]:
        return list(candidates)[:top_n]


def get_reranker() -> Reranker:
    choice = settings.reranker
    if choice == "none":
        return NoopReranker()
    if choice == "cohere":
        return CohereReranker()
    return LLMReranker()
