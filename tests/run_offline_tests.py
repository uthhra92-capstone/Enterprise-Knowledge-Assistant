"""Offline test suite - exercises every module with NO network / API calls.

    python tests/run_offline_tests.py

Uses a deterministic bag-of-words fake embedder and a fake LLM, so the full
pipeline (loaders -> chunking -> vector store -> hybrid retrieval -> RRF ->
rerank -> grounding gate -> answer -> citations) runs for free. Live LLM
behaviour (real grounded answers, memory rewriting) is covered separately by
`python eval.py` and by the manual cases in TESTING.md.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# --- make the project importable and force an API-free configuration ----------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy")  # constructors only, never called
os.environ["RERANKER"] = "none"                           # pipeline uses NoopReranker

from eka.chunking import split_documents                                  # noqa: E402
from eka.corpus import delete_source_file, list_source_files, save_upload  # noqa: E402
from eka.history import ConversationStore                                  # noqa: E402
from eka.loaders import detect_kind, load_documents                       # noqa: E402
from eka.memory import ConversationMemory, condense_question             # noqa: E402
from eka.pipeline import KnowledgeAssistant                               # noqa: E402
from eka.reranker import LLMReranker, NoopReranker, RerankResult          # noqa: E402
from eka.retrievers import (                                              # noqa: E402
    Candidate,
    build_bm25,
    reciprocal_rank_fusion,
    HybridRetriever,
)
from eka.vector_store import (                                            # noqa: E402
    FaissVectorStore,
    NumpyVectorStore,
    resolve_vector_store,
)
from langchain_core.documents import Document                             # noqa: E402

DATA_DIR = ROOT / "data" / "documents"


# --------------------------------------------------------------------- fakes ---
class FakeEmbeddings:
    """Hashing bag-of-words vectors: similar text -> similar vector, deterministic."""

    dim = 512

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[hash(tok) % self.dim] += 1.0
        norm = float(np.linalg.norm(v))
        return (v / norm if norm else v).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeStructured:
    def __init__(self, model_cls):
        self.model_cls = model_cls

    def invoke(self, messages):
        text = messages[-1][1] if isinstance(messages[-1], tuple) else messages[-1].content
        n = len(re.findall(r"\[(\d+)\]", text)) or 1
        from eka.reranker import PassageScore

        return self.model_cls(
            scores=[PassageScore(index=i, relevance=float(n - i + 1)) for i in range(1, n + 1)]
        )


class FakeLLM:
    def __init__(self, answer: str = "Employees receive 20 days of annual leave per year [1]."):
        self._answer = answer

    def invoke(self, prompt):
        if isinstance(prompt, str):  # condense_question passes a plain string
            tail = prompt.split("Follow-up question:")[-1].strip().splitlines()[0]
            return SimpleNamespace(content=f"About the leave policy: {tail}")
        return SimpleNamespace(content=self._answer)  # answer() passes a message list

    def with_structured_output(self, model_cls):
        return FakeStructured(model_cls)


def _docs(*pairs) -> list[Document]:
    out = []
    for src, body in pairs:
        d = Document(page_content=body, metadata={"source": src})
        out.append(d)
    return split_documents(out, chunk_size=400, chunk_overlap=40)


# --------------------------------------------------------------------- tests ---
def test_loaders_multiformat():
    docs = load_documents(DATA_DIR)
    sources = {d.metadata["source"] for d in docs}
    assert len(docs) > 0, "no documents loaded"
    exts = {Path(s).suffix.lower() for s in sources}
    assert {".md", ".txt"} <= exts, f"expected .md and .txt, got {exts}"
    assert all("source" in d.metadata for d in docs)
    assert any("Leave_Policy" in s for s in sources)


def test_detect_kind():
    assert detect_kind(Path("x.pdf")) == "pdf"
    assert detect_kind(Path("x.docx")) == "docx"
    assert detect_kind(Path("x.md")) == "text"
    assert detect_kind(Path("x.zip")) is None


def test_chunking_ids_unique():
    docs = load_documents(DATA_DIR)
    chunks = split_documents(docs)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(chunks) > len(docs), "expected splitting to increase count"
    assert len(ids) == len(set(ids)), "chunk_id values must be unique"
    assert all(len(c.page_content) <= 2000 for c in chunks)


def test_vector_store_roundtrip_both_backends():
    chunks = _docs(
        ("A.md", "annual leave carry forward five days by 31 March"),
        ("B.md", "travel per diem international sixty dollars domestic"),
        ("C.md", "password fourteen characters multi factor authentication"),
    )
    emb = FakeEmbeddings()
    for cls in (NumpyVectorStore, FaissVectorStore):
        with tempfile.TemporaryDirectory() as tmp:
            store = cls.from_documents(chunks, emb)
            store.save(tmp)
            reloaded = cls.load(tmp, emb)
            assert len(reloaded) == len(chunks)
            hits = reloaded.search("carry forward annual leave", k=3)
            assert hits and hits[0][0].metadata["source"] == "A.md", f"{cls.__name__}: wrong top hit"
            assert hits[0][1] >= hits[-1][1], "results must be sorted by score desc"
            qv = reloaded.embed_query("carry forward annual leave")
            assert reloaded.similarity(qv, hits[0][0].metadata["chunk_id"]) > 0.1


def test_faiss_is_default_backend():
    assert resolve_vector_store().backend == "faiss"


def test_rrf_fusion_merges_and_orders():
    a = _docs(("d.md", "one two three four five six seven eight nine ten"))
    d1, d2 = a[0], a[-1]
    # d1 is rank-0 in list X and rank-1 in list Y -> should win
    fused = reciprocal_rank_fusion([[d1, d2], [d2, d1]], k=60)
    ids = [doc.metadata["chunk_id"] for doc, _ in fused]
    assert set(ids) == {d1.metadata["chunk_id"], d2.metadata["chunk_id"]}, "must dedupe by chunk_id"
    assert fused[0][0].metadata["chunk_id"] == d1.metadata["chunk_id"]


def test_hybrid_retriever_combines_vector_and_bm25():
    chunks = _docs(
        ("Leave.md", "annual leave entitlement is twenty days carry forward five days"),
        ("Travel.md", "per diem international travel is one hundred dollars per day"),
        ("IT.md", "passwords must be at least fourteen characters with mfa enabled"),
    )
    emb = FakeEmbeddings()
    store = NumpyVectorStore.from_documents(chunks, emb)
    hybrid = HybridRetriever(store, build_bm25(chunks, k=5))
    cands = hybrid.retrieve("how many annual leave days and carry forward")
    assert cands, "hybrid retriever returned nothing"
    assert isinstance(cands[0], Candidate)
    assert cands[0].source == "Leave.md"
    assert any(c.vector_rank is not None for c in cands)
    assert any(c.bm25_rank is not None for c in cands)


def test_noop_reranker_truncates():
    chunks = _docs(("d.md", " ".join(f"w{i}" for i in range(300))))
    cands = [Candidate(document=c, fused_score=1.0 / (i + 1)) for i, c in enumerate(chunks)]
    out = NoopReranker().rerank("q", cands, top_n=2)
    assert len(out) == 2


def test_llm_reranker_reorders_by_score():
    chunks = _docs(("d.md", " ".join(f"w{i}" for i in range(400))))
    cands = [Candidate(document=c, fused_score=0.01) for c in chunks[:4]]
    out = LLMReranker(llm=FakeLLM()).rerank("q", cands, top_n=3)
    assert len(out) == 3
    assert out[0].rerank_score is not None
    assert out[0].rerank_score >= out[-1].rerank_score, "must be sorted by rerank score"


def test_memory_windowing_and_condense():
    mem = ConversationMemory(window=2)
    for i in range(4):
        mem.add_user(f"q{i}")
        mem.add_assistant(f"a{i}")
    assert len(mem.recent_messages()) == 2, "window must cap recent messages"
    assert "q3" in mem.as_text() and "q0" not in mem.as_text()

    # no history -> question returned unchanged (no LLM call)
    assert condense_question(ConversationMemory(), "What is the leave policy?") == "What is the leave policy?"
    # with history -> rewritten via (fake) LLM
    mem2 = ConversationMemory()
    mem2.add_user("What is the leave policy?")
    mem2.add_assistant("Employees receive 20 days.")
    rewritten = condense_question(mem2, "what about carry-forward?", llm=FakeLLM())
    assert rewritten != "what about carry-forward?" and len(rewritten) > 0


def test_sqlite_history_crud_and_owner_isolation():
    with tempfile.TemporaryDirectory() as tmp:
        store = ConversationStore(Path(tmp) / "c.db")
        store.append_message("c1", "user", "What is the leave policy?", owner="alice")
        store.append_message("c1", "assistant", "20 days [1].", sources=["Leave_Policy.md"], owner="alice")
        store.append_message("c2", "user", "per diem?", owner="bob")

        assert [c.id for c in store.list_conversations("alice")] == ["c1"]
        assert [c.id for c in store.list_conversations("bob")] == ["c2"]
        assert len(store.list_conversations()) == 2  # unscoped sees all

        msgs = store.load_messages("c1")
        assert msgs[0].content == "What is the leave policy?"
        assert msgs[1].sources == ["Leave_Policy.md"]
        assert store.list_conversations("alice")[0].title != "New chat"  # auto-titled

        store.delete_conversation("c1")
        assert store.list_conversations("alice") == []


def test_sqlite_history_migrates_old_schema():
    import sqlite3

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "old.db"
        con = sqlite3.connect(path)
        con.executescript(
            "CREATE TABLE conversations (id TEXT PRIMARY KEY, title TEXT NOT NULL DEFAULT 'New chat',"
            " created_at TEXT NOT NULL, updated_at TEXT NOT NULL);"
            "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL,"
            " role TEXT, content TEXT, standalone_question TEXT, sources TEXT, created_at TEXT NOT NULL);"
        )
        con.execute("INSERT INTO conversations VALUES ('old','Legacy','2026-01-01T00:00:00+00:00','2026-01-01T00:00:00+00:00')")
        con.commit()
        con.close()

        store = ConversationStore(path)  # should ALTER TABLE ADD COLUMN owner
        check = sqlite3.connect(path)
        try:
            cols = {r[1] for r in check.execute("PRAGMA table_info(conversations)")}
        finally:
            check.close()
        assert "owner" in cols
        store.append_message("old", "user", "hello")  # still usable


def test_corpus_add_list_delete():
    before = {f.name for f in list_source_files()}
    save_upload("__offline_test__.txt", b"temporary file for the offline test suite")
    names = {f.name for f in list_source_files()}
    assert "__offline_test__.txt" in names
    kinds = {f.name: f.kind for f in list_source_files()}
    assert kinds["__offline_test__.txt"] == "text"
    assert delete_source_file("__offline_test__.txt") is True
    assert {f.name for f in list_source_files()} == before


def test_corpus_rejects_unsafe_names():
    # traversal is reduced to a bare filename, never escapes the folder
    p = save_upload("../../escape.txt", b"x")
    assert p.resolve().parent == (ROOT / "data" / "documents").resolve()
    assert p.name == "escape.txt"
    delete_source_file("escape.txt")


def _build_assistant(answer: str) -> KnowledgeAssistant:
    docs = load_documents(DATA_DIR)
    chunks = split_documents(docs)
    emb = FakeEmbeddings()
    store = resolve_vector_store().from_documents(chunks, emb)
    bm25 = build_bm25(store.documents, 8)
    return KnowledgeAssistant(store, bm25, llm=FakeLLM(answer=answer))


def test_pipeline_grounded_answer_with_citation():
    ka = _build_assistant("Employees receive 20 days of annual leave; up to 5 days carry forward [1].")
    ans = ka.answer("How many annual leave days do employees get and what is the carry-forward rule?")
    assert ans.grounded is True
    assert ans.text.strip() != ""
    assert ans.retrieved, "retrieved chunks should be reported"
    assert ans.cited_sources, "a grounded answer must list sources"
    assert any("Leave_Policy" in s for s in ans.cited_sources), ans.cited_sources
    assert ans.best_similarity >= 0.2


def test_pipeline_not_found_for_unrelated_question():
    ka = _build_assistant("this should never be shown")
    # tokens with essentially no overlap with any policy document -> below the gate
    ans = ka.answer("Blorptastic quuxflibbet zorbnak wibblesnort thrummy?")
    assert ans.grounded is False, f"expected not-found, got: {ans.text!r} (sim={ans.best_similarity})"
    assert ans.cited_sources == []
    assert "could not find" in ans.text.lower()


def test_pipeline_memory_rewrites_followup():
    ka = _build_assistant("Up to 5 days may be carried forward, used by 31 March [1].")
    mem = ConversationMemory()
    mem.add_user("What is the leave policy?")
    mem.add_assistant("Employees receive 20 days of annual leave.")
    ans = ka.answer("what about carry-forward?", mem)
    assert ans.standalone_question != "what about carry-forward?", "follow-up should be condensed"


def test_ingest_writes_index_files(monkeypatch=None):
    import eka.pipeline as pipeline_mod

    orig = pipeline_mod.get_embeddings
    pipeline_mod.get_embeddings = lambda: FakeEmbeddings()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            pipeline_mod.KnowledgeAssistant.ingest(data_dir=DATA_DIR, storage_dir=tmp)
            files = {p.name for p in Path(tmp).iterdir()}
            assert {"index.faiss", "embeddings.npy", "chunks.jsonl", "manifest.json"} <= files, files
            import json

            manifest = json.loads((Path(tmp) / "manifest.json").read_text())
            assert manifest["vector_backend"] == "faiss"
            assert manifest["num_chunks"] > 0
            assert manifest["num_sources"] >= 8
    finally:
        pipeline_mod.get_embeddings = orig


# --------------------------------------------------------------------- runner ---
def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed, failed = 0, []
    print(f"running {len(tests)} offline tests\n" + "-" * 60)
    for fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed.append((fn.__name__, exc))
            print(f"FAIL  {fn.__name__}\n      {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"ok    {fn.__name__}")
    print("-" * 60)
    print(f"{passed}/{len(tests)} passed")
    if failed:
        print(f"{len(failed)} FAILED")
        return 1
    print("ALL OFFLINE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
