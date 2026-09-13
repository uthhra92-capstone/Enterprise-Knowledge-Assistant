# Testing

Three layers: an automated offline suite (free), a scripted live run, and a
manual UI checklist.

## 1. Automated offline tests — no API key, no network

```bash
python tests/run_offline_tests.py
```

18 tests covering every module with a deterministic fake embedder and fake LLM:

| Area | Checks |
|---|---|
| `loaders` | loads `.md` + `.txt`, magic-byte type detection, metadata |
| `chunking` | splitting increases count, `chunk_id` values unique |
| `vector_store` | FAISS **and** NumPy: `from_documents → save → load → search`, sorted results, `similarity()` lookup, FAISS is the default backend |
| `retrievers` | `build_bm25`, RRF dedupes by `chunk_id` and orders correctly, `HybridRetriever` returns candidates carrying both vector and BM25 ranks |
| `reranker` | `NoopReranker` truncates to `top_n`; `LLMReranker` reorders by score and degrades to fusion order on failure |
| `memory` | window caps recent turns; `condense_question` is a no-op with no history and rewrites with history |
| `history` | SQLite create/append/list/load/delete, auto-title, **per-owner isolation**, **old-schema migration** |
| `corpus` | add / list / delete, type detection, unsafe path names reduced to a bare filename |
| `pipeline` | full `answer()`: grounded answer + citations, not-found path (grounding gate), follow-up condensation, and `ingest()` writing `index.faiss` / `embeddings.npy` / `chunks.jsonl` / `manifest.json` |

Expected tail: `18/18 passed` / `ALL OFFLINE TESTS PASSED`.

## 2. Scripted live run — needs `OPENAI_API_KEY`

```bash
python ingest.py      # build the FAISS index from data/documents/
python eval.py        # run a 6-question conversation through the real pipeline
```

Costs a fraction of a cent on `gpt-4o-mini`. The captured result is in
[`SAMPLE_OUTPUT.md`](SAMPLE_OUTPUT.md). What to look for:

- answers state the correct figures from the source files;
- questions 2–3 print a `rewritten ->` line (conversational memory);
- each answer's `[n]` markers match the `sources` line;
- question 6 returns exactly *"I could not find this information in the available documents."* with no sources.

## 3. Manual UI checklist — `streamlit run app.py`

| # | Step | Expected result | Requirement |
|---|---|---|---|
| 1 | Start with no `storage/` index | Sidebar shows "No index yet"; **📁 Documents** lists the 9 sample files | Ingestion / error messages |
| 2 | Click **🧱 Rebuild index** | Success message; sidebar shows Sources 9, Chunks ~19, Vector store `faiss` | Ingestion, embeddings, vector DB |
| 3 | Ask *"What is the annual leave entitlement?"* | Answer says 20 days (25 after 5 years); **Sources** lists `Leave_Policy.md` | Retrieval, citations, grounding |
| 4 | Expand **🔍 Retrieval detail** | 4 chunks listed with vector / fused / rerank scores | Hybrid search + reranking visible |
| 5 | Ask follow-up *"What about carry-forward?"* | Answer about the 5-day / 31 March rule; expander shows a rewritten query | Conversational memory |
| 6 | Ask *"Who won the 2022 World Cup?"* | *"I could not find this information in the available documents."*, no sources | Hallucination handling |
| 7 | Click **🧹 Clear / reset** | Transcript clears; a fresh chat starts | Clear/reset conversation |
| 8 | Send a message, then **➕ New chat**, then reopen the previous chat from **Past chats** | The earlier conversation reloads with its messages and sources | Conversation history (persistent) |
| 9 | Upload a new `.txt` in **📁 Documents**, click **🧱 Rebuild index**, ask about its content | The new file is answered from and cited | Document management |
| 10 | Remove `OPENAI_API_KEY` from `.env` and restart | Clear red error, app stops gracefully | Basic error messages |
| 11 | Open the app in a second browser / device | Its **Past chats** list is empty — it does not see the other device's history | Per-visitor isolation |

## Known cosmetic issue

`langchain-openai`'s `with_structured_output` (used by the LLM reranker) emits a
`PydanticSerializationUnexpectedValue ... field_name='parsed'` warning on stderr
with current `pydantic` versions. It does not affect results. Silence it by
adding to the top of `eval.py` / `app.py`:

```python
import warnings
warnings.filterwarnings("ignore", message=".*parsed.*", module="pydantic.*")
```
