# Enterprise Knowledge Assistant — Advanced RAG

A production-oriented **Retrieval-Augmented Generation** application that answers
employee questions from a collection of private company documents. It goes well
beyond "chat with a PDF":

| Requirement | How it is implemented |
|---|---|
| Document ingestion (≥2 formats) | `eka/loaders.py` — `.pdf`, `.docx`, `.txt`, `.md` |
| Chunking | `eka/chunking.py` — `RecursiveCharacterTextSplitter`, stable `chunk_id` |
| Embeddings | `eka/embeddings.py` — OpenAI `text-embedding-3-small` |
| Local vector store | `eka/vector_store.py` — **FAISS** `IndexFlatIP` (default), with a zero-dependency NumPy store as automatic fallback (`VECTOR_BACKEND`) |
| Semantic retrieval | `LocalVectorStore.search()` |
| Keyword retrieval (BM25) | `langchain_community.retrievers.BM25Retriever` |
| Hybrid search | `eka/retrievers.py` — **Reciprocal Rank Fusion** of vector + BM25 |
| Reranking | `eka/reranker.py` — LLM relevance judge (or Cohere, or none) |
| Conversational memory | `eka/memory.py` — windowed history + **history-aware query rewriting**; `eka/history.py` persists every chat to **SQLite** with a sidebar "past chats" picker |
| Source citations | `[n]` markers in the prompt → parsed back to file names |
| Hallucination handling | strict system prompt + **semantic-similarity grounding gate** |
| UI | `app.py` — Streamlit chat, history, reset, sources, retrieval inspector, in-app **document upload / remove** |
| Structured architecture | small single-responsibility modules under `eka/` |
| Deployment | ready for **Streamlit Community Cloud** (`.streamlit/secrets.toml`, `st.secrets` → env) |

## Architecture

```
data/documents/*.{pdf,docx,txt,md}
        │  load  →  chunk  →  embed  →  persist
        ▼
   storage/  (index.faiss · embeddings.npy · chunks.jsonl · manifest.json · conversations.db)
        │
        ▼
 question ─► condense with memory ─► ┌─ semantic (vector) search ─┐
                                    │                            ├─► RRF fusion
                                    └─ keyword (BM25) search ─────┘
                                                     │
                                              rerank (LLM judge)
                                                     │
                                        grounding gate (min similarity)
                                                     │
                             LLM + recent chat turns + numbered context
                                                     │
                                    grounded answer + [n] citations → sources
                                                     │
                                              Streamlit chat UI
```

## Requirements

- **Python 3.11 – 3.14** (see the note below)
- An **OpenAI API key**

### Python 3.14 note

The code targets 3.14 and uses no removed/deprecated stdlib. Every dependency
(including `faiss-cpu 1.15`, `numpy`, `lxml`, `tiktoken`) has a Python 3.14 wheel,
so `pip install -r requirements.txt` works as-is. If a future environment lacks a
FAISS wheel, set `VECTOR_BACKEND=numpy` in `.env` — the app then uses the built-in
NumPy vector store and everything else is unchanged.

## Setup (Windows PowerShell)

```powershell
cd "C:\Users\New\Desktop\Capstone Project"

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

notepad .env          # paste your OPENAI_API_KEY
```

macOS / Linux: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

## Run

```powershell
# 1) Put some .pdf / .docx / .txt / .md files in data/documents/
#    (or add them from the app sidebar after it starts)

# 2) Build the index
python ingest.py

# 3) Launch the app
streamlit run app.py
```

Then open http://localhost:8501 and ask questions about your documents. Follow-ups
work — e.g. *"What is the leave policy?"* then *"What about carry-forward?"* (the
second question is rewritten with the conversation before retrieval). Ask about
something not in the documents and you get *"I could not find this information in
the available documents."*

Optional non-UI demo: `python eval.py`

## Managing documents

Two ways to change the corpus:

- **From the app** — sidebar → **📁 Documents**: upload files, remove files with 🗑,
  then click **🧱 Rebuild index**.
- **On disk** — drop files into `data/documents/` (subfolders fine) and run
  `python ingest.py`.

Files with a missing/incorrect extension are detected by content. Scanned PDFs
with no text layer are reported and skipped (they need OCR first).

## Deploy to Streamlit Community Cloud

1. Push this folder to a **GitHub repo**. If your documents are confidential, use a
   **private** repo, or don't commit `data/documents/` (see the commented lines in
   `.gitignore`) and upload them from the app after deploy.
2. Go to <https://share.streamlit.io> → **New app**, pick the repo, set the main
   file to `app.py`, and choose **Python 3.12** or **3.13** under *Advanced*.
3. Open **Advanced → Secrets** and paste (see `.streamlit/secrets.toml.example`):
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```
4. **Deploy.** On first load there is no index — open the sidebar, add documents
   (or they're already in the repo), and click **🧱 Rebuild index**.

Notes: Streamlit Cloud's filesystem is **ephemeral** — the FAISS index and
`conversations.db` are rebuilt/reset whenever the app restarts or redeploys.
That's fine for a demo; for durable storage use a host with a persistent volume
or an external database.

## Configuration

Everything is driven by `.env`: models, chunk size,
the `TOP_K_*` retrieval widths, `RRF_K`, `RERANKER` (`llm` / `cohere` / `none`),
`MEMORY_WINDOW`, and `SCORE_THRESHOLD` (the grounding gate — raise it to make the
assistant more conservative about answering).

## Conversation memory & history

Two layers:

1. **In-session memory** (`eka/memory.py`) — a windowed buffer (`MEMORY_WINDOW`
   turns). It feeds *history-aware query rewriting*: `condense_question()` turns
   "What about carry-forward?" into a standalone query before retrieval, and the
   recent turns are also passed to the LLM when composing the answer.
2. **Persistent history** (`eka/history.py`) — every turn is written to
   `storage/conversations.db` (SQLite, standard library, no extra dependency).
   The sidebar shows a **Past chats** list: click a title to reopen it (memory is
   rehydrated from the DB), **➕ New chat** to start fresh, 🗑 to delete one.
   Conversations are auto-titled from your first question. Cited sources are
   stored per message and re-rendered when you reopen a chat.

Inspect it directly if you like:

```bash
sqlite3 storage/conversations.db "SELECT role, substr(content,1,60) FROM messages ORDER BY id;"
```

## How hallucination is mitigated

1. **System prompt** forbids outside knowledge and requires `[n]` citations for
   every claim.
2. **Grounding gate** — if the best retrieved chunk's cosine similarity to the
   (rewritten) question is below `SCORE_THRESHOLD`, the LLM is never called and the
   assistant returns the fixed "not found" message.
3. **Citation parsing** — the answer's `[n]` markers are mapped back to real file
   names and shown as **Sources**; the "not found" response yields no sources.
4. The **retrieval inspector** expander shows every candidate chunk with its
   vector / fusion / rerank scores so answers are auditable.

## Project layout

```
Capstone Project/
├─ app.py                 # Streamlit UI
├─ ingest.py              # build/rebuild the index
├─ eval.py                # scripted demo (memory + hallucination probes)
├─ requirements.txt
├─ .env
├─ .python-version                 # pin for pyenv / Render (3.12)
├─ .streamlit/
│  ├─ config.toml                  # theme
│  └─ secrets.toml.example         # deploy secrets template
├─ data/documents/                 # your source files (put them here)
├─ storage/                        # generated index + chat DB (gitignored)
└─ eka/
   ├─ config.py                    # env-driven settings
   ├─ loaders.py                   # pdf / docx / txt / md → Document (+ magic-byte sniff)
   ├─ corpus.py                    # list / add / remove corpus files (used by the UI)
   ├─ chunking.py                  # recursive splitter + chunk_id
   ├─ embeddings.py                # OpenAI embeddings factory
   ├─ vector_store.py              # FAISS index (+ NumPy fallback) + persistence
   ├─ retrievers.py                # BM25 + vector + reciprocal rank fusion
   ├─ reranker.py                  # LLM / Cohere / noop rerankers
   ├─ memory.py                    # in-session windowed memory + query condensation
   ├─ history.py                   # SQLite persistence of past conversations
   ├─ prompts.py                   # system / answer / condense / rerank prompts
   └─ pipeline.py                  # KnowledgeAssistant: ingest() + answer()
```

## Notes / possible extensions

- FAISS is the default store; a NumPy store is the fallback. To add ChromaDB,
  implement the same interface (`from_documents`, `save`, `load`, `search`,
  `similarity`) in `eka/vector_store.py` and register it in `resolve_vector_store`.
- Add streaming token output in the UI with `st.write_stream`.
- Add per-document access control by filtering candidates on a `metadata` field.
