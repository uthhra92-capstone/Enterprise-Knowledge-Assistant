"""Streamlit UI for the Enterprise Knowledge Assistant.

    streamlit run app.py
"""

from __future__ import annotations

import os

import streamlit as st

# On Streamlit Community Cloud there is no local .env file; secrets configured in
# the app dashboard are exposed via st.secrets. Copy them into the environment
# BEFORE eka.config reads them. A local run with a real .env is unaffected
# (setdefault never overwrites an existing variable).
try:
    for _key, _value in st.secrets.items():
        if isinstance(_value, str):
            os.environ.setdefault(_key, _value)
except Exception:  # noqa: BLE001 - no secrets file is the normal local case
    pass

from eka import corpus  # noqa: E402
from eka.config import settings  # noqa: E402
from eka.history import ConversationStore  # noqa: E402
from eka.memory import ConversationMemory  # noqa: E402
from eka.pipeline import KnowledgeAssistant  # noqa: E402

st.set_page_config(page_title="Enterprise Knowledge Assistant", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading knowledge index…")
def load_assistant() -> KnowledgeAssistant:
    return KnowledgeAssistant.load()


@st.cache_resource(show_spinner=False)
def get_store() -> ConversationStore:
    return ConversationStore()


def rebuild_index() -> None:
    with st.spinner("Rebuilding index from the documents folder …"):
        KnowledgeAssistant.ingest()
    load_assistant.clear()
    st.session_state["corpus_dirty"] = False


def open_conversation(store: ConversationStore, conversation_id: str) -> None:
    """Point the session at *conversation_id* and hydrate memory from SQLite."""
    memory = ConversationMemory()
    for message in store.load_messages(conversation_id):
        if message.role == "user":
            memory.add_user(message.content)
        else:
            memory.add_assistant(
                message.content,
                sources=message.sources,
                standalone_question=message.standalone_question,
            )
    st.session_state.conversation_id = conversation_id
    st.session_state.memory = memory


def start_new_conversation() -> None:
    """Begin a fresh, empty conversation (previous one stays saved in history)."""
    st.session_state.conversation_id = ConversationStore.new_id()
    st.session_state.memory = ConversationMemory()


def clear_current_conversation(store: ConversationStore) -> None:
    """Reset the conversation: wipe its messages from memory *and* SQLite."""
    current = st.session_state.get("conversation_id")
    if current:
        store.delete_conversation(current)
    start_new_conversation()


def render_document_manager() -> None:
    """Upload / remove corpus files, then rebuild the index."""
    st.subheader("📁 Documents")

    upload_round = st.session_state.get("upload_round", 0)
    uploads = st.file_uploader(
        "Add files to the knowledge base",
        type=corpus.UPLOAD_TYPES,
        accept_multiple_files=True,
        key=f"uploader_{upload_round}",
    )
    if uploads:
        saved, failed = [], []
        for item in uploads:
            try:
                corpus.save_upload(item.name, item.getvalue())
                saved.append(item.name)
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{item.name}: {exc}")
        if saved:
            st.session_state["corpus_dirty"] = True
        st.session_state["upload_round"] = upload_round + 1  # resets the widget
        st.session_state["upload_report"] = {"saved": saved, "failed": failed}
        st.rerun()

    report = st.session_state.pop("upload_report", None)
    if report:
        if report["saved"]:
            st.success("Added: " + ", ".join(report["saved"]))
        for line in report["failed"]:
            st.error(line)

    files = corpus.list_source_files()
    if not files:
        st.caption("No documents yet — upload PDF / DOCX / TXT / MD files above.")
    else:
        st.caption(f"{len(files)} file(s) in `{corpus.documents_dir()}`")
        for source in files:
            name_col, del_col = st.columns([0.8, 0.2])
            flag = "" if source.ingestable else " ⚠️"
            name_col.markdown(f"`{source.name}`{flag}")
            name_col.caption(f"{source.size_kb:,.0f} KB · {source.kind or 'unrecognised type'}")
            if del_col.button("🗑", key=f"delfile_{source.name}", help="Remove this file"):
                corpus.delete_source_file(source.name)
                st.session_state["corpus_dirty"] = True
                st.rerun()

    if st.session_state.get("corpus_dirty"):
        st.warning("Documents changed — rebuild the index to apply.")

    if st.button("🧱 Rebuild index", use_container_width=True):
        try:
            rebuild_index()
            st.success("Index rebuilt.")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Rebuild failed: {exc}")


# --------------------------------------------------------------- top of sidebar
with st.sidebar:
    st.header("📚 Knowledge Assistant")
    st.caption("Advanced RAG · hybrid search · reranking · cited answers")

    manifest = KnowledgeAssistant.manifest()
    if manifest:
        st.subheader("Index")
        st.write(f"**Sources:** {manifest.get('num_sources', '?')}")
        st.write(f"**Chunks:** {manifest.get('num_chunks', '?')}")
        st.write(f"**Vector store:** `{manifest.get('vector_backend', '?')}`")
        st.write(f"**Embeddings:** `{manifest.get('embedding_model', '?')}`")
        st.write(f"**Built:** {manifest.get('created_at', '?')}")
        with st.expander("Documents in index"):
            for name in manifest.get("sources", []):
                st.write(f"- {name}")
    else:
        st.info("No index built yet.")

    render_document_manager()

    st.subheader("Settings")
    st.write(f"**LLM:** `{settings.llm_model}`")
    st.write(f"**Reranker:** `{settings.reranker}`")
    st.write(
        f"**Retrieval:** vec {settings.top_k_vector} + bm25 {settings.top_k_bm25} "
        f"→ fuse {settings.top_k_fused} → final {settings.top_k_final}"
    )


# ----------------------------------------------------------------- pre-flight
st.title("Enterprise Knowledge Assistant")

if not settings.has_openai_key:
    st.error(
        "`OPENAI_API_KEY` is not configured. Add it to a `.env` file (local) or to "
        "the app's Secrets (Streamlit Cloud), then restart."
    )
    st.stop()

if not KnowledgeAssistant.index_exists():
    st.warning(
        "No index yet. Add documents in the sidebar (**📁 Documents**), then click "
        "**🧱 Rebuild index**."
    )
    st.stop()

try:
    assistant = load_assistant()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load the index: {exc}")
    st.stop()

store = get_store()

# --- per-visitor identity -------------------------------------------------------
# Chat history lives in ONE SQLite file on the server. Scope it to this browser
# so two devices don't see each other's chats. The token rides in the URL
# (?s=...) so a refresh keeps your history; a fresh browser gets a new one.
if "owner_id" not in st.session_state:
    token = st.query_params.get("s")
    if not token:
        token = ConversationStore.new_id()
        st.query_params["s"] = token
    st.session_state.owner_id = token
owner_id: str = st.session_state.owner_id

# --- resolve the active conversation ---------------------------------------
if "conversation_id" not in st.session_state:
    recent = store.list_conversations(owner_id, limit=1)
    if recent:
        open_conversation(store, recent[0].id)
    else:
        start_new_conversation()

memory: ConversationMemory = st.session_state.memory
conversation_id: str = st.session_state.conversation_id


# ----------------------------------------------------------------- transcript
for turn in memory.turns:
    with st.chat_message(turn.role):
        st.markdown(turn.content)
        if turn.role == "assistant" and turn.sources:
            st.markdown("**Sources**")
            for source in turn.sources:
                st.markdown(f"- `{source}`")

placeholder = "Ask a question about your documents …"
question = st.chat_input(placeholder)

if question:
    with st.chat_message("user"):
        st.markdown(question)

    answer = None
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents and composing a grounded answer…"):
                answer = assistant.answer(question, memory)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Something went wrong while answering: {exc}")

        if answer is not None:
            st.markdown(answer.text)

            if answer.cited_sources:
                st.markdown("**Sources**")
                for source in answer.cited_sources:
                    st.markdown(f"- `{source}`")
            elif not answer.grounded:
                st.caption("No sufficiently relevant passage was found in the documents.")

            with st.expander(
                f"🔍 Retrieval detail — {len(answer.retrieved)} chunks · "
                f"best similarity {answer.best_similarity:.3f} · {answer.elapsed_s:.1f}s"
            ):
                if answer.standalone_question != question:
                    st.caption(f"Rewritten query: *{answer.standalone_question}*")
                for i, chunk in enumerate(answer.retrieved, 1):
                    st.markdown(
                        f"**{i}. `{chunk.source}`** — "
                        f"vector={chunk.vector_score:.3f} · "
                        f"fused={chunk.fused_score:.5f} · "
                        f"rerank={chunk.rerank_score if chunk.rerank_score is not None else '–'} · "
                        f"(vec rank {chunk.vector_rank}, bm25 rank {chunk.bm25_rank})"
                    )
                    st.caption(chunk.snippet + " …")

    if answer is not None:
        memory.add_user(question)
        memory.add_assistant(
            answer.text,
            sources=answer.cited_sources,
            standalone_question=answer.standalone_question,
        )
        store.append_message(conversation_id, "user", question, owner=owner_id)
        store.append_message(
            conversation_id,
            "assistant",
            answer.text,
            standalone_question=answer.standalone_question,
            sources=answer.cited_sources,
            owner=owner_id,
        )


# ------------------------------------------------------ bottom of sidebar: history
# Rendered last so a message sent this run already shows up in the list.
with st.sidebar:
    st.subheader("💬 Conversation")

    new_col, clear_col = st.columns(2)
    if new_col.button("➕ New chat", use_container_width=True, help="Start a fresh chat; this one stays saved"):
        start_new_conversation()
        st.rerun()
    if clear_col.button(
        "🧹 Clear / reset",
        use_container_width=True,
        help="Erase the current conversation (removes it from history too)",
    ):
        clear_current_conversation(store)
        st.rerun()

    st.markdown("**Past chats**")
    conversations = store.list_conversations(owner_id, limit=50)
    if not conversations:
        st.caption("Your conversations will be saved here.")
    for conv in conversations:
        is_active = conv.id == conversation_id
        row = st.columns([0.82, 0.18])
        label = f"{'🟢 ' if is_active else ''}{conv.title}"
        if row[0].button(
            label,
            key=f"open_{conv.id}",
            use_container_width=True,
            help=f"{conv.message_count} messages · updated {conv.updated_at}",
        ):
            open_conversation(store, conv.id)
            st.rerun()
        if row[1].button("🗑", key=f"del_{conv.id}", help="Delete this conversation"):
            store.delete_conversation(conv.id)
            if is_active:
                st.session_state.conversation_id = ConversationStore.new_id()
                st.session_state.memory = ConversationMemory()
            st.rerun()

    st.caption(f"Stored in `{store.db_path}`")
