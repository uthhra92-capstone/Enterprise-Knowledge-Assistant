"""Conversation memory + history-aware query rewriting.

A follow-up like "what about carry-forward?" is meaningless to a retriever on its
own. ``condense_question`` uses the recent turns to rewrite it into a standalone
query ("What is the carry-forward rule in the leave policy?") before retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

from .config import settings
from .prompts import CONDENSE_PROMPT


@dataclass
class ChatTurn:
    role: str  # "user" | "assistant"
    content: str
    sources: list[str] | None = None          # assistant turns: cited file names
    standalone_question: str | None = None     # assistant turns: memory-rewritten query


@dataclass
class ConversationMemory:
    window: int = field(default_factory=lambda: settings.memory_window)
    turns: list[ChatTurn] = field(default_factory=list)

    def add_user(self, content: str) -> None:
        self.turns.append(ChatTurn("user", content))

    def add_assistant(
        self,
        content: str,
        sources: list[str] | None = None,
        standalone_question: str | None = None,
    ) -> None:
        self.turns.append(
            ChatTurn("assistant", content, sources=sources, standalone_question=standalone_question)
        )

    def clear(self) -> None:
        self.turns.clear()

    def recent_messages(self) -> list[BaseMessage]:
        """Recent turns as LangChain messages, for the answering prompt."""
        messages: list[BaseMessage] = []
        for turn in self.turns[-self.window :]:
            if turn.role == "user":
                messages.append(HumanMessage(turn.content))
            else:
                messages.append(AIMessage(turn.content))
        return messages

    def as_text(self) -> str:
        """Recent turns as plain text, for the condense prompt."""
        lines = []
        for turn in self.turns[-self.window :]:
            speaker = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{speaker}: {turn.content}")
        return "\n".join(lines)


def condense_question(
    memory: ConversationMemory, question: str, llm: ChatOpenAI | None = None
) -> str:
    """Rewrite *question* into a standalone query using the conversation so far."""
    if not memory.turns:
        return question

    llm = llm or ChatOpenAI(
        model=settings.llm_model,
        temperature=0.0,
        api_key=settings.openai_api_key or None,
        base_url=settings.openai_base_url,
        max_retries=settings.max_retries,
        timeout=settings.request_timeout,
    )
    prompt = CONDENSE_PROMPT.format(chat_history=memory.as_text(), question=question)
    try:
        rewritten = llm.invoke(prompt).content.strip()
        return rewritten or question
    except Exception as exc:  # noqa: BLE001 - never block a query on this
        print(f"[memory] condense_question failed, using raw question: {exc}")
        return question
