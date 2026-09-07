"""Quick sanity/demo harness - run a set of questions through the pipeline.

    python eval.py

Needs a built index (python ingest.py) and OPENAI_API_KEY. The last two
questions deliberately probe conversational memory and hallucination handling.
"""

from __future__ import annotations

import sys

from eka.config import settings
from eka.memory import ConversationMemory
from eka.pipeline import KnowledgeAssistant

# Generic probes; the last two exercise memory and hallucination handling.
# Adjust the wording to match whatever documents you have indexed.
CONVERSATION = [
    "Give me a short summary of what these documents cover.",
    "What does the disciplinary / misconduct policy say?",
    "What are the possible outcomes or penalties?",   # follow-up: relies on memory
    "What is the process for raising a complaint or grievance?",
    "What is the company's policy on time travel to the year 3000?",  # not in docs
]


def main() -> int:
    if not settings.has_openai_key:
        print("OPENAI_API_KEY is not set.")
        return 1
    if not KnowledgeAssistant.index_exists():
        print("No index found. Run:  python ingest.py")
        return 1

    assistant = KnowledgeAssistant.load()
    memory = ConversationMemory()

    for question in CONVERSATION:
        print("\n" + "=" * 88)
        print(f"Q: {question}")
        answer = assistant.answer(question, memory)
        if answer.standalone_question != question:
            print(f"   (rewritten -> {answer.standalone_question})")
        print(f"\nA: {answer.text}")
        print(f"\n   sources        : {', '.join(answer.cited_sources) or '(none)'}")
        print(f"   grounded       : {answer.grounded}  best_sim={answer.best_similarity:.3f}")
        print(f"   retrieved      : {', '.join(f'{c.source}[{c.rerank_score}]' for c in answer.retrieved)}")
        print(f"   latency        : {answer.elapsed_s:.2f}s")
        memory.add_user(question)
        memory.add_assistant(answer.text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
