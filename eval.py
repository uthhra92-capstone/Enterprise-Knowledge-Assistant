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

# Probes for the bundled Northwind Dynamics sample documents.
# Q3 is a bare follow-up (memory), Q6 is not covered by any document (hallucination).
CONVERSATION = [
    "What is the annual leave entitlement?",
    "How does carry-forward work and by when must it be used?",
    "What about the rules for sick leave?",              # follow-up: relies on memory
    "What is the per diem for international business travel?",
    "How much parental leave does a primary caregiver get?",
    "What is the company's policy on cryptocurrency trading bonuses?",  # not in docs
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
