"""Build (or rebuild) the local knowledge index from ./data/documents.

    python ingest.py
"""

from __future__ import annotations

import sys

from eka.config import settings
from eka.pipeline import KnowledgeAssistant


def main() -> int:
    if not settings.has_openai_key:
        print("ERROR: OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
        return 1

    try:
        assistant = KnowledgeAssistant.ingest()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}")
        return 1

    info = KnowledgeAssistant.manifest()
    print("\nDone.")
    print(f"  sources : {info.get('num_sources', '?')}")
    print(f"  chunks  : {info.get('num_chunks', len(assistant.vector_store))}")
    print(f"  model   : {info.get('embedding_model', settings.embedding_model)}")
    print("\nNow run:  streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
