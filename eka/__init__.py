"""Enterprise Knowledge Assistant (eka).

An advanced Retrieval-Augmented-Generation package:

    documents -> loaders -> chunking -> embeddings -> local vector store
                                                          |
    query -> (semantic search + BM25 keyword search) -> RRF fusion
                                                          |
                                     -> reranking -> grounded context
                                                          |
                              -> LLM + conversation memory -> answer + citations
"""

__version__ = "1.0.0"
