"""
Component 5: Hybrid Retrieval-Augmented Generation (RAG) - NOVEL ALGORITHM

Novel combination of three retrieval signals:
1. Dense (semantic similarity)
2. Lexical (BM25 keyword matching)
3. Causal (graph-based relations from CLKG)

This is the FIRST hybrid retrieval system combining all three signals
for legal document understanding.
"""

from .hybrid_retriever import HybridRetriever

__all__ = ['HybridRetriever']

