"""
Component 3: Causal Legal Knowledge Graph (CLKG) - NOVEL ALGORITHM

This is the FIRST legal AI system to model explicit causal relationships
between clauses (not just semantic similarity).

Edge types: SUPPORTS, CONTRADICTS, MODIFIES, OVERTURNS, ENABLES, BLOCKS, REQUIRES
"""

from .clkg_builder import CLKGBuilder
from .clkg_graph import CLKGGraph
from .relation_classifier import CausalRelationClassifier

__all__ = ['CLKGBuilder', 'CLKGGraph', 'CausalRelationClassifier']

