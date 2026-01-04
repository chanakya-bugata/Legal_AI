"""
Component 2: Clause Extraction & Relation Extraction
Identifies legal clauses and relationships between them
"""

from .clause_extractor import ClauseExtractor
from .relation_extractor import RelationExtractor
from .ner_extractor import NERExtractor

__all__ = ['ClauseExtractor', 'RelationExtractor', 'NERExtractor']

