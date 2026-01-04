"""
CLKG Builder - Constructs Causal Legal Knowledge Graph from clauses

NOVEL ALGORITHM: First system to build explicit causal relationships
between legal clauses.
"""

from typing import List, Dict
import numpy as np
from .clkg_graph import CLKGGraph, Clause, CausalEdge, CausalRelationType
from .relation_classifier import CausalRelationClassifier
from src.document_processing.document_encoder import DocumentEncoder


class CLKGBuilder:
    """
    Builds CLKG from extracted clauses
    
    Algorithm:
    1. Extract entities and obligations from each clause
    2. Fine-tune relation extraction model
    3. Predict relations between clause pairs
    4. Validate with domain experts (optional)
    5. Build graph
    """
    
    def __init__(
        self,
        encoder: DocumentEncoder,
        relation_classifier: CausalRelationClassifier,
        confidence_threshold: float = 0.7
    ):
        """
        Args:
            encoder: Document encoder for clause embeddings
            relation_classifier: Model to classify causal relations
            confidence_threshold: Minimum confidence for edge creation
        """
        self.encoder = encoder
        self.relation_classifier = relation_classifier
        self.confidence_threshold = confidence_threshold
    
    def build_graph(self, clauses: List[Dict]) -> CLKGGraph:
        """
        Build CLKG from list of clauses
        
        Args:
            clauses: List of clause dictionaries with 'text', 'id', etc.
        
        Returns:
            CLKGGraph instance
        """
        graph = CLKGGraph()
        
        # Step 1: Create clause nodes
        clause_objects = []
        for clause_dict in clauses:
            clause = Clause(
                id=clause_dict.get('id', f"clause_{len(clause_objects)}"),
                text=clause_dict['text'],
                start_pos=clause_dict.get('start', 0),
                end_pos=clause_dict.get('end', len(clause_dict['text'])),
                entities=clause_dict.get('entities', []),
                obligations=clause_dict.get('obligations', []),
                conditions=clause_dict.get('conditions', []),
                risk_score=clause_dict.get('risk_score', 0.0)
            )
            clause_objects.append(clause)
            graph.add_clause(clause)
        
        # Step 2: Extract clause embeddings
        clause_texts = [c.text for c in clause_objects]
        clause_embeddings = self.encoder.encode_clauses(clause_texts)
        
        # Step 3: Predict relations between all clause pairs
        for i, clause_i in enumerate(clause_objects):
            for j, clause_j in enumerate(clause_objects):
                if i != j:
                    # Encode clause pair
                    pair_embedding = self._encode_clause_pair(
                        clause_embeddings[i],
                        clause_embeddings[j]
                    )
                    
                    # Predict relation type
                    relation_type, confidence = self.relation_classifier.predict(
                        clause_i.text,
                        clause_j.text,
                        pair_embedding
                    )
                    
                    # Add edge if confidence is high enough
                    if confidence >= self.confidence_threshold:
                        explanation = self._generate_explanation(
                            clause_i, clause_j, relation_type
                        )
                        
                        edge = CausalEdge(
                            source_id=clause_i.id,
                            target_id=clause_j.id,
                            relation_type=relation_type,
                            confidence=confidence,
                            explanation=explanation
                        )
                        graph.add_edge(edge)
        
        return graph
    
    def _encode_clause_pair(
        self,
        embedding_i: np.ndarray,
        embedding_j: np.ndarray
    ) -> np.ndarray:
        """
        Encode clause pair for relation classification
        
        Strategies:
        - Concatenation: [emb_i, emb_j]
        - Element-wise operations: [emb_i, emb_j, emb_i - emb_j, emb_i * emb_j]
        """
        # Use concatenation + difference + element-wise product
        diff = embedding_i - embedding_j
        product = embedding_i * embedding_j
        
        pair_embedding = np.concatenate([
            embedding_i,
            embedding_j,
            diff,
            product
        ])
        
        return pair_embedding
    
    def _generate_explanation(
        self,
        clause_i: Clause,
        clause_j: Clause,
        relation_type: CausalRelationType
    ) -> str:
        """Generate human-readable explanation for relation"""
        
        templates = {
            CausalRelationType.SUPPORTS: (
                f"Clause '{clause_i.text[:50]}...' supports "
                f"'{clause_j.text[:50]}...' by enabling its fulfillment."
            ),
            CausalRelationType.CONTRADICTS: (
                f"Clause '{clause_i.text[:50]}...' contradicts "
                f"'{clause_j.text[:50]}...' - these provisions conflict."
            ),
            CausalRelationType.MODIFIES: (
                f"Clause '{clause_i.text[:50]}...' modifies the scope of "
                f"'{clause_j.text[:50]}...'."
            ),
            CausalRelationType.OVERTURNS: (
                f"Clause '{clause_i.text[:50]}...' overturns or replaces "
                f"'{clause_j.text[:50]}...'."
            ),
            CausalRelationType.ENABLES: (
                f"Clause '{clause_i.text[:50]}...' is a prerequisite for "
                f"'{clause_j.text[:50]}...'."
            ),
            CausalRelationType.BLOCKS: (
                f"Clause '{clause_i.text[:50]}...' prevents "
                f"'{clause_j.text[:50]}...'."
            ),
            CausalRelationType.REQUIRES: (
                f"If '{clause_i.text[:50]}...' occurs, then "
                f"'{clause_j.text[:50]}...' is mandatory."
            )
        }
        
        return templates.get(relation_type, "Causal relationship detected.")

