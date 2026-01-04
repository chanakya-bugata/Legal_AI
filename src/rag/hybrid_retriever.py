"""
Hybrid Retriever - Combines Dense + Lexical + Causal Retrieval

NOVEL ALGORITHM: First system to combine three orthogonal retrieval
signals for legal document understanding.
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from src.clkg.clkg_graph import CLKGGraph
from src.document_processing.document_encoder import DocumentEncoder


class HybridRetriever:
    """
    Hybrid retrieval combining:
    1. Dense retrieval (semantic similarity)
    2. Lexical retrieval (BM25)
    3. Causal retrieval (graph-based)
    """
    
    def __init__(
        self,
        clauses: List[Dict],
        graph: CLKGGraph,
        encoder: DocumentEncoder,
        dense_weight: float = 0.5,
        lexical_weight: float = 0.3,
        causal_weight: float = 0.2
    ):
        """
        Args:
            clauses: List of clause dictionaries
            graph: CLKG graph
            encoder: Document encoder for embeddings
            dense_weight: Weight for dense retrieval
            lexical_weight: Weight for lexical retrieval
            causal_weight: Weight for causal retrieval
        """
        self.clauses = clauses
        self.graph = graph
        self.encoder = encoder
        
        # Normalize weights
        total = dense_weight + lexical_weight + causal_weight
        self.dense_weight = dense_weight / total
        self.lexical_weight = lexical_weight / total
        self.causal_weight = causal_weight / total
        
        # Build indices
        self._build_dense_index()
        self._build_lexical_index()
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve relevant clauses using hybrid approach
        
        Returns:
            List of clause dictionaries with relevance scores
        """
        # Signal 1: Dense retrieval
        dense_results = self._dense_retrieve(query, top_k=10)
        dense_scores = {r['id']: r['score'] for r in dense_results}
        
        # Signal 2: Lexical retrieval
        lexical_results = self._lexical_retrieve(query, top_k=10)
        lexical_scores = {r['id']: r['score'] for r in lexical_results}
        
        # Signal 3: Causal retrieval
        causal_results = self._causal_retrieve(query, top_k=10)
        causal_scores = {r['id']: r['score'] for r in causal_results}
        
        # Normalize scores to [0, 1]
        dense_scores = self._normalize_scores(dense_scores)
        lexical_scores = self._normalize_scores(lexical_scores)
        causal_scores = self._normalize_scores(causal_scores)
        
        # Combine scores
        combined_scores = {}
        all_ids = set(dense_scores.keys()) | set(lexical_scores.keys()) | set(causal_scores.keys())
        
        for clause_id in all_ids:
            score = (
                self.dense_weight * dense_scores.get(clause_id, 0.0) +
                self.lexical_weight * lexical_scores.get(clause_id, 0.0) +
                self.causal_weight * causal_scores.get(clause_id, 0.0)
            )
            combined_scores[clause_id] = score
        
        # Sort and return top-k
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Format results
        results = []
        for clause_id, score in sorted_results:
            clause = next((c for c in self.clauses if c.get('id') == clause_id), None)
            if clause:
                results.append({
                    'id': clause_id,
                    'text': clause.get('text', ''),
                    'score': score,
                    'dense_score': dense_scores.get(clause_id, 0.0),
                    'lexical_score': lexical_scores.get(clause_id, 0.0),
                    'causal_score': causal_scores.get(clause_id, 0.0)
                })
        
        return results
    
    def _dense_retrieve(self, query: str, top_k: int) -> List[Dict]:
        """Semantic similarity using dense embeddings"""
        # Encode query
        query_embedding = self.encoder.encode_text(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Compute similarities
        similarities = cosine_similarity(query_embedding, self.clause_embeddings)[0]
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            clause = self.clauses[idx]
            results.append({
                'id': clause.get('id', f'clause_{idx}'),
                'score': float(similarities[idx])
            })
        
        return results
    
    def _lexical_retrieve(self, query: str, top_k: int) -> List[Dict]:
        """Keyword matching using BM25"""
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Score all clauses
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            clause = self.clauses[idx]
            results.append({
                'id': clause.get('id', f'clause_{idx}'),
                'score': float(scores[idx])
            })
        
        return results
    
    def _causal_retrieve(self, query: str, top_k: int) -> List[Dict]:
        """Graph-based retrieval using CLKG"""
        # Find most relevant clause for query
        query_embedding = self.encoder.encode_text(query)
        
        most_relevant = None
        best_similarity = -1
        
        for clause in self.clauses:
            clause_id = clause.get('id')
            if clause_id not in self.graph.clauses:
                continue
            
            clause_text = clause.get('text', '')
            clause_embedding = self.encoder.encode_text(clause_text)
            
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                clause_embedding.reshape(1, -1)
            )[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                most_relevant = clause_id
        
        if most_relevant is None:
            return []
        
        # BFS from most_relevant in CLKG
        visited = set()
        queue = [(most_relevant, 0, 1.0)]  # (clause_id, distance, score)
        results = []
        
        while queue and len(results) < top_k * 2:
            clause_id, distance, score = queue.pop(0)
            
            if clause_id in visited or distance > 2:
                continue
            
            visited.add(clause_id)
            
            # Add to results
            score_decay = 1.0 / (1.0 + distance)
            results.append({
                'id': clause_id,
                'score': score * score_decay
            })
            
            # Explore neighbors
            clause = self.graph.clauses.get(clause_id)
            if clause:
                neighbors = self.graph.get_neighbors(clause_id)
                for neighbor in neighbors:
                    if neighbor.id not in visited:
                        queue.append((neighbor.id, distance + 1, score * 0.7))
        
        return results[:top_k]
    
    def _build_dense_index(self):
        """Build dense embedding index"""
        clause_texts = [c.get('text', '') for c in self.clauses]
        self.clause_embeddings = self.encoder.encode_clauses(clause_texts)
    
    def _build_lexical_index(self):
        """Build BM25 index"""
        corpus = [c.get('text', '').lower().split() for c in self.clauses]
        self.bm25 = BM25Okapi(corpus)
    
    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to [0, 1]"""
        if not scores:
            return {}
        
        max_score = max(scores.values())
        min_score = min(scores.values())
        
        if max_score == min_score:
            return {k: 1.0 for k in scores.keys()}
        
        return {
            k: (v - min_score) / (max_score - min_score)
            for k, v in scores.items()
        }

