"""
Baseline Comparison Framework

Compares:
1. Baseline: Standard RAG (dense only)
2. Proposed: Hybrid RAG (dense + lexical + causal)

For ablation studies and evaluation.
"""

from typing import List, Dict
import numpy as np
from sklearn.metrics import average_precision_score
from src.rag.hybrid_retriever import HybridRetriever
from src.document_processing.document_encoder import DocumentEncoder
from src.clkg.clkg_graph import CLKGGraph


class BaselineRetriever:
    """Baseline: Dense retrieval only (standard RAG)"""
    
    def __init__(self, clauses: List[Dict], encoder: DocumentEncoder):
        self.clauses = clauses
        self.encoder = encoder
        self._build_index()
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Dense retrieval only"""
        query_embedding = self.encoder.encode_text(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        similarities = np.dot(query_embedding, self.clause_embeddings.T)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [
            {
                'id': self.clauses[i].get('id', f'clause_{i}'),
                'text': self.clauses[i].get('text', ''),
                'score': float(similarities[i])
            }
            for i in top_indices
        ]
    
    def _build_index(self):
        clause_texts = [c.get('text', '') for c in self.clauses]
        self.clause_embeddings = self.encoder.encode_clauses(clause_texts).T


def evaluate_retrieval(
    retriever,
    queries: List[str],
    ground_truth: List[List[str]],  # List of relevant clause IDs per query
    top_k: int = 5
) -> Dict:
    """
    Evaluate retrieval performance
    
    Returns:
        Dictionary with MAP@k, Precision@k, Recall@k
    """
    all_precisions = []
    all_recalls = []
    all_aps = []
    
    for query, true_clause_ids in zip(queries, ground_truth):
        retrieved = retriever.retrieve(query, top_k=top_k)
        retrieved_ids = [r['id'] for r in retrieved]
        
        # Precision@k
        relevant_retrieved = len(set(retrieved_ids) & set(true_clause_ids))
        precision = relevant_retrieved / top_k if top_k > 0 else 0.0
        all_precisions.append(precision)
        
        # Recall@k
        recall = relevant_retrieved / len(true_clause_ids) if true_clause_ids else 0.0
        all_recalls.append(recall)
        
        # Average Precision
        y_true = [1 if r_id in true_clause_ids else 0 for r_id in retrieved_ids]
        y_scores = [r['score'] for r in retrieved]
        if any(y_true):
            ap = average_precision_score(y_true, y_scores)
            all_aps.append(ap)
    
    return {
        'map@k': np.mean(all_aps) if all_aps else 0.0,
        'precision@k': np.mean(all_precisions),
        'recall@k': np.mean(all_recalls),
        'num_queries': len(queries)
    }


def compare_baseline_vs_proposed(
    clauses: List[Dict],
    graph: CLKGGraph,
    encoder: DocumentEncoder,
    queries: List[str],
    ground_truth: List[List[str]]
) -> Dict:
    """
    Compare baseline vs proposed system
    
    Returns:
        Dictionary with comparison results
    """
    # Baseline
    baseline_retriever = BaselineRetriever(clauses, encoder)
    baseline_results = evaluate_retrieval(baseline_retriever, queries, ground_truth)
    
    # Proposed (Hybrid)
    hybrid_retriever = HybridRetriever(clauses, graph, encoder)
    proposed_results = evaluate_retrieval(hybrid_retriever, queries, ground_truth)
    
    # Compute improvements
    improvements = {
        'map@k': (proposed_results['map@k'] - baseline_results['map@k']) / baseline_results['map@k'] * 100,
        'precision@k': (proposed_results['precision@k'] - baseline_results['precision@k']) / baseline_results['precision@k'] * 100,
        'recall@k': (proposed_results['recall@k'] - baseline_results['recall@k']) / baseline_results['recall@k'] * 100
    }
    
    return {
        'baseline': baseline_results,
        'proposed': proposed_results,
        'improvements': improvements
    }

