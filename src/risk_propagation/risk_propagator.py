"""
Risk Propagator - Main interface for GNN-based risk propagation
"""

import torch
import numpy as np
from typing import List, Dict, Optional
from .gnn_model import RiskPropagationGNN
from src.clkg.clkg_graph import CLKGGraph, Clause, CausalRelationType


class RiskPropagator:
    """
    Propagates risk through clause dependencies using GNN
    
    Algorithm:
    1. Compute initial risk scores per clause
    2. Build graph from CLKG
    3. Propagate risk through GNN
    4. Apply contradiction penalties
    5. Return updated risk scores
    """
    
    def __init__(
        self,
        gnn_model: RiskPropagationGNN,
        device: str = "cpu"
    ):
        """
        Args:
            gnn_model: Trained GNN model
            device: 'cpu' or 'cuda'
        """
        self.gnn_model = gnn_model.to(device)
        self.device = device
    
    def propagate_risks(
        self,
        graph: CLKGGraph,
        clause_embeddings: np.ndarray,
        initial_risks: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Propagate risks through clause dependencies
        
        Args:
            graph: CLKG graph
            clause_embeddings: Embeddings for each clause [N, 768]
            initial_risks: Optional initial risk scores per clause
        
        Returns:
            Dictionary mapping clause_id -> updated risk score
        """
        # Step 1: Prepare node features
        clause_ids = list(graph.clauses.keys())
        node_features = torch.tensor(clause_embeddings, dtype=torch.float32).to(self.device)
        
        # Step 2: Convert graph to edge index
        edge_index, edge_attr = self._graph_to_edge_index(graph, clause_ids)
        
        # Step 3: Forward pass through GNN
        self.gnn_model.eval()
        with torch.no_grad():
            propagated_scores = self.gnn_model(node_features, edge_index, edge_attr)
        
        # Step 4: Apply contradiction penalties
        updated_risks = {}
        for i, clause_id in enumerate(clause_ids):
            risk = propagated_scores[i].item()
            
            # Check for contradictions
            contradictions = graph.get_contradictions(clause_id)
            if contradictions:
                # Increase risk due to inconsistency
                num_contradictions = len(contradictions)
                contradiction_penalty = 0.1 * num_contradictions
                risk = min(1.0, risk + contradiction_penalty)
            
            updated_risks[clause_id] = risk
        
        return updated_risks
    
    def _graph_to_edge_index(
        self,
        graph: CLKGGraph,
        clause_ids: List[str]
    ) -> tuple:
        """
        Convert CLKG to PyTorch Geometric format
        
        Returns:
            edge_index: [2, E] tensor
            edge_attr: [E, num_edge_types] tensor (one-hot encoded)
        """
        # Create mapping from clause_id to index
        id_to_idx = {clause_id: i for i, clause_id in enumerate(clause_ids)}
        
        edge_index = []
        edge_types = []
        
        # Relation type to index mapping
        relation_to_idx = {
            CausalRelationType.SUPPORTS: 0,
            CausalRelationType.CONTRADICTS: 1,
            CausalRelationType.MODIFIES: 2,
            CausalRelationType.OVERTURNS: 3,
            CausalRelationType.ENABLES: 4,
            CausalRelationType.BLOCKS: 5,
            CausalRelationType.REQUIRES: 6
        }
        
        for edge in graph.edges:
            source_idx = id_to_idx.get(edge.source_id)
            target_idx = id_to_idx.get(edge.target_id)
            
            if source_idx is not None and target_idx is not None:
                edge_index.append([source_idx, target_idx])
                
                # One-hot encode relation type
                relation_idx = relation_to_idx.get(edge.relation_type, 0)
                one_hot = [0.0] * 7
                one_hot[relation_idx] = 1.0
                edge_types.append(one_hot)
        
        if not edge_index:
            # No edges - return empty tensors
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
            edge_attr_tensor = torch.empty((0, 7), dtype=torch.float32)
        else:
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr_tensor = torch.tensor(edge_types, dtype=torch.float32)
        
        return edge_index_tensor.to(self.device), edge_attr_tensor.to(self.device)
    
    def detect_cascade_risks(
        self,
        graph: CLKGGraph,
        risk_scores: Dict[str, float],
        threshold: float = 0.7
    ) -> List[Dict]:
        """
        Detect cascade risks (chain reactions)
        
        Returns:
            List of cascade risk dictionaries with:
            - chain: List of clause IDs in cascade
            - total_risk: Combined risk score
            - explanation: Text explanation
        """
        cascades = []
        
        # Find contradiction chains
        chains = graph.find_contradiction_chains()
        
        for chain in chains:
            # Compute combined risk
            chain_risks = [risk_scores.get(clause_id, 0.0) for clause_id in chain]
            total_risk = max(chain_risks)  # Use max for worst-case
            
            # Add penalty for chain length
            chain_penalty = 0.05 * (len(chain) - 1)
            total_risk = min(1.0, total_risk + chain_penalty)
            
            if total_risk >= threshold:
                cascades.append({
                    'chain': chain,
                    'total_risk': total_risk,
                    'explanation': (
                        f"Risk cascade detected: {len(chain)} clauses form "
                        f"a contradiction chain with combined risk {total_risk:.2f}"
                    )
                })
        
        return cascades

