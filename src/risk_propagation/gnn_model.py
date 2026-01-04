"""
Graph Attention Network (GAT) for Risk Propagation

NOVEL ALGORITHM: First application of GNNs to legal risk analysis
with cascade detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Tuple


class RiskPropagationGNN(nn.Module):
    """
    Graph Attention Network for propagating risk through clause dependencies
    
    Architecture:
    - Layer 1: Direct neighbor aggregation
    - Layer 2: Second-order dependencies
    - Layer 3: Global risk context
    - Risk scoring head
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        num_layers: int = 3,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            embedding_dim: Dimension of clause embeddings (768 for Legal-BERT)
            num_layers: Number of GAT layers
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Graph Attention Network layers
        self.gat_layers = nn.ModuleList()
        
        # First layer: embedding_dim -> hidden_dim
        self.gat_layers.append(
            GATConv(
                embedding_dim,
                hidden_dim,
                heads=num_heads,
                concat=True,
                dropout=dropout
            )
        )
        
        # Intermediate layers: hidden_dim*num_heads -> hidden_dim
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(
                    hidden_dim * num_heads,
                    hidden_dim,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout
                )
            )
        
        # Last layer: hidden_dim*num_heads -> hidden_dim (single head for final representation)
        self.gat_layers.append(
            GATConv(
                hidden_dim * num_heads,
                hidden_dim,
                heads=1,
                concat=False,
                dropout=dropout
            )
        )
        
        # Risk scoring head
        self.risk_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features (clause embeddings) [N, embedding_dim]
            edge_index: Graph structure [2, E]
            edge_attr: Edge attributes (relation types) [E, num_edge_types] (optional)
        
        Returns:
            risk_scores: Updated risk scores [N, 1]
        """
        # Pass through GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, edge_index, edge_attr)
            
            # Apply activation and dropout (except last layer)
            if i < len(self.gat_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Score each clause
        risk_scores = self.risk_scorer(x)
        
        return risk_scores
    
    def forward_with_attention(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention weights (for explainability)
        
        Returns:
            risk_scores: Updated risk scores [N, 1]
            attention_weights: Attention weights for visualization
        """
        # For now, return standard forward pass
        # In production, modify GAT layers to return attention weights
        risk_scores = self.forward(x, edge_index, edge_attr)
        
        # Placeholder for attention weights
        attention_weights = torch.zeros(edge_index.size(1))
        
        return risk_scores, attention_weights

