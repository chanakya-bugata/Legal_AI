"""
Component 4: GNN-Based Risk Propagation - NOVEL ALGORITHM

Novel application of Graph Neural Networks to detect cascading risks
through clause dependencies. Unlike baseline systems that score clauses
independently, this propagates risk through the dependency graph.
"""

from .risk_propagator import RiskPropagator
from .gnn_model import RiskPropagationGNN

__all__ = ['RiskPropagator', 'RiskPropagationGNN']

