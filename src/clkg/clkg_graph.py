"""
Causal Legal Knowledge Graph (CLKG) Data Structure

NOVEL CONTRIBUTION: First system to model explicit causal relationships
between legal clauses (not just similarity).
"""

from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum


class CausalRelationType(Enum):
    """Causal relation types between clauses"""
    SUPPORTS = "SUPPORTS"          # A enables fulfillment of B
    CONTRADICTS = "CONTRADICTS"    # A conflicts with B
    MODIFIES = "MODIFIES"          # A changes scope of B
    OVERTURNS = "OVERTURNS"        # A voids/replaces B
    ENABLES = "ENABLES"            # A is prerequisite for B
    BLOCKS = "BLOCKS"              # A prevents B
    REQUIRES = "REQUIRES"          # B mandatory if A occurs


@dataclass
class Clause:
    """Represents a legal clause"""
    id: str
    text: str
    start_pos: int
    end_pos: int
    entities: List[Dict] = None  # Parties, amounts, dates
    obligations: List[Dict] = None  # What each party must do
    conditions: List[Dict] = None  # When obligations apply
    risk_score: float = 0.0  # Initial risk score (0-1)
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = []
        if self.obligations is None:
            self.obligations = []
        if self.conditions is None:
            self.conditions = []


@dataclass
class CausalEdge:
    """Represents a causal relationship between two clauses"""
    source_id: str
    target_id: str
    relation_type: CausalRelationType
    confidence: float  # 0.0-1.0
    explanation: str  # Human-readable explanation
    
    def __repr__(self):
        return f"{self.source_id} --[{self.relation_type.value}]--> {self.target_id} (conf: {self.confidence:.2f})"


class CLKGGraph:
    """
    Causal Legal Knowledge Graph
    
    Nodes: Clauses
    Edges: Causal relationships with semantic labels
    """
    
    def __init__(self):
        self.clauses: Dict[str, Clause] = {}
        self.edges: List[CausalEdge] = []
        self.adjacency: Dict[str, List[str]] = {}  # clause_id -> list of connected clause_ids
    
    def add_clause(self, clause: Clause):
        """Add a clause node to the graph"""
        self.clauses[clause.id] = clause
        self.adjacency[clause.id] = []
    
    def add_edge(self, edge: CausalEdge):
        """Add a causal relationship edge"""
        # Validate
        if edge.source_id not in self.clauses:
            raise ValueError(f"Source clause {edge.source_id} not in graph")
        if edge.target_id not in self.clauses:
            raise ValueError(f"Target clause {edge.target_id} not in graph")
        
        self.edges.append(edge)
        self.adjacency[edge.source_id].append(edge.target_id)
    
    def get_neighbors(
        self,
        clause_id: str,
        relation_type: Optional[CausalRelationType] = None
    ) -> List[Clause]:
        """
        Get neighboring clauses
        
        Args:
            clause_id: ID of clause
            relation_type: Optional filter by relation type
        
        Returns:
            List of neighboring clauses
        """
        neighbors = []
        
        for edge in self.edges:
            if edge.source_id == clause_id:
                if relation_type is None or edge.relation_type == relation_type:
                    neighbors.append(self.clauses[edge.target_id])
        
        return neighbors
    
    def get_contradictions(self, clause_id: str) -> List[Clause]:
        """Get all clauses that contradict the given clause"""
        return self.get_neighbors(clause_id, CausalRelationType.CONTRADICTS)
    
    def get_supports(self, clause_id: str) -> List[Clause]:
        """Get all clauses that support the given clause"""
        return self.get_neighbors(clause_id, CausalRelationType.SUPPORTS)
    
    def find_contradiction_chains(self, max_length: int = 5) -> List[List[str]]:
        """
        Find chains of contradictions (cascade detection)
        
        Returns:
            List of clause ID chains that form contradiction paths
        """
        chains = []
        visited = set()
        
        def dfs(current_id: str, path: List[str]):
            if len(path) > max_length:
                return
            
            if current_id in visited:
                return
            
            visited.add(current_id)
            path.append(current_id)
            
            # Find contradictions
            contradictions = self.get_contradictions(current_id)
            for contradicted_clause in contradictions:
                if contradicted_clause.id not in path:
                    dfs(contradicted_clause.id, path.copy())
            
            # If path has multiple clauses, it's a chain
            if len(path) >= 2:
                chains.append(path)
        
        # Start DFS from each clause
        for clause_id in self.clauses:
            visited.clear()
            dfs(clause_id, [])
        
        return chains
    
    def get_statistics(self) -> Dict:
        """Get graph statistics"""
        return {
            'num_clauses': len(self.clauses),
            'num_edges': len(self.edges),
            'num_contradictions': len([
                e for e in self.edges
                if e.relation_type == CausalRelationType.CONTRADICTS
            ]),
            'num_supports': len([
                e for e in self.edges
                if e.relation_type == CausalRelationType.SUPPORTS
            ]),
            'avg_degree': len(self.edges) / len(self.clauses) if self.clauses else 0
        }
    
    def to_networkx(self):
        """Convert to NetworkX graph for visualization"""
        try:
            import networkx as nx
            
            G = nx.DiGraph()
            
            # Add nodes
            for clause_id, clause in self.clauses.items():
                G.add_node(clause_id, text=clause.text[:50], risk=clause.risk_score)
            
            # Add edges
            for edge in self.edges:
                G.add_edge(
                    edge.source_id,
                    edge.target_id,
                    relation=edge.relation_type.value,
                    confidence=edge.confidence,
                    explanation=edge.explanation
                )
            
            return G
        except ImportError:
            raise ImportError("NetworkX required for visualization. Install with: pip install networkx")

