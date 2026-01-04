"""
Main Pipeline: End-to-End Legal Document Analysis

Integrates all 6 components:
1. Document Processing
2. Clause Extraction
3. CLKG Construction
4. Risk Propagation
5. Hybrid RAG
6. Generation (placeholder)
"""

from typing import Dict, List, Optional
from src.document_processing.pdf_parser import PDFParser
from src.document_processing.document_encoder import DocumentEncoder
from src.clause_extraction.clause_extractor import ClauseExtractor
from src.clkg.clkg_builder import CLKGBuilder
from src.clkg.relation_classifier import CausalRelationClassifier
from src.risk_propagation.risk_propagator import RiskPropagator
from src.risk_propagation.gnn_model import RiskPropagationGNN
from src.rag.hybrid_retriever import HybridRetriever


class LegalIntelligencePipeline:
    """
    Complete pipeline for legal document analysis
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize all components
        
        Args:
            device: 'cpu' or 'cuda'
        """
        self.device = device
        
        # Component 1: Document Processing
        self.pdf_parser = PDFParser(use_ocr=False)
        self.encoder = DocumentEncoder(device=device)
        
        # Component 2: Clause Extraction
        self.clause_extractor = ClauseExtractor()
        
        # Component 3: CLKG
        self.relation_classifier = CausalRelationClassifier()
        self.clkg_builder = CLKGBuilder(
            encoder=self.encoder,
            relation_classifier=self.relation_classifier
        )
        
        # Component 4: Risk Propagation
        gnn_model = RiskPropagationGNN()
        self.risk_propagator = RiskPropagator(gnn_model, device=device)
        
        # Component 5: RAG (initialized after document processing)
        self.retriever = None
    
    def process_document(self, pdf_path: str) -> Dict:
        """
        Process a legal document end-to-end
        
        Returns:
            Dictionary with:
            - clauses: Extracted clauses
            - clkg: Causal knowledge graph
            - risks: Risk scores
            - statistics: Graph statistics
        """
        # Step 1: Parse PDF
        print("Step 1: Parsing PDF...")
        parsed_doc = self.pdf_parser.parse(pdf_path)
        
        # Step 2: Encode document
        print("Step 2: Encoding document...")
        encoded_doc = self.encoder.encode_document(parsed_doc['text'])
        
        # Step 3: Extract clauses
        print("Step 3: Extracting clauses...")
        tokenizer = self.encoder.legal_bert_tokenizer
        clauses = self.clause_extractor.extract_clauses(
            parsed_doc['text'],
            tokenizer
        )
        
        # Format clauses for CLKG
        clause_dicts = []
        for i, clause in enumerate(clauses):
            clause_dicts.append({
                'id': f'clause_{i}',
                'text': clause['text'],
                'start': clause.get('start', 0),
                'end': clause.get('end', len(clause['text']))
            })
        
        # Step 4: Build CLKG
        print("Step 4: Building Causal Legal Knowledge Graph...")
        clkg = self.clkg_builder.build_graph(clause_dicts)
        
        # Step 5: Propagate risks
        print("Step 5: Propagating risks through GNN...")
        clause_texts = [c['text'] for c in clause_dicts]
        clause_embeddings = self.encoder.encode_clauses(clause_texts)
        risks = self.risk_propagator.propagate_risks(
            clkg,
            clause_embeddings
        )
        
        # Update clause risk scores in graph
        for clause_id, risk_score in risks.items():
            if clause_id in clkg.clauses:
                clkg.clauses[clause_id].risk_score = risk_score
        
        # Step 6: Initialize RAG retriever
        print("Step 6: Initializing hybrid retrieval...")
        self.retriever = HybridRetriever(
            clauses=clause_dicts,
            graph=clkg,
            encoder=self.encoder
        )
        
        # Get statistics
        stats = clkg.get_statistics()
        stats['num_clauses'] = len(clauses)
        stats['avg_risk'] = sum(risks.values()) / len(risks) if risks else 0.0
        
        return {
            'clauses': clause_dicts,
            'clkg': clkg,
            'risks': risks,
            'statistics': stats,
            'document_text': parsed_doc['text']
        }
    
    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """
        Query the document using hybrid RAG
        
        Args:
            query_text: Natural language query
            top_k: Number of results to return
        
        Returns:
            List of relevant clauses with scores
        """
        if self.retriever is None:
            raise ValueError("Must process document first. Call process_document()")
        
        return self.retriever.retrieve(query_text, top_k=top_k)
    
    def get_risk_analysis(self) -> Dict:
        """
        Get comprehensive risk analysis
        
        Returns:
            Dictionary with risk scores, cascades, and explanations
        """
        if self.retriever is None:
            raise ValueError("Must process document first. Call process_document()")
        
        clkg = self.retriever.graph
        risks = {
            clause_id: clause.risk_score
            for clause_id, clause in clkg.clauses.items()
        }
        
        # Detect cascades
        cascades = self.risk_propagator.detect_cascade_risks(clkg, risks)
        
        # Get high-risk clauses
        high_risk = [
            {
                'id': clause_id,
                'text': clkg.clauses[clause_id].text[:100] + '...',
                'risk': risk_score
            }
            for clause_id, risk_score in risks.items()
            if risk_score >= 0.7
        ]
        high_risk.sort(key=lambda x: x['risk'], reverse=True)
        
        return {
            'risks': risks,
            'cascades': cascades,
            'high_risk_clauses': high_risk,
            'statistics': clkg.get_statistics()
        }

