"""
Multi-Modal Document Encoder
Combines LayoutLMv3 (layout-aware) + Legal-BERT (legal domain)
"""

from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Optional
import torch
import numpy as np


class DocumentEncoder:
    """
    Encodes legal documents using:
    1. LayoutLMv3: Captures text + layout structure
    2. Legal-BERT: Legal domain-specific embeddings
    3. Fusion: Combines both representations
    """
    
    def __init__(
        self,
        legal_bert_model: str = "nlpaueb/legal-bert-base-uncased",
        layoutlm_model: str = "microsoft/layoutlmv3-base",
        device: str = "cpu"
    ):
        """
        Args:
            legal_bert_model: Legal-BERT model name
            layoutlm_model: LayoutLMv3 model name
            device: 'cpu' or 'cuda'
        """
        self.device = device
        
        # Load Legal-BERT
        self.legal_bert_tokenizer = AutoTokenizer.from_pretrained(legal_bert_model)
        self.legal_bert_model = AutoModel.from_pretrained(legal_bert_model).to(device)
        self.legal_bert_model.eval()
        
        # Load LayoutLMv3 (for layout-aware encoding)
        # Note: LayoutLMv3 requires image input, so we'll use it when available
        # For now, we'll focus on Legal-BERT and add LayoutLMv3 later
        self.layoutlm_tokenizer = None
        self.layoutlm_model = None
        
        # Try to load LayoutLMv3 (may fail if not available)
        try:
            self.layoutlm_tokenizer = AutoTokenizer.from_pretrained(layoutlm_model)
            self.layoutlm_model = AutoModel.from_pretrained(layoutlm_model).to(device)
            self.layoutlm_model.eval()
        except Exception as e:
            print(f"LayoutLMv3 not available, using Legal-BERT only: {e}")
    
    def encode_text(self, text: str, max_length: int = 512) -> np.ndarray:
        """
        Encode text using Legal-BERT
        
        Returns:
            768-dimensional embedding vector
        """
        # Tokenize
        inputs = self.legal_bert_tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Encode
        with torch.no_grad():
            outputs = self.legal_bert_model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding[0]  # Return as 1D array
    
    def encode_document(
        self,
        text: str,
        layout_info: Optional[Dict] = None,
        chunk_size: int = 512,
        overlap: int = 50
    ) -> Dict:
        """
        Encode entire document with chunking
        
        Args:
            text: Full document text
            layout_info: Optional layout information (for LayoutLMv3)
            chunk_size: Token chunk size
            overlap: Overlap between chunks
        
        Returns:
            Dictionary with:
            - embeddings: List of chunk embeddings
            - document_embedding: Average embedding
            - chunks: List of text chunks
        """
        # Split into chunks
        chunks = self._chunk_text(text, chunk_size, overlap)
        
        # Encode each chunk
        chunk_embeddings = []
        for chunk in chunks:
            embedding = self.encode_text(chunk)
            chunk_embeddings.append(embedding)
        
        # Average pooling for document-level embedding
        document_embedding = np.mean(chunk_embeddings, axis=0)
        
        return {
            'embeddings': chunk_embeddings,
            'document_embedding': document_embedding,
            'chunks': chunks,
            'num_chunks': len(chunks)
        }
    
    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks"""
        # Simple word-based chunking
        words = text.split()
        chunks = []
        
        i = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk = ' '.join(chunk_words)
            chunks.append(chunk)
            i += chunk_size - overlap
        
        return chunks
    
    def encode_clauses(self, clauses: List[str]) -> np.ndarray:
        """
        Encode multiple clauses
        
        Returns:
            Matrix of shape (num_clauses, 768)
        """
        embeddings = []
        for clause in clauses:
            emb = self.encode_text(clause)
            embeddings.append(emb)
        
        return np.array(embeddings)

