"""
Component 1: Multi-Modal Document Encoder
Extracts text, layout, and structure from legal documents
"""

from .pdf_parser import PDFParser
from .document_encoder import DocumentEncoder

__all__ = ['PDFParser', 'DocumentEncoder']

