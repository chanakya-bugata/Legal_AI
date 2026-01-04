"""
Causal Relation Classifier
Classifies the type of causal relationship between two clauses
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Tuple
import numpy as np
from .clkg_graph import CausalRelationType


class CausalRelationClassifier(nn.Module):
    """
    Classifies causal relations between clause pairs
    
    Input: Two clause texts
    Output: Relation type + confidence
    """
    
    def __init__(
        self,
        model_name: str = "nlpaueb/legal-bert-base-uncased",
        num_relations: int = 7  # Number of relation types
    ):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        # Input: [CLS] token from concatenated clause pair
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_relations)
        self.num_relations = num_relations
        
        # Map relation indices to enum
        self.relation_map = {
            0: CausalRelationType.SUPPORTS,
            1: CausalRelationType.CONTRADICTS,
            2: CausalRelationType.MODIFIES,
            3: CausalRelationType.OVERTURNS,
            4: CausalRelationType.ENABLES,
            5: CausalRelationType.BLOCKS,
            6: CausalRelationType.REQUIRES
        }
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass"""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
    
    def predict(
        self,
        clause_i_text: str,
        clause_j_text: str,
        pair_embedding: np.ndarray = None
    ) -> Tuple[CausalRelationType, float]:
        """
        Predict relation type between two clauses
        
        Args:
            clause_i_text: First clause text
            clause_j_text: Second clause text
            pair_embedding: Optional pre-computed embedding
        
        Returns:
            (relation_type, confidence)
        """
        # Format input: "[CLS] clause_i [SEP] clause_j [SEP]"
        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        
        encoded = tokenizer(
            clause_i_text,
            clause_j_text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        self.eval()
        with torch.no_grad():
            logits = self.forward(
                encoded['input_ids'],
                encoded['attention_mask']
            )
            probs = torch.softmax(logits, dim=-1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()
        
        relation_type = self.relation_map.get(pred_idx, CausalRelationType.SUPPORTS)
        
        return relation_type, confidence

