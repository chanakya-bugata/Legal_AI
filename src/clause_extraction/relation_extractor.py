"""
Relation Extraction between clauses
Extracts: REQUIRES, PROHIBITS, MODIFIES, etc.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple


class RelationExtractor(nn.Module):
    """
    Extracts relations between clause pairs
    """
    
    def __init__(self, model_name: str = "nlpaueb/legal-bert-base-uncased"):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        # Relation types: REQUIRES, PROHIBITS, MODIFIES, etc.
        self.classifier = nn.Linear(self.bert.config.hidden_size, 10)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass"""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        
        return logits
    
    def extract_relation(
        self,
        clause_i: str,
        clause_j: str,
        tokenizer
    ) -> Tuple[str, float]:
        """Extract relation between two clauses"""
        encoded = tokenizer(
            clause_i,
            clause_j,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        self.eval()
        with torch.no_grad():
            logits = self.forward(encoded['input_ids'], encoded['attention_mask'])
            probs = torch.softmax(logits, dim=-1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()
        
        relation_types = [
            'REQUIRES', 'PROHIBITS', 'MODIFIES', 'SUPPORTS',
            'CONTRADICTS', 'ENABLES', 'BLOCKS', 'OVERTURNS',
            'RELATED', 'NONE'
        ]
        
        return relation_types[pred_idx], confidence

