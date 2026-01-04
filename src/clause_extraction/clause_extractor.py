"""
Clause Extraction using Token Classification (BIO tagging)
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple
import numpy as np


class ClauseExtractor(nn.Module):
    """
    Extracts legal clauses using token-level classification
    Labels: B-Clause (beginning), I-Clause (inside), O (outside)
    """
    
    def __init__(
        self,
        model_name: str = "nlpaueb/legal-bert-base-uncased",
        num_labels: int = 3  # B-Clause, I-Clause, O
    ):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.num_labels = num_labels
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask
            labels: BIO labels for training
        
        Returns:
            logits: Classification logits [batch_size, seq_len, num_labels]
            loss: Cross-entropy loss (if labels provided)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Flatten for loss calculation
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss,
                labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
            return loss, logits
        
        return logits
    
    def extract_clauses(self, text: str, tokenizer) -> List[Dict]:
        """
        Extract clauses from text
        
        Returns:
            List of clause dictionaries with:
            - text: Clause text
            - start: Start position in original text
            - end: End position
            - confidence: Prediction confidence
        """
        # Tokenize
        encoding = tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Predict
        self.eval()
        with torch.no_grad():
            logits = self.forward(
                encoding['input_ids'],
                encoding['attention_mask']
            )
            predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()
        
        # Convert predictions to clauses
        clauses = self._predictions_to_clauses(
            text,
            predictions,
            encoding,
            tokenizer
        )
        
        return clauses
    
    def _predictions_to_clauses(
        self,
        text: str,
        predictions: np.ndarray,
        encoding: Dict,
        tokenizer
    ) -> List[Dict]:
        """Convert BIO predictions to clause spans"""
        clauses = []
        current_clause = None
        
        # Map token predictions back to text
        tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
        
        for i, (token, pred) in enumerate(zip(tokens, predictions)):
            if pred == 0:  # B-Clause
                # Start new clause
                if current_clause:
                    clauses.append(current_clause)
                current_clause = {'tokens': [token], 'start_idx': i}
            elif pred == 1:  # I-Clause
                # Continue current clause
                if current_clause:
                    current_clause['tokens'].append(token)
            else:  # O
                # End current clause
                if current_clause:
                    current_clause['end_idx'] = i
                    clauses.append(current_clause)
                    current_clause = None
        
        # Handle last clause
        if current_clause:
            current_clause['end_idx'] = len(tokens)
            clauses.append(current_clause)
        
        # Convert token indices to text spans
        clause_texts = []
        for clause in clauses:
            # Reconstruct clause text (simplified - in production, use proper mapping)
            clause_text = ' '.join(clause['tokens']).replace(' ##', '')
            clause_texts.append({
                'text': clause_text,
                'start': clause.get('start_idx', 0),
                'end': clause.get('end_idx', len(tokens))
            })
        
        return clause_texts

