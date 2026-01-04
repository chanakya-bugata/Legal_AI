"""
Named Entity Recognition for Legal Documents
Extracts: Parties, amounts, dates, obligations, conditions
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict


class NERExtractor(nn.Module):
    """
    Extracts legal entities from text
    
    Entity types:
    - PARTY: Legal parties (Buyer, Seller, etc.)
    - AMOUNT: Monetary amounts
    - DATE: Dates and time periods
    - OBLIGATION: Legal obligations
    - CONDITION: Conditions and prerequisites
    """
    
    def __init__(self, model_name: str = "nlpaueb/legal-bert-base-uncased"):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        # 6 labels: PARTY, AMOUNT, DATE, OBLIGATION, CONDITION, O
        self.classifier = nn.Linear(self.bert.config.hidden_size, 6)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass"""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, 6)
            active_labels = torch.where(
                active_loss,
                labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
            return loss, logits
        
        return logits
    
    def extract_entities(self, text: str, tokenizer) -> List[Dict]:
        """Extract entities from text"""
        encoding = tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        self.eval()
        with torch.no_grad():
            logits = self.forward(encoding['input_ids'], encoding['attention_mask'])
            predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()
        
        # Convert to entity spans (simplified)
        entities = []
        entity_types = ['PARTY', 'AMOUNT', 'DATE', 'OBLIGATION', 'CONDITION', 'O']
        
        current_entity = None
        for i, pred in enumerate(predictions):
            if pred < 5:  # Not 'O'
                entity_type = entity_types[pred]
                if current_entity is None or current_entity['type'] != entity_type:
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {'type': entity_type, 'start': i, 'end': i}
                else:
                    current_entity['end'] = i
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        return entities

