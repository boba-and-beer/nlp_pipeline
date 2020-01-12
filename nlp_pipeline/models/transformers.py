"""
Create a custom transformer model as below
"""
from transformers import PreTrainedModel
import torch.nn as nn

class CustomTransformerModel(nn.Module):
    def __init__(self, transformer_model: PreTrainedModel):
        """Custom Transformer Model"""
        super(CustomTransformerModel, self).__init__()
        self.transformer = transformer_model

    def forward(self, input_ids, attention_mask, engineered_features=None):
        results = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            engineered_features=None,
        )
        logits = results[0]
        return logits