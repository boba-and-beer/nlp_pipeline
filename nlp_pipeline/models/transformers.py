"""
Create a custom transformer model as below
"""
from transformers import PreTrainedModel
from nlp_pipeline.models.pooler import *
import torch.nn as nn
import torch


class CustomTransformerModel(nn.Module):
    def __init__(self, transformer_model: PreTrainedModel):
        """Simplest Transformer Model"""
        super(CustomTransformerModel, self).__init__()
        self.transformer = transformer_model

    def forward(self, input_ids, attention_mask, engineered_features=None):
        results = self.transformer(
            input_ids, attention_mask=attention_mask, engineered_features=None,
        )
        logits = results[0]
        return logits

class CTMWithFeatures(CustomTransformerModel):
    """
    Combines Bert Model With Features
    """
    def __init__(self, transformer_model: PreTrainedModel, num_of_features: int):
        super(CTMWithFeatures, self).__init__(transformer_model)
        self.transformer = transformer_model
        self.pooler = SimplePooler(num_of_features)

        # Review dropout
        self.dropout = nn.Dropout(transformer_model.dropout_rate)
        self.classifier = nn.Linear(
            num_of_features + 768, self.transformer.config.num_labels
        )

    # def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, 
    # engineered_features=None):
    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, 
        engineered_features=None, **kwargs):
        
        results = self.transformer(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            inputs_embeds=None, 
            engineered_features=None
        )

        # Creating a multi-layer perceptron with engineered features
        pooled_feat = self.pooler(engineered_features)
        logits = results[0]
        
        # Now we forward this to other layers
        output = torch.cat((results, pooled_feat), dim=1)
        output = self.dropout(output)
        output = self.classifier(output)
        return output

class CTMEncoded(CustomTransformerModel):
    """
    Combines Bert Model With Features
    """
    def __init__(self, transformer_model: PreTrainedModel, num_of_features: int):
        super(CTMEncoded, self).__init__(transformer_model)
        self.transformer = transformer_model
        self.pooler = SimplePooler(num_of_features)

        # Review dropout
        self.dropout = nn.Dropout(transformer_model.dropout_rate)
        self.classifier = nn.Linear(
            num_of_features + 768, self.transformer.config.num_labels
        )

    # def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, 
    # engineered_features=None):
    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, 
        engineered_features=None, **kwargs):
        
        results = self.transformer(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            inputs_embeds=None, 
            engineered_features=None
        )

        # Creating a multi-layer perceptron with engineered features
        pooled_feat = self.pooler(engineered_features)
        logits = results[0]
        
        # Now we forward this to other layers
        output = torch.cat((results, pooled_feat), dim=1)
        output = self.dropout(output)
        output = self.classifier(output)
        return output