"""
Convert text to IDS  and then IDS to tensors 
"""
from transformers import PreTrainedTokenizer, PretrainedConfig
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from transformers import XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig
from transformers import XLMForSequenceClassification, XLMTokenizer, XLMConfig
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig

from typing import Union
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm

class Text2Tensor:
    """
    Class that converts texts to tensors to be placed inside datasets.
    """
    def __init__(self, model_type: Union[Path, str]):
        """
        Convert things to tensors.
        """
        self.model_type = model_type
        MODEL_CLASSES = {
            'bert': (BertForSequenceClassification, BertTokenizer, BertConfig),
            'xlnet': (XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig),
            'xlm': (XLMForSequenceClassification, XLMTokenizer, XLMConfig),
            'roberta': (RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig),
            'distilbert': (DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig)
        }
        if self.model_type == 'bert':
            self.model_name = 'bert-base-uncased'
        self.tokeniser_class = MODEL_CLASSES[model_type][1]
        self.config_class = MODEL_CLASSES[model_type][2]
        self.create_tokeniser()
        self.create_config()
    
    def create_tokeniser(self):
        self.tokeniser = self.tokeniser_class.from_pretrained(self.model_name)
        self.tokeniser.pretrained_init_configuration = self.config_class
        self.pad_first = bool(self.model_type in ["xlnet"])
        self.pad_idx = self.tokeniser.pad_token_id
    
    def create_config(self):
        self.custom_config_class = self.config_class.from_pretrained(self.model_name)

    def convert_text_to_tensor(self, text: pd.Series, head_len=None, **encode_params):
        """
        Currently only supports encoding for the head
        """
        train_tensors = []
        attention_masks = []
        for i in tqdm(range(len(text))):
            input_ids, attention_mask = self.encode_head_tail(text.iloc[i],  
            head_len=head_len, **encode_params)
            train_tensors.append(input_ids)
            attention_masks.append(attention_mask)
        return train_tensors, attention_masks
    
    def encode_head_tail(self, text: str, head_len=None, max_length=None,**kwargs):
        """
        Encode the head and tail of the length.
        """
        # The proportion of head and tail tokens
        
        if max_length is None:
            max_length = self.custom_config_class.max_position_embeddings
        
        if head_len is None:
            head_len = max_length
        tail_len = max_length - head_len

        encoded_tokens = self.tokeniser.encode_plus(
            text=text, pad_to_max_length=True, 
            truncation_strategy='do_not_truncate', return_tensors="pt", **kwargs
        )

        input_ids = encoded_tokens['input_ids']
        attention_mask = encoded_tokens["attention_mask"]

        # Taking the head and tail
        head_tokens = input_ids[:, :head_len]
        tail_tokens= input_ids[:, -tail_len:]
        input_ids = torch.cat((head_tokens, tail_tokens), dim=1)

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        return input_ids.flatten(), attention_mask.flatten()