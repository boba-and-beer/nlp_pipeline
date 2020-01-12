"""
Convert text to IDS  and then IDS to tensors 
"""
from transformers import PreTrainedTokenizer, PretrainedConfig
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    RobertaConfig,
)
from transformers import XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig
from transformers import XLMForSequenceClassification, XLMTokenizer, XLMConfig
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    DistilBertConfig,
)

from typing import Union
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class Text2Tensor:
    """
    Class that converts texts to tensors to be placed inside datasets.
    """

    MODEL_CLASSES = {
        "bert": (BertForSequenceClassification, BertTokenizer, BertConfig),
        "xlnet": (XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig),
        "xlm": (XLMForSequenceClassification, XLMTokenizer, XLMConfig),
        "roberta": (RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig),
        "distilbert": (
            DistilBertForSequenceClassification,
            DistilBertTokenizer,
            DistilBertConfig,
        ),
    }

    def choose_model(self, model_name):
        if model_name == "bert":
            self.model_name = "bert-base-uncased"
            self.tokeniser_class = self.MODEL_CLASSES[model_name][1]
            config_class = self.MODEL_CLASSES[model_name][2]
        self.config = config_class.from_pretrained(self.model_name)
        self.create_tokeniser()

    def create_tokeniser(self):
        self.tokeniser = self.tokeniser_class.from_pretrained(self.model_name)
        self.tokeniser.pretrained_init_configuration = self.config
        self.pad_first = bool(self.model_name in ["xlnet"])
        self.pad_idx = self.tokeniser.pad_token_id

    def convert_text_to_tensor(
        self, text: pd.Series, head_len=None, return_tensors="pt", max_length=None, **encode_params
    ):
        """
        Text -> Match to ID
        """
        train_tensors = []
        attention_masks = []
        for i in tqdm(range(len(text))):
            input_ids, attention_mask = self.encode_head_tail(
                text.iloc[i],
                head_len=head_len,
                return_tensors=return_tensors,
                max_length=max_length,
                **encode_params
            )
            train_tensors.append(input_ids)
            attention_masks.append(attention_mask)
        return train_tensors, attention_masks

    def encode_head_tail(
        self,
        text: str,
        head_len=None,
        max_length=None,
        add_special_tokens=True,
        return_tensors="pt",
        **kwargs
    ):
        """
        Encode the head and tail of the length.
        """
        # The proportion of head and tail tokens

        try:
            encoded_tokens = self.tokeniser.encode_plus(
                text=text,
                pad_to_max_length=True,
                max_length=max_length,
                truncation_strategy="do_not_truncate",
                return_tensors=return_tensors,
                add_special_tokens=add_special_tokens,
            )
        except:
            import pdb
            pdb.set_trace()

        input_ids = encoded_tokens["input_ids"]
        attention_mask = encoded_tokens["attention_mask"]

        if max_length is None:
            max_length = self.config.max_position_embeddings

        if head_len is None:
            head_len = max_length
        tail_len = max_length - head_len

        # Taking the head and tail
        if isinstance(input_ids[0], int):
            head_tokens = input_ids[:head_len]
            if tail_len == 0:
                input_ids = head_tokens
            else:
                tail_tokens = input_ids[-tail_len:]
                input_ids = head_tokens + tail_tokens
            attention_mask = attention_mask[:(max_length)]
            return input_ids, attention_mask
        else:
            head_tokens = input_ids[:, :head_len]
            attention_mask = attention_mask[:, :head_len]
            if tail_len == 0:
                input_ids = head_tokens
            else:
                tail_tokens = input_ids[:, -tail_len:]
                input_ids = torch.cat((head_tokens, tail_tokens), dim=1)

            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            
            
            if input_ids.size()[1] > 512:
                import pdb; pdb.set_trace()
            return input_ids.flatten(), attention_mask.flatten()
           