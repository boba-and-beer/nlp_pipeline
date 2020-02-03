"""
Specific things used for google quest challenge
1. Features Created
2. Dataset Definition
"""
from typing import List
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, Sampler
from utils import *

# Define the dataset here 

def get_features(train, test, pretrain_model):
    """Generate a few features here"""
    # Mix question title and body together
    df = pd.concat([train, test], sort=True)
    df["question"] = df["question_title"] + df["question_body"]

    # Generate the features
    # 0
    df["q_word_count"] = df["question"].apply(lambda x: len(x.split()))
    df["a_word_count"] = df["answer"].apply(lambda x: len(x.split()))

    # 1
    df["q_char_count"] = df["question"].apply(lambda x: len(x.replace(" ", "")))
    df["a_char_count"] = df["answer"].apply(lambda x: len(x.replace(" ", "")))

    # 2
    df["a_word_density"] = df["a_word_count"] / (df["a_char_count"] + 1)

    # 3
    df["q_count_numbers"] = df["question"].apply(count_numbers)
    df["ans_count_numbers"] = df["answer"].apply(count_numbers)

    # 4
    df["q_brackets"] = df["question"].apply(lambda x: x.count("()"))
    df["a_brackets"] = df["answer"].apply(lambda x: x.count("()"))

    df["q_or"] = df["question"].apply(lambda x: x.count("or"))
    df["a_or"] = df["answer"].apply(lambda x: x.count("or"))

    # 5
    df["q_total_length"] = df["question"].apply(len)
    df["a_total_length"] = df["answer"].apply(len)

    # 6
    df["q_capitals"] = df["question"].apply(
        lambda comment: sum(1 for c in comment if c.isupper())
    )
    df["a_capitals"] = df["answer"].apply(
        lambda comment: sum(1 for c in comment if c.isupper())
    )

    # 7
    df["q_caps_vs_length"] = df.apply(
        lambda row: float(row["q_capitals"]) / float(row["q_total_length"]), axis=1
    )
    df["a_caps_vs_length"] = df.apply(
        lambda row: float(row["a_capitals"]) / float(row["q_total_length"]), axis=1
    )

    # 9
    df["q_num_question_marks"] = df["question"].apply(lambda x: x.count("?"))
    df["a_num_question_marks"] = df["answer"].apply(lambda x: x.count("?"))

    df["q_eq"] = df["question"].apply(lambda x: x.count("="))
    df["a_eq"] = df["answer"].apply(lambda x: x.count("="))

    # 10
    df["q_num_punctuation"] = df["question"].apply(
        lambda x: sum(x.count(w) for w in ".,;:")
    )
    df["a_num_punctuation"] = df["answer"].apply(
        lambda x: sum(x.count(w) for w in ".,;:")
    )

    # 12
    df["q_num_unique_words"] = df["question"].apply(
        lambda x: len(set(w for w in x.split()))
    )
    df["a_num_unique_words"] = df["answer"].apply(
        lambda x: len(set(w for w in x.split()))
    )

    # 13
    df["q_words_vs_unique"] = df["q_num_unique_words"] / df["q_word_count"]
    df["a_words_vs_unique"] = df["a_num_unique_words"] / df["a_word_count"]

    # 14 - num of lines
    df["q_num_of_lines"] = df["question"].apply(lambda x: x.count("\n"))
    df["a_num_of_lines"] = df["answer"].apply(lambda x: x.count("\n"))

    # ask yourself why you know
    df["q_why"] = df["question"].apply(lambda x: x.lower().count("why"))
    df["a_why"] = df["answer"].apply(lambda x: x.lower().count("why"))

    df["q_how"] = df["question"].apply(lambda x: x.lower().count("how "))
    df["a_how"] = df["answer"].apply(lambda x: x.lower().count("how "))

    # Adding categorical column data
    dummy_cols = pd.get_dummies(df["category"], drop_first=False, prefix="category")
    df = pd.concat([df, dummy_cols], axis=1)

    # Adding subcategories - change the 4
    df = df.reset_index()
    df["subcategory"] = (
        df["host"]
        .str.replace(".stackexchange", "")
        .str.replace(".com", "")
        .str.replace(".net", "")
        .str.replace("meta.", "")
    )

    subcategory_text = df["subcategory"]
    df.drop(["category", "subcategory"], axis=1, inplace=True)

    df_subcat, _ = pretrain_model.convert_text_to_tensor(
        text=subcategory_text,
        max_length=4,
        add_special_tokens=False,
        return_tensors=None,
    )

    subcat_df = pd.DataFrame(
        df_subcat, columns=["subcat_0", "subcat_1", "subcat_2", "subcat_3"]
    )

    # Normalize
    ss = StandardScaler(with_mean=True, with_std=True)

    ANSWER_FEATURES = [col for col in df.columns if col.startswith("a_")]
    QUESTION_FEATURES = [col for col in df.columns if col.startswith("q_")]
    subcat_features = [col for col in subcat_df.columns if col.startswith("subcat_")]

    # For each column we have to perform standard scaling
    train_features = df.head(len(train))
    train_subcat = subcat_df.head(len(train))
    test_features = df.tail(len(test))
    test_subcat = subcat_df.tail(len(test))

    # Combining the trian features
    train_features = pd.concat([train_features, train_subcat], axis=1)

    test_subcat.index = test_features.index
    test_features = pd.concat([test_features, test_subcat], axis=1)

    # Normalize the training data and then apply to test data
    train_features.loc[:, ANSWER_FEATURES + QUESTION_FEATURES] = ss.fit_transform(
        train_features.loc[:, ANSWER_FEATURES + QUESTION_FEATURES]
    )

    test_features[ANSWER_FEATURES + QUESTION_FEATURES] = ss.transform(
        test_features[ANSWER_FEATURES + QUESTION_FEATURES]
    )

    # Converting to test features cos it's a bit weird (removing bottom NA values)
    test_features = test_features.iloc[: test_features.shape[0]]

    return train_features, test_features, ss, ANSWER_FEATURES, QUESTION_FEATURES


#####################################
### Dataset Definition
#####################################
# class QuestDataset(Dataset):
#     def __init__(
#         self,
#         input_ids: List[torch.Tensor]=None,
#         attention_mask: List[torch.Tensor]=None,
#         inputs_embeds: List[torch.Tensor]=None,
#         label_tensors: torch.FloatTensor = None,
#         engineered_features=None,
#     ):
#         self.input_ids = input_ids
#         self.attention_mask = attention_mask
#         self.inputs_embeds = inputs_embeds
#         self.label_tensors = label_tensors
#         self.engineered_features = engineered_features
#         self.y = label_tensors

#     def __len__(self):
#         return len(self.y)

#     def __getitem__(self, idx):
        
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
        
#         sample = {
#             'input_ids': self.input_ids[idx] if self.input_ids is not None else None, 
#             'attention_mask': self.attention_mask[idx] if self.attention_mask is not None else None, 
#             'inputs_embeds': self.inputs_embeds[idx] if self.inputs_embeds is not None else None,
#             'engineered_features': self.engineered_features[idx] if self.engineered_features is not None else None
#         }
#         # Return objects even if they are None
#         prop_sample = {k: v for k, v in sample.items() if v is not None}
#         # Create a list of values
#         prop_sample = list(zip([v for k, v in sample.items() if v is not None]))
#         return self.x[idx], self.y[idx]
        
class QuestDataset(Dataset):
    def __init__(
        self,
        input_ids: List[torch.Tensor]=None,
        attention_mask: List[torch.Tensor]=None,
        label_tensors: torch.FloatTensor = None,
        **kwargs
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label_tensors = label_tensors
        # Create an obvious x
        self.x = list(zip(self.input_ids, self.attention_mask))
        self.y = label_tensors
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.x[idx], self.y[idx]

class EmbedDataset(Dataset):
    """
    Feed input embeddings through this. No features, that does not seem to help.
    """
    def __init__(
        self,
        inputs_embeds: List[torch.Tensor],
        label_tensors: torch.FloatTensor
    ):
        self.inputs_embeds = inputs_embeds
        self.label_tensors = label_tensors
        self.y = label_tensors

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.inputs_embeds[idx], self.y[idx]

class SimpleCustomBatch:
    def __init__(self, data):
        # Returns a list of of the iterated object
        transposed_data = list(zip(*data))

    # Custom memory pinning method on custom type 
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self
    
def collate_wrapper(batch):
    return SimpleCustomBatch(batch)