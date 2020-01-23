"""
Utils file to copy all of the pytorch functions over 
"""
from scipy.stats import spearmanr
import torch
import numpy as np
from fastai.text import *
from fastai.callback import *
from fastai import *
from fastai.callbacks.general_sched import *
import wandb
from wandb.fastai import WandbCallback
from functools import partial
from utils import *

# Convert tokens to ID
import pandas as pd

pd.set_option("display.max_columns", 500)

# Low level library
import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as topti

# Spearmanr scores
from scipy.stats import spearmanr

# Used to create learner objects
from fastai.text import *
from torch.utils.data import Dataset, DataLoader, random_split
from fastai.callbacks.tracker import *
from sklearn.model_selection import GroupShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# Progress bar
import types
from tqdm import tqdm

# Text preprocessing
from keras.preprocessing.sequence import pad_sequences

# from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset

# Transformers
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    DistilBertConfig,
)
from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig
from transformers import BertPreTrainedModel, BertModel

# AdamW Optimiser
from transformers import AdamW
from functools import partial

# For the learner object
import fastai

# Optimizer
from ranger import Ranger

# The usual stuff
from pathlib import Path
import sys
import random

def seed_all(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_dir(model_dir: Path):
    """Make the directory if it is not there."""
    while not model_dir.exists():
        parent = model_dir.parent
        child = model_dir
        while not parent.exists():
            parent = parent.parent
            child = child.parent
        if not parent.exists():
            parent.mkdir()
        if not child.exists():
            child.mkdir()


def count_numbers(word):
    digit_count = 0
    for letter in word:
        if letter.isdigit():
            digit_count += 1
    return digit_count


def add_to_list(filehandler: str, name: str):
    """Add a file to a list"""
    if os.path.exists(filehandler):
        list_of_names = pickle.load(open(filehandler, "rb"))
        list_of_names.append(name)
    else:
        list_of_names = [name]
    pickle.dump(list_of_names, open(filehandler, "wb"))


def normalize_float_list(float_list, length_mean, length_std):
    new_float_list = []
    for i, val in enumerate(float_list):
        new_float = (val - length_mean) / length_std
        new_float_list.append(new_float)
    return new_float_list


def count_word_list(text):
    word_length_list = []
    for passage in text:
        word_length_list.append(torch.FloatTensor(np.array([len(passage.split())])))
    return word_length_list


def count_question_marks(text):
    question_mark_count_list = []
    for passage in text:
        question_mark_count = torch.FloatTensor(np.array([passage.count("?")]))
        question_mark_count_list.append(question_mark_count)
    return question_mark_count_list


def combine_tensors(
    question_tensors, answer_tensors, question_mask_tensors, answer_mask_tensors
):
    tensors = []
    masks = []
    for i in range(len(question_tensors)):
        full_tensor = torch.cat((question_tensors[i], answer_tensors[i]), dim=0)
        tensors.append(full_tensor)
        full_tensor = torch.cat(
            (question_mask_tensors[i], answer_mask_tensors[i]), dim=0
        )
        masks.append(full_tensor)
    return tensors, masks


# Note: These functions have to be stored in the model itself.
def flattenAnneal(learn: Learner, lr: float, n_epochs: int, start_pct: float):
    n = len(learn.data.train_dl)
    anneal_start = int(n * n_epochs * start_pct)
    anneal_end = int(n * n_epochs) - anneal_start
    lr_array = np.array([lr / (2.6 ** i) for i in range(len(learn.layer_groups))])
    phases = [
        TrainingPhase(anneal_start).schedule_hp("lr", lr_array),
        TrainingPhase(anneal_end).schedule_hp("lr", lr_array, anneal=annealing_cos),
    ]
    sched = GeneralScheduler(learn, phases)
    learn.callbacks.append(sched)
    learn.fit(n_epochs)


def fit_sgd_warm(learn, n_cycles, lr, mom, cycle_len, cycle_mult):
    n = len(learn.data.train_dl)
    lr_array = np.array(
        [
            lr / (2.6 ** 9),
            lr / (2.6 ** 8),
            lr / (2.6 ** 7),
            lr / (2.6 ** 6),
            lr / (2.6 ** 5),
            lr / (2.6 ** 4),
            lr / (2.6 ** 3),
            lr / (2.6 ** 2),
            lr / (2.6 ** 1),
            lr,
        ]
    )
    phases = [
        (
            TrainingPhase(n * (cycle_len * cycle_mult ** i))
            .schedule_hp("lr", lr_array, anneal=annealing_cos)
            .schedule_hp("mom", mom)
        )
        for i in range(n_cycles)
    ]
    sched = GeneralScheduler(learn, phases)
    learn.callbacks.append(sched)
    if cycle_mult != 1:
        total_epochs = int(
            cycle_len * (1 - (cycle_mult) ** n_cycles) / (1 - cycle_mult)
        )
    else:
        total_epochs = n_cycles * cycle_len
    learn.fit(total_epochs)
