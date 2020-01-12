"""
Unit testse for the text2tensor class.
"""
from .text_to_tensor import *
import pytest

@pytest.fixture
def series_text():
    text = pd.Series(['this is weird'])
    return text

@pytest.fixture
def train_input_ids():
    model = Text2Tensor('bert')
    text = "strange this is but a weird thing that does not exist. You are dumbing down."
    train_input, train_att = model.encode_head_tail(text, head_len=2)
    return train_input

def test_text_encode_head(train_input_ids):
    # Test this works with more encodings
    model = Text2Tensor('bert')
    text = "strange this is but a weird thing that does not exist. You are dumbing down."
    train_input, train_att = model.encode_head_tail(text, max_length=5, head_len=2)
    assert train_input.size()[0] == 5
    assert torch.all(train_input[0:2].eq(train_input_ids[0:2]))
    assert torch.all(train_input[-3:].eq(train_input_ids[-3:]))

def test_text_encode_plus(series_text):
    # Test this works
    model = Text2Tensor('bert')
    train_input, train_att = model.convert_text_to_tensor(series_text)
    assert 1 == 1

def test_text_encode_plus_padding_param(series_text):
    # Test this works with more encodings
    model = Text2Tensor('bert')
    train_input, train_att = model.convert_text_to_tensor(series_text, max_length=500)
    assert train_input[0].size()[1] == 500

def test_text_encode_plus_head_len(train_input_ids):
    # Test this works with more encodings
    model = Text2Tensor('bert')
    text = "strange this is but a weird thing that does not exist. You are dumbing down."
    train_input, train_att = model.encode_head_tail(text, max_length=10, head_len=2)
    assert train_input.size()[0] == 10
