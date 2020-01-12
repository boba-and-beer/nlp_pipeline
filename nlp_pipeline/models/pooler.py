"""
This module contains poolers.
"""
import torch.nn as nn

class BertPooler(nn.Module):
    """
    Generic Bert Pooler
    """
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # We take take the mean of all the hidden states
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class MeanPooler(nn.Module):
    def __init__(self, config):
        super(MeanPooler, self).__init__()
        self.meanpool = nn.AvgPool1d(kernel_size=512 * 2)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        second_last_layer = hidden_states[-2]
        last_layer = hidden_states[-1]
        both_layers = torch.cat((second_last_layer, last_layer), dim=1)
        mean_tensor = self.meanpool(both_layers.permute(0, 2, 1))
        pooled_output = self.dense(mean_tensor.reshape((-1, 768)))
        pooled_output = self.activation(pooled_output)
        return pooled_output

