import torch.nn as nn
import torch
import copy
import os
#from transformers import *
#from transformers import BertForMaskedLM
import logging
import math

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def swish(x):
    return x * torch.sigmoid(x)


def _gelu_python(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        This is now written in C in torch.nn.functional
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


if torch.__version__ < "1.4.0":
    gelu = _gelu_python
else:
    gelu = F.gelu


def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))

def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError("function {} not found in ACT2FN mapping {}".format(activation_string, list(ACT2FN.keys())))


ACT2FN = {
    "relu": F.relu,
    "swish": swish,
    "gelu": gelu,
    "tanh": torch.tanh,
    "gelu_new": gelu_new,
    "gelu_fast": gelu_fast,
}

BertLayerNorm = torch.nn.LayerNorm

class Last_layer(nn.Module):
    def __init__(self, config, o_dim):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, o_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(o_dim))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.decoder(hidden_states)

        return output

class Disc_only(nn.Module):
    def __init__(self, base_model_, config):
        super().__init__()
        self.base_model = base_model_

        self.config = config
        self.disc_head = Last_layer(config, 2)

        self.disc_criterion = nn.CrossEntropyLoss()
        #torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, input_ids, lm_label):

        replaced_mask = input_ids == lm_label

        outputs = self.base_model(input_ids, output_hidden_states=True)
        lm_outputs = self.disc_head(outputs[0])

        lm_label = torch.ones_like(lm_label).cuda()
        lm_label = lm_label.masked_fill(replaced_mask, 0)

        disc_loss = self.disc_criterion(lm_outputs.view(-1, 2), lm_label.view(-1))
        
        lm_score = torch.max(lm_outputs, dim = -1)[1]
        lm_score = len(lm_score[lm_score == lm_label]) / lm_label.shape[0] / lm_label.shape[1]

        return disc_loss, lm_score