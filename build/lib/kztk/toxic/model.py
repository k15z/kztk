import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from nltk.tokenize import sent_tokenize

charmap = {c: i for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")}
input_size = len(charmap)
def char_vector(text):
    vec = torch.zeros(input_size, len(text))
    for i, c in enumerate(text):
        if c in charmap:
            vec[charmap[c],i] = 1.0
    return vec

SPACE = char_vector(" ")
def add_spaces(sentences):
    res = []
    for s in sentences:
        res.append(s)
        res.append(SPACE)
    return res[:-1] if len(res) > 1 else res

class ToxicModel(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(ToxicModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.conv_1 = nn.Conv1d(input_size, hidden_size, 3, padding=1)
        self.conv_2 = nn.Conv1d(input_size + hidden_size, hidden_size, 3, padding=1)
        self.conv_3 = nn.Conv1d(input_size + 2*hidden_size, hidden_size, 3, padding=1)
        self.conv_4 = nn.Conv1d(input_size + 3*hidden_size, hidden_size, 3, padding=1)
        self.norm_1 = nn.InstanceNorm1d(4*hidden_size, affine=True)
        self.linear_1 = nn.Linear(4*hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, self.output_size)

    def forward(self, text_vec):
        hidden = text_vec
        hidden = torch.cat([hidden, F.rrelu(self.conv_1(hidden))], dim=1)
        hidden = torch.cat([hidden, F.rrelu(self.conv_2(hidden))], dim=1)
        hidden = torch.cat([hidden, F.rrelu(self.conv_3(hidden))], dim=1)
        hidden = torch.cat([hidden, F.rrelu(self.conv_4(hidden))], dim=1)
        hidden = hidden[:,self.input_size:,:] # drop raw input
        hidden = self.norm_1(hidden) if hidden.size(2) > 1 else hidden
        hidden = torch.max(hidden, dim=2)[0]
        hidden = self.linear_2(F.rrelu(self.linear_1(hidden)))
        return hidden

class ToxicWrapper(object):

    def __init__(self, trnn):
        self.trnn = trnn

    def predict_one(self, x):
        x = torch.cat(add_spaces(x), dim=1) # join sentences
        x = autograd.Variable(x.unsqueeze(0))
        y = F.sigmoid(self.trnn(x)).data[0]
        result = {}
        for k, v in zip(["toxic", "severe", "obscene", "threat", "insult", "hate"], y):
            result[k] = v
        return result
