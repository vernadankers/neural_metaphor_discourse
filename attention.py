"""Attention modules for the metaphor & discourse project.

- GeneralAttention
- HierarchicalAttention
"""
import torch
import torch.nn as nn
from allennlp.nn.util import masked_softmax


class GeneralAttention(nn.Module):
    def __init__(self, hidden, sparsemax=False):
        super().__init__()
        self.linear = nn.Linear(hidden, 1)
        self.normaliser = masked_softmax
        self.weights = []

    def forward(self, _, context, masks):
        context = torch.cat(context, dim=1)
        weights = self.linear(context).squeeze(-1)
        weights = self.normaliser(weights, torch.cat(masks, dim=1).cuda())
        context = torch.bmm(weights.unsqueeze(dim=1), context)

        if not self.training:
            self.weights = []
            for w, m in zip(weights, torch.cat(masks, dim=1)):
                w = w.cpu().tolist()
                m = m.cpu().tolist()
                self.weights.append(" ".join([str(round(x, 3)) \
                                    for x, y in zip(w, m) if y != 0]))
        return context


class HierarchicalAttention(nn.Module):
    def __init__(self, hidden, normaliser="softmax", bert=False):
        super().__init__()
        self.linear1 = nn.Linear(hidden, 1)
        self.linear2 = nn.Linear(hidden, 1)
        self.bert = bert
        if not bert:
            self.lstm = nn.LSTM(input_size=hidden, hidden_size=int(hidden / 2),
                                batch_first=True, bidirectional=True)
        else:
            self.transformer = torch.nn.TransformerEncoderLayer(
                768, 8, dim_feedforward=2048, dropout=0.1, activation='relu'
            )
        self.normaliser = normaliser
        self.weights = []

    def normalise(self, inp, mask=None):
        if mask is not None:
            return masked_softmax(inp, mask)
        return torch.softmax(inp, dim=-1)

    def forward(self, focus, context, context_masks):

        if not self.training:
            self.weights = [[] for _ in context]

        context_output = []
        for i, (sentence, masks) in enumerate(zip(context, context_masks)):
            weights = self.linear1(sentence).squeeze(-1)
            weights = self.normalise(weights, masks.cuda())

            if not self.training:
                for w, m in zip(weights, masks):
                    w = w.cpu().tolist()
                    m = m.cpu().tolist()
                    self.weights[i].append(" ".join([str(round(x, 3)) \
                                           for x, y in zip(w, m) if y != 0]))
            sentence = torch.bmm(weights.unsqueeze(dim=1), sentence)
            context_output.append(sentence)

        if not self.training:
            self.weights = [" ".join(list(x)) for x in zip(*self.weights)]

        if self.bert:
            context = self.transformer(torch.cat(context_output, dim=1))
        else:
            context, _ = self.lstm(torch.cat(context_output, dim=1))
        weights = self.linear2(context).squeeze(-1)
        weights = self.normalise(weights)
        if not self.training:
            for i, w in enumerate(weights):
                w = w.cpu().tolist()
                self.weights[i] = self.weights[i] + " " + \
                    " ".join([str(round(x, 3)) for x in w])

        context = torch.bmm(weights.unsqueeze(dim=1), context)
        return context
