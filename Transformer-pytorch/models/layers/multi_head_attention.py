import torch
from torch.nn import functional as F
from torch import nn
import math

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h # head

        assert d_model % h == 0, 'd_model is not divisible by h'

        self.d_k = d_model // h

        # Define the weight matrix
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)


    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):

        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_scores = F.softmax(attention_scores, dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores
    
    def forward(self, q, k, v, mask):

        query = self.w_q(q) # torch.Size([8, 350, 512])
        key = self.w_k(k) # torch.Size([8, 350, 512])
        value = self.w_v(v) # torch.Size([8, 350, 512])

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)

from torch.autograd import Variable

# d_model  = 512
# h = 8
# dropout = 0.1
# mha = MultiHeadAttentionBlock(d_model, h, dropout)

# q = Variable(torch.rand(1, 128, d_model))
# k = Variable(torch.rand(1, 128, d_model))
# v = Variable(torch.rand(1, 128, d_model))

# mask = None

# output = mha(q, k, v, mask)

# print(output.size())