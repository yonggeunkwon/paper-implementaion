from torch import nn
import math

class TokenEmbedding(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)


    def forward(self, x):
        print("forward_start")
        print("x : ", x)
        print(x.shape)
        return self.embedding(x) * math.sqrt(self.d_model)