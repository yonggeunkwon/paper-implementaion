import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model:int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Positional Encoding을 위한 matrix 만들기. (seq_len, d_model)shape의 0으로 찬 matrix임
        pe = torch.zeros(seq_len, d_model) # torch.Size([350, 512])

        # Creating a tensor representing positions
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1) # torch.Size([350, 1])

        # Create the division term for the positional encoding formula
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # torch.Size([256])

        # Apply sine to 짝수 차수 
        pe[:, 0::2] = torch.sin(position * div_term) # ([350, 256])

        # Apply cosine to 홀수 차수
        pe[:, 1::2] = torch.cos(position * div_term) # ([350, 256])

        # pe.shape = torch.Size([350, 512])
        # ------------------
        pe = pe.unsqueeze(0)  # torch.Size([1, 350, 512])

        self.register_buffer('pe', pe)

    
    def forward(self, x):
        print("--------------------Positional encoding start--------------------")
        print("x : ", x)
        print("x.shape : ", x.shape)
        print(self.pe[:, :x.shape[1], :].shape)
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

# seq_len = 128
# position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(-1)
# print(position)
# print(position.shape)
