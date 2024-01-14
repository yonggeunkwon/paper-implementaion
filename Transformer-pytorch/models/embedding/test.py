import torch
from torch.nn import functional as F

input = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]])
embedding_matrix = torch.rand(10, 3)

print(input.shape)

print(F.embedding(input, embedding_matrix))
print(F.embedding(input, embedding_matrix).shape)