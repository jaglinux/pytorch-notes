import torch

batch_size = 32
M, K, N = 64, 128, 256

A = torch.randn(batch_size, M, K)
B = torch.randn(batch_size, K, N)
C = torch.bmm(A, B)

print(C.shape)  # torch.Size([32, 64, 256])
