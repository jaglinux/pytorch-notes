import torch

batch_size = 2
M, K, N = 5, 2, 5

A = torch.randn(batch_size, M, K)
B = torch.randn(batch_size, K, N)
C = torch.bmm(A, B)

print(C.shape)  # Should print torch.Size([32, 64, 256])
print("Input Tensor A ", A)
print("Input Tensor B ", B)
print("Output tensor bmm  ", C)

result = []
def sequence_mm(batch_size, A, B):
    for i in range(batch_size):
        temp = torch.mm(A[i], B[i])
        result.append(temp)

sequence_mm(batch_size, A, B)
print("Output tensor manual ", result)

result = torch.stack(result)
print("Tensor equal results ", torch.equal(C, result))
