import torch

# Create two INT8 tensors
matrix1 = torch.randint(-10, 10, (17, 16), dtype=torch.int8, device='cuda')
matrix2 = torch.randint(-10, 10, (16, 16), dtype=torch.int8, device='cuda')
# Perform matrix multiplication
result = torch._int_mm(matrix1, matrix2)

# Print the result
print(result)
print(result.dtype)
