import torch


# Check if CUDA is available
if torch.cuda.is_available():
    # Enable TF32 for matrix multiplications
    torch.backends.cuda.matmul.allow_tf32 = True

    # Create two tensors on the GPU
    a = torch.randn(1000, 1000, device='cuda')
    b = torch.randn(1000, 1000, device='cuda')

    # Perform matrix multiplication using TF32
    c = torch.matmul(a, b)

    print("TF32 is enabled for matrix multiplication ", c)

    torch.backends.cuda.matmul.allow_tf32 = False
    # Perform matrix multiplication using FP32
    c = torch.matmul(a, b)

    print("FP32 is enabled for matrix multiplication ", c)
else:
    print("CUDA is not available")
