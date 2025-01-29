import torch
import time
import datetime

from torch.profiler import profile, ProfilerActivity

# use HIPBLASLT_ALLOW_TF32=1
# Check if CUDA is available
if torch.cuda.is_available():
    # Enable TF32 for matrix multiplications
    torch.backends.cuda.matmul.allow_tf32 = True

    # Create two tensors on the GPU
    a = torch.randn(1000, 1000, device='cuda')
    b = torch.randn(1000, 1000, device='cuda')

    # Perform matrix multiplication using TF32
    start_time = time.time()
    c = torch.matmul(a, b)
    end_time = time.time()

    print("TF32 is enabled for matrix multiplication ", c)
    print(f"time taken is  {(end_time-start_time) * 1000} ms")

    #torch.backends.cuda.matmul.allow_tf32 = False
    # Perform matrix multiplication using FP32
    #c = torch.matmul(a, b)

    #print("FP32 is enabled for matrix multiplication ", c)
    # Profiling
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d_%H-%M-%S")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_stack=True,
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            c = torch.matmul(a, b)
        s.synchronize()
        torch.cuda.current_stream().wait_stream(s)
    prof.export_chrome_trace(f'matmul_{now}.json')
else:
    print("CUDA is not available")
