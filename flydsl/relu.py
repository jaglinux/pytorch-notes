import torch
import flydsl.compiler as flyc
import flydsl.expr as fx 
from flydsl.expr import gpu

@flyc.kernel
def relu_kernel(dram_in:fx.Tensor, dram_out:fx.Tensor, N: fx.Constexpr[int]):
    t_id = gpu.thread_idx.x
    b_id = gpu.block_idx.x
    block_dim = gpu.block_dim.x # same as BLOCK_SIZE

    global_thread_id = b_id * block_dim + t_id

    if global_thread_id < N:
        register_a = dram_in[global_thread_id]
        if register_a < 0:
            register_result = 0.0
        else:
            register_result = register_a
        dram_out[global_thread_id] = register_result


@flyc.jit
def launch_relu(x: fx.Tensor, y: fx.Tensor, N: fx.Constexpr[int]):
    BLOCK_SIZE = 256
    GRID_SIZE = (N+BLOCK_SIZE-1) // BLOCK_SIZE
    relu_kernel(x, y, N).launch(grid=(GRID_SIZE,1,1), block=(BLOCK_SIZE,1,1))

if __name__ == "__main__":
    N = 500
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    launch_relu(x, y, N)
    pt_ref = torch.relu(x)
    print(y)
