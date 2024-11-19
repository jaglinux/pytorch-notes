import time
import torch


B, in_dim, out_dim = 32, 100, 10

def run(mod, l_fn):
    x = torch.randn(B, in_dim, device="cuda")
    y = torch.randn(B, out_dim, device="cuda")
    y_pred = mod(x)
    l = l_fn(y_pred, y)
    mod.zero_grad()
    l.backward()

if __name__=="__main__":
    t0 = time.perf_counter_ns()

    torch.profiler._utils._init_for_cuda_graphs()
    prof = torch.profiler.profile()
    model = torch.nn.Sequential(
        torch.nn.Linear(in_dim, out_dim),
        torch.nn.ReLU()
    ).to(device="cuda")
    
    loss_fn = torch.nn.MSELoss(size_average=False)
    print(f"Prep Time={(time.perf_counter_ns() - t0)/1000}us")
    t0 = time.perf_counter_ns()
    with prof:
        for _ in range(20):
            run(model, loss_fn)
    torch.cuda.synchronize("cuda")
    print(f"Bake Time={(time.perf_counter_ns() - t0)/1000}us")
    if hasattr(prof, "export_chrome_trace"):
        prof.export_chrome_trace(f"profiler.json")
