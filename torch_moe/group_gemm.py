import torch


def build_offsets(group_sizes: list[int], device: str) -> torch.Tensor:
    """
    Convert per-group sizes to cumulative int32 offsets expected by torch._grouped_mm.
    """
    return torch.cumsum(
        torch.tensor(group_sizes, device=device, dtype=torch.int32),
        dim=0,
        dtype=torch.int32,
    )


def grouped_mm_2d3d_example() -> None:
    """
    Standard MoE-style case:
      - x is 2D: [total_tokens, K]
      - w is 3D: [num_groups, K, N]
      - offs slices rows of x for each group
    """
    device = "cuda"
    dtype = torch.bfloat16

    # Three experts/groups; router has already assigned tokens to each group.
    group_sizes = [4, 2, 3]
    offs = build_offsets(group_sizes, device)

    total_tokens = int(offs[-1].item())
    num_groups = len(group_sizes)
    K = 16
    N = 8

    # x: routed token activations after dispatch, flattened as [total_tokens, hidden_size].
    x = torch.randn(total_tokens, K, device=device, dtype=dtype)
    # w: per-expert FFN projection weights selected by routing, one [K, N] matrix per expert.
    w = torch.randn(num_groups, K, N, device=device, dtype=dtype)

    # torch._grouped_mm returns [total_tokens, N] for 2d x 3d.
    y = torch._grouped_mm(x, w, offs=offs, out_dtype=dtype)

    # Simple correctness check against a Python reference loop.
    y_ref = torch.empty_like(y)
    start = 0
    for i, end in enumerate(offs.tolist()):
        if end > start:
            y_ref[start:end] = x[start:end] @ w[i]
        start = end
    torch.testing.assert_close(y, y_ref, atol=5e-2, rtol=5e-2)

    print("2d x 3d grouped_mm OK")
    print(f"x: {tuple(x.shape)}, w: {tuple(w.shape)}, offs: {offs.tolist()}, y: {tuple(y.shape)}")


if __name__ == "__main__":
    torch.manual_seed(0)
    grouped_mm_2d3d_example()

