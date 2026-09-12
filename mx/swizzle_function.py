def gfx950_to_blocked_32_8(scale):
    rows, cols = scale.shape
    padded_rows = _ceil_div(rows, 32) * 32
    padded_cols = _ceil_div(cols, 8) * 8
    padded = scale
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (padded_rows, padded_cols), device=scale.device, dtype=scale.dtype
        )
        padded[:rows, :cols] = scale
    blocks = padded.view(padded_rows // 32, 2, 16, padded_cols // 8, 2, 4)
    return blocks.permute(0, 3, 5, 2, 4, 1).flatten()
