#!/usr/bin/env python3
"""Call FP8 tensorwise and MX FP8 scaled_mm (v1, v2 NO_SWIZZLE, v2 SWIZZLE_32_8)."""

import argparse
import sys
import time

import torch
from torch.nn.functional import ScalingType, SwizzleType, scaled_mm
from triton.testing import do_bench


def _to_mxfp(data_hp, block_size=32):
    orig_shape = data_hp.shape
    data_hp = data_hp.reshape(*orig_shape[:-1], orig_shape[-1] // block_size, block_size)
    max_abs = torch.amax(torch.abs(data_hp), -1).unsqueeze(-1)
    data_hp = data_hp.to(torch.float32)
    max_abs = max_abs.to(torch.float32)
    max_pos = torch.finfo(torch.float8_e4m3fn).max
    e8m0_bias = 127
    descale = max_abs / max_pos
    exponent = torch.where(
        torch.isnan(descale),
        0xFF,
        (
            torch.clamp(
                torch.ceil(torch.log2(descale)),
                min=-e8m0_bias,
                max=e8m0_bias,
            )
            + e8m0_bias
        ).to(torch.uint8),
    )
    descale_fp = torch.where(
        exponent == 0,
        1.0,
        torch.exp2(e8m0_bias - exponent.to(torch.float32)),
    )
    data_lp = torch.clamp(data_hp * descale_fp, min=-max_pos, max=max_pos)
    data_lp = data_lp.to(torch.float8_e4m3fn).reshape(orig_shape)
    scale = exponent.view(torch.float8_e8m0fnu).squeeze(-1)
    return scale, data_lp


def _dequant_mxfp8(data_lp, scale, block_size=32):
    return data_lp.float() * scale.float().repeat_interleave(block_size, dim=1)


def _ceil_div(x, y):
    return (x + y - 1) // y


def _to_blocked_32_8(scale):
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


def _tflops(m, n, k, latency_ms):
    return 2 * m * n * k / (latency_ms * 1e-3) * 1e-12


def _bench(fn, m, n, k):
    time.sleep(0.5)
    latency_ms = do_bench(fn, warmup=100, rep=500)
    return latency_ms, _tflops(m, n, k, latency_ms)


def _summarize(name, out, ref, m, n, k, latency_ms, tflops):
    out_f = out.float().flatten()
    ref_f = ref.float().flatten()
    diff = (out.float() - ref.float()).abs().mean().item()
    print()
    print(f"=== {name} ===")
    print(f"shape:  {tuple(out.shape)}")
    print(f"dtype:  {out.dtype}")
    print(f"out[:4]: {out_f[:4].tolist()}")
    print(f"ref[:4]: {ref_f[:4].tolist()}")
    print(f"diff:   {diff}")
    print(f"time:   {latency_ms:.3f} ms")
    print(f"tflops: {tflops:.2f}")
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-1, rtol=1e-1)


def call_fp8(a_hp, b_hp, m, k, n):
    a = a_hp.to(torch.float8_e4m3fn)
    b = b_hp.to(torch.float8_e4m3fn).t()
    scale_a = torch.tensor(1.0, device=a.device)
    scale_b = torch.tensor(1.0, device=a.device)

    def gemm():
        return torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)

    out = gemm()
    latency_ms, tflops = _bench(gemm, m, n, k)
    ref = (a.float() @ b.float()) * scale_a * scale_b
    return out, ref, latency_ms, tflops


def call_mxfp8_v1(a, b, scale_a, scale_b, m, k, n):
    # v1 has no swizzle arg; ROCm takes flattened unswizzled e8m0 scales.
    scale_a_flat = scale_a.flatten()
    scale_b_flat = scale_b.flatten()
    b_col = b.t()

    def gemm():
        return torch._scaled_mm(
            a, b_col, scale_a=scale_a_flat, scale_b=scale_b_flat, out_dtype=torch.bfloat16
        )

    out = gemm()
    latency_ms, tflops = _bench(gemm, m, n, k)
    ref = _dequant_mxfp8(a, scale_a) @ _dequant_mxfp8(b, scale_b).t()
    return out, ref, latency_ms, tflops


def call_mxfp8_v2_no_swizzle(a, b, scale_a, scale_b, m, k, n):
    b_col = b.t()

    def gemm():
        return scaled_mm(
            a,
            b_col,
            scale_a,
            ScalingType.BlockWise1x32,
            scale_b,
            ScalingType.BlockWise1x32,
            swizzle_a=SwizzleType.NO_SWIZZLE,
            swizzle_b=SwizzleType.NO_SWIZZLE,
            output_dtype=torch.bfloat16,
        )

    out = gemm()
    latency_ms, tflops = _bench(gemm, m, n, k)
    ref = _dequant_mxfp8(a, scale_a) @ _dequant_mxfp8(b, scale_b).t()
    return out, ref, latency_ms, tflops


def call_mxfp8_v2_swizzle_32_8(a, b, scale_a, scale_b, m, k, n):
    scale_a_sw = _to_blocked_32_8(scale_a)
    scale_b_sw = _to_blocked_32_8(scale_b)
    b_col = b.t()

    def gemm():
        return scaled_mm(
            a,
            b_col,
            scale_a_sw,
            ScalingType.BlockWise1x32,
            scale_b_sw,
            ScalingType.BlockWise1x32,
            swizzle_a=SwizzleType.SWIZZLE_32_8,
            swizzle_b=SwizzleType.SWIZZLE_32_8,
            output_dtype=torch.bfloat16,
        )

    out = gemm()
    latency_ms, tflops = _bench(gemm, m, n, k)
    ref = _dequant_mxfp8(a, scale_a) @ _dequant_mxfp8(b, scale_b).t()
    return out, ref, latency_ms, tflops


def _parse_dims(dims):
    if len(dims) == 1:
        m = k = n = dims[0]
    elif len(dims) == 3:
        m, k, n = dims
    else:
        raise SystemExit("pass 1 size (m=k=n) or 3 sizes (m k n)")
    return m, k, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dims", nargs="+", type=int, help="size, or m k n")
    args = parser.parse_args()
    m, k, n = _parse_dims(args.dims)

    if not torch.cuda.is_available():
        print("CUDA/ROCm not available", file=sys.stderr)
        return 1

    device = "cuda"
    a_hp = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    b_hp = torch.randn((n, k), device=device, dtype=torch.bfloat16)

    print(f"m={m}")
    print(f"k={k}")
    print(f"n={n}")

    out, ref, latency_ms, tflops = call_fp8(a_hp, b_hp, m, k, n)
    _summarize("fp8", out, ref, m, n, k, latency_ms, tflops)

    scale_a, a_mx = _to_mxfp(a_hp.contiguous())
    scale_b, b_mx = _to_mxfp(b_hp.contiguous())

    out, ref, latency_ms, tflops = call_mxfp8_v1(a_mx, b_mx, scale_a, scale_b, m, k, n)
    _summarize("mxfp8 v1", out, ref, m, n, k, latency_ms, tflops)

    out, ref, latency_ms, tflops = call_mxfp8_v2_no_swizzle(
        a_mx, b_mx, scale_a, scale_b, m, k, n
    )
    _summarize("mxfp8 v2 no swizzle", out, ref, m, n, k, latency_ms, tflops)

    out, ref, latency_ms, tflops = call_mxfp8_v2_swizzle_32_8(
        a_mx, b_mx, scale_a, scale_b, m, k, n
    )
    _summarize("mxfp8 v2 SWIZZLE_32_8", out, ref, m, n, k, latency_ms, tflops)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

'''
python run_scaled_mm.py 8192

m=8192
k=8192
n=8192

=== fp8 ===
shape:  (8192, 8192)
dtype:  torch.bfloat16
out[:4]: [-43.0, 58.25, 64.5, 96.0]
ref[:4]: [-42.93224334716797, 58.14308166503906, 64.42999267578125, 96.16051483154297]
diff:   0.10160034149885178
time:   0.420 ms
tflops: 2620.75

=== mxfp8 v1 ===
shape:  (8192, 8192)
dtype:  torch.bfloat16
out[:4]: [-42.75, 58.0, 64.0, 96.0]
ref[:4]: [-42.94017791748047, 58.150978088378906, 64.42330932617188, 96.14852905273438]
diff:   0.2027888000011444
time:   0.627 ms
tflops: 1753.18

=== mxfp8 v2 no swizzle ===
shape:  (8192, 8192)
dtype:  torch.bfloat16
out[:4]: [-42.75, 58.0, 64.0, 96.0]
ref[:4]: [-42.94017791748047, 58.150978088378906, 64.42330932617188, 96.14852905273438]
diff:   0.2027888000011444
time:   0.627 ms
tflops: 1754.71

=== mxfp8 v2 SWIZZLE_32_8 ===
shape:  (8192, 8192)
dtype:  torch.bfloat16
out[:4]: [-43.0, 58.25, 64.5, 96.0]
ref[:4]: [-42.94017791748047, 58.150978088378906, 64.42330932617188, 96.14852905273438]
diff:   0.10160454362630844
time:   0.586 ms
tflops: 1876.62

'''
