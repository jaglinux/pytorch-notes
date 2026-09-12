#!/usr/bin/env python3
"""Call MX FP4 scaled_mm (v1, v2 NO_SWIZZLE, v2 SWIZZLE_32_8) vs bf16.

There is no plain FP4 GEMM. Quantization matches
test/test_scaled_matmul_cuda.py MX FP4 (block size 32, e8m0 scales).
"""

import argparse
import sys
import time

import torch
from torch.nn.functional import ScalingType, SwizzleType, scaled_mm
from torch.testing._internal.common_quantized import _bfloat16_to_float4_e2m1fn_x2
from triton.testing import do_bench

_FP4_MAX = 6.0
_SQNR_MIN = 12.0


def _to_mxfp4(data_hp, block_size=32):
    orig_shape = data_hp.shape
    data_hp = data_hp.reshape(*orig_shape[:-1], orig_shape[-1] // block_size, block_size)
    max_abs = torch.amax(torch.abs(data_hp), -1).unsqueeze(-1)
    data_hp = data_hp.to(torch.float32)
    max_abs = max_abs.to(torch.float32)
    e8m0_bias = 127
    descale = max_abs / _FP4_MAX
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
    data_lp = torch.clamp(data_hp * descale_fp, min=-_FP4_MAX, max=_FP4_MAX)
    data_lp = data_lp.reshape(orig_shape)
    data_pack = _bfloat16_to_float4_e2m1fn_x2(data_lp.to(torch.bfloat16))
    scale = exponent.view(torch.float8_e8m0fnu).squeeze(-1)
    return scale, data_pack, data_lp


def _dequant_mxfp4(data_lp, scale, block_size=32):
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


def _sqnr(ref, out):
    ps = torch.norm(ref.float())
    pn = torch.norm(ref.float() - out.float())
    return (20 * torch.log10(ps / pn)).item()


def _summarize(name, out, ref, latency_ms, tflops, bf16_ms=None):
    out_f = out.float().flatten()
    ref_f = ref.float().flatten()
    diff = (out.float() - ref.float()).abs().mean().item()
    sqnr = _sqnr(ref, out)
    print()
    print(f"=== {name} ===")
    print(f"shape:  {tuple(out.shape)}")
    print(f"dtype:  {out.dtype}")
    print(f"out[:4]: {out_f[:4].tolist()}")
    print(f"ref[:4]: {ref_f[:4].tolist()}")
    print(f"diff:   {diff}")
    print(f"sqnr:   {sqnr:.2f} dB")
    print(f"time:   {latency_ms:.3f} ms")
    print(f"tflops: {tflops:.2f}")
    speedup = 1.0 if bf16_ms is None else bf16_ms / latency_ms
    print(f"speedup vs bf16: {speedup:.2f}x")
    if name != "bf16" and sqnr < _SQNR_MIN:
        raise AssertionError(f"{name} sqnr {sqnr:.2f} dB < {_SQNR_MIN}")


def call_bf16(a_hp, b_hp, m, k, n):
    b_col = b_hp.t()

    def gemm():
        return a_hp @ b_col

    out = gemm()
    latency_ms, tflops = _bench(gemm, m, n, k)
    return out, out, latency_ms, tflops


def call_mxfp4_v1(a, b, scale_a, scale_b, a_lp, b_lp, m, k, n):
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
    ref = _dequant_mxfp4(a_lp, scale_a) @ _dequant_mxfp4(b_lp, scale_b).t()
    return out, ref, latency_ms, tflops


def call_mxfp4_v2_no_swizzle(a, b, scale_a, scale_b, a_lp, b_lp, m, k, n):
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
    ref = _dequant_mxfp4(a_lp, scale_a) @ _dequant_mxfp4(b_lp, scale_b).t()
    return out, ref, latency_ms, tflops


def call_mxfp4_v2_swizzle_32_8(a, b, scale_a, scale_b, a_lp, b_lp, m, k, n):
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
    ref = _dequant_mxfp4(a_lp, scale_a) @ _dequant_mxfp4(b_lp, scale_b).t()
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
    if k % 32 != 0:
        raise SystemExit("K must be divisible by 32 for MX FP4")

    device = "cuda"
    a_hp = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    b_hp = torch.randn((n, k), device=device, dtype=torch.bfloat16)

    print(f"m={m}")
    print(f"k={k}")
    print(f"n={n}")

    out, ref, bf16_ms, tflops = call_bf16(a_hp, b_hp, m, k, n)
    _summarize("bf16", out, ref, bf16_ms, tflops)

    scale_a, a_mx, a_lp = _to_mxfp4(a_hp.contiguous())
    scale_b, b_mx, b_lp = _to_mxfp4(b_hp.contiguous())
    mx_ref = _dequant_mxfp4(a_lp, scale_a) @ _dequant_mxfp4(b_lp, scale_b).t()

    out, _, latency_ms, tflops = call_mxfp4_v1(
        a_mx, b_mx, scale_a, scale_b, a_lp, b_lp, m, k, n
    )
    _summarize("mxfp4 v1", out, mx_ref, latency_ms, tflops, bf16_ms)

    out, _, latency_ms, tflops = call_mxfp4_v2_no_swizzle(
        a_mx, b_mx, scale_a, scale_b, a_lp, b_lp, m, k, n
    )
    _summarize("mxfp4 v2 no swizzle", out, mx_ref, latency_ms, tflops, bf16_ms)

    out, _, latency_ms, tflops = call_mxfp4_v2_swizzle_32_8(
        a_mx, b_mx, scale_a, scale_b, a_lp, b_lp, m, k, n
    )
    _summarize("mxfp4 v2 SWIZZLE_32_8", out, mx_ref, latency_ms, tflops, bf16_ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


'''
m=8192
k=8192
n=8192

=== bf16 ===
shape:  (8192, 8192)
dtype:  torch.bfloat16
out[:4]: [-44.25, -35.5, 62.0, -92.5]
ref[:4]: [-44.25, -35.5, 62.0, -92.5]
diff:   0.0
sqnr:   inf dB
time:   0.856 ms
tflops: 1285.08
speedup vs bf16: 1.00x

=== mxfp4 v1 ===
shape:  (8192, 8192)
dtype:  torch.bfloat16
out[:4]: [-37.25, -33.0, 35.25, -86.5]
ref[:4]: [-44.33106994628906, -35.513607025146484, 61.927459716796875, -92.72473907470703]
diff:   11.734745979309082
sqnr:   15.78 dB
time:   0.398 ms
tflops: 2763.29
speedup vs bf16: 2.15x

=== mxfp4 v2 no swizzle ===
shape:  (8192, 8192)
dtype:  torch.bfloat16
out[:4]: [-37.25, -33.0, 35.25, -86.5]
ref[:4]: [-44.33106994628906, -35.513607025146484, 61.927459716796875, -92.72473907470703]
diff:   11.734745979309082
sqnr:   15.78 dB
time:   0.398 ms
tflops: 2760.15
speedup vs bf16: 2.15x

=== mxfp4 v2 SWIZZLE_32_8 ===
shape:  (8192, 8192)
dtype:  torch.bfloat16
out[:4]: [-37.25, -33.25, 35.5, -87.0]
ref[:4]: [-44.33106994628906, -35.513607025146484, 61.927459716796875, -92.72473907470703]
diff:   11.73892593383789
sqnr:   15.78 dB
time:   0.269 ms
tflops: 4085.77
speedup vs bf16: 3.18x

'''
