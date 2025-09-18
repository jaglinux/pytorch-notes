import torch

M, N, K = 32 , 32 ,32
BLOCK_SIZE = 32
#torch.set_printoptions(threshold=M*N)

# largest power of 2 representable in `torch.float8_e4m3fn`
F8E4M3_LARGEST_POW2 = 8
# largest power of 2 representable in `torch.float4_e2m1fn_x2`
FP4E2M1FN_LARGEST_POW2 = 1.0
# max value of `torch.float8_e4m3fn` (448)
F8E4M3_MAX_VAL = torch.finfo(torch.float8_e4m3fn).max
# exponent bias of `torch.float8_e8m0fnu`
F8E8M0_EXP_BIAS = 127
# exponent and mantissa bits of `torch.float4_e2m1fn_x2`
FP4_EBITS, FP4_MBITS = 2, 1
FP4_MAX_VAL = 6.0

def data_to_mx_scale(x, block_size, recipe):
    # simple implementation of https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
    # section 6.3, not all edge cases (such as NaN) are handled/tested
    if recipe == "mxfp8":
        largest_pow2 = F8E4M3_LARGEST_POW2
    elif recipe == "mxfp4":
        largest_pow2 = FP4E2M1FN_LARGEST_POW2
    else:
        raise ValueError(f"data_to_mx_scale(): Unsupported mx recipe: {recipe}")
    orig_shape = x.shape
    x = x.reshape(-1, block_size)
    max_abs = torch.amax(torch.abs(x), 1)
    print("----x------ ", x)
    print("------max_abs------- ", max_abs, max_abs.shape)
    largest_p2_lt_max_abs = torch.floor(torch.log2(max_abs))
    print("------largest_p2_lt_max_abs------- ", largest_p2_lt_max_abs, largest_p2_lt_max_abs.shape)
    
    scale_e8m0_unbiased = largest_p2_lt_max_abs - largest_pow2
    print("------scale_e8m0_unbiased------- ", scale_e8m0_unbiased, scale_e8m0_unbiased.shape)
    scale_e8m0_unbiased = torch.clamp(scale_e8m0_unbiased, -1 * F8E8M0_EXP_BIAS, F8E8M0_EXP_BIAS)
    print("------scale_e8m0_unbiased clamp------- ", scale_e8m0_unbiased, scale_e8m0_unbiased.shape)
    scale_e8m0_biased = scale_e8m0_unbiased + F8E8M0_EXP_BIAS
    print("------scale_e8m0_biased------- ", scale_e8m0_biased, scale_e8m0_biased.shape)
    scale_e8m0_biased = scale_e8m0_biased.to(torch.uint8)
    print("------scale_e8m0_biased uint8------- ", scale_e8m0_biased, scale_e8m0_biased.shape)
    scale_e8m0_biased = scale_e8m0_biased.view(torch.float8_e8m0fnu)
    print("------scale_e8m0_biased view------- ", scale_e8m0_biased, scale_e8m0_biased.shape)
    a = scale_e8m0_biased.reshape(orig_shape[0], -1)
    print("------return value ------- ", a, a.shape)
    return a

def create_mx_fp8_data(m, n, k, device="cpu"):
    A_ref = torch.randn((m, k), device=device, dtype=torch.bfloat16) * 1000
    B_ref = torch.randn((n, k), device=device, dtype=torch.bfloat16) * 1000
    data_to_mx_scale(A_ref, BLOCK_SIZE, recipe="mxfp8")
    print(A_ref, A_ref.shape, A_ref.numel())
    print(B_ref, B_ref.shape, B_ref.numel())

create_mx_fp8_data(M, N, K)


'''
----x------  tensor([[ -194., -1008., -1800.,  ..., -1728.,  -462.,  -182.],
        [ 1456.,   652., -1024.,  ...,   127.,  -201., -2320.],
        [  388.,   113., -1736.,  ...,  -748.,   372., -1104.],
        ...,
        [ 1264.,  1808.,  -145.,  ...,  -716., -1896.,  -360.],
        [ 1408.,   138.,   660.,  ...,  -450., -1760.,     0.],
        [  300., -1008., -1360.,  ...,  -600.,   394.,  -676.]],
       dtype=torch.bfloat16)
------max_abs-------  tensor([1960., 2320., 2544., 2240., 2176., 1832., 1920., 2864., 1896., 2976.,
        2832., 2496., 2224., 1880., 2032., 1696., 2592., 1928., 2160., 2368.,
        1720., 2864., 2272., 2224., 1832., 2336., 2016., 2960., 2576., 2096.,
        2448., 2656.], dtype=torch.bfloat16) torch.Size([32])
------largest_p2_lt_max_abs-------  tensor([10., 11., 11., 11., 11., 10., 10., 11., 10., 11., 11., 11., 11., 10.,
        11., 10., 11., 10., 11., 11., 10., 11., 11., 11., 10., 11., 11., 11.,
        11., 11., 11., 11.], dtype=torch.bfloat16) torch.Size([32])
------scale_e8m0_unbiased-------  tensor([2., 3., 3., 3., 3., 2., 2., 3., 2., 3., 3., 3., 3., 2., 3., 2., 3., 2.,
        3., 3., 2., 3., 3., 3., 2., 3., 3., 3., 3., 3., 3., 3.],
       dtype=torch.bfloat16) torch.Size([32])
------scale_e8m0_unbiased clamp-------  tensor([2., 3., 3., 3., 3., 2., 2., 3., 2., 3., 3., 3., 3., 2., 3., 2., 3., 2.,
        3., 3., 2., 3., 3., 3., 2., 3., 3., 3., 3., 3., 3., 3.],
       dtype=torch.bfloat16) torch.Size([32])
------scale_e8m0_biased-------  tensor([129., 130., 130., 130., 130., 129., 129., 130., 129., 130., 130., 130.,
        130., 129., 130., 129., 130., 129., 130., 130., 129., 130., 130., 130.,
        129., 130., 130., 130., 130., 130., 130., 130.], dtype=torch.bfloat16) torch.Size([32])
------scale_e8m0_biased uint8-------  tensor([129, 130, 130, 130, 130, 129, 129, 130, 129, 130, 130, 130, 130, 129,
        130, 129, 130, 129, 130, 130, 129, 130, 130, 130, 129, 130, 130, 130,
        130, 130, 130, 130], dtype=torch.uint8) torch.Size([32])
------scale_e8m0_biased view-------  tensor([4., 8., 8., 8., 8., 4., 4., 8., 4., 8., 8., 8., 8., 4., 8., 4., 8., 4.,
        8., 8., 4., 8., 8., 8., 4., 8., 8., 8., 8., 8., 8., 8.],
       dtype=torch.float8_e8m0fnu) torch.Size([32])
------return value -------  tensor([[4.],
        [8.],
        [8.],
        [8.],
        [8.],
        [4.],
        [4.],
        [8.],
        [4.],
        [8.],
        [8.],
        [8.],
        [8.],
        [4.],
        [8.],
        [4.],
        [8.],
        [4.],
        [8.],
        [8.],
        [4.],
        [8.],
        [8.],
        [8.],
        [4.],
        [8.],
        [8.],
        [8.],
        [8.],
        [8.],
        [8.],
        [8.]], dtype=torch.float8_e8m0fnu) torch.Size([32, 1])
tensor([[ -194., -1008., -1800.,  ..., -1728.,  -462.,  -182.],
        [ 1456.,   652., -1024.,  ...,   127.,  -201., -2320.],
        [  388.,   113., -1736.,  ...,  -748.,   372., -1104.],
        ...,
        [ 1264.,  1808.,  -145.,  ...,  -716., -1896.,  -360.],
        [ 1408.,   138.,   660.,  ...,  -450., -1760.,     0.],
        [  300., -1008., -1360.,  ...,  -600.,   394.,  -676.]],
       dtype=torch.bfloat16) torch.Size([32, 32]) 1024
tensor([[ -238.0000,   432.0000,   185.0000,  ...,  -576.0000,  3104.0000,
          -216.0000],
        [ -528.0000,   372.0000,  1360.0000,  ...,  -776.0000,   740.0000,
          -820.0000],
        [  976.0000,   676.0000,   368.0000,  ...,   828.0000,  -548.0000,
           920.0000],
        ...,
        [  468.0000,  -444.0000,  -792.0000,  ...,  -119.5000,   233.0000,
           165.0000],
        [   29.1250,  -114.5000,  -976.0000,  ..., -1824.0000,    38.5000,
         -1080.0000],
        [-2336.0000,  -288.0000,  -888.0000,  ...,  -872.0000,    92.0000,
           676.0000]], dtype=torch.bfloat16) torch.Size([32, 32]) 1024
'''
    
