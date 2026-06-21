import torch
import triton
import triton.language as tl

# Softmax Kernel for 2D matrix

@triton.jit
def softmax_2D(
    a_ptr, out_ptr,
    M, N,
    stride_am, stride_an,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(axis=0)

    # Pass 1: Find row max value
    max_val_temp = -float('inf')

    for k in range(0, tl.cdiv(N, BLOCK_N)):
        col_offsets = k * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = col_offsets < N

        a_ptrs = a_ptr + pid_m * stride_am + col_offsets * stride_an
        a = tl.load(a_ptrs, mask=mask, other=-float('inf'))

        max_val_temp = tl.maximum(max_val_temp, tl.max(a, axis=0))

    # Pass 2: compute exp(x - max) and sum:
    exp_sum_temp = 0.0
    
    for k in range(0, tl.cdiv(N, BLOCK_N)):
        col_offsets = k * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = col_offsets < N

        a_ptrs = a_ptr + pid_m * stride_am + col_offsets * stride_an
        a = tl.load(a_ptrs, mask=mask, other=-float('inf'))

        exp_sum_temp += tl.sum(tl.exp(a - max_val_temp), axis=0)

    # Pass 3: Normalize and write back

    for k in range(0, tl.cdiv(N, BLOCK_N)):
        col_offsets = k * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = col_offsets < N

        a_ptrs = a_ptr + pid_m * stride_am + col_offsets * stride_an
        a = tl.load(a_ptrs, mask=mask, other = 0.0)

        tl.store(out_ptr + pid_m * stride_am + col_offsets * stride_an, tl.exp(a - max_val_temp) / exp_sum_temp, mask=mask)


def softmax_2d(a: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and a.ndim == 2
    M, N = a.shape
    out = torch.empty_like(a)
    BLOCK_N = 1024
    grid = (M,)
    softmax_2D[grid](
        a, out,
        M, N,
        a.stride(0), a.stride(1),
        BLOCK_M=1, BLOCK_N=BLOCK_N,
    )
    return out


def test_correctness():
    torch.manual_seed(0)
    for M, N in [(4, 128), (128, 512), (100, 777), (256, 1024)]:
        a = torch.randn(M, N, device="cuda", dtype=torch.float32)
        ref = torch.softmax(a, dim=1)
        out = softmax_2d(a)
        assert torch.allclose(out, ref, atol=1e-5), f"MISMATCH at ({M}, {N})"
        print(f"[PASS] ({M} x {N})")


if __name__ == "__main__":
    test_correctness()