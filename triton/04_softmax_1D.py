import torch
import triton
import triton.language as tl

# Softmax Kernel for 1D vector

@triton.jit
def softmax_1D(
    a_ptr, out_ptr,
    M,
    BLOCK_M: tl.constexpr,
):
    # Pass 1 - find the row max:
    max_val = -float('inf')
    exp_sum = 0.0

    for k in range(0, tl.cdiv(M, BLOCK_M)):
        col_offsets = k * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = col_offsets < M

        a_ptrs = a_ptr + col_offsets
        a = tl.load(a_ptrs, mask=mask, other=-float('inf'))

        max_val = tl.maximum(max_val, tl.max(a, axis=0))

    # Pass 2 - compute exp(x - max) and sum:

    for k in range(0, tl.cdiv(M, BLOCK_M)):
        col_offsets = k * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = col_offsets < M

        a_ptrs = a_ptr + col_offsets
        a = tl.load(a_ptrs, mask=mask, other=-float('inf'))

        exp_sum += tl.sum(tl.exp(a - max_val), axis=0)

    # Pass 3 - Normalize and write back

    for k in range(0, tl.cdiv(M, BLOCK_M)):
        col_offsets = k * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = col_offsets < M

        a_ptrs = a_ptr + col_offsets
        a = tl.load(a_ptrs, mask=mask, other=0.0)

        tl.store(out_ptr + col_offsets, tl.exp(a - max_val) / exp_sum, mask=mask)


def softmax_1d(a: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and a.ndim == 1
    M = a.shape[0]
    out = torch.empty_like(a)
    BLOCK_M = 1024
    softmax_1D[(1,)](a, out, M, BLOCK_M=BLOCK_M)
    return out


def test_correctness():
    torch.manual_seed(0)

    for N in [128, 1000, 1024, 7777]:
        a = torch.randn(N, device="cuda", dtype=torch.float32)
        ref = torch.softmax(a, dim=0)
        out = softmax_1d(a)
        assert torch.allclose(out, ref, atol=1e-5), f"MISMATCH at N={N}"
        print(f"[PASS] N={N}")


if __name__ == "__main__":
    test_correctness()
