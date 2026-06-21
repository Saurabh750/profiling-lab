"""
Triton Kernel 03: Row-wise Reduction (Sum)
==========================================

Goal: given matrix A of shape (M, N), compute out[i] = sum(A[i, :]) for each row i.
Output shape: (M,)

WHAT'S NEW
-----------
So far every program wrote back a tile — the output was the same shape as the input.
Reductions are different: many inputs collapse to fewer outputs.

Here each program handles exactly ONE row and produces ONE scalar output.

NEW OPERATION: tl.sum(tensor, axis)
-------------------------------------
  a = tl.load(...)           # shape (BLOCK_N,)
  result = tl.sum(a, axis=0) # scalar — sum of all elements along axis 0

tl.sum works on a tile (a vector of BLOCK_N values) and reduces it to a single value.
This is a warp-level reduction under the hood.

THE INNER K-LOOP (key pattern for matmul)
------------------------------------------
When a row has N columns and N > BLOCK_N, one tile can't hold the whole row.
Solution: loop over the row in BLOCK_N-sized chunks and accumulate:

  acc = 0.0
  for k in range(0, tl.cdiv(N, BLOCK_N)):
      load chunk k of the row
      acc += tl.sum(chunk, axis=0)
  store acc

This is EXACTLY the K-loop in matmul, just with tl.sum instead of tl.dot.
In matmul: each program loops over K-tiles of A and B, accumulating into a
           (BLOCK_M x BLOCK_N) output tile.
Here:      each program loops over K-tiles of one row, accumulating into a scalar.

TWO KERNELS
------------
Kernel A: simple — assumes N <= BLOCK_N (one tile covers the whole row).
           Clean introduction to tl.sum.
Kernel B: general — loops over K tiles. Works for any N.
           The pattern you will use in matmul.
"""

import torch
import triton
import triton.language as tl


# ===========================================================================
# Kernel A: Simple — one tile per row, N must fit in BLOCK_N
# ===========================================================================

@triton.jit
def row_sum_kernel_simple(
    a_ptr, out_ptr,
    M, N,
    stride_am, stride_an,
    BLOCK_N: tl.constexpr,   # must be >= N and a power of 2
):
    pid_m = tl.program_id(axis=0)   # one program per row

    col_offsets = tl.arange(0, BLOCK_N)            # [0, 1, ..., BLOCK_N-1]
    mask = col_offsets < N                          # guard for actual N < BLOCK_N

    a_ptrs = a_ptr + pid_m * stride_am + col_offsets * stride_an
    a = tl.load(a_ptrs, mask=mask, other=0.0)      # shape (BLOCK_N,)

    # Reduce the tile to a scalar
    result = tl.sum(a, axis=0)                     # scalar

    tl.store(out_ptr + pid_m, result)


def row_sum_simple(a: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and a.ndim == 2
    M, N = a.shape
    out = torch.empty(M, device=a.device, dtype=a.dtype)

    # BLOCK_N must be >= N and a power of 2
    BLOCK_N = triton.next_power_of_2(N)

    grid = (M,)   # one program per row
    row_sum_kernel_simple[grid](
        a, out, M, N,
        a.stride(0), a.stride(1),
        BLOCK_N=BLOCK_N,
    )
    return out


# ===========================================================================
# Kernel B: General — inner K-loop, works for any N
# ===========================================================================

@triton.jit
def row_sum_kernel_general(
    a_ptr, out_ptr,
    M, N,
    stride_am, stride_an,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)   # one program per row

    # Accumulator starts at zero. Each K-loop iteration adds to it.
    # Use a Python float (scalar), NOT tl.zeros((1,), ...) which creates a rank-1
    # block — you can't tl.store a block into a scalar pointer.
    acc = 0.0

    # Loop over the row in BLOCK_N-sized chunks
    # tl.cdiv(N, BLOCK_N) = ceil(N / BLOCK_N) = number of tiles in this row
    for k in range(0, tl.cdiv(N, BLOCK_N)):
        col_offsets = k * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = col_offsets < N

        a_ptrs = a_ptr + pid_m * stride_am + col_offsets * stride_an
        a = tl.load(a_ptrs, mask=mask, other=0.0)  # shape (BLOCK_N,)

        # Reduce this chunk and add to accumulator
        acc += tl.sum(a, axis=0)

    tl.store(out_ptr + pid_m, acc)


def row_sum_general(a: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and a.ndim == 2
    M, N = a.shape
    out = torch.empty(M, device=a.device, dtype=torch.float32)

    BLOCK_N = 1024  # tile size for the K-loop

    grid = (M,)
    row_sum_kernel_general[grid](
        a, out, M, N,
        a.stride(0), a.stride(1),
        BLOCK_N=BLOCK_N,
    )
    return out


# ===========================================================================
# Tests and benchmark
# ===========================================================================

def test_correctness():
    torch.manual_seed(0)

    # Test simple kernel: N must be small enough to fit in one BLOCK_N tile
    for M, N in [(128, 64), (256, 512), (100, 300)]:
        A = torch.randn(M, N, device="cuda", dtype=torch.float32)
        ref = A.sum(dim=1)
        out = row_sum_simple(A)
        assert torch.allclose(out, ref, atol=1e-4), f"Simple MISMATCH at ({M},{N})"
        print(f"[PASS] simple  ({M:4d} x {N:4d})")

    # Test general kernel: works for any N, including large and non-power-of-2
    for M, N in [(128, 64), (256, 1000), (512, 8192), (100, 7777)]:
        A = torch.randn(M, N, device="cuda", dtype=torch.float32)
        ref = A.sum(dim=1)
        out = row_sum_general(A)
        assert torch.allclose(out, ref, atol=1e-4), f"General MISMATCH at ({M},{N})"
        print(f"[PASS] general ({M:4d} x {N:4d})")


def benchmark():
    print("\nBenchmark: Triton row sum vs PyTorch .sum(dim=1)")
    print(f"{'Shape':>14}  {'Torch (ms)':>12}  {'Triton (ms)':>12}  {'Speedup':>8}")
    print("-" * 54)
    for M, N in [(1024, 1024), (2048, 2048), (4096, 4096), (1024, 16384)]:
        A = torch.randn(M, N, device="cuda", dtype=torch.float32)
        ms_torch  = triton.testing.do_bench(lambda: A.sum(dim=1))
        ms_triton = triton.testing.do_bench(lambda: row_sum_general(A))
        print(f"{f'{M}x{N}':>14}  {ms_torch:>12.4f}  {ms_triton:>12.4f}  {ms_torch/ms_triton:>8.2f}x")


if __name__ == "__main__":
    test_correctness()
    benchmark()
