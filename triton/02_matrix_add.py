"""
Triton Kernel 02: 2D Matrix Addition
======================================

Goal: add two 2D matrices elementwise: C = A + B, where A, B, C are (M, N)

WHAT'S NEW VS 1D
-----------------
In 1D we had a 1D grid of programs, each covering a 1D tile.
In 2D we have a grid of programs, each covering a 2D tile (BLOCK_M x BLOCK_N).

Grid layout (for M=8, N=8, BLOCK_M=4, BLOCK_N=4):

  col_blocks ->  0         1
               +--------+--------+
  row_block 0  | pid(0,0)| pid(0,1)|
               +--------+--------+
  row_block 1  | pid(1,0)| pid(1,1)|
               +--------+--------+

Each program covers a (BLOCK_M x BLOCK_N) tile of the output matrix.

2D POINTER ARITHMETIC (shared by both kernels below)
------------------------------------------------------
A matrix stored in row-major order in memory:
  A[r, c] lives at: a_ptr + r * stride_row + c * stride_col

For a contiguous (M, N) tensor: stride_row = N, stride_col = 1.

To load a (BLOCK_M x BLOCK_N) tile we use broadcasting:
  row_offsets[:, None]  shape (BLOCK_M, 1)
  col_offsets[None, :]  shape (1, BLOCK_N)
  -> broadcast to (BLOCK_M, BLOCK_N) pointer block

TWO APPROACHES TO THE GRID
----------------------------
Kernel A — 1D flat grid + manual decode:
  grid = (num_row_blocks * num_col_blocks,)
  pid = tl.program_id(0)
  pid_m = pid // num_col_blocks
  pid_n = pid %  num_col_blocks

  Advantage: full control over which (pid_m, pid_n) a flat pid maps to.
  This is needed in matmul (Step 5) for L2 cache swizzling — reordering
  the mapping is how you improve cache hit rate across programs.

Kernel B — 2D grid (simpler, more intuitive):
  grid = (num_row_blocks, num_col_blocks)
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)

  Advantage: directly readable. No decoding needed.
  Limitation: no easy way to swizzle — GPU decides execution order.

For this kernel both produce identical results. Learn B first to build
intuition, then understand A as preparation for matmul.
"""

import torch
import triton
import triton.language as tl


# ===========================================================================
# Kernel A: 1D flat grid with manual (pid_m, pid_n) decode
# ===========================================================================

@triton.jit
def matrix_add_kernel_1d_grid(
    a_ptr, b_ptr, c_ptr,
    M, N,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_col_blocks = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_col_blocks
    pid_n = pid %  num_col_blocks

    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # (BLOCK_M,)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # (BLOCK_N,)

    a_ptrs = a_ptr + row_offsets[:, None] * stride_am + col_offsets[None, :] * stride_an
    b_ptrs = b_ptr + row_offsets[:, None] * stride_bm + col_offsets[None, :] * stride_bn
    c_ptrs = c_ptr + row_offsets[:, None] * stride_cm + col_offsets[None, :] * stride_cn

    mask = (row_offsets[:, None] < M) & (col_offsets[None, :] < N)

    a = tl.load(a_ptrs, mask=mask, other=0.0)
    b = tl.load(b_ptrs, mask=mask, other=0.0)
    tl.store(c_ptrs, a + b, mask=mask)


def matrix_add_1d_grid(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape and a.is_cuda and a.ndim == 2
    M, N = a.shape
    c = torch.empty_like(a)
    BLOCK_M, BLOCK_N = 32, 32
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    matrix_add_kernel_1d_grid[grid](
        a, b, c, M, N,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return c


# ===========================================================================
# Kernel B: 2D grid — pid_m and pid_n come directly from program_id
# ===========================================================================

@triton.jit
def matrix_add_kernel_2d_grid(
    a_ptr, b_ptr, c_ptr,
    M, N,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # No decode needed — each grid axis maps directly to a matrix axis
    pid_m = tl.program_id(axis=0)   # row-block index
    pid_n = tl.program_id(axis=1)   # col-block index

    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # (BLOCK_M,)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # (BLOCK_N,)

    a_ptrs = a_ptr + row_offsets[:, None] * stride_am + col_offsets[None, :] * stride_an
    b_ptrs = b_ptr + row_offsets[:, None] * stride_bm + col_offsets[None, :] * stride_bn
    c_ptrs = c_ptr + row_offsets[:, None] * stride_cm + col_offsets[None, :] * stride_cn

    mask = (row_offsets[:, None] < M) & (col_offsets[None, :] < N)

    a = tl.load(a_ptrs, mask=mask, other=0.0)
    b = tl.load(b_ptrs, mask=mask, other=0.0)
    tl.store(c_ptrs, a + b, mask=mask)


def matrix_add_2d_grid(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape and a.is_cuda and a.ndim == 2
    M, N = a.shape
    c = torch.empty_like(a)
    BLOCK_M, BLOCK_N = 32, 32
    # 2-element tuple: (programs along axis 0, programs along axis 1)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matrix_add_kernel_2d_grid[grid](
        a, b, c, M, N,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return c


# ===========================================================================
# Tests and benchmark
# ===========================================================================

def test_correctness():
    torch.manual_seed(0)
    cases = [
        (128, 128, "128x128 square"),
        (100, 70,  "100x70 non-square (tests boundary masking)"),
    ]
    for M, N, label in cases:
        A = torch.randn(M, N, device="cuda")
        B = torch.randn(M, N, device="cuda")
        ref = A + B
        assert torch.allclose(matrix_add_1d_grid(A, B), ref), f"1D grid MISMATCH on {label}"
        assert torch.allclose(matrix_add_2d_grid(A, B), ref), f"2D grid MISMATCH on {label}"
        print(f"[PASS] {label} — both kernels match")

    # Non-contiguous tensor: stride_col=2 instead of 1
    A = torch.randn(64, 128, device="cuda")[:, ::2]
    B = torch.randn(64, 64,  device="cuda")
    ref = A + B
    assert torch.allclose(matrix_add_1d_grid(A, B), ref), "1D grid MISMATCH on non-contiguous"
    assert torch.allclose(matrix_add_2d_grid(A, B), ref), "2D grid MISMATCH on non-contiguous"
    print("[PASS] non-contiguous tensor (stride_col=2) — both kernels match")


def benchmark():
    print("\nBenchmark: 1D-grid vs 2D-grid vs PyTorch")
    print(f"{'Shape':>12}  {'Torch':>10}  {'1D-grid':>10}  {'2D-grid':>10}")
    print("-" * 48)
    for size in [128, 512, 1024, 2048, 4096]:
        A = torch.randn(size, size, device="cuda")
        B = torch.randn(size, size, device="cuda")
        ms_torch = triton.testing.do_bench(lambda: A + B)
        ms_1d    = triton.testing.do_bench(lambda: matrix_add_1d_grid(A, B))
        ms_2d    = triton.testing.do_bench(lambda: matrix_add_2d_grid(A, B))
        print(f"{f'{size}x{size}':>12}  {ms_torch:>10.4f}  {ms_1d:>10.4f}  {ms_2d:>10.4f}")


if __name__ == "__main__":
    test_correctness()
    benchmark()

"""
Concrete example. Matrix A is shape (8, 8), contiguous → stride_am=8, stride_an=1. BLOCK_M=4, BLOCK_N=4. Grid is (2,
  2) → 4 programs total.

  Focus on the program at pid_m=1, pid_n=0 (bottom-left tile):

  ---
  Step 1: row and col offsets
  row_offsets = 1*4 + [0,1,2,3] = [4, 5, 6, 7]   # logical row indices
  col_offsets = 0*4 + [0,1,2,3] = [0, 1, 2, 3]   # logical col indices

  ---
  Step 2: reshape for broadcasting
  row_offsets[:, None]  →  [[4],    shape (4, 1)
                             [5],
                             [6],
                             [7]]

  col_offsets[None, :]  →  [[0, 1, 2, 3]]          shape (1, 4)

  ---
  Step 3: multiply by strides to get memory offsets
  row_offsets[:, None] * stride_am(=8)  →  [[32],    (each row index * 8)
                                             [40],
                                             [48],
                                             [56]]

  col_offsets[None, :] * stride_an(=1)  →  [[0, 1, 2, 3]]

  ---
  Step 4: add together (broadcast) → 2D pointer block
  a_ptrs = a_ptr + [[32, 33, 34, 35],    shape (4, 4)
                     [40, 41, 42, 43],
                     [48, 49, 50, 51],
                     [56, 57, 58, 59]]

  ---
  Verify against the actual matrix layout:
  A[4,0] → 4*8 + 0 = 32  ✓
  A[4,3] → 4*8 + 3 = 35  ✓
  A[7,3] → 7*8 + 3 = 59  ✓

  This program loaded exactly rows 4–7, cols 0–3. The other three programs cover the remaining tiles:

  pid(0,0) → rows 0-3, cols 0-3    pid(0,1) → rows 0-3, cols 4-7
  pid(1,0) → rows 4-7, cols 0-3    pid(1,1) → rows 4-7, cols 4-7

  The [:, None] and [None, :] reshape trick is just NumPy-style broadcasting to generate all (row, col) pointer
  combinations in one shot instead of a nested loop.
"""