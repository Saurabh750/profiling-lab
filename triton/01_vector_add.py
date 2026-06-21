"""
Triton Kernel 01: 1D Vector Addition
=====================================

Goal: add two 1D vectors elementwise: C = A + B

THE CORE TRITON MENTAL MODEL
-----------------------------
In CUDA, a kernel function runs once per thread. Thread knows its position via
threadIdx, blockIdx, blockDim.

In Triton, a kernel function runs once per "program instance" (think: one CUDA block).
But instead of a single thread doing one element, each program instance operates on
a TILE - a contiguous chunk of BLOCK_SIZE elements - all at once using vectorized ops.

So if your vector has N=1024 elements and BLOCK_SIZE=256, you launch 4 program instances:
  program 0 handles elements [0..255]
  program 1 handles elements [256..511]
  program 2 handles elements [512..767]
  program 3 handles elements [768..1023]

THE CANONICAL TRITON KERNEL LOOP
---------------------------------
Every Triton kernel follows this pattern:

  1. pid = tl.program_id(0)          # which tile am I? (like blockIdx.x)
  2. offsets = pid * BLOCK + arange  # absolute indices for my tile
  3. mask = offsets < N              # boundary guard for the last tile
  4. a = tl.load(a_ptr + offsets, mask=mask)   # load my tile
  5. c = compute(a, b)               # do work
  6. tl.store(c_ptr + offsets, c, mask=mask)   # write back

Once you understand this loop, every Triton kernel is a variation on it.

POINTER ARITHMETIC IN TRITON
------------------------------
Triton works with raw pointers (like C). If a_ptr is a pointer to the start of
tensor A, then:
  a_ptr + 5   points to element A[5]
  a_ptr + offsets  (where offsets is a vector)  points to A[offsets[0]], A[offsets[1]], ...

tl.load(ptr_vector, mask) does a vectorized gather load.
tl.store(ptr_vector, values, mask) does a vectorized scatter store.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# The kernel: decorated with @triton.jit, runs on the GPU
# ---------------------------------------------------------------------------

@triton.jit
def vector_add_kernel(
    a_ptr,      # pointer to first input vector A
    b_ptr,      # pointer to second input vector B
    c_ptr,      # pointer to output vector C
    N,          # total number of elements
    BLOCK_SIZE: tl.constexpr,  # tile size - must be constexpr so Triton can unroll/optimize
):
    # Step 1: which program instance (tile) am I?
    # axis=0 means the first (and only) grid dimension here
    pid = tl.program_id(axis=0)

    # Step 2: compute the absolute indices for all elements in my tile
    # arange(0, BLOCK_SIZE) gives [0, 1, 2, ..., BLOCK_SIZE-1]
    # pid * BLOCK_SIZE shifts it to the right tile
    # e.g. pid=2, BLOCK_SIZE=4 -> [8, 9, 10, 11]
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Step 3: create a boolean mask for out-of-bounds elements
    # The last tile may be partial (N might not be divisible by BLOCK_SIZE)
    # Without the mask, we'd read/write garbage memory past the end of the array
    mask = offsets < N

    # Step 4: load tiles from A and B
    # other=0.0 fills masked-out lanes with 0 (doesn't matter for loads
    # that we mask on store, but good practice)
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)

    # Step 5: compute
    c = a + b

    # Step 6: store result - masked so we don't corrupt memory past the array
    tl.store(c_ptr + offsets, c, mask=mask)


# ---------------------------------------------------------------------------
# The launcher: runs on the CPU, sets up the grid and calls the kernel
# ---------------------------------------------------------------------------

def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape, "A and B must have the same shape"
    assert a.is_cuda and b.is_cuda, "Tensors must be on GPU"

    N = a.numel()
    c = torch.empty_like(a)

    BLOCK_SIZE = 1024  # each program handles 1024 elements

    # The grid is a tuple of ints or lambdas that tells Triton how many
    # program instances to launch along each dimension.
    # Here we need ceil(N / BLOCK_SIZE) programs along axis 0.
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    # Launch the kernel. Arguments map 1-to-1 to kernel parameters.
    vector_add_kernel[grid](
        a, b, c,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return c


# ---------------------------------------------------------------------------
# Test and benchmark
# ---------------------------------------------------------------------------

def test_correctness():
    torch.manual_seed(0)
    N = 10_000  # intentionally not a multiple of BLOCK_SIZE to test masking
    a = torch.randn(N, device="cuda", dtype=torch.float32)
    b = torch.randn(N, device="cuda", dtype=torch.float32)

    c_triton = vector_add(a, b)
    c_torch = a + b

    # allclose checks numerical equality within a tolerance
    assert torch.allclose(c_triton, c_torch), "MISMATCH!"
    print(f"[PASS] N={N}: max abs error = {(c_triton - c_torch).abs().max().item():.2e}")


def benchmark():
    """Compare Triton vs torch for various sizes."""
    print("\nBenchmark: Triton vs PyTorch vector add")
    print(f"{'N':>12}  {'Torch (ms)':>12}  {'Triton (ms)':>12}  {'Speedup':>8}")
    print("-" * 52)

    for N in [2**i for i in range(14, 26)]:  # 16K to 64M elements
        a = torch.randn(N, device="cuda", dtype=torch.float32)
        b = torch.randn(N, device="cuda", dtype=torch.float32)

        # Triton's built-in benchmarking utility: runs warmup + timed reps
        ms_torch = triton.testing.do_bench(lambda: a + b)
        ms_triton = triton.testing.do_bench(lambda: vector_add(a, b))

        speedup = ms_torch / ms_triton
        print(f"{N:>12,}  {ms_torch:>12.4f}  {ms_triton:>12.4f}  {speedup:>8.2f}x")


if __name__ == "__main__":
    test_correctness()
    benchmark()
