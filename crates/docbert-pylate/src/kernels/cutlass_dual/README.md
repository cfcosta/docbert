# Vendored CUTLASS dual GEMM

These headers are copied from NVIDIA CUTLASS `examples/45_dual_gemm` at
commit `7d49e6c7e2f8896c47f586706e67e1fb215529dc`, the same commit whose
`include/` headers the build already pins for FlashAttention and the
rounded GeGLU epilogue. They were ported through vs1 without changing
the vendored interfaces. The example directory is not part of that header
cache, so the six files are kept here under their original BSD-3-Clause
license headers.

One local change: `threadblock/dual_epilogue.h` skips loading the C source
tiles when neither output op needs them. Upstream always loads them, which
would re-read a full output-sized tensor that the round-only ops ignore.
