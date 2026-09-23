// The BF16 operations preserve candle-kernels' rope_thd rounding.
__device__ __forceinline__ unsigned short mul_bf16(unsigned short a, unsigned short b) {
    unsigned short r;
    asm("{ .reg .b16 z; mov.b16 z, 0x8000; fma.rn.bf16 %0,%1,%2,z; }"
        : "=h"(r) : "h"(a), "h"(b));
    return r;
}
__device__ __forceinline__ unsigned short add_bf16(unsigned short a, unsigned short b) {
    unsigned short r;
    asm("{ .reg .b16 one; mov.b16 one, 0x3f80; fma.rn.bf16 %0,%1,one,%2; }"
        : "=h"(r) : "h"(a), "h"(b));
    return r;
}
__device__ __forceinline__ unsigned short sub_bf16(unsigned short a, unsigned short b) {
    unsigned short r;
    asm("{ .reg .b16 neg; mov.b16 neg, 0xbf80; fma.rn.bf16 %0,%2,neg,%1; }"
        : "=h"(r) : "h"(a), "h"(b));
    return r;
}
extern "C" __global__ void apply_rotary_pair_bf16(
    const unsigned short *q, const unsigned short *k,
    const unsigned short *cos, const unsigned short *sin,
    unsigned short *out, unsigned int count, unsigned int tokens,
    unsigned int heads, unsigned int dim
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count / 2) return;
    unsigned int row = i / (dim / 2), col = i % (dim / 2);
    unsigned int a = row * dim + col, b = a + dim / 2;
    // Query tables broadcast across batches; packed inputs have batch size one.
    unsigned int cs = ((row / heads) % tokens) * (dim / 2) + col;
    unsigned short c = cos[cs], s = sin[cs];
    unsigned short q0 = q[a], q1 = q[b], k0 = k[a], k1 = k[b];
    out[a] = sub_bf16(mul_bf16(q0, c), mul_bf16(q1, s));
    out[b] = add_bf16(mul_bf16(q0, s), mul_bf16(q1, c));
    out[count + a] = sub_bf16(mul_bf16(k0, c), mul_bf16(k1, s));
    out[count + b] = add_bf16(mul_bf16(k0, s), mul_bf16(k1, c));
}
