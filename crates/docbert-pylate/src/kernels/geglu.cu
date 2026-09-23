// Preserve candle-kernels 0.11's BF16 normal-CDF rounding and both products.
#include <math.h>

__device__ __forceinline__ unsigned short round_bf16(float x) {
    unsigned short result;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(result) : "f"(x));
    return result;
}

__device__ __forceinline__ unsigned short mul_bf16(unsigned short a, unsigned short b) {
    unsigned short result;
    // The negative-zero addend matches Ampere __hmul, including signed zero.
    asm("{ .reg .b16 z; mov.b16 z, 0x8000; fma.rn.bf16 %0, %1, %2, z; }"
        : "=h"(result) : "h"(a), "h"(b));
    return result;
}

__device__ __forceinline__ unsigned int mul_bf16_pair(unsigned int a, unsigned int b) {
    unsigned int result;
    asm("{ .reg .b32 z; mov.b32 z, 0x80008000; fma.rn.bf16x2 %0, %1, %2, z; }"
        : "=r"(result) : "r"(a), "r"(b));
    return result;
}

extern "C" __global__ void apply_geglu_bf16(
    const unsigned short *input,
    unsigned short *output,
    unsigned int count,
    unsigned int inter
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        unsigned int row = i / inter;
        unsigned int column = i % inter;
        unsigned int offset = row * 2 * inter + column;
        unsigned short activation = input[offset];
        float x = __uint_as_float((unsigned int)activation << 16);
        unsigned short cdf = round_bf16(normcdff(x));
        unsigned short gelu = mul_bf16(activation, cdf);
        output[i] = mul_bf16(gelu, input[offset + inter]);
    }
}

extern "C" __global__ void apply_geglu_bf16_pair(
    const unsigned short *input,
    unsigned short *output,
    unsigned int count,
    unsigned int inter
) {
    unsigned int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) {
        // Even widths and storage offsets keep both halves aligned.
        unsigned int row = i / inter;
        unsigned int column = i % inter;
        unsigned int offset = row * 2 * inter + column;
        unsigned int activation = *reinterpret_cast<const unsigned int *>(input + offset);
        unsigned int gate = *reinterpret_cast<const unsigned int *>(input + offset + inter);
        unsigned int cdf0 = round_bf16(normcdff(__uint_as_float((activation & 0xffff) << 16)));
        unsigned int cdf1 = round_bf16(normcdff(__uint_as_float(activation & 0xffff0000)));
        unsigned int gelu = mul_bf16_pair(activation, cdf0 | (cdf1 << 16));
        reinterpret_cast<unsigned int *>(output)[i / 2] = mul_bf16_pair(gelu, gate);
    }
}
