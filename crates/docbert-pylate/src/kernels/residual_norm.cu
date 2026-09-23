// Preserve Candle's BF16 add rounding and LayerNorm reduction order.
#include <math.h>
__device__ __forceinline__ unsigned short add_bf16(unsigned short a, unsigned short b) {
    unsigned short r;
    asm("{ .reg .b16 one; mov.b16 one, 0x3f80; fma.rn.bf16 %0,%1,one,%2; }"
        : "=h"(r) : "h"(a), "h"(b));
    return r;
}
__device__ __forceinline__ float decode_bf16(unsigned short a) {
    return __uint_as_float(static_cast<unsigned int>(a) << 16);
}
extern "C" __global__ void add_and_normalize_bf16(
    const unsigned short *x, const unsigned short *y, const unsigned short *weight,
    unsigned short *out, unsigned int count, int ncols, float eps
) {
    const unsigned int start = blockIdx.x * ncols;
    const int tid = threadIdx.x;
    float2 mean_var = make_float2(0.f, 0.f);
    for (int col = tid; col < ncols; col += blockDim.x) {
        const unsigned short sum = add_bf16(x[start + col], y[start + col]);
        out[start + col] = sum;
        const float xi = decode_bf16(sum);
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }
    // Same lane traversal and XOR shuffle order as candle-kernels/reduce.cu.
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        mean_var.x += __shfl_xor_sync(0xffffffff, mean_var.x, mask, 32);
        mean_var.y += __shfl_xor_sync(0xffffffff, mean_var.y, mask, 32);
    }
    if (blockDim.x > 32) {
        __shared__ float2 sums[32];
        const int warp = tid / 32, lane = tid % 32;
        if (lane == 0) sums[warp] = mean_var;
        __syncthreads();
        mean_var = sums[lane];
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            mean_var.x += __shfl_xor_sync(0xffffffff, mean_var.x, mask, 32);
            mean_var.y += __shfl_xor_sync(0xffffffff, mean_var.y, mask, 32);
        }
    }
    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);
    for (int col = tid; col < ncols; col += blockDim.x) {
        const float lhs = (decode_bf16(out[start + col]) - mean) * inv_std;
        // The trunk LayerNorm has a persistent +0 bias. Preserve its FMA.
        const float value = __fmaf_rn(lhs, decode_bf16(weight[col]), 0.f);
        unsigned short result;
        asm("cvt.rn.bf16.f32 %0,%1;" : "=h"(result) : "f"(value));
        out[count + start + col] = result;
    }
}
