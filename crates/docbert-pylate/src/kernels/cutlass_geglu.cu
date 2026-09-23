// Round each projection to BF16 before applying Candle's rounded GeGLU.
#include <cutlass/bfloat16.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cuda_runtime.h>
#include "cutlass_dual/device/dual_gemm.h"

using Bfloat16 = cutlass::bfloat16_t;

__device__ unsigned short multiply_bf16(unsigned short a, unsigned short b) {
    unsigned short result;
    asm("{ .reg .b16 z; mov.b16 z, 0x8000; fma.rn.bf16 %0, %1, %2, z; }"
        : "=h"(result) : "h"(a), "h"(b));
    return result;
}

struct RoundedGeGlu {
    using ElementOutput = Bfloat16;
    using ElementAccumulator = Bfloat16;
    using ElementCompute = float;
    static int const kCount = 8;
    using FragmentOutput = cutlass::Array<Bfloat16, kCount>;
    using FragmentAccumulator = cutlass::Array<Bfloat16, kCount>;
    struct Params {};

    CUTLASS_HOST_DEVICE RoundedGeGlu(Params const &) {}

    CUTLASS_DEVICE FragmentOutput operator()(
        FragmentAccumulator const &activation,
        FragmentAccumulator const &gate
    ) const {
        FragmentOutput output;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kCount; ++i) {
            Bfloat16 rounded_activation = activation[i];
            Bfloat16 rounded_gate = gate[i];
            Bfloat16 cdf(normcdff(float(rounded_activation)));
            output[i] = Bfloat16::bitcast(multiply_bf16(
                multiply_bf16(rounded_activation.raw(), cdf.raw()), rounded_gate.raw()));
        }
        return output;
    }
};

using RoundOnly = cutlass::epilogue::thread::LinearCombination<
    Bfloat16, 8, float, float, cutlass::epilogue::thread::ScaleType::Nothing>;
using DualGeGlu = cutlass::gemm::device::DualGemm<
    Bfloat16, cutlass::layout::RowMajor, Bfloat16,
    cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor,
    Bfloat16, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 32>,
    cutlass::gemm::GemmShape<64, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    RoundOnly, RoundOnly, RoundedGeGlu,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    3, false, false, false>;

extern "C" int apply_docbert_dual_geglu(
    void const *input, void const *weight, void *output,
    int rows, int inter, int width, void *stream
) {
    auto weights = static_cast<Bfloat16 const *>(weight);
    auto result = static_cast<Bfloat16 *>(output);
    // Both B operands alias the fused Wi allocation. C and D0/D1 are unused.
    DualGeGlu::Arguments arguments(
        cutlass::gemm::DualGemmMode::kGemm, {rows, inter, width},
        {static_cast<Bfloat16 const *>(input), width},
        {weights, width}, {result, inter}, {nullptr, inter},
        {weights + inter * width, width},
        {result, inter}, {nullptr, inter}, {result, inter});
    DualGeGlu operation;
    auto status = operation.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) return int(status);
    status = operation.initialize(arguments, nullptr, static_cast<cudaStream_t>(stream));
    if (status != cutlass::Status::kSuccess) return 100 + int(status);
    return int(operation(static_cast<cudaStream_t>(stream)));
}
