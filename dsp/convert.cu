#include "dsp/convert.hpp"

#include <cuda_runtime.h>

namespace orca::dsp {
namespace {

constexpr int kTile = 256;

__device__ __forceinline__ short saturatingRoundToI16Device(float value) {
  if (isnan(value)) {
    return 0;
  }
  value = fminf(32767.0f, fmaxf(-32768.0f, value));
  return static_cast<short>(__float2int_rn(value));
}

__global__ void k0IngressConvertKernel(const common::ci16* __restrict__ input,
                                       common::cf32* __restrict__ output,
                                       std::size_t elementCount) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= elementCount) {
    return;
  }

  const common::ci16 v = input[idx];
  output[idx] = common::cf32{static_cast<float>(v.re), static_cast<float>(v.im)};
}

__global__ void k5EgressPackKernel(const common::cf32* __restrict__ input,
                                   common::ci16* __restrict__ output,
                                   std::size_t elementCount) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= elementCount) {
    return;
  }

  const common::cf32 v = input[idx];
  output[idx] = common::ci16{saturatingRoundToI16Device(v.re), saturatingRoundToI16Device(v.im)};
}

int gridSize(std::size_t elementCount) noexcept {
  return static_cast<int>((elementCount + kTile - 1) / kTile);
}

cudaStream_t toCudaStream(void* stream) noexcept { return reinterpret_cast<cudaStream_t>(stream); }

}  // namespace

void k0IngressConvertCuda(const common::ci16* input, common::cf32* output,
                          std::size_t elementCount, void* stream) {
  if (elementCount == 0) {
    return;
  }
  k0IngressConvertKernel<<<gridSize(elementCount), kTile, 0, toCudaStream(stream)>>>(
      input, output, elementCount);
}

void k5EgressPackCuda(const common::cf32* input, common::ci16* output,
                      std::size_t elementCount, void* stream) {
  if (elementCount == 0) {
    return;
  }
  k5EgressPackKernel<<<gridSize(elementCount), kTile, 0, toCudaStream(stream)>>>(
      input, output, elementCount);
}

}  // namespace orca::dsp
