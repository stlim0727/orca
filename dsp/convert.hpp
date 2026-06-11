#pragma once

#include <cstddef>

#include "common/complex.hpp"

namespace orca::dsp {

void k0IngressConvertCpu(const common::ci16* input, common::cf32* output,
                         std::size_t elementCount) noexcept;

void k5EgressPackCpu(const common::cf32* input, common::ci16* output,
                     std::size_t elementCount) noexcept;

#if ORCA_WITH_CUDA
enum class CudaLaunchStatus {
  success = 0,
  invalidArgument,
  launchFailure,
};

// Launches flat Spec-E K0/K5 conversion kernels on the supplied CUDA stream. The stream is
// intentionally passed as void* so non-CUDA translation units do not need CUDA headers.
CudaLaunchStatus k0IngressConvertCuda(const common::ci16* input, common::cf32* output,
                                       std::size_t elementCount, void* stream = nullptr) noexcept;

CudaLaunchStatus k5EgressPackCuda(const common::cf32* input, common::ci16* output,
                                   std::size_t elementCount, void* stream = nullptr) noexcept;
#endif

}  // namespace orca::dsp
