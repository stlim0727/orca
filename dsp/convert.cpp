// Host (CPU) implementation of the K0/K5 converts — also the golden model
// the CUDA kernels must match bit-exactly (Spec E §E.7).

#include "dsp/convert.hpp"

namespace orca {
namespace dsp {

void convertK0(const ci16* src, cf32* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = toCf32(src[i]);
}

void convertK5(const cf32* src, ci16* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = toCi16(src[i]);
}

}  // namespace dsp
}  // namespace orca
