#include "dsp/convert.hpp"

namespace orca::dsp {

void k0IngressConvertCpu(const common::ci16* input, common::cf32* output,
                         std::size_t elementCount) noexcept {
  for (std::size_t i = 0; i < elementCount; ++i) {
    output[i] = common::toCf32(input[i]);
  }
}

void k5EgressPackCpu(const common::cf32* input, common::ci16* output,
                     std::size_t elementCount) noexcept {
  for (std::size_t i = 0; i < elementCount; ++i) {
    output[i] = common::toCi16(input[i]);
  }
}

}  // namespace orca::dsp
