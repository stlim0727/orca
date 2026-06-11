#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace orca::common {

struct cf32 {
  float re;
  float im;
};

struct ci16 {
  std::int16_t re;
  std::int16_t im;
};

// Host-side stand-in for CUDA's half2 storage shape. CUDA kernels use native half support
// internally when needed; Stage 1 only requires the layout-compatible pair.
struct half2_shim {
  std::uint16_t re;
  std::uint16_t im;
};

constexpr cf32 toCf32(ci16 v) noexcept {
  return cf32{static_cast<float>(v.re), static_cast<float>(v.im)};
}

inline std::int16_t saturatingRoundToI16(float value) noexcept {
  if (std::isnan(value)) {
    return 0;
  }

  const float clamped = std::max(
      static_cast<float>(std::numeric_limits<std::int16_t>::min()),
      std::min(static_cast<float>(std::numeric_limits<std::int16_t>::max()), value));
  return static_cast<std::int16_t>(std::lrint(clamped));
}

inline ci16 toCi16(cf32 v) noexcept {
  return ci16{saturatingRoundToI16(v.re), saturatingRoundToI16(v.im)};
}

static_assert(sizeof(cf32) == 8, "cf32 must be two 32-bit floats");
static_assert(sizeof(ci16) == 4, "ci16 must be two 16-bit signed integers");
static_assert(sizeof(half2_shim) == 4, "half2 storage must remain 4 bytes");

}  // namespace orca::common
