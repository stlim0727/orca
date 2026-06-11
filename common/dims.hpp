#pragma once

#include <cstddef>
#include <cstdint>

namespace orca::common {

constexpr std::uint32_t roundUp(std::uint32_t value, std::uint32_t multiple) noexcept {
  return ((value + multiple - 1U) / multiple) * multiple;
}

constexpr std::uint32_t C = 2;
constexpr std::uint32_t U = 32;
constexpr std::uint32_t numTx = 64;
constexpr std::uint32_t numRx = 4;
constexpr std::uint32_t numUeTx = 2;
constexpr std::uint32_t numSc = 3276;
constexpr std::uint32_t numScP = roundUp(numSc, 32);
constexpr std::uint32_t rankMax = 4;
constexpr std::uint32_t N_ring = 4;
constexpr std::uint32_t MAX_ALLOCS = 512;

static_assert(numScP == 3296, "Phase-1 padded subcarrier count must match Spec E");

}  // namespace orca::common
