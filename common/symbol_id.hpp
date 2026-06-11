#pragma once

#include <cstdint>

namespace orca::common {

enum class Direction : std::uint8_t { dl = 0, ul = 1 };

struct SymbolId {
  std::uint16_t cell;
  Direction dir;
  std::uint16_t sfn;
  std::uint16_t slot;
  std::uint8_t sym;

  constexpr bool operator==(const SymbolId& other) const noexcept {
    return cell == other.cell && dir == other.dir && sfn == other.sfn && slot == other.slot &&
           sym == other.sym;
  }
};

constexpr std::uint64_t continuousSymbolCounter(std::uint16_t sfn, std::uint16_t slot,
                                                std::uint8_t sym,
                                                std::uint16_t slotsPerFrame = 20,
                                                std::uint8_t symbolsPerSlot = 14) noexcept {
  return (static_cast<std::uint64_t>(sfn) * slotsPerFrame + slot) * symbolsPerSlot + sym;
}

}  // namespace orca::common
