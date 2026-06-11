#pragma once

#include <cstddef>

#include "common/complex.hpp"
#include "common/dims.hpp"

namespace orca::common::layout {

constexpr std::size_t idxHdl(std::size_t cell, std::size_t ue, std::size_t rx,
                             std::size_t tx, std::size_t sc) noexcept {
  return (((cell * U + ue) * numRx + rx) * numTx + tx) * numScP + sc;
}

constexpr std::size_t idxXdl(std::size_t cell, std::size_t layer,
                             std::size_t sc) noexcept {
  return (cell * rankMax + layer) * numScP + sc;
}

constexpr std::size_t idxY(std::size_t cell, std::size_t tx, std::size_t sc) noexcept {
  return (cell * numTx + tx) * numScP + sc;
}

constexpr std::size_t idxRdl(std::size_t ue, std::size_t rx, std::size_t sc) noexcept {
  return (ue * numRx + rx) * numScP + sc;
}

constexpr std::size_t idxXul(std::size_t ue, std::size_t ueTx, std::size_t sc) noexcept {
  return (ue * numUeTx + ueTx) * numScP + sc;
}

constexpr std::size_t idxRul(std::size_t cell, std::size_t rxRu, std::size_t sc) noexcept {
  return (cell * numTx + rxRu) * numScP + sc;
}

constexpr std::size_t idxZ(std::size_t cell, std::size_t layer, std::size_t sc) noexcept {
  return (cell * rankMax + layer) * numScP + sc;
}

constexpr std::size_t hDlRowBytes() noexcept { return numScP * sizeof(half2_shim); }
constexpr std::size_t cf32RowBytes() noexcept { return numScP * sizeof(cf32); }

static_assert(hDlRowBytes() == 13184, "H_dl row stride must match Spec E");
static_assert(cf32RowBytes() == 26368, "cf32 tensor row stride must match Spec E");
static_assert(hDlRowBytes() % 128 == 0, "H_dl rows must be 128-byte aligned");
static_assert(cf32RowBytes() % 128 == 0, "cf32 rows must be 128-byte aligned");

}  // namespace orca::common::layout
