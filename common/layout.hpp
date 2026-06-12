#pragma once
// Flat row-major index functions for every hot tensor, exactly per Spec E §E.11.
// Golden rule: `sc` is the innermost (unit-stride) axis of every tensor, padded
// to numScP so each row starts on a 128-byte boundary.

#include <cstddef>

#include "common/complex.hpp"
#include "common/dims.hpp"

namespace orca {
namespace layout {

using dims::C;
using dims::numRx;
using dims::numScP;
using dims::numTx;
using dims::numUeTx;
using dims::rankMax;
using dims::U;

// H_dl[c][u][rx][tx][sc]  (half2)
constexpr size_t idxH(size_t c, size_t u, size_t rx, size_t tx, size_t sc) {
    return ((((c * U + u) * numRx + rx) * numTx + tx) * numScP) + sc;
}

// x_dl[c][l][sc]  (cf32)
constexpr size_t idxXdl(size_t c, size_t l, size_t sc) {
    return ((c * rankMax + l) * numScP) + sc;
}

// y[c][tx][sc]  (cf32, DL mid P·x)
constexpr size_t idxY(size_t c, size_t tx, size_t sc) {
    return ((c * numTx + tx) * numScP) + sc;
}

// r_dl[u][rx][sc]  (cf32, DL out → vUE)
constexpr size_t idxRdl(size_t u, size_t rx, size_t sc) {
    return ((u * numRx + rx) * numScP) + sc;
}

// x_ul[u][uetx][sc]  (cf32, ← vUE)
constexpr size_t idxXul(size_t u, size_t uetx, size_t sc) {
    return ((u * numUeTx + uetx) * numScP) + sc;
}

// r_ul[c][rx_ru][sc]  (cf32, UL mid H_h·x; rx_ru spans the 64 RU antennas)
constexpr size_t idxRul(size_t c, size_t rxRu, size_t sc) {
    return ((c * numTx + rxRu) * numScP) + sc;
}

// z[c][l][sc]  (cf32, UL out → vDU)
constexpr size_t idxZ(size_t c, size_t l, size_t sc) {
    return ((c * rankMax + l) * numScP) + sc;
}

// Total element counts (allocation extents).
constexpr size_t elemsH   = static_cast<size_t>(C) * U * numRx * numTx * numScP;
constexpr size_t elemsXdl = static_cast<size_t>(C) * rankMax * numScP;
constexpr size_t elemsY   = static_cast<size_t>(C) * numTx * numScP;
constexpr size_t elemsRdl = static_cast<size_t>(U) * numRx * numScP;
constexpr size_t elemsXul = static_cast<size_t>(U) * numUeTx * numScP;
constexpr size_t elemsRul = static_cast<size_t>(C) * numTx * numScP;
constexpr size_t elemsZ   = static_cast<size_t>(C) * rankMax * numScP;

// Row byte strides (Spec E §E.11: 13184 for half2 rows, 26368 for cf32 rows).
constexpr size_t rowBytesHalf2 = static_cast<size_t>(numScP) * sizeof(half2c);
constexpr size_t rowBytesCf32  = static_cast<size_t>(numScP) * sizeof(cf32);

static_assert(rowBytesHalf2 == 13184, "H row stride (Spec E §E.11)");
static_assert(rowBytesCf32 == 26368, "cf32 row stride (Spec E §E.11)");
static_assert(rowBytesHalf2 % 128 == 0, "half2 rows start 128-B aligned");
static_assert(rowBytesCf32 % 128 == 0, "cf32 rows start 128-B aligned");

}  // namespace layout
}  // namespace orca
