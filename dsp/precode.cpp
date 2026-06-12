#include "dsp/precode.hpp"

#include <cstring>

#include "common/dims.hpp"
#include "common/layout.hpp"

namespace orca {
namespace dsp {

uint32_t precodeGolden(const cf32* x_dl, const BeamCodebook& book,
                       const Alloc* allocs, uint32_t numAllocs, cf32* y) {
    std::memset(y, 0, sizeof(cf32) * layout::elemsY);

    uint32_t rejects = 0;
    for (uint32_t i = 0; i < numAllocs; ++i) {
        const Alloc& a = allocs[i];
        if (a.dir != 0) continue;  // UL allocations are not precoded
        const bool ok =
            a.cell < dims::C && a.rank >= 1 && a.rank <= dims::rankMax &&
            a.scStart < dims::numSc && a.scLen >= 1 &&
            a.scLen <= dims::numSc - a.scStart;
        if (!ok) {
            ++rejects;
            continue;
        }

        // Gather W[tx][l] from the codebook (Spec E §E.4 step 2).
        const cf32* W[dims::rankMax] = {};
        bool beamsOk = true;
        for (uint8_t l = 0; l < a.rank; ++l) {
            W[l] = book.beam(a.beamId[l]);
            if (!W[l]) beamsOk = false;
        }
        if (!beamsOk) {
            ++rejects;
            continue;
        }

        for (uint32_t tx = 0; tx < dims::numTx; ++tx) {
            for (uint32_t sc = a.scStart; sc < uint32_t{a.scStart} + a.scLen;
                 ++sc) {
                cf32 acc{0.0f, 0.0f};
                for (uint8_t l = 0; l < a.rank; ++l)
                    acc = cmac(acc, W[l][tx],
                               x_dl[layout::idxXdl(a.cell, l, sc)]);
                y[layout::idxY(a.cell, tx, sc)] = acc;
            }
        }
    }
    return rejects;
}

}  // namespace dsp
}  // namespace orca
