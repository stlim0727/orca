#include "dsp/combine.hpp"

#include <cstring>

#include "common/dims.hpp"
#include "common/layout.hpp"

namespace orca {
namespace dsp {

uint32_t combineGolden(const cf32* r_ul, const BeamCodebook& book,
                       const Alloc* allocs, uint32_t numAllocs, cf32* z) {
    std::memset(z, 0, sizeof(cf32) * layout::elemsZ);

    uint32_t rejects = 0;
    for (uint32_t i = 0; i < numAllocs; ++i) {
        const Alloc& a = allocs[i];
        if (a.dir != 1) continue;  // DL allocations are not combined
        const bool ok =
            a.cell < dims::C && a.rank >= 1 && a.rank <= dims::rankMax &&
            a.scStart < dims::numSc && a.scLen >= 1 &&
            a.scLen <= dims::numSc - a.scStart;
        if (!ok) {
            ++rejects;
            continue;
        }

        const cf32* Cw[dims::rankMax] = {};
        bool beamsOk = true;
        for (uint8_t l = 0; l < a.rank; ++l) {
            Cw[l] = book.beam(a.beamId[l]);
            if (!Cw[l]) beamsOk = false;
        }
        if (!beamsOk) {
            ++rejects;
            continue;
        }

        for (uint8_t l = 0; l < a.rank; ++l) {
            for (uint32_t sc = a.scStart; sc < uint32_t{a.scStart} + a.scLen;
                 ++sc) {
                cf32 acc{0.0f, 0.0f};
                for (uint32_t rx = 0; rx < dims::numTx; ++rx)
                    acc = cmac(acc, Cw[l][rx],
                               r_ul[layout::idxRul(a.cell, rx, sc)]);
                z[layout::idxZ(a.cell, l, sc)] = acc;
            }
        }
    }
    return rejects;
}

}  // namespace dsp
}  // namespace orca
