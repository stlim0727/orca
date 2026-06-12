#include "channel/cir_writer.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace orca {
namespace channel {

CirWriter::CirWriter(const CirHeader& hdr, const CellDesc* cells,
                     const UeArrayDesc& ue)
    : hdr_(hdr), cells_(cells, cells + hdr.numCells), ue_(ue) {
    std::memcpy(hdr_.magic, "ORCACIR1", 8);
    hdr_.version        = kCirVersion;
    hdr_.pathRecBytes   = uint32_t{sizeof(PathRecord)};
    hdr_.linkBlockBytes = linkBlockStride(hdr_.pMax);

    const uint64_t totalBlocks =
        uint64_t{hdr_.numCells} * numGp(hdr_);
    links_.assign(totalBlocks * hdr_.linkBlockBytes, 0u);
}

void CirWriter::setLink(uint32_t cell, uint64_t gp, uint16_t numPaths,
                        uint16_t flags, float pathlossDb,
                        const PathRecord* paths) {
    if (cell >= hdr_.numCells || gp >= numGp(hdr_)) return;
    uint8_t* dst = links_.data() + linkOffset(hdr_, cell, gp);

    // Clear the entire block so re-writes with fewer paths leave no stale bytes.
    std::memset(dst, 0, hdr_.linkBlockBytes);

    const uint32_t n = std::min(uint32_t{numPaths}, hdr_.pMax);
    // Store the clamped count so the on-disk block is self-consistent.
    LinkBlockHeader lbh{static_cast<uint16_t>(n), flags, pathlossDb};
    std::memcpy(dst, &lbh, sizeof(lbh));

    if (n > 0 && paths != nullptr)
        std::memcpy(dst + sizeof(LinkBlockHeader), paths,
                    n * sizeof(PathRecord));
}

bool CirWriter::write(const std::string& path) const {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;

    auto wOk = [f](const void* p, size_t sz) {
        return std::fwrite(p, 1, sz, f) == sz;
    };

    const bool ok = wOk(&hdr_, sizeof(hdr_))
                 && wOk(cells_.data(), cells_.size() * sizeof(CellDesc))
                 && wOk(&ue_, sizeof(ue_))
                 && wOk(links_.data(), links_.size());

    std::fclose(f);
    return ok;
}

}  // namespace channel
}  // namespace orca
