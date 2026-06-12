#include "channel/cir_loader.hpp"

#include <cstdio>

namespace orca {
namespace channel {

bool CirTable::load(const std::string& path, std::string& errMsg) {
    valid_ = false;

    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { errMsg = "cannot open file"; return false; }

    auto rOk = [f](void* p, size_t sz) {
        return std::fread(p, 1, sz, f) == sz;
    };

    if (!rOk(&hdr_, sizeof(hdr_))) {
        errMsg = "header read failed"; goto fail;
    }
    if (!cirMagicOk(hdr_)) { errMsg = "bad magic"; goto fail; }
    if (hdr_.version != kCirVersion) { errMsg = "version mismatch"; goto fail; }
    if (hdr_.pathRecBytes != sizeof(PathRecord)) {
        errMsg = "pathRecBytes mismatch"; goto fail;
    }
    if (hdr_.linkBlockBytes != linkBlockStride(hdr_.pMax)) {
        errMsg = "linkBlockBytes mismatch"; goto fail;
    }
    if (hdr_.numCells == 0) { errMsg = "numCells is zero"; goto fail; }
    if (hdr_.nx == 0 || hdr_.ny == 0 || hdr_.nz == 0) {
        errMsg = "zero grid extent"; goto fail;
    }

    cells_.resize(hdr_.numCells);
    if (!rOk(cells_.data(), cells_.size() * sizeof(CellDesc))) {
        errMsg = "cell descriptor read failed"; goto fail;
    }

    if (!rOk(&ue_, sizeof(ue_))) {
        errMsg = "UE array descriptor read failed"; goto fail;
    }

    {
        const uint64_t dataBytes =
            uint64_t{hdr_.numCells} * numGp(hdr_) * hdr_.linkBlockBytes;
        data_.resize(dataBytes);
        if (!rOk(data_.data(), dataBytes)) {
            errMsg = "link block data read failed"; goto fail;
        }
    }

    std::fclose(f);
    valid_ = true;
    return true;

fail:
    std::fclose(f);
    return false;
}

const LinkBlockHeader* CirTable::linkBlock(uint32_t cell, uint64_t gp) const {
    if (!valid_ || cell >= hdr_.numCells || gp >= numGp(hdr_)) return nullptr;
    return reinterpret_cast<const LinkBlockHeader*>(
        data_.data() + linkOffset(hdr_, cell, gp));
}

}  // namespace channel
}  // namespace orca
