#pragma once
// Spec G §G.8 CIR table runtime loader: read-only host-resident table with
// O(1) (cell, gp) lookup.  Validate magic/version/config on load; hard-stop
// on mismatch (mirrors Spec D/F handshake discipline).

#include <cstdint>
#include <string>
#include <vector>

#include "channel/cir_table.hpp"

namespace orca {
namespace channel {

class CirTable {
public:
    // Load from file. Returns false and sets errMsg on any validation failure.
    bool load(const std::string& path, std::string& errMsg);

    bool            valid()   const { return valid_; }
    const CirHeader& header() const { return hdr_; }

    // Cell descriptor by index (call only if valid()).
    const CellDesc& cell(uint32_t c) const { return cells_[c]; }

    // UE-array descriptor (call only if valid()).
    const UeArrayDesc& ueArray() const { return ue_; }

    // Pointer to the link-block header for (cell, gp); nullptr if OOB or invalid.
    const LinkBlockHeader* linkBlock(uint32_t cell, uint64_t gp) const;

    // Pointer to the PathRecord array following a link-block header.
    const PathRecord* paths(const LinkBlockHeader* blk) const {
        return reinterpret_cast<const PathRecord*>(
            reinterpret_cast<const uint8_t*>(blk) + sizeof(LinkBlockHeader));
    }

private:
    bool                  valid_ = false;
    CirHeader             hdr_{};
    std::vector<CellDesc> cells_;
    UeArrayDesc           ue_{};
    std::vector<uint8_t>  data_;  // raw link-block region (all cells × all gp)
};

}  // namespace channel
}  // namespace orca
