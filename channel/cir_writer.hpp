#pragma once
// Spec G §G.7 CIR table writer: assemble and serialize the binary format.
// Used by the offline toolchain and by tests to generate synthetic tables.

#include <cstdint>
#include <string>
#include <vector>

#include "channel/cir_table.hpp"

namespace orca {
namespace channel {

class CirWriter {
public:
    // Initializes with a valid header skeleton (magic/version/pathRecBytes/
    // linkBlockBytes are overwritten to the correct values).
    CirWriter(const CirHeader& hdr,
              const CellDesc* cells,   // hdr.numCells entries
              const UeArrayDesc& ue);

    // Set the link block for (cell, gp). If numPaths > hdr.pMax, only the
    // first pMax paths are stored. Silently ignored for OOB (cell, gp).
    void setLink(uint32_t cell, uint64_t gp, uint16_t numPaths, uint16_t flags,
                 float pathlossDb, const PathRecord* paths);

    // Write the table to a file. Returns true on success.
    bool write(const std::string& path) const;

private:
    CirHeader             hdr_;
    std::vector<CellDesc> cells_;
    UeArrayDesc           ue_;
    std::vector<uint8_t>  links_;  // raw link-block region
};

}  // namespace channel
}  // namespace orca
