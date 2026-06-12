#pragma once
// Spec G §G.7 on-disk CIR table format: POD types, constants, stride helpers.
// All fields are little-endian (the toolchain and host are both LE; no swapping).
// Mmap-friendly: fixed-stride link blocks → O(1) (cell, gp) lookup.

#include <cstdint>
#include <cstring>

namespace orca {
namespace channel {

// ---- PathRecord — 40 bytes (G.7) -------------------------------------------

struct PathRecord {
    float    tau;        // excess delay (s), relative to first arrival
    float    gainRe;     // complex path gain (linear), real part
    float    gainIm;     // imaginary part
    float    aodAz;      // departure azimuth, cell-array frame (rad)
    float    aodEl;      // departure elevation (rad)
    float    aoaAz;      // arrival azimuth, UE-array frame (rad)
    float    aoaEl;      // arrival elevation (rad)
    uint8_t  flags;      // bit0=LoS, bit1=ground-reflect
    uint8_t  _pad0[3];   // alignment
    float    _rsv[2];    // reserved (zero)
};
static_assert(sizeof(PathRecord) == 40, "PathRecord must be 40 B (G.7)");

// PathRecord flags (G.6).
constexpr uint8_t kPathLoS     = 0x01u;
constexpr uint8_t kPathGndRefl = 0x02u;

// ---- LinkBlockHeader — 8 bytes (G.7) ----------------------------------------
// Precedes PathRecord paths[pMax] in each link block.

struct LinkBlockHeader {
    uint16_t numPaths;   // actual path count (≤ pMax)
    uint16_t flags;      // bit0 = kLinkNoCoverage
    float    pathlossDb; // aggregate path-loss (dB), for interferer ranking (G.5)
};
static_assert(sizeof(LinkBlockHeader) == 8, "LinkBlockHeader must be 8 B");

constexpr uint16_t kLinkNoCoverage = 0x0001u;

// Total byte stride of one link block in the file.
inline uint32_t linkBlockStride(uint32_t pMax) {
    return uint32_t{sizeof(LinkBlockHeader)} + pMax * uint32_t{sizeof(PathRecord)};
}

// ---- CellDesc — 40 bytes (G.7) ----------------------------------------------
// pos[3]=12 + boresight[3]=12 + numTx=4 + arrayType=4 + elemSpacing=4 + _rsv=4.

struct CellDesc {
    float    pos[3];       // world position (m)
    float    boresight[3]; // array boresight unit vector
    uint32_t numTx;        // TRX antennas
    uint32_t arrayType;    // 0=ULA, 1=UPA, 2=dual-pol-UPA
    float    elemSpacing;  // element spacing (m)
    float    _rsv;         // reserved
};
static_assert(sizeof(CellDesc) == 40, "CellDesc must be 40 B (G.7)");

// ---- UeArrayDesc — 20 bytes (G.7) -------------------------------------------

struct UeArrayDesc {
    uint32_t numRx;         // UE receive antennas
    uint32_t numUeTx;       // UE transmit antennas (subset of rx, Spec E §E.6)
    uint8_t  ueTxToRx[4];  // which rx elements transmit (default {0,1,0,0})
    uint32_t arrayType;
    float    elemSpacing;
};
static_assert(sizeof(UeArrayDesc) == 20, "UeArrayDesc must be 20 B");

// ---- CirHeader — 96 bytes (G.7) ---------------------------------------------
// Explicit _pad0 ensures 8-byte alignment of carrierHz on all platforms.

struct CirHeader {
    char     magic[8];        // "ORCACIR1"
    uint32_t version;         // kCirVersion = 1
    uint32_t numCells;
    uint32_t gridDims;        // 2 or 3
    uint32_t nx, ny, nz;     // grid extents (nz=1 for 2-D grids)
    float    originX, originY, originZ;
    float    spacingX, spacingY, spacingZ;
    float    ueHeight;
    uint32_t _pad0;           // explicit alignment pad before double
    double   carrierHz;       // carrier frequency f_c (Hz)
    uint32_t pMax;            // max paths/link (stride)
    uint32_t pathRecBytes;    // sizeof(PathRecord) = 40
    uint32_t linkBlockBytes;  // = linkBlockStride(pMax)
    float    normRefDb;       // normalization reference (NaN = none, G.6.5)
    float    pruneThreshDb;   // prune threshold (dB, G.6.1)
    float    maxExcessDelayS; // max excess delay (s, G.6.4)
};
static_assert(sizeof(CirHeader) == 96, "CirHeader must be 96 B");

// Magic bytes and version.
inline constexpr uint32_t kCirVersion = 1u;

inline bool cirMagicOk(const CirHeader& h) {
    return std::memcmp(h.magic, "ORCACIR1", 8) == 0;
}

// Number of grid points.
inline uint64_t numGp(const CirHeader& h) {
    return uint64_t{h.nx} * h.ny * h.nz;
}

// Byte offset into the link-block region for (cell, gp).
inline uint64_t linkOffset(const CirHeader& h, uint32_t cell, uint64_t gp) {
    return (uint64_t{cell} * numGp(h) + gp) * h.linkBlockBytes;
}

}  // namespace channel
}  // namespace orca
