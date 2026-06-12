#pragma once
// Spec B §B.2 — the 20-byte fixed ORU-fronthaul header. Every field is
// byte-aligned; multi-byte fields are network byte order on the wire. The
// pack/unpack below write bytes explicitly, so they are host-endianness-
// independent and bit-exact (the golden-model contract).

#include <cstdint>

#include "common/dims.hpp"

namespace orca {
namespace fh {

constexpr uint8_t kFhVersion = 1;
constexpr uint32_t kHeaderBytes = 20;

enum class MsgType : uint8_t {  // §B.2 msgtyp
    UPlane = 0,
    CPlane = 1,
    SPlane = 2,
    Telemetry = 3,
};

enum class FhDir : uint8_t { DL = 0, UL = 1 };  // §B.2 dir

enum class Cmp : uint8_t {  // §B.2 cmp
    Int16 = 0,
    Bfp = 1,
    Int12 = 2,
    Reserved = 3,
};

// Host-order view of the header (wire layout in pack/unpack).
struct FhHeader {
    uint8_t ver;          // 4 bits used
    MsgType msgtyp;       // 4 bits used
    FhDir dir;            // 4 bits used
    Cmp cmp;              // 4 bits used
    uint8_t iqWidth;      // bits per I or Q
    uint8_t reserved0;
    uint16_t eAxC;
    uint8_t sectionId;
    uint8_t numSections;  // 0 = unknown
    uint16_t sfn;
    uint8_t slot;
    uint8_t sym;          // 0..13
    uint16_t startPrb;
    uint16_t numPrb;
    uint16_t seqNum;
    uint16_t udCompParam;
};

// --- big-endian byte helpers ---
inline void storeBe16(uint8_t* p, uint16_t v) {
    p[0] = static_cast<uint8_t>(v >> 8);
    p[1] = static_cast<uint8_t>(v & 0xFF);
}
inline uint16_t loadBe16(const uint8_t* p) {
    return static_cast<uint16_t>((uint16_t{p[0]} << 8) | p[1]);
}

// Serializes into exactly kHeaderBytes at `out`.
inline void pack(const FhHeader& h, uint8_t* out) {
    out[0] = static_cast<uint8_t>((h.ver & 0x0F) << 4) |
             (static_cast<uint8_t>(h.msgtyp) & 0x0F);
    out[1] = static_cast<uint8_t>((static_cast<uint8_t>(h.dir) & 0x0F) << 4) |
             (static_cast<uint8_t>(h.cmp) & 0x0F);
    out[2] = h.iqWidth;
    out[3] = h.reserved0;
    storeBe16(out + 4, h.eAxC);
    out[6] = h.sectionId;
    out[7] = h.numSections;
    storeBe16(out + 8, h.sfn);
    out[10] = h.slot;
    out[11] = h.sym;
    storeBe16(out + 12, h.startPrb);
    storeBe16(out + 14, h.numPrb);
    storeBe16(out + 16, h.seqNum);
    storeBe16(out + 18, h.udCompParam);
}

// Accepts a decoded header for processing: known version/enums and in-range
// symbol/PRB fields. unpack() itself is a raw decoder (bit-exact, never
// rejects); callers must validate() before acting on the framing.
inline bool validate(const FhHeader& h) {
    if (h.ver != kFhVersion) return false;
    if (static_cast<uint8_t>(h.msgtyp) > static_cast<uint8_t>(MsgType::Telemetry))
        return false;
    if (static_cast<uint8_t>(h.dir) > static_cast<uint8_t>(FhDir::UL)) return false;
    if (h.cmp == Cmp::Reserved ||
        static_cast<uint8_t>(h.cmp) > static_cast<uint8_t>(Cmp::Reserved))
        return false;
    if (h.sym > 13) return false;
    if (h.slot >= dims::slotsPerFrame(dims::mu)) return false;
    // PRB range gates the reassembly path; only U-plane sections carry IQ
    // (S-plane/Telemetry legitimately leave the PRB fields 0).
    if (h.msgtyp == MsgType::UPlane) {
        if (h.startPrb >= dims::numPrb) return false;
        if (h.numPrb == 0 || h.numPrb > dims::numPrb - h.startPrb) return false;
    }
    return true;
}

// Deserializes exactly kHeaderBytes from `in`.
inline FhHeader unpack(const uint8_t* in) {
    FhHeader h;
    h.ver = in[0] >> 4;
    h.msgtyp = static_cast<MsgType>(in[0] & 0x0F);
    h.dir = static_cast<FhDir>(in[1] >> 4);
    h.cmp = static_cast<Cmp>(in[1] & 0x0F);
    h.iqWidth = in[2];
    h.reserved0 = in[3];
    h.eAxC = loadBe16(in + 4);
    h.sectionId = in[6];
    h.numSections = in[7];
    h.sfn = loadBe16(in + 8);
    h.slot = in[10];
    h.sym = in[11];
    h.startPrb = loadBe16(in + 12);
    h.numPrb = loadBe16(in + 14);
    h.seqNum = loadBe16(in + 16);
    h.udCompParam = loadBe16(in + 18);
    return h;
}

}  // namespace fh
}  // namespace orca
