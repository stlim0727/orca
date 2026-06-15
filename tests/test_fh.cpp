// Sub-stage 1d test: Spec B framing — 20-byte header bit-exact round trip +
// known wire bytes (network order), eAxC encode/decode + cell extraction,
// U-plane int16 payload round trip, C-plane section round trip + truncation
// handling + Alloc mapping.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "fh/cplane.hpp"
#include "fh/eaxc.hpp"
#include "fh/fh_header.hpp"
#include "fh/uplane.hpp"
#include "tests/check.hpp"

using namespace orca;
using namespace orca::fh;

static void testHeader() {
    FhHeader h{};
    h.ver = kFhVersion;
    h.msgtyp = MsgType::CPlane;
    h.dir = FhDir::UL;
    h.cmp = Cmp::Bfp;
    h.iqWidth = 12;
    h.reserved0 = 0;
    h.eAxC = 0xBEEF;
    h.sectionId = 5;
    h.numSections = 9;
    h.sfn = 0x1234;
    h.slot = 19;
    h.sym = 13;
    h.startPrb = 271;
    h.numPrb = 2;
    h.seqNum = 0xCAFE;
    h.udCompParam = 0x0102;

    uint8_t buf[kHeaderBytes];
    pack(h, buf);

    // Known wire bytes: nibble packing and network byte order.
    CHECK(buf[0] == 0x11);  // ver=1 | msgtyp=1 (CPlane)
    CHECK(buf[1] == 0x11);  // dir=1 (UL) | cmp=1 (BFP)
    CHECK(buf[4] == 0xBE && buf[5] == 0xEF);    // eAxC big-endian
    CHECK(buf[8] == 0x12 && buf[9] == 0x34);    // sfn big-endian
    CHECK(buf[12] == 0x01 && buf[13] == 0x0F);  // startPrb 271 = 0x010F
    CHECK(buf[16] == 0xCA && buf[17] == 0xFE);  // seqNum

    const FhHeader r = unpack(buf);
    CHECK(r.ver == h.ver && r.msgtyp == h.msgtyp && r.dir == h.dir &&
          r.cmp == h.cmp);
    CHECK(r.iqWidth == h.iqWidth && r.eAxC == h.eAxC);
    CHECK(r.sectionId == h.sectionId && r.numSections == h.numSections);
    CHECK(r.sfn == h.sfn && r.slot == h.slot && r.sym == h.sym);
    CHECK(r.startPrb == h.startPrb && r.numPrb == h.numPrb);
    CHECK(r.seqNum == h.seqNum && r.udCompParam == h.udCompParam);

    // Re-pack is byte-identical (bit-exact contract).
    uint8_t buf2[kHeaderBytes];
    pack(r, buf2);
    CHECK(std::memcmp(buf, buf2, kHeaderBytes) == 0);

    // validate(): the C-plane header above is acceptable framing.
    CHECK(validate(r));
    FhHeader bad = r;
    bad.ver = 2;
    CHECK(!validate(bad));
    bad = r;
    bad.cmp = Cmp::Reserved;
    CHECK(!validate(bad));
    bad = r;
    bad.sym = 14;
    CHECK(!validate(bad));
    bad = r;
    bad.slot = 20;  // µ=1: slots 0..19
    CHECK(!validate(bad));

    // U-plane PRB range is gated; S-plane with zero PRB fields is fine.
    FhHeader up = r;
    up.msgtyp = MsgType::UPlane;
    up.startPrb = 272;
    up.numPrb = 2;  // 272+2 > 273
    CHECK(!validate(up));
    up.numPrb = 1;
    CHECK(validate(up));
    up.numPrb = 0;
    CHECK(!validate(up));
    FhHeader sp = r;
    sp.msgtyp = MsgType::SPlane;
    sp.startPrb = 0;
    sp.numPrb = 0;
    CHECK(validate(sp));
}

static void testEaxc() {
    // Default widths 4/2/2/8, MSB→LSB.
    const Eaxc e{0xA, 0x2, 0x1, 0x7F};
    const uint16_t v = encodeEaxc(e);
    // 1010 | 10 | 01 | 01111111 = 0xA9 0x7F
    CHECK(v == 0xA97F);

    const Eaxc d = decodeEaxc(v);
    CHECK(d.duPort == 0xA && d.bandSector == 0x2 && d.ccId == 0x1 &&
          d.ruPort == 0x7F);

    // Cell = {BandSector, CC_ID} = (2<<2)|1 = 9.
    CHECK(cellOf(v) == 9);

    // Fields are masked to their widths (no bleed into neighbors).
    const uint16_t w = encodeEaxc(Eaxc{0xFF, 0xFF, 0xFF, 0x1FF});
    const Eaxc m = decodeEaxc(w);
    CHECK(m.duPort == 0xF && m.bandSector == 0x3 && m.ccId == 0x3 &&
          m.ruPort == 0xFF);

    // Non-default widths (more cells): 2/4/2/8.
    const EaxcConfig wide{2, 4, 2, 8};
    CHECK(wide.valid());
    const uint16_t v2 = encodeEaxc(Eaxc{1, 0xB, 2, 3}, wide);
    const Eaxc d2 = decodeEaxc(v2, wide);
    CHECK(d2.duPort == 1 && d2.bandSector == 0xB && d2.ccId == 2 &&
          d2.ruPort == 3);
    CHECK(cellOf(v2, wide) == ((0xB << 2) | 2));

    // Degenerate-but-valid config: one full-width 16-bit field (mask(16)).
    const EaxcConfig full{0, 0, 0, 16};
    CHECK(full.valid());
    const uint16_t v3 = encodeEaxc(Eaxc{0, 0, 0, 0xBEEF}, full);
    CHECK(v3 == 0xBEEF);
    CHECK(decodeEaxc(v3, full).ruPort == 0xBEEF);
    CHECK(cellOf(v3, full) == 0);

    // Invalid configs are detectable before use.
    const EaxcConfig badCfg{8, 8, 8, 8};
    CHECK(!badCfg.valid());
}

static void testUplane() {
    constexpr uint32_t kSc = 24;  // 2 PRBs
    ci16 src[kSc], back[kSc];
    for (uint32_t i = 0; i < kSc; ++i)
        src[i] = ci16{int16_t(int(i) * 100 - 1200), int16_t(-int(i) * 7)};
    src[0] = ci16{INT16_MIN, INT16_MAX};  // extremes survive the wire

    uint8_t wire[uplaneBytes(2)];
    CHECK(sizeof(wire) == kSc * kBytesPerSc);
    CHECK(uplaneBytes(dims::numPrb + 1) == 0);  // out-of-range → malformed
    packUplane(src, kSc, wire);

    // Spot-check network order: INT16_MIN = 0x8000.
    CHECK(wire[0] == 0x80 && wire[1] == 0x00);

    unpackUplane(wire, kSc, back);
    for (uint32_t i = 0; i < kSc; ++i)
        CHECK(back[i].re == src[i].re && back[i].im == src[i].im);
}

static void testCplane() {
    CplaneSection s{};
    s.prbStart = 100;
    s.numPrb = 8;
    s.ueId = 17;
    s.numLayers = 3;
    s.dir = 1;
    s.beamId[0] = 1000;
    s.beamId[1] = 1001;
    s.beamId[2] = 1002;

    uint8_t buf[64];
    const uint32_t written = packCplane(s, buf, sizeof(buf));
    CHECK(written == 8 + 2 * 3);

    CplaneSection r{};
    CHECK(parseCplane(buf, written, r) == written);
    CHECK(r.prbStart == 100 && r.numPrb == 8 && r.ueId == 17);
    CHECK(r.numLayers == 3 && r.dir == 1);
    CHECK(r.beamId[0] == 1000 && r.beamId[1] == 1001 && r.beamId[2] == 1002);
    CHECK(r.beamId[3] == 0);  // unused layers zeroed

    // Truncated input is rejected at both lengths (header and beam list).
    CHECK(parseCplane(buf, 7, r) == 0);
    CHECK(parseCplane(buf, written - 1, r) == 0);

    // Invalid rank rejected on both paths.
    CplaneSection bad = s;
    bad.numLayers = 5;
    CHECK(packCplane(bad, buf, sizeof(buf)) == 0);
    buf[6] = 0;  // corrupt rank on the wire
    CHECK(parseCplane(buf, written, r) == 0);

    // Alloc mapping: PRB → subcarrier range, cell from eAxC.
    Alloc a{};
    CHECK(toAlloc(s, /*cell=*/1, a));
    CHECK(a.cell == 1 && a.ueId == 17);
    CHECK(a.scStart == 1200 && a.scLen == 96);
    CHECK(a.dir == 1 && a.rank == 3);
    CHECK(a.beamId[2] == 1002 && a.beamId[3] == 0);

    // Out-of-carrier PRB ranges never become allocations.
    CplaneSection oob = s;
    oob.prbStart = 270;
    oob.numPrb = 4;  // 270+4 > 273
    CHECK(!toAlloc(oob, 1, a));
    oob.prbStart = 273;
    oob.numPrb = 1;
    CHECK(!toAlloc(oob, 1, a));
    oob = s;
    oob.numPrb = 0;
    CHECK(!toAlloc(oob, 1, a));

    // Capacity too small → pack refuses.
    CHECK(packCplane(s, buf, 10) == 0);
}

int main() {
    testHeader();
    testEaxc();
    testUplane();
    testCplane();

    return orca::test::report("test_fh");
}
