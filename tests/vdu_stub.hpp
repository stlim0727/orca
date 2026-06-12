#pragma once
// Synthetic vDU (AGENT.md 1e): emits DL symbols with a deterministic IQ
// pattern as Spec B packets (C-plane first, then per-layer U-plane sections)
// and verifies that UL packets return the same IQ bit-exact (the Stage-1
// identity loopback contract).

#include <cstdint>
#include <vector>

#include "common/complex.hpp"
#include "common/dims.hpp"
#include "fh/cplane.hpp"
#include "fh/eaxc.hpp"
#include "fh/fh_header.hpp"
#include "fh/uplane.hpp"

namespace orca {
namespace testutil {

class VduStub {
  public:
    explicit VduStub(fh::EaxcConfig eaxc = {}) : eaxc_(eaxc) {}

    // Deterministic per-sample pattern — pure function of the symbol key,
    // layer and subcarrier (golden-model style; int16-exact).
    static ci16 patternIq(uint16_t sfn, uint8_t slot, uint8_t sym,
                          uint8_t layer, uint32_t sc) {
        const uint32_t x = (uint32_t{sfn} * 131u + uint32_t{slot} * 17u +
                            uint32_t{sym} * 7u + uint32_t{layer} * 1009u + sc);
        return ci16{static_cast<int16_t>((x * 2654435761u) >> 17),
                    static_cast<int16_t>((x * 40503u) >> 16)};
    }

    // Builds one DL symbol for `cell`: 1 C-plane packet (full-band, rank
    // layers) + rank · numSections U-plane packets. numSections splits the
    // band to exercise reassembly.
    std::vector<std::vector<uint8_t>> buildDlSymbol(uint16_t sfn, uint8_t slot,
                                                    uint8_t sym, uint8_t rank,
                                                    uint32_t numSections,
                                                    uint16_t cell = 0,
                                                    uint16_t ueId = 1) {
        std::vector<std::vector<uint8_t>> pkts;
        pkts.push_back(buildCplane(sfn, slot, sym, rank, cell, ueId));
        for (uint8_t l = 0; l < rank; ++l) {
            uint16_t prb = 0;
            for (uint32_t sec = 0; sec < numSections; ++sec) {
                const uint16_t left = static_cast<uint16_t>(dims::numPrb - prb);
                const uint16_t take = static_cast<uint16_t>(
                    sec + 1 == numSections
                        ? left
                        : dims::numPrb / numSections);
                pkts.push_back(
                    buildUplane(sfn, slot, sym, l, prb, take, cell, sec,
                                static_cast<uint8_t>(numSections)));
                prb = static_cast<uint16_t>(prb + take);
            }
        }
        return pkts;
    }

    // Verifies one UL U-plane packet bit-exact against the pattern. Returns
    // the number of mismatched samples (0 = exact); -1 = malformed packet.
    int verifyUlPacket(const uint8_t* data, uint32_t len) const {
        if (len < fh::kHeaderBytes) return -1;
        const fh::FhHeader h = fh::unpack(data);
        if (!fh::validate(h) || h.msgtyp != fh::MsgType::UPlane ||
            h.dir != fh::FhDir::UL)
            return -1;
        const uint32_t want = fh::uplaneBytes(h.numPrb);
        if (want == 0 || len != fh::kHeaderBytes + want) return -1;
        const uint8_t layer =
            static_cast<uint8_t>(fh::decodeEaxc(h.eAxC, eaxc_).ruPort);
        const uint32_t numSc = h.numPrb * dims::scPerPrb;
        std::vector<ci16> rx(numSc);
        fh::unpackUplane(data + fh::kHeaderBytes, numSc, rx.data());
        int bad = 0;
        for (uint32_t i = 0; i < numSc; ++i) {
            const uint32_t sc = h.startPrb * dims::scPerPrb + i;
            const ci16 e = patternIq(h.sfn, h.slot, h.sym, layer, sc);
            if (rx[i].re != e.re || rx[i].im != e.im) ++bad;
        }
        return bad;
    }

  private:
    std::vector<uint8_t> buildCplane(uint16_t sfn, uint8_t slot, uint8_t sym,
                                     uint8_t rank, uint16_t cell,
                                     uint16_t ueId) {
        fh::CplaneSection s{};
        s.prbStart = 0;
        s.numPrb = dims::numPrb;
        s.ueId = ueId;
        s.numLayers = rank;
        s.dir = 0;  // DL
        for (uint8_t l = 0; l < rank; ++l)
            s.beamId[l] = static_cast<uint16_t>(100 + l);

        std::vector<uint8_t> pkt(fh::kHeaderBytes + fh::cplaneBytes(rank));
        fh::FhHeader h = baseHeader(sfn, slot, sym, cell, /*layer=*/0);
        h.msgtyp = fh::MsgType::CPlane;
        fh::pack(h, pkt.data());
        fh::packCplane(s, pkt.data() + fh::kHeaderBytes,
                       static_cast<uint32_t>(pkt.size() - fh::kHeaderBytes));
        return pkt;
    }

    std::vector<uint8_t> buildUplane(uint16_t sfn, uint8_t slot, uint8_t sym,
                                     uint8_t layer, uint16_t startPrb,
                                     uint16_t numPrb, uint16_t cell,
                                     uint32_t sectionId, uint8_t numSections) {
        const uint32_t numSc = numPrb * dims::scPerPrb;
        std::vector<ci16> iq(numSc);
        for (uint32_t i = 0; i < numSc; ++i)
            iq[i] = patternIq(sfn, slot, sym, layer,
                              startPrb * dims::scPerPrb + i);

        std::vector<uint8_t> pkt(fh::kHeaderBytes + fh::uplaneBytes(numPrb));
        fh::FhHeader h = baseHeader(sfn, slot, sym, cell, layer);
        h.msgtyp = fh::MsgType::UPlane;
        h.sectionId = static_cast<uint8_t>(sectionId);
        h.numSections = numSections;
        h.startPrb = startPrb;
        h.numPrb = numPrb;
        fh::pack(h, pkt.data());
        fh::packUplane(iq.data(), numSc, pkt.data() + fh::kHeaderBytes);
        return pkt;
    }

    fh::FhHeader baseHeader(uint16_t sfn, uint8_t slot, uint8_t sym,
                            uint16_t cell, uint8_t layer) const {
        fh::FhHeader h{};
        h.ver = fh::kFhVersion;
        h.dir = fh::FhDir::DL;
        h.cmp = fh::Cmp::Int16;
        h.iqWidth = 16;
        h.eAxC = fh::encodeEaxc(
            fh::Eaxc{0, static_cast<uint16_t>(cell >> eaxc_.ccIdBits),
                     static_cast<uint16_t>(cell & fh::mask(eaxc_.ccIdBits)),
                     layer},
            eaxc_);
        h.sfn = sfn;
        h.slot = slot;
        h.sym = sym;
        h.startPrb = 0;
        h.numPrb = dims::numPrb;
        h.seqNum = seq_++;
        return h;
    }

    fh::EaxcConfig eaxc_;
    mutable uint16_t seq_ = 0;
};

}  // namespace testutil
}  // namespace orca
