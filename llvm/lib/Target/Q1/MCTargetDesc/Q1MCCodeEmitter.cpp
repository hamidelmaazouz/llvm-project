#include "MCTargetDesc/Q1MCTargetDesc.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace llvm;

#define DEBUG_TYPE "mccodeemitter"

STATISTIC(MCNumEmitted, "Number of MC instructions emitted");

namespace {

class Q1MCCodeEmitter : public MCCodeEmitter {
  Q1MCCodeEmitter(const Q1MCCodeEmitter &) = delete;
  void operator=(const Q1MCCodeEmitter &) = delete;
  MCContext &Ctx;
  MCInstrInfo const &MCII;

public:
  Q1MCCodeEmitter(MCContext &Ctx, MCInstrInfo const &MCII)
      : Ctx(Ctx), MCII(MCII) {}

  ~Q1MCCodeEmitter() override = default;

  void encodeInstruction(const MCInst &MI, SmallVectorImpl<char> &CB,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const override;

  /// TableGen'erated function for getting the binary encoding for an
  /// instruction.
  uint64_t getBinaryCodeForInstr(const MCInst &MI,
                                 SmallVectorImpl<MCFixup> &Fixups,
                                 const MCSubtargetInfo &STI) const;

  /// Return binary encoding of operand. If the machine operand requires
  /// relocation, record the relocation and return zero.
  uint64_t getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                             SmallVectorImpl<MCFixup> &Fixups,
                             const MCSubtargetInfo &STI) const;
};
} // namespace

void Q1MCCodeEmitter::encodeInstruction(const MCInst &MI,
                                        SmallVectorImpl<char> &CB,
                                        SmallVectorImpl<MCFixup> &Fixups,
                                        const MCSubtargetInfo &STI) const {
  uint64_t Bits = getBinaryCodeForInstr(MI, Fixups, STI);

  support::endian::write<uint32_t>(CB, Bits, llvm::endianness::little);

  ++MCNumEmitted; // Keep track of the # of mi's emitted.
}

uint64_t Q1MCCodeEmitter::getMachineOpValue(const MCInst &MI,
                                            const MCOperand &MO,
                                            SmallVectorImpl<MCFixup> &Fixups,
                                            const MCSubtargetInfo &STI) const {
  if (MO.isReg())
    return Ctx.getRegisterInfo()->getEncodingValue(MO.getReg());
  if (MO.isImm())
    return static_cast<uint64_t>(MO.getImm());
  return 0;
}

MCCodeEmitter *llvm::createQ1MCCodeEmitter(const MCInstrInfo &MCII,
                                           MCContext &Ctx) {
  return new Q1MCCodeEmitter(Ctx, MCII);
}

#include "Q1GenMCCodeEmitter.inc"