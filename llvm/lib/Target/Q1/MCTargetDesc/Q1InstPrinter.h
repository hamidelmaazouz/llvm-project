#ifndef LLVM_LIB_TARGET_Q1_MCTARGETDESC_Q1INSTPRINTER_H
#define LLVM_LIB_TARGET_Q1_MCTARGETDESC_Q1INSTPRINTER_H

// #include "MCTargetDesc/Q1MCTargetDesc.h"
#include "llvm/MC/MCInstPrinter.h"

namespace llvm {

class Q1InstPrinter : public MCInstPrinter {
public:
  Q1InstPrinter(const MCAsmInfo &MAI, const MCInstrInfo &MII,
                const MCRegisterInfo &MRI)
      : MCInstPrinter(MAI, MII, MRI) {}

  void printInst(const MCInst *MI, uint64_t Address, StringRef Annot,
                 const MCSubtargetInfo &STI, raw_ostream &O) override;
  void printRegName(raw_ostream &O, MCRegister Reg) const override;

  void printOperand(const MCInst *MI, unsigned OpNo, raw_ostream &O);
  void printRegReg(const MCInst *MI, unsigned OpNo, const MCSubtargetInfo &STI,
                   raw_ostream &O);
  static const char *getRegisterName(MCRegister Reg);
};
} // namespace llvm

#endif
