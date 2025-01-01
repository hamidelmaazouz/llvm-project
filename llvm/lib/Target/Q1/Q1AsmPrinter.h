#ifndef LLVM_LIB_TARGET_Q1_Q1ASMPRINTER_H
#define LLVM_LIB_TARGET_Q1_Q1ASMPRINTER_H

#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/MC/MCStreamer.h"

using namespace llvm;

namespace {
class Q1AsmPrinter : public AsmPrinter {
public:
  explicit Q1AsmPrinter(TargetMachine &TM, std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer)) {}

  StringRef getPassName() const override { return "Q1 Assembly Printer"; }

  void emitInstruction(const MachineInstr *MI) override;

  /// Lower a MachineInstr to an MCInst
  void lowerInstr(const MachineInstr *MI, MCInst &OutMI) const;

  /// Lower a MachineOperand to an MCOperand
  bool lowerOperand(const MachineOperand &MO, MCOperand &OutMOp) const;
};
} // namespace

#endif // LLVM_LIB_TARGET_Q1_Q1ASMPRINTER_H
