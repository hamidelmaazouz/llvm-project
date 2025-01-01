#include "Q1AsmPrinter.h"
#include "MCTargetDesc/Q1MCTargetDesc.h"
#include "TargetInfo/Q1TargetInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

void Q1AsmPrinter::emitInstruction(const MachineInstr *MI) {
  MCInst OutInst;
  lowerInstr(MI, OutInst);
  EmitToStreamer(*OutStreamer, OutInst);
}

bool Q1AsmPrinter::lowerOperand(const MachineOperand &MO,
                                MCOperand &OutMOp) const {
  switch (MO.getType()) {
  default:
    report_fatal_error("lowerOperand: unknown operand type");
  case MachineOperand::MO_Register:
    OutMOp = MCOperand::createReg(MO.getReg());
    break;
  case MachineOperand::MO_Immediate:
    OutMOp = MCOperand::createImm(MO.getImm());
    break;
  }
  return true;
}

void Q1AsmPrinter::lowerInstr(const MachineInstr *MI, MCInst &OutMI) const {
  OutMI.setOpcode(MI->getOpcode());
  for (auto &MO : MI->operands()) {
    MCOperand MCOp;
    if (lowerOperand(MO, MCOp))
      OutMI.addOperand(MCOp);
  }
}

// Force static initialization.
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeQ1AsmPrinter() {
  RegisterAsmPrinter<Q1AsmPrinter> X(getTheQ1Target());
}
