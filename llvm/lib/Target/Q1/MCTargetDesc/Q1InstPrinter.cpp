#include "Q1InstPrinter.h"
// #include "Q1BaseInfo.h"
// #include "Q1MCExpr.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
using namespace llvm;

#define DEBUG_TYPE "asm-printer"

// Include the auto-generated portion of the assembly writer.
#define PRINT_ALIAS_INSTR
// #include "Q1GenAsmWriter.inc"

void Q1InstPrinter::printInst(const MCInst *MI, uint64_t Address,
                              StringRef Annot, const MCSubtargetInfo &STI,
                              raw_ostream &O) {
  O << "\t";
  O << "Q1";
}

void Q1InstPrinter::printRegName(raw_ostream &O, MCRegister Reg) const {
  markup(O, Markup::Register) << getRegisterName(Reg);
}

void Q1InstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                 raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNo);

  if (MO.isReg()) {
    printRegName(O, MO.getReg());
    return;
  }

  if (MO.isImm()) {
    markup(O, Markup::Immediate) << formatImm(MO.getImm());
    return;
  }

  assert(MO.isExpr() && "Unknown operand kind in printOperand");
  MO.getExpr()->print(O, &MAI);
}

void Q1InstPrinter::printRegReg(const MCInst *MI, unsigned OpNo,
                                const MCSubtargetInfo &STI, raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNo);

  assert(MO.isReg() && "printRegReg can only print register operands");
  printRegName(O, MO.getReg());

  O << "(";
  const MCOperand &MO1 = MI->getOperand(OpNo + 1);
  assert(MO1.isReg() && "printRegReg can only print register operands");
  printRegName(O, MO1.getReg());
  O << ")";
}

const char *Q1InstPrinter::getRegisterName(MCRegister Reg) {
  unsigned RegNo = Reg.id();
  return ("R" + std::to_string(RegNo)).c_str();
}
