#include "Q1RegisterInfo.h"
#include "MCTargetDesc/Q1MCTargetDesc.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

#define GET_REGINFO_TARGET_DESC
#include "Q1GenRegisterInfo.inc"

using namespace llvm;

Q1RegisterInfo::Q1RegisterInfo() : Q1GenRegisterInfo(Q1::R0) {}

const MCPhysReg *
Q1RegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  return CSR_NoRegs_SaveList;
}

BitVector Q1RegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  return Reserved;
}

bool Q1RegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator MI,
                                         int SPAdj, unsigned FIOperandNum,
                                         RegScavenger *RS) const {
  return false;
}

Register Q1RegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  return Q1::R0;
}
