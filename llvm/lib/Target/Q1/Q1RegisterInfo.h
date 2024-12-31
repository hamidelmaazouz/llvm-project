#ifndef LLVM_LIB_TARGET_Q1_Q1REGISTERINFO_H
#define LLVM_LIB_TARGET_Q1_Q1REGISTERINFO_H

#include "Q1FrameLowering.h"

#define GET_REGINFO_HEADER
#include "Q1GenRegisterInfo.inc"

namespace llvm {

struct Q1RegisterInfo : public Q1GenRegisterInfo {
  Q1RegisterInfo();

  const MCPhysReg *getCalleeSavedRegs(const MachineFunction *MF) const override;

  BitVector getReservedRegs(const MachineFunction &MF) const override;

  bool eliminateFrameIndex(MachineBasicBlock::iterator MI, int SPAdj,
                           unsigned FIOperandNum,
                           RegScavenger *RS = nullptr) const override;

  Register getFrameRegister(const MachineFunction &MF) const override;
};

} // namespace llvm

#endif
