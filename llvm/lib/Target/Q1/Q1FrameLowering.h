#ifndef LLVM_LIB_TARGET_Q1_Q1FRAMELOWERING_H
#define LLVM_LIB_TARGET_Q1_Q1FRAMELOWERING_H

#include "llvm/CodeGen/TargetFrameLowering.h"

namespace llvm {

class Q1FrameLowering : public TargetFrameLowering {
public:
  Q1FrameLowering();

  void emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override;

protected:
  bool hasFPImpl(const MachineFunction &MF) const override;
};
} // namespace llvm

#endif
