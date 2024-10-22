#ifndef LLVM_Q1FRAMELOWERING_H
#define LLVM_Q1FRAMELOWERING_H

#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/Support/TypeSize.h"

namespace llvm {
class Q1FrameLowering : public TargetFrameLowering {

public:
  void emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
};
} // namespace llvm

#endif // LLVM_Q1FRAMELOWERING_H
