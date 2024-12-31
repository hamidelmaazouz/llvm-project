#include "Q1FrameLowering.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/Support/Alignment.h"

using namespace llvm;

Q1FrameLowering::Q1FrameLowering()
    : TargetFrameLowering(TargetFrameLowering::StackGrowsDown, Align(4), 0,
                          Align(4), false) {}

void Q1FrameLowering::emitPrologue(MachineFunction &MF,
                                   MachineBasicBlock &MBB) const {}

void Q1FrameLowering::emitEpilogue(MachineFunction &MF,
                                   MachineBasicBlock &MBB) const {}

bool Q1FrameLowering::hasFPImpl(const MachineFunction &MF) const {
  return false;
}
