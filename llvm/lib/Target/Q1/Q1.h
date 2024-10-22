#ifndef LLVM_LIB_TARGET_Q1_Q1_H
#define LLVM_LIB_TARGET_Q1_Q1_H

#include "MCTargetDesc/Q1MCTargetDesc.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class FunctionPass;
class TargetMachine;
class Q1TargetMachine;

FunctionPass *createQ1ISelDag(Q1TargetMachine &TM, CodeGenOptLevel OptLevel);
} // namespace llvm

#endif // LLVM_LIB_TARGET_Q1_Q1_H
