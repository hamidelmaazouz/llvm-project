#ifndef LLVM_LIB_TARGET_Q1_Q1_H
#define LLVM_LIB_TARGET_Q1_Q1_H

#include "llvm/Support/CodeGen.h"

namespace llvm {
class FunctionPass;
class InstructionSelector;
class PassRegistry;
class Q1RegisterBankInfo;
class Q1Subtarget;
class Q1TargetMachine;

FunctionPass *createQ1ISelDag(Q1TargetMachine &TM, CodeGenOptLevel OptLevel);

void initializeQ1DAGToDAGISelLegacyPass(PassRegistry &);

InstructionSelector *createQ1InstructionSelector(const Q1TargetMachine &TM,
                                                 const Q1Subtarget &,
                                                 const Q1RegisterBankInfo &);
} // namespace llvm
#endif
