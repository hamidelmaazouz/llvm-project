#include "TargetInfo/Q1TargetInfo.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

Target &llvm::getTheQ1Target() {
  static Target TheQ1Target;
  return TheQ1Target;
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeQ1TargetInfo() {
  RegisterTarget<Triple::q1, /*HasJIT=*/false> X(getTheQ1Target(), "Q1",
                                                 "QBlox Q1 processor", "Q1");
}
