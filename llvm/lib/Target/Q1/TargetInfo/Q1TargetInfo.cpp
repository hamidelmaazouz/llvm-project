#include "Q1TargetInfo.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

Target &llvm::getTheQ1Target() {
  static Target TheQ1Target;
  return TheQ1Target;
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeQ1TargetInfo() {
  RegisterTarget<Triple::q1, /*HasJIT=*/true> X(getTheQ1Target(), "q1",
                                                "Q1 Sequence processor", "Q1");
}
