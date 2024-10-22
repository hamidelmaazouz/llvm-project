#ifndef LLVM_LIB_TARGET_Q1_MCTARGETDESC_Q1MCASMINFO_H
#define LLVM_LIB_TARGET_Q1_MCTARGETDESC_Q1MCASMINFO_H

#include "llvm/MC/MCAsmInfoELF.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
class Triple;

class Q1MCAsmInfo : public MCAsmInfoELF {
public:
  explicit Q1MCAsmInfo(const Triple &TT);
};

} // namespace llvm

#endif
