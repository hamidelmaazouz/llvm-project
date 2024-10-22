#ifndef LLVM_LIB_TARGET_Q1_MCTARGETDESC_Q1MCASMINFO_H
#define LLVM_LIB_TARGET_Q1_MCTARGETDESC_Q1MCASMINFO_H

#include "llvm/MC/MCAsmInfoELF.h"

namespace llvm {
class Triple;

class Q1MCAsmInfo : public MCAsmInfoELF {
  void anchor() override;

public:
  explicit Q1MCAsmInfo(const Triple &TargetTriple);
};

} // namespace llvm

#endif
