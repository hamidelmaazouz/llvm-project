#ifndef LLVM_LIB_TARGET_Q1_Q1INSTRINFO_H
#define LLVM_LIB_TARGET_Q1_Q1INSTRINFO_H

#include "Q1RegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "Q1GenInstrInfo.inc"

namespace llvm {

class Q1Subtarget;

class Q1InstrInfo : public Q1GenInstrInfo {
protected:
  const Q1Subtarget &STI;
  const Q1RegisterInfo RI;

  virtual void anchor();

public:
  explicit Q1InstrInfo(Q1Subtarget &STI);
  const Q1RegisterInfo &getRegisterInfo() const { return RI; }
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_Q1_Q1INSTRINFO_H
