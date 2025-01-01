#ifndef LLVM_LIB_TARGET_Q1_Q1TARGETMACHINE_H
#define LLVM_LIB_TARGET_Q1_Q1TARGETMACHINE_H

#include "MCTargetDesc/Q1MCTargetDesc.h"
#include "Q1Subtarget.h"
#include "llvm/CodeGen/CodeGenTargetMachineImpl.h"
#include "llvm/CodeGen/SelectionDAGTargetInfo.h"
#include "llvm/Target/TargetLoweringObjectFile.h"

namespace llvm {
class Q1TargetMachine : public CodeGenTargetMachineImpl {
  std::unique_ptr<TargetLoweringObjectFile> TLOF;
  mutable StringMap<std::unique_ptr<Q1Subtarget>> SubtargetMap;

public:
  Q1TargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                  StringRef FS, const TargetOptions &Options,
                  std::optional<Reloc::Model> RM,
                  std::optional<CodeModel::Model> CM, CodeGenOptLevel OL,
                  bool JIT);
  ~Q1TargetMachine() override;

  const Q1Subtarget *getSubtargetImpl(const Function &F) const override;

  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  TargetLoweringObjectFile *getObjFileLowering() const override {
    return TLOF.get();
  }
};
} // namespace llvm

#endif
