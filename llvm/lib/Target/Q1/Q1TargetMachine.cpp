#include "Q1TargetMachine.h"
#include "Q1.h"
#include "TargetInfo/Q1TargetInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeQ1Target() {
  RegisterTargetMachine<Q1TargetMachine> X(getTheQ1Target());
  auto *PR = PassRegistry::getPassRegistry();
  initializeQ1DAGToDAGISelLegacyPass(*PR);
}

namespace {
std::string computeDataLayout(const Triple &TT, const TargetOptions &Options) {
  std::string Ret = "";
  // Dummy - Bytes in memory are stored using the little-endian schema
  Ret += "e";

  // Dummy - Name mangling
  Ret += DataLayout::getManglingComponent(TT);

  // Dummy - Data alignment
  Ret += "-i8:8:32-i16:16:32";

  // Dummy - Native register sizes
  Ret += "-n32";

  return Ret;
}

Reloc::Model getEffectiveRelocModel(const Triple &TT,
                                    std::optional<Reloc::Model> RM) {
  return RM.value_or(Reloc::Static);
}
} // namespace

Q1TargetMachine::Q1TargetMachine(const Target &T, const Triple &TT,
                                 StringRef CPU, StringRef FS,
                                 const TargetOptions &Options,
                                 std::optional<Reloc::Model> RM,
                                 std::optional<CodeModel::Model> CM,
                                 CodeGenOptLevel OL, bool JIT)
    : CodeGenTargetMachineImpl(T, computeDataLayout(TT, Options), TT, CPU, FS,
                               Options, getEffectiveRelocModel(TT, RM),
                               getEffectiveCodeModel(CM, CodeModel::Small), OL),
      TLOF(std::make_unique<TargetLoweringObjectFileELF>()) {
  initAsmInfo();
}

Q1TargetMachine::~Q1TargetMachine() = default;

const Q1Subtarget *Q1TargetMachine::getSubtargetImpl(const Function &F) const {
  Attribute CPUAttr = F.getFnAttribute("target-cpu");
  Attribute FSAttr = F.getFnAttribute("target-features");

  std::string CPU = !CPUAttr.hasAttribute(Attribute::None)
                        ? CPUAttr.getValueAsString().str()
                        : TargetCPU;
  std::string FS = !FSAttr.hasAttribute(Attribute::None)
                       ? FSAttr.getValueAsString().str()
                       : TargetFS;

  auto &I = SubtargetMap[CPU + FS];
  if (!I) {
    // This needs to be done before we create a new
    // subtarget since any creation will depend on the
    // TM and the code generation flags on the function
    // that reside in TargetOptions.
    resetTargetOptions(F);
    I = std::make_unique<Q1Subtarget>(TargetTriple, CPU, FS, *this);
  }
  return I.get();
}

namespace {
class Q1PassConfig : public TargetPassConfig {
public:
  Q1PassConfig(Q1TargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {}

  bool addInstSelector() override;
  void addPreEmitPass() override;
};
} // namespace

TargetPassConfig *Q1TargetMachine::createPassConfig(PassManagerBase &PM) {
  return new Q1PassConfig(*this, PM);
}

bool Q1PassConfig::addInstSelector() {
  addPass(createQ1ISelDag(getTM<Q1TargetMachine>(), getOptLevel()));
  return false;
}

void Q1PassConfig::addPreEmitPass() {
  // TODO Add pass for div-by-zero check.
}
