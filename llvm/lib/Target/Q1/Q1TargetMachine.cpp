#include "Q1TargetMachine.h"
#include "MCTargetDesc/Q1BaseInfo.h"
#include "Q1.h"
#include "Q1MachineFunctionInfo.h"
#include "Q1TargetObjectFile.h"
#include "Q1TargetTransformInfo.h"
#include "TargetInfo/Q1TargetInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/GlobalISel/CSEInfo.h"
#include "llvm/CodeGen/GlobalISel/IRTranslator.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelect.h"
#include "llvm/CodeGen/GlobalISel/Legalizer.h"
#include "llvm/CodeGen/GlobalISel/RegBankSelect.h"
#include "llvm/CodeGen/MIRParser/MIParser.h"
#include "llvm/CodeGen/MIRYamlMapping.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/MacroFusion.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Vectorize/LoopIdiomVectorize.h"
#include <optional>
using namespace llvm;

static cl::opt<bool> EnableRedundantCopyElimination(
    "Q1-enable-copyelim",
    cl::desc("Enable the redundant copy elimination pass"), cl::init(true),
    cl::Hidden);

// FIXME: Unify control over GlobalMerge.
static cl::opt<cl::boolOrDefault>
    EnableGlobalMerge("Q1-enable-global-merge", cl::Hidden,
                      cl::desc("Enable the global merge pass"));

static cl::opt<bool>
    EnableMachineCombiner("Q1-enable-machine-combiner",
                          cl::desc("Enable the machine combiner pass"),
                          cl::init(true), cl::Hidden);

static cl::opt<unsigned> RVVVectorBitsMaxOpt(
    "Q1-v-vector-bits-max",
    cl::desc("Assume V extension vector registers are at most this big, "
             "with zero meaning no maximum size is assumed."),
    cl::init(0), cl::Hidden);

static cl::opt<int> RVVVectorBitsMinOpt(
    "Q1-v-vector-bits-min",
    cl::desc("Assume V extension vector registers are at least this big, "
             "with zero meaning no minimum size is assumed. A value of -1 "
             "means use Zvl*b extension. This is primarily used to enable "
             "autovectorization with fixed width vectors."),
    cl::init(-1), cl::Hidden);

static cl::opt<bool> EnableQ1CopyPropagation(
    "Q1-enable-copy-propagation",
    cl::desc("Enable the copy propagation with RISC-V copy instr"),
    cl::init(true), cl::Hidden);

static cl::opt<bool> EnableQ1DeadRegisterElimination(
    "Q1-enable-dead-defs", cl::Hidden,
    cl::desc("Enable the pass that removes dead"
             " definitons and replaces stores to"
             " them with stores to x0"),
    cl::init(true));

static cl::opt<bool>
    EnableSinkFold("Q1-enable-sink-fold",
                   cl::desc("Enable sinking and folding of instruction copies"),
                   cl::init(true), cl::Hidden);

static cl::opt<bool>
    EnableLoopDataPrefetch("Q1-enable-loop-data-prefetch", cl::Hidden,
                           cl::desc("Enable the loop data prefetch pass"),
                           cl::init(true));

static cl::opt<bool> EnableMISchedLoadClustering(
    "Q1-misched-load-clustering", cl::Hidden,
    cl::desc("Enable load clustering in the machine scheduler"),
    cl::init(true));

static cl::opt<bool> EnableVSETVLIAfterRVVRegAlloc(
    "Q1-vsetvl-after-rvv-regalloc", cl::Hidden,
    cl::desc("Insert vsetvls after vector register allocation"),
    cl::init(true));

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeQ1Target() {
  RegisterTargetMachine<Q1TargetMachine> X(getTheQ1Target());
  auto *PR = PassRegistry::getPassRegistry();
  initializeGlobalISel(*PR);
  initializeQ1O0PreLegalizerCombinerPass(*PR);
  initializeQ1PreLegalizerCombinerPass(*PR);
  initializeQ1PostLegalizerCombinerPass(*PR);
  initializeKCFIPass(*PR);
  initializeQ1DeadRegisterDefinitionsPass(*PR);
  initializeQ1MakeCompressibleOptPass(*PR);
  initializeQ1GatherScatterLoweringPass(*PR);
  initializeQ1CodeGenPreparePass(*PR);
  initializeQ1PostRAExpandPseudoPass(*PR);
  initializeQ1MergeBaseOffsetOptPass(*PR);
  initializeQ1OptWInstrsPass(*PR);
  initializeQ1PreRAExpandPseudoPass(*PR);
  initializeQ1ExpandPseudoPass(*PR);
  initializeQ1VectorPeepholePass(*PR);
  initializeQ1InsertVSETVLIPass(*PR);
  initializeQ1InsertReadWriteCSRPass(*PR);
  initializeQ1InsertWriteVXRMPass(*PR);
  initializeQ1DAGToDAGISelLegacyPass(*PR);
  initializeQ1MoveMergePass(*PR);
  initializeQ1PushPopOptPass(*PR);
}

static StringRef computeDataLayout(const Triple &TT,
                                   const TargetOptions &Options) {
  StringRef ABIName = Options.MCOptions.getABIName();
  if (TT.isArch64Bit()) {
    if (ABIName == "lp64e")
      return "e-m:e-p:64:64-i64:64-i128:128-n32:64-S64";

    return "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128";
  }
  assert(TT.isArch32Bit() && "only RV32 and RV64 are currently supported");

  if (ABIName == "ilp32e")
    return "e-m:e-p:32:32-i64:64-n32-S32";

  return "e-m:e-p:32:32-i64:64-n32-S128";
}

static Reloc::Model getEffectiveRelocModel(const Triple &TT,
                                           std::optional<Reloc::Model> RM) {
  return RM.value_or(Reloc::Static);
}

Q1TargetMachine::Q1TargetMachine(const Target &T, const Triple &TT,
                                       StringRef CPU, StringRef FS,
                                       const TargetOptions &Options,
                                       std::optional<Reloc::Model> RM,
                                       std::optional<CodeModel::Model> CM,
                                       CodeGenOptLevel OL, bool JIT)
    : LLVMTargetMachine(T, computeDataLayout(TT, Options), TT, CPU, FS, Options,
                        getEffectiveRelocModel(TT, RM),
                        getEffectiveCodeModel(CM, CodeModel::Small), OL),
      TLOF(std::make_unique<Q1ELFTargetObjectFile>()) {
  initAsmInfo();

  // RISC-V supports the MachineOutliner.
  setMachineOutliner(true);
  setSupportsDefaultOutlining(true);

  if (TT.isOSFuchsia() && !TT.isArch64Bit())
    report_fatal_error("Fuchsia is only supported for 64-bit");
}

MachineFunctionInfo *Q1TargetMachine::createMachineFunctionInfo(
    BumpPtrAllocator &Allocator, const Function &F,
    const TargetSubtargetInfo *STI) const {
  return Q1MachineFunctionInfo::create<Q1MachineFunctionInfo>(Allocator,
                                                                    F, STI);
}

TargetTransformInfo
Q1TargetMachine::getTargetTransformInfo(const Function &F) const {
  return TargetTransformInfo(Q1TTIImpl(this, F));
}

// A RISC-V hart has a single byte-addressable address space of 2^XLEN bytes
// for all memory accesses, so it is reasonable to assume that an
// implementation has no-op address space casts. If an implementation makes a
// change to this, they can override it here.
bool Q1TargetMachine::isNoopAddrSpaceCast(unsigned SrcAS,
                                             unsigned DstAS) const {
  return true;
}

namespace {

class RVVRegisterRegAlloc : public RegisterRegAllocBase<RVVRegisterRegAlloc> {
public:
  RVVRegisterRegAlloc(const char *N, const char *D, FunctionPassCtor C)
      : RegisterRegAllocBase(N, D, C) {}
};

static bool onlyAllocateRVVReg(const TargetRegisterInfo &TRI,
                               const MachineRegisterInfo &MRI,
                               const Register Reg) {
  const TargetRegisterClass *RC = MRI.getRegClass(Reg);
  return Q1RegisterInfo::isRVVRegClass(RC);
}

static FunctionPass *useDefaultRegisterAllocator() { return nullptr; }

static llvm::once_flag InitializeDefaultRVVRegisterAllocatorFlag;

/// -Q1-rvv-regalloc=<fast|basic|greedy> command line option.
/// This option could designate the rvv register allocator only.
/// For example: -Q1-rvv-regalloc=basic
static cl::opt<RVVRegisterRegAlloc::FunctionPassCtor, false,
               RegisterPassParser<RVVRegisterRegAlloc>>
    RVVRegAlloc("Q1-rvv-regalloc", cl::Hidden,
                cl::init(&useDefaultRegisterAllocator),
                cl::desc("Register allocator to use for RVV register."));

static void initializeDefaultRVVRegisterAllocatorOnce() {
  RegisterRegAlloc::FunctionPassCtor Ctor = RVVRegisterRegAlloc::getDefault();

  if (!Ctor) {
    Ctor = RVVRegAlloc;
    RVVRegisterRegAlloc::setDefault(RVVRegAlloc);
  }
}

static FunctionPass *createBasicRVVRegisterAllocator() {
  return createBasicRegisterAllocator(onlyAllocateRVVReg);
}

static FunctionPass *createGreedyRVVRegisterAllocator() {
  return createGreedyRegisterAllocator(onlyAllocateRVVReg);
}

static FunctionPass *createFastRVVRegisterAllocator() {
  return createFastRegisterAllocator(onlyAllocateRVVReg, false);
}

static RVVRegisterRegAlloc basicRegAllocRVVReg("basic",
                                               "basic register allocator",
                                               createBasicRVVRegisterAllocator);
static RVVRegisterRegAlloc
    greedyRegAllocRVVReg("greedy", "greedy register allocator",
                         createGreedyRVVRegisterAllocator);

static RVVRegisterRegAlloc fastRegAllocRVVReg("fast", "fast register allocator",
                                              createFastRVVRegisterAllocator);

class Q1PassConfig : public TargetPassConfig {
public:
  Q1PassConfig(Q1TargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {
    if (TM.getOptLevel() != CodeGenOptLevel::None)
      substitutePass(&PostRASchedulerID, &PostMachineSchedulerID);
    setEnableSinkAndFold(EnableSinkFold);
    EnableLoopTermFold = true;
  }

  Q1TargetMachine &getQ1TargetMachine() const {
    return getTM<Q1TargetMachine>();
  }

  ScheduleDAGInstrs *
  createMachineScheduler(MachineSchedContext *C) const override {
    ScheduleDAGMILive *DAG = nullptr;
    if (EnableMISchedLoadClustering) {
      DAG = createGenericSchedLive(C);
      DAG->addMutation(createLoadClusterDAGMutation(
          DAG->TII, DAG->TRI, /*ReorderWhileClustering=*/true));
    }
    return DAG;
  }

  void addIRPasses() override;
  bool addPreISel() override;
  void addCodeGenPrepare() override;
  bool addInstSelector() override;
  bool addIRTranslator() override;
  void addPreLegalizeMachineIR() override;
  bool addLegalizeMachineIR() override;
  void addPreRegBankSelect() override;
  bool addRegBankSelect() override;
  bool addGlobalInstructionSelect() override;
  void addPreEmitPass() override;
  void addPreEmitPass2() override;
  void addPreSched2() override;
  void addMachineSSAOptimization() override;
  FunctionPass *createRVVRegAllocPass(bool Optimized);
  bool addRegAssignAndRewriteFast() override;
  bool addRegAssignAndRewriteOptimized() override;
  void addPreRegAlloc() override;
  void addPostRegAlloc() override;
  void addFastRegAlloc() override;

  std::unique_ptr<CSEConfigBase> getCSEConfig() const override;
};
} // namespace

TargetPassConfig *Q1TargetMachine::createPassConfig(PassManagerBase &PM) {
  return new Q1PassConfig(*this, PM);
}

std::unique_ptr<CSEConfigBase> Q1PassConfig::getCSEConfig() const {
  return getStandardCSEConfigForOpt(TM->getOptLevel());
}

FunctionPass *Q1PassConfig::createRVVRegAllocPass(bool Optimized) {
  // Initialize the global default.
  llvm::call_once(InitializeDefaultRVVRegisterAllocatorFlag,
                  initializeDefaultRVVRegisterAllocatorOnce);

  RegisterRegAlloc::FunctionPassCtor Ctor = RVVRegisterRegAlloc::getDefault();
  if (Ctor != useDefaultRegisterAllocator)
    return Ctor();

  if (Optimized)
    return createGreedyRVVRegisterAllocator();

  return createFastRVVRegisterAllocator();
}

bool Q1PassConfig::addRegAssignAndRewriteFast() {
  addPass(createRVVRegAllocPass(false));
  if (EnableVSETVLIAfterRVVRegAlloc)
    addPass(createQ1InsertVSETVLIPass());
  if (TM->getOptLevel() != CodeGenOptLevel::None &&
      EnableQ1DeadRegisterElimination)
    addPass(createQ1DeadRegisterDefinitionsPass());
  return TargetPassConfig::addRegAssignAndRewriteFast();
}

bool Q1PassConfig::addRegAssignAndRewriteOptimized() {
  addPass(createRVVRegAllocPass(true));
  addPass(createVirtRegRewriter(false));
  if (EnableVSETVLIAfterRVVRegAlloc)
    addPass(createQ1InsertVSETVLIPass());
  if (TM->getOptLevel() != CodeGenOptLevel::None &&
      EnableQ1DeadRegisterElimination)
    addPass(createQ1DeadRegisterDefinitionsPass());
  return TargetPassConfig::addRegAssignAndRewriteOptimized();
}

void Q1PassConfig::addIRPasses() {
  addPass(createAtomicExpandLegacyPass());
  addPass(createQ1ZacasABIFixPass());

  if (getOptLevel() != CodeGenOptLevel::None) {
    if (EnableLoopDataPrefetch)
      addPass(createLoopDataPrefetchPass());

    addPass(createQ1GatherScatterLoweringPass());
    addPass(createInterleavedAccessPass());
    addPass(createQ1CodeGenPreparePass());
  }

  TargetPassConfig::addIRPasses();
}

bool Q1PassConfig::addPreISel() {
  if (TM->getOptLevel() != CodeGenOptLevel::None) {
    // Add a barrier before instruction selection so that we will not get
    // deleted block address after enabling default outlining. See D99707 for
    // more details.
    addPass(createBarrierNoopPass());
  }

  if (EnableGlobalMerge == cl::BOU_TRUE) {
    addPass(createGlobalMergePass(TM, /* MaxOffset */ 2047,
                                  /* OnlyOptimizeForSize */ false,
                                  /* MergeExternalByDefault */ true));
  }

  return false;
}

void Q1PassConfig::addCodeGenPrepare() {
  if (getOptLevel() != CodeGenOptLevel::None)
    addPass(createTypePromotionLegacyPass());
  TargetPassConfig::addCodeGenPrepare();
}

bool Q1PassConfig::addInstSelector() {
  addPass(createQ1ISelDag(getQ1TargetMachine(), getOptLevel()));

  return false;
}

bool Q1PassConfig::addIRTranslator() {
  addPass(new IRTranslator(getOptLevel()));
  return false;
}

void Q1PassConfig::addPreLegalizeMachineIR() {
  if (getOptLevel() == CodeGenOptLevel::None) {
    addPass(createQ1O0PreLegalizerCombiner());
  } else {
    addPass(createQ1PreLegalizerCombiner());
  }
}

bool Q1PassConfig::addLegalizeMachineIR() {
  addPass(new Legalizer());
  return false;
}

void Q1PassConfig::addPreRegBankSelect() {
  if (getOptLevel() != CodeGenOptLevel::None)
    addPass(createQ1PostLegalizerCombiner());
}

bool Q1PassConfig::addRegBankSelect() {
  addPass(new RegBankSelect());
  return false;
}

bool Q1PassConfig::addGlobalInstructionSelect() {
  addPass(new InstructionSelect(getOptLevel()));
  return false;
}

void Q1PassConfig::addPreSched2() {
  addPass(createQ1PostRAExpandPseudoPass());

  // Emit KCFI checks for indirect calls.
  addPass(createKCFIPass());
}

void Q1PassConfig::addPreEmitPass() {
  // TODO: It would potentially be better to schedule copy propagation after
  // expanding pseudos (in addPreEmitPass2). However, performing copy
  // propagation after the machine outliner (which runs after addPreEmitPass)
  // currently leads to incorrect code-gen, where copies to registers within
  // outlined functions are removed erroneously.
  if (TM->getOptLevel() >= CodeGenOptLevel::Default &&
      EnableQ1CopyPropagation)
    addPass(createMachineCopyPropagationPass(true));
  addPass(&BranchRelaxationPassID);
  addPass(createQ1MakeCompressibleOptPass());
}

void Q1PassConfig::addPreEmitPass2() {
  if (TM->getOptLevel() != CodeGenOptLevel::None) {
    addPass(createQ1MoveMergePass());
    // Schedule PushPop Optimization before expansion of Pseudo instruction,
    // ensuring return instruction is detected correctly.
    addPass(createQ1PushPopOptimizationPass());
  }
  addPass(createQ1IndirectBranchTrackingPass());
  addPass(createQ1ExpandPseudoPass());

  // Schedule the expansion of AMOs at the last possible moment, avoiding the
  // possibility for other passes to break the requirements for forward
  // progress in the LR/SC block.
  addPass(createQ1ExpandAtomicPseudoPass());

  // KCFI indirect call checks are lowered to a bundle.
  addPass(createUnpackMachineBundles([&](const MachineFunction &MF) {
    return MF.getFunction().getParent()->getModuleFlag("kcfi");
  }));
}

void Q1PassConfig::addMachineSSAOptimization() {
  addPass(createQ1VectorPeepholePass());

  TargetPassConfig::addMachineSSAOptimization();

  if (EnableMachineCombiner)
    addPass(&MachineCombinerID);

  if (TM->getTargetTriple().isQ164()) {
    addPass(createQ1OptWInstrsPass());
  }
}

void Q1PassConfig::addPreRegAlloc() {
  addPass(createQ1PreRAExpandPseudoPass());
  if (TM->getOptLevel() != CodeGenOptLevel::None)
    addPass(createQ1MergeBaseOffsetOptPass());

  addPass(createQ1InsertReadWriteCSRPass());
  addPass(createQ1InsertWriteVXRMPass());
  addPass(createQ1LandingPadSetupPass());

  // Run Q1InsertVSETVLI after PHI elimination. On O1 and above do it after
  // register coalescing so needVSETVLIPHI doesn't need to look through COPYs.
  if (!EnableVSETVLIAfterRVVRegAlloc) {
    if (TM->getOptLevel() == CodeGenOptLevel::None)
      insertPass(&PHIEliminationID, &Q1InsertVSETVLIID);
    else
      insertPass(&RegisterCoalescerID, &Q1InsertVSETVLIID);
  }
}

void Q1PassConfig::addFastRegAlloc() {
  addPass(&InitUndefID);
  TargetPassConfig::addFastRegAlloc();
}


void Q1PassConfig::addPostRegAlloc() {
  if (TM->getOptLevel() != CodeGenOptLevel::None &&
      EnableRedundantCopyElimination)
    addPass(createQ1RedundantCopyEliminationPass());
}

void Q1TargetMachine::registerPassBuilderCallbacks(PassBuilder &PB) {
  PB.registerLateLoopOptimizationsEPCallback([=](LoopPassManager &LPM,
                                                 OptimizationLevel Level) {
    LPM.addPass(LoopIdiomVectorizePass(LoopIdiomVectorizeStyle::Predicated));
  });
}

yaml::MachineFunctionInfo *
Q1TargetMachine::createDefaultFuncInfoYAML() const {
  return new yaml::Q1MachineFunctionInfo();
}

yaml::MachineFunctionInfo *
Q1TargetMachine::convertFuncInfoToYAML(const MachineFunction &MF) const {
  const auto *MFI = MF.getInfo<Q1MachineFunctionInfo>();
  return new yaml::Q1MachineFunctionInfo(*MFI);
}

bool Q1TargetMachine::parseMachineFunctionInfo(
    const yaml::MachineFunctionInfo &MFI, PerFunctionMIParsingState &PFS,
    SMDiagnostic &Error, SMRange &SourceRange) const {
  const auto &YamlMFI =
      static_cast<const yaml::Q1MachineFunctionInfo &>(MFI);
  PFS.MF.getInfo<Q1MachineFunctionInfo>()->initializeBaseYamlFields(YamlMFI);
  return false;
}
