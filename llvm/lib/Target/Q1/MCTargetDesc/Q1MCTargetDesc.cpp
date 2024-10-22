#include "MCTargetDesc/Q1MCTargetDesc.h"
#include "MCTargetDesc/Q1InstPrinter.h"
#include "MCTargetDesc/Q1MCAsmInfo.h"
#include "TargetInfo/Q1TargetInfo.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

#define GET_REGINFO_MC_DESC
#include "Q1GenRegisterInfo.inc"

#define GET_INSTRINFO_MC_DESC
#include "Q1GenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "Q1GenSubtargetInfo.inc"

static MCAsmInfo *createQ1MCAsmInfo(const MCRegisterInfo &MRI, const Triple &TT,
                                    const MCTargetOptions &Options) {
  MCAsmInfo *MAI = new Q1MCAsmInfo(TT);

  unsigned SP = MRI.getDwarfRegNum(Q1::R1, true);
  MCCFIInstruction Inst = MCCFIInstruction::cfiDefCfa(nullptr, SP, 0);
  MAI->addInitialFrameState(Inst);

  return MAI;
}

static MCRegisterInfo *createQ1MCRegisterInfo(const Triple &TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitQ1MCRegisterInfo(X, Q1::R1);
  return X;
}

static MCInstrInfo *createQ1MCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitQ1MCInstrInfo(X);
  return X;
}

static MCSubtargetInfo *createQ1MCSubtargetInfo(const Triple &TT, StringRef CPU,
                                                StringRef FS) {
  return createQ1MCSubtargetInfoImpl(TT, CPU, /*TuneCPU*/ CPU, FS);
}

static MCInstPrinter *createQ1MCInstPrinter(const Triple &T,
                                            unsigned SyntaxVariant,
                                            const MCAsmInfo &MAI,
                                            const MCInstrInfo &MII,
                                            const MCRegisterInfo &MRI) {
  return new Q1InstPrinter(MAI, MII, MRI);
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeQ1TargetMC() {
  // Register the MCAsmInfo.
  TargetRegistry::RegisterMCAsmInfo(getTheQ1Target(), createQ1MCAsmInfo);

  // Register the MCCodeEmitter.
  TargetRegistry::RegisterMCCodeEmitter(getTheQ1Target(),
                                        createQ1MCCodeEmitter);

  // Register the MCRegisterInfo.
  TargetRegistry::RegisterMCRegInfo(getTheQ1Target(), createQ1MCRegisterInfo);

  // Register the MCInstrInfo.
  TargetRegistry::RegisterMCInstrInfo(getTheQ1Target(), createQ1MCInstrInfo);

  // Register the MCSubtargetInfo.
  TargetRegistry::RegisterMCSubtargetInfo(getTheQ1Target(),
                                          createQ1MCSubtargetInfo);

  // Register the MCInstPrinter.
  TargetRegistry::RegisterMCInstPrinter(getTheQ1Target(),
                                        createQ1MCInstPrinter);
}