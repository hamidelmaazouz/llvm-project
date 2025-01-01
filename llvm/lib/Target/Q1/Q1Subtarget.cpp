#include "Q1Subtarget.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

#define DEBUG_TYPE "Q1-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "Q1GenSubtargetInfo.inc"

void Q1Subtarget::anchor() {}

Q1Subtarget::Q1Subtarget(const Triple &TT, StringRef CPU, StringRef FS,
                         const TargetMachine &TM)
    : Q1GenSubtargetInfo(TT, CPU, CPU, FS), FrameLowering(), InstrInfo(*this),
      TLInfo(TM, *this) {}
