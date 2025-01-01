#include "Q1InstrInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFrameInfo.h"

using namespace llvm;

#define GET_INSTRINFO_CTOR_DTOR
#include "Q1GenInstrInfo.inc"

// Pin the vtable to this file.
void Q1InstrInfo::anchor() {}

Q1InstrInfo::Q1InstrInfo(Q1Subtarget &STI) : Q1GenInstrInfo(), STI(STI), RI() {}
