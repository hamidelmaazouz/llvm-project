#ifndef LLVM_LIB_TARGET_Q1_MCTARGETDESC_Q1MCTARGETDESC_H
#define LLVM_LIB_TARGET_Q1_MCTARGETDESC_Q1MCTARGETDESC_H

#include "llvm/Support/DataTypes.h"

#include <memory>

namespace llvm {

class MCAsmBackend;
class MCCodeEmitter;
class MCContext;
class MCInstrInfo;
class MCObjectTargetWriter;
class MCRegisterInfo;
class MCSubtargetInfo;
class MCTargetOptions;
class StringRef;
class Target;
class Triple;
class raw_pwrite_stream;
class raw_ostream;

MCCodeEmitter *createQ1MCCodeEmitter(const MCInstrInfo &MCII, MCContext &Ctx);
} // namespace llvm

#define GET_REGINFO_ENUM
#include "Q1GenRegisterInfo.inc"

#define GET_INSTRINFO_ENUM
#include "Q1GenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "Q1GenSubtargetInfo.inc"

#endif
