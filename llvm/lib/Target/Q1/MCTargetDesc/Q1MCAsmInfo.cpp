#include "Q1MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"

using namespace llvm;

Q1MCAsmInfo::Q1MCAsmInfo(const Triple &TT) {
  CommentString = "#";
  SupportsDebugInformation = false;
  ExceptionsType = ExceptionHandling::None;
  
  IsLittleEndian = true;
  CodePointerSize = 4;
  CalleeSaveStackSlotSize = 4;
}
