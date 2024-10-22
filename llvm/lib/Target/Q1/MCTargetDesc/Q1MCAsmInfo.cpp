#include "Q1MCAsmInfo.h"
//#include "MCTargetDesc/Q1MCExpr.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/TargetParser/Triple.h"
using namespace llvm;

void Q1MCAsmInfo::anchor() {}

Q1MCAsmInfo::Q1MCAsmInfo(const Triple &TT) {
  SupportsDebugInformation = true;
  Data32bitsDirective = "\t.long\t";
  CodePointerSize = 4;
  CommentString = "#";
}
