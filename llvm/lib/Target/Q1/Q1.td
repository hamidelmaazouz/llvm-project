include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// Registers, calling conventions, instruction descriptions.
//===----------------------------------------------------------------------===//

include "Q1RegisterInfo.td"
include "Q1CallingConv.td"
include "Q1InstrInfo.td"

//===----------------------------------------------------------------------===//
// Define the Q1 target.
//===----------------------------------------------------------------------===//

def Q1InstrInfo : InstrInfo {
  let guessInstructionProperties = 0;
}

def Q1AsmParser : AsmParser {
  let ShouldEmitMatchRegisterAltName = 1;
  let AllowDuplicateRegisterNames = 1;
}

def Q1AsmWriter : AsmWriter {
  int PassSubtarget = 1;
}

def Q1 : Target {
  let InstructionSet = Q1InstrInfo;
  let AssemblyParsers = [Q1AsmParser];
  let AssemblyWriters = [Q1AsmWriter];
  let AllowRegisterRenaming = 1;
}
