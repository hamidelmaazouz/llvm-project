include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// Q1 subtarget features and instruction predicates.
//===----------------------------------------------------------------------===//

include "Q1Features.td"


//===----------------------------------------------------------------------===//
// Registers, calling conventions, instruction descriptions.
//===----------------------------------------------------------------------===//

include "Q1RegisterInfo.td"
include "Q1CallingConv.td"
include "Q1InstrFormats.td"
include "Q1InstrInfo.td"


//===----------------------------------------------------------------------===//
// Define the Q1 target.
//===----------------------------------------------------------------------===//

def Q1InstrInfo : InstrInfo;

def Q1AsmParser : AsmParser;

def Q1AsmWriter : AsmWriter;

def Q1 : Target {
  let InstructionSet = Q1InstrInfo;
  let AssemblyParsers = [Q1AsmParser];
  let AssemblyWriters = [Q1AsmWriter];
}
