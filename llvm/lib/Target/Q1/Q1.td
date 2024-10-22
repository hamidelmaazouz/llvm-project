include "llvm/Target/Target.td"

include "Q1RegisterInfo.td"
include "Q1InstrFormats.td"
include "Q1InstrInfo.td"

def Q1InstrInfo : InstrInfo;

def Q1AsmParser : AsmParser;

def Q1AsmWriter : AsmWriter;

def Q1 : Target {
  let InstructionSet = Q1InstrInfo;
  let AssemblyParsers = [Q1AsmParser];
  let AssemblyWriters = [Q1AsmWriter];
}
