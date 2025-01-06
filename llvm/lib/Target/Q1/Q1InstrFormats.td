//===----------------------------------------------------------------------===//
// Operand formats
//===----------------------------------------------------------------------===//

class Q1RegisterOperand<RegisterClass RC = GP> : RegisterOperand<RC> {
  let DecoderMethod = "decode" # RC # "RegisterClass";
}

def GPROpnd : Q1RegisterOperand<GP>;

class Q1UImmLeafOperand<int numBits = 32> :
    Operand<Q1VT>,
    ImmLeaf<Q1VT, "return isUInt<" # numBits # ">(Imm);">;

def UImmOpnd : Q1UImmLeafOperand<32>;


//===----------------------------------------------------------------------===//
// Instruction formats
//===----------------------------------------------------------------------===//

class Q1InstCommon<dag ins, dag outs, string asmStr, list<dag> pattern> : Instruction {
    let Namespace = "Q1";
    dag InOperandList = ins;
    dag OutOperandList = outs;
    let AsmString = asmStr;
    let Pattern = pattern;
    bits<32> Inst;
    bits<32> SoftFail;
    let Size = 4;
}

/// Formats for CL feature/extension

class Q1InstCL<dag ins, dag outs, string asmStr, list<dag> pattern> : Q1InstCommon<ins, outs, asmStr, pattern>;

class Q1InstCL00<string asmStr> : Q1InstCL<(ins), (outs), asmStr, []>;

class Q1InstCL10<dag ins, string asmStr> : Q1InstCL<ins, (outs), asmStr, []>;

/// Formats for RF feature/extension

class Q1InstRF<dag ins, dag outs, string asmStr, list<dag> pattern> : Q1InstCommon<ins, outs, asmStr, pattern>;

class Q1InstRF00<string asmStr> : Q1InstCL<(ins), (outs), asmStr, []>;

class Q1InstRF10<dag ins, string asmStr> : Q1InstCL<ins, (outs), asmStr, []>;

/// Formats for QC feature/extension

class Q1InstQC<dag ins, dag outs, string asmStr, list<dag> pattern> : Q1InstCommon<ins, outs, asmStr, pattern>;

/// Formats for QR feature/extension

class Q1InstQR<dag ins, dag outs, string asmStr, list<dag> pattern> : Q1InstCommon<ins, outs, asmStr, pattern>;


//===----------------------------------------------------------------------===//
// Pattern formats
//===----------------------------------------------------------------------===//

class PatReg<SDPatternOperator OpNode, Q1InstCommon Inst, ValueType vt = Q1VT>
    : Pat<(vt (OpNode (vt GPROpnd:$rs))), (Inst GPROpnd:$rs)>;

class PatRegReg<SDPatternOperator OpNode, Q1InstCommon Inst, ValueType vt1 = Q1VT, ValueType vt2 = Q1VT>
    : Pat<(vt1 (OpNode (vt1 GPROpnd:$rs1), (vt2 GPROpnd:$rs2))), (Inst GPROpnd:$rs1, GPROpnd:$rs2)>;

class PatRegImm<SDPatternOperator OpNode, Q1InstCommon Inst, ImmLeaf ImmType, ValueType vt = Q1VT>
    : Pat<(vt (OpNode (vt GPROpnd:$rs), ImmType:$is)), (Inst GPROpnd:$rs, ImmType:$is)>;
