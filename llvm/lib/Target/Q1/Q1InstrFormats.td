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

// Classical - Control instructions
class Q1InstCC<string asmStr> : Q1InstCommon<(ins), (outs), asmStr, []>;

// Classical - Jump instructions
class Q1InstCJ<dag ins, string asmStr> : Q1InstCommon<ins, (outs), asmStr, []>;
