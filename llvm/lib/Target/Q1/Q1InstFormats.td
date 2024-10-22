class InstQ1<dag ins, dag outs, string asmstr, list<dag> pattern> : Instruction {
    field bits<32> Inst;
    let Namespace = "Q1";
    dag InOperandList = ins;
    dag OutOperandList = outs;
    let AsmString = asmstr;
    let Pattern = pattern;
}
