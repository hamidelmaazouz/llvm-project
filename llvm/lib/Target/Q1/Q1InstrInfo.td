def NOP : InstQ1<
    (),
    (),
    "nop",
    []
>

def STOP : InstQ1<
    (),
    (),
    "stop",
    []
>

def ADDrr : InstQ1<
    (ins GRRegs:$src1, GRRegs:$src2),
    (outs GRRegs:$dst),
    "add $src1, $src2, $dst",
    [(set (add i32:$src1, i32:$src2), i32:$dst)]
> {
 bits<6> src1;
 bits<6> src2;
 bits<6> dst;

 let Inst{0-5} = src1;         // Operand 1
 let Inst{6-11} = src2;        // Operand 2
 let Inst{12-17} = dst;        // Destination
 let Inst{18-24} = 0b1100;     // Opcode
 let Inst{25-31} = 0b0000000;  // Something else
}
