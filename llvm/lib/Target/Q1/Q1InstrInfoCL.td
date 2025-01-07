//===----------------------------------------------------------------------===//
// Control instructions
//===----------------------------------------------------------------------===//

def ILLEGAL : Q1InstCL00<"illegal">;
def STOP : Q1InstCL00<"stop">;
def NOP : Q1InstCL00<"nop">;


//===----------------------------------------------------------------------===//
// Jump instruction classes
//===----------------------------------------------------------------------===//

def JMP_I : Q1InstCL10<(ins UImmOpnd:$address), "jmp $address">;
def JMP_R : Q1InstCL10<(ins GPROpnd:$address), "jmp $address">;

def JGE_RII : Q1InstCL10<(ins GPROpnd:$a, UImmOpnd:$b, UImmOpnd:$address), "jge $a, $b, $address">;
def JGE_RIR : Q1InstCL10<(ins GPROpnd:$a, UImmOpnd:$b, GPROpnd:$address), "jge $a, $b, $address">;

def JLT_RII : Q1InstCL10<(ins GPROpnd:$a, UImmOpnd:$b, UImmOpnd:$address), "jlt $a, $b, $address">;
def JLT_RIR : Q1InstCL10<(ins GPROpnd:$a, UImmOpnd:$b, GPROpnd:$address), "jlt $a, $b, $address">;

def LOOP_RI : Q1InstCL10<(ins GPROpnd:$a, UImmOpnd:$address), "loop $a, $address">;
def LOOP_RR : Q1InstCL10<(ins GPROpnd:$a, GPROpnd:$address), "loop $a, $address">;


//===----------------------------------------------------------------------===//
// Arithmetic (and Logic) instructions
//===----------------------------------------------------------------------===//

def MOVE_IR : Q1InstCL<(ins UImmOpnd:$source), (outs GPROpnd:$destination), "move $source, $destination", []>;
def MOVE_RR : Q1InstCL<(ins GPROpnd:$source), (outs GPROpnd:$destination), "move $source, $destination", []>;

def NOT_IR : Q1InstCL<(ins UImmOpnd:$source), (outs GPROpnd:$destination), "not $source, $destination", []>;
def NOT_RR : Q1InstCL<(ins GPROpnd:$source), (outs GPROpnd:$destination), "not $source, $destination", []>;

def ADD_RIR : Q1InstCL<(ins GPROpnd:$a, UImmOpnd:$b), (outs GPROpnd:$destination), "add $a, $b, $destination", []>;
def ADD_RRR : Q1InstCL<(ins GPROpnd:$a, GPROpnd:$b), (outs GPROpnd:$destination), "add $a, $b, $destination", []>;
def : PatRegReg<add, ADD_RRR>;

def SUB_RIR : Q1InstCL<(ins GPROpnd:$a, UImmOpnd:$b), (outs GPROpnd:$destination), "sub $a, $b, $destination", []>;
def SUB_RRR : Q1InstCL<(ins GPROpnd:$a, GPROpnd:$b), (outs GPROpnd:$destination), "sub $a, $b, $destination", []>;
def : PatRegReg<sub, SUB_RRR>;

def AND_RIR : Q1InstCL<(ins GPROpnd:$a, UImmOpnd:$b), (outs GPROpnd:$destination), "and $a, $b, $destination", []>;
def AND_RRR : Q1InstCL<(ins GPROpnd:$a, GPROpnd:$b), (outs GPROpnd:$destination), "and $a, $b, $destination", []>;
def : PatRegReg<and, AND_RRR>;

def OR_RIR : Q1InstCL<(ins GPROpnd:$a, UImmOpnd:$b), (outs GPROpnd:$destination), "or $a, $b, $destination", []>;
def OR_RRR : Q1InstCL<(ins GPROpnd:$a, GPROpnd:$b), (outs GPROpnd:$destination), "or $a, $b, $destination", []>;
def : PatRegReg<or, OR_RRR>;

def XOR_RIR : Q1InstCL<(ins GPROpnd:$a, UImmOpnd:$b), (outs GPROpnd:$destination), "xor $a, $b, $destination", []>;
def XOR_RRR : Q1InstCL<(ins GPROpnd:$a, GPROpnd:$b), (outs GPROpnd:$destination), "xor $a, $b, $destination", []>;
def : PatRegReg<xor, XOR_RRR>;

def ASL_RIR : Q1InstCL<(ins GPROpnd:$a, UImmOpnd:$b), (outs GPROpnd:$destination), "asl $a, $b, $destination", []>;
def ASL_RRR : Q1InstCL<(ins GPROpnd:$a, GPROpnd:$b), (outs GPROpnd:$destination), "asl $a, $b, $destination", []>;

def ASR_RIR : Q1InstCL<(ins GPROpnd:$a, UImmOpnd:$b), (outs GPROpnd:$destination), "asr $a, $b, $destination", []>;
def ASR_RRR : Q1InstCL<(ins GPROpnd:$a, GPROpnd:$b), (outs GPROpnd:$destination), "asr $a, $b, $destination", []>;
