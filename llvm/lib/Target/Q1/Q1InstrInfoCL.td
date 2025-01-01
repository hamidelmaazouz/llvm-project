//===----------------------------------------------------------------------===//
// Classical - Control instructions
//===----------------------------------------------------------------------===//

def ILLEGAL : Q1InstCC<"illegal">;
def STOP : Q1InstCC<"stop">;
def NOP : Q1InstCC<"nop">;


//===----------------------------------------------------------------------===//
// Classical - Jump instruction classes
//===----------------------------------------------------------------------===//

def JMP_I : Q1InstCJ<(ins i32imm:$address), "jmp $address">;
def JMP_R : Q1InstCJ<(ins GPROpnd:$address), "jmp $address">;

def JGE_RII : Q1InstCJ<(ins GPROpnd:$a, i32imm:$b, i32imm:$address), "jge $a, $b, $address">;
def JGE_RIR : Q1InstCJ<(ins GPROpnd:$a, i32imm:$b, GPROpnd:$address), "jge $a, $b, $address">;

def JLT_RII : Q1InstCJ<(ins GPROpnd:$a, i32imm:$b, i32imm:$address), "jlt $a, $b, $address">;
def JLT_RIR : Q1InstCJ<(ins GPROpnd:$a, i32imm:$b, GPROpnd:$address), "jlt $a, $b, $address">;

def LOOP_RI : Q1InstCJ<(ins GPROpnd:$a, i32imm:$address), "loop $a, $address">;
def LOOP_RR : Q1InstCJ<(ins GPROpnd:$a, GPROpnd:$address), "loop $a, $address">;


//===----------------------------------------------------------------------===//
// Classical - Arithmetic (and Logic) instructions
//===----------------------------------------------------------------------===//

def MOVE_IR : Q1InstCommon<(ins i32imm:$source), (outs GPROpnd:$destination), "move $source, $destination", []>;
def MOVE_RR : Q1InstCommon<(ins GPROpnd:$source), (outs GPROpnd:$destination), "move $source, $destination", []>;

def NOT_IR : Q1InstCommon<(ins i32imm:$source), (outs GPROpnd:$destination), "not $source, $destination", []>;
def NOT_RR : Q1InstCommon<(ins GPROpnd:$source), (outs GPROpnd:$destination), "not $source, $destination", []>;

def ADD_RIR : Q1InstCommon<(ins GPROpnd:$a, i32imm:$b), (outs GPROpnd:$destination), "add $a, $b, $destination", []>;
def ADD_RRR : Q1InstCommon<(ins GPROpnd:$a, GPROpnd:$b), (outs GPROpnd:$destination), "add $a, $b, $destination", []>;

def SUB_RIR : Q1InstCommon<(ins GPROpnd:$a, i32imm:$b), (outs GPROpnd:$destination), "sub $a, $b, $destination", []>;
def SUB_RRR : Q1InstCommon<(ins GPROpnd:$a, GPROpnd:$b), (outs GPROpnd:$destination), "sub $a, $b, $destination", []>;

def AND_RIR : Q1InstCommon<(ins GPROpnd:$a, i32imm:$b), (outs GPROpnd:$destination), "and $a, $b, $destination", []>;
def AND_RRR : Q1InstCommon<(ins GPROpnd:$a, GPROpnd:$b), (outs GPROpnd:$destination), "and $a, $b, $destination", []>;

def OR_RIR : Q1InstCommon<(ins GPROpnd:$a, i32imm:$b), (outs GPROpnd:$destination), "or $a, $b, $destination", []>;
def OR_RRR : Q1InstCommon<(ins GPROpnd:$a, GPROpnd:$b), (outs GPROpnd:$destination), "or $a, $b, $destination", []>;

def XOR_RIR : Q1InstCommon<(ins GPROpnd:$a, i32imm:$b), (outs GPROpnd:$destination), "xor $a, $b, $destination", []>;
def XOR_RRR : Q1InstCommon<(ins GPROpnd:$a, GPROpnd:$b), (outs GPROpnd:$destination), "xor $a, $b, $destination", []>;

def ASL_RIR : Q1InstCommon<(ins GPROpnd:$a, i32imm:$b), (outs GPROpnd:$destination), "asl $a, $b, $destination", []>;
def ASL_RRR : Q1InstCommon<(ins GPROpnd:$a, GPROpnd:$b), (outs GPROpnd:$destination), "asl $a, $b, $destination", []>;

def ASR_RIR : Q1InstCommon<(ins GPROpnd:$a, i32imm:$b), (outs GPROpnd:$destination), "asr $a, $b, $destination", []>;
def ASR_RRR : Q1InstCommon<(ins GPROpnd:$a, GPROpnd:$b), (outs GPROpnd:$destination), "asr $a, $b, $destination", []>;
