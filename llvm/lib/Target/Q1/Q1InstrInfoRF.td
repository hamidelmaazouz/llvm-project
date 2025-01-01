//===----------------------------------------------------------------------===//
// Classical - Parameter operation instructions
//===----------------------------------------------------------------------===//

def SET_MRK_I : Q1InstCJ<(ins i32imm:$mask), "set_mrk $mask">;
def SET_MRK_R : Q1InstCJ<(ins GPROpnd:$mask), "set_mrk $mask">;

def SET_FREQ_I : Q1InstCJ<(ins i32imm:$value), "set_freq $value">;
def SET_FREQ_R : Q1InstCJ<(ins GPROpnd:$value), "set_freq $value">;

def RESET_PH : Q1InstCC<"reset_ph">;

def SET_PH_I : Q1InstCJ<(ins i32imm:$value), "set_ph $value">;
def SET_PH_R : Q1InstCJ<(ins GPROpnd:$value), "set_ph $value">;

def SET_PH_DELTA_I : Q1InstCJ<(ins i32imm:$value), "set_ph_delta $value">;
def SET_PH_DELTA_R : Q1InstCJ<(ins GPROpnd:$value), "set_ph_delta $value">;

def SET_AWG_GAIN_I : Q1InstCJ<(ins i32imm:$value), "set_awg_gain $value">;
def SET_AWG_GAIN_R : Q1InstCJ<(ins GPROpnd:$value), "set_awg_gain $value">;

def SET_AWG_OFFS_I : Q1InstCJ<(ins i32imm:$value), "set_awg_offs $value">;
def SET_AWG_OFFS_R : Q1InstCJ<(ins GPROpnd:$value), "set_awg_offs $value">;
