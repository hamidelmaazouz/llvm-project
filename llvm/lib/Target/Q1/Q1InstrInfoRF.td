//===----------------------------------------------------------------------===//
// Parameter operation instructions
//===----------------------------------------------------------------------===//

def SET_MRK_I : Q1InstRF10<(ins UImmOpnd:$mask), "set_mrk $mask">;
def SET_MRK_R : Q1InstRF10<(ins GPROpnd:$mask), "set_mrk $mask">;

def SET_FREQ_I : Q1InstRF10<(ins UImmOpnd:$value), "set_freq $value">;
def SET_FREQ_R : Q1InstRF10<(ins GPROpnd:$value), "set_freq $value">;

def RESET_PH : Q1InstRF00<"reset_ph">;

def SET_PH_I : Q1InstRF10<(ins UImmOpnd:$value), "set_ph $value">;
def SET_PH_R : Q1InstRF10<(ins GPROpnd:$value), "set_ph $value">;

def SET_PH_DELTA_I : Q1InstRF10<(ins UImmOpnd:$value), "set_ph_delta $value">;
def SET_PH_DELTA_R : Q1InstRF10<(ins GPROpnd:$value), "set_ph_delta $value">;

def SET_AWG_GAIN_I : Q1InstRF10<(ins UImmOpnd:$value), "set_awg_gain $value">;
def SET_AWG_GAIN_R : Q1InstRF10<(ins GPROpnd:$value), "set_awg_gain $value">;

def SET_AWG_OFFS_I : Q1InstRF10<(ins UImmOpnd:$value), "set_awg_offs $value">;
def SET_AWG_OFFS_R : Q1InstRF10<(ins GPROpnd:$value), "set_awg_offs $value">;
