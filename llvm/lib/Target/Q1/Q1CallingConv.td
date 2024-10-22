def RetCC_Q1 : CallingConv<[
    CCIfType<[i32], CCAssignToReg<[R0]>>,
    CCIfType<[i32], CCAssignToStack<4, 4>>
]>;


def CC_Q1 : CallingConv<[
    CCIfType<[i32], CCAssignToReg<[R0, R1]>>,
    CCIfType<[i32], CCAssignToStack<4, 4>>
]>;


def CC_Save : CalleeSavedRegs<(add R2, R3)>;

