# RUN: llvm-mc --triple=q1 %s \
# RUN:     | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: jmp R0
jmp R0
# CHECK-ASM-AND-OBJ: jmp 2025
jmp 2025

# CHECK-ASM-AND-OBJ: jge R10, 10, 2030
jge R10, 10, 2030
# CHECK-ASM-AND-OBJ: jge R10, 10, R11
jge R10, 10, R11

# CHECK-ASM-AND-OBJ: jlt R20, 20, 2050
jlt R20, 20, 2050
# CHECK-ASM-AND-OBJ: jlt R20, 20, R51
jlt R20, 20, R51

# CHECK-ASM-AND-OBJ: loop R63, 10000
loop R63, 10000
# CHECK-ASM-AND-OBJ: loop R63, R40
loop R63, R40
