# RUN: llvm-mc %s --triple=q1 \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: illegal
illegal
# CHECK-ASM-AND-OBJ: stop
stop
# CHECK-ASM-AND-OBJ: nop
nop
