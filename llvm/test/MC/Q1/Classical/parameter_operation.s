# RUN: llvm-mc --triple=q1 %s \
# RUN:     | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: set_mrk 11
set_mrk 11
# CHECK-ASM-AND-OBJ: set_mrk R0
set_mrk R0

# CHECK-ASM-AND-OBJ: set_freq 10000
set_freq 10000
# CHECK-ASM-AND-OBJ: set_freq R0
set_freq R0

# CHECK-ASM-AND-OBJ: reset_ph
reset_ph

# CHECK-ASM-AND-OBJ: set_ph 10000
set_ph 10000
# CHECK-ASM-AND-OBJ: set_ph R0
set_ph R0

# CHECK-ASM-AND-OBJ: set_ph_delta 10000
set_ph_delta 10000
# CHECK-ASM-AND-OBJ: set_ph_delta R0
set_ph_delta R0

# CHECK-ASM-AND-OBJ: set_awg_gain 10000
set_awg_gain 10000
# CHECK-ASM-AND-OBJ: set_awg_gain R0
set_awg_gain R0

# CHECK-ASM-AND-OBJ: set_awg_offs 10000
set_awg_offs 10000
# CHECK-ASM-AND-OBJ: set_awg_offs R0
set_awg_offs R0
