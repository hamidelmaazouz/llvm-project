# RUN: llvm-mc --triple=q1 %s \
# RUN:     | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: move 0, R0
move 0, R0
# CHECK-ASM-AND-OBJ: move R0, R1
move R0, R1

# CHECK-ASM-AND-OBJ: not 0, R57
not 0, R57
# CHECK-ASM-AND-OBJ: not R60, R57
not R60, R57

# CHECK-ASM-AND-OBJ: add R0, 1, R0
add R0, 1, R0
# CHECK-ASM-AND-OBJ: add R0, R1, R0
add R0, R1, R0

# CHECK-ASM-AND-OBJ: sub R0, 1, R0
sub R0, 1, R0
# CHECK-ASM-AND-OBJ: sub R0, R1, R0
sub R0, R1, R0

# CHECK-ASM-AND-OBJ: and R0, 1, R0
and R0, 1, R0
# CHECK-ASM-AND-OBJ: and R0, R1, R0
and R0, R1, R0

# CHECK-ASM-AND-OBJ: or R0, 1, R0
or R0, 1, R0
# CHECK-ASM-AND-OBJ: or R0, R1, R0
or R0, R1, R0

# CHECK-ASM-AND-OBJ: xor R0, 1, R0
xor R0, 1, R0
# CHECK-ASM-AND-OBJ: xor R0, R1, R0
xor R0, R1, R0

# CHECK-ASM-AND-OBJ: asl R0, 1, R0
asl R0, 1, R0
# CHECK-ASM-AND-OBJ: asl R0, R1, R0
asl R0, R1, R0

# CHECK-ASM-AND-OBJ: asr R0, 1, R0
asr R0, 1, R0
# CHECK-ASM-AND-OBJ: asr R0, R1, R0
asr R0, R1, R0
