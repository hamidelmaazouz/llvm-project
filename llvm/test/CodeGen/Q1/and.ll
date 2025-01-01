; RUN: llc -mtriple=q1 --verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefix=Q1

define i32 @and_i32_i32_i32(i32 %a, i32 %b) {
; Q1-LABEL: and_i32_i32_i32:
; Q1-NEXT:    and a0, b0, res0
  %res = and i32 %a, %b
  ret i32 %res
}
