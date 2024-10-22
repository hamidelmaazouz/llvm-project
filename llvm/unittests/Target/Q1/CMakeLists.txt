include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/Q1
  ${LLVM_BINARY_DIR}/lib/Target/Q1
  )

set(LLVM_LINK_COMPONENTS
  Q1CodeGen
  Q1Desc
  Q1Info
  Analysis
  CodeGen
  Core
  MC
  SelectionDAG
  TargetParser
  )

add_llvm_target_unittest(Q1Tests
  MCInstrAnalysisTest.cpp
        Q1InstrInfoTest.cpp
  )
