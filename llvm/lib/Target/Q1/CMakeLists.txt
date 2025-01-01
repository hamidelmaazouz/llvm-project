add_llvm_component_group(Q1)

set(LLVM_TARGET_DEFINITIONS Q1.td)
#tablegen(LLVM Q1GenCompressInstEmitter.inc -gen-compress-inst-emitter)
#tablegen(LLVM Q1GenMacroFusion.inc -gen-macro-fusion-pred)
#tablegen(LLVM Q1GenDisassemblerTables.inc -gen-disassembler)
#tablegen(LLVM Q1GenMCPseudoLowering.inc -gen-pseudo-lowering)
#tablegen(LLVM Q1GenRegisterBank.inc -gen-register-bank)
#tablegen(LLVM Q1GenSearchableTables.inc -gen-searchable-tables)
tablegen(LLVM Q1GenDAGISel.inc -gen-dag-isel)
tablegen(LLVM Q1GenCallingConv.inc -gen-callingconv)
tablegen(LLVM Q1GenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM Q1GenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM Q1GenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM Q1GenRegisterInfo.inc -gen-register-info)
tablegen(LLVM Q1GenInstrInfo.inc -gen-instr-info)
tablegen(LLVM Q1GenSubtargetInfo.inc -gen-subtarget)

#set(LLVM_TARGET_DEFINITIONS Q1GISel.td)
#tablegen(LLVM Q1GenGlobalISel.inc -gen-global-isel)
#tablegen(LLVM Q1GenO0PreLegalizeGICombiner.inc -gen-global-isel-combiner
#              -combiners="Q1O0PreLegalizerCombiner")
#tablegen(LLVM Q1GenPreLegalizeGICombiner.inc -gen-global-isel-combiner
#              -combiners="Q1PreLegalizerCombiner")
#tablegen(LLVM Q1GenPostLegalizeGICombiner.inc -gen-global-isel-combiner
#              -combiners="Q1PostLegalizerCombiner")

add_public_tablegen_target(Q1CommonTableGen)

add_llvm_target(Q1CodeGen
  Q1TargetMachine.cpp
  Q1FrameLowering.cpp
  Q1RegisterInfo.cpp
  Q1InstrInfo.cpp
  Q1AsmPrinter.cpp
  Q1Subtarget.cpp
  Q1ISelDAGToDAG.cpp
  Q1ISelLowering.cpp

  LINK_COMPONENTS
  AsmPrinter
  CodeGen
  Core
  MC
  SelectionDAG
  Support
  Target
  TargetParser
  Q1Desc
  Q1Info

  ADD_TO_COMPONENT
  Q1
  )

add_subdirectory(AsmParser)
#add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
#add_subdirectory(MCA)
add_subdirectory(TargetInfo)
