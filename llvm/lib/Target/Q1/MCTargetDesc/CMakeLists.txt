add_llvm_component_library(LLVMQ1Desc
  Q1InstPrinter.cpp
  Q1MCAsmInfo.cpp
  Q1MCCodeEmitter.cpp   
  Q1MCTargetDesc.cpp

  LINK_COMPONENTS
  MC
  Q1Info
  Support
  TargetParser

  ADD_TO_COMPONENT
  Q1
)
