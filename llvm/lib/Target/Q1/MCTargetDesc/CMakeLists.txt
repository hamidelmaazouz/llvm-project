add_llvm_component_library(LLVMQ1Desc
  Q1MCTargetDesc.cpp
  Q1MCAsmInfo.cpp
  Q1InstPrinter.cpp
  Q1MCCodeEmitter.cpp

  LINK_COMPONENTS
  MC
  Support
  Q1Info

  ADD_TO_COMPONENT
  Q1
  )
