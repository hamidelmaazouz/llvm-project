add_llvm_component_library(LLVMQ1Info
        Q1TargetInfo.cpp

        LINK_COMPONENTS
        MC
        Support

        ADD_TO_COMPONENT
        Q1
)
