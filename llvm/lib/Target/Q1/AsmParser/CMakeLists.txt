add_llvm_component_library(LLVMQ1AsmParser
        Q1AsmParser.cpp

        LINK_COMPONENTS
        MC
        MCParser
        Q1Desc
        Q1Info
        Support
        TargetParser

        ADD_TO_COMPONENT
        Q1
)