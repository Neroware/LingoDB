add_mlir_dialect_library(MLIRGraph
        GraphDialect.cpp
        GraphOps.cpp
        GraphOpsEnums.cpp
        GraphTypes.cpp
        GraphAttributes.cpp
        ADDITIONAL_HEADER_DIRS
        ../../../../../include/gengodb/compiler/Dialect/Graph

        DEPENDS
        MLIRGraphOpsIncGen
        MLIRSubOperatorOpsIncGen
        MLIRDBOpsIncGen

        LINK_LIBS mlir-support PUBLIC
        MLIRIR
        runtime_funcs_ptr
)