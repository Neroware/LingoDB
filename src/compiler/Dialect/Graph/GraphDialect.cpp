#include "lingodb/compiler/Dialect/Graph/GraphDialect.h"

#include "lingodb/compiler/Dialect/Graph/GraphOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace lingodb::compiler::dialect::graph;

#define GET_ATTRDEF_CLASSES
#include "lingodb/compiler/Dialect/Graph/GraphOpsAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "lingodb/compiler/Dialect/Graph/GraphOpsTypes.cpp.inc"

void GraphDialect::initialize() {
    addOperations<
    #define GET_OP_LIST
    #include "lingodb/compiler/Dialect/Graph/GraphOps.cpp.inc"
    
          >();
    
       addTypes<
    #define GET_TYPEDEF_LIST
    #include "lingodb/compiler/Dialect/Graph/GraphOpsTypes.cpp.inc"
    
          >();
       addAttributes<
    #define GET_ATTRDEF_LIST
    #include "lingodb/compiler/Dialect/Graph/GraphOpsAttributes.cpp.inc"
    
          >();
    
       // mlir::arith::CmpIOp::attachInterface<ArithCmpICmpInterface>(*getContext());
}
#include "lingodb/compiler/Dialect/Graph/GraphOpsDialect.cpp.inc"