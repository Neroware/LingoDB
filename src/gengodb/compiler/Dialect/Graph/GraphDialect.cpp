#include "gengodb/compiler/Dialect/Graph/GraphDialect.h"

#include "gengodb/compiler/Dialect/Graph/GraphOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace lingodb::compiler::dialect::graph;

void GraphDialect::initialize() {
    addOperations<
    #define GET_OP_LIST
    #include "gengodb/compiler/Dialect/Graph/GraphOps.cpp.inc"
    
          >();
    
    registerTypes();
    registerAttrs();
}
#include "gengodb/compiler/Dialect/Graph/GraphOpsDialect.cpp.inc"