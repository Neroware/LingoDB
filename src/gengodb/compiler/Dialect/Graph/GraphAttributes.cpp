#include "gengodb/compiler/Dialect/Graph/GraphDialect.h"
#include "gengodb/compiler/Dialect/Graph/GraphOpsAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include <llvm/ADT/TypeSwitch.h>

#define GET_ATTRDEF_CLASSES
#include "gengodb/compiler/Dialect/Graph/GraphOpsAttributes.cpp.inc"

void gengodb::compiler::dialect::graph::GraphDialect::registerAttrs() {
   addAttributes<
#define GET_ATTRDEF_LIST
#include "gengodb/compiler/Dialect/Graph/GraphOpsAttributes.cpp.inc"

      >();
}