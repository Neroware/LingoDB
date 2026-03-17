#ifndef GENGODB_COMPILER_DIALECT_GRAPH_GRAPHOPSATTRIBUTES_H
#define GENGODB_COMPILER_DIALECT_GRAPH_GRAPHOPSATTRIBUTES_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#define GET_ATTRDEF_CLASSES
#include "gengodb/compiler/Dialect/Graph/GraphOpsAttributes.h.inc"

#endif // GENGODB_COMPILER_DIALECT_GRAPH_GRAPHOPSATTRIBUTES_H