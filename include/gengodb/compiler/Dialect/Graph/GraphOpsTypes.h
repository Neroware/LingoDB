#ifndef GENGODB_COMPILER_DIALECT_GRAPH_GRAPHOPSTYPES_H
#define GENGODB_COMPILER_DIALECT_GRAPH_GRAPHOPSTYPES_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

using namespace lingodb::compiler::dialect::subop;

#define GET_TYPEDEF_CLASSES
#include "gengodb/compiler/Dialect/Graph/GraphOpsTypes.h.inc"

#endif // GENGODB_COMPILER_DIALECT_GRAPH_GRAPHOPSTYPES_H
