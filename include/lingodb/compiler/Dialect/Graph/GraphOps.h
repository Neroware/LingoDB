#ifndef LINGODB_COMPILER_DIALECT_GRAPH_GRAPHOPS_H
#define LINGODB_COMPILER_DIALECT_GRAPH_GRAPHOPS_H

#include "lingodb/compiler/Dialect/Graph/GraphOpsTypes.h"
#include "lingodb/compiler/Dialect/Graph/GraphOpsAttributes.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "lingodb/compiler/Dialect/Graph/GraphOps.h.inc"

#endif // LINGODB_COMPILER_DIALECT_GRAPH_GRAPHOPS_H
