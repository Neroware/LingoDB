#include "lingodb/compiler/Dialect/Graph/GraphOps.h"

#include "lingodb/compiler/Dialect/Graph/GraphDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

#define GET_OP_CLASSES
#include "lingodb/compiler/Dialect/Graph/GraphOps.cpp.inc"