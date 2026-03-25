#include "gengodb/compiler/Dialect/GPM/IR/GPMOps.h"

#include "gengodb/compiler/Dialect/GPM/IR/GPMDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace lingodb::compiler::dialect;
using namespace gengodb::compiler::dialect;

#define GET_OP_CLASSES
#include "gengodb/compiler/Dialect/GPM/IR/GPMOps.cpp.inc"