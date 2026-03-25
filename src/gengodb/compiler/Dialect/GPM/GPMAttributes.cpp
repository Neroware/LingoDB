#include "gengodb/compiler/Dialect/GPM/IR/GPMDialect.h"
#include "gengodb/compiler/Dialect/GPM/IR/GPMOpsAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include <llvm/ADT/TypeSwitch.h>

#define GET_ATTRDEF_CLASSES
#include "gengodb/compiler/Dialect/GPM/IR/GPMOpsAttributes.cpp.inc"

void gengodb::compiler::dialect::gpm::GPMDialect::registerAttrs() {
   addAttributes<
#define GET_ATTRDEF_LIST
#include "gengodb/compiler/Dialect/GPM/IR/GPMOpsAttributes.cpp.inc"

      >();
}