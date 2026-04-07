#ifndef GENGODB_COMPILER_DIALECT_GPM_IR_GPMOPSINTERFACES_H
#define GENGODB_COMPILER_DIALECT_GPM_IR_GPMOPSINTERFACES_H

#include "llvm/ADT/SmallPtrSet.h"

#include "lingodb/compiler/Dialect/RelAlg/ColumnSet.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOpsTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

namespace gengodb::compiler::dialect::gpm::detail {
using namespace lingodb::compiler::dialect::relalg;
llvm::SmallVector<VariableTermAttr, 4> getBoundVariables(mlir::Operation* op);
llvm::SmallVector<VariableTermAttr, 4> getUnboundVariables(mlir::Operation* op);
ColumnSet getCreatedColumns(mlir::Operation* op);
ColumnSet getFreeColumns(mlir::Operation* op);
} // namespace gengodb::compiler::dialect::gpm::detail
class GPMOperator;
#define GET_OP_CLASSES
#include "gengodb/compiler/Dialect/GPM/IR/GPMOpsInterfaces.h.inc"

#endif //GENGODB_COMPILER_DIALECT_GPM_IR_GPMOPSINTERFACES_H
