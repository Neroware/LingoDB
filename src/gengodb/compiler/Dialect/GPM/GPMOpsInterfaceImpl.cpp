#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"

#include "gengodb/compiler/Dialect/GPM/IR/GPMDialect.h"
#include "gengodb/compiler/Dialect/GPM/IR/GPMOps.h"

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"

namespace gengodb::compiler::dialect::gpm::detail {
using namespace lingodb::compiler::dialect::relalg;
llvm::SmallVector<VariableTermAttr, 4> getBoundVariables(mlir::Operation* op) {
    llvm::SmallVector<VariableTermAttr, 4> vx;
    for (auto named : op->getAttrs()) {
        if (const auto& v = mlir::dyn_cast_or_null<VariableTermAttr>(named.getValue())) {
            if (v.hasBinding()) {
                vx.push_back(v);
            }
        }
    }
    return vx;
}
llvm::SmallVector<VariableTermAttr, 4> getUnboundVariables(mlir::Operation* op) {
    llvm::SmallVector<VariableTermAttr, 4> vx;
    for (auto named : op->getAttrs()) {
        if (const auto& v = mlir::dyn_cast_or_null<VariableTermAttr>(named.getValue())) {
            if (!v.hasBinding()) {
                vx.push_back(v);
            }
        }
    }
    return vx;
}
ColumnSet getCreatedColumns(mlir::Operation* op) {
    const auto vx = getUnboundVariables(op);
    ColumnSet columns;
    for (const auto& v : vx) {
        columns.insert(v.getProducedBinding().getColumnPtr().get());
    }
    return columns;
}
ColumnSet getFreeColumns(mlir::Operation* op) {
    const auto vx = getBoundVariables(op);
    ColumnSet columns;
    for (const auto& v : vx) {
        columns.insert(v.getBindingReference().getColumnPtr().get());
    }
    return columns;
}

} // namespace gengodb::compiler::dialect::gpm::detail

#include "gengodb/compiler/Dialect/GPM/IR/GPMOpsInterfaces.cpp.inc"