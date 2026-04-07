#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"

#include "gengodb/compiler/Dialect/GPM/IR/GPMDialect.h"
#include "gengodb/compiler/Dialect/GPM/IR/GPMOps.h"

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"

namespace gengodb::compiler::dialect::gpm::detail {
using namespace lingodb::compiler::dialect::relalg;
inline auto filterVariableTerms(mlir::Operation* op, bool isBound) {
    return llvm::make_filter_range(
      op->getAttrs(), [&](mlir::NamedAttribute attr) {
        auto v = mlir::dyn_cast<VariableTermAttr>(attr.getValue());
        return v && (v.hasBinding() == isBound);
      });
}
ColumnSet getCreatedVariables(mlir::Operation* op) {
    ColumnSet columns;
    for (auto x : filterVariableTerms(op, false)) {
        columns.insert(mlir::cast<VariableTermAttr>(x.getValue())
            .getProducedBinding().getColumnPtr().get());
    }
    return columns;
}
ColumnSet getBoundVariables(mlir::Operation* op) {
    ColumnSet columns;
    for (auto x : filterVariableTerms(op, true)) {
        columns.insert(mlir::cast<VariableTermAttr>(x.getValue())
            .getBindingReference().getColumnPtr().get());
    }
    return columns;
}
ColumnSet getAllVariables(mlir::Operation* op) {
    return getBoundVariables(op)
        .insert(getCreatedVariables(op));
}

} // namespace gengodb::compiler::dialect::gpm::detail

#include "gengodb/compiler/Dialect/GPM/IR/GPMOpsInterfaces.cpp.inc"