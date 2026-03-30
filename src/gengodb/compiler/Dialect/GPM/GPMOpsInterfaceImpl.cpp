#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"

#include "gengodb/compiler/Dialect/GPM/IR/GPMDialect.h"
#include "gengodb/compiler/Dialect/GPM/IR/GPMOps.h"

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"

namespace gengodb::compiler::dialect::gpm::detail {
using namespace lingodb::compiler::dialect::relalg;
ColumnSet getBoundVariables(mlir::Operation* op) {
    ColumnSet bindings;
    for (auto named : op->getAttrs()) {
        if (const auto& v = mlir::dyn_cast_or_null<VariableTermAttr>(named.getValue())) {
            if (v.hasBinding()) {
                bindings.insert(v.getBindingReference().getColumnPtr().get());
            }
        }
    }
    return bindings;
}
ColumnSet getUnboundVariables(mlir::Operation* op) {
    ColumnSet bindings;
    for (auto named : op->getAttrs()) {
        if (const auto& v = mlir::dyn_cast_or_null<VariableTermAttr>(named.getValue())) {
            if (!v.hasBinding()) {
                bindings.insert(v.getProducedBinding().getColumnPtr().get());
            }
        }
    }
    return bindings;
}
ColumnSet getAllVariables(mlir::Operation* op) {
    ColumnSet vars;
    for (auto named : op->getAttrs()) {
        if (const auto& v = mlir::dyn_cast_or_null<VariableTermAttr>(named.getValue())) {
            if (v.hasBinding()) {
                vars.insert(v.getBindingReference().getColumnPtr().get());
            }
            else {
                vars.insert(v.getProducedBinding().getColumnPtr().get());
            }
        }
    }
    return vars;
}

} // namespace gengodb::compiler::dialect::gpm::detail

#include "gengodb/compiler/Dialect/GPM/IR/GPMOpsInterfaces.cpp.inc"