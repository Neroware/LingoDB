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

::mlir::LogicalResult gpm::TriplePatternOp::verify() {
    auto isValidTerm = [](mlir::Attribute attr) {
        return mlir::isa<
            IdentifierTermAttr,
            BNodeTermAttr,
            VariableTermAttr
        >(attr);
    };
    if (!isValidTerm(getS())) {
        return emitOpError("subject must be a GPM term attribute");
    }
    if (!isValidTerm(getP())) {
        return emitOpError("predicate must be a GPM term attribute");
    } 
    if (!isValidTerm(getO())) {
        return emitOpError("object must be a GPM term attribute");
    }
    if (mlir::isa<BNodeTermAttr>(getP())) {
        return emitOpError("predicate cannot be a blank node");
    }
    return mlir::success();
}