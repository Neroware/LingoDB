#ifndef LINGODB_COMPILER_CONVERSION_GRAPH_GRAPHTYPEIDENTIFIER_H
#define LINGODB_COMPILER_CONVERSION_GRAPH_GRAPHTYPEIDENTIFIER_H
#include <cstdint>
#include "lingodb/compiler/Dialect/DB/IR/DBTypes.h"
#include "lingodb/semantics/Datatypes.h"

using namespace lingodb::semantics;

typedef uint64_t graph_type_id_t;
namespace lingodb::compiler::dialect {
namespace graph {

struct GraphTypeIdentifier {
    XSDType getXsdType(mlir::Type type) {
        if (auto iType = mlir::dyn_cast_or_null<mlir::IntegerType>(type)) {
            if (iType.isSignlessInteger(64))    return XSD_TYPE_UNSIGNED_LONG;
            if (iType.isSignedInteger(64))      return XSD_TYPE_LONG;
            if (iType.isSignlessInteger(32))    return XSD_TYPE_UNSIGNED_INT;
            if (iType.isSignedInteger(32))      return XSD_TYPE_INT;
            if (iType.isSignlessInteger(16))    return XSD_TYPE_UNSIGNED_SHORT;
            if (iType.isSignedInteger(16))      return XSD_TYPE_SHORT;
            if (iType.isSignlessInteger(8))     return XSD_TYPE_UNSIGNED_BYTE;
            if (iType.isSignedInteger(8))       return XSD_TYPE_BYTE;
            if (iType.isInteger(128))           return XSD_TYPE_INTEGER;
        }
        if (mlir::dyn_cast_or_null<mlir::Float32Type>(type)) {
                                                return XSD_TYPE_FLOAT;
        }
        if (mlir::dyn_cast_or_null<mlir::Float64Type>(type)) {
                                                return XSD_TYPE_DOUBLE;
        }
        if (auto sType = mlir::dyn_cast_or_null<db::StringType>(type)) {
                                                return XSD_TYPE_STRING;
        }
        return XSD_UNDEFINED;
    }
};

} // graph
} // lingodb::compiler::dialect

#endif // LINGODB_COMPILER_CONVERSION_GRAPH_GRAPHTYPEIDENTIFIER_H