#ifndef LINGODB_COMPILER_DIALECT_GRAPH_GRAPHDIALECT_H
#define LINGODB_COMPILER_DIALECT_GRAPH_GRAPHDIALECT_H
#include <memory>
namespace llvm {
class hash_code; // NOLINT (readability-identifier-naming)
template <typename T>
llvm::hash_code hash_value(std::shared_ptr<T> arg); // NOLINT (readability-identifier-naming)
} // end namespace llvm
#include "mlir/IR/Dialect.h"

#include "lingodb/compiler/Dialect/Graph/GraphOpsDialect.h.inc"

#ifndef MLIR_HASHCODE_SHARED_PTR
#define MLIR_HASHCODE_SHARED_PTR
namespace llvm {
template <typename T>
llvm::hash_code hash_value(std::shared_ptr<T> arg) { // NOLINT (readability-identifier-naming)
   return hash_value(arg.get());
}
} // end namespace llvm
#endif // MLIR_HASHCODE_SHARED_PTR
#endif // LINGODB_COMPILER_DIALECT_GRAPH_GRAPHDIALECT_H
