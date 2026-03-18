#ifndef GENGODB_RUNTIME_GRAPHHELPER_H
#define GENGODB_RUNTIME_GRAPHHELPER_H

#include "gengodb/runtime/Graph.h"
#include "gengodb/runtime/PropertyGraph.h"

namespace lingodb::runtime {
    
struct GraphHelper {
    static GraphBase* allocAndPopulateBuiltinGraph(int32_t builtin);
    static GraphBase* allocGraphState(size_t nodeBufLen, size_t relBufLen, size_t propBufLen);
    static void createGraph(lingodb::runtime::VarLen32 meta);
    static GraphBase* getGraph(lingodb::runtime::VarLen32 description);
}; // GraphHelper

} // namespace lingodb::runtime

#endif // GENGODB_RUNTIME_GRAPHHELPER_H